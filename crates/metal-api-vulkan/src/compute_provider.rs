//! Compute provider over the shared Vulkan executor: owned bytes, staged
//! lease copies and host-memory no-copy imports.

use crate::{
    execute_pool_sequence_with_status, render, Binding, BoundDispatch, FloatControls2Support,
    LandingTarget, LandingUpdate, PendingExecution, PoolBinding, PoolKey, PoolKind, SequenceTail,
    SpirvFeaturePolicy, TranslatedComputePipeline, VulkanContext, VulkanExecutor,
    VulkanPipelineArtifact,
};
use metal_api_core::completion::wire::CompletionOutbox;
use metal_api_core::completion::{AbandonmentOutcome, CompletionRecord, ObservationDeadline};
pub use metal_api_core::provider::CompiledComputePipeline;
use metal_api_core::provider::{
    allocate_device_epoch, AliasMode, AllocationId, AttachmentFormat, BufferSource, BufferView,
    BufferWriteback, CompletionDisposition, CompletionPolicy, CompletionReadback, CompletionToken,
    ComputeProvider, ComputeTrace, DeviceEpoch, DispatchKind, FieldValue, FunctionIdentity,
    FunctionSource, HeapId, HeapResource, IndirectCommandDescriptor, IndirectCommandKind, LeaseId,
    LeaseImporter, LeaseRegistry, PipelineCompileRequest, PipelineContract, PipelineId,
    PipelineProvider, PresentDescriptor, ProviderCapabilities, ProviderError, ProviderErrorClass,
    ProviderHealth, ProviderPhase, ProviderSubmission, QueuePriority, RenderAttachment,
    RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot, Retryability,
    SemanticDigest, ShaderSource, StagedLease, StorageMode, SubmissionId, TerminalState, TracePass,
    ValidatedComputeTrace, ViewId,
};
use metal_api_core::provider::{
    queue_priorities_for_device, BorrowedLease, BorrowedLeaseRegistry, NoCopyLeaseImporter,
};
use metal_api_core::{AirSource, BufferBinding, Device, Function, Size};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

const TRANSLATOR_REVISION: &[u8] = b"43c46ac8a24adf1a6e872b8a52c706ec9614fad0";
const GPU_DEADLINE: Duration = Duration::from_secs(20);

/// How many present target identities the provider keeps resident across
/// submissions at once.
///
/// The contract's own ceiling (`MAX_PRESENT_TARGETS`) bounds the targets *one
/// trace* names; this is the provider-internal budget for the registry those
/// traces fill over a process lifetime. A target's identity is the
/// `(allocation, view)` pair the present names (`research/docs/24` §5.2), so a
/// guest presenting many surfaces would otherwise grow one image per identity
/// with no release surface at all. The registry evicts the least recently used
/// entry beyond this budget, and a target is also retired as soon as the lease
/// its allocation was imported under is released.
pub const PRESENT_TARGET_BUDGET: usize = 8;

/// How many provider-resident render target identities the provider keeps
/// across submissions at once (`research/docs/23` §76, R7).
///
/// The R4a present registry's rule, generalised to the render rail: a resident
/// target's identity is the `(allocation, view)` pair of the attachment that
/// declares it, so a guest rendering many surfaces would otherwise keep one
/// full-size image per identity for the process's lifetime with no release
/// surface at all. The registry evicts the least recently used entry beyond this
/// budget, and an identity is also retired as soon as the lease its allocation
/// was imported under is released or the device epoch advances. A later
/// `LoadOp::Resident` for an evicted or retired identity is refused by name
/// (`resident_target_evicted` / `resident_target_released` /
/// `resident_target_stale`) — never served from the bytes the image used to
/// hold.
pub const RESIDENT_TARGET_BUDGET: usize = 8;

/// One heap placement a provider executed: which heap a resource landed in,
/// which allocation it belongs to, and the byte range it occupies there.
///
/// This is the falsifiable placement observation `research/docs/25` §6 Step 3
/// asks for: two resources placed in the same heap at different offsets are
/// reported as two records sharing one `heap_id`, not merely "looks shared".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeapPlacementObservation {
    pub heap_id: HeapId,
    pub allocation_id: AllocationId,
    pub offset: u64,
    pub byte_size: u64,
}

/// One indirect replay a provider executed: the command kind, the half-open
/// range it replayed and how many commands it encoded (`research/docs/25`
/// §5.1). The capture reports this record, not the suite's request, so a rail
/// that ran a direct draw cannot claim an indirect replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IcbReplayObservation {
    pub kind: IndirectCommandKind,
    pub start: u32,
    pub count: u32,
    pub commands: u32,
}

/// The provider-side half of a heap payload's mapping to the trace's owned
/// allocations: the slab size and, for each owned allocation, its binding
/// offset inside the slab.
struct HeapPlan {
    slab_size: u64,
    offsets: BTreeMap<u64, u64>,
    /// Each owned allocation's full byte size from its `AllocationRecord`, so
    /// the device buffer spans the whole allocation and matches the
    /// placement's `byte_size` (`research/docs/25` §6 Step 3).
    sizes: BTreeMap<u64, u64>,
    observations: Vec<HeapPlacementObservation>,
}

struct RegisteredPipeline {
    metadata: CompiledComputePipeline,
    artifact: Arc<VulkanPipelineArtifact>,
}

/// One host-registered render pipeline: the trace-table entry the provider
/// minted for it and the two compiled stage modules behind that identity.
///
/// The shape mirrors [`RegisteredPipeline`] on purpose. A trace's pipeline
/// table is the only place a pass says which pipeline it runs, so both rails
/// check the caller-supplied table entry against what the owner registered
/// before anything executes (`validate_pipeline_identity`). The ids share one
/// counter and one namespace: a compute pass naming a render registration is
/// refused as an unknown pipeline, and a render pass naming a compute
/// registration is refused as an unknown render pipeline.
struct RegisteredRenderPipeline {
    metadata: CompiledComputePipeline,
    stages: Arc<render::RenderStages>,
}

/// One render pipeline a host asks a provider context to own.
///
/// `vertex_spirv`/`fragment_spirv` are the two compiled stage modules the
/// graphics pipeline is built from; `contract` is the render contract that
/// names their entries and the attachment format they were compiled against, so
/// the value a trace carries and the value the provider builds are checked
/// against each other rather than trusted. The `logical_digest` is a
/// caller-issued fixture/parity identity, exactly as it is for
/// [`VulkanComputeProvider::compile_pipeline`]: it identifies the case, it does
/// not prove that two modules are equal.
pub struct RenderPipelineRequest {
    pub contract: RenderPipelineContract,
    pub vertex_spirv: Vec<u8>,
    pub fragment_spirv: Vec<u8>,
    pub logical_digest: SemanticDigest,
}

/// One render pipeline a host asks a provider context to own, built from two
/// stages the translator produced.
///
/// This is [`RenderPipelineRequest`]'s sibling for modules the rail did not
/// compile itself: `vertex`/`fragment` carry the SPIR-V the translator returned
/// beside the reflection of the AIR it came from, and the registration gate
/// checks each reflection against `contract` field by field
/// (`render_stage_reflection_mismatch`, `render_stage_unsupported_interface`,
/// `render_stage_translation_unavailable`). The modules are executed as
/// registered — the rail binds each module's own entry point — so the pipeline
/// runs the translation and not the reviewed module of the format list.
///
/// The contract's entry names are the AIR function names the translations
/// started from, which is the identity the pipeline table reports.
pub struct TranslatedRenderPipelineRequest {
    pub contract: RenderPipelineContract,
    pub vertex: crate::TranslatedRenderStage,
    pub fragment: crate::TranslatedRenderStage,
    pub logical_digest: SemanticDigest,
}

/// One planned render pass: the descriptor a trace carries and the stage
/// modules the provider registered for the pipeline it names.
struct PlannedRenderPass {
    pass: RenderPassDescriptor,
    stages: Arc<render::RenderStages>,
}

struct CompletionSlot {
    record: Arc<CompletionRecord>,
    pending: Option<PendingExecution>,
    pool: Vec<BufferView>,
    /// The texture pool the deferred readback maps its storage image landings
    /// through (`research/docs/26` §21.4, C2). A storage image's landing is
    /// keyed by a texture view identity rather than a buffer pool key, so the
    /// slot carries the pool the same way it carries the buffer one.
    textures: Vec<metal_api_core::provider::TextureView>,
    /// Render writebacks produced synchronously during an async submission.
    /// The render rail completes inside `submit` even when the compute half is
    /// deferred, so its bytes ride alongside the deferred pool readback and are
    /// merged into the completion record once the compute fence retires.
    render_writebacks: Vec<BufferWriteback>,
    /// Heap placement observations recorded when the async submission is
    /// planned, published once its writebacks become visible.
    heap_observations: Vec<HeapPlacementObservation>,
    deadline: ObservationDeadline,
}

/// Retains no-copy leases for one submission until a pending execution owns
/// them. If any step before the hand-off fails, Drop returns the retains so
/// the owner is not blocked by a submission that never reached the queue.
struct BorrowedRetains {
    registry: Arc<BorrowedLeaseRegistry>,
    lease_ids: Vec<LeaseId>,
    armed: bool,
}

impl BorrowedRetains {
    fn new(registry: Arc<BorrowedLeaseRegistry>, lease_ids: Vec<LeaseId>) -> Self {
        Self {
            registry,
            lease_ids,
            armed: false,
        }
    }

    fn retain(&mut self) -> Result<(), ProviderError> {
        self.registry.retain_all(&self.lease_ids)?;
        self.armed = true;
        Ok(())
    }

    fn take(&mut self) -> Option<(Arc<BorrowedLeaseRegistry>, Vec<LeaseId>)> {
        if !self.armed {
            return None;
        }
        self.armed = false;
        Some((
            Arc::clone(&self.registry),
            std::mem::take(&mut self.lease_ids),
        ))
    }
}

impl Drop for BorrowedRetains {
    fn drop(&mut self) {
        if self.armed {
            self.registry.retire_all(&self.lease_ids);
        }
    }
}

/// One provider identity sharing the standalone executor's Vulkan device owner.
///
/// This implementation admits up to eight serial exact-thread dispatches
/// selecting registered pipelines over an initialized view pool, with owned
/// bytes, staged lease imports and host readback. Each pass maps a subset of
/// that pool to its pipeline's bindings.
/// By default `submit` waits for GPU completion and readback, and `wait` only
/// observes the recorded terminal result. `with_async_execution(true)` records
/// and submits on the calling thread, returns `Submitted`, and defers the
/// device-fence wait and readback to `wait`/`readback`; no worker is created
/// per submission. Tokens and metadata are process-local, and no-copy leases
/// are imported through `VK_EXT_external_memory_host` when the device
/// advertises it and refused otherwise. Callers can explicitly release
/// registered pipelines and completion records.
pub struct VulkanComputeProvider {
    /// The current device owner. Rebuild replaces the whole executor with a
    /// freshly created device, so the field sits behind a mutex: every
    /// submission path reads the current owner, and a rebuild must not strand
    /// the `Arc` clones those paths hold. The dead owner stays alive exactly
    /// as long as those clones do.
    executor: Mutex<Arc<VulkanExecutor>>,
    /// Current device epoch, kept as the raw counter so `rebuild_after_device_loss`
    /// can advance it through `&self`. `device_epoch()` is the only reader that
    /// materializes the typed value, so the two cannot drift.
    epoch: AtomicU64,
    /// Device-owned capability snapshot, recomputed on rebuild because a fresh
    /// device can answer a different limits/format/fault snapshot.
    capabilities: Mutex<ProviderCapabilities>,
    next_pipeline: AtomicU64,
    next_submission: AtomicU64,
    pipelines: Mutex<BTreeMap<PipelineId, Arc<RegisteredPipeline>>>,
    render_pipelines: Mutex<BTreeMap<PipelineId, Arc<RegisteredRenderPipeline>>>,
    present_targets: Mutex<BTreeMap<(AllocationId, ViewId), PresentTargetEntry>>,
    /// Monotonic use stamp behind the registry's least-recently-used order:
    /// every lookup and every insert stamps its identity with the next value.
    present_target_stamp: AtomicU64,
    /// Cumulative present targets retired before the device epoch ended: the
    /// budget's evictions plus the retirements a lease release drives.
    present_target_evictions: AtomicU64,
    /// The provider-resident render targets the R7 increment adds
    /// (`research/docs/23` §76): the same `(allocation, view)` key and the same
    /// budget/LRU shape as the present registry, but armed by a render pass's
    /// `LoadOp::Resident` / `StoreOp::Resident` instead of a present action, and
    /// loadable by a later submission.
    resident_targets: Mutex<BTreeMap<(AllocationId, ViewId), ResidentTargetEntry>>,
    /// Monotonic use stamp behind the resident registry's least-recently-used
    /// order, exactly as [`Self::present_target_stamp`] orders the present one.
    resident_target_stamp: AtomicU64,
    /// Cumulative resident targets retired before the device epoch ended: the
    /// budget's evictions plus the retirements a lease release or an epoch
    /// advance drives.
    resident_target_evictions: AtomicU64,
    /// The identities the registry has retired in the *current* epoch, with the
    /// rule that retired them. A `LoadOp::Resident` for one of them is refused
    /// with that rule's name; a `StoreOp::Resident` creates the identity again
    /// and clears its tombstone.
    resident_target_tombstones: Mutex<BTreeMap<(AllocationId, ViewId), ResidentTargetRetirement>>,
    /// The allocation each imported lease reserves. A release retires the
    /// present targets *and* the resident render targets created for that
    /// allocation, exactly as the native rail's `drop_present_targets_for` does
    /// (`research/docs/24` §5.2, `research/docs/23` §76).
    lease_allocations: Mutex<BTreeMap<LeaseId, AllocationId>>,
    completions: Mutex<BTreeMap<SubmissionId, CompletionSlot>>,
    heap_observations: Mutex<Vec<HeapPlacementObservation>>,
    icb_observations: Mutex<Vec<IcbReplayObservation>>,
    retire_tx: Mutex<Option<mpsc::Sender<PendingExecution>>>,
    observation_deadline: Duration,
    async_execution: bool,
    completion_outbox: Option<Arc<CompletionOutbox>>,
    staging: LeaseRegistry,
    borrowed: Arc<BorrowedLeaseRegistry>,
}

/// One resident present target: the provider-owned image and the stamp that
/// orders it in the registry's least-recently-used eviction.
struct PresentTargetEntry {
    image: Arc<render::ProviderTargetImage>,
    last_used: u64,
}

/// One resident render target: the provider-owned image, the shape the trace
/// declared for it, and the bookkeeping the registry's budget and the rail's
/// published layout need (`research/docs/23` §76, R7).
struct ResidentTargetEntry {
    image: Arc<render::ProviderTargetImage>,
    /// The format the identity was created with. A later pass naming the same
    /// identity with a different format is refused by name rather than rendered
    /// into an image of the wrong format.
    format: AttachmentFormat,
    width: u64,
    height: u64,
    /// Whether a completed pass has defined the image's bytes. An entry is
    /// created before its first pass runs, so this is what keeps a refused or
    /// failed pass from leaving an image a later `LoadOp::Resident` could read
    /// as "the target's contents".
    defined: bool,
    last_used: u64,
}

/// Why a resident target identity is no longer in the registry
/// (`research/docs/23` §76, R7).
///
/// The tombstone is what makes the refusal nameable: a `LoadOp::Resident` for
/// an identity that is gone says *which* rule retired it, so the guest can
/// re-render the frame instead of guessing whether it forgot to store one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentTargetRetirement {
    /// The registry's budget evicted the least recently used identity.
    Budget,
    /// The lease the identity's allocation was imported under was released.
    LeaseReleased,
    /// The device epoch advanced: every image of the dead device is gone.
    EpochAdvance,
}

impl ResidentTargetRetirement {
    /// The refusal slug a load of this retired identity states.
    const fn slug(self) -> &'static str {
        match self {
            Self::Budget => "resident_target_evicted",
            Self::LeaseReleased => "resident_target_released",
            Self::EpochAdvance => "resident_target_stale",
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Budget => "budget",
            Self::LeaseReleased => "lease_released",
            Self::EpochAdvance => "epoch_advance",
        }
    }
}

/// The provider-side capability snapshot for one device owner.
///
/// [`VulkanExecutor::provider_capabilities`] answers the device's own limits;
/// this overlay adds the provider-shaped windows (`max_passes`, ranged
/// aliasing, staged leases and, when the device advertises
/// `VK_EXT_external_memory_host`, borrowed no-copy leases). Both
/// [`VulkanComputeProvider::with_executor`] and the rebuild path run it, so a
/// fresh device never keeps a stale snapshot from the dead one.
fn build_provider_capabilities(executor: &VulkanExecutor) -> ProviderCapabilities {
    let mut capabilities = executor.provider_capabilities();
    capabilities.max_passes = 8;
    // Ranged aliasing is admitted: every pool entry owns its own device
    // buffer, so two disjoint views of one allocation never share GPU
    // bytes. A pass binding one view cannot observe another view's writes
    // because each view's footprint proof bounds its accesses inside its
    // own half-open range, and admission rejects overlapping ranges
    // outright. Writeback stays byte-exact per view (the offset is
    // re-based onto the allocation), and the reserved allocation is still
    // exclusive for the commit-to-completion window.
    capabilities.alias_mode = AliasMode::DistinctViews;
    if !capabilities
        .storage_modes
        .contains(&StorageMode::StagedLease)
    {
        capabilities.storage_modes.push(StorageMode::StagedLease);
    }
    if executor.context.external_memory_host_alignment() > 0 {
        capabilities.storage_modes.push(StorageMode::BorrowedNoCopy);
    }
    capabilities
}

impl VulkanComputeProvider {
    pub fn new() -> Result<Self, ProviderError> {
        let executor = VulkanExecutor::new().map_err(|error| {
            refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Resource,
                "device_initialization_failed",
            )
            .with_detail(error.to_string())
        })?;
        Self::with_executor(executor)
    }

    /// Use the same device and queue lock as an existing snapshot executor.
    pub fn with_executor(executor: Arc<VulkanExecutor>) -> Result<Self, ProviderError> {
        let epoch = allocate_device_epoch()?;
        let capabilities = build_provider_capabilities(&executor);
        Ok(Self {
            executor: Mutex::new(executor),
            epoch: AtomicU64::new(epoch.get()),
            capabilities: Mutex::new(capabilities),
            next_pipeline: AtomicU64::new(1),
            next_submission: AtomicU64::new(1),
            pipelines: Mutex::new(BTreeMap::new()),
            render_pipelines: Mutex::new(BTreeMap::new()),
            present_targets: Mutex::new(BTreeMap::new()),
            present_target_stamp: AtomicU64::new(0),
            present_target_evictions: AtomicU64::new(0),
            resident_targets: Mutex::new(BTreeMap::new()),
            resident_target_stamp: AtomicU64::new(0),
            resident_target_evictions: AtomicU64::new(0),
            resident_target_tombstones: Mutex::new(BTreeMap::new()),
            lease_allocations: Mutex::new(BTreeMap::new()),
            completions: Mutex::new(BTreeMap::new()),
            heap_observations: Mutex::new(Vec::new()),
            icb_observations: Mutex::new(Vec::new()),
            retire_tx: Mutex::new(None),
            observation_deadline: GPU_DEADLINE,
            async_execution: false,
            completion_outbox: None,
            staging: LeaseRegistry::new(),
            borrowed: Arc::new(BorrowedLeaseRegistry::new()),
        })
    }

    /// Select deferred submission. The default synchronous mode is retained
    /// for the direct trace rail and existing captures.
    pub fn with_async_execution(mut self, async_execution: bool) -> Self {
        self.async_execution = async_execution;
        self
    }

    /// Rebuild the device in place after a confirmed device loss.
    ///
    /// The one precondition is the exact terminal state the core lifecycle
    /// keeps: `DeviceLost`. A usable context, an abandonment-exhausted
    /// context, or a provider that still carries a completion outbox is
    /// refused fail-closed (`rebuild_requires_device_loss` /
    /// `rebuild_with_completion_outbox`) instead of being partially rebuilt.
    /// The outbox refusal is deliberate: the outbox is scoped to the dead
    /// epoch, and silently publishing fresh-epoch tokens into it would break
    /// its `device_epoch` contract.
    ///
    /// On success the provider owns a brand-new `vk::Device`, queues and
    /// capability snapshot, and its `DeviceEpoch` has advanced. Every
    /// registration bound to the dead device is dropped — pipelines, render
    /// pipelines, present-target images, completion records and the
    /// last-submission observations — because Vulkan invalidates every device
    /// child with the device. Old-epoch completion tokens keep their existing
    /// `device_epoch_mismatch` refusal and old-epoch leases their existing
    /// `lease_epoch_mismatch` refusal; neither is silently re-admitted. The
    /// caller re-registers pipelines and re-validates the trace, exactly as
    /// it would for a freshly created provider.
    pub fn rebuild_after_device_loss(&self) -> Result<(), ProviderError> {
        // Fail closed before touching anything: rebuild is only the repair for
        // a terminal `DeviceLost`, and the outbox is epoch-scoped state that
        // cannot survive the epoch advance.
        {
            let executor = self.lock_executor()?;
            let lifecycle = executor.context.lock_lifecycle();
            if !lifecycle.ended_by_device_loss() {
                return Err(refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Resource,
                    "rebuild_requires_device_loss",
                )
                .with_field(
                    "terminal",
                    FieldValue::Text(terminal_state_name(lifecycle.state()).to_owned()),
                ));
            }
            if self.completion_outbox.is_some() {
                return Err(refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Resource,
                    "rebuild_with_completion_outbox",
                )
                .with_detail(
                    "the completion outbox is scoped to the dead device epoch; recreate the \
                     provider with a fresh outbox instead",
                ));
            }
        }

        // Build the replacement before mutating anything: if the new device
        // cannot be created, the provider stays exactly as it was (terminal
        // and refusing), so the refusal still names a failed rebuild.
        let fresh = VulkanExecutor::new().map_err(|error| {
            refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Resource,
                "rebuild_device_initialization_failed",
            )
            .with_detail(error.to_string())
        })?;
        let epoch = allocate_device_epoch()?;
        let capabilities = build_provider_capabilities(&fresh);

        // Install the fresh owner and the advanced epoch. Terminal states are
        // monotonic, so the precondition above cannot have unwound between the
        // check and the swap; poisoning here is recovered the same way the
        // lifecycle lock is, because a panic must not leave the owner stranded
        // on a dead device.
        {
            let mut executor = self
                .executor
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *executor = fresh;
        }
        self.epoch.store(epoch.get(), Ordering::Relaxed);
        *self
            .capabilities
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = capabilities;
        self.pipelines
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.render_pipelines
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.present_targets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        // Every resident target image belongs to the dead device, so the whole
        // registry goes with it — and each identity that was alive leaves a
        // tombstone naming the epoch advance, so a trace that later loads one of
        // them is refused with `resident_target_stale` instead of being served
        // an image from a device that no longer exists
        // (`research/docs/23` §76, R7).
        let retired_resident = {
            let mut registry = self
                .resident_targets
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::take(&mut *registry)
                .into_iter()
                .collect::<Vec<((AllocationId, ViewId), ResidentTargetEntry)>>()
        };
        self.retire_resident_targets(&retired_resident, ResidentTargetRetirement::EpochAdvance)?;
        self.completions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.heap_observations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.icb_observations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        // The retirement worker captured the dead context when it spawned;
        // closing the channel lets it drain and exit so a future deferred
        // submission spawns a worker bound to the new device instead.
        {
            let mut retire = self
                .retire_tx
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *retire = None;
        }
        Ok(())
    }

    /// Build the explicitly declared heap-aliasing test snapshot.
    ///
    /// This exists only for the heap-aliasing hazard fixture (`research/docs/25`
    /// §7.1): the production capability snapshot built by [`Self::with_executor`]
    /// keeps `supports_heap_aliasing = false`, and every production caller must
    /// keep using that snapshot. The fixture flips exactly this one capability
    /// bit so the aliasing shape reaches execution through real admission, not
    /// through a bypass. Nothing else in the snapshot changes, so the fixture
    /// still fails closed if it accidentally loses this marker.
    pub fn with_heap_aliasing_test_snapshot(mut self) -> Self {
        self.capabilities
            .get_mut()
            .expect("provider owns its capability snapshot")
            .supports_heap_aliasing = true;
        self
    }

    pub fn async_execution(&self) -> bool {
        self.async_execution
    }

    /// Heap placements the provider most recently executed successfully.
    ///
    /// Every successful submission replaces the previous vector instead of
    /// appending to it, so this stays bounded to one submission rather than
    /// growing across a long-running process. Each record names the heap, the
    /// owned allocation placed in it, and the `[offset, offset + byte_size)`
    /// range it occupies. Two resources in the same heap therefore appear as
    /// two records sharing one `heap_id`, which is the falsifiable observation
    /// `research/docs/25` §6 Step 3 requires rather than a "looks shared"
    /// assertion.
    pub fn heap_placement_observations(&self) -> Vec<HeapPlacementObservation> {
        self.heap_observations
            .lock()
            .expect("heap observation lock poisoned")
            .clone()
    }

    /// The indirect replay the provider executed last, if the last submission
    /// carried one. Like the heap observation, a successful submission replaces
    /// the vector instead of appending to it.
    pub fn icb_replay_observations(&self) -> Vec<IcbReplayObservation> {
        self.icb_observations
            .lock()
            .expect("icb observation lock poisoned")
            .clone()
    }

    /// Record one indirect replay, replacing whatever the previous submission
    /// left behind (the same bounded shape the heap observation uses).
    fn publish_icb_observation(
        &self,
        kind: IndirectCommandKind,
        start: u32,
        count: u32,
        commands: u32,
    ) {
        let mut observations = self
            .icb_observations
            .lock()
            .expect("icb observation lock poisoned");
        observations.clear();
        observations.push(IcbReplayObservation {
            kind,
            start,
            count,
            commands,
        });
    }

    /// Publish admission, terminal transitions and device health through
    /// `outbox`. The outbox must be scoped to this provider's device epoch.
    /// Without an outbox the provider keeps its in-process behavior.
    pub fn with_completion_outbox(
        mut self,
        outbox: Arc<CompletionOutbox>,
    ) -> Result<Self, ProviderError> {
        if outbox.device_epoch() != self.device_epoch() {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Args,
                "completion_outbox_epoch_mismatch",
            )
            .with_field("expected", FieldValue::Unsigned(self.device_epoch().get()))
            .with_field("actual", FieldValue::Unsigned(outbox.device_epoch().get())));
        }
        self.completion_outbox = Some(outbox);
        Ok(self)
    }

    /// Completion outbox attached to this provider, if any.
    pub fn completion_outbox(&self) -> Option<&Arc<CompletionOutbox>> {
        self.completion_outbox.as_ref()
    }

    /// Bound how long a deferred submission may remain non-terminal before
    /// `wait` publishes unknown completion. The default is 20 seconds.
    pub fn with_observation_deadline(mut self, limit: Duration) -> Self {
        self.observation_deadline = limit;
        self
    }

    pub fn device_epoch(&self) -> DeviceEpoch {
        DeviceEpoch::new(self.epoch.load(Ordering::Relaxed))
    }

    pub fn device_name(&self) -> String {
        self.lock_executor()
            .expect("executor lock poisoned")
            .device_name()
            .to_owned()
    }

    /// What this provider's device reported about
    /// `VK_KHR_shader_float_controls2` (R8).
    ///
    /// The two readings — extension name and `shaderFloatControls2` — are the
    /// capability snapshot a caller records, and [`Self::spirv_feature_policy`]
    /// is the gate answer derived from them.
    pub fn float_controls2_support(&self) -> FloatControls2Support {
        self.lock_executor()
            .expect("executor lock poisoned")
            .float_controls2_support()
    }

    /// The SPIR-V capability policy this provider's device answers with (R8).
    ///
    /// A caller that translates outside the provider — the render rail's
    /// translated-stage arm translates each stage itself — asks for this and
    /// hands it to `TranslatedRenderStage::translate_with_policy`, so the
    /// module it registers is the module this device validated. The provider
    /// re-asks the same policy at registration regardless.
    pub fn spirv_feature_policy(&self) -> SpirvFeaturePolicy {
        self.lock_executor()
            .expect("executor lock poisoned")
            .spirv_feature_policy()
    }

    /// Report whether this provider can still admit new work.
    ///
    /// Health and admission are the same query on the same lifecycle, so a
    /// provider that reports `Usable` here also admits the next submission.
    pub fn health(&self) -> ProviderHealth {
        self.lock_executor()
            .expect("executor lock poisoned")
            .context
            .health()
    }

    /// Report `(abandoned submissions, abandoned bytes)` for this context.
    pub fn abandonment_stats(&self) -> (u64, u64) {
        self.lock_executor()
            .expect("executor lock poisoned")
            .context
            .abandonment_stats()
    }

    /// Translate and register a function. The logical digest is a caller-issued
    /// fixture/parity identity; it is not used to reuse an artifact or to prove
    /// equality of differently encoded modules. Each compile gets its own ID.
    pub fn compile_pipeline(
        &self,
        function: &Function,
        logical_digest: SemanticDigest,
    ) -> Result<CompiledComputePipeline, ProviderError> {
        self.ensure_usable()?;
        // The device answers for its own capability subset: the gate runs under
        // the policy this device derived, and the lock is taken for the read
        // only — translation itself must not hold it.
        let policy = self.spirv_feature_policy();
        let translated = TranslatedComputePipeline::translate_with_policy(function, policy)
            .map_err(|error| {
                refusal(
                    ProviderPhase::Compile,
                    ProviderErrorClass::Compile,
                    "pipeline_translation_failed",
                )
                .with_detail(error.to_string())
            })?;
        let revision = SemanticDigest::new("git-commit", TRANSLATOR_REVISION.to_vec())
            .expect("non-empty pinned translator identity");
        let contract = translated
            .provider_contract(Some(revision))
            .map_err(|error| {
                refusal(
                    ProviderPhase::Compile,
                    ProviderErrorClass::Compile,
                    "pipeline_reflection_failed",
                )
                .with_detail(error.to_string())
            })?;
        self.ensure_usable()?;
        let metadata = CompiledComputePipeline {
            device_epoch: self.device_epoch(),
            pipeline_id: PipelineId::new(next_identity(
                &self.next_pipeline,
                "pipeline_identity_exhausted",
            )?),
            function: FunctionIdentity {
                logical_digest,
                entry_name: function.name().to_owned(),
                source: match function.air_source() {
                    AirSource::SanitizedLl(_) => FunctionSource::SanitizedLl,
                    AirSource::Binary(_) => FunctionSource::BinaryAir,
                },
            },
            contract,
            // A compute registration has no render half: a render pass naming
            // this id finds no render contract here and is refused by core
            // admission.
            render: None,
        };
        let registered = RegisteredPipeline {
            metadata: metadata.clone(),
            artifact: Arc::new(VulkanPipelineArtifact {
                context: Arc::clone(&self.lock_executor()?.context),
                translated,
            }),
        };
        self.pipelines
            .lock()
            .map_err(|_| registry_poisoned())?
            .insert(metadata.pipeline_id, Arc::new(registered));
        Ok(metadata)
    }

    /// Register one offscreen render pipeline from its two compiled stages.
    ///
    /// The returned metadata is the entry the trace's pipeline table carries,
    /// the same hand-off [`Self::compile_pipeline`] gives the compute rail. The
    /// stage modules stay in this provider context; a trace names the pipeline
    /// by id and the provider refuses a table entry that does not match the
    /// registration.
    ///
    /// The table entry is a [`CompiledComputePipeline`] because that is the one
    /// entry shape `ComputeTrace` has today (`research/docs/23` §6 Step 4 grows
    /// it). It carries the registration's vertex entry as the function entry
    /// name, an exact-thread contract that binds nothing, and the reviewed
    /// contract in the entry's `render` half, which is what core admission
    /// compares the pass's attachment against. It is deliberately unreachable
    /// from the compute rail: a compute pass naming this id finds no artifact in
    /// the compute registry and is refused as `unknown_pipeline`.
    pub fn register_render_pipeline(
        &self,
        request: RenderPipelineRequest,
    ) -> Result<CompiledComputePipeline, ProviderError> {
        let RenderPipelineRequest {
            contract,
            vertex_spirv,
            fragment_spirv,
            logical_digest,
        } = request;
        let stages = render::RenderStages {
            contract,
            vertex_spirv,
            fragment_spirv,
            vertex_translation: None,
            fragment_translation: None,
        };
        self.register_render_stages(stages, logical_digest)
    }

    /// Register one offscreen render pipeline from two translated stages.
    ///
    /// The sibling of [`Self::register_render_pipeline`] for modules the rail
    /// did not compile itself (`research/docs/23`, R2 increment): the caller
    /// hands the SPIR-V the translator returned *and* the reflection of the AIR
    /// it came from, and the registration gate checks each reflection against
    /// the contract — stage, entry, vertex attributes against the declared
    /// vertex layout, render targets against the declared colour format list,
    /// and the varyings of the two stages against each other. Nothing about the
    /// pipeline is inferred from the module bytes.
    ///
    /// The returned metadata is the same table entry
    /// [`Self::register_render_pipeline`] mints, and both registrations flow
    /// through one registration gate and one registry slot.
    pub fn register_translated_render_pipeline(
        &self,
        request: TranslatedRenderPipelineRequest,
    ) -> Result<CompiledComputePipeline, ProviderError> {
        let TranslatedRenderPipelineRequest {
            contract,
            vertex,
            fragment,
            logical_digest,
        } = request;
        let stages = render::RenderStages {
            contract,
            vertex_spirv: vertex.spirv,
            fragment_spirv: fragment.spirv,
            vertex_translation: Some(vertex.reflection),
            fragment_translation: Some(fragment.reflection),
        };
        self.register_render_stages(stages, logical_digest)
    }

    /// The one registration path both render entry points take.
    ///
    /// Structural and interface validation ([`render::RenderStages::validate`]),
    /// the table metadata and the registry insertion live here so the reviewed
    /// and the translated registration cannot drift apart: they are the same
    /// pipeline to the trace table, the registry and the execution path.
    fn register_render_stages(
        &self,
        stages: render::RenderStages,
        logical_digest: SemanticDigest,
    ) -> Result<CompiledComputePipeline, ProviderError> {
        self.ensure_usable()?;
        // The device answers for the modules as well (R8): the registration
        // gate re-asks the capability subset the translation gate asked, with
        // this provider's own policy, so a caller cannot register a module its
        // device could not create — whether it translated elsewhere or handed
        // the rail a module of its own. It is asked before the module's own
        // accounting, in the same order the translation entry point asks it.
        render::validate_module_capabilities(&stages, self.spirv_feature_policy())?;
        stages.validate()?;
        let function = FunctionIdentity {
            logical_digest,
            entry_name: stages.contract.vertex_entry.clone(),
            // The registration hands the rail compiled stage modules, i.e. a
            // Metal-side binary rather than source text. The field is table
            // metadata the render rail never reads; the modules themselves are
            // what the pipeline is built from.
            source: FunctionSource::Metallib,
        };
        function.validate().map_err(|error| {
            refusal(
                ProviderPhase::Compile,
                ProviderErrorClass::Args,
                "render_pipeline_contract_invalid",
            )
            .with_detail(error.to_string())
        })?;
        let metadata = CompiledComputePipeline {
            device_epoch: self.device_epoch(),
            pipeline_id: PipelineId::new(next_identity(
                &self.next_pipeline,
                "pipeline_identity_exhausted",
            )?),
            function,
            contract: render_pipeline_table_contract(),
            render: Some(stages.contract.clone()),
        };
        self.ensure_usable()?;
        let registered = Arc::new(RegisteredRenderPipeline {
            metadata: metadata.clone(),
            stages: Arc::new(stages),
        });
        self.render_pipelines
            .lock()
            .map_err(|_| registry_poisoned())?
            .insert(metadata.pipeline_id, registered);
        Ok(metadata)
    }

    /// Stop accepting new submissions using this render pipeline.
    ///
    /// Symmetric with [`PipelineProvider::release_pipeline`]: the epoch and the
    /// registered identity are verified before the entry is removed, so a stale
    /// or foreign value cannot release another context's registration. A
    /// submission that is already encoding holds its own `Arc`, so releasing
    /// never pulls the stage modules out from under running work.
    pub fn release_render_pipeline(
        &self,
        metadata: &CompiledComputePipeline,
    ) -> Result<(), ProviderError> {
        check_epoch(self.device_epoch(), metadata.device_epoch)?;
        let mut registrations = self
            .render_pipelines
            .lock()
            .map_err(|_| registry_poisoned())?;
        let registered = registrations
            .get(&metadata.pipeline_id)
            .ok_or_else(|| unknown_render_pipeline(metadata.pipeline_id))?;
        if registered.metadata != *metadata {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Resource,
                "render_pipeline_identity_mismatch",
            ));
        }
        registrations.remove(&metadata.pipeline_id);
        Ok(())
    }

    /// Stop accepting new submissions using this pipeline. An in-flight submit
    /// retains its own Arc until completion, independent of registry removal.
    pub fn release_pipeline(
        &self,
        pipeline: &CompiledComputePipeline,
    ) -> Result<(), ProviderError> {
        PipelineProvider::release_pipeline(self, pipeline)
    }

    /// Forget a completion observation. Removing a running deferred record
    /// does not cancel device work: the provider hands the pending submission
    /// to a shared retirement thread that waits for its fence and then releases
    /// the handles. The same retirement path reclaims a submission whose
    /// observation deadline expired, so a timed-out submission does not poison
    /// the shared context unless its fence never signals.
    pub fn release_completion(&self, token: CompletionToken) -> Result<(), ProviderError> {
        self.validate_token(token)?;
        let slot = self
            .completions
            .lock()
            .map_err(|_| registry_poisoned())?
            .remove(&token.submission_id)
            .ok_or_else(|| unknown_completion(token))?;
        if let Some(pending) = slot.pending {
            self.retire(pending);
        }
        Ok(())
    }

    /// Plan the render passes of one admitted trace, before any device work.
    ///
    /// Planning is where a render-bearing trace meets the provider's own
    /// registry and the rail's pass order: a pipeline the context never
    /// registered, a table entry that disagrees with the registration, or an
    /// order the rail cannot honour is refused here, so a submission that
    /// cannot run end to end executes nothing at all. The deferral path is
    /// refused outright — the render rail completes inside `submit` while the
    /// deferred path records now and reports at `wait`, so admitting render
    /// work there would report bytes no rail read
    /// (`research/docs/23` §6 Step 5 owns that shape).
    fn plan_render_passes(
        &self,
        trace: &ComputeTrace,
    ) -> Result<Vec<PlannedRenderPass>, ProviderError> {
        if !trace.has_render_passes() {
            return Ok(Vec::new());
        }
        refuse_reordered_render_reads(trace)?;
        let registrations = self
            .render_pipelines
            .lock()
            .map_err(|_| registry_poisoned())?;
        let mut plan = Vec::with_capacity(trace.render_passes().count());
        for pass in trace.render_passes() {
            let registered = registrations
                .get(&pass.pipeline)
                .ok_or_else(|| unknown_render_pipeline(pass.pipeline))?;
            let requested = trace.pipeline(pass.pipeline).map_err(|error| {
                refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Resource,
                    "render_pipeline_identity_mismatch",
                )
                .with_detail(error.to_string())
            })?;
            validate_pipeline_identity(requested, &registered.metadata)?;
            plan.push(PlannedRenderPass {
                pass: pass.clone(),
                stages: Arc::clone(&registered.stages),
            });
        }
        Ok(plan)
    }

    /// Resolve the indirect dispatch the compute rail replays, or `None` when
    /// the trace carries no indirect command or carries the render rail's draw
    /// command.
    ///
    /// The compute rail owns `Dispatch` (`research/docs/25` §6 Step 4): the
    /// payload's `threadgroups` is the single workgroup count the rail encodes
    /// and replays. A dispatch that arrives next to a render pass, or a trace
    /// whose compute passes do not give the command exactly one target, is a
    /// shape mismatch the first increment refuses fail-closed rather than
    /// silently replaying into one pass or dropping the command.
    fn indirect_dispatch_threadgroups(
        &self,
        trace: &ComputeTrace,
    ) -> Result<Option<[u32; 3]>, ProviderError> {
        let Some(payload) = trace.indirect.as_deref() else {
            return Ok(None);
        };
        let IndirectCommandDescriptor::Dispatch { threadgroups } = payload.command else {
            return Ok(None);
        };
        if trace.has_render_passes() {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Capability,
                "icb_command_unsupported",
            )
            .with_detail("an indirect dispatch replays a compute pass, not a render pass"));
        }
        let compute_passes = trace.compute_passes().count();
        if compute_passes != 1 {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Capability,
                "icb_command_unsupported",
            )
            .with_field(
                "passes",
                FieldValue::Unsigned(u64::try_from(compute_passes).unwrap_or(u64::MAX)),
            )
            .with_detail("an indirect dispatch replays exactly one compute pass"));
        }
        Ok(Some(threadgroups))
    }

    /// Execute the planned render passes in trace order, after the compute
    /// sequence, and turn each stored attachment readback into a buffer
    /// writeback. A `StoreOp::DontCare` attachment lands no writeback: its
    /// bytes are discarded by the pass, so they disappear from the observable
    /// surface instead of being published (`docs/23` §3.6, v19).
    ///
    /// Every attachment's bytes leave the rail through the same channel a
    /// compute pass uses: one [`BufferWriteback`] per attachment for the view
    /// and allocation the trace declared, at the view's own offset inside the
    /// allocation, so resource admission, lease bookkeeping and readback
    /// consumers need no second path. An attachment that no buffer view covers
    /// has no such landing rail and is refused instead of being executed and
    /// dropped. A presenting pass keeps the pre-MRT single-attachment shape.
    fn execute_render_passes(
        &self,
        trace: &ComputeTrace,
        pool: &[BufferView],
        plan: &[PlannedRenderPass],
        resources: &ResourceTableSnapshot,
    ) -> Result<Vec<BufferWriteback>, ProviderError> {
        // The render rail's indirect payload replays one draw or indexed draw
        // command into exactly one plain render pass (`research/docs/25` §6
        // Step 4). The guards run before the empty-plan early return on
        // purpose: a trace that carries an indirect draw but no render pass has
        // nothing to replay into, and reporting success while dropping the
        // command would be fail-open. A dispatch payload never reaches this
        // guard: it is owned by the compute rail, and
        // `indirect_dispatch_threadgroups` already refused the
        // render-pass-plus-dispatch shape before the render plan was built.
        let render_replays_indirect = trace.indirect.as_deref().is_some_and(|payload| {
            matches!(
                payload.command.kind(),
                IndirectCommandKind::Draw | IndirectCommandKind::DrawIndexed
            )
        });
        if render_replays_indirect {
            if plan.is_empty() {
                return Err(refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Capability,
                    "icb_command_unsupported",
                )
                .with_detail("the first indirect increment needs a render pass to replay into"));
            }
            if plan.len() != 1 {
                return Err(refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Capability,
                    "icb_command_unsupported",
                )
                .with_field(
                    "passes",
                    FieldValue::Unsigned(u64::try_from(plan.len()).unwrap_or(u64::MAX)),
                )
                .with_detail("the first indirect increment replays into exactly one render pass"));
            }
            if plan.iter().any(|planned| planned.pass.present.is_some()) {
                return Err(refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Capability,
                    "icb_command_unsupported",
                )
                .with_detail(
                    "the first indirect increment replays plain render passes, not presenting ones",
                ));
            }
        }
        if plan.is_empty() {
            return Ok(Vec::new());
        }
        // Both rails share one lease channel (`research/docs/23` §71, R3c): the
        // same staged and no-copy registries this provider imports into, the
        // admitted snapshot that is authoritative for every reservation, and
        // the device's own host-import alignment. The render rail resolves its
        // vertex and index inputs through it exactly as the compute rail
        // resolves its pool bindings.
        let host_import_alignment = self
            .lock_executor()
            .map_err(|_| registry_poisoned())?
            .context
            .external_memory_host_alignment();
        let leases = render::RenderLeaseContext {
            staging: &self.staging,
            borrowed: &self.borrowed,
            resources,
            device_epoch: self.device_epoch(),
            host_import_alignment,
        };
        let host_readback = trace.completion_policy == CompletionPolicy::HostReadback;
        let mut writebacks = Vec::with_capacity(plan.len());
        for planned in plan {
            if let Some(present) = &planned.pass.present {
                // The present rail renders exactly one attachment into the
                // provider-owned target; the pre-MRT gate stays in place rather
                // than being widened, so present keeps its single-attachment
                // byte behaviour (`research/docs/24` §3.5 shape one).
                let [attachment] = planned.pass.color_attachments.as_slice() else {
                    return Err(refusal(
                        ProviderPhase::Resolve,
                        ProviderErrorClass::Args,
                        "trace_contract_invalid",
                    )
                    .with_detail("the present rail executes exactly one colour attachment"));
                };
                // The present rail hands its target on through the present
                // action, which is the R4a registry's own shape
                // (`docs/24` §5.2): a pass that also declares the render rail's
                // resident target would be asking two registries to own one
                // identity, so it is refused by name instead of executed with
                // one of them silently winning (`research/docs/23` §76, R7).
                if attachment.declares_resident_target() {
                    return Err(refusal(
                        ProviderPhase::Resolve,
                        ProviderErrorClass::Capability,
                        "resident_target_present_unsupported",
                    )
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
                // The declared view serves two purposes: it is the attachment's
                // landing for the writeback channel, and it is the source of
                // the previous contents a `LoadOp::Load` pass uploads before it
                // opens (`research/docs/23` §3.3/§74). A loading pass therefore
                // needs the declaration even when the trace asks for no host
                // readback.
                let declared = pool.iter().find(|view| {
                    view.view_id == attachment.view_id
                        && view.allocation_id == attachment.allocation_id
                });
                let loading = matches!(attachment.load, metal_api_core::provider::LoadOp::Load);
                let storing = matches!(attachment.store, metal_api_core::provider::StoreOp::Store);
                // A stored attachment's bytes land through the writeback
                // channel and a loading attachment uploads the trace's own
                // bytes, so either needs the declaring view. A discarded
                // attachment needs no landing declaration — unless it also
                // loads, whose previous bytes still come from the declaration
                // (`docs/23` §3.6, v19).
                let view = if (host_readback && storing) || loading {
                    Some(declared.ok_or_else(|| {
                        refusal(
                            ProviderPhase::Resolve,
                            ProviderErrorClass::Capability,
                            "render_attachment_landing_unsupported",
                        )
                        .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
                        .with_field(
                            "allocation",
                            FieldValue::Unsigned(attachment.allocation_id.get()),
                        )
                        .with_detail(
                            "attachment bytes land through the buffer writeback channel and \
                             `LoadOp::Load` uploads the trace's own bytes, and this trace \
                             declares no buffer view covering the attachment",
                        )
                    })?)
                } else {
                    None
                };
                // A loading attachment's previous contents are resolved by the
                // rail from this same declaration: the caller hands the view
                // over and the rail decides its source arm — trace-owned bytes,
                // the provider's staged copy, or the owner's imported window —
                // so a lease-backed declaration is never snapshotted here
                // (`research/docs/23` §74, R5b).
                let previous = view.filter(|_| loading);
                let target = self.present_target(present, attachment)?;
                let executor = self.lock_executor()?;
                let texels = render::execute_present_render(
                    &executor.context,
                    &planned.stages,
                    &planned.pass,
                    &target,
                    previous,
                    Some(&leases),
                )?;
                if let Some(view) = view {
                    writebacks.push(BufferWriteback {
                        view_id: view.view_id,
                        allocation_id: view.allocation_id,
                        offset: view.offset,
                        bytes: texels,
                    });
                }
                continue;
            }

            // The offscreen shape: resolve one landing view and one
            // previous-contents declaration per attachment, in location order,
            // then hand the render rail the whole list and publish one
            // writeback per attachment that has a landing. The declaration is
            // what the rail resolves into bytes, so a lease-backed attachment
            // load is imported (or refused by name) inside the rail rather
            // than being snapshotted here (`research/docs/23` §74, R5b).
            let mut views = Vec::with_capacity(planned.pass.color_attachments.len());
            let mut previous = Vec::with_capacity(planned.pass.color_attachments.len());
            // The provider-resident targets this pass declares, in location
            // order (`research/docs/23` §76, R7). The identity is the
            // attachment's own pair, so the registry and the contract cannot
            // disagree about *which* target a pass means.
            let mut resident = Vec::with_capacity(planned.pass.color_attachments.len());
            let mut resident_identities = Vec::new();
            for attachment in &planned.pass.color_attachments {
                let declared = pool.iter().find(|view| {
                    view.view_id == attachment.view_id
                        && view.allocation_id == attachment.allocation_id
                });
                // A resident target is resolved — or refused by name — before
                // any device object exists: the registry decides whether the
                // identity holds bytes a load may read, and a pass that renders
                // into a resident identity without declaring it is refused
                // rather than silently overwriting the provider's bytes.
                if attachment.declares_resident_target() {
                    let identity = (attachment.allocation_id, attachment.view_id);
                    let image =
                        self.resident_target(attachment, attachment.loads_resident_target())?;
                    resident_identities.push(identity);
                    resident.push(Some(image));
                } else {
                    if self.resident_target_identity_is_resident(
                        attachment.allocation_id,
                        attachment.view_id,
                    ) {
                        return Err(refusal(
                            ProviderPhase::Resolve,
                            ProviderErrorClass::Capability,
                            "resident_target_undeclared",
                        )
                        .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
                        .with_field(
                            "allocation",
                            FieldValue::Unsigned(attachment.allocation_id.get()),
                        )
                        .with_detail(
                            "the provider holds this identity's image and the pass declares \
                             neither `LoadOp::Resident` nor `StoreOp::Resident` for it, so the \
                             trace would be reading or overwriting bytes it never named",
                        ));
                    }
                    resident.push(None);
                }
                // The landing view is the writeback channel's declaration, and
                // `LoadOp::Load` uploads the trace's own bytes through it. A
                // resident store has neither: its bytes stay in the provider's
                // image and the pass publishes no writeback for it, so it needs
                // no landing view even when the trace asks for a host readback
                // (`research/docs/23` §76, R7). Every other store arm keeps the
                // pre-R7 rule unchanged.
                let loading = matches!(attachment.load, metal_api_core::provider::LoadOp::Load);
                let landing_needed = host_readback
                    && attachment.store != metal_api_core::provider::StoreOp::Resident;
                let view = if landing_needed || loading {
                    Some(declared.ok_or_else(|| {
                        refusal(
                            ProviderPhase::Resolve,
                            ProviderErrorClass::Capability,
                            "render_attachment_landing_unsupported",
                        )
                        .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
                        .with_field(
                            "allocation",
                            FieldValue::Unsigned(attachment.allocation_id.get()),
                        )
                        .with_detail(
                            "attachment bytes land through the buffer writeback channel and \
                             `LoadOp::Load` uploads the trace's own bytes, and this trace \
                             declares no buffer view covering the attachment",
                        )
                    })?)
                } else {
                    None
                };
                previous.push(view.filter(|_| loading));
                views.push(view);
            }
            // The stored depth attachment's landing view, resolved before the
            // pass runs for the same reason the colour ones are: the bytes it
            // receives have to be named by the trace, and a storing surface
            // without a declaration is refused instead of executed
            // (`research/docs/23` §3.3, v43).
            let depth_view = match planned.pass.depth.as_ref() {
                Some(depth) => match (depth.store, depth.identity) {
                    (Some(metal_api_core::provider::DepthStoreOp::Store), Some(identity)) => {
                        if !host_readback {
                            // A trace that publishes no readback keeps its depth
                            // texels on the device, exactly as a colour
                            // attachment does, and needs no landing view.
                            None
                        } else {
                            Some(
                                pool.iter()
                                    .find(|view| {
                                        view.view_id == identity.view_id
                                            && view.allocation_id == identity.allocation_id
                                    })
                                    .ok_or_else(|| {
                                        refusal(
                                    ProviderPhase::Resolve,
                                    ProviderErrorClass::Capability,
                                    "render_depth_landing_unsupported",
                                )
                                .with_field(
                                    "view",
                                    FieldValue::Unsigned(identity.view_id.get()),
                                )
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
                            )
                        }
                    }
                    _ => None,
                },
                None => None,
            };
            // The stored stencil attachment's landing view, resolved the same
            // way the depth one is (`research/docs/23` §3.3, v49): the bytes it
            // receives have to be named by the trace, and a storing surface
            // without a declaration is refused instead of executed.
            let stencil_view = match planned.pass.stencil.as_ref() {
                Some(stencil) => match (stencil.store, stencil.identity) {
                    (Some(metal_api_core::provider::StoreOp::Store), Some(identity)) => {
                        if !host_readback {
                            // A trace that publishes no readback keeps its
                            // stencil texels on the device, exactly as a colour
                            // or depth landing does, and needs no landing view.
                            None
                        } else {
                            Some(
                                pool.iter()
                                    .find(|view| {
                                        view.view_id == identity.view_id
                                            && view.allocation_id == identity.allocation_id
                                    })
                                    .ok_or_else(|| {
                                        refusal(
                                    ProviderPhase::Resolve,
                                    ProviderErrorClass::Capability,
                                    "render_stencil_landing_unsupported",
                                )
                                .with_field(
                                    "view",
                                    FieldValue::Unsigned(identity.view_id.get()),
                                )
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
                            )
                        }
                    }
                    _ => None,
                },
                None => None,
            };
            let executor = self.lock_executor()?;
            // The resident slice the rail borrows for this pass, in location
            // order: `Some` exactly for the attachments whose declaration
            // named the provider's image (`research/docs/23` §76, R7).
            let resident_refs: Vec<_> = resident.iter().map(|image| image.as_deref()).collect();
            let outcome = match trace.indirect.as_deref() {
                Some(payload) => {
                    let outcome = render::execute_indirect_render_pass(
                        &executor.context,
                        &planned.stages,
                        &planned.pass,
                        &payload.command,
                        &previous,
                        &resident_refs,
                        Some(&leases),
                    );
                    if outcome.is_ok() {
                        // Publish what was actually replayed: the command kind,
                        // the range and the one command the first increment
                        // encodes (`research/docs/25` §5.1).
                        self.publish_icb_observation(
                            payload.command.kind(),
                            payload.range.start,
                            payload.range.count,
                            1,
                        );
                    }
                    outcome
                }
                None => render::execute_render_pass(
                    &executor.context,
                    &planned.stages,
                    &planned.pass,
                    &previous,
                    &resident_refs,
                    Some(&leases),
                ),
            };
            let readback = match outcome {
                Ok(readback) => {
                    // The pass completed, so the bytes the resident targets
                    // hold are the ones this pass left there: a later
                    // `LoadOp::Resident` for those identities resolves instead
                    // of being refused as undefined.
                    self.note_resident_targets(&resident_identities, true)?;
                    readback
                }
                Err(error) => {
                    // A pass that was refused or failed defines nothing: the
                    // identities it named stay unloadable until a later pass
                    // renders them again, rather than serving bytes of unknown
                    // state (`research/docs/23` §76, R7).
                    self.note_resident_targets(&resident_identities, false)?;
                    return Err(error);
                }
            };
            for (view, texels) in views.into_iter().zip(readback.attachments) {
                // `None` is the discarded attachment: no bytes, no writeback,
                // whatever the view resolution above produced (`docs/23`
                // §3.6, v19).
                let Some(bytes) = texels else { continue };
                if let Some(view) = view {
                    writebacks.push(BufferWriteback {
                        view_id: view.view_id,
                        allocation_id: view.allocation_id,
                        offset: view.offset,
                        bytes,
                    });
                }
            }
            // A writable stage buffer is a landing like a stored attachment
            // (`research/docs/23` §3.3, v86): one complete writeback for the
            // view the trace declared, in the same byte-keyed channel and in
            // the pass's canonical binding order. The bytes are only published
            // when the trace asked for a host readback — the same rule the
            // attachment landings follow, and the reason the rail reads them
            // unconditionally is that the read is a mapping the bytes already
            // live in.
            if host_readback {
                for landing in readback.stage_buffers {
                    let view = planned.pass.stage_buffers.iter().find(|stage| {
                        stage.stage == landing.stage && stage.view.metal_binding == landing.index
                    });
                    // The pair rules held every readable or writable stage
                    // buffer to a declaration, so a landing without its view
                    // is a rail state the contract cannot produce; skipping it
                    // would publish a partial set of writebacks, which is why
                    // the lookup is an expectation rather than a filter.
                    let view = view.expect("every stage buffer landing has its own view");
                    writebacks.push(BufferWriteback {
                        view_id: view.view.view_id,
                        allocation_id: view.view.allocation_id,
                        offset: view.view.offset,
                        bytes: landing.bytes,
                    });
                }
            }
            // The depth landing follows the colour ones, in the same channel
            // and in the same (allocation, view) order the writeback contract
            // states (`research/docs/23` §3.3, v43).
            if let (Some(view), Some(texels)) = (depth_view, readback.depth) {
                writebacks.push(BufferWriteback {
                    view_id: view.view_id,
                    allocation_id: view.allocation_id,
                    offset: view.offset,
                    bytes: texels,
                });
            }
            // The stencil landing follows the depth one, one byte per texel,
            // in the same channel and the same (allocation, view) order the
            // writeback contract states (`research/docs/23` §3.3, v49).
            if let (Some(view), Some(texels)) = (stencil_view, readback.stencil) {
                writebacks.push(BufferWriteback {
                    view_id: view.view_id,
                    allocation_id: view.allocation_id,
                    offset: view.offset,
                    bytes: texels,
                });
            }
        }
        Ok(writebacks)
    }

    /// Resolve the provider-owned present target for one present descriptor,
    /// creating it (and, for a `Sentinel` initial state, pre-filling it) on
    /// first use. The image is keyed by the target's allocation/view identity
    /// and reused across submissions, so it survives the pass's own drop scope
    /// (`docs/24` §5.2).
    ///
    /// The registry is bounded: an insert that takes it past
    /// [`PRESENT_TARGET_BUDGET`] evicts the least recently used entries, and
    /// every use re-stamps its identity, so the victim is the identity a
    /// process least recently presented rather than the lowest key. The image
    /// is an `Arc`, so a submission that already holds one keeps it alive
    /// until its own pass returns — the same lifetime the layout lock
    /// serializes on.
    fn present_target(
        &self,
        present: &PresentDescriptor,
        attachment: &RenderAttachment,
    ) -> Result<Arc<render::ProviderTargetImage>, ProviderError> {
        let key = (present.target.allocation_id, present.target.view_id);
        {
            let mut registry = self
                .present_targets
                .lock()
                .map_err(|_| registry_poisoned())?;
            if let Some(entry) = registry.get_mut(&key) {
                entry.last_used = self.next_present_target_stamp();
                return Ok(Arc::clone(&entry.image));
            }
        }
        // The image is created outside the registry lock: it runs device calls
        // and a sentinel pre-fill, and the registry is only the admission
        // point that decides which identities stay resident.
        let mut image = render::ProviderTargetImage::create(
            Arc::clone(&self.lock_executor()?.context),
            attachment.format,
            attachment.width,
            attachment.height,
        )?;
        if let Some(sentinel) = present.target.initial.sentinel() {
            image.preset_sentinel(attachment.format, sentinel)?;
        }
        let image = Arc::new(image);
        let mut registry = self
            .present_targets
            .lock()
            .map_err(|_| registry_poisoned())?;
        let stamp = self.next_present_target_stamp();
        if let Some(entry) = registry.get_mut(&key) {
            // Two submissions raced to create the same identity: the first
            // insert stays authoritative and this one's image is dropped at
            // the end of the call, once no submission can still hold it.
            entry.last_used = stamp;
            return Ok(Arc::clone(&entry.image));
        }
        registry.insert(
            key,
            PresentTargetEntry {
                image: Arc::clone(&image),
                last_used: stamp,
            },
        );
        // The budget is a provider-internal policy, so the eviction happens
        // here rather than in the contract: the targets themselves are dropped
        // after the registry lock is released, so their images' device teardown
        // cannot run under it.
        let mut retired = Vec::new();
        while registry.len() > PRESENT_TARGET_BUDGET {
            let victim = registry
                .iter()
                .filter(|(identity, _)| **identity != key)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(identity, _)| *identity);
            match victim.and_then(|victim| registry.remove(&victim)) {
                Some(entry) => retired.push(entry),
                None => break,
            }
        }
        drop(registry);
        if !retired.is_empty() {
            self.present_target_evictions
                .fetch_add(retired.len() as u64, Ordering::Relaxed);
        }
        drop(retired);
        Ok(image)
    }

    /// The next least-recently-used stamp; only the registry lock's holders
    /// call it, so the order is the lock's serial order.
    fn next_present_target_stamp(&self) -> u64 {
        self.present_target_stamp.fetch_add(1, Ordering::Relaxed)
    }

    /// Resolve the provider-resident target of one attachment that declares it,
    /// creating the identity on a resident store
    /// (`research/docs/23` §76, R7).
    ///
    /// `loading` is the trace's own decision: a `LoadOp::Resident` attachment
    /// keeps the image's contents, so the identity has to exist *and* hold
    /// bytes a completed pass defined. Every way it can fail is refused by name
    /// rather than served from a fresh image or from bytes no pass defined:
    ///
    /// - a resident load of an identity no pass ever stored is
    ///   `resident_target_unavailable`;
    /// - one the budget evicted, a lease release retired, or an epoch advance
    ///   cleared states that rule (`resident_target_evicted`,
    ///   `resident_target_released`, `resident_target_stale`);
    /// - one whose creating pass never completed is
    ///   `resident_target_undefined`;
    /// - one whose shape changed is `resident_target_shape_changed`.
    fn resident_target(
        &self,
        attachment: &RenderAttachment,
        loading: bool,
    ) -> Result<Arc<render::ProviderTargetImage>, ProviderError> {
        let key = (attachment.allocation_id, attachment.view_id);
        {
            let mut registry = self
                .resident_targets
                .lock()
                .map_err(|_| registry_poisoned())?;
            if let Some(entry) = registry.get_mut(&key) {
                if entry.format != attachment.format
                    || entry.width != attachment.width
                    || entry.height != attachment.height
                {
                    return Err(refusal(
                        ProviderPhase::Resolve,
                        ProviderErrorClass::Capability,
                        "resident_target_shape_changed",
                    )
                    .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
                    .with_field(
                        "allocation",
                        FieldValue::Unsigned(attachment.allocation_id.get()),
                    )
                    .with_field(
                        "format",
                        FieldValue::Unsigned(u64::from(attachment.format.code())),
                    )
                    .with_field("width", FieldValue::Unsigned(attachment.width))
                    .with_field("height", FieldValue::Unsigned(attachment.height))
                    .with_field(
                        "expected_format",
                        FieldValue::Unsigned(u64::from(entry.format.code())),
                    )
                    .with_field("expected_width", FieldValue::Unsigned(entry.width))
                    .with_field("expected_height", FieldValue::Unsigned(entry.height))
                    .with_detail(
                        "the resident target's identity is reused for one image, so a pass that \
                         declares a different shape for it is refused instead of rendered into \
                         an image of the wrong format or extent",
                    ));
                }
                if loading && !entry.defined {
                    return Err(refusal(
                        ProviderPhase::Resolve,
                        ProviderErrorClass::Capability,
                        "resident_target_undefined",
                    )
                    .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
                    .with_field(
                        "allocation",
                        FieldValue::Unsigned(attachment.allocation_id.get()),
                    )
                    .with_detail(
                        "the resident target's image exists but no completed pass has defined \
                         its bytes: the pass that created it was refused or failed",
                    ));
                }
                entry.last_used = self.next_resident_target_stamp();
                return Ok(Arc::clone(&entry.image));
            }
            if loading {
                // The identity is gone. The tombstone names the rule that
                // retired it; an identity with no tombstone was never stored in
                // this epoch at all.
                let tombstones = self
                    .resident_target_tombstones
                    .lock()
                    .map_err(|_| registry_poisoned())?;
                let retirement = tombstones.get(&key).copied();
                let mut error = refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Capability,
                    retirement.map_or("resident_target_unavailable", |rule| rule.slug()),
                )
                .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
                .with_field(
                    "allocation",
                    FieldValue::Unsigned(attachment.allocation_id.get()),
                );
                if let Some(rule) = retirement {
                    error =
                        error.with_field("retired_by", FieldValue::Text(rule.name().to_owned()));
                }
                return Err(error.with_detail(
                    "a `LoadOp::Resident` attachment keeps the bytes the provider holds for its \
                     identity; this identity holds none, so the pass has to render the frame \
                     again instead of reading an image that is gone",
                ));
            }
        }
        // The image is created outside the registry lock: it runs device calls,
        // and the registry is only the admission point that decides which
        // identities stay resident.
        let image = Arc::new(render::ProviderTargetImage::create(
            Arc::clone(&self.lock_executor()?.context),
            attachment.format,
            attachment.width,
            attachment.height,
        )?);
        let mut registry = self
            .resident_targets
            .lock()
            .map_err(|_| registry_poisoned())?;
        let stamp = self.next_resident_target_stamp();
        if let Some(entry) = registry.get_mut(&key) {
            // Two submissions raced to create the same identity: the first
            // insert stays authoritative and this one's image is dropped at the
            // end of the call, once no submission can still hold it.
            entry.last_used = stamp;
            return Ok(Arc::clone(&entry.image));
        }
        registry.insert(
            key,
            ResidentTargetEntry {
                image: Arc::clone(&image),
                format: attachment.format,
                width: attachment.width,
                height: attachment.height,
                // The pass that creates the identity has not run yet, so its
                // bytes are not defined until it completes.
                defined: false,
                last_used: stamp,
            },
        );
        // The budget is a provider-internal policy, exactly as it is for the
        // present registry: the identities beyond it are retired after the
        // registry lock is released, so their images' device teardown cannot
        // run under it, and each one leaves a tombstone a later resident load
        // is refused by name with.
        let mut retired = Vec::new();
        while registry.len() > RESIDENT_TARGET_BUDGET {
            let Some(victim) = registry
                .iter()
                .filter(|(identity, _)| **identity != key)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(identity, _)| *identity)
            else {
                break;
            };
            if let Some(entry) = registry.remove(&victim) {
                retired.push((victim, entry));
            }
        }
        drop(registry);
        self.retire_resident_targets(&retired, ResidentTargetRetirement::Budget)?;
        drop(retired);
        // A stored identity is alive again, so its tombstone goes with it.
        self.resident_target_tombstones
            .lock()
            .map_err(|_| registry_poisoned())?
            .remove(&key);
        Ok(image)
    }

    /// Retire resident target identities and record the rule that retired them
    /// (`research/docs/23` §76, R7).
    ///
    /// The tombstone is what makes a later `LoadOp::Resident` refusal nameable,
    /// and the counter follows the present registry's rule: budget evictions and
    /// lease-release retirements are counted, the epoch advance that clears the
    /// whole registry is not — it is observable through the epoch itself.
    fn retire_resident_targets(
        &self,
        retired: &[((AllocationId, ViewId), ResidentTargetEntry)],
        rule: ResidentTargetRetirement,
    ) -> Result<(), ProviderError> {
        if retired.is_empty() {
            return Ok(());
        }
        {
            let mut tombstones = self
                .resident_target_tombstones
                .lock()
                .map_err(|_| registry_poisoned())?;
            for (identity, _) in retired {
                tombstones.insert(*identity, rule);
            }
        }
        if rule != ResidentTargetRetirement::EpochAdvance {
            self.resident_target_evictions
                .fetch_add(retired.len() as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    /// The next least-recently-used stamp of the resident registry; only that
    /// registry's lock holders call it, so the order is the lock's serial
    /// order.
    fn next_resident_target_stamp(&self) -> u64 {
        self.resident_target_stamp.fetch_add(1, Ordering::Relaxed)
    }

    /// Whether the provider currently holds a resident target for one
    /// attachment identity (`research/docs/23` §76, R7).
    ///
    /// A pass that renders into an identity the provider holds has to declare
    /// it — a resident load or a resident store — so this is the question
    /// behind `resident_target_undeclared`: the provider's bytes are not
    /// silently overwritten by a pass that never named them.
    fn resident_target_identity_is_resident(
        &self,
        allocation_id: AllocationId,
        view_id: ViewId,
    ) -> bool {
        self.resident_targets
            .lock()
            .map(|registry| registry.contains_key(&(allocation_id, view_id)))
            .unwrap_or(false)
    }

    /// State whether the bytes of the identities a pass declared are defined
    /// now that the pass has completed (`research/docs/23` §76, R7).
    ///
    /// A completed pass defines them: it cleared the image, kept contents a
    /// previous pass defined, or stored the raster into it. A pass that was
    /// refused or failed defines nothing, and its identity stays unloadable
    /// until a later pass renders it again.
    fn note_resident_targets(
        &self,
        identities: &[(AllocationId, ViewId)],
        defined: bool,
    ) -> Result<(), ProviderError> {
        if identities.is_empty() {
            return Ok(());
        }
        let mut registry = self
            .resident_targets
            .lock()
            .map_err(|_| registry_poisoned())?;
        for identity in identities {
            if let Some(entry) = registry.get_mut(identity) {
                entry.defined = defined;
            }
        }
        Ok(())
    }

    /// Drop any present target reserved for the allocation of a released
    /// lease.
    ///
    /// A present target lives across submissions until its allocation's lease
    /// is released (`docs/24` §5.2), so releasing the lease has to retire the
    /// target too — the same rule the native rail states in its own
    /// `drop_provider_targets_for`. The lease→allocation mapping is removed in
    /// the same call, so a double release cannot retire a sibling
    /// allocation's targets.
    fn drop_provider_targets_for(&self, lease_id: LeaseId) -> Result<(), ProviderError> {
        let allocation_id = self
            .lease_allocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&lease_id);
        let Some(allocation_id) = allocation_id else {
            return Ok(());
        };
        let retired = {
            let mut registry = self
                .present_targets
                .lock()
                .map_err(|_| registry_poisoned())?;
            let before = registry.len();
            registry.retain(|(allocation, _), _| *allocation != allocation_id);
            before - registry.len()
        };
        if retired > 0 {
            self.present_target_evictions
                .fetch_add(retired as u64, Ordering::Relaxed);
        }
        // The resident registry is retired by the same rule and for the same
        // reason (`research/docs/23` §76, R7): the images a released allocation
        // backed are gone, so a later `LoadOp::Resident` names the release
        // (`resident_target_released`) instead of reading an image whose
        // backing the owner has taken back.
        let retired_resident = {
            let mut registry = self
                .resident_targets
                .lock()
                .map_err(|_| registry_poisoned())?;
            let identities: Vec<_> = registry
                .keys()
                .filter(|(allocation, _)| *allocation == allocation_id)
                .copied()
                .collect();
            identities
                .into_iter()
                .filter_map(|identity| registry.remove(&identity).map(|entry| (identity, entry)))
                .collect::<Vec<_>>()
        };
        self.retire_resident_targets(&retired_resident, ResidentTargetRetirement::LeaseReleased)?;
        Ok(())
    }

    /// Number of provider-owned present targets still alive. Exposed for the
    /// cross-submission test that proves a target is reused, not recreated,
    /// across two presents of the same allocation/view.
    #[doc(hidden)]
    pub fn present_target_count(&self) -> usize {
        self.present_targets
            .lock()
            .map(|registry| registry.len())
            .unwrap_or(0)
    }

    /// Cumulative number of present targets retired before the device epoch
    /// ended, by either normal-path surface: the budget's eviction or the
    /// release of the lease an allocation was reserved under. The device-loss
    /// teardown that clears the registry is not counted here; it is observable
    /// through the epoch advance instead.
    #[doc(hidden)]
    pub fn present_target_evictions(&self) -> u64 {
        self.present_target_evictions.load(Ordering::Relaxed)
    }

    /// Number of provider-resident render targets still alive
    /// (`research/docs/23` §76, R7).
    ///
    /// The observation the resident chain's own test reads: two submissions
    /// that name the same identity keep one image, and a submission that names
    /// a new one adds exactly one entry until the budget evicts.
    #[doc(hidden)]
    pub fn resident_target_count(&self) -> usize {
        self.resident_targets
            .lock()
            .map(|registry| registry.len())
            .unwrap_or(0)
    }

    /// Cumulative resident targets retired before the device epoch ended: the
    /// budget's evictions plus the retirements a lease release drives
    /// (`research/docs/23` §76, R7). The device-loss teardown that clears the
    /// registry is not counted here, exactly as the present registry's
    /// counter states; the epoch advance is its observation.
    #[doc(hidden)]
    pub fn resident_target_evictions(&self) -> u64 {
        self.resident_target_evictions.load(Ordering::Relaxed)
    }

    /// Whether one `(allocation, view)` identity is resident, i.e. whether a
    /// later `LoadOp::Resident` for it resolves instead of being refused
    /// (`research/docs/23` §76, R7).
    #[doc(hidden)]
    pub fn resident_target_is_live(&self, allocation_id: AllocationId, view_id: ViewId) -> bool {
        self.resident_target_identity_is_resident(allocation_id, view_id)
    }

    /// Cumulative present acquire / present completions of the presentation
    /// rail. Smoke tests use it to prove a presenting case reports one of each.
    #[doc(hidden)]
    pub fn present_counts(&self) -> (usize, usize) {
        self.lock_executor()
            .expect("executor lock poisoned")
            .present_counts()
    }

    fn retire(&self, pending: PendingExecution) {
        let mut slot = match self.retire_tx.lock() {
            Ok(slot) => slot,
            Err(poisoned) => poisoned.into_inner(),
        };
        let sender = slot.get_or_insert_with(|| {
            let (tx, rx) = mpsc::channel::<PendingExecution>();
            let context = Arc::clone(
                &self
                    .lock_executor()
                    .expect("executor lock poisoned")
                    .context,
            );
            let outbox = self.completion_outbox.clone();
            let _ = std::thread::Builder::new()
                .name("vulkan-provider-retire".into())
                .spawn(move || {
                    while let Ok(mut pending) = rx.recv() {
                        match pending.wait(crate::FENCE_TIMEOUT_NS) {
                            Ok(true) => {}
                            Ok(false) => {
                                if context.record_abandonment(pending.owned_bytes())
                                    == AbandonmentOutcome::Admitted
                                {
                                    pending.retain_after_budgeted_abandon();
                                }
                            }
                            Err(error) if error.class == ProviderErrorClass::DeviceLost => {
                                // `PendingExecution::wait` already routed the
                                // driver's loss through the core lifecycle and
                                // attached the fault evidence to this error;
                                // these keep the retirement thread's own
                                // guarantee explicit for the handles it owns.
                                pending.mark_device_lost();
                                context.mark_device_lost();
                            }
                            Err(_) => {
                                if context.record_abandonment(pending.owned_bytes())
                                    == AbandonmentOutcome::Admitted
                                {
                                    pending.retain_after_budgeted_abandon();
                                }
                            }
                        }
                        if let Some(outbox) = &outbox {
                            sync_context_health(&context, outbox);
                        }
                    }
                });
            tx
        });
        let _ = sender.send(pending);
    }

    fn wait_inner(
        &self,
        token: CompletionToken,
        timeout: Duration,
    ) -> Result<CompletionDisposition, ProviderError> {
        self.validate_token(token)?;
        let (record, pending, pool, textures, render_writebacks, heap_observations, deadline) = {
            let mut completions = self.completions.lock().map_err(|_| registry_poisoned())?;
            let slot = completions
                .get_mut(&token.submission_id)
                .ok_or_else(|| unknown_completion(token))?;
            if !slot.record.is_running() {
                (
                    Arc::clone(&slot.record),
                    None,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    slot.deadline,
                )
            } else if let Some(pending) = slot.pending.take() {
                (
                    Arc::clone(&slot.record),
                    Some(pending),
                    slot.pool.clone(),
                    slot.textures.clone(),
                    slot.render_writebacks.clone(),
                    slot.heap_observations.clone(),
                    slot.deadline,
                )
            } else {
                (
                    Arc::clone(&slot.record),
                    None,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    slot.deadline,
                )
            }
        };
        let Some(mut pending) = pending else {
            return record.wait(token, timeout);
        };
        if deadline.expired() {
            self.retire(pending);
            return Err(self.fail_deadline(&record, token));
        }
        match pending.wait(duration_to_nanos(deadline.clamp(timeout))) {
            Ok(true) => match pending
                .read_updates()
                .and_then(|updates| map_writebacks(&pool, &textures, updates, token))
            {
                Ok(writebacks) => {
                    drop(pending);
                    let mut merged = BTreeMap::new();
                    for writeback in writebacks.into_iter().chain(render_writebacks) {
                        merged.insert((writeback.allocation_id, writeback.view_id), writeback);
                    }
                    record.complete(merged.into_values().collect());
                    self.publish_heap_observations(heap_observations);
                    Ok(CompletionDisposition::CompletedVisible { token })
                }
                Err(error) => {
                    drop(pending);
                    record.fail(error.clone());
                    Err(error)
                }
            },
            Ok(false) => {
                if deadline.expired() {
                    self.retire(pending);
                    return Err(self.fail_deadline(&record, token));
                }
                let mut pending = Some(pending);
                let mut completions = self.completions.lock().map_err(|_| registry_poisoned())?;
                if let Some(slot) = completions.get_mut(&token.submission_id) {
                    if slot.record.is_running() && slot.pending.is_none() {
                        slot.pending = pending.take();
                    }
                }
                drop(completions);
                if let Some(pending) = pending {
                    self.retire(pending);
                }
                Ok(CompletionDisposition::TimedOut { token })
            }
            Err(error) => {
                drop(pending);
                record.fail(error.clone());
                Err(error)
            }
        }
    }

    fn cancel_inner(&self, token: CompletionToken) -> Result<CompletionDisposition, ProviderError> {
        self.validate_token(token)?;
        let (record, pending) = {
            let mut completions = self.completions.lock().map_err(|_| registry_poisoned())?;
            let slot = completions
                .get_mut(&token.submission_id)
                .ok_or_else(|| unknown_completion(token))?;
            (Arc::clone(&slot.record), slot.pending.take())
        };
        if let Some(pending) = pending {
            self.retire(pending);
        }
        record.cancel();
        record.wait(token, Duration::ZERO)
    }

    fn terminal_record(
        &self,
        token: CompletionToken,
        writebacks: Vec<BufferWriteback>,
    ) -> Arc<CompletionRecord> {
        match &self.completion_outbox {
            Some(outbox) => {
                let _ = outbox.submitted(token);
                let record = CompletionRecord::running_with_observer(token, outbox.observer());
                record.complete(writebacks);
                record
            }
            None => CompletionRecord::completed(writebacks),
        }
    }

    fn failed_record(&self, token: CompletionToken, error: ProviderError) -> Arc<CompletionRecord> {
        match &self.completion_outbox {
            Some(outbox) => {
                let _ = outbox.submitted(token);
                let record = CompletionRecord::running_with_observer(token, outbox.observer());
                record.fail(error);
                record
            }
            None => CompletionRecord::failed(error),
        }
    }

    fn sync_completion_health(&self) {
        if let Some(outbox) = &self.completion_outbox {
            sync_context_health(
                &self
                    .lock_executor()
                    .expect("executor lock poisoned")
                    .context,
                outbox,
            );
        }
    }

    fn fail_deadline(&self, record: &CompletionRecord, token: CompletionToken) -> ProviderError {
        let error = refusal(
            ProviderPhase::Wait,
            ProviderErrorClass::Execute,
            "vulkan-completion-unknown",
        )
        .with_completion(CompletionDisposition::SubmittedUnknown { token: Some(token) });
        record.fail(error.clone());
        error
    }

    fn validate_token(&self, token: CompletionToken) -> Result<(), ProviderError> {
        token.validate().map_err(|error| {
            refusal(
                ProviderPhase::Wait,
                ProviderErrorClass::Args,
                "invalid_completion_token",
            )
            .with_detail(error.to_string())
        })?;
        check_epoch(self.device_epoch(), token.device_epoch)
    }

    fn ensure_usable(&self) -> Result<(), ProviderError> {
        ensure_context_usable(&self.lock_executor()?.context)
    }

    /// Lock the current device owner for one read or write.
    ///
    /// Rebuild swaps the owner in place, so every submission path must read the
    /// owner through this guard instead of a copy taken earlier. Poisoning is a
    /// registry failure, not a signal to keep going on a possibly dead device.
    fn lock_executor(&self) -> Result<MutexGuard<'_, Arc<VulkanExecutor>>, ProviderError> {
        self.executor.lock().map_err(|_| registry_poisoned())
    }
}

impl VulkanComputeProvider {
    /// Staged lease registry owned by this provider.
    pub fn lease_registry(&self) -> &LeaseRegistry {
        &self.staging
    }

    /// No-copy lease registry owned by this provider.
    pub fn borrowed_registry(&self) -> &Arc<BorrowedLeaseRegistry> {
        &self.borrowed
    }

    /// Map a trace's heap payload onto its owned allocations and record the
    /// placement observations (`research/docs/25-heaps与ICB设计.md` §6 Step 3).
    ///
    /// The mapping is: the trace's distinct owned allocations in ascending
    /// identity order, zipped one-to-one with `placements`. Staged and borrowed
    /// views keep their own backing and are not part of the heap, so a count
    /// or size mismatch is a typed refusal instead of a silent drop.
    fn plan_heap_placements(
        &self,
        trace: &ComputeTrace,
        pool: &[BufferView],
        resources: &ResourceTableSnapshot,
    ) -> Result<Option<HeapPlan>, ProviderError> {
        let Some(heap) = &trace.heap else {
            return Ok(None);
        };
        // The first increment binds buffers only; a texture placement has no
        // Vulkan image binding yet (`research/docs/25` §6 Step 3).
        if let Some(placement) = heap
            .placements
            .iter()
            .find(|placement| matches!(placement.resource, HeapResource::Texture { .. }))
        {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Capability,
                "heap_placement_unsupported",
            )
            .with_field("resource", FieldValue::Text("texture".to_string()))
            .with_field("heap", FieldValue::Unsigned(placement.heap_id.get())));
        }
        let mut owned = BTreeSet::<u64>::new();
        for resource in pool {
            if matches!(resource.source, BufferSource::OwnedBytes(_)) {
                owned.insert(resource.allocation_id.get());
            }
        }
        let owned: Vec<u64> = owned.into_iter().collect();
        if heap.placements.len() != owned.len() {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Args,
                "heap_placement_mismatch",
            )
            .with_field(
                "placements",
                FieldValue::Unsigned(heap.placements.len() as u64),
            )
            .with_field("allocations", FieldValue::Unsigned(owned.len() as u64)));
        }
        let mut heap_ids = BTreeSet::<u64>::new();
        let mut offsets = BTreeMap::<u64, u64>::new();
        let mut sizes = BTreeMap::<u64, u64>::new();
        let mut observations = Vec::with_capacity(owned.len());
        for (placement, allocation) in heap.placements.iter().zip(owned.iter()) {
            heap_ids.insert(placement.heap_id.get());
            let record = resources
                .allocation(AllocationId::new(*allocation))
                .ok_or_else(|| {
                    refusal(
                        ProviderPhase::Resolve,
                        ProviderErrorClass::Resource,
                        "heap_placement_mismatch",
                    )
                    .with_field("allocation", FieldValue::Unsigned(*allocation))
                })?;
            if placement.resource.byte_size() != record.size {
                return Err(refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Args,
                    "heap_placement_mismatch",
                )
                .with_field("allocation", FieldValue::Unsigned(*allocation))
                .with_field(
                    "placement_size",
                    FieldValue::Unsigned(placement.resource.byte_size()),
                )
                .with_field("allocation_size", FieldValue::Unsigned(record.size)));
            }
            offsets.insert(*allocation, placement.offset);
            sizes.insert(*allocation, record.size);
            observations.push(HeapPlacementObservation {
                heap_id: placement.heap_id,
                allocation_id: AllocationId::new(*allocation),
                offset: placement.offset,
                byte_size: placement.resource.byte_size(),
            });
        }
        if heap_ids.len() != 1 {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Args,
                "heap_placement_mismatch",
            )
            .with_field(
                "distinct_heaps",
                FieldValue::Unsigned(heap_ids.len() as u64),
            ));
        }
        Ok(Some(HeapPlan {
            slab_size: heap.descriptor.size,
            offsets,
            sizes,
            observations,
        }))
    }

    fn publish_heap_observations(&self, observations: Vec<HeapPlacementObservation>) {
        // Keep only the latest successful submission's placements: each
        // completed submission replaces the previous vector instead of
        // appending, so a long-running process never grows an unbounded log.
        *self
            .heap_observations
            .lock()
            .expect("heap observation lock poisoned") = observations;
    }
}

impl LeaseImporter for VulkanComputeProvider {
    fn import_staged_lease(&self, staged: StagedLease) -> Result<(), ProviderError> {
        if staged.reservation.lease.owner_epoch != self.device_epoch() {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Args,
                "lease_epoch_mismatch",
            )
            .with_field("expected", FieldValue::Unsigned(self.device_epoch().get()))
            .with_field(
                "actual",
                FieldValue::Unsigned(staged.reservation.lease.owner_epoch.get()),
            ));
        }
        let lease_id = staged.lease_id();
        let allocation_id = staged.reservation.lease.allocation_id;
        self.staging.import(staged)?;
        self.lease_allocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(lease_id, allocation_id);
        Ok(())
    }

    fn release_staged_lease(&self, lease_id: LeaseId) -> Result<(), ProviderError> {
        self.staging.release(lease_id)?;
        self.drop_provider_targets_for(lease_id)
    }
}

impl NoCopyLeaseImporter for VulkanComputeProvider {
    fn no_copy_alignment(&self) -> u64 {
        self.lock_executor()
            .expect("executor lock poisoned")
            .context
            .external_memory_host_alignment()
    }

    unsafe fn import_borrowed_lease(&self, borrowed: BorrowedLease) -> Result<(), ProviderError> {
        if borrowed.reservation.lease.owner_epoch != self.device_epoch() {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Args,
                "lease_epoch_mismatch",
            )
            .with_field("expected", FieldValue::Unsigned(self.device_epoch().get()))
            .with_field(
                "actual",
                FieldValue::Unsigned(borrowed.reservation.lease.owner_epoch.get()),
            ));
        }
        let alignment = self.no_copy_alignment();
        if alignment == 0 {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Capability,
                "storage_mode_unsupported",
            ));
        }
        let alignment = usize::try_from(alignment).unwrap_or(usize::MAX);
        if !borrowed.host_pointer.is_multiple_of(alignment) {
            return Err(borrowed_alignment_error(
                borrowed.lease_id(),
                borrowed.host_pointer,
                alignment as u64,
            ));
        }
        let lease_id = borrowed.lease_id();
        let allocation_id = borrowed.reservation.lease.allocation_id;
        self.borrowed.import(borrowed)?;
        self.lease_allocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(lease_id, allocation_id);
        Ok(())
    }

    fn release_borrowed_lease(&self, lease_id: LeaseId) -> Result<(), ProviderError> {
        self.borrowed.release(lease_id)?;
        self.drop_provider_targets_for(lease_id)
    }
}

impl PipelineProvider for VulkanComputeProvider {
    fn device_epoch(&self) -> DeviceEpoch {
        DeviceEpoch::new(self.epoch.load(Ordering::Relaxed))
    }

    fn compile(
        &self,
        request: PipelineCompileRequest,
    ) -> Result<CompiledComputePipeline, ProviderError> {
        request.validate().map_err(|error| {
            refusal(
                ProviderPhase::Compile,
                ProviderErrorClass::Args,
                "invalid_compile_request",
            )
            .with_detail(error.to_string())
        })?;
        let device = Device::new(self.lock_executor()?.clone());
        let library = match request.source {
            ShaderSource::SanitizedLl(source) => device.new_library_with_air(source),
            ShaderSource::BinaryAir(bytes) => device.new_library_with_binary_air(bytes),
            ShaderSource::MetalSource(_) => {
                return Err(refusal(
                    ProviderPhase::Compile,
                    ProviderErrorClass::Capability,
                    "shader_source_unsupported",
                ))
            }
        }
        .map_err(|error| {
            refusal(
                ProviderPhase::Compile,
                ProviderErrorClass::Args,
                "invalid_library_source",
            )
            .with_detail(error.to_string())
        })?;
        let function = library.function(request.entry_name).map_err(|error| {
            refusal(
                ProviderPhase::Compile,
                ProviderErrorClass::Args,
                "invalid_compile_request",
            )
            .with_detail(error.to_string())
        })?;
        self.compile_pipeline(&function, request.logical_digest)
    }

    fn release_pipeline(&self, metadata: &CompiledComputePipeline) -> Result<(), ProviderError> {
        check_epoch(self.device_epoch(), metadata.device_epoch)?;
        let mut pipelines = self.pipelines.lock().map_err(|_| registry_poisoned())?;
        let registered = pipelines
            .get(&metadata.pipeline_id)
            .ok_or_else(|| unknown_pipeline(metadata.pipeline_id))?;
        if registered.metadata != *metadata {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Resource,
                "pipeline_identity_mismatch",
            ));
        }
        pipelines.remove(&metadata.pipeline_id);
        Ok(())
    }

    fn release_completion(&self, token: CompletionToken) -> Result<(), ProviderError> {
        VulkanComputeProvider::release_completion(self, token)
    }
}

impl ComputeProvider for VulkanComputeProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities
            .lock()
            .expect("capability lock poisoned")
            .clone()
    }

    fn health(&self) -> ProviderHealth {
        VulkanComputeProvider::health(self)
    }

    /// Install an owner queue-priority marking on the device queues.
    ///
    /// The marking is expanded to one tier per device queue
    /// ([`queue_priorities_for_device`]) and the installed table is returned, so
    /// an owner that sent the marking over the command channel gets the same
    /// view a process-local caller reads from
    /// [`VulkanExecutor::queue_priorities`]. Nothing else about submission
    /// changes: the tier table is read by every queue selection and by nothing
    /// else.
    fn set_queue_priorities(
        &self,
        tiers: &[QueuePriority],
    ) -> Result<Vec<QueuePriority>, ProviderError> {
        let executor = self.lock_executor()?;
        let installed = queue_priorities_for_device(executor.queue_count(), tiers);
        executor.set_queue_priorities(&installed).map_err(|error| {
            refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Internal,
                "queue_priorities_refused",
            )
            .with_detail(error.to_string())
        })?;
        Ok(installed)
    }

    fn submit(&self, admitted: ValidatedComputeTrace) -> Result<ProviderSubmission, ProviderError> {
        let trace = admitted.trace();
        check_epoch(self.device_epoch(), trace.device_epoch)?;
        // A ValidatedComputeTrace may have been admitted against another
        // capability snapshot. Only the receiving owner can authorize execution.
        self.capabilities
            .lock()
            .map_err(|_| registry_poisoned())?
            .admit(trace, admitted.resources())?;
        // The indirect dispatch the compute rail replays, resolved and shape
        // checked before any compute resource exists. The render rail owns the
        // draw half; `None` here means the compute sequence dispatches directly.
        let indirect_dispatch = self.indirect_dispatch_threadgroups(trace)?;
        // Render work is planned before the compute sequence runs, so a trace
        // this provider cannot execute end to end is refused with no execution
        // at all rather than after its compute passes already wrote bytes.
        let render_plan = self.plan_render_passes(trace)?;
        let artifacts = {
            let registry = self.pipelines.lock().map_err(|_| registry_poisoned())?;
            trace
                .compute_passes()
                .map(|pass| {
                    let requested = trace.pipeline(pass.pipeline).map_err(|error| {
                        refusal(
                            ProviderPhase::Resolve,
                            ProviderErrorClass::Resource,
                            "pipeline_identity_mismatch",
                        )
                        .with_detail(error.to_string())
                    })?;
                    let registered = registry
                        .get(&pass.pipeline)
                        .ok_or_else(|| unknown_pipeline(pass.pipeline))?;
                    // A texture declaration states the sampler its module was
                    // lowered against (`research/docs/26` §21.3, C1b). The
                    // state the registered module carries is the declaration's
                    // own source, so a request that names a different state is
                    // refused by binding and both halves before the identity
                    // walk reports it as a generic contract mismatch.
                    refuse_foreign_texture_samplers(requested, &registered.metadata)?;
                    validate_pipeline_identity(requested, &registered.metadata)?;
                    Ok(registered.artifact.clone())
                })
                .collect::<Result<Vec<_>, ProviderError>>()?
        };
        let pool = trace.serial_resources().map_err(|error| {
            refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Resource,
                "resource_contract_invalid",
            )
            .with_detail(error.to_string())
        })?;
        let heap_plan = self.plan_heap_placements(trace, &pool, admitted.resources())?;
        let mut dispatches = Vec::with_capacity(trace.passes.len());
        for pass in trace.compute_passes() {
            let grid = narrow_dimensions(pass.dispatch.grid)?.dimensions();
            let local = narrow_dimensions(pass.dispatch.threads_per_threadgroup)?.dimensions();
            let mut bindings = pass
                .buffers
                .iter()
                .map(|view| {
                    let position = pool
                        .iter()
                        .position(|resource| resource.view_id == view.view_id)
                        .expect("validated resource pool");
                    Binding {
                        metal_index: view.metal_binding,
                        key: PoolKey::buffer(position as u32),
                        width: usize::try_from(view.length).unwrap_or(usize::MAX),
                    }
                })
                .collect::<Vec<_>>();
            // Sampled textures share the Metal argument index space with
            // buffers, so they join the same binding map with a texture pool
            // key (`research/docs/16` §4.7).
            for texture in &pass.textures {
                let width =
                    usize::try_from(texture.expected_bytes().unwrap_or(0)).unwrap_or(usize::MAX);
                bindings.push(Binding {
                    metal_index: texture.metal_binding,
                    key: PoolKey {
                        kind: PoolKind::Texture,
                        index: texture.metal_binding,
                    },
                    width,
                });
            }
            dispatches.push(BoundDispatch {
                grid,
                local,
                bindings,
            });
        }
        let token = CompletionToken {
            submission_id: SubmissionId::new(next_identity(
                &self.next_submission,
                "submission_identity_exhausted",
            )?),
            device_epoch: self.device_epoch(),
        };
        let alignment = self.no_copy_alignment();
        // Owned views of one allocation share a single device buffer, so the
        // backing is created once and the readback is sliced per view
        // (`research/docs/15` §3). Only allocations with more than one owned
        // view are shared; a lone owned view keeps its exact-length buffer, and
        // staged or borrowed views keep their own binding because their backing
        // is owned elsewhere. Each view carries its own snapshot bytes and they
        // are copied in at the view's own offset; a view that cannot read
        // copies nothing in, which step 4 of `research/docs/15` measures.
        let overflow = || {
            refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Resource,
                "buffer_range_overflow",
            )
        };
        let mut owned_per_allocation = BTreeMap::<AllocationId, usize>::new();
        for resource in pool.iter() {
            if matches!(resource.source, BufferSource::OwnedBytes(_)) {
                *owned_per_allocation
                    .entry(resource.allocation_id)
                    .or_default() += 1;
            }
        }
        let mut buffers = Vec::with_capacity(pool.len());
        let mut borrowed_leases = Vec::new();
        for (position, resource) in pool.iter().enumerate() {
            // The validated pool has at most 64 resources. First-use Metal
            // binding labels may repeat across different passes.
            let index = position as u32;
            match &resource.source {
                BufferSource::OwnedBytes(bytes) => {
                    // A heap-bearing trace binds every owned allocation into
                    // the heap slab instead of its own device memory. The
                    // allocation's buffer is bound at its placement offset and
                    // the view still addresses its own window inside it
                    // (`research/docs/25` §6 Step 3).
                    if let Some(plan) = &heap_plan {
                        if let Some(heap_offset) = plan.offsets.get(&resource.allocation_id.get()) {
                            let allocation_size = plan
                                .sizes
                                .get(&resource.allocation_id.get())
                                .copied()
                                .ok_or_else(&overflow)?;
                            buffers.push(PoolBinding::HeapOwned {
                                index,
                                allocation: resource.allocation_id.get(),
                                offset: usize::try_from(resource.offset).map_err(|_| overflow())?,
                                length: usize::try_from(resource.length).map_err(|_| overflow())?,
                                access: resource.access,
                                bytes: bytes.clone(),
                                allocation_size: usize::try_from(allocation_size)
                                    .map_err(|_| overflow())?,
                                heap_offset: usize::try_from(*heap_offset)
                                    .map_err(|_| overflow())?,
                                heap_size: usize::try_from(plan.slab_size)
                                    .map_err(|_| overflow())?,
                            });
                            continue;
                        }
                    }
                    // A lone owned view keeps its exact-length buffer; only a
                    // repeated allocation shares one backing across its views.
                    if owned_per_allocation
                        .get(&resource.allocation_id)
                        .copied()
                        .unwrap_or(0)
                        < 2
                    {
                        buffers.push(PoolBinding::Owned(BufferBinding {
                            index,
                            bytes: bytes.clone(),
                        }));
                        continue;
                    }
                    buffers.push(PoolBinding::SharedOwned {
                        index,
                        allocation: resource.allocation_id.get(),
                        offset: usize::try_from(resource.offset).map_err(|_| overflow())?,
                        length: usize::try_from(resource.length).map_err(|_| overflow())?,
                        // A view that cannot read uploads nothing: its snapshot
                        // bytes are never observable. Every other view copies
                        // in exactly its own bytes (`research/docs/15` step 4).
                        access: resource.access,
                        bytes: bytes.clone(),
                    });
                }
                BufferSource::StagedLease(lease_id) => {
                    let bytes = self.staging.view_bytes(
                        *lease_id,
                        resource,
                        self.device_epoch(),
                        admitted.resources(),
                    )?;
                    buffers.push(PoolBinding::Owned(BufferBinding { index, bytes }));
                }
                BufferSource::BorrowedNoCopy(lease_id) => {
                    if alignment == 0 {
                        return Err(refusal(
                            ProviderPhase::Resolve,
                            ProviderErrorClass::Capability,
                            "storage_mode_unsupported",
                        ));
                    }
                    let view = self.borrowed.view_pointer(
                        *lease_id,
                        resource,
                        self.device_epoch(),
                        admitted.resources(),
                    )?;
                    let alignment = usize::try_from(alignment).unwrap_or(usize::MAX);
                    if !view.pointer.is_multiple_of(alignment) {
                        return Err(borrowed_alignment_error(
                            *lease_id,
                            view.pointer,
                            alignment as u64,
                        ));
                    }
                    borrowed_leases.push(*lease_id);
                    buffers.push(PoolBinding::Imported {
                        index,
                        pointer: view.pointer,
                        len: view.len,
                        capacity: view.capacity,
                    });
                }
            }
        }
        let mut retains = BorrowedRetains::new(Arc::clone(&self.borrowed), borrowed_leases);
        retains.retain()?;
        let textures = trace.serial_texture_resources().map_err(|error| {
            refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Resource,
                "resource_contract_invalid",
            )
            .with_detail(error.to_string())
        })?;
        if self.async_execution {
            let executor = self.lock_executor()?.clone();
            let queue_index = executor.context.pick_queue();
            let pending = {
                let _execution = executor
                    .context
                    .lock_queue(queue_index)
                    .map_err(|_| registry_poisoned())?;
                ensure_executor_usable(&executor)?;
                PendingExecution::submit(
                    &executor.context,
                    queue_index,
                    &artifacts,
                    &buffers,
                    &dispatches,
                    SequenceTail {
                        borrowed: retains.take(),
                        textures: &textures,
                        indirect_dispatch,
                    },
                )?
            };
            // The indirect dispatch replay is encoded and submitted above, so
            // this is the point where its record becomes true (`docs/25` §5.1).
            if let Some(payload) = trace.indirect.as_deref() {
                self.publish_icb_observation(
                    payload.command.kind(),
                    payload.range.start,
                    payload.range.count,
                    1,
                );
            }
            // The render rail completes inside `submit` even in deferred mode:
            // each render/present pass executes here, so a present tail's
            // acquire/present counters and the target's terminal layout are
            // advanced before `submit` returns. `wait` only decides whether
            // those writebacks become host-visible; `cancel` or a deadline
            // abandons the observation and in-flight reclaim but does NOT roll
            // the present action back (`docs/24` §9 leaves a truly cancellable
            // present as a future contract). The render passes do not share the
            // compute queue: they select a graphics-capable family through
            // `render::select_graphics_queue`, so ordering against the compute
            // dispatch is guaranteed by core admission
            // (`AttachmentComputeConflict` / `RenderPassOrderUnsupported`)
            // rather than by a shared queue submission. Its bytes are merged
            // with the deferred pool readback at `wait`.
            let render_writebacks = match self.execute_render_passes(
                trace,
                &pool,
                &render_plan,
                admitted.resources(),
            ) {
                Ok(writebacks) => writebacks,
                Err(error) => {
                    self.retire(pending);
                    self.sync_completion_health();
                    return Err(attach_token(error, token));
                }
            };
            let record = match &self.completion_outbox {
                Some(outbox) => {
                    let _ = outbox.submitted(token);
                    CompletionRecord::running_with_observer(token, outbox.observer())
                }
                None => CompletionRecord::running(),
            };
            self.completions
                .lock()
                .map_err(|_| registry_poisoned())?
                .insert(
                    token.submission_id,
                    CompletionSlot {
                        record,
                        pending: Some(pending),
                        pool,
                        textures: textures.clone(),
                        render_writebacks,
                        heap_observations: heap_plan
                            .as_ref()
                            .map(|plan| plan.observations.clone())
                            .unwrap_or_default(),
                        deadline: ObservationDeadline::new(self.observation_deadline),
                    },
                );
            self.sync_completion_health();
            return Ok(ProviderSubmission {
                completion: CompletionDisposition::Submitted { token },
                writebacks: Vec::new(),
            });
        }
        let executor = self.lock_executor()?.clone();
        let result = execute_on_context(
            &executor,
            &artifacts,
            &buffers,
            &dispatches,
            &mut retains,
            &textures,
            indirect_dispatch,
        )
        .and_then(|updates| {
            // The compute dispatch -- direct or indirect -- is complete here,
            // so an indirect replay's record becomes true at this point
            // (`research/docs/25` §5.1).
            if let Some(payload) = trace.indirect.as_deref() {
                self.publish_icb_observation(
                    payload.command.kind(),
                    payload.range.start,
                    payload.range.count,
                    1,
                );
            }
            // Compute and render writebacks share one channel and one rule: one
            // complete writeback per written view, keyed by identity. A view
            // both rails could have written is refused by core admission
            // (`AttachmentComputeConflict`), and the render rail runs last, so
            // the map keeps the bytes a repeated attachment write ends with.
            let mut merged = BTreeMap::new();
            for writeback in map_writebacks(&pool, &textures, updates, token)?
                .into_iter()
                .chain(self.execute_render_passes(
                    trace,
                    &pool,
                    &render_plan,
                    admitted.resources(),
                )?)
            {
                merged.insert((writeback.allocation_id, writeback.view_id), writeback);
            }
            let writebacks: Vec<BufferWriteback> = merged.into_values().collect();
            let output = ProviderSubmission {
                completion: CompletionDisposition::CompletedVisible { token },
                writebacks,
            };
            output.validate_for_trace(trace).map_err(|error| {
                output_error(token, "writeback_contract_invalid").with_detail(error.to_string())
            })?;
            Ok(output)
        })
        .map_err(|error| attach_token(error, token));
        let observation = match &result {
            Ok(output) => Some(self.terminal_record(token, output.writebacks.clone())),
            Err(error) if error.completion.token().is_some() => {
                Some(self.failed_record(token, error.clone()))
            }
            Err(_) => None,
        };
        if result.is_ok() {
            if let Some(plan) = &heap_plan {
                self.publish_heap_observations(plan.observations.clone());
            }
        }
        if let Some(observation) = observation {
            self.completions
                .lock()
                .map_err(|_| registry_poisoned())?
                .insert(
                    token.submission_id,
                    CompletionSlot {
                        record: observation,
                        pending: None,
                        pool: Vec::new(),
                        textures: Vec::new(),
                        render_writebacks: Vec::new(),
                        heap_observations: Vec::new(),
                        deadline: ObservationDeadline::new(self.observation_deadline),
                    },
                );
        }
        self.sync_completion_health();
        result
    }

    fn wait(
        &self,
        token: CompletionToken,
        timeout: Duration,
    ) -> Result<CompletionDisposition, ProviderError> {
        let result = self.wait_inner(token, timeout);
        self.sync_completion_health();
        result
    }

    fn cancel(&self, token: CompletionToken) -> Result<CompletionDisposition, ProviderError> {
        let result = self.cancel_inner(token);
        self.sync_completion_health();
        result
    }

    fn readback(&self, token: CompletionToken) -> Result<CompletionReadback, ProviderError> {
        self.validate_token(token)?;
        let record = {
            let completions = self.completions.lock().map_err(|_| registry_poisoned())?;
            let slot = completions
                .get(&token.submission_id)
                .ok_or_else(|| unknown_completion(token))?;
            Arc::clone(&slot.record)
        };
        record.readback(token)
    }
}

impl Drop for VulkanComputeProvider {
    fn drop(&mut self) {
        let mut completions = match self.completions.lock() {
            Ok(completions) => completions,
            Err(poisoned) => poisoned.into_inner(),
        };
        let pending = std::mem::take(&mut *completions)
            .into_values()
            .filter_map(|slot| slot.pending)
            .collect::<Vec<_>>();
        drop(completions);
        for pending in pending {
            self.retire(pending);
        }
    }
}

fn next_identity(counter: &AtomicU64, slug: &'static str) -> Result<u64, ProviderError> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| refusal(ProviderPhase::Resolve, ProviderErrorClass::Internal, slug))
}

fn check_epoch(expected: DeviceEpoch, actual: DeviceEpoch) -> Result<(), ProviderError> {
    if actual != expected {
        return Err(refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Resource,
            "device_epoch_mismatch",
        )
        .with_field("expected", FieldValue::Unsigned(expected.get()))
        .with_field("actual", FieldValue::Unsigned(actual.get())));
    }
    Ok(())
}

fn validate_pipeline_identity(
    requested: &CompiledComputePipeline,
    metadata: &CompiledComputePipeline,
) -> Result<(), ProviderError> {
    check_epoch(metadata.device_epoch, requested.device_epoch)?;
    if requested.pipeline_id != metadata.pipeline_id {
        return Err(refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Resource,
            "pipeline_identity_mismatch",
        ));
    }
    if requested.function != metadata.function {
        return Err(refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Resource,
            "pipeline_function_mismatch",
        ));
    }
    if requested.contract != metadata.contract {
        return Err(refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Resource,
            "pipeline_contract_mismatch",
        ));
    }
    // The render half is part of the registration too: a trace that carries a
    // different (or absent) render contract would be admitted against a format
    // this context did not compile the stages for, so it is refused here even
    // though core admission already compares the pass with the entry
    // (review item I3, 2026-09-14: core is the first gate, this is the
    // registry's own).
    if requested.render != metadata.render {
        return Err(refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Resource,
            "render_pipeline_contract_mismatch",
        ));
    }
    Ok(())
}

/// Refuse a trace whose compute passes would be reordered against a render
/// pass's stores.
///
/// This increment executes every compute pass first and every render pass
/// afterwards, in trace order within each group (see
/// [`VulkanComputeProvider::execute_render_passes`]). Compute passes that come
/// before a render pass therefore run in the order the trace asked for, and so
/// do render passes among themselves. The one shape that would silently change
/// meaning is a compute pass that *follows* a render pass and binds a view that
/// render pass stores: it would observe pre-render bytes where the trace's
/// serial order defines post-render ones. That shape is refused rather than
/// executed in an order the bytes would not reflect — core admission already
/// refuses the write/write half of the pair with
/// `AttachmentComputeConflict`.
///
/// Core admission owns both halves now (`ContractError::
/// RenderPassOrderUnsupported`, slug `render_pass_order_unsupported`, review
/// item I4, 2026-09-14), and every submitted trace reached it through
/// `ProviderCapabilities::admit`. This walk stays as the rail's own defense for
/// a value-level plan: it compares view identities, so it is at least as strict
/// as the contract's byte ranges and never admits a trace the contract refused.
fn refuse_reordered_render_reads(trace: &ComputeTrace) -> Result<(), ProviderError> {
    let mut render_written = BTreeMap::<ViewId, usize>::new();
    for (index, entry) in trace.passes.iter().enumerate() {
        match entry {
            TracePass::Render(pass) => {
                for attachment in &pass.color_attachments {
                    render_written.entry(attachment.view_id).or_insert(index);
                }
            }
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

fn narrow_dimensions(wide: [u64; 3]) -> Result<Size, ProviderError> {
    let mut values = [0; 3];
    for (axis, value) in wide.into_iter().enumerate() {
        values[axis] = u32::try_from(value).map_err(|_| {
            refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Capability,
                "dispatch_dimension_overflow",
            )
            .with_field("axis", FieldValue::Unsigned(axis as u64))
            .with_field("requested", FieldValue::Unsigned(value))
            .with_field("maximum", FieldValue::Unsigned(u64::from(u32::MAX)))
        })?;
    }
    Size::new(values[0], values[1], values[2]).map_err(|error| {
        refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Args,
            "dispatch_dimension_invalid",
        )
        .with_detail(error.to_string())
    })
}

fn map_writebacks(
    pool: &[BufferView],
    textures: &[metal_api_core::provider::TextureView],
    updates: Vec<LandingUpdate>,
    token: CompletionToken,
) -> Result<Vec<BufferWriteback>, ProviderError> {
    let mut writebacks = Vec::with_capacity(updates.len());
    for update in updates {
        match update.target {
            LandingTarget::Buffer(index) => {
                let view = usize::try_from(index)
                    .ok()
                    .and_then(|position| pool.get(position))
                    .ok_or_else(|| output_error(token, "writeback_unknown_binding"))?;
                let offset = view
                    .offset
                    .checked_add(update.offset as u64)
                    .ok_or_else(|| output_error(token, "writeback_range_overflow"))?;
                writebacks.push(BufferWriteback {
                    view_id: view.view_id,
                    allocation_id: view.allocation_id,
                    offset,
                    bytes: update.bytes,
                });
            }
            // A storage image landing is keyed by the texture pool's Metal
            // index, exactly as the descriptor writer resolves the image
            // (`research/docs/26` §21.4, C2). The bytes are the view's whole
            // tightly packed extent, so the landing starts at the allocation's
            // own zero and carries the view identity the writeback channel
            // validates against. A landing whose texture is not in the pool is
            // a rail bug, and it is reported instead of being dropped.
            LandingTarget::Texture(index) => {
                let texture = textures
                    .iter()
                    .find(|texture| texture.metal_binding == index)
                    .ok_or_else(|| output_error(token, "writeback_unknown_binding"))?;
                writebacks.push(BufferWriteback {
                    view_id: texture.view_id,
                    allocation_id: texture.allocation_id,
                    offset: u64::try_from(update.offset)
                        .map_err(|_| output_error(token, "writeback_range_overflow"))?,
                    bytes: update.bytes,
                });
            }
        }
    }
    writebacks.sort_by_key(|w| (w.allocation_id, w.view_id));
    Ok(writebacks)
}

/// Serialize device work on the selected queue with the standalone executor,
/// then run the prepared sequence. The worker path calls this directly; the
/// synchronous path calls it on the submitting thread.
fn execute_on_context(
    executor: &Arc<VulkanExecutor>,
    artifacts: &[Arc<VulkanPipelineArtifact>],
    buffers: &[PoolBinding],
    dispatches: &[BoundDispatch],
    retains: &mut BorrowedRetains,
    textures: &[metal_api_core::provider::TextureView],
    indirect_dispatch: Option<[u32; 3]>,
) -> Result<Vec<LandingUpdate>, ProviderError> {
    // The synchronous path goes through the same queue policy as the deferred
    // object path (`research/docs/21` §4): the tier table decides which idle
    // queue receives the work, and a one-queue device keeps answering zero.
    let queue_index = executor.context.pick_queue();
    let _execution = executor
        .context
        .lock_queue(queue_index)
        .map_err(|_| registry_poisoned())?;
    ensure_executor_usable(executor)?;
    execute_pool_sequence_with_status(
        &executor.context,
        artifacts,
        buffers,
        dispatches,
        SequenceTail {
            borrowed: retains.take(),
            textures,
            indirect_dispatch,
        },
        queue_index,
    )
}

fn ensure_executor_usable(executor: &VulkanExecutor) -> Result<(), ProviderError> {
    ensure_context_usable(&executor.context)
}

fn sync_context_health(context: &VulkanContext, outbox: &CompletionOutbox) {
    let health = context.health();
    if outbox.health() != health {
        let _ = outbox.publish_device(health);
    }
}

/// Refuse new work on a terminal context.
///
/// The context's `metal_api_core` lifecycle is the only admission authority,
/// so this is a pass-through: the structured refusal the provider returns is
/// exactly the one the lifecycle produced, `terminal` field and abandoned
/// counters included. There is no second mapping to keep in step.
fn ensure_context_usable(context: &VulkanContext) -> Result<(), ProviderError> {
    context.admit()
}

fn duration_to_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn attach_token(mut error: ProviderError, token: CompletionToken) -> ProviderError {
    error.completion = match error.completion {
        CompletionDisposition::NotSubmitted => CompletionDisposition::NotSubmitted,
        CompletionDisposition::Failed { .. } => {
            CompletionDisposition::Failed { token: Some(token) }
        }
        CompletionDisposition::DeviceLost { .. } => {
            CompletionDisposition::DeviceLost { token: Some(token) }
        }
        _ => CompletionDisposition::SubmittedUnknown { token: Some(token) },
    };
    error
}

fn output_error(token: CompletionToken, slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Readback, ProviderErrorClass::Internal, slug)
        .with_completion(CompletionDisposition::Failed { token: Some(token) })
}

fn refusal(phase: ProviderPhase, class: ProviderErrorClass, slug: &'static str) -> ProviderError {
    let mut error = ProviderError::new(phase, class, slug).expect("static non-empty refusal slug");
    error.retryability = Retryability::Never;
    error
}

fn registry_poisoned() -> ProviderError {
    refusal(
        ProviderPhase::Resolve,
        ProviderErrorClass::Internal,
        "provider_registry_poisoned",
    )
}

/// The refusal-field name of one terminal state, so a caller can read which
/// non-loss state refused a rebuild without matching on the enum itself.
fn terminal_state_name(state: TerminalState) -> &'static str {
    match state {
        TerminalState::Usable => "usable",
        TerminalState::Exhausted { .. } => "abandonment_exhausted",
        TerminalState::DeviceLost => "device_lost",
    }
}

fn unknown_pipeline(id: PipelineId) -> ProviderError {
    refusal(
        ProviderPhase::Resolve,
        ProviderErrorClass::Resource,
        "unknown_pipeline",
    )
    .with_field("pipeline", FieldValue::Unsigned(id.get()))
}

fn unknown_render_pipeline(id: PipelineId) -> ProviderError {
    refusal(
        ProviderPhase::Resolve,
        ProviderErrorClass::Resource,
        "unknown_render_pipeline",
    )
    .with_field("pipeline", FieldValue::Unsigned(id.get()))
}

/// The pipeline-table contract one render registration carries.
///
/// `ComputeTrace` has a single pipeline entry shape and core admission
/// validates every entry's contract, so a render registration carries the
/// most permissive exact-thread contract: no bindings, no push constants and no
/// fixed grid. Nothing reads it as a compute contract — the compute rail
/// resolves artifacts out of its own registry, where a render registration does
/// not exist.
fn render_pipeline_table_contract() -> PipelineContract {
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

/// Refuse a compute texture declaration the registered module does not carry
/// (`research/docs/26` §21.3–§21.4, C1b/C2).
///
/// The halves of one declaration are checked in the order the execution path
/// resolves them. The access decides which descriptor the binding is executed
/// as — sampled textures are combined image samplers, storage images are
/// `STORAGE_IMAGE` descriptors — so a request that names the other one is
/// refused by name (`compute_texture_access_unsupported`) with both accesses.
/// The shape follows (`compute_texture_type_unsupported`,
/// `compute_texture_format_unsupported`), because the module's `OpTypeImage`
/// was decorated with one dimensionality and one format. When those agree the
/// sampler state is restated: from C1b on the rail creates one `VkSampler` per
/// AIR-embedded constexpr sampler, with the state the module's own AIR was
/// lowered against, so a request naming another filtering or address mode would
/// change which texels the module reads without changing the module and is
/// refused with the binding, the declared state and the module's state rather
/// than a silently substituted sampler.
fn refuse_foreign_texture_samplers(
    requested: &CompiledComputePipeline,
    registered: &CompiledComputePipeline,
) -> Result<(), ProviderError> {
    for declared in &requested.contract.texture_bindings {
        let Some(module) = registered
            .contract
            .texture_bindings
            .iter()
            .find(|candidate| candidate.metal_binding == declared.metal_binding)
        else {
            // A binding the module does not declare is the identity walk's
            // refusal, not this gate's.
            continue;
        };
        // The access is the first half of the same restatement
        // (`research/docs/26` §21.4, C2): a declaration that names a sampled
        // binding where the module writes is refused by name before the sampler
        // comparison, because the two accesses are executed by different
        // descriptors and a substituted one would change what the module can
        // observe. Core's pair rules already refuse a *view* that disagrees
        // with a request's own contract; this gate is the request's contract
        // against the module that will run.
        if declared.access != module.access {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Capability,
                "compute_texture_access_unsupported",
            )
            .with_field(
                "binding",
                FieldValue::Unsigned(u64::from(declared.metal_binding)),
            )
            .with_field(
                "declared_access",
                FieldValue::Text(format!("{:?}", declared.access)),
            )
            .with_field(
                "module_access",
                FieldValue::Text(format!("{:?}", module.access)),
            ));
        }
        // The shape is the second half of the restatement. The module's
        // `OpTypeImage` was decorated with one dimensionality and one format,
        // and a descriptor of another shape would either be refused by Vulkan
        // or read as bytes the module never declared. Each half names itself so
        // a fix needs no second lookup.
        if declared.texture_type != module.texture_type {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Capability,
                "compute_texture_type_unsupported",
            )
            .with_field(
                "binding",
                FieldValue::Unsigned(u64::from(declared.metal_binding)),
            )
            .with_field(
                "declared_type",
                FieldValue::Text(format!("{:?}", declared.texture_type)),
            )
            .with_field(
                "module_type",
                FieldValue::Text(format!("{:?}", module.texture_type)),
            ));
        }
        if declared.format != module.format {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Capability,
                "compute_texture_format_unsupported",
            )
            .with_field(
                "binding",
                FieldValue::Unsigned(u64::from(declared.metal_binding)),
            )
            .with_field(
                "declared_format",
                FieldValue::Text(format!("{:?}", declared.format)),
            )
            .with_field(
                "module_format",
                FieldValue::Text(format!("{:?}", module.format)),
            ));
        }
        // A storage declaration carries no sampler on either side (core
        // enforces that invariant), so the sampler comparison below runs for
        // sampled bindings only — where both halves are present.
        if declared.sampler == module.sampler {
            continue;
        }
        let (declared_sampler, module_sampler) = match (declared.sampler, module.sampler) {
            (Some(declared_sampler), Some(module_sampler)) => (declared_sampler, module_sampler),
            // Unreachable through core validation: an access either carries a
            // sampler on both sides or on neither. Kept as a named refusal so a
            // hand-built contract cannot slip past with a half-stated pair.
            (declared_sampler, module_sampler) => {
                return Err(refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Capability,
                    "compute_texture_sampler_unsupported",
                )
                .with_field(
                    "binding",
                    FieldValue::Unsigned(u64::from(declared.metal_binding)),
                )
                .with_field(
                    "declared_sampler",
                    FieldValue::Text(format!("{declared_sampler:?}")),
                )
                .with_field(
                    "module_sampler",
                    FieldValue::Text(format!("{module_sampler:?}")),
                ));
            }
        };
        if declared_sampler != module_sampler {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Capability,
                "compute_texture_sampler_unsupported",
            )
            .with_field(
                "binding",
                FieldValue::Unsigned(u64::from(declared.metal_binding)),
            )
            .with_field(
                "filter",
                FieldValue::Text(format!("{:?}", declared_sampler.filter)),
            )
            .with_field(
                "address",
                FieldValue::Text(format!("{:?}", declared_sampler.address)),
            )
            .with_field(
                "module_filter",
                FieldValue::Text(format!("{:?}", module_sampler.filter)),
            )
            .with_field(
                "module_address",
                FieldValue::Text(format!("{:?}", module_sampler.address)),
            ));
        }
    }
    Ok(())
}

fn unknown_completion(token: CompletionToken) -> ProviderError {
    refusal(
        ProviderPhase::Wait,
        ProviderErrorClass::Resource,
        "unknown_completion",
    )
    .with_completion(CompletionDisposition::SubmittedUnknown { token: Some(token) })
}

fn borrowed_alignment_error(lease_id: LeaseId, pointer: usize, alignment: u64) -> ProviderError {
    refusal(
        ProviderPhase::Resolve,
        ProviderErrorClass::Capability,
        "lease_alignment_unsupported",
    )
    .with_field("lease", FieldValue::Unsigned(lease_id.get()))
    .with_field("pointer", FieldValue::Unsigned(pointer as u64))
    .with_field("alignment", FieldValue::Unsigned(alignment))
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::provider::ComputePass;
    use metal_api_core::provider::{
        AllocationId, AttachmentFormat, BufferAccess, ClearColor, Dispatch, DispatchKind,
        DispatchType, LoadOp, OperationId, ProviderLifecycle, RenderAttachment, SamplerPolicy,
        StoreOp, ViewId, PROVIDER_SCHEMA_VERSION,
    };

    /// An admitted-shaped trace whose pass list is the only thing under test.
    ///
    /// `refuse_reordered_render_reads` walks the pass list, so this fixture
    /// deliberately carries no pipeline table and no resources: the values are
    /// the ones the ordering rule reads and nothing else.
    fn ordering_trace(passes: Vec<TracePass>) -> ComputeTrace {
        ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(3),
            operation_id: OperationId::new(5),
            pipelines: Vec::new(),
            encoder_dispatch_type: DispatchType::Serial,
            passes,
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        }
    }

    fn ordering_render_pass(view: u64) -> RenderPassDescriptor {
        RenderPassDescriptor {
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
            pipeline: PipelineId::new(1),
            color_attachments: vec![RenderAttachment {
                view_id: ViewId::new(view),
                allocation_id: AllocationId::new(2),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                load: LoadOp::Clear(ClearColor::new([0xfe; 4])),
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

    fn ordering_compute_pass(binding_view: u64) -> ComputePass {
        ComputePass {
            pipeline: PipelineId::new(2),
            buffers: vec![BufferView {
                view_id: ViewId::new(binding_view),
                metal_binding: 0,
                allocation_id: AllocationId::new(2),
                offset: 0,
                length: 16,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![0; 16]),
            }],
            textures: Vec::new(),
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
        }
    }

    /// A compute pass that *follows* a render pass storing the view it binds
    /// would silently observe pre-render bytes, because this increment runs
    /// every compute pass before every render pass. The trace is refused
    /// instead; the other three orders stay admissible.
    #[test]
    fn a_compute_pass_after_a_render_pass_may_not_observe_its_attachment() {
        let refused = refuse_reordered_render_reads(&ordering_trace(vec![
            TracePass::Render(ordering_render_pass(9)),
            TracePass::Compute(ordering_compute_pass(9)),
        ]))
        .expect_err("a compute pass observing a stored attachment is refused");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_pass_order_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.fields.get("pass"), Some(&FieldValue::Unsigned(1)));
        assert_eq!(
            refused.fields.get("render_pass"),
            Some(&FieldValue::Unsigned(0))
        );
        assert_eq!(refused.fields.get("view"), Some(&FieldValue::Unsigned(9)));

        // The same two passes in trace order: the compute pass reads the bytes
        // the trace says it reads.
        refuse_reordered_render_reads(&ordering_trace(vec![
            TracePass::Compute(ordering_compute_pass(9)),
            TracePass::Render(ordering_render_pass(9)),
        ]))
        .expect("a compute pass before the store keeps its own bytes");

        // A compute pass after the store that binds a different view does not
        // observe the attachment at all.
        refuse_reordered_render_reads(&ordering_trace(vec![
            TracePass::Render(ordering_render_pass(9)),
            TracePass::Compute(ordering_compute_pass(11)),
        ]))
        .expect("an unrelated view carries no hazard");
    }

    /// The provider no longer spells its own terminal refusals: the core
    /// lifecycle owns the terminal state and its refusal, and the Vulkan side
    /// returns that error unchanged. These assertions are the ones the two
    /// local builders used to carry, now checked against the single source.
    #[test]
    fn unavailable_provider_errors_require_recreation() {
        let mut lost_lifecycle = ProviderLifecycle::new(1, 4096);
        lost_lifecycle.mark_device_lost();
        let lost = crate::terminal_refusal(&lost_lifecycle).expect_err("lost device refuses work");
        assert_eq!(lost.phase, ProviderPhase::Resolve);
        assert_eq!(lost.class, ProviderErrorClass::DeviceLost);
        assert_eq!(lost.slug, "device_lost");
        assert_eq!(lost.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(
            lost.completion,
            CompletionDisposition::DeviceLost { token: None }
        );
        assert_eq!(
            lost.fields.get("terminal"),
            Some(&FieldValue::Text("device_lost".into()))
        );

        let mut exhausted_lifecycle = ProviderLifecycle::new(1, 4096);
        assert_eq!(
            exhausted_lifecycle.record_abandonment(512),
            AbandonmentOutcome::Exhausted
        );
        let exhausted = crate::terminal_refusal(&exhausted_lifecycle)
            .expect_err("exhausted budget refuses work");
        assert_eq!(exhausted.phase, ProviderPhase::Resolve);
        assert_eq!(exhausted.class, ProviderErrorClass::Resource);
        assert_eq!(exhausted.slug, "provider_unavailable");
        assert_eq!(exhausted.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(exhausted.completion, CompletionDisposition::NotSubmitted);
        assert_eq!(
            exhausted.fields.get("terminal"),
            Some(&FieldValue::Text("abandonment_budget".into()))
        );
        assert_eq!(
            exhausted.fields.get("abandoned_submissions"),
            Some(&FieldValue::Unsigned(1))
        );
        assert_eq!(
            exhausted.fields.get("abandoned_bytes"),
            Some(&FieldValue::Unsigned(512))
        );
    }

    #[test]
    fn a_texture_contract_must_restate_the_sampler_its_module_carries() {
        use metal_api_core::provider::{SamplerAddressMode, SamplerFilter};

        // The registered module carries nearest+clamp; a request that restates
        // exactly that state is the declaration the module was lowered
        // against.
        let mut module_trace = texture_sampler_trace(SamplerPolicy::synthesized_read());
        let module = module_trace.pipelines.remove(0);
        refuse_foreign_texture_samplers(&module, &module)
            .expect("the module's own sampler state is the declaration");

        // Either half of the state can move the request away from the module,
        // and each refusal names the binding, the declared state and the
        // module's own state (`research/docs/26` §21.3, C1b).
        for (filter, address) in [
            (SamplerFilter::Linear, SamplerAddressMode::ClampToEdge),
            (SamplerFilter::Nearest, SamplerAddressMode::Repeat),
        ] {
            let mut requested_trace = texture_sampler_trace(SamplerPolicy { filter, address });
            let requested = requested_trace.pipelines.remove(0);
            let refusal = refuse_foreign_texture_samplers(&requested, &module)
                .expect_err("a state the module was not lowered against is a refusal");
            eprintln!("compute texture sampler refusal: {refusal:?}");
            assert_eq!(refusal.slug, "compute_texture_sampler_unsupported");
            assert_eq!(refusal.class, ProviderErrorClass::Capability);
            assert_eq!(
                refusal.fields.get("binding"),
                Some(&FieldValue::Unsigned(0))
            );
            assert_eq!(
                refusal.fields.get("filter"),
                Some(&FieldValue::Text(format!("{filter:?}")))
            );
            assert_eq!(
                refusal.fields.get("address"),
                Some(&FieldValue::Text(format!("{address:?}")))
            );
            assert_eq!(
                refusal.fields.get("module_filter"),
                Some(&FieldValue::Text("Nearest".into()))
            );
            assert_eq!(
                refusal.fields.get("module_address"),
                Some(&FieldValue::Text("ClampToEdge".into()))
            );
        }
    }

    /// One compute pass whose pipeline contract declares a texture binding with
    /// the given sampler state. Only the gate under test reads it: no device,
    /// no resources and no texture view are needed.
    fn texture_sampler_trace(sampler: SamplerPolicy) -> ComputeTrace {
        use metal_api_core::provider::{
            FunctionSource, TextureAccess, TextureBindingContract, TextureFootprintProof,
            TextureFormat, TextureType,
        };

        let mut trace = ordering_trace(vec![TracePass::Compute(ordering_compute_pass(7))]);
        trace.pipelines.push(CompiledComputePipeline {
            device_epoch: trace.device_epoch,
            pipeline_id: PipelineId::new(2),
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("fixture", vec![9]).unwrap(),
                entry_name: "read_texture_2d".to_owned(),
                source: FunctionSource::BinaryAir,
            },
            contract: PipelineContract {
                dispatch_kind: DispatchKind::ThreadsExact,
                required_local_size: None,
                fixed_grid: None,
                push_constant_offset: 0,
                push_constant_bytes: 0,
                buffer_bindings: Vec::new(),
                texture_bindings: vec![TextureBindingContract {
                    metal_binding: 0,
                    access: TextureAccess::Sampled,
                    texture_type: TextureType::D2,
                    format: TextureFormat::R32Uint,
                    sampler: Some(sampler),
                    footprint: TextureFootprintProof::WholeView,
                }],
                shader_capabilities: Vec::new(),
                translator_revision: None,
            },
            render: None,
        });
        trace
    }

    #[test]
    fn writebacks_use_pool_identity_when_later_resources_repeat_binding_labels() {
        let token = CompletionToken {
            device_epoch: DeviceEpoch::new(1),
            submission_id: SubmissionId::new(2),
        };
        let pool: Vec<_> = [(330, 430, 20), (340, 440, 48)]
            .into_iter()
            .map(|(allocation, view, offset)| BufferView {
                view_id: ViewId::new(view),
                metal_binding: 9,
                allocation_id: AllocationId::new(allocation),
                offset,
                length: 4,
                access: BufferAccess::Write,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![0; 4]),
            })
            .collect();
        let updates = vec![
            LandingUpdate {
                target: LandingTarget::Buffer(1),
                offset: 0,
                bytes: vec![2; 4],
            },
            LandingUpdate {
                target: LandingTarget::Buffer(0),
                offset: 0,
                bytes: vec![1; 4],
            },
        ];
        let writes = map_writebacks(&pool, &[], updates, token).unwrap();
        assert_eq!(writes[0].view_id, ViewId::new(430));
        assert_eq!(writes[0].allocation_id, AllocationId::new(330));
        assert_eq!(writes[0].offset, 20);
        assert_eq!(writes[0].bytes, vec![1; 4]);
        assert_eq!(writes[1].view_id, ViewId::new(440));
        assert_eq!(writes[1].allocation_id, AllocationId::new(340));
        assert_eq!(writes[1].offset, 48);
        assert_eq!(writes[1].bytes, vec![2; 4]);
        assert_eq!(
            map_writebacks(
                &pool,
                &[],
                vec![LandingUpdate {
                    target: LandingTarget::Buffer(9),
                    offset: 0,
                    bytes: vec![3; 4]
                }],
                token
            )
            .unwrap_err()
            .slug,
            "writeback_unknown_binding"
        );
    }

    #[test]
    fn identity_exhaustion_never_wraps_or_reuses_a_value() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(next_identity(&counter, "exhausted").unwrap(), u64::MAX - 1);
        assert_eq!(
            next_identity(&counter, "exhausted").unwrap_err().slug,
            "exhausted"
        );
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn wide_dimensions_refuse_truncation() {
        let error = narrow_dimensions([u64::from(u32::MAX) + 1, 1, 1]).unwrap_err();
        assert_eq!(error.slug, "dispatch_dimension_overflow");
        assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
        assert_eq!(
            narrow_dimensions([10, 3, 1]).unwrap().dimensions(),
            [10, 3, 1]
        );
    }

    #[test]
    fn failure_tokens_only_attach_after_submission() {
        let token = CompletionToken {
            device_epoch: DeviceEpoch::new(1),
            submission_id: SubmissionId::new(2),
        };
        let before = refusal(
            ProviderPhase::Encode,
            ProviderErrorClass::Execute,
            "encode_failed",
        );
        assert_eq!(
            attach_token(before.clone(), token).completion,
            CompletionDisposition::NotSubmitted
        );
        let after = before.with_completion(CompletionDisposition::SubmittedUnknown { token: None });
        assert_eq!(
            attach_token(after, token).completion,
            CompletionDisposition::SubmittedUnknown { token: Some(token) }
        );
    }
}
