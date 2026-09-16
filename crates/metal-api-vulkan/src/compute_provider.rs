//! Compute provider over the shared Vulkan executor: owned bytes, staged
//! lease copies and host-memory no-copy imports.

use crate::{
    execute_pool_sequence_with_status, render, Binding, BoundDispatch, PendingExecution,
    PoolBinding, PoolKey, PoolKind, SequenceTail, TranslatedComputePipeline, VulkanContext,
    VulkanExecutor, VulkanPipelineArtifact,
};
use metal_api_core::completion::wire::CompletionOutbox;
use metal_api_core::completion::{AbandonmentOutcome, CompletionRecord, ObservationDeadline};
pub use metal_api_core::provider::CompiledComputePipeline;
use metal_api_core::provider::{
    allocate_device_epoch, AliasMode, AllocationId, BufferSource, BufferView, BufferWriteback,
    CompletionDisposition, CompletionPolicy, CompletionReadback, CompletionToken, ComputeProvider,
    ComputeTrace, DeviceEpoch, DispatchKind, FieldValue, FunctionIdentity, FunctionSource, HeapId,
    HeapResource, IndirectCommandDescriptor, IndirectCommandKind, LeaseId, LeaseImporter,
    LeaseRegistry, PipelineCompileRequest, PipelineContract, PipelineId, PipelineProvider,
    PresentDescriptor, ProviderCapabilities, ProviderError, ProviderErrorClass, ProviderHealth,
    ProviderPhase, ProviderSubmission, QueuePriority, RenderAttachment, RenderPassDescriptor,
    RenderPipelineContract, ResourceTableSnapshot, Retryability, SemanticDigest, ShaderSource,
    StagedLease, StorageMode, SubmissionId, TerminalState, TracePass, ValidatedComputeTrace,
    ViewId,
};
use metal_api_core::provider::{
    queue_priorities_for_device, BorrowedLease, BorrowedLeaseRegistry, NoCopyLeaseImporter,
};
use metal_api_core::{AirSource, BufferBinding, BufferUpdate, Device, Function, Size};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

const TRANSLATOR_REVISION: &[u8] = b"43c46ac8a24adf1a6e872b8a52c706ec9614fad0";
const GPU_DEADLINE: Duration = Duration::from_secs(20);

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
    present_targets: Mutex<BTreeMap<(AllocationId, ViewId), Arc<render::PresentTargetImage>>>,
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
        let translated = TranslatedComputePipeline::translate(function).map_err(|error| {
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
        self.ensure_usable()?;
        let stages = render::RenderStages {
            contract: request.contract,
            vertex_spirv: request.vertex_spirv,
            fragment_spirv: request.fragment_spirv,
        };
        stages.validate()?;
        let function = FunctionIdentity {
            logical_digest: request.logical_digest,
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
                // The declared view serves two purposes: it is the attachment's
                // landing for the writeback channel, and it is the source of
                // the previous bytes a `LoadOp::Load` pass uploads before it
                // opens (`research/docs/23` §3.3). A loading pass therefore
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
                let previous = if loading {
                    match view.map(|view| &view.source) {
                        Some(BufferSource::OwnedBytes(bytes)) => Some(bytes.as_slice()),
                        _ => {
                            return Err(refusal(
                                ProviderPhase::Resolve,
                                ProviderErrorClass::Capability,
                                "attachment_load_op_unsupported",
                            )
                            .with_field("load_op", FieldValue::Text("load".to_owned()))
                            .with_detail(
                                "the first `LoadOp::Load` increment uploads trace-owned bytes only",
                            ));
                        }
                    }
                } else {
                    None
                };
                let target = self.present_target(present, attachment)?;
                let executor = self.lock_executor()?;
                let texels = render::execute_present_render(
                    &executor.context,
                    &planned.stages,
                    &planned.pass,
                    &target,
                    previous,
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
            // previous-byte source per attachment, in location order, then hand
            // the render rail the whole list and publish one writeback per
            // attachment that has a landing.
            let mut views = Vec::with_capacity(planned.pass.color_attachments.len());
            let mut previous = Vec::with_capacity(planned.pass.color_attachments.len());
            for attachment in &planned.pass.color_attachments {
                let declared = pool.iter().find(|view| {
                    view.view_id == attachment.view_id
                        && view.allocation_id == attachment.allocation_id
                });
                let loading = matches!(attachment.load, metal_api_core::provider::LoadOp::Load);
                let view = if host_readback || loading {
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
                let source = if loading {
                    match view.map(|view| &view.source) {
                        Some(BufferSource::OwnedBytes(bytes)) => Some(bytes.as_slice()),
                        _ => {
                            return Err(refusal(
                                ProviderPhase::Resolve,
                                ProviderErrorClass::Capability,
                                "attachment_load_op_unsupported",
                            )
                            .with_field("load_op", FieldValue::Text("load".to_owned()))
                            .with_detail(
                                "the first `LoadOp::Load` increment uploads trace-owned bytes only",
                            ));
                        }
                    }
                } else {
                    None
                };
                previous.push(source);
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
            let readback = match trace.indirect.as_deref() {
                Some(payload) => {
                    let readback = render::execute_indirect_render_pass(
                        &executor.context,
                        &planned.stages,
                        &planned.pass,
                        &payload.command,
                        &previous,
                    )?;
                    // Publish what was actually replayed: the command kind,
                    // the range and the one command the first increment
                    // encodes (`research/docs/25` §5.1).
                    self.publish_icb_observation(
                        payload.command.kind(),
                        payload.range.start,
                        payload.range.count,
                        1,
                    );
                    readback
                }
                None => render::execute_render_pass(
                    &executor.context,
                    &planned.stages,
                    &planned.pass,
                    &previous,
                )?,
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
    fn present_target(
        &self,
        present: &PresentDescriptor,
        attachment: &RenderAttachment,
    ) -> Result<Arc<render::PresentTargetImage>, ProviderError> {
        let key = (present.target.allocation_id, present.target.view_id);
        if let Some(existing) = self
            .present_targets
            .lock()
            .map_err(|_| registry_poisoned())?
            .get(&key)
        {
            return Ok(Arc::clone(existing));
        }
        let mut image = render::PresentTargetImage::create(
            Arc::clone(&self.lock_executor()?.context),
            attachment.format,
            attachment.width,
            attachment.height,
        )?;
        if let Some(sentinel) = present.target.initial.sentinel() {
            image.preset_sentinel(attachment.format, sentinel)?;
        }
        let mut registry = self
            .present_targets
            .lock()
            .map_err(|_| registry_poisoned())?;
        let existing = registry.entry(key).or_insert_with(|| Arc::new(image));
        Ok(Arc::clone(existing))
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
        let (record, pending, pool, render_writebacks, heap_observations, deadline) = {
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
                    slot.deadline,
                )
            } else if let Some(pending) = slot.pending.take() {
                (
                    Arc::clone(&slot.record),
                    Some(pending),
                    slot.pool.clone(),
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
                .and_then(|updates| map_writebacks(&pool, updates, token))
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
        self.staging.import(staged)
    }

    fn release_staged_lease(&self, lease_id: LeaseId) -> Result<(), ProviderError> {
        self.staging.release(lease_id)
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
        self.borrowed.import(borrowed)
    }

    fn release_borrowed_lease(&self, lease_id: LeaseId) -> Result<(), ProviderError> {
        self.borrowed.release(lease_id)
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
            let render_writebacks = match self.execute_render_passes(trace, &pool, &render_plan) {
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
            for writeback in map_writebacks(&pool, updates, token)?
                .into_iter()
                .chain(self.execute_render_passes(trace, &pool, &render_plan)?)
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
    updates: Vec<BufferUpdate>,
    token: CompletionToken,
) -> Result<Vec<BufferWriteback>, ProviderError> {
    let mut writebacks = Vec::with_capacity(updates.len());
    for update in updates {
        let view = usize::try_from(update.index)
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
) -> Result<Vec<BufferUpdate>, ProviderError> {
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
        shader_capabilities: Vec::new(),
        translator_revision: None,
    }
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
        DispatchType, LoadOp, OperationId, ProviderLifecycle, RenderAttachment, StoreOp, ViewId,
        PROVIDER_SCHEMA_VERSION,
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
            BufferUpdate {
                index: 1,
                offset: 0,
                bytes: vec![2; 4],
            },
            BufferUpdate {
                index: 0,
                offset: 0,
                bytes: vec![1; 4],
            },
        ];
        let writes = map_writebacks(&pool, updates, token).unwrap();
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
                vec![BufferUpdate {
                    index: 9,
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
