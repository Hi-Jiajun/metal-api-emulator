//! Compute provider over the shared Vulkan executor: owned bytes, staged
//! lease copies and host-memory no-copy imports.

use crate::{
    execute_pool_sequence_with_status, Binding, BoundDispatch, PendingExecution, PoolBinding,
    PoolKey, PoolKind, TranslatedComputePipeline, VulkanContext, VulkanExecutor,
    VulkanPipelineArtifact,
};
use metal_api_core::completion::wire::CompletionOutbox;
use metal_api_core::completion::{AbandonmentOutcome, CompletionRecord, ObservationDeadline};
pub use metal_api_core::provider::CompiledComputePipeline;
use metal_api_core::provider::{
    allocate_device_epoch, AliasMode, AllocationId, BufferSource, BufferView, BufferWriteback,
    CompletionDisposition, CompletionReadback, CompletionToken, ComputeProvider, DeviceEpoch,
    FieldValue, FunctionIdentity, FunctionSource, LeaseId, LeaseImporter, LeaseRegistry,
    PipelineCompileRequest, PipelineId, PipelineProvider, ProviderCapabilities, ProviderError,
    ProviderErrorClass, ProviderHealth, ProviderPhase, ProviderSubmission, QueuePriority,
    Retryability, SemanticDigest, ShaderSource, StagedLease, StorageMode, SubmissionId,
    ValidatedComputeTrace,
};
use metal_api_core::provider::{
    queue_priorities_for_device, BorrowedLease, BorrowedLeaseRegistry, NoCopyLeaseImporter,
};
use metal_api_core::{AirSource, BufferBinding, BufferUpdate, Device, Function, Size};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const TRANSLATOR_REVISION: &[u8] = b"43c46ac8a24adf1a6e872b8a52c706ec9614fad0";
const GPU_DEADLINE: Duration = Duration::from_secs(20);

struct RegisteredPipeline {
    metadata: CompiledComputePipeline,
    artifact: Arc<VulkanPipelineArtifact>,
}

struct CompletionSlot {
    record: Arc<CompletionRecord>,
    pending: Option<PendingExecution>,
    pool: Vec<BufferView>,
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
    executor: Arc<VulkanExecutor>,
    epoch: DeviceEpoch,
    capabilities: ProviderCapabilities,
    next_pipeline: AtomicU64,
    next_submission: AtomicU64,
    pipelines: Mutex<BTreeMap<PipelineId, Arc<RegisteredPipeline>>>,
    completions: Mutex<BTreeMap<SubmissionId, CompletionSlot>>,
    retire_tx: Mutex<Option<mpsc::Sender<PendingExecution>>>,
    observation_deadline: Duration,
    async_execution: bool,
    completion_outbox: Option<Arc<CompletionOutbox>>,
    staging: LeaseRegistry,
    borrowed: Arc<BorrowedLeaseRegistry>,
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
        Ok(Self {
            executor,
            epoch,
            capabilities,
            next_pipeline: AtomicU64::new(1),
            next_submission: AtomicU64::new(1),
            pipelines: Mutex::new(BTreeMap::new()),
            completions: Mutex::new(BTreeMap::new()),
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

    pub fn async_execution(&self) -> bool {
        self.async_execution
    }

    /// Publish admission, terminal transitions and device health through
    /// `outbox`. The outbox must be scoped to this provider's device epoch.
    /// Without an outbox the provider keeps its in-process behavior.
    pub fn with_completion_outbox(
        mut self,
        outbox: Arc<CompletionOutbox>,
    ) -> Result<Self, ProviderError> {
        if outbox.device_epoch() != self.epoch {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Args,
                "completion_outbox_epoch_mismatch",
            )
            .with_field("expected", FieldValue::Unsigned(self.epoch.get()))
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
        self.epoch
    }

    pub fn device_name(&self) -> &str {
        self.executor.device_name()
    }

    /// Report whether this provider can still admit new work.
    ///
    /// Health and admission are the same query on the same lifecycle, so a
    /// provider that reports `Usable` here also admits the next submission.
    pub fn health(&self) -> ProviderHealth {
        self.executor.context.health()
    }

    /// Report `(abandoned submissions, abandoned bytes)` for this context.
    pub fn abandonment_stats(&self) -> (u64, u64) {
        self.executor.context.abandonment_stats()
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
            device_epoch: self.epoch,
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
        };
        let registered = RegisteredPipeline {
            metadata: metadata.clone(),
            artifact: Arc::new(VulkanPipelineArtifact {
                context: Arc::clone(&self.executor.context),
                translated,
            }),
        };
        self.pipelines
            .lock()
            .map_err(|_| registry_poisoned())?
            .insert(metadata.pipeline_id, Arc::new(registered));
        Ok(metadata)
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

    fn retire(&self, pending: PendingExecution) {
        let mut slot = match self.retire_tx.lock() {
            Ok(slot) => slot,
            Err(poisoned) => poisoned.into_inner(),
        };
        let sender = slot.get_or_insert_with(|| {
            let (tx, rx) = mpsc::channel::<PendingExecution>();
            let context = Arc::clone(&self.executor.context);
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
        let (record, pending, pool, deadline) = {
            let mut completions = self.completions.lock().map_err(|_| registry_poisoned())?;
            let slot = completions
                .get_mut(&token.submission_id)
                .ok_or_else(|| unknown_completion(token))?;
            if !slot.record.is_running() {
                (Arc::clone(&slot.record), None, Vec::new(), slot.deadline)
            } else if let Some(pending) = slot.pending.take() {
                (
                    Arc::clone(&slot.record),
                    Some(pending),
                    slot.pool.clone(),
                    slot.deadline,
                )
            } else {
                (Arc::clone(&slot.record), None, Vec::new(), slot.deadline)
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
                    record.complete(writebacks);
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
            sync_context_health(&self.executor.context, outbox);
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
        check_epoch(self.epoch, token.device_epoch)
    }

    fn ensure_usable(&self) -> Result<(), ProviderError> {
        ensure_context_usable(&self.executor.context)
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
}

impl LeaseImporter for VulkanComputeProvider {
    fn import_staged_lease(&self, staged: StagedLease) -> Result<(), ProviderError> {
        if staged.reservation.lease.owner_epoch != self.epoch {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Args,
                "lease_epoch_mismatch",
            )
            .with_field("expected", FieldValue::Unsigned(self.epoch.get()))
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
        self.executor.context.external_memory_host_alignment()
    }

    unsafe fn import_borrowed_lease(&self, borrowed: BorrowedLease) -> Result<(), ProviderError> {
        if borrowed.reservation.lease.owner_epoch != self.epoch {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Args,
                "lease_epoch_mismatch",
            )
            .with_field("expected", FieldValue::Unsigned(self.epoch.get()))
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
        self.epoch
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
        let device = Device::new(self.executor.clone());
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
        check_epoch(self.epoch, metadata.device_epoch)?;
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
        self.capabilities.clone()
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
        let installed = queue_priorities_for_device(self.executor.queue_count(), tiers);
        self.executor
            .set_queue_priorities(&installed)
            .map_err(|error| {
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
        check_epoch(self.epoch, trace.device_epoch)?;
        // A ValidatedComputeTrace may have been admitted against another
        // capability snapshot. Only the receiving owner can authorize execution.
        self.capabilities.admit(trace, admitted.resources())?;
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
            device_epoch: self.epoch,
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
                        self.epoch,
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
                        self.epoch,
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
            let result = self.submit_async(
                pool,
                textures,
                artifacts,
                buffers,
                dispatches,
                token,
                &mut retains,
            );
            self.sync_completion_health();
            return result;
        }
        let result = execute_on_context(
            &self.executor,
            &artifacts,
            &buffers,
            &dispatches,
            &mut retains,
            &textures,
        )
        .and_then(|updates| {
            let writebacks = map_writebacks(&pool, updates, token)?;
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

impl VulkanComputeProvider {
    #[allow(clippy::too_many_arguments)]
    fn submit_async(
        &self,
        pool: Vec<BufferView>,
        textures: Vec<metal_api_core::provider::TextureView>,
        artifacts: Vec<Arc<VulkanPipelineArtifact>>,
        buffers: Vec<PoolBinding>,
        dispatches: Vec<BoundDispatch>,
        token: CompletionToken,
        retains: &mut BorrowedRetains,
    ) -> Result<ProviderSubmission, ProviderError> {
        let queue_index = self.executor.context.pick_queue();
        let pending = {
            let _execution = self
                .executor
                .context
                .lock_queue(queue_index)
                .map_err(|_| registry_poisoned())?;
            ensure_executor_usable(&self.executor)?;
            PendingExecution::submit(
                &self.executor.context,
                queue_index,
                &artifacts,
                &buffers,
                &dispatches,
                retains.take(),
                &textures,
            )?
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
                    deadline: ObservationDeadline::new(self.observation_deadline),
                },
            );
        Ok(ProviderSubmission {
            completion: CompletionDisposition::Submitted { token },
            writebacks: Vec::new(),
        })
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
        retains.take(),
        textures,
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

fn unknown_pipeline(id: PipelineId) -> ProviderError {
    refusal(
        ProviderPhase::Resolve,
        ProviderErrorClass::Resource,
        "unknown_pipeline",
    )
    .with_field("pipeline", FieldValue::Unsigned(id.get()))
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
    use metal_api_core::provider::{AllocationId, BufferAccess, ProviderLifecycle, ViewId};

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
