//! Owned-byte implementation of the first compute provider slice.

use crate::{
    execute_pipeline_sequence_with_status, BoundDispatch, PendingExecution,
    TranslatedComputePipeline, VulkanExecutor, VulkanPipelineArtifact,
};
use metal_api_core::completion::{CompletionRecord, ObservationDeadline};
pub use metal_api_core::provider::CompiledComputePipeline;
use metal_api_core::provider::{
    allocate_device_epoch, BufferSource, BufferView, BufferWriteback, CompletionDisposition,
    CompletionReadback, CompletionToken, ComputeProvider, DeviceEpoch, FieldValue,
    FunctionIdentity, FunctionSource, PipelineCompileRequest, PipelineId, PipelineProvider,
    ProviderCapabilities, ProviderError, ProviderErrorClass, ProviderPhase, ProviderSubmission,
    Retryability, SemanticDigest, ShaderSource, SubmissionId, ValidatedComputeTrace,
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

/// One provider identity sharing the standalone executor's Vulkan device owner.
///
/// This implementation admits up to eight serial exact-thread dispatches
/// selecting registered pipelines over an initialized view pool, with owned bytes and
/// host readback. Each pass maps a subset of that pool to its pipeline's bindings.
/// By default `submit` waits for GPU completion and readback, and `wait` only
/// observes the recorded terminal result. `with_async_execution(true)` records
/// and submits on the calling thread, returns `Submitted`, and defers the
/// device-fence wait and readback to `wait`/`readback`; no worker is created
/// per submission. Tokens and metadata are process-local, and no-copy leases
/// are refused. Callers can explicitly release registered pipelines and
/// completion records.
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
    /// to a shared reaper that waits for its fence and then releases the
    /// handles. Unknown completion resources stay with the executor's
    /// abandonment policy.
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
            let _ = std::thread::Builder::new()
                .name("vulkan-provider-retire".into())
                .spawn(move || {
                    while let Ok(mut pending) = rx.recv() {
                        let _ = pending.wait(crate::FENCE_TIMEOUT_NS);
                    }
                });
            tx
        });
        let _ = sender.send(pending);
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
        self.executor.context.ensure_usable().map_err(|error| {
            let mut result = refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Resource,
                "provider_unavailable",
            )
            .with_detail(error.to_string());
            result.retryability = Retryability::RetryAfterRecreate;
            result
        })
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

    fn submit(&self, admitted: ValidatedComputeTrace) -> Result<ProviderSubmission, ProviderError> {
        let trace = admitted.trace();
        check_epoch(self.epoch, trace.device_epoch)?;
        // A ValidatedComputeTrace may have been admitted against another
        // capability snapshot. Only the receiving owner can authorize execution.
        self.capabilities.admit(trace, admitted.resources())?;
        let artifacts = {
            let registry = self.pipelines.lock().map_err(|_| registry_poisoned())?;
            trace
                .passes
                .iter()
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
        for pass in &trace.passes {
            let grid = narrow_dimensions(pass.dispatch.grid)?.dimensions();
            let local = narrow_dimensions(pass.dispatch.threads_per_threadgroup)?.dimensions();
            let bindings = pass
                .buffers
                .iter()
                .map(|view| {
                    let position = pool
                        .iter()
                        .position(|resource| resource.view_id == view.view_id)
                        .expect("validated resource pool");
                    (view.metal_binding, position as u32)
                })
                .collect();
            dispatches.push(BoundDispatch {
                grid,
                local,
                bindings,
            });
        }
        let buffers = pool
            .iter()
            .enumerate()
            .map(|(position, resource)| {
                let BufferSource::OwnedBytes(bytes) = &resource.source else {
                    return Err(refusal(
                        ProviderPhase::Resolve,
                        ProviderErrorClass::Capability,
                        "storage_mode_unsupported",
                    ));
                };
                Ok(BufferBinding {
                    // The validated pool has at most 64 resources. First-use
                    // Metal binding labels may repeat across different passes.
                    index: position as u32,
                    bytes: bytes.clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let token = CompletionToken {
            submission_id: SubmissionId::new(next_identity(
                &self.next_submission,
                "submission_identity_exhausted",
            )?),
            device_epoch: self.epoch,
        };
        if self.async_execution {
            return self.submit_async(pool, artifacts, buffers, dispatches, token);
        }
        let result = execute_on_context(&self.executor, &artifacts, buffers, &dispatches)
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
            Ok(output) => Some(CompletionRecord::completed(output.writebacks.clone())),
            Err(error) if error.completion.token().is_some() => {
                Some(CompletionRecord::failed(error.clone()))
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
        result
    }

    fn wait(
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
            drop(pending);
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
                    drop(pending);
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
    fn submit_async(
        &self,
        pool: Vec<BufferView>,
        artifacts: Vec<Arc<VulkanPipelineArtifact>>,
        buffers: Vec<BufferBinding>,
        dispatches: Vec<BoundDispatch>,
        token: CompletionToken,
    ) -> Result<ProviderSubmission, ProviderError> {
        let pending = {
            let _execution = self
                .executor
                .context
                .execution_lock
                .lock()
                .map_err(|_| registry_poisoned())?;
            ensure_executor_usable(&self.executor)?;
            PendingExecution::submit(&self.executor.context, &artifacts, &buffers, &dispatches)?
        };
        let record = CompletionRecord::running();
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

/// Serialize device work with the standalone executor, then run the prepared
/// sequence. The worker path calls this directly; the synchronous path calls
/// it on the submitting thread.
fn execute_on_context(
    executor: &Arc<VulkanExecutor>,
    artifacts: &[Arc<VulkanPipelineArtifact>],
    buffers: Vec<BufferBinding>,
    dispatches: &[BoundDispatch],
) -> Result<Vec<BufferUpdate>, ProviderError> {
    let _execution = executor
        .context
        .execution_lock
        .lock()
        .map_err(|_| registry_poisoned())?;
    ensure_executor_usable(executor)?;
    execute_pipeline_sequence_with_status(&executor.context, artifacts, buffers, dispatches)
}

fn ensure_executor_usable(executor: &VulkanExecutor) -> Result<(), ProviderError> {
    executor.context.ensure_usable().map_err(|error| {
        let mut result = refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Resource,
            "provider_unavailable",
        )
        .with_detail(error.to_string());
        result.retryability = Retryability::RetryAfterRecreate;
        result
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::provider::{AllocationId, BufferAccess, ViewId};

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
