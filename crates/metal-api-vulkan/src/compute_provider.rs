//! Synchronous, owned-byte implementation of the first compute provider slice.

use crate::{
    execute_submission_with_status, TranslatedComputePipeline, VulkanExecutor,
    VulkanPipelineArtifact,
};
use metal_api_core::provider::{
    BufferSource, BufferWriteback, CompletionDisposition, CompletionToken, ComputeProvider,
    ComputeTrace, DeviceEpoch, FieldValue, FunctionIdentity, FunctionSource, PipelineContract,
    PipelineId, ProviderCapabilities, ProviderError, ProviderErrorClass, ProviderPhase,
    ProviderSubmission, Retryability, SemanticDigest, SubmissionId, ValidatedComputeTrace,
};
use metal_api_core::{AirSource, BufferBinding, BufferUpdate, ComputeSubmission, Function, Size};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

static NEXT_DEVICE_EPOCH: AtomicU64 = AtomicU64::new(1);
const TRANSLATOR_REVISION: &[u8] = b"9e0e99a41dc3cb8bb7e288b531f1698a79fd4b1c";

/// Neutral metadata for a pipeline registered by one provider context.
/// The provider retains the actual artifact and checks this metadata at submit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledComputePipeline {
    pub device_epoch: DeviceEpoch,
    pub pipeline_id: PipelineId,
    pub function: FunctionIdentity,
    pub contract: PipelineContract,
}

struct RegisteredPipeline {
    metadata: CompiledComputePipeline,
    artifact: Arc<VulkanPipelineArtifact>,
}

/// One provider identity sharing the standalone executor's Vulkan device owner.
///
/// This implementation admits one serial exact-thread dispatch with owned
/// bytes and host readback. `submit` waits for GPU completion and readback;
/// `wait` only observes the recorded terminal result. Tokens and metadata are
/// process-local, and no-copy leases and asynchronous submission are refused.
/// Callers can explicitly release registered pipelines and completion records.
pub struct VulkanComputeProvider {
    executor: Arc<VulkanExecutor>,
    epoch: DeviceEpoch,
    capabilities: ProviderCapabilities,
    next_pipeline: AtomicU64,
    next_submission: AtomicU64,
    pipelines: Mutex<BTreeMap<PipelineId, Arc<RegisteredPipeline>>>,
    completions: Mutex<BTreeMap<SubmissionId, Result<CompletionDisposition, ProviderError>>>,
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
        let epoch = DeviceEpoch::new(next_identity(&NEXT_DEVICE_EPOCH, "device_epoch_exhausted")?);
        let capabilities = executor.provider_capabilities();
        Ok(Self {
            executor,
            epoch,
            capabilities,
            next_pipeline: AtomicU64::new(1),
            next_submission: AtomicU64::new(1),
            pipelines: Mutex::new(BTreeMap::new()),
            completions: Mutex::new(BTreeMap::new()),
        })
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
    pub fn release_pipeline(&self, pipeline: PipelineId) -> Result<(), ProviderError> {
        self.pipelines
            .lock()
            .map_err(|_| registry_poisoned())?
            .remove(&pipeline)
            .ok_or_else(|| unknown_pipeline(pipeline))?;
        Ok(())
    }

    /// Forget a terminal observation. This releases no GPU resources; unknown
    /// completion resources stay with the executor's abandonment policy.
    pub fn release_completion(&self, token: CompletionToken) -> Result<(), ProviderError> {
        self.validate_token(token)?;
        let _observation = self
            .completions
            .lock()
            .map_err(|_| registry_poisoned())?
            .remove(&token.submission_id)
            .ok_or_else(|| unknown_completion(token))?;
        Ok(())
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
        let pass = &trace.passes[0];
        let registered = self
            .pipelines
            .lock()
            .map_err(|_| registry_poisoned())?
            .get(&pass.pipeline)
            .cloned()
            .ok_or_else(|| unknown_pipeline(pass.pipeline))?;
        validate_pipeline_identity(trace, &registered.metadata)?;
        let grid = narrow_dimensions(pass.dispatch.grid)?;
        let local = narrow_dimensions(pass.dispatch.threads_per_threadgroup)?;
        let buffers = pass
            .buffers
            .iter()
            .map(|view| {
                let BufferSource::OwnedBytes(bytes) = &view.source else {
                    return Err(refusal(
                        ProviderPhase::Resolve,
                        ProviderErrorClass::Capability,
                        "storage_mode_unsupported",
                    ));
                };
                Ok(BufferBinding {
                    index: view.metal_binding,
                    bytes: bytes.clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let submission = ComputeSubmission {
            pipeline: registered.artifact.clone(),
            buffers,
            threads_per_grid: grid,
            threads_per_threadgroup: local,
        };
        let token = CompletionToken {
            submission_id: SubmissionId::new(next_identity(
                &self.next_submission,
                "submission_identity_exhausted",
            )?),
            device_epoch: self.epoch,
        };
        let result = {
            let _execution = self
                .executor
                .context
                .execution_lock
                .lock()
                .map_err(|_| registry_poisoned())?;
            self.ensure_usable()?;
            execute_submission_with_status(
                &self.executor.context,
                registered.artifact.clone(),
                submission,
            )
        };
        let result = result
            .and_then(|updates| {
                let writebacks = map_writebacks(trace, updates, token)?;
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
            Ok(output) => Some(Ok(output.completion)),
            Err(error) if error.completion.token().is_some() => Some(Err(error.clone())),
            Err(_) => None,
        };
        if let Some(observation) = observation {
            self.completions
                .lock()
                .map_err(|_| {
                    let completion = match &observation {
                        Ok(disposition) => *disposition,
                        Err(error) => error.completion,
                    };
                    registry_poisoned().with_completion(completion)
                })?
                .insert(token.submission_id, observation);
        }
        result
    }

    fn wait(
        &self,
        token: CompletionToken,
        _timeout: Duration,
    ) -> Result<CompletionDisposition, ProviderError> {
        // Every token made observable by this synchronous provider is terminal.
        // A timed-out underlying submit is SubmittedUnknown, not a waitable job.
        self.validate_token(token)?;
        self.completions
            .lock()
            .map_err(|_| registry_poisoned())?
            .get(&token.submission_id)
            .cloned()
            .ok_or_else(|| unknown_completion(token))?
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
    trace: &ComputeTrace,
    metadata: &CompiledComputePipeline,
) -> Result<(), ProviderError> {
    check_epoch(metadata.device_epoch, trace.device_epoch)?;
    if trace.passes.first().map(|p| p.pipeline) != Some(metadata.pipeline_id) {
        return Err(refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Resource,
            "pipeline_identity_mismatch",
        ));
    }
    if trace.function != metadata.function {
        return Err(refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Resource,
            "pipeline_function_mismatch",
        ));
    }
    if trace.pipeline_contract != metadata.contract {
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
    trace: &ComputeTrace,
    updates: Vec<BufferUpdate>,
    token: CompletionToken,
) -> Result<Vec<BufferWriteback>, ProviderError> {
    let mut writebacks = Vec::with_capacity(updates.len());
    for update in updates {
        let view = trace.passes[0]
            .buffers
            .iter()
            .find(|v| v.metal_binding == update.index)
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
