//! Experimental synchronous compute objects over the shared provider contract.
//!
//! Dispatch records freeze pipeline and view bindings. Buffer contents are read
//! once at commit, when the complete command is admitted and submitted once.
//! Buffer locks are held through synchronous execution and checked writeback;
//! concurrent CPU access and commands using those buffers wait for that boundary.
//! This module does not extend the older [`crate::ComputeExecutor`] object API.

use crate::provider::{
    self as contract, AllocationId, AllocationRecord, BufferSource, CompiledComputePipeline,
    CompletionDisposition, CompletionPolicy, CompletionToken, ContractError, Dispatch,
    DispatchKind, DispatchType, OperationId, PipelineCompileRequest, PipelineId, PipelineProvider,
    ProviderCapabilities, ProviderError, ProviderSubmission, ResourceTableSnapshot, ViewId,
    MAX_SERIAL_RESOURCES, PROVIDER_SCHEMA_VERSION,
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
    IdentityExhausted,
    InvalidPipelineMetadata,
    PassLimit { requested: usize, maximum: usize },
    ProviderPanicked,
    SynchronousCompletionRequired,
    CompletionObservationMismatch,
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
            Self::IdentityExhausted => f.write_str("object identity space exhausted"),
            Self::InvalidPipelineMetadata => {
                f.write_str("provider returned inconsistent pipeline metadata")
            }
            Self::PassLimit { requested, maximum } => {
                write!(f, "command has {requested} passes; maximum is {maximum}")
            }
            Self::ProviderPanicked => f.write_str("provider panicked"),
            Self::SynchronousCompletionRequired => {
                f.write_str("object API requires synchronous completed host readback")
            }
            Self::CompletionObservationMismatch => {
                f.write_str("provider wait disagrees with submitted completion")
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
            }),
        })
    }

    pub fn new_command_queue(&self) -> CommandQueue {
        CommandQueue {
            owner: Arc::clone(&self.state),
        }
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

struct BufferInner {
    owner: Arc<DeviceState>,
    allocation_id: AllocationId,
    length: usize,
    bytes: Mutex<Vec<u8>>,
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
        Ok(lock(&self.inner.bytes, "provider buffer")?.clone())
    }
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        let end = checked_range(offset, bytes.len(), self.inner.length)?;
        lock(&self.inner.bytes, "provider buffer")?[offset..end].copy_from_slice(bytes);
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
                    recording_error: None,
                    status: CommandBufferStatus::Recording,
                    failure: None,
                    submission: None,
                    completion: None,
                }),
                completion: Condvar::new(),
            }),
        }
    }
}

#[derive(Clone)]
struct RecordedPass {
    pipeline: Pipeline,
    buffers: BTreeMap<u32, BufferView>,
    dispatch: Dispatch,
}
struct CommandInner {
    passes: Vec<RecordedPass>,
    encoder_open: bool,
    recording_error: Option<Error>,
    status: CommandBufferStatus,
    failure: Option<Error>,
    submission: Option<ProviderSubmission>,
    completion: Option<CompletionToken>,
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

/// Single-use synchronous command buffer. No buffer bytes change if admission,
/// execution, completion observation or writeback validation fails.
pub struct CommandBuffer {
    shared: Arc<CommandShared>,
}
impl CommandBuffer {
    pub fn status(&self) -> Result<CommandBufferStatus, Error> {
        Ok(lock(&self.shared.inner, "provider command")?.status)
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
            dispatch_count: 0,
            ended: false,
        })
    }

    pub fn commit(&self) -> Result<(), Error> {
        let passes = {
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
            inner.passes.clone()
        };
        let mut token = None;
        let result = catch_unwind(AssertUnwindSafe(|| self.execute(&passes, &mut token)))
            .unwrap_or(Err(Error::ProviderPanicked));
        let mut inner = lock(&self.shared.inner, "provider command")?;
        inner.completion = token;
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

    fn execute(
        &self,
        passes: &[RecordedPass],
        token: &mut Option<CompletionToken>,
    ) -> Result<ProviderSubmission, Error> {
        let owner = &self.shared.owner;
        let mut buffers = BTreeMap::<AllocationId, &Buffer>::new();
        for pass in passes {
            for view in pass.buffers.values() {
                buffers.insert(view.allocation_id(), &view.buffer);
            }
        }
        // All commands acquire their complete union in allocation order. Keep
        // guards outside provider_call's unwind boundary so a provider panic
        // cannot poison otherwise unchanged host storage.
        let mut guards = Vec::with_capacity(buffers.len());
        let mut positions = BTreeMap::new();
        let mut resources = ResourceTableSnapshot::new();
        for (id, buffer) in buffers {
            positions.insert(id, guards.len());
            guards.push(lock(&buffer.inner.bytes, "provider buffer")?);
            resources.insert_allocation(AllocationRecord {
                allocation_id: id,
                owner_epoch: owner.epoch,
                size: buffer.inner.length as u64,
            })?;
        }
        let mut pipelines = BTreeMap::<PipelineId, CompiledComputePipeline>::new();
        let mut trace_passes = Vec::with_capacity(passes.len());
        for pass in passes {
            let metadata = pass.pipeline.metadata();
            if let Some(previous) = pipelines.insert(metadata.pipeline_id, metadata.clone()) {
                if previous != *metadata {
                    return Err(Error::InvalidPipelineMetadata);
                }
            }
            let mut views = Vec::with_capacity(pass.buffers.len());
            for (binding, view) in &pass.buffers {
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
            trace_passes.push(contract::ComputePass {
                pipeline: metadata.pipeline_id,
                buffers: views,
                dispatch: pass.dispatch,
            });
        }
        let trace = contract::ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: owner.epoch,
            operation_id: OperationId::new(next_id()?),
            pipelines: pipelines.into_values().collect(),
            encoder_dispatch_type: DispatchType::Serial,
            passes: trace_passes,
            completion_policy: CompletionPolicy::HostReadback,
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
        if !matches!(
            submission.completion,
            CompletionDisposition::CompletedVisible { .. }
        ) {
            return Err(Error::SynchronousCompletionRequired);
        }
        let completed_token = token.ok_or(Error::SynchronousCompletionRequired)?;
        let observed = provider_call(|| owner.provider.wait(completed_token, Duration::ZERO))?;
        if observed != submission.completion {
            return Err(Error::CompletionObservationMismatch);
        }
        let mut writes = Vec::with_capacity(submission.writebacks.len());
        for write in &submission.writebacks {
            let position = *positions
                .get(&write.allocation_id)
                .ok_or(ContractError::UnknownAllocation(write.allocation_id))?;
            let offset = usize::try_from(write.offset)
                .map_err(|_| ContractError::ArithmeticOverflow("writeback offset"))?;
            let end = checked_range(offset, write.bytes.len(), guards[position].len())?;
            writes.push((position, offset, end, &write.bytes));
        }
        // Every recoverable failure precedes the first copy.
        for (position, offset, end, bytes) in writes {
            guards[position][offset..end].copy_from_slice(bytes);
        }
        Ok(submission)
    }

    pub fn wait_until_completed(&self) -> Result<(), Error> {
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
                    inner = self
                        .shared
                        .completion
                        .wait(inner)
                        .map_err(|_| ApiError::StatePoisoned("provider command"))?;
                }
            }
        }
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
    dispatch_count: usize,
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
        if let Some((first, _)) = self.buffers.iter().find(|(other, bound)| {
            **other != index && bound.allocation_id() == view.allocation_id()
        }) {
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
    pub fn dispatch_threads(&mut self, grid: Size, local: Size) -> Result<(), Error> {
        self.ensure_open()?;
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
        let unique = inner
            .passes
            .iter()
            .flat_map(|pass| pass.buffers.values())
            .chain(self.buffers.values())
            .map(|view| view.view_id)
            .collect::<BTreeSet<_>>();
        if unique.len() > MAX_SERIAL_RESOURCES {
            return Err(ContractError::SerialResourceLimit {
                requested: unique.len(),
                maximum: MAX_SERIAL_RESOURCES,
            }
            .into());
        }
        inner.passes.push(RecordedPass {
            pipeline: pipeline.clone(),
            buffers: self.buffers.clone(),
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: grid.dimensions().map(u64::from),
                threads_per_threadgroup: local.dimensions().map(u64::from),
            },
        });
        self.dispatch_count += 1;
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

#[cfg(test)]
mod tests;
