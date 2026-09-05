//! Metal handles stay behind one lock. No guest pointers or caller-owned
//! memory are passed to Metal; submission copies admitted view contents.

use crate::{bounded_contract, refusal, unknown_completion};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    Buffer, CommandBuffer, CommandBufferRef, CommandQueue, ComputeCommandEncoderRef,
    ComputePipelineState, Device, MTLCommandBufferStatus, MTLGPUFamily, MTLHazardTrackingMode,
    MTLResourceOptions, MTLSize,
};
use metal_api_core::provider::*;
use objc::{msg_send, runtime::Object, sel, sel_impl};
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

const GPU_DEADLINE: Duration = Duration::from_secs(20);

#[derive(Clone)]
struct RegisteredPipeline {
    metadata: CompiledComputePipeline,
    pipeline: ComputePipelineState,
}

struct State {
    device: Device,
    queue: CommandQueue,
    pipelines: BTreeMap<PipelineId, RegisteredPipeline>,
    completions: BTreeMap<SubmissionId, Result<CompletionDisposition, ProviderError>>,
    next_pipeline: u64,
    next_submission: u64,
    abandoned: bool,
}

/// Synchronous native provider with at most eight serial passes over one buffer
/// pool, allowing pipeline changes and binding permutations for exact fixtures.
///
/// A submission has a 20-second observation deadline. Unknown retirement or a
/// GPU error permanently disables new work in this context and retains its
/// submitted backing until process exit. `wait` reads the recorded terminal
/// observation; releasing that record does not retire GPU resources.
pub struct NativeMetalProvider {
    epoch: DeviceEpoch,
    name: String,
    capabilities: ProviderCapabilities,
    state: Mutex<State>,
}

impl NativeMetalProvider {
    pub fn new() -> Result<Self, ProviderError> {
        objc::rc::autoreleasepool(|| {
            let device = Device::system_default().ok_or_else(|| {
                refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Capability,
                    "native_metal_device_unavailable",
                )
            })?;
            if device.name().trim().is_empty()
                || !device.has_unified_memory()
                || !device.supports_family(MTLGPUFamily::Apple4)
            {
                return Err(refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Capability,
                    "native_metal_device_ineligible",
                )
                .with_detail(
                    "requires a named device with unified memory and Apple GPU family 4",
                ));
            }
            // newCommandQueue is retained, and nil is checked before wrapping.
            let queue = unsafe {
                let pointer: *mut metal::MTLCommandQueue =
                    msg_send![device.as_ref(), newCommandQueue];
                if pointer.is_null() {
                    return Err(resource_error("command_queue_allocation_failed"));
                }
                CommandQueue::from_ptr(pointer)
            };
            let dimensions = device.max_threads_per_threadgroup();
            let local = [dimensions.width, dimensions.height, dimensions.depth];
            let capabilities = ProviderCapabilities {
                max_passes: 8,
                supports_threads_exact: true,
                supports_threadgroups: false,
                supports_serial: true,
                supports_concurrent: false,
                max_local_size: local,
                max_invocations: local.into_iter().fold(1_u64, u64::saturating_mul).min(1024),
                max_group_count: [1024; 3],
                max_storage_buffer_descriptors: 31,
                max_buffer_range: device.max_buffer_length().min(1024 * 1024),
                max_push_constant_bytes: 0,
                alias_mode: AliasMode::Refused,
                storage_modes: vec![StorageMode::OwnedBytes],
                host_readback: true,
                submit_only: false,
            };
            Ok(Self {
                epoch: allocate_device_epoch()?,
                name: device.name().into(),
                capabilities,
                state: Mutex::new(State {
                    device,
                    queue,
                    pipelines: BTreeMap::new(),
                    completions: BTreeMap::new(),
                    next_pipeline: 1,
                    next_submission: 1,
                    abandoned: false,
                }),
            })
        })
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    fn lock(&self) -> Result<MutexGuard<'_, State>, ProviderError> {
        self.state.lock().map_err(|_| {
            refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Internal,
                "provider_registry_poisoned",
            )
        })
    }

    fn check_epoch(&self, epoch: DeviceEpoch) -> Result<(), ProviderError> {
        if epoch != self.epoch {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Resource,
                "device_epoch_mismatch",
            )
            .with_field("expected", FieldValue::Unsigned(self.epoch.get()))
            .with_field("actual", FieldValue::Unsigned(epoch.get())));
        }
        Ok(())
    }

    fn check_token(&self, token: CompletionToken) -> Result<(), ProviderError> {
        self.check_epoch(token.device_epoch)?;
        token.validate().map_err(|error| {
            refusal(
                ProviderPhase::Wait,
                ProviderErrorClass::Args,
                "invalid_completion_token",
            )
            .with_detail(error.to_string())
        })
    }
}

impl PipelineProvider for NativeMetalProvider {
    fn device_epoch(&self) -> DeviceEpoch {
        self.epoch
    }

    fn compile(
        &self,
        request: PipelineCompileRequest,
    ) -> Result<CompiledComputePipeline, ProviderError> {
        let contract = bounded_contract(&request)?;
        let mut state = self.lock()?;
        state.ensure_usable()?;
        objc::rc::autoreleasepool(|| {
            let ShaderSource::MetalSource(source) = &request.source else {
                unreachable!("checked bounded source")
            };
            let options = metal::CompileOptions::new();
            let library = state
                .device
                .new_library_with_source(source, &options)
                .map_err(|error| compile_error("metal_source_compile_failed").with_detail(error))?;
            let function = library
                .get_function(&request.entry_name, None)
                .map_err(|error| {
                    compile_error("metal_function_resolution_failed").with_detail(error)
                })?;
            let pipeline = unsafe {
                let mut error: *mut Object = std::ptr::null_mut();
                let pointer: *mut metal::MTLComputePipelineState = msg_send![state.device.as_ref(),
                    newComputePipelineStateWithFunction:function.as_ref() error:&mut error];
                if pointer.is_null() {
                    return Err(compile_error("metal_pipeline_compile_failed")
                        .with_detail(error_description(error)));
                }
                ComputePipelineState::from_ptr(pointer)
            };
            let metadata = CompiledComputePipeline {
                device_epoch: self.epoch,
                pipeline_id: PipelineId::new(next_id(&mut state.next_pipeline)?),
                function: FunctionIdentity {
                    logical_digest: request.logical_digest,
                    entry_name: request.entry_name,
                    source: FunctionSource::MetalSource,
                },
                contract,
            };
            state.pipelines.insert(
                metadata.pipeline_id,
                RegisteredPipeline {
                    metadata: metadata.clone(),
                    pipeline,
                },
            );
            Ok(metadata)
        })
    }

    fn release_pipeline(&self, pipeline: &CompiledComputePipeline) -> Result<(), ProviderError> {
        self.check_epoch(pipeline.device_epoch)?;
        let mut state = self.lock()?;
        let registered = state
            .pipelines
            .get(&pipeline.pipeline_id)
            .ok_or_else(|| unknown_pipeline(pipeline.pipeline_id))?;
        if registered.metadata != *pipeline {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Resource,
                "pipeline_identity_mismatch",
            ));
        }
        state.pipelines.remove(&pipeline.pipeline_id);
        Ok(())
    }

    fn release_completion(&self, token: CompletionToken) -> Result<(), ProviderError> {
        self.check_token(token)?;
        let _record = self
            .lock()?
            .completions
            .remove(&token.submission_id)
            .ok_or_else(|| unknown_completion(token))?;
        Ok(())
    }
}

impl ComputeProvider for NativeMetalProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities.clone()
    }

    fn submit(&self, admitted: ValidatedComputeTrace) -> Result<ProviderSubmission, ProviderError> {
        let trace = admitted.trace();
        self.check_epoch(trace.device_epoch)?;
        self.capabilities.admit(trace, admitted.resources())?;
        let mut state = self.lock()?;
        state.ensure_usable()?;
        // Resolve and retain every pass's pipeline under the same registry
        // lock, checking all metadata and local limits before GPU allocation.
        let mut pipelines = Vec::with_capacity(trace.passes.len());
        for (pass_index, pass) in trace.passes.iter().enumerate() {
            let metadata = trace.pipeline(pass.pipeline).map_err(|error| {
                refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Args,
                    "pipeline_table_invalid",
                )
                .with_detail(error.to_string())
            })?;
            let registered = state
                .pipelines
                .get(&pass.pipeline)
                .ok_or_else(|| unknown_pipeline(pass.pipeline))?;
            if metadata.function != registered.metadata.function {
                return Err(refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Resource,
                    "pipeline_function_mismatch",
                ));
            }
            if metadata.contract != registered.metadata.contract {
                return Err(refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Resource,
                    "pipeline_contract_mismatch",
                ));
            }
            if metadata != &registered.metadata {
                return Err(refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Resource,
                    "pipeline_identity_mismatch",
                ));
            }
            if pass.buffers.iter().any(|view| view.metal_binding > 30) {
                return Err(refusal(
                    ProviderPhase::Encode,
                    ProviderErrorClass::Capability,
                    "metal_buffer_binding_limit",
                )
                .with_field("pass_index", FieldValue::Unsigned(pass_index as u64)));
            }
            let local = pass.dispatch.threads_per_threadgroup;
            let invocations = local.into_iter().try_fold(1_u64, u64::checked_mul);
            if invocations
                .is_none_or(|value| value > registered.pipeline.max_total_threads_per_threadgroup())
            {
                return Err(refusal(
                    ProviderPhase::Encode,
                    ProviderErrorClass::Capability,
                    "pipeline_local_size_limit",
                )
                .with_field("pass_index", FieldValue::Unsigned(pass_index as u64)));
            }
            pipelines.push(registered.pipeline.clone());
        }
        let token = CompletionToken {
            device_epoch: self.epoch,
            submission_id: SubmissionId::new(next_id(&mut state.next_submission)?),
        };
        let result = objc::rc::autoreleasepool(|| execute(&mut state, trace, pipelines, token));
        let observation = match &result {
            Ok(submission) => Some(Ok(submission.completion)),
            Err(error) if error.completion.token().is_some() => Some(Err(error.clone())),
            Err(_) => None,
        };
        if let Some(observation) = observation {
            state.completions.insert(token.submission_id, observation);
        }
        result
    }

    fn wait(
        &self,
        token: CompletionToken,
        _timeout: Duration,
    ) -> Result<CompletionDisposition, ProviderError> {
        self.check_token(token)?;
        self.lock()?
            .completions
            .get(&token.submission_id)
            .cloned()
            .ok_or_else(|| unknown_completion(token))?
    }
}

impl State {
    fn ensure_usable(&self) -> Result<(), ProviderError> {
        if self.abandoned {
            let mut error = resource_error("provider_unavailable");
            error.retryability = Retryability::RetryAfterRecreate;
            return Err(error);
        }
        Ok(())
    }
}

struct SubmissionResources {
    // Retain the whole context as well as the explicit command dependencies.
    _device: Device,
    _queue: CommandQueue,
    pipelines: Vec<ComputePipelineState>,
    command: CommandBuffer,
    buffers: Vec<Buffer>,
}

struct PendingSubmission {
    resources: Option<SubmissionResources>,
    submitted: bool,
}

impl Drop for PendingSubmission {
    fn drop(&mut self) {
        if self.submitted {
            // Also protects an unwind during commit/status observation. A
            // poisoned mutex prevents another submit after such an unwind.
            if let Some(resources) = self.resources.take() {
                std::mem::forget(resources);
            }
        }
    }
}

fn execute(
    state: &mut State,
    trace: &ComputeTrace,
    pipelines: Vec<ComputePipelineState>,
    token: CompletionToken,
) -> Result<ProviderSubmission, ProviderError> {
    let pool = trace.serial_resources().map_err(|error| {
        refusal(
            ProviderPhase::Encode,
            ProviderErrorClass::Args,
            "serial_buffer_pool_invalid",
        )
        .with_detail(error.to_string())
    })?;
    let pool_positions: BTreeMap<_, _> = pool
        .iter()
        .enumerate()
        .map(|(index, view)| (view.view_id, index))
        .collect();
    let mut buffers = Vec::with_capacity(pool.len());
    for view in &pool {
        let BufferSource::OwnedBytes(bytes) = &view.source else {
            return Err(refusal(
                ProviderPhase::Encode,
                ProviderErrorClass::Capability,
                "storage_mode_unsupported",
            ));
        };
        // OwnedBytes contains the view itself, not the entire logical
        // allocation. Binding offset is zero; writebacks retain view.offset.
        let buffer = unsafe {
            let pointer: *mut metal::MTLBuffer = msg_send![state.device.as_ref(),
                newBufferWithBytes:bytes.as_ptr().cast::<std::ffi::c_void>()
                length:view.length options:MTLResourceOptions::StorageModeShared];
            if pointer.is_null() {
                return Err(resource_error("metal_buffer_allocation_failed"));
            }
            Buffer::from_ptr(pointer)
        };
        if buffer.contents().is_null() {
            return Err(resource_error("metal_buffer_mapping_failed"));
        }
        // MTLDevice-created resources default to tracked hazards. This is the
        // ordering guarantee used by the directly bound serial passes below.
        if buffer.hazard_tracking_mode() != MTLHazardTrackingMode::Tracked {
            return Err(resource_error("metal_buffer_hazard_tracking_unavailable"));
        }
        buffers.push(buffer);
    }
    let command = unsafe {
        let pointer: *mut metal::MTLCommandBuffer = msg_send![state.queue.as_ref(), commandBuffer];
        if pointer.is_null() {
            return Err(resource_error("metal_command_buffer_allocation_failed"));
        }
        // commandBuffer is autoreleased, so retain it for the pending guard.
        CommandBufferRef::from_ptr(pointer).to_owned()
    };
    let mut pending = PendingSubmission {
        resources: Some(SubmissionResources {
            _device: state.device.clone(),
            _queue: state.queue.clone(),
            pipelines,
            command,
            buffers,
        }),
        submitted: false,
    };
    let resources = pending.resources.as_ref().expect("pending resources");
    // The default compute encoder dispatches serially. Directly bound tracked
    // resources on MTLCommandQueue carry writes across encoder boundaries:
    // https://developer.apple.com/documentation/metal/resource-synchronization
    // Each pass sees earlier writes; the initial bytes are uploaded only once.
    for (pass_index, pass) in trace.passes.iter().enumerate() {
        let encoder = unsafe {
            let pointer: *mut metal::MTLComputeCommandEncoder =
                msg_send![resources.command.as_ref(), computeCommandEncoder];
            if pointer.is_null() {
                return Err(resource_error("metal_encoder_allocation_failed"));
            }
            ComputeCommandEncoderRef::from_ptr(pointer)
        };
        encoder.set_compute_pipeline_state(&resources.pipelines[pass_index]);
        for view in &pass.buffers {
            // serial_resources validated that each pass permutes this pool.
            let buffer = &resources.buffers[pool_positions[&view.view_id]];
            encoder.set_buffer(u64::from(view.metal_binding), Some(buffer), 0);
        }
        let [gx, gy, gz] = pass.dispatch.grid;
        let [lx, ly, lz] = pass.dispatch.threads_per_threadgroup;
        encoder.dispatch_threads(MTLSize::new(gx, gy, gz), MTLSize::new(lx, ly, lz));
        encoder.end_encoding();
    }
    pending.submitted = true;
    resources.command.commit();
    let started = Instant::now();
    loop {
        match resources.command.status() {
            MTLCommandBufferStatus::Completed => break,
            MTLCommandBufferStatus::Error => {
                state.abandoned = true;
                let detail = unsafe {
                    let error: *mut Object = msg_send![resources.command.as_ref(), error];
                    error_description(error)
                };
                return Err(refusal(
                    ProviderPhase::Wait,
                    ProviderErrorClass::Execute,
                    "metal_command_failed",
                )
                .with_detail(detail)
                .with_completion(CompletionDisposition::Failed { token: Some(token) }));
            }
            _ if started.elapsed() >= GPU_DEADLINE => {
                state.abandoned = true;
                return Err(refusal(
                    ProviderPhase::Wait,
                    ProviderErrorClass::Execute,
                    "metal_completion_unknown",
                )
                .with_completion(CompletionDisposition::SubmittedUnknown { token: Some(token) }));
            }
            _ => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    // Shared memory on the admitted device is now CPU visible. Only a known
    // completed command permits the guard to release its backing resources.
    pending.submitted = false;
    let mut writebacks = Vec::new();
    for (view, buffer) in pool.iter().zip(&resources.buffers) {
        if view.access.is_writable() {
            let bytes = unsafe {
                // Admission bounded length to 1 MiB, contents was checked
                // before commit, and this completed buffer remains retained.
                std::slice::from_raw_parts(buffer.contents().cast::<u8>(), view.length as usize)
                    .to_vec()
            };
            writebacks.push(BufferWriteback {
                view_id: view.view_id,
                allocation_id: view.allocation_id,
                offset: view.offset,
                bytes,
            });
        }
    }
    writebacks.sort_by_key(|writeback| (writeback.allocation_id, writeback.view_id));
    let submission = ProviderSubmission {
        completion: CompletionDisposition::CompletedVisible { token },
        writebacks,
    };
    submission.validate_for_trace(trace).map_err(|error| {
        refusal(
            ProviderPhase::Readback,
            ProviderErrorClass::Internal,
            "writeback_contract_invalid",
        )
        .with_detail(error.to_string())
        .with_completion(CompletionDisposition::Failed { token: Some(token) })
    })?;
    Ok(submission)
}

fn next_id(counter: &mut u64) -> Result<u64, ProviderError> {
    let value = *counter;
    *counter = counter.checked_add(1).ok_or_else(|| {
        refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Internal,
            "provider_identity_exhausted",
        )
    })?;
    Ok(value)
}

fn compile_error(slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Compile, ProviderErrorClass::Compile, slug)
}
fn resource_error(slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Resolve, ProviderErrorClass::Resource, slug)
}
fn unknown_pipeline(id: PipelineId) -> ProviderError {
    refusal(
        ProviderPhase::Resolve,
        ProviderErrorClass::Resource,
        "unknown_pipeline",
    )
    .with_field("pipeline", FieldValue::Unsigned(id.get()))
}

/// Called only inside an autorelease pool with nil or a live NSError pointer.
unsafe fn error_description(error: *mut Object) -> String {
    if error.is_null() {
        return "Metal returned no error description".into();
    }
    let description: *mut Object = msg_send![error, localizedDescription];
    if description.is_null() {
        return "Metal returned no error description".into();
    }
    let bytes: *const std::ffi::c_char = msg_send![description, UTF8String];
    if bytes.is_null() {
        return "Metal returned no error description".into();
    }
    CStr::from_ptr(bytes).to_string_lossy().into_owned()
}
