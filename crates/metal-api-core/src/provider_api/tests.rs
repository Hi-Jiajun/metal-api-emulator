use super::*;
use crate::provider::{
    allocate_device_epoch, AliasMode, BufferAccess, BufferBindingContract, BufferWriteback,
    ComputeProvider, ComputeTrace, FootprintProof, FunctionIdentity, PipelineContract,
    ProviderErrorClass, ProviderPhase, SemanticDigest, ShaderSource, StorageMode, SubmissionId,
    ValidatedComputeTrace,
};
use std::sync::atomic::AtomicUsize;

const GOOD: usize = 0;
const BAD_LAST_WRITE: usize = 1;
const SUBMITTED: usize = 2;
const FAIL: usize = 3;
const PANIC_SUBMIT: usize = 4;
const WRONG_WAIT: usize = 5;
const MISSING_WRITE: usize = 6;
const PANIC_WAIT: usize = 7;
const BAD_METADATA: usize = 8;
const PANIC_RELEASE: usize = 9;

struct FakeProvider {
    epoch: contract::DeviceEpoch,
    mode: AtomicUsize,
    pipelines: Mutex<BTreeSet<PipelineId>>,
    next_pipeline: AtomicU64,
    traces: Mutex<Vec<ComputeTrace>>,
    released_pipelines: AtomicUsize,
    released_completions: Mutex<Vec<CompletionToken>>,
    release_order: Mutex<Vec<&'static str>>,
    gate: Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>,
}

impl FakeProvider {
    fn new() -> Self {
        Self {
            epoch: allocate_device_epoch().unwrap(),
            mode: AtomicUsize::new(GOOD),
            pipelines: Mutex::new(BTreeSet::new()),
            next_pipeline: AtomicU64::new(1),
            traces: Mutex::new(Vec::new()),
            released_pipelines: AtomicUsize::new(0),
            released_completions: Mutex::new(Vec::new()),
            release_order: Mutex::new(Vec::new()),
            gate: None,
        }
    }
    fn error(&self, token: CompletionToken) -> ProviderError {
        ProviderError::new(
            ProviderPhase::Submit,
            ProviderErrorClass::Execute,
            "synthetic_failure",
        )
        .unwrap()
        .with_completion(CompletionDisposition::SubmittedUnknown { token: Some(token) })
    }
}

impl ComputeProvider for FakeProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            max_passes: 8,
            supports_threads_exact: true,
            supports_threadgroups: false,
            supports_serial: true,
            supports_concurrent: false,
            max_local_size: [1024; 3],
            max_invocations: 1024,
            max_group_count: [65535; 3],
            max_storage_buffer_descriptors: 128,
            max_buffer_range: 65536,
            max_push_constant_bytes: 0,
            alias_mode: AliasMode::Refused,
            storage_modes: vec![StorageMode::OwnedBytes],
            host_readback: true,
            submit_only: false,
        }
    }
    fn submit(&self, admitted: ValidatedComputeTrace) -> Result<ProviderSubmission, ProviderError> {
        let trace = admitted.trace();
        for pipeline in &trace.pipelines {
            assert!(self
                .pipelines
                .lock()
                .unwrap()
                .contains(&pipeline.pipeline_id));
        }
        let submission_id = {
            let mut traces = self.traces.lock().unwrap();
            traces.push(trace.clone());
            traces.len() as u64
        };
        let token = CompletionToken {
            submission_id: SubmissionId::new(submission_id),
            device_epoch: self.epoch,
        };
        if let Some((entered, release)) = &self.gate {
            entered.wait();
            release.wait();
        }
        let mode = self.mode.load(Ordering::SeqCst);
        if mode == FAIL {
            return Err(self.error(token));
        }
        if mode == PANIC_SUBMIT {
            panic!("synthetic submit panic");
        }
        if mode == SUBMITTED {
            return Ok(ProviderSubmission {
                completion: CompletionDisposition::Submitted { token },
                writebacks: vec![],
            });
        }
        let resources = trace.serial_resources().unwrap();
        let mut contents = resources
            .iter()
            .map(|view| {
                let BufferSource::OwnedBytes(bytes) = &view.source else {
                    panic!("owned bytes required");
                };
                (view.view_id, bytes.clone())
            })
            .collect::<BTreeMap<_, _>>();
        for pass in &trace.passes {
            let input = pass
                .buffers
                .iter()
                .find(|view| view.access == BufferAccess::Read)
                .map(|view| contents[&view.view_id].clone());
            for view in &pass.buffers {
                if view.access.is_writable() {
                    let bytes = contents.get_mut(&view.view_id).unwrap();
                    if let Some(input) = &input {
                        bytes.copy_from_slice(input);
                    }
                    for byte in bytes {
                        *byte = byte.wrapping_add(1);
                    }
                }
            }
        }
        let mut writebacks = resources
            .iter()
            .filter(|view| view.access.is_writable())
            .map(|view| BufferWriteback {
                allocation_id: view.allocation_id,
                view_id: view.view_id,
                offset: view.offset,
                bytes: contents[&view.view_id].clone(),
            })
            .collect::<Vec<_>>();
        writebacks.sort_by_key(|write| (write.allocation_id, write.view_id));
        if mode == BAD_LAST_WRITE {
            writebacks.last_mut().unwrap().offset += 1;
        }
        if mode == MISSING_WRITE {
            writebacks.pop();
        }
        Ok(ProviderSubmission {
            completion: CompletionDisposition::CompletedVisible { token },
            writebacks,
        })
    }
    fn wait(
        &self,
        token: CompletionToken,
        _timeout: Duration,
    ) -> Result<CompletionDisposition, ProviderError> {
        match self.mode.load(Ordering::SeqCst) {
            PANIC_WAIT => panic!("synthetic wait panic"),
            WRONG_WAIT => Ok(CompletionDisposition::TimedOut { token }),
            _ => Ok(CompletionDisposition::CompletedVisible { token }),
        }
    }
}

impl PipelineProvider for FakeProvider {
    fn device_epoch(&self) -> contract::DeviceEpoch {
        self.epoch
    }
    fn compile(
        &self,
        request: PipelineCompileRequest,
    ) -> Result<CompiledComputePipeline, ProviderError> {
        let bindings = if request.entry_name == "copy" {
            vec![(4, BufferAccess::Read), (9, BufferAccess::Write)]
        } else if let Some(count) = request.entry_name.strip_prefix("wide:") {
            (0..count.parse().unwrap())
                .map(|slot| (slot, BufferAccess::ReadWrite))
                .collect()
        } else {
            vec![(request.entry_name.parse().unwrap(), BufferAccess::ReadWrite)]
        };
        let pipeline_id = PipelineId::new(self.next_pipeline.fetch_add(1, Ordering::SeqCst));
        self.pipelines.lock().unwrap().insert(pipeline_id);
        let mut metadata = CompiledComputePipeline {
            device_epoch: self.epoch,
            pipeline_id,
            function: FunctionIdentity {
                entry_name: request.entry_name,
                logical_digest: request.logical_digest,
                source: request.source.kind(),
            },
            contract: PipelineContract {
                dispatch_kind: DispatchKind::ThreadsExact,
                required_local_size: None,
                fixed_grid: None,
                push_constant_offset: 0,
                push_constant_bytes: 0,
                buffer_bindings: bindings
                    .into_iter()
                    .map(|(metal_binding, access)| BufferBindingContract {
                        metal_binding,
                        access,
                        footprint: FootprintProof::Static { max_bytes: 4 },
                    })
                    .collect(),
                shader_capabilities: vec![],
                translator_revision: None,
            },
        };
        if self.mode.load(Ordering::SeqCst) == BAD_METADATA {
            metadata.function.entry_name = "wrong".into();
        }
        Ok(metadata)
    }
    fn release_pipeline(&self, pipeline: &CompiledComputePipeline) -> Result<(), ProviderError> {
        assert!(self.pipelines.lock().unwrap().remove(&pipeline.pipeline_id));
        self.released_pipelines.fetch_add(1, Ordering::SeqCst);
        self.release_order.lock().unwrap().push("pipeline");
        if self.mode.load(Ordering::SeqCst) == PANIC_RELEASE {
            panic!("synthetic pipeline release panic");
        }
        Ok(())
    }
    fn release_completion(&self, token: CompletionToken) -> Result<(), ProviderError> {
        self.released_completions.lock().unwrap().push(token);
        self.release_order.lock().unwrap().push("completion");
        if self.mode.load(Ordering::SeqCst) == PANIC_RELEASE {
            panic!("synthetic completion release panic");
        }
        Ok(())
    }
}

fn setup() -> (Arc<FakeProvider>, Device) {
    let provider = Arc::new(FakeProvider::new());
    let device = Device::new(provider.clone());
    (provider, device)
}
fn request(entry: &str) -> PipelineCompileRequest {
    PipelineCompileRequest {
        entry_name: entry.into(),
        logical_digest: SemanticDigest::new("fixture", vec![1]).unwrap(),
        source: ShaderSource::SanitizedLl("owned test".into()),
    }
}
fn pipeline(device: &Device, entry: &str) -> Pipeline {
    device.compile_pipeline(request(entry)).unwrap()
}
fn buffer(device: &Device, byte: u8) -> (Buffer, BufferView) {
    let buffer = device.new_buffer_with_bytes(vec![byte; 8]).unwrap();
    let view = buffer.view(2, 4).unwrap();
    (buffer, view)
}
fn dispatch(encoder: &mut ComputeCommandEncoder) -> Result<(), Error> {
    encoder.dispatch_threads(Size::new(1, 1, 1).unwrap(), Size::new(1, 1, 1).unwrap())
}
fn command(device: &Device, pipeline: &Pipeline, views: &[(u32, &BufferView)]) -> CommandBuffer {
    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(pipeline).unwrap();
    for (binding, view) in views {
        encoder.set_buffer(*binding, view).unwrap();
    }
    dispatch(&mut encoder).unwrap();
    encoder.end_encoding().unwrap();
    command
}

#[test]
fn one_submit_preserves_recorded_dispatches_late_views_and_commit_time_contents() {
    let (provider, device) = setup();
    let first = pipeline(&device, "0");
    let copy = pipeline(&device, "copy");
    let last = pipeline(&device, "7");
    let (a, av) = buffer(&device, 1);
    let (b, bv) = buffer(&device, 9);
    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&first).unwrap();
    encoder.set_buffer(0, &av).unwrap();
    dispatch(&mut encoder).unwrap();
    encoder.clear_buffers().unwrap();
    encoder.set_compute_pipeline_state(&copy).unwrap();
    encoder.set_buffer(4, &av).unwrap();
    encoder.set_buffer(9, &bv).unwrap();
    dispatch(&mut encoder).unwrap();
    encoder.clear_buffers().unwrap();
    encoder.set_compute_pipeline_state(&last).unwrap();
    encoder.set_buffer(7, &bv).unwrap();
    dispatch(&mut encoder).unwrap();
    encoder.end_encoding().unwrap();
    a.write(2, &[3; 4]).unwrap();
    command.commit().unwrap();
    command.wait_until_completed().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Completed);
    assert_eq!(a.read().unwrap(), vec![1, 1, 4, 4, 4, 4, 1, 1]);
    assert_eq!(b.read().unwrap(), vec![9, 9, 6, 6, 6, 6, 9, 9]);
    let traces = provider.traces.lock().unwrap();
    assert_eq!(traces.len(), 1);
    assert_eq!(traces[0].passes.len(), 3);
    assert_eq!(traces[0].pipelines.len(), 3);
    assert_eq!(
        traces[0].passes[0].buffers[0].source,
        BufferSource::OwnedBytes(vec![3; 4])
    );
    assert_eq!(
        traces[0].passes[1].buffers[0].source,
        BufferSource::OwnedBytes(vec![3; 4])
    );
    assert_eq!(traces[0].passes[0].buffers[0].view_id, av.view_id());
    assert_eq!(command.submission().unwrap().writebacks.len(), 2);
}

#[test]
fn pipeline_and_view_owners_live_until_last_recorded_command_drops() {
    let (provider, device) = setup();
    let pipeline = pipeline(&device, "0");
    let (buffer, view) = buffer(&device, 1);
    let command = command(&device, &pipeline, &[(0, &view)]);
    drop(pipeline);
    drop(view);
    drop(buffer);
    drop(device);
    assert_eq!(provider.released_pipelines.load(Ordering::SeqCst), 0);
    command.commit().unwrap();
    assert_eq!(provider.released_completions.lock().unwrap().len(), 0);
    drop(command);
    assert_eq!(provider.released_pipelines.load(Ordering::SeqCst), 1);
    assert_eq!(provider.released_completions.lock().unwrap().len(), 1);
    assert_eq!(
        *provider.release_order.lock().unwrap(),
        vec!["completion", "pipeline"]
    );
}

#[test]
fn drop_retirement_panics_do_not_escape_or_skip_remaining_owners() {
    let (provider, device) = setup();
    let pipeline = pipeline(&device, "0");
    let (_, view) = buffer(&device, 1);
    let command = command(&device, &pipeline, &[(0, &view)]);
    command.commit().unwrap();
    drop(pipeline);
    provider.mode.store(PANIC_RELEASE, Ordering::SeqCst);
    assert!(catch_unwind(AssertUnwindSafe(|| drop(command))).is_ok());
    assert_eq!(
        *provider.release_order.lock().unwrap(),
        vec!["completion", "pipeline"]
    );
    assert_eq!(provider.released_completions.lock().unwrap().len(), 1);
    assert_eq!(provider.released_pipelines.load(Ordering::SeqCst), 1);
}

#[test]
fn cloned_device_shares_identity_but_wrapping_same_provider_does_not() {
    let (provider, device) = setup();
    let foreign = Device::new(provider);
    let local_pipeline = pipeline(&device.clone(), "0");
    let foreign_pipeline = pipeline(&foreign, "0");
    let (_, local_view) = buffer(&device, 1);
    let (_, foreign_view) = buffer(&foreign, 1);
    assert_ne!(local_view.allocation_id(), foreign_view.allocation_id());
    assert_ne!(local_view.view_id(), foreign_view.view_id());
    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.compute_command_encoder().unwrap();
    assert_eq!(
        encoder.set_compute_pipeline_state(&foreign_pipeline),
        Err(Error::Api(ApiError::ForeignPipeline))
    );
    assert_eq!(
        encoder.set_buffer(0, &foreign_view),
        Err(Error::ForeignBuffer)
    );
    encoder.set_compute_pipeline_state(&local_pipeline).unwrap();
    encoder.set_buffer(0, &local_view).unwrap();
    dispatch(&mut encoder).unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();
}

#[test]
fn ordering_and_dropped_encoder_failures_are_observable() {
    let (provider, device) = setup();
    let command = device.new_command_queue().command_buffer();
    assert_eq!(
        command.commit(),
        Err(Error::Api(ApiError::NoEncodedCommands))
    );
    assert_eq!(
        command.wait_until_completed(),
        Err(Error::Api(ApiError::CommandBufferNotCommitted))
    );
    let encoder = command.compute_command_encoder().unwrap();
    assert!(matches!(
        command.compute_command_encoder(),
        Err(Error::Api(ApiError::EncoderAlreadyOpen))
    ));
    assert_eq!(command.commit(), Err(Error::Api(ApiError::EncoderNotEnded)));
    drop(encoder);
    assert_eq!(command.commit(), Err(Error::Api(ApiError::EncoderNotEnded)));
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Failed);
    assert_eq!(
        command.wait_until_completed(),
        Err(Error::Api(ApiError::EncoderNotEnded))
    );
    assert_eq!(
        command.commit(),
        Err(Error::Api(ApiError::CommandBufferAlreadyCommitted))
    );
    assert!(provider.traces.lock().unwrap().is_empty());
}

#[test]
fn empty_encoder_and_duplicate_commit_are_refused() {
    let (_, device) = setup();
    let empty = device.new_command_queue().command_buffer();
    assert_eq!(
        empty.compute_command_encoder().unwrap().end_encoding(),
        Err(Error::Api(ApiError::MissingDispatch))
    );
    assert_eq!(empty.commit(), Err(Error::Api(ApiError::MissingDispatch)));
    let pipeline = pipeline(&device, "0");
    let (_, view) = buffer(&device, 1);
    let command = command(&device, &pipeline, &[(0, &view)]);
    command.commit().unwrap();
    assert_eq!(
        command.commit(),
        Err(Error::Api(ApiError::CommandBufferAlreadyCommitted))
    );
}

#[test]
fn exact_binding_layout_and_aliases_are_checked_before_submission() {
    let (provider, device) = setup();
    let copy = pipeline(&device, "copy");
    let (a, av) = buffer(&device, 1);
    let (_, bv) = buffer(&device, 2);
    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.compute_command_encoder().unwrap();
    assert_eq!(
        dispatch(&mut encoder),
        Err(Error::Api(ApiError::MissingPipeline))
    );
    encoder.set_compute_pipeline_state(&copy).unwrap();
    assert_eq!(
        dispatch(&mut encoder),
        Err(Error::Contract(ContractError::MissingBinding(4)))
    );
    encoder.set_buffer(4, &av).unwrap();
    assert!(matches!(
        encoder.set_buffer(9, &av),
        Err(Error::Api(ApiError::AliasedBufferBindings { .. }))
    ));
    assert!(matches!(
        encoder.set_buffer(9, &a.view(0, 4).unwrap()),
        Err(Error::Api(ApiError::AliasedBufferBindings { .. }))
    ));
    encoder.set_buffer(9, &bv).unwrap();
    let (_, extra) = buffer(&device, 3);
    encoder.set_buffer(100, &extra).unwrap();
    assert_eq!(
        dispatch(&mut encoder),
        Err(Error::Contract(ContractError::UnknownBinding(100)))
    );
    assert!(provider.traces.lock().unwrap().is_empty());
}

#[test]
fn alternate_views_of_one_allocation_across_passes_are_refused() {
    let (provider, device) = setup();
    let pipeline = pipeline(&device, "0");
    let (a, av) = buffer(&device, 1);
    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    encoder.set_buffer(0, &av).unwrap();
    dispatch(&mut encoder).unwrap();
    encoder.set_buffer(0, &a.view(2, 4).unwrap()).unwrap();
    dispatch(&mut encoder).unwrap();
    encoder.end_encoding().unwrap();
    assert!(
        matches!(command.commit(), Err(Error::Provider(error)) if error.slug == "buffer_alias_unsupported")
    );
    assert!(provider.traces.lock().unwrap().is_empty());
    assert_eq!(a.read().unwrap(), vec![1; 8]);
}

#[test]
fn empty_out_of_bounds_and_too_short_views_are_refused() {
    let (provider, device) = setup();
    assert!(matches!(
        device.new_buffer_with_bytes(vec![]),
        Err(Error::Api(ApiError::EmptyBuffer))
    ));
    let (a, _) = buffer(&device, 1);
    assert!(matches!(
        a.view(0, 0),
        Err(Error::Contract(ContractError::ZeroLength(_)))
    ));
    assert!(a.view(usize::MAX, 1).is_err());
    assert!(a.view(6, 4).is_err());
    assert!(a.write(usize::MAX, &[1]).is_err());
    let pipeline = pipeline(&device, "0");
    let command = command(&device, &pipeline, &[(0, &a.view(2, 3).unwrap())]);
    assert!(
        matches!(command.commit(), Err(Error::Provider(error)) if error.slug == "buffer_footprint_exceeds_view")
    );
    assert!(provider.traces.lock().unwrap().is_empty());
}

#[test]
fn pass_and_resource_limits_accept_boundaries_and_refuse_excess() {
    for count in [8, 9] {
        let (provider, device) = setup();
        let pipeline = pipeline(&device, "0");
        let (_, view) = buffer(&device, 1);
        let command = device.new_command_queue().command_buffer();
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        encoder.set_buffer(0, &view).unwrap();
        for _ in 0..8 {
            dispatch(&mut encoder).unwrap();
        }
        if count == 9 {
            assert_eq!(
                dispatch(&mut encoder),
                Err(Error::PassLimit {
                    requested: 9,
                    maximum: 8
                })
            );
        }
        encoder.end_encoding().unwrap();
        command.commit().unwrap();
        assert_eq!(provider.traces.lock().unwrap()[0].passes.len(), 8);
    }
    for count in [64, 65] {
        let (provider, device) = setup();
        let pipeline = pipeline(&device, &format!("wide:{count}"));
        let views = (0..count).map(|_| buffer(&device, 1).1).collect::<Vec<_>>();
        let command = device.new_command_queue().command_buffer();
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        for (binding, view) in views.iter().enumerate() {
            encoder.set_buffer(binding as u32, view).unwrap();
        }
        if count == 64 {
            dispatch(&mut encoder).unwrap();
            encoder.end_encoding().unwrap();
            command.commit().unwrap();
        } else {
            assert_eq!(
                dispatch(&mut encoder),
                Err(Error::Contract(ContractError::SerialResourceLimit {
                    requested: 65,
                    maximum: 64
                }))
            );
        }
        assert_eq!(
            provider.traces.lock().unwrap().len(),
            usize::from(count == 64)
        );
    }
}

#[test]
fn provider_failures_and_invalid_results_never_partially_land() {
    for mode in [
        BAD_LAST_WRITE,
        SUBMITTED,
        FAIL,
        PANIC_SUBMIT,
        WRONG_WAIT,
        MISSING_WRITE,
        PANIC_WAIT,
    ] {
        let (provider, device) = setup();
        provider.mode.store(mode, Ordering::SeqCst);
        let pipeline = pipeline(&device, "wide:2");
        let (a, av) = buffer(&device, 1);
        let (b, bv) = buffer(&device, 2);
        let command = command(&device, &pipeline, &[(0, &av), (1, &bv)]);
        let error = command.commit().unwrap_err();
        assert_eq!(command.status().unwrap(), CommandBufferStatus::Failed);
        assert_eq!(command.wait_until_completed(), Err(error.clone()));
        assert_eq!(command.submission(), Err(error.clone()));
        match mode {
            SUBMITTED => assert_eq!(error, Error::SynchronousCompletionRequired),
            FAIL => assert!(
                matches!(error, Error::Provider(error) if error.slug == "synthetic_failure")
            ),
            PANIC_SUBMIT | PANIC_WAIT => assert_eq!(error, Error::ProviderPanicked),
            WRONG_WAIT => assert_eq!(error, Error::CompletionObservationMismatch),
            _ => assert!(matches!(error, Error::Contract(_))),
        }
        assert_eq!(a.read().unwrap(), vec![1; 8], "mode={mode}");
        assert_eq!(b.read().unwrap(), vec![2; 8], "mode={mode}");
        a.write(0, &[3]).unwrap(); // Provider panic does not poison host storage.
        drop(command);
        assert_eq!(
            provider.released_completions.lock().unwrap().len(),
            usize::from(mode != PANIC_SUBMIT)
        );
    }
}

#[test]
fn invalid_compile_metadata_is_refused_and_retired() {
    let (provider, device) = setup();
    provider.mode.store(BAD_METADATA, Ordering::SeqCst);
    assert!(matches!(
        device.compile_pipeline(request("0")),
        Err(Error::InvalidPipelineMetadata)
    ));
    assert_eq!(provider.released_pipelines.load(Ordering::SeqCst), 1);
}

#[test]
fn waiters_wake_when_provider_panics_and_cpu_writes_resume() {
    let entered = Arc::new(std::sync::Barrier::new(2));
    let release = Arc::new(std::sync::Barrier::new(2));
    let mut provider = FakeProvider::new();
    provider.gate = Some((Arc::clone(&entered), Arc::clone(&release)));
    provider.mode.store(PANIC_SUBMIT, Ordering::SeqCst);
    let provider = Arc::new(provider);
    let device = Device::new(provider);
    let pipeline = pipeline(&device, "0");
    let (a, av) = buffer(&device, 1);
    let command = command(&device, &pipeline, &[(0, &av)]);
    std::thread::scope(|scope| {
        let submit = scope.spawn(|| command.commit());
        entered.wait();
        assert_eq!(command.status().unwrap(), CommandBufferStatus::Committed);
        let waiter = scope.spawn(|| command.wait_until_completed());
        let writer = scope.spawn(|| a.write(2, &[7; 4]));
        release.wait();
        assert_eq!(submit.join().unwrap(), Err(Error::ProviderPanicked));
        assert_eq!(waiter.join().unwrap(), Err(Error::ProviderPanicked));
        writer.join().unwrap().unwrap();
    });
    assert_eq!(a.read().unwrap(), vec![1, 1, 7, 7, 7, 7, 1, 1]);
}

#[test]
fn concurrent_commands_with_opposite_binding_order_keep_both_updates() {
    let (_, device) = setup();
    let pipeline = pipeline(&device, "wide:2");
    let (a, av) = buffer(&device, 1);
    let (b, bv) = buffer(&device, 2);
    let first = command(&device, &pipeline, &[(0, &av), (1, &bv)]);
    let second = command(&device, &pipeline, &[(0, &bv), (1, &av)]);
    std::thread::scope(|scope| {
        let first = scope.spawn(|| first.commit());
        let second = scope.spawn(|| second.commit());
        first.join().unwrap().unwrap();
        second.join().unwrap().unwrap();
    });
    assert_eq!(a.read().unwrap(), vec![1, 1, 3, 3, 3, 3, 1, 1]);
    assert_eq!(b.read().unwrap(), vec![2, 2, 4, 4, 4, 4, 2, 2]);
}
