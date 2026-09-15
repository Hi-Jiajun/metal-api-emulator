use super::*;
use crate::provider::{
    allocate_device_epoch, AliasMode, AttachmentFormat, BufferAccess, BufferBindingContract,
    BufferWriteback, CompletionReadback, ComputeProvider, ComputeTrace, FootprintProof,
    FunctionIdentity, FunctionSource, PipelineContract, PipelineId, PresentMode,
    ProviderErrorClass, ProviderHealth, ProviderPhase, RenderPipelineContract, Retryability,
    SemanticDigest, ShaderSource, StorageMode, SubmissionId, ValidatedComputeTrace,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout,
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
const ASYNC_GOOD: usize = 10;
const ASYNC_TIMEOUT_THEN_GOOD: usize = 11;
const ASYNC_FAILED: usize = 12;
const ASYNC_UNKNOWN: usize = 13;
const ASYNC_BAD_READBACK: usize = 14;
const ASYNC_MISSING_READBACK: usize = 15;
const ASYNC_DEVICE_LOST: usize = 16;
const CANCEL_GOOD: usize = 17;
const CANCEL_RACE_COMPLETED: usize = 18;
const CANCEL_REFUSED: usize = 19;
const SUBMIT_DEVICE_LOST: usize = 20;
const SUBMIT_EXHAUSTED: usize = 21;

const fn is_async_mode(mode: usize) -> bool {
    matches!(
        mode,
        SUBMITTED
            | ASYNC_GOOD
            | ASYNC_TIMEOUT_THEN_GOOD
            | ASYNC_FAILED
            | ASYNC_UNKNOWN
            | ASYNC_BAD_READBACK
            | ASYNC_MISSING_READBACK
            | ASYNC_DEVICE_LOST
            | CANCEL_GOOD
            | CANCEL_RACE_COMPLETED
            | CANCEL_REFUSED
    )
}

struct FakeProvider {
    epoch: contract::DeviceEpoch,
    mode: AtomicUsize,
    pipelines: Mutex<BTreeSet<PipelineId>>,
    next_pipeline: AtomicU64,
    traces: Mutex<Vec<ComputeTrace>>,
    released_pipelines: AtomicUsize,
    released_completions: Mutex<Vec<CompletionToken>>,
    release_order: Mutex<Vec<&'static str>>,
    readbacks: Mutex<BTreeMap<SubmissionId, CompletionReadback>>,
    cancelled: Mutex<Vec<CompletionToken>>,
    alias_mode: AliasMode,
    wait_calls: AtomicUsize,
    gate: Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>,
    render: bool,
    vertex_input: bool,
    heap: bool,
    icb: bool,
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
            readbacks: Mutex::new(BTreeMap::new()),
            cancelled: Mutex::new(Vec::new()),
            alias_mode: AliasMode::Refused,
            wait_calls: AtomicUsize::new(0),
            gate: None,
            render: false,
            vertex_input: false,
            heap: false,
            icb: false,
        }
    }
    fn with_alias_mode(mut self, alias_mode: AliasMode) -> Self {
        self.alias_mode = alias_mode;
        self
    }
    fn with_render(mut self) -> Self {
        self.render = true;
        self
    }
    fn with_vertex_input(mut self) -> Self {
        self.vertex_input = true;
        self
    }
    fn with_heap(mut self) -> Self {
        self.heap = true;
        self
    }
    fn with_icb(mut self) -> Self {
        self.icb = true;
        self
    }
    /// The attachment load operation of the most recent trace, for the
    /// encoder-side load test: the provider rail is what acts on it, so the
    /// trace has to carry the load the caller recorded.
    fn last_render_load(&self) -> Option<LoadOp> {
        self.traces.lock().unwrap().last().and_then(|trace| {
            trace
                .render_passes()
                .next()
                .map(|pass| pass.color_attachments[0].load)
        })
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
            alias_mode: self.alias_mode,
            storage_modes: vec![StorageMode::OwnedBytes],
            host_readback: true,
            submit_only: false,
            supports_render_passes: self.render,
            max_color_attachments: u32::from(self.render),
            max_attachment_dimension: [u64::from(self.render) * 4096; 2],
            supported_color_formats: self
                .render
                .then_some(AttachmentFormat::Rgba8Unorm)
                .into_iter()
                .collect(),
            max_vertex_buffers: if self.vertex_input {
                MAX_VERTEX_BUFFERS as u32
            } else {
                0
            },
            supported_vertex_formats: if self.vertex_input {
                VertexFormat::ADMITTED.to_vec()
            } else {
                Vec::new()
            },
            supported_index_formats: if self.vertex_input {
                IndexFormat::ADMITTED.to_vec()
            } else {
                Vec::new()
            },
            supports_presentation: self.render,
            max_present_targets: u32::from(self.render),
            supported_present_modes: self
                .render
                .then_some(PresentMode::Fifo)
                .into_iter()
                .collect(),
            max_present_image_count: u32::from(self.render),
            supports_heaps: self.heap,
            max_heap_bytes: u64::from(self.heap) * 65536,
            supported_heap_storage_modes: self
                .heap
                .then_some(StorageMode::OwnedBytes)
                .into_iter()
                .collect(),
            supports_heap_aliasing: false,
            supports_indirect_command_buffers: self.icb,
            max_indirect_commands: u32::from(self.icb) * 8,
            supported_indirect_commands: self
                .icb
                .then_some(IndirectCommandKind::Dispatch)
                .into_iter()
                .chain(self.icb.then_some(IndirectCommandKind::Draw))
                .chain(self.icb.then_some(IndirectCommandKind::DrawIndexed))
                .collect(),
        }
    }
    fn health(&self) -> ProviderHealth {
        match self.mode.load(Ordering::SeqCst) {
            SUBMIT_DEVICE_LOST | ASYNC_DEVICE_LOST => ProviderHealth::DeviceLost,
            SUBMIT_EXHAUSTED => ProviderHealth::Exhausted,
            _ => ProviderHealth::Usable,
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
        if mode == SUBMIT_DEVICE_LOST {
            let mut error = ProviderError::new(
                ProviderPhase::Submit,
                ProviderErrorClass::DeviceLost,
                "device_lost",
            )
            .unwrap()
            .with_completion(CompletionDisposition::DeviceLost { token: Some(token) });
            error.retryability = Retryability::RetryAfterRecreate;
            return Err(error);
        }
        if mode == SUBMIT_EXHAUSTED {
            let mut error = ProviderError::new(
                ProviderPhase::Submit,
                ProviderErrorClass::Resource,
                "provider_unavailable",
            )
            .unwrap()
            .with_completion(CompletionDisposition::NotSubmitted);
            error.retryability = Retryability::RetryAfterRecreate;
            return Err(error);
        }
        if mode == FAIL {
            return Err(self.error(token));
        }
        if mode == PANIC_SUBMIT {
            panic!("synthetic submit panic");
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
        for pass in trace.compute_passes() {
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
        // A render pass stores the reviewed fragment texel into its colour
        // attachment. The attachment view is declared read-only by the compute
        // pass, so the compute loop above leaves it out; the render half lands
        // its own writeback at the view's offset.
        for pass in trace.render_passes() {
            for attachment in &pass.color_attachments {
                let view = resources
                    .iter()
                    .find(|view| {
                        view.view_id == attachment.view_id
                            && view.allocation_id == attachment.allocation_id
                    })
                    .expect("the attachment view is in the serial pool");
                let texels = usize::try_from(attachment.expected_bytes().unwrap()).unwrap();
                // The render rail runs last, so its texels replace the
                // pre-render bytes the compute loop parked for the same view.
                writebacks.retain(|write| {
                    write.allocation_id != attachment.allocation_id
                        || write.view_id != attachment.view_id
                });
                writebacks.push(BufferWriteback {
                    allocation_id: attachment.allocation_id,
                    view_id: attachment.view_id,
                    offset: view.offset,
                    bytes: [0x40, 0x80, 0xc0, 0xff].repeat(texels / 4),
                });
            }
        }
        writebacks.sort_by_key(|write| (write.allocation_id, write.view_id));
        if mode == BAD_LAST_WRITE {
            writebacks.last_mut().unwrap().offset += 1;
        }
        if mode == MISSING_WRITE {
            writebacks.pop();
        }
        if is_async_mode(mode) {
            let mut readback_writebacks = writebacks.clone();
            if mode == ASYNC_BAD_READBACK {
                readback_writebacks.last_mut().unwrap().offset += 1;
            }
            if mode == ASYNC_MISSING_READBACK {
                readback_writebacks.pop();
            }
            self.readbacks.lock().unwrap().insert(
                token.submission_id,
                CompletionReadback {
                    completion: CompletionDisposition::CompletedVisible { token },
                    writebacks: readback_writebacks,
                },
            );
            return Ok(ProviderSubmission {
                completion: CompletionDisposition::Submitted { token },
                writebacks: vec![],
            });
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
        let call = self.wait_calls.fetch_add(1, Ordering::SeqCst) + 1;
        match self.mode.load(Ordering::SeqCst) {
            PANIC_WAIT => panic!("synthetic wait panic"),
            WRONG_WAIT => Ok(CompletionDisposition::TimedOut { token }),
            ASYNC_FAILED => Ok(CompletionDisposition::Failed { token: Some(token) }),
            ASYNC_UNKNOWN => Ok(CompletionDisposition::SubmittedUnknown { token: Some(token) }),
            ASYNC_DEVICE_LOST => Ok(CompletionDisposition::DeviceLost { token: Some(token) }),
            ASYNC_TIMEOUT_THEN_GOOD if call <= 2 => Ok(CompletionDisposition::TimedOut { token }),
            _ => Ok(CompletionDisposition::CompletedVisible { token }),
        }
    }

    fn readback(&self, token: CompletionToken) -> Result<CompletionReadback, ProviderError> {
        if self.mode.load(Ordering::SeqCst) == SUBMITTED {
            return Err(ProviderError::new(
                ProviderPhase::Wait,
                ProviderErrorClass::Capability,
                "completion_readback_unsupported",
            )
            .unwrap()
            .with_completion(CompletionDisposition::SubmittedUnknown { token: Some(token) }));
        }
        self.readbacks
            .lock()
            .unwrap()
            .get(&token.submission_id)
            .cloned()
            .ok_or_else(|| {
                ProviderError::new(
                    ProviderPhase::Wait,
                    ProviderErrorClass::Resource,
                    "unknown_completion",
                )
                .unwrap()
                .with_completion(CompletionDisposition::SubmittedUnknown { token: Some(token) })
            })
    }

    fn cancel(&self, token: CompletionToken) -> Result<CompletionDisposition, ProviderError> {
        self.cancelled.lock().unwrap().push(token);
        match self.mode.load(Ordering::SeqCst) {
            CANCEL_RACE_COMPLETED => Ok(CompletionDisposition::CompletedVisible { token }),
            CANCEL_REFUSED => Err(ProviderError::new(
                ProviderPhase::Wait,
                ProviderErrorClass::Capability,
                "completion_cancel_unsupported",
            )
            .unwrap()
            .with_completion(CompletionDisposition::SubmittedUnknown { token: Some(token) })),
            _ => Ok(CompletionDisposition::Cancelled { token }),
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
        } else if request.entry_name == "declare" {
            // A render case's declaring pass: one read-only view of the
            // attachment the render pass later stores into.
            vec![(0, BufferAccess::Read)]
        } else if let Some(list) = request.entry_name.strip_prefix("read:") {
            // A declaring pass for a render case whose draw reads more than one
            // view (`read:0,1`): every named slot is read-only, so the pass
            // leaves no writeback and no compute write for the draw's own
            // read of the same bytes to conflict with.
            list.split(',')
                .map(|slot| (slot.parse().unwrap(), BufferAccess::Read))
                .collect()
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
            render: None,
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
        self.readbacks.lock().unwrap().remove(&token.submission_id);
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
        traces[0].passes[0]
            .as_compute()
            .expect("recorded pass is a compute pass")
            .buffers[0]
            .source,
        BufferSource::OwnedBytes(vec![3; 4])
    );
    assert_eq!(
        traces[0].passes[1]
            .as_compute()
            .expect("recorded pass is a compute pass")
            .buffers[0]
            .source,
        BufferSource::OwnedBytes(vec![3; 4])
    );
    assert_eq!(
        traces[0].passes[0]
            .as_compute()
            .expect("recorded pass is a compute pass")
            .buffers[0]
            .view_id,
        av.view_id()
    );
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
fn submit_time_device_errors_never_land() {
    for mode in [SUBMIT_DEVICE_LOST, SUBMIT_EXHAUSTED] {
        let (provider, device) = setup();
        provider.mode.store(mode, Ordering::SeqCst);
        let pipeline = pipeline(&device, "wide:2");
        let (a, av) = buffer(&device, 1);
        let (b, bv) = buffer(&device, 2);
        let command = command(&device, &pipeline, &[(0, &av), (1, &bv)]);
        let error = command.commit().unwrap_err();
        let Error::Provider(provider_error) = &error else {
            panic!("mode={mode}: {error:?}");
        };
        assert_eq!(provider_error.phase, ProviderPhase::Submit, "mode={mode}");
        assert_eq!(
            provider_error.retryability,
            Retryability::RetryAfterRecreate,
            "mode={mode}"
        );
        match mode {
            SUBMIT_DEVICE_LOST => {
                assert_eq!(
                    provider_error.class,
                    ProviderErrorClass::DeviceLost,
                    "mode={mode}"
                );
                assert_eq!(provider_error.slug, "device_lost", "mode={mode}");
                assert!(
                    matches!(
                        provider_error.completion,
                        CompletionDisposition::DeviceLost { token: Some(_) }
                    ),
                    "mode={mode}"
                );
            }
            SUBMIT_EXHAUSTED => {
                assert_eq!(
                    provider_error.class,
                    ProviderErrorClass::Resource,
                    "mode={mode}"
                );
                assert_eq!(provider_error.slug, "provider_unavailable", "mode={mode}");
                assert_eq!(
                    provider_error.completion,
                    CompletionDisposition::NotSubmitted,
                    "mode={mode}"
                );
            }
            _ => unreachable!(),
        }
        assert_eq!(command.status().unwrap(), CommandBufferStatus::Failed);
        assert_eq!(command.wait_until_completed(), Err(error.clone()));
        assert_eq!(command.submission(), Err(error.clone()));
        assert_eq!(a.read().unwrap(), vec![1; 8], "mode={mode}");
        assert_eq!(b.read().unwrap(), vec![2; 8], "mode={mode}");
        drop(command);
        assert_eq!(
            provider.released_completions.lock().unwrap().len(),
            usize::from(mode == SUBMIT_DEVICE_LOST),
            "mode={mode}"
        );
    }
}

#[test]
fn device_reports_provider_health_without_waiting_for_a_submit() {
    let (provider, device) = setup();
    assert_eq!(device.health(), ProviderHealth::Usable);
    provider.mode.store(SUBMIT_DEVICE_LOST, Ordering::SeqCst);
    assert_eq!(device.health(), ProviderHealth::DeviceLost);
    provider.mode.store(SUBMIT_EXHAUSTED, Ordering::SeqCst);
    assert_eq!(device.health(), ProviderHealth::Exhausted);
    provider.mode.store(ASYNC_DEVICE_LOST, Ordering::SeqCst);
    assert_eq!(device.health(), ProviderHealth::DeviceLost);
}

#[test]
fn async_submission_finalizes_on_wait_and_lands_validated_readback() {
    let (provider, device) = setup();
    provider.mode.store(ASYNC_GOOD, Ordering::SeqCst);
    let pipeline = pipeline(&device, "wide:2");
    let (a, av) = buffer(&device, 1);
    let (b, bv) = buffer(&device, 2);
    let command = command(&device, &pipeline, &[(0, &av), (1, &bv)]);
    command.commit().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Committed);
    assert!(matches!(
        command.submission().unwrap().completion,
        CompletionDisposition::Submitted { .. }
    ));
    let (sender, receiver) = std::sync::mpsc::channel();
    let reader = a.clone();
    let handle = std::thread::spawn(move || sender.send(reader.read()).unwrap());
    assert!(receiver.recv_timeout(Duration::from_millis(50)).is_err());
    command.wait_until_completed().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Completed);
    assert_eq!(
        receiver
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap(),
        vec![1, 1, 2, 2, 2, 2, 1, 1]
    );
    handle.join().unwrap();
    assert_eq!(b.read().unwrap(), vec![2, 2, 3, 3, 3, 3, 2, 2]);
    let submission = command.submission().unwrap();
    assert!(matches!(
        submission.completion,
        CompletionDisposition::CompletedVisible { .. }
    ));
    assert_eq!(submission.writebacks.len(), 2);
    drop(command);
    assert_eq!(provider.released_completions.lock().unwrap().len(), 1);
}

#[test]
fn async_wait_retries_nonterminal_timeouts_until_visible() {
    let (provider, device) = setup();
    provider
        .mode
        .store(ASYNC_TIMEOUT_THEN_GOOD, Ordering::SeqCst);
    let pipeline = pipeline(&device, "wide:1");
    let (a, av) = buffer(&device, 1);
    let command = command(&device, &pipeline, &[(0, &av)]);
    command.commit().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Committed);
    command.wait_until_completed().unwrap();
    assert_eq!(provider.wait_calls.load(Ordering::SeqCst), 3);
    assert_eq!(a.read().unwrap(), vec![1, 1, 2, 2, 2, 2, 1, 1]);
}

#[test]
fn async_terminal_failures_and_unknown_completion_never_land() {
    for mode in [ASYNC_FAILED, ASYNC_UNKNOWN, ASYNC_DEVICE_LOST] {
        let (provider, device) = setup();
        provider.mode.store(mode, Ordering::SeqCst);
        let pipeline = pipeline(&device, "wide:1");
        let (a, av) = buffer(&device, 1);
        let command = command(&device, &pipeline, &[(0, &av)]);
        command.commit().unwrap();
        let error = command.wait_until_completed().unwrap_err();
        assert!(
            matches!(error, Error::CompletionUnavailable(_)),
            "mode={mode}"
        );
        assert_eq!(command.status().unwrap(), CommandBufferStatus::Failed);
        assert_eq!(a.read().unwrap(), vec![1; 8], "mode={mode}");
        assert!(matches!(
            command.submission().unwrap().completion,
            CompletionDisposition::Submitted { .. }
        ));
        drop(command);
        assert_eq!(provider.released_completions.lock().unwrap().len(), 1);
    }
}

#[test]
fn async_readback_contract_failures_never_partially_land() {
    for mode in [ASYNC_BAD_READBACK, ASYNC_MISSING_READBACK] {
        let (provider, device) = setup();
        provider.mode.store(mode, Ordering::SeqCst);
        let pipeline = pipeline(&device, "wide:2");
        let (a, av) = buffer(&device, 1);
        let (b, bv) = buffer(&device, 2);
        let command = command(&device, &pipeline, &[(0, &av), (1, &bv)]);
        command.commit().unwrap();
        let error = command.wait_until_completed().unwrap_err();
        assert!(matches!(error, Error::Contract(_)), "mode={mode}");
        assert_eq!(command.status().unwrap(), CommandBufferStatus::Failed);
        assert_eq!(a.read().unwrap(), vec![1; 8], "mode={mode}");
        assert_eq!(b.read().unwrap(), vec![2; 8], "mode={mode}");
        drop(command);
        assert_eq!(provider.released_completions.lock().unwrap().len(), 1);
    }
}

#[test]
fn async_provider_without_readback_support_fails_at_wait() {
    let (provider, device) = setup();
    provider.mode.store(SUBMITTED, Ordering::SeqCst);
    let pipeline = pipeline(&device, "wide:1");
    let (a, av) = buffer(&device, 1);
    let command = command(&device, &pipeline, &[(0, &av)]);
    command.commit().unwrap();
    let error = command.wait_until_completed().unwrap_err();
    assert!(
        matches!(&error, Error::Provider(error) if error.slug == "completion_readback_unsupported")
    );
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Failed);
    assert_eq!(a.read().unwrap(), vec![1; 8]);
    drop(command);
    assert_eq!(provider.released_completions.lock().unwrap().len(), 1);
}

#[test]
fn dropping_pending_command_releases_reservations_and_completion_record() {
    let (provider, device) = setup();
    provider.mode.store(ASYNC_GOOD, Ordering::SeqCst);
    let pipeline = pipeline(&device, "wide:1");
    let (a, av) = buffer(&device, 1);
    let command = command(&device, &pipeline, &[(0, &av)]);
    command.commit().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Committed);
    drop(command);
    assert_eq!(provider.released_completions.lock().unwrap().len(), 1);
    assert_eq!(a.read().unwrap(), vec![1; 8]);
    a.write(0, &[9]).unwrap();
    assert_eq!(a.read().unwrap()[0], 9);
}

#[test]
fn concurrent_async_commands_serialize_on_overlapping_reservations() {
    let (provider, device) = setup();
    provider.mode.store(ASYNC_GOOD, Ordering::SeqCst);
    let pipeline = pipeline(&device, "wide:2");
    let (a, av) = buffer(&device, 1);
    let (b, bv) = buffer(&device, 2);
    let first = command(&device, &pipeline, &[(0, &av), (1, &bv)]);
    let second = command(&device, &pipeline, &[(0, &bv), (1, &av)]);
    first.commit().unwrap();
    let second = std::thread::spawn(move || {
        second.commit().unwrap();
        second
    });
    first.wait_until_completed().unwrap();
    let second = second.join().unwrap();
    second.wait_until_completed().unwrap();
    assert_eq!(a.read().unwrap(), vec![1, 1, 3, 3, 3, 3, 1, 1]);
    assert_eq!(b.read().unwrap(), vec![2, 2, 4, 4, 4, 4, 2, 2]);
}

/// Ranged reservations: while one asynchronous command keeps a range of an
/// allocation reserved, a sibling commit of a disjoint range of the same
/// allocation must still be accepted. A whole-allocation reservation would
/// park it until the first command completed. The channel timeout keeps the
/// regression a failure instead of a hang.
#[test]
fn disjoint_ranges_of_one_allocation_stay_in_flight_together() {
    let (provider, device) = setup();
    provider.mode.store(ASYNC_GOOD, Ordering::SeqCst);
    let pipeline = pipeline(&device, "0");
    let shared = device.new_buffer_with_bytes(vec![7_u8; 8]).unwrap();
    let low = shared.view(0, 4).unwrap();
    let high = shared.view(4, 4).unwrap();
    let first = command(&device, &pipeline, &[(0, &low)]);
    let second = command(&device, &pipeline, &[(0, &high)]);
    first.commit().unwrap();
    let second = std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel();
        let sibling = scope.spawn(move || {
            let outcome = second.commit();
            let _ = tx.send(());
            (outcome, second)
        });
        let committed = rx.recv_timeout(Duration::from_secs(10)).is_ok();
        if !committed {
            // Unblock the sibling so the scope can join before reporting.
            first.wait_until_completed().unwrap();
        }
        assert!(
            committed,
            "a disjoint range of one allocation waited for an in-flight sibling (whole-allocation reservation)"
        );
        let (outcome, second) = sibling.join().unwrap();
        assert_eq!(outcome, Ok(()));
        second
    });
    first.wait_until_completed().unwrap();
    second.wait_until_completed().unwrap();
}

/// The negative side of the same probe: an overlapping range must keep waiting
/// for the in-flight command to complete.
#[test]
fn overlapping_ranges_of_one_allocation_still_serialize() {
    let (provider, device) = setup();
    provider.mode.store(ASYNC_GOOD, Ordering::SeqCst);
    let pipeline = pipeline(&device, "0");
    let shared = device.new_buffer_with_bytes(vec![7_u8; 8]).unwrap();
    let low = shared.view(0, 4).unwrap();
    let overlapping = shared.view(2, 4).unwrap();
    let first = command(&device, &pipeline, &[(0, &low)]);
    let second = command(&device, &pipeline, &[(0, &overlapping)]);
    first.commit().unwrap();
    let second = std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel();
        let sibling = scope.spawn(move || {
            let outcome = second.commit();
            let _ = tx.send(());
            (outcome, second)
        });
        let blocked = rx.recv_timeout(Duration::from_millis(500)).is_err();
        first.wait_until_completed().unwrap();
        assert!(blocked, "an overlapping range did not serialize");
        let (outcome, second) = sibling.join().unwrap();
        assert_eq!(outcome, Ok(()));
        second
    });
    second.wait_until_completed().unwrap();
}

/// While a command is parked inside submit its ranges are reserved but the
/// host bytes are not: the trace snapshot is already complete. A CPU write to
/// a disjoint range must therefore succeed, while one that overlaps the
/// reserved range must keep waiting. Holding the bytes guard across submit
/// would make the disjoint writer wait too.
#[test]
fn a_parked_submit_does_not_hold_host_bytes_for_a_disjoint_range() {
    let entered = Arc::new(std::sync::Barrier::new(2));
    let release = Arc::new(std::sync::Barrier::new(2));
    let mut provider = FakeProvider::new();
    provider.gate = Some((Arc::clone(&entered), Arc::clone(&release)));
    provider.mode.store(ASYNC_GOOD, Ordering::SeqCst);
    let device = Device::new(Arc::new(provider));
    let pipeline = pipeline(&device, "0");
    let shared = device.new_buffer_with_bytes(vec![7_u8; 8]).unwrap();
    let low = shared.view(0, 4).unwrap();
    let first = command(&device, &pipeline, &[(0, &low)]);
    std::thread::scope(|scope| {
        let submit = scope.spawn(|| first.commit());
        entered.wait();
        let shared_ref = &shared;
        let (disjoint_tx, disjoint_rx) = std::sync::mpsc::channel();
        let disjoint = scope.spawn(move || {
            let outcome = shared_ref.write(4, &[9; 4]);
            let _ = disjoint_tx.send(());
            outcome
        });
        let wrote = disjoint_rx.recv_timeout(Duration::from_secs(10)).is_ok();
        let (overlap_tx, overlap_rx) = std::sync::mpsc::channel();
        let shared_ref = &shared;
        let overlapping = scope.spawn(move || {
            let outcome = shared_ref.write(0, &[5; 4]);
            let _ = overlap_tx.send(());
            outcome
        });
        let overlapped = overlap_rx.recv_timeout(Duration::from_millis(500)).is_ok();
        release.wait();
        assert_eq!(submit.join().unwrap(), Ok(()));
        first.wait_until_completed().unwrap();
        assert!(
            wrote,
            "a disjoint CPU write waited for a parked submit: the host bytes are still held across submit"
        );
        assert!(
            !overlapped,
            "an overlapping CPU write did not wait for its reserved range"
        );
        disjoint.join().unwrap().unwrap();
        overlapping.join().unwrap().unwrap();
    });
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

#[test]
fn cancelling_a_pending_command_releases_reservations_and_provider_slot() {
    let (provider, device) = setup();
    provider.mode.store(CANCEL_GOOD, Ordering::SeqCst);
    let pipeline = pipeline(&device, "wide:1");
    let (a, av) = buffer(&device, 1);
    let command = command(&device, &pipeline, &[(0, &av)]);
    command.commit().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Committed);
    command.cancel().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Failed);
    assert_eq!(provider.cancelled.lock().unwrap().len(), 1);
    // The host reservation is released without waiting for device retirement.
    assert_eq!(a.read().unwrap(), vec![1; 8]);
    let error = command.wait_until_completed().unwrap_err();
    assert!(matches!(
        error,
        Error::CompletionUnavailable(CompletionDisposition::Cancelled { .. })
    ));
    drop(command);
    assert_eq!(provider.released_completions.lock().unwrap().len(), 1);
}

#[test]
fn cancel_race_loss_and_refusal_never_land_results() {
    for mode in [CANCEL_RACE_COMPLETED, CANCEL_REFUSED] {
        let (provider, device) = setup();
        provider.mode.store(mode, Ordering::SeqCst);
        let pipeline = pipeline(&device, "wide:1");
        let (a, av) = buffer(&device, 1);
        let command = command(&device, &pipeline, &[(0, &av)]);
        command.commit().unwrap();
        assert!(command.cancel().is_err(), "mode={mode}");
        assert_eq!(command.status().unwrap(), CommandBufferStatus::Failed);
        assert_eq!(a.read().unwrap(), vec![1; 8], "mode={mode}");
        drop(command);
        assert_eq!(provider.released_completions.lock().unwrap().len(), 1);
    }
}

#[test]
fn cancelling_a_recording_command_is_refused() {
    let (provider, device) = setup();
    provider.mode.store(CANCEL_GOOD, Ordering::SeqCst);
    let pipeline = pipeline(&device, "wide:1");
    let (_, av) = buffer(&device, 1);
    let command = command(&device, &pipeline, &[(0, &av)]);
    assert!(matches!(
        command.cancel(),
        Err(Error::Api(ApiError::CommandBufferNotCommitted))
    ));
    assert!(provider.cancelled.lock().unwrap().is_empty());
}

#[test]
fn disjoint_views_of_one_allocation_are_admitted_under_distinct_views() {
    let provider = Arc::new(FakeProvider::new().with_alias_mode(AliasMode::DistinctViews));
    let device = Device::new(provider.clone());
    let pipeline = pipeline(&device, "0");
    let buffer = device.new_buffer_with_bytes(vec![3; 16]).unwrap();
    let first = buffer.view(0, 4).unwrap();
    let second = buffer.view(8, 4).unwrap();
    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    encoder.set_buffer(0, &first).unwrap();
    dispatch(&mut encoder).unwrap();
    encoder.clear_buffers().unwrap();
    encoder.set_buffer(0, &second).unwrap();
    dispatch(&mut encoder).unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();
    assert_eq!(command.submission().unwrap().writebacks.len(), 2);
    assert_eq!(provider.traces.lock().unwrap().len(), 1);
}

#[test]
fn overlapping_views_of_one_allocation_stay_refused_under_distinct_views() {
    let provider = Arc::new(FakeProvider::new().with_alias_mode(AliasMode::DistinctViews));
    let device = Device::new(provider.clone());
    let pipeline = pipeline(&device, "0");
    let buffer = device.new_buffer_with_bytes(vec![3; 16]).unwrap();
    let first = buffer.view(0, 4).unwrap();
    let second = buffer.view(2, 4).unwrap();
    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    encoder.set_buffer(0, &first).unwrap();
    dispatch(&mut encoder).unwrap();
    encoder.clear_buffers().unwrap();
    encoder.set_buffer(0, &second).unwrap();
    dispatch(&mut encoder).unwrap();
    encoder.end_encoding().unwrap();
    assert!(matches!(
        command.commit(),
        Err(Error::Provider(error)) if error.slug == "buffer_alias_unsupported"
    ));
    assert!(provider.traces.lock().unwrap().is_empty());
}

fn render_metadata(provider: &FakeProvider) -> CompiledComputePipeline {
    CompiledComputePipeline {
        device_epoch: provider.device_epoch(),
        pipeline_id: PipelineId::new(9001),
        function: FunctionIdentity {
            entry_name: "vertex_main".into(),
            logical_digest: SemanticDigest::new("fixture", vec![2]).unwrap(),
            source: FunctionSource::Metallib,
        },
        contract: PipelineContract {
            dispatch_kind: DispatchKind::ThreadsExact,
            required_local_size: None,
            fixed_grid: None,
            push_constant_offset: 0,
            push_constant_bytes: 0,
            buffer_bindings: Vec::new(),
            shader_capabilities: Vec::new(),
            translator_revision: None,
        },
        render: Some(RenderPipelineContract {
            vertex_entry: "vertex_main".into(),
            fragment_entry: "fragment_main".into(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::None,
        }),
    }
}

#[test]
fn render_pipeline_wraps_render_metadata_and_refuses_compute_only() {
    let (provider, device) = setup();
    assert!(device.render_pipeline(&render_metadata(&provider)).is_ok());
    let mut compute_only = render_metadata(&provider);
    compute_only.render = None;
    assert!(matches!(
        device.render_pipeline(&compute_only),
        Err(Error::InvalidPipelineMetadata)
    ));
}

#[test]
fn render_encoder_refuses_a_foreign_pipeline() {
    let (_provider, device) = setup();
    let (other_provider, other_device) = setup();
    let foreign = other_device
        .render_pipeline(&render_metadata(&other_provider))
        .unwrap();
    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.render_command_encoder().unwrap();
    assert!(matches!(
        encoder.set_render_pipeline_state(&foreign),
        Err(Error::Api(ApiError::ForeignPipeline))
    ));
}

#[test]
fn a_recording_can_load_its_attachment_instead_of_clearing_it() {
    // The encoder-side load shape (`research/docs/23` §3.3): the attachment
    // keeps the bytes the view already holds, which the object API snapshots at
    // commit exactly as it does for the draw inputs. The declaring compute pass
    // binds the same view, so the trace's own declaration is what the provider
    // uploads.
    let provider = Arc::new(FakeProvider::new().with_render());
    let device = Device::new(provider.clone());
    let declaring = device.compile_pipeline(request("declare")).unwrap();
    let render_metadata = render_metadata(&provider);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &attachment_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    {
        let mut encoder = command.render_command_encoder().unwrap();
        encoder.set_render_pipeline_state(&render).unwrap();
        encoder
            .draw_render_pass(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Load,
                None,
            )
            .expect("a loading pass records like a clearing one");
        encoder.end_encoding().unwrap();
    }
    command.commit().unwrap();
    command.wait_until_completed().unwrap();
    // The trace the provider actually received carries `LoadOp::Load`: the
    // provider rail is what uploads the attachment's previous bytes, so a
    // recording that claims to load has to reach it as a load rather than as a
    // clear the provider had to reinterpret.
    assert_eq!(provider.last_render_load(), Some(LoadOp::Load));
}

#[test]
fn render_encoder_refuses_attachment_extent_mismatch_and_missing_pipeline() {
    let (provider, device) = setup();
    let render_pipeline = device.render_pipeline(&render_metadata(&provider)).unwrap();
    let buffer = device.new_buffer_with_bytes(vec![0; 4]).unwrap();
    let view = buffer.view(0, 4).unwrap();
    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.render_command_encoder().unwrap();
    assert!(matches!(
        encoder.draw_render_pass(
            &view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0; 4]),
            None
        ),
        Err(Error::Api(ApiError::MissingPipeline))
    ));
    encoder.set_render_pipeline_state(&render_pipeline).unwrap();
    assert!(matches!(
        encoder.draw_render_pass(
            &view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0; 4]),
            None
        ),
        Err(Error::Contract(
            ContractError::AttachmentExtentMismatch { .. }
        ))
    ));
}

#[test]
fn render_encoder_refuses_a_foreign_buffer() {
    let (provider, device) = setup();
    let (_other_provider, other_device) = setup();
    let render_pipeline = device.render_pipeline(&render_metadata(&provider)).unwrap();
    let foreign = other_device.new_buffer_with_bytes(vec![0; 16]).unwrap();
    let view = foreign.view(0, 16).unwrap();
    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render_pipeline).unwrap();
    assert!(matches!(
        encoder.draw_render_pass(
            &view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0; 4]),
            None
        ),
        Err(Error::ForeignBuffer)
    ));
}

#[test]
fn render_encoder_end_encoding_requires_a_draw() {
    let (_provider, device) = setup();
    let command = device.new_command_queue().command_buffer();
    let encoder = command.render_command_encoder().unwrap();
    assert!(matches!(
        encoder.end_encoding(),
        Err(Error::Api(ApiError::MissingDispatch))
    ));
}

#[test]
fn object_render_commit_routes_the_attachment_through_submit_and_lands_texels() {
    let provider = Arc::new(FakeProvider::new().with_render());
    let device = Device::new(provider.clone());

    let declaring = device.compile_pipeline(request("declare")).unwrap();
    let render_metadata = render_metadata(&provider);
    // `register_render_pipeline` is a concrete-context entry point, not part of
    // `PipelineProvider`; the fake admits the table entry here the way the real
    // rails register it before any command names it.
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &attachment_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    {
        let mut encoder = command.render_command_encoder().unwrap();
        encoder.set_render_pipeline_state(&render).unwrap();
        encoder
            .draw_render_pass(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Clear([0xfe; 4]),
                Some(PresentInitial::Sentinel([0xef; 4])),
            )
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    command.commit().unwrap();
    command.wait_until_completed().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Completed);
    assert_eq!(
        attachment.read().unwrap(),
        [0x40, 0x80, 0xc0, 0xff].repeat(4),
        "the render pass lands the reviewed fragment texel into the attachment"
    );

    let traces = provider.traces.lock().unwrap();
    assert_eq!(traces.len(), 1);
    assert_eq!(traces[0].passes.len(), 2);
    assert!(traces[0].passes[0].as_compute().is_some());
    let render_pass = traces[0].passes[1].as_render().expect("render pass");
    assert_eq!(render_pass.color_attachments.len(), 1);
    assert_eq!(
        render_pass.color_attachments[0].view_id,
        attachment_view.view_id()
    );
    assert_eq!(
        render_pass.color_attachments[0].allocation_id,
        attachment_view.allocation_id()
    );
    assert!(
        render_pass.present.is_some(),
        "the present tail rides the same recorded render pass"
    );
}

#[test]
fn heap_placement_publishes_placements_in_allocation_order() {
    let provider = Arc::new(FakeProvider::new().with_heap());
    let device = Device::new(provider.clone());
    let copy = pipeline(&device, "copy");
    let (first, first_view) = buffer(&device, 1);
    let (second, second_view) = buffer(&device, 2);

    let heap = device.new_heap(64, StorageMode::OwnedBytes, false).unwrap();
    // Place the later allocation first: commit still publishes the payload in
    // ascending allocation order, the exact order the provider's placement map
    // zips against the trace's owned allocations.
    heap.place(&second, 8).unwrap();
    heap.place(&first, 0).unwrap();

    let command = device.new_command_queue().command_buffer();
    command.set_heap(&heap).unwrap();
    let mut encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&copy).unwrap();
    encoder.set_buffer(4, &first_view).unwrap();
    encoder.set_buffer(9, &second_view).unwrap();
    dispatch(&mut encoder).unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();
    command.wait_until_completed().unwrap();

    let traces = provider.traces.lock().unwrap();
    assert_eq!(traces.len(), 1);
    let payload = traces[0].heap.as_ref().expect("heap payload");
    assert_eq!(payload.descriptor.size, 64);
    assert_eq!(payload.descriptor.storage_mode, StorageMode::OwnedBytes);
    assert_eq!(payload.placements.len(), 2);
    assert_eq!(payload.placements[0].offset, 0);
    assert_eq!(payload.placements[1].offset, 8);
    assert_eq!(payload.placements[0].resource.byte_size(), 8);
}

#[test]
fn heap_place_refuses_duplicate_overflow_overlap_and_foreign_buffers() {
    let provider = Arc::new(FakeProvider::new().with_heap());
    let device = Device::new(provider.clone());
    let other = Device::new(provider.clone());
    let (first, _) = buffer(&device, 1);
    let (second, _) = buffer(&device, 2);
    let (foreign, _) = buffer(&other, 3);

    let heap = device.new_heap(32, StorageMode::OwnedBytes, false).unwrap();
    assert_eq!(heap.place(&foreign, 0), Err(Error::ForeignBuffer));
    heap.place(&first, 0).unwrap();
    assert!(matches!(
        heap.place(&first, 16),
        Err(Error::HeapPlacementDuplicate { .. })
    ));
    assert!(matches!(
        heap.place(&second, 4),
        Err(Error::Contract(ContractError::HeapPlacementOverlap { .. }))
    ));
    assert!(matches!(
        heap.place(&second, 25),
        Err(Error::Contract(ContractError::HeapPlacementOverflow { .. }))
    ));
}

#[test]
fn heap_construction_refuses_zero_size_and_aliasing() {
    let device = Device::new(Arc::new(FakeProvider::new()));
    assert!(matches!(
        device.new_heap(0, StorageMode::OwnedBytes, false),
        Err(Error::Contract(ContractError::ZeroLength(_)))
    ));
    assert!(matches!(
        device.new_heap(64, StorageMode::OwnedBytes, true),
        Err(Error::Contract(ContractError::HeapAliasingUnsupported))
    ));
}

#[test]
fn indirect_dispatch_publishes_the_replayed_payload() {
    let provider = Arc::new(FakeProvider::new().with_icb());
    let device = Device::new(provider.clone());
    let copy = pipeline(&device, "copy");
    let (_, first_view) = buffer(&device, 1);
    let (_, second_view) = buffer(&device, 2);
    let icb = device
        .new_indirect_command_buffer(
            IndirectCommandKind::Dispatch,
            1,
            vec![IndirectCommandKind::Dispatch],
            IndirectCommandRange { start: 0, count: 1 },
            IndirectCommandDescriptor::Dispatch {
                threadgroups: [1, 1, 1],
            },
        )
        .unwrap();

    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&copy).unwrap();
    encoder.set_buffer(4, &first_view).unwrap();
    encoder.set_buffer(9, &second_view).unwrap();
    encoder
        .dispatch_indirect(
            &icb,
            Size::new(1, 1, 1).unwrap(),
            Size::new(1, 1, 1).unwrap(),
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();
    command.wait_until_completed().unwrap();

    let traces = provider.traces.lock().unwrap();
    assert_eq!(traces.len(), 1);
    let payload = traces[0].indirect.as_ref().expect("indirect payload");
    assert_eq!(payload.command.kind(), IndirectCommandKind::Dispatch);
    assert_eq!(payload.buffer.max_commands, 1);
    assert_eq!(payload.range.start, 0);
    assert_eq!(payload.range.count, 1);
}

#[test]
fn indirect_encoder_refuses_kind_mismatch_direct_conflict_and_second_buffer() {
    let provider = Arc::new(FakeProvider::new().with_icb());
    let device = Device::new(provider.clone());
    let copy = pipeline(&device, "copy");
    let (_, first_view) = buffer(&device, 1);
    let (_, second_view) = buffer(&device, 2);
    let dispatch_icb = device
        .new_indirect_command_buffer(
            IndirectCommandKind::Dispatch,
            1,
            vec![IndirectCommandKind::Dispatch],
            IndirectCommandRange { start: 0, count: 1 },
            IndirectCommandDescriptor::Dispatch {
                threadgroups: [1, 1, 1],
            },
        )
        .unwrap();
    let draw_icb = device
        .new_indirect_command_buffer(
            IndirectCommandKind::Draw,
            1,
            vec![IndirectCommandKind::Draw],
            IndirectCommandRange { start: 0, count: 1 },
            IndirectCommandDescriptor::Draw {
                vertex_count: 3,
                instance_count: 1,
            },
        )
        .unwrap();

    let bind = |encoder: &mut ComputeCommandEncoder| {
        encoder.set_buffer(4, &first_view).unwrap();
        encoder.set_buffer(9, &second_view).unwrap();
    };
    let shape = |encoder: &mut ComputeCommandEncoder, icb: &IndirectCommandBuffer| {
        encoder.dispatch_indirect(
            icb,
            Size::new(1, 1, 1).unwrap(),
            Size::new(1, 1, 1).unwrap(),
        )
    };

    // A dispatch encoder refuses a draw command.
    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&copy).unwrap();
    bind(&mut encoder);
    assert!(matches!(
        shape(&mut encoder, &draw_icb),
        Err(Error::IndirectKindMismatch {
            accepted: &[IndirectCommandKind::Dispatch],
            actual: IndirectCommandKind::Draw
        })
    ));
    // The same encoder refuses a direct dispatch after an indirect one.
    shape(&mut encoder, &dispatch_icb).unwrap();
    assert!(matches!(
        encoder.dispatch_threads(Size::new(1, 1, 1).unwrap(), Size::new(1, 1, 1).unwrap()),
        Err(Error::IndirectDirectConflict)
    ));
    encoder.end_encoding().unwrap();

    // A second encoder cannot add another indirect command buffer.
    let mut second = command.compute_command_encoder().unwrap();
    second.set_compute_pipeline_state(&copy).unwrap();
    bind(&mut second);
    assert!(matches!(
        shape(&mut second, &dispatch_icb),
        Err(Error::IndirectAlreadyRecorded)
    ));
}

#[test]
fn render_draw_indirect_replays_the_attachment() {
    let provider = Arc::new(FakeProvider::new().with_render().with_icb());
    let device = Device::new(provider.clone());

    let declaring = device.compile_pipeline(request("declare")).unwrap();
    let render_metadata = render_metadata(&provider);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let icb = device
        .new_indirect_command_buffer(
            IndirectCommandKind::Draw,
            1,
            vec![IndirectCommandKind::Draw],
            IndirectCommandRange { start: 0, count: 1 },
            IndirectCommandDescriptor::Draw {
                vertex_count: 3,
                instance_count: 1,
            },
        )
        .unwrap();

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &attachment_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    {
        let mut encoder = command.render_command_encoder().unwrap();
        encoder.set_render_pipeline_state(&render).unwrap();
        encoder
            .draw_indirect(
                &icb,
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Clear([0xfe; 4]),
                None,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    command.commit().unwrap();
    command.wait_until_completed().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Completed);
    assert_eq!(
        attachment.read().unwrap(),
        [0x40, 0x80, 0xc0, 0xff].repeat(4)
    );

    let traces = provider.traces.lock().unwrap();
    assert_eq!(traces.len(), 1);
    let payload = traces[0].indirect.as_ref().expect("indirect payload");
    assert_eq!(payload.command.kind(), IndirectCommandKind::Draw);
    assert!(traces[0].passes[1].as_render().is_some());
}

/// [`render_metadata`] with a vertex layout that binds caller-held streams, the
/// way a pipeline compiled against `MTLVertexDescriptor` metadata names them.
fn render_metadata_with_layout(
    provider: &FakeProvider,
    layout: VertexLayout,
) -> CompiledComputePipeline {
    let mut metadata = render_metadata(provider);
    metadata.render = Some(RenderPipelineContract {
        vertex_entry: "vertex_main".into(),
        fragment_entry: "fragment_main".into(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: layout,
    });
    metadata
}

/// One vertex stream: eight bytes per vertex, one `float32x2` at offset 0.
fn stream_layout(location: u32) -> VertexBufferLayout {
    VertexBufferLayout {
        stride: 8,
        attributes: vec![VertexAttribute {
            location,
            offset: 0,
            format: VertexFormat::Float32x2,
        }],
    }
}

/// The pass-shaped view a bound draw input has to become: `metal_binding` as the
/// pass's label, the view's own range, the contract's read-only access and the
/// bytes the trace has to carry.
///
/// The expectation is stated here rather than read back from the code under
/// test, so a pass that spells a different label, range, access or byte count
/// fails the comparison.
fn stream_view(view: &BufferView, metal_binding: u32, bytes: Vec<u8>) -> contract::BufferView {
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

/// Register a render table entry with `layout` and wrap it for the encoder: the
/// fake keeps the registration the way a rail's own context does before any
/// command names it (`Device::render_pipeline`).
fn render_pipeline_with_layout(
    provider: &FakeProvider,
    device: &Device,
    layout: VertexLayout,
) -> RenderPipeline {
    let metadata = render_metadata_with_layout(provider, layout);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(metadata.pipeline_id);
    device.render_pipeline(&metadata).unwrap()
}

#[test]
fn render_encoder_refuses_repeated_out_of_range_and_foreign_inputs() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let (_other_provider, other_device) = setup();
    let render = render_pipeline_with_layout(
        &provider,
        &device,
        VertexLayout::Buffers(vec![stream_layout(0), stream_layout(1)]),
    );
    let (_, stream_view) = buffer(&device, 0x11);
    let (_, second_stream) = buffer(&device, 0x22);
    let (_, index_view) = buffer(&device, 0x33);
    let (_, foreign) = buffer(&other_device, 0x44);

    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    // Ownership is answered first, then the binding the encoder already holds:
    // the order a heap placement uses.
    assert_eq!(
        encoder.set_vertex_buffer(0, &foreign),
        Err(Error::ForeignBuffer)
    );
    assert_eq!(
        encoder.set_index_buffer(&foreign, IndexFormat::Uint32),
        Err(Error::ForeignBuffer)
    );
    encoder.set_vertex_buffer(0, &stream_view).unwrap();
    assert_eq!(
        encoder.set_vertex_buffer(0, &second_stream),
        Err(Error::VertexBufferAlreadyBound { index: 0 })
    );
    assert_eq!(
        encoder.set_vertex_buffer(MAX_VERTEX_BUFFERS as u32, &second_stream),
        Err(Error::VertexBufferIndexOutOfRange {
            index: MAX_VERTEX_BUFFERS as u32,
            maximum: MAX_VERTEX_BUFFERS,
        })
    );
    // The cap is on the pass's own list, so the last admitted index is the one
    // below it.
    encoder
        .set_vertex_buffer(MAX_VERTEX_BUFFERS as u32 - 1, &second_stream)
        .unwrap();
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    assert_eq!(
        encoder.set_index_buffer(&index_view, IndexFormat::Uint32),
        Err(Error::IndexBufferAlreadyBound)
    );
}

#[test]
fn direct_draws_refuse_missing_inputs_and_counts_below_the_milestone() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let render = render_pipeline_with_layout(
        &provider,
        &device,
        VertexLayout::Buffers(vec![stream_layout(0)]),
    );
    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let (_, stream_view) = buffer(&device, 0x11);
    let (_, index_view) = buffer(&device, 0x22);

    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    // Neither shape has the input it draws from: the vertex-buffer draw has no
    // stream, the indexed draw no index buffer. The vertex_id-only shape is
    // `draw_render_pass`, which binds no input at all.
    assert_eq!(
        encoder.draw_primitives(
            &attachment_view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            FULL_SCREEN_TRIANGLE_VERTICES,
            None,
        ),
        Err(Error::MissingVertexBuffer)
    );
    assert_eq!(
        encoder.draw_indexed_primitives(
            &attachment_view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            FULL_SCREEN_TRIANGLE_VERTICES,
            None,
        ),
        Err(Error::MissingIndexBuffer)
    );
    encoder.set_vertex_buffer(0, &stream_view).unwrap();
    assert_eq!(
        encoder.draw_primitives(
            &attachment_view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            FULL_SCREEN_TRIANGLE_VERTICES - 1,
            None,
        ),
        Err(Error::Contract(
            ContractError::DrawVertexCountBelowMinimum {
                minimum: FULL_SCREEN_TRIANGLE_VERTICES,
                actual: FULL_SCREEN_TRIANGLE_VERTICES - 1,
            }
        ))
    );
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint32)
        .unwrap();
    assert_eq!(
        encoder.draw_indexed_primitives(
            &attachment_view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            FULL_SCREEN_TRIANGLE_VERTICES - 1,
            None,
        ),
        Err(Error::Contract(
            ContractError::DrawVertexCountBelowMinimum {
                minimum: FULL_SCREEN_TRIANGLE_VERTICES,
                actual: FULL_SCREEN_TRIANGLE_VERTICES - 1,
            }
        ))
    );
}

#[test]
fn direct_draws_refuse_a_pipeline_layout_that_disagrees_with_the_pass() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let two_streams = render_pipeline_with_layout(
        &provider,
        &device,
        VertexLayout::Buffers(vec![stream_layout(0), stream_layout(1)]),
    );
    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let (_, stream_view) = buffer(&device, 0x11);

    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&two_streams).unwrap();
    // The vertex_id shape binds nothing, so a pipeline whose layout binds two
    // streams is refused when the pass is recorded: the pass the caller asked
    // for and the pipeline it names cannot both be executed.
    assert_eq!(
        encoder.draw_render_pass(
            &attachment_view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            None,
        ),
        Err(Error::Contract(
            ContractError::VertexLayoutBindingMismatch {
                pipeline_buffers: 2,
                pass_buffers: 0,
            }
        ))
    );
    // One bound stream against a two-stream layout is the same refusal, and it
    // names both counts.
    encoder.set_vertex_buffer(0, &stream_view).unwrap();
    assert_eq!(
        encoder.draw_primitives(
            &attachment_view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            FULL_SCREEN_TRIANGLE_VERTICES,
            None,
        ),
        Err(Error::Contract(
            ContractError::VertexLayoutBindingMismatch {
                pipeline_buffers: 2,
                pass_buffers: 1,
            }
        ))
    );
}

#[test]
fn indexed_draw_records_the_bound_streams_in_binding_order() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    // Only the attachment needs a compute declaration: a draw input carries its
    // own bytes, so the two streams and the index buffer never enter the pool.
    let declaring = pipeline(&device, "read:0");
    let render = render_pipeline_with_layout(
        &provider,
        &device,
        VertexLayout::Buffers(vec![stream_layout(0), stream_layout(1)]),
    );

    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let (_, first_stream) = buffer(&device, 0x11);
    let (_, second_stream) = buffer(&device, 0x22);
    let index = device.new_buffer_with_bytes(vec![0; 12]).unwrap();
    let index_view = index.view(0, 12).unwrap();

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &attachment_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    {
        let mut encoder = command.render_command_encoder().unwrap();
        encoder.set_render_pipeline_state(&render).unwrap();
        // Bound in the reverse order of the pipeline's layout: the pass's own
        // binding list is positional, so the first stream still lands first.
        encoder.set_vertex_buffer(1, &second_stream).unwrap();
        encoder.set_vertex_buffer(0, &first_stream).unwrap();
        encoder
            .set_index_buffer(&index_view, IndexFormat::Uint32)
            .unwrap();
        encoder
            .draw_indexed_primitives(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Clear([0xfe; 4]),
                FULL_SCREEN_TRIANGLE_VERTICES,
                None,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    command.commit().unwrap();
    command.wait_until_completed().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Completed);
    assert_eq!(
        attachment.read().unwrap(),
        [0x40, 0x80, 0xc0, 0xff].repeat(4)
    );

    let traces = provider.traces.lock().unwrap();
    assert_eq!(traces.len(), 1);
    assert_eq!(
        traces[0].passes[0]
            .as_compute()
            .expect("the declaring compute pass")
            .buffers
            .iter()
            .map(|view| view.view_id)
            .collect::<Vec<_>>(),
        vec![attachment_view.view_id()],
        "only the attachment is declared: a draw input carries its own bytes"
    );
    let pass = traces[0]
        .render_passes()
        .next()
        .expect("the indexed pass reached the trace");
    assert_eq!(
        pass.vertices, FULL_SCREEN_TRIANGLE_VERTICES,
        "an indexed draw's `vertices` is its index count"
    );
    assert_eq!(
        pass.vertex_buffers,
        vec![
            stream_view(&first_stream, 0, vec![0x11; 4]),
            stream_view(&second_stream, 1, vec![0x22; 4]),
        ],
        "entry i is binding i, whatever order the caller bound them in, and the \
         pass carries each stream's own bytes"
    );
    assert_eq!(
        pass.indices,
        Some(IndexBufferBinding {
            view: stream_view(&index_view, 0, vec![0; 12]),
            format: IndexFormat::Uint32,
        })
    );
}

#[test]
fn indexed_draw_without_streams_keeps_the_vertex_id_shape() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    // The index buffer needs no declaration of its own: the pass carries its
    // bytes, so the compute pass only has to bring the attachment in.
    let declaring = pipeline(&device, "read:0");
    let render = render_pipeline_with_layout(&provider, &device, VertexLayout::None);

    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let index = device.new_buffer_with_bytes(vec![0; 12]).unwrap();
    let index_view = index.view(0, 12).unwrap();

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &attachment_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    {
        let mut encoder = command.render_command_encoder().unwrap();
        encoder.set_render_pipeline_state(&render).unwrap();
        encoder
            .set_index_buffer(&index_view, IndexFormat::Uint32)
            .unwrap();
        encoder
            .draw_indexed_primitives(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Clear([0xfe; 4]),
                FULL_SCREEN_TRIANGLE_VERTICES,
                None,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    command.commit().unwrap();
    command.wait_until_completed().unwrap();
    assert_eq!(
        attachment.read().unwrap(),
        [0x40, 0x80, 0xc0, 0xff].repeat(4)
    );

    let traces = provider.traces.lock().unwrap();
    assert_eq!(traces.len(), 1);
    let pass = traces[0]
        .render_passes()
        .next()
        .expect("the index-only pass reached the trace");
    assert!(
        pass.vertex_buffers.is_empty(),
        "an index-only draw selects through vertex_id"
    );
    assert_eq!(
        pass.indices,
        Some(IndexBufferBinding {
            view: stream_view(&index_view, 0, vec![0; 12]),
            format: IndexFormat::Uint32,
        })
    );
}

#[test]
fn a_stream_bound_at_a_skipped_index_has_no_position_to_land_at() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let render = render_pipeline_with_layout(
        &provider,
        &device,
        VertexLayout::Buffers(vec![stream_layout(0)]),
    );
    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);

    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    // Binding index 1 alone leaves no view for entry 0 of the pass's positional
    // list, so the draw is refused rather than drawing stream 1's bytes at
    // binding 0 — the position is the binding, and the contract holds the entry
    // to the label the view carries.
    encoder.set_vertex_buffer(1, &stream).unwrap();
    assert_eq!(
        encoder.draw_primitives(
            &attachment_view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            FULL_SCREEN_TRIANGLE_VERTICES,
            None,
        ),
        Err(Error::Contract(
            ContractError::VertexBufferBindingMismatch {
                index: 0,
                metal_binding: 1,
            }
        ))
    );
}

#[test]
fn draw_inputs_carry_the_bytes_the_command_commits_with() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0");
    let render = render_pipeline_with_layout(
        &provider,
        &device,
        VertexLayout::Buffers(vec![stream_layout(0)]),
    );

    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let (stream, bound_view) = buffer(&device, 0x11);

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &attachment_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    {
        let mut encoder = command.render_command_encoder().unwrap();
        encoder.set_render_pipeline_state(&render).unwrap();
        encoder.set_vertex_buffer(0, &bound_view).unwrap();
        encoder
            .draw_primitives(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Clear([0xfe; 4]),
                FULL_SCREEN_TRIANGLE_VERTICES,
                None,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    // A draw input is not a copy taken while recording: the bytes the command
    // commits with are the ones the trace uploads, the same boundary a compute
    // binding's bytes are taken at.
    stream.write(2, &[0x33; 4]).unwrap();
    command.commit().unwrap();
    command.wait_until_completed().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Completed);

    let traces = provider.traces.lock().unwrap();
    assert_eq!(traces.len(), 1);
    let pass = traces[0]
        .render_passes()
        .next()
        .expect("the stream draw reached the trace");
    assert_eq!(
        pass.vertex_buffers,
        vec![stream_view(&bound_view, 0, vec![0x33; 4])],
        "the pass carries the bytes the command committed with"
    );
}

#[test]
fn render_draw_indirect_replays_draw_indexed_and_refuses_bound_inputs() {
    let provider = Arc::new(
        FakeProvider::new()
            .with_render()
            .with_icb()
            .with_vertex_input(),
    );
    let device = Device::new(provider.clone());
    let declaring = device.compile_pipeline(request("declare")).unwrap();
    let render_metadata = render_metadata(&provider);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let (_, stream_view) = buffer(&device, 0x11);
    let indexed_icb = device
        .new_indirect_command_buffer(
            IndirectCommandKind::DrawIndexed,
            1,
            vec![IndirectCommandKind::DrawIndexed],
            IndirectCommandRange { start: 0, count: 1 },
            IndirectCommandDescriptor::DrawIndexed {
                index_count: 3,
                instance_count: 1,
            },
        )
        .unwrap();
    let dispatch_icb = device
        .new_indirect_command_buffer(
            IndirectCommandKind::Dispatch,
            1,
            vec![IndirectCommandKind::Dispatch],
            IndirectCommandRange { start: 0, count: 1 },
            IndirectCommandDescriptor::Dispatch {
                threadgroups: [1, 1, 1],
            },
        )
        .unwrap();

    // A render encoder still refuses a dispatch command, and the refusal names
    // the whole accepted set rather than one member of it.
    let probe = device.new_command_queue().command_buffer();
    {
        let mut encoder = probe.render_command_encoder().unwrap();
        encoder.set_render_pipeline_state(&render).unwrap();
        let error = encoder
            .draw_indirect(
                &dispatch_icb,
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Clear([0xfe; 4]),
                None,
            )
            .unwrap_err();
        assert_eq!(
            error,
            Error::IndirectKindMismatch {
                accepted: &[IndirectCommandKind::Draw, IndirectCommandKind::DrawIndexed],
                actual: IndirectCommandKind::Dispatch,
            }
        );
        assert!(
            error.to_string().contains("DrawIndexed"),
            "the refusal spells the accepted set: {error}"
        );
        // A stream bound for a direct draw cannot also be replayed: the ICB
        // supplies the draw's own input.
        encoder.set_vertex_buffer(0, &stream_view).unwrap();
        assert_eq!(
            encoder.draw_indirect(
                &indexed_icb,
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Clear([0xfe; 4]),
                None,
            ),
            Err(Error::IndirectReplayInputConflict {
                vertex_buffers: 1,
                index_buffer: false,
            })
        );
    }

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &attachment_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    {
        let mut encoder = command.render_command_encoder().unwrap();
        encoder.set_render_pipeline_state(&render).unwrap();
        encoder
            .draw_indirect(
                &indexed_icb,
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Clear([0xfe; 4]),
                None,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    command.commit().unwrap();
    command.wait_until_completed().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Completed);
    assert_eq!(
        attachment.read().unwrap(),
        [0x40, 0x80, 0xc0, 0xff].repeat(4)
    );

    let traces = provider.traces.lock().unwrap();
    assert_eq!(
        traces.len(),
        1,
        "the refused probe never reached a provider"
    );
    let payload = traces[0].indirect.as_ref().expect("indirect payload");
    assert_eq!(payload.command.kind(), IndirectCommandKind::DrawIndexed);
    let pass = traces[0]
        .render_passes()
        .next()
        .expect("the replayed pass reached the trace");
    assert_eq!(pass.vertices, 3, "the ICB's own index count");
    assert!(
        pass.vertex_buffers.is_empty() && pass.indices.is_none(),
        "the replay reads its input from the ICB payload, not from the pass"
    );
}
