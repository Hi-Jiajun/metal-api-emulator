use super::*;
use crate::provider::{
    allocate_device_epoch, AliasMode, AttachmentFormat, BufferAccess, BufferBindingContract,
    BufferWriteback, CompletionReadback, ComputeProvider, ComputeTrace, ContractError, FieldValue,
    FootprintProof, FunctionIdentity, FunctionSource, PipelineContract, PipelineId, PresentMode,
    ProviderErrorClass, ProviderHealth, ProviderPhase, RenderPipelineContract, Retryability,
    SemanticDigest, ShaderSource, StorageMode, SubmissionId, TextureAccess, TextureBindingContract,
    TextureFormat, TextureSource, TracePass, ValidatedComputeTrace, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexLayout,
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
    fragment_texture: bool,
    stage_buffers: bool,
    staged_lease: bool,
    depth_resolve: bool,
    stencil_resolve: bool,
    heap: bool,
    icb: bool,
    compute_textures: bool,
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
            fragment_texture: false,
            stage_buffers: false,
            staged_lease: false,
            depth_resolve: false,
            stencil_resolve: false,
            heap: false,
            icb: false,
            compute_textures: false,
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
    fn with_fragment_texture(mut self) -> Self {
        self.fragment_texture = true;
        self
    }
    fn with_stage_buffers(mut self) -> Self {
        self.stage_buffers = true;
        self
    }
    fn with_staged_lease(mut self) -> Self {
        self.staged_lease = true;
        self
    }
    fn with_depth_resolve(mut self) -> Self {
        self.depth_resolve = true;
        self
    }
    fn with_stencil_resolve(mut self) -> Self {
        self.stencil_resolve = true;
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
    /// Declare the compute texture face the object rails execute
    /// (`research/docs/26` §21.3–21.4): one sampled `R32Uint` texture and one
    /// `R32Float` storage image, whose landing the fixture publishes through
    /// the identity-keyed writeback channel.
    fn with_compute_textures(mut self) -> Self {
        self.compute_textures = true;
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

    /// The fragment textures the most recent render pass carried, in binding
    /// order, for the encoder-side render-sampler test
    /// (`research/docs/23` §3.3, v70).
    fn last_render_textures(&self) -> Vec<contract::TextureView> {
        self.traces
            .lock()
            .unwrap()
            .last()
            .and_then(|trace| {
                trace
                    .render_passes()
                    .next()
                    .map(|pass| pass.textures.clone())
            })
            .unwrap_or_default()
    }

    /// The stage buffers the most recent render pass carried, in canonical
    /// order, for the encoder-side stage-buffer tests
    /// (`research/docs/23` §3.3, v87).
    fn last_render_stage_buffers(&self) -> Vec<contract::StageBufferView> {
        self.last_render_pass()
            .map(|pass| pass.stage_buffers.clone())
            .unwrap_or_default()
    }

    /// The runtime samplers the most recent render pass carried, in canonical
    /// order, for the encoder-side runtime-sampler test
    /// (`research/docs/23` §3.3, v102).
    fn last_render_samplers(&self) -> Vec<contract::RenderSamplerBinding> {
        self.last_render_pass()
            .map(|pass| pass.samplers.clone())
            .unwrap_or_default()
    }

    /// The most recent render pass the provider received.
    fn last_render_pass(&self) -> Option<contract::RenderPassDescriptor> {
        self.traces
            .lock()
            .unwrap()
            .last()
            .and_then(|trace| trace.render_passes().next().cloned())
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
            // The object API's stage-buffer entry point
            // (`research/docs/23` §3.3, v87): the fixture provider declares the
            // face when a test asks for it, and the staged arm's source mode
            // when a test binds a staged lease.
            supports_render_stage_buffers: self.stage_buffers,
            max_render_stage_buffers: if self.stage_buffers {
                MAX_RENDER_STAGE_BUFFERS as u32
            } else {
                0
            },
            // The object API's compute texture face (`research/docs/23` §3.3,
            // v98): the fixture provider declares the sampled arm when a test
            // asks for it.
            supports_compute_texture_sampling: self.compute_textures,
            max_compute_textures: u32::from(self.compute_textures),
            supported_compute_texture_formats: if self.compute_textures {
                vec![TextureFormat::R32Uint, TextureFormat::R32Float]
            } else {
                Vec::new()
            },
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
            storage_modes: if self.staged_lease {
                vec![StorageMode::OwnedBytes, StorageMode::StagedLease]
            } else {
                vec![StorageMode::OwnedBytes]
            },
            host_readback: true,
            submit_only: false,
            supports_render_passes: self.render,
            // The fixture provider executes the contract's full attachment
            // list, so a multi-attachment trace is admitted rather than
            // refused at the rail's own one-attachment capability.
            max_color_attachments: if self.render {
                MAX_COLOR_ATTACHMENTS as u32
            } else {
                0
            },
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
            // The object rails execute the direct vertex-input shape, so the
            // fixture provider declares the same two instancing bits; the
            // instanced draw call itself is the next increment's encoder work
            // (`research/docs/23` §3.3, v31).
            supports_render_instancing: self.vertex_input,
            max_render_instances: if self.vertex_input { 4 } else { 0 },
            // The object rails record and execute the pass-wide raster, so the
            // fixture provider declares the two multisample bits beside its
            // render ones (`research/docs/23` §3.3, v51/v52).
            supports_render_multisample: self.render,
            max_render_sample_count: if self.render { 4 } else { 0 },
            // The fixture provider executes the Sample0 depth resolve when
            // asked to (`with_depth_resolve`); the default keeps the "cannot
            // resolve" shape and a resolving pass is refused during admission
            // (`research/docs/23` §3.3, v57/v58).
            supports_render_depth_resolve: self.render && self.depth_resolve,
            depth_resolve_modes: if self.depth_resolve {
                1u32 << u32::from(contract::DepthResolveFilter::Sample0.code())
            } else {
                0
            },
            // The fixture provider executes the Sample0 stencil resolve when
            // asked to (`with_stencil_resolve`); the default keeps the "cannot
            // resolve" shape and a resolving pass is refused during admission
            // (`research/docs/23` §3.3, v60/v70).
            supports_render_stencil_resolve: self.render && self.stencil_resolve,
            stencil_resolve_modes: if self.stencil_resolve {
                1u32 << u32::from(contract::StencilResolveFilter::Sample0.code())
            } else {
                0
            },
            // The fixture provider executes the render sampler only when the
            // texture flag is set on it, exactly as the depth resolve above
            // (`research/docs/23` §3.3, v70).
            supports_render_texture_sampling: self.render && self.fragment_texture,
            max_render_textures: u32::from(self.fragment_texture),
            supported_render_texture_formats: self
                .fragment_texture
                .then_some(TextureFormat::Rgba8Unorm)
                .into_iter()
                .collect(),
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
        // A storage image lands through the same identity-keyed channel a
        // buffer view uses (`research/docs/26` §21.4, C2): one writeback per
        // writable texture view, whole extent at offset zero. The fixture's
        // transform is a deterministic byte bump, so a handle that skipped the
        // landing cannot read as the landed bytes by accident.
        for texture in trace.serial_texture_resources().unwrap() {
            if texture.access != TextureAccess::Storage {
                continue;
            }
            let TextureSource::OwnedBytes(bytes) = &texture.source else {
                panic!("owned bytes required");
            };
            writebacks.push(BufferWriteback {
                allocation_id: texture.allocation_id,
                view_id: texture.view_id,
                offset: 0,
                bytes: bytes.iter().map(|byte| byte.wrapping_add(0x10)).collect(),
            });
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
        } else if request.entry_name == "sampled_texture" || request.entry_name == "storage_texture"
        {
            // The texture fixtures bind no buffers: the declaration is the
            // whole shape, exactly as a texture-only compute module's is.
            Vec::new()
        } else {
            vec![(request.entry_name.parse().unwrap(), BufferAccess::ReadWrite)]
        };
        let texture_bindings = match request.entry_name.as_str() {
            "sampled_texture" => vec![TextureBindingContract::sampled_r32uint(0)],
            "storage_texture" => vec![TextureBindingContract::storage(0, TextureFormat::R32Float)],
            _ => Vec::new(),
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
                texture_bindings,
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
            texture_bindings: Vec::new(),
            shader_capabilities: Vec::new(),
            translator_revision: None,
        },
        render: Some(RenderPipelineContract {
            stage_buffers: Vec::new(),
            vertex_entry: "vertex_main".into(),
            fragment_entry: "fragment_main".into(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::None,
            textures: Vec::new(),
        }),
    }
}

/// [`render_metadata`] with the one fragment texture its pass binds declared
/// (`research/docs/23` §3.3, v100).
///
/// The declaration is what the registration states about the module the
/// fragment stage came from, so a recording that binds the texture has to name
/// a pipeline whose contract states it — the sampled object fixture is the
/// reviewed pair, whose own MSL sibling carries a nearest/clamp-to-edge
/// sampler.
fn render_metadata_with_fragment_texture(provider: &FakeProvider) -> CompiledComputePipeline {
    let mut metadata = render_metadata(provider);
    if let Some(render) = metadata.render.as_mut() {
        render.textures = vec![TextureBindingContract::sampled(
            0,
            TextureFormat::Rgba8Unorm,
            crate::provider::SamplerPolicy {
                filter: crate::provider::SamplerFilter::Nearest,
                address: crate::provider::SamplerAddressMode::ClampToEdge,
            },
        )];
    }
    metadata
}

/// The render metadata one runtime-sampler case's pipeline carries
/// (`research/docs/23` §3.3, v102): the reviewed sampling shape's entries, with
/// a contract that pairs the one texture with the runtime `[[sampler(0)]]`
/// argument the encoder states.
fn render_metadata_with_runtime_sampler(provider: &FakeProvider) -> CompiledComputePipeline {
    let mut metadata = render_metadata(provider);
    if let Some(render) = metadata.render.as_mut() {
        render.textures = vec![TextureBindingContract::sampled_runtime(
            0,
            TextureFormat::Rgba8Unorm,
            0,
        )];
    }
    metadata
}

#[test]
fn a_recording_states_the_runtime_samplers_it_bound() {
    let provider = Arc::new(FakeProvider::new().with_render().with_fragment_texture());
    let device = Device::new(provider.clone());
    let render_metadata = render_metadata_with_runtime_sampler(&provider);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();
    let texels: Vec<u8> = (0..4u8)
        .flat_map(|y| (0..4u8).flat_map(move |x| [x, y, x.wrapping_add(y), 0xff]))
        .collect();
    let sampled = device
        .new_texture_with_bytes(TextureFormat::Rgba8Unorm, 4, 4, texels)
        .unwrap();
    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();

    // The declaring compute pass is what puts the attachment's allocation into
    // the trace, exactly as the texture test's setup does
    // (`research/docs/23` §3.6).
    let declaring = device.compile_pipeline(request("declare")).unwrap();
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
        // The encoder states the Metal state the `[[sampler(0)]]` argument
        // executes with, exactly as `setFragmentSamplerState(_:index:)` does
        // (`research/docs/23` §3.3, v102).
        let policy = contract::SamplerPolicy {
            filter: contract::SamplerFilter::Linear,
            address: contract::SamplerAddressMode::Repeat,
        };
        encoder.set_fragment_sampler_state(0, policy).unwrap();
        assert!(matches!(
            encoder.set_fragment_sampler_state(0, policy),
            Err(Error::FragmentSamplerAlreadyBound { index: 0 })
        ));
        assert!(matches!(
            encoder.set_fragment_sampler_state(MAX_RENDER_SAMPLERS as u32, policy),
            Err(Error::FragmentSamplerIndexOutOfRange { index, maximum: 16 })
                if index == MAX_RENDER_SAMPLERS as u32
        ));
        encoder.set_fragment_texture(0, &sampled).unwrap();
        encoder
            .draw_render_pass(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Clear([0xfe; 4]),
                None,
            )
            .expect("the runtime sampler pairing records");
        encoder.end_encoding().unwrap();
    }
    command.commit().unwrap();
    command.wait_until_completed().unwrap();

    // The state reaches the trace as the request's own declaration: exactly the
    // pair the registration names, in canonical order.
    let samplers = provider.last_render_samplers();
    assert_eq!(samplers.len(), 1, "one runtime sampler reaches the trace");
    assert_eq!(samplers[0].metal_binding, 0);
    assert_eq!(samplers[0].policy.filter, contract::SamplerFilter::Linear);
    assert_eq!(
        samplers[0].policy.address,
        contract::SamplerAddressMode::Repeat
    );
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
fn a_recording_binds_one_fragment_texture_in_binding_order() {
    // The object-API half of the render sampler (`research/docs/23` §3.3,
    // v70): the encoder binds one texture, the pass it records carries the
    // binding positionally with the texture's own bytes, and the two binding
    // refusals are the vertex streams' own (a repeated index, an index past
    // the contract's cap).
    let provider = Arc::new(FakeProvider::new().with_render().with_fragment_texture());
    let device = Device::new(provider.clone());
    let declaring = device.compile_pipeline(request("declare")).unwrap();
    let render_metadata = render_metadata_with_fragment_texture(&provider);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let texels: Vec<u8> = (0..4u8)
        .flat_map(|y| (0..4u8).flat_map(move |x| [x, y, x.wrapping_add(y), 0xff]))
        .collect();
    let sampled = device
        .new_texture_with_bytes(TextureFormat::Rgba8Unorm, 4, 4, texels.clone())
        .unwrap();
    let foreign = {
        let other = Device::new(Arc::new(
            FakeProvider::new().with_render().with_fragment_texture(),
        ));
        other
            .new_texture_with_bytes(TextureFormat::Rgba8Unorm, 4, 4, texels)
            .unwrap()
    };

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
        assert!(matches!(
            encoder.set_fragment_texture(0, &foreign),
            Err(Error::ForeignTexture)
        ));
        encoder.set_fragment_texture(0, &sampled).unwrap();
        assert!(matches!(
            encoder.set_fragment_texture(0, &sampled),
            Err(Error::FragmentTextureAlreadyBound { index: 0 })
        ));
        // The binding space is the contract's ceiling (`v102`): the entry at or
        // past [`MAX_RENDER_TEXTURES`] is the one refused by name, and the
        // indexes below it are the ones the wider contract admits.
        assert!(matches!(
            encoder.set_fragment_texture(MAX_RENDER_TEXTURES as u32, &sampled),
            Err(Error::FragmentTextureIndexOutOfRange {
                index,
                maximum: 8,
            }) if index == MAX_RENDER_TEXTURES as u32
        ));
        encoder
            .draw_render_pass(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Clear([0xfe; 4]),
                None,
            )
            .expect("a sampled pass records like the pre-v70 one");
        encoder.end_encoding().unwrap();
    }
    command.commit().unwrap();
    command.wait_until_completed().unwrap();

    // The trace the provider received carries the binding: position zero, the
    // texture's own view identity and its bytes, which is what makes the
    // sampling falsifiable instead of the driver's default.
    let textures = provider.last_render_textures();
    assert_eq!(textures.len(), 1, "one fragment texture reaches the trace");
    assert_eq!(textures[0].metal_binding, 0);
    assert_eq!(textures[0].view_id, sampled.view_id());
    assert_eq!(textures[0].format, TextureFormat::Rgba8Unorm);
    assert_eq!(textures[0].width, 4);
    assert_eq!(textures[0].height, 4);
    assert_eq!(
        textures[0].source,
        TextureSource::OwnedBytes(
            (0..4u8)
                .flat_map(|y| (0..4u8).flat_map(move |x| [x, y, x.wrapping_add(y), 0xff]))
                .collect()
        )
    );
}

/// The render metadata one stage-buffer case's pipeline carries
/// (`research/docs/23` §3.3, v83-v87): the reviewed contract's fields with the
/// slots the case declares.
fn render_metadata_with_stage_buffers(
    provider: &FakeProvider,
    stage_buffers: Vec<contract::StageBufferBinding>,
) -> CompiledComputePipeline {
    let mut metadata = render_metadata(provider);
    metadata
        .render
        .as_mut()
        .expect("render metadata carries a render half")
        .stage_buffers = stage_buffers;
    metadata
}

#[test]
fn a_render_pass_binds_its_stage_buffers_in_canonical_order() {
    // The object-API half of the stage-buffer face (`research/docs/23` §3.3,
    // v83-v87): the encoder states the slot and the bytes, the recorded
    // pipeline's declaration states the access, and the pass the provider
    // receives carries both — in the contract's canonical order whatever order
    // the bindings were recorded in.
    let provider = Arc::new(FakeProvider::new().with_render().with_stage_buffers());
    let device = Device::new(provider.clone());
    // The declaring pass reads the attachment the render pass stores into and
    // the view that becomes the writable slot's pool entry, exactly as a
    // render case's declaring pass declares both.
    let declaring = pipeline(&device, "declare");
    let render_metadata = render_metadata_with_stage_buffers(
        &provider,
        vec![
            contract::StageBufferBinding {
                stage: RenderPipelineStage::Vertex,
                index: 0,
                access: BufferAccess::Read,
                footprint: FootprintProof::Static { max_bytes: 8 },
            },
            contract::StageBufferBinding {
                stage: RenderPipelineStage::Fragment,
                index: 0,
                access: BufferAccess::Read,
                footprint: FootprintProof::Static { max_bytes: 4 },
            },
        ],
    );
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let positions = device.new_buffer_with_bytes((0..8u8).collect()).unwrap();
    let positions_view = positions.view(0, 8).unwrap();
    let tint = device.new_buffer_with_bytes(vec![0x11; 4]).unwrap();
    let tint_view = tint.view(0, 4).unwrap();
    let foreign = {
        let other = Device::new(Arc::new(
            FakeProvider::new().with_render().with_stage_buffers(),
        ));
        other
            .new_buffer_with_bytes(vec![0x22; 4])
            .unwrap()
            .view(0, 4)
            .unwrap()
    };

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
        assert!(matches!(
            encoder.set_stage_buffer(RenderPipelineStage::Vertex, 0, &foreign),
            Err(Error::ForeignBuffer)
        ));
        // Bound fragment-first on purpose: the pass still carries the canonical
        // order, so a binding's position cannot depend on the call order.
        encoder
            .set_stage_buffer(RenderPipelineStage::Fragment, 0, &tint_view)
            .unwrap();
        encoder
            .set_stage_buffer(RenderPipelineStage::Vertex, 0, &positions_view)
            .unwrap();
        assert!(matches!(
            encoder.set_stage_buffer(RenderPipelineStage::Vertex, 0, &positions_view),
            Err(Error::StageBufferAlreadyBound {
                stage: RenderPipelineStage::Vertex,
                index: 0,
            })
        ));
        assert!(matches!(
            encoder.set_stage_buffer(
                RenderPipelineStage::Vertex,
                MAX_RENDER_STAGE_BUFFER_INDEX,
                &positions_view,
            ),
            Err(Error::Contract(
                ContractError::RenderStageBufferIndexExceeded { index, .. }
            )) if index == MAX_RENDER_STAGE_BUFFER_INDEX
        ));
        encoder
            .draw_render_pass(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Clear([0xfe; 4]),
                None,
            )
            .expect("a stage-buffer pass records like the pre-v83 one");
        encoder.end_encoding().unwrap();
    }
    command.commit().unwrap();
    command.wait_until_completed().unwrap();

    let slots = provider.last_render_stage_buffers();
    assert_eq!(slots.len(), 2, "both bound slots reach the trace");
    assert_eq!(slots[0].stage, RenderPipelineStage::Vertex);
    assert_eq!(slots[0].view.metal_binding, 0);
    assert_eq!(slots[0].view.access, BufferAccess::Read);
    assert_eq!(slots[0].view.view_id, positions_view.view_id());
    assert_eq!(
        slots[0].view.source,
        BufferSource::OwnedBytes((0..8u8).collect())
    );
    assert_eq!(slots[1].stage, RenderPipelineStage::Fragment);
    assert_eq!(slots[1].view.access, BufferAccess::Read);
    assert_eq!(slots[1].view.view_id, tint_view.view_id());
    assert_eq!(
        slots[1].view.source,
        BufferSource::OwnedBytes(vec![0x11; 4])
    );
}

#[test]
fn a_writable_stage_buffer_slot_carries_the_declarations_own_access() {
    // A writable slot is the landing the writeback channel publishes
    // (`research/docs/23` §3.3, v86). The encoder states no access of its own,
    // so the pass carries the recorded pipeline's declaration, and the view is
    // the one the declaring compute pass already bound — the trace's own pool
    // entry the landing is keyed by.
    let provider = Arc::new(FakeProvider::new().with_render().with_stage_buffers());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_with_stage_buffers(
        &provider,
        vec![contract::StageBufferBinding {
            stage: RenderPipelineStage::Fragment,
            index: 1,
            access: BufferAccess::Write,
            footprint: FootprintProof::Static { max_bytes: 4 },
        }],
    );
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let sink = device.new_buffer_with_bytes(vec![0xcd; 4]).unwrap();
    let sink_view = sink.view(0, 4).unwrap();

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &attachment_view).unwrap();
        encoder.set_buffer(1, &sink_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    {
        let mut encoder = command.render_command_encoder().unwrap();
        encoder.set_render_pipeline_state(&render).unwrap();
        encoder
            .set_stage_buffer(RenderPipelineStage::Fragment, 1, &sink_view)
            .unwrap();
        encoder
            .draw_render_pass(
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

    let slots = provider.last_render_stage_buffers();
    assert_eq!(slots.len(), 1);
    assert_eq!(slots[0].view.access, BufferAccess::Write);
    assert_eq!(slots[0].view.view_id, sink_view.view_id());
    assert_eq!(slots[0].view.allocation_id, sink_view.allocation_id());
    assert_eq!(
        slots[0].view.source,
        BufferSource::OwnedBytes(vec![0xcd; 4])
    );
}

#[test]
fn a_stage_buffer_slot_the_pipeline_does_not_declare_is_refused() {
    let provider = Arc::new(FakeProvider::new().with_render().with_stage_buffers());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "declare");
    let render_metadata = render_metadata_with_stage_buffers(
        &provider,
        vec![contract::StageBufferBinding {
            stage: RenderPipelineStage::Vertex,
            index: 0,
            access: BufferAccess::Read,
            footprint: FootprintProof::Static { max_bytes: 4 },
        }],
    );
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let other = device.new_buffer_with_bytes(vec![0x11; 4]).unwrap();
    let other_view = other.view(0, 4).unwrap();

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &attachment_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    // The pipeline's declaration is the vertex slot alone: binding the
    // fragment slot would fill a descriptor no use covers, so the pass refuses
    // the pairing by name instead of dropping the binding.
    encoder
        .set_stage_buffer(RenderPipelineStage::Fragment, 0, &other_view)
        .unwrap();
    assert!(matches!(
        encoder.draw_render_pass(
            &attachment_view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            None,
        ),
        Err(Error::Contract(
            ContractError::UndeclaredStageBufferBinding {
                stage: RenderPipelineStage::Fragment,
                index: 0,
            }
        ))
    ));
    drop(encoder);
}

#[test]
fn a_lease_bound_stage_buffer_names_the_imported_reservation() {
    // The staged arm the trace rail already executes
    // (`research/docs/23` §90, R9i): the caller imports the lease through its
    // own provider channel and binds the reservation here, so the object rail's
    // pass carries the same `BufferSource` the trace rail's does. A commit that
    // completes proves the snapshot carried the owner's registration: admission
    // refuses a lease the resource table does not hold (`lease_not_admitted`
    // one layer down, `UnknownAllocation` here).
    let provider = Arc::new(
        FakeProvider::new()
            .with_render()
            .with_stage_buffers()
            .with_staged_lease(),
    );
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "declare");
    let render_metadata = render_metadata_with_stage_buffers(
        &provider,
        vec![contract::StageBufferBinding {
            stage: RenderPipelineStage::Fragment,
            index: 0,
            access: BufferAccess::Read,
            footprint: FootprintProof::Static { max_bytes: 4 },
        }],
    );
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let lease_id = contract::LeaseId::new(4242);
    let allocation = AllocationId::new(4241);
    let lease = StageBufferLease {
        reservation: LeaseReservation {
            lease: contract::BufferLease {
                lease_id,
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
            },
            offset: 0,
            length: 4,
        },
        allocation_size: 4096,
        arm: StageBufferLeaseArm::StagedLease,
    };

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
        // A reservation that reaches past the owner's registration is refused
        // by name, exactly as the trace's own resource table refuses it.
        assert!(matches!(
            encoder.set_stage_buffer_lease(
                RenderPipelineStage::Fragment,
                0,
                StageBufferLease {
                    allocation_size: 2,
                    ..lease
                },
            ),
            Err(Error::Contract(ContractError::LeaseRangeOutOfBounds { .. }))
        ));
        encoder
            .set_stage_buffer_lease(RenderPipelineStage::Fragment, 0, lease)
            .unwrap();
        encoder
            .draw_render_pass(
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

    let slots = provider.last_render_stage_buffers();
    assert_eq!(slots.len(), 1);
    assert_eq!(slots[0].view.allocation_id, allocation);
    assert_eq!(slots[0].view.offset, 0);
    assert_eq!(slots[0].view.length, 4);
    assert_eq!(slots[0].view.source, BufferSource::StagedLease(lease_id));
}

#[test]
fn a_writable_stage_buffer_slot_refuses_a_lease_binding() {
    // An imported lease has no host image the writeback channel could land in,
    // so the pairing is refused by name instead of executed as a pass whose
    // landing no rail could publish (`research/docs/23` §3.3, v86/v87).
    let provider = Arc::new(
        FakeProvider::new()
            .with_render()
            .with_stage_buffers()
            .with_staged_lease(),
    );
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "declare");
    let render_metadata = render_metadata_with_stage_buffers(
        &provider,
        vec![contract::StageBufferBinding {
            stage: RenderPipelineStage::Fragment,
            index: 0,
            access: BufferAccess::Write,
            footprint: FootprintProof::Static { max_bytes: 4 },
        }],
    );
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let attachment = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let attachment_view = attachment.view(0, 16).unwrap();
    let lease = StageBufferLease {
        reservation: LeaseReservation {
            lease: contract::BufferLease {
                lease_id: contract::LeaseId::new(4242),
                allocation_id: AllocationId::new(4241),
                owner_epoch: provider.device_epoch(),
            },
            offset: 0,
            length: 4,
        },
        allocation_size: 4096,
        arm: StageBufferLeaseArm::StagedLease,
    };

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &attachment_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder
        .set_stage_buffer_lease(RenderPipelineStage::Fragment, 0, lease)
        .unwrap();
    assert!(matches!(
        encoder.draw_render_pass(
            &attachment_view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            None,
        ),
        Err(Error::WritableStageBufferLeaseUnsupported {
            stage: RenderPipelineStage::Fragment,
            index: 0,
        })
    ));
    drop(encoder);
}

#[test]
fn a_draw_records_two_clear_attachments_in_location_order() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_multi(
        &provider,
        vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm],
    );
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let first = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let first_view = first.view(0, 16).unwrap();
    let second = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let second_view = second.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &first_view).unwrap();
        encoder.set_buffer(1, &second_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    {
        let mut encoder = command.render_command_encoder().unwrap();
        encoder.set_render_pipeline_state(&render).unwrap();
        encoder.set_vertex_buffer(0, &stream).unwrap();
        encoder
            .draw_primitives_with_attachments(
                &[
                    RenderColorAttachment {
                        view: &first_view,
                        format: AttachmentFormat::Rgba8Unorm,
                        load: RenderAttachmentLoad::Clear([0xa1; 4]),
                        store: StoreOp::Store,
                    },
                    RenderColorAttachment {
                        view: &second_view,
                        format: AttachmentFormat::Rgba8Unorm,
                        load: RenderAttachmentLoad::Clear([0xb2; 4]),
                        store: StoreOp::Store,
                    },
                ],
                2,
                2,
                FULL_SCREEN_TRIANGLE_VERTICES,
                None,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    command.commit().unwrap();
    command.wait_until_completed().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Completed);
    assert_eq!(first.read().unwrap(), [0x40, 0x80, 0xc0, 0xff].repeat(4));
    assert_eq!(second.read().unwrap(), [0x40, 0x80, 0xc0, 0xff].repeat(4));

    // The pass carries both attachments in list order, which is the location
    // order the fragment stage's outputs land at, and each one keeps the load
    // the caller recorded for its own position.
    let traces = provider.traces.lock().unwrap();
    assert_eq!(traces.len(), 1);
    let pass = traces[0].render_passes().next().expect("render pass");
    assert_eq!(pass.color_attachments.len(), 2);
    assert_eq!(pass.color_attachments[0].view_id, first_view.view_id());
    assert_eq!(pass.color_attachments[1].view_id, second_view.view_id());
    assert_eq!(
        pass.color_attachments[0].load,
        LoadOp::Clear(ClearColor::new([0xa1; 4]))
    );
    assert_eq!(
        pass.color_attachments[1].load,
        LoadOp::Clear(ClearColor::new([0xb2; 4]))
    );
}

#[test]
fn two_loading_attachments_keep_their_own_snapshotted_bytes() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    // One declaring compute pass binds both attachment views, so the trace's
    // own view declarations carry each attachment's bytes for the rail to
    // upload before the loading pass opens.
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_multi(
        &provider,
        vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm],
    );
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let first = device.new_buffer_with_bytes(vec![0xaa; 16]).unwrap();
    let first_view = first.view(0, 16).unwrap();
    let second = device.new_buffer_with_bytes(vec![0xbb; 16]).unwrap();
    let second_view = second.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &first_view).unwrap();
        encoder.set_buffer(1, &second_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    {
        let mut encoder = command.render_command_encoder().unwrap();
        encoder.set_render_pipeline_state(&render).unwrap();
        encoder.set_vertex_buffer(0, &stream).unwrap();
        encoder
            .draw_primitives_with_attachments(
                &[
                    RenderColorAttachment {
                        view: &first_view,
                        format: AttachmentFormat::Rgba8Unorm,
                        load: RenderAttachmentLoad::Load,
                        store: StoreOp::Store,
                    },
                    RenderColorAttachment {
                        view: &second_view,
                        format: AttachmentFormat::Rgba8Unorm,
                        load: RenderAttachmentLoad::Load,
                        store: StoreOp::Store,
                    },
                ],
                2,
                2,
                FULL_SCREEN_TRIANGLE_VERTICES,
                None,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    command.commit().unwrap();
    command.wait_until_completed().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Completed);

    let traces = provider.traces.lock().unwrap();
    assert_eq!(traces.len(), 1);
    // Each attachment's load reaches the provider as `LoadOp::Load`, never as
    // a clear the provider would have to reinterpret, and the two views stay
    // in location order.
    let pass = traces[0].render_passes().next().expect("render pass");
    assert_eq!(pass.color_attachments.len(), 2);
    assert_eq!(pass.color_attachments[0].view_id, first_view.view_id());
    assert_eq!(pass.color_attachments[0].load, LoadOp::Load);
    assert_eq!(pass.color_attachments[1].view_id, second_view.view_id());
    assert_eq!(pass.color_attachments[1].load, LoadOp::Load);
    // The two declarations snapshot each attachment's own bytes: the values
    // differ and both are preserved in the trace the provider uploads from.
    let declaring_pass = traces[0].compute_passes().next().expect("declaring pass");
    let first_bytes = declaring_pass
        .buffers
        .iter()
        .find(|view| view.view_id == first_view.view_id())
        .expect("first attachment declaration");
    let second_bytes = declaring_pass
        .buffers
        .iter()
        .find(|view| view.view_id == second_view.view_id())
        .expect("second attachment declaration");
    assert_ne!(first_bytes.source, second_bytes.source);
    assert_eq!(first_bytes.source, BufferSource::OwnedBytes(vec![0xaa; 16]));
    assert_eq!(
        second_bytes.source,
        BufferSource::OwnedBytes(vec![0xbb; 16])
    );
}

#[test]
fn a_draw_projects_each_attachment_store_to_its_own_location() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_multi(
        &provider,
        vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm],
    );
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let first = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let first_view = first.view(0, 16).unwrap();
    let second = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let second_view = second.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &first_view).unwrap();
        encoder.set_buffer(1, &second_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    {
        let mut encoder = command.render_command_encoder().unwrap();
        encoder.set_render_pipeline_state(&render).unwrap();
        encoder.set_vertex_buffer(0, &stream).unwrap();
        encoder
            .draw_primitives_with_attachments(
                &[
                    RenderColorAttachment {
                        view: &first_view,
                        format: AttachmentFormat::Rgba8Unorm,
                        load: RenderAttachmentLoad::Clear([0xa1; 4]),
                        store: StoreOp::Store,
                    },
                    RenderColorAttachment {
                        view: &second_view,
                        format: AttachmentFormat::Rgba8Unorm,
                        load: RenderAttachmentLoad::Clear([0xb2; 4]),
                        store: StoreOp::DontCare,
                    },
                ],
                2,
                2,
                FULL_SCREEN_TRIANGLE_VERTICES,
                None,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    command.commit().unwrap();
    command.wait_until_completed().unwrap();
    assert_eq!(command.status().unwrap(), CommandBufferStatus::Completed);

    // The pass carries one store decision per location, in list order: the
    // stored attachment stays observable and the discarded one is handed to
    // the provider as `DontCare` at its own location, never flattened to one
    // shared op.
    let traces = provider.traces.lock().unwrap();
    assert_eq!(traces.len(), 1);
    let pass = traces[0].render_passes().next().expect("render pass");
    assert_eq!(pass.color_attachments.len(), 2);
    assert_eq!(pass.color_attachments[0].view_id, first_view.view_id());
    assert_eq!(pass.color_attachments[0].store, StoreOp::Store);
    assert_eq!(pass.color_attachments[1].view_id, second_view.view_id());
    assert_eq!(pass.color_attachments[1].store, StoreOp::DontCare);
}

#[test]
fn a_draw_refuses_to_discard_every_attachment() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let render = render_pipeline_with_layout(
        &provider,
        &device,
        VertexLayout::Buffers(vec![stream_layout(0)]),
    );
    let (_, stream) = buffer(&device, 0x11);
    let first = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let first_view = first.view(0, 16).unwrap();
    let second = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let second_view = second.view(0, 16).unwrap();

    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    // An all-discarded pass has no observable landing point, so "nothing
    // landed" could pass as "landed correctly"; the contract refuses the
    // whole shape at recording time, when the descriptor is validated.
    assert_eq!(
        encoder.draw_primitives_with_attachments(
            &[
                RenderColorAttachment {
                    view: &first_view,
                    format: AttachmentFormat::Rgba8Unorm,
                    load: RenderAttachmentLoad::Clear([0xfe; 4]),
                    store: StoreOp::DontCare,
                },
                RenderColorAttachment {
                    view: &second_view,
                    format: AttachmentFormat::Rgba8Unorm,
                    load: RenderAttachmentLoad::Clear([0xfd; 4]),
                    store: StoreOp::DontCare,
                },
            ],
            2,
            2,
            FULL_SCREEN_TRIANGLE_VERTICES,
            None,
        ),
        Err(Error::Contract(
            ContractError::AllRenderAttachmentsDiscarded
        ))
    );
    // The refusal landed no pass, so the encoder can still record a valid
    // pass that stores one attachment and end cleanly.
    encoder
        .draw_primitives_with_attachments(
            &[RenderColorAttachment {
                view: &first_view,
                format: AttachmentFormat::Rgba8Unorm,
                load: RenderAttachmentLoad::Clear([0xfe; 4]),
                store: StoreOp::Store,
            }],
            2,
            2,
            FULL_SCREEN_TRIANGLE_VERTICES,
            None,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
}

#[test]
fn multi_attachment_draws_refuse_empty_duplicate_and_excess_lists() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let render = render_pipeline_with_layout(
        &provider,
        &device,
        VertexLayout::Buffers(vec![stream_layout(0)]),
    );
    let (_, stream) = buffer(&device, 0x11);
    let pool = device.new_buffer_with_bytes(vec![0xfe; 80]).unwrap();
    let views = (0..5)
        .map(|position| pool.view(position * 16, 16).unwrap())
        .collect::<Vec<_>>();

    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    let shapes = views
        .iter()
        .map(|view| RenderColorAttachment {
            view,
            format: AttachmentFormat::Rgba8Unorm,
            load: RenderAttachmentLoad::Clear([0xfe; 4]),
            store: StoreOp::Store,
        })
        .collect::<Vec<_>>();

    // An empty attachment list names no target for any fragment output.
    assert_eq!(
        encoder.draw_primitives_with_attachments(&[], 2, 2, FULL_SCREEN_TRIANGLE_VERTICES, None,),
        Err(Error::EmptyRenderAttachmentList)
    );
    // The same `(allocation, view)` identity twice would name one target for
    // two locations.
    assert_eq!(
        encoder.draw_primitives_with_attachments(
            &[shapes[0], shapes[0]],
            2,
            2,
            FULL_SCREEN_TRIANGLE_VERTICES,
            None,
        ),
        Err(Error::DuplicateRenderAttachment)
    );
    // Five locations exceed the contract's cap; the refusal names both sides.
    assert_eq!(
        encoder.draw_primitives_with_attachments(
            &shapes,
            2,
            2,
            FULL_SCREEN_TRIANGLE_VERTICES,
            None,
        ),
        Err(Error::RenderAttachmentLimitExceeded {
            requested: 5,
            maximum: 4,
        })
    );
    // None of the refusals landed a pass, so the encoder can still record a
    // valid draw and end cleanly.
    encoder
        .draw_primitives_with_attachments(&[shapes[0]], 2, 2, FULL_SCREEN_TRIANGLE_VERTICES, None)
        .unwrap();
    encoder.end_encoding().unwrap();
}

#[test]
fn an_instanced_draw_records_the_instance_count_and_refuses_zero() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    // The bound stream means the pipeline has to carry the reviewed layout, or
    // the recorded pass would disagree with the pipeline about its bindings.
    let render_metadata = render_metadata_multi(&provider, vec![AttachmentFormat::Rgba8Unorm]);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let target = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let view = target.view(0, 16).unwrap();
    let scratch = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let scratch_view = scratch.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);
    let index = device
        .new_buffer_with_bytes(vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        .unwrap();
    let index_view = index.view(0, 16).unwrap();

    let command = device.new_command_queue().command_buffer();
    // The declaring pass is what puts the attachment view in the trace's
    // resource table, exactly as every other render fixture declares it.
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &view).unwrap();
        encoder.set_buffer(1, &scratch_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();

    // Zero instances is not a draw: the encoder refuses it with the same
    // contract error a zero-count pass would produce
    // (`research/docs/23` §3.3, v31/v32), before any pass is recorded.
    assert_eq!(
        encoder.draw_primitives_instanced(
            &view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            FULL_SCREEN_TRIANGLE_VERTICES,
            0,
            None,
        ),
        Err(ContractError::ZeroLength("render instance count").into())
    );
    assert_eq!(
        encoder.draw_indexed_primitives_instanced(
            &view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            6,
            0,
            None,
        ),
        Err(ContractError::ZeroLength("render instance count").into())
    );

    // The two-instance shapes record; the commit is what turns them into the
    // pass the rails read, and the pass carries the count the API named.
    encoder
        .draw_indexed_primitives_instanced(
            &view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            6,
            2,
            None,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();
    let trace = provider.traces.lock().unwrap().last().cloned().unwrap();
    let pass = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .expect("the recorded draw is a render pass");
    assert_eq!(pass.instance_count, 2);
    assert_eq!(pass.vertices, 6);
    assert!(pass.indices.is_some());
    // The declaring compute pass is still the trace's first pass, so the
    // attachment resolves exactly as it does for every other render case.
    assert_eq!(trace.passes.len(), 2);
}

#[test]
fn a_base_vertex_draw_records_the_offset_and_needs_an_index_buffer() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_multi(&provider, vec![AttachmentFormat::Rgba8Unorm]);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let target = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let view = target.view(0, 16).unwrap();
    let scratch = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let scratch_view = scratch.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);
    let index = device
        .new_buffer_with_bytes(vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        .unwrap();
    let index_view = index.view(0, 16).unwrap();

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &view).unwrap();
        encoder.set_buffer(1, &scratch_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();

    // The offset is an indexed-draw parameter, so an encoder without an index
    // buffer is refused before any pass is recorded — the same rule the
    // contract states as `BaseVertexRequiresIndices`
    // (`research/docs/23` §3.3, v34/v35).
    assert_eq!(
        encoder.draw_indexed_primitives_base_vertex(
            &view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            6,
            1,
            1,
            None,
        ),
        Err(Error::MissingIndexBuffer)
    );

    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    encoder
        .draw_indexed_primitives_base_vertex(
            &view,
            AttachmentFormat::Rgba8Unorm,
            2,
            2,
            RenderAttachmentLoad::Clear([0xfe; 4]),
            6,
            1,
            1,
            None,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();

    let trace = provider.traces.lock().unwrap().last().cloned().unwrap();
    let pass = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .expect("the recorded draw is a render pass");
    assert_eq!(pass.base_vertex, 1);
    assert_eq!(pass.instance_count, 1);
    assert_eq!(pass.vertices, 6);
    assert!(pass.indices.is_some());
}

#[test]
fn a_depth_draw_records_the_surface_and_refuses_a_mismatched_extent() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_multi(&provider, vec![AttachmentFormat::Rgba8Unorm]);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let target = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let view = target.view(0, 16).unwrap();
    let scratch = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let scratch_view = scratch.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);
    let index = device
        .new_buffer_with_bytes(vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        .unwrap();
    let index_view = index.view(0, 16).unwrap();

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &view).unwrap();
        encoder.set_buffer(1, &scratch_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();

    // The depth surface is a second raster with the pass's own extent: a
    // surface a different size than the colour attachments is refused by the
    // contract's own rule when the pass is recorded
    // (`research/docs/23` §3.3, v36/v37).
    assert_eq!(
        encoder.draw_indexed_primitives_with_depth(
            &[RenderColorAttachment {
                view: &view,
                format: AttachmentFormat::Rgba8Unorm,
                load: RenderAttachmentLoad::Clear([0xfe; 4]),
                store: StoreOp::Store,
            }],
            2,
            2,
            6,
            1,
            RenderDepthAttachment {
                width: 4,
                height: 2,
                load: RenderDepthLoad::Clear(1.0),
                store: None,
                identity: None,
            },
            Some(RenderDepthTest {
                compare: contract::CompareFunction::Less,
                write: true,
            }),
            None,
        ),
        Err(ContractError::DepthExtentMismatch {
            raster: [2, 2],
            depth: [4, 2],
        }
        .into())
    );

    // The reviewed shape records, and the committed trace carries both the
    // rail-owned surface and the state.
    encoder
        .draw_indexed_primitives_with_depth(
            &[RenderColorAttachment {
                view: &view,
                format: AttachmentFormat::Rgba8Unorm,
                load: RenderAttachmentLoad::Clear([0xfe; 4]),
                store: StoreOp::Store,
            }],
            2,
            2,
            6,
            1,
            RenderDepthAttachment {
                width: 2,
                height: 2,
                load: RenderDepthLoad::Clear(1.0),
                store: None,
                identity: None,
            },
            Some(RenderDepthTest {
                compare: contract::CompareFunction::Less,
                write: true,
            }),
            None,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();

    let trace = provider.traces.lock().unwrap().last().cloned().unwrap();
    let pass = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .expect("the recorded draw is a render pass");
    let depth = pass.depth.as_ref().expect("the pass opens a depth surface");
    assert_eq!((depth.width, depth.height), (2, 2));
    assert_eq!(depth.load, contract::DepthLoadOp::Clear(1.0f32.to_bits()));
    let test = pass.depth_test.expect("the pass tests depths");
    assert_eq!(test.compare, contract::CompareFunction::Less);
    assert!(test.write);

    // v44: the recording can keep the surface and name where its texels land,
    // and the recorded pass carries exactly that pair — the same two fields the
    // trace contract validates, because the object API's depth surface is one
    // description rather than a second one (`research/docs/23` §3.3, v43/v44).
    let stored = device.new_command_queue().command_buffer();
    {
        let mut declaring_encoder = stored.compute_command_encoder().unwrap();
        declaring_encoder
            .set_compute_pipeline_state(&declaring)
            .unwrap();
        declaring_encoder.set_buffer(0, &view).unwrap();
        declaring_encoder.set_buffer(1, &scratch_view).unwrap();
        dispatch(&mut declaring_encoder).unwrap();
        declaring_encoder.end_encoding().unwrap();
    }
    let mut encoder = stored.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    let identity = contract::RenderDepthIdentity {
        allocation_id: scratch_view.allocation_id(),
        view_id: scratch_view.view_id(),
    };
    encoder
        .draw_indexed_primitives_with_depth(
            &[RenderColorAttachment {
                view: &view,
                format: AttachmentFormat::Rgba8Unorm,
                load: RenderAttachmentLoad::Clear([0xfe; 4]),
                store: StoreOp::Store,
            }],
            2,
            2,
            6,
            1,
            RenderDepthAttachment {
                width: 2,
                height: 2,
                load: RenderDepthLoad::Clear(1.0),
                store: Some(contract::DepthStoreOp::Store),
                identity: Some(identity),
            },
            Some(RenderDepthTest {
                compare: contract::CompareFunction::Less,
                write: true,
            }),
            None,
        )
        .expect("a recording that keeps its depth surface is the v44 shape");
    encoder.end_encoding().unwrap();
    stored.commit().unwrap();

    let trace = provider.traces.lock().unwrap().last().cloned().unwrap();
    let depth = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .and_then(|pass| pass.depth.clone())
        .expect("the recorded pass opens a depth surface");
    assert_eq!(depth.store, Some(contract::DepthStoreOp::Store));
    assert_eq!(depth.identity, Some(identity));

    // The pair is still one decision: keeping the surface without naming a
    // landing is refused when the pass is recorded, exactly as the trace
    // contract refuses it.
    let mut encoder = device
        .new_command_queue()
        .command_buffer()
        .render_command_encoder()
        .unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    let refused = encoder
        .draw_indexed_primitives_with_depth(
            &[RenderColorAttachment {
                view: &view,
                format: AttachmentFormat::Rgba8Unorm,
                load: RenderAttachmentLoad::Clear([0xfe; 4]),
                store: StoreOp::Store,
            }],
            2,
            2,
            6,
            1,
            RenderDepthAttachment {
                width: 2,
                height: 2,
                load: RenderDepthLoad::Clear(1.0),
                store: Some(contract::DepthStoreOp::Store),
                identity: None,
            },
            Some(RenderDepthTest {
                compare: contract::CompareFunction::Less,
                write: true,
            }),
            None,
        )
        .expect_err("a stored depth surface needs its landing");
    assert_eq!(
        refused,
        ContractError::DepthStoreIdentityMismatch {
            store: Some(contract::DepthStoreOp::Store.code()),
            identity: false,
        }
        .into()
    );
}

#[test]
fn a_depth_bearing_multisample_draw_records_both_halves() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_multi(&provider, vec![AttachmentFormat::Rgba8Unorm]);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let target = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let view = target.view(0, 16).unwrap();
    let scratch = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let scratch_view = scratch.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);
    let index = device
        .new_buffer_with_bytes(vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        .unwrap();
    let index_view = index.view(0, 16).unwrap();
    let attachment = RenderColorAttachment {
        view: &view,
        format: AttachmentFormat::Rgba8Unorm,
        load: RenderAttachmentLoad::Clear([0x11, 0x22, 0x33, 0x44]),
        store: StoreOp::Store,
    };
    let state = contract::MultisampleState {
        sample_count: contract::SampleCount::Four,
    };
    let depth = RenderDepthAttachment {
        width: 2,
        height: 2,
        load: RenderDepthLoad::Clear(1.0),
        store: None,
        identity: None,
    };
    let test = RenderDepthTest {
        compare: contract::CompareFunction::Less,
        write: true,
    };

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &view).unwrap();
        encoder.set_buffer(1, &scratch_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();

    // The combined entry records both halves of the reviewed shape: the
    // pass-wide raster and the rail-owned depth surface the pass tests
    // (`research/docs/23` §3.3, v53/v54).
    encoder
        .draw_indexed_primitives_with_multisample_depth(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            depth,
            Some(test),
            state,
            None,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();

    let trace = provider.traces.lock().unwrap().last().cloned().unwrap();
    let pass = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .expect("the recorded draw is a render pass");
    assert_eq!(pass.multisample, Some(state));
    let recorded = pass.depth.as_ref().expect("the pass opens a depth surface");
    assert_eq!((recorded.width, recorded.height), (2, 2));
    assert_eq!(recorded.store, None);
    assert_eq!(
        pass.depth_test,
        Some(contract::DepthTest {
            compare: contract::CompareFunction::Less,
            write: true,
        })
    );

    // A stored depth surface beside the raster is the shape the contract
    // refuses — resolving it needs the depth resolve filters — so the
    // recording refuses it too instead of recording a pass admission would
    // reject (`research/docs/23` §3.3, v53).
    let mut encoder = device
        .new_command_queue()
        .command_buffer()
        .render_command_encoder()
        .unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    assert_eq!(
        encoder.draw_indexed_primitives_with_multisample_depth(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            RenderDepthAttachment {
                width: 2,
                height: 2,
                load: RenderDepthLoad::Clear(1.0),
                store: Some(contract::DepthStoreOp::Store),
                identity: None,
            },
            Some(test),
            state,
            None,
        ),
        Err(ContractError::MultisampleDepthStoreUnsupported.into())
    );
}

#[test]
fn a_stored_depth_multisample_draw_records_its_resolve() {
    let provider = Arc::new(
        FakeProvider::new()
            .with_render()
            .with_vertex_input()
            .with_depth_resolve(),
    );
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_multi(&provider, vec![AttachmentFormat::Rgba8Unorm]);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let target = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let view = target.view(0, 16).unwrap();
    let scratch = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let scratch_view = scratch.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);
    let index = device
        .new_buffer_with_bytes(vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        .unwrap();
    let index_view = index.view(0, 16).unwrap();
    let attachment = RenderColorAttachment {
        view: &view,
        format: AttachmentFormat::Rgba8Unorm,
        load: RenderAttachmentLoad::Clear([0x11, 0x22, 0x33, 0x44]),
        store: StoreOp::Store,
    };
    let state = contract::MultisampleState {
        sample_count: contract::SampleCount::Four,
    };
    let identity = contract::RenderDepthIdentity {
        allocation_id: scratch_view.allocation_id(),
        view_id: scratch_view.view_id(),
    };
    let depth = RenderDepthAttachment {
        width: 2,
        height: 2,
        load: RenderDepthLoad::Clear(1.0),
        store: Some(contract::DepthStoreOp::Store),
        identity: Some(identity),
    };
    let test = RenderDepthTest {
        compare: contract::CompareFunction::Less,
        write: true,
    };

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &view).unwrap();
        encoder.set_buffer(1, &scratch_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();

    // The resolving entry records all three halves of the reviewed shape: the
    // pass-wide raster, the stored depth surface the pass tests and the
    // resolve that reduces its samples (`research/docs/23` §3.3, v57/v58).
    encoder
        .draw_indexed_primitives_with_multisample_depth_resolve(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            depth,
            Some(test),
            state,
            contract::DepthResolveFilter::Sample0,
            None,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();

    let trace = provider.traces.lock().unwrap().last().cloned().unwrap();
    let pass = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .expect("the recorded draw is a render pass");
    assert_eq!(pass.multisample, Some(state));
    assert_eq!(
        pass.depth_resolve,
        Some(contract::MultisampleDepthResolve {
            filter: contract::DepthResolveFilter::Sample0,
        })
    );
    let recorded = pass.depth.as_ref().expect("the pass opens a depth surface");
    assert_eq!(recorded.store, Some(contract::DepthStoreOp::Store));
    assert_eq!(recorded.identity, Some(identity));

    // The resolve only means something beside a stored multisampled depth
    // surface: a single-sample raster, a surface the pass drops and a stored
    // surface without a landing are all refused at recording with the
    // contract's own errors (`research/docs/23` §3.3, v51/v57).
    let mut refusal = device
        .new_command_queue()
        .command_buffer()
        .render_command_encoder()
        .unwrap();
    refusal.set_render_pipeline_state(&render).unwrap();
    refusal.set_vertex_buffer(0, &stream).unwrap();
    refusal
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    assert_eq!(
        refusal.draw_indexed_primitives_with_multisample_depth_resolve(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            RenderDepthAttachment {
                width: 2,
                height: 2,
                load: RenderDepthLoad::Clear(1.0),
                store: Some(contract::DepthStoreOp::Store),
                identity: Some(identity),
            },
            Some(test),
            contract::MultisampleState {
                sample_count: contract::SampleCount::One,
            },
            contract::DepthResolveFilter::Sample0,
            None,
        ),
        Err(ContractError::SingleSampleMultisampleState.into())
    );
    assert_eq!(
        refusal.draw_indexed_primitives_with_multisample_depth_resolve(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            RenderDepthAttachment {
                width: 2,
                height: 2,
                load: RenderDepthLoad::Clear(1.0),
                store: None,
                identity: None,
            },
            Some(test),
            state,
            contract::DepthResolveFilter::Sample0,
            None,
        ),
        Err(ContractError::DepthResolveWithoutStoredDepth { store: None }.into())
    );
    assert_eq!(
        refusal.draw_indexed_primitives_with_multisample_depth_resolve(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            RenderDepthAttachment {
                width: 2,
                height: 2,
                load: RenderDepthLoad::Clear(1.0),
                store: Some(contract::DepthStoreOp::Store),
                identity: None,
            },
            Some(test),
            state,
            contract::DepthResolveFilter::Sample0,
            None,
        ),
        Err(ContractError::DepthStoreIdentityMismatch {
            store: Some(contract::DepthStoreOp::Store.code()),
            identity: false,
        }
        .into())
    );
}

#[test]
fn a_stencil_bearing_multisample_draw_records_both_halves() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_multi(&provider, vec![AttachmentFormat::Rgba8Unorm]);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let target = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let view = target.view(0, 16).unwrap();
    let scratch = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let scratch_view = scratch.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);
    let index = device
        .new_buffer_with_bytes(vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        .unwrap();
    let index_view = index.view(0, 16).unwrap();
    let attachment = RenderColorAttachment {
        view: &view,
        format: AttachmentFormat::Rgba8Unorm,
        load: RenderAttachmentLoad::Clear([0x11, 0x22, 0x33, 0x44]),
        store: StoreOp::Store,
    };
    let state = contract::MultisampleState {
        sample_count: contract::SampleCount::Four,
    };
    let stencil = RenderStencilAttachment {
        width: 2,
        height: 2,
        load: RenderStencilLoad::Clear(0),
        store: None,
        identity: None,
    };
    let test = RenderStencilTest {
        compare: contract::StencilCompare::Equal,
        fail_op: contract::StencilOp::Keep,
        depth_fail_op: contract::StencilOp::Keep,
        pass_op: contract::StencilOp::IncrementWrap,
        read_mask: 0xff,
        write_mask: 0xff,
        reference: 0,
    };

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &view).unwrap();
        encoder.set_buffer(1, &scratch_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    encoder
        .draw_indexed_primitives_with_multisample_stencil(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            stencil,
            Some(test),
            state,
            None,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();

    let trace = provider.traces.lock().unwrap().last().cloned().unwrap();
    let pass = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .expect("the recorded draw is a render pass");
    assert_eq!(pass.multisample, Some(state));
    let recorded = pass
        .stencil
        .as_ref()
        .expect("the pass opens a stencil surface");
    assert_eq!((recorded.width, recorded.height), (2, 2));
    assert_eq!(recorded.store, None);
    assert_eq!(pass.depth, None);
    let recorded_test = pass.stencil_test.expect("the pass tests the stencil");
    assert_eq!(recorded_test.compare, contract::StencilCompare::Equal);
    assert_eq!(recorded_test.pass_op, contract::StencilOp::IncrementWrap);

    // A stored stencil surface beside the raster is the shape the contract
    // refuses — the stencil resolve is a later increment — so the recording
    // refuses it too (`research/docs/23` §3.3, v55).
    let mut encoder = device
        .new_command_queue()
        .command_buffer()
        .render_command_encoder()
        .unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    assert_eq!(
        encoder.draw_indexed_primitives_with_multisample_stencil(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            RenderStencilAttachment {
                width: 2,
                height: 2,
                load: RenderStencilLoad::Clear(0),
                store: Some(StoreOp::Store),
                identity: None,
            },
            Some(test),
            state,
            None,
        ),
        Err(ContractError::MultisampleStencilStoreUnsupported.into())
    );
}

#[test]
fn a_combined_depth_stencil_multisample_draw_records_both_faces() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_multi(&provider, vec![AttachmentFormat::Rgba8Unorm]);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let target = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let view = target.view(0, 16).unwrap();
    let scratch = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let scratch_view = scratch.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);
    let index = device
        .new_buffer_with_bytes(vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        .unwrap();
    let index_view = index.view(0, 16).unwrap();
    let attachment = RenderColorAttachment {
        view: &view,
        format: AttachmentFormat::Rgba8Unorm,
        load: RenderAttachmentLoad::Clear([0x11, 0x22, 0x33, 0x44]),
        store: StoreOp::Store,
    };
    let state = contract::MultisampleState {
        sample_count: contract::SampleCount::Four,
    };
    let depth = RenderDepthAttachment {
        width: 2,
        height: 2,
        load: RenderDepthLoad::Clear(1.0),
        store: None,
        identity: None,
    };
    let depth_test = RenderDepthTest {
        compare: contract::CompareFunction::Less,
        write: true,
    };
    let stencil = RenderStencilAttachment {
        width: 2,
        height: 2,
        load: RenderStencilLoad::Clear(0),
        store: None,
        identity: None,
    };
    let stencil_test = RenderStencilTest {
        compare: contract::StencilCompare::Equal,
        fail_op: contract::StencilOp::Keep,
        depth_fail_op: contract::StencilOp::IncrementWrap,
        pass_op: contract::StencilOp::Keep,
        read_mask: 0xff,
        write_mask: 0xff,
        reference: 0,
    };

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &view).unwrap();
        encoder.set_buffer(1, &scratch_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();

    // The combined entry records every half of the v66 shape at once: the
    // pass-wide raster, the one rail-owned surface both faces share, and the
    // two tests the pass drives that surface with (`research/docs/23` §3.3,
    // v66/v68).
    encoder
        .draw_indexed_primitives_with_multisample_depth_stencil(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            state,
            depth,
            stencil,
            Some(depth_test),
            Some(stencil_test),
            None,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();

    let trace = provider.traces.lock().unwrap().last().cloned().unwrap();
    let pass = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .expect("the recorded draw is a render pass");
    assert_eq!(pass.multisample, Some(state));
    let recorded_depth = pass.depth.as_ref().expect("the pass opens a depth surface");
    assert_eq!((recorded_depth.width, recorded_depth.height), (2, 2));
    assert_eq!(recorded_depth.store, None);
    let recorded_stencil = pass
        .stencil
        .as_ref()
        .expect("the pass opens a stencil surface");
    assert_eq!((recorded_stencil.width, recorded_stencil.height), (2, 2));
    assert_eq!(recorded_stencil.store, None);
    assert_eq!(
        pass.depth_test,
        Some(contract::DepthTest {
            compare: contract::CompareFunction::Less,
            write: true,
        })
    );
    let recorded_test = pass
        .stencil_test
        .expect("the pass tests the stencil surface");
    assert_eq!(recorded_test.compare, contract::StencilCompare::Equal);
    assert_eq!(
        recorded_test.depth_fail_op,
        contract::StencilOp::IncrementWrap
    );
    // Both faces are rail-owned here: the pair states no resolve, exactly the
    // shape whose observation is the colour attachment's resolve.
    assert_eq!(pass.depth_resolve, None);
    assert_eq!(pass.stencil_resolve, None);

    let mut refusal = device
        .new_command_queue()
        .command_buffer()
        .render_command_encoder()
        .unwrap();
    refusal.set_render_pipeline_state(&render).unwrap();
    refusal.set_vertex_buffer(0, &stream).unwrap();
    refusal
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    // A single-sample state is what the absent state means (`research/docs/23`
    // §3.3, v51), so the combined entry refuses it like its siblings do.
    assert_eq!(
        refusal.draw_indexed_primitives_with_multisample_depth_stencil(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            contract::MultisampleState {
                sample_count: contract::SampleCount::One,
            },
            depth,
            stencil,
            Some(depth_test),
            Some(stencil_test),
            None,
        ),
        Err(ContractError::SingleSampleMultisampleState.into())
    );
    // The two faces share one surface, so a pair that keeps one while dropping
    // the other is refused by name in both directions (`research/docs/23`
    // §3.3, v66/v68) — the recording cannot silently drop a face.
    let stored_depth = RenderDepthAttachment {
        store: Some(contract::DepthStoreOp::Store),
        identity: Some(contract::RenderDepthIdentity {
            allocation_id: scratch_view.allocation_id(),
            view_id: scratch_view.view_id(),
        }),
        ..depth
    };
    assert_eq!(
        refusal.draw_indexed_primitives_with_multisample_depth_stencil(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            state,
            stored_depth,
            stencil,
            Some(depth_test),
            Some(stencil_test),
            None,
        ),
        Err(ContractError::MultisampleCombinedSurfaceUnsupported {
            depth_stored: true,
            stencil_stored: false,
        }
        .into())
    );
    let stored_stencil = RenderStencilAttachment {
        store: Some(StoreOp::Store),
        identity: Some(contract::RenderStencilIdentity {
            allocation_id: scratch_view.allocation_id(),
            view_id: scratch_view.view_id(),
        }),
        ..stencil
    };
    assert_eq!(
        refusal.draw_indexed_primitives_with_multisample_depth_stencil(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            state,
            depth,
            stored_stencil,
            Some(depth_test),
            Some(stencil_test),
            None,
        ),
        Err(ContractError::MultisampleCombinedSurfaceUnsupported {
            depth_stored: false,
            stencil_stored: true,
        }
        .into())
    );
    // The v60 stored shape keeps both faces, but only through the two resolves
    // the v58/v60 entries carry; this entry states none, so it refuses the
    // shape with the contract's own first refusal rather than recording a pass
    // admission would reject (`research/docs/23` §3.3, v57/v60/v68).
    assert_eq!(
        refusal.draw_indexed_primitives_with_multisample_depth_stencil(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            state,
            stored_depth,
            stored_stencil,
            Some(depth_test),
            Some(stencil_test),
            None,
        ),
        Err(ContractError::MultisampleDepthStoreUnsupported.into())
    );
}

/// The stored sibling of the combined pair (`research/docs/23` §3.3,
/// v60/v70): both faces kept through their own resolve, recorded in one call.
/// The recording is the v60 shape the trace rails execute, so the pass it
/// becomes carries both landed identities and both filters — and every way the
/// pair can be half-stated is refused at recording time with the contract's own
/// error rather than silently dropped.
#[test]
fn a_stored_combined_depth_stencil_draw_records_both_resolves() {
    let provider = Arc::new(
        FakeProvider::new()
            .with_render()
            .with_vertex_input()
            .with_depth_resolve()
            .with_stencil_resolve(),
    );
    let device = Device::new(provider.clone());
    // Three read-only views: the colour attachment the draw reads, and the two
    // landings the stored pair writes its texels into. The declaring pass
    // declares all three, which is what puts them in the trace's resources.
    let declaring = pipeline(&device, "read:0,1,2");
    let render_metadata = render_metadata_multi(&provider, vec![AttachmentFormat::Rgba8Unorm]);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let target = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let view = target.view(0, 16).unwrap();
    let depth_surface = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let depth_view = depth_surface.view(0, 16).unwrap();
    let stencil_surface = device.new_buffer_with_bytes(vec![0xfc; 16]).unwrap();
    // The stencil landing is one byte per texel: four texels, four bytes
    // (`research/docs/23` §3.3, v49).
    let stencil_view = stencil_surface.view(0, 4).unwrap();
    let (_, stream) = buffer(&device, 0x11);
    let index = device
        .new_buffer_with_bytes(vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        .unwrap();
    let index_view = index.view(0, 16).unwrap();
    let attachment = RenderColorAttachment {
        view: &view,
        format: AttachmentFormat::Rgba8Unorm,
        load: RenderAttachmentLoad::Clear([0x11, 0x22, 0x33, 0x44]),
        store: StoreOp::Store,
    };
    let state = contract::MultisampleState {
        sample_count: contract::SampleCount::Four,
    };
    let stored_depth = RenderDepthAttachment {
        width: 2,
        height: 2,
        load: RenderDepthLoad::Clear(1.0),
        store: Some(contract::DepthStoreOp::Store),
        identity: Some(contract::RenderDepthIdentity {
            allocation_id: depth_view.allocation_id(),
            view_id: depth_view.view_id(),
        }),
    };
    let rail_owned_depth = RenderDepthAttachment {
        store: None,
        identity: None,
        ..stored_depth
    };
    let stored_stencil = RenderStencilAttachment {
        width: 2,
        height: 2,
        load: RenderStencilLoad::Clear(0),
        store: Some(StoreOp::Store),
        identity: Some(contract::RenderStencilIdentity {
            allocation_id: stencil_view.allocation_id(),
            view_id: stencil_view.view_id(),
        }),
    };
    let rail_owned_stencil = RenderStencilAttachment {
        store: None,
        identity: None,
        ..stored_stencil
    };
    let depth_test = RenderDepthTest {
        compare: contract::CompareFunction::Less,
        write: true,
    };
    let stencil_test = RenderStencilTest {
        compare: contract::StencilCompare::Equal,
        fail_op: contract::StencilOp::Keep,
        depth_fail_op: contract::StencilOp::Keep,
        pass_op: contract::StencilOp::IncrementWrap,
        read_mask: 0xff,
        write_mask: 0xff,
        reference: 0,
    };

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &view).unwrap();
        encoder.set_buffer(1, &depth_view).unwrap();
        encoder.set_buffer(2, &stencil_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();

    // One call records the whole v60 shape: the raster, the one shared surface
    // kept through both faces' landings, and the two filters that make the
    // stored texels observable.
    encoder
        .draw_indexed_primitives_with_multisample_depth_stencil_resolve(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            state,
            stored_depth,
            contract::DepthResolveFilter::Sample0,
            stored_stencil,
            contract::StencilResolveFilter::Sample0,
            Some(depth_test),
            Some(stencil_test),
            None,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();

    let trace = provider.traces.lock().unwrap().last().cloned().unwrap();
    let pass = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .expect("the recorded draw is a render pass");
    assert_eq!(pass.multisample, Some(state));
    let recorded_depth = pass.depth.as_ref().expect("the pass opens a depth surface");
    assert_eq!(recorded_depth.store, Some(contract::DepthStoreOp::Store));
    assert_eq!(recorded_depth.identity, stored_depth.identity);
    let recorded_stencil = pass
        .stencil
        .as_ref()
        .expect("the pass opens a stencil surface");
    assert_eq!(recorded_stencil.store, Some(StoreOp::Store));
    assert_eq!(recorded_stencil.identity, stored_stencil.identity);
    assert_eq!(
        pass.depth_resolve,
        Some(contract::MultisampleDepthResolve {
            filter: contract::DepthResolveFilter::Sample0,
        })
    );
    assert_eq!(
        pass.stencil_resolve,
        Some(contract::MultisampleStencilResolve {
            filter: contract::StencilResolveFilter::Sample0,
        })
    );

    let mut refusal = device
        .new_command_queue()
        .command_buffer()
        .render_command_encoder()
        .unwrap();
    refusal.set_render_pipeline_state(&render).unwrap();
    refusal.set_vertex_buffer(0, &stream).unwrap();
    refusal
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    // A single-sample state is what the absent state means (`research/docs/23`
    // §3.3, v51), so the resolving entry refuses it like its siblings do.
    assert_eq!(
        refusal.draw_indexed_primitives_with_multisample_depth_stencil_resolve(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            contract::MultisampleState {
                sample_count: contract::SampleCount::One,
            },
            stored_depth,
            contract::DepthResolveFilter::Sample0,
            stored_stencil,
            contract::StencilResolveFilter::Sample0,
            Some(depth_test),
            Some(stencil_test),
            None,
        ),
        Err(ContractError::SingleSampleMultisampleState.into())
    );
    // The pair keeps both faces or neither (`research/docs/23` §3.3, v66): a
    // lopsided store decision is refused by name in both directions, exactly as
    // the v68 entry refuses it.
    assert_eq!(
        refusal.draw_indexed_primitives_with_multisample_depth_stencil_resolve(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            state,
            stored_depth,
            contract::DepthResolveFilter::Sample0,
            rail_owned_stencil,
            contract::StencilResolveFilter::Sample0,
            Some(depth_test),
            Some(stencil_test),
            None,
        ),
        Err(ContractError::MultisampleCombinedSurfaceUnsupported {
            depth_stored: true,
            stencil_stored: false,
        }
        .into())
    );
    assert_eq!(
        refusal.draw_indexed_primitives_with_multisample_depth_stencil_resolve(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            state,
            rail_owned_depth,
            contract::DepthResolveFilter::Sample0,
            stored_stencil,
            contract::StencilResolveFilter::Sample0,
            Some(depth_test),
            Some(stencil_test),
            None,
        ),
        Err(ContractError::MultisampleCombinedSurfaceUnsupported {
            depth_stored: false,
            stencil_stored: true,
        }
        .into())
    );
    // The v66 pair keeps neither face — the v68 entry's shape — so the
    // resolving entry refuses it with the contract's own "resolve beside a
    // dropped surface" error rather than recording two filters admission would
    // refuse (`research/docs/23` §3.3, v57/v60/v70).
    assert_eq!(
        refusal.draw_indexed_primitives_with_multisample_depth_stencil_resolve(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            state,
            rail_owned_depth,
            contract::DepthResolveFilter::Sample0,
            rail_owned_stencil,
            contract::StencilResolveFilter::Sample0,
            Some(depth_test),
            Some(stencil_test),
            None,
        ),
        Err(ContractError::DepthResolveWithoutStoredDepth { store: None }.into())
    );
    // A kept face names where its texels land (`research/docs/23` §3.3,
    // v43/v49): a stored face without an identity is refused, one face at a
    // time, with the contract's own error.
    assert_eq!(
        refusal.draw_indexed_primitives_with_multisample_depth_stencil_resolve(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            state,
            RenderDepthAttachment {
                identity: None,
                ..stored_depth
            },
            contract::DepthResolveFilter::Sample0,
            stored_stencil,
            contract::StencilResolveFilter::Sample0,
            Some(depth_test),
            Some(stencil_test),
            None,
        ),
        Err(ContractError::DepthStoreIdentityMismatch {
            store: Some(contract::DepthStoreOp::Store.code()),
            identity: false,
        }
        .into())
    );
    assert_eq!(
        refusal.draw_indexed_primitives_with_multisample_depth_stencil_resolve(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            state,
            stored_depth,
            contract::DepthResolveFilter::Sample0,
            RenderStencilAttachment {
                identity: None,
                ..stored_stencil
            },
            contract::StencilResolveFilter::Sample0,
            Some(depth_test),
            Some(stencil_test),
            None,
        ),
        Err(ContractError::StencilStoreIdentityMismatch {
            store: Some(StoreOp::Store),
            identity: false,
        }
        .into())
    );
}

#[test]
fn a_multisample_draw_records_the_raster_and_needs_an_index_buffer() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_multi(&provider, vec![AttachmentFormat::Rgba8Unorm]);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let target = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let view = target.view(0, 16).unwrap();
    let scratch = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let scratch_view = scratch.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);
    let index = device
        .new_buffer_with_bytes(vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        .unwrap();
    let index_view = index.view(0, 16).unwrap();
    let attachment = RenderColorAttachment {
        view: &view,
        format: AttachmentFormat::Rgba8Unorm,
        load: RenderAttachmentLoad::Clear([0x22, 0x44, 0x66, 0x89]),
        store: StoreOp::Store,
    };
    let state = contract::MultisampleState {
        sample_count: contract::SampleCount::Four,
    };

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &view).unwrap();
        encoder.set_buffer(1, &scratch_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();

    // The multisample entry is an indexed one like the other state-carrying
    // entries, so an encoder without an index buffer is refused before a pass
    // is recorded (`research/docs/23` §3.3, v51/v52).
    assert_eq!(
        encoder.draw_indexed_primitives_with_multisample(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            state,
            None,
        ),
        Err(Error::MissingIndexBuffer)
    );

    // Stating the single-sample raster is refused at recording time: the
    // absent field is what that shape means, so the entry cannot record a
    // second encoding of it (`research/docs/23` §3.3, v51).
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    assert_eq!(
        encoder.draw_indexed_primitives_with_multisample(
            std::slice::from_ref(&attachment),
            2,
            2,
            6,
            1,
            contract::MultisampleState {
                sample_count: contract::SampleCount::One,
            },
            None,
        ),
        Err(ContractError::SingleSampleMultisampleState.into())
    );

    encoder
        .draw_indexed_primitives_with_multisample(&[attachment], 2, 2, 6, 1, state, None)
        .unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();

    let trace = provider.traces.lock().unwrap().last().cloned().unwrap();
    let pass = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .expect("the recorded draw is a render pass");
    assert_eq!(pass.multisample, Some(state));
    assert!(pass.depth.is_none() && pass.stencil.is_none());
}

#[test]
fn a_stencil_draw_records_the_surface_and_state_and_needs_an_index_buffer() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_multi(&provider, vec![AttachmentFormat::Rgba8Unorm]);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let target = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let view = target.view(0, 16).unwrap();
    let scratch = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let scratch_view = scratch.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);
    let index = device
        .new_buffer_with_bytes(vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        .unwrap();
    let index_view = index.view(0, 16).unwrap();
    let attachment = RenderColorAttachment {
        view: &view,
        format: AttachmentFormat::Rgba8Unorm,
        load: RenderAttachmentLoad::Clear([0xfe; 4]),
        store: StoreOp::Store,
    };

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &view).unwrap();
        encoder.set_buffer(1, &scratch_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();

    // The stencil entry is an indexed one like the other state-carrying
    // entries, so an encoder without an index buffer is refused before a pass
    // is recorded (`research/docs/23` §3.3, v47/v48).
    let stencil = RenderStencilAttachment {
        width: 2,
        height: 2,
        load: RenderStencilLoad::Clear(0),
        store: None,
        identity: None,
    };
    let test = RenderStencilTest {
        compare: contract::StencilCompare::Equal,
        fail_op: contract::StencilOp::Keep,
        depth_fail_op: contract::StencilOp::Keep,
        pass_op: contract::StencilOp::IncrementWrap,
        read_mask: 0xff,
        write_mask: 0xff,
        reference: 0,
    };
    assert_eq!(
        encoder.draw_indexed_primitives_with_stencil(
            &[attachment],
            2,
            2,
            6,
            1,
            stencil,
            Some(test),
            None,
        ),
        Err(Error::MissingIndexBuffer)
    );

    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    encoder
        .draw_indexed_primitives_with_stencil(&[attachment], 2, 2, 6, 1, stencil, Some(test), None)
        .unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();

    let trace = provider.traces.lock().unwrap().last().cloned().unwrap();
    let pass = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .expect("the recorded draw is a render pass");
    let recorded = pass
        .stencil
        .as_ref()
        .expect("the pass opens a stencil surface");
    assert_eq!((recorded.width, recorded.height), (2, 2));
    assert_eq!(recorded.format, contract::StencilFormat::Stencil8);
    assert_eq!(recorded.load, contract::StencilLoadOp::Clear(0));
    let recorded_test = pass.stencil_test.expect("the pass tests stencils");
    assert_eq!(recorded_test.compare, contract::StencilCompare::Equal);
    assert_eq!(recorded_test.pass_op, contract::StencilOp::IncrementWrap);
    assert_eq!(recorded_test.read_mask, 0xff);
    assert_eq!(recorded_test.reference, 0);

    // A stencil surface of another extent is refused by the contract's own
    // rule when the pass is recorded.
    let mut encoder = device
        .new_command_queue()
        .command_buffer()
        .render_command_encoder()
        .unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();
    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    assert_eq!(
        encoder.draw_indexed_primitives_with_stencil(
            &[attachment],
            2,
            2,
            6,
            1,
            RenderStencilAttachment {
                width: 2,
                height: 4,
                load: RenderStencilLoad::Clear(0),
                store: None,
                identity: None,
            },
            Some(test),
            None,
        ),
        Err(ContractError::StencilExtentMismatch {
            raster: [2, 2],
            stencil: [2, 4],
        }
        .into())
    );
}

#[test]
fn a_culling_draw_records_the_state_and_needs_an_index_buffer() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_multi(&provider, vec![AttachmentFormat::Rgba8Unorm]);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let target = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let view = target.view(0, 16).unwrap();
    let scratch = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let scratch_view = scratch.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);
    let index = device
        .new_buffer_with_bytes(vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        .unwrap();
    let index_view = index.view(0, 16).unwrap();

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &view).unwrap();
        encoder.set_buffer(1, &scratch_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();

    // The object API's culling entry is an indexed one, so an encoder without
    // an index buffer is refused before a pass is recorded — the same rule the
    // other indexed entries state (`research/docs/23` §3.3, v39/v41).
    assert_eq!(
        encoder.draw_indexed_primitives_with_cull(
            &[RenderColorAttachment {
                view: &view,
                format: AttachmentFormat::Rgba8Unorm,
                load: RenderAttachmentLoad::Clear([0xfe; 4]),
                store: StoreOp::Store,
            }],
            2,
            2,
            6,
            1,
            contract::RenderPassCull {
                mode: contract::CullMode::Back,
                winding: contract::Winding::CounterClockwise,
            },
            None,
        ),
        Err(Error::MissingIndexBuffer)
    );

    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    encoder
        .draw_indexed_primitives_with_cull(
            &[RenderColorAttachment {
                view: &view,
                format: AttachmentFormat::Rgba8Unorm,
                load: RenderAttachmentLoad::Clear([0xfe; 4]),
                store: StoreOp::Store,
            }],
            2,
            2,
            6,
            1,
            contract::RenderPassCull {
                mode: contract::CullMode::Back,
                winding: contract::Winding::CounterClockwise,
            },
            None,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();

    let trace = provider.traces.lock().unwrap().last().cloned().unwrap();
    let pass = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .expect("the recorded draw is a render pass");
    let cull = pass.cull.expect("the pass carries the culling state");
    assert_eq!(cull.mode, contract::CullMode::Back);
    assert_eq!(cull.winding, contract::Winding::CounterClockwise);
    assert!(pass.blend.is_none());
    assert!(pass.depth.is_none());
}

#[test]
fn a_blending_draw_records_the_state_and_needs_an_index_buffer() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let render_metadata = render_metadata_multi(&provider, vec![AttachmentFormat::Rgba8Unorm]);
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let target = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let view = target.view(0, 16).unwrap();
    let scratch = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let scratch_view = scratch.view(0, 16).unwrap();
    let (_, stream) = buffer(&device, 0x11);
    let index = device
        .new_buffer_with_bytes(vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        .unwrap();
    let index_view = index.view(0, 16).unwrap();
    let blend = [contract::BlendAttachment {
        enabled: true,
        source_rgb: contract::BlendFactor::SourceAlpha,
        destination_rgb: contract::BlendFactor::OneMinusSourceAlpha,
        source_alpha: contract::BlendFactor::SourceAlpha,
        destination_alpha: contract::BlendFactor::OneMinusSourceAlpha,
        operation: contract::BlendOperation::Add,
        alpha_operation: contract::BlendOperation::Add,
        write_mask: contract::ColorWriteMask::ALL,
    }];

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &view).unwrap();
        encoder.set_buffer(1, &scratch_view).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command.render_command_encoder().unwrap();
    encoder.set_render_pipeline_state(&render).unwrap();
    encoder.set_vertex_buffer(0, &stream).unwrap();

    // The blending entry is an indexed one, so an encoder without an index
    // buffer is refused before a pass is recorded (`research/docs/23` §3.3,
    // v40/v42).
    assert_eq!(
        encoder.draw_indexed_primitives_with_blend(
            &[RenderColorAttachment {
                view: &view,
                format: AttachmentFormat::Rgba8Unorm,
                load: RenderAttachmentLoad::Clear([0xfe; 4]),
                store: StoreOp::Store,
            }],
            2,
            2,
            6,
            1,
            &blend,
            None,
        ),
        Err(Error::MissingIndexBuffer)
    );

    encoder
        .set_index_buffer(&index_view, IndexFormat::Uint16)
        .unwrap();
    encoder
        .draw_indexed_primitives_with_blend(
            &[RenderColorAttachment {
                view: &view,
                format: AttachmentFormat::Rgba8Unorm,
                load: RenderAttachmentLoad::Clear([0xfe; 4]),
                store: StoreOp::Store,
            }],
            2,
            2,
            6,
            1,
            &blend,
            None,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command.commit().unwrap();

    let trace = provider.traces.lock().unwrap().last().cloned().unwrap();
    let pass = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .expect("the recorded draw is a render pass");
    let recorded = pass
        .blend
        .as_ref()
        .expect("the pass carries the blend state");
    assert_eq!(recorded.attachments, blend.to_vec());
    assert!(pass.cull.is_none());
    assert!(pass.depth.is_none());
}

#[test]
fn an_indexed_draw_records_two_attachments_through_the_shared_path() {
    let provider = Arc::new(FakeProvider::new().with_render().with_vertex_input());
    let device = Device::new(provider.clone());
    let declaring = pipeline(&device, "read:0,1");
    let mut render_metadata = render_metadata_multi(
        &provider,
        vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm],
    );
    // The index-only shape binds no stream, so the matching pipeline has no
    // vertex layout either.
    if let Some(render) = &mut render_metadata.render {
        render.vertex_layout = VertexLayout::None;
    }
    provider
        .pipelines
        .lock()
        .unwrap()
        .insert(render_metadata.pipeline_id);
    let render = device.render_pipeline(&render_metadata).unwrap();

    let first = device.new_buffer_with_bytes(vec![0xfe; 16]).unwrap();
    let first_view = first.view(0, 16).unwrap();
    let second = device.new_buffer_with_bytes(vec![0xfd; 16]).unwrap();
    let second_view = second.view(0, 16).unwrap();
    let index = device.new_buffer_with_bytes(vec![0; 12]).unwrap();
    let index_view = index.view(0, 12).unwrap();

    let command = device.new_command_queue().command_buffer();
    {
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&declaring).unwrap();
        encoder.set_buffer(0, &first_view).unwrap();
        encoder.set_buffer(1, &second_view).unwrap();
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
            .draw_indexed_primitives_with_attachments(
                &[
                    RenderColorAttachment {
                        view: &first_view,
                        format: AttachmentFormat::Rgba8Unorm,
                        load: RenderAttachmentLoad::Clear([0xa1; 4]),
                        store: StoreOp::Store,
                    },
                    RenderColorAttachment {
                        view: &second_view,
                        format: AttachmentFormat::Rgba8Unorm,
                        load: RenderAttachmentLoad::Clear([0xb2; 4]),
                        store: StoreOp::Store,
                    },
                ],
                2,
                2,
                FULL_SCREEN_TRIANGLE_VERTICES,
                None,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    command.commit().unwrap();
    command.wait_until_completed().unwrap();
    assert_eq!(first.read().unwrap(), [0x40, 0x80, 0xc0, 0xff].repeat(4));
    assert_eq!(second.read().unwrap(), [0x40, 0x80, 0xc0, 0xff].repeat(4));

    let traces = provider.traces.lock().unwrap();
    let pass = traces[0].render_passes().next().expect("render pass");
    assert_eq!(pass.color_attachments.len(), 2);
    assert_eq!(pass.color_attachments[0].view_id, first_view.view_id());
    assert_eq!(pass.color_attachments[1].view_id, second_view.view_id());
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
fn heap_construction_refuses_zero_size_and_defers_aliasing_to_admission() {
    let device = Device::new(Arc::new(FakeProvider::new()));
    assert!(matches!(
        device.new_heap(0, StorageMode::OwnedBytes, false),
        Err(Error::Contract(ContractError::ZeroLength(_)))
    ));
    // Declaring aliasing is well formed; the capability decision belongs to
    // admission, so construction succeeds and a provider whose snapshot keeps
    // `supports_heap_aliasing = false` refuses the commit.
    device
        .new_heap(64, StorageMode::OwnedBytes, true)
        .expect("an aliasing heap declares before admission");
}

#[test]
fn an_aliasing_heap_is_refused_at_commit_until_the_snapshot_admits_it() {
    let provider = Arc::new(FakeProvider::new().with_heap());
    let device = Device::new(provider.clone());
    let copy = pipeline(&device, "copy");
    let (first, first_view) = buffer(&device, 1);
    let (second, second_view) = buffer(&device, 2);

    let heap = device
        .new_heap(64, StorageMode::OwnedBytes, true)
        .expect("aliasing declares at construction");
    heap.place(&first, 0).expect("first placement");
    heap.place(&second, 0)
        .expect("overlapping placement declares aliasing");

    let command = device.new_command_queue().command_buffer();
    command.set_heap(&heap).unwrap();
    let mut encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&copy).unwrap();
    encoder.set_buffer(4, &first_view).unwrap();
    encoder.set_buffer(9, &second_view).unwrap();
    dispatch(&mut encoder).unwrap();
    encoder.end_encoding().unwrap();
    assert!(matches!(
        command.commit(),
        Err(Error::Provider(error)) if error.slug == "heap_alias_unsupported"
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
        stage_buffers: Vec::new(),
        vertex_entry: "vertex_main".into(),
        fragment_entry: "fragment_main".into(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: layout,
        textures: Vec::new(),
    });
    metadata
}

/// The render contract for a multi-attachment draw: one compiled format per
/// location and one bound stream at binding 0, so a direct draw through vertex
/// buffers has a matching pipeline and every attachment lands at its own
/// location.
fn render_metadata_multi(
    provider: &FakeProvider,
    formats: Vec<AttachmentFormat>,
) -> CompiledComputePipeline {
    let mut metadata = render_metadata(provider);
    metadata.render = Some(RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: "vertex_main".into(),
        fragment_entry: "fragment_main".into(),
        color_formats: formats,
        vertex_layout: VertexLayout::Buffers(vec![stream_layout(0)]),
        textures: Vec::new(),
    });
    metadata
}

/// One vertex stream: eight bytes per vertex, one `float32x2` at offset 0.
fn stream_layout(location: u32) -> VertexBufferLayout {
    VertexBufferLayout {
        stride: 8,
        step: crate::provider::VertexStep::PerVertex,
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
                fragment_textures: 0,
                fragment_samplers: 0,
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

#[test]
fn a_compute_texture_declaration_without_the_capability_is_refused_by_name() {
    // C1's capability gate reached through the object rail: the declaration is
    // the module's own, the binding is recorded, and admission still refuses
    // the pass before the provider is asked for anything.
    let (provider, device) = setup();
    let pipeline = pipeline(&device, "sampled_texture");
    let texture = device
        .new_texture_with_bytes(TextureFormat::R32Uint, 2, 2, vec![7_u8; 16])
        .unwrap();
    let command = device.new_command_queue().command_buffer();
    let mut encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    encoder.set_texture(0, &texture).unwrap();
    dispatch(&mut encoder).unwrap();
    encoder.end_encoding().unwrap();
    let refusal = command
        .commit()
        .expect_err("a provider without the compute texture bit refuses the pass");
    let Error::Provider(refusal) = refusal else {
        panic!("the capability gate reaches the caller as a provider refusal");
    };
    eprintln!("object rail compute texture capability refusal: {refusal:?}");
    assert_eq!(refusal.slug, "compute_texture_input_unsupported");
    assert_eq!(refusal.fields.get("pass"), Some(&FieldValue::Unsigned(0)));
    assert_eq!(
        refusal.fields.get("textures"),
        Some(&FieldValue::Unsigned(1))
    );
    assert_eq!(
        provider.traces.lock().unwrap().len(),
        0,
        "the refusal precedes submission"
    );
}

#[test]
fn a_storage_texture_lands_into_its_own_handle_on_both_object_rails() {
    for (mode, deferred) in [(GOOD, false), (ASYNC_GOOD, true)] {
        let provider = Arc::new(FakeProvider::new().with_compute_textures());
        provider.mode.store(mode, Ordering::SeqCst);
        let device = Device::new(provider.clone());
        let pipeline = pipeline(&device, "storage_texture");
        assert_eq!(
            pipeline.metadata().contract.texture_bindings,
            vec![TextureBindingContract::storage(0, TextureFormat::R32Float)]
        );
        let initial = (0..16_u8).collect::<Vec<_>>();
        let texture = device
            .new_storage_texture_with_bytes(TextureFormat::R32Float, 2, 2, initial.clone())
            .unwrap();
        assert_eq!(texture.access(), TextureAccess::Storage);
        assert_eq!(texture.read().unwrap(), initial);
        let expected = initial
            .iter()
            .map(|byte| byte.wrapping_add(0x10))
            .collect::<Vec<_>>();
        let command = device.new_command_queue().command_buffer();
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        encoder.set_texture(0, &texture).unwrap();
        dispatch(&mut encoder).unwrap();
        encoder.end_encoding().unwrap();
        command.commit().unwrap();
        if deferred {
            assert_eq!(command.status().unwrap(), CommandBufferStatus::Committed);
            assert!(matches!(
                command.submission().unwrap().completion,
                CompletionDisposition::Submitted { .. }
            ));
            assert!(
                command.submission().unwrap().writebacks.is_empty(),
                "a deferred commit publishes no landing yet"
            );
            // The handle's bytes are held in flight: a reader waits for the
            // write's reservation exactly as a buffer reader waits for its
            // range, and observes the landed bytes once the fence retires.
            let (sender, receiver) = std::sync::mpsc::channel();
            let reader = texture.clone();
            let handle = std::thread::spawn(move || sender.send(reader.read()).unwrap());
            assert!(
                receiver.recv_timeout(Duration::from_millis(50)).is_err(),
                "a texture read waits out the in-flight write"
            );
            command.wait_until_completed().unwrap();
            assert_eq!(
                receiver
                    .recv_timeout(Duration::from_secs(2))
                    .unwrap()
                    .unwrap(),
                expected
            );
            handle.join().unwrap();
        } else {
            assert_eq!(command.status().unwrap(), CommandBufferStatus::Completed);
            assert_eq!(texture.read().unwrap(), expected);
        }
        let submission = command.submission().unwrap();
        let writeback = submission
            .writebacks
            .iter()
            .find(|write| write.view_id == texture.view_id())
            .expect("the storage image's landing is keyed by its own view");
        assert_eq!(writeback.allocation_id, texture.allocation_id());
        assert_eq!(writeback.offset, 0);
        assert_eq!(writeback.bytes, expected);
        eprintln!(
            "object rail storage landing deferred={deferred}: {:?}",
            texture.read().unwrap()
        );
        drop(command);
        assert_eq!(provider.released_completions.lock().unwrap().len(), 1);
    }
}
