//! Metal handles stay behind one lock. No guest pointers or caller-owned
//! memory are passed to Metal; submission copies admitted view contents or
//! staged lease windows.

use crate::{
    bounded_contract, classify_command_buffer_error, device_lost_refusal, heap, icb,
    lifecycle::NativeLifecycle, refusal, render, unknown_completion, CommandBufferFailure,
};
use block::ConcreteBlock;
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    Buffer, CommandBuffer, CommandBufferRef, CommandQueue, ComputeCommandEncoderRef,
    ComputePipelineState, Device, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
    MTLCommandBufferStatus, MTLGPUFamily, MTLHazardTrackingMode, MTLIndirectCommandType, MTLOrigin,
    MTLPixelFormat, MTLRegion, MTLResourceOptions, MTLSize, MTLStorageMode, MTLTextureType,
    MTLTextureUsage, NSRange, NSUInteger, Texture, TextureDescriptor,
};
use metal_api_core::completion::wire::CompletionOutbox;
use metal_api_core::completion::{AbandonmentBudget, CompletionRecord, ObservationDeadline};
use metal_api_core::provider::*;
use objc::{msg_send, runtime::Object, sel, sel_impl};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CStr;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

const GPU_DEADLINE: Duration = Duration::from_secs(20);

#[derive(Clone)]
struct RegisteredPipeline {
    metadata: CompiledComputePipeline,
    pipeline: ComputePipelineState,
}

/// One host-registered render pipeline: the trace-table entry this context
/// minted for it and the reviewed contract behind that identity.
///
/// The shape mirrors [`RegisteredPipeline`] on purpose. A trace's pipeline table
/// is the only place a pass says which pipeline it runs, so both rails check the
/// caller-supplied entry against what the owner registered before anything
/// executes. The ids share one counter and one namespace: a compute pass naming
/// a render registration is refused as an unknown pipeline, and a render pass
/// naming a compute registration is refused as an unknown render pipeline.
struct RegisteredRenderPipeline {
    metadata: CompiledComputePipeline,
    contract: RenderPipelineContract,
}

/// One render pipeline a host asks a native context to own.
///
/// The rail compiles one reviewed MSL module (`crate::render::REVIEWED_SOURCE`),
/// so the request carries no source: `contract` names the two entries and the
/// attachment format that module was reviewed for, and `logical_digest` is the
/// caller-issued fixture identity [`PipelineProvider::compile`] also takes.
/// Registering a contract the reviewed module does not carry is refused with
/// `native_render_source_not_reviewed`, exactly as an unreviewed MSL fixture is.
pub struct NativeRenderPipelineRequest {
    pub contract: RenderPipelineContract,
    pub logical_digest: SemanticDigest,
}

struct State {
    device: Device,
    queue: CommandQueue,
    pipelines: BTreeMap<PipelineId, RegisteredPipeline>,
    next_pipeline: u64,
    next_submission: u64,
    /// Present targets by (allocation, view). The texture is created on the
    /// first present and reused across submissions until the allocation's
    /// lease is released (`research/docs/24` §6 Step 7).
    present_targets: BTreeMap<(AllocationId, ViewId), PresentTargetRef>,
}

/// One present target's device-side texture. Held for the life of the
/// allocation's lease, not one submission.
struct PresentTargetRef {
    texture: Texture,
}

#[derive(Clone)]
struct CompletionSlot {
    record: Arc<CompletionRecord>,
    deadline: ObservationDeadline,
    owned_bytes: u64,
    /// Render-bearing deferred submissions finish their render rail inside
    /// `submit` and park the merged writebacks here. `wait`/`readback` land
    /// them into the record, so `cancel` can still abandon the observation
    /// before either lands it. `None` for synchronous and compute-only
    /// submissions.
    deferred_writebacks: Option<Vec<BufferWriteback>>,
}

fn trace_owned_bytes(trace: &ComputeTrace) -> u64 {
    trace.compute_passes().fold(0_u64, |total, pass| {
        pass.buffers
            .iter()
            .fold(total, |total, view| total.saturating_add(view.length))
    })
}

/// Native provider with at most eight serial passes over one buffer pool,
/// allowing pipeline changes and binding permutations for exact fixtures.
///
/// A submission has a configurable observation deadline (20 seconds by
/// default). A deadline expiry records `SubmittedUnknown` but leaves the
/// context usable: the completion handler retains the device, queue, pipeline
/// and buffer references until Metal reports a terminal status, then releases
/// them. A completion handler that observes `Error` permanently disables new
/// work in this context. `wait` reads the recorded terminal observation;
/// releasing that record does not retire GPU resources. By default `submit`
/// waits for GPU completion and readback. Calling
/// [`NativeMetalProvider::with_async_execution`] with `true` instead returns
/// `Submitted` immediately and fills the completion record from an
/// `MTLCommandBuffer` completion handler.
/// Device-buffer copy-in and copy-out operations. One of each per touched
/// allocation, not per view: owned views of one allocation share one MTLBuffer
/// (`research/docs/15` §3.3).
#[derive(Default)]
struct CopyCounters {
    uploads: AtomicUsize,
    readbacks: AtomicUsize,
}

/// Cumulative acquire/present completions of the present rail
/// (`research/docs/24` §1.4, §5.3). The accessor mirrors
/// [`NativeMetalProvider::buffer_copy_counts`] so the capture harness can
/// assert "one acquire, one present" without a second observation channel.
#[derive(Default)]
struct PresentCounters {
    acquires: AtomicUsize,
    presents: AtomicUsize,
}

pub struct NativeMetalProvider {
    epoch: DeviceEpoch,
    name: String,
    capabilities: ProviderCapabilities,
    /// The reviewed 2/4/8 sample counts the device admits, as the
    /// contract-code bitmask the device probe built (`research/docs/23` §3.3,
    /// v61). The capture runner reads the device-gated sample-count cases
    /// against it; the snapshot's ceiling alone cannot say which of the
    /// counts a device lacks.
    render_sample_counts: u32,
    state: Mutex<State>,
    /// Render registrations, keyed by pipeline id.
    ///
    /// A separate mutex from `state` because the trace path resolves a
    /// registration while a submission holds the device lock. The lock order is
    /// `state` then `render_pipelines` everywhere — registration mints its
    /// pipeline id from `State::next_pipeline` before touching this map — so the
    /// two locks cannot deadlock.
    render_pipelines: Mutex<BTreeMap<PipelineId, RegisteredRenderPipeline>>,
    completions: Mutex<BTreeMap<SubmissionId, CompletionSlot>>,
    /// Admission, health and the abandonment counters of this instance.
    ///
    /// One `metal_api_core` lifecycle is the only terminal-state authority, so
    /// the provider cannot report one health while refusing on another. It is
    /// shared with the Metal completion handlers, which run on a driver thread
    /// after `submit` has returned and therefore cannot borrow the provider.
    lifecycle: Arc<NativeLifecycle>,
    async_execution: bool,
    observation_deadline: Duration,
    completion_outbox: Option<Arc<CompletionOutbox>>,
    staging: LeaseRegistry,
    /// Lease id → allocation id, populated on import and consumed on release,
    /// so releasing a lease can drop the present targets reserved for its
    /// allocation (`research/docs/24` §6 Step 7).
    lease_allocations: Mutex<BTreeMap<LeaseId, AllocationId>>,
    borrowed: Arc<BorrowedLeaseRegistry>,
    counters: Arc<CopyCounters>,
    present_counters: Arc<PresentCounters>,
    /// Heap placements the provider most recently executed successfully. Every
    /// successful heap-bearing submission replaces the vector instead of
    /// appending, so it stays bounded to one submission (`research/docs/25`
    /// §6 Step 3).
    heap_observations: Arc<Mutex<Vec<heap::HeapPlacementObservation>>>,
    /// Indirect replays the provider most recently executed. Like the heap
    /// observation, a successful ICB-bearing submission replaces the vector
    /// instead of appending (`research/docs/25` §6 Step 7b).
    icb_observations: Arc<Mutex<Vec<icb::IcbReplayObservation>>>,
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
            // The render bits come from the rail itself (`crate::render`) so the
            // snapshot and the rail cannot disagree; the unit tests assert that
            // agreement against core admission on a host without Metal.
            let render_bits = render::capability_bits();
            // The vertex-input bits come from the same rail value, for the same
            // reason: `render::plan_vertex_input` owns the stride and index
            // footprints, and this snapshot publishes exactly the formats and
            // the stream count that rail translates.
            let vertex_bits = render::vertex_input_capability_bits();
            let instancing_bits = render::instancing_capability_bits();
            // The multisample bits come from the device probe
            // (`render::device_multisample_capability_bits`,
            // `research/docs/23` §3.3, v51/v61): the snapshot publishes the
            // largest of the reviewed 2x/4x/8x rasters the device's
            // `supportsTextureSampleCount:` answer admits, and the plan holds
            // the raster to the counts the encoder builds.
            let multisample_bits = render::device_multisample_capability_bits(&device);
            // The depth-resolve bits come from the device probe
            // (`render::device_depth_resolve_capability_bits`,
            // `research/docs/23` §3.3, v57c): the snapshot publishes the
            // Sample0|Min|Max bits when the device answers the Apple-family
            // question — the v57e `--depth-resolve-selftest` run measured the
            // Apple Paravirtual device executing all three filters
            // (`f4d70e4`, CI run `35112569688`).
            let depth_resolve_bits = render::device_depth_resolve_capability_bits(&device);
            // The stencil-resolve bits come from the same device probe
            // (`render::device_stencil_resolve_capability_bits`,
            // `research/docs/23` §3.3, v60): the v59
            // `--stencil-resolve-selftest` run measured the Apple Paravirtual
            // device executing both filters (`2b877b8`, CI run
            // `35120171655`), so the mask carries Sample0|DepthResolvedSample.
            let stencil_resolve_bits = render::device_stencil_resolve_capability_bits(&device);
            // The heap bits stay closed until `--heap-selftest` passes on an
            // Apple GPU; they come from one spelling (`crate::heap`) so the
            // snapshot and the flip condition cannot drift.
            let heap_bits = heap::heap_capability_bits();
            // The ICB bits stay closed until `--icb-selftest` passes on an
            // Apple GPU; they come from one spelling (`crate::icb`) so the
            // snapshot and the flip condition cannot drift.
            let icb_bits = icb::icb_capability_bits();
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
                // Same ranged-aliasing argument as the Vulkan provider: each
                // admitted view is copied into its own MTLBuffer that starts
                // at the view, so disjoint views of one allocation never share
                // device bytes, and admission refuses overlapping ranges.
                alias_mode: AliasMode::DistinctViews,
                storage_modes: vec![
                    StorageMode::OwnedBytes,
                    StorageMode::StagedLease,
                    StorageMode::BorrowedNoCopy,
                ],
                host_readback: true,
                submit_only: false,
                // Render-bearing device snapshot, flipped in Step 7. The flip
                // condition was the single-device check in
                // `conformance/RENDER-CAPTURE.md` §5, and the observation behind
                // it is CI run `34774478149` (`native-oracle-build`, commit
                // `fb4f8da`): on an Apple Paravirtual device
                // `native-oracle --render-selftest` ran the same reviewed module
                // and the same `runRenderCase` a suite would, read the 2x2
                // attachment back as `4080c0ff` four times, and printed
                // `render_selftest: PASS`. A green job whose log said `SKIP`
                // would not have been that evidence.
                //
                // The bits stay the rail's own limits, and the trace path below
                // executes `render::plan_trace` + `render::encode_offscreen_render`
                // (`research/docs/23` §4.2, §6 Steps 6-7).
                supports_render_passes: render_bits.supports_render_passes,
                max_color_attachments: render_bits.max_color_attachments,
                max_attachment_dimension: render_bits.max_attachment_dimension,
                supported_color_formats: render_bits.supported_color_formats,
                // Vertex input is executed by this rail as of the vertex-input
                // increment: a pass that binds streams is translated into an
                // `MTLVertexDescriptor` plus `setVertexBuffer` /
                // `drawIndexedPrimitives` by `render.rs`, whose own plan proves
                // every stride and index footprint on the host beforehand
                // (`research/docs/23` §3.3). The flip condition is the
                // `--vertex-selftest` observation recorded on
                // `render::vertex_input_capability_bits`; before the flip these
                // three bits were at their defaults and core admission refused
                // such a trace instead of executing it with positions the trace
                // did not ask for.
                max_vertex_buffers: vertex_bits.max_vertex_buffers,
                supported_vertex_formats: vertex_bits.supported_vertex_formats,
                supported_index_formats: vertex_bits.supported_index_formats,
                // Instancing is executed by this rail as of v31: the plan
                // carries each binding's step function and the draw carries the
                // pass's instance count, both proved on the host before a
                // device object exists (`research/docs/23` §3.3). The flip
                // condition is the reviewed `instanced_pair_4x4` case on the
                // Apple rail, recorded on
                // `render::instancing_capability_bits`.
                supports_render_instancing: instancing_bits.supports_render_instancing,
                max_render_instances: instancing_bits.max_render_instances,
                // Multisampling is executed by this rail as of v51: the plan
                // carries the pass-wide raster, the encoder creates one
                // multisampled texture per colour location and resolves it
                // into the attachment's own texture, both proved on the host
                // before a device object exists (`research/docs/23` §3.3).
                // The flip condition is the reviewed `msaa_edge_4x4` case on
                // the Apple rail, and the v61 increment widens the declared
                // ceiling to the device's own 2x/4x/8x answer, recorded on
                // `render::device_multisample_capability_bits`.
                supports_render_multisample: multisample_bits.supports_render_multisample,
                max_render_sample_count: multisample_bits.max_render_sample_count,
                // The depth resolve is executed by this rail as of v57c: the
                // encoder opens the four-sample depth surface with
                // `storeAction = .multisampleResolve` and lands the Sample0
                // reduction in the v43 shared-storage readback texture. The
                // bits come from the device probe
                // (`render::device_depth_resolve_capability_bits`); the v57e
                // self-test measured Min and Max on the Apple Paravirtual
                // device, so the mask also carries those two bits from v57f
                // on (`research/docs/23` §3.3, v57c/v57e).
                supports_render_depth_resolve: depth_resolve_bits.supports_render_depth_resolve,
                depth_resolve_modes: depth_resolve_bits.depth_resolve_modes,
                // The stencil resolve is executed by this rail as of v60: the
                // encoder opens the combined depth-stencil surface with
                // `storeAction = .multisampleResolve` and lands the two
                // reductions in their single-sample shared textures. The bits
                // come from the device probe
                // (`render::device_stencil_resolve_capability_bits`), which
                // declares both filters on the v59 self-test evidence
                // (`research/docs/23` §3.3, v60).
                supports_render_stencil_resolve: stencil_resolve_bits
                    .supports_render_stencil_resolve,
                stencil_resolve_modes: stencil_resolve_bits.stencil_resolve_modes,
                // The render-sampler bits stay closed until the rail executes
                // the shape (`research/docs/23` §3.3, v70): the contract and
                // the wire ship them fail-closed, so a texture-bearing pass is
                // refused during admission instead of being executed with a
                // cleared sampling result the trace did not ask for.
                supports_render_texture_sampling: render_bits.supports_render_texture_sampling,
                max_render_textures: render_bits.max_render_textures,
                supported_render_texture_formats: render_bits
                    .supported_render_texture_formats
                    .clone(),
                // The present bits come from the same rail value as the render
                // bits, so this snapshot cannot claim a present action the rail
                // does not run (`research/docs/24` §4.2, §6 Step 3).
                supports_presentation: render_bits.supports_presentation,
                max_present_targets: render_bits.max_present_targets,
                supported_present_modes: render_bits.supported_present_modes,
                max_present_image_count: render_bits.max_present_image_count,
                supports_heaps: heap_bits.supports_heaps,
                max_heap_bytes: heap_bits.max_heap_bytes,
                supported_heap_storage_modes: heap_bits.supported_heap_storage_modes,
                supports_heap_aliasing: heap_bits.supports_heap_aliasing,
                supports_indirect_command_buffers: icb_bits.supports_indirect_command_buffers,
                max_indirect_commands: icb_bits.max_indirect_commands,
                supported_indirect_commands: icb_bits.supported_indirect_commands,
            };
            Ok(Self {
                epoch: allocate_device_epoch()?,
                name: device.name().into(),
                capabilities,
                render_sample_counts: multisample_bits.render_sample_counts,
                state: Mutex::new(State {
                    device,
                    queue,
                    pipelines: BTreeMap::new(),
                    next_pipeline: 1,
                    next_submission: 1,
                    present_targets: BTreeMap::new(),
                }),
                render_pipelines: Mutex::new(BTreeMap::new()),
                completions: Mutex::new(BTreeMap::new()),
                counters: Arc::new(CopyCounters::default()),
                present_counters: Arc::new(PresentCounters::default()),
                lifecycle: Arc::new(NativeLifecycle::new()),
                async_execution: false,
                observation_deadline: GPU_DEADLINE,
                completion_outbox: None,
                staging: LeaseRegistry::new(),
                lease_allocations: Mutex::new(BTreeMap::new()),
                borrowed: Arc::new(BorrowedLeaseRegistry::new()),
                heap_observations: Arc::new(Mutex::new(Vec::new())),
                icb_observations: Arc::new(Mutex::new(Vec::new())),
            })
        })
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// The reviewed sample counts this device admits, as the contract-code
    /// bitmask (`research/docs/23` §3.3, v61): bit `i` = `SampleCount` code
    /// `i`.
    pub fn render_sample_counts(&self) -> u32 {
        self.render_sample_counts
    }

    /// Select deferred submission. The default synchronous mode is retained
    /// for existing direct captures and for providers that need immediate
    /// readback.
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

    /// Bound how many deferred submissions may become unobservable before the
    /// provider refuses new work. The default tolerates eight abandonments.
    pub fn with_abandonment_budget(mut self, budget: AbandonmentBudget) -> Self {
        self.lifecycle = Arc::new(NativeLifecycle::with_budget(budget));
        self
    }

    /// Report whether this provider can still admit new work.
    ///
    /// Health and admission are the same query on the same lifecycle, so a
    /// provider that reports `Usable` here also admits the next submission.
    pub fn health(&self) -> ProviderHealth {
        self.lifecycle.health()
    }

    /// Report `(abandoned submissions, abandoned bytes)` for this provider.
    pub fn abandonment_stats(&self) -> (u64, u64) {
        self.lifecycle.abandonment()
    }

    /// Give up on one submission whose completion is no longer observable.
    ///
    /// The shared lifecycle charges the budget, so the provider turns terminal
    /// in the same transition that reaches the bound.
    fn record_abandonment(&self, bytes: u64) {
        self.lifecycle.record_abandonment(bytes);
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

    fn completions(
        &self,
    ) -> Result<MutexGuard<'_, BTreeMap<SubmissionId, CompletionSlot>>, ProviderError> {
        self.completions.lock().map_err(|_| {
            refusal(
                ProviderPhase::Wait,
                ProviderErrorClass::Internal,
                "provider_registry_poisoned",
            )
        })
    }

    fn slot(&self, token: CompletionToken) -> Result<CompletionSlot, ProviderError> {
        self.completions()?
            .get(&token.submission_id)
            .cloned()
            .ok_or_else(|| unknown_completion(token))
    }

    /// Land a render-bearing deferred submission's parked writebacks into its
    /// record. Idempotent: a terminal transition that already won (a cancel or
    /// a deadline failure) is not overwritten, and a compute-only submission
    /// parks nothing.
    fn land_deferred(&self, slot: &CompletionSlot) {
        if let Some(writebacks) = &slot.deferred_writebacks {
            slot.record.complete(writebacks.clone());
        }
    }

    fn running_record(&self, token: CompletionToken) -> Arc<CompletionRecord> {
        match &self.completion_outbox {
            Some(outbox) => {
                let _ = outbox.submitted(token);
                CompletionRecord::running_with_observer(token, outbox.observer())
            }
            None => CompletionRecord::running(),
        }
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
        self.publish_health(self.health());
    }

    fn publish_health(&self, health: ProviderHealth) {
        if let Some(outbox) = &self.completion_outbox {
            if outbox.health() != health {
                let _ = outbox.publish_device(health);
            }
        }
    }

    fn fail_deadline(&self, slot: &CompletionSlot, token: CompletionToken) -> ProviderError {
        let error = refusal(
            ProviderPhase::Wait,
            ProviderErrorClass::Execute,
            "metal_completion_unknown",
        )
        .with_completion(CompletionDisposition::SubmittedUnknown { token: Some(token) });
        slot.record.fail(error.clone());
        self.record_abandonment(slot.owned_bytes);
        error
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

fn device_lost_error(token: CompletionToken, detail: String) -> ProviderError {
    device_lost_refusal(ProviderPhase::Wait, Some(token)).with_detail(detail)
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
        self.lifecycle.admit()?;
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
                // A compute registration has no render half; a render pass
                // naming this id is refused by core admission.
                render: None,
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
        let slot = self
            .completions()?
            .remove(&token.submission_id)
            .ok_or_else(|| unknown_completion(token))?;
        if slot.record.is_running() {
            self.record_abandonment(slot.owned_bytes);
        }
        Ok(())
    }
}

impl ComputeProvider for NativeMetalProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> ProviderHealth {
        NativeMetalProvider::health(self)
    }

    fn submit(&self, admitted: ValidatedComputeTrace) -> Result<ProviderSubmission, ProviderError> {
        let trace = admitted.trace();
        self.check_epoch(trace.device_epoch)?;
        self.capabilities.admit(trace, admitted.resources())?;
        let mut state = self.lock()?;
        self.lifecycle.admit()?;
        // Resolve and retain every pass's pipeline under the same registry
        // lock, checking all metadata and local limits before GPU allocation.
        let mut pipelines = Vec::with_capacity(trace.passes.len());
        for (pass_index, pass) in trace.compute_passes().enumerate() {
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
        let borrowed_leases = trace
            .compute_passes()
            .flat_map(|pass| pass.buffers.iter())
            .filter_map(|view| match view.source {
                BufferSource::BorrowedNoCopy(lease_id) => Some(lease_id),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut retains = BorrowedRetains::new(Arc::clone(&self.borrowed), borrowed_leases);
        retains.retain()?;
        let resolve = |view: &BufferView| -> Result<ResolvedBuffer, ProviderError> {
            match &view.source {
                BufferSource::OwnedBytes(bytes) => Ok(ResolvedBuffer::Owned(bytes.clone())),
                BufferSource::StagedLease(lease_id) => self
                    .staging
                    .view_bytes(*lease_id, view, self.epoch, admitted.resources())
                    .map(ResolvedBuffer::Owned),
                BufferSource::BorrowedNoCopy(lease_id) => {
                    let resolved = self.borrowed.view_pointer(
                        *lease_id,
                        view,
                        self.epoch,
                        admitted.resources(),
                    )?;
                    let alignment = self.no_copy_alignment();
                    if alignment == 0 {
                        return Err(refusal(
                            ProviderPhase::Resolve,
                            ProviderErrorClass::Capability,
                            "storage_mode_unsupported",
                        ));
                    }
                    if !(resolved.base_len as u64).is_multiple_of(alignment) {
                        return Err(borrowed_length_error(
                            *lease_id,
                            resolved.base_len as u64,
                            alignment,
                        ));
                    }
                    let alignment = usize::try_from(alignment).unwrap_or(usize::MAX);
                    if !resolved.base_pointer.is_multiple_of(alignment) {
                        return Err(borrowed_alignment_error(
                            *lease_id,
                            resolved.base_pointer,
                            alignment as u64,
                        ));
                    }
                    if !resolved.offset.is_multiple_of(4) {
                        return Err(borrowed_offset_error(*lease_id, resolved.offset as u64));
                    }
                    Ok(ResolvedBuffer::Borrowed {
                        pointer: resolved.base_pointer as *mut u8,
                        length: resolved.base_len,
                        offset: resolved.offset as u64,
                    })
                }
            }
        };
        if self.async_execution {
            let result = self.submit_async(
                &mut state,
                trace,
                admitted.resources(),
                pipelines,
                token,
                &resolve,
                retains,
            );
            self.publish_health(self.lifecycle.health());
            return result;
        }
        let result = objc::rc::autoreleasepool(|| {
            self.execute(
                &mut state,
                trace,
                admitted.resources(),
                pipelines,
                token,
                &resolve,
                retains,
            )
        });
        let observation = match &result {
            Ok(submission) => Some(self.terminal_record(token, submission.writebacks.clone())),
            Err(error) if error.completion.token().is_some() => {
                Some(self.failed_record(token, error.clone()))
            }
            Err(_) => None,
        };
        if let Some(record) = observation {
            self.completions()?.insert(
                token.submission_id,
                CompletionSlot {
                    record,
                    deadline: ObservationDeadline::new(self.observation_deadline),
                    owned_bytes: trace_owned_bytes(trace),
                    deferred_writebacks: None,
                },
            );
        }
        self.publish_health(self.lifecycle.health());
        result
    }

    fn wait(
        &self,
        token: CompletionToken,
        timeout: Duration,
    ) -> Result<CompletionDisposition, ProviderError> {
        self.check_token(token)?;
        let slot = self.slot(token)?;
        self.land_deferred(&slot);
        if !slot.record.is_running() {
            let result = slot.record.wait(token, timeout);
            self.sync_completion_health();
            return result;
        }
        if slot.deadline.expired() {
            let result = Err(self.fail_deadline(&slot, token));
            self.sync_completion_health();
            return result;
        }
        let observed = slot.record.wait(token, slot.deadline.clamp(timeout))?;
        if matches!(observed, CompletionDisposition::TimedOut { .. }) && slot.deadline.expired() {
            let result = Err(self.fail_deadline(&slot, token));
            self.sync_completion_health();
            return result;
        }
        self.sync_completion_health();
        Ok(observed)
    }

    fn cancel(&self, token: CompletionToken) -> Result<CompletionDisposition, ProviderError> {
        self.check_token(token)?;
        let slot = self.slot(token)?;
        slot.record.cancel();
        let result = slot.record.wait(token, Duration::ZERO);
        self.sync_completion_health();
        result
    }

    fn readback(&self, token: CompletionToken) -> Result<CompletionReadback, ProviderError> {
        self.check_token(token)?;
        let slot = self.slot(token)?;
        self.land_deferred(&slot);
        slot.record.readback(token)
    }
}

struct SubmissionResources {
    // Retain the whole context as well as the explicit command dependencies.
    _device: Device,
    _queue: CommandQueue,
    pipelines: Vec<ComputePipelineState>,
    command: CommandBuffer,
    buffers: Vec<BoundBuffer>,
    /// Sampled textures by their contract view id (`research/docs/16` §4.8).
    textures: BTreeMap<ViewId, TextureRef>,
    /// The indirect command buffer a compute dispatch replays from, retained
    /// for the submission's whole lifetime (`research/docs/25` §6 Step 7b).
    indirect: Option<IndirectCommandBuffer>,
    // Keep owner mappings imported and retained until Metal retires the work.
    _borrowed: BorrowedRetains,
}

/// One pool buffer plus the byte offset of its admitted view.
///
/// Owned and staged views are copied into a buffer that starts at the view,
/// so their offset is zero. Borrowed views map the whole reservation, so the
/// view is addressed with a binding offset instead.
struct BoundBuffer {
    buffer: Buffer,
    offset: u64,
}

/// One owned MTLTexture created from a sampled view's initial bytes.
struct TextureRef {
    texture: Texture,
}

/// Retains no-copy leases for one submission until its Metal resources own
/// them. If encoding fails first, Drop returns the retains so the owner is not
/// blocked by a submission that never reached the queue.
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
}

impl Drop for BorrowedRetains {
    fn drop(&mut self) {
        if self.armed {
            self.registry.retire_all(&self.lease_ids);
        }
    }
}

struct EncodedSubmission {
    pending: crate::PendingSubmission<SubmissionResources>,
    pool: Vec<BufferView>,
    /// Heap placement observations planned with the submission, published once
    /// its command buffer reaches a terminal success.
    heap_observations: Option<Vec<heap::HeapPlacementObservation>>,
    /// The indirect replay planned with the submission, consumed by the
    /// dispatch encode path here and by the draw encode path in `render.rs`.
    icb_replay: Option<icb::IcbPlan>,
}

/// One admitted view resolved for binding. Owned and staged views return their
/// snapshot bytes; borrowed views return the owner pointer Metal maps directly.
enum ResolvedBuffer {
    Owned(Vec<u8>),
    Borrowed {
        pointer: *mut u8,
        length: usize,
        offset: u64,
    },
}

/// Resolves one admitted view to the exact binding used by Metal.
type BufferResolver<'a> = dyn Fn(&BufferView) -> Result<ResolvedBuffer, ProviderError> + 'a;

fn encode(
    state: &mut State,
    counters: &CopyCounters,
    trace: &ComputeTrace,
    resources: &ResourceTableSnapshot,
    pipelines: Vec<ComputePipelineState>,
    resolve: &BufferResolver<'_>,
    retains: BorrowedRetains,
) -> Result<EncodedSubmission, ProviderError> {
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
    // The heap payload maps the trace's owned allocations onto one shared slab;
    // a heap-less trace keeps the per-allocation buffer path unchanged
    // (`research/docs/25` §6 Step 7).
    let heap_plan = heap::plan_heap_placements(trace, &pool, resources)?;
    let heap_observations = heap_plan.as_ref().map(|plan| plan.observations.clone());
    let heap_offsets = heap_plan.as_ref().map(|plan| &plan.offsets);
    // The indirect payload maps onto one replayed command; a trace without one
    // keeps the direct draw/dispatch shape. The plan is pure, so a trace the
    // first increment cannot replay is refused before any Metal object exists.
    let icb_replay = icb::plan_replay(trace)?;
    let heap_slab = match &heap_plan {
        Some(plan) => {
            let slab = state
                .device
                .new_buffer(plan.slab_size, MTLResourceOptions::StorageModeShared);
            if slab.as_ptr().is_null() {
                return Err(resource_error("metal_heap_slab_allocation_failed"));
            }
            Some(slab)
        }
        None => None,
    };
    // Owned views of one allocation share one MTLBuffer, bound with the view's
    // own offset, so the image is uploaded once (`research/docs/15` §3). A lone
    // owned view keeps its exact-length buffer at offset zero. The image spans
    // the largest end offset across the allocation's views and is zero-filled
    // between them: nothing reads or writes there, because each view's
    // footprint proof bounds its own accesses.
    let overflow = || {
        refusal(
            ProviderPhase::Encode,
            ProviderErrorClass::Args,
            "buffer_range_overflow",
        )
    };
    let mut owned_per_allocation = BTreeMap::<AllocationId, usize>::new();
    for view in &pool {
        if matches!(view.source, BufferSource::OwnedBytes(_)) {
            *owned_per_allocation.entry(view.allocation_id).or_default() += 1;
        }
    }
    let mut shared_images = BTreeMap::<AllocationId, Vec<u8>>::new();
    for view in &pool {
        let BufferSource::OwnedBytes(_) = &view.source else {
            continue;
        };
        // A heap-placed allocation lives in the shared slab, not in a
        // per-allocation shared image, so it never enters this merge.
        if heap_offsets.is_some_and(|offsets| offsets.contains_key(&view.allocation_id)) {
            continue;
        }
        if owned_per_allocation
            .get(&view.allocation_id)
            .copied()
            .unwrap_or(0)
            < 2
        {
            continue;
        }
        let end = view.offset.checked_add(view.length).ok_or_else(overflow)?;
        let size = usize::try_from(end).map_err(|_| overflow())?;
        shared_images
            .entry(view.allocation_id)
            .and_modify(|image| {
                if image.len() < size {
                    image.resize(size, 0);
                }
            })
            .or_insert_with(|| vec![0_u8; size]);
    }
    for view in &pool {
        let BufferSource::OwnedBytes(bytes) = &view.source else {
            continue;
        };
        let Some(image) = shared_images.get_mut(&view.allocation_id) else {
            continue;
        };
        let start = usize::try_from(view.offset).map_err(|_| overflow())?;
        let end = start.checked_add(bytes.len()).ok_or_else(overflow)?;
        if end > image.len() {
            return Err(overflow());
        }
        image[start..end].copy_from_slice(bytes);
    }
    // The first view of an allocation creates its MTLBuffer and owns that
    // reference; every sibling retains the same object explicitly, which
    // balances the release its own wrapper performs on drop.
    let mut shared_buffers = BTreeMap::<AllocationId, *mut metal::MTLBuffer>::new();
    let mut buffers = Vec::with_capacity(pool.len());
    for view in &pool {
        let (buffer, offset) = match resolve(view)? {
            ResolvedBuffer::Owned(bytes)
                if heap_offsets
                    .is_some_and(|offsets| offsets.contains_key(&view.allocation_id)) =>
            {
                unsafe {
                    // A heap-placed owned view uploads its own bytes into the
                    // slab at the allocation's placement offset plus the view's
                    // own offset inside that allocation, and binds the slab at
                    // the same absolute offset (`research/docs/25` §6 Step 7).
                    let placement_offset = heap_offsets
                        .expect("heap branch requires placement offsets")[&view.allocation_id];
                    let binding_offset = placement_offset
                        .checked_add(view.offset)
                        .ok_or_else(overflow)?;
                    let slab = heap_slab.as_ref().expect("heap branch requires a slab");
                    let destination = slab
                        .contents()
                        .cast::<u8>()
                        .add(usize::try_from(binding_offset).map_err(|_| overflow())?);
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
                    counters.uploads.fetch_add(1, Ordering::Relaxed);
                    (slab.clone(), binding_offset)
                }
            }
            ResolvedBuffer::Owned(_) if shared_images.contains_key(&view.allocation_id) => unsafe {
                let image = &shared_images[&view.allocation_id];
                let pointer: *mut metal::MTLBuffer = match shared_buffers.get(&view.allocation_id) {
                    Some(existing) => msg_send![*existing, retain],
                    None => {
                        let created: *mut metal::MTLBuffer = msg_send![state.device.as_ref(),
                            newBufferWithBytes:image.as_ptr().cast::<std::ffi::c_void>()
                            length:image.len() options:MTLResourceOptions::StorageModeShared];
                        if created.is_null() {
                            return Err(resource_error("metal_buffer_allocation_failed"));
                        }
                        counters.uploads.fetch_add(1, Ordering::Relaxed);
                        shared_buffers.insert(view.allocation_id, created);
                        created
                    }
                };
                (Buffer::from_ptr(pointer), view.offset)
            },
            ResolvedBuffer::Owned(bytes) => unsafe {
                // The resolved bytes contain the view itself, not the entire
                // logical allocation. Binding offset is zero; writebacks
                // retain view.offset.
                let pointer: *mut metal::MTLBuffer = msg_send![state.device.as_ref(),
                    newBufferWithBytes:bytes.as_ptr().cast::<std::ffi::c_void>()
                    length:view.length options:MTLResourceOptions::StorageModeShared];
                if pointer.is_null() {
                    return Err(resource_error("metal_buffer_allocation_failed"));
                }
                counters.uploads.fetch_add(1, Ordering::Relaxed);
                (Buffer::from_ptr(pointer), 0)
            },
            ResolvedBuffer::Borrowed {
                pointer,
                length,
                offset,
            } => unsafe {
                // Metal maps the whole owner reservation directly; a nil
                // deallocator keeps the owner responsible for unmapping it.
                let buffer: *mut metal::MTLBuffer = msg_send![state.device.as_ref(),
                    newBufferWithBytesNoCopy:pointer.cast::<std::ffi::c_void>()
                    length:length
                    options:MTLResourceOptions::StorageModeShared
                    deallocator:std::ptr::null::<std::ffi::c_void>()];
                if buffer.is_null() {
                    return Err(resource_error("metal_no_copy_buffer_allocation_failed"));
                }
                (Buffer::from_ptr(buffer), offset)
            },
        };
        if buffer.contents().is_null() {
            return Err(resource_error("metal_buffer_mapping_failed"));
        }
        // MTLDevice-created resources default to tracked hazards. This is the
        // ordering guarantee used by the directly bound serial passes below.
        if buffer.hazard_tracking_mode() != MTLHazardTrackingMode::Tracked {
            return Err(resource_error("metal_buffer_hazard_tracking_unavailable"));
        }
        buffers.push(BoundBuffer { buffer, offset });
    }
    // Sampled textures (`research/docs/16` §4.8): the first increment accepts
    // D2, single-sample R32Uint owned bytes, uploaded once per submission.
    let mut bound_textures = BTreeMap::<ViewId, TextureRef>::new();
    for texture in trace.serial_texture_resources().map_err(|error| {
        refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Resource,
            "texture_contract_invalid",
        )
        .with_detail(error.to_string())
    })? {
        texture.validate_shape().map_err(|error| {
            refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Resource,
                "texture_shape_invalid",
            )
            .with_detail(error.to_string())
        })?;
        if texture.texture_type != TextureType::D2
            || texture.format != TextureFormat::R32Uint
            || texture.sample_count != 1
            || texture.depth != 1
            || texture.array_length != 1
            || texture.access != TextureAccess::Sampled
        {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Capability,
                "texture_shape_unsupported",
            ));
        }
        let TextureSource::OwnedBytes(bytes) = &texture.source else {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Capability,
                "texture_source_unsupported",
            ));
        };
        let width = texture.width;
        let height = texture.height;
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(MTLPixelFormat::R32Uint);
        descriptor.set_width(width);
        descriptor.set_height(height);
        descriptor.set_mipmap_level_count(1);
        descriptor.set_usage(MTLTextureUsage::ShaderRead);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        let created = state.device.new_texture(descriptor.as_ref());
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width,
                height,
                depth: 1,
            },
        };
        let stride = NSUInteger::try_from(width.saturating_mul(4)).unwrap_or(NSUInteger::MAX);
        // `replace_region` takes the *source* stride and owns the texture-side
        // layout, so this upload cannot repeat the Vulkan rail's defect: there
        // the host had to guess the destination `VkSubresourceLayout.rowPitch`,
        // while Metal keeps that distance inside the driver. The tightly packed
        // stride is the AIR fixture's row order and nothing else.
        created.replace_region(region, 0, bytes.as_ptr().cast(), stride);
        // A texture upload is a copy-in like a buffer upload, so the v11 count
        // contract sees one operation per touched allocation
        // (`research/docs/18` step 3).
        counters.uploads.fetch_add(1, Ordering::Relaxed);
        bound_textures.insert(texture.view_id, TextureRef { texture: created });
    }
    let command = unsafe {
        let pointer: *mut metal::MTLCommandBuffer = msg_send![state.queue.as_ref(), commandBuffer];
        if pointer.is_null() {
            return Err(resource_error("metal_command_buffer_allocation_failed"));
        }
        // commandBuffer is autoreleased, so retain it for the pending guard.
        CommandBufferRef::from_ptr(pointer).to_owned()
    };
    // The dispatch replay's device object is created here, after the buffers it
    // re-binds but before the command buffer that executes it, so it is
    // retained for the submission's whole lifetime (`research/docs/25` §6
    // Step 7b). The draw replay's object is created in `render.rs`.
    let indirect = match &icb_replay {
        Some(plan) if matches!(plan.command, icb::IcbCommand::Dispatch { .. }) => {
            let descriptor = IndirectCommandBufferDescriptor::new();
            descriptor.set_command_types(MTLIndirectCommandType::ConcurrentDispatch);
            descriptor.set_max_kernel_buffer_bind_count(31);
            let buffer = state.device.new_indirect_command_buffer_with_descriptor(
                &descriptor,
                u64::from(plan.max_commands),
                MTLResourceOptions::StorageModeShared,
            );
            if buffer.as_ptr().is_null() {
                return Err(resource_error(
                    "metal_indirect_command_buffer_allocation_failed",
                ));
            }
            Some(buffer)
        }
        _ => None,
    };
    let pending = crate::PendingSubmission {
        resources: Some(SubmissionResources {
            _device: state.device.clone(),
            _queue: state.queue.clone(),
            pipelines,
            command,
            buffers,
            textures: bound_textures,
            indirect,
            _borrowed: retains,
        }),
        submitted: false,
    };
    let resources = pending.resources.as_ref().expect("pending resources");
    // The default compute encoder dispatches serially. Directly bound tracked
    // resources on MTLCommandQueue carry writes across encoder boundaries:
    // https://developer.apple.com/documentation/metal/resource-synchronization
    // Each pass sees earlier writes; the initial bytes are uploaded only once.
    for (pass_index, pass) in trace.compute_passes().enumerate() {
        let encoder = unsafe {
            let pointer: *mut metal::MTLComputeCommandEncoder =
                msg_send![resources.command.as_ref(), computeCommandEncoder];
            if pointer.is_null() {
                return Err(resource_error("metal_encoder_allocation_failed"));
            }
            ComputeCommandEncoderRef::from_ptr(pointer)
        };
        // A dispatch replay is encoded entirely on the indirect command: its
        // own pipeline and buffer bindings, then one `concurrentDispatchThread-
        // groups` and an `executeCommandsInBuffer` from the encoder. Nothing is
        // bound on the encoder itself, which is what makes a direct dispatch
        // unable to stand in for it.
        if let Some(plan) = &icb_replay {
            let icb::IcbCommand::Dispatch { threadgroups } = plan.command else {
                unreachable!("a dispatch replay was planned for a non-dispatch command");
            };
            let buffer = resources
                .indirect
                .as_ref()
                .expect("the dispatch ICB was created during encode");
            let command = buffer.indirect_compute_command_at_index(u64::from(plan.range.start));
            command.set_compute_pipeline_state(&resources.pipelines[pass_index]);
            for view in &pass.buffers {
                let bound = &resources.buffers[pool_positions[&view.view_id]];
                command.set_kernel_buffer(
                    u64::from(view.metal_binding),
                    Some(&bound.buffer),
                    bound.offset,
                );
            }
            command.concurrent_dispatch_threadgroups(
                MTLSize::new(
                    u64::from(threadgroups[0]),
                    u64::from(threadgroups[1]),
                    u64::from(threadgroups[2]),
                ),
                MTLSize::new(
                    pass.dispatch.threads_per_threadgroup[0],
                    pass.dispatch.threads_per_threadgroup[1],
                    pass.dispatch.threads_per_threadgroup[2],
                ),
            );
            let range = NSRange::new(u64::from(plan.range.start), u64::from(plan.range.count));
            unsafe {
                let _: () = msg_send![
                    encoder,
                    executeCommandsInBuffer: buffer.as_ref()
                    withRange: range
                ];
            }
            encoder.end_encoding();
            continue;
        }
        encoder.set_compute_pipeline_state(&resources.pipelines[pass_index]);
        for view in &pass.buffers {
            // serial_resources validated that each pass binds a subset of this pool.
            let bound = &resources.buffers[pool_positions[&view.view_id]];
            encoder.set_buffer(
                u64::from(view.metal_binding),
                Some(&bound.buffer),
                bound.offset,
            );
        }
        for texture in &pass.textures {
            let bound = resources
                .textures
                .get(&texture.view_id)
                .ok_or_else(|| resource_error("metal_texture_not_uploaded"))?;
            encoder.set_texture(u64::from(texture.metal_binding), Some(&bound.texture));
        }
        let [gx, gy, gz] = pass.dispatch.grid;
        let [lx, ly, lz] = pass.dispatch.threads_per_threadgroup;
        encoder.dispatch_threads(MTLSize::new(gx, gy, gz), MTLSize::new(lx, ly, lz));
        encoder.end_encoding();
    }
    Ok(EncodedSubmission {
        pending,
        pool,
        heap_observations,
        icb_replay,
    })
}

impl NativeMetalProvider {
    /// Run one synchronous submission to a terminal command-buffer status.
    ///
    /// The lifecycle and the copy counters are the provider's, which is why
    /// this is a method: a terminal command buffer has to record its outcome on
    /// the same admission state `submit` read before the work was encoded.
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        state: &mut State,
        trace: &ComputeTrace,
        resources: &ResourceTableSnapshot,
        pipelines: Vec<ComputePipelineState>,
        token: CompletionToken,
        resolve: &BufferResolver<'_>,
        retains: BorrowedRetains,
    ) -> Result<ProviderSubmission, ProviderError> {
        let EncodedSubmission {
            mut pending,
            pool,
            heap_observations,
            icb_replay,
        } = encode(
            state,
            &self.counters,
            trace,
            resources,
            pipelines,
            resolve,
            retains,
        )?;
        // Render work is planned after the compute objects exist but before the
        // compute command buffer is committed: the ordering rule, the reviewed
        // allowlist, the attachment landing and the load op are all decided
        // from values, so a trace this provider cannot execute end to end is
        // refused with nothing on the queue.
        let render_contracts = self.render_contracts(trace)?;
        let render_plan = render::plan_trace(
            trace,
            &pool,
            &render_contracts,
            self.capabilities.depth_resolve_modes,
            self.capabilities.stencil_resolve_modes,
        )?;
        pending.submitted = true;
        let resources = pending.resources.as_ref().expect("encoded resources");
        resources.command.commit();
        let started = Instant::now();
        loop {
            match resources.command.status() {
                MTLCommandBufferStatus::Completed => break,
                MTLCommandBufferStatus::Error => {
                    let (detail, code) = unsafe {
                        let error: *mut Object = msg_send![resources.command.as_ref(), error];
                        (error_description(error), command_buffer_error_code(error))
                    };
                    if classify_command_buffer_error(code) == CommandBufferFailure::DeviceLost {
                        self.lifecycle.mark_device_lost();
                        return Err(device_lost_error(token, detail));
                    }
                    // Every other terminal command-buffer failure keeps the
                    // `metal_command_failed` path and seals the instance
                    // without charging the abandonment budget
                    // (`docs/PROVIDER-B1.md`, "Execution failures and
                    // visibility").
                    self.lifecycle.mark_unobservable_submission();
                    return Err(refusal(
                        ProviderPhase::Wait,
                        ProviderErrorClass::Execute,
                        "metal_command_failed",
                    )
                    .with_detail(detail)
                    .with_completion(CompletionDisposition::Failed { token: Some(token) }));
                }
                _ if started.elapsed() >= GPU_DEADLINE => {
                    // The command buffer never reached a terminal status, so
                    // the instance stops trusting the device; this is a
                    // classified unknown completion, not booked abandoned GPU
                    // work.
                    self.lifecycle.mark_unobservable_submission();
                    return Err(refusal(
                        ProviderPhase::Wait,
                        ProviderErrorClass::Execute,
                        "metal_completion_unknown",
                    )
                    .with_completion(CompletionDisposition::SubmittedUnknown {
                        token: Some(token),
                    }));
                }
                _ => std::thread::sleep(Duration::from_millis(1)),
            }
        }
        // Shared memory on the admitted device is now CPU visible. Only a known
        // completed command permits the guard to release its backing resources.
        pending.submitted = false;
        // The render rail runs last, on the same queue and after the compute
        // command buffer reached its terminal status, and the merge is what
        // lands its texels: the attachment's view is already a written view of
        // the compute pool, so its pre-render bytes are replaced rather than
        // reported alongside.
        let render_writebacks =
            self.execute_render_passes(state, &render_plan, icb_replay.as_ref())?;
        let writebacks = render::merge_writebacks(
            collect_writebacks(
                &pool,
                &resources.buffers,
                &self.counters,
                &discarded_only_attachments(trace),
            ),
            render_writebacks,
        );
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
        if let Some(observations) = heap_observations {
            self.publish_heap_observations(observations);
        }
        if let Some(plan) = &icb_replay {
            self.publish_icb_observation(plan.observation());
        }
        Ok(submission)
    }
}

/// The `(allocation, view)` identities a trace discards and never stores.
///
/// Core's resource table marks every attachment view writable, because the
/// render pass is what writes it; a discarded-only attachment is the one case
/// where no bytes ever leave the rail for that view (`research/docs/23` §3.6,
/// v19). The compute-side collect has to skip those views: reading them back
/// would both fabricate a pre-render writeback and charge a copy-out the pass
/// never performs. A view that any pass stores stays in the collect, so a mixed
/// store/discard identity cannot slip out of the count.
fn discarded_only_attachments(trace: &ComputeTrace) -> BTreeSet<(AllocationId, ViewId)> {
    let mut stored = BTreeSet::new();
    let mut discarded = BTreeSet::new();
    for attachment in trace
        .render_passes()
        .flat_map(|pass| pass.color_attachments.iter())
    {
        let identity = (attachment.allocation_id, attachment.view_id);
        match attachment.store {
            StoreOp::Store => {
                stored.insert(identity);
            }
            StoreOp::DontCare => {
                discarded.insert(identity);
            }
        }
    }
    discarded.difference(&stored).copied().collect()
}

fn collect_writebacks(
    pool: &[BufferView],
    buffers: &[BoundBuffer],
    counters: &CopyCounters,
    discarded: &BTreeSet<(AllocationId, ViewId)>,
) -> Vec<BufferWriteback> {
    let mut read_buffers = BTreeSet::<usize>::new();
    let mut writebacks = Vec::new();
    for (view, bound) in pool.iter().zip(buffers) {
        if discarded.contains(&(view.allocation_id, view.view_id)) {
            continue;
        }
        if view.access.is_writable() {
            read_buffers.insert(bound.buffer.as_ptr() as usize);
            let bytes = unsafe {
                // Admission bounded length to 1 MiB, contents was checked
                // before commit, and this completed buffer remains retained.
                let contents = bound
                    .buffer
                    .contents()
                    .cast::<u8>()
                    .add(bound.offset as usize);
                std::slice::from_raw_parts(contents, view.length as usize).to_vec()
            };
            writebacks.push(BufferWriteback {
                view_id: view.view_id,
                allocation_id: view.allocation_id,
                offset: view.offset,
                bytes,
            });
        }
    }
    counters
        .readbacks
        .fetch_add(read_buffers.len(), Ordering::Relaxed);
    writebacks.sort_by_key(|writeback| (writeback.allocation_id, writeback.view_id));
    writebacks
}

impl NativeMetalProvider {
    /// Cumulative device-buffer copy-in / copy-out operations. Tests use it to
    /// prove that several views of one allocation share one copy.
    #[doc(hidden)]
    pub fn buffer_copy_counts(&self) -> (usize, usize) {
        (
            self.counters.uploads.load(Ordering::Relaxed),
            self.counters.readbacks.load(Ordering::Relaxed),
        )
    }

    /// Cumulative acquire / present completions of the present rail. The
    /// capture harness asserts the present milestone's one acquire and one
    /// present from this pair, the same shape `buffer_copy_counts` gives the
    /// copy path (`research/docs/24` §1.4, §5.3).
    #[doc(hidden)]
    pub fn present_counts(&self) -> (usize, usize) {
        (
            self.present_counters.acquires.load(Ordering::Relaxed),
            self.present_counters.presents.load(Ordering::Relaxed),
        )
    }

    /// Heap placements the native provider most recently executed.
    ///
    /// Every successful heap-bearing submission replaces the previous vector
    /// instead of appending to it, so this stays bounded to one submission
    /// rather than growing across a long-running process. Each record names the
    /// heap, the owned allocation placed in it, and the
    /// `[offset, offset + byte_size)` range it occupies; two resources in one
    /// heap therefore appear as two records sharing one `heap_id`, which is the
    /// falsifiable observation `research/docs/25` §6 Step 3 requires rather
    /// than a "looks shared" assertion.
    pub fn heap_placement_observations(&self) -> Vec<heap::HeapPlacementObservation> {
        self.heap_observations
            .lock()
            .expect("heap observation lock poisoned")
            .clone()
    }

    /// Record one heap placement set, replacing whatever the previous
    /// submission left behind (the same bounded shape the Vulkan rail uses).
    fn publish_heap_observations(&self, observations: Vec<heap::HeapPlacementObservation>) {
        *self
            .heap_observations
            .lock()
            .expect("heap observation lock poisoned") = observations;
    }

    /// The indirect replay the native provider executed last, if the last
    /// submission carried one. Like the heap observation, a successful
    /// ICB-bearing submission replaces the vector instead of appending to it.
    pub fn icb_replay_observations(&self) -> Vec<icb::IcbReplayObservation> {
        self.icb_observations
            .lock()
            .expect("icb observation lock poisoned")
            .clone()
    }

    /// Record one indirect replay, replacing whatever the previous submission
    /// left behind (the same bounded shape the heap observation uses).
    fn publish_icb_observation(&self, observation: icb::IcbReplayObservation) {
        let mut observations = self
            .icb_observations
            .lock()
            .expect("icb observation lock poisoned");
        observations.clear();
        observations.push(observation);
    }

    /// Staged lease registry owned by this provider.
    pub fn lease_registry(&self) -> &LeaseRegistry {
        &self.staging
    }

    /// No-copy lease registry owned by this provider.
    pub fn borrowed_registry(&self) -> &BorrowedLeaseRegistry {
        &self.borrowed
    }

    /// Drop any present target reserved for the allocation of a released
    /// lease.
    ///
    /// A present target lives across submissions until its allocation's lease
    /// is released, so releasing the lease has to retire the target too
    /// (`research/docs/24` §6 Step 7). The lease→allocation mapping is removed
    /// in the same call, so a double release cannot retire a sibling
    /// allocation's targets.
    fn drop_present_targets_for(&self, lease_id: LeaseId) -> Result<(), ProviderError> {
        let allocation_id = self
            .lease_allocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&lease_id);
        if let Some(allocation_id) = allocation_id {
            let mut state = self.lock()?;
            state
                .present_targets
                .retain(|(allocation, _), _| *allocation != allocation_id);
        }
        Ok(())
    }

    /// Register one render pipeline: the trace-table entry a render pass names
    /// and the reviewed contract the rail compiles.
    ///
    /// The render sibling of [`PipelineProvider::compile`]. It validates the
    /// contract, refuses an entry pair the reviewed module does not carry
    /// (`native_render_source_not_reviewed`), mints the pipeline identity from
    /// the same counter compute pipelines use, and hands back the table entry a
    /// trace has to carry. The module itself stays in the rail, which compiles
    /// the reviewed bytes at execution.
    pub fn register_render_pipeline(
        &self,
        request: NativeRenderPipelineRequest,
    ) -> Result<CompiledComputePipeline, ProviderError> {
        let mut state = self.lock()?;
        self.lifecycle.admit()?;
        request
            .contract
            .validate()
            .map_err(|error| render_contract_error(error.to_string()))?;
        render::review_contract(&request.contract)?;
        let function = FunctionIdentity {
            logical_digest: request.logical_digest,
            entry_name: request.contract.vertex_entry.clone(),
            // The rail hands Metal source, which is the representation this
            // registration is compiled from, exactly as `compile` does for a
            // reviewed compute fixture.
            source: FunctionSource::MetalSource,
        };
        function
            .validate()
            .map_err(|error| render_contract_error(error.to_string()))?;
        let metadata = CompiledComputePipeline {
            device_epoch: self.epoch,
            pipeline_id: PipelineId::new(next_id(&mut state.next_pipeline)?),
            function,
            contract: render_table_contract(),
            // The half that makes this a render registration: core admission
            // compares a render pass's attachment with this contract, and the
            // render rail re-checks it against the registration below.
            render: Some(request.contract.clone()),
        };
        self.render_pipelines()?.insert(
            metadata.pipeline_id,
            RegisteredRenderPipeline {
                metadata: metadata.clone(),
                contract: request.contract,
            },
        );
        Ok(metadata)
    }

    /// Stop accepting new submissions that name this render registration.
    ///
    /// Symmetric with [`PipelineProvider::release_pipeline`]: the epoch and the
    /// registered identity are verified before the entry is removed, so a stale
    /// or foreign value cannot release another context's registration. A
    /// submission that already resolved this id holds its own copy of the
    /// contract, so releasing never pulls the reviewed contract out from under
    /// work that is being planned or encoded.
    pub fn release_render_pipeline(
        &self,
        metadata: &CompiledComputePipeline,
    ) -> Result<(), ProviderError> {
        self.check_epoch(metadata.device_epoch)?;
        let mut registrations = self.render_pipelines()?;
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

    fn render_pipelines(
        &self,
    ) -> Result<MutexGuard<'_, BTreeMap<PipelineId, RegisteredRenderPipeline>>, ProviderError> {
        self.render_pipelines.lock().map_err(|_| {
            refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Internal,
                "provider_registry_poisoned",
            )
        })
    }

    /// The reviewed contracts a trace's render passes name, keyed by pipeline id.
    ///
    /// A render pass's pipeline id is caller-supplied table data, so the entry
    /// the trace carries is checked against the registration that owns the same
    /// id before the contract is handed to the rail: a trace cannot name a
    /// render pipeline this context never registered, and it cannot pass a table
    /// entry that disagrees with one.
    fn render_contracts(
        &self,
        trace: &ComputeTrace,
    ) -> Result<BTreeMap<PipelineId, RenderPipelineContract>, ProviderError> {
        let mut contracts = BTreeMap::new();
        if !trace.has_render_passes() {
            return Ok(contracts);
        }
        let registrations = self.render_pipelines()?;
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
            if requested != &registered.metadata {
                return Err(refusal(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Resource,
                    "render_pipeline_identity_mismatch",
                )
                .with_field("pipeline", FieldValue::Unsigned(pass.pipeline.get())));
            }
            contracts.insert(pass.pipeline, registered.contract.clone());
        }
        Ok(contracts)
    }

    /// Execute the planned render passes in trace order, after the compute
    /// sequence, and turn each attachment readback into a buffer writeback —
    /// the stored depth attachment's own included when the pass has one and the
    /// stored stencil attachment's own when it has that one
    /// (`research/docs/23` §3.3, v43/v49).
    ///
    /// The attachment's bytes leave the rail through the same channel a compute
    /// pass uses — one [`BufferWriteback`] for the view and allocation the trace
    /// declared, at the view's own offset inside the allocation — so resource
    /// admission, lease bookkeeping and readback consumers need no second path
    /// (`research/docs/23` §6 Step 7).
    fn execute_render_passes(
        &self,
        state: &mut State,
        plan: &[render::TraceRenderPlan<'_>],
        icb_replay: Option<&icb::IcbPlan>,
    ) -> Result<Vec<BufferWriteback>, ProviderError> {
        let mut writebacks = Vec::with_capacity(plan.len());
        for planned in plan {
            // One copy-in per sampled texture the encoder uploads into its own
            // `MTLTexture` (`research/docs/23` §3.3, v70). A texture upload is
            // a copy-in like a buffer upload, so the v11 count contract sees
            // one operation per touched allocation, matching the Vulkan rail.
            self.counters
                .uploads
                .fetch_add(planned.plan.textures.len(), Ordering::Relaxed);
            match &planned.present {
                // A present action hands its one attachment on to the target
                // texture, whose readback is the pass's single writeback.
                Some(present) => {
                    let texels = self.execute_present_render(state, planned, present)?;
                    writebacks.push(planned.writeback(texels));
                }
                // An offscreen pass reads every attachment back, one writeback
                // per landing view in location order, plus the stored depth
                // and stencil surfaces' own when the pass has them (v43/v49).
                None if icb_replay.is_some() => {
                    let readback = render::encode_indirect_offscreen_render(
                        &state.device,
                        &state.queue,
                        &planned.plan,
                        icb_replay.expect("the indirect draw was planned"),
                    )?;
                    writebacks.extend(planned.writebacks(readback));
                }
                None => {
                    let readback = render::encode_offscreen_render(
                        &state.device,
                        &state.queue,
                        &planned.plan,
                    )?;
                    writebacks.extend(planned.writebacks(readback));
                }
            }
        }
        Ok(writebacks)
    }

    /// Execute one present action: acquire the target, preset the sentinel if
    /// the target declares one, render the pass's attachment into it, present
    /// it, and read the target back (`research/docs/24` §6 Step 7). The target
    /// texture is created once and reused across submissions until the
    /// allocation's lease is released.
    fn execute_present_render(
        &self,
        state: &mut State,
        planned: &render::TraceRenderPlan<'_>,
        present: &render::PresentPlan<'_>,
    ) -> Result<Vec<u8>, ProviderError> {
        let key = (
            present.descriptor.target.allocation_id,
            present.descriptor.target.view_id,
        );
        let texture = if let Some(entry) = state.present_targets.get(&key) {
            entry.texture.clone()
        } else {
            let texture = render::present_target_texture(
                &state.device,
                planned.plan.attachments[0].format,
                planned.plan.extent,
            )?;
            state.present_targets.insert(
                key,
                PresentTargetRef {
                    texture: texture.clone(),
                },
            );
            texture
        };
        // acquire: take the target's ownership before the pass writes it.
        self.present_counters
            .acquires
            .fetch_add(1, Ordering::Relaxed);
        // The sentinel makes "the present never happened" falsifiable: a
        // present that claims success without running the render reads back the
        // sentinel, not the fragment texel (`research/docs/24` §3.1).
        if let Some(sentinel) = &present.sentinel {
            render::upload_texels(&texture, &planned.plan, sentinel);
        }
        let texels =
            render::encode_present_render(&state.device, &state.queue, &planned.plan, &texture)?;
        // present: hand the target on after the pass completed.
        self.present_counters
            .presents
            .fetch_add(1, Ordering::Relaxed);
        Ok(texels)
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_async(
        &self,
        state: &mut State,
        trace: &ComputeTrace,
        resources: &ResourceTableSnapshot,
        pipelines: Vec<ComputePipelineState>,
        token: CompletionToken,
        resolve: &BufferResolver<'_>,
        retains: BorrowedRetains,
    ) -> Result<ProviderSubmission, ProviderError> {
        let EncodedSubmission {
            mut pending,
            pool,
            heap_observations,
            icb_replay,
        } = encode(
            state,
            &self.counters,
            trace,
            resources,
            pipelines,
            resolve,
            retains,
        )?;
        if trace.has_render_passes() {
            // Render-bearing deferred submission. The render rail runs on the
            // same Metal queue, so it serializes after the compute command
            // buffer. Committing compute first and then executing the render
            // synchronously blocks until both complete; the shared-storage
            // compute buffers are then CPU-visible, so they are read back
            // directly instead of through a completion handler. The record
            // stays `Running` and the merged writebacks are parked, so `cancel`
            // can still abandon the observation before `wait` lands it — the
            // same shape the Vulkan object rail reports.
            let render_contracts = self.render_contracts(trace)?;
            let render_plan = render::plan_trace(
                trace,
                &pool,
                &render_contracts,
                self.capabilities.depth_resolve_modes,
                self.capabilities.stencil_resolve_modes,
            )?;
            pending.submitted = true;
            let resources = pending.resources.as_ref().expect("encoded resources");
            resources.command.commit();
            // The compute command buffer is committed and the render rail now
            // serializes after it on the same queue, so the submission is no
            // longer in the "commit could unwind" window the submitted guard
            // exists for. Clear the flag before executing the render passes so
            // a Metal render failure returns through `?` with the normal error
            // path and releases `SubmissionResources` instead of `mem::forget`
            // leaking the whole bundle (the sync rail clears it before render
            // for the same reason).
            pending.submitted = false;
            let render_writebacks =
                self.execute_render_passes(state, &render_plan, icb_replay.as_ref())?;
            // The render command buffer serialized after the compute command
            // buffer, so a completed render implies a terminal compute status.
            match resources.command.status() {
                MTLCommandBufferStatus::Completed => {}
                MTLCommandBufferStatus::Error => {
                    let (detail, code) = unsafe {
                        let error: *mut Object = msg_send![resources.command.as_ref(), error];
                        (error_description(error), command_buffer_error_code(error))
                    };
                    if classify_command_buffer_error(code) == CommandBufferFailure::DeviceLost {
                        self.lifecycle.mark_device_lost();
                        return Err(device_lost_error(token, detail));
                    }
                    self.lifecycle.mark_unobservable_submission();
                    return Err(refusal(
                        ProviderPhase::Wait,
                        ProviderErrorClass::Execute,
                        "metal_command_failed",
                    )
                    .with_detail(detail)
                    .with_completion(CompletionDisposition::Failed { token: Some(token) }));
                }
                _ => {
                    self.lifecycle.mark_unobservable_submission();
                    return Err(refusal(
                        ProviderPhase::Wait,
                        ProviderErrorClass::Internal,
                        "metal_completion_unknown",
                    )
                    .with_completion(CompletionDisposition::SubmittedUnknown {
                        token: Some(token),
                    }));
                }
            }
            let compute_writebacks = collect_writebacks(
                &pool,
                &resources.buffers,
                &self.counters,
                &discarded_only_attachments(trace),
            );
            let merged = render::merge_writebacks(compute_writebacks, render_writebacks);
            let record = self.running_record(token);
            self.completions()?.insert(
                token.submission_id,
                CompletionSlot {
                    record,
                    deadline: ObservationDeadline::new(self.observation_deadline),
                    owned_bytes: trace_owned_bytes(trace),
                    deferred_writebacks: Some(merged),
                },
            );
            if let Some(observations) = &heap_observations {
                self.publish_heap_observations(observations.clone());
            }
            if let Some(plan) = &icb_replay {
                self.publish_icb_observation(plan.observation());
            }
            return Ok(ProviderSubmission {
                completion: CompletionDisposition::Submitted { token },
                writebacks: Vec::new(),
            });
        }
        let SubmissionResources {
            _device,
            _queue,
            pipelines: retained_pipelines,
            command,
            buffers,
            textures: _textures,
            indirect: _indirect,
            _borrowed,
        } = pending.resources.take().expect("encoded resources");
        pending.submitted = true;
        let record = self.running_record(token);
        self.completions()?.insert(
            token.submission_id,
            CompletionSlot {
                record: Arc::clone(&record),
                deadline: ObservationDeadline::new(self.observation_deadline),
                owned_bytes: trace_owned_bytes(trace),
                deferred_writebacks: None,
            },
        );
        // The completion handler runs after this call returned, so it holds the
        // shared lifecycle instead of borrowing the provider.
        let lifecycle = Arc::clone(&self.lifecycle);
        let outbox = self.completion_outbox.clone();
        let counters = Arc::clone(&self.counters);
        let heap_observations_arc = Arc::clone(&self.heap_observations);
        let icb_observations_arc = Arc::clone(&self.icb_observations);
        // The handler outlives this call, so the discarded-only identity set is
        // computed here and moved into the block rather than borrowed.
        let discarded_only = discarded_only_attachments(trace);
        let handler = ConcreteBlock::new(move |command: &CommandBufferRef| {
            // Retain the device, queue and compiled pipelines for the whole
            // device execution; the block itself is retained by the command
            // buffer until it is invoked.
            let _retain = (
                &_device,
                &_queue,
                &retained_pipelines,
                &_indirect,
                &_borrowed,
            );
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                objc::rc::autoreleasepool(|| match command.status() {
                    MTLCommandBufferStatus::Completed => Ok(collect_writebacks(
                        &pool,
                        &buffers,
                        &counters,
                        &discarded_only,
                    )),
                    MTLCommandBufferStatus::Error => {
                        let (detail, code) = unsafe {
                            let error: *mut Object = msg_send![command, error];
                            (error_description(error), command_buffer_error_code(error))
                        };
                        if classify_command_buffer_error(code) == CommandBufferFailure::DeviceLost {
                            lifecycle.mark_device_lost();
                            Err(device_lost_error(token, detail))
                        } else {
                            lifecycle.mark_unobservable_submission();
                            Err(refusal(
                                ProviderPhase::Wait,
                                ProviderErrorClass::Execute,
                                "metal_command_failed",
                            )
                            .with_detail(detail)
                            .with_completion(CompletionDisposition::Failed { token: Some(token) }))
                        }
                    }
                    _ => Err(refusal(
                        ProviderPhase::Wait,
                        ProviderErrorClass::Internal,
                        "metal_completion_handler_without_terminal_status",
                    )
                    .with_completion(CompletionDisposition::SubmittedUnknown {
                        token: Some(token),
                    })),
                })
            }));
            match outcome {
                Ok(Ok(writebacks)) => {
                    if let Some(observations) = &heap_observations {
                        *heap_observations_arc
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                            observations.clone();
                    }
                    if let Some(plan) = &icb_replay {
                        let mut observations = icb_observations_arc
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        observations.clear();
                        observations.push(plan.observation());
                    }
                    record.complete(writebacks)
                }
                Ok(Err(error)) => record.fail(error),
                Err(_) => record.fail(
                    refusal(
                        ProviderPhase::Wait,
                        ProviderErrorClass::Internal,
                        "metal_completion_handler_panicked",
                    )
                    .with_completion(CompletionDisposition::SubmittedUnknown {
                        token: Some(token),
                    }),
                ),
            }
            if let Some(outbox) = &outbox {
                let health = lifecycle.health();
                if outbox.health() != health {
                    let _ = outbox.publish_device(health);
                }
            }
        });
        let block = handler.copy();
        command.add_completed_handler(&block);
        command.commit();
        Ok(ProviderSubmission {
            completion: CompletionDisposition::Submitted { token },
            writebacks: Vec::new(),
        })
    }
}

impl LeaseImporter for NativeMetalProvider {
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
        self.drop_present_targets_for(lease_id)
    }
}

impl NoCopyLeaseImporter for NativeMetalProvider {
    fn no_copy_alignment(&self) -> u64 {
        page_size()
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
        if !borrowed.reservation.length.is_multiple_of(alignment) {
            return Err(borrowed_length_error(
                borrowed.lease_id(),
                borrowed.reservation.length,
                alignment,
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
        self.drop_present_targets_for(lease_id)
    }
}

/// Page size Metal requires for `newBufferWithBytesNoCopy:` on macOS.
fn page_size() -> u64 {
    // SAFETY: `sysconf` has no preconditions for `_SC_PAGESIZE`.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if size <= 0 {
        0
    } else {
        size as u64
    }
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

fn borrowed_length_error(lease_id: LeaseId, length: u64, alignment: u64) -> ProviderError {
    refusal(
        ProviderPhase::Resolve,
        ProviderErrorClass::Capability,
        "lease_length_unsupported",
    )
    .with_field("lease", FieldValue::Unsigned(lease_id.get()))
    .with_field("length", FieldValue::Unsigned(length))
    .with_field("alignment", FieldValue::Unsigned(alignment))
}

fn borrowed_offset_error(lease_id: LeaseId, offset: u64) -> ProviderError {
    refusal(
        ProviderPhase::Resolve,
        ProviderErrorClass::Capability,
        "lease_offset_unsupported",
    )
    .with_field("lease", FieldValue::Unsigned(lease_id.get()))
    .with_field("offset", FieldValue::Unsigned(offset))
    .with_field("alignment", FieldValue::Unsigned(4))
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

/// The refusal a render pass gets when it names a pipeline this context never
/// registered as a render pipeline. Mirror of [`unknown_pipeline`]: the two
/// registries share one id namespace, so a render pass naming a compute
/// registration and a compute pass naming a render registration are refused
/// symmetrically.
fn unknown_render_pipeline(id: PipelineId) -> ProviderError {
    refusal(
        ProviderPhase::Resolve,
        ProviderErrorClass::Resource,
        "unknown_render_pipeline",
    )
    .with_field("pipeline", FieldValue::Unsigned(id.get()))
}

/// A render pipeline contract the context cannot register or execute.
fn render_contract_error(detail: String) -> ProviderError {
    refusal(
        ProviderPhase::Compile,
        ProviderErrorClass::Args,
        "render_pipeline_contract_invalid",
    )
    .with_detail(detail)
}

/// The compute half of the pipeline-table entry one render registration carries.
///
/// `ComputeTrace` has a single pipeline entry shape and core admission validates
/// every entry's contract, so a render registration carries the most permissive
/// exact-thread contract: no bindings, no push constants and no fixed grid.
/// Nothing reads it as a compute contract — the compute rail resolves artifacts
/// out of [`State::pipelines`], where a render registration does not exist. The
/// entry's render half, which is what core admission compares a render pass
/// against, is set by [`NativeMetalProvider::register_render_pipeline`].
fn render_table_contract() -> PipelineContract {
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

/// Called only inside an autorelease pool with nil or a live NSError pointer.
unsafe fn command_buffer_error_code(error: *mut Object) -> Option<i64> {
    if error.is_null() {
        return None;
    }
    Some(msg_send![error, code])
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
