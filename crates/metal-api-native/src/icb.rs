//! Native indirect-command rail (`research/docs/25-heaps与ICB设计.md` §6 Step 7b).
//!
//! The native provider replays the trace's optional `indirect` payload through
//! a real `MTLIndirectCommandBuffer`: a compute dispatch is encoded on one
//! `MTLIndirectComputeCommand` and replayed with `executeCommandsInBuffer`,
//! and a non-indexed draw is encoded on one `MTLIndirectRenderCommand` and
//! replayed the same way. The planning half is pure and is compiled both for
//! the macOS provider and for the host-side unit tests that pin it on a
//! machine that cannot load Metal; only the encode bodies in `native.rs` and
//! `render.rs` need a device.
//!
//! The capability bits stay closed until the Apple `--icb-selftest` produces
//! the single-device evidence the flip condition names; `icb_capability_bits`
//! is the one spelling of that condition so the snapshot and the tests cannot
//! drift from it.
#![allow(dead_code)] // ICB bits stay closed until the Apple selftest flips them

use crate::refusal;
use metal_api_core::provider::{
    ComputeTrace, IndirectCommandDescriptor, IndirectCommandKind, IndirectCommandRange,
    ProviderError, ProviderErrorClass, ProviderPhase,
};

/// One indirect replay the native provider executed: the command kind, the
/// half-open range it replayed and how many commands it encoded
/// (`research/docs/25` §5.1). The capture reports this record, not the suite's
/// request, so a rail that ran a direct draw/dispatch cannot claim an indirect
/// replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IcbReplayObservation {
    pub kind: IndirectCommandKind,
    pub start: u32,
    pub count: u32,
    pub commands: u32,
}

/// The ICB bits this provider declares, in one value so the macOS capability
/// snapshot (`native.rs`) and the host-side unit tests cannot drift.
///
/// Flip evidence (`research/docs/25` §6 Step 7b): a CI run whose
/// `native-oracle-build` job ran "Run native ICB self-test when a Metal device
/// is eligible". The probe must report an eligible Apple Paravirtual device and
/// the oracle must encode one compute dispatch on a `MTLIndirectCommandBuffer`,
/// replay it with `executeCommandsInBuffer`, and read `icb_selftest: PASS
/// (fefefefe)` back — the copied word, not the `ffffffff` sentinel the write
/// buffer was preset with. A green job whose log said `SKIP` is not that
/// evidence. Until the flip every field stays at its default, so core admission
/// refuses an ICB-bearing trace with `icb_unsupported` instead of silently
/// dropping the replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct IcbCapabilityBits {
    pub(crate) supports_indirect_command_buffers: bool,
    pub(crate) max_indirect_commands: u32,
    pub(crate) supported_indirect_commands: Vec<IndirectCommandKind>,
}

/// The pre-flip ICB bits: no indirect command buffers. The Apple selftest is
/// what turns them on, and the parent flips them only after that CI evidence.
pub(crate) fn icb_capability_bits() -> IcbCapabilityBits {
    IcbCapabilityBits {
        supports_indirect_command_buffers: false,
        max_indirect_commands: 0,
        supported_indirect_commands: Vec::new(),
    }
}

/// One command the rail replayed, narrowed to what the encode body needs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IcbCommand {
    Dispatch {
        threadgroups: [u32; 3],
    },
    Draw {
        vertex_count: u32,
        instance_count: u32,
    },
}

impl IcbCommand {
    pub(crate) fn kind(&self) -> IndirectCommandKind {
        match self {
            Self::Dispatch { .. } => IndirectCommandKind::Dispatch,
            Self::Draw { .. } => IndirectCommandKind::Draw,
        }
    }
}

/// The provider-side half of an indirect payload's mapping onto one command:
/// the command to encode, the fixed command count and the replay range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct IcbPlan {
    pub(crate) command: IcbCommand,
    pub(crate) max_commands: u32,
    pub(crate) range: IndirectCommandRange,
}

impl IcbPlan {
    /// The observation a successful replay publishes: one command, the plan's
    /// kind and the plan's replay range (`research/docs/25` §5.1). All three
    /// publish sites (sync, async render-bearing, async compute-only) derive
    /// the record from this one shape so none of them can drift.
    pub(crate) fn observation(&self) -> IcbReplayObservation {
        IcbReplayObservation {
            kind: self.command.kind(),
            start: self.range.start,
            count: self.range.count,
            commands: 1,
        }
    }
}

/// Map a trace's indirect payload onto one replayed command and record the
/// rules the first increment can honour (`research/docs/25` §6 Step 7b).
///
/// The dispatch path mirrors the Vulkan rail: one indirect command replays
/// exactly one full-workgroup compute pass, and the encoded threadgroups must
/// equal that pass's planned group count. The draw path replays exactly one
/// non-indexed render pass. Everything wider is a typed refusal rather than a
/// silent fallback to the direct shape.
pub(crate) fn plan_replay(trace: &ComputeTrace) -> Result<Option<IcbPlan>, ProviderError> {
    let Some(indirect) = &trace.indirect else {
        return Ok(None);
    };
    let plan = match indirect.command {
        IndirectCommandDescriptor::Dispatch { threadgroups } => {
            if trace.has_render_passes() {
                return Err(icb_refusal(
                    "an indirect dispatch replays a compute pass, not a render pass",
                ));
            }
            let mut passes = trace.compute_passes();
            let pass = passes.next().ok_or_else(|| {
                icb_refusal("an indirect dispatch replays exactly one compute pass")
            })?;
            if passes.next().is_some() {
                return Err(icb_refusal(
                    "an indirect dispatch replays exactly one compute pass",
                ));
            }
            let local = pass.dispatch.threads_per_threadgroup;
            if local.contains(&0) {
                return Err(icb_refusal(
                    "indirect dispatch threadgroup dimensions must be nonzero",
                ));
            }
            // A `ThreadsExact` dispatch launches ceil(grid / local) threadgroups
            // per axis; the ICB command has to name that count or the footprint
            // proof would describe a different launch than the one replayed.
            let group_count = [
                u32::try_from(pass.dispatch.grid[0].div_ceil(local[0]))
                    .map_err(|_| icb_refusal("indirect dispatch group count exceeds u32"))?,
                u32::try_from(pass.dispatch.grid[1].div_ceil(local[1]))
                    .map_err(|_| icb_refusal("indirect dispatch group count exceeds u32"))?,
                u32::try_from(pass.dispatch.grid[2].div_ceil(local[2]))
                    .map_err(|_| icb_refusal("indirect dispatch group count exceeds u32"))?,
            ];
            if group_count != threadgroups {
                return Err(icb_refusal(format!(
                    "indirect dispatch threadgroups {threadgroups:?} disagree with the planned group count {group_count:?}"
                )));
            }
            IcbCommand::Dispatch { threadgroups }
        }
        IndirectCommandDescriptor::Draw {
            vertex_count,
            instance_count,
        } => {
            if trace.compute_passes().next().is_some() {
                return Err(icb_refusal(
                    "an indirect draw replays a render pass, not a compute pass",
                ));
            }
            let mut passes = trace.render_passes();
            let pass = passes
                .next()
                .ok_or_else(|| icb_refusal("an indirect draw replays exactly one render pass"))?;
            if passes.next().is_some() {
                return Err(icb_refusal(
                    "an indirect draw replays exactly one render pass",
                ));
            }
            if pass.present.is_some() {
                return Err(icb_refusal(
                    "the first indirect increment replays offscreen draws only",
                ));
            }
            IcbCommand::Draw {
                vertex_count,
                instance_count,
            }
        }
        IndirectCommandDescriptor::DrawIndexed { .. } => {
            return Err(icb_refusal(
                "the first indirect increment replays non-indexed draws only",
            ));
        }
    };
    Ok(Some(IcbPlan {
        command: plan,
        max_commands: indirect.buffer.max_commands,
        range: indirect.range,
    }))
}

/// The capability refusal both rails publish for a well-formed indirect command
/// this increment cannot replay (`research/docs/25` §4.3).
fn icb_refusal(detail: impl Into<String>) -> ProviderError {
    refusal(
        ProviderPhase::Resolve,
        ProviderErrorClass::Capability,
        "icb_command_unsupported",
    )
    .with_detail(detail.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::provider::{
        AliasMode, AllocationId, AllocationRecord, AttachmentFormat, BufferAccess,
        BufferBindingContract, BufferSource, BufferView, ClearColor, CompiledComputePipeline,
        CompletionPolicy, ComputePass, DeviceEpoch, Dispatch, DispatchKind, DispatchType,
        FootprintProof, FunctionIdentity, FunctionSource, IndirectCommandBufferDescriptor,
        IndirectCommandPayload, LoadOp, OperationId, PipelineContract, PipelineId,
        ProviderCapabilities, RenderAttachment, RenderPassDescriptor, ResourceTableSnapshot,
        SemanticDigest, StorageMode, StoreOp, TracePass, ViewId, PROVIDER_SCHEMA_VERSION,
    };

    fn pipeline() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(1),
            pipeline_id: PipelineId::new(1),
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("icb-fixture", vec![1]).unwrap(),
                entry_name: "copy_word".to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: PipelineContract {
                dispatch_kind: DispatchKind::ThreadsExact,
                required_local_size: None,
                fixed_grid: None,
                push_constant_offset: 0,
                push_constant_bytes: 0,
                buffer_bindings: vec![
                    BufferBindingContract {
                        metal_binding: 0,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 4 },
                    },
                    BufferBindingContract {
                        metal_binding: 1,
                        access: BufferAccess::Write,
                        footprint: FootprintProof::Static { max_bytes: 4 },
                    },
                ],
                shader_capabilities: Vec::new(),
                translator_revision: None,
            },
            render: None,
        }
    }

    fn compute_pass(grid: [u64; 3], local: [u64; 3]) -> TracePass {
        TracePass::Compute(ComputePass {
            pipeline: PipelineId::new(1),
            buffers: vec![
                BufferView {
                    view_id: ViewId::new(1),
                    metal_binding: 0,
                    allocation_id: AllocationId::new(1),
                    offset: 0,
                    length: 4,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![0xfe; 4]),
                },
                BufferView {
                    view_id: ViewId::new(2),
                    metal_binding: 1,
                    allocation_id: AllocationId::new(2),
                    offset: 0,
                    length: 4,
                    access: BufferAccess::Write,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![0xff; 4]),
                },
            ],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid,
                threads_per_threadgroup: local,
            },
            textures: Vec::new(),
        })
    }

    fn render_pass() -> TracePass {
        TracePass::Render(RenderPassDescriptor {
            pipeline: PipelineId::new(1),
            color_attachments: vec![RenderAttachment {
                view_id: ViewId::new(3),
                allocation_id: AllocationId::new(3),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                load: LoadOp::Clear(ClearColor::new([0xfe; 4])),
                store: StoreOp::Store,
            }],
            viewport: [0, 0, 2, 2],
            vertices: 3,
            present: None,
        })
    }

    fn trace_with(
        passes: Vec<TracePass>,
        indirect: Option<IndirectCommandPayload>,
    ) -> ComputeTrace {
        ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(1),
            operation_id: OperationId::new(1),
            pipelines: vec![pipeline()],
            encoder_dispatch_type: DispatchType::Serial,
            passes,
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: indirect.map(Box::new),
        }
    }

    fn dispatch_payload(threadgroups: [u32; 3]) -> IndirectCommandPayload {
        IndirectCommandPayload {
            buffer: IndirectCommandBufferDescriptor {
                max_commands: 1,
                kinds: vec![IndirectCommandKind::Dispatch],
            },
            command: IndirectCommandDescriptor::Dispatch { threadgroups },
            range: IndirectCommandRange { start: 0, count: 1 },
        }
    }

    fn draw_payload() -> IndirectCommandPayload {
        IndirectCommandPayload {
            buffer: IndirectCommandBufferDescriptor {
                max_commands: 1,
                kinds: vec![IndirectCommandKind::Draw],
            },
            command: IndirectCommandDescriptor::Draw {
                vertex_count: 3,
                instance_count: 1,
            },
            range: IndirectCommandRange { start: 0, count: 1 },
        }
    }

    fn snapshot() -> ResourceTableSnapshot {
        let mut snapshot = ResourceTableSnapshot::new();
        for (allocation, size) in [(1, 4), (2, 4), (3, 16)] {
            snapshot
                .insert_allocation(AllocationRecord {
                    allocation_id: AllocationId::new(allocation),
                    owner_epoch: DeviceEpoch::new(1),
                    size,
                })
                .unwrap();
        }
        snapshot
    }

    fn capabilities(bits: &IcbCapabilityBits) -> ProviderCapabilities {
        ProviderCapabilities {
            max_passes: 8,
            supports_threads_exact: true,
            supports_threadgroups: false,
            supports_serial: true,
            supports_concurrent: false,
            max_local_size: [1024, 1024, 1024],
            max_invocations: 1024,
            max_group_count: [1024, 1024, 1024],
            max_storage_buffer_descriptors: 31,
            max_buffer_range: 1024,
            max_push_constant_bytes: 0,
            alias_mode: AliasMode::DistinctViews,
            storage_modes: vec![StorageMode::OwnedBytes],
            host_readback: true,
            submit_only: false,
            supports_render_passes: false,
            max_color_attachments: 0,
            max_attachment_dimension: [0, 0],
            supported_color_formats: Vec::new(),
            supports_presentation: false,
            max_present_targets: 0,
            supported_present_modes: Vec::new(),
            max_present_image_count: 0,
            supports_heaps: false,
            max_heap_bytes: 0,
            supported_heap_storage_modes: Vec::new(),
            supports_heap_aliasing: false,
            supports_indirect_command_buffers: bits.supports_indirect_command_buffers,
            max_indirect_commands: bits.max_indirect_commands,
            supported_indirect_commands: bits.supported_indirect_commands.clone(),
        }
    }

    #[test]
    fn icb_capability_bits_stay_closed_until_the_selftest_flips_them() {
        let bits = icb_capability_bits();
        assert!(!bits.supports_indirect_command_buffers);
        assert_eq!(bits.max_indirect_commands, 0);
        assert!(bits.supported_indirect_commands.is_empty());

        let snapshot = snapshot();
        let trace = trace_with(
            vec![compute_pass([1, 1, 1], [1, 1, 1])],
            Some(dispatch_payload([1, 1, 1])),
        );
        let refused = capabilities(&bits).admit(&trace, &snapshot).unwrap_err();
        assert_eq!(refused.slug, "icb_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
    }

    #[test]
    fn no_indirect_payload_is_no_replay() {
        let trace = trace_with(vec![compute_pass([1, 1, 1], [1, 1, 1])], None);
        assert!(plan_replay(&trace).unwrap().is_none());
    }

    #[test]
    fn dispatch_replay_accepts_the_planned_group_count() {
        let trace = trace_with(
            vec![compute_pass([2, 3, 4], [1, 1, 1])],
            Some(dispatch_payload([2, 3, 4])),
        );
        let plan = plan_replay(&trace)
            .unwrap()
            .expect("dispatch payload plans");
        assert_eq!(
            plan.command,
            IcbCommand::Dispatch {
                threadgroups: [2, 3, 4]
            }
        );
        assert_eq!(plan.max_commands, 1);
        assert_eq!(plan.range, IndirectCommandRange { start: 0, count: 1 });
        assert_eq!(plan.command.kind(), IndirectCommandKind::Dispatch);
    }

    #[test]
    fn dispatch_replay_refuses_a_mismatched_group_count() {
        let trace = trace_with(
            vec![compute_pass([2, 3, 4], [1, 1, 1])],
            Some(dispatch_payload([1, 1, 1])),
        );
        let error = plan_replay(&trace).unwrap_err();
        assert_eq!(error.slug, "icb_command_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
    }

    #[test]
    fn dispatch_replay_refuses_multiple_compute_passes() {
        let trace = trace_with(
            vec![
                compute_pass([1, 1, 1], [1, 1, 1]),
                compute_pass([1, 1, 1], [1, 1, 1]),
            ],
            Some(dispatch_payload([1, 1, 1])),
        );
        let error = plan_replay(&trace).unwrap_err();
        assert_eq!(error.slug, "icb_command_unsupported");
    }

    #[test]
    fn dispatch_replay_refuses_a_render_pass() {
        let trace = trace_with(
            vec![compute_pass([1, 1, 1], [1, 1, 1]), render_pass()],
            Some(dispatch_payload([1, 1, 1])),
        );
        let error = plan_replay(&trace).unwrap_err();
        assert_eq!(error.slug, "icb_command_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
    }

    #[test]
    fn compute_only_dispatch_publishes_the_expected_replay_observation() {
        // The async compute-only submission cannot run on a host without a
        // Metal device, so this pins the pure half: the exact observation the
        // completion handler publishes once the command buffer completes.
        let trace = trace_with(
            vec![compute_pass([2, 3, 4], [1, 1, 1])],
            Some(dispatch_payload([2, 3, 4])),
        );
        let plan = plan_replay(&trace)
            .unwrap()
            .expect("dispatch payload plans");
        assert_eq!(
            plan.observation(),
            IcbReplayObservation {
                kind: IndirectCommandKind::Dispatch,
                start: 0,
                count: 1,
                commands: 1,
            }
        );
    }

    #[test]
    fn draw_replay_accepts_one_render_pass() {
        let trace = trace_with(vec![render_pass()], Some(draw_payload()));
        let plan = plan_replay(&trace).unwrap().expect("draw payload plans");
        assert_eq!(
            plan.command,
            IcbCommand::Draw {
                vertex_count: 3,
                instance_count: 1,
            }
        );
        assert_eq!(plan.command.kind(), IndirectCommandKind::Draw);
    }

    #[test]
    fn draw_replay_refuses_multiple_render_passes() {
        let trace = trace_with(vec![render_pass(), render_pass()], Some(draw_payload()));
        let error = plan_replay(&trace).unwrap_err();
        assert_eq!(error.slug, "icb_command_unsupported");
    }

    #[test]
    fn draw_replay_refuses_a_compute_pass() {
        let trace = trace_with(
            vec![compute_pass([1, 1, 1], [1, 1, 1]), render_pass()],
            Some(draw_payload()),
        );
        let error = plan_replay(&trace).unwrap_err();
        assert_eq!(error.slug, "icb_command_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
    }

    #[test]
    fn indexed_draw_is_refused() {
        let payload = IndirectCommandPayload {
            buffer: IndirectCommandBufferDescriptor {
                max_commands: 1,
                kinds: vec![IndirectCommandKind::DrawIndexed],
            },
            command: IndirectCommandDescriptor::DrawIndexed {
                index_count: 3,
                instance_count: 1,
            },
            range: IndirectCommandRange { start: 0, count: 1 },
        };
        let trace = trace_with(vec![render_pass()], Some(payload));
        let error = plan_replay(&trace).unwrap_err();
        assert_eq!(error.slug, "icb_command_unsupported");
    }
}
