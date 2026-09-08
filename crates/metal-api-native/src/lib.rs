//! Bounded native Metal compute provider for independent offline comparison.
//!
//! The provider accepts only the six exact MSL fixtures shipped with the
//! conformance suite. Their manually audited contracts are not reflection of
//! arbitrary MSL. Execution is synchronous, uses fresh shared buffers, and is
//! restricted to unified-memory devices supporting Apple GPU family 4.

#[cfg(any(target_os = "macos", test))]
use metal_api_core::provider::{
    AffineAccess, AffineTerm, BufferAccess, BufferBindingContract, CompletionDisposition,
    CompletionToken, DispatchKind, FootprintProof, PipelineCompileRequest, PipelineContract,
    ShaderSource,
};
use metal_api_core::provider::{ProviderError, ProviderErrorClass, ProviderPhase, Retryability};

#[cfg(target_os = "macos")]
mod native;
#[cfg(target_os = "macos")]
pub use native::NativeMetalProvider;

#[cfg(not(target_os = "macos"))]
mod unsupported;
#[cfg(not(target_os = "macos"))]
pub use unsupported::NativeMetalProvider;

fn refusal(phase: ProviderPhase, class: ProviderErrorClass, slug: &'static str) -> ProviderError {
    let mut error = ProviderError::new(phase, class, slug).expect("non-empty native error slug");
    error.retryability = Retryability::Never;
    error
}

/// A device that reported `DeviceRemoved` stays unusable until recreation.
#[cfg(any(target_os = "macos", test))]
fn device_lost_refusal(phase: ProviderPhase, token: Option<CompletionToken>) -> ProviderError {
    let mut error = refusal(
        phase,
        ProviderErrorClass::DeviceLost,
        "metal_device_removed",
    );
    error.retryability = Retryability::RetryAfterRecreate;
    error.completion = CompletionDisposition::DeviceLost { token };
    error
}

/// `MTLCommandBufferError::DeviceRemoved` from Metal 0.33.
#[cfg(any(target_os = "macos", test))]
pub(crate) const DEVICE_REMOVED_ERROR_CODE: i64 = 11;

/// How a terminal `MTLCommandBufferStatus::Error` must be classified.
#[cfg(any(target_os = "macos", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CommandBufferFailure {
    DeviceLost,
    Other,
}

/// Classify a command-buffer error code without loading Metal.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn classify_command_buffer_error(code: Option<i64>) -> CommandBufferFailure {
    if code == Some(DEVICE_REMOVED_ERROR_CODE) {
        CommandBufferFailure::DeviceLost
    } else {
        CommandBufferFailure::Other
    }
}

#[cfg(any(target_os = "macos", test))]
fn unknown_completion(token: CompletionToken) -> ProviderError {
    // The record may have been explicitly released. Its absence cannot prove
    // that a submission never happened or that backing is safe to release.
    refusal(
        ProviderPhase::Wait,
        ProviderErrorClass::Resource,
        "unknown_completion",
    )
    .with_completion(CompletionDisposition::SubmittedUnknown { token: Some(token) })
}

#[cfg(any(target_os = "macos", test))]
const COPY: &str = include_str!("../../../conformance/shaders/copy_word.metal");
#[cfg(any(target_os = "macos", test))]
const INDEXED: &str = include_str!("../../../conformance/shaders/indexed_boundary.metal");
#[cfg(any(target_os = "macos", test))]
const TRANSFORM: &str = include_str!("../../../conformance/shaders/transform_3d.metal");
#[cfg(any(target_os = "macos", test))]
const MIX: &str = include_str!("../../../conformance/shaders/mix_3d.metal");
#[cfg(any(target_os = "macos", test))]
const REMAP: &str = include_str!("../../../conformance/shaders/remap_3d.metal");
#[cfg(any(target_os = "macos", test))]
const COPY_3D: &str = include_str!("../../../conformance/shaders/copy_3d.metal");

/// Exact byte equality is essential: a matching entry name or digest cannot
/// establish the footprint of caller-supplied source.
#[cfg(any(target_os = "macos", test))]
fn bounded_contract(request: &PipelineCompileRequest) -> Result<PipelineContract, ProviderError> {
    request.validate().map_err(|error| {
        refusal(
            ProviderPhase::Compile,
            ProviderErrorClass::Args,
            "invalid_compile_request",
        )
        .with_detail(error.to_string())
    })?;
    let ShaderSource::MetalSource(source) = &request.source else {
        return Err(refusal(
            ProviderPhase::Compile,
            ProviderErrorClass::Capability,
            "shader_source_unsupported",
        ));
    };
    let static_word = || FootprintProof::Static { max_bytes: 4 };
    let affine = |strides: &[u64]| FootprintProof::Affine {
        accesses: vec![AffineAccess {
            base_offset: 0,
            access_size: 4,
            terms: strides
                .iter()
                .enumerate()
                .map(|(axis, stride)| AffineTerm {
                    axis: axis as u8,
                    stride: *stride,
                })
                .collect(),
        }],
    };
    let binding = |metal_binding, access, footprint| BufferBindingContract {
        metal_binding,
        access,
        footprint,
    };
    let (grid, buffer_bindings) = match (request.entry_name.as_str(), source.as_str()) {
        ("copy_word", COPY) => (
            [1, 1, 1],
            vec![
                binding(0, BufferAccess::Read, static_word()),
                binding(1, BufferAccess::Write, static_word()),
            ],
        ),
        ("kernel_dispatch_threads_boundary_barrier", INDEXED) => (
            [10, 3, 1],
            vec![binding(0, BufferAccess::Write, affine(&[4, 40]))],
        ),
        ("transform_3d", TRANSFORM) | ("mix_3d", MIX) => (
            [5, 3, 2],
            vec![
                binding(0, BufferAccess::ReadWrite, affine(&[4, 20, 60])),
                binding(2, BufferAccess::Read, static_word()),
                binding(5, BufferAccess::Write, affine(&[4, 20, 60])),
            ],
        ),
        ("remap_3d", REMAP) => (
            [5, 3, 2],
            vec![
                binding(1, BufferAccess::Read, static_word()),
                binding(3, BufferAccess::Read, affine(&[4, 20, 60])),
                binding(7, BufferAccess::Write, affine(&[4, 20, 60])),
            ],
        ),
        ("copy_3d", COPY_3D) => (
            [5, 3, 2],
            vec![
                binding(4, BufferAccess::Read, affine(&[4, 20, 60])),
                binding(9, BufferAccess::Write, affine(&[4, 20, 60])),
            ],
        ),
        _ => {
            return Err(refusal(
                ProviderPhase::Compile,
                ProviderErrorClass::Capability,
                "native_shader_not_allowlisted",
            ));
        }
    };
    Ok(PipelineContract {
        dispatch_kind: DispatchKind::ThreadsExact,
        required_local_size: None,
        fixed_grid: Some(grid),
        push_constant_offset: 0,
        push_constant_bytes: 0,
        buffer_bindings,
        shader_capabilities: Vec::new(),
        translator_revision: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use metal_api_core::completion::wire::{
        CompletionMessage, CompletionOutbox, CompletionSink, CompletionUpdate,
    };
    use metal_api_core::provider::{DeviceEpoch, SemanticDigest, SubmissionId};
    #[cfg(target_os = "macos")]
    use std::sync::{Arc, Mutex};

    #[cfg(target_os = "macos")]
    #[derive(Default)]
    struct RecordingSink {
        messages: Mutex<Vec<CompletionMessage>>,
    }

    #[cfg(target_os = "macos")]
    impl CompletionSink for RecordingSink {
        fn deliver(&self, message: CompletionMessage) {
            self.messages
                .lock()
                .expect("recording completion sink")
                .push(message);
        }
    }

    #[test]
    fn device_removed_is_distinct_from_other_command_buffer_errors() {
        assert_eq!(
            classify_command_buffer_error(Some(DEVICE_REMOVED_ERROR_CODE)),
            CommandBufferFailure::DeviceLost
        );
        assert_eq!(
            classify_command_buffer_error(Some(2)),
            CommandBufferFailure::Other
        );
        assert_eq!(
            classify_command_buffer_error(None),
            CommandBufferFailure::Other
        );
    }

    #[test]
    fn device_loss_requires_recreation_and_keeps_the_observed_token() {
        let unavailable = device_lost_refusal(ProviderPhase::Resolve, None);
        assert_eq!(unavailable.phase, ProviderPhase::Resolve);
        assert_eq!(unavailable.class, ProviderErrorClass::DeviceLost);
        assert_eq!(unavailable.slug, "metal_device_removed");
        assert_eq!(unavailable.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(
            unavailable.completion,
            CompletionDisposition::DeviceLost { token: None }
        );

        let token = CompletionToken {
            submission_id: SubmissionId::new(7),
            device_epoch: DeviceEpoch::new(3),
        };
        let observed = device_lost_refusal(ProviderPhase::Wait, Some(token));
        assert_eq!(observed.phase, ProviderPhase::Wait);
        assert_eq!(
            observed.completion,
            CompletionDisposition::DeviceLost { token: Some(token) }
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_device_removed_code_matches_the_local_classifier() {
        assert_eq!(
            metal::MTLCommandBufferError::DeviceRemoved as i64,
            DEVICE_REMOVED_ERROR_CODE
        );
    }

    fn request(entry: &str, source: &str) -> PipelineCompileRequest {
        PipelineCompileRequest {
            entry_name: entry.into(),
            logical_digest: SemanticDigest::new("fixture", vec![1]).unwrap(),
            source: ShaderSource::MetalSource(source.into()),
        }
    }

    #[test]
    fn only_exact_source_and_entry_can_claim_a_fixture_contract() {
        for (entry, source) in [
            ("copy_word", COPY),
            ("kernel_dispatch_threads_boundary_barrier", INDEXED),
            ("transform_3d", TRANSFORM),
            ("mix_3d", MIX),
            ("remap_3d", REMAP),
            ("copy_3d", COPY_3D),
        ] {
            let contract = bounded_contract(&request(entry, source)).unwrap();
            contract.validate().unwrap();
            assert_eq!(contract.required_local_size, None);
            assert_eq!(contract.push_constant_bytes, 0);
            for invalid in [
                request("wrong_entry", source),
                request(entry, &format!("{source}\n")),
            ] {
                let error = bounded_contract(&invalid).unwrap_err();
                assert_eq!(error.slug, "native_shader_not_allowlisted");
                assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
            }
        }
        let changed = COPY.replace("output[0]", "output[100]");
        assert!(bounded_contract(&request("copy_word", &changed)).is_err());
        assert!(bounded_contract(&request("transform_3d", MIX)).is_err());
        assert!(bounded_contract(&request("mix_3d", TRANSFORM)).is_err());
        assert!(bounded_contract(&request("remap_3d", TRANSFORM)).is_err());
        assert!(bounded_contract(&request("transform_3d", REMAP)).is_err());
        assert!(bounded_contract(&request("copy_3d", COPY)).is_err());
        assert!(bounded_contract(&request("copy_word", COPY_3D)).is_err());
    }

    #[test]
    fn bounds_and_accesses_cover_each_fixture_at_its_fixed_grid() {
        for (entry, source, grid, expected) in [
            (
                "copy_word",
                COPY,
                [1, 1, 1],
                vec![(0, BufferAccess::Read, 4), (1, BufferAccess::Write, 4)],
            ),
            (
                "kernel_dispatch_threads_boundary_barrier",
                INDEXED,
                [10, 3, 1],
                vec![(0, BufferAccess::Write, 120)],
            ),
            (
                "transform_3d",
                TRANSFORM,
                [5, 3, 2],
                vec![
                    (0, BufferAccess::ReadWrite, 120),
                    (2, BufferAccess::Read, 4),
                    (5, BufferAccess::Write, 120),
                ],
            ),
            (
                "mix_3d",
                MIX,
                [5, 3, 2],
                vec![
                    (0, BufferAccess::ReadWrite, 120),
                    (2, BufferAccess::Read, 4),
                    (5, BufferAccess::Write, 120),
                ],
            ),
            (
                "remap_3d",
                REMAP,
                [5, 3, 2],
                vec![
                    (1, BufferAccess::Read, 4),
                    (3, BufferAccess::Read, 120),
                    (7, BufferAccess::Write, 120),
                ],
            ),
            (
                "copy_3d",
                COPY_3D,
                [5, 3, 2],
                vec![(4, BufferAccess::Read, 120), (9, BufferAccess::Write, 120)],
            ),
        ] {
            let contract = bounded_contract(&request(entry, source)).unwrap();
            assert_eq!(contract.fixed_grid, Some(grid));
            let actual: Vec<_> = contract
                .buffer_bindings
                .iter()
                .map(|binding| {
                    let end = match &binding.footprint {
                        FootprintProof::Static { max_bytes } => *max_bytes,
                        FootprintProof::Affine { accesses } => accesses
                            .iter()
                            .map(|access| {
                                access.base_offset
                                    + access.access_size
                                    + access
                                        .terms
                                        .iter()
                                        .map(|term| (grid[term.axis as usize] - 1) * term.stride)
                                        .sum::<u64>()
                            })
                            .max()
                            .unwrap(),
                        FootprintProof::Unbounded => panic!("allowlisted shader needs a proof"),
                    };
                    (binding.metal_binding, binding.access, end)
                })
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn remapped_fixture_has_its_own_access_and_affine_proofs() {
        let contract = bounded_contract(&request("remap_3d", REMAP)).unwrap();
        assert_eq!(contract.buffer_bindings[0].metal_binding, 1);
        assert_eq!(contract.buffer_bindings[0].access, BufferAccess::Read);
        assert_eq!(
            contract.buffer_bindings[0].footprint,
            FootprintProof::Static { max_bytes: 4 }
        );
        let indexed_proof = FootprintProof::Affine {
            accesses: vec![AffineAccess {
                base_offset: 0,
                access_size: 4,
                terms: vec![
                    AffineTerm { axis: 0, stride: 4 },
                    AffineTerm {
                        axis: 1,
                        stride: 20,
                    },
                    AffineTerm {
                        axis: 2,
                        stride: 60,
                    },
                ],
            }],
        };
        assert_eq!(contract.buffer_bindings[1].metal_binding, 3);
        assert_eq!(contract.buffer_bindings[1].access, BufferAccess::Read);
        assert_eq!(contract.buffer_bindings[1].footprint, indexed_proof);
        assert_eq!(contract.buffer_bindings[2].metal_binding, 7);
        assert_eq!(contract.buffer_bindings[2].access, BufferAccess::Write);
        assert_eq!(contract.buffer_bindings[2].footprint, indexed_proof);
        for changed in [
            REMAP.replace("buffer(3)", "buffer(0)"),
            REMAP.replace("input[index]", "input[index + 1]"),
            REMAP.replace("device const uint *input", "device uint *input"),
        ] {
            assert_eq!(
                bounded_contract(&request("remap_3d", &changed))
                    .unwrap_err()
                    .slug,
                "native_shader_not_allowlisted"
            );
        }
    }

    #[test]
    fn subset_copy_fixture_pins_slots_accesses_and_three_dimensional_bounds() {
        let contract = bounded_contract(&request("copy_3d", COPY_3D)).unwrap();
        let footprint = FootprintProof::Affine {
            accesses: vec![AffineAccess {
                base_offset: 0,
                access_size: 4,
                terms: vec![
                    AffineTerm { axis: 0, stride: 4 },
                    AffineTerm {
                        axis: 1,
                        stride: 20,
                    },
                    AffineTerm {
                        axis: 2,
                        stride: 60,
                    },
                ],
            }],
        };
        assert_eq!(contract.fixed_grid, Some([5, 3, 2]));
        assert_eq!(
            contract.buffer_bindings,
            vec![
                BufferBindingContract {
                    metal_binding: 4,
                    access: BufferAccess::Read,
                    footprint: footprint.clone(),
                },
                BufferBindingContract {
                    metal_binding: 9,
                    access: BufferAccess::Write,
                    footprint,
                },
            ]
        );
        for changed in [
            COPY_3D.replace("buffer(4)", "buffer(0)"),
            COPY_3D.replace("buffer(9)", "buffer(5)"),
            COPY_3D.replace("input[index]", "input[index + 1]"),
            COPY_3D.replace("gid.z * 3", "gid.z * 4"),
            COPY_3D.replace("device const uint *input", "device uint *input"),
        ] {
            assert_ne!(changed, COPY_3D);
            assert_eq!(
                bounded_contract(&request("copy_3d", &changed))
                    .unwrap_err()
                    .slug,
                "native_shader_not_allowlisted"
            );
        }
    }

    #[test]
    fn unsupported_representations_are_typed_compile_refusals() {
        for source in [
            ShaderSource::SanitizedLl("define void @copy_word() {}".into()),
            ShaderSource::BinaryAir(vec![1]),
        ] {
            let mut input = request("copy_word", COPY);
            input.source = source;
            let error = bounded_contract(&input).unwrap_err();
            assert_eq!(error.phase, ProviderPhase::Compile);
            assert_eq!(error.class, ProviderErrorClass::Capability);
            assert_eq!(error.slug, "shader_source_unsupported");
        }
        assert_eq!(
            bounded_contract(&request("", COPY)).unwrap_err().class,
            ProviderErrorClass::Args
        );
    }

    #[test]
    fn missing_completion_record_never_claims_not_submitted_or_retired() {
        let token = CompletionToken {
            device_epoch: DeviceEpoch::new(1),
            submission_id: SubmissionId::new(2),
        };
        let error = unknown_completion(token);
        assert_eq!(error.phase, ProviderPhase::Wait);
        assert_eq!(error.class, ProviderErrorClass::Resource);
        assert_eq!(error.slug, "unknown_completion");
        assert_eq!(
            error.completion,
            CompletionDisposition::SubmittedUnknown { token: Some(token) }
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn non_macos_reports_unavailable_without_loading_metal() {
        let error = NativeMetalProvider::new().err().unwrap();
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.slug, "native_metal_platform_unavailable");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn deadline_expiry_keeps_the_native_provider_usable() {
        use metal_api_core::provider::{
            AllocationId, AllocationRecord, BufferSource, BufferView, CompletionPolicy,
            ComputePass, ComputeProvider, ComputeTrace, Dispatch, DispatchType, OperationId,
            PipelineProvider, ProviderHealth, ResourceTableSnapshot, ViewId,
            PROVIDER_SCHEMA_VERSION,
        };
        use std::time::Duration;

        let Ok(provider) = NativeMetalProvider::new() else {
            eprintln!("skipping native deadline test: no eligible Metal device");
            return;
        };
        let provider = provider
            .with_async_execution(true)
            .with_observation_deadline(Duration::ZERO);
        let sink = Arc::new(RecordingSink::default());
        let outbox =
            Arc::new(CompletionOutbox::new(provider.device_epoch(), sink.clone()).unwrap());
        let provider = provider.with_completion_outbox(outbox).unwrap();
        let pipeline = provider
            .compile(request("copy_word", COPY))
            .expect("reviewed copy_word fixture compiles");
        let make_trace = || ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: provider.device_epoch(),
            operation_id: OperationId::new(1),
            pipelines: vec![pipeline.clone()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![ComputePass {
                pipeline: pipeline.pipeline_id,
                buffers: vec![
                    BufferView {
                        view_id: ViewId::new(1),
                        metal_binding: 0,
                        allocation_id: AllocationId::new(1),
                        offset: 0,
                        length: 4,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(0x1122_3344_u32.to_le_bytes().to_vec()),
                    },
                    BufferView {
                        view_id: ViewId::new(2),
                        metal_binding: 1,
                        allocation_id: AllocationId::new(2),
                        offset: 0,
                        length: 4,
                        access: BufferAccess::Write,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(vec![0; 4]),
                    },
                ],
                dispatch: Dispatch {
                    kind: DispatchKind::ThreadsExact,
                    grid: [1, 1, 1],
                    threads_per_threadgroup: [1, 1, 1],
                },
            }],
            completion_policy: CompletionPolicy::HostReadback,
        };
        let resources = || {
            let mut snapshot = ResourceTableSnapshot::new();
            for allocation in 1..=2 {
                snapshot
                    .insert_allocation(AllocationRecord {
                        allocation_id: AllocationId::new(allocation),
                        owner_epoch: provider.device_epoch(),
                        size: 4,
                    })
                    .unwrap();
            }
            snapshot
        };
        let first = provider
            .submit(
                provider
                    .capabilities()
                    .validate_trace(make_trace(), resources())
                    .unwrap(),
            )
            .unwrap();
        let first_token = first.completion.token().unwrap();
        let error = provider.wait(first_token, Duration::ZERO).unwrap_err();
        assert_eq!(error.slug, "metal_completion_unknown");
        assert_eq!(provider.health(), ProviderHealth::Usable);
        let (abandoned, bytes) = provider.abandonment_stats();
        assert_eq!(abandoned, 1);
        assert!(bytes > 0);
        // The completion handler retains and releases the timed-out submission;
        // an observation deadline must not permanently disable the context.
        let second = provider
            .submit(
                provider
                    .capabilities()
                    .validate_trace(make_trace(), resources())
                    .unwrap(),
            )
            .expect("provider must accept work after a deadline observation");
        let second_token = second.completion.token().unwrap();
        assert!(matches!(
            second.completion,
            CompletionDisposition::Submitted { .. }
        ));
        provider.release_completion(first_token).unwrap();
        provider.release_completion(second_token).unwrap();
        provider.release_pipeline(&pipeline).unwrap();

        let observed = sink
            .messages
            .lock()
            .expect("recording completion sink")
            .iter()
            .filter_map(|message| match message {
                CompletionMessage::Token(update) => {
                    Some((update.token, update.sequence.get(), update.update.clone()))
                }
                CompletionMessage::Device(_) => None,
            })
            .collect::<Vec<_>>();
        assert!(observed.len() >= 3, "outbox stream: {observed:?}");
        assert_eq!(observed[0], (first_token, 1, CompletionUpdate::Submitted));
        assert_eq!(
            observed[1],
            (first_token, 2, CompletionUpdate::SubmittedUnknown)
        );
        assert_eq!(observed[2], (second_token, 1, CompletionUpdate::Submitted));
        // The completion handler may still publish the second token's terminal
        // transition after the test releases the slot.
        assert!(observed[3..].iter().all(|(token, sequence, update)| {
            *token == second_token && *sequence == 2 && update.is_terminal()
        }));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn staged_lease_import_copies_and_retires_a_window() {
        use metal_api_core::provider::{
            AllocationId, AllocationRecord, BufferLease, BufferSource, BufferView,
            CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace, Dispatch, DispatchType,
            LeaseId, LeaseImporter, LeaseLedger, LeaseObservation, LeaseReservation, OperationId,
            PipelineProvider, ResourceTableSnapshot, StagedLease, ViewId, PROVIDER_SCHEMA_VERSION,
        };

        let Ok(provider) = NativeMetalProvider::new() else {
            eprintln!("skipping native staged lease test: no eligible Metal device");
            return;
        };
        let pipeline = provider
            .compile(request("copy_word", COPY))
            .expect("reviewed copy_word fixture compiles");
        let lease_id = LeaseId::new(97);
        let reservation = LeaseReservation {
            lease: BufferLease {
                lease_id,
                allocation_id: AllocationId::new(197),
                owner_epoch: provider.device_epoch(),
            },
            offset: 0,
            length: 8,
        };
        let mut input = 0x6745_2301_u32.to_le_bytes().to_vec();
        input.extend_from_slice(&[0_u8; 4]);
        provider
            .import_staged_lease(StagedLease::new(reservation, input).unwrap())
            .unwrap();

        let trace = ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: provider.device_epoch(),
            operation_id: OperationId::new(97),
            pipelines: vec![pipeline.clone()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![ComputePass {
                pipeline: pipeline.pipeline_id,
                buffers: vec![
                    BufferView {
                        view_id: ViewId::new(297),
                        metal_binding: 0,
                        allocation_id: AllocationId::new(197),
                        offset: 0,
                        length: 4,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source: BufferSource::StagedLease(lease_id),
                    },
                    BufferView {
                        view_id: ViewId::new(298),
                        metal_binding: 1,
                        allocation_id: AllocationId::new(198),
                        offset: 0,
                        length: 4,
                        access: BufferAccess::Write,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(vec![0xab; 4]),
                    },
                ],
                dispatch: Dispatch {
                    kind: DispatchKind::ThreadsExact,
                    grid: [1, 1, 1],
                    threads_per_threadgroup: [1, 1, 1],
                },
            }],
            completion_policy: CompletionPolicy::HostReadback,
        };
        let resources = || {
            let mut snapshot = ResourceTableSnapshot::new();
            snapshot
                .insert_allocation(AllocationRecord {
                    allocation_id: AllocationId::new(197),
                    owner_epoch: provider.device_epoch(),
                    size: 16,
                })
                .unwrap();
            snapshot
                .insert_allocation(AllocationRecord {
                    allocation_id: AllocationId::new(198),
                    owner_epoch: provider.device_epoch(),
                    size: 32,
                })
                .unwrap();
            snapshot.insert_lease(reservation).unwrap();
            snapshot
        };

        let submission = provider
            .submit(
                provider
                    .capabilities()
                    .validate_trace(trace.clone(), resources())
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(submission.writebacks.len(), 1);
        assert_eq!(
            submission.writebacks[0].bytes,
            0x6745_2301_u32.to_le_bytes()
        );
        let token = submission.completion.token().unwrap();
        let mut ledger = LeaseLedger::new();
        ledger.register(reservation).unwrap();
        ledger.bind(lease_id, token).unwrap();
        assert_eq!(
            ledger.observe(token, submission.completion).unwrap(),
            LeaseObservation::Retired
        );
        assert!(ledger.release_ready(lease_id));
        provider.release_staged_lease(lease_id).unwrap();
        assert!(provider.lease_registry().is_empty());

        let error = provider
            .submit(
                provider
                    .capabilities()
                    .validate_trace(trace.clone(), resources())
                    .unwrap(),
            )
            .unwrap_err();
        assert_eq!(error.slug, "lease_not_imported");
        provider.release_completion(token).unwrap();
        provider.release_pipeline(&pipeline).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn borrowed_lease_imports_without_copying_and_retires() {
        use metal_api_core::provider::{
            AllocationId, AllocationRecord, BorrowedLease, BufferLease, BufferSource, BufferView,
            CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace, Dispatch, DispatchType,
            LeaseId, LeaseLedger, LeaseObservation, LeaseReservation, NoCopyLeaseImporter,
            OperationId, PipelineProvider, ResourceTableSnapshot, ViewId, PROVIDER_SCHEMA_VERSION,
        };

        let Ok(provider) = NativeMetalProvider::new() else {
            eprintln!("skipping native borrowed lease test: no eligible Metal device");
            return;
        };
        let pipeline = provider
            .compile(request("copy_word", COPY))
            .expect("reviewed copy_word fixture compiles");
        let alignment = provider.no_copy_alignment();
        assert!(
            alignment > 0,
            "native provider must advertise page alignment"
        );
        let lease_id = LeaseId::new(98);
        let reservation = LeaseReservation {
            lease: BufferLease {
                lease_id,
                allocation_id: AllocationId::new(197),
                owner_epoch: provider.device_epoch(),
            },
            offset: 0,
            // Metal maps whole no-copy reservations, so the owner window must
            // be a page multiple.
            length: alignment,
        };
        let layout =
            std::alloc::Layout::from_size_align(alignment as usize, alignment as usize).unwrap();
        // SAFETY: the layout is non-zero and the allocation is freed below.
        let owner = unsafe { std::alloc::alloc(layout) };
        assert!(!owner.is_null(), "page-aligned owner allocation failed");
        // SAFETY: `owner` covers one writable page.
        unsafe {
            std::ptr::write_bytes(owner, 0xcd, alignment as usize);
            std::ptr::copy_nonoverlapping(0xaaaa_aaaa_u32.to_le_bytes().as_ptr(), owner, 4);
        }

        // SAFETY: the owner allocation outlives the import and is released
        // after both submissions retire below.
        let error = unsafe {
            provider
                .import_borrowed_lease(BorrowedLease::new(reservation, owner as usize + 1).unwrap())
        }
        .expect_err("misaligned owner pointer must be refused");
        assert_eq!(error.slug, "lease_alignment_unsupported");
        // SAFETY: `owner` covers one writable page; the short reservation is
        // refused before the mapping is imported.
        let error = unsafe {
            provider.import_borrowed_lease(
                BorrowedLease::new(
                    LeaseReservation {
                        length: 8,
                        ..reservation
                    },
                    owner as usize,
                )
                .unwrap(),
            )
        }
        .expect_err("non-page-multiple reservation must be refused");
        assert_eq!(error.slug, "lease_length_unsupported");
        // SAFETY: as above.
        unsafe {
            provider
                .import_borrowed_lease(BorrowedLease::new(reservation, owner as usize).unwrap())
                .unwrap();
        }

        let resources = || {
            let mut snapshot = ResourceTableSnapshot::new();
            snapshot
                .insert_allocation(AllocationRecord {
                    allocation_id: AllocationId::new(197),
                    owner_epoch: provider.device_epoch(),
                    size: alignment,
                })
                .unwrap();
            snapshot
                .insert_allocation(AllocationRecord {
                    allocation_id: AllocationId::new(198),
                    owner_epoch: provider.device_epoch(),
                    size: 32,
                })
                .unwrap();
            snapshot.insert_lease(reservation).unwrap();
            snapshot
        };
        let trace = |lease_access: BufferAccess, owned: Vec<u8>, view: u64| {
            let owned_access = match lease_access {
                BufferAccess::Read => BufferAccess::Write,
                BufferAccess::Write => BufferAccess::Read,
                other => panic!("borrowed fixture cannot use {other:?}"),
            };
            let lease_view = |binding| BufferView {
                view_id: ViewId::new(view),
                metal_binding: binding,
                allocation_id: AllocationId::new(197),
                offset: 0,
                length: 4,
                access: lease_access,
                attribute_stride: None,
                source: BufferSource::BorrowedNoCopy(lease_id),
            };
            let owned_view = |binding| BufferView {
                view_id: ViewId::new(view + 1),
                metal_binding: binding,
                allocation_id: AllocationId::new(198),
                offset: 0,
                length: 4,
                access: owned_access,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(owned.clone()),
            };
            let buffers = match lease_access {
                BufferAccess::Read => vec![lease_view(0), owned_view(1)],
                BufferAccess::Write => vec![owned_view(0), lease_view(1)],
                other => panic!("borrowed fixture cannot use {other:?}"),
            };
            ComputeTrace {
                schema_version: PROVIDER_SCHEMA_VERSION,
                device_epoch: provider.device_epoch(),
                operation_id: OperationId::new(view),
                pipelines: vec![pipeline.clone()],
                encoder_dispatch_type: DispatchType::Serial,
                passes: vec![ComputePass {
                    pipeline: pipeline.pipeline_id,
                    buffers,
                    dispatch: Dispatch {
                        kind: DispatchKind::ThreadsExact,
                        grid: [1, 1, 1],
                        threads_per_threadgroup: [1, 1, 1],
                    },
                }],
                completion_policy: CompletionPolicy::HostReadback,
            }
        };

        // The owner changes its mapping after import; a provider that had
        // snapshotted the bytes would observe the old word.
        // SAFETY: `owner` covers one writable page.
        unsafe {
            std::ptr::copy_nonoverlapping(0xbbbb_bbbb_u32.to_le_bytes().as_ptr(), owner, 4);
        }
        let read_trace = trace(BufferAccess::Read, vec![0; 4], 398);
        let read = provider
            .submit(
                provider
                    .capabilities()
                    .validate_trace(read_trace.clone(), resources())
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(read.writebacks.len(), 1);
        assert_eq!(read.writebacks[0].bytes, 0xbbbb_bbbb_u32.to_le_bytes());
        let read_token = read.completion.token().unwrap();

        let word = 0x1234_5678_u32;
        let write_trace = trace(BufferAccess::Write, word.to_le_bytes().to_vec(), 400);
        let write = provider
            .submit(
                provider
                    .capabilities()
                    .validate_trace(write_trace.clone(), resources())
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(write.writebacks.len(), 1);
        assert_eq!(write.writebacks[0].bytes, word.to_le_bytes());
        // SAFETY: `owner` covers one initialized page.
        let observed = unsafe { std::slice::from_raw_parts(owner, 4) };
        assert_eq!(observed, word.to_le_bytes(), "GPU write was not in place");
        let write_token = write.completion.token().unwrap();

        let mut ledger = LeaseLedger::new();
        ledger.register(reservation).unwrap();
        for (token, disposition) in [
            (read_token, read.completion),
            (write_token, write.completion),
        ] {
            ledger.bind(lease_id, token).unwrap();
            assert_eq!(
                ledger.observe(token, disposition).unwrap(),
                LeaseObservation::Retired
            );
        }
        assert!(ledger.release_ready(lease_id));
        assert_eq!(provider.borrowed_registry().outstanding(lease_id), Some(0));
        provider.release_borrowed_lease(lease_id).unwrap();

        let error = provider
            .submit(
                provider
                    .capabilities()
                    .validate_trace(write_trace.clone(), resources())
                    .unwrap(),
            )
            .unwrap_err();
        assert_eq!(error.slug, "lease_not_imported");
        provider.release_completion(read_token).unwrap();
        provider.release_completion(write_token).unwrap();
        provider.release_pipeline(&pipeline).unwrap();
        // SAFETY: the layout matches the allocation above and no submission
        // retains the mapping anymore.
        unsafe { std::alloc::dealloc(owner, layout) };
    }
}
