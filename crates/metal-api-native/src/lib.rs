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
    use metal_api_core::provider::{DeviceEpoch, SemanticDigest, SubmissionId};

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
}
