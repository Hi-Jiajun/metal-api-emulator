//! Vulkan-to-neutral provider contract mapping.
//!
//! This module translates reflection and selected-device limits into the pure
//! values owned by `metal-api-core::provider`. It deliberately does not expose
//! Vulkan descriptors, region plans, handles, or guest memory pointers.

use ash::vk;
use metal2vulkan::reflect::{
    BufferFootprint, KernelDispatch, ResourceAccess, ResourceKind, ShaderReflection,
};
use metal_api_core::provider::{
    AliasMode, BufferAccess, BufferBindingContract, DispatchKind, FootprintProof, PipelineContract,
    ProviderCapabilities, SemanticDigest, StorageMode,
};
use metal_api_core::ExecutorError;

fn failure(message: impl Into<String>) -> ExecutorError {
    ExecutorError::new(message)
}

pub(crate) fn capabilities_from_limits(limits: &vk::PhysicalDeviceLimits) -> ProviderCapabilities {
    ProviderCapabilities {
        supports_threads_exact: true,
        supports_threadgroups: false,
        supports_serial: true,
        supports_concurrent: false,
        max_local_size: limits.max_compute_work_group_size.map(u64::from),
        max_invocations: u64::from(limits.max_compute_work_group_invocations),
        max_group_count: limits.max_compute_work_group_count.map(u64::from),
        max_storage_buffer_descriptors: limits
            .max_per_stage_descriptor_storage_buffers
            .min(limits.max_descriptor_set_storage_buffers)
            .min(limits.max_per_stage_resources),
        max_buffer_range: u64::from(limits.max_storage_buffer_range),
        alias_mode: AliasMode::Refused,
        storage_modes: vec![StorageMode::OwnedBytes],
        host_readback: true,
        submit_only: false,
    }
}

pub(crate) fn pipeline_contract(
    reflection: &ShaderReflection,
    translator_revision: Option<SemanticDigest>,
) -> Result<PipelineContract, ExecutorError> {
    let dispatch_kind = match reflection.kernel_dispatch {
        Some(KernelDispatch::ThreadsDynamic { .. } | KernelDispatch::ThreadsFixed { .. }) => {
            DispatchKind::ThreadsExact
        }
        Some(KernelDispatch::Workgroups) => DispatchKind::Threadgroups,
        None => return Err(failure("provider contract has no kernel dispatch kind")),
    };
    let required_local_size = reflection
        .local_size
        .ok_or_else(|| failure("provider contract has no reflected local size"))?
        .map(u64::from);
    let push_constant_bytes = reflection
        .kernel_dispatch
        .and_then(KernelDispatch::push_constant_range)
        .map_or(0, |range| range.size);

    let mut buffer_bindings = Vec::with_capacity(reflection.bindings.len());
    for binding in &reflection.bindings {
        if binding.kind != ResourceKind::Buffer {
            return Err(failure(format!(
                "provider contract only maps Metal buffers, found {:?} at {}",
                binding.kind, binding.metal_index
            )));
        }
        let access = map_access(binding.access, binding.metal_index)?;
        let footprint = binding
            .footprint
            .as_ref()
            .ok_or_else(|| failure(format!("buffer {} has no footprint", binding.metal_index)))?;
        buffer_bindings.push(BufferBindingContract {
            metal_binding: binding.metal_index,
            access,
            footprint: map_footprint(footprint, binding.metal_index)?,
        });
    }

    let contract = PipelineContract {
        dispatch_kind,
        required_local_size,
        push_constant_bytes,
        buffer_bindings,
        // Capability names are provider admission metadata. A normalized
        // cross-provider vocabulary is intentionally still an open decision.
        shader_capabilities: Vec::new(),
        translator_revision,
    };
    contract
        .validate()
        .map_err(|error| failure(format!("provider pipeline contract: {error}")))?;
    Ok(contract)
}

fn map_access(access: Option<ResourceAccess>, index: u32) -> Result<BufferAccess, ExecutorError> {
    match access {
        Some(ResourceAccess::Unused) => Ok(BufferAccess::Unused),
        Some(ResourceAccess::ReadOnly) => Ok(BufferAccess::Read),
        Some(ResourceAccess::WriteOnly) => Ok(BufferAccess::Write),
        Some(ResourceAccess::ReadWrite) => Ok(BufferAccess::ReadWrite),
        Some(other) => Err(failure(format!(
            "buffer {index} has non-buffer access classification {other:?}"
        ))),
        None => Err(failure(format!(
            "buffer {index} has no access classification"
        ))),
    }
}

fn map_footprint(footprint: &BufferFootprint, index: u32) -> Result<FootprintProof, ExecutorError> {
    if footprint.has_unbounded_access {
        return Ok(FootprintProof::Unbounded);
    }
    if !footprint.strided_accesses.is_empty() {
        return Ok(FootprintProof::Affine);
    }
    let mut max_bytes = 0_u64;
    for range in &footprint.static_ranges {
        let end = range
            .offset
            .checked_add(range.size)
            .ok_or_else(|| failure(format!("buffer {index} footprint overflows u64")))?;
        max_bytes = max_bytes.max(end);
    }
    Ok(FootprintProof::Static { max_bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::reflect::{
        BufferByteRange, BufferIndexSource, BufferStrideTerm, BufferStridedAccess,
    };

    #[test]
    fn footprint_mapping_preserves_bounded_and_unbounded_states() {
        let bounded = BufferFootprint {
            static_ranges: vec![BufferByteRange { offset: 4, size: 8 }],
            strided_accesses: Vec::new(),
            has_unbounded_access: false,
        };
        assert_eq!(
            map_footprint(&bounded, 0).unwrap(),
            FootprintProof::Static { max_bytes: 12 }
        );

        let affine = BufferFootprint {
            static_ranges: Vec::new(),
            strided_accesses: vec![BufferStridedAccess {
                base_offset: 0,
                access_size: 4,
                terms: vec![BufferStrideTerm {
                    source: BufferIndexSource::GlobalInvocationIdX,
                    stride: 4,
                }],
            }],
            has_unbounded_access: false,
        };
        assert_eq!(map_footprint(&affine, 0).unwrap(), FootprintProof::Affine);

        let unbounded = BufferFootprint {
            has_unbounded_access: true,
            ..BufferFootprint::default()
        };
        assert_eq!(
            map_footprint(&unbounded, 0).unwrap(),
            FootprintProof::Unbounded
        );
    }

    #[test]
    fn access_mapping_rejects_missing_or_non_buffer_classifications() {
        assert_eq!(
            map_access(Some(ResourceAccess::ReadOnly), 3).unwrap(),
            BufferAccess::Read
        );
        assert!(map_access(None, 3).is_err());
        assert!(map_access(Some(ResourceAccess::Sampled), 3).is_err());
    }
}
