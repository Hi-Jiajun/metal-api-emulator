//! Adapter from the Metal-like API object model to reims-vgpu's persistent
//! Vulkan compute engine.
//!
//! The adapter is intentionally outside the product guest path. It lets the
//! same source-level application exercise either the standalone Vulkan
//! executor or the engine that reims uses for guest work, without changing
//! QEMU, packet routing, display, or backend selection.
//!
//! The engine is process-global and serialized. Its synchronous entry waits
//! for submitted work, but does not impose an end-to-end deadline on engine
//! lock acquisition, device initialization, or pipeline compilation.

use metal2vulkan::reflect::{KernelDispatch, ResourceAccess};
use metal_api_core::{
    BufferUpdate, ComputeExecutor, ComputeSubmission, ExecutorError, Function, PipelineArtifact,
};
use metal_api_vulkan::TranslatedComputePipeline;
use reims_vgpu::backend::vulkan::engine::{
    execute_compute_request_sync, ComputeBufferOutput, ComputeBufferResource, ComputeDispatch,
    ComputeRequest,
};
use reims_vgpu::model::{DeviceId, DeviceState};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

fn failure(message: impl Into<String>) -> ExecutorError {
    ExecutorError::new(message)
}

/// Source-level executor backed by reims-vgpu's process-global Vulkan engine.
pub struct ReimsVulkanExecutor {
    identity: Arc<()>,
    state: Arc<DeviceState>,
}

impl ReimsVulkanExecutor {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            identity: Arc::new(()),
            state: Arc::new(DeviceState::new(DeviceId(1), 12)),
        })
    }

    pub const fn device_name(&self) -> &'static str {
        "reims-vgpu persistent Vulkan engine"
    }
}

struct ReimsPipelineArtifact {
    executor_identity: Arc<()>,
    translated: TranslatedComputePipeline,
}

impl ComputeExecutor for ReimsVulkanExecutor {
    fn new_compute_pipeline(&self, function: &Function) -> Result<PipelineArtifact, ExecutorError> {
        Ok(Arc::new(ReimsPipelineArtifact {
            executor_identity: Arc::clone(&self.identity),
            translated: TranslatedComputePipeline::translate(function)?,
        }))
    }

    fn execute(&self, submission: ComputeSubmission) -> Result<Vec<BufferUpdate>, ExecutorError> {
        let artifact = Arc::downcast::<ReimsPipelineArtifact>(Arc::clone(&submission.pipeline))
            .map_err(|_| failure("pipeline artifact is not a reims Vulkan pipeline"))?;
        if !Arc::ptr_eq(&artifact.executor_identity, &self.identity) {
            return Err(failure(
                "pipeline artifact belongs to another reims Vulkan executor",
            ));
        }

        let grid = submission.threads_per_grid.dimensions();
        let local = submission.threads_per_threadgroup.dimensions();
        artifact
            .translated
            .validate_buffers(&submission.buffers, grid)?;
        artifact.translated.validate_threadgroup(local)?;
        let reflection = artifact.translated.reflection();
        let contract = reflection
            .kernel_dispatch
            .ok_or_else(|| failure("translated kernel has no dispatch contract"))?;
        if !matches!(contract, KernelDispatch::ThreadsDynamic { .. }) {
            return Err(failure(format!(
                "translated kernel returned unexpected dispatch contract {contract:?}"
            )));
        }
        let dispatch = ComputeDispatch::exact_threads(contract, local, grid)
            .map_err(|error| failure(format!("plan exact dispatch: {error}")))?;

        let supplied = submission
            .buffers
            .iter()
            .map(|binding| (binding.index, binding))
            .collect::<BTreeMap<_, _>>();
        let mut descriptor_contracts = BTreeMap::new();
        let mut expected_writable = BTreeSet::new();
        let mut storage_buffers = Vec::with_capacity(reflection.bindings.len());
        for reflected in &reflection.bindings {
            let descriptor = reflected.descriptor.ok_or_else(|| {
                failure(format!(
                    "Metal buffer {} has no Vulkan descriptor",
                    reflected.metal_index
                ))
            })?;
            let binding = supplied
                .get(&reflected.metal_index)
                .expect("shared validation requires every reflected buffer");
            let writable = !matches!(
                reflected.access,
                Some(ResourceAccess::Unused | ResourceAccess::ReadOnly)
            );
            if writable {
                expected_writable.insert(reflected.metal_index);
            }
            if descriptor_contracts
                .insert(
                    descriptor.binding,
                    (reflected.metal_index, writable, binding.bytes.len()),
                )
                .is_some()
            {
                return Err(failure(format!(
                    "duplicate Vulkan descriptor binding {}",
                    descriptor.binding
                )));
            }
            storage_buffers.push(ComputeBufferResource {
                binding: descriptor.binding,
                bytes: binding.bytes.clone(),
                writable,
            });
        }

        let request = ComputeRequest {
            spirv: spirv_words(artifact.translated.spirv())?,
            entry: "main".to_string(),
            dispatch,
            storage_buffers,
            ..ComputeRequest::default()
        };
        let output = execute_compute_request_sync(&self.state, &request)
            .map_err(|error| failure(format!("reims Vulkan compute: {error}")))?;
        if !output.images.is_empty() {
            return Err(failure(
                "reims Vulkan compute returned images for a buffer-only request",
            ));
        }

        map_buffer_updates(output.buffers, &descriptor_contracts, &expected_writable)
    }
}

type BufferOutputContract = (u32, bool, usize);

fn map_buffer_updates(
    buffers: Vec<ComputeBufferOutput>,
    descriptor_contracts: &BTreeMap<u32, BufferOutputContract>,
    expected_writable: &BTreeSet<u32>,
) -> Result<Vec<BufferUpdate>, ExecutorError> {
    let mut seen = BTreeSet::new();
    let mut updates = Vec::with_capacity(buffers.len());
    for buffer in buffers {
        let &(metal_index, writable, expected_len) =
            descriptor_contracts.get(&buffer.binding).ok_or_else(|| {
                failure(format!(
                    "reims Vulkan compute returned unknown descriptor binding {}",
                    buffer.binding
                ))
            })?;
        if !writable {
            return Err(failure(format!(
                "reims Vulkan compute returned read-only Metal buffer {metal_index}"
            )));
        }
        if !seen.insert(metal_index) {
            return Err(failure(format!(
                "reims Vulkan compute returned Metal buffer {metal_index} more than once"
            )));
        }
        if buffer.bytes.len() != expected_len {
            return Err(failure(format!(
                "reims Vulkan compute returned Metal buffer {metal_index} length {}, expected {expected_len}",
                buffer.bytes.len()
            )));
        }
        updates.push(BufferUpdate {
            index: metal_index,
            offset: 0,
            bytes: buffer.bytes,
        });
    }
    if &seen != expected_writable {
        return Err(failure(format!(
            "reims Vulkan compute returned writable Metal buffers {seen:?}, expected {expected_writable:?}"
        )));
    }
    updates.sort_by_key(|update| update.index);
    Ok(updates)
}

fn spirv_words(bytes: &[u8]) -> Result<Vec<u32>, ExecutorError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(failure("translated SPIR-V is not word aligned"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte chunk")))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spirv_words_preserve_little_endian_encoding() {
        assert_eq!(
            spirv_words(&[3, 2, 35, 7, 1, 0, 0, 0]).unwrap(),
            [0x0723_0203, 1]
        );
        assert!(spirv_words(&[0, 1, 2]).is_err());
    }

    fn output_contracts() -> (BTreeMap<u32, BufferOutputContract>, BTreeSet<u32>) {
        (
            BTreeMap::from([(7, (0, true, 4)), (8, (1, false, 4))]),
            BTreeSet::from([0]),
        )
    }

    #[test]
    fn output_mapping_requires_every_writable_buffer_at_its_exact_length() {
        let (contracts, expected) = output_contracts();
        let updates = map_buffer_updates(
            vec![ComputeBufferOutput {
                binding: 7,
                bytes: vec![1, 2, 3, 4],
            }],
            &contracts,
            &expected,
        )
        .unwrap();
        assert_eq!(updates[0].index, 0);
        assert_eq!(updates[0].bytes, [1, 2, 3, 4]);

        let missing = map_buffer_updates(Vec::new(), &contracts, &expected).unwrap_err();
        assert!(missing.message().contains("expected {0}"));

        let short = map_buffer_updates(
            vec![ComputeBufferOutput {
                binding: 7,
                bytes: vec![1, 2, 3],
            }],
            &contracts,
            &expected,
        )
        .unwrap_err();
        assert!(short.message().contains("length 3, expected 4"));
    }

    #[test]
    fn output_mapping_rejects_read_only_and_unknown_bindings() {
        let (contracts, expected) = output_contracts();
        let read_only = map_buffer_updates(
            vec![ComputeBufferOutput {
                binding: 8,
                bytes: vec![0; 4],
            }],
            &contracts,
            &expected,
        )
        .unwrap_err();
        assert!(read_only.message().contains("read-only Metal buffer 1"));

        let unknown = map_buffer_updates(
            vec![ComputeBufferOutput {
                binding: 9,
                bytes: vec![0; 4],
            }],
            &contracts,
            &expected,
        )
        .unwrap_err();
        assert!(unknown.message().contains("unknown descriptor binding 9"));
    }
}
