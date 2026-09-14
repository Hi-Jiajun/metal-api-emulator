//! End-to-end heap placement rail (`research/docs/25` §6 Step 3): one trace
//! whose two owned buffers are placed at different offsets inside one heap
//! slab, submitted through the provider's admission and execution path.
//!
//! The case proves three things at once: the copy kernel's output bytes are
//! correct after both buffers share one `VkDeviceMemory` slab (rather than
//! each owning its own), the placement observations report the same `heap_id`
//! at the two different offsets, and a placement whose byte size disagrees
//! with its allocation is refused with `heap_placement_mismatch` instead of
//! being silently narrowed.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferAccess, BufferSource, BufferView,
    CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass, ComputeProvider,
    ComputeTrace, Dispatch, DispatchKind, DispatchType, HeapDescriptor, HeapId, HeapPayload,
    HeapPlacement, HeapResource, OperationId, ResourceTableSnapshot, SemanticDigest, StorageMode,
    TracePass, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;

const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

const INPUT_VIEW: ViewId = ViewId::new(901);
const INPUT_ALLOCATION: AllocationId = AllocationId::new(910);
const OUTPUT_VIEW: ViewId = ViewId::new(902);
const OUTPUT_ALLOCATION: AllocationId = AllocationId::new(911);
const HEAP: HeapId = HeapId::new(61);

fn executor() -> Option<Arc<VulkanExecutor>> {
    match VulkanExecutor::new() {
        Ok(executor) => Some(executor),
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            None
        }
    }
}

fn heap_payload(placements: Vec<HeapPlacement>) -> HeapPayload {
    HeapPayload {
        descriptor: HeapDescriptor {
            size: 4096,
            storage_mode: StorageMode::OwnedBytes,
            allows_aliasing: false,
        },
        placements,
    }
}

fn placement(offset: u64, byte_size: u64) -> HeapPlacement {
    HeapPlacement {
        heap_id: HEAP,
        offset,
        resource: HeapResource::Buffer { byte_size },
    }
}

fn trace(
    provider: &VulkanComputeProvider,
    pipeline: &CompiledComputePipeline,
    heap: HeapPayload,
) -> ComputeTrace {
    ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(31),
        pipelines: vec![pipeline.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![TracePass::Compute(ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers: vec![
                BufferView {
                    view_id: INPUT_VIEW,
                    metal_binding: 0,
                    allocation_id: INPUT_ALLOCATION,
                    offset: 0,
                    length: 4,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(0x6745_2301u32.to_le_bytes().to_vec()),
                },
                BufferView {
                    view_id: OUTPUT_VIEW,
                    metal_binding: 1,
                    allocation_id: OUTPUT_ALLOCATION,
                    offset: 0,
                    length: 4,
                    access: BufferAccess::Write,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(0xabab_ababu32.to_le_bytes().to_vec()),
                },
            ],
            textures: Vec::new(),
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
        })],
        completion_policy: CompletionPolicy::HostReadback,
        heap: Some(Box::new(heap)),
        indirect: None,
    }
}

fn resources(provider: &VulkanComputeProvider) -> ResourceTableSnapshot {
    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [(INPUT_ALLOCATION, 4), (OUTPUT_ALLOCATION, 4)] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("heap fixture allocation");
    }
    resources
}

fn pipeline(
    provider: &VulkanComputeProvider,
    device: &Device,
    case: &[u8],
) -> CompiledComputePipeline {
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("fixture library loads")
        .function("copy_word")
        .expect("fixture entry exists");
    provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest"),
        )
        .expect("compute pipeline registers")
}

#[test]
fn two_buffers_land_in_one_slab_and_report_their_offsets() {
    let Some(executor) = executor() else {
        return;
    };
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider");
    let device = Device::new(executor as Arc<dyn ComputeExecutor>);
    let pipeline = pipeline(&provider, &device, b"heap_e2e");
    let value = trace(
        &provider,
        &pipeline,
        heap_payload(vec![placement(0, 4), placement(256, 4)]),
    );
    let admitted = provider
        .capabilities()
        .validate_trace(value.clone(), resources(&provider))
        .expect("heap trace admits");
    let result = provider.submit(admitted).expect("heap trace submits");
    let CompletionDisposition::CompletedVisible { token } = result.completion else {
        panic!("heap trace did not complete: {:?}", result.completion);
    };
    assert_eq!(
        provider
            .wait(token, std::time::Duration::ZERO)
            .expect("wait"),
        CompletionDisposition::CompletedVisible { token }
    );

    let writeback = result
        .writebacks
        .iter()
        .find(|writeback| writeback.view_id == OUTPUT_VIEW)
        .expect("output writeback");
    assert_eq!(writeback.allocation_id, OUTPUT_ALLOCATION);
    assert_eq!(writeback.offset, 0);
    assert_eq!(writeback.bytes, 0x6745_2301u32.to_le_bytes());

    let observations = provider.heap_placement_observations();
    assert_eq!(observations.len(), 2);
    assert_eq!(observations[0].heap_id, HEAP);
    assert_eq!(observations[0].allocation_id, INPUT_ALLOCATION);
    assert_eq!(observations[0].offset, 0);
    assert_eq!(observations[0].byte_size, 4);
    assert_eq!(observations[1].heap_id, HEAP);
    assert_eq!(observations[1].allocation_id, OUTPUT_ALLOCATION);
    assert_eq!(observations[1].offset, 256);
    assert_eq!(observations[1].byte_size, 4);
}

#[test]
fn a_placement_size_mismatch_is_refused() {
    let Some(executor) = executor() else {
        return;
    };
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider");
    let device = Device::new(executor as Arc<dyn ComputeExecutor>);
    let pipeline = pipeline(&provider, &device, b"heap_e2e_refusal");
    let value = trace(
        &provider,
        &pipeline,
        heap_payload(vec![placement(0, 8), placement(256, 8)]),
    );
    let admitted = provider
        .capabilities()
        .validate_trace(value, resources(&provider))
        .expect("structurally valid heap trace admits");
    match provider.submit(admitted) {
        Err(error)
            if error.slug == "heap_placement_mismatch"
                && error.completion == CompletionDisposition::NotSubmitted => {}
        other => panic!("size mismatch was not refused with heap_placement_mismatch: {other:?}"),
    }
}
