//! End-to-end indirect dispatch rail (`research/docs/25` §6 Step 4): one
//! compute-only trace whose single dispatch is replayed from a CPU-encoded
//! `VkDispatchIndirectCommand` instead of `vkCmdDispatch`.
//!
//! The falsifiable claim is byte equality with a direct dispatch of the same
//! reviewed copy kernel: the same grid, the same local size, and the same
//! output bytes. The refusal half pins the shape rules fail-closed — a
//! threadgroups payload that disagrees with the planned group count, and a
//! trace whose compute passes give the command more than one target.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferAccess, BufferSource, BufferView,
    CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass, ComputeProvider,
    ComputeTrace, Dispatch, DispatchKind, DispatchType, IndirectCommandBufferDescriptor,
    IndirectCommandDescriptor, IndirectCommandKind, IndirectCommandPayload, IndirectCommandRange,
    OperationId, PipelineId, ProviderErrorClass, ResourceTableSnapshot, SemanticDigest, TracePass,
    ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;

const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

const INPUT_VIEW: ViewId = ViewId::new(1001);
const INPUT_ALLOCATION: AllocationId = AllocationId::new(1010);
const OUTPUT_VIEW: ViewId = ViewId::new(1002);
const OUTPUT_ALLOCATION: AllocationId = AllocationId::new(1011);
const OUTPUT2_VIEW: ViewId = ViewId::new(1003);
const OUTPUT2_ALLOCATION: AllocationId = AllocationId::new(1012);

const INPUT_WORD: [u8; 4] = 0x6745_2301u32.to_le_bytes();

fn executor() -> Option<Arc<VulkanExecutor>> {
    match VulkanExecutor::new() {
        Ok(executor) => Some(executor),
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            None
        }
    }
}

fn pipeline(provider: &VulkanComputeProvider, device: &Device) -> CompiledComputePipeline {
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("fixture library loads")
        .function("copy_word")
        .expect("fixture entry exists");
    provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", b"indirect_dispatch_e2e".to_vec())
                .expect("digest"),
        )
        .expect("compute pipeline registers")
}

fn compute_pass(pipeline: PipelineId, output: (ViewId, AllocationId)) -> ComputePass {
    ComputePass {
        pipeline,
        buffers: vec![
            BufferView {
                view_id: INPUT_VIEW,
                metal_binding: 0,
                allocation_id: INPUT_ALLOCATION,
                offset: 0,
                length: 4,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(INPUT_WORD.to_vec()),
            },
            BufferView {
                view_id: output.0,
                metal_binding: 1,
                allocation_id: output.1,
                offset: 0,
                length: 4,
                access: BufferAccess::Write,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![0xab; 4]),
            },
        ],
        textures: Vec::new(),
        dispatch: Dispatch {
            kind: DispatchKind::ThreadsExact,
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
        },
    }
}

fn indirect_dispatch(threadgroups: [u32; 3]) -> IndirectCommandPayload {
    IndirectCommandPayload {
        buffer: IndirectCommandBufferDescriptor {
            max_commands: 1,
            kinds: vec![IndirectCommandKind::Dispatch],
        },
        command: IndirectCommandDescriptor::Dispatch { threadgroups },
        range: IndirectCommandRange { start: 0, count: 1 },
    }
}

fn trace(
    provider: &VulkanComputeProvider,
    pipeline: &CompiledComputePipeline,
    passes: Vec<ComputePass>,
    indirect: Option<IndirectCommandPayload>,
) -> ComputeTrace {
    ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(41),
        pipelines: vec![pipeline.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: passes.into_iter().map(TracePass::Compute).collect(),
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: indirect.map(Box::new),
    }
}

fn resources(provider: &VulkanComputeProvider) -> ResourceTableSnapshot {
    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [
        (INPUT_ALLOCATION, 4),
        (OUTPUT_ALLOCATION, 4),
        (OUTPUT2_ALLOCATION, 4),
    ] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("indirect dispatch fixture allocation");
    }
    resources
}

fn output_bytes(
    submission: &metal_api_core::provider::ProviderSubmission,
    view: ViewId,
) -> Vec<u8> {
    let writeback = submission
        .writebacks
        .iter()
        .find(|writeback| writeback.view_id == view)
        .unwrap_or_else(|| panic!("view {view:?} has no writeback"));
    writeback.bytes.clone()
}

#[test]
fn an_indirect_dispatch_replays_the_same_output_bytes_as_a_direct_dispatch() {
    let Some(executor) = executor() else {
        return;
    };
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider");
    let device = Device::new(executor as Arc<dyn ComputeExecutor>);
    let pipeline = pipeline(&provider, &device);

    let direct = trace(
        &provider,
        &pipeline,
        vec![compute_pass(
            pipeline.pipeline_id,
            (OUTPUT_VIEW, OUTPUT_ALLOCATION),
        )],
        None,
    );
    let direct_admitted = provider
        .capabilities()
        .validate_trace(direct.clone(), resources(&provider))
        .expect("the direct trace is admitted");
    let direct_result = provider
        .submit(direct_admitted)
        .expect("the direct dispatch completes");
    assert!(matches!(
        direct_result.completion,
        CompletionDisposition::CompletedVisible { .. }
    ));
    let direct_bytes = output_bytes(&direct_result, OUTPUT_VIEW);
    assert_eq!(direct_bytes, INPUT_WORD);

    let indirect = trace(
        &provider,
        &pipeline,
        vec![compute_pass(
            pipeline.pipeline_id,
            (OUTPUT_VIEW, OUTPUT_ALLOCATION),
        )],
        Some(indirect_dispatch([1, 1, 1])),
    );
    let indirect_admitted = provider
        .capabilities()
        .validate_trace(indirect.clone(), resources(&provider))
        .expect("the dispatch payload is in the first increment's admitted set");
    let indirect_result = provider
        .submit(indirect_admitted)
        .expect("the indirect dispatch completes");
    let indirect_bytes = output_bytes(&indirect_result, OUTPUT_VIEW);
    eprintln!("indirect dispatch readback: {:02x?}", indirect_bytes);
    assert_eq!(indirect_bytes, INPUT_WORD);
    assert_eq!(indirect_bytes, direct_bytes);
}

#[test]
fn an_indirect_dispatch_whose_threadgroups_disagree_with_the_plan_is_refused() {
    let Some(executor) = executor() else {
        return;
    };
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider");
    let device = Device::new(executor as Arc<dyn ComputeExecutor>);
    let pipeline = pipeline(&provider, &device);

    // The kernel dispatches one workgroup, but the payload claims two: the
    // footprint proof would describe a different launch than the rail replays.
    let value = trace(
        &provider,
        &pipeline,
        vec![compute_pass(
            pipeline.pipeline_id,
            (OUTPUT_VIEW, OUTPUT_ALLOCATION),
        )],
        Some(indirect_dispatch([2, 1, 1])),
    );
    let admitted = provider
        .capabilities()
        .validate_trace(value.clone(), resources(&provider))
        .expect("the payload itself is admissible");
    let refused = provider
        .submit(admitted)
        .expect_err("a mismatched threadgroups payload must not replay");
    assert_eq!(refused.slug, "icb_command_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}

#[test]
fn an_indirect_dispatch_with_two_compute_passes_is_refused() {
    let Some(executor) = executor() else {
        return;
    };
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider");
    let device = Device::new(executor as Arc<dyn ComputeExecutor>);
    let pipeline = pipeline(&provider, &device);

    let value = trace(
        &provider,
        &pipeline,
        vec![
            compute_pass(pipeline.pipeline_id, (OUTPUT_VIEW, OUTPUT_ALLOCATION)),
            compute_pass(pipeline.pipeline_id, (OUTPUT2_VIEW, OUTPUT2_ALLOCATION)),
        ],
        Some(indirect_dispatch([1, 1, 1])),
    );
    let admitted = provider
        .capabilities()
        .validate_trace(value.clone(), resources(&provider))
        .expect("admission admits the two-pass shape");
    let refused = provider
        .submit(admitted)
        .expect_err("one indirect dispatch needs exactly one compute pass");
    assert_eq!(refused.slug, "icb_command_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}
