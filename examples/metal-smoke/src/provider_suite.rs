//! Live checks of canonical provider submission against the snapshot executor.

use super::{
    assemble_owned_air, execute_copy_word, execute_indexed_boundary_dispatch,
    indexed_boundary_golden, wrap_air_bitcode,
};
use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferSource, BufferView, CompletionDisposition,
    CompletionPolicy, CompletionToken, ComputePass, ComputeProvider, ComputeTrace, Dispatch,
    DispatchKind, DispatchType, FootprintProof, OperationId, PipelineCompileRequest,
    PipelineProvider, ProviderError, ProviderSubmission, ResourceTableSnapshot, SemanticDigest,
    ShaderSource, SubmissionId, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{Device, Library};
use metal_api_vulkan::{CompiledComputePipeline, VulkanComputeProvider, VulkanExecutor};
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;

/// Exercise provider admission, GPU execution, writeback identity, and completion.
/// Both paths share one Vulkan executor but compile and submit independently.
pub fn run_provider_suite(executor: Arc<VulkanExecutor>) -> Result<(), Box<dyn Error>> {
    println!("Metal API provider: standalone Vulkan");
    println!("Metal API Vulkan device: {}", executor.device_name());
    let provider =
        VulkanComputeProvider::with_executor(executor.clone()).map_err(provider_error)?;
    let peer = VulkanComputeProvider::with_executor(executor.clone()).map_err(provider_error)?;
    let device = Device::new(executor);
    let source = include_str!("../shaders/kernel_copy_word.ll");
    run_copy(
        &provider,
        &device,
        device.new_library_with_air(source)?,
        "textual",
        1,
    )?;
    let raw = assemble_owned_air(source)?;
    let wrapped = wrap_air_bitcode(&raw)?;
    for (index, (encoding, air)) in [("raw", raw), ("wrapped", wrapped)].into_iter().enumerate() {
        run_copy(
            &provider,
            &device,
            device.new_library_with_binary_air(air)?,
            encoding,
            index as u64 + 2,
        )?;
    }
    run_indexed_and_refusals(&provider, &peer, &device)?;
    let unknown_token = CompletionToken {
        submission_id: SubmissionId::new(u64::MAX),
        device_epoch: provider.device_epoch(),
    };
    expect_unknown_completion(provider.wait(unknown_token, Duration::ZERO), unknown_token)?;
    println!("PASS provider_refusal slug=unknown_completion");
    println!("PASS suite provider=standalone Vulkan snapshot_parity=4");
    Ok(())
}

fn provider_error(error: ProviderError) -> Box<dyn Error> {
    format!("provider failure: {error:?}").into()
}

fn run_copy(
    provider: &VulkanComputeProvider,
    device: &Device,
    library: Library,
    encoding: &str,
    operation: u64,
) -> Result<(), Box<dyn Error>> {
    let function = library.function("copy_word")?;
    let pipeline = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", b"copy_word".to_vec())?,
        )
        .map_err(provider_error)?;
    let trace = make_trace(
        &pipeline,
        operation,
        Dispatch {
            kind: DispatchKind::ThreadsExact,
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
        },
        vec![
            (0, 8, 0x6745_2301_u32.to_le_bytes().to_vec()),
            (1, 16, 0xabab_abab_u32.to_le_bytes().to_vec()),
        ],
    )?;
    let result = submit_and_wait(provider, &trace)?;
    let expected = 0x6745_2301_u32.to_le_bytes();
    check_writeback(&trace, &result, 1, &expected)?;
    let reference = execute_copy_word(device, library)?;
    if reference.to_le_bytes() != expected {
        return Err("copy_word snapshot executor disagrees with the provider golden".into());
    }
    release_case(provider, &pipeline, &result)?;
    println!(
        "PASS provider_copy_word encoding={encoding} output={reference:#010x} writeback_offset=16 snapshot_parity=exact"
    );
    Ok(())
}

fn run_indexed_and_refusals(
    provider: &VulkanComputeProvider,
    peer: &VulkanComputeProvider,
    device: &Device,
) -> Result<(), Box<dyn Error>> {
    let library = device.new_library_with_air(include_str!(
        "../shaders/kernel_dispatch_threads_boundary_barrier.ll"
    ))?;
    let function = library.function("kernel_dispatch_threads_boundary_barrier")?;
    let pipeline = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", b"indexed_boundary".to_vec())?,
        )
        .map_err(provider_error)?;
    let trace = make_trace(
        &pipeline,
        4,
        Dispatch {
            kind: DispatchKind::ThreadsExact,
            grid: [10, 3, 1],
            threads_per_threadgroup: [8, 2, 1],
        },
        vec![(0, 32, vec![0xaa; 30 * size_of::<u32>()])],
    )?;
    let result = submit_and_wait(provider, &trace)?;
    let expected = indexed_boundary_golden();
    check_writeback(&trace, &result, 0, &expected)?;
    if execute_indexed_boundary_dispatch(device)? != expected {
        return Err("indexed snapshot executor disagrees with the provider golden".into());
    }
    println!(
        "PASS provider_indexed_boundary_dispatch words=30 regions=4 writeback_offset=32 snapshot_parity=exact"
    );

    // Admission can verify an internally consistent proof but cannot establish
    // that it belongs to the compiled pipeline. The provider must check that.
    let mut forged = trace.clone();
    let FootprintProof::Affine { accesses } =
        &mut forged.pipelines[0].contract.buffer_bindings[0].footprint
    else {
        return Err("indexed provider fixture must carry an affine footprint proof".into());
    };
    for access in accesses {
        access.base_offset = 0;
        access.access_size = 1;
        access.terms.clear();
    }
    if forged.pipelines[0].contract == trace.pipelines[0].contract {
        return Err("forged fixture failed to change the pipeline contract".into());
    }
    let admitted = provider
        .capabilities()
        .validate_trace(forged.clone(), resources_for_trace(&forged)?)
        .map_err(provider_error)?;
    expect_refusal(provider.submit(admitted), "pipeline_contract_mismatch")?;
    println!("PASS provider_refusal slug=pipeline_contract_mismatch admitted_forgery=true");

    let admitted = peer
        .capabilities()
        .validate_trace(trace.clone(), resources_for_trace(&trace)?)
        .map_err(provider_error)?;
    expect_refusal(peer.submit(admitted), "device_epoch_mismatch")?;
    let token = result
        .completion
        .token()
        .ok_or("completed result has no token")?;
    expect_refusal(peer.wait(token, Duration::ZERO), "device_epoch_mismatch")?;
    println!("PASS provider_refusal slug=device_epoch_mismatch shared_executor=trace_and_token");
    // Compilation and release must carry owner identity through the shared API.
    expect_refusal(
        PipelineProvider::release_pipeline(peer, &pipeline),
        "device_epoch_mismatch",
    )?;
    let mut changed_pipeline = pipeline.clone();
    changed_pipeline.function.entry_name = "forged".into();
    expect_refusal(
        PipelineProvider::release_pipeline(provider, &changed_pipeline),
        "pipeline_identity_mismatch",
    )?;
    expect_refusal(
        PipelineProvider::compile(
            provider,
            PipelineCompileRequest {
                entry_name: "unsupported_msl".into(),
                logical_digest: SemanticDigest::new("fixture", vec![1])?,
                source: ShaderSource::MetalSource("kernel void unsupported_msl() {}".into()),
            },
        ),
        "shader_source_unsupported",
    )?;
    println!("PASS shared_compile_refusals foreign_release=checked metadata=checked msl=refused");

    release_case(provider, &pipeline, &result)?;
    expect_unknown_completion(provider.wait(token, Duration::ZERO), token)?;
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources_for_trace(&trace)?)
        .map_err(provider_error)?;
    expect_refusal(provider.submit(admitted), "unknown_pipeline")?;
    println!("PASS provider_release completion=unknown_completion pipeline=unknown_pipeline");
    Ok(())
}

fn make_trace(
    pipeline: &CompiledComputePipeline,
    operation: u64,
    dispatch: Dispatch,
    bindings: Vec<(u32, u64, Vec<u8>)>,
) -> Result<ComputeTrace, Box<dyn Error>> {
    let mut buffers = Vec::with_capacity(bindings.len());
    for (index, offset, bytes) in bindings {
        let access = pipeline
            .contract
            .buffer_bindings
            .iter()
            .find(|binding| binding.metal_binding == index)
            .ok_or("fixture binding is missing from pipeline reflection")?
            .access;
        buffers.push(BufferView {
            view_id: ViewId::new(200 + u64::from(index)),
            metal_binding: index,
            allocation_id: AllocationId::new(100 + u64::from(index)),
            offset,
            length: u64::try_from(bytes.len())?,
            access,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(bytes),
        });
    }
    Ok(ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: pipeline.device_epoch,
        operation_id: OperationId::new(operation),
        pipelines: vec![pipeline.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers,
            dispatch,
        }],
        completion_policy: CompletionPolicy::HostReadback,
    })
}

fn resources_for_trace(trace: &ComputeTrace) -> Result<ResourceTableSnapshot, Box<dyn Error>> {
    let mut resources = ResourceTableSnapshot::new();
    for view in &trace.passes[0].buffers {
        resources.insert_allocation(AllocationRecord {
            allocation_id: view.allocation_id,
            owner_epoch: trace.device_epoch,
            size: view.offset + view.length + 8,
        })?;
    }
    Ok(resources)
}

fn submit_and_wait(
    provider: &VulkanComputeProvider,
    trace: &ComputeTrace,
) -> Result<ProviderSubmission, Box<dyn Error>> {
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources_for_trace(trace)?)
        .map_err(provider_error)?;
    let result = provider.submit(admitted).map_err(provider_error)?;
    result.validate_for_trace(trace)?;
    let CompletionDisposition::CompletedVisible { token } = result.completion else {
        return Err(format!("host readback did not complete: {:?}", result.completion).into());
    };
    let waited = provider
        .wait(token, Duration::ZERO)
        .map_err(provider_error)?;
    if waited != (CompletionDisposition::CompletedVisible { token }) {
        return Err(format!("completed token did not stay visible: {waited:?}").into());
    }
    Ok(result)
}

fn check_writeback(
    trace: &ComputeTrace,
    result: &ProviderSubmission,
    binding: u32,
    expected: &[u8],
) -> Result<(), Box<dyn Error>> {
    let view = trace.passes[0]
        .buffers
        .iter()
        .find(|view| view.metal_binding == binding)
        .ok_or("output binding is missing from fixture")?;
    let [writeback] = result.writebacks.as_slice() else {
        return Err("provider fixture requires exactly one writable view".into());
    };
    if writeback.allocation_id != view.allocation_id
        || writeback.view_id != view.view_id
        || writeback.offset != view.offset
        || writeback.bytes != expected
    {
        return Err(
            format!("provider writeback did not match binding {binding}: {writeback:?}").into(),
        );
    }
    // Land the returned bytes using their allocation-relative offset and
    // compare the entire backing, including guards outside the view.
    let start = usize::try_from(view.offset)?;
    let end = start + expected.len();
    let mut actual_allocation = vec![0x5a; end + 8];
    let mut expected_allocation = actual_allocation.clone();
    expected_allocation[start..end].copy_from_slice(expected);
    let write_start = usize::try_from(writeback.offset)?;
    actual_allocation[write_start..write_start + writeback.bytes.len()]
        .copy_from_slice(&writeback.bytes);
    if actual_allocation != expected_allocation {
        return Err("provider writeback changed bytes outside its allocation view".into());
    }
    Ok(())
}

fn release_case(
    provider: &VulkanComputeProvider,
    pipeline: &CompiledComputePipeline,
    result: &ProviderSubmission,
) -> Result<(), Box<dyn Error>> {
    provider
        .release_completion(
            result
                .completion
                .token()
                .ok_or("completed result has no token")?,
        )
        .map_err(provider_error)?;
    provider
        .release_pipeline(pipeline)
        .map_err(provider_error)?;
    Ok(())
}

fn expect_unknown_completion(
    result: Result<CompletionDisposition, ProviderError>,
    token: CompletionToken,
) -> Result<(), Box<dyn Error>> {
    match result {
        Err(error)
            if error.slug == "unknown_completion"
                && error.completion
                    == (CompletionDisposition::SubmittedUnknown { token: Some(token) }) =>
        {
            Ok(())
        }
        other => {
            Err(format!("unknown token should preserve uncertain completion: {other:?}").into())
        }
    }
}

fn expect_refusal<T>(result: Result<T, ProviderError>, slug: &str) -> Result<(), Box<dyn Error>> {
    match result {
        Err(error)
            if error.slug == slug && error.completion == CompletionDisposition::NotSubmitted =>
        {
            Ok(())
        }
        Err(error) => Err(format!("expected pre-submit refusal {slug}, got {error:?}").into()),
        Ok(_) => Err(format!("provider accepted fixture requiring refusal {slug}").into()),
    }
}
