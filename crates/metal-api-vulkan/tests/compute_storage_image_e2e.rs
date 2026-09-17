//! End-to-end compute storage images (`research/docs/26` §21.4, C2): kernels
//! that write a `texture2d<float, access::write|read_write>` argument land their
//! texels in the trace's writeback channel, and a request that does not restate
//! the module's own declaration is refused by name before any device object
//! exists.
//!
//! The falsifiable half is the pair of write-only fixtures: they are
//! byte-identical except for one constant, and the landed bytes have to follow
//! the module that ran (`[1000.0 .. 1015.0]` against `[2000.0 .. 2015.0]` over
//! a 4x4 view). The read-modify-write fixture lands `[1.0 .. 16.0]` from an
//! initial `[0.0 .. 15.0]`, which is the reading a rail that skipped the upload
//! or bound a write-only descriptor cannot produce: it would read zeroes and
//! land a uniform one.
//!
//! The refusal half pins the three facts a storage declaration carries —
//! access, shape and reach. A declaration that names a sampled binding where the
//! module writes is refused by name (`compute_texture_access_unsupported`) with
//! both accesses; a view whose format disagrees with the module's is refused by
//! the provider's own format gate (`compute_texture_format_unsupported`) with
//! both formats; and a declaration whose reach cannot be stated
//! (`TextureFootprintProof::Unbounded`) is refused by core admission with
//! `compute_texture_footprint_unsupported` and the proof it carried.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferWriteback, CompiledComputePipeline,
    CompletionDisposition, CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace, Dispatch,
    DispatchKind, DispatchType, FieldValue, OperationId, ResourceTableSnapshot, SemanticDigest,
    TextureAccess, TextureBindingContract, TextureFootprintProof, TextureFormat, TextureSource,
    TextureType, TextureView, TracePass, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, ComputeSubmission, Device, Size};
use metal_api_vulkan::{VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;
use std::time::Duration;

const BIAS_1000_AIR: &str = include_str!("fixtures/write_texture_2d_bias_1000.ll");
const BIAS_2000_AIR: &str = include_str!("fixtures/write_texture_2d_bias_2000.ll");
const INCREMENT_AIR: &str = include_str!("fixtures/read_write_texture_2d_increment.ll");

const TEXTURE_VIEW: ViewId = ViewId::new(940);
const TEXTURE_ALLOCATION: AllocationId = AllocationId::new(941);

const TEXTURE_WIDTH: u64 = 4;
const TEXTURE_HEIGHT: u64 = 4;
const TEXTURE_BYTES: u64 = TEXTURE_WIDTH * TEXTURE_HEIGHT * 4;

fn executor() -> Option<Arc<VulkanExecutor>> {
    match VulkanExecutor::new() {
        Ok(executor) => Some(executor),
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            None
        }
    }
}

/// The initial contents of the storage image, one `f32` per texel.
fn texture_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn zeros() -> Vec<u8> {
    texture_bytes(&[0.0_f32; (TEXTURE_WIDTH * TEXTURE_HEIGHT) as usize])
}

fn increment_seed() -> Vec<u8> {
    texture_bytes(
        &(0..(TEXTURE_WIDTH * TEXTURE_HEIGHT) as u32)
            .map(|cell| cell as f32)
            .collect::<Vec<_>>(),
    )
}

fn texture_view(access: TextureAccess, bytes: Vec<u8>) -> TextureView {
    TextureView {
        view_id: TEXTURE_VIEW,
        metal_binding: 0,
        allocation_id: TEXTURE_ALLOCATION,
        texture_type: TextureType::D2,
        format: TextureFormat::R32Float,
        width: TEXTURE_WIDTH,
        height: TEXTURE_HEIGHT,
        depth: 1,
        array_length: 1,
        sample_count: 1,
        access,
        source: TextureSource::OwnedBytes(bytes),
    }
}

fn trace(provider: &VulkanComputeProvider, pipeline: &CompiledComputePipeline) -> ComputeTrace {
    ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(78),
        pipelines: vec![pipeline.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![TracePass::Compute(ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers: Vec::new(),
            textures: vec![texture_view(TextureAccess::Storage, zeros())],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [TEXTURE_WIDTH, TEXTURE_HEIGHT, 1],
                threads_per_threadgroup: [TEXTURE_WIDTH, TEXTURE_HEIGHT, 1],
            },
        })],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    }
}

fn resources(provider: &VulkanComputeProvider) -> ResourceTableSnapshot {
    let mut resources = ResourceTableSnapshot::new();
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: TEXTURE_ALLOCATION,
            owner_epoch: provider.device_epoch(),
            size: TEXTURE_BYTES,
        })
        .expect("storage image allocation");
    resources
}

/// Compile one storage-image fixture into a provider-owned pipeline.
fn compile(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
    source: &str,
) -> CompiledComputePipeline {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let library = device
        .new_library_with_air(source)
        .expect("fixture library loads");
    let entry = if source.contains("read_write_texture_2d") {
        "read_write_texture_2d"
    } else {
        "write_texture_2d"
    };
    let function = library.function(entry).expect("fixture entry exists");
    provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", source.as_bytes().to_vec())
                .expect("digest"),
        )
        .expect("the storage image fixture registers")
}

/// Run one fixture end to end and decode the landed texels as `f32`s.
fn landed_floats(
    provider: &VulkanComputeProvider,
    pipeline: &CompiledComputePipeline,
    initial: Vec<u8>,
    access: TextureAccess,
) -> (Vec<f32>, BufferWriteback) {
    let mut value = trace(provider, pipeline);
    let TracePass::Compute(pass) = &mut value.passes[0] else {
        unreachable!("one compute pass");
    };
    pass.textures = vec![texture_view(access, initial)];
    let admitted = provider
        .capabilities()
        .validate_trace(value, resources(provider))
        .expect("the storage image trace is structurally admissible");
    let submission = provider
        .submit(admitted)
        .expect("the storage image executes");
    let writeback = submission
        .writebacks
        .iter()
        .find(|writeback| writeback.view_id == TEXTURE_VIEW)
        .expect("the storage image lands a writeback")
        .clone();
    let floats = writeback
        .bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    (floats, writeback)
}

#[test]
fn a_storage_image_lands_the_bytes_its_module_writes() {
    let Some(executor) = executor() else {
        return;
    };
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider");
    let bias_1000 = compile(&provider, &executor, BIAS_1000_AIR);
    let bias_2000 = compile(&provider, &executor, BIAS_2000_AIR);
    let increment = compile(&provider, &executor, INCREMENT_AIR);

    // The declaration the rail derived is the module's own storage shape.
    for pipeline in [&bias_1000, &bias_2000, &increment] {
        assert_eq!(pipeline.contract.texture_bindings.len(), 1);
        assert_eq!(
            pipeline.contract.texture_bindings[0],
            TextureBindingContract::storage(0, TextureFormat::R32Float),
            "the registered declaration is the module's own storage image"
        );
    }

    let (first, writeback) = landed_floats(&provider, &bias_1000, zeros(), TextureAccess::Storage);
    let (second, _) = landed_floats(&provider, &bias_2000, zeros(), TextureAccess::Storage);
    let (incremented, _) = landed_floats(
        &provider,
        &increment,
        increment_seed(),
        TextureAccess::Storage,
    );

    // Raw readings, both halves, so the separation is inspectable.
    eprintln!("C2 write-only landing bias 1000: {first:?}");
    eprintln!("C2 write-only landing bias 2000: {second:?}");
    eprintln!("C2 read-modify-write landing:    {incremented:?}");

    let expected = |bias: f32| (0..16).map(|cell| cell as f32 + bias).collect::<Vec<_>>();
    assert_eq!(
        first,
        expected(1000.0),
        "the 1000-biased module lands 1000..1015"
    );
    assert_eq!(
        second,
        expected(2000.0),
        "the 2000-biased module lands 2000..2015"
    );
    assert_eq!(
        incremented,
        (1..=16).map(|cell| cell as f32).collect::<Vec<_>>(),
        "the read-modify-write module lands the uploaded texels plus one"
    );
    assert_ne!(first, second, "the two modules separate on this fixture");
    assert_eq!(
        writeback.bytes.len() as u64,
        TEXTURE_BYTES,
        "the landing is the view's whole tightly packed extent"
    );
    assert_eq!(writeback.offset, 0, "a storage image has no byte offset");
    assert_eq!(writeback.allocation_id, TEXTURE_ALLOCATION);
}

#[test]
fn a_request_that_names_another_texture_declaration_is_refused_by_name() {
    let Some(executor) = executor() else {
        return;
    };
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider");
    let registered = compile(&provider, &executor, BIAS_1000_AIR);

    // A request that declares a *sampled* binding where the module writes is
    // refused with both accesses before any descriptor exists.
    let mut sampled = registered.clone();
    sampled.contract.texture_bindings = vec![TextureBindingContract::sampled(
        0,
        TextureFormat::R32Float,
        metal_api_core::provider::SamplerPolicy::synthesized_read(),
    )];
    let mut value = trace(&provider, &sampled);
    let TracePass::Compute(pass) = &mut value.passes[0] else {
        unreachable!("one compute pass");
    };
    pass.textures = vec![texture_view(TextureAccess::Sampled, zeros())];
    let admitted = provider
        .capabilities()
        .validate_trace(value, resources(&provider))
        .expect("the sampled declaration is structurally admissible");
    let refusal = provider
        .submit(admitted)
        .expect_err("a sampled declaration against a storage module is refused");
    eprintln!("C2 storage access refusal: {refusal:?}");
    assert_eq!(refusal.slug, "compute_texture_access_unsupported");
    assert_eq!(
        refusal.fields.get("binding"),
        Some(&FieldValue::Unsigned(0))
    );
    assert_eq!(
        refusal.fields.get("declared_access"),
        Some(&FieldValue::Text("Sampled".into()))
    );
    assert_eq!(
        refusal.fields.get("module_access"),
        Some(&FieldValue::Text("Storage".into()))
    );

    // A request whose format disagrees with the module's is refused with both
    // formats.
    let mut foreign_format = registered.clone();
    foreign_format.contract.texture_bindings =
        vec![TextureBindingContract::storage(0, TextureFormat::R32Uint)];
    let mut value = trace(&provider, &foreign_format);
    let TracePass::Compute(pass) = &mut value.passes[0] else {
        unreachable!("one compute pass");
    };
    let mut view = texture_view(TextureAccess::Storage, zeros());
    view.format = TextureFormat::R32Uint;
    pass.textures = vec![view];
    let admitted = provider
        .capabilities()
        .validate_trace(value, resources(&provider))
        .expect("the foreign format declaration is structurally admissible");
    let refusal = provider
        .submit(admitted)
        .expect_err("a format the module does not carry is refused");
    eprintln!("C2 storage format refusal: {refusal:?}");
    assert_eq!(refusal.slug, "compute_texture_format_unsupported");
    assert_eq!(
        refusal.fields.get("declared_format"),
        Some(&FieldValue::Text("R32Uint".into()))
    );
    assert_eq!(
        refusal.fields.get("module_format"),
        Some(&FieldValue::Text("R32Float".into()))
    );

    // A declaration whose reach cannot be stated is refused by core admission
    // before a provider sees it.
    let mut unbounded = registered.clone();
    unbounded.contract.texture_bindings[0].footprint = TextureFootprintProof::Unbounded;
    let value = trace(&provider, &unbounded);
    let refusal = provider
        .capabilities()
        .validate_trace(value, resources(&provider))
        .expect_err("an unbounded storage reach is refused");
    eprintln!("C2 storage footprint refusal: {refusal:?}");
    assert_eq!(refusal.slug, "compute_texture_footprint_unsupported");
    assert_eq!(
        refusal.detail.as_deref(),
        Some("texture 0 states reach Unbounded, which the first increment cannot size"),
        "the core pair rule names the binding and the reach it carried"
    );

    // The sample-shaped request half: a *view* that declares a read-only access
    // against the module's storage declaration is the core pair rule, refused
    // with the binding and both accesses.
    let mut value = trace(&provider, &registered);
    let TracePass::Compute(pass) = &mut value.passes[0] else {
        unreachable!("one compute pass");
    };
    pass.textures = vec![texture_view(TextureAccess::Sampled, zeros())];
    let refusal = provider
        .capabilities()
        .validate_trace(value, resources(&provider))
        .expect_err("a read-only view against a storage declaration is refused");
    eprintln!("C2 storage view-access refusal: {refusal:?}");
    assert_eq!(refusal.slug, "trace_contract_invalid");
    assert!(
        refusal
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("Sampled") && detail.contains("Storage")),
        "the refusal detail has to name both accesses, got {:?}",
        refusal.detail
    );
}

#[test]
fn the_executor_api_refuses_a_storage_image_landing_by_name() {
    // The standalone executor returns buffer-keyed updates, so a storage image
    // has no channel there; the refusal is the boundary's, not the device's,
    // and it happens before any device object exists.
    let Some(executor) = executor() else {
        return;
    };
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let library = device
        .new_library_with_air(BIAS_1000_AIR)
        .expect("fixture library loads");
    let function = library
        .function("write_texture_2d")
        .expect("fixture entry exists");
    let pipeline = executor
        .new_compute_pipeline(&function)
        .expect("the storage fixture pipeline creates");
    let refusal = executor
        .execute(ComputeSubmission {
            pipeline,
            buffers: Vec::new(),
            textures: vec![texture_view(TextureAccess::Storage, zeros())],
            threads_per_grid: Size::new(TEXTURE_WIDTH as u32, TEXTURE_HEIGHT as u32, 1).unwrap(),
            threads_per_threadgroup: Size::new(TEXTURE_WIDTH as u32, TEXTURE_HEIGHT as u32, 1)
                .unwrap(),
        })
        .expect_err("the executor API has no storage image landing channel");
    eprintln!("C2 executor storage refusal: {refusal}");
    assert!(
        refusal.to_string().contains("buffer updates only"),
        "the refusal has to name the channel it is missing, got {refusal}"
    );
}

#[test]
fn the_deferred_completion_path_publishes_the_storage_image_landing() {
    // The landing is not a synchronous-only shortcut: with deferred execution
    // `submit` returns `Submitted` with no writebacks and the texels appear
    // when the fence retires, on the same identity-keyed channel.
    let Some(executor) = executor() else {
        return;
    };
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor))
        .expect("provider")
        .with_async_execution(true);
    let pipeline = compile(&provider, &executor, BIAS_1000_AIR);
    let value = trace(&provider, &pipeline);
    let admitted = provider
        .capabilities()
        .validate_trace(value, resources(&provider))
        .expect("the storage image trace is structurally admissible");
    let submission = provider
        .submit(admitted)
        .expect("the deferred submission is accepted");
    let CompletionDisposition::Submitted { token } = submission.completion else {
        panic!(
            "a deferred submission returns Submitted, got {:?}",
            submission.completion
        );
    };
    assert!(
        submission.writebacks.is_empty(),
        "a deferred submission publishes no writeback before `wait`"
    );
    let disposition = provider
        .wait(token, Duration::from_secs(20))
        .expect("the fence retires");
    assert!(matches!(
        disposition,
        CompletionDisposition::CompletedVisible { .. }
    ));
    let readback = provider
        .readback(token)
        .expect("the landing is published when the fence retires");
    let writeback = readback
        .writebacks
        .iter()
        .find(|writeback| writeback.view_id == TEXTURE_VIEW)
        .expect("the deferred path lands the storage image");
    let floats = writeback
        .bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    eprintln!("C2 deferred landing bias 1000: {floats:?}");
    assert_eq!(
        floats,
        (0..16).map(|cell| cell as f32 + 1000.0).collect::<Vec<_>>()
    );
}

#[test]
fn a_sampled_object_against_a_storage_declaration_is_refused_by_name() {
    // The object face carries the storage image on both rails from E-CO1 on
    // (`tests/compute_object_texture_e2e.rs`), but the mistake this test names
    // stays expressible: a `new_texture_with_bytes` handle is `Sampled`
    // (`research/docs/16` §4.4), so pairing it with a storage-image pipeline is
    // still refused by the core pair rule with the binding and both accesses
    // instead of binding a read-only descriptor to a writable module.
    let Some(executor) = executor() else {
        return;
    };
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider");
    let device = metal_api_core::provider_api::Device::new(Arc::new(provider));
    let pipeline = device
        .compile_pipeline(metal_api_core::provider::PipelineCompileRequest {
            entry_name: "write_texture_2d".to_owned(),
            logical_digest: SemanticDigest::new(
                "metal-smoke-fixture-v1",
                b"object_storage_image".to_vec(),
            )
            .expect("digest"),
            source: metal_api_core::provider::ShaderSource::SanitizedLl(BIAS_1000_AIR.to_owned()),
        })
        .expect("pipeline");
    let texture = device
        .new_texture_with_bytes(
            TextureFormat::R32Float,
            TEXTURE_WIDTH,
            TEXTURE_HEIGHT,
            zeros(),
        )
        .expect("texture object");
    let queue = device.new_command_queue();
    let command = queue.command_buffer();
    {
        let mut encoder = command.compute_command_encoder().expect("encoder");
        encoder
            .set_compute_pipeline_state(&pipeline)
            .expect("pipeline state");
        encoder.set_texture(0, &texture).expect("texture binding");
        encoder
            .dispatch_threads(
                metal_api_core::Size::new(TEXTURE_WIDTH as u32, TEXTURE_HEIGHT as u32, 1).unwrap(),
                metal_api_core::Size::new(TEXTURE_WIDTH as u32, TEXTURE_HEIGHT as u32, 1).unwrap(),
            )
            .expect("dispatch");
        encoder.end_encoding().expect("end encoding");
    }
    let refusal = command
        .commit()
        .expect_err("the object face does not execute a storage image binding");
    eprintln!("C2 object storage refusal: {refusal:?}");
    let metal_api_core::provider_api::Error::Provider(refusal) = refusal else {
        panic!("the pair rule reaches the caller as a provider refusal");
    };
    assert_eq!(refusal.slug, "trace_contract_invalid");
    assert_eq!(
        refusal.detail.as_deref(),
        Some("texture 0 is declared Sampled, but the contract pairs it with Storage"),
        "the refusal names the binding and both accesses"
    );
}
