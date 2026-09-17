//! End-to-end compute texture sampling (`research/docs/26` §21.3, C1b): three
//! translations that differ only in the AIR-embedded constexpr sampler state
//! sample one `R32Float` texture and land different bytes.
//!
//! The falsifiable claim is that the state the module carries is the state the
//! pass executes with. All three fixtures share a byte-identical kernel body
//! (two `air.sample_texture_2d` calls, fixed coordinates, one 8-byte `float2`
//! store); only the `@__air_sampler_state` words move between
//! nearest+clamp-to-edge, nearest+repeat and linear+clamp-to-edge. Against a
//! row of `[0, 4, 8, 12]` the two readings are:
//!
//! * nearest + clamp-to-edge: `[12, 4]` — `u=1.375` clamps to texel 3 and the
//!   second sample point lands in texel 1;
//! * nearest + repeat: `[4, 4]` — `u=1.375` wraps to the centre of texel 1;
//! * linear + clamp-to-edge: `[12, 3]` — the second sample blends a quartile of
//!   texel 0 into texel 1.
//!
//! The refusal half pins the declaration: the registered pipeline's contract
//! *is* the module's own state, and a request that declares another state is
//! refused by name (`compute_texture_sampler_unsupported`) with the binding
//! and both state halves.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferAccess, BufferSource, BufferView,
    CompiledComputePipeline, CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace,
    Dispatch, DispatchKind, DispatchType, OperationId, ResourceTableSnapshot, SamplerAddressMode,
    SamplerFilter, SamplerPolicy, SemanticDigest, TextureAccess, TextureFormat, TextureSource,
    TextureType, TextureView, TracePass, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{BufferBinding, ComputeExecutor, Device, Size};
use metal_api_vulkan::{VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;

const NEAREST_CLAMP_AIR: &str = include_str!("fixtures/sample_texture_2d_nearest_clamp.ll");
const NEAREST_REPEAT_AIR: &str = include_str!("fixtures/sample_texture_2d_nearest_repeat.ll");
const LINEAR_CLAMP_AIR: &str = include_str!("fixtures/sample_texture_2d_linear_clamp.ll");

const TEXTURE_VIEW: ViewId = ViewId::new(920);
const TEXTURE_ALLOCATION: AllocationId = AllocationId::new(921);
const OUTPUT_VIEW: ViewId = ViewId::new(922);
const OUTPUT_ALLOCATION: AllocationId = AllocationId::new(923);

const TEXTURE_BYTES: u64 = 4 * 4 * 4;

/// The sampled row: texel `i` holds `4 * i` in every row of the 4x4 texture.
fn texture_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(TEXTURE_BYTES as usize);
    for _row in 0..4 {
        for value in [0.0_f32, 4.0, 8.0, 12.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn executor() -> Option<Arc<VulkanExecutor>> {
    match VulkanExecutor::new() {
        Ok(executor) => Some(executor),
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            None
        }
    }
}

/// Run one fixture through the synchronous executor entry point and decode the
/// two `f32` readings its store landed.
fn readings(executor: &Arc<VulkanExecutor>, source: &str) -> [f32; 2] {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let library = device
        .new_library_with_air(source)
        .expect("fixture library loads");
    let function = library
        .function("sample_texture_2d")
        .expect("fixture entry exists");
    let pipeline = executor
        .new_compute_pipeline(&function)
        .expect("the sampler fixture pipeline creates");
    let submission = metal_api_core::ComputeSubmission {
        pipeline,
        buffers: vec![BufferBinding {
            index: 0,
            bytes: vec![0_u8; 8],
        }],
        textures: vec![TextureView {
            view_id: TEXTURE_VIEW,
            metal_binding: 0,
            allocation_id: TEXTURE_ALLOCATION,
            texture_type: TextureType::D2,
            format: TextureFormat::R32Float,
            width: 4,
            height: 4,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            access: TextureAccess::Sampled,
            source: TextureSource::OwnedBytes(texture_bytes()),
        }],
        threads_per_grid: Size::new(1, 1, 1).unwrap(),
        threads_per_threadgroup: Size::new(1, 1, 1).unwrap(),
    };
    let updates = executor
        .execute(submission)
        .expect("the sampler fixture executes");
    assert_eq!(updates.len(), 1, "one writeable buffer has one update");
    let bytes = &updates[0].bytes;
    assert_eq!(bytes.len(), 8, "the fixture stores exactly two floats");
    let first = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let second = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
    [first, second]
}

fn policy(filter: SamplerFilter, address: SamplerAddressMode) -> SamplerPolicy {
    SamplerPolicy { filter, address }
}

fn compile_provider_pipeline(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
    source: &str,
) -> CompiledComputePipeline {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let library = device
        .new_library_with_air(source)
        .expect("fixture library loads");
    let function = library
        .function("sample_texture_2d")
        .expect("fixture entry exists");
    provider
        .compile_pipeline(
            &function,
            SemanticDigest::new(
                "metal-smoke-fixture-v1",
                b"compute_texture_sampler".to_vec(),
            )
            .expect("digest"),
        )
        .expect("the sampler fixture registers")
}

fn trace(provider: &VulkanComputeProvider, pipeline: &CompiledComputePipeline) -> ComputeTrace {
    ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(77),
        pipelines: vec![pipeline.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![TracePass::Compute(ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers: vec![BufferView {
                view_id: OUTPUT_VIEW,
                metal_binding: 0,
                allocation_id: OUTPUT_ALLOCATION,
                offset: 0,
                length: 8,
                access: BufferAccess::Write,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![0_u8; 8]),
            }],
            textures: vec![TextureView {
                view_id: TEXTURE_VIEW,
                metal_binding: 0,
                allocation_id: TEXTURE_ALLOCATION,
                texture_type: TextureType::D2,
                format: TextureFormat::R32Float,
                width: 4,
                height: 4,
                depth: 1,
                array_length: 1,
                sample_count: 1,
                access: TextureAccess::Sampled,
                source: TextureSource::OwnedBytes(texture_bytes()),
            }],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
        })],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    }
}

fn resources(provider: &VulkanComputeProvider) -> ResourceTableSnapshot {
    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [(TEXTURE_ALLOCATION, TEXTURE_BYTES), (OUTPUT_ALLOCATION, 8)] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("sampler fixture allocation");
    }
    resources
}

#[test]
fn the_air_sampler_state_is_the_state_the_pass_executes_with() {
    let Some(executor) = executor() else {
        return;
    };

    let nearest_clamp = readings(&executor, NEAREST_CLAMP_AIR);
    let nearest_repeat = readings(&executor, NEAREST_REPEAT_AIR);
    let linear_clamp = readings(&executor, LINEAR_CLAMP_AIR);

    // Raw readings, both halves, so the separation is inspectable rather than
    // asserted-only.
    eprintln!("C1b reading nearest+clamp:  {nearest_clamp:?}");
    eprintln!("C1b reading nearest+repeat: {nearest_repeat:?}");
    eprintln!("C1b reading linear+clamp:   {linear_clamp:?}");

    assert_eq!(
        nearest_clamp,
        [12.0, 4.0],
        "u=1.375 clamps to texel 3; the second sample lands in texel 1"
    );
    assert_eq!(
        nearest_repeat,
        [4.0, 4.0],
        "u=1.375 wraps to the centre of texel 1 under repeat"
    );
    assert_eq!(
        linear_clamp,
        [12.0, 3.0],
        "the second sample blends a quartile of texel 0 into texel 1"
    );
    assert_ne!(
        nearest_clamp, linear_clamp,
        "the two filters separate on this fixture"
    );
    assert_ne!(
        nearest_clamp, nearest_repeat,
        "the two address modes separate on this fixture"
    );
}

#[test]
fn a_declaration_that_moves_away_from_the_modules_sampler_is_refused_by_name() {
    let Some(executor) = executor() else {
        return;
    };
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider");
    let registered = compile_provider_pipeline(&provider, &executor, LINEAR_CLAMP_AIR);

    // The contract the rail derived *is* the module's own state: the fixture
    // carries a linear, clamp-to-edge AIR sampler, and the registered contract
    // restates exactly that.
    assert_eq!(registered.contract.texture_bindings.len(), 1);
    assert_eq!(
        registered.contract.texture_bindings[0].sampler,
        Some(policy(
            SamplerFilter::Linear,
            SamplerAddressMode::ClampToEdge
        )),
        "the registered declaration is the module's own AIR state"
    );
    assert_eq!(
        registered.contract.texture_bindings[0].format,
        TextureFormat::R32Float
    );

    // A request that declares nearest+clamp against that module is refused
    // before any device object exists.
    let mut tampered = registered.clone();
    tampered.contract.texture_bindings[0].sampler = Some(policy(
        SamplerFilter::Nearest,
        SamplerAddressMode::ClampToEdge,
    ));
    let tampered_trace = trace(&provider, &tampered);
    let admitted = provider
        .capabilities()
        .validate_trace(tampered_trace, resources(&provider))
        .expect("the tampered declaration is structurally admissible");
    let refusal = provider
        .submit(admitted)
        .expect_err("a foreign sampler declaration is refused");
    eprintln!("C1b sampler refusal: {refusal:?}");
    assert_eq!(refusal.slug, "compute_texture_sampler_unsupported");
    assert_eq!(
        refusal.fields.get("binding"),
        Some(&metal_api_core::provider::FieldValue::Unsigned(0))
    );
    assert_eq!(
        refusal.fields.get("filter"),
        Some(&metal_api_core::provider::FieldValue::Text(
            "Nearest".into()
        ))
    );
    assert_eq!(
        refusal.fields.get("module_filter"),
        Some(&metal_api_core::provider::FieldValue::Text("Linear".into()))
    );
    assert_eq!(
        refusal.fields.get("address"),
        Some(&metal_api_core::provider::FieldValue::Text(
            "ClampToEdge".into()
        ))
    );

    // The same trace with the module's own state runs end to end and lands the
    // module's own bytes, which keeps the refusal above from being a blanket
    // rejection of the shape.
    let registered_trace = trace(&provider, &registered);
    let admitted = provider
        .capabilities()
        .validate_trace(registered_trace, resources(&provider))
        .expect("the module's own declaration is admitted");
    let submission = provider
        .submit(admitted)
        .expect("the registered sampler executes");
    let writeback = submission
        .writebacks
        .iter()
        .find(|writeback| writeback.view_id == OUTPUT_VIEW)
        .expect("the fixture writes its output buffer");
    let first = f32::from_le_bytes(writeback.bytes[0..4].try_into().unwrap());
    let second = f32::from_le_bytes(writeback.bytes[4..8].try_into().unwrap());
    eprintln!("C1b provider-path reading linear+clamp: [{first}, {second}]");
    assert_eq!([first, second], [12.0, 3.0]);
}
