//! End-to-end compute textures through the shared object API
//! (`research/docs/26` §21.3–21.4, E-CO1).
//!
//! C1/C1b gave the trace rail a sampled compute texture; C2 gave it a storage
//! image whose texels land through the identity-keyed writeback channel. The
//! object rail could record a texture binding from `f017d25` on, but every view
//! it built declared `Sampled`, so the writable half of the face — and the
//! landing a `Texture` handle would have to observe — was outside the object
//! model. This file pins the two halves on both object rails:
//!
//! * a sampled texture's *content* moves the readback of both the synchronous
//!   and the deferred object rail (`[100..115]` against `[115..100]`), and the
//!   handle itself stays the declaration's bytes because a sampled texture is a
//!   read-only source;
//! * a storage image's landing arrives on both object rails as the same bytes
//!   the trace rail publishes, byte for byte, and the handle's own bytes are
//!   the landing (`Texture::read`);
//! * the recorder pairs the texture face before any execution — a declaration
//!   the pass does not bind and a binding the contract does not declare are
//!   refused by name at `dispatch` — and the shapes the module does not declare
//!   stay refused at admission with the core pair rule's trace-contract slug.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferWriteback, CompletionDisposition, CompletionPolicy,
    ComputePass, ComputeProvider, ComputeTrace, ContractError, Dispatch, DispatchKind,
    DispatchType, OperationId, PipelineCompileRequest, ProviderSubmission, ResourceTableSnapshot,
    SamplerAddressMode, SamplerFilter, SamplerPolicy, SemanticDigest, ShaderSource, TextureAccess,
    TextureBindingContract, TextureFormat, TextureSource, TextureType, TextureView, TracePass,
    ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{CommandBufferStatus, ComputeExecutor, Device, Size};
use metal_api_vulkan::{VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;

const SAMPLED_CELL_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_read_texture_2d_cell.ll");
const BIAS_1000_AIR: &str = include_str!("fixtures/write_texture_2d_bias_1000.ll");
const BIAS_2000_AIR: &str = include_str!("fixtures/write_texture_2d_bias_2000.ll");
const INCREMENT_AIR: &str = include_str!("fixtures/read_write_texture_2d_increment.ll");
const NEAREST_CLAMP_AIR: &str = include_str!("fixtures/sample_texture_2d_nearest_clamp.ll");
const NEAREST_REPEAT_AIR: &str = include_str!("fixtures/sample_texture_2d_nearest_repeat.ll");
const LINEAR_CLAMP_AIR: &str = include_str!("fixtures/sample_texture_2d_linear_clamp.ll");

const SAMPLED_ENTRY: &str = "read_texture_2d_cell";
const WIDTH: u64 = 4;
const HEIGHT: u64 = 4;
const EXTENT: usize = (WIDTH * HEIGHT * 4) as usize;

/// The trace rail's storage-image identity, the one C2's readings use.
const TRACE_VIEW: ViewId = ViewId::new(940);
const TRACE_ALLOCATION: AllocationId = AllocationId::new(941);

fn executor() -> Option<Arc<VulkanExecutor>> {
    match VulkanExecutor::new() {
        Ok(executor) => Some(executor),
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            None
        }
    }
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn grid() -> Size {
    Size::new(WIDTH as u32, HEIGHT as u32, 1).expect("grid")
}

fn local() -> Size {
    Size::new(WIDTH as u32, HEIGHT as u32, 1).expect("local size")
}

/// One object-API provider over the shared executor, synchronous or deferred.
fn object_device(
    executor: &Arc<VulkanExecutor>,
    async_execution: bool,
) -> metal_api_core::provider_api::Device {
    let provider = VulkanComputeProvider::with_executor(Arc::clone(executor))
        .expect("provider")
        .with_async_execution(async_execution);
    metal_api_core::provider_api::Device::new(Arc::new(provider))
}

fn compile(
    device: &metal_api_core::provider_api::Device,
    entry: &str,
    source: &str,
    label: &str,
) -> metal_api_core::provider_api::Pipeline {
    device
        .compile_pipeline(PipelineCompileRequest {
            entry_name: entry.to_owned(),
            logical_digest: SemanticDigest::new(
                "metal-smoke-fixture-v1",
                label.as_bytes().to_vec(),
            )
            .expect("digest"),
            source: ShaderSource::SanitizedLl(source.to_owned()),
        })
        .expect("the fixture registers")
}

/// Commit, observe both completion dispositions, and wait for the landing.
fn commit_and_wait(
    command: &metal_api_core::provider_api::CommandBuffer,
    async_execution: bool,
) -> ProviderSubmission {
    command.commit().expect("commit");
    if async_execution {
        assert_eq!(
            command.status().expect("status"),
            CommandBufferStatus::Committed,
            "a deferred commit leaves the command pending"
        );
        assert!(
            matches!(
                command.submission().expect("submission").completion,
                CompletionDisposition::Submitted { .. }
            ),
            "a deferred commit returns a submitted token and no landing yet"
        );
    }
    command.wait_until_completed().expect("completion");
    assert_eq!(
        command.status().expect("status"),
        CommandBufferStatus::Completed
    );
    let submission = command.submission().expect("submission");
    assert!(matches!(
        submission.completion,
        CompletionDisposition::CompletedVisible { .. }
    ));
    submission
}

/// Run one sampled-texture cell dispatch on one object rail and return the
/// output buffer's bytes.
fn sampled_cell_readback(
    executor: &Arc<VulkanExecutor>,
    async_execution: bool,
    texels: &[u32],
) -> (Vec<u8>, Vec<u8>) {
    let device = object_device(executor, async_execution);
    let pipeline = compile(
        &device,
        SAMPLED_ENTRY,
        SAMPLED_CELL_AIR,
        if async_execution {
            "object-sampled-async"
        } else {
            "object-sampled"
        },
    );
    assert_eq!(
        pipeline.metadata().contract.texture_bindings.len(),
        1,
        "the sampled fixture declares one texture"
    );
    assert_eq!(
        pipeline.metadata().contract.texture_bindings[0].access,
        TextureAccess::Sampled
    );
    let texture = device
        .new_texture_with_bytes(TextureFormat::R32Uint, WIDTH, HEIGHT, u32_bytes(texels))
        .expect("texture object");
    let output = device
        .new_buffer_with_bytes(vec![0_u8; EXTENT])
        .expect("output buffer");
    let queue = device.new_command_queue();
    let command = queue.command_buffer();
    {
        let mut encoder = command.compute_command_encoder().expect("encoder");
        encoder
            .set_compute_pipeline_state(&pipeline)
            .expect("pipeline state");
        encoder.set_texture(0, &texture).expect("texture binding");
        encoder
            .set_buffer(0, &output.view(0, EXTENT).expect("output view"))
            .expect("buffer binding");
        encoder.dispatch_threads(grid(), local()).expect("dispatch");
        encoder.end_encoding().expect("end encoding");
    }
    let submission = commit_and_wait(&command, async_execution);
    // A sampled texture is a read-only source: no published writeback names
    // it, and the handle still holds the declaration's own bytes.
    assert!(
        submission
            .writebacks
            .iter()
            .all(|write| write.allocation_id != texture.allocation_id()),
        "a sampled texture is never a writeback target"
    );
    assert_eq!(
        texture.read().expect("texture read"),
        u32_bytes(texels),
        "the sampled handle's bytes are the declaration's"
    );
    (output.read().expect("readback"), u32_bytes(texels))
}

#[test]
fn sampled_texture_content_moves_both_object_rails() {
    let Some(executor) = executor() else {
        return;
    };
    let ascending = (0..16_u32).collect::<Vec<_>>();
    let descending = (0..16_u32).rev().collect::<Vec<_>>();
    for async_execution in [false, true] {
        let (first, first_declaration) =
            sampled_cell_readback(&executor, async_execution, &ascending);
        let (second, second_declaration) =
            sampled_cell_readback(&executor, async_execution, &descending);
        assert_eq!(first_declaration, u32_bytes(&ascending));
        assert_eq!(second_declaration, u32_bytes(&descending));
        let first = decode_u32(&first);
        let second = decode_u32(&second);
        eprintln!("E-CO1 sampled object rail async={async_execution} ascending content: {first:?}");
        eprintln!(
            "E-CO1 sampled object rail async={async_execution} descending content: {second:?}"
        );
        assert_eq!(
            first,
            (100..116).collect::<Vec<u32>>(),
            "ascending content 0..15 lands 100..115"
        );
        assert_eq!(
            second,
            (100..116).rev().collect::<Vec<u32>>(),
            "descending content 15..0 lands the same cells in the other order"
        );
        assert_ne!(
            first, second,
            "the content change separates the two readings"
        );
    }
}

/// The trace rail's landing for one storage fixture: the same raw trace C2's
/// reading uses, so the object rail is compared against the rail that owned
/// the shape first.
fn trace_rail_landing(
    executor: &Arc<VulkanExecutor>,
    entry: &str,
    source: &str,
    initial: Vec<u8>,
) -> Vec<u8> {
    let provider = VulkanComputeProvider::with_executor(Arc::clone(executor)).expect("provider");
    let compile_device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let library = compile_device
        .new_library_with_air(source)
        .expect("fixture library loads");
    let function = library.function(entry).expect("fixture entry exists");
    let pipeline = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", source.as_bytes().to_vec())
                .expect("digest"),
        )
        .expect("the storage fixture registers");
    let pipeline_id = pipeline.pipeline_id;
    let mut resources = ResourceTableSnapshot::new();
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: TRACE_ALLOCATION,
            owner_epoch: provider.device_epoch(),
            size: EXTENT as u64,
        })
        .expect("storage image allocation");
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(78),
        pipelines: vec![pipeline],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![TracePass::Compute(ComputePass {
            pipeline: pipeline_id,
            buffers: Vec::new(),
            textures: vec![TextureView {
                view_id: TRACE_VIEW,
                metal_binding: 0,
                allocation_id: TRACE_ALLOCATION,
                texture_type: TextureType::D2,
                format: TextureFormat::R32Float,
                width: WIDTH,
                height: HEIGHT,
                depth: 1,
                array_length: 1,
                sample_count: 1,
                access: TextureAccess::Storage,
                source: TextureSource::OwnedBytes(initial),
            }],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [WIDTH, HEIGHT, 1],
                threads_per_threadgroup: [WIDTH, HEIGHT, 1],
            },
        })],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let admitted = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect("the storage image trace is structurally admissible");
    let submission = provider
        .submit(admitted)
        .expect("the storage image executes");
    submission
        .writebacks
        .iter()
        .find(|writeback| writeback.view_id == TRACE_VIEW)
        .expect("the storage image lands a writeback")
        .bytes
        .clone()
}

/// Run one storage fixture on one object rail and return the handle's landing
/// plus the published writeback.
fn object_rail_landing(
    executor: &Arc<VulkanExecutor>,
    async_execution: bool,
    entry: &str,
    source: &str,
    initial: Vec<u8>,
) -> (Vec<u8>, BufferWriteback) {
    let device = object_device(executor, async_execution);
    let pipeline = compile(
        &device,
        entry,
        source,
        if async_execution {
            "object-storage-async"
        } else {
            "object-storage"
        },
    );
    assert_eq!(
        pipeline.metadata().contract.texture_bindings,
        vec![TextureBindingContract::storage(0, TextureFormat::R32Float)],
        "the object rail's declaration is the module's own storage shape"
    );
    let texture = device
        .new_storage_texture_with_bytes(TextureFormat::R32Float, WIDTH, HEIGHT, initial)
        .expect("storage texture object");
    assert_eq!(texture.access(), TextureAccess::Storage);
    let queue = device.new_command_queue();
    let command = queue.command_buffer();
    {
        let mut encoder = command.compute_command_encoder().expect("encoder");
        encoder
            .set_compute_pipeline_state(&pipeline)
            .expect("pipeline state");
        encoder.set_texture(0, &texture).expect("texture binding");
        encoder.dispatch_threads(grid(), local()).expect("dispatch");
        encoder.end_encoding().expect("end encoding");
    }
    let submission = commit_and_wait(&command, async_execution);
    let writeback = submission
        .writebacks
        .iter()
        .find(|writeback| {
            writeback.allocation_id == texture.allocation_id()
                && writeback.view_id == texture.view_id()
        })
        .expect("the storage image lands through the object's own identity")
        .clone();
    assert_eq!(writeback.offset, 0, "a storage image has no byte offset");
    assert_eq!(writeback.bytes.len(), EXTENT);
    assert_eq!(
        texture.read().expect("texture read"),
        writeback.bytes,
        "the handle's bytes are the landing the submission published"
    );
    (writeback.bytes.clone(), writeback)
}

#[test]
fn storage_image_landing_matches_the_trace_rail_on_both_object_rails() {
    let Some(executor) = executor() else {
        return;
    };
    let zeros = f32_bytes(&[0.0_f32; 16]);
    let increment_seed = f32_bytes(&(0..16_u32).map(|cell| cell as f32).collect::<Vec<_>>());
    let cases: [(&str, &str, Vec<u8>, Vec<f32>); 3] = [
        (
            "write_texture_2d",
            BIAS_1000_AIR,
            zeros.clone(),
            (0..16_u32).map(|cell| cell as f32 + 1000.0).collect(),
        ),
        (
            "write_texture_2d",
            BIAS_2000_AIR,
            zeros,
            (0..16_u32).map(|cell| cell as f32 + 2000.0).collect(),
        ),
        (
            "read_write_texture_2d",
            INCREMENT_AIR,
            increment_seed,
            (1..=16_u32).map(|cell| cell as f32).collect(),
        ),
    ];
    for (entry, source, initial, expected) in cases {
        let expected = f32_bytes(&expected);
        let trace = trace_rail_landing(&executor, entry, source, initial.clone());
        eprintln!(
            "E-CO1 trace-rail storage landing {entry}: {:?}",
            decode_f32(&trace)
        );
        assert_eq!(trace, expected, "the trace rail lands the module's texels");
        for async_execution in [false, true] {
            let (landed, _writeback) =
                object_rail_landing(&executor, async_execution, entry, source, initial.clone());
            eprintln!(
                "E-CO1 object storage landing {entry} async={async_execution}: {:?}",
                decode_f32(&landed)
            );
            assert_eq!(
                landed, trace,
                "the object rail's landing is the trace rail's, byte for byte"
            );
        }
    }
}

#[test]
fn the_object_rail_executes_the_modules_own_constexpr_sampler() {
    let Some(executor) = executor() else {
        return;
    };
    let row = [0.0_f32, 4.0, 8.0, 12.0];
    let texture_bytes = [row, row, row, row]
        .into_iter()
        .flatten()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<u8>>();
    let cases = [
        (
            "nearest-clamp",
            NEAREST_CLAMP_AIR,
            SamplerPolicy {
                filter: SamplerFilter::Nearest,
                address: SamplerAddressMode::ClampToEdge,
            },
            [12.0_f32, 4.0],
        ),
        (
            "nearest-repeat",
            NEAREST_REPEAT_AIR,
            SamplerPolicy {
                filter: SamplerFilter::Nearest,
                address: SamplerAddressMode::Repeat,
            },
            [4.0_f32, 4.0],
        ),
        (
            "linear-clamp",
            LINEAR_CLAMP_AIR,
            SamplerPolicy {
                filter: SamplerFilter::Linear,
                address: SamplerAddressMode::ClampToEdge,
            },
            [12.0_f32, 3.0],
        ),
    ];
    for async_execution in [false, true] {
        for (label, source, policy, expected) in cases {
            let device = object_device(&executor, async_execution);
            let pipeline = compile(&device, "sample_texture_2d", source, label);
            assert_eq!(
                pipeline.metadata().contract.texture_bindings.len(),
                1,
                "the sampler fixture declares one texture"
            );
            assert_eq!(
                pipeline.metadata().contract.texture_bindings[0].sampler,
                Some(policy),
                "the object rail's declaration is the module's own state"
            );
            let texture = device
                .new_texture_with_bytes(
                    TextureFormat::R32Float,
                    WIDTH,
                    HEIGHT,
                    texture_bytes.clone(),
                )
                .expect("sampled texture object");
            let output = device
                .new_buffer_with_bytes(vec![0_u8; 8])
                .expect("output buffer");
            let queue = device.new_command_queue();
            let command = queue.command_buffer();
            {
                let mut encoder = command.compute_command_encoder().expect("encoder");
                encoder
                    .set_compute_pipeline_state(&pipeline)
                    .expect("pipeline state");
                encoder.set_texture(0, &texture).expect("texture binding");
                encoder
                    .set_buffer(0, &output.view(0, 8).expect("output view"))
                    .expect("buffer binding");
                encoder
                    .dispatch_threads(
                        Size::new(1, 1, 1).expect("sampler grid"),
                        Size::new(1, 1, 1).expect("sampler local size"),
                    )
                    .expect("dispatch");
                encoder.end_encoding().expect("end encoding");
            }
            commit_and_wait(&command, async_execution);
            let bytes = output.read().expect("readback");
            let first = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
            let second = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
            eprintln!(
                "E-CO1 object sampler {label} async={async_execution}: [{first:?}, {second:?}]"
            );
            assert_eq!(
                [first, second],
                expected,
                "the module's own sampler state is the state the object rail executed"
            );
        }
    }
}

#[test]
fn the_object_recorder_pairs_texture_bindings_before_execution() {
    let Some(executor) = executor() else {
        return;
    };
    let device = object_device(&executor, false);
    let sampled = compile(
        &device,
        SAMPLED_ENTRY,
        SAMPLED_CELL_AIR,
        "object-pair-sampled",
    );
    let storage = compile(
        &device,
        "write_texture_2d",
        BIAS_1000_AIR,
        "object-pair-storage",
    );
    let storage_texture = device
        .new_storage_texture_with_bytes(
            TextureFormat::R32Float,
            WIDTH,
            HEIGHT,
            f32_bytes(&[0.0_f32; 16]),
        )
        .expect("storage texture object");
    let output = device
        .new_buffer_with_bytes(vec![0_u8; EXTENT])
        .expect("output buffer");

    // (a) The contract declares texture 0; the pass binds none there.
    let queue = device.new_command_queue();
    {
        let command = queue.command_buffer();
        let mut encoder = command.compute_command_encoder().expect("encoder");
        encoder
            .set_compute_pipeline_state(&sampled)
            .expect("pipeline state");
        encoder
            .set_buffer(0, &output.view(0, EXTENT).expect("output view"))
            .expect("buffer binding");
        let refusal = encoder
            .dispatch_threads(grid(), local())
            .expect_err("a declared texture binding that the pass does not bind is refused");
        eprintln!("E-CO1 object missing texture binding refusal: {refusal}");
        assert_eq!(
            refusal,
            metal_api_core::provider_api::Error::Contract(ContractError::MissingTextureBinding {
                binding: 0
            })
        );
        assert_eq!(
            refusal.to_string(),
            "pipeline contract declares texture binding 0, but the pass binds no texture there"
        );
    }

    // (b) The pass binds two textures; the contract declares one.
    {
        let command = queue.command_buffer();
        let mut encoder = command.compute_command_encoder().expect("encoder");
        encoder
            .set_compute_pipeline_state(&storage)
            .expect("pipeline state");
        encoder
            .set_texture(0, &storage_texture)
            .expect("declared binding");
        encoder
            .set_texture(1, &storage_texture)
            .expect("undeclared binding");
        let refusal = encoder
            .dispatch_threads(grid(), local())
            .expect_err("a texture binding the contract does not declare is refused");
        eprintln!("E-CO1 object undeclared texture binding refusal: {refusal}");
        assert_eq!(
            refusal,
            metal_api_core::provider_api::Error::Contract(
                ContractError::UndeclaredTextureBinding { binding: 1 }
            )
        );
        assert_eq!(
            refusal.to_string(),
            "pass binds texture 1, but the pipeline contract declares no texture there"
        );
    }
}

#[test]
fn an_object_texture_the_module_does_not_declare_is_refused_by_name() {
    let Some(executor) = executor() else {
        return;
    };
    let device = object_device(&executor, false);
    let storage_pipeline = compile(
        &device,
        "write_texture_2d",
        BIAS_1000_AIR,
        "object-mismatch-storage",
    );

    // (a) A sampled object where the module writes: the object model can still
    // express this mistake, so the pair rule has to refuse it with both
    // accesses.
    let sampled = device
        .new_texture_with_bytes(
            TextureFormat::R32Float,
            WIDTH,
            HEIGHT,
            f32_bytes(&[0.0_f32; 16]),
        )
        .expect("sampled texture object");
    let queue = device.new_command_queue();
    {
        let command = queue.command_buffer();
        let mut encoder = command.compute_command_encoder().expect("encoder");
        encoder
            .set_compute_pipeline_state(&storage_pipeline)
            .expect("pipeline state");
        encoder.set_texture(0, &sampled).expect("texture binding");
        encoder.dispatch_threads(grid(), local()).expect("dispatch");
        encoder.end_encoding().expect("end encoding");
        let refusal = command
            .commit()
            .expect_err("a sampled object against a storage module is refused");
        eprintln!("E-CO1 object sampled-against-storage refusal: {refusal}");
        let metal_api_core::provider_api::Error::Provider(refusal) = refusal else {
            panic!("the pair rule reaches the caller as a provider refusal");
        };
        assert_eq!(refusal.slug, "trace_contract_invalid");
        assert_eq!(
            refusal.detail.as_deref(),
            Some("texture 0 is declared Sampled, but the contract pairs it with Storage")
        );
    }

    // (b) A storage object whose format the module does not carry: the object
    // model declares `R32Uint`, the module reflects `R32Float`.
    let foreign = device
        .new_storage_texture_with_bytes(
            TextureFormat::R32Uint,
            WIDTH,
            HEIGHT,
            u32_bytes(&[0_u32; 16]),
        )
        .expect("storage texture object");
    {
        let command = queue.command_buffer();
        let mut encoder = command.compute_command_encoder().expect("encoder");
        encoder
            .set_compute_pipeline_state(&storage_pipeline)
            .expect("pipeline state");
        encoder.set_texture(0, &foreign).expect("texture binding");
        encoder.dispatch_threads(grid(), local()).expect("dispatch");
        encoder.end_encoding().expect("end encoding");
        let refusal = command
            .commit()
            .expect_err("a storage object of another format is refused");
        eprintln!("E-CO1 object foreign-format refusal: {refusal}");
        let metal_api_core::provider_api::Error::Provider(refusal) = refusal else {
            panic!("the pair rule reaches the caller as a provider refusal");
        };
        assert_eq!(refusal.slug, "trace_contract_invalid");
        assert_eq!(
            refusal.detail.as_deref(),
            Some("texture 0 states format R32Uint, but the contract pairs it with R32Float")
        );
    }
}
