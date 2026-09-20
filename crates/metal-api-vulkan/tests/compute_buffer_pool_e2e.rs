//! The compute half's pooled host-visible upload buffers' own rail cases
//! (`crate::compute_buffer_pool`).
//!
//! The increment hands a creation of a given shape — a `VkBuffer` of one size
//! and one usage — the buffer and memory a creation of the *same shape* built
//! before. That is only allowed to be a timing change, so this file is the
//! byte-level oracle the increment's own report reads. One submission of the
//! `copy_word` fixture states two owned upload buffers of the same shape: the
//! word the pass reads and the word it writes.
//!
//! * one dispatch runs twice against one provider: the second submission must
//!   be served from the pool (one hit per upload buffer) and must publish the
//!   *same* word;
//! * the second arm states *different bytes* for the same shape and must land
//!   *its own* word — the one failure direction a pooled upload buffer has
//!   ("the next submission binds what the last one wrote there");
//! * the same dispatches must publish the same bytes with the mechanism
//!   switched **off**, so the two arms of the reading are compared with each
//!   other and not only with themselves;
//! * switching the mechanism off must drop what it held, and the counters that
//!   name the directions (`hits`, `misses`, `disabled`, `returns`) must be
//!   readable from the provider, so a round that shows no reuse can tell "the
//!   shapes never repeated" from "the pool refused them".

use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferAccess, BufferSource, BufferView,
    CompiledComputePipeline, CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace,
    Dispatch, DispatchKind, DispatchType, IndirectCommandBufferDescriptor,
    IndirectCommandDescriptor, IndirectCommandKind, IndirectCommandPayload, IndirectCommandRange,
    OperationId, ProviderErrorClass, ResourceTableSnapshot, SemanticDigest, TracePass,
    ValidatedComputeTrace, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    ComputeBufferPoolCounts, DeviceLossPoint, VulkanComputeProvider, VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed fixture both this file's arms run: one `uint` in, one `uint`
/// out, so the word the pass landed is the word it read.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");
const COPY_WORD_ENTRY: &str = "copy_word";

const INPUT_VIEW: ViewId = ViewId::new(8410);
const INPUT_ALLOCATION: AllocationId = AllocationId::new(8411);
const OUTPUT_VIEW: ViewId = ViewId::new(8412);
const OUTPUT_ALLOCATION: AllocationId = AllocationId::new(8413);

/// How many upload buffers one submission creates: the word it reads and the
/// word it writes. The two have the same shape (four bytes,
/// `STORAGE_BUFFER`), so they are two entries of one key rather than two keys.
const UPLOADS_PER_SUBMISSION: u64 = 2;

fn digest(case: &[u8]) -> SemanticDigest {
    SemanticDigest::new("compute-buffer-pool-fixture-v1", case.to_vec()).expect("digest")
}

fn provider_with_device() -> Option<(Arc<VulkanExecutor>, VulkanComputeProvider)> {
    let executor = match VulkanExecutor::new() {
        Ok(executor) => executor,
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            return None;
        }
    };
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context");
    Some((executor, provider))
}

fn compile_copy_word(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
) -> CompiledComputePipeline {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function(COPY_WORD_ENTRY)
        .expect("the fixture entry exists");
    provider
        .compile_pipeline(&function, digest(b"compute-buffer-pool-copy-word"))
        .expect("the copy_word pipeline registers")
}

/// One compute pass whose read word is `word` and whose written word starts as
/// the same four zero bytes every arm states.
fn trace(
    provider: &VulkanComputeProvider,
    pipeline: &CompiledComputePipeline,
    word: u32,
) -> ComputeTrace {
    ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(83),
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
                    source: BufferSource::OwnedBytes(word.to_le_bytes().to_vec()),
                },
                BufferView {
                    view_id: OUTPUT_VIEW,
                    metal_binding: 1,
                    allocation_id: OUTPUT_ALLOCATION,
                    offset: 0,
                    length: 4,
                    access: BufferAccess::Write,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![0_u8; 4]),
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
        heap: None,
        indirect: None,
    }
}

fn resources(provider: &VulkanComputeProvider) -> ResourceTableSnapshot {
    let mut resources = ResourceTableSnapshot::new();
    for allocation in [INPUT_ALLOCATION, OUTPUT_ALLOCATION] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size: 4,
            })
            .expect("fixture allocation");
    }
    resources
}

/// Submit one `copy_word` dispatch and return the word its writeback landed.
fn admit(provider: &VulkanComputeProvider, trace: ComputeTrace) -> ValidatedComputeTrace {
    provider
        .capabilities()
        .validate_trace(trace, resources(provider))
        .expect("the copy_word trace is admitted")
}

fn submit(provider: &VulkanComputeProvider, pipeline: &CompiledComputePipeline, word: u32) -> u32 {
    let trace = trace(provider, pipeline, word);
    let submitted = provider
        .submit(admit(provider, trace.clone()))
        .expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    let bytes = submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == OUTPUT_VIEW)
        .expect("the pass writes its output buffer")
        .bytes;
    u32::from_le_bytes(bytes[0..4].try_into().expect("one word"))
}

/// One dispatch runs twice: the second submission takes every upload buffer the
/// first one built, and the word it lands is the word the fresh path landed.
#[test]
fn a_repeated_dispatch_is_served_from_the_pool_and_lands_the_same_word() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let pipeline = compile_copy_word(&provider, &executor);

    // Arm one: the mechanism on, which is its default.
    provider.set_compute_buffer_pool(true);
    let before: ComputeBufferPoolCounts = provider.compute_buffer_pool_counts();
    let first = submit(&provider, &pipeline, 0x1122_3344);
    let after_first = provider.compute_buffer_pool_counts();
    let second = submit(&provider, &pipeline, 0x1122_3344);
    let after_second = provider.compute_buffer_pool_counts();

    eprintln!(
        "compute pool on: first {first:#010x} second {second:#010x}; hits {} -> {} -> {}, \
         misses {}, entries {}, held {} bytes",
        before.hits,
        after_first.hits,
        after_second.hits,
        after_second.misses,
        after_second.entries,
        after_second.held_bytes,
    );
    assert_eq!(first, 0x1122_3344, "the fresh path lands the word it read");
    assert_eq!(
        second, first,
        "a submission served from the pool lands exactly the word the fresh path landed"
    );
    assert_eq!(
        after_first.hits - before.hits,
        0,
        "the first submission of a shape has nothing to take"
    );
    assert_eq!(
        after_first.misses - before.misses,
        UPLOADS_PER_SUBMISSION,
        "the first submission builds one buffer per owned view"
    );
    assert_eq!(
        after_second.hits - after_first.hits,
        UPLOADS_PER_SUBMISSION,
        "the second submission takes one buffer per owned view"
    );
    assert_eq!(
        after_second.misses - after_first.misses,
        0,
        "a served creation is never also counted as a build"
    );
    assert_eq!(
        after_second.returns - after_first.returns,
        UPLOADS_PER_SUBMISSION,
        "a submission whose fence was observed hands every buffer it took back"
    );
    assert_eq!(
        after_second.entries, UPLOADS_PER_SUBMISSION as usize,
        "the pool holds one buffer per shape this dispatch states"
    );
    assert!(
        after_second.held_bytes > 0,
        "a held buffer is an allocation the pool kept"
    );
    assert_eq!(
        (after_second.evictions, after_second.flushes),
        (0, 0),
        "one dispatch's shapes are far below the cap"
    );

    // Arm two: the same two dispatches with the mechanism switched off. The
    // switch drops what it held, and the fresh path lands the same bytes.
    provider.set_compute_buffer_pool(false);
    let off_before = provider.compute_buffer_pool_counts();
    assert_eq!(
        off_before.entries, 0,
        "switching the mechanism off drops what it held"
    );
    let third = submit(&provider, &pipeline, 0x1122_3344);
    let fourth = submit(&provider, &pipeline, 0x1122_3344);
    let off_after = provider.compute_buffer_pool_counts();

    eprintln!(
        "compute pool off: third {third:#010x} fourth {fourth:#010x}; hits {} -> {}, \
         disabled {} -> {}, entries {}",
        off_before.hits, off_after.hits, off_before.disabled, off_after.disabled, off_after.entries,
    );
    assert_eq!(
        (third, fourth),
        (first, second),
        "the two arms of the reading land the same word"
    );
    assert_eq!(
        off_after.hits, off_before.hits,
        "nothing is served while the switch is off"
    );
    assert_eq!(
        off_after.disabled - off_before.disabled,
        2 * UPLOADS_PER_SUBMISSION,
        "every creation of both submissions reports the switch, not a miss"
    );
    assert_eq!(off_after.entries, 0, "the switch off holds nothing");
    assert_eq!(
        off_after.returns - off_before.returns,
        0,
        "nothing is handed back while the switch is off"
    );
}

/// The second arm states different bytes for the same shape: the served buffer
/// must carry the *new* payload, not the one the previous submission left there.
#[test]
fn a_served_buffer_carries_the_new_arms_bytes() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let pipeline = compile_copy_word(&provider, &executor);
    provider.set_compute_buffer_pool(true);

    let first = submit(&provider, &pipeline, 0x1122_3344);
    let before = provider.compute_buffer_pool_counts();
    let second = submit(&provider, &pipeline, 0x5566_7788);
    let after = provider.compute_buffer_pool_counts();

    assert_eq!(first, 0x1122_3344);
    assert_eq!(
        second, 0x5566_7788,
        "the served buffer holds this arm's bytes: {second:#010x}"
    );
    assert_eq!(
        after.hits - before.hits,
        UPLOADS_PER_SUBMISSION,
        "the second arm is served rather than rebuilt"
    );
    assert_eq!(
        after.misses - before.misses,
        0,
        "the second arm builds nothing of its own"
    );
}

/// A submission that observed a device loss destroys every pair it took instead
/// of handing it back: the queue's work was never proven retired, and a pair
/// taken out of a device that stopped answering must not come back.
///
/// The pool is filled by one healthy submission first, so the lost one is
/// reading the same shapes: it takes both of them (the counters say so) and the
/// pool must come out of it empty rather than holding anything.
#[test]
fn a_lost_submission_destroys_the_pairs_it_took() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let pipeline = compile_copy_word(&provider, &executor);
    provider.set_compute_buffer_pool(true);

    let healthy = submit(&provider, &pipeline, 0x1122_3344);
    let filled = provider.compute_buffer_pool_counts();
    assert_eq!(healthy, 0x1122_3344);
    assert_eq!(
        filled.entries, UPLOADS_PER_SUBMISSION as usize,
        "the healthy submission's shapes are the ones the lost one will take"
    );

    // The substituted driver answer fails the submission through the real loss
    // path, after the build has already taken the two pairs.
    executor.inject_driver_device_loss_for_test(DeviceLossPoint::Submit);
    let error = provider
        .submit(admit(&provider, trace(&provider, &pipeline, 0x1122_3344)))
        .expect_err("the substituted driver answer refuses the submission");
    assert_eq!(error.class, ProviderErrorClass::DeviceLost);

    let after = provider.compute_buffer_pool_counts();
    eprintln!(
        "compute pool after a device loss: hits {} -> {}, entries {} -> {}, returns {}, \
         evictions {}, flushes {}",
        filled.hits,
        after.hits,
        filled.entries,
        after.entries,
        after.returns,
        after.evictions,
        after.flushes,
    );
    assert_eq!(
        after.hits - filled.hits,
        UPLOADS_PER_SUBMISSION,
        "the lost submission took the shapes before the driver answered"
    );
    assert_eq!(
        after.returns - filled.returns,
        0,
        "a lost submission hands nothing back"
    );
    assert_eq!(
        after.entries, 0,
        "the pairs a lost submission took are destroyed, not held"
    );
    assert_eq!(
        (after.evictions, after.flushes),
        (filled.evictions, filled.flushes),
        "the destruction is neither an eviction nor a switch going off"
    );
}

/// The indirect replay's own twelve-byte `INDIRECT_BUFFER` is one more shape
/// this pool serves: a second indirect submission of the same dispatch takes
/// all three of its pairs — the two owned views and the command — rather than
/// building any of them again.
#[test]
fn the_indirect_replays_buffer_is_pooled_too() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let pipeline = compile_copy_word(&provider, &executor);
    provider.set_compute_buffer_pool(true);

    let indirect = IndirectCommandPayload {
        buffer: IndirectCommandBufferDescriptor {
            max_commands: 1,
            kinds: vec![IndirectCommandKind::Dispatch],
        },
        command: IndirectCommandDescriptor::Dispatch {
            threadgroups: [1, 1, 1],
        },
        range: IndirectCommandRange { start: 0, count: 1 },
    };
    // The two owned views plus the command: three shapes, one of them the
    // indirect replay's own.
    let per_submission = UPLOADS_PER_SUBMISSION + 1;
    let run = |word: u32| {
        let trace = ComputeTrace {
            indirect: Some(Box::new(indirect.clone())),
            ..trace(&provider, &pipeline, word)
        };
        let admitted = provider
            .capabilities()
            .validate_trace(trace.clone(), resources(&provider))
            .expect("the indirect trace is admitted");
        let submitted = provider
            .submit(admitted)
            .expect("the indirect submission completes");
        submitted
            .validate_for_trace(&trace)
            .expect("the writebacks cover the trace");
        submitted
            .writebacks
            .into_iter()
            .find(|writeback| writeback.view_id == OUTPUT_VIEW)
            .expect("the pass writes its output buffer")
            .bytes
    };

    let before = provider.compute_buffer_pool_counts();
    let first = run(0x1122_3344);
    let after_first = provider.compute_buffer_pool_counts();
    let second = run(0x1122_3344);
    let after_second = provider.compute_buffer_pool_counts();

    eprintln!(
        "compute pool indirect on: hits {} -> {} -> {}, misses {}, entries {}, held {} bytes",
        before.hits,
        after_first.hits,
        after_second.hits,
        after_second.misses,
        after_second.entries,
        after_second.held_bytes,
    );
    assert_eq!(first, second, "the two indirect arms land the same word");
    assert_eq!(
        first,
        0x1122_3344u32.to_le_bytes().to_vec(),
        "the replay reads the same input word as the direct path"
    );
    assert_eq!(
        after_second.hits - after_first.hits,
        per_submission,
        "the second indirect submission takes both views and the command"
    );
    assert_eq!(
        after_first.misses - before.misses,
        per_submission,
        "the first indirect submission builds both views and the command"
    );
    assert_eq!(
        after_second.entries, per_submission as usize,
        "the pool holds one pair per shape the indirect submission states"
    );
}
