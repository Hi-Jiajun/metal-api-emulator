//! The compute half's shape-decided pipeline reuse's own rail cases
//! (`crate::compute_pipeline_reuse`).
//!
//! A submission hands the next submission of the *same shape* the module, the
//! descriptor-set layout, the pipeline layout and the pipelines a previous
//! submission built. That is only allowed to be a timing change, so this file
//! is the byte-level oracle the increment's own report reads:
//!
//! * one dispatch runs twice against one provider: the second submission must
//!   be served from the table and must publish the *same* word;
//! * a **different kernel** must never be served another module's group — its
//!   SPIR-V words and its three-binding layout are a different shape — and the
//!   first kernel's own entry must still be there afterwards, which is the one
//!   failure direction a shape table has ("the next submission runs through
//!   somebody else's pipeline");
//! * the same dispatches must publish the same bytes with the mechanism
//!   switched **off**, so the two arms of the reading are compared with each
//!   other and not only with themselves;
//! * a submission that observed a device loss must destroy the group it took
//!   instead of handing it back.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferAccess, BufferSource, BufferView,
    CompiledComputePipeline, CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace,
    Dispatch, DispatchKind, DispatchType, OperationId, ProviderErrorClass, ResourceTableSnapshot,
    SemanticDigest, TracePass, ValidatedComputeTrace, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    ComputePipelineReuseCounts, DeviceLossPoint, VulkanComputeProvider, VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed fixture the copy arms run: one `uint` in, one `uint` out, so
/// the word the pass landed is the word it read.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");
const COPY_WORD_ENTRY: &str = "copy_word";

/// A second reviewed fixture, and the shape the table must not confuse with the
/// first: three bindings rather than two, and a different module. It lands its
/// *witness* word, so an arm that ran through the wrong pipeline lands a
/// different word rather than the same one by luck.
const WITNESS_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word_with_witness.ll");
const WITNESS_ENTRY: &str = "copy_word_with_witness";

const INPUT_VIEW: ViewId = ViewId::new(8420);
const INPUT_ALLOCATION: AllocationId = AllocationId::new(8421);
const OUTPUT_VIEW: ViewId = ViewId::new(8422);
const OUTPUT_ALLOCATION: AllocationId = AllocationId::new(8423);
const WITNESS_VIEW: ViewId = ViewId::new(8424);
const WITNESS_ALLOCATION: AllocationId = AllocationId::new(8425);

/// One submission builds exactly one group: `create_pipeline_objects` states
/// one `PipelineObjects` per compute pass, and this fixture states one pass.
const GROUPS_PER_SUBMISSION: u64 = 1;

fn digest(case: &[u8]) -> SemanticDigest {
    SemanticDigest::new("compute-pipeline-reuse-fixture-v1", case.to_vec()).expect("digest")
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

fn compile(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
    air: &str,
    entry: &str,
    case: &[u8],
) -> CompiledComputePipeline {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let function = device
        .new_library_with_air(air)
        .expect("the fixture library loads")
        .function(entry)
        .expect("the fixture entry exists");
    provider
        .compile_pipeline(&function, digest(case))
        .expect("the fixture pipeline registers")
}

fn buffer_view(
    view_id: ViewId,
    allocation_id: AllocationId,
    metal_binding: u32,
    access: BufferAccess,
    bytes: Vec<u8>,
) -> BufferView {
    BufferView {
        view_id,
        metal_binding,
        allocation_id,
        offset: 0,
        length: 4,
        access,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(bytes),
    }
}

/// One `copy_word` pass whose read word is `word` and whose written word starts
/// as the same four zero bytes every arm states.
fn copy_trace(
    provider: &VulkanComputeProvider,
    pipeline: &CompiledComputePipeline,
    word: u32,
) -> ComputeTrace {
    ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(84),
        pipelines: vec![pipeline.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![TracePass::Compute(ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers: vec![
                buffer_view(
                    INPUT_VIEW,
                    INPUT_ALLOCATION,
                    0,
                    BufferAccess::Read,
                    word.to_le_bytes().to_vec(),
                ),
                buffer_view(
                    OUTPUT_VIEW,
                    OUTPUT_ALLOCATION,
                    1,
                    BufferAccess::Write,
                    vec![0_u8; 4],
                ),
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

/// The three-binding fixture's pass. Its output lands the witness word, so the
/// bytes say which module ran.
fn witness_trace(
    provider: &VulkanComputeProvider,
    pipeline: &CompiledComputePipeline,
    input: u32,
    witness: u32,
) -> ComputeTrace {
    ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(85),
        pipelines: vec![pipeline.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![TracePass::Compute(ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers: vec![
                buffer_view(
                    INPUT_VIEW,
                    INPUT_ALLOCATION,
                    0,
                    BufferAccess::Read,
                    input.to_le_bytes().to_vec(),
                ),
                buffer_view(
                    OUTPUT_VIEW,
                    OUTPUT_ALLOCATION,
                    1,
                    BufferAccess::Write,
                    vec![0_u8; 4],
                ),
                buffer_view(
                    WITNESS_VIEW,
                    WITNESS_ALLOCATION,
                    2,
                    BufferAccess::Read,
                    witness.to_le_bytes().to_vec(),
                ),
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
    for allocation in [INPUT_ALLOCATION, OUTPUT_ALLOCATION, WITNESS_ALLOCATION] {
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

fn admit(provider: &VulkanComputeProvider, trace: ComputeTrace) -> ValidatedComputeTrace {
    provider
        .capabilities()
        .validate_trace(trace, resources(provider))
        .expect("the fixture trace is admitted")
}

/// Submit one trace and return the word its writeback landed.
fn run(provider: &VulkanComputeProvider, trace: ComputeTrace) -> u32 {
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

/// One dispatch runs twice: the second submission takes the group the first one
/// built, and the word it lands is the word the fresh path landed.
#[test]
fn a_repeated_dispatch_is_served_from_the_table_and_lands_the_same_word() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let pipeline = compile(
        &provider,
        &executor,
        COPY_WORD_AIR,
        COPY_WORD_ENTRY,
        b"compute-pipeline-reuse-copy-word",
    );

    // Arm one: the mechanism on, which is its default.
    provider.set_compute_pipeline_reuse(true);
    let before = provider.compute_pipeline_reuse_counts();
    let first = run(&provider, copy_trace(&provider, &pipeline, 0x1122_3344));
    let after_first = provider.compute_pipeline_reuse_counts();
    let second = run(&provider, copy_trace(&provider, &pipeline, 0x1122_3344));
    let after_second = provider.compute_pipeline_reuse_counts();

    eprintln!(
        "compute pipeline reuse on: first {first:#010x} second {second:#010x}; hits {} -> {} -> {}, \
         misses {}, returns {}, entries {}, evictions {}",
        before.hits,
        after_first.hits,
        after_second.hits,
        after_second.misses,
        after_second.returns,
        after_second.entries,
        after_second.evictions,
    );
    assert_eq!(first, 0x1122_3344, "the fresh path lands the word it read");
    assert_eq!(
        second, first,
        "a submission served from the table lands exactly the word the fresh path landed"
    );
    assert_eq!(
        after_first.hits - before.hits,
        0,
        "the first submission of a shape has nothing to take"
    );
    assert_eq!(
        after_first.misses - before.misses,
        GROUPS_PER_SUBMISSION,
        "the first submission builds its own group"
    );
    assert_eq!(
        after_first.returns - before.returns,
        GROUPS_PER_SUBMISSION,
        "a submission whose fence was observed hands its group back"
    );
    assert_eq!(
        after_second.hits - after_first.hits,
        GROUPS_PER_SUBMISSION,
        "the second submission takes the group the first one handed back"
    );
    assert_eq!(
        after_second.misses - after_first.misses,
        0,
        "a served creation is never also counted as a build"
    );
    assert_eq!(
        after_second.entries, GROUPS_PER_SUBMISSION as usize,
        "the table holds one group per shape this fixture states"
    );
    assert_eq!(
        (after_second.evictions, after_second.flushes),
        (before.evictions, before.flushes),
        "one fixture is far below the cap and no contract surface moved"
    );

    // Arm two: the same two dispatches with the mechanism switched off. The
    // switch drops what it held, and the fresh path lands the same bytes.
    provider.set_compute_pipeline_reuse(false);
    let off_before = provider.compute_pipeline_reuse_counts();
    assert_eq!(
        off_before.entries, 0,
        "switching the mechanism off drops what it held"
    );
    let third = run(&provider, copy_trace(&provider, &pipeline, 0x1122_3344));
    let fourth = run(&provider, copy_trace(&provider, &pipeline, 0x1122_3344));
    let off_after = provider.compute_pipeline_reuse_counts();

    eprintln!(
        "compute pipeline reuse off: third {third:#010x} fourth {fourth:#010x}; hits {} -> {}, \
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
        2 * GROUPS_PER_SUBMISSION,
        "every creation of both submissions reports the switch, not a miss"
    );
    assert_eq!(off_after.entries, 0, "the switch off holds nothing");
    assert_eq!(
        off_after.returns - off_before.returns,
        0,
        "nothing is handed back while the switch is off"
    );
    assert_eq!(
        off_after.misses, off_before.misses,
        "a disabled creation is never also counted as a miss"
    );
}

/// A second kernel is a second shape. Its submission must build its own group
/// (and land its own word) rather than take the first kernel's, and the first
/// kernel's entry must still be there: two modules never meet in one entry.
#[test]
fn a_different_kernel_is_never_served_another_modules_group() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let copy = compile(
        &provider,
        &executor,
        COPY_WORD_AIR,
        COPY_WORD_ENTRY,
        b"compute-pipeline-reuse-copy-word",
    );
    let witness = compile(
        &provider,
        &executor,
        WITNESS_AIR,
        WITNESS_ENTRY,
        b"compute-pipeline-reuse-witness",
    );
    provider.set_compute_pipeline_reuse(true);

    // The first kernel's shape is resident after its own submission.
    let copy_first = run(&provider, copy_trace(&provider, &copy, 0x1122_3344));
    let before = provider.compute_pipeline_reuse_counts();
    assert_eq!(copy_first, 0x1122_3344);
    assert_eq!(
        before.entries, 1,
        "the copy kernel's shape is what the second kernel must not take"
    );

    // The second kernel's first submission builds its own group and lands its
    // own word: the witness, not the input.
    let witness_first = run(
        &provider,
        witness_trace(&provider, &witness, 0x1122_3344, 0xAAAA_BBBB),
    );
    let after_witness = provider.compute_pipeline_reuse_counts();
    eprintln!(
        "two shapes: witness landed {witness_first:#010x}; hits {} -> {}, misses {} -> {}, \
         mismatches {}, entries {}",
        before.hits,
        after_witness.hits,
        before.misses,
        after_witness.misses,
        after_witness.mismatches,
        after_witness.entries,
    );
    assert_eq!(
        witness_first, 0xAAAA_BBBB,
        "the witness kernel runs its own module and lands its own word"
    );
    assert_eq!(
        after_witness.hits - before.hits,
        0,
        "a different module never takes another module's group"
    );
    assert_eq!(
        after_witness.misses - before.misses,
        GROUPS_PER_SUBMISSION,
        "the second shape builds its own group"
    );
    assert_eq!(
        after_witness.entries, 2,
        "both shapes are resident under their own keys"
    );

    // Both shapes then repeat: each takes its own entry back and lands its own
    // word, which is the arm that would fail if the two collided.
    let copy_second = run(&provider, copy_trace(&provider, &copy, 0x5566_7788));
    let witness_second = run(
        &provider,
        witness_trace(&provider, &witness, 0x5566_7788, 0xCCDD_EEFF),
    );
    let after = provider.compute_pipeline_reuse_counts();
    assert_eq!(
        copy_second, 0x5566_7788,
        "the copy kernel still lands the word it read"
    );
    assert_eq!(
        witness_second, 0xCCDD_EEFF,
        "the witness kernel still lands its own witness"
    );
    assert_eq!(
        after.hits - after_witness.hits,
        2 * GROUPS_PER_SUBMISSION,
        "each shape is served from its own entry"
    );
    assert_eq!(
        after.misses - after_witness.misses,
        0,
        "neither repeat builds anything"
    );
    assert_eq!(after.entries, 2, "both entries came back");
}

/// A submission that observed a device loss destroys the group it took instead
/// of handing it back: the queue's work was never proven retired, and objects
/// taken out of a device that stopped answering must not come back.
///
/// The table is filled by one healthy submission first, so the lost one is
/// reading the same shape: it takes the entry (the counters say so) and the
/// table must come out of it empty rather than holding anything.
#[test]
fn a_lost_submission_destroys_the_group_it_took() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let pipeline = compile(
        &provider,
        &executor,
        COPY_WORD_AIR,
        COPY_WORD_ENTRY,
        b"compute-pipeline-reuse-copy-word",
    );
    provider.set_compute_pipeline_reuse(true);

    let healthy = run(&provider, copy_trace(&provider, &pipeline, 0x1122_3344));
    let filled = provider.compute_pipeline_reuse_counts();
    assert_eq!(healthy, 0x1122_3344);
    assert_eq!(
        filled.entries, GROUPS_PER_SUBMISSION as usize,
        "the healthy submission's shape is the one the lost one will take"
    );

    // The substituted driver answer fails the submission through the real loss
    // path, after the creation has already taken the group.
    executor.inject_driver_device_loss_for_test(DeviceLossPoint::Submit);
    let error = provider
        .submit(admit(
            &provider,
            copy_trace(&provider, &pipeline, 0x1122_3344),
        ))
        .expect_err("the substituted driver answer refuses the submission");
    assert_eq!(error.class, ProviderErrorClass::DeviceLost);

    let after = provider.compute_pipeline_reuse_counts();
    eprintln!(
        "compute pipeline reuse after a device loss: hits {} -> {}, entries {} -> {}, \
         returns {}, dropped {}",
        filled.hits, after.hits, filled.entries, after.entries, after.returns, after.dropped,
    );
    assert_eq!(
        after.hits - filled.hits,
        GROUPS_PER_SUBMISSION,
        "the lost submission took the shape before the driver answered"
    );
    assert_eq!(
        after.returns - filled.returns,
        0,
        "a lost submission hands nothing back"
    );
    assert_eq!(
        after.entries, 0,
        "the group a lost submission took is destroyed, not held"
    );
    assert_eq!(
        (after.evictions, after.flushes),
        (filled.evictions, filled.flushes),
        "the destruction is neither an eviction nor a switch going off"
    );
}

/// The counters one healthy run leaves behind are the whole partition the
/// profile line prints, read from the provider rather than from a window: every
/// creation is a hit, a miss, a mismatch or a disabled ask, and every hand-back
/// is a return or a drop.
#[test]
fn the_counters_partition_every_creation_and_hand_back() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let pipeline = compile(
        &provider,
        &executor,
        COPY_WORD_AIR,
        COPY_WORD_ENTRY,
        b"compute-pipeline-reuse-copy-word",
    );
    provider.set_compute_pipeline_reuse(true);
    let before: ComputePipelineReuseCounts = provider.compute_pipeline_reuse_counts();
    for word in [0x0102_0304, 0x0506_0708, 0x090A_0B0C] {
        assert_eq!(run(&provider, copy_trace(&provider, &pipeline, word)), word);
    }
    let after = provider.compute_pipeline_reuse_counts();

    let creations = (after.hits - before.hits)
        + (after.misses - before.misses)
        + (after.mismatches - before.mismatches)
        + (after.disabled - before.disabled);
    let hand_backs = (after.returns - before.returns) + (after.dropped - before.dropped);
    eprintln!(
        "three submissions: creations {creations}, hand-backs {hand_backs}, hits {}, misses {}, \
         entries {}",
        after.hits, after.misses, after.entries,
    );
    assert_eq!(
        creations,
        3 * GROUPS_PER_SUBMISSION,
        "three submissions state three creations"
    );
    assert_eq!(
        hand_backs,
        3 * GROUPS_PER_SUBMISSION,
        "every observed submission hands its group back"
    );
    assert_eq!(
        after.mismatches, before.mismatches,
        "a digest collision is not what served these submissions"
    );
    assert_eq!(after.entries, 1, "one shape stays resident");
}
