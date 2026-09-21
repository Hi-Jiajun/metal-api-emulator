//! Native heap rail (`research/docs/25-heaps与ICB设计.md` §6 Step 7a).
//!
//! The native provider places the trace's owned allocations into one shared
//! `MTLBuffer` slab instead of per-allocation buffers, mirroring the Vulkan
//! rail's single `VkDeviceMemory` slab plus per-allocation offset binding. The
//! planning half is pure and is compiled both for the macOS provider and for
//! the host-side unit tests that pin it on a machine that cannot load Metal;
//! only the slab encode body in `native.rs` needs a device.
//!
//! The capability bits stay closed until the Apple `--heap-selftest` produces
//! the single-device evidence the flip condition names; `heap_capability_bits`
//! is the one spelling of that condition so the snapshot and the tests cannot
//! drift from it.
#![allow(dead_code)] // heap bits stay closed until the Apple selftest flips them

use crate::refusal;
use metal_api_core::provider::{
    AllocationId, BufferSource, BufferView, ComputeTrace, FieldValue, HeapId, HeapResource,
    ProviderError, ProviderErrorClass, ProviderPhase, ResourceTableSnapshot, StorageMode,
};
use std::collections::{BTreeMap, BTreeSet};

/// One heap placement the native provider executed: which heap a resource
/// landed in, which allocation it belongs to, and the byte range it occupies
/// there (`research/docs/25` §6 Step 3). Two resources placed in one heap are
/// reported as two records sharing one `heap_id`, not merely "looks shared".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeapPlacementObservation {
    pub heap_id: HeapId,
    pub allocation_id: AllocationId,
    pub offset: u64,
    pub byte_size: u64,
}

/// The heap bits this provider declares, in one value so the macOS capability
/// snapshot (`native.rs`) and the host-side unit tests cannot drift.
///
/// Flip evidence (`research/docs/25` §6 Step 7a): CI run `34838302150`, job
/// `native-oracle-build`, step "Run native heap self-test when a Metal device
/// is eligible". The probe reported an eligible Apple Paravirtual device and
/// the oracle allocated two buffers from one `MTLHeap`, ran the reviewed
/// `copy_word` kernel and read `heap_selftest: PASS (fefefefe)` back — the
/// copied word, not the `ffffffff` sentinel the write buffer was preset with.
/// A green job whose log said `SKIP` is not that evidence. Before the flip
/// every field stayed at its default, so core admission refused a heap-bearing
/// trace with `heap_unsupported` instead of silently dropping the placement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HeapCapabilityBits {
    pub(crate) supports_heaps: bool,
    pub(crate) max_heap_bytes: u64,
    pub(crate) supported_heap_storage_modes: Vec<StorageMode>,
    pub(crate) supports_heap_aliasing: bool,
}

/// The first heap increment's bits: one owned-byte slab, no aliasing, and the
/// same 64 MiB ceiling the Vulkan rail declares.
pub(crate) fn heap_capability_bits() -> HeapCapabilityBits {
    HeapCapabilityBits {
        supports_heaps: true,
        max_heap_bytes: 64 * 1024 * 1024,
        supported_heap_storage_modes: vec![StorageMode::OwnedBytes],
        supports_heap_aliasing: false,
    }
}

/// The provider-side half of a heap payload's mapping to the trace's owned
/// allocations: the slab size and, for each owned allocation, its placement
/// offset and full byte size.
#[derive(Debug)]
pub(crate) struct HeapPlan {
    pub(crate) slab_size: u64,
    pub(crate) offsets: BTreeMap<AllocationId, u64>,
    pub(crate) sizes: BTreeMap<AllocationId, u64>,
    pub(crate) observations: Vec<HeapPlacementObservation>,
}

/// Map a trace's heap payload onto its owned allocations and record the
/// placement observations (`research/docs/25` §6 Step 3).
///
/// The mapping is the same one the Vulkan rail uses: the trace's distinct
/// owned allocations in ascending identity order, zipped one-to-one with the
/// payload's placements. Staged and borrowed views keep their own backing and
/// are not part of the heap, so a count or size mismatch is a typed refusal
/// rather than a silent drop.
pub(crate) fn plan_heap_placements(
    trace: &ComputeTrace,
    pool: &[BufferView],
    resources: &ResourceTableSnapshot,
) -> Result<Option<HeapPlan>, ProviderError> {
    let Some(heap) = &trace.heap else {
        return Ok(None);
    };
    // The first increment binds buffers only; a texture placement has no Metal
    // heap binding yet (`research/docs/25` §6 Step 7).
    if let Some(placement) = heap
        .placements
        .iter()
        .find(|placement| matches!(placement.resource, HeapResource::Texture { .. }))
    {
        return Err(refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Capability,
            "heap_placement_unsupported",
        )
        .with_field("resource", FieldValue::Text("texture".to_string()))
        .with_field("heap", FieldValue::Unsigned(placement.heap_id.get())));
    }
    let mut owned = BTreeSet::<AllocationId>::new();
    for resource in pool {
        // The zero-fill declaration is placed like the trace's own payloads,
        // so a heap trace that states it is planned by the same rule the arm it
        // replaced would take (`BufferSource::ZeroFill`, statement economy
        // W2-A). The arm itself is refused later, by its own name, in the
        // resolve walk rather than by this count.
        if matches!(
            resource.source,
            BufferSource::OwnedBytes(_) | BufferSource::ZeroFill { .. }
        ) {
            owned.insert(resource.allocation_id);
        }
    }
    let owned: Vec<AllocationId> = owned.into_iter().collect();
    if heap.placements.len() != owned.len() {
        return Err(refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Args,
            "heap_placement_mismatch",
        )
        .with_field(
            "placements",
            FieldValue::Unsigned(heap.placements.len() as u64),
        )
        .with_field("allocations", FieldValue::Unsigned(owned.len() as u64)));
    }
    let mut heap_ids = BTreeSet::<u64>::new();
    let mut offsets = BTreeMap::<AllocationId, u64>::new();
    let mut sizes = BTreeMap::<AllocationId, u64>::new();
    let mut observations = Vec::with_capacity(owned.len());
    for (placement, allocation) in heap.placements.iter().zip(owned.iter()) {
        heap_ids.insert(placement.heap_id.get());
        let record = resources.allocation(*allocation).ok_or_else(|| {
            refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Resource,
                "heap_placement_mismatch",
            )
            .with_field("allocation", FieldValue::Unsigned(allocation.get()))
        })?;
        if placement.resource.byte_size() != record.size {
            return Err(refusal(
                ProviderPhase::Resolve,
                ProviderErrorClass::Args,
                "heap_placement_mismatch",
            )
            .with_field("allocation", FieldValue::Unsigned(allocation.get()))
            .with_field(
                "placement_size",
                FieldValue::Unsigned(placement.resource.byte_size()),
            )
            .with_field("allocation_size", FieldValue::Unsigned(record.size)));
        }
        offsets.insert(*allocation, placement.offset);
        sizes.insert(*allocation, record.size);
        observations.push(HeapPlacementObservation {
            heap_id: placement.heap_id,
            allocation_id: *allocation,
            offset: placement.offset,
            byte_size: placement.resource.byte_size(),
        });
    }
    if heap_ids.len() != 1 {
        return Err(refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Args,
            "heap_placement_mismatch",
        )
        .with_field(
            "distinct_heaps",
            FieldValue::Unsigned(heap_ids.len() as u64),
        ));
    }
    Ok(Some(HeapPlan {
        slab_size: heap.descriptor.size,
        offsets,
        sizes,
        observations,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::provider::{
        AllocationRecord, BufferAccess, BufferBindingContract, BufferView, CompiledComputePipeline,
        CompletionPolicy, ComputePass, ComputeTrace, DeviceEpoch, Dispatch, DispatchKind,
        DispatchType, FootprintProof, FunctionIdentity, FunctionSource, HeapDescriptor,
        HeapPayload, HeapPlacement, OperationId, PipelineContract, PipelineId, SemanticDigest,
        TracePass, ViewId, PROVIDER_SCHEMA_VERSION,
    };

    fn snapshot(sizes: &[(u64, u64)]) -> ResourceTableSnapshot {
        let mut snapshot = ResourceTableSnapshot::new();
        for (allocation, size) in sizes {
            snapshot
                .insert_allocation(AllocationRecord {
                    allocation_id: AllocationId::new(*allocation),
                    owner_epoch: DeviceEpoch::new(1),
                    size: *size,
                })
                .unwrap();
        }
        snapshot
    }

    fn trace_with(heap: Option<HeapPayload>, buffers: Vec<BufferView>) -> ComputeTrace {
        ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(1),
            operation_id: OperationId::new(1),
            pipelines: vec![CompiledComputePipeline {
                device_epoch: DeviceEpoch::new(1),
                pipeline_id: PipelineId::new(1),
                function: FunctionIdentity {
                    logical_digest: SemanticDigest::new("heap-fixture", vec![1]).unwrap(),
                    entry_name: "copy_word".to_owned(),
                    source: FunctionSource::MetalSource,
                },
                contract: PipelineContract {
                    dispatch_kind: DispatchKind::ThreadsExact,
                    required_local_size: None,
                    fixed_grid: None,
                    push_constant_offset: 0,
                    push_constant_bytes: 0,
                    buffer_bindings: vec![
                        BufferBindingContract {
                            metal_binding: 0,
                            access: BufferAccess::Read,
                            footprint: FootprintProof::Static { max_bytes: 4 },
                        },
                        BufferBindingContract {
                            metal_binding: 1,
                            access: BufferAccess::Write,
                            footprint: FootprintProof::Static { max_bytes: 4 },
                        },
                    ],
                    texture_bindings: Vec::new(),
                    shader_capabilities: Vec::new(),
                    translator_revision: None,
                },
                render: None,
            }],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![TracePass::Compute(ComputePass {
                pipeline: PipelineId::new(1),
                buffers,
                dispatch: Dispatch {
                    kind: DispatchKind::ThreadsExact,
                    grid: [1, 1, 1],
                    threads_per_threadgroup: [1, 1, 1],
                },
                textures: Vec::new(),
            })],
            completion_policy: CompletionPolicy::HostReadback,
            heap: heap.map(Box::new),
            indirect: None,
        }
    }

    fn view(
        view_id: u64,
        metal_binding: u32,
        allocation_id: u64,
        offset: u64,
        length: u64,
        access: BufferAccess,
    ) -> BufferView {
        BufferView {
            view_id: ViewId::new(view_id),
            metal_binding,
            allocation_id: AllocationId::new(allocation_id),
            offset,
            length,
            access,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0_u8; length as usize]),
        }
    }

    fn heap_payload(placements: Vec<(u64, u64, u64)>) -> HeapPayload {
        HeapPayload {
            descriptor: HeapDescriptor {
                size: 512,
                storage_mode: StorageMode::OwnedBytes,
                allows_aliasing: false,
            },
            placements: placements
                .into_iter()
                .map(|(_allocation, offset, size)| HeapPlacement {
                    heap_id: HeapId::new(7),
                    offset,
                    resource: HeapResource::Buffer { byte_size: size },
                })
                .collect(),
        }
    }

    #[test]
    fn no_heap_payload_is_no_plan() {
        let trace = trace_with(
            None,
            vec![
                view(1, 0, 1, 0, 4, BufferAccess::Read),
                view(2, 1, 2, 0, 4, BufferAccess::Write),
            ],
        );
        let pool = trace.serial_resources().unwrap();
        let resources = snapshot(&[(1, 4), (2, 4)]);
        assert!(plan_heap_placements(&trace, &pool, &resources)
            .unwrap()
            .is_none());
    }

    #[test]
    fn zips_owned_allocations_one_to_one_and_records_offsets() {
        let trace = trace_with(
            Some(heap_payload(vec![(900, 0, 16), (920, 256, 12)])),
            vec![
                view(910, 0, 900, 0, 16, BufferAccess::Read),
                view(930, 1, 920, 4, 4, BufferAccess::Write),
            ],
        );
        let pool = trace.serial_resources().unwrap();
        let resources = snapshot(&[(900, 16), (920, 12)]);
        let plan = plan_heap_placements(&trace, &pool, &resources)
            .unwrap()
            .expect("heap payload plans");
        assert_eq!(plan.slab_size, 512);
        assert_eq!(plan.offsets[&AllocationId::new(900)], 0);
        assert_eq!(plan.offsets[&AllocationId::new(920)], 256);
        assert_eq!(plan.sizes[&AllocationId::new(900)], 16);
        assert_eq!(plan.sizes[&AllocationId::new(920)], 12);
        assert_eq!(
            plan.observations,
            vec![
                HeapPlacementObservation {
                    heap_id: HeapId::new(7),
                    allocation_id: AllocationId::new(900),
                    offset: 0,
                    byte_size: 16,
                },
                HeapPlacementObservation {
                    heap_id: HeapId::new(7),
                    allocation_id: AllocationId::new(920),
                    offset: 256,
                    byte_size: 12,
                },
            ]
        );
    }

    #[test]
    fn texture_placement_is_refused() {
        let mut payload = heap_payload(vec![(900, 0, 16)]);
        payload.placements[0].resource = HeapResource::Texture { byte_size: 16 };
        let trace = trace_with(
            Some(payload),
            vec![
                view(910, 0, 900, 0, 16, BufferAccess::Read),
                view(930, 1, 920, 0, 4, BufferAccess::Write),
            ],
        );
        let pool = trace.serial_resources().unwrap();
        let resources = snapshot(&[(900, 16), (920, 4)]);
        let error = plan_heap_placements(&trace, &pool, &resources).unwrap_err();
        assert_eq!(error.slug, "heap_placement_unsupported");
    }

    #[test]
    fn placement_count_mismatch_is_refused() {
        let trace = trace_with(
            Some(heap_payload(vec![(900, 0, 16), (920, 256, 12)])),
            vec![
                view(910, 0, 900, 0, 16, BufferAccess::Read),
                view(930, 1, 920, 0, 4, BufferAccess::Write),
            ],
        );
        let pool = trace.serial_resources().unwrap();
        let resources = snapshot(&[(900, 16), (920, 4)]);
        let error = plan_heap_placements(&trace, &pool, &resources).unwrap_err();
        assert_eq!(error.slug, "heap_placement_mismatch");
    }

    #[test]
    fn allocation_size_mismatch_is_refused() {
        let trace = trace_with(
            Some(heap_payload(vec![(900, 0, 16)])),
            vec![
                view(910, 0, 900, 0, 16, BufferAccess::Read),
                view(930, 1, 920, 0, 4, BufferAccess::Write),
            ],
        );
        let pool = trace.serial_resources().unwrap();
        let resources = snapshot(&[(900, 32), (920, 4)]);
        let error = plan_heap_placements(&trace, &pool, &resources).unwrap_err();
        assert_eq!(error.slug, "heap_placement_mismatch");
    }

    #[test]
    fn multiple_distinct_heaps_are_refused() {
        let trace = trace_with(
            Some(HeapPayload {
                descriptor: HeapDescriptor {
                    size: 512,
                    storage_mode: StorageMode::OwnedBytes,
                    allows_aliasing: false,
                },
                placements: vec![
                    HeapPlacement {
                        heap_id: HeapId::new(7),
                        offset: 0,
                        resource: HeapResource::Buffer { byte_size: 16 },
                    },
                    HeapPlacement {
                        heap_id: HeapId::new(8),
                        offset: 256,
                        resource: HeapResource::Buffer { byte_size: 12 },
                    },
                ],
            }),
            vec![
                view(910, 0, 900, 0, 16, BufferAccess::Read),
                view(930, 1, 920, 4, 4, BufferAccess::Write),
            ],
        );
        let pool = trace.serial_resources().unwrap();
        let resources = snapshot(&[(900, 16), (920, 12)]);
        let error = plan_heap_placements(&trace, &pool, &resources).unwrap_err();
        assert_eq!(error.slug, "heap_placement_mismatch");
    }
}
