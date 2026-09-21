//! The provider's half of the statement payload table (statement economy W4,
//! task E-SW3).
//!
//! The table itself is the contract's (`metal_api_core::statement_payload`):
//! one slot per payload a statement filed, scoped to a device epoch, with the
//! caps both ends read. This module is what a *provider* does with it — resolve
//! the two arms a decoded statement may carry, in the trace's own wire order:
//!
//! - `TextureSource::OwnedInSlot` carries its bytes and a slot they are filed
//!   under. The declaration becomes the ordinary `OwnedBytes` arm here, so
//!   nothing downstream of this module can tell the two apart, and the bytes
//!   are staged for the table.
//! - `TextureSource::SlottedBytes` carries a slot, a length and a digest. The
//!   bytes come out of the table and the arm again becomes `OwnedBytes`; a slot
//!   the table does not hold, or one whose length or digest disagrees, is a
//!   refusal **by name** — the arm names bytes the statement did not state, and
//!   sampling anything else would be a frame no statement asked for.
//!
//! # Why the staging is not an optimization
//!
//! A statement's own declarations are applied **after all of them resolve**.
//! A refusal part way through a statement therefore leaves the table exactly as
//! it was, which is what the sender's ledger does with a statement whose frame
//! did not resolve on this side: both ends stay equal after every statement
//! both ends processed. Applying as we go would leave the provider holding
//! entries the sender never committed, and the *next* statement that referenced
//! one would be refused — a lost draw bought by an optimization.

use metal_api_core::provider::{
    ComputeTrace, FieldValue, ProviderError, ProviderErrorClass, ProviderPhase, TextureSource,
};
use metal_api_core::statement_payload::{PayloadLookup, PayloadTable, PayloadTableFull};

/// What this provider's table has done over its own life, for the rail's own
/// readings and its tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StatementPayloadCounts {
    /// Declarations that filed their bytes in the table.
    pub declare_n: u64,
    /// Their bytes.
    pub declare_bytes: u64,
    /// References the table resolved.
    pub reference_n: u64,
    /// The bytes those references named — the payload the wire did not carry.
    pub reference_bytes: u64,
    /// References whose slot the table did not hold.
    pub unknown_n: u64,
    /// References whose slot held other bytes.
    pub mismatch_n: u64,
    /// Declarations the table's own bounds refused (a wiring finding: the
    /// sender's ledger holds to the same constants).
    pub full_n: u64,
}

/// Resolve one decoded statement's payload-table arms in place.
///
/// Returns whether the statement carried any of them, so a caller can skip the
/// bookkeeping on the statements that carry none — which is every statement of
/// a round that states no such arm, and the shape every round before this
/// increment had.
pub(crate) fn resolve_statement_payloads(
    table: &mut PayloadTable,
    trace: &mut ComputeTrace,
    counts: &mut StatementPayloadCounts,
) -> Result<bool, ProviderError> {
    table.scope(trace.device_epoch);
    let mut staged: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut resolved = false;
    for texture in trace.texture_declarations_mut() {
        // Take the arm first: the rewrite below has to read the bytes it is
        // about to state, and the placeholder is overwritten in every branch
        // that does not refuse.
        let taken = std::mem::replace(&mut texture.source, TextureSource::PassEntrySnapshot);
        match taken {
            TextureSource::OwnedInSlot { slot, bytes } => {
                resolved = true;
                counts.declare_n = counts.declare_n.saturating_add(1);
                counts.declare_bytes = counts
                    .declare_bytes
                    .saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
                staged.push((slot, bytes.clone()));
                texture.source = TextureSource::OwnedBytes(bytes);
            }
            TextureSource::SlottedBytes {
                slot,
                length,
                digest,
            } => {
                resolved = true;
                // A declaration earlier in this same statement is the table's
                // own next state: the wire states it before the reference, and
                // both ends read the statement in that order.
                let held = staged
                    .iter()
                    .find(|(staged_slot, _)| *staged_slot == slot)
                    .map(|(_, bytes)| bytes.as_slice())
                    .map(|bytes| {
                        if crate::statement_payload_table::matches(bytes, length, digest) {
                            Some(bytes)
                        } else {
                            None
                        }
                    });
                let bytes = match held {
                    Some(Some(bytes)) => bytes.to_vec(),
                    Some(None) => return Err(slot_mismatch(&texture, slot, length, digest)),
                    None => match table.lookup(slot, digest, length) {
                        PayloadLookup::Hit(bytes) => {
                            counts.reference_bytes = counts.reference_bytes.saturating_add(length);
                            bytes.to_vec()
                        }
                        PayloadLookup::Unknown => {
                            counts.unknown_n = counts.unknown_n.saturating_add(1);
                            return Err(slot_unknown(&texture, slot, length, digest));
                        }
                        PayloadLookup::Mismatch => {
                            counts.mismatch_n = counts.mismatch_n.saturating_add(1);
                            return Err(slot_mismatch(&texture, slot, length, digest));
                        }
                    },
                };
                counts.reference_n = counts.reference_n.saturating_add(1);
                texture.source = TextureSource::OwnedBytes(bytes);
            }
            other => texture.source = other,
        }
    }
    // Every arm resolved: the statement's own declarations take effect, which
    // is the point after which the two ends hold the same table.
    for (slot, bytes) in staged {
        if let Err(full) = table.declare(slot, bytes) {
            counts.full_n = counts.full_n.saturating_add(1);
            return Err(table_full(slot, full));
        }
    }
    Ok(resolved)
}

/// Whether the staged bytes are the payload a reference names.
fn matches(bytes: &[u8], length: u64, digest: u128) -> bool {
    u64::try_from(bytes.len()).unwrap_or(u64::MAX) == length
        && metal_api_core::statement_payload::payload_digest(bytes) == digest
}

fn refusal(slug: &'static str) -> ProviderError {
    let mut error = ProviderError::new(ProviderPhase::Resolve, ProviderErrorClass::Args, slug)
        .expect("static provider refusal slug");
    error.retryability = metal_api_core::provider::Retryability::Never;
    error
}

fn slot_unknown(
    texture: &metal_api_core::provider::TextureView,
    slot: u32,
    length: u64,
    digest: u128,
) -> ProviderError {
    refusal("statement_payload_slot_unknown")
        .with_field("view", FieldValue::Unsigned(texture.view_id.get()))
        .with_field("slot", FieldValue::Unsigned(u64::from(slot)))
        .with_field("length", FieldValue::Unsigned(length))
        .with_field("digest_hi", FieldValue::Unsigned((digest >> 64) as u64))
        .with_field("digest_lo", FieldValue::Unsigned(digest as u64))
        .with_detail(
            "the declaration names bytes an earlier statement filed in this provider's \
             statement payload table, and this device epoch holds no entry for that slot: an \
             entry is filed by the declaration arm of the statement that carried the bytes, so \
             a slot nothing filed is a statement whose bytes never reached this provider",
        )
}

fn slot_mismatch(
    texture: &metal_api_core::provider::TextureView,
    slot: u32,
    length: u64,
    digest: u128,
) -> ProviderError {
    refusal("statement_payload_slot_mismatch")
        .with_field("view", FieldValue::Unsigned(texture.view_id.get()))
        .with_field("slot", FieldValue::Unsigned(u64::from(slot)))
        .with_field("length", FieldValue::Unsigned(length))
        .with_field("digest_hi", FieldValue::Unsigned((digest >> 64) as u64))
        .with_field("digest_lo", FieldValue::Unsigned(digest as u64))
        .with_detail(
            "the declaration names bytes an earlier statement filed in this provider's \
             statement payload table, and the entry this slot holds is not them: its length or \
             its digest disagrees with the arm. The slot was written by another declaration, \
             so sampling it would be a frame no statement stated",
        )
}

fn table_full(slot: u32, full: PayloadTableFull) -> ProviderError {
    let (bound, held) = match full {
        PayloadTableFull::Slots => ("slots", u64::from(slot)),
        PayloadTableFull::Bytes => ("bytes", u64::from(slot)),
    };
    refusal("statement_payload_table_full")
        .with_field("slot", FieldValue::Unsigned(u64::from(slot)))
        .with_field("bound", FieldValue::Text(bound.to_owned()))
        .with_field("held", FieldValue::Unsigned(held))
        .with_detail(
            "the declaration files its bytes in the statement payload table, and the table's \
             own bound cannot hold them: the bounds are the contract's constants, and the \
             sender's ledger holds to the same two, so this is a wiring finding rather than a \
             second policy",
        )
}

#[cfg(test)]
mod tests {
    use super::{resolve_statement_payloads, StatementPayloadCounts};
    use metal_api_core::provider::{
        AllocationId, CompletionPolicy, ComputePass, ComputeTrace, DeviceEpoch, Dispatch,
        DispatchKind, DispatchType, OperationId, PipelineId, ProviderErrorClass, TextureAccess,
        TextureFormat, TextureSource, TextureType, TextureView, TracePass, ViewId,
        PROVIDER_SCHEMA_VERSION,
    };
    use metal_api_core::statement_payload::{payload_digest, PayloadTable};

    const PAYLOAD: usize = 4096;

    fn texture(view: u64, source: TextureSource) -> TextureView {
        TextureView {
            view_id: ViewId::new(view),
            metal_binding: 0,
            allocation_id: AllocationId::new(view),
            texture_type: TextureType::D2,
            format: TextureFormat::Rgba8Unorm,
            width: u64::try_from(PAYLOAD / 4).unwrap_or(1),
            height: 1,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            access: TextureAccess::Sampled,
            source,
        }
    }

    fn trace(textures: Vec<TextureView>) -> ComputeTrace {
        ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(1),
            operation_id: OperationId::new(1),
            pipelines: Vec::new(),
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![TracePass::Compute(ComputePass {
                pipeline: PipelineId::new(1),
                buffers: Vec::new(),
                textures,
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

    /// The bytes one resolved statement's single declaration states.
    fn resolved_bytes(trace: &ComputeTrace) -> &[u8] {
        match &trace.texture_declarations()[0].source {
            TextureSource::OwnedBytes(bytes) => bytes,
            other => panic!("a resolved statement states bytes, not {other:?}"),
        }
    }

    #[test]
    fn a_reference_resolves_to_the_bytes_an_earlier_statement_filed() {
        let payload = vec![0x5a; PAYLOAD];
        let digest = payload_digest(&payload);
        let mut table = PayloadTable::new();
        let mut counts = StatementPayloadCounts::default();

        let mut declaring = trace(vec![texture(
            1,
            TextureSource::OwnedInSlot {
                slot: 4,
                bytes: payload.clone(),
            },
        )]);
        assert!(
            resolve_statement_payloads(&mut table, &mut declaring, &mut counts)
                .expect("a declaration resolves"),
            "the statement carried an arm"
        );
        assert_eq!(
            resolved_bytes(&declaring),
            payload.as_slice(),
            "the declaration states the bytes it carried, byte for byte"
        );
        assert_eq!(table.entries(), 1);
        assert_eq!(table.used_bytes(), PAYLOAD as u64);

        let mut referencing = trace(vec![texture(
            2,
            TextureSource::SlottedBytes {
                slot: 4,
                length: PAYLOAD as u64,
                digest,
            },
        )]);
        assert!(
            resolve_statement_payloads(&mut table, &mut referencing, &mut counts)
                .expect("a reference the table holds resolves")
        );
        assert_eq!(
            resolved_bytes(&referencing),
            payload.as_slice(),
            "the reference states the bytes the earlier statement filed"
        );
        assert_eq!(counts.declare_n, 1);
        assert_eq!(counts.declare_bytes, PAYLOAD as u64);
        assert_eq!(counts.reference_n, 1);
        assert_eq!(counts.reference_bytes, PAYLOAD as u64);
        assert_eq!(counts.unknown_n, 0);
        assert_eq!(counts.mismatch_n, 0);
    }

    #[test]
    fn a_reference_the_table_cannot_answer_is_refused_by_name() {
        let payload = vec![0x5a; PAYLOAD];
        let digest = payload_digest(&payload);
        let mut table = PayloadTable::new();
        let mut counts = StatementPayloadCounts::default();
        let mut declaring = trace(vec![texture(
            1,
            TextureSource::OwnedInSlot {
                slot: 4,
                bytes: payload.clone(),
            },
        )]);
        resolve_statement_payloads(&mut table, &mut declaring, &mut counts)
            .expect("a declaration resolves");

        // A slot nothing filed.
        let mut unknown = trace(vec![texture(
            2,
            TextureSource::SlottedBytes {
                slot: 9,
                length: PAYLOAD as u64,
                digest,
            },
        )]);
        let refusal = resolve_statement_payloads(&mut table, &mut unknown, &mut counts)
            .expect_err("a slot nothing filed is refused");
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "statement_payload_slot_unknown");
        assert_eq!(counts.unknown_n, 1);

        // The slot's own payload, another digest.
        let mut mismatched = trace(vec![texture(
            3,
            TextureSource::SlottedBytes {
                slot: 4,
                length: PAYLOAD as u64,
                digest: payload_digest(&[0x5b; PAYLOAD]),
            },
        )]);
        let refusal = resolve_statement_payloads(&mut table, &mut mismatched, &mut counts)
            .expect_err("another payload's digest is refused");
        assert_eq!(refusal.slug, "statement_payload_slot_mismatch");
        assert_eq!(counts.mismatch_n, 1);

        // The slot's own digest, another length.
        let mut short = trace(vec![texture(
            4,
            TextureSource::SlottedBytes {
                slot: 4,
                length: (PAYLOAD - 4) as u64,
                digest,
            },
        )]);
        let refusal = resolve_statement_payloads(&mut table, &mut short, &mut counts)
            .expect_err("another length is refused");
        assert_eq!(refusal.slug, "statement_payload_slot_mismatch");
    }

    #[test]
    fn a_refused_statement_files_nothing() {
        let payload = vec![0x5a; PAYLOAD];
        let digest = payload_digest(&payload);
        let mut table = PayloadTable::new();
        let mut counts = StatementPayloadCounts::default();
        // One statement that files a payload *and* names one the table does not
        // hold: the declaration must not take effect, because the statement it
        // belonged to never resolved.
        let mut mixed = trace(vec![
            texture(
                1,
                TextureSource::OwnedInSlot {
                    slot: 4,
                    bytes: payload.clone(),
                },
            ),
            texture(
                2,
                TextureSource::SlottedBytes {
                    slot: 9,
                    length: PAYLOAD as u64,
                    digest,
                },
            ),
        ]);
        let refusal = resolve_statement_payloads(&mut table, &mut mixed, &mut counts)
            .expect_err("one unresolvable reference refuses the statement");
        assert_eq!(refusal.slug, "statement_payload_slot_unknown");
        assert_eq!(table.entries(), 0, "the declaration never took effect");
        assert_eq!(table.used_bytes(), 0);
        // And the next statement declares the same payload rather than naming
        // it, which is what the sender's own ledger does with a plan that never
        // crossed.
        let mut again = trace(vec![texture(
            3,
            TextureSource::OwnedInSlot {
                slot: 4,
                bytes: payload.clone(),
            },
        )]);
        resolve_statement_payloads(&mut table, &mut again, &mut counts)
            .expect("the next statement's declaration resolves");
        assert_eq!(table.entries(), 1);
    }

    #[test]
    fn a_reference_inside_one_statement_resolves_against_its_own_earlier_declaration() {
        let payload = vec![0x5a; PAYLOAD];
        let digest = payload_digest(&payload);
        let mut table = PayloadTable::new();
        let mut counts = StatementPayloadCounts::default();
        let mut trace = trace(vec![
            texture(
                1,
                TextureSource::OwnedInSlot {
                    slot: 2,
                    bytes: payload.clone(),
                },
            ),
            texture(
                2,
                TextureSource::SlottedBytes {
                    slot: 2,
                    length: PAYLOAD as u64,
                    digest,
                },
            ),
        ]);
        resolve_statement_payloads(&mut table, &mut trace, &mut counts)
            .expect("the reference resolves against the declaration beside it");
        let bytes: Vec<&[u8]> = trace
            .texture_declarations()
            .iter()
            .map(|texture| match &texture.source {
                TextureSource::OwnedBytes(bytes) => bytes.as_slice(),
                other => panic!("a resolved statement states bytes, not {other:?}"),
            })
            .collect();
        assert_eq!(bytes, vec![payload.as_slice(), payload.as_slice()]);
    }
}
