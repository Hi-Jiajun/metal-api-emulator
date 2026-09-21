//! The statement payload table (the statement economy's W4, task E-SW3).
//!
//! # What it is for
//!
//! W1's section account prices one statement's own bytes, and the texture
//! payload — the sampled declarations' tightly packed extents — is the largest
//! single slice of it that is still real, read data. The census that followed
//! (`.agents/tasks/root/e_statement_texture_payload-report.md`) measured what
//! that slice is made of over one production-pose round: **265 distinct
//! payloads over a whole boot, 98% of the delivered bytes a re-send of bytes an
//! earlier statement already carried, and 0.0003% a repeat of bytes the *same*
//! statement carried**.
//!
//! The arm that takes those bytes off the wire is therefore a cross-statement
//! one: a declaration *files* its bytes in a table the provider keeps
//! ([`crate::provider::TextureSource::OwnedInSlot`]) and a later declaration
//! *names* that slot, its length and a digest instead of carrying the payload
//! ([`crate::provider::TextureSource::SlottedBytes`]).
//!
//! # The two ends, and why they cannot disagree silently
//!
//! Both ends hold a table: the provider stores the bytes ([`PayloadTable`]),
//! the sender keeps the ledger of what it filed ([`PayloadLedger`] — slots,
//! lengths and digests, never the bytes). The sender is the only one that
//! *decides* anything: it looks its ledger up before it states an arm, and a
//! slot is written only by a declaration it states. So the provider's table is
//! a shadow of the sender's ledger, and the two are equal after every statement
//! both ends processed, which is every statement that crossed.
//!
//! Equality is not *assumed* where a wrong answer would be silent: the
//! reference arm carries the payload's **digest and length**, and the provider
//! refuses a reference its own entry does not match, by name. A collision or a
//! desync is therefore a named refusal — never another payload's bytes. The
//! digest is a 128-bit FNV-1a fold ([`payload_digest`]): its job is to make a
//! desync loud, not to be a cryptographic commitment, and both ends compute it
//! with this one function so the two cannot drift.
//!
//! # What a table holds, and how it is bounded
//!
//! Both ends bound one epoch's table by the *same* two constants
//! ([`PAYLOAD_TABLE_SLOTS`], [`PAYLOAD_TABLE_BYTES`]), so the sender never
//! declares an entry the provider would refuse: a full table is a *policy* the
//! sender reads off its own ledger (it replaces its least recently used entry,
//! stating the replacement into that entry's own slot), and the provider's own
//! caps are a wiring guard rather than a second policy. A device epoch change
//! empties both, because the bytes a provider holds do not survive it.

use std::collections::BTreeMap;

use crate::provider::DeviceEpoch;

/// The most payload slots one epoch's table may hold.
pub const PAYLOAD_TABLE_SLOTS: u32 = 4096;

/// The most payload bytes one epoch's table may hold.
///
/// The census that sized this saw 392 MB of distinct payloads in one boot, so
/// this is roomy for the pose it was measured in while staying a bound a
/// provider can promise: past it the sender stops filing new payloads and
/// carries them, which is the shape every round before this arm had.
pub const PAYLOAD_TABLE_BYTES: u64 = 512 << 20;

/// A 128-bit FNV-1a fold of one payload.
///
/// One function for the two ends: the sender folds the bytes it states, the
/// provider folds the bytes it receives, and a reference only resolves when
/// the two folds and the lengths agree. See the module docs for what this
/// digest is and is not for.
pub fn payload_digest(bytes: &[u8]) -> u128 {
    const OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
    const PRIME: u128 = 0x0000000001000000000000000000013b;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u128::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// What one slot holds, as the sender's ledger knows it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LedgerEntry {
    digest: u128,
    length: u64,
    /// The statement ordinal this entry was last used in, for the ledger's own
    /// replacement policy.
    used_at: u64,
}

/// What the sender should state for one payload it is about to declare.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PayloadPlan {
    /// The table already holds these bytes: state the reference arm for this
    /// slot and carry none of them.
    Reuse(u32),
    /// The table has room (a fresh slot, or the least recently used one): state
    /// the declaration arm into this slot.
    Declare(u32),
    /// Neither fits — the payload is larger than the whole budget, or
    /// replacing the least recently used entry would still not make room: state
    /// the ordinary arm and carry the bytes.
    Carry,
}

/// The sending side's ledger of what it has filed: one entry per slot, with no
/// payload bytes of its own.
#[derive(Clone, Debug, Default)]
pub struct PayloadLedger {
    epoch: Option<DeviceEpoch>,
    entries: BTreeMap<u32, LedgerEntry>,
    /// The next slot no declaration has used yet, in this epoch.
    next_slot: u32,
    used_bytes: u64,
    /// The statement ordinal this ledger is planning for, which is what
    /// [`Self::plan`]'s replacement policy orders by.
    clock: u64,
}

impl PayloadLedger {
    /// An empty ledger, scoped to no epoch yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Scope the ledger to `epoch`, emptying it when the epoch changed.
    ///
    /// Returns whether the ledger was emptied. Both ends call this on the
    /// statement they are about to state, so a device epoch change empties the
    /// two tables at the same statement.
    pub fn scope(&mut self, epoch: DeviceEpoch) -> bool {
        if self.epoch == Some(epoch) {
            return false;
        }
        self.epoch = Some(epoch);
        self.entries.clear();
        self.next_slot = 0;
        self.used_bytes = 0;
        self.clock = 0;
        true
    }

    /// Whether `epoch` is the epoch this ledger is scoped to.
    pub fn scoped_to(&self, epoch: DeviceEpoch) -> bool {
        self.epoch == Some(epoch)
    }

    /// What to state for a payload of these bytes: `None` when the bytes are
    /// empty, which is a shape the contract refuses on its own.
    ///
    /// A plan is a proposal: nothing in the ledger changes until the statement
    /// it belongs to is stated, which is what [`Self::plan`] taking `&mut self`
    /// only to advance the ordinal means — a statement that never crossed
    /// leaves no entry behind on either end.
    pub fn plan(&mut self, digest: u128, length: u64) -> PayloadPlan {
        self.clock = self.clock.saturating_add(1);
        let ordinal = self.clock;
        if let Some(slot) = self
            .entries
            .iter()
            .find(|(_, entry)| entry.digest == digest && entry.length == length)
            .map(|(slot, _)| *slot)
        {
            if let Some(entry) = self.entries.get_mut(&slot) {
                entry.used_at = ordinal;
            }
            return PayloadPlan::Reuse(slot);
        }
        if length > PAYLOAD_TABLE_BYTES {
            return PayloadPlan::Carry;
        }
        if self.next_slot < PAYLOAD_TABLE_SLOTS {
            // The entry is filed by `commit`, so a statement that never crossed
            // leaves the slot unclaimed: the next statement plans the same
            // number again.
            return PayloadPlan::Declare(self.next_slot);
        }
        let victim = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.used_at)
            .map(|(slot, entry)| (*slot, entry.length));
        match victim {
            Some((slot, held)) if self.used_bytes - held + length <= PAYLOAD_TABLE_BYTES => {
                PayloadPlan::Declare(slot)
            }
            _ => PayloadPlan::Carry,
        }
    }

    /// File the entry one [`PayloadPlan::Declare`] planned, which the caller
    /// does once the statement that stated it crossed.
    pub fn commit(&mut self, slot: u32, digest: u128, length: u64) {
        let ordinal = self.clock;
        let previous = self.entries.insert(
            slot,
            LedgerEntry {
                digest,
                length,
                used_at: ordinal,
            },
        );
        if let Some(previous) = previous {
            self.used_bytes = self.used_bytes.saturating_sub(previous.length);
        }
        self.used_bytes = self.used_bytes.saturating_add(length);
        // The slot a fresh plan named is claimed here rather than in `plan`:
        // only a statement both ends processed may consume one.
        if slot == self.next_slot {
            self.next_slot = self.next_slot.saturating_add(1);
        }
    }

    /// The slots this ledger holds, and the bytes they stand for.
    pub fn entries(&self) -> usize {
        self.entries.len()
    }

    /// The bytes the ledger's entries stand for.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }
}

/// What a slot holds, as the provider's table knows it.
#[derive(Clone, Debug, Eq, PartialEq)]
struct StoredPayload {
    digest: u128,
    bytes: Vec<u8>,
}

/// What a reference arm's lookup found.
#[derive(Debug)]
pub enum PayloadLookup<'a> {
    /// The entry matches the reference's slot, length and digest: these are
    /// the bytes the sender filed.
    Hit(&'a [u8]),
    /// No entry this epoch holds that slot.
    Unknown,
    /// The slot holds other bytes: the length or the digest disagrees.
    Mismatch,
}

/// Why a declaration could not be filed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PayloadTableFull {
    /// Every slot is in use.
    Slots,
    /// Filing these bytes would pass the table's byte bound.
    Bytes,
}

/// The provider's own table: the payloads the statements it read filed, keyed
/// by slot, scoped to one device epoch.
#[derive(Debug, Default)]
pub struct PayloadTable {
    epoch: Option<DeviceEpoch>,
    entries: BTreeMap<u32, StoredPayload>,
    used_bytes: u64,
}

impl PayloadTable {
    /// An empty table, scoped to no epoch yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Scope the table to `epoch`, emptying it when the epoch changed.
    pub fn scope(&mut self, epoch: DeviceEpoch) -> bool {
        if self.epoch == Some(epoch) {
            return false;
        }
        self.epoch = Some(epoch);
        self.entries.clear();
        self.used_bytes = 0;
        true
    }

    /// The bytes one reference arm names, or why it names none.
    pub fn lookup(&self, slot: u32, digest: u128, length: u64) -> PayloadLookup<'_> {
        match self.entries.get(&slot) {
            Some(entry)
                if entry.digest == digest
                    && u64::try_from(entry.bytes.len()).unwrap_or(u64::MAX) == length =>
            {
                PayloadLookup::Hit(&entry.bytes)
            }
            Some(_) => PayloadLookup::Mismatch,
            None => PayloadLookup::Unknown,
        }
    }

    /// File one declaration's bytes under `slot`, replacing whatever the slot
    /// held.
    pub fn declare(&mut self, slot: u32, bytes: Vec<u8>) -> Result<(), PayloadTableFull> {
        if !self.entries.contains_key(&slot)
            && u32::try_from(self.entries.len()).unwrap_or(u32::MAX) >= PAYLOAD_TABLE_SLOTS
        {
            return Err(PayloadTableFull::Slots);
        }
        let length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        let held = self
            .entries
            .get(&slot)
            .map(|entry| u64::try_from(entry.bytes.len()).unwrap_or(u64::MAX))
            .unwrap_or(0);
        if !payload_fits(self.used_bytes, held, length) {
            return Err(PayloadTableFull::Bytes);
        }
        let digest = payload_digest(&bytes);
        self.entries.insert(slot, StoredPayload { digest, bytes });
        self.used_bytes = self.used_bytes - held + length;
        Ok(())
    }

    /// The slots the table holds, and the bytes they stand for.
    pub fn entries(&self) -> usize {
        self.entries.len()
    }

    /// The bytes the table holds.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }
}

/// Whether a payload of `incoming` bytes fits the table's byte bound when the
/// entry it replaces held `held` of the `used` bytes.
///
/// The predicate both ends read: the sender's ledger refuses to *plan* a
/// declaration this says no to, and the provider refuses to file one — the same
/// arithmetic on the same constant, so a plan the provider would take back is
/// impossible.
fn payload_fits(used: u64, held: u64, incoming: u64) -> bool {
    used.saturating_sub(held).saturating_add(incoming) <= PAYLOAD_TABLE_BYTES
}

#[cfg(test)]
mod tests {
    use super::{
        payload_digest, PayloadLedger, PayloadLookup, PayloadPlan, PayloadTable, PayloadTableFull,
        PAYLOAD_TABLE_BYTES, PAYLOAD_TABLE_SLOTS,
    };
    use crate::provider::DeviceEpoch;

    fn epoch() -> DeviceEpoch {
        DeviceEpoch::new(1)
    }

    #[test]
    fn the_digest_reads_the_bytes_and_nothing_else() {
        assert_eq!(payload_digest(&[]), payload_digest(&[]));
        assert_eq!(payload_digest(&[7; 4096]), payload_digest(&[7; 4096]));
        assert_ne!(payload_digest(&[7; 4096]), payload_digest(&[7; 4095]));
        assert_ne!(payload_digest(&[0x11; 64]), payload_digest(&[0x12; 64]));
        // A single bit in the middle of a large payload moves the fold: the
        // digest is the guard that turns a desync into a refusal, so it has to
        // read every byte rather than a prefix.
        let mut bytes = vec![0_u8; 1 << 20];
        let mut tampered = bytes.clone();
        bytes[1 << 19] = 1;
        tampered[1 << 19] = 1;
        assert_eq!(payload_digest(&bytes), payload_digest(&tampered));
        tampered[1 << 19] = 2;
        assert_ne!(payload_digest(&bytes), payload_digest(&tampered));
    }

    #[test]
    fn a_second_statement_reuses_the_slot_the_first_one_declared() {
        let bytes = vec![0x5a; 4096];
        let digest = payload_digest(&bytes);
        let mut ledger = PayloadLedger::new();
        ledger.scope(epoch());
        let first = ledger.plan(digest, 4096);
        let PayloadPlan::Declare(slot) = first else {
            panic!("a fresh ledger declares, not {first:?}");
        };
        ledger.commit(slot, digest, 4096);
        assert_eq!(ledger.entries(), 1);
        assert_eq!(ledger.used_bytes(), 4096);
        assert_eq!(ledger.plan(digest, 4096), PayloadPlan::Reuse(slot));
        // The same *length* with other bytes is another payload: the ledger
        // plans a declaration for it rather than reusing a slot whose digest
        // disagrees.
        assert!(matches!(
            ledger.plan(payload_digest(&[0x5b; 4096]), 4096),
            PayloadPlan::Declare(_)
        ));
    }

    #[test]
    fn a_statement_that_never_crossed_leaves_no_slot_behind() {
        let bytes = vec![1_u8; 1024];
        let digest = payload_digest(&bytes);
        let mut ledger = PayloadLedger::new();
        ledger.scope(epoch());
        let planned = ledger.plan(digest, 1024);
        // The plan is not committed — the statement never crossed — so the
        // ledger holds nothing and plans the same slot again.
        assert_eq!(planned, ledger.plan(digest, 1024));
        assert_eq!(ledger.entries(), 0);
        assert_eq!(ledger.used_bytes(), 0);
    }

    #[test]
    fn a_reference_its_slot_does_not_match_is_refused_by_name() {
        let bytes = vec![0x33; 2048];
        let digest = payload_digest(&bytes);
        let mut table = PayloadTable::new();
        table.scope(epoch());
        assert!(table.declare(7, bytes.clone()).is_ok());
        assert_eq!(table.entries(), 1);
        assert_eq!(table.used_bytes(), 2048);
        match table.lookup(7, digest, 2048) {
            PayloadLookup::Hit(held) => assert_eq!(held, bytes.as_slice()),
            other => panic!("the entry's own bytes have to resolve, not {other:?}"),
        }
        // Another payload's digest, the same entry's length.
        assert!(matches!(
            table.lookup(7, payload_digest(&[0x34; 2048]), 2048),
            PayloadLookup::Mismatch
        ));
        // The entry's own digest, another length.
        assert!(matches!(
            table.lookup(7, digest, 2047),
            PayloadLookup::Mismatch
        ));
        // And a slot nothing filed.
        assert!(matches!(
            table.lookup(8, digest, 2048),
            PayloadLookup::Unknown
        ));
    }

    #[test]
    fn a_full_table_replaces_its_least_recently_used_entry() {
        let mut ledger = PayloadLedger::new();
        ledger.scope(epoch());
        let mut slots = Vec::new();
        for index in 0..PAYLOAD_TABLE_SLOTS {
            // One distinct payload per slot, so every plan is a declaration.
            let digest = payload_digest(&[
                u8::try_from(index & 0xff).unwrap_or(0),
                u8::try_from(index >> 8).unwrap_or(0),
            ]);
            let PayloadPlan::Declare(slot) = ledger.plan(digest, 1) else {
                panic!("slot {index} has to fit an empty table");
            };
            ledger.commit(slot, digest, 1);
            slots.push(slot);
        }
        assert_eq!(
            ledger.entries(),
            usize::try_from(PAYLOAD_TABLE_SLOTS).unwrap()
        );
        // The least recently used entry is slot 0 (nothing touched it since),
        // so the next payload past the slot bound declares *into* it rather
        // than failing: the sender is the only end that decides, and a full
        // table is its own policy to keep working.
        let fresh = payload_digest(&[0xfe; 8]);
        assert_eq!(ledger.plan(fresh, 8), PayloadPlan::Declare(slots[0]));
        // A payload larger than the table's whole budget has no slot at all.
        assert_eq!(
            ledger.plan(payload_digest(&[0x01]), PAYLOAD_TABLE_BYTES + 1),
            PayloadPlan::Carry
        );
    }

    #[test]
    fn a_new_epoch_empties_both_ends() {
        let bytes = vec![0x77; 512];
        let digest = payload_digest(&bytes);
        let mut ledger = PayloadLedger::new();
        let mut table = PayloadTable::new();
        ledger.scope(epoch());
        table.scope(epoch());
        let PayloadPlan::Declare(slot) = ledger.plan(digest, 512) else {
            panic!("an empty ledger declares");
        };
        ledger.commit(slot, digest, 512);
        assert!(table.declare(slot, bytes).is_ok());
        // The epoch change empties the sender's ledger and the provider's
        // table at the same statement, so nothing a device epoch did not hold
        // can be named afterwards.
        let next = DeviceEpoch::new(2);
        assert!(ledger.scope(next), "the sender's ledger empties");
        assert!(table.scope(next), "the provider's table empties");
        assert_eq!(ledger.entries(), 0);
        assert_eq!(table.entries(), 0);
        assert!(matches!(
            table.lookup(slot, digest, 512),
            PayloadLookup::Unknown
        ));
        // A ledger scoped twice to the same epoch keeps what it holds.
        assert!(!ledger.scope(next), "the same epoch is not a change");
    }

    #[test]
    fn the_provider_refuses_a_declaration_its_own_bounds_cannot_hold() {
        let mut table = PayloadTable::new();
        table.scope(epoch());
        assert_eq!(
            table.declare(1, vec![0; 16]),
            Ok(()),
            "an empty table files its first entry"
        );
        // Replacing an entry with a smaller payload reports its own bytes, not
        // the entry's: the byte bound counts what the table holds.
        assert_eq!(table.declare(1, vec![0; 8]), Ok(()));
        assert_eq!(table.used_bytes(), 8);
        // The byte bound, read as the predicate both ends use: a payload fits
        // when what the table holds minus the entry it replaces plus the
        // incoming bytes stays inside the constant. Asserting it here rather
        // than with a half-gigabyte allocation is the same reading.
        for (used, held, incoming, fits) in [
            (0, 0, PAYLOAD_TABLE_BYTES, true),
            (0, 0, PAYLOAD_TABLE_BYTES + 1, false),
            // Replacing the least recently used entry reports *its* bytes
            // against the bound, not the incoming payload's alone.
            (PAYLOAD_TABLE_BYTES, 8, 8, true),
            (PAYLOAD_TABLE_BYTES, 8, 9, false),
            (PAYLOAD_TABLE_BYTES, 4096, 4096, true),
            (PAYLOAD_TABLE_BYTES, 4096, 8192, false),
            (PAYLOAD_TABLE_BYTES, 0, 1, false),
        ] {
            assert_eq!(
                super::payload_fits(used, held, incoming),
                fits,
                "used={used} held={held} incoming={incoming}"
            );
        }
    }

    #[test]
    fn the_provider_files_no_more_slots_than_the_constant_allows() {
        let mut table = PayloadTable::new();
        table.scope(epoch());
        for slot in 0..PAYLOAD_TABLE_SLOTS {
            assert_eq!(
                table.declare(slot, vec![u8::try_from(slot % 251).unwrap_or(0)]),
                Ok(()),
                "slot {slot} fits an empty table"
            );
        }
        assert_eq!(
            table.entries(),
            usize::try_from(PAYLOAD_TABLE_SLOTS).unwrap()
        );
        // One slot past the bound is refused by name: the sender's ledger holds
        // to the same constant, so this is a wiring guard rather than a second
        // policy.
        assert_eq!(
            table.declare(PAYLOAD_TABLE_SLOTS, vec![0; 1]),
            Err(PayloadTableFull::Slots)
        );
        // Replacing a slot the table already holds is not a new slot.
        assert_eq!(table.declare(0, vec![9; 1]), Ok(()));
    }
}
