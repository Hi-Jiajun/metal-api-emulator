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
//! ([`PAYLOAD_TABLE_SLOTS`], [`PAYLOAD_TABLE_BYTES`]), and [`PayloadLedger::plan`]
//! reads **both of them on every branch it can take**, through `payload_fits`
//! — the one predicate [`PayloadTable::declare`] applies. So the sender never
//! declares an entry the provider would refuse: a full table is a *policy* the
//! sender reads off its own ledger (it replaces its least recently used entry,
//! stating the replacement into that entry's own slot), and the provider's own
//! caps are a wiring guard rather than a second policy. A device epoch change
//! empties both, because the bytes a provider holds do not survive it.
//!
//! One reading that paragraph does not cover is *within* a statement, and the
//! sentence is a statement rather than a declaration: the provider files a
//! statement's declarations together ([`PayloadTable::declare_all`]), so the
//! sender has to plan them that way too ([`PayloadLedger::plan_statement`]).
//! Planning declaration by declaration against the ledger *as the statement
//! found it* — its own earlier declarations are filed only once the statement
//! crossed — states declarations that each fit the bound and do not fit it
//! together: a small one into a fresh slot, a larger one into the entry it
//! replaces, together one payload past the byte bound. The provider then has to
//! refuse the statement by name (there is no third reading: a filing is a
//! statement's, and filing half of one leaves the two `used_bytes` apart), and
//! a wiring that can refuse a statement the ledger read as fitting is exactly
//! what the module above promises cannot happen.
//!
//! So a statement's plans are taken as one batch, over a projection of the
//! ledger the statement's own earlier plans move as the walk goes: the entry a
//! plan replaces reports the length it holds *then*, a fresh slot leaves the
//! free list, and the replacement policy orders by the readings the statement's
//! own reuse marks left. Every branch still reads both bounds through
//! `payload_fits` and [`PAYLOAD_TABLE_SLOTS`] — the predicate the provider
//! applies — so a statement the table cannot hold whole is absorbed by
//! replacement or by carrying the bytes ([`PayloadPlan::Carry`]) rather than
//! refused. [`PayloadTable::declare_all`] stays where it was, the second line:
//! a statement whose plan the provider's own walk does not take files none of
//! it and is refused by name.
//!
//! Reading one bound and not the other is the shape census v65 caught, and it
//! is worth stating as a shape rather than as a caution. The *first* branch of a
//! plan — a slot no declaration has used yet — used to return without asking
//! `payload_fits`, on the reading that a slot number below
//! [`PAYLOAD_TABLE_SLOTS`] is room. But the two constants are one budget, and
//! the byte bound binds first by orders of magnitude: 4 096 slots of the
//! 8 294 400-byte payload this pose samples is 32 GiB against a 512 MiB bound.
//! One boot's filings reached 531 265 024 B with 100 slots consumed, the next
//! payload was planned into a fresh slot the provider's own bound could not
//! hold, and a refusal kills the statement that stated it — seven of them, and
//! 3 468 skipped draws behind them. A branch of the plan that does not read the
//! whole budget is a branch that states declarations the provider takes back.

use std::collections::BTreeMap;

use crate::provider::DeviceEpoch;

/// The most payload slots one epoch's table may hold.
pub const PAYLOAD_TABLE_SLOTS: u32 = 4096;

/// The most payload bytes one epoch's table may hold.
///
/// The census that sized this saw 392 MB of distinct payloads in one boot, so
/// this is roomy for the pose it was measured in while staying a bound a
/// provider can promise: past it the sender stops filing *into fresh slots* —
/// it replaces its least recently used entry while the entry it replaces makes
/// room, and a payload even that cannot hold is carried, which is the shape
/// every round before this arm had.
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

/// The ledger a statement's walk reads: the entries, the free slot and the
/// bytes, as the statement that is being planned has moved them so far.
///
/// A statement's own declarations are not filed until the statement both ends
/// processed, so the walk they are planned on cannot be the ledger itself: what
/// the statement itself has already planned stands in for the slots it names,
/// while a filing — the fresh slot one consumes, the bytes one adds — stays in
/// the projection. It is one copy of a table the two constants bound (4 096
/// entries at the very most, and a pose's entries are one per payload of the
/// byte bound), paid once per statement the sender plans — a single
/// declaration's [`PayloadLedger::plan`] is a statement of one, so the cost of
/// the common case is the same copy.
#[derive(Clone, Debug)]
struct LedgerProjection {
    entries: BTreeMap<u32, LedgerEntry>,
    next_slot: u32,
    used_bytes: u64,
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
    /// The statement ordinal this ledger is planning for, which is what the
    /// replacement policy orders by — one ordinal per declaration, so a batch's
    /// own declarations sit in the order the statement states them.
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

    /// What to state for a payload of these bytes, planned against the ledger
    /// as the statement that carries it found it.
    ///
    /// A plan is a proposal: nothing in the ledger's *filings* changes until the
    /// statement it belongs to is stated, which is what this method taking
    /// `&mut self` only to advance the ordinal means — a statement that never
    /// crossed leaves no entry behind on either end. A statement that carries
    /// more than one declaration plans them together instead
    /// ([`Self::plan_statement`]), because the two ends file them together.
    pub fn plan(&mut self, digest: u128, length: u64) -> PayloadPlan {
        let mut projection = self.projection();
        self.plan_statement_into(&mut projection, &[(digest, length)])
            .pop()
            .expect("one payload, one plan")
    }

    /// Plan **one statement's** declarations in one call: what to state for
    /// each of them, in the order the statement states them.
    ///
    /// The unit the wire files is the statement, not the declaration: the
    /// provider applies a statement's declarations together, over a walk of its
    /// own table that the statement's own earlier declarations move as it goes
    /// ([`PayloadTable::declare_all`]), and a statement's filings are committed
    /// only once every arm of it resolved. Planning declaration by declaration
    /// against the ledger as the statement found it therefore states
    /// declarations that each fit the bound and do not fit it together — a small
    /// one into a fresh slot and a larger one into the entry it replaces, one
    /// payload past the byte bound between them — and the provider has to refuse
    /// that statement by name. This call plans the batch instead, over a
    /// projection of the ledger the statement's own earlier plans move: the
    /// entry a replacement reports is the length the walk left there, a fresh slot
    /// spends itself, and the least recently used entry the policy orders by is
    /// the one this statement's own plans have left coldest.
    ///
    /// Every branch still reads both bounds, through `payload_fits` and
    /// [`PAYLOAD_TABLE_SLOTS`] — the predicate the provider applies — so a
    /// statement the table cannot hold whole is absorbed by replacing its least
    /// recently used entry or by carrying those bytes
    /// ([`PayloadPlan::Carry`]), never by refusing the statement.
    ///
    /// Naming an entry is not a filing: a payload the projection already holds
    /// is stated as a reference (also when the statement's *own* earlier
    /// declaration is what holds it — the provider resolves a reference against
    /// what this same statement staged before reaching its table), and the
    /// entries a statement names are marked as used as the walk goes, which is
    /// the reading [`Self::plan`] takes. What the batch defers is the filings:
    /// the slot each declaration names is claimed by [`Self::commit_statement`],
    /// which the caller calls once the statement crossed, and a statement that
    /// never crossed commits nothing.
    pub fn plan_statement(&mut self, payloads: &[(u128, u64)]) -> Vec<PayloadPlan> {
        let mut projection = self.projection();
        self.plan_statement_into(&mut projection, payloads)
    }

    /// The same walk, on a projection the caller owns.
    fn plan_statement_into(
        &mut self,
        projection: &mut LedgerProjection,
        payloads: &[(u128, u64)],
    ) -> Vec<PayloadPlan> {
        let mut plans = Vec::with_capacity(payloads.len());
        for (digest, length) in payloads {
            // One declaration, one ordinal: a plan's replacement policy orders
            // by the statements the entries it holds were last used in, and a
            // statement's own declarations are used in it one after the other,
            // exactly as they were when each of them was planned on its own.
            self.clock = self.clock.saturating_add(1);
            let ordinal = self.clock;
            plans.push(self.plan_into(projection, *digest, *length, ordinal));
        }
        plans
    }

    /// The ledger as it stands, for one statement's walk to move.
    fn projection(&self) -> LedgerProjection {
        LedgerProjection {
            entries: self.entries.clone(),
            next_slot: self.next_slot,
            used_bytes: self.used_bytes,
        }
    }

    /// One declaration's plan against `projection`.
    ///
    /// The identity a reference is stated from is the projection's — the
    /// statement's own earlier declarations included, so a payload the walk
    /// already replaced under another slot is planned again rather than named
    /// through the slot that no longer holds it — and so is the room a
    /// declaration reads. Both bounds are read on the projection, which is the
    /// ledger itself for a statement's first declaration.
    fn plan_into(
        &mut self,
        projection: &mut LedgerProjection,
        digest: u128,
        length: u64,
        ordinal: u64,
    ) -> PayloadPlan {
        if let Some(slot) = projection
            .entries
            .iter()
            .find(|(_, entry)| entry.digest == digest && entry.length == length)
            .map(|(slot, _)| *slot)
        {
            // A reuse is not a filing: the entry is the table's already, and
            // the statement naming it is what keeps it off the replacement
            // policy's front. The ledger's own entry is marked too, exactly as
            // `plan` marks it, so the order a later statement replaces by is
            // the order this statement's own naming left — whoever ends up
            // committing anything.
            if let Some(entry) = projection.entries.get_mut(&slot) {
                entry.used_at = ordinal;
            }
            if let Some(entry) = self.entries.get_mut(&slot) {
                entry.used_at = ordinal;
            }
            return PayloadPlan::Reuse(slot);
        }
        // A fresh slot is room only when the *byte* bound holds too. The two
        // constants are one budget and the byte bound binds first (see the
        // module docs): a plan that reads only `next_slot` states a declaration
        // the provider's own `payload_fits` refuses, and a refusal kills the
        // statement rather than carrying that one payload.
        if projection.next_slot < PAYLOAD_TABLE_SLOTS
            && payload_fits(projection.used_bytes, 0, length)
        {
            // The entry is filed by `commit_statement`, so a statement that
            // never crossed leaves the slot unclaimed: the next statement plans
            // the same number again. The projection does spend it, because the
            // rest of *this* statement states into the slots after it.
            let slot = projection.next_slot;
            projection.next_slot = projection.next_slot.saturating_add(1);
            projection.used_bytes = projection.used_bytes.saturating_add(length);
            projection.entries.insert(
                slot,
                LedgerEntry {
                    digest,
                    length,
                    used_at: ordinal,
                },
            );
            return PayloadPlan::Declare(slot);
        }
        let victim = projection
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.used_at)
            .map(|(slot, entry)| (*slot, entry.length));
        match victim {
            Some((slot, held)) if payload_fits(projection.used_bytes, held, length) => {
                projection.entries.insert(
                    slot,
                    LedgerEntry {
                        digest,
                        length,
                        used_at: ordinal,
                    },
                );
                projection.used_bytes = projection.used_bytes - held + length;
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

    /// File **one statement's** declarations, all of them at once: what the
    /// caller does once the provider's reader resolved the statement that
    /// stated them.
    ///
    /// `declared` is the [`PayloadPlan::Declare`] arms the statement actually
    /// stated — a [`PayloadPlan::Reuse`] names an entry the table already holds
    /// and files nothing — in the order the statement states them. A statement's
    /// filings take effect together, which is why they are handed over together:
    /// the provider files them the same way ([`PayloadTable::declare_all`]), and
    /// a statement whose frame did not resolve is one whose caller commits none
    /// of this.
    pub fn commit_statement(&mut self, declared: &[(u32, u128, u64)]) {
        for (slot, digest, length) in declared {
            self.commit(*slot, *digest, *length);
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

/// What refused one of a statement's declarations, and the reading of the bound
/// that refused it.
///
/// The two bounds are one budget, and a refusal is read against the one that
/// bound it — so the number a caller logs beside the slot has to be the number
/// *that* bound read, and it is not the same number on both branches: the byte
/// bound refuses a declaration for the bytes the entry it names already stands
/// for, the slot bound for the slots the table already holds (a slot bound
/// refuses a slot no entry holds, so the entry's own bytes are not what it
/// read).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PayloadRefusal {
    /// The slot the refused declaration named.
    pub slot: u32,
    /// Which of the two bounds refused it.
    pub bound: PayloadTableFull,
    /// What the refusing bound read: the bytes the entry at [`Self::slot`]
    /// stands for (zero when the slot holds no entry), or the slots the table
    /// holds.
    pub held: u64,
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

    /// File every declaration one statement stated, **all of them or none**.
    ///
    /// A statement's declarations take effect together, because the sender
    /// commits its whole plan only once this end resolved the statement — and
    /// drops that plan whole when this end refuses it. Filing the first k−1
    /// declarations of a statement whose k-th one the bounds cannot hold would
    /// therefore leave this table holding entries the sender's ledger never
    /// committed: the two `used_bytes` diverge, and from then on this end's
    /// bound refuses declarations the ledger read as fitting — the shape it6's
    /// user run read as one `statement_payload_table_full` refusal at
    /// t≈209.5 s, with the ledger and the table one payload apart under it.
    /// (The sender plans a statement as one batch over the same walk this
    /// pre-check takes — `PayloadLedger::plan_statement` — so a statement a
    /// well-wired sender states is one this walk takes: this reading is the
    /// second line, and it is what keeps a refusal from being a desync when the
    /// two ends do disagree.)
    ///
    /// The statement is therefore read once over a shadow of the table (the
    /// same slot bound and the same `payload_fits` [`Self::declare`] applies,
    /// each declaration's bytes standing in for the slot it names as the walk
    /// goes), and only a statement every one of whose declarations the shadow
    /// takes is filed at all. The refusal reports the slot and the bound that
    /// refused it, so a caller can still name what the statement carried.
    pub fn declare_all(&mut self, staged: Vec<(u32, Vec<u8>)>) -> Result<(), PayloadRefusal> {
        let lengths: Vec<(u32, u64)> = staged
            .iter()
            .map(|(slot, bytes)| (*slot, u64::try_from(bytes.len()).unwrap_or(u64::MAX)))
            .collect();
        admit_declarations(
            self.used_bytes,
            self.entries
                .iter()
                .map(|(slot, entry)| (*slot, u64::try_from(entry.bytes.len()).unwrap_or(u64::MAX))),
            &lengths,
        )?;
        for (slot, bytes) in staged {
            // The walk above read the same bounds on a shadow of this table, so
            // every declaration here is one it took.
            let filed = self.declare(slot, bytes);
            debug_assert!(
                filed.is_ok(),
                "the declaration pre-check took every declaration, not {filed:?}"
            );
        }
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

/// Whether the table could take every one of one statement's declarations, in
/// the wire's own order, **without filing any of them**.
///
/// `used` is what the table holds and `occupied` is the slot each of its
/// entries stands under with the bytes it stands for; `staged` is the
/// statement's own declarations, as the slot each names and the bytes it
/// carries. The walk is [`PayloadTable::declare`]'s own reading — the slot
/// bound on a slot no entry holds yet, and `payload_fits` on the bytes — with
/// the statement's own declarations standing in for the slots they name as it
/// goes. That is what makes a statement that names one slot twice, or one whose
/// own earlier declaration spends the room a later one needs, read exactly the
/// way filing it would.
fn admit_declarations(
    used: u64,
    occupied: impl Iterator<Item = (u32, u64)>,
    staged: &[(u32, u64)],
) -> Result<(), PayloadRefusal> {
    let mut held: BTreeMap<u32, u64> = occupied.collect();
    let mut used = used;
    for (slot, length) in staged {
        if !held.contains_key(slot)
            && u32::try_from(held.len()).unwrap_or(u32::MAX) >= PAYLOAD_TABLE_SLOTS
        {
            return Err(PayloadRefusal {
                slot: *slot,
                bound: PayloadTableFull::Slots,
                held: u64::try_from(held.len()).unwrap_or(u64::MAX),
            });
        }
        let replaced = held.get(slot).copied().unwrap_or(0);
        if !payload_fits(used, replaced, *length) {
            return Err(PayloadRefusal {
                slot: *slot,
                bound: PayloadTableFull::Bytes,
                held: replaced,
            });
        }
        used = used - replaced + *length;
        held.insert(*slot, *length);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        admit_declarations, payload_digest, PayloadLedger, PayloadLookup, PayloadPlan,
        PayloadRefusal, PayloadTable, PayloadTableFull, PAYLOAD_TABLE_BYTES, PAYLOAD_TABLE_SLOTS,
    };
    use crate::provider::DeviceEpoch;

    fn epoch() -> DeviceEpoch {
        DeviceEpoch::new(1)
    }

    /// What one statement's plan did, for a caller that reads the arms back.
    type StatementPlans = Vec<PayloadPlan>;

    /// State one statement to **both** ends, the way each of them reads it, and
    /// check the two tables still stand for the same thing.
    ///
    /// `Some(payloads)` is a statement that crossed the wire: the sender plans
    /// every byte-carrying declaration of it **as one batch**, before it states
    /// the frame, the provider resolves the arms of the frame it decoded — a
    /// reference out of what this same statement already staged, else out of its
    /// table, a declaration staged — and files the declarations only once
    /// **all** of them resolved, in the wire's own order; the sender commits the
    /// batch after that, which is the point the two tables are equal at.
    /// `None` is a trace that states no lease at all: it keeps the in-process
    /// path, the sender never plans for it (R's `crosses_the_wire`), and there
    /// is no wire for the provider to read, so neither end moves.
    fn state_statement(
        ledger: &mut PayloadLedger,
        table: &mut PayloadTable,
        payloads: Option<&[Vec<u8>]>,
    ) -> StatementPlans {
        let Some(payloads) = payloads else {
            assert_eq!(
                (ledger.entries(), ledger.used_bytes()),
                (table.entries(), table.used_bytes()),
                "a statement that never crossed moved one of the two ends"
            );
            return Vec::new();
        };
        let digests: Vec<u128> = payloads.iter().map(|bytes| payload_digest(bytes)).collect();
        let lengths: Vec<u64> = payloads
            .iter()
            .map(|bytes| u64::try_from(bytes.len()).unwrap_or(u64::MAX))
            .collect();
        // The statement's own declarations, for the batch plan; the parameter
        // `payloads` stays the bytes each of them carries, which the checks
        // below read them back from.
        let declarations: Vec<(u128, u64)> = digests
            .iter()
            .copied()
            .zip(lengths.iter().copied())
            .collect();
        let plans: StatementPlans = ledger.plan_statement(&declarations);
        // The provider's reader, in the wire's own order: nothing is filed
        // until every arm of the statement has resolved.
        let mut staged: Vec<(u32, &[u8])> = Vec::new();
        for (index, plan) in plans.iter().enumerate() {
            match *plan {
                PayloadPlan::Declare(slot) => staged.push((slot, payloads[index].as_slice())),
                PayloadPlan::Reuse(slot) => {
                    let held = staged
                        .iter()
                        .find(|(staged_slot, _)| *staged_slot == slot)
                        .map(|(_, staged)| *staged)
                        .or_else(
                            || match table.lookup(slot, digests[index], lengths[index]) {
                                PayloadLookup::Hit(held) => Some(held),
                                _ => None,
                            },
                        );
                    assert_eq!(
                        held.map(|held| held.to_vec()),
                        Some(payloads[index].clone()),
                        "the ledger named slot {slot} and the provider's table does not hold \
                         those bytes: the two ends are not holding the same table"
                    );
                }
                PayloadPlan::Carry => {}
            }
        }
        for (slot, bytes) in &staged {
            if let Err(full) = table.declare(*slot, bytes.to_vec()) {
                panic!(
                    "the sender stated Declare({slot}) and the provider's own bound refused it \
                     as {full:?}: the ledger holds {} entries standing for {} bytes against the \
                     two constants, so the plan read one bound and not the other. The refusal \
                     kills the whole statement rather than carrying this one payload.",
                    ledger.entries(),
                    ledger.used_bytes(),
                );
            }
        }
        let declared: Vec<(u32, u128, u64)> = plans
            .iter()
            .enumerate()
            .filter_map(|(index, plan)| match *plan {
                PayloadPlan::Declare(slot) => Some((slot, digests[index], lengths[index])),
                _ => None,
            })
            .collect();
        ledger.commit_statement(&declared);
        assert_eq!(
            (ledger.entries(), ledger.used_bytes()),
            (table.entries(), table.used_bytes()),
            "the two ends hold different tables after a statement both of them processed"
        );
        plans
    }

    /// The payload this pose's sampled declarations carry: 1920x1080, four
    /// bytes a texel, which is the size census v65's filings were made of.
    const SAMPLED_PAYLOAD: usize = 8_294_400;

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

    /// The declaration phase is all or nothing: a statement whose k-th
    /// declaration the bounds cannot hold files none of the k−1 before it,
    /// because the sender commits its plan only for a statement both ends
    /// processed — it drops the whole plan when this end refuses one.
    #[test]
    fn a_statement_the_table_cannot_hold_whole_files_none_of_it() {
        let slots = usize::try_from(PAYLOAD_TABLE_SLOTS).unwrap();
        let last = PAYLOAD_TABLE_SLOTS - 1;
        let mut table = PayloadTable::new();
        table.scope(epoch());
        for slot in 0..last {
            assert_eq!(table.declare(slot, vec![1]), Ok(()));
        }
        let before = (table.entries(), table.used_bytes());
        assert_eq!(before, (slots - 1, u64::try_from(slots - 1).unwrap()));
        // The statement fills the table's last free slot and then names one
        // more: the second declaration is the one the slot bound refuses, and
        // it takes the first one back with it. The refusal reads the *slot*
        // bound, so what it reports is the slots the table holds — the slot the
        // declaration named travels in its own field.
        assert_eq!(
            table.declare_all(vec![(last, vec![2; 8]), (PAYLOAD_TABLE_SLOTS, vec![3; 8]),]),
            Err(PayloadRefusal {
                slot: PAYLOAD_TABLE_SLOTS,
                bound: PayloadTableFull::Slots,
                held: u64::try_from(slots).unwrap(),
            })
        );
        assert_eq!(
            (table.entries(), table.used_bytes()),
            before,
            "the refusal filed the statement's first declaration"
        );
        // A statement's own two declarations into one slot are taken whole:
        // the walk reads the table the statement itself builds, so the second
        // one replaces the slot the first one filled rather than asking for a
        // fresh one.
        assert_eq!(
            table.declare_all(vec![(last, vec![4; 8]), (last, vec![5; 16])]),
            Ok(())
        );
        assert_eq!(table.entries(), slots);
        assert_eq!(table.used_bytes(), before.1 + 16);
    }

    /// Census it6's shape (E-SW3F2's §2), with both halves of the reading
    /// beside each other: the statement planned declaration by declaration is
    /// one the provider has to refuse, and the statement planned as the batch
    /// the wire files is one it takes.
    ///
    /// The table holds a 4 KiB entry and the bulk of the byte budget, so 4 KiB
    /// of the bound is left, and the statement states that same 4 KiB payload
    /// and an 8 KiB one. Planned one at a time — against the ledger *as the
    /// statement found it*, because a statement's own filings are not made until
    /// it crossed — the two declarations each fit and do not fit together: the
    /// small one plans the fresh slot 2 and the larger one the entry it
    /// replaces. The provider's walk refuses that statement by name (the byte
    /// bound, slot 0, the 4 KiB that entry stands for) and files none of it.
    /// Planned as one batch, the statement's second payload is *carried* — the
    /// shape every round before this arm had — and the statement crosses with
    /// both ends holding the same table.
    #[test]
    fn a_statements_own_declarations_are_planned_as_the_statement_it_is() {
        /// The entry the cross-statement plan replaces.
        const VICTIM_SLOT: u32 = 0;
        /// The bulk of the budget, leaving the statement one 4 KiB payload of
        /// room to plan a fresh slot into.
        const BULK_SLOT: u32 = 1;
        const ROOM: usize = 4096;
        const CROSSING: usize = 8192;
        /// The slot the per-declaration reading plans for the small payload,
        /// which no declaration has used yet.
        const FRESH_SLOT: u32 = 2;

        let bulk = usize::try_from(PAYLOAD_TABLE_BYTES).unwrap() - ROOM - ROOM;
        let fresh = vec![0x33; ROOM];
        let crossing = vec![0x44; CROSSING];
        let fresh_digest = payload_digest(&fresh);
        let crossing_digest = payload_digest(&crossing);

        let mut table = PayloadTable::new();
        table.scope(epoch());
        assert_eq!(table.declare(VICTIM_SLOT, vec![0x11; ROOM]), Ok(()));
        assert_eq!(table.declare(BULK_SLOT, vec![0x22; bulk]), Ok(()));
        let mut ledger = PayloadLedger::new();
        ledger.scope(epoch());
        ledger.commit(VICTIM_SLOT, payload_digest(&[0x11]), ROOM as u64);
        ledger.commit(BULK_SLOT, payload_digest(&[0x22]), bulk as u64);
        assert_eq!(
            (ledger.entries(), ledger.used_bytes()),
            (table.entries(), table.used_bytes()),
            "the two ends open the statement on the same table"
        );

        // The control arm: the statement's declarations planned one at a time.
        let mut one_at_a_time = ledger.clone();
        assert_eq!(
            [
                one_at_a_time.plan(fresh_digest, ROOM as u64),
                one_at_a_time.plan(crossing_digest, CROSSING as u64),
            ],
            [
                PayloadPlan::Declare(FRESH_SLOT),
                PayloadPlan::Declare(VICTIM_SLOT)
            ],
            "each declaration fits the bound it reads on its own"
        );
        assert_eq!(
            table.declare_all(vec![
                (FRESH_SLOT, fresh.clone()),
                (VICTIM_SLOT, crossing.clone()),
            ]),
            Err(PayloadRefusal {
                slot: VICTIM_SLOT,
                bound: PayloadTableFull::Bytes,
                held: ROOM as u64,
            }),
            "the provider refuses the statement its own two declarations cross"
        );
        assert_eq!(
            (ledger.entries(), ledger.used_bytes()),
            (table.entries(), table.used_bytes()),
            "the refusal filed none of the control statement"
        );

        // The arm this increment is for: the same statement, planned as one
        // batch against the room the statement's own walk leaves.
        let plans = ledger.plan_statement(&[
            (fresh_digest, ROOM as u64),
            (crossing_digest, CROSSING as u64),
        ]);
        assert_eq!(
            plans,
            [PayloadPlan::Declare(FRESH_SLOT), PayloadPlan::Carry],
            "the batch absorbs the payload the bound cannot hold by carrying it"
        );
        // The statement's arms are the ones the plan named: the 4 KiB payload
        // declared into slot 2, and the 8 KiB one on the ordinary arm, which is
        // what `Carry` means. The provider's walk takes them, and the two ends
        // hold the same table after the statement crossed.
        assert_eq!(
            table.declare_all(vec![(FRESH_SLOT, fresh.clone())]),
            Ok(()),
            "the provider takes the batch the sender planned"
        );
        ledger.commit_statement(&[(FRESH_SLOT, fresh_digest, ROOM as u64)]);
        assert_eq!(
            (ledger.entries(), ledger.used_bytes()),
            (table.entries(), table.used_bytes()),
            "the batch statement left the two ends apart"
        );
        assert_eq!(
            table.used_bytes(),
            PAYLOAD_TABLE_BYTES,
            "the table is full again"
        );
        match table.lookup(FRESH_SLOT, fresh_digest, ROOM as u64) {
            PayloadLookup::Hit(bytes) => assert_eq!(bytes, fresh.as_slice()),
            other => panic!("the batch statement did not file its declaration: {other:?}"),
        }
    }

    /// The other half of the same reading, taken off lengths alone (so it costs
    /// no half gigabyte): a statement whose own declarations cannot be filed in
    /// the room the table opens with is absorbed by the table's own policy —
    /// the first declaration states into the entry it replaces and the second
    /// into the room that replacement freed — and a payload the whole budget
    /// cannot hold is *carried*, with the statement's other declarations still
    /// stated. Neither shape is a refusal.
    #[test]
    fn a_statements_own_declarations_are_absorbed_by_the_tables_own_policy() {
        let mut ledger = PayloadLedger::new();
        ledger.scope(epoch());
        // One entry standing for the whole budget, so a fresh slot has no room
        // at all.
        ledger.commit(0, payload_digest(&[0xb0]), PAYLOAD_TABLE_BYTES);
        let first = payload_digest(&[0xb1]);
        let second = payload_digest(&[0xb2]);
        assert_eq!(
            ledger.plan_statement(&[(first, 1), (second, 1)]),
            [PayloadPlan::Declare(0), PayloadPlan::Declare(1)],
            "the statement replaces what it cannot fit beside, then takes the room it freed"
        );
        // Planning files nothing: the ledger still stands where it stood.
        assert_eq!(
            (ledger.entries(), ledger.used_bytes()),
            (1, PAYLOAD_TABLE_BYTES)
        );
        // A payload larger than the whole budget is carried, and the statement
        // still states the declaration the table can take.
        assert_eq!(
            ledger.plan_statement(&[(first, PAYLOAD_TABLE_BYTES + 1), (second, 1)]),
            [PayloadPlan::Carry, PayloadPlan::Declare(0)],
        );
    }

    /// Committing a batch is committing each of its declarations, in the
    /// statement's own order: the two readings leave the same ledger — the same
    /// slots, the same bytes, the same free slot.
    ///
    /// What the batch changes is above the commit, and this test pins both
    /// halves: three declarations of one statement that the byte bound holds are
    /// planned into the three slots after the table's, where planning each of
    /// them against the ledger *as the statement found it* named the same fresh
    /// slot three times (the shape that made a statement's own declarations
    /// overwrite one another, and the room reading this cut replaces).
    #[test]
    fn a_batch_commits_what_the_same_declarations_commit_one_at_a_time() {
        let payloads = [
            (payload_digest(&[0x01]), SAMPLED_PAYLOAD as u64),
            (payload_digest(&[0x02]), 4096),
            (payload_digest(&[0x03]), SAMPLED_PAYLOAD as u64),
        ];

        let mut batch = PayloadLedger::new();
        batch.scope(epoch());
        let plans = batch.plan_statement(&payloads);
        assert_eq!(
            plans,
            [
                PayloadPlan::Declare(0),
                PayloadPlan::Declare(1),
                PayloadPlan::Declare(2)
            ],
            "a statement's own declarations take the slots after the table's"
        );
        let mut one_at_a_time = PayloadLedger::new();
        one_at_a_time.scope(epoch());
        let singles: Vec<PayloadPlan> = payloads
            .iter()
            .map(|(digest, length)| one_at_a_time.plan(*digest, *length))
            .collect();
        assert_eq!(
            singles,
            [
                PayloadPlan::Declare(0),
                PayloadPlan::Declare(0),
                PayloadPlan::Declare(0)
            ],
            "planned one at a time against the ledger the statement found, all three name the \
             same fresh slot: each of them states a filing the next one replaces"
        );

        // The commit is the same reading either way: these declarations, filed
        // in the statement's own order.
        let declared: [(u32, u128, u64); 3] = [
            (0, payloads[0].0, payloads[0].1),
            (1, payloads[1].0, payloads[1].1),
            (2, payloads[2].0, payloads[2].1),
        ];
        batch.commit_statement(&declared);
        for (slot, digest, length) in &declared {
            one_at_a_time.commit(*slot, *digest, *length);
        }
        assert_eq!(
            (batch.entries(), batch.used_bytes(), batch.next_slot),
            (
                one_at_a_time.entries(),
                one_at_a_time.used_bytes(),
                one_at_a_time.next_slot
            ),
            "the batch's filings are the filings of each declaration of it"
        );
    }

    /// A payload one statement states **twice** is stated as a declaration and
    /// then as a reference to it: the second one is what the statement's own
    /// first declaration is about to hold, so naming it carries none of the
    /// bytes (and the provider resolves a reference against what this same
    /// statement staged before it reaches its table).
    #[test]
    fn a_statement_names_what_its_own_first_declaration_staged() {
        let mut ledger = PayloadLedger::new();
        let mut table = PayloadTable::new();
        ledger.scope(epoch());
        table.scope(epoch());
        let payload = vec![0x5a; SAMPLED_PAYLOAD];
        let plans = state_statement(&mut ledger, &mut table, Some(&[payload.clone(), payload]));
        assert_eq!(
            plans,
            vec![PayloadPlan::Declare(0), PayloadPlan::Reuse(0)],
            "the statement declares the payload once and names it the second time"
        );
        assert_eq!(
            (ledger.entries(), ledger.used_bytes()),
            (1, SAMPLED_PAYLOAD as u64)
        );
        assert_eq!(
            (table.entries(), table.used_bytes()),
            (1, SAMPLED_PAYLOAD as u64)
        );
    }

    /// The same reading, taken off lengths alone, so the byte bound's own arm
    /// of the pre-check costs nothing to keep beside the walk that spends half
    /// a gigabyte: `used` and the entries the slots already stand for are the
    /// table's state, and the statement's own declarations take their slots as
    /// the walk goes.
    #[test]
    fn the_declaration_precheck_reads_the_statement_in_the_wires_order() {
        let bound = PAYLOAD_TABLE_BYTES;
        let full: Vec<(u32, u64)> = (0..PAYLOAD_TABLE_SLOTS - 1).map(|slot| (slot, 1)).collect();
        let slot_bound = u64::from(PAYLOAD_TABLE_SLOTS);
        for (used, occupied, staged, expected) in [
            // A statement whose own first declaration spends the room its
            // second one needs: 8 KiB then 8 KiB, with 16 KiB of the bound
            // left. One byte less of room and the second one is the refusal —
            // of the slot that holds *no* entry, so the byte bound read zero
            // bytes for it and the refusal says so.
            (
                bound - 16 * 1024 + 1,
                Vec::new(),
                vec![(0, 8 * 1024), (1, 8 * 1024)],
                Err(PayloadRefusal {
                    slot: 1,
                    bound: PayloadTableFull::Bytes,
                    held: 0,
                }),
            ),
            (
                bound - 16 * 1024,
                Vec::new(),
                vec![(0, 8 * 1024), (1, 8 * 1024)],
                Ok(()),
            ),
            // it6's shape: the statement's small declaration plans a fresh slot
            // and its larger one the entry it replaces, and it is the room the
            // small one took that makes that replacement impossible. The
            // refusal names the slot the replacement landed in and the bytes
            // that entry already stands for — the number a reader used to have
            // to mistake for the table's own reading.
            (
                bound - 4 * 1024,
                vec![(0, 4 * 1024), (1, bound - 8 * 1024)],
                vec![(2, 4 * 1024), (0, 8 * 1024)],
                Err(PayloadRefusal {
                    slot: 0,
                    bound: PayloadTableFull::Bytes,
                    held: 4 * 1024,
                }),
            ),
            // One slot named twice: the second declaration replaces what the
            // statement's own first one filed, so the bytes it reports against
            // the bound are its own and not the whole budget's.
            (bound - 16, Vec::new(), vec![(7, 16), (7, 16)], Ok(())),
            // The slot bound is read on the slots no entry holds yet: the last
            // free slot is the statement's to fill, one more is not — while the
            // same statement's second declaration into the slot its own first
            // one filled is a replacement, not a fresh slot.
            (
                slot_bound - 1,
                full.clone(),
                vec![(PAYLOAD_TABLE_SLOTS - 1, 1), (PAYLOAD_TABLE_SLOTS, 1)],
                Err(PayloadRefusal {
                    slot: PAYLOAD_TABLE_SLOTS,
                    bound: PayloadTableFull::Slots,
                    // The statement's own first declaration filled the last
                    // free slot before the walk reached this one, so the slot
                    // bound read the bound itself.
                    held: slot_bound,
                }),
            ),
            (
                slot_bound - 1,
                full,
                vec![(PAYLOAD_TABLE_SLOTS - 1, 1), (PAYLOAD_TABLE_SLOTS - 1, 2)],
                Ok(()),
            ),
        ] {
            assert_eq!(
                admit_declarations(used, occupied.into_iter(), &staged),
                expected,
                "used={used} staged={staged:?}"
            );
        }
    }

    /// Census v65's shape, with both ends fed the same statements: one boot's
    /// [`SAMPLED_PAYLOAD`]-byte declarations, statements that state no lease at
    /// all interleaved between them, and the filings walking up to the byte
    /// bound and across it.
    ///
    /// The byte bound is the one that binds, and it binds while `next_slot` is
    /// three orders of magnitude below [`PAYLOAD_TABLE_SLOTS`]: 64 of these
    /// payloads are 530 841 600 B of the 512 MiB budget, and the 65th would be
    /// 539 136 000 B. A plan that reads only the slot bound states that 65th
    /// declaration into a fresh slot, the provider's own predicate takes it
    /// back, and the refusal kills the statement — which is what the census read
    /// as seven refusals and 3 468 skipped draws, and what this walk fails on
    /// before it walks on a plan that reads both bounds.
    #[test]
    fn the_two_ends_hold_the_same_table_over_the_shape_census_v65_walked() {
        let capacity = usize::try_from(PAYLOAD_TABLE_BYTES / SAMPLED_PAYLOAD as u64).unwrap();
        assert_eq!(capacity, 64, "the byte bound holds 64 of these payloads");
        let mut ledger = PayloadLedger::new();
        let mut table = PayloadTable::new();
        ledger.scope(epoch());
        table.scope(epoch());
        let mut filings = 0_u32;
        // Eight statements past the bound: the sender has to keep stating
        // declarations, and each of them has to be one the provider takes.
        for index in 0..capacity + 8 {
            if index % 8 == 3 {
                // A trace that states no lease at all, between the ones that
                // cross: the in-process population census v65 counted beside
                // the 52% that crossed.
                assert!(state_statement(&mut ledger, &mut table, None).is_empty());
            }
            let mut bytes = vec![0_u8; SAMPLED_PAYLOAD];
            bytes[0] = u8::try_from(index).unwrap_or(0);
            bytes[1] = 0x5a;
            let plans = state_statement(&mut ledger, &mut table, Some(&[bytes]));
            assert_eq!(plans.len(), 1);
            match plans[0] {
                PayloadPlan::Declare(slot) if index < capacity => {
                    assert_eq!(
                        slot, filings,
                        "a payload the byte bound still holds files into the next slot"
                    );
                    filings += 1;
                }
                PayloadPlan::Declare(slot) => {
                    // Past the bound a fresh slot no longer fits, so the plan
                    // replaces the least recently used entry — one the table
                    // already holds, which is why the provider can file it.
                    assert!(
                        slot < filings,
                        "past the byte bound the plan stated a fresh slot {slot}"
                    );
                }
                PayloadPlan::Carry => {
                    panic!("the walk's least recently used entry always makes room for these");
                }
                PayloadPlan::Reuse(slot) => panic!("every payload of this walk is new, not {slot}"),
            }
        }
        // Every statement of the walk is equal on both ends, and the table
        // holds exactly what the budget holds.
        assert_eq!(ledger.entries(), capacity);
        assert_eq!(table.entries(), capacity);
        assert_eq!(ledger.used_bytes(), (capacity * SAMPLED_PAYLOAD) as u64);
        assert!(ledger.used_bytes() <= PAYLOAD_TABLE_BYTES);
    }

    /// The other half of the same walk: payloads the byte bound can hold many
    /// more of than the slot bound can, so [`PAYLOAD_TABLE_SLOTS`] is what runs
    /// out, and the plan past it states a replacement into the least recently
    /// used entry — a slot both ends already hold, never a fresh one.
    #[test]
    fn the_two_ends_replace_the_least_recently_used_entry_together() {
        let slots = usize::try_from(PAYLOAD_TABLE_SLOTS).unwrap();
        let mut ledger = PayloadLedger::new();
        let mut table = PayloadTable::new();
        ledger.scope(epoch());
        table.scope(epoch());
        for index in 0..slots + 1 {
            let bytes = vec![
                u8::try_from(index & 0xff).unwrap_or(0),
                u8::try_from((index >> 8) & 0xff).unwrap_or(0),
                0x77,
            ];
            let plans = state_statement(&mut ledger, &mut table, Some(&[bytes]));
            match plans[0] {
                // The first `slots` declarations file fresh slots, in order;
                // the one past the slot bound replaces the entry nothing has
                // touched since it was filed.
                PayloadPlan::Declare(slot) => assert_eq!(
                    slot,
                    if index < slots {
                        u32::try_from(index).unwrap_or(u32::MAX)
                    } else {
                        0
                    },
                    "declaration {index} named the wrong slot"
                ),
                other => panic!("declaration {index} is not a declaration: {other:?}"),
            }
        }
        assert_eq!(ledger.entries(), slots);
        assert_eq!(table.entries(), slots);
        assert_eq!(ledger.used_bytes(), table.used_bytes());
    }

    /// The same reading, taken off lengths alone, so it costs nothing to keep
    /// beside the walk above: every declaration the ledger *states* is one the
    /// predicate the provider applies takes — a fresh slot is room only while
    /// the whole budget holds it, and past that the entry a replacement reports
    /// is the one the table already holds for that slot.
    #[test]
    fn a_stated_declaration_is_one_the_providers_own_predicate_takes() {
        let mut ledger = PayloadLedger::new();
        ledger.scope(epoch());
        let mut filed = std::collections::BTreeSet::new();
        let mut declarations = 0_u32;
        for index in 0..200_u32 {
            let length = SAMPLED_PAYLOAD as u64;
            let digest = payload_digest(&index.to_be_bytes());
            let used = ledger.used_bytes();
            let PayloadPlan::Declare(slot) = ledger.plan(digest, length) else {
                continue;
            };
            // Every payload of this walk has the same length, so the entry a
            // replacement reports is either this length or no entry at all.
            let held = if filed.contains(&slot) { length } else { 0 };
            assert!(
                super::payload_fits(used, held, length),
                "declaration {index} stated Declare({slot}) for {length} B with {used} B held \
                 and {held} B in the slot: the provider's own predicate refuses it"
            );
            filed.insert(slot);
            ledger.commit(slot, digest, length);
            declarations += 1;
        }
        assert_eq!(declarations, 200, "every payload of this walk is new");
        assert_eq!(ledger.entries(), filed.len());
        assert!(ledger.used_bytes() <= PAYLOAD_TABLE_BYTES);
    }
}
