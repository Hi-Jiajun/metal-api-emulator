//! Per-section byte accounting for one submission statement.
//!
//! A submission crosses the owner→provider wire as one *statement*: a request
//! tag, a trace header, a pipeline table, a pass table — which carries every
//! buffer and texture view, and the payloads those views declare — and a
//! resource table. Which of those sections a statement's bytes are actually in
//! is a question only the encoder can answer, and it is the question
//! `openspec/changes/render-statement-economy` (W1) is decided on: the change
//! may only shrink a section the wire is *measured* to spend its bytes on.
//! The static account this reading is the runtime half of is the
//! `e_statement_wire_recon` table (one row per section, with the encoder line
//! and the byte formula each row is).
//!
//! # What it costs, and what it must not change
//!
//! **Off by default.** Off, this module is one relaxed load per encoded payload
//! ([`enabled`]) and nothing else: no clock, no lock, no allocation, and — the
//! property a byte-identity test pins — not one byte of any frame. On, the
//! encoder reads its own writer's offset at each section boundary, so every
//! number here is a function of the bytes that were written and cannot
//! disagree with them. The switch is [`SWITCH`], and the owner rail also arms
//! it when its own frame profile is on, so a round that turns the profile on
//! reads the sections without a second switch in its launcher.
//!
//! Two entry points answer with the same sections:
//!
//! * [`crate::command_codec::CommandCodec::encode_request_payload`] keeps the
//!   process-wide accumulators below current when [`enabled`] says so — the
//!   shape a running round reads, through [`take`];
//! * [`crate::command_codec::CommandCodec::encode_request_payload_accounted`]
//!   hands one payload's account back to its caller and touches no global at
//!   all — the shape a test or a rail reads a single frame with.
//!
//! # The sections
//!
//! The six positional sections tile the payload exactly, in the order the
//! encoder writes them, so a reader can check the account against the frame it
//! came from rather than take it on faith:
//!
//! ```text
//! total_bytes == tag_bytes + trace_header_bytes + pipeline_table_bytes
//!              + pass_table_bytes + resource_table_bytes + tail_bytes
//! ```
//!
//! * `tag` is the request tag that selects the submission's shape;
//! * `trace_header` is the trace's schema, device epoch, operation id and
//!   pipeline count — the fixed 26 bytes the recon table prices as §2;
//! * `pipeline_table` is the pipelines that follow that count;
//! * `pass_table` is the dispatch type, the pass count and every pass, which is
//!   where the views (and therefore the payloads) live;
//! * `resource_table` is the allocations and lease reservations that close a
//!   submission;
//! * `tail` is what is left between the last pass and the end of the payload —
//!   the completion policy and any heap/ICB tail. It is reported rather than
//!   folded into a neighbour so that the identity above can fail loudly.
//!
//! The views are the one thing that is **not** a tile: a view sits inside
//! whichever pass, stage-buffer block or vertex input states it, so the view
//! fields are roll-ups over the whole payload, disjoint from each other and
//! reported beside the tiles rather than summed into them:
//!
//! * `views_n` / `view_bytes` — how many `BufferView`s crossed, and the bytes
//!   all of them together took (`put_view` from its `view_id` to the end of its
//!   source);
//! * `view_payload_bytes` — the part of that which was the view's own
//!   `OwnedBytes` payload, which is the batch of bytes the provider then has to
//!   copy, upload and release, and the number the zero-fill arm of the change
//!   is read against (`submit_binding_copies_bytes` on the provider side);
//! * `view_declared_bytes` — the sum of the views' declared `length`s. A
//!   payload that is a zero-filled buffer is `length` bytes long by
//!   construction, so this is the extent a zero-fill declaration may stand in
//!   for, and the pair is what says whether a payload is exactly its view;
//! * `textures_n` / `texture_bytes` / `texture_payload_bytes` — the same three
//!   readings for `TextureView`s.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

/// The environment switch: `1`, `on`, `true` or `yes` arms the account, and
/// anything else — unset included — leaves it off.
pub const SWITCH: &str = "METAL_API_IPC_STATEMENT_ACCOUNTING";

const UNSET: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;

static ENABLED: AtomicU8 = AtomicU8::new(UNSET);

/// One statement's sections, in bytes.
///
/// `frames` is how many submission payloads the account covers: the section
/// fields are summed over them, so a reader dividing by `frames` gets the
/// per-statement mean, and a reader checking the identity above is checking a
/// sum of identities.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StatementAccount {
    /// How many submission payloads this account covers.
    pub frames: u64,
    /// The request tag byte that selected the submission's shape.
    pub tag_bytes: u64,
    /// The trace's schema, epoch, operation id and pipeline count.
    pub trace_header_bytes: u64,
    /// The pipeline table that follows that count.
    pub pipeline_table_bytes: u64,
    /// The dispatch type, the pass count and every pass.
    pub pass_table_bytes: u64,
    /// The allocations and lease reservations.
    pub resource_table_bytes: u64,
    /// The completion policy and any heap/ICB tail.
    pub tail_bytes: u64,
    /// The whole payload, frame header excluded.
    pub total_bytes: u64,
    /// How many `BufferView`s crossed.
    pub views_n: u64,
    /// The bytes those views took in all, payloads included.
    pub view_bytes: u64,
    /// The `OwnedBytes` payload inside those views.
    pub view_payload_bytes: u64,
    /// The sum of those views' declared `length`s.
    pub view_declared_bytes: u64,
    /// How many `TextureView`s crossed.
    pub textures_n: u64,
    /// The bytes those textures took in all, payloads included.
    pub texture_bytes: u64,
    /// The `OwnedBytes` payload inside those textures.
    pub texture_payload_bytes: u64,
}

impl StatementAccount {
    /// The bytes the six positional sections account for.
    pub fn attributed_bytes(&self) -> u64 {
        self.tag_bytes
            .saturating_add(self.trace_header_bytes)
            .saturating_add(self.pipeline_table_bytes)
            .saturating_add(self.pass_table_bytes)
            .saturating_add(self.resource_table_bytes)
            .saturating_add(self.tail_bytes)
    }

    /// What the sections do not account for, which is zero for a frame the
    /// encoder priced end to end. It is computed rather than stored so that a
    /// section that is added later cannot silently take bytes out of the
    /// identity: the reading grows, and this number is what says so.
    pub fn residual_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.attributed_bytes())
    }

    /// Accumulate `other` into this account, saturating rather than wrapping.
    pub fn add(&mut self, other: &StatementAccount) {
        self.frames = self.frames.saturating_add(other.frames);
        self.tag_bytes = self.tag_bytes.saturating_add(other.tag_bytes);
        self.trace_header_bytes = self
            .trace_header_bytes
            .saturating_add(other.trace_header_bytes);
        self.pipeline_table_bytes = self
            .pipeline_table_bytes
            .saturating_add(other.pipeline_table_bytes);
        self.pass_table_bytes = self.pass_table_bytes.saturating_add(other.pass_table_bytes);
        self.resource_table_bytes = self
            .resource_table_bytes
            .saturating_add(other.resource_table_bytes);
        self.tail_bytes = self.tail_bytes.saturating_add(other.tail_bytes);
        self.total_bytes = self.total_bytes.saturating_add(other.total_bytes);
        self.views_n = self.views_n.saturating_add(other.views_n);
        self.view_bytes = self.view_bytes.saturating_add(other.view_bytes);
        self.view_payload_bytes = self
            .view_payload_bytes
            .saturating_add(other.view_payload_bytes);
        self.view_declared_bytes = self
            .view_declared_bytes
            .saturating_add(other.view_declared_bytes);
        self.textures_n = self.textures_n.saturating_add(other.textures_n);
        self.texture_bytes = self.texture_bytes.saturating_add(other.texture_bytes);
        self.texture_payload_bytes = self
            .texture_payload_bytes
            .saturating_add(other.texture_payload_bytes);
    }
}

/// The switch's own parser, apart from the process-global it caches into so a
/// unit test can read every spelling without a switch it cannot put back.
pub fn parse_switch(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "on" | "true" | "yes")
    )
}

/// Whether this process prices its statements.
///
/// The first call resolves [`SWITCH`] and caches the answer; afterwards this is
/// a relaxed load, which is the whole cost the off arm pays. [`set_enabled`]
/// overrides it either way — that is how the owner rail arms the account when
/// its own frame profile is on.
pub fn enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        OFF => false,
        ON => true,
        _ => {
            let on = parse_switch(std::env::var(SWITCH).ok().as_deref());
            ENABLED.store(if on { ON } else { OFF }, Ordering::Relaxed);
            on
        }
    }
}

/// Arm or disarm the account, overriding whatever [`SWITCH`] would answer.
pub fn set_enabled(on: bool) {
    ENABLED.store(if on { ON } else { OFF }, Ordering::Relaxed);
}

/// The process-wide accumulators, one per field of [`StatementAccount`], all
/// `Relaxed`: a reading that is a few bytes stale is still this statement's
/// bytes, and no admission decision reads any of them.
struct Counters {
    frames: AtomicU64,
    tag_bytes: AtomicU64,
    trace_header_bytes: AtomicU64,
    pipeline_table_bytes: AtomicU64,
    pass_table_bytes: AtomicU64,
    resource_table_bytes: AtomicU64,
    tail_bytes: AtomicU64,
    total_bytes: AtomicU64,
    views_n: AtomicU64,
    view_bytes: AtomicU64,
    view_payload_bytes: AtomicU64,
    view_declared_bytes: AtomicU64,
    textures_n: AtomicU64,
    texture_bytes: AtomicU64,
    texture_payload_bytes: AtomicU64,
}

static COUNTERS: Counters = Counters {
    frames: AtomicU64::new(0),
    tag_bytes: AtomicU64::new(0),
    trace_header_bytes: AtomicU64::new(0),
    pipeline_table_bytes: AtomicU64::new(0),
    pass_table_bytes: AtomicU64::new(0),
    resource_table_bytes: AtomicU64::new(0),
    tail_bytes: AtomicU64::new(0),
    total_bytes: AtomicU64::new(0),
    views_n: AtomicU64::new(0),
    view_bytes: AtomicU64::new(0),
    view_payload_bytes: AtomicU64::new(0),
    view_declared_bytes: AtomicU64::new(0),
    textures_n: AtomicU64::new(0),
    texture_bytes: AtomicU64::new(0),
    texture_payload_bytes: AtomicU64::new(0),
};

/// Add one statement's account to the process-wide accumulators.
///
/// Called by the codec itself; a caller that wants one frame's account rather
/// than the running total asks
/// [`CommandCodec::encode_request_payload_accounted`](crate::command_codec::CommandCodec::encode_request_payload_accounted)
/// instead, which records into nothing.
pub fn record(account: StatementAccount) {
    use std::sync::atomic::Ordering::Relaxed;
    let pairs: [(_, u64); 15] = [
        (&COUNTERS.frames, account.frames),
        (&COUNTERS.tag_bytes, account.tag_bytes),
        (&COUNTERS.trace_header_bytes, account.trace_header_bytes),
        (&COUNTERS.pipeline_table_bytes, account.pipeline_table_bytes),
        (&COUNTERS.pass_table_bytes, account.pass_table_bytes),
        (&COUNTERS.resource_table_bytes, account.resource_table_bytes),
        (&COUNTERS.tail_bytes, account.tail_bytes),
        (&COUNTERS.total_bytes, account.total_bytes),
        (&COUNTERS.views_n, account.views_n),
        (&COUNTERS.view_bytes, account.view_bytes),
        (&COUNTERS.view_payload_bytes, account.view_payload_bytes),
        (&COUNTERS.view_declared_bytes, account.view_declared_bytes),
        (&COUNTERS.textures_n, account.textures_n),
        (&COUNTERS.texture_bytes, account.texture_bytes),
        (
            &COUNTERS.texture_payload_bytes,
            account.texture_payload_bytes,
        ),
    ];
    for (counter, value) in pairs {
        counter.fetch_add(value, Relaxed);
    }
}

/// The accumulators as they stand, leaving them in place.
pub fn snapshot() -> StatementAccount {
    use std::sync::atomic::Ordering::Relaxed;
    StatementAccount {
        frames: COUNTERS.frames.load(Relaxed),
        tag_bytes: COUNTERS.tag_bytes.load(Relaxed),
        trace_header_bytes: COUNTERS.trace_header_bytes.load(Relaxed),
        pipeline_table_bytes: COUNTERS.pipeline_table_bytes.load(Relaxed),
        pass_table_bytes: COUNTERS.pass_table_bytes.load(Relaxed),
        resource_table_bytes: COUNTERS.resource_table_bytes.load(Relaxed),
        tail_bytes: COUNTERS.tail_bytes.load(Relaxed),
        total_bytes: COUNTERS.total_bytes.load(Relaxed),
        views_n: COUNTERS.views_n.load(Relaxed),
        view_bytes: COUNTERS.view_bytes.load(Relaxed),
        view_payload_bytes: COUNTERS.view_payload_bytes.load(Relaxed),
        view_declared_bytes: COUNTERS.view_declared_bytes.load(Relaxed),
        textures_n: COUNTERS.textures_n.load(Relaxed),
        texture_bytes: COUNTERS.texture_bytes.load(Relaxed),
        texture_payload_bytes: COUNTERS.texture_payload_bytes.load(Relaxed),
    }
}

/// The accumulators, emptied by the read: what the statements encoded since the
/// last read carried, which is how the owner rail puts one frame's sections on
/// its own frame line. Each field is swapped on its own, so a statement encoded
/// between two of the swaps is counted by one read or the other and never by
/// both.
pub fn take() -> StatementAccount {
    use std::sync::atomic::Ordering::Relaxed;
    StatementAccount {
        frames: COUNTERS.frames.swap(0, Relaxed),
        tag_bytes: COUNTERS.tag_bytes.swap(0, Relaxed),
        trace_header_bytes: COUNTERS.trace_header_bytes.swap(0, Relaxed),
        pipeline_table_bytes: COUNTERS.pipeline_table_bytes.swap(0, Relaxed),
        pass_table_bytes: COUNTERS.pass_table_bytes.swap(0, Relaxed),
        resource_table_bytes: COUNTERS.resource_table_bytes.swap(0, Relaxed),
        tail_bytes: COUNTERS.tail_bytes.swap(0, Relaxed),
        total_bytes: COUNTERS.total_bytes.swap(0, Relaxed),
        views_n: COUNTERS.views_n.swap(0, Relaxed),
        view_bytes: COUNTERS.view_bytes.swap(0, Relaxed),
        view_payload_bytes: COUNTERS.view_payload_bytes.swap(0, Relaxed),
        view_declared_bytes: COUNTERS.view_declared_bytes.swap(0, Relaxed),
        textures_n: COUNTERS.textures_n.swap(0, Relaxed),
        texture_bytes: COUNTERS.texture_bytes.swap(0, Relaxed),
        texture_payload_bytes: COUNTERS.texture_payload_bytes.swap(0, Relaxed),
    }
}

/// Empty the accumulators without reading them.
pub fn reset() {
    let _ = take();
}

/// The sections the encoder tiles a payload with, in write order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Section {
    /// The request tag.
    Tag,
    /// Schema, epoch, operation id and pipeline count.
    TraceHeader,
    /// The pipeline table.
    PipelineTable,
    /// Dispatch type, pass count and every pass.
    PassTable,
    /// Allocations and lease reservations.
    ResourceTable,
    /// Completion policy and heap/ICB tail.
    Tail,
}

/// One statement under construction: the section being written and the offset
/// it started at, plus the account those sections are being summed into.
///
/// The draft is owned by the [`Encoder`](crate::command_codec::CommandCodec)
/// that armed it, so a payload that is refused part way through drops it and
/// records nothing.
pub(crate) struct Draft {
    section: Section,
    start: usize,
    account: StatementAccount,
}

impl Draft {
    /// A draft whose first section starts at offset zero.
    pub(crate) fn new() -> Self {
        Self {
            section: Section::Tag,
            start: 0,
            account: StatementAccount::default(),
        }
    }

    /// Close the section being written at `end` and open `section` there.
    pub(crate) fn begin(&mut self, end: usize, section: Section) {
        self.close(end);
        self.section = section;
        self.start = end;
    }

    /// Note one buffer view written between `start` and `end`: `declared` is
    /// the view's own `length`, `payload` the `OwnedBytes` bytes it carried
    /// (zero for every reference arm).
    pub(crate) fn note_view(&mut self, start: usize, end: usize, declared: u64, payload: u64) {
        self.account.views_n = self.account.views_n.saturating_add(1);
        self.account.view_bytes = self
            .account
            .view_bytes
            .saturating_add(saturating_span(start, end));
        self.account.view_payload_bytes = self.account.view_payload_bytes.saturating_add(payload);
        self.account.view_declared_bytes =
            self.account.view_declared_bytes.saturating_add(declared);
    }

    /// Note one texture view written between `start` and `end`: `payload` is
    /// the `OwnedBytes` bytes it carried.
    pub(crate) fn note_texture(&mut self, start: usize, end: usize, payload: u64) {
        let span = saturating_span(start, end);
        self.account.textures_n = self.account.textures_n.saturating_add(1);
        self.account.texture_bytes = self.account.texture_bytes.saturating_add(span);
        self.account.texture_payload_bytes =
            self.account.texture_payload_bytes.saturating_add(payload);
    }

    /// Close the last section at `end` and finish the statement.
    pub(crate) fn finish(mut self, end: usize) -> StatementAccount {
        self.close(end);
        self.account.total_bytes = end as u64;
        self.account.frames = 1;
        self.account
    }

    fn close(&mut self, end: usize) {
        let span = saturating_span(self.start, end);
        match self.section {
            Section::Tag => self.account.tag_bytes = self.account.tag_bytes.saturating_add(span),
            Section::TraceHeader => {
                self.account.trace_header_bytes =
                    self.account.trace_header_bytes.saturating_add(span)
            }
            Section::PipelineTable => {
                self.account.pipeline_table_bytes =
                    self.account.pipeline_table_bytes.saturating_add(span)
            }
            Section::PassTable => {
                self.account.pass_table_bytes = self.account.pass_table_bytes.saturating_add(span)
            }
            Section::ResourceTable => {
                self.account.resource_table_bytes =
                    self.account.resource_table_bytes.saturating_add(span)
            }
            Section::Tail => self.account.tail_bytes = self.account.tail_bytes.saturating_add(span),
        }
    }
}

/// The bytes between two offsets, or zero if the writer went backwards, which
/// it never does: an offset that moved the wrong way is a bug in the reading
/// and must not become a wrapped byte count.
fn saturating_span(start: usize, end: usize) -> u64 {
    end.saturating_sub(start) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The switch reads the spellings a launcher writes and nothing else.
    #[test]
    fn the_switch_reads_the_spellings_a_launcher_writes() {
        for on in ["1", "on", "ON", " true ", "yes", "Yes"] {
            assert!(parse_switch(Some(on)), "{on:?} is an ask");
        }
        for off in ["", "0", "off", "no", "false", "maybe", "2"] {
            assert!(!parse_switch(Some(off)), "{off:?} is not an ask");
        }
        assert!(!parse_switch(None), "unset is not an ask");
    }

    /// The identity a reader checks an account against its frame with.
    #[test]
    fn the_sections_tile_the_payload_and_the_residual_says_when_they_do_not() {
        let mut account = StatementAccount {
            frames: 1,
            tag_bytes: 1,
            trace_header_bytes: 26,
            pipeline_table_bytes: 400,
            pass_table_bytes: 1_000,
            resource_table_bytes: 64,
            tail_bytes: 1,
            total_bytes: 1_492,
            ..StatementAccount::default()
        };
        assert_eq!(account.attributed_bytes(), 1_492);
        assert_eq!(account.residual_bytes(), 0);

        // A seventh section taking bytes out of the total is what the residual
        // is for: it is reported, not hidden.
        account.total_bytes += 8;
        assert_eq!(account.residual_bytes(), 8);

        // And the accumulators are the same identity, summed.
        let same = account;
        account.add(&same);
        assert_eq!(account.frames, 2);
        assert_eq!(account.total_bytes, 3_000);
    }

    /// A draft closes every section at the offsets it is told, so the account
    /// is the writer's own offsets rather than a second walk of the values.
    #[test]
    fn a_draft_prices_the_offsets_it_is_closed_at() {
        let mut draft = Draft::new();
        draft.begin(1, Section::TraceHeader);
        draft.begin(27, Section::PipelineTable);
        draft.begin(427, Section::PassTable);
        draft.note_view(427, 500, 64, 0);
        draft.begin(1_000, Section::ResourceTable);
        draft.begin(1_064, Section::Tail);
        let account = draft.finish(1_065);
        assert_eq!(account.frames, 1);
        assert_eq!(account.tag_bytes, 1);
        assert_eq!(account.trace_header_bytes, 26);
        assert_eq!(account.pipeline_table_bytes, 400);
        assert_eq!(account.pass_table_bytes, 573);
        assert_eq!(account.resource_table_bytes, 64);
        assert_eq!(account.tail_bytes, 1);
        assert_eq!(account.total_bytes, 1_065);
        assert_eq!(account.residual_bytes(), 0);
        assert_eq!(account.views_n, 1);
        assert_eq!(account.view_bytes, 73);
        assert_eq!(account.view_payload_bytes, 0);
        assert_eq!(account.view_declared_bytes, 64);
    }

    /// The counters take and leave nothing behind.
    #[test]
    fn the_accumulators_read_once() {
        reset();
        record(StatementAccount {
            frames: 2,
            total_bytes: 100,
            view_payload_bytes: 80,
            ..StatementAccount::default()
        });
        let taken = take();
        assert_eq!(taken.frames, 2);
        assert_eq!(taken.total_bytes, 100);
        assert_eq!(taken.view_payload_bytes, 80);
        assert_eq!(snapshot(), StatementAccount::default());
        assert_eq!(take(), StatementAccount::default());
    }
}
