//! Default-off phase timing for one provider submission.
//!
//! `provider.submit` is one call that returns only after the device work it
//! submitted has retired and its host-visible bytes have been read back (the
//! reims rail's `FrameSpan::ProvSubmit` measures exactly that call). That bar is
//! where the guest's draw cost lives, but it is a *total*: planning, resource
//! creation, recording, `vkQueueSubmit`, `vkWaitForFences` and the host-mapped
//! readback are all inside it, and none of them can be told apart from outside
//! this crate.
//!
//! This module splits that total into disjoint bars, driven by
//! `METAL_API_VULKAN_PHASE_PROFILE`:
//!
//! * unset (the default) or any value other than a truthy one: off. Every call
//!   site is a `let _bar = Bar::enter(Phase::X)` that resolves to `None` after
//!   one relaxed load — **no clock read, no lock, no allocation**.
//! * `1` / `on` / `true` / `yes`: on. Each bar reads the monotonic clock twice
//!   and adds to a thread-local table; every
//!   `METAL_API_VULKAN_PHASE_PROFILE_EVERY` (default 256) completed submissions
//!   that thread writes one line to **stderr** (the QEMU process's own stderr,
//!   i.e. the round's boot log):
//!
//!   ```text
//!   PHASE submit n=256 total_us=... admit_us=... plan_us=... pool_us=...
//!   resource_build_us=... record_us=... queue_submit_us=... fence_wait_us=...
//!   fence_wait_idle_n=... fence_wait_idle_us=... fence_wait_blocked_n=...
//!   fence_wait_blocked_us=... fence_wait_timeout_n=... read_updates_us=...
//!   render_total_us=... render_setup_us=... render_record_us=...
//!   render_submit_us=... render_wait_us=... (with render_wait's own
//!   idle/blocked/timeout fields) render_readback_us=... writebacks_us=...
//!   settle_us=... fence_wait_skipped_n=... plan_settle_us=... render_us=...
//!   readback_rect_n=... readback_rect_bytes=... readback_rect_extent_bytes=...
//!   readback_full_n=... readback_full_bytes=... readback_switch_n=...
//!   readback_shape_n=... readback_bounds_n=... readback_whole_n=...
//!   ```
//!
//! Fields are **sums over the line's own window** (`n` submissions), not means,
//! so a reader can add lines together and divide by the summed `n` without
//! weighting error. µs fields carry three decimals, i.e. nanosecond resolution.
//!
//! The `readback_*` fields are the one exception in *unit*, not in window: they
//! count the stored attachments this window's submissions published — how many
//! `vkCmdCopyImageToBuffer` stages carried only the pass's written rectangle
//! (`readback_rect_n`) and how many bytes the host then read
//! (`readback_rect_bytes`, versus `readback_rect_extent_bytes`, what those
//! attachments' whole extents occupy), how many carried the whole attachment
//! (`readback_full_n` / `readback_full_bytes`) and which fact sent them there
//! (`readback_switch_n` for the `METAL_API_VULKAN_FULL_READBACK` control arm,
//! `readback_shape_n` for a shape whose seed this rail does not hold,
//! `readback_bounds_n` for a declared rect it cannot prove,
//! `readback_whole_n` for a rectangle that covered the whole attachment anyway).
//! They are window sums like every other field, so the same add-and-divide rule
//! answers "bytes read back per submission" (`docs/WRITTEN-RECT-READBACK.md`).
//!
//! The accumulator is **thread-local**, and a line is emitted by the thread that
//! filled its own window. That is what makes each line self-consistent: with one
//! process-wide table, two threads submitting at once would land their bars in
//! the same window, and the line's `n`, its fields and its enclosing `total`
//! would describe different populations.
//!
//! The bars are deliberately disjoint: `enter`-style nesting would charge the
//! child's time to the parent as well, and then "how much of the total is the
//! readback" would have no answer. `total` encloses the call and `render_total`
//! encloses the render half; every other field is a region inside one of them,
//! so `sum(fields) <= total` and `sum(render children) <= render_total` are
//! identities a reader can check, and the difference is the seam between the
//! bars (function calls, `Arc` clones, the queue lock) — the residual, not a
//! missing bar.
//!
//! A phase may be charged at more than one disjoint region: `settle` covers the
//! completion bookkeeping of both the synchronous and the deferred arm, and
//! `writebacks` covers the mapping and the contract validation of the merged
//! result. Every field is still a sum of disjoint time.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::readback_rect::ReadbackFallback;

/// One bar of the submission profile.
///
/// The discriminant is the slot index, so `Phase::QueueSubmit as usize` and
/// [`PHASE_NAMES`] have to stay in step; `phase_names_cover_every_phase` pins
/// that down the way the draw rail pins its own table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Phase {
    /// The enclosing bar: one whole `submit` call, success or refusal.
    Total,
    /// Epoch check, capability admission and the indirect-command resolve.
    Admit,
    /// Everything that decides *what* will run: the render plan, the heap
    /// placement plan, the registered pipelines, the serial resource pool and
    /// the per-pass dispatch list.
    Plan,
    /// The submission's own inputs: device bindings for every pooled view,
    /// including the owned-byte copies, the staged-lease resolution and the
    /// gathered guest runs, plus the borrow retains that keep them alive.
    Pool,
    /// Device objects for this submission: pipeline objects, buffers, textures,
    /// static samplers, descriptor sets and the indirect buffer.
    ResourceBuild,
    /// Command-buffer recording.
    Record,
    /// Completion fence creation and `vkQueueSubmit`.
    QueueSubmit,
    /// `vkWaitForFences` on the submission's completion fence, reported with an
    /// idle/blocked split — see [`Local::note_fence_wait`].
    FenceWait,
    /// The host-mapped readback of every writable view's bytes.
    ReadUpdates,
    /// The enclosing bar for the render half executed inside the same `submit`
    /// (render/present passes and their own submissions). Like [`Phase::Total`]
    /// it encloses children, so it is an aggregate a reader checks the others
    /// against rather than a bar to add to them.
    RenderTotal,
    /// Everything an offscreen render pass does before it records: the
    /// attachment/pipeline resolution, the device objects, and the readback
    /// destination buffers the pass will copy out into.
    RenderSetup,
    /// Command-buffer recording for one render pass, including the
    /// `vkCmdCopyImageToBuffer` that stages the pass's stored attachments.
    RenderRecord,
    /// The render half's completion fence and `vkQueueSubmit`.
    RenderSubmit,
    /// The render half's `vkWaitForFences`. This is where the pass's own
    /// copy-out executes, so it is device time as much as it is latency.
    RenderWait,
    /// The render half's host-visible readback: the mapped copy of every stored
    /// attachment, the depth/stencil surfaces and the writable stage buffers.
    RenderReadback,
    /// Writeback mapping and the contract validation of the merged result.
    Writebacks,
    /// Completion bookkeeping: the terminal observation, its record insert and
    /// the health synchronisation.
    Settle,
}

const PHASE_COUNT: usize = Phase::Settle as usize + 1;

/// The printed field name of each phase, in slot order.
const PHASE_NAMES: [&str; PHASE_COUNT] = [
    "total",
    "admit",
    "plan",
    "pool",
    "resource_build",
    "record",
    "queue_submit",
    "fence_wait",
    "read_updates",
    "render_total",
    "render_setup",
    "render_record",
    "render_submit",
    "render_wait",
    "render_readback",
    "writebacks",
    "settle",
];

/// The slots the printed `plan_settle_us` field aggregates: the CPU-only matter
/// around the device work — what will run (`plan`), onto what (`pool`), and what
/// its completion left to do (`settle`).
const PLAN_SETTLE_SLOTS: [usize; 3] = [
    Phase::Plan as usize,
    Phase::Pool as usize,
    Phase::Settle as usize,
];

/// The slots the printed `render_us` field aggregates: the disjoint regions of
/// the render half. `render_total` is their enclosing bar and is printed beside
/// them rather than added to them.
const RENDER_SLOTS: [usize; 5] = [
    Phase::RenderSetup as usize,
    Phase::RenderRecord as usize,
    Phase::RenderSubmit as usize,
    Phase::RenderWait as usize,
    Phase::RenderReadback as usize,
];

/// The bars that are a fence wait, and therefore carry the idle/blocked split.
const WAIT_SLOTS: [usize; 2] = [Phase::FenceWait as usize, Phase::RenderWait as usize];

/// Submissions per emitted line, and therefore the `n=` field.
const EVERY_DEFAULT: u64 = 256;

/// A fence wait that returns inside this many nanoseconds is counted as an
/// immediate return rather than as device latency.
///
/// `vkWaitForFences` on a signaled fence costs a driver call (single-digit
/// microseconds here); a fence the queue has not retired cannot return before
/// the device work has run. The two populations are far apart, so the threshold
/// classifies them rather than splitting one population — and both buckets are
/// printed, so a reader who disagrees with the cut can still see the raw sums
/// (and move it with `METAL_API_VULKAN_PHASE_IDLE_NS`).
const IDLE_NS_DEFAULT: u64 = 100_000;

/// The readback region one stored attachment was published through
/// (`docs/WRITTEN-RECT-READBACK.md` §2), as an emitted line counts it.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ReadbackRegion {
    /// Only the pass's written rectangle left the device. `bytes` is what the
    /// host copied out of the mapping and `extent_bytes` what the attachment's
    /// whole extent occupies — the cost the same attachment had before the
    /// increment.
    Rect { bytes: u64, extent_bytes: u64 },
    /// The whole attachment left the device.
    Full { bytes: u64 },
    /// The whole attachment left the device because this rail could not prove a
    /// narrower rectangle: the control switch, the shape's seed, the declared
    /// bounds, or a rectangle that covered the whole attachment anyway.
    Fallback(ReadbackFallback),
}

/// One window's readback regions, in the same units the counter line prints.
#[derive(Default)]
struct ReadbackCounts {
    rect_n: u64,
    rect_bytes: u64,
    rect_extent_bytes: u64,
    full_n: u64,
    full_bytes: u64,
    switch_n: u64,
    shape_n: u64,
    bounds_n: u64,
    whole_n: u64,
}

impl ReadbackCounts {
    fn note(&mut self, region: ReadbackRegion) {
        match region {
            ReadbackRegion::Rect {
                bytes,
                extent_bytes,
            } => {
                self.rect_n += 1;
                self.rect_bytes += bytes;
                self.rect_extent_bytes += extent_bytes;
            }
            ReadbackRegion::Full { bytes } => {
                self.full_n += 1;
                self.full_bytes += bytes;
            }
            ReadbackRegion::Fallback(fallback) => match fallback {
                ReadbackFallback::Switch => self.switch_n += 1,
                ReadbackFallback::Shape => self.shape_n += 1,
                ReadbackFallback::Bounds => self.bounds_n += 1,
                ReadbackFallback::Whole => self.whole_n += 1,
            },
        }
    }
}

/// Count one stored attachment's readback region for the emitting thread's
/// window. A no-op — one relaxed load — while the profile is off, exactly as a
/// bar is.
#[inline]
pub(crate) fn note_readback(region: ReadbackRegion) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| local.borrow_mut().readback.note(region));
}

/// One thread's window of the profile.
#[derive(Default)]
struct Local {
    ns: [u64; PHASE_COUNT],
    calls: [u64; PHASE_COUNT],
    wait_idle_ns: [u64; PHASE_COUNT],
    wait_idle_calls: [u64; PHASE_COUNT],
    wait_blocked_ns: [u64; PHASE_COUNT],
    wait_blocked_calls: [u64; PHASE_COUNT],
    wait_timeout_calls: [u64; PHASE_COUNT],
    /// Waits with no fence to wait for (`PendingExecution::wait`'s `!submitted`
    /// early return): the only waits that are exactly free.
    fence_skipped_calls: u64,
    /// `total` bars closed since the last emitted line.
    window: u64,
    /// The stored attachments this window's submissions read back, by region.
    readback: ReadbackCounts,
}

impl Local {
    #[inline]
    fn charge(&mut self, phase: Phase, ns: u64) {
        let slot = phase as usize;
        self.ns[slot] += ns;
        self.calls[slot] += 1;
    }

    /// Bank one `vkWaitForFences`, classified by how long the call itself took.
    ///
    /// The classification is the answer to "how much of the wait is the device
    /// and how much is the driver call": a wait that returns immediately means
    /// the queue had already retired the work, and the microseconds that remain
    /// here buy no GPU time.
    #[inline]
    fn note_fence_wait(&mut self, phase: Phase, ns: u64, timed_out: bool) {
        self.charge(phase, ns);
        let slot = phase as usize;
        if ns < idle_ns() {
            self.wait_idle_ns[slot] += ns;
            self.wait_idle_calls[slot] += 1;
        } else {
            self.wait_blocked_ns[slot] += ns;
            self.wait_blocked_calls[slot] += 1;
        }
        if timed_out {
            self.wait_timeout_calls[slot] += 1;
        }
    }

    /// One submission closed. Emits the window's line when it is full.
    ///
    /// Banked from the enclosing bar's `Drop`, so the submission the line is
    /// closed by is already inside it: children drop before their parent, and
    /// counting at entry would publish a line whose `total` was missing the
    /// submission whose fields it reports.
    fn note_total(&mut self, ns: u64) {
        self.charge(Phase::Total, ns);
        self.window += 1;
        if self.window < every() {
            return;
        }
        let n = self.window;
        let mut fields = String::with_capacity(600);
        let mut plan_settle_ns = 0u64;
        let mut render_ns = 0u64;
        for (slot, name) in PHASE_NAMES.iter().enumerate() {
            let ns = std::mem::take(&mut self.ns[slot]);
            self.calls[slot] = 0;
            if PLAN_SETTLE_SLOTS.contains(&slot) {
                plan_settle_ns += ns;
            }
            if RENDER_SLOTS.contains(&slot) {
                render_ns += ns;
            }
            if WAIT_SLOTS.contains(&slot) {
                let idle_calls = std::mem::take(&mut self.wait_idle_calls[slot]);
                let idle_us = micros(std::mem::take(&mut self.wait_idle_ns[slot]));
                let blocked_calls = std::mem::take(&mut self.wait_blocked_calls[slot]);
                let blocked_us = micros(std::mem::take(&mut self.wait_blocked_ns[slot]));
                let timed_out = std::mem::take(&mut self.wait_timeout_calls[slot]);
                fields.push_str(&format!(
                    " {name}_us={:.3} {name}_idle_n={idle_calls} {name}_idle_us={idle_us:.3} \
                     {name}_blocked_n={blocked_calls} {name}_blocked_us={blocked_us:.3} \
                     {name}_timeout_n={timed_out}",
                    micros(ns)
                ));
                continue;
            }
            fields.push_str(&format!(" {name}_us={:.3}", micros(ns)));
        }
        let skipped = std::mem::take(&mut self.fence_skipped_calls);
        let readback = std::mem::take(&mut self.readback);
        self.window = 0;
        let plan_settle_us = micros(plan_settle_ns);
        let render_us = micros(render_ns);
        eprintln!(
            "PHASE submit n={n}{fields} fence_wait_skipped_n={skipped} \
             plan_settle_us={plan_settle_us:.3} render_us={render_us:.3} \
             readback_rect_n={} readback_rect_bytes={} readback_rect_extent_bytes={} \
             readback_full_n={} readback_full_bytes={} readback_switch_n={} \
             readback_shape_n={} readback_bounds_n={} readback_whole_n={}",
            readback.rect_n,
            readback.rect_bytes,
            readback.rect_extent_bytes,
            readback.full_n,
            readback.full_bytes,
            readback.switch_n,
            readback.shape_n,
            readback.bounds_n,
            readback.whole_n,
        );
    }
}

thread_local! {
    /// One window per submitting thread. A thread that never submits never
    /// touches this, and a thread that does gets a table only its own bars
    /// write — which is what lets a line's `n` and its fields be the same
    /// population.
    static LOCAL: RefCell<Local> = RefCell::new(Local::default());
}

#[inline]
fn micros(ns: u64) -> f64 {
    ns as f64 / 1_000.0
}

/// Whether the profile is on, read once from the process environment.
#[inline]
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_VULKAN_PHASE_PROFILE")
                .ok()
                .as_deref(),
        )
    })
}

fn every() -> u64 {
    static EVERY: OnceLock<u64> = OnceLock::new();
    *EVERY.get_or_init(|| {
        parse_env_u64("METAL_API_VULKAN_PHASE_PROFILE_EVERY")
            .filter(|every| *every > 0)
            .unwrap_or(EVERY_DEFAULT)
    })
}

fn idle_ns() -> u64 {
    static IDLE: OnceLock<u64> = OnceLock::new();
    *IDLE.get_or_init(|| parse_env_u64("METAL_API_VULKAN_PHASE_IDLE_NS").unwrap_or(IDLE_NS_DEFAULT))
}

/// The enable word, read exactly as a round's launcher spells it.
fn parse_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1" | "on" | "ON" | "true" | "yes")
    )
}

fn parse_env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse().ok()
}

/// A running bar, banked when it drops.
///
/// `None` when the profile is off, which keeps every call site a
/// `let _bar = Bar::enter(..)` with no branch in it: the switch is read once
/// here rather than at each bracket.
pub(crate) struct Bar {
    slot: Phase,
    started: Instant,
}

impl Bar {
    #[inline]
    pub(crate) fn enter(phase: Phase) -> Option<Self> {
        if !enabled() {
            return None;
        }
        Some(Self {
            slot: phase,
            started: Instant::now(),
        })
    }

    #[inline]
    pub(crate) fn enter_fence_wait(phase: Phase) -> Option<FenceWaitBar> {
        if !enabled() {
            return None;
        }
        Some(FenceWaitBar {
            phase,
            started: Instant::now(),
            timed_out: false,
        })
    }
}

impl Drop for Bar {
    #[inline]
    fn drop(&mut self) {
        let phase = self.slot;
        let ns = elapsed_ns(self.started.elapsed());
        LOCAL.with(|local| {
            let mut local = local.borrow_mut();
            if phase == Phase::Total {
                local.note_total(ns);
            } else {
                local.charge(phase, ns);
            }
        });
    }
}

/// The `vkWaitForFences` bar: same shape as [`Bar`], but it reports the idle /
/// blocked split instead of a plain charge.
pub(crate) struct FenceWaitBar {
    phase: Phase,
    started: Instant,
    timed_out: bool,
}

impl Drop for FenceWaitBar {
    #[inline]
    fn drop(&mut self) {
        let ns = elapsed_ns(self.started.elapsed());
        let phase = self.phase;
        let timed_out = self.timed_out;
        LOCAL.with(|local| local.borrow_mut().note_fence_wait(phase, ns, timed_out));
    }
}

impl FenceWaitBar {
    /// The driver answered with nothing yet (`TIMEOUT` / `NOT_READY`), so the
    /// caller may retry. Counted apart from the retired waits because a retry
    /// loop is a different finding from a slow queue.
    #[inline]
    pub(crate) fn mark_timed_out(&mut self) {
        self.timed_out = true;
    }
}

/// A wait with no fence behind it: exactly free, and exactly countable.
#[inline]
pub(crate) fn note_fence_wait_skipped() {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| local.borrow_mut().fence_skipped_calls += 1);
}

#[inline]
fn elapsed_ns(elapsed: Duration) -> u64 {
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The printed table and the phase enum must agree slot for slot: a missing
    /// name is lost time and a shifted name is time charged to a neighbour,
    /// which reads like an answer and is worse than no reading at all.
    #[test]
    fn phase_names_cover_every_phase() {
        assert_eq!(PHASE_NAMES.len(), PHASE_COUNT);
        assert_eq!(PHASE_NAMES[Phase::Total as usize], "total");
        assert_eq!(PHASE_NAMES[Phase::FenceWait as usize], "fence_wait");
        for (slot, name) in PHASE_NAMES.iter().enumerate() {
            assert!(!name.is_empty(), "slot {slot} has no field name");
            assert!(
                PHASE_NAMES
                    .iter()
                    .take(slot)
                    .all(|previous| previous != name),
                "duplicate field name {name}"
            );
        }
        assert!(PLAN_SETTLE_SLOTS.iter().all(|slot| *slot < PHASE_COUNT));
    }

    /// Off by default: a process without the variable must not read a clock or
    /// reach the accumulator from any call site.
    #[test]
    fn the_profile_is_off_unless_the_variable_says_otherwise() {
        for value in [
            None,
            Some(""),
            Some("0"),
            Some("off"),
            Some("no"),
            Some("2"),
        ] {
            assert!(
                !parse_enabled(value),
                "{value:?} must not enable the profile"
            );
        }
        for value in ["1", "on", "ON", "true", "yes", " 1 "] {
            assert!(
                parse_enabled(Some(value)),
                "{value} must enable the profile"
            );
        }
    }

    /// The buckets have to partition the wait: their sum is the field, or a
    /// reader cannot tell device latency from driver-call overhead.
    #[test]
    fn the_fence_wait_buckets_partition_the_wait() {
        let mut local = Local::default();
        local.note_fence_wait(Phase::FenceWait, 1_000, false);
        local.note_fence_wait(Phase::FenceWait, IDLE_NS_DEFAULT, false);
        local.note_fence_wait(Phase::FenceWait, 5_000_000, true);
        local.note_fence_wait(Phase::RenderWait, 7, false);
        let slot = Phase::FenceWait as usize;
        let total = local.ns[Phase::FenceWait as usize];
        assert_eq!(
            total,
            local.wait_idle_ns[slot] + local.wait_blocked_ns[slot]
        );
        assert_eq!(total, 1_000 + IDLE_NS_DEFAULT + 5_000_000);
        assert_eq!(local.wait_idle_calls[slot], 1);
        assert_eq!(local.wait_blocked_calls[slot], 2);
        assert_eq!(local.wait_timeout_calls[slot], 1);
        assert_eq!(local.calls[slot], 3);
        // Each wait bar keeps its own buckets: a render wait must not land in
        // the submission-level wait's idle reading.
        let render = Phase::RenderWait as usize;
        assert_eq!(local.ns[render], 7);
        assert_eq!(local.wait_idle_calls[render], 1);
        assert_eq!(local.calls[render], 1);
    }

    /// A window drains on the `total` bar that fills it, and a drained window
    /// starts empty: a line whose counters kept the previous window's time
    /// would double a phase in the table a reader is summing.
    #[test]
    fn a_window_closes_on_the_total_bar_that_fills_it() {
        let mut local = Local::default();
        let window = every();
        for _ in 0..window {
            local.charge(Phase::Plan, 10);
            local.note_total(100);
        }
        assert_eq!(local.window, 0, "the window must have been drained");
        assert_eq!(local.ns[Phase::Total as usize], 0);
        assert_eq!(local.ns[Phase::Plan as usize], 0);
        local.charge(Phase::Plan, 7);
        assert_eq!(local.ns[Phase::Plan as usize], 7);
    }
}
