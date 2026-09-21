//! One derivation of the submission's serial resource pool.
//!
//! # What this is for
//!
//! `docs/SUBMIT-PHASE-PROFILE.md` divides one `VulkanComputeProvider::submit`,
//! and the shape round (tag `ct1`, E `537c1d1` × R `a31bf9f9`) read the
//! submission line's own distribution: the top 5% of submissions carry 39.3% of
//! the round's `total`, and two of the lanes that tail is made of are the same
//! work done twice —
//!
//! | lane | mean | in the top 5% | what it is |
//! |---|---|---|---|
//! | `plan_resources_us` | 15.08 ms/frame | **96.0%** | `ComputeTrace::serial_resources_ref` in `plan` |
//! | `submit_validate_derive_us` | 15.43 ms/frame | **93.5%** | the same call again, in `submit_validate` |
//!
//! The two calls are the same pure function over the same borrowed trace, and
//! the pairs agree submission by submission: over the round's tail, the
//! plan-side mean is 4 151.2 µs and the validate-side mean is 4 138.6 µs
//! (−12.6 µs apart), while the *shape* the tail is made of — a single
//! unbatched pass over a large declaration — is what makes each call expensive
//! (`views_bytes` > 8 MiB: 887.4 µs in `plan` against 153.1 µs under 64 KiB).
//! A submission therefore derived its pool twice and paid the walk twice.
//!
//! The mechanism is a hand-over rather than a cache: the table `plan` derived
//! is still alive when the terminal validation runs (the same `pool` the
//! executor, the render rail and the pool's own geometry already read), so the
//! validation walks **that** table instead of deriving a second one. Nothing is
//! memoized, nothing is keyed, and no second derivation happens anywhere: the
//! submission's derivation count is the reading
//! (`pool_derivations_n`: two per submission off, one on).
//!
//! # Why the device cannot tell
//!
//! * **The table is the same table.** `serial_resources_ref` is a pure walk
//!   over `&self`: same length, same first-use order, same merged accesses,
//!   same declared bytes. The trace is immutable for the whole call, and the
//!   plan-side table is derived from the same trace the validation would have
//!   derived its own from.
//! * **The borrow cannot dangle.** The table is declared in the plan, before
//!   the executor, the bindings and the render rail that also read it, and it
//!   is dropped at the end of the call after the validation's walk — the order
//!   the pre-cut code's second table had anyway (it was dropped in
//!   `submit_release`).
//! * **Nothing else changes.** The walk's inputs are the same references; the
//!   writeback mapping, the coverage walk, the texture pool and the release
//!   accounting all read what they read before. The only observable difference
//!   is the second derivation's own microseconds and its share of the byte
//!   counters.
//!
//! # The switch and its counters
//!
//! **Unset is on** (flipped 2026-09-22 once the round's two arms had read it:
//! `pool_derivations_n` 2.000 → 1.000 on every submission, `views_n` 2.014 →
//! 1.004 and the tail's top-5 % `total` 21 143 → 17 417 µs). The control words
//! `0` / `off` / `false` / `no` derive the validation's own table again — the
//! pre-cut path a round compares against.
//!
//! Two counters read the mechanism beside the bars it moves:
//!
//! * `pool_derivations_n` — the pool derivations one submission made. Two off
//!   (plan and validation), one on. The count does not depend on how much the
//!   submission declares, so a submission with an empty pool reads 2 → 1 as
//!   well.
//! * `submit_resource_borrows_n` / `_bytes` — the declared bytes the
//!   derivations moved, which read `2N` off and `N` on for a submission that
//!   declares `N` (`crate::serial_resources_borrow` owns that reading).
//!
//! On the sample line the same count is printed per submission as
//! `pool_derivations_n`, which is the reading a round's two arms compare
//! distributions of, not means of.

use std::sync::OnceLock;

/// Whether the terminal validation walks the plan's own resource table instead
/// of deriving a second one, read once from the process environment.
pub(crate) fn enabled_from_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_VULKAN_SUBMIT_POOL_ONCE")
                .ok()
                .as_deref(),
        )
    })
}

/// The switch's own reading: **on unless the variable turns it off**
/// (`0`/`off`/`false`/`no`), the tenth cut's flip after its round read it:
/// `pool_derivations_n` 2.000 → 1.000 on every submission, `views_n` 2.014 →
/// 1.004, `submit_validate_derive` 247.1 → 0.6 µs a submission (−99.8 %) and
/// the tail's top-5 % `total` 21 143 → 17 417 µs (−17.6 %), with the three
/// untouched lanes moving 1.5–8 % on a host that differs 7 % between arms.
/// The control words restore the pre-cut path byte for byte.
fn parse_enabled(value: Option<&str>) -> bool {
    !matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On unless a control word turns it off: the pre-cut path stays reachable
    /// as the control arm.
    #[test]
    fn the_switch_is_on_unless_a_control_word_turns_it_off() {
        assert!(parse_enabled(None));
        assert!(parse_enabled(Some("")));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("off")));
        assert!(!parse_enabled(Some("false")));
        assert!(!parse_enabled(Some("no")));
        assert!(!parse_enabled(Some("OFF")));
        assert!(parse_enabled(Some("1")));
        assert!(parse_enabled(Some("on")));
        assert!(parse_enabled(Some("ON")));
        assert!(parse_enabled(Some("true")));
        assert!(parse_enabled(Some("yes")));
        assert!(parse_enabled(Some(" yes ")));
    }
}
