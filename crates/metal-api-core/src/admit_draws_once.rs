//! One construction of a multi-draw list's single-draw passes per admission walk.
//!
//! A multi-draw pass (`TracePass::RenderDraws`) is judged as "N draws in one
//! pass is N passes, one draw each": `RenderDrawsDescriptor::materialize`
//! builds those N passes, and every rule the contract states is written
//! against them. Two places build them on one admission walk:
//!
//! 1. the list's **own** validation, `RenderDrawsDescriptor::validate`, which
//!    builds `head.with_draw(draw)` for every draw of the tail, validates it,
//!    and **drops it** — the value is only ever read;
//! 2. the walk's one materialization
//!    (`ComputeTrace::render_draw_passes`, `crate::admit_shared_draws`), the N
//!    passes the four render gates then read.
//!
//! The tenth cut collapsed (2) from one per gate to one per walk. This cut
//! removes the *other* copy: the list's validation hands the passes it built to
//! the walk instead of dropping them, so one walk builds each of a list's
//! single-draw passes once, in the region it was always built in.
//!
//! Driven by `METAL_API_CORE_ADMIT_DRAWS_ONCE`:
//!
//! * unset (the default), or any word that is not a truthy one: off. The walk
//!   validates the trace through [`ComputeTrace::validate`] and then
//!   materializes the render entries for itself, exactly as before.
//! * `1` / `on` / `true` / `yes`: on. The walk validates the trace through
//!   [`ComputeTrace::validate_collecting_draws`], which runs the very same
//!   checks in the very same order and keeps the passes the list's own
//!   validation built; the gates read those.
//!
//! # Why the answers cannot differ
//!
//! 1. `validate_collecting_draws` is `validate` with a sink: every check is the
//!    same call, in the same order, on the same values (`admit_profile`'s
//!    `draw_list_materialize_n` / `draw_list_materialize_passes` meter the
//!    second construction, and the new `draw_list_validate_build_n` meters the
//!    first, so a round can read both arms).
//! 2. The passes the walk hands its gates are the values
//!    `RenderDrawPasses` would have built: the head borrowed rather than
//!    cloned (the walk holds `&ComputeTrace` throughout, and the gates only
//!    read the pass they are given), and one `head.with_draw(draw)` per tail
//!    draw, in declaration order.
//! 3. Nothing else reads the walk's materialization, and a refused trace still
//!    refuses at the same check with the same name: the ceiling test and the
//!    per-draw validation sit exactly where they sat.
//!
//! So the two arms differ in how many times the same bytes are built and in
//! nothing else: off, one list builds its draws twice per walk (once to
//! validate and once to run the gates); on, once.
//!
//! [`ComputeTrace::validate`]: crate::provider::ComputeTrace::validate
//! [`ComputeTrace::validate_collecting_draws`]: crate::provider::ComputeTrace::validate_collecting_draws

use std::sync::OnceLock;

/// A test's own arm, so one process can drive a walk both ways and compare the
/// answers (`0` follows the environment, `1` is on, `2` is off).
#[cfg(test)]
static ARM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether one admission walk builds a list's single-draw passes once.
///
/// Read once from the process environment; **off unless a truthy word turns it
/// on** — the cut lands off and the round that prices it compares the two arms
/// of this switch. Every call site is one relaxed load.
#[inline]
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    match ARM.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_CORE_ADMIT_DRAWS_ONCE")
                .ok()
                .as_deref(),
        )
    })
}

/// Force one arm for this process, or hand the question back to the
/// environment with `None`.
///
/// The mechanism changes how many times a walk builds the same passes and
/// nothing else, so a test that pins the two arms' answers can also be the
/// reading that the arms are interchangeable.
#[cfg(test)]
pub(crate) fn set_arm(arm: Option<bool>) {
    ARM.store(
        match arm {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// The enable word, read exactly as a round's launcher spells it.
fn parse_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1" | "on" | "ON" | "true" | "yes")
    )
}

#[cfg(test)]
mod tests {
    use super::parse_enabled;

    /// Off is the default, and only the truthy words turn the cut on: a word
    /// this cut does not know keeps the pre-cut path.
    #[test]
    fn the_switch_is_off_unless_a_truthy_word_turns_it_on() {
        assert!(!parse_enabled(None));
        assert!(!parse_enabled(Some("")));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("off")));
        assert!(!parse_enabled(Some("false")));
        assert!(!parse_enabled(Some("no")));
        assert!(!parse_enabled(Some("nope")));
        assert!(!parse_enabled(Some("2")));
        assert!(parse_enabled(Some("1")));
        assert!(parse_enabled(Some("on")));
        assert!(parse_enabled(Some("ON")));
        assert!(parse_enabled(Some(" true ")));
        assert!(parse_enabled(Some("yes")));
    }
}
