//! One structural validation of the trace per admission walk.
//!
//! [`ComputeTrace::validate`](crate::provider::ComputeTrace::validate) is the
//! admission walk's own first region — and it is *also* the first line of two
//! later regions, because the two entries those regions call are public
//! entry points a caller may reach without having admitted anything first:
//!
//! * `ComputeTrace::validate_serial_buffer_reuse` states the serial
//!   resource-reuse subset and opens with `self.validate()?`;
//! * `ResourceTableSnapshot::validate_trace` states the resource table's own
//!   agreement with the trace and opens with `trace.validate()?`.
//!
//! Inside [`ProviderCapabilities::admit`](crate::provider::ProviderCapabilities::admit)
//! the trace has already been validated by the time those two regions run: the
//! walk's first region validates it and refuses there, `&ComputeTrace` is
//! immutable for the whole walk, and `validate()` is a pure function of that
//! borrow. The two later calls therefore re-answer, byte for byte, a question
//! this walk has already answered.
//!
//! The tenth profile round priced exactly that on its arm-on production pose
//! (`sa2b`, 2026-09-21, one walk of `route=validate`): `trace_validate` 23.70 +
//! `serial_reuse` 23.16 + `resources` 20.92 = **67.8 µs per walk**, 55 % of the
//! 123 µs walk — and the two later regions are nothing but the repeated
//! validation, because their own bodies are a handful of `Vec`/`BTreeMap` scans
//! (the same round's `sa1` split table reads the same three bars at 24.21 /
//! 23.49 / 21.42 µs on a 263 µs walk, i.e. the later two bars *are* the first).
//!
//! This cut makes one walk validate once, driven by
//! `METAL_API_CORE_VALIDATE_ONCE`:
//!
//! * unset (the default), or any word that is not one of the four enable words:
//!   off. Every region calls its own public entry and the trace is validated
//!   three times, statement for statement as before.
//! * `1` / `on` / `true` / `yes` (case-insensitive and whitespace-trimmed): on.
//!   The first region keeps its `trace.validate()`, and the two later regions
//!   call **crate-private** entries that are the very same bodies with that one
//!   first line dropped
//!   (`ComputeTrace::validate_serial_buffer_reuse_after_validate`,
//!   `ResourceTableSnapshot::validate_trace_after_validate`).
//!
//! # Why the answers cannot differ
//!
//! 1. `validate()` is a pure function of `&ComputeTrace`: it reads the trace and
//!    returns an answer, and the trace owns no interior mutability a reader can
//!    observe. The walk holds `&ComputeTrace` across every region, so nothing
//!    (in the walk or beside it) can hand the later regions a different value
//!    than the one the first region read.
//! 2. The first region **refuses the walk** on `Err`, so the later regions are
//!    reached only when `validate()`'s answer was `Ok` — which is why dropping
//!    their repeated call cannot turn an admitted trace into a refused one or
//!    the other way around, and cannot move a refusal to another name.
//! 3. What the later regions keep is their own bodies, unchanged: the serial
//!    pool's own rules still run, the resource table's own agreement walk still
//!    runs, in the same order, and each still refuses by the same name.
//!
//! So the two arms differ in how many times the same question is asked of the
//! same bytes, and in nothing else. `metal_api_core::admit_profile`'s three
//! bars name the difference: on the cut's arm the later two bars fall to their
//! bodies, and the walk's total falls by the two removed passes.
//!
//! # Why the entries are crate-private
//!
//! A caller that has *not* validated the trace must not be able to skip the
//! validation, so "skip it" is not a public mode of either entry: the two
//! `_after_validate` functions are `pub(crate)` and the walk is their only
//! caller. Outside this crate — the reims rail, `metal-api-vulkan`, the native
//! provider — the public entries keep validating, whatever the switch says.

use std::sync::OnceLock;

/// A test's own arm, so one process can drive a walk both ways and compare the
/// answers (`0` follows the environment, `1` is on, `2` is off).
#[cfg(test)]
static ARM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether one admission walk validates the trace once instead of three times.
///
/// Read once from the process environment; **off unless a word turns it on** —
/// the cut's own switch is spelled this way so an environment that says nothing
/// (and every environment a round does not touch on purpose) keeps the
/// statement-for-statement pre-cut path. Every call site is one relaxed load.
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
            std::env::var("METAL_API_CORE_VALIDATE_ONCE")
                .ok()
                .as_deref(),
        )
    })
}

/// Force one arm for this process, or hand the question back to the
/// environment with `None`.
///
/// The mechanism changes how many times a walk asks the trace the same question
/// and nothing else, so a test that pins the two arms' answers can also be the
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
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "on" | "true" | "yes")
    )
}

#[cfg(test)]
mod tests {
    use super::parse_enabled;

    /// Off is the default, and only the four enable words turn the cut on: a
    /// word this cut does not know keeps the pre-cut path.
    #[test]
    fn the_switch_is_off_unless_an_enable_word_turns_it_on() {
        assert!(!parse_enabled(None));
        assert!(!parse_enabled(Some("")));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("off")));
        assert!(!parse_enabled(Some("OFF")));
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
