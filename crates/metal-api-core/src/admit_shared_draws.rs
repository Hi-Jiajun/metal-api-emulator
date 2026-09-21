//! One materialization of the render entries per admission walk.
//!
//! [`ComputeTrace::render_draw_passes`](crate::provider::ComputeTrace::render_draw_passes)
//! states a multi-draw pass — B-2's `TracePass::RenderDraws` — as one
//! single-draw pass per draw, which is what makes "N draws in one pass" exactly
//! "N passes, one draw each" for every rule the contract states. The iterator
//! is lazy and borrows a single-draw entry, but a **list** entry it has to
//! materialize: `RenderDrawsDescriptor::materialize` clones the pass state and
//! every draw's own declarations — its vertex streams, sampled textures,
//! runtime samplers and stage buffers, the last of which carry the view's own
//! bytes or runs.
//!
//! Four admission gates walk those entries (`admit_render_passes`,
//! `admit_render_texture_inputs`, `admit_render_pixel_samplers`,
//! `admit_render_stage_buffer_inputs`), and each one builds **its own**
//! iterator. On the tenth profile round's production pose (`sa1`, 2026-09-21)
//! the four bars read 48.7 / 48.1 / 47.7 / 46.6 µs per walk — 74 % of a 263 µs
//! walk — while the gates' own bodies are a handful of `Vec` scans: the
//! microseconds are the materializations, paid once per gate.
//!
//! This cut materializes once per walk and hands the same values to every gate,
//! driven by `METAL_API_CORE_ADMIT_SHARED_DRAWS`:
//!
//! * unset (the default), or `0` / `off` / `false` / `no` (case-insensitive and
//!   whitespace-trimmed): off. Every gate builds its own iterator, byte for byte
//!   as before.
//! * `1` / `on` / `true` / `yes`: on. The walk builds one
//!   `Vec<(usize, Cow<RenderPassDescriptor>)>` before the first gate and each
//!   gate walks that.
//!
//! # Why the answers cannot differ
//!
//! 1. `materialize` is a pure function of the list it is called on: it clones
//!    the head and builds one pass per tail draw through `with_draw`. Two calls
//!    on the same list produce equal values.
//! 2. The gates only *read* the pass they are handed (`&RenderPassDescriptor`);
//!    nothing on the walk mutates a materialized pass or the trace it came
//!    from (`&ComputeTrace` outlives every borrow here).
//! 3. The gate order, the per-entry order and the refusal each gate raises are
//!    unchanged: the same values in the same order reach the same checks.
//!
//! So the two arms differ in how many times the same bytes are cloned, and in
//! nothing else. `metal_api_core::admit_profile`'s
//! `draw_list_materialize_n` / `_passes` meter counts the events on both arms,
//! and the four gate bars fall by the work this cut removes.

use std::sync::OnceLock;

/// A test's own arm, so one process can drive the walk both ways and compare
/// the answers (`0` follows the environment, `1` is on, `2` is off).
#[cfg(test)]
static ARM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether the walk materializes the render entries once instead of per gate.
///
/// Read once from the process environment; off is the default and every call
/// site is one relaxed load.
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
            std::env::var("METAL_API_CORE_ADMIT_SHARED_DRAWS")
                .ok()
                .as_deref(),
        )
    })
}

/// Force one arm for this process, or hand the question back to the
/// environment with `None`.
///
/// The mechanism changes how many times a walk clones a list and nothing else,
/// so a test that pins the two arms' answers can also be the reading that the
/// arms are interchangeable.
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
    match value.map(str::trim).map(str::to_ascii_lowercase) {
        Some(word) => matches!(word.as_str(), "1" | "on" | "true" | "yes"),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::parse_enabled;

    /// Off is the default, and the spellings a launcher may write are the ones
    /// the other cuts' switches accept.
    #[test]
    fn the_switch_is_off_unless_a_truthy_word_says_otherwise() {
        assert!(!parse_enabled(None));
        assert!(!parse_enabled(Some("")));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("off")));
        assert!(!parse_enabled(Some("OFF")));
        assert!(!parse_enabled(Some("false")));
        assert!(!parse_enabled(Some("no")));
        assert!(!parse_enabled(Some("nope")));
        assert!(parse_enabled(Some("1")));
        assert!(parse_enabled(Some("on")));
        assert!(parse_enabled(Some("ON")));
        assert!(parse_enabled(Some(" true ")));
        assert!(parse_enabled(Some("yes")));
    }
}
