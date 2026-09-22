//! Handing a draw list's per-draw objects back once the fence has signalled.
//!
//! # What this is for
//!
//! A render pass that carries an ordered list of draws (`research/docs/23`
//! §3.3, G3-B/B-2) builds the draws after the head into objects of their own —
//! their pipelines, their uploaded vertex/index/stage buffers and their sampled
//! textures (`OffscreenObjects::build_draw_objects`). Those objects are read by
//! the command buffer until the pass's fence, exactly like the pass's own set,
//! and the pass's own set hands its four pooled families back *before* it drops
//! (`OffscreenObjects::release_reusable`, `release_pooled_textures`,
//! `release_imported_windows`, `release_pooled_uploads`). The list's members did
//! not: `finish_prepared_offscreen_pass` took `extra_draws` and dropped them, so
//! every member after the head built its pair and destroyed it again instead of
//! handing it to the pool the head's own set fills.
//!
//! The teardown round (`ct2`, E `ecfb012` × R `9ab840c6`, production pose, 28 152
//! submissions) read exactly that:
//!
//! | reading | value | what it says |
//! |---|---|---|
//! | `buf_miss_n` (upload pool) | 30 391 | a creation the pool did not serve |
//! | `td_unreturned_n` | 30 329 | one destroyed while still holding its pool key |
//! | of which `stage` / `vertex` / `input_index` | 13 955 / 8 050 / 8 052 | the three families a per-draw set owns |
//! | misses that found the pool empty | 3 | not a cold pool: the shape was never handed back |
//! | misses that followed an eviction | 0 | not the caps either |
//! | `td_memory_us` | 358.4 µs a submission | 149 345 ns per `vkFreeMemory`, against 760 ns per `vkDestroyBuffer` |
//!
//! Every one of those 30 329 objects sat in a submission the batch did not
//! carry (the draw list is its own submission scope), and a set that is dropped
//! without its hand-back is a shape the pool can never hold: the next draw of
//! the same list asks again and builds again.
//!
//! # What the release does, and what it cannot break
//!
//! With the switch on, each of the list's per-draw sets hands its four families
//! back in the order the pass's own set already uses, after the fence has proven
//! the device done with them and before the set drops:
//!
//! | family | hand-back | what the pool does with it |
//! |---|---|---|
//! | pipeline, layouts, modules | `release_reusable` | shape cache (`crate::render_setup_reuse`) |
//! | sampled texture backings | `release_pooled_textures` | `crate::render_texture_pool` |
//! | owner-window imports | `release_imported_windows` | `crate::render_import_pool` |
//! | host-visible upload pairs | `release_pooled_uploads` | `crate::render_buffer_pool` |
//!
//! Nothing else about the member changes: the sets are still built, recorded and
//! submitted exactly as before, and the bytes a member uploads are still its own
//! (a pooled pair is an allocation the driver is handed once, not a copy of
//! anybody's texels — `crate::render_buffer_pool` states that rule).
//!
//! Three properties keep the release the same *kind* of act as the pass's own:
//!
//! * **It runs after the fence.** The release sits in
//!   `finish_prepared_offscreen_pass`, after `vkWaitForFences` has retired the
//!   submission — the same place the pass's own four releases run, and the only
//!   point at which the device is provably done with the handles.
//! * **A member's readback is not affected.** The pass's readback
//!   (`read_back_offscreen`) reads the *pass's* objects and decisions; a
//!   member's own set contributes no attachment images and no writeback
//!   destination to it (`OffscreenObjects::stage_buffer_readback_bytes` is
//!   called on the pass's set alone), so the release cannot move a byte a
//!   readback would have read.
//! * **Off is the pre-cut path, byte for byte.** The switch is **off unless
//!   asked for**: with it unset the list's members drop exactly as they always
//!   have, destroying what they built.
//!
//! # The switch and the readings beside it
//!
//! `METAL_API_VULKAN_DRAW_OBJECT_RELEASE=1` (also `on`, `true`, `yes`) turns the
//! hand-back on; unset and every other spelling is the control arm a round
//! compares against.
//!
//! The counters that move are the ones the round above reads: the per-kind
//! `buf_return_<kind>_n` rises by the objects the switch stops destroying,
//! `td_unreturned_<kind>_n` and `buf_miss_<kind>_n` fall to the pass's own share,
//! `pool_return_n` rises for the texture backings, and `render_teardown_us` and
//! `render_setup_us` fall together (one object's create/allocate/bind *and* its
//! free stop happening per member). `render_release_draws_us` is the hand-back's
//! own cost, so "give it back is free" is a reading rather than an assumption.

use std::sync::OnceLock;

/// Whether a draw list's per-draw objects hand their pooled families back
/// before they drop, read once from the process environment.
pub(crate) fn enabled_from_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_VULKAN_DRAW_OBJECT_RELEASE")
                .ok()
                .as_deref(),
        )
    })
}

/// The switch's own reading: off unless a word asks for it.
///
/// The pre-cut path is the default because the cut is a *mechanism* change to
/// when a device object dies, and a round has to be able to state the arm it
/// compares against without setting anything. The control words are the same
/// five the other post-cut switches accept, and every other value (including
/// unset and the empty string) is the control arm.
fn parse_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "on" | "true" | "yes")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off unless a word asks for it: the pre-cut path stays the default arm,
    /// and the words that turn it on are the five the other switches accept.
    #[test]
    fn the_switch_is_off_unless_a_word_asks_for_it() {
        assert!(!parse_enabled(None));
        assert!(!parse_enabled(Some("")));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("off")));
        assert!(!parse_enabled(Some("false")));
        assert!(!parse_enabled(Some("no")));
        assert!(parse_enabled(Some("1")));
        assert!(parse_enabled(Some("on")));
        assert!(parse_enabled(Some("ON")));
        assert!(parse_enabled(Some("true")));
        assert!(parse_enabled(Some("yes")));
        assert!(parse_enabled(Some(" yes ")));
    }
}
