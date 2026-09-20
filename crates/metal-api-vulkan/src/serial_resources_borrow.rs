//! The submission's resource pool borrowing the trace's own declarations.
//!
//! # What this is for
//!
//! `docs/SUBMIT-PHASE-PROFILE.md` divides one `VulkanComputeProvider::submit`,
//! and the sixth cut split the seam its fifth cut left. That split's ranking
//! (`docs/SUBMIT-BINDING-BORROW.md` §5) named the same batch of bytes three
//! times over: every submission declares about **1.18 MB** of buffer views with
//! their own bytes (`BufferSource::OwnedBytes`), and the call held **three
//! copies** of them.
//!
//! | copy | where it is made | freed by | state |
//! |---|---|---|---|
//! | the plan's serial resource pool | `ComputeTrace::serial_resources`, derived in `plan` | the release tail | **this cut** |
//! | one per pooled binding | `Vec::clone` of the view's bytes | `submit_release_bindings` | removed by the sixth cut |
//! | the terminal validation's own tables | `ComputeTrace::serial_resources` derived again in `submit_validate` | the seam inside `submit_validate` (43.0 µs/submission) | **this cut** |
//!
//! The derivation is a pure function of the borrowed trace: it walks the
//! declarations, merges each view's access over its uses and returns a table in
//! first-use order. What it returns is a *view of the trace*, so a caller that
//! only reads it — a plan sizing bindings and uploading their bytes, a
//! validation walking writebacks against identities and ranges — has no reason
//! to own it, and owning it is what clones the bytes.
//!
//! This module's mechanism is `ComputeTrace::serial_resources_ref`: the pool is
//! derived as a table of [`metal_api_core::provider::SerialResource`] — each
//! entry the trace's own declaration plus the access this submission merged
//! over it — so `plan` holds no copy and `submit_validate` hands the same table
//! to the combined check instead of deriving a second one. The copy's cost was
//! paid twice per derivation, once to clone and once to free, and the mechanism
//! removes both halves of both derivations.
//!
//! # Why the device cannot tell
//!
//! * **The bytes are the same bytes.** A pooled binding hands `create_buffers`
//!   exactly the slice it handed before: the trace's own view source, same
//!   pointer, same length, same order. Nothing about the buffer the driver
//!   creates, the mapping, the upload region or the recorded commands reads the
//!   host vector's identity — the upload is one `copy_nonoverlapping`.
//! * **The borrow cannot dangle.** The derived table borrows the trace, which
//!   `submit` owns for the whole call (`ValidatedComputeTrace` is its own
//!   binding); the table is declared after the trace and dropped before it, and
//!   the owned table the control arm keeps is declared *before* the table that
//!   borrows it, so it outlives its own borrow. The deferred arm stores no
//!   borrow: the completion slot carries the pool's identity and geometry,
//!   which is all a deferred readback resolves a landing with, and the two
//!   tables themselves drop at the end of the call exactly as they do on the
//!   synchronous path.
//! * **Nothing else changes.** The pool has the same length, the same order,
//!   the same merged accesses and the same declared bytes; the planner's view
//!   windows, the binding table's shape, the render rail's resolutions and the
//!   writeback mapping are untouched. The render rail keeps its own references
//!   into the trace's declarations, which is what it already read.
//!
//! # The switch and its counters
//!
//! `METAL_API_VULKAN_SUBMIT_RESOURCE_BORROW=1` (also `on`, `ON`, `true`,
//! `yes`) turns the mechanism on; **off is the default** and every other value
//! — including unset — leaves both derivations owning their views, which is the
//! pre-cut path.
//!
//! The phase line counts both arms beside the bars they move
//! (`submit_resource_copies_n` / `_bytes` for the bytes a derivation copied,
//! `submit_resource_borrows_n` / `_bytes` for the same bytes lent), so a round
//! reads how many bytes the mechanism took off `plan` and `submit_validate`
//! rather than inferring it. Off, one submission's two derivations report two
//! copies of its declared bytes; on, it reports two borrows of them and no
//! copies.

use std::sync::OnceLock;

/// Whether the submission's resource pool borrows the trace's declarations
/// instead of cloning them, read once from the process environment.
pub(crate) fn enabled_from_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_VULKAN_SUBMIT_RESOURCE_BORROW")
                .ok()
                .as_deref(),
        )
    })
}

/// The switch's own reading: **on unless the variable turns it off**
/// (`0`/`off`/`false`/`no`), the default B-1 took on 2026-09-20. Unset is on:
/// the mechanism removes two copies the submission made for itself (the seventh
/// knife's A/B read the four bars together −389.6/−402.7 µs per submission, the
/// same sum differing by 13.1 between two identical control arms, with the byte
/// counters showing 2.04–2.42 MB of copies become 0 and 2.48 MB of borrows),
/// and the off arm stays reachable as the control.
fn parse_enabled(value: Option<&str>) -> bool {
    !matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On unless the variable turns it off: the pre-cut path stays reachable as
    /// the control a round compares against.
    #[test]
    fn the_switch_is_on_unless_the_variable_turns_it_off() {
        assert!(parse_enabled(None));
        assert!(parse_enabled(Some("")));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("off")));
        assert!(!parse_enabled(Some("no")));
        assert!(!parse_enabled(Some("false")));
        assert!(!parse_enabled(Some("OFF")));
        assert!(!parse_enabled(Some("No ")));
        assert!(!parse_enabled(Some("False")));
        assert!(parse_enabled(Some("1")));
        assert!(parse_enabled(Some("on")));
        assert!(parse_enabled(Some("borrow-resources")));
    }
}
