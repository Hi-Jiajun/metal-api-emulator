//! The submission's pooled bindings borrowing the bytes they upload.
//!
//! # What this is for
//!
//! `docs/SUBMIT-PHASE-PROFILE.md` divides one `VulkanComputeProvider::submit`,
//! and the sixth cut split the seam its fifth cut left. The `sp16` round read
//! the release the call's own tail pays — the values `total` was declared
//! before, dropped after the last numbered bar — at 164.8 µs/submission, 5.13 %
//! of a submission and the largest region of that seam. Its three children
//! divide it, and the largest is `submit_release_bindings_us` (88.8
//! µs/submission): the pooled bindings, one per view, each carrying its own
//! copy of the view's bytes.
//!
//! Those bytes come from three sources and only the first is a copy the
//! submission makes *for itself*:
//!
//! | source | what the binding holds | can it be borrowed? |
//! |---|---|---|
//! | the trace's snapshot (`BufferSource::OwnedBytes`) | `Vec::clone` of the view's bytes | **yes** — the serial resource pool holds the same bytes for the whole call |
//! | a staged lease (`BufferSource::StagedLease`) | a fresh `Vec` copied out of the staging registry's lock | no: the registry's bytes live behind a mutex the upload does not hold |
//! | gathered guest runs (`BufferSource::GuestRuns`) | a fresh `Vec` the gather just built | no: the gather is what produces them |
//!
//! This module's mechanism is the first row: hand the binding a borrow of the
//! table the submission already holds instead of a second copy of it. The
//! copy's cost is paid twice — once to `Vec::clone` in `pool`, once to `free`
//! in the release — and the mechanism removes both halves.
//!
//! # Why the device cannot tell
//!
//! * **The bytes are the same bytes.** `BindingBytes::Borrowed(bytes)` hands
//!   `create_buffers` exactly the slice `BindingBytes::Copied(bytes.clone())`
//!   handed it: same pointer at upload time, same length, same order. Nothing
//!   about the buffer the driver creates, the mapping, the upload region or the
//!   recorded commands reads the host vector's identity — the upload is one
//!   `copy_nonoverlapping` out of the slice.
//! * **The borrow cannot dangle.** The bytes are the serial resource pool's own
//!   view sources: the pool is a local of the same `submit` call, declared
//!   before the bindings and dropped after them, and the bindings are passed on
//!   by reference and never stored. The borrow checker enforces both ends — the
//!   deferred arm has to drop the bindings before it moves the pool into its
//!   completion slot, which is why that drop is written out there.
//! * **Nothing else changes.** The other two sources keep their own `Vec`s, so
//!   the binding table's shape, its widths (`len()`), the planner's view
//!   windows and the writeback mapping are untouched.
//!
//! # The switch and its counters
//!
//! `METAL_API_VULKAN_SUBMIT_BINDING_BORROW=1` (also `on`, `ON`, `true`, `yes`)
//! turns the mechanism on; **off is the default** and every other value —
//! including unset — leaves the copy where it was. Off is the pre-cut path.
//!
//! The phase line counts both arms beside the bar they move
//! (`submit_binding_copies_n` / `_bytes` for the copies, `submit_binding_borrows_n`
//! / `_bytes` for the borrows), so a round reads how many bytes the mechanism
//! took off `pool` and the release rather than inferring it.

use std::sync::OnceLock;

/// Whether the submission's trace-supplied bindings borrow their bytes, read
/// once from the process environment.
pub(crate) fn enabled_from_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_VULKAN_SUBMIT_BINDING_BORROW")
                .ok()
                .as_deref(),
        )
    })
}

/// The switch's own reading: the control words turn the mechanism on and
/// everything else — including unset — leaves it off.
fn parse_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1" | "on" | "ON" | "true" | "yes")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off unless the variable says otherwise: this is the pre-cut path, and a
    /// round that leaves the variable alone has to read the submission the cut
    /// found.
    #[test]
    fn the_switch_is_off_unless_the_variable_says_otherwise() {
        assert!(!parse_enabled(None));
        assert!(!parse_enabled(Some("")));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("off")));
        assert!(!parse_enabled(Some("no")));
        assert!(!parse_enabled(Some("false")));
        assert!(!parse_enabled(Some("borrow-bindings")));
        assert!(parse_enabled(Some("1")));
        assert!(parse_enabled(Some("on")));
        assert!(parse_enabled(Some("ON")));
        assert!(parse_enabled(Some("ON ")));
        assert!(parse_enabled(Some("true")));
        assert!(parse_enabled(Some("yes")));
    }
}
