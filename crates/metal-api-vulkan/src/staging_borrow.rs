//! The bytes a staged lease hands a binding: the registry's own, not a copy of
//! them.
//!
//! # What this is for
//!
//! One submission's bindings come from three places, and two of them have been
//! cut already. The sixth cut (`crate::submit_binding_borrow`) lets a binding
//! hold a *borrow* of the trace's own snapshot bytes, which the submission's
//! serial resource pool already keeps for the whole call. The seventh cut
//! (`crate::serial_resources_borrow`) lends the derived pool itself. What is
//! left is the arm both those cuts' tables listed as "cannot be borrowed": a
//! **staged lease**'s window (`BufferSource::StagedLease` /
//! `TextureSource::StagedLease`).
//!
//! The bytes of that arm live in the provider's staging registry
//! ([`LeaseRegistry`]), imported by the owner before the submission and released
//! after it completes. The registry's copying entry point
//! ([`LeaseRegistry::view_bytes`] / `::texture_bytes`) clones the window out of
//! its own lock, so every resolved view pays one allocation, one `memcpy` of
//! the window and one `free` — the *third* copy of the same owner bytes the two
//! earlier cuts priced, and the `sp20` round read it at 0.29 MB per submission
//! (11–13 GB per 300 s round).
//!
//! This module is that arm: the registry hands the binding a **handle on the
//! bytes it already holds** ([`LeaseWindowBytes`]) instead of a copy of them,
//! and nothing else about the submission changes.
//!
//! # Why a reference could not be lent, and a handle can
//!
//! The two earlier cuts lent a `&[u8]` because the bytes they lent belonged to
//! something the submission already held for the whole call: the trace, via the
//! serial resource pool. Here the holder is the registry, and its bytes sit
//! behind its own `Mutex`. Lending a reference would mean lending the guard with
//! it, which is not a longer-lived-object question but a deadlock:
//!
//! * the same submission resolves more than one staged window (one binding
//!   each), and the second resolution would take the lock the first is still
//!   holding — `std::sync::Mutex` is not reentrant;
//! * the owner releases the same lease on the submission's completion path
//!   (`provider_owner`'s `settle` → `release_staged_lease` → `LeaseRegistry::release`),
//!   which takes the same lock, and a guard held across the upload would make
//!   that release wait on work that cannot finish until it returns;
//! * every other thread's `import`/`release` would block for the length of a
//!   submission's device work rather than for the length of one map lookup.
//!
//! So the registry lends the *bytes* rather than a borrow of them: it holds each
//! import by handle (`Arc<StagedLease>`, `LeaseRegistry`'s storage) and answers a
//! window with another handle on the same allocation ([`LeaseWindowBytes`]). The
//! lock is taken for the checks and dropped before the handle escapes, the bytes
//! are not copied, and the window outlives the import exactly as long as the
//! copy outlived it — the property the callers actually relied on.
//!
//! # Why the device cannot tell
//!
//! * **The bytes are the same bytes.** `BindingBytes::Staged(window)` hands
//!   `create_buffers` exactly the slice `BindingBytes::Copied(registry.view_bytes(..))`
//!   handed it: the window is checked against the same reservation, by the same
//!   names in the same order (`LeaseRegistry::view_window` *is* the copying
//!   entry point's window, minus the `to_vec`), and the upload reads it with the
//!   same pointer, length and order. The driver never learned the host vector's
//!   identity.
//! * **The bytes cannot dangle or be dropped early.** The handle owns a
//!   reference to the import, so the bytes stay alive as long as any binding
//!   that reads them — including past `release_staged_lease`, which is exactly
//!   the lifetime the copy gave (`Arc` is what makes it checkable rather than
//!   argued).
//! * **Nothing else changes.** The checks, their order and their refusals are the
//!   registry's own; the binding table's shape, its widths (`len()`), the
//!   planner's view windows, the writeback mapping and the recording are
//!   untouched. The only difference is which allocation the bytes live in: the
//!   import's, rather than a fresh one per resolution.
//!
//! # The switch and its counters
//!
//! `METAL_API_VULKAN_STAGING_BORROW=1` (also `on`, `ON`, `true`, `yes`) turns the
//! mechanism on; **off is the default** and every other value — including unset —
//! leaves the copy where it was, which is the pre-cut path.
//!
//! The phase line counts both arms beside the bar they move
//! (`staging_window_copies_n` / `_bytes` for the copies,
//! `staging_window_shares_n` / `_bytes` for the windows lent by handle), so a
//! round reads how many bytes the mechanism took out of the submission rather
//! than inferring it.

use std::sync::OnceLock;

use metal_api_core::provider::{
    BufferView, DeviceEpoch, LeaseId, LeaseRegistry, LeaseWindowBytes, ProviderError,
    ResourceTableSnapshot, TextureView,
};

use crate::BindingBytes;

/// Whether a staged lease's window is lent by handle instead of copied, read
/// once from the process environment.
pub(crate) fn enabled_from_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_VULKAN_STAGING_BORROW")
                .ok()
                .as_deref(),
        )
    })
}

/// The switch's own reading: **on unless the variable turns it off**
/// (`0`/`off`/`false`/`no`, case-insensitively) — the eighth cut was flipped
/// on once its A/B had priced it (the `sp23`/`sb` arms read
/// `staging_window_us` 66.231 → 0.949 → 62.093 µs per submission, 65× the
/// spread between the two identical control arms, with 6.517 → 0 copies per
/// submission and 0 → 6.557 windows lent per submission). The off arm stays
/// reachable as the control a round compares against.
fn parse_enabled(value: Option<&str>) -> bool {
    !matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// Resolve one staged **view** into the bytes a binding uploads: the registry's
/// window by handle (on) or its copy (off).
pub(crate) fn resolve_view<'a>(
    registry: &LeaseRegistry,
    lease_id: LeaseId,
    view: &BufferView,
    device_epoch: DeviceEpoch,
    resources: &ResourceTableSnapshot,
) -> Result<BindingBytes<'a>, ProviderError> {
    resolve_view_with(
        enabled_from_env(),
        registry,
        lease_id,
        view,
        device_epoch,
        resources,
    )
}

/// [`resolve_view`] with the arm stated rather than read from the process: the
/// shape its own tests drive both arms through, and the shape the two rails
/// reach by setting the variable for their whole process.
fn resolve_view_with<'a>(
    lend_by_handle: bool,
    registry: &LeaseRegistry,
    lease_id: LeaseId,
    view: &BufferView,
    device_epoch: DeviceEpoch,
    resources: &ResourceTableSnapshot,
) -> Result<BindingBytes<'a>, ProviderError> {
    resolve(
        lend_by_handle,
        || registry.view_bytes(lease_id, view, device_epoch, resources),
        || registry.view_window(lease_id, view, device_epoch, resources),
    )
}

/// Resolve one staged **texture** into the bytes a pass uploads: the same two
/// arms, landed through the registry's texture entry points.
pub(crate) fn resolve_texture<'a>(
    registry: &LeaseRegistry,
    lease_id: LeaseId,
    texture: &TextureView,
    device_epoch: DeviceEpoch,
    resources: &ResourceTableSnapshot,
) -> Result<BindingBytes<'a>, ProviderError> {
    resolve(
        enabled_from_env(),
        || registry.texture_bytes(lease_id, texture, device_epoch, resources),
        || registry.texture_window(lease_id, texture, device_epoch, resources),
    )
}

/// The region both arms are resolved in, timed and counted
/// (`phase_profile::Phase::StagingWindow`), and the arm itself.
///
/// The bar covers whichever arm runs, which is what makes the two comparable:
/// off it is the copy, on it is the handle. The refused path is inside the bar
/// too — a refusal costs a lock and a check on either arm.
fn resolve<'a>(
    lend_by_handle: bool,
    copy: impl FnOnce() -> Result<Vec<u8>, ProviderError>,
    lend: impl FnOnce() -> Result<LeaseWindowBytes, ProviderError>,
) -> Result<BindingBytes<'a>, ProviderError> {
    let _window = crate::phase_profile::Bar::enter(crate::phase_profile::Phase::StagingWindow);
    if lend_by_handle {
        let window = lend()?;
        crate::phase_profile::note_staging_window_share(window.len() as u64);
        Ok(BindingBytes::Staged(window))
    } else {
        let bytes = copy()?;
        crate::phase_profile::note_staging_window_copy(bytes.len() as u64);
        Ok(BindingBytes::Copied(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::provider::{
        AllocationId, AllocationRecord, BufferAccess, BufferLease, BufferSource, LeaseReservation,
        StagedLease, TextureAccess, TextureFormat, TextureSource, TextureType, ViewId,
    };

    /// One registry with one imported lease, one admitted allocation and one
    /// view naming a window inside both: the smallest fixture the two arms can
    /// be driven through.
    fn fixture() -> (LeaseRegistry, BufferView, ResourceTableSnapshot) {
        let registry = LeaseRegistry::new();
        let reservation = LeaseReservation {
            lease: BufferLease {
                lease_id: LeaseId::new(1),
                allocation_id: AllocationId::new(2),
                owner_epoch: DeviceEpoch::new(1),
            },
            offset: 4096,
            length: 8192,
        };
        let bytes: Vec<u8> = (0..8192).map(|i| (i % 251) as u8).collect();
        registry
            .import(StagedLease::new(reservation, bytes).unwrap())
            .unwrap();
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(2),
                owner_epoch: DeviceEpoch::new(1),
                size: 65536,
            })
            .unwrap();
        resources.insert_lease(reservation).unwrap();
        let view = BufferView {
            view_id: ViewId::new(7),
            metal_binding: 0,
            allocation_id: AllocationId::new(2),
            offset: 4112,
            length: 1024,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::StagedLease(LeaseId::new(1)),
        };
        (registry, view, resources)
    }

    /// The two arms hand the device the same bytes, and only one of them moves
    /// them.
    ///
    /// The bytes are asserted three ways: against the window the registry's own
    /// copy answers with, against the offset the view names inside the
    /// reservation (so a window read from the wrong end would fail), and by
    /// allocation — two resolutions of one view share a pointer on the handle
    /// arm and do not on the copy arm, which is the mechanism itself stated as a
    /// reading rather than argued.
    #[test]
    fn the_two_arms_hand_over_the_same_bytes_from_different_allocations() {
        let (registry, view, resources) = fixture();
        let epoch = DeviceEpoch::new(1);
        let lease = LeaseId::new(1);

        let lent = resolve_view_with(true, &registry, lease, &view, epoch, &resources).unwrap();
        let copied = resolve_view_with(false, &registry, lease, &view, epoch, &resources).unwrap();
        assert!(matches!(lent, BindingBytes::Staged(_)));
        assert!(matches!(copied, BindingBytes::Copied(_)));
        assert_eq!(lent.len(), view.length as usize);
        assert_eq!(lent.as_slice(), copied.as_slice());

        // The window is the view's own offset inside the reservation, in order:
        // the fixture's bytes are `index % 251`, and the view opens 16 bytes into
        // the reservation.
        let expected: Vec<u8> = (16..16 + 1024).map(|i| (i % 251) as u8).collect();
        assert_eq!(lent.as_slice(), expected.as_slice());
        assert_eq!(copied.as_slice(), expected.as_slice());

        // The handle arm points into the registry's import — the same allocation
        // every time — while the copy arm allocates per resolution. Asserting
        // both directions is what makes "no copy" a reading rather than a claim:
        // a mechanism that copied into a cached buffer would pass the byte
        // equality above and fail here.
        let lent_again =
            resolve_view_with(true, &registry, lease, &view, epoch, &resources).unwrap();
        let copied_again =
            resolve_view_with(false, &registry, lease, &view, epoch, &resources).unwrap();
        assert_eq!(lent.as_slice().as_ptr(), lent_again.as_slice().as_ptr());
        assert_ne!(copied.as_slice().as_ptr(), copied_again.as_slice().as_ptr());
        assert_ne!(lent.as_slice().as_ptr(), copied.as_slice().as_ptr());

        // Both arms run the registry's checks, in its order, under its names: a
        // view past the reservation is refused the same way on either arm.
        let mut outside = view.clone();
        outside.offset = 4096 + 8192;
        outside.length = 16;
        let lent_refusal =
            resolve_view_with(true, &registry, lease, &outside, epoch, &resources).unwrap_err();
        let copied_refusal =
            resolve_view_with(false, &registry, lease, &outside, epoch, &resources).unwrap_err();
        assert_eq!(lent_refusal.slug, "lease_range_out_of_bounds");
        assert_eq!(copied_refusal.slug, lent_refusal.slug);
    }

    /// The texture arm is the same mechanism through the registry's other entry
    /// point: a texture's texels are the reservation's first bytes, and the two
    /// arms read the same ones.
    #[test]
    fn the_texture_arm_reads_the_same_texels_on_either_arm() {
        let (registry, _, resources) = fixture();
        let epoch = DeviceEpoch::new(1);
        let lease = LeaseId::new(1);
        // The registry's texture arm reads the reservation's own start for the
        // texture's tightly packed extent (`LeaseRegistry::texture_bytes`), so
        // the fixture's four-byte RGBA8 texture is the reservation's first four
        // bytes.
        let texture = TextureView {
            view_id: ViewId::new(9),
            metal_binding: 0,
            allocation_id: AllocationId::new(2),
            texture_type: TextureType::D2,
            format: TextureFormat::Rgba8Unorm,
            width: 1,
            height: 1,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            access: TextureAccess::Sampled,
            source: TextureSource::StagedLease(lease),
        };
        let copied = resolve_texture(&registry, lease, &texture, epoch, &resources).unwrap();
        let expected: Vec<u8> = (0..4).map(|i| (i % 251) as u8).collect();
        assert_eq!(copied.as_slice(), expected.as_slice());
        let window = registry
            .texture_window(lease, &texture, epoch, &resources)
            .unwrap();
        assert_eq!(window.as_slice(), copied.as_slice());
    }

    /// Off unless the variable says otherwise: the copy stays the default and
    /// the pre-cut path stays reachable as the control a round compares against.
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
        assert!(parse_enabled(Some("borrow-staging")));
        assert!(parse_enabled(Some("1")));
        assert!(parse_enabled(Some("on")));
        assert!(parse_enabled(Some("ON")));
        assert!(parse_enabled(Some("true")));
        assert!(parse_enabled(Some("Yes")));
    }
}
