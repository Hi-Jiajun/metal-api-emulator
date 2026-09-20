//! The memory type a host-readback staging buffer is backed by.
//!
//! # What this is for
//!
//! Every staging buffer this rail reads — the colour, depth and stencil
//! readback destinations, the writable stage buffers' copy-out destinations and
//! the kept-frame landing's own destination — is a `TRANSFER_DST` buffer the
//! device writes and the host then reads once, through the mapping the
//! allocation returns.
//!
//! The sp7 round (`docs/READBACK-MEMORY.md` carries the reading) measured both
//! of that shape's consumers and found the same number twice:
//!
//! | region | bytes | time | rate |
//! |---|---|---|---|
//! | `readback_full` (whole-extent attachments) | 307 328 B/submit | 1 816.6 µs/submit | 169 MB/s |
//! | `landing_fetch` (kept frames) | 8 294 400 B/landing | 49 206.8 µs/landing | 168 MB/s |
//!
//! One rate at two very different sizes, with no fixed per-call part (the
//! smaller read is exactly 8.29/0.82 of a 4.87 ms read, and 4.87 ms is what the
//! whole 1 816.6 µs/submit divides into) — that is a property of the *memory the
//! mapping points at*, not of the copy or of the buffer. Both regions read
//! through `HOST_VISIBLE | HOST_COHERENT`, which is the flag pair this rail has
//! always asked for, and `VulkanContext::memory_type` answers it with the
//! *first* type that satisfies it. On a discrete device that is the host-visible
//! window over device-local memory: uncached reads over the bus, which is where
//! 168 MB/s comes from. The device's own writes into those buffers are fine
//! (the landing's fence wait is 1 679 µs for 8.29 MB ≈ 4.9 GB/s) — it is the
//! host's read of them that is slow.
//!
//! # The choice
//!
//! A readback buffer wants the memory the host reads *fast*: still visible and
//! still coherent — coherent because the read happens after the fence and this
//! rail never invalidates a mapped range — and additionally cached on the host
//! when the device states such a type. So the selection asks for
//!
//! ```text
//! HOST_VISIBLE | HOST_COHERENT | HOST_CACHED   (preferred)
//! HOST_VISIBLE | HOST_COHERENT                  (fallback: exactly the old pair)
//! ```
//!
//! and a device with only one of the two gets the same answer it got before this
//! module existed. Which type was taken is stated once per process on stderr and
//! counted per window, so a round can tell a device that has no cached type from
//! one that does not use the one it has.
//!
//! # The switch
//!
//! `METAL_API_VULKAN_CACHED_READBACK=0` (also `off`, `no`, `false`) drops the
//! preference: every staging buffer is selected exactly as it was before, which
//! is the control arm an A/B states. Anything else — including unset — prefers
//! the cached type.

use std::sync::OnceLock;

use ash::vk;

/// The flags a readback staging buffer's memory has always been chosen with:
/// the host can map it, and a read of that mapping is coherent without an
/// invalidate (this rail never invalidates).
pub(crate) fn plain_flags() -> vk::MemoryPropertyFlags {
    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
}

/// [`plain_flags`] plus the host's own cache: the same mapping contract, with
/// reads that do not go out to the device's memory window one line at a time.
pub(crate) fn cached_flags() -> vk::MemoryPropertyFlags {
    plain_flags() | vk::MemoryPropertyFlags::HOST_CACHED
}

/// The memory type one staging buffer was given, and which arm chose it.
///
/// (`Debug` is spelled by hand: ash's flag type does not carry one, and a
/// choice is worth being able to print in a test's own failure message.)
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Choice {
    /// The index handed to `vkAllocateMemory`.
    pub(crate) index: u32,
    /// The chosen type's own flags, for the line the process prints once.
    pub(crate) flags: vk::MemoryPropertyFlags,
    /// Whether this choice came from the cached preference.
    pub(crate) cached: bool,
}

impl std::fmt::Debug for Choice {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Choice {{ index: {}, flags: {:#x}, cached: {} }}",
            self.index,
            self.flags.as_raw(),
            self.cached
        )
    }
}

/// Choose among one device's memory types for a readback staging buffer.
///
/// Pure over the type list and the buffer's own `memory_type_bits`, so the
/// preference order is testable without a device. The failure states the flags
/// the fallback asked for — the same sentence the selection printed before this
/// module existed, so a device that cannot host a readback buffer fails with the
/// message it always did.
pub(crate) fn choose(
    types: &[vk::MemoryType],
    bits: u32,
    prefer_cached: bool,
) -> Result<Choice, String> {
    if prefer_cached {
        if let Some(choice) = find(types, bits, cached_flags(), true) {
            return Ok(choice);
        }
    }
    find(types, bits, plain_flags(), false).ok_or_else(|| {
        format!(
            "no Vulkan memory type satisfies flags {:#x} for mask {bits:#x}",
            plain_flags().as_raw()
        )
    })
}

fn find(
    types: &[vk::MemoryType],
    bits: u32,
    required: vk::MemoryPropertyFlags,
    cached: bool,
) -> Option<Choice> {
    types
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            bits & (1 << index) != 0 && memory_type.property_flags.contains(required)
        })
        .map(|(index, memory_type)| Choice {
            index: index as u32,
            flags: memory_type.property_flags,
            cached,
        })
}

/// The selection one allocation should make, with the process's switch folded
/// in. The failure is the sentence the caller states as its own refusal, so a
/// device that cannot host a readback buffer fails where it always did.
pub(crate) fn select(types: &[vk::MemoryType], bits: u32) -> Result<Choice, String> {
    let choice = choose(types, bits, preference_enabled())?;
    note(&choice);
    Ok(choice)
}

/// Count one selection and, once per process, state the type it landed on.
///
/// The line is printed only while the phase profile is on: it is a reading
/// about a round, not a product message, and a process that is not being
/// measured should not grow a line on stderr for it.
pub(crate) fn note(choice: &Choice) {
    crate::phase_profile::note_staging_memory(choice.cached);
    static ANNOUNCED: OnceLock<()> = OnceLock::new();
    ANNOUNCED.get_or_init(|| {
        if crate::phase_profile::enabled() {
            eprintln!(
                "STAGING readback memory type_index={} flags={:#x} cached={}",
                choice.index,
                choice.flags.as_raw(),
                u8::from(choice.cached)
            );
        }
    });
}

/// Whether the cached type is preferred, read once from the process
/// environment.
pub(crate) fn preference_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_VULKAN_CACHED_READBACK")
                .ok()
                .as_deref(),
        )
    })
}

/// The switch word, read exactly as a round's launcher spells it.
fn parse_enabled(value: Option<&str>) -> bool {
    !matches!(
        value.map(str::trim),
        Some("0" | "off" | "OFF" | "no" | "NO" | "false" | "FALSE")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory_type(flags: vk::MemoryPropertyFlags) -> vk::MemoryType {
        vk::MemoryType {
            property_flags: flags,
            heap_index: 0,
        }
    }

    /// The device the sp7 round ran on, as the standard discrete layout: a
    /// device-local type, the host-visible window over it, write-combined host
    /// memory and cached host memory. Both arms have to land on the type their
    /// flags state, and the preference must not change what the mask allows.
    #[test]
    fn the_preference_takes_the_cached_type_and_the_fallback_takes_the_first() {
        let types = [
            memory_type(vk::MemoryPropertyFlags::DEVICE_LOCAL),
            memory_type(
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            ),
            memory_type(plain_flags()),
            memory_type(cached_flags()),
        ];
        let all = u32::MAX;
        let cached = choose(&types, all, true).expect("a cached type exists");
        assert_eq!(cached.index, 3);
        assert!(cached.cached);
        let plain = choose(&types, all, false).expect("a host-visible type exists");
        assert_eq!(
            plain.index, 1,
            "the first satisfying type is the device window"
        );
        assert!(!plain.cached);
    }

    /// A mask that excludes the cached type has to fall back rather than fail,
    /// and a device with no host-visible type at all has to fail with the
    /// fallback's own sentence.
    #[test]
    fn the_mask_and_the_failure_keep_their_meanings() {
        let types = [
            memory_type(vk::MemoryPropertyFlags::DEVICE_LOCAL),
            memory_type(plain_flags()),
            memory_type(cached_flags()),
        ];
        // Buffers whose requirements allow only the first two types.
        let choice = choose(&types, 0b011, true).expect("the plain type is allowed");
        assert_eq!(choice.index, 1);
        assert!(!choice.cached);
        let choice = choose(&types, 1 << 2, true).expect("only the cached type is allowed");
        assert_eq!(choice.index, 2);
        assert!(choice.cached);
        let failure = choose(
            &[memory_type(vk::MemoryPropertyFlags::DEVICE_LOCAL)],
            u32::MAX,
            true,
        )
        .expect_err("no host-visible type");
        assert!(
            failure.contains(&format!("flags {:#x}", plain_flags().as_raw())),
            "the sentence states the fallback's flags: {failure}"
        );
    }

    /// Off by default only for the words a launcher spells: the switch has to
    /// read as a round states it, and anything else keeps the preference.
    #[test]
    fn the_switch_reads_the_words_a_round_spells() {
        for value in ["0", "off", "OFF", "no", "NO", "false", "FALSE"] {
            assert!(!parse_enabled(Some(value)), "{value} must turn it off");
        }
        for value in [None, Some(""), Some("1"), Some("on"), Some("true")] {
            assert!(parse_enabled(value), "{value:?} must keep the preference");
        }
    }
}
