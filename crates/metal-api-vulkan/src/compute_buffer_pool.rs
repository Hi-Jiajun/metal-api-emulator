//! Pooling the compute rail's own host-visible upload buffers.
//!
//! # What this is for
//!
//! `docs/SUBMIT-PHASE-PROFILE.md` splits one submission. The g3a round
//! (`docs/COMPUTE-BUFFER-POOL.md` carries the reading) found the compute half
//! repeating, per submission, the pair of costs the render half had already
//! stopped paying in its own fourth cut: `rb_buffer` 195.4 µs/submission —
//! one `vkCreateBuffer`, one `vkGetBufferMemoryRequirements`, one
//! `vkAllocateMemory` and one `vkBindBufferMemory` for the submission's own
//! staged bytes, 191.7 µs each — and `submit_td_buffers` 155.9 µs/submission
//! destroying that same pair again. Together ≈351 µs/submission, 12.9% of one
//! submission, for an object whose *identity* is two fields the driver is
//! handed.
//!
//! The render half answered the same question with
//! [`crate::render_buffer_pool`]: keep the pair whose shape repeated and give
//! it back after the fence. This module is the compute half's counterpart, and
//! it is deliberately a second pool rather than a sharing of the first — the
//! two rails have their own switches and their own counters, so a round can
//! read one rail's reuse without the other's numbers standing in for it.
//!
//! | object | why the key decides it |
//! |---|---|
//! | `VkBuffer` | the size and usage flags the driver was handed |
//! | `VkDeviceMemory` | the memory type and size the device's own requirements asked for, which those two fields decide |
//!
//! Nothing else is pooled. The map, the write of the declaration's own bytes,
//! the recording, the submission and the readback all stay where they were, so
//! every byte a submission binds is the declaration's own: this is a reuse of
//! allocation, not of contents.
//!
//! # Why the key cannot lie
//!
//! The key is **read back from the structure that is about to be handed to the
//! driver** — the `size` and `usage` of the `VkBufferCreateInfo` this half is
//! about to state. Two creations meet in the pool exactly when the driver would
//! be handed the same creation twice, and the comparison is a field-by-field
//! equality of that key — there is no digest here to collide, because the
//! population is a handful of shapes and the scan is the cheap half of what it
//! saves.
//!
//! # What the pool serves, and what keeps its own path
//!
//! * [`crate::ExecutionResources::create_owned_backing`] — the submission's own
//!   host-visible buffer for a staged, borrowed-and-copied or gathered view, and
//!   the shared backing a repeated allocation's views address. This is the
//!   population the g3a reading counted (`rb_buffer_n` 1.02/submission).
//! * the `INDIRECT_BUFFER` an indirect dispatch replays from
//!   (`ExecutionResources::create_indirect_dispatch`) — the same pair, one
//!   shape (a `DispatchIndirectCommand`'s twelve bytes).
//!
//! Four sites keep their own path, each because the shape does not decide it:
//!
//! * an owner-window import (`import_host_buffer`): its identity is the owner's
//!   pointer, and it already has a pool of its own on the render face
//!   ([`crate::render_import_pool`]) — an imported pair is not a shape and must
//!   not be handed to a creation that named a different window;
//! * a heap placement (`create_heap_buffers`): its memory is the submission's
//!   single slab and its binding offset comes from the heap plan, so the pair's
//!   identity is `(size, usage)` *plus* a placement inside a memory whose type
//!   is the intersection of every placement's requirements — pooling it means
//!   pooling that association, which is a different mechanism than this one;
//! * the storage image's transfer buffer (`create_storage_texture`): it is
//!   built and torn down inside the storage image's own region, where its
//!   lifetime is the image's, not the submission's;
//! * a creation that ran while the switch was off.
//!
//! # What a hit skips, and what it does not
//!
//! On a hit the submission skips `vkCreateBuffer`, `vkGetBufferMemoryRequirements`,
//! `vkAllocateMemory` and `vkBindBufferMemory`. It still maps the memory and
//! writes the declaration's bytes into it exactly as the fresh path did, still
//! binds the buffer the same way, still records, submits, waits and reads back
//! identically. A buffer leaves the pool with the submission that took it, so
//! the pool never holds a buffer a command buffer can still be reading; the
//! submission whose fence was observed hands it back, and a submission that
//! never reached its fence, or observed a device loss, destroys it instead.
//!
//! # The switch, the counters and the failure posture
//!
//! `METAL_API_VULKAN_COMPUTE_BUFFER_POOL=0` (also `off`, `no`, `false`) turns
//! the whole mechanism off: no take, no hold, one relaxed load per buffer, which
//! is the arm a round's control run states. Anything else — including unset —
//! leaves it on.
//!
//! The profile line prints the window's own counters (`compute_buffer_hit_n`,
//! `compute_buffer_miss_n`, `compute_buffer_disabled_n`,
//! `compute_buffer_return_n`, `compute_buffer_drop_n`), so a round can tell a
//! pool that is not being asked from one that is refusing; the cumulative
//! reading is [`ComputeBufferPoolCounts`].
//!
//! A buffer is given back only by a submission that observed its fence, and only
//! while the mechanism is still on; a submission that failed before its fence,
//! one that observed a device loss, or a pool switched off mid-flight destroys
//! the buffer instead (the fail-closed direction, exactly as the fresh path
//! always did). Entries the cap evicts are destroyed under the lock that removed
//! them, and a switch that goes off drops everything it held, so no device
//! object outlives the pool's own reference to it.

use std::collections::VecDeque;
use std::sync::OnceLock;

use ash::vk;

/// The shape that decides one host-visible upload buffer, read back from the
/// structure the driver is handed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ComputeBufferKey {
    /// The `size` of the `VkBufferCreateInfo` (the declaration's own byte
    /// length, which is what the caller passes as the buffer's size).
    byte_length: u64,
    /// The `usage` bits of the same structure.
    usage: u32,
}

impl ComputeBufferKey {
    /// The key of the buffer a creation is about to state.
    pub(crate) fn new(byte_length: u64, usage: vk::BufferUsageFlags) -> Self {
        Self {
            byte_length,
            usage: usage.as_raw(),
        }
    }

    /// The byte length the driver was handed, which is what the pool's byte cap
    /// counts. The allocation the driver makes for it is never smaller than
    /// this, so the cap is a bound on the keys' sizes rather than on the
    /// device's own rounding — a conservative reading of the same number.
    fn byte_length(self) -> u64 {
        self.byte_length
    }
}

/// The two device objects one pooled buffer owns.
pub(crate) struct ComputeUploadedBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
}

impl ComputeUploadedBuffer {
    /// Release the pair in the order the submission always released them.
    pub(crate) fn destroy(&self, device: &ash::Device) {
        unsafe {
            if self.buffer != vk::Buffer::null() {
                device.destroy_buffer(self.buffer, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                device.free_memory(self.memory, None);
            }
        }
    }
}

struct Entry {
    key: ComputeBufferKey,
    uploaded: ComputeUploadedBuffer,
}

/// What one buffer's use of the pool came to, as the profile line counts it.
///
/// The first three partition a creation's `take`: the pool held a buffer of
/// this shape and handed it over, it held none and the declaration built its
/// own, or the switch was off and nothing was asked. The last two partition a
/// completed submission's hand-back: the pool kept the buffer, or it destroyed
/// it (the switch went off mid-submission, or the shape does not fit the cap).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ComputeBufferOutcome {
    /// The pool held a buffer of this shape.
    Hit,
    /// The pool held none; the creation built its own.
    Miss,
    /// The switch is off: no take, no hold.
    Disabled,
    /// A completed submission handed its buffer back and the pool kept it.
    Returned,
    /// A buffer was destroyed instead of held.
    Dropped,
}

/// How many shapes one device keeps resident.
///
/// A submission names a handful of upload buffers (the staged views one
/// dispatch reads, a shared backing, and the indirect replay's own command),
/// and the shapes repeat for the whole round; the cap is what keeps a
/// pathological stream of distinct shapes from growing the provider's own
/// device memory, and the eviction counter says when it was reached.
pub(crate) const ENTRY_CAP: usize = 64;

/// How many bytes of buffer shape one device keeps resident.
///
/// The compute half's own uploads are small: a staged view's bytes, the
/// gathered windows of a guest-runs binding, and the twelve bytes of an
/// indirect command. The cap leaves room for a few dozen of the largest shape a
/// desktop frame states and is two orders of magnitude below what the render
/// half's own pool holds, because nothing here is an 8.3 MB attachment. A
/// single buffer larger than the cap is not held at all: it is destroyed when
/// the submission hands it back, which is the same end it had before this
/// module existed.
pub(crate) const BYTE_CAP: u64 = 64 * 1024 * 1024;

/// The resident upload buffers one device hands back.
pub(crate) struct ComputeBufferPool {
    /// The `VkDevice` the entries belong to. Kept so an evicted or flushed
    /// entry can be destroyed while the device is alive; the context's own
    /// teardown destroys whatever is still here with the device.
    device: ash::Device,
    enabled: bool,
    entries: VecDeque<Entry>,
    held_bytes: u64,
    hits: u64,
    misses: u64,
    disabled: u64,
    returns: u64,
    evictions: u64,
    flushes: u64,
    dropped: u64,
}

impl ComputeBufferPool {
    /// The pool a device starts with.
    pub(crate) fn new(device: ash::Device) -> Self {
        Self {
            device,
            enabled: enabled_from_env(),
            entries: VecDeque::new(),
            held_bytes: 0,
            hits: 0,
            misses: 0,
            disabled: 0,
            returns: 0,
            evictions: 0,
            flushes: 0,
            dropped: 0,
        }
    }

    /// Whether the mechanism is on for this device.
    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    /// Turn the mechanism on or off. Switching it off drops what it held.
    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        if self.enabled && !enabled {
            self.clear();
        }
        self.enabled = enabled;
    }

    /// Drop every entry, destroying each pair under the lock that removed it.
    pub(crate) fn clear(&mut self) {
        while let Some(entry) = self.entries.pop_front() {
            entry.uploaded.destroy(&self.device);
        }
        self.held_bytes = 0;
        self.flushes += 1;
    }

    /// The buffer a creation of this shape may have, if the pool holds one, and
    /// the outcome the profile line counts.
    ///
    /// A buffer leaves the pool with the caller, so the pool never holds one a
    /// command buffer can still be reading or a host can still be writing.
    pub(crate) fn take(
        &mut self,
        key: ComputeBufferKey,
    ) -> (Option<ComputeUploadedBuffer>, ComputeBufferOutcome) {
        if !self.enabled {
            self.disabled += 1;
            return (None, ComputeBufferOutcome::Disabled);
        }
        // The comparison is the decision: the first entry whose key equals this
        // one is the shape the driver would be handed twice, and a key is
        // exactly those fields. The scan is linear because the population is a
        // handful of shapes.
        let Some(index) = self.entries.iter().position(|entry| entry.key == key) else {
            self.misses += 1;
            return (None, ComputeBufferOutcome::Miss);
        };
        let entry = self.entries.remove(index).expect("index just found");
        self.held_bytes = self.held_bytes.saturating_sub(entry.key.byte_length());
        self.hits += 1;
        (Some(entry.uploaded), ComputeBufferOutcome::Hit)
    }

    /// Take a buffer back from the submission that used it, or destroy it when
    /// the mechanism is off or the cap cannot hold it.
    pub(crate) fn give(
        &mut self,
        key: ComputeBufferKey,
        uploaded: ComputeUploadedBuffer,
    ) -> ComputeBufferOutcome {
        if !self.enabled {
            uploaded.destroy(&self.device);
            self.dropped += 1;
            return ComputeBufferOutcome::Dropped;
        }
        let bytes = key.byte_length();
        if bytes > BYTE_CAP {
            uploaded.destroy(&self.device);
            self.dropped += 1;
            return ComputeBufferOutcome::Dropped;
        }
        while !self.entries.is_empty()
            && (self.entries.len() >= ENTRY_CAP || self.held_bytes + bytes > BYTE_CAP)
        {
            if let Some(entry) = self.entries.pop_front() {
                self.held_bytes = self.held_bytes.saturating_sub(entry.key.byte_length());
                entry.uploaded.destroy(&self.device);
                self.evictions += 1;
            }
        }
        self.held_bytes += bytes;
        self.returns += 1;
        self.entries.push_back(Entry { key, uploaded });
        ComputeBufferOutcome::Returned
    }

    /// The counters one reading reports.
    pub(crate) fn counts(&self) -> ComputeBufferPoolCounts {
        ComputeBufferPoolCounts {
            entries: self.entries.len(),
            held_bytes: self.held_bytes,
            hits: self.hits,
            misses: self.misses,
            disabled: self.disabled,
            returns: self.returns,
            evictions: self.evictions,
            flushes: self.flushes,
            dropped: self.dropped,
        }
    }
}

/// What the pool has seen over a device's life: the partition of every creation
/// and every hand-back (`crate::compute_buffer_pool`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputeBufferPoolCounts {
    pub entries: usize,
    pub held_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub disabled: u64,
    pub returns: u64,
    pub evictions: u64,
    pub flushes: u64,
    pub dropped: u64,
}

/// Whether the mechanism is on, read once from the process environment.
pub(crate) fn enabled_from_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_VULKAN_COMPUTE_BUFFER_POOL")
                .ok()
                .as_deref(),
        )
    })
}

/// The switch's own reading: exactly the control words turn the mechanism off,
/// everything else — including unset — leaves it on.
fn parse_enabled(value: Option<&str>) -> bool {
    !matches!(
        value,
        Some("0" | "off" | "OFF" | "no" | "NO" | "false" | "FALSE")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The switch's own arm: only the five spellings a round's control run
    /// states turn the mechanism off, and unset leaves it on.
    #[test]
    fn the_switch_is_off_only_for_the_control_words() {
        for value in ["0", "off", "OFF", "no", "NO", "false", "FALSE"] {
            assert!(!parse_enabled(Some(value)), "{value} must disable the pool");
        }
        for value in [
            None,
            Some("1"),
            Some("on"),
            Some("yes"),
            Some("true"),
            Some(""),
        ] {
            assert!(parse_enabled(value), "{value:?} must leave the pool on");
        }
    }

    /// The key is exactly the structure the driver is handed: the same size and
    /// usage is the same key, and either field differing is a different one.
    #[test]
    fn the_key_is_the_buffers_own_shape() {
        let one = ComputeBufferKey::new(12, vk::BufferUsageFlags::INDIRECT_BUFFER);
        let same = ComputeBufferKey::new(12, vk::BufferUsageFlags::INDIRECT_BUFFER);
        let longer = ComputeBufferKey::new(16, vk::BufferUsageFlags::INDIRECT_BUFFER);
        let other_usage = ComputeBufferKey::new(12, vk::BufferUsageFlags::STORAGE_BUFFER);
        assert_eq!(one, same);
        assert_ne!(one, longer);
        assert_ne!(one, other_usage);
        assert_eq!(one.byte_length(), 12);
    }
}
