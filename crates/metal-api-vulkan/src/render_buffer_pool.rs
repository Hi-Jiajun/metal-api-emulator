//! Pooling the rail-owned host-visible upload buffers one offscreen pass
//! creates.
//!
//! # What this is for
//!
//! `docs/SUBMIT-PHASE-PROFILE.md` splits one submission. The sp10 round
//! (`docs/RENDER-BUFFER-POOL.md` carries the reading) split the pass's teardown
//! and put 398.4 of `render_teardown`'s 507.7 µs/submit in one group:
//! `teardown_buffers_us` — 5.1 `VkBuffer`s and 5.3 `VkDeviceMemory`s destroyed
//! per submission, ~80 µs each on this host, for the buffers a pass *uploads
//! into*: a `Load` attachment's previous bytes, the vertex streams and index
//! buffer a draw binds, an indirect replay's two buffers, and the stage buffers
//! a translated module reads.
//!
//! Every one of them is created by
//! [`crate::render::OffscreenObjects::create_host_visible_buffer`] — one
//! `vkCreateBuffer`, one `vkAllocateMemory` and one `vkBindBufferMemory`, then
//! the bytes are mapped in — and every one of them is destroyed with the pass.
//! A guest that draws the same shape draw after draw — which is what a
//! compositor does — pays that construction *and* that destruction again for
//! every one of them. This module keeps the pair whose identity is decided by
//! the buffer's own shape and nothing else:
//!
//! | object | why the key decides it |
//! |---|---|
//! | `VkBuffer` | the size and usage flags the driver was handed |
//! | `VkDeviceMemory` | the memory type and size the device's own requirements asked for, which those two fields decide |
//!
//! Nothing else is pooled. The mapping, the write of the declaration's bytes,
//! the recording, the submission and the readback all stay where they were, so
//! every byte a pass binds is the declaration's own: this is a reuse of
//! allocation, not of contents.
//!
//! # Why the key cannot lie
//!
//! The key is **read back from the structure that is about to be handed to the
//! driver** — the `size` and `usage` of the `VkBufferCreateInfo` this rail is
//! about to state. Two declarations meet in the pool exactly when the driver
//! would be handed the same creation twice, and the comparison is a
//! field-by-field equality of that key — there is no digest here to collide,
//! because the population is a handful of shapes and the scan is the cheap half
//! of what it saves.
//!
//! # What a hit skips, and what it does not
//!
//! On a hit the pass skips `vkCreateBuffer`, `vkGetBufferMemoryRequirements`,
//! `vkAllocateMemory` and `vkBindBufferMemory`. It still maps the memory and
//! writes the declaration's bytes into it exactly as the fresh path did, still
//! binds the buffer the same way, still records, submits, waits and reads back
//! identically. A buffer leaves the pool with the pass that took it, so the
//! pool never holds a buffer a command buffer can still be reading; the pass
//! that reached its fence hands it back, and a pass that failed destroys it
//! instead.
//!
//! # The switch, the counters and the failure posture
//!
//! `METAL_API_VULKAN_RENDER_BUFFER_POOL=0` (also `off`, `no`, `false`) turns
//! the whole mechanism off: no take, no hold, one relaxed load per buffer, which
//! is the arm a round's control run states. Anything else — including unset —
//! leaves it on.
//!
//! The profile line prints the window's own counters (`buffer_hit_n`,
//! `buffer_miss_n`, `buffer_disabled_n`, `buffer_return_n`, `buffer_drop_n`), so
//! a round can tell a pool that is not being asked from one that is refusing;
//! the cumulative reading is [`RenderBufferPoolCounts`].
//!
//! A buffer is given back only by a pass that reached its readback without an
//! error, and only while the mechanism is still on; a pass that failed before
//! its fence, or a pool switched off mid-flight, destroys the buffer instead
//! (the fail-closed direction, exactly as the fresh path always did). Entries
//! the cap evicts are destroyed under the lock that removed them, and a switch
//! that goes off drops everything it held, so no device object outlives the
//! pool's own reference to it.

use std::collections::VecDeque;
use std::sync::OnceLock;

use ash::vk;

/// The shape that decides one host-visible upload buffer, read back from the
/// structure the driver is handed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct UploadKey {
    /// The `size` of the `VkBufferCreateInfo` (the declaration's own byte
    /// length, which is what the caller passes as `byte_length`).
    byte_length: u64,
    /// The `usage` bits of the same structure.
    usage: u32,
}

impl UploadKey {
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
    pub(crate) fn byte_length(self) -> u64 {
        self.byte_length
    }

    /// The usage bits the driver was handed, as the key stores them. Read by
    /// the miss-key line (`crate::phase_profile`), which spells the shape a
    /// take asked for the way the creation site would have stated it.
    pub(crate) fn usage(self) -> u32 {
        self.usage
    }
}

/// The two device objects one upload buffer owns.
pub(crate) struct UploadedBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
}

impl UploadedBuffer {
    /// Release the pair in the order the pass always released them.
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
    key: UploadKey,
    uploaded: UploadedBuffer,
}

/// What one buffer's use of the pool came to, as the profile line counts it.
///
/// The first three partition a creation's `take`: the pool held a buffer of
/// this shape and handed it over, it held none and the declaration built its
/// own, or the switch was off and nothing was asked. The last two partition a
/// completed pass's hand-back: the pool kept the buffer, or it destroyed it
/// (the switch went off mid-pass, or the shape does not fit the cap).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum UploadOutcome {
    /// The pool held a buffer of this shape.
    Hit,
    /// The pool held none; the declaration built its own. The census says what
    /// the pool *did* hold, which is what separates "the shape is not resident"
    /// from "the pool was empty".
    Miss(MissCensus),
    /// The switch is off: no take, no hold.
    Disabled,
    /// A completed pass handed its buffer back and the pool kept it.
    Returned,
    /// A buffer was destroyed instead of held.
    Dropped,
}

/// What the pool held when a take missed, as the profile counts it.
///
/// A miss is one fact — the pair was built — and three different findings:
/// the pool held nothing at all (the shape was never handed back, or the cap
/// evicted it), it held shapes and this one differs on the byte length, or it
/// held shapes and this one differs on the usage. The census states all three,
/// and it is a *witness* rather than a partition: an entry can agree with the
/// asked key on the length and differ on the usage, and another entry the other
/// way round, in which case both agreement counts move for the same miss.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MissCensus {
    /// How many entries the pool held when this take asked.
    pub(crate) held: usize,
    /// How many of them have the byte length this take asked for.
    pub(crate) same_length: usize,
    /// How many of them have the usage this take asked for.
    pub(crate) same_usage: usize,
    /// How many entries the pool evicted since the previous take — the "the cap
    /// took my shape" arm, which a miss cannot otherwise be told apart from.
    pub(crate) evicted_since_last_ask: u64,
}

impl MissCensus {
    /// The census one miss states, read from the keys the pool held and the key
    /// the take asked for.
    ///
    /// A free-standing constructor rather than inline arithmetic in
    /// [`RenderBufferPool::take`] because it is the whole answer a round reads
    /// and the one part of the pool a unit test can state without a device: the
    /// keys are values, so `a_miss_states_what_the_pool_held` reads the three
    /// arms (empty, length differs, usage differs) off it directly.
    pub(crate) fn of<'a>(
        held: impl Iterator<Item = &'a UploadKey>,
        asked: UploadKey,
        evicted_since_last_ask: u64,
    ) -> Self {
        let mut census = Self {
            evicted_since_last_ask,
            ..Self::default()
        };
        for key in held {
            census.held += 1;
            if key.byte_length == asked.byte_length {
                census.same_length += 1;
            }
            if key.usage == asked.usage {
                census.same_usage += 1;
            }
        }
        census
    }
}

/// How many shapes one device keeps resident.
///
/// A pass names a handful of upload buffers (a `Load`'s previous bytes, one to
/// four vertex streams, an index buffer, up to two indirect buffers, and the
/// stage buffers a translated module reads), and the shapes repeat for the
/// whole round; the cap is what keeps a pathological stream of distinct shapes
/// from growing the provider's own device memory, and the eviction counter says
/// when it was reached.
pub(crate) const ENTRY_CAP: usize = 64;

/// How many bytes of buffer shape one device keeps resident.
///
/// The census's own upload buffers are small (an index buffer's three `u32`s,
/// the indirect commands, a stage buffer's view), with the one large shape
/// being a `Load` attachment's previous bytes (1920x1080x4 = 8.3 MB); the cap
/// is a few dozen of those plus room for the small ones. A single buffer larger
/// than the cap is not held at all: it is destroyed when the pass hands it
/// back, which is the same end it had before this module existed.
pub(crate) const BYTE_CAP: u64 = 192 * 1024 * 1024;

/// The resident upload buffers one device hands back.
pub(crate) struct RenderBufferPool {
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
    /// How many evictions the pool had performed the last time a take asked.
    /// The difference to `evictions` at the next take is what
    /// [`MissCensus::evicted_since_last_ask`] reports.
    evictions_at_last_ask: u64,
}

impl RenderBufferPool {
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
            evictions_at_last_ask: 0,
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
    pub(crate) fn take(&mut self, key: UploadKey) -> (Option<UploadedBuffer>, UploadOutcome) {
        let evicted_since_last_ask = self.evictions.saturating_sub(self.evictions_at_last_ask);
        self.evictions_at_last_ask = self.evictions;
        if !self.enabled {
            self.disabled += 1;
            return (None, UploadOutcome::Disabled);
        }
        // The comparison is the decision: the first entry whose key equals this
        // one is the shape the driver would be handed twice, and a key is
        // exactly those fields. The scan is linear because the population is a
        // handful of shapes.
        let Some(index) = self.entries.iter().position(|entry| entry.key == key) else {
            self.misses += 1;
            return (
                None,
                UploadOutcome::Miss(MissCensus::of(
                    self.entries.iter().map(|entry| &entry.key),
                    key,
                    evicted_since_last_ask,
                )),
            );
        };
        let entry = self.entries.remove(index).expect("index just found");
        self.held_bytes = self.held_bytes.saturating_sub(entry.key.byte_length());
        self.hits += 1;
        (Some(entry.uploaded), UploadOutcome::Hit)
    }

    /// Take a buffer back from the pass that used it, or destroy it when the
    /// mechanism is off or the cap cannot hold it.
    pub(crate) fn give(&mut self, key: UploadKey, uploaded: UploadedBuffer) -> UploadOutcome {
        if !self.enabled {
            uploaded.destroy(&self.device);
            self.dropped += 1;
            return UploadOutcome::Dropped;
        }
        let bytes = key.byte_length();
        if bytes > BYTE_CAP {
            uploaded.destroy(&self.device);
            self.dropped += 1;
            return UploadOutcome::Dropped;
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
        UploadOutcome::Returned
    }

    /// The counters one reading reports.
    pub(crate) fn counts(&self) -> RenderBufferPoolCounts {
        RenderBufferPoolCounts {
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

/// What the pool has seen over a device's life: the partition of every
/// creation and every hand-back (`crate::render_buffer_pool`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderBufferPoolCounts {
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
            std::env::var("METAL_API_VULKAN_RENDER_BUFFER_POOL")
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
        let one = UploadKey::new(12, vk::BufferUsageFlags::INDEX_BUFFER);
        let same = UploadKey::new(12, vk::BufferUsageFlags::INDEX_BUFFER);
        let longer = UploadKey::new(16, vk::BufferUsageFlags::INDEX_BUFFER);
        let other_usage = UploadKey::new(12, vk::BufferUsageFlags::STORAGE_BUFFER);
        assert_eq!(one, same);
        assert_ne!(one, longer);
        assert_ne!(one, other_usage);
        assert_eq!(one.byte_length(), 12);
    }

    /// A miss's census separates the three arms a round has to tell apart: a
    /// pool that held nothing, one whose entries differ on the byte length, and
    /// one whose entries differ on the usage. The agreement counts are
    /// witnesses, so an entry that agrees on one axis is counted on that axis
    /// even while another entry agrees on the other.
    #[test]
    fn a_miss_states_what_the_pool_held() {
        let asked = UploadKey::new(8_294_400, vk::BufferUsageFlags::TRANSFER_SRC);

        // The cold arm: nothing held at all.
        let empty = MissCensus::of(std::iter::empty(), asked, 0);
        assert_eq!(empty, MissCensus::default());
        assert_eq!(empty.held, 0);

        // One held entry with the asked usage and another length: only the
        // usage axis could have served, which is what "the length kept it out"
        // means.
        let other_length = UploadKey::new(20_480, vk::BufferUsageFlags::TRANSFER_SRC);
        let census = MissCensus::of([other_length].iter(), asked, 0);
        assert_eq!(census.held, 1);
        assert_eq!(census.same_usage, 1);
        assert_eq!(census.same_length, 0);

        // A held entry with the asked length and another usage, beside one with
        // neither: both counts are witnesses over the whole population.
        let other_usage = UploadKey::new(8_294_400, vk::BufferUsageFlags::STORAGE_BUFFER);
        let neither = UploadKey::new(64, vk::BufferUsageFlags::INDIRECT_BUFFER);
        let census = MissCensus::of([other_usage, neither].iter(), asked, 3);
        assert_eq!(census.held, 2);
        assert_eq!(census.same_length, 1);
        assert_eq!(census.same_usage, 0);
        assert_eq!(census.evicted_since_last_ask, 3);
    }
}
