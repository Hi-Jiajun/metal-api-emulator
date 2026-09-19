//! Pooling the host-pointer import one sampled declaration makes of its owner's
//! window.
//!
//! # What this is for
//!
//! `docs/SUBMIT-PHASE-PROFILE.md` splits one submission; the sp4 round
//! (`docs/RENDER-IMPORT-POOL.md` carries the reading) put `setup_textures`'s
//! one remaining cost in `texture_import_us`: the no-copy arm of a sampled
//! declaration imports the owner's window as a `TRANSFER_SRC` buffer, and it
//! imports it **again for every pass that samples that window**. On this host
//! that import is not a copy — the bytes stay where the owner put them — but it
//! is five driver calls, and the expensive one is
//! `vkAllocateMemory` over a host range the driver has to make usable as device
//! memory: ~1.5 ms per import in the sp4 reading, 1.1 ms per submission, 98% of
//! `setup_textures`.
//!
//! A guest that samples the same window draw after draw — which is what a
//! compositor does — pays that per pass. This module keeps the buffer, its
//! imported memory and the requirement the import was checked against, and
//! hands them back to the next declaration that names the same window:
//!
//! | object | why the key decides it |
//! |---|---|
//! | `VkBuffer` | the owner's own range (`pointer`, `len`) and the usage the declaration states |
//! | `VkDeviceMemory` | the same range, imported once |
//! | requirement size | what the device asked for that range, which the lease's capacity check reads |
//!
//! Nothing else is pooled, and no byte is copied by this module: a hit skips
//! the import, not the read. The pass that takes the import still records the
//! same `vkCmdCopyBufferToImage` out of the same owner's pages, so every texel
//! it samples is the declaration's own.
//!
//! # Why the key cannot lie
//!
//! The key is `(pointer, len, usage)` — the three fields of the import the
//! driver is handed (`VkBufferCreateInfo::size` is `len`,
//! `VkImportMemoryHostPointerInfoEXT::host_pointer` is `pointer`, and the usage
//! flags decide the buffer's own usage bits). Two declarations meet in the pool
//! exactly when the driver would be asked to import the same range twice, and
//! the comparison is a field-by-field equality of that key — no digest, because
//! the population is a handful of windows and the scan is the cheap half of
//! what it saves.
//!
//! A `pointer` that a later window re-uses *is* the same host range: the import
//! describes the range the owner's declaration names, so a re-mapped window at
//! the same address is read at execution time exactly as a fresh import of that
//! address would be. What the pool never does is keep a *stale* range forever:
//! the caps below bound both the number of windows and the host bytes they can
//! cover, and the oldest entry is destroyed to make room.
//!
//! # The switch, the counters and the failure posture
//!
//! `METAL_API_VULKAN_RENDER_IMPORT_POOL=0` (also `off`, `no`, `false`) turns the
//! whole mechanism off: no take, no hold, one relaxed load per declaration,
//! which is the arm a round's control run states. Anything else — including
//! unset — leaves it on.
//!
//! The profile line prints the window's own counters (`import_hit_n`,
//! `import_miss_n`, `import_disabled_n`, `import_return_n`, `import_drop_n`), so
//! a round can tell a pool that is not being asked from one that is refusing;
//! the cumulative reading is [`RenderImportPoolCounts`].
//!
//! An import is given back only by a pass that reached its readback without an
//! error, and only while the mechanism is still on; a pass that failed before
//! its fence, or a pool switched off mid-flight, destroys the import instead
//! (the fail-closed direction, exactly as the fresh path always did). Entries
//! the cap evicts are destroyed under the lock that removed them, and a switch
//! that goes off drops everything it held, so no device object outlives the
//! pool's own reference to it.

use std::collections::VecDeque;
use std::sync::OnceLock;

use ash::vk;

/// The range and usage that decide an imported owner window.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ImportKey {
    pointer: usize,
    len: u64,
    usage: u32,
}

impl ImportKey {
    /// The key of the import one declaration is about to state.
    pub(crate) fn new(pointer: usize, len: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            pointer,
            len: u64::try_from(len).unwrap_or(u64::MAX),
            usage: usage.as_raw(),
        }
    }

    /// The host bytes this window covers, which is what the byte cap counts.
    pub(crate) fn bytes(&self) -> u64 {
        self.len
    }
}

/// One imported owner window: the buffer over the owner's pages, the memory
/// object that makes them usable as device memory, and the requirement the
/// import was checked against (the lease's capacity check reads it again on a
/// hit, so the pool has to carry it).
pub(crate) struct Imported {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) requirements_size: u64,
}

impl Imported {
    /// Release the two objects, in the order the pass always released them.
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
    key: ImportKey,
    imported: Imported,
}

/// What one declaration's use of the pool came to, as the profile line counts
/// it.
///
/// The first three partition a declaration's `take`: the pool held this
/// window's import and handed it over, it held none and the declaration
/// imported the range itself, or the switch was off and nothing was asked. The
/// last two partition a completed pass's hand-back: the pool kept the import,
/// or it destroyed it (the switch went off mid-pass, or the window does not fit
/// the cap).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ImportOutcome {
    /// The pool held this window's import.
    Hit,
    /// The pool held none; the declaration imported the range itself.
    Miss,
    /// The switch is off: no take, no hold.
    Disabled,
    /// A completed pass handed its import back and the pool kept it.
    Returned,
    /// An import was destroyed instead of held.
    Dropped,
}

/// How many windows one device keeps imported.
///
/// The guest desktop samples its compositor's own surfaces, a handful at a
/// time; the cap is what keeps a pathological stream of distinct windows from
/// holding their host ranges imported forever, and the eviction counter says
/// when it was reached.
pub(crate) const ENTRY_CAP: usize = 64;

/// How many host bytes of owner windows one device keeps imported.
///
/// An imported range is the *owner's* memory, not the provider's, but the
/// driver has been handed it and may have made it device-visible, so the pool
/// bounds what it holds the same way the sampled-backing pool does: a 1920x1080
/// frame is 8.3 MB, and the cap is twenty-four of those. A window larger than
/// the cap is not held at all: it is destroyed when the pass hands it back,
/// which is the same end it had before this module existed.
pub(crate) const BYTE_CAP: u64 = 192 * 1024 * 1024;

/// The resident owner windows one device hands back.
pub(crate) struct RenderImportPool {
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

impl RenderImportPool {
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

    /// Drop every entry, destroying each import under the lock that removed it.
    pub(crate) fn clear(&mut self) {
        while let Some(entry) = self.entries.pop_front() {
            entry.imported.destroy(&self.device);
        }
        self.held_bytes = 0;
        self.flushes += 1;
    }

    /// The import this window may already have, if the pool holds one, and what
    /// the profile line counts.
    ///
    /// An import leaves the pool with the caller, so the pool never holds a
    /// buffer a pass is recording with.
    pub(crate) fn take(&mut self, key: ImportKey) -> (Option<Imported>, ImportOutcome) {
        if !self.enabled {
            self.disabled += 1;
            return (None, ImportOutcome::Disabled);
        }
        // The comparison is the decision: the first entry whose key equals this
        // one is the range the driver would be handed twice, and a key is
        // exactly those fields. The scan is linear because the population is a
        // handful of windows.
        let Some(index) = self.entries.iter().position(|entry| entry.key == key) else {
            self.misses += 1;
            return (None, ImportOutcome::Miss);
        };
        let entry = self.entries.remove(index).expect("index just found");
        self.held_bytes = self.held_bytes.saturating_sub(entry.key.bytes());
        self.hits += 1;
        (Some(entry.imported), ImportOutcome::Hit)
    }

    /// Take an import back from the pass that used it, or destroy it when the
    /// mechanism is off or the cap cannot hold it.
    pub(crate) fn give(&mut self, key: ImportKey, imported: Imported) -> ImportOutcome {
        if !self.enabled {
            imported.destroy(&self.device);
            self.dropped += 1;
            return ImportOutcome::Dropped;
        }
        let bytes = key.bytes();
        if bytes > BYTE_CAP {
            imported.destroy(&self.device);
            self.dropped += 1;
            return ImportOutcome::Dropped;
        }
        while !self.entries.is_empty()
            && (self.entries.len() >= ENTRY_CAP || self.held_bytes + bytes > BYTE_CAP)
        {
            if let Some(entry) = self.entries.pop_front() {
                self.held_bytes = self.held_bytes.saturating_sub(entry.key.bytes());
                entry.imported.destroy(&self.device);
                self.evictions += 1;
            }
        }
        self.held_bytes += bytes;
        self.returns += 1;
        self.entries.push_back(Entry { key, imported });
        ImportOutcome::Returned
    }

    /// The counters one reading reports.
    pub(crate) fn counts(&self) -> RenderImportPoolCounts {
        RenderImportPoolCounts {
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

/// What a pool reading reports, beside the profile line's own window counters.
///
/// `hits` counts declarations served from the pool and `misses` counts the ones
/// that imported their own window; `returns` counts imports a completed pass
/// handed back, `evictions` the entries the cap destroyed, `flushes` the times
/// a switch-off or a contract-surface change dropped everything, and `dropped`
/// the imports destroyed instead of held (the pool was off at hand-back time,
/// or the window did not fit the cap). `entries` and `held_bytes` are what the
/// pool holds now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderImportPoolCounts {
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
        !matches!(
            std::env::var("METAL_API_VULKAN_RENDER_IMPORT_POOL")
                .ok()
                .as_deref(),
            Some("0" | "off" | "OFF" | "no" | "NO" | "false" | "FALSE")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(pointer: usize, len: usize) -> ImportKey {
        ImportKey::new(pointer, len, vk::BufferUsageFlags::TRANSFER_SRC)
    }

    /// The key states exactly the fields the driver is handed: a declaration
    /// meets a pooled import when the range *and* the usage are equal, and
    /// never otherwise.
    #[test]
    fn only_the_same_range_meets_again() {
        assert_eq!(key(0x1000, 4096), key(0x1000, 4096));
        assert_ne!(key(0x1000, 4096), key(0x2000, 4096));
        assert_ne!(key(0x1000, 4096), key(0x1000, 8192));
        assert_ne!(
            key(0x1000, 4096),
            ImportKey::new(0x1000, 4096, vk::BufferUsageFlags::TRANSFER_DST)
        );
        assert_eq!(key(0x1000, 4096).bytes(), 4096);
    }
}
