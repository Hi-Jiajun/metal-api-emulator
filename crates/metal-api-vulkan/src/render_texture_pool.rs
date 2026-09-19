//! Pooling the *backing* of the sampled textures one offscreen pass creates.
//!
//! # What this is for
//!
//! `docs/SUBMIT-PHASE-PROFILE.md` splits one submission; the `setup_*` fields
//! split the render half's assembly. The sp1 round (`docs/TEXTURE-BACKING-POOL.md`
//! carries the reading) put `setup_textures` at the top of that split: for every
//! sampled declaration a pass creates a `VkImage`, allocates and binds its
//! memory, creates an image view, uploads the texels into it and destroys all
//! of it with the pass. A guest that samples the same shapes draw after draw —
//! which is what a compositor does — pays that construction again for every one
//! of them, and on this host the construction is the expensive half.
//!
//! This module keeps the three device objects whose identity is decided by the
//! image's own *shape* and nothing else, and hands them back to the next
//! declaration of the same shape:
//!
//! | object | why the shape decides it |
//! |---|---|
//! | `VkImage` | the image type, the format, the three extents, the tiling (which the byte arm decides) |
//! | `VkDeviceMemory` | the image's own requirements, which those fields decide |
//! | `VkImageView` | the image, its format and the view type |
//!
//! Nothing else is pooled. The sampler, the descriptor set, the layout, the
//! render pass, the framebuffers, the readback buffers, the command pool, the
//! fence and — for the no-copy arm — the imported owner window stay per pass,
//! so a pass's *content* is written exactly where it always was: this is a
//! reuse of backing storage, not of state. The bytes are uploaded into the
//! handed-back image by the same `map`/copy/unmap the fresh image took, so
//! every texel a pass samples is the declaration's own.
//!
//! # Why the key cannot lie
//!
//! The key is **read back from the structure that is about to be handed to the
//! driver** — the `VkImageCreateInfo` this rail is about to state, and the
//! `VkImageViewCreateInfo` it states beside it: the image type, the format, the
//! extent, the view type, and the one bit that is not a field of either
//! structure but decides both halves (`device_copy` chooses `OPTIMAL` +
//! `TRANSFER_DST` + `UNDEFINED`, or `LINEAR` + host-visible + `PREINITIALIZED`).
//! Two declarations meet in the pool exactly when the driver would be handed
//! the same image twice, and the comparison is a field-by-field equality of
//! that key — there is no digest here to collide, because the population is a
//! handful of shapes and the scan is the cheap half of what it saves.
//!
//! # What a hit skips, and what it does not
//!
//! On a hit the pass skips `vkCreateImage`, `vkAllocateMemory`,
//! `vkBindImageMemory` and `vkCreateImageView`. It still uploads the texels
//! exactly as before (the byte arms' `map`/copy/unmap, the no-copy arm's
//! `vkCmdCopyBufferToImage`), still creates its sampler and descriptor set,
//! still records, submits, waits and reads back identically. Every image the
//! pool holds is in `GENERAL` when it is handed back — the layout each arm
//! publishes before the descriptor binds it — so the pass that takes it states
//! that layout as the entry layout of its own barriers
//! ([`crate::render::SampledTextureObjects::entry_layout`]).
//!
//! # The switch, the counters and the failure posture
//!
//! `METAL_API_VULKAN_TEXTURE_BACKING_POOL=0` (also `off`, `no`, `false`) turns
//! the whole mechanism off: no take, no hold, one relaxed load per declaration,
//! which is the arm a round's control run states. Anything else — including
//! unset — leaves it on.
//!
//! The profile line prints the window's own counters (`pool_hit_n`,
//! `pool_miss_n`, `pool_disabled_n`, `pool_evict_n`, `pool_drop_n`), so a round
//! can tell a pool that is not being asked from one that is refusing; the
//! cumulative reading is [`RenderTexturePoolCounts`].
//!
//! A backing is given back only by a pass that reached its readback without an
//! error, and only while the mechanism is still on; a pass that failed before
//! its fence, or a pool switched off mid-flight, destroys the backing instead
//! (the fail-closed direction, exactly as the fresh path always did). Entries
//! the cap evicts are destroyed under the lock that removed them, and a switch
//! that goes off drops everything it held, so no device object outlives the
//! pool's own reference to it.

use std::collections::VecDeque;
use std::sync::OnceLock;

use ash::vk;

/// The shape that decides a sampled image's backing, read back from the
/// structures the driver is handed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct BackingKey {
    image_type: i32,
    format: i32,
    extent: [u32; 3],
    view_type: i32,
    /// Whether the texels reach the image through a device copy
    /// (`OPTIMAL`, `TRANSFER_DST`, `DEVICE_LOCAL`, `UNDEFINED`) or through a
    /// host write (`LINEAR`, host-visible, `PREINITIALIZED`).
    device_copy: bool,
}

impl BackingKey {
    /// The key of the image `create_render_textures` is about to state.
    pub(crate) fn new(
        image_type: vk::ImageType,
        format: vk::Format,
        extent: [u32; 3],
        view_type: vk::ImageViewType,
        device_copy: bool,
    ) -> Self {
        Self {
            image_type: image_type.as_raw(),
            format: format.as_raw(),
            extent,
            view_type: view_type.as_raw(),
            device_copy,
        }
    }
}

/// The three device objects one sampled declaration creates before it writes a
/// byte, and the requirements the host upload maps with.
pub(crate) struct Backing {
    pub(crate) image: vk::Image,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) requirements: vk::MemoryRequirements,
    pub(crate) view: vk::ImageView,
}

impl Backing {
    /// The image's own allocation size, which is what the byte cap counts.
    fn bytes(&self) -> u64 {
        self.requirements.size
    }

    /// Release the three objects, in the order the pass always released them.
    pub(crate) fn destroy(&self, device: &ash::Device) {
        unsafe {
            if self.view != vk::ImageView::null() {
                device.destroy_image_view(self.view, None);
            }
            if self.image != vk::Image::null() {
                device.destroy_image(self.image, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                device.free_memory(self.memory, None);
            }
        }
    }
}

struct Entry {
    key: BackingKey,
    backing: Backing,
}

/// What one declaration's use of the pool came to, as the profile line counts
/// it.
///
/// The first three partition a declaration's `take`: the pool held a backing of
/// this shape and handed it over, it held none and the declaration built its
/// own, or the switch was off and nothing was asked. The last two partition a
/// completed pass's hand-back: the pool kept the backing, or it destroyed it
/// (the switch went off mid-pass, or the shape does not fit the cap).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PoolOutcome {
    /// The pool held a backing of this shape.
    Hit,
    /// The pool held none; the declaration built its own.
    Miss,
    /// The switch is off: no take, no hold.
    Disabled,
    /// A completed pass handed its backing back and the pool kept it.
    Returned,
    /// A backing was destroyed instead of held.
    Dropped,
}

/// How many shapes one device keeps resident.
///
/// The guest desktop samples a handful of shapes and repeats them (the sp1
/// round's own passes name one to four declarations each, and the shapes repeat
/// for the whole round); the cap is what keeps a pathological stream of
/// distinct shapes from growing the provider's own device memory, and the
/// eviction counter says when it was reached.
pub(crate) const ENTRY_CAP: usize = 32;

/// How many bytes of sampled backing one device keeps resident.
///
/// The compositor's own source is 1920x1080x4 = 8.3 MB per shape, so the cap is
/// fifteen of those plus room for the small ones. A single backing larger than
/// the cap is not held at all: it is destroyed when the pass hands it back,
/// which is the same end it had before this module existed.
pub(crate) const BYTE_CAP: u64 = 128 * 1024 * 1024;

/// The resident shapes one device hands back.
pub(crate) struct RenderTexturePool {
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

impl RenderTexturePool {
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

    /// Drop every entry, destroying each backing under the lock that removed
    /// it.
    pub(crate) fn clear(&mut self) {
        while let Some(entry) = self.entries.pop_front() {
            entry.backing.destroy(&self.device);
        }
        self.held_bytes = 0;
        self.flushes += 1;
    }

    /// The backing a declaration of this shape may have, if the pool holds one,
    /// and the outcome the profile line counts.
    ///
    /// A backing leaves the pool with the caller, so the pool never holds an
    /// image a pass is uploading into or recording with.
    pub(crate) fn take(&mut self, key: BackingKey) -> (Option<Backing>, PoolOutcome) {
        if !self.enabled {
            self.disabled += 1;
            return (None, PoolOutcome::Disabled);
        }
        // The comparison is the decision: the first entry whose key equals this
        // one is the shape the driver would be handed twice, and a key is
        // exactly those fields. The scan is linear because the population is a
        // handful of shapes.
        let Some(index) = self.entries.iter().position(|entry| entry.key == key) else {
            self.misses += 1;
            return (None, PoolOutcome::Miss);
        };
        let entry = self.entries.remove(index).expect("index just found");
        self.held_bytes = self.held_bytes.saturating_sub(entry.backing.bytes());
        self.hits += 1;
        (Some(entry.backing), PoolOutcome::Hit)
    }

    /// Take a backing back from the pass that used it, or destroy it when the
    /// mechanism is off or the cap cannot hold it.
    pub(crate) fn give(&mut self, key: BackingKey, backing: Backing) -> PoolOutcome {
        if !self.enabled {
            backing.destroy(&self.device);
            self.dropped += 1;
            return PoolOutcome::Dropped;
        }
        let bytes = backing.bytes();
        if bytes > BYTE_CAP {
            backing.destroy(&self.device);
            self.dropped += 1;
            return PoolOutcome::Dropped;
        }
        while !self.entries.is_empty()
            && (self.entries.len() >= ENTRY_CAP || self.held_bytes + bytes > BYTE_CAP)
        {
            if let Some(entry) = self.entries.pop_front() {
                self.held_bytes = self.held_bytes.saturating_sub(entry.backing.bytes());
                entry.backing.destroy(&self.device);
                self.evictions += 1;
            }
        }
        self.held_bytes += bytes;
        self.returns += 1;
        self.entries.push_back(Entry { key, backing });
        PoolOutcome::Returned
    }

    /// The counters one reading reports.
    pub(crate) fn counts(&self) -> RenderTexturePoolCounts {
        RenderTexturePoolCounts {
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
/// that built their own backing; `returns` counts backings a completed pass
/// handed back, `evictions` the entries the cap destroyed, `flushes` the times
/// a switch-off or a contract-surface change dropped everything, and `dropped`
/// the backings destroyed instead of held (the pool was off at hand-back time,
/// or the shape did not fit the cap). `entries` and `held_bytes` are what the
/// pool holds now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderTexturePoolCounts {
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
            std::env::var("METAL_API_VULKAN_TEXTURE_BACKING_POOL")
                .ok()
                .as_deref(),
            Some("0" | "off" | "OFF" | "no" | "NO" | "false" | "FALSE")
        )
    })
}
