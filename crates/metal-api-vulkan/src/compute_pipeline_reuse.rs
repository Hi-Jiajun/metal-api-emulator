//! Reuse of the *shape-decided* device objects one compute submission builds.
//!
//! # What this is for
//!
//! `docs/SUBMIT-PHASE-PROFILE.md` divides one submission. The g3a round (the
//! reading is carried in `docs/COMPUTE-PIPELINE-REUSE.md`) found the compute
//! half building its pipeline objects again for every submission: `rb_pipeline`
//! 118.6 µs/submission in that round, 98.0 / 96.7 / 89.0 µs in the g3c arms,
//! with `rb_pipeline_n` a flat 1.020/submission. `PipelineObjects::create`
//! builds a shader module, the descriptor-set layout its reflection names, the
//! pipeline layout over that set, and one `VkPipeline` per local size the plan
//! dispatches — and destroys all of them with the submission
//! (`submit_td_pipeline`, 4.7–6.5 µs/submission).
//!
//! A guest that dispatches the same kernel with the same threadgroup size
//! submission after submission — which is what a compositor does — pays that
//! construction again for every one of them. The render half answered the same
//! question in its own earlier cut ([`crate::render_setup_reuse`]); this module
//! is the compute half's counterpart, and it is deliberately a second table
//! rather than a sharing of the first: the two rails have their own switches
//! and their own counters, so a round can read one rail's reuse without the
//! other's numbers standing in for it.
//!
//! # Why the key cannot lie
//!
//! The key is **read back from the structures that are about to be handed to
//! the driver**, and it states every field those structures state:
//!
//! | object | the fields the key reads back |
//! |---|---|
//! | `VkShaderModule` | the SPIR-V words of `VkShaderModuleCreateInfo::code` |
//! | `VkDescriptorSetLayout` | the ordered `VkDescriptorSetLayoutBinding`s, by binding number, that `vkCreateDescriptorSetLayout` is handed |
//! | `VkPipelineLayout` | that one set layout, plus the `VkPushConstantRange` the reflection's kernel contract names |
//! | `VkPipeline` (per local size) | the module, the entry name (`main`, the same constant at every site), the `COMPUTE` stage bit, the layout, and the specialization data — which is exactly the local size's three words |
//!
//! Those four rows are the whole `VkComputePipelineCreateInfo` chain and the
//! three objects it stands on; every field the rail never states (a structure's
//! `flags` word, a never-set `pNext` chain, `VkPipelineCache::null()`) is the
//! driver's default for every creation alike, so it carries no shape. A digest
//! of those fields buckets the table; the comparison of them is the decision,
//! and a digest collision costs a comparison rather than a wrong reuse.
//!
//! Two creations meet here exactly when the driver would be handed the same
//! creation twice, and the objects the driver made for one of two
//! definition-identical creations are interchangeable by construction: a
//! `VkDescriptorSetLayout` compatibility is defined on the bindings it was
//! created with, and a `VkPipelineLayout`'s on the set layouts and push
//! constants it was created with. That is why a reused pipeline layout may be
//! paired with a submission's own fresh descriptor sets: the sets are allocated
//! from a layout whose *definition* this key compared.
//!
//! # What a hit skips, and what it does not
//!
//! On a hit the submission skips `vkCreateShaderModule`,
//! `vkCreateDescriptorSetLayout`, `vkCreatePipelineLayout` and the
//! `vkCreateComputePipelines` it would have made for each local size. It still
//! records, submits and reads back exactly as before, still writes the
//! declaration bytes into its own buffers, and still allocates its own
//! descriptor pools, sets and command buffers. The bytes a submission produces
//! are therefore unchanged by construction; the plan's own rail cases
//! (`tests/compute_pipeline_reuse_e2e.rs`) compare them anyway, arm against
//! arm.
//!
//! # The switch, the counters and the failure posture
//!
//! `METAL_API_VULKAN_COMPUTE_PIPELINE_REUSE=0` (also `off`, `no`, `false`)
//! turns the whole mechanism off: no take, no hold, one relaxed load per
//! creation, which is the arm a round's control run states. Anything else —
//! including unset — leaves it on.
//!
//! The counters the profile line prints are in [`crate::phase_profile`]
//! (`compute_pipeline_hit_n`, `compute_pipeline_miss_n`,
//! `compute_pipeline_mismatch_n`, `compute_pipeline_disabled_n`,
//! `compute_pipeline_return_n`, `compute_pipeline_drop_n`), so a round can tell
//! a table that is not being asked from one that is refusing; the cumulative
//! reading is [`ComputePipelineReuseCounts`].
//!
//! A group leaves the table with the submission that took it, and only a
//! submission whose fence was observed hands it back
//! (`ExecutionResources::drop`), so the table never holds objects a command
//! buffer can still be executing. A submission that failed before its fence,
//! one that observed a device loss, and a creation that ran while the switch
//! was off all destroy what they built, exactly as the fresh path always did —
//! the fail-closed direction. Entries the cap evicts are destroyed under the
//! lock that removed them, and a switch that goes off drops everything it held,
//! so no device object outlives the table's own reference to it.
//!
//! # Invalidation and lifetime
//!
//! * the SPIR-V words are the key, so a released registration cannot be served
//!   an entry minted for a different module: a re-registered function whose
//!   translation differs states different words and meets no entry;
//! * a compute registration *leaving* the registry still flushes the table
//!   (`VulkanComputeProvider::release_pipeline`), so entries minted while a
//!   retired registration was live do not outlive it — the same rule the render
//!   half states for its own table;
//! * the switch going off destroys every entry, and every entry an entry's own
//!   device object owns is destroyed by the same lock that removed it;
//! * a device loss ends the context that owns the table, and the table's own
//!   teardown destroys whatever it still holds before the device it was built
//!   on goes away;
//! * unlike the render half there is no unkeyed arm: a compute creation always
//!   states its module's words, its layout bindings, its push-constant range
//!   and the local sizes it is about to build, so every creation can be keyed.

use std::collections::{BTreeMap, VecDeque};
use std::sync::OnceLock;

use ash::vk;

/// One `VkDescriptorSetLayoutBinding`, as the fields the driver reads.
///
/// The immutable-sampler list is empty at the site this rail creates a layout
/// from and the flag word is the default; both are asserted rather than keyed,
/// so a future site that states either is a deliberate change here rather than
/// a silent one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct LayoutBinding {
    pub(crate) binding: u32,
    pub(crate) descriptor_type: i32,
    pub(crate) descriptor_count: u32,
    pub(crate) stage_flags: u32,
}

impl LayoutBinding {
    /// The binding `vkCreateDescriptorSetLayout` is about to be handed.
    pub(crate) fn of(binding: &vk::DescriptorSetLayoutBinding<'_>) -> Self {
        debug_assert!(
            binding.p_immutable_samplers.is_null(),
            "a layout with immutable samplers needs the samplers in its definition"
        );
        Self {
            binding: binding.binding,
            descriptor_type: binding.descriptor_type.as_raw(),
            descriptor_count: binding.descriptor_count,
            stage_flags: binding.stage_flags.as_raw(),
        }
    }
}

/// One `VkPushConstantRange`, as the two fields the driver reads.
///
/// The stage bit is `COMPUTE` at every site this rail states a range from, so
/// it is asserted rather than keyed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct PushConstantRange {
    pub(crate) offset: u32,
    pub(crate) size: u32,
}

impl PushConstantRange {
    pub(crate) fn of(range: &vk::PushConstantRange) -> Self {
        debug_assert!(
            range.stage_flags == vk::ShaderStageFlags::COMPUTE,
            "a range on another stage is a different definition and must be keyed"
        );
        Self {
            offset: range.offset,
            size: range.size,
        }
    }
}

/// The shape that decides one compute pipeline group, read back from the
/// structures the driver is about to be handed.
///
/// `digest` buckets the table; the comparison of the four fields beside it is
/// the decision. `local_sizes` is the set of pipelines one group carries, in
/// the ascending order the group's own map holds them, so two plans that build
/// the same pipelines in a different region order share one entry.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct ComputePipelineKey {
    digest: u64,
    words: Vec<u32>,
    bindings: Vec<LayoutBinding>,
    push_constant: PushConstantRange,
    local_sizes: Vec<[u32; 3]>,
}

impl ComputePipelineKey {
    /// The key of the group a creation is about to state.
    pub(crate) fn new(
        words: &[u32],
        bindings: &[LayoutBinding],
        push_constant: PushConstantRange,
        mut local_sizes: Vec<[u32; 3]>,
    ) -> Self {
        local_sizes.sort_unstable();
        local_sizes.dedup();
        let mut digest = Digest::new();
        digest.u64(words.len() as u64);
        for word in words {
            digest.u64(u64::from(*word));
        }
        digest.u64(bindings.len() as u64);
        for binding in bindings {
            digest.u64(u64::from(binding.binding));
            digest.u64(binding.descriptor_type as u64);
            digest.u64(u64::from(binding.descriptor_count));
            digest.u64(u64::from(binding.stage_flags));
        }
        digest.u64(u64::from(push_constant.offset));
        digest.u64(u64::from(push_constant.size));
        digest.u64(local_sizes.len() as u64);
        for size in &local_sizes {
            for axis in size {
                digest.u64(u64::from(*axis));
            }
        }
        Self {
            digest: digest.finish(),
            words: words.to_vec(),
            bindings: bindings.to_vec(),
            push_constant,
            local_sizes,
        }
    }

    /// The fields the driver would be handed twice, compared field by field.
    fn same_shape(&self, other: &Self) -> bool {
        self.words == other.words
            && self.bindings == other.bindings
            && self.push_constant == other.push_constant
            && self.local_sizes == other.local_sizes
    }
}

/// The device objects one compute pipeline group owns.
///
/// These are the four objects `PipelineObjects` builds and releases as one
/// unit: the module, the set layout the reflection names, the layout over it,
/// and one pipeline per local size.
///
/// # Who destroys them
///
/// The value itself does, through `Drop`, whenever it still holds a device
/// handle. The table holds its entries **disarmed** (`device: None`), so an
/// entry the table keeps is destroyed by the table's own device handle when the
/// cap evicts it, the switch drops it, or the context tears the table down; a
/// value a submission holds is armed, so a submission that fails before its
/// fence — or one that observed a device loss — releases exactly what it built,
/// which is the fail-closed direction the fresh path always had. Taking an
/// entry out arms what left, and handing one back disarms what came in.
pub(crate) struct ReusablePipelineGroup {
    pub(crate) shader: vk::ShaderModule,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) pipelines: BTreeMap<[u32; 3], vk::Pipeline>,
    /// The device the handles were made on while this value owns them.
    device: Option<ash::Device>,
}

impl ReusablePipelineGroup {
    /// The empty group a submission starts from: every handle is the driver's
    /// null, so a creation that fails before it mints one destroys nothing.
    pub(crate) fn empty(device: ash::Device) -> Self {
        Self {
            shader: vk::ShaderModule::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipelines: BTreeMap::new(),
            device: Some(device),
        }
    }

    /// Take the device handle out, so the caller that now holds the handles
    /// owns releasing them and this table does not.
    pub(crate) fn disarm(&mut self) -> Option<ash::Device> {
        self.device.take()
    }

    /// Hand the device handle back to a value that left the table.
    pub(crate) fn arm(&mut self, device: ash::Device) {
        self.device = Some(device);
    }

    /// Release the handles on the device the caller names, whatever this value
    /// holds. A disarmed value is released by the table that armed itself.
    pub(crate) fn destroy_with(&mut self, device: &ash::Device) {
        self.device = None;
        unsafe {
            for pipeline in self.pipelines.values().copied() {
                if pipeline != vk::Pipeline::null() {
                    device.destroy_pipeline(pipeline, None);
                }
            }
            self.pipelines.clear();
            if self.pipeline_layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.pipeline_layout, None);
                self.pipeline_layout = vk::PipelineLayout::null();
            }
            if self.set_layout != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(self.set_layout, None);
                self.set_layout = vk::DescriptorSetLayout::null();
            }
            if self.shader != vk::ShaderModule::null() {
                device.destroy_shader_module(self.shader, None);
                self.shader = vk::ShaderModule::null();
            }
        }
    }
}

impl Drop for ReusablePipelineGroup {
    fn drop(&mut self) {
        if let Some(device) = self.device.take() {
            // `destroy_with` clears `device`, so this borrow is over before the
            // value it names is released.
            self.destroy_with(&device);
        }
    }
}

struct Entry {
    key: ComputePipelineKey,
    group: ReusablePipelineGroup,
}

/// What one creation's use of the table came to, as the profile line counts it.
///
/// The first four partition a creation's take: the table held objects of this
/// shape and handed them over, it held none and the creation built its own, an
/// entry shared the digest but not the shape (fail-closed), or the switch was
/// off and nothing was asked. The last two partition a completed submission's
/// hand-back: the table kept the objects, or it destroyed them (the switch went
/// off mid-submission, or the shape does not fit the cap).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ComputePipelineOutcome {
    /// The table held objects of this shape.
    Hit,
    /// The table held none; the creation built its own.
    Miss,
    /// An entry shared the digest but not the shape.
    Mismatch,
    /// The switch is off: no take, no hold.
    Disabled,
    /// A completed submission handed its objects back and the table kept them.
    Returned,
    /// Objects were destroyed instead of held.
    Dropped,
}

/// How many shapes one device keeps resident.
///
/// A guest desktop dispatches a handful of kernels (a copy, a composite, a
/// clear) and repeats them for the whole round, so the population is small; the
/// cap is what keeps a pathological stream of distinct kernels from growing the
/// provider's own memory with modules the device never runs again, and the
/// eviction counter says when it was reached.
pub(crate) const ENTRY_CAP: usize = 64;

/// The resident pipeline groups one device hands back.
pub(crate) struct ComputePipelineReuse {
    /// The `VkDevice` the entries belong to. Kept so an evicted or flushed
    /// entry can be destroyed while the device is alive; the context's own
    /// teardown destroys whatever is still here with the device.
    device: ash::Device,
    enabled: bool,
    entries: VecDeque<Entry>,
    hits: u64,
    misses: u64,
    mismatches: u64,
    disabled: u64,
    returns: u64,
    evictions: u64,
    flushes: u64,
    dropped: u64,
}

impl ComputePipelineReuse {
    /// The table a device starts with.
    pub(crate) fn new(device: ash::Device) -> Self {
        Self {
            device,
            enabled: enabled_from_env(),
            entries: VecDeque::new(),
            hits: 0,
            misses: 0,
            mismatches: 0,
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

    /// Drop every entry, destroying each group under the lock that removed it.
    pub(crate) fn clear(&mut self) {
        while let Some(mut entry) = self.entries.pop_front() {
            entry.group.destroy_with(&self.device);
        }
        self.flushes += 1;
    }

    /// The objects a creation of this shape may have, if the table holds them,
    /// and the outcome the profile line counts.
    ///
    /// The objects leave the table with the caller, so the table never holds
    /// objects a command buffer can still be executing. A caller that fails
    /// before handing them back destroys them (its own teardown owns them by
    /// then), which is the fail-closed arm: the next creation of that shape
    /// builds the objects again instead of finding them half-used.
    pub(crate) fn take(
        &mut self,
        key: &ComputePipelineKey,
    ) -> (Option<ReusablePipelineGroup>, ComputePipelineOutcome) {
        if !self.enabled {
            self.disabled += 1;
            return (None, ComputePipelineOutcome::Disabled);
        }
        let mut mismatched = false;
        let mut found = None;
        for index in 0..self.entries.len() {
            if self.entries[index].key.digest != key.digest {
                continue;
            }
            if self.entries[index].key.same_shape(key) {
                found = Some(index);
                break;
            }
            mismatched = true;
        }
        let Some(index) = found else {
            if mismatched {
                self.mismatches += 1;
                return (None, ComputePipelineOutcome::Mismatch);
            }
            self.misses += 1;
            return (None, ComputePipelineOutcome::Miss);
        };
        self.hits += 1;
        let mut entry = self.entries.remove(index).expect("index just found");
        // What leaves the table is the submission's to release until it hands
        // it back: a submission that fails before its fence destroys it.
        entry.group.arm(self.device.clone());
        (Some(entry.group), ComputePipelineOutcome::Hit)
    }

    /// Take a group back from the submission that used it, or destroy it when
    /// the mechanism is off or the cap cannot hold it.
    ///
    /// A group another entry already holds for the same shape is destroyed
    /// rather than kept twice: two submissions of one shape racing on a cold
    /// table both build, and the loser's objects are surplus.
    pub(crate) fn give(
        &mut self,
        key: ComputePipelineKey,
        mut group: ReusablePipelineGroup,
    ) -> ComputePipelineOutcome {
        if !self.enabled {
            group.destroy_with(&self.device);
            self.dropped += 1;
            return ComputePipelineOutcome::Dropped;
        }
        if self.entries.iter().any(|entry| entry.key.same_shape(&key)) {
            group.destroy_with(&self.device);
            self.dropped += 1;
            return ComputePipelineOutcome::Dropped;
        }
        while self.entries.len() >= ENTRY_CAP {
            if let Some(mut evicted) = self.entries.pop_front() {
                evicted.group.destroy_with(&self.device);
                self.evictions += 1;
            }
        }
        // The table owns its entries: their own device handle goes, and the
        // table's releases them.
        debug_assert!(group.disarm().is_some(), "a handed-back group is armed");
        self.returns += 1;
        self.entries.push_back(Entry { key, group });
        ComputePipelineOutcome::Returned
    }

    /// The counters one reading reports.
    pub(crate) fn counts(&self) -> ComputePipelineReuseCounts {
        ComputePipelineReuseCounts {
            entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            mismatches: self.mismatches,
            disabled: self.disabled,
            returns: self.returns,
            evictions: self.evictions,
            flushes: self.flushes,
            dropped: self.dropped,
        }
    }
}

/// What the table has seen over a device's life: the partition of every
/// creation and every hand-back (`crate::compute_pipeline_reuse`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputePipelineReuseCounts {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub mismatches: u64,
    pub disabled: u64,
    pub returns: u64,
    pub evictions: u64,
    pub flushes: u64,
    pub dropped: u64,
}

/// FNV-1a, the digest this crate's other keys use. It picks the bucket; the
/// comparison above decides.
struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn u64(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn finish(self) -> u64 {
        self.0
    }
}

/// Whether the mechanism is on, read once from the process environment.
pub(crate) fn enabled_from_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_VULKAN_COMPUTE_PIPELINE_REUSE")
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

    fn binding(binding: u32, descriptor_type: i32) -> LayoutBinding {
        LayoutBinding {
            binding,
            descriptor_type,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE.as_raw(),
        }
    }

    fn key(
        words: &[u32],
        bindings: &[LayoutBinding],
        push_constant: PushConstantRange,
        local_sizes: &[[u32; 3]],
    ) -> ComputePipelineKey {
        ComputePipelineKey::new(words, bindings, push_constant, local_sizes.to_vec())
    }

    /// The switch's own arm: only the seven spellings a round's control run
    /// states turn the mechanism off, and unset leaves it on.
    #[test]
    fn the_switch_is_off_only_for_the_control_words() {
        for value in ["0", "off", "OFF", "no", "NO", "false", "FALSE"] {
            assert!(
                !parse_enabled(Some(value)),
                "{value} must disable the table"
            );
        }
        for value in [
            None,
            Some("1"),
            Some("on"),
            Some("yes"),
            Some("true"),
            Some(""),
        ] {
            assert!(parse_enabled(value), "{value:?} must leave the table on");
        }
    }

    /// The key is exactly the structures the driver is handed: any one field
    /// differing is a different shape, and a different region order is not.
    #[test]
    fn the_key_is_the_creations_own_shape() {
        let words = [0x0723_0203, 0x0001_0000, 0x0002_0000];
        let other_words = [0x0723_0203, 0x0001_0000, 0x0003_0000];
        let bindings = [binding(0, 7), binding(1, 3)];
        let other_bindings = [binding(0, 7), binding(1, 0)];
        let range = PushConstantRange {
            offset: 0,
            size: 12,
        };
        let other_range = PushConstantRange {
            offset: 4,
            size: 12,
        };

        let one = key(&words, &bindings, range, &[[8, 1, 1]]);
        let same = key(&words, &bindings, range, &[[8, 1, 1]]);
        let reordered = key(&words, &bindings, range, &[[4, 1, 1], [8, 1, 1]]);
        let reordered_same = key(&words, &bindings, range, &[[8, 1, 1], [4, 1, 1]]);
        assert_eq!(one, same);
        assert_eq!(reordered, reordered_same);

        assert_ne!(one, key(&other_words, &bindings, range, &[[8, 1, 1]]));
        assert_ne!(one, key(&words, &other_bindings, range, &[[8, 1, 1]]));
        assert_ne!(one, key(&words, &bindings, other_range, &[[8, 1, 1]]));
        assert_ne!(one, key(&words, &bindings, range, &[[4, 1, 1]]));
        assert_ne!(one, key(&words, &bindings, range, &[[8, 1, 1], [4, 1, 1]]));

        // A digest that matches is still not a decision: two keys whose words
        // differ are refused by the comparison however they bucket.
        let mut forged = key(&words, &bindings, range, &[[8, 1, 1]]);
        forged.words[2] = 0x0003_0000;
        assert_eq!(forged.digest, one.digest, "the digest is only a bucket");
        assert!(!forged.same_shape(&one), "the comparison is the decision");
    }
}
