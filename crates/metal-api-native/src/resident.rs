//! The provider-resident render target registry (`research/docs/23` §76, R7).
//!
//! The Vulkan rail's R7 increment keeps a render pass's frame in an image the
//! provider owns under the attachment's own `(allocation, view)` identity, so a
//! later pass loads those bytes with `LoadOp::Resident` and the frame never has
//! to leave the guest ([`metal_api_core::provider::LoadOp::Resident`] /
//! [`metal_api_core::provider::StoreOp::Resident`]). This module is the Metal
//! rail's own copy of that policy, and it is deliberately the *policy* alone:
//! the registry is generic over the image type the caller stores, so every
//! decision it makes — which identities are alive, which one the budget evicts
//! first, which refusal a load of a retired identity names, when a pass makes
//! its identity loadable again — is testable on a host that cannot load Metal.
//! The one device-shaped half, creating the `MTLTexture`, is the closure the
//! provider passes in.
//!
//! The registry reuses the R4a present target registry's shape
//! (`native.rs::State::present_targets`, `research/docs/24` §6 Step 7): one
//! entry per identity, held across submissions, retired when the lease its
//! allocation was imported under is released. What R7 adds is the budget and
//! the tombstones:
//!
//! - the registry keeps at most [`RESIDENT_TARGET_BUDGET`] identities, evicting
//!   the least recently used one when a pass creates a new identity beyond it;
//! - every identity it retires leaves a *tombstone* naming the rule that
//!   retired it, so a later `LoadOp::Resident` is refused with
//!   `resident_target_evicted` / `resident_target_released` /
//!   `resident_target_stale` instead of being served from a fresh image — the
//!   fail-closed rule the whole arm exists for;
//! - a `StoreOp::Resident` pass re-creates a retired identity and clears its
//!   tombstone, because the pass is what defines the new image's bytes.

use metal_api_core::provider::{
    AllocationId, AttachmentFormat, FieldValue, ProviderError, ProviderErrorClass, ProviderPhase,
    RenderAttachment, ViewId,
};
use std::collections::BTreeMap;

/// How many provider-resident render target identities the provider keeps
/// across submissions at once (`research/docs/23` §76, R7).
///
/// The R4a present registry's rule, generalised to the render rail: a resident
/// target's identity is the `(allocation, view)` pair of the attachment that
/// declares it, so a guest rendering many surfaces would otherwise keep one
/// full-size image per identity for the process's lifetime with no release
/// surface at all. The value is the Vulkan rail's own budget
/// (`crates/metal-api-vulkan/src/compute_provider.rs`), so the two rails refuse
/// and retire at the same rate.
pub(crate) const RESIDENT_TARGET_BUDGET: usize = 8;

/// One resident target's identity: the pair the attachment itself names, so the
/// registry and the contract cannot disagree about which target a pass means.
pub(crate) type ResidentIdentity = (AllocationId, ViewId);

/// Why a resident target identity is no longer in the registry
/// (`research/docs/23` §76, R7).
///
/// The tombstone is what makes the refusal nameable: a `LoadOp::Resident` for
/// an identity that is gone says *which* rule retired it, so the guest can
/// render the frame again instead of guessing whether it forgot to store one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResidentTargetRetirement {
    /// The registry's budget evicted the least recently used identity.
    Budget,
    /// The lease the identity's allocation was imported under was released.
    LeaseReleased,
    /// The device was lost: every image of the dead device is gone.
    EpochAdvance,
}

impl ResidentTargetRetirement {
    /// The refusal slug a load of this retired identity states.
    pub(crate) const fn slug(self) -> &'static str {
        match self {
            Self::Budget => "resident_target_evicted",
            Self::LeaseReleased => "resident_target_released",
            Self::EpochAdvance => "resident_target_stale",
        }
    }

    /// The `retired_by` field's spelling of the same rule.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Budget => "budget",
            Self::LeaseReleased => "lease_released",
            Self::EpochAdvance => "epoch_advance",
        }
    }
}

/// One resident target: the provider-owned image, the shape the trace declared
/// for it, and the bookkeeping the budget needs.
struct ResidentTargetEntry<Image> {
    image: Image,
    /// The format the identity was created with. A later pass naming the same
    /// identity with a different format is refused by name rather than rendered
    /// into an image of the wrong format.
    format: AttachmentFormat,
    width: u64,
    height: u64,
    /// Whether a completed pass has defined the image's bytes. An entry is
    /// created before its first pass runs, so this is what keeps a refused or
    /// failed pass from leaving an image a later `LoadOp::Resident` could read
    /// as "the target's contents".
    defined: bool,
    /// The least-recently-used stamp that orders the budget's eviction.
    last_used: u64,
}

/// The provider-resident render targets one provider keeps
/// (`research/docs/23` §76, R7).
///
/// `Image` is the caller's own provider-owned image type: the macOS provider
/// stores `MTLTexture`s, and the host-side tests store a stand-in so the same
/// decisions are exercised without a device.
pub(crate) struct ResidentRegistry<Image> {
    entries: BTreeMap<ResidentIdentity, ResidentTargetEntry<Image>>,
    /// The identities the registry has retired in the *current* device epoch,
    /// with the rule that retired them. A `LoadOp::Resident` for one of them is
    /// refused with that rule's name; a `StoreOp::Resident` creates the identity
    /// again and clears its tombstone.
    tombstones: BTreeMap<ResidentIdentity, ResidentTargetRetirement>,
    /// Monotonic use stamp behind the registry's least-recently-used order:
    /// every lookup and every insert stamps its identity with the next value.
    stamp: u64,
    /// Cumulative resident targets retired before the device was lost: the
    /// budget's evictions plus the retirements a lease release drives.
    evictions: u64,
}

impl<Image> Default for ResidentRegistry<Image> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Image> ResidentRegistry<Image> {
    /// An empty registry, as the provider creates it with its device.
    pub(crate) const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            tombstones: BTreeMap::new(),
            stamp: 0,
            evictions: 0,
        }
    }

    /// The number of identities the registry currently holds.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Cumulative identities retired before the device was lost: the budget's
    /// evictions plus the retirements a lease release drives. The device-loss
    /// teardown is not counted here; it is observable through the tombstones it
    /// leaves behind (every load after it names `resident_target_stale`).
    pub(crate) fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Whether the registry currently holds one attachment's identity.
    ///
    /// A pass that renders into an identity the provider holds has to declare
    /// it — a resident load or a resident store — so this is the question
    /// behind `resident_target_undeclared`: the provider's bytes are not
    /// silently overwritten by a pass that never named them.
    pub(crate) fn is_live(&self, allocation_id: AllocationId, view_id: ViewId) -> bool {
        self.entries.contains_key(&(allocation_id, view_id))
    }

    /// State whether the bytes of the identities a pass declared are defined
    /// now that the pass has reached a terminal command buffer
    /// (`research/docs/23` §76, R7).
    ///
    /// A completed pass defines them: it cleared the image, kept contents a
    /// previous pass defined, or stored the raster into it. A pass that was
    /// refused or failed defines nothing, and its identity stays unloadable
    /// until a later pass renders it again.
    pub(crate) fn note(&mut self, identities: &[ResidentIdentity], defined: bool) {
        for identity in identities {
            if let Some(entry) = self.entries.get_mut(identity) {
                entry.defined = defined;
            }
        }
    }
}

impl<Image: Clone> ResidentRegistry<Image> {
    /// Resolve one attachment's resident target, creating the identity on a
    /// resident store (`research/docs/23` §76, R7).
    ///
    /// `loading` is the trace's own decision: a `LoadOp::Resident` attachment
    /// keeps the image's contents, so the identity has to exist *and* hold
    /// bytes a completed pass defined. Every way it can fail is refused by name
    /// rather than served from a fresh image or from bytes no pass defined:
    ///
    /// - a resident load of an identity no pass ever stored is
    ///   `resident_target_unavailable`;
    /// - one the budget evicted, a lease release retired, or a device loss
    ///   cleared states that rule (`resident_target_evicted`,
    ///   `resident_target_released`, `resident_target_stale`);
    /// - one whose creating pass never completed is `resident_target_undefined`;
    /// - one whose shape changed is `resident_target_shape_changed`.
    ///
    /// `create` builds the provider's own image for a new identity; it runs
    /// only after the registry has decided that the identity has to be created
    /// (the device-shaped half of this call).
    pub(crate) fn resolve(
        &mut self,
        attachment: &RenderAttachment,
        loading: bool,
        create: impl FnOnce(AttachmentFormat, u64, u64) -> Result<Image, ProviderError>,
    ) -> Result<Image, ProviderError> {
        let identity = (attachment.allocation_id, attachment.view_id);
        if let Some(entry) = self.entries.get_mut(&identity) {
            if entry.format != attachment.format
                || entry.width != attachment.width
                || entry.height != attachment.height
            {
                return Err(resident_target_refusal(
                    "resident_target_shape_changed",
                    attachment,
                )
                .with_field(
                    "format",
                    FieldValue::Unsigned(u64::from(attachment.format.code())),
                )
                .with_field("width", FieldValue::Unsigned(attachment.width))
                .with_field("height", FieldValue::Unsigned(attachment.height))
                .with_field(
                    "expected_format",
                    FieldValue::Unsigned(u64::from(entry.format.code())),
                )
                .with_field("expected_width", FieldValue::Unsigned(entry.width))
                .with_field("expected_height", FieldValue::Unsigned(entry.height))
                .with_detail(
                    "the resident target's identity is reused for one image, so a pass that \
                     declares a different shape for it is refused instead of rendered into an \
                     image of the wrong format or extent",
                ));
            }
            if loading && !entry.defined {
                return Err(
                    resident_target_refusal("resident_target_undefined", attachment).with_detail(
                        "the resident target's image exists but no completed pass has defined \
                         its bytes: the pass that created it was refused or failed",
                    ),
                );
            }
            entry.last_used = self.stamp;
            self.stamp += 1;
            return Ok(entry.image.clone());
        }
        if loading {
            // The identity is gone. The tombstone names the rule that retired
            // it; an identity with no tombstone was never stored in this epoch
            // at all.
            let retirement = self.tombstones.get(&identity).copied();
            let mut error = resident_target_refusal(
                retirement.map_or("resident_target_unavailable", |rule| rule.slug()),
                attachment,
            );
            if let Some(rule) = retirement {
                error = error.with_field("retired_by", FieldValue::Text(rule.name().to_owned()));
            }
            return Err(error.with_detail(
                "a `LoadOp::Resident` attachment keeps the bytes the provider holds for its \
                 identity; this identity holds none, so the pass has to render the frame again \
                 instead of reading an image that is gone",
            ));
        }
        let image = create(attachment.format, attachment.width, attachment.height)?;
        let stamp = self.stamp;
        self.stamp += 1;
        self.entries.insert(
            identity,
            ResidentTargetEntry {
                image: image.clone(),
                format: attachment.format,
                width: attachment.width,
                height: attachment.height,
                // The pass that creates the identity has not run yet, so its
                // bytes are not defined until it completes.
                defined: false,
                last_used: stamp,
            },
        );
        // The budget is a provider-internal policy, exactly as it is for the
        // present registry: the identities beyond it are retired here, and each
        // one leaves a tombstone a later resident load is refused by name with.
        let mut retired = Vec::new();
        while self.entries.len() > RESIDENT_TARGET_BUDGET {
            // The identity this call just created is never its own round's
            // victim: the pass that named it is the reason it exists.
            let Some(victim) = self
                .entries
                .iter()
                .filter(|(candidate, _)| **candidate != identity)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(candidate, _)| *candidate)
            else {
                break;
            };
            if let Some(entry) = self.entries.remove(&victim) {
                retired.push((victim, entry));
            }
        }
        self.retire(&retired, ResidentTargetRetirement::Budget);
        // A stored identity is alive again, so its tombstone goes with it.
        self.tombstones.remove(&identity);
        Ok(image)
    }

    /// Retire every identity of one allocation and record the rule
    /// (`research/docs/23` §76, R7): the release of the lease an allocation was
    /// imported under is what retires its images.
    ///
    /// Returns the number of identities retired, which is what the provider's
    /// eviction counter is charged.
    pub(crate) fn retire_allocation(&mut self, allocation_id: AllocationId) -> usize {
        let identities: Vec<ResidentIdentity> = self
            .entries
            .keys()
            .filter(|(allocation, _)| *allocation == allocation_id)
            .copied()
            .collect();
        let retired: Vec<_> = identities
            .into_iter()
            .filter_map(|identity| {
                self.entries
                    .remove(&identity)
                    .map(|entry| (identity, entry))
            })
            .collect();
        self.retire(&retired, ResidentTargetRetirement::LeaseReleased);
        retired.len()
    }

    /// Retire the whole registry because the device that owned every image was
    /// lost (`research/docs/23` §76, R7).
    ///
    /// Every identity leaves an [`ResidentTargetRetirement::EpochAdvance`]
    /// tombstone, so a later resident load names `resident_target_stale`
    /// instead of reading an image of the dead device. The teardown is not
    /// charged to the eviction counter — it is observable through the tombstones
    /// themselves, exactly as the Vulkan rail's epoch advance is observable
    /// through its own.
    pub(crate) fn clear(&mut self) {
        let identities: Vec<ResidentIdentity> = self.entries.keys().copied().collect();
        let retired: Vec<_> = identities
            .into_iter()
            .filter_map(|identity| {
                self.entries
                    .remove(&identity)
                    .map(|entry| (identity, entry))
            })
            .collect();
        self.retire(&retired, ResidentTargetRetirement::EpochAdvance);
    }

    /// Record the rule that retired these identities, and charge the ones the
    /// normal-path surfaces retired.
    fn retire(
        &mut self,
        retired: &[(ResidentIdentity, ResidentTargetEntry<Image>)],
        rule: ResidentTargetRetirement,
    ) {
        if retired.is_empty() {
            return;
        }
        for (identity, _) in retired {
            self.tombstones.insert(*identity, rule);
        }
        if rule != ResidentTargetRetirement::EpochAdvance {
            self.evictions += retired.len() as u64;
        }
    }
}

/// The capability refusal one attachment's resident target states.
///
/// The identity fields are the same on every arm, so they are set here once;
/// the caller adds the arm's own fields and detail.
fn resident_target_refusal(slug: &'static str, attachment: &RenderAttachment) -> ProviderError {
    crate::refusal(ProviderPhase::Resolve, ProviderErrorClass::Capability, slug)
        .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
        .with_field(
            "allocation",
            FieldValue::Unsigned(attachment.allocation_id.get()),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::provider::{LoadOp, StoreOp};

    /// The attachment every case declares: one 2x2 `Rgba8Unorm`, its load and
    /// store replaced per case.
    fn attachment(view: u64, allocation: u64) -> RenderAttachment {
        RenderAttachment {
            view_id: ViewId::new(view),
            allocation_id: AllocationId::new(allocation),
            format: AttachmentFormat::Rgba8Unorm,
            width: 2,
            height: 2,
            load: LoadOp::Resident,
            store: StoreOp::Resident,
        }
    }

    /// The image stand-in the host tests store: the registry never looks inside
    /// it, so a tagged identity is enough to prove which entry a call served.
    fn tag(attachment: &RenderAttachment) -> String {
        format!(
            "{}/{}",
            attachment.allocation_id.get(),
            attachment.view_id.get()
        )
    }

    /// One resident store, the pass that creates an identity.
    fn store(registry: &mut ResidentRegistry<String>, attachment: &RenderAttachment) -> String {
        registry
            .resolve(attachment, false, |_, _, _| Ok(tag(attachment)))
            .expect("a resident store creates the identity")
    }

    /// One resident load, the pass that keeps an identity's bytes.
    fn load(
        registry: &mut ResidentRegistry<String>,
        attachment: &RenderAttachment,
    ) -> Result<String, ProviderError> {
        registry.resolve(attachment, true, |_, _, _| {
            panic!("a resident load never creates an image")
        })
    }

    /// A resident load of an identity no pass ever stored is refused by name
    /// instead of being served a fresh image (`research/docs/23` §76, R7).
    #[test]
    fn a_load_of_an_identity_no_pass_stored_is_unavailable() {
        let mut registry = ResidentRegistry::<String>::new();
        let attachment = attachment(7, 9);
        let error = load(&mut registry, &attachment).expect_err("nothing was ever stored");
        assert_eq!(error.slug, "resident_target_unavailable");
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(error.class, ProviderErrorClass::Capability);
        let fields = &error.fields;
        assert_eq!(
            fields.get("view"),
            Some(&FieldValue::Unsigned(ViewId::new(7).get()))
        );
        assert_eq!(
            fields.get("allocation"),
            Some(&FieldValue::Unsigned(AllocationId::new(9).get()))
        );
        assert!(!fields.contains_key("retired_by"));
    }

    /// An entry created by a resident store is not loadable until the pass that
    /// created it completed: "the entry is there" is not "the bytes are there"
    /// (`research/docs/23` §76, R7).
    #[test]
    fn a_created_identity_is_undefined_until_its_pass_completes() {
        let mut registry = ResidentRegistry::<String>::new();
        let attachment = attachment(7, 9);
        assert_eq!(store(&mut registry, &attachment), "9/7");
        let error = load(&mut registry, &attachment).expect_err("the creating pass has not run");
        assert_eq!(error.slug, "resident_target_undefined");

        // The completed pass is what makes the bytes loadable, and the load
        // serves the image the store created rather than a new one.
        registry.note(&[(attachment.allocation_id, attachment.view_id)], true);
        assert_eq!(load(&mut registry, &attachment), Ok("9/7".to_owned()));

        // A refused or failed pass takes the definition back: the identity is
        // unloadable again until another pass renders it.
        registry.note(&[(attachment.allocation_id, attachment.view_id)], false);
        assert_eq!(
            load(&mut registry, &attachment)
                .expect_err("the failed pass defines nothing")
                .slug,
            "resident_target_undefined"
        );
    }

    /// A later pass that names the same identity with a different shape is
    /// refused by name, whichever half of the shape changed
    /// (`research/docs/23` §76, R7).
    #[test]
    fn a_shape_change_is_refused_by_name() {
        let mut registry = ResidentRegistry::<String>::new();
        let created = attachment(7, 9);
        store(&mut registry, &created);
        registry.note(&[(created.allocation_id, created.view_id)], true);

        let mut wider = created;
        wider.width = 4;
        let error = load(&mut registry, &wider).expect_err("the extent changed");
        assert_eq!(error.slug, "resident_target_shape_changed");
        let fields = &error.fields;
        assert_eq!(fields.get("width"), Some(&FieldValue::Unsigned(4)));
        assert_eq!(fields.get("expected_width"), Some(&FieldValue::Unsigned(2)));

        let mut narrower_format = created;
        narrower_format.format = AttachmentFormat::Rgba16Float;
        let error = load(&mut registry, &narrower_format).expect_err("the format changed");
        assert_eq!(error.slug, "resident_target_shape_changed");
        assert_eq!(
            error.fields.get("expected_format"),
            Some(&FieldValue::Unsigned(u64::from(
                AttachmentFormat::Rgba8Unorm.code()
            )))
        );
        // A shape change is not a retirement: the entry is still there, and
        // the identity's own shape still loads.
        assert!(registry.is_live(created.allocation_id, created.view_id));
        assert_eq!(load(&mut registry, &created), Ok("9/7".to_owned()));
        assert_eq!(registry.evictions(), 0);
    }

    /// The budget bounds the registry, the victim is the least recently used
    /// identity, and a load of an evicted identity is refused by name instead of
    /// being served the bytes the image used to hold (`research/docs/23` §76,
    /// R7).
    #[test]
    fn the_budget_evicts_the_least_recently_used_identity() {
        let mut registry = ResidentRegistry::<String>::new();
        let mut identities = Vec::new();
        for index in 0..RESIDENT_TARGET_BUDGET + 2 {
            let attachment = attachment(7 + index as u64, 9 + index as u64);
            store(&mut registry, &attachment);
            registry.note(&[(attachment.allocation_id, attachment.view_id)], true);
            identities.push(attachment);
        }
        assert_eq!(registry.len(), RESIDENT_TARGET_BUDGET);
        assert_eq!(registry.evictions(), 2);

        let evicted = &identities[0];
        assert!(!registry.is_live(evicted.allocation_id, evicted.view_id));
        let error = load(&mut registry, evicted).expect_err("the budget evicted this identity");
        assert_eq!(error.slug, "resident_target_evicted");
        assert_eq!(
            error.fields.get("retired_by"),
            Some(&FieldValue::Text("budget".to_owned()))
        );
        // The identity's own shape is not the reason it is gone, so the refusal
        // is the tombstone's rule and never the shape arm.
        assert_eq!(registry.len(), RESIDENT_TARGET_BUDGET);

        // The most recently used identity is the one the budget kept.
        let survivor = identities.last().expect("the loop stored identities");
        assert!(registry.is_live(survivor.allocation_id, survivor.view_id));

        // Re-storing the evicted identity creates it again and clears the
        // tombstone: the pass is what defines the new image.
        store(&mut registry, evicted);
        registry.note(&[(evicted.allocation_id, evicted.view_id)], true);
        assert_eq!(load(&mut registry, evicted), Ok("9/7".to_owned()));
        assert_eq!(registry.len(), RESIDENT_TARGET_BUDGET);
        assert_eq!(registry.evictions(), 3);
    }

    /// Releasing the lease of one allocation retires that allocation's
    /// identities only, and the refusal names the release
    /// (`research/docs/23` §76, R7).
    #[test]
    fn a_lease_release_retires_one_allocations_identities() {
        let mut registry = ResidentRegistry::<String>::new();
        let released = attachment(7, 9);
        let sibling = attachment(8, 10);
        store(&mut registry, &released);
        store(&mut registry, &sibling);
        registry.note(&[(released.allocation_id, released.view_id)], true);
        registry.note(&[(sibling.allocation_id, sibling.view_id)], true);

        assert_eq!(registry.retire_allocation(released.allocation_id), 1);
        assert_eq!(registry.evictions(), 1);
        let error = load(&mut registry, &released).expect_err("the lease was released");
        assert_eq!(error.slug, "resident_target_released");
        assert_eq!(
            error.fields.get("retired_by"),
            Some(&FieldValue::Text("lease_released".to_owned()))
        );
        // The sibling allocation's identity is untouched, and a second release
        // of the same allocation retires nothing.
        assert_eq!(registry.retire_allocation(released.allocation_id), 0);
        assert_eq!(registry.evictions(), 1);
        assert_eq!(load(&mut registry, &sibling), Ok("10/8".to_owned()));
    }

    /// Losing the device clears every image and leaves the `epoch_advance`
    /// tombstone behind, which is what a later load names — and the teardown is
    /// not charged to the eviction counter (`research/docs/23` §76, R7).
    #[test]
    fn losing_the_device_stales_every_identity() {
        let mut registry = ResidentRegistry::<String>::new();
        let attachment = attachment(7, 9);
        store(&mut registry, &attachment);
        registry.note(&[(attachment.allocation_id, attachment.view_id)], true);
        let evictions = registry.evictions();

        registry.clear();
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.evictions(), evictions);
        let error = load(&mut registry, &attachment).expect_err("the device is gone");
        assert_eq!(error.slug, "resident_target_stale");
        assert_eq!(
            error.fields.get("retired_by"),
            Some(&FieldValue::Text("epoch_advance".to_owned()))
        );

        // A store after the loss creates the identity again under the new
        // device and clears the stale tombstone.
        store(&mut registry, &attachment);
        registry.note(&[(attachment.allocation_id, attachment.view_id)], true);
        assert_eq!(load(&mut registry, &attachment), Ok("9/7".to_owned()));
    }
}
