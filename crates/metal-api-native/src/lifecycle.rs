//! Native Metal admission and terminal-state authority.
//!
//! One `metal_api_core::ProviderLifecycle` owns everything a submission can be
//! refused for: the bounded abandonment budget, the abandoned counters and the
//! terminal cause. The provider keeps no second copy of that state, so it can
//! not report one health while refusing on another (`docs/PROVIDER-B1.md` §7).
//!
//! Two things stay provider-specific, and this module is the whole of them:
//!
//! - The device-loss *slug*. The core refusal spells the generic `device_lost`
//!   for every provider; native Metal has published `metal_device_removed`
//!   since the 2026-09-08 device-removal increment, and `docs/PROVIDER-B1.md`
//!   §7 records that the two slugs stay distinct.
//! - A handle a Metal completion handler can share. A handler runs on a driver
//!   thread after `submit` has returned, so it cannot borrow the provider; it
//!   holds an `Arc<NativeLifecycle>` and locks the one shared lifecycle instead
//!   of a separate atomic per state flag.

use metal_api_core::completion::{AbandonmentBudget, AbandonmentOutcome};
use metal_api_core::provider::{
    ProviderError, ProviderHealth, ProviderLifecycle, TerminalRefusal, TerminalRefusalReason,
};
use std::sync::{Mutex, MutexGuard};

/// Slug this provider reports for an observed device loss.
///
/// The refusal stays structurally the lifecycle's; only this spelling is the
/// provider's (`docs/PROVIDER-B1.md` §7: `device_lost` on Vulkan,
/// `metal_device_removed` on native Metal).
pub(crate) const DEVICE_LOST_SLUG: &str = "metal_device_removed";

/// Default bound of one native provider instance: eight abandoned submissions
/// or 64 MiB, whichever comes first (`docs/PROVIDER-B1.md` §7).
pub(crate) const DEFAULT_ABANDONMENT_BUDGET: AbandonmentBudget =
    AbandonmentBudget::new(8, 64 * 1024 * 1024);

/// Admission, health and abandonment counters of one native provider instance.
pub(crate) struct NativeLifecycle {
    lifecycle: Mutex<ProviderLifecycle>,
}

impl NativeLifecycle {
    /// A lifecycle bounded by [`DEFAULT_ABANDONMENT_BUDGET`].
    pub(crate) fn new() -> Self {
        Self::with_budget(DEFAULT_ABANDONMENT_BUDGET)
    }

    /// A lifecycle bounded by `budget`, as `with_abandonment_budget` builds it.
    pub(crate) fn with_budget(budget: AbandonmentBudget) -> Self {
        Self {
            lifecycle: Mutex::new(ProviderLifecycle::new(
                budget.max_submissions(),
                budget.max_bytes(),
            )),
        }
    }

    /// Lock the lifecycle for one transition or query.
    ///
    /// Terminal states are monotonic and admission never unwinds through this
    /// mutex, so a poison left by an unrelated panic still protects the last
    /// recorded state; recovering the guard keeps refusal and health available
    /// instead of turning one panic into a second, unrelated failure.
    fn lock(&self) -> MutexGuard<'_, ProviderLifecycle> {
        self.lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Admit one submission against this provider's terminal state.
    pub(crate) fn admit(&self) -> Result<(), ProviderError> {
        self.lock().admit().map_err(admission_refusal)
    }

    /// Health of the same state that admission reads.
    pub(crate) fn health(&self) -> ProviderHealth {
        self.lock().health()
    }

    /// `(abandoned submissions, abandoned bytes)` recorded so far.
    pub(crate) fn abandonment(&self) -> (u64, u64) {
        self.lock().abandonment()
    }

    /// Give up on one submission whose completion is no longer observable.
    ///
    /// The budget decides: the provider stays usable while the bound holds and
    /// turns terminal in the same transition that reaches it. A terminal
    /// lifecycle neither re-charges nor re-reports the counters.
    pub(crate) fn record_abandonment(&self, bytes: u64) -> AbandonmentOutcome {
        self.lock().record_abandonment(bytes)
    }

    /// Fail the instance closed on a submission that is not abandoned GPU
    /// work, without charging the budget.
    ///
    /// This is the transition for every terminal `MTLCommandBuffer` failure
    /// that is not a device removal, and for a wait that never reaches a
    /// terminal command-buffer status (`docs/PROVIDER-B1.md`, "Execution
    /// failures and visibility"): the counters keep reporting real abandoned
    /// work, and new work is refused with the same `provider_unavailable`
    /// refusal.
    pub(crate) fn mark_unobservable_submission(&self) {
        self.lock().mark_unobservable_submission();
    }

    /// Mark the device as lost (`MTLCommandBufferError::DeviceRemoved`).
    pub(crate) fn mark_device_lost(&self) {
        self.lock().mark_device_lost();
    }
}

impl Default for NativeLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

/// Map the lifecycle's refusal onto the provider error this crate publishes.
///
/// Phase, class, retryability, completion disposition and the structured
/// `terminal` / `abandoned_submissions` / `abandoned_bytes` fields are the
/// lifecycle's and cross unchanged, and the branch is on the typed
/// [`TerminalRefusalReason`] rather than on any message text. The one
/// provider-specific detail is the device-loss slug: the core refusal spells
/// the generic `device_lost`, while native Metal publishes
/// [`DEVICE_LOST_SLUG`] (`docs/PROVIDER-B1.md` §7).
pub(crate) fn admission_refusal(refusal: TerminalRefusal) -> ProviderError {
    let reason = refusal.reason();
    let mut error = refusal.into_error();
    if reason == TerminalRefusalReason::DeviceLost {
        error.slug = DEVICE_LOST_SLUG.to_owned();
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::provider::{
        CompletionDisposition, FieldValue, ProviderErrorClass, ProviderPhase, Retryability,
    };

    /// The terminal cause every refusal carries, whatever its slug spelling.
    fn assert_terminal_field(refusal: &ProviderError, terminal: &str) {
        assert_eq!(
            refusal.fields.get("terminal"),
            Some(&FieldValue::Text(terminal.to_owned())),
            "terminal field is wrong: {refusal:?}"
        );
    }

    /// The counters the abandonment refusal carries. A device-loss refusal has
    /// none: a device loss is not an abandonment and never charges the budget.
    fn assert_abandonment_counters(refusal: &ProviderError, submissions: u64, bytes: u64) {
        assert_eq!(
            refusal.fields.get("abandoned_submissions"),
            Some(&FieldValue::Unsigned(submissions)),
            "abandoned submissions are wrong: {refusal:?}"
        );
        assert_eq!(
            refusal.fields.get("abandoned_bytes"),
            Some(&FieldValue::Unsigned(bytes)),
            "abandoned bytes are wrong: {refusal:?}"
        );
    }

    /// Health and admission have to describe one state. This mirrors the check
    /// the Vulkan context carries, against the same core lifecycle.
    fn assert_health_and_refusal_agree(lifecycle: &NativeLifecycle) {
        let health = lifecycle.health();
        match lifecycle.admit() {
            Ok(()) => assert_eq!(health, ProviderHealth::Usable),
            Err(error) => {
                assert!(
                    !health.is_usable(),
                    "usable lifecycle refused work: {error:?}"
                );
                let expected = match health {
                    ProviderHealth::DeviceLost => "device_lost",
                    ProviderHealth::Exhausted => "abandonment_budget",
                    ProviderHealth::Usable => unreachable!("handled by the match arm above"),
                };
                assert_terminal_field(&error, expected);
                assert_eq!(
                    error.fields.contains_key("abandoned_submissions"),
                    health == ProviderHealth::Exhausted,
                    "abandonment counters disagree with the reported health: {error:?}"
                );
                if health == ProviderHealth::Exhausted {
                    assert_unsigned_field(&error, "abandoned_submissions");
                    assert_unsigned_field(&error, "abandoned_bytes");
                }
                assert_eq!(error.phase, ProviderPhase::Resolve);
                assert_eq!(error.retryability, Retryability::RetryAfterRecreate);
            }
        }
    }

    fn assert_unsigned_field(error: &ProviderError, key: &str) {
        match error.fields.get(key) {
            Some(FieldValue::Unsigned(_)) => {}
            other => panic!("{key} is not an unsigned field: {other:?}"),
        }
    }

    #[test]
    fn exhausted_budget_refuses_new_work_idempotently() {
        let lifecycle = NativeLifecycle::with_budget(AbandonmentBudget::new(1, 4096));
        assert_eq!(lifecycle.health(), ProviderHealth::Usable);
        assert_eq!(lifecycle.abandonment(), (0, 0));
        lifecycle.admit().expect("a usable lifecycle admits work");

        // One unobservable submission reaches the submission bound, which is
        // the whole budget here, so the first abandonment ends the instance.
        assert_eq!(
            lifecycle.record_abandonment(4096),
            AbandonmentOutcome::Exhausted
        );
        assert_eq!(lifecycle.health(), ProviderHealth::Exhausted);

        let refusal = lifecycle
            .admit()
            .expect_err("an exhausted lifecycle refuses work");
        assert_eq!(refusal.phase, ProviderPhase::Resolve);
        assert_eq!(refusal.class, ProviderErrorClass::Resource);
        assert_eq!(refusal.slug, "provider_unavailable");
        assert_eq!(refusal.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(refusal.completion, CompletionDisposition::NotSubmitted);
        assert_terminal_field(&refusal, "abandonment_budget");
        assert_abandonment_counters(&refusal, 1, 4096);

        // Retries answer the same structured refusal, and a terminal lifecycle
        // neither re-charges its budget nor drifts into another reason.
        for _ in 0..3 {
            assert_eq!(
                lifecycle
                    .admit()
                    .expect_err("an exhausted lifecycle admits"),
                refusal
            );
            assert_eq!(
                lifecycle.record_abandonment(4096),
                AbandonmentOutcome::Exhausted
            );
            assert_eq!(lifecycle.abandonment(), (1, 4096));
            assert_eq!(lifecycle.health(), ProviderHealth::Exhausted);
        }
        assert_health_and_refusal_agree(&lifecycle);
    }

    #[test]
    fn device_loss_and_budget_exhaustion_stay_distinguishable() {
        let lost = NativeLifecycle::new();
        let exhausted = NativeLifecycle::with_budget(AbandonmentBudget::new(1, 4096));
        assert_eq!(
            exhausted.record_abandonment(64),
            AbandonmentOutcome::Exhausted
        );
        lost.mark_device_lost();

        assert_eq!(lost.health(), ProviderHealth::DeviceLost);
        assert_eq!(exhausted.health(), ProviderHealth::Exhausted);

        let lost_refusal = lost.admit().expect_err("a lost device refuses work");
        assert_eq!(lost_refusal.phase, ProviderPhase::Resolve);
        assert_eq!(lost_refusal.class, ProviderErrorClass::DeviceLost);
        // The provider keeps its own device-loss slug; only that spelling
        // differs from the core's generic `device_lost`.
        assert_eq!(lost_refusal.slug, DEVICE_LOST_SLUG);
        assert_eq!(lost_refusal.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(
            lost_refusal.completion,
            CompletionDisposition::DeviceLost { token: None }
        );
        assert_terminal_field(&lost_refusal, "device_lost");

        let exhausted_refusal = exhausted
            .admit()
            .expect_err("an exhausted lifecycle refuses work");
        assert_eq!(exhausted_refusal.class, ProviderErrorClass::Resource);
        assert_eq!(exhausted_refusal.slug, "provider_unavailable");
        assert_ne!(lost_refusal, exhausted_refusal);

        // A device loss is not an abandonment: it never consumes the budget,
        // and a later abandonment never relabels the observed loss.
        assert_eq!(lost.abandonment(), (0, 0));
        assert_eq!(lost.record_abandonment(4096), AbandonmentOutcome::Exhausted);
        assert_eq!(lost.abandonment(), (0, 0));
        assert_eq!(lost.health(), ProviderHealth::DeviceLost);
        assert_eq!(
            lost.admit().expect_err("a lost device stays lost").slug,
            DEVICE_LOST_SLUG
        );
        assert_health_and_refusal_agree(&lost);
        assert_health_and_refusal_agree(&exhausted);
    }

    #[test]
    fn health_and_admission_read_one_lifecycle() {
        let lifecycle = NativeLifecycle::new();
        assert_eq!(lifecycle.health(), ProviderHealth::Usable);
        assert_eq!(lifecycle.abandonment(), (0, 0));
        assert_health_and_refusal_agree(&lifecycle);

        // Control: a tolerated abandonment moves the counters, keeps the
        // provider admitting work and produces no terminal cause.
        assert_eq!(
            lifecycle.record_abandonment(4096),
            AbandonmentOutcome::Admitted
        );
        assert_eq!(lifecycle.abandonment(), (1, 4096));
        assert_eq!(lifecycle.health(), ProviderHealth::Usable);
        assert_health_and_refusal_agree(&lifecycle);

        // Control: the bound is what decides, not the first abandonment.
        let bounded = NativeLifecycle::with_budget(AbandonmentBudget::new(2, 8192));
        assert_eq!(
            bounded.record_abandonment(4096),
            AbandonmentOutcome::Admitted
        );
        assert_eq!(bounded.health(), ProviderHealth::Usable);
        assert_eq!(
            bounded.record_abandonment(4096),
            AbandonmentOutcome::Exhausted
        );
        assert_eq!(bounded.abandonment(), (2, 8192));
        assert_eq!(bounded.health(), ProviderHealth::Exhausted);
        assert_health_and_refusal_agree(&bounded);
    }

    #[test]
    fn a_failed_command_buffer_seals_the_provider_without_charging_the_budget() {
        let lifecycle = NativeLifecycle::new();
        // `metal_command_failed` is this provider's terminal, non-device-loss
        // command-buffer outcome: new work stops, but the submission is not
        // booked as abandoned GPU work.
        lifecycle.mark_unobservable_submission();

        assert_eq!(lifecycle.health(), ProviderHealth::Exhausted);
        assert_eq!(lifecycle.abandonment(), (0, 0));
        let refusal = lifecycle
            .admit()
            .expect_err("a sealed provider refuses work");
        assert_eq!(refusal.class, ProviderErrorClass::Resource);
        assert_eq!(refusal.slug, "provider_unavailable");
        assert_eq!(refusal.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(refusal.completion, CompletionDisposition::NotSubmitted);
        assert_terminal_field(&refusal, "abandonment_budget");
        assert_abandonment_counters(&refusal, 0, 0);

        // Sealing is idempotent, and an observed device loss still wins over
        // the abandonment cause.
        lifecycle.mark_unobservable_submission();
        assert_eq!(
            lifecycle.admit().expect_err("a sealed provider admits"),
            refusal
        );
        lifecycle.mark_device_lost();
        assert_eq!(lifecycle.health(), ProviderHealth::DeviceLost);
        assert_eq!(
            lifecycle.admit().expect_err("a lost device admits").slug,
            DEVICE_LOST_SLUG
        );
        assert_eq!(lifecycle.abandonment(), (0, 0));
    }
}
