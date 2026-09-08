//! Transport-independent completion notifications for a provider in another
//! process.
//!
//! The in-process [`CompletionRecord`](super::CompletionRecord) shares state
//! through `Arc`, `Mutex` and `Condvar`. A provider that crosses a process
//! boundary cannot share that memory; it publishes the same terminal semantics
//! as messages over an arbitrary transport (pipe, socket, shared ring or RPC).
//! This module defines only those messages and the receiver-side mirror. It
//! performs no I/O, owns no GPU resources and deliberately has no serialization
//! dependency: the transport chooses the encoding.
//!
//! # Ordering and duplication
//!
//! Every notification carries a per-stream [`CompletionSequence`]. Duplicate,
//! reordered and coalesced notifications are safe to replay. Terminal
//! observations are final: the first `CompletedVisible`, `Cancelled`, `Failed`
//! or `SubmittedUnknown` wins, exactly like
//! [`CompletionRecord`](super::CompletionRecord). When two conflicting terminal
//! notifications arrive in different orders, the lower sequence wins so the
//! mirror converges without depending on delivery order. A `Submitted`
//! notification is only an acknowledgement or heartbeat.
//!
//! Device health is the monotonic lattice `Usable < Exhausted < DeviceLost`.
//! `Exhausted` stops new admissions but leaves in-flight tokens observable;
//! `DeviceLost` is terminal, marks every non-terminal token as `DeviceLost` and
//! is teardown evidence for lease release. A token that already completed keeps
//! its `CompletedVisible` result if the device is lost later.
//!
//! # What is not on the wire
//!
//! `TimedOut` is an observer-local decision and is never published: the owner
//! clamps its own wait, and a timeout does not change provider state.
//! `NotSubmitted` has no token and is not a notification. Host-visible bytes
//! are also not carried here; `CompletedVisible` means the owner may request
//! the readback over the data channel.

use crate::provider::{
    disposition_retires_resources, CompletionDisposition, CompletionToken, ContractError,
    DeviceEpoch, LeaseLedger, LeaseObservation, ProviderError, ProviderErrorClass, ProviderHealth,
    ProviderPhase, Retryability, SubmissionId,
};
use std::collections::BTreeMap;

/// Monotonic revision of one completion stream.
///
/// Sequences are scoped to a token or to the device health stream and start at
/// [`CompletionSequence::FIRST`]. A receiver accepts a notification only when
/// it is newer than the last one accepted for that stream, so retries and
/// reordering cannot regress state. Gaps are allowed: a transport may coalesce
/// or drop non-terminal heartbeats.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompletionSequence(u64);

impl CompletionSequence {
    /// First sequence a sender assigns to a stream.
    pub const FIRST: Self = Self(1);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    pub const fn next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub fn validate(self) -> Result<(), ContractError> {
        if self.is_zero() {
            return Err(ContractError::InvalidIdentity("completion sequence"));
        }
        Ok(())
    }
}

/// Compact failure summary carried by a [`CompletionUpdate::Failed`] message.
///
/// The full [`ProviderError`] also carries diagnostic fields and free-form
/// detail. Those are useful in-process but make a control message
/// transport-specific; the mirror keeps only the stable classification needed
/// to decide retry and lease retirement. A caller may log the detail from a
/// separate diagnostic channel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionFailure {
    pub phase: ProviderPhase,
    pub class: ProviderErrorClass,
    pub slug: String,
    pub retryability: Retryability,
}

impl CompletionFailure {
    pub fn new(
        phase: ProviderPhase,
        class: ProviderErrorClass,
        slug: impl Into<String>,
    ) -> Result<Self, ContractError> {
        let slug = slug.into();
        if slug.trim().is_empty() {
            return Err(ContractError::EmptyField("completion failure slug"));
        }
        Ok(Self {
            phase,
            class,
            slug,
            retryability: Retryability::Unknown,
        })
    }

    pub fn from_provider_error(error: &ProviderError) -> Self {
        Self {
            phase: error.phase,
            class: error.class,
            slug: error.slug.clone(),
            retryability: error.retryability,
        }
    }

    /// Rebuild the structured refusal for a token on the owner side.
    pub fn to_provider_error(
        &self,
        token: CompletionToken,
    ) -> Result<ProviderError, ContractError> {
        token.validate()?;
        let mut error = ProviderError::new(self.phase, self.class, self.slug.clone())?;
        error.retryability = self.retryability;
        error.completion = CompletionDisposition::Failed { token: Some(token) };
        Ok(error)
    }
}

/// Status of one submission token as published by the provider process.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompletionUpdate {
    /// The provider admitted the submission. Non-terminal.
    Submitted,
    /// Host-visible results are ready; readback is requested separately.
    CompletedVisible,
    /// The submission was cancelled before a terminal result was observed.
    /// Cancellation is terminal for observation but is not retirement evidence.
    Cancelled,
    /// The submission failed. The summary carries the refusal classification.
    Failed(CompletionFailure),
    /// The provider can no longer observe the token. Terminal for observation
    /// but not retirement evidence.
    SubmittedUnknown,
}

impl CompletionUpdate {
    pub fn is_terminal(&self) -> bool {
        !matches!(self, Self::Submitted)
    }
}

/// One per-token notification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionTokenUpdate {
    pub token: CompletionToken,
    pub sequence: CompletionSequence,
    pub update: CompletionUpdate,
}

/// One device-wide health notification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionDeviceUpdate {
    pub device_epoch: DeviceEpoch,
    pub sequence: CompletionSequence,
    pub health: ProviderHealth,
}

/// One transport-independent completion notification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompletionMessage {
    Token(CompletionTokenUpdate),
    Device(CompletionDeviceUpdate),
}

impl CompletionMessage {
    /// Validate the identities and sequence before encoding or applying.
    pub fn validate(&self) -> Result<(), ContractError> {
        match self {
            Self::Token(update) => {
                update.sequence.validate()?;
                update.token.validate()
            }
            Self::Device(update) => {
                update.sequence.validate()?;
                if update.device_epoch.is_zero() {
                    return Err(ContractError::InvalidIdentity("completion device epoch"));
                }
                Ok(())
            }
        }
    }
}

/// Result of applying one notification to a [`CompletionMirror`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MirrorOutcome {
    /// The notification advanced the mirror, possibly only its sequence.
    Applied,
    /// The notification was a duplicate, stale or weaker than the recorded
    /// state and changed nothing.
    Ignored,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TokenMirrorState {
    sequence: CompletionSequence,
    status: TokenMirrorStatus,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TokenMirrorStatus {
    Submitted,
    CompletedVisible,
    Cancelled,
    Failed(CompletionFailure),
    SubmittedUnknown,
    DeviceLost,
}

impl TokenMirrorStatus {
    fn is_terminal(&self) -> bool {
        !matches!(self, Self::Submitted)
    }

    fn disposition(&self, token: CompletionToken) -> CompletionDisposition {
        match self {
            Self::Submitted => CompletionDisposition::Submitted { token },
            Self::CompletedVisible => CompletionDisposition::CompletedVisible { token },
            Self::Cancelled => CompletionDisposition::Cancelled { token },
            Self::Failed(_) => CompletionDisposition::Failed { token: Some(token) },
            Self::SubmittedUnknown => {
                CompletionDisposition::SubmittedUnknown { token: Some(token) }
            }
            Self::DeviceLost => CompletionDisposition::DeviceLost { token: Some(token) },
        }
    }
}

/// Receiver-side mirror of one provider instance's completion stream.
///
/// The mirror is a plain value: it accepts validated notifications in any
/// order, stores the strongest observation per token and never blocks. The
/// provider process owns the real [`CompletionRecord`](super::CompletionRecord);
/// this type is the owner's converged view of that record.
///
/// The mirror is scoped to the provider-supplied [`DeviceEpoch`]. That epoch is
/// opaque cross-process identity, not a value allocated by the owner's local
/// `allocate_device_epoch`, so the owner must not compare it with local epochs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionMirror {
    device_epoch: DeviceEpoch,
    health: ProviderHealth,
    device_sequence: Option<CompletionSequence>,
    tokens: BTreeMap<SubmissionId, TokenMirrorState>,
}

impl CompletionMirror {
    pub fn new(device_epoch: DeviceEpoch) -> Result<Self, ContractError> {
        if device_epoch.is_zero() {
            return Err(ContractError::InvalidIdentity("completion device epoch"));
        }
        Ok(Self {
            device_epoch,
            health: ProviderHealth::Usable,
            device_sequence: None,
            tokens: BTreeMap::new(),
        })
    }

    pub const fn device_epoch(&self) -> DeviceEpoch {
        self.device_epoch
    }

    pub const fn health(&self) -> ProviderHealth {
        self.health
    }

    /// Last sequence accepted for the device health stream.
    pub const fn device_sequence(&self) -> Option<CompletionSequence> {
        self.device_sequence
    }

    /// Last sequence accepted for one token. A transport may use this to
    /// resume a stream after reconnecting.
    pub fn token_sequence(&self, token: CompletionToken) -> Option<CompletionSequence> {
        if token.device_epoch != self.device_epoch {
            return None;
        }
        self.tokens
            .get(&token.submission_id)
            .map(|state| state.sequence)
    }

    /// Apply one notification. See the module documentation for the ordering
    /// rules; a duplicate or stale notification returns [`MirrorOutcome::Ignored`].
    pub fn apply(&mut self, message: CompletionMessage) -> Result<MirrorOutcome, ContractError> {
        message.validate()?;
        match message {
            CompletionMessage::Token(update) => self.apply_token(update),
            CompletionMessage::Device(update) => self.apply_device(update),
        }
    }

    fn apply_token(
        &mut self,
        update: CompletionTokenUpdate,
    ) -> Result<MirrorOutcome, ContractError> {
        self.check_epoch(update.token.device_epoch)?;
        if self.health == ProviderHealth::DeviceLost {
            return Ok(MirrorOutcome::Ignored);
        }
        let incoming_terminal = update.update.is_terminal();
        let status = status_from_update(update.update);
        match self.tokens.entry(update.token.submission_id) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(TokenMirrorState {
                    sequence: update.sequence,
                    status,
                });
                Ok(MirrorOutcome::Applied)
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                let state = entry.get_mut();
                let apply = match (incoming_terminal, state.status.is_terminal()) {
                    (true, false) => true,
                    (false, true) => false,
                    (true, true) => update.sequence < state.sequence,
                    (false, false) => update.sequence > state.sequence,
                };
                if !apply {
                    return Ok(MirrorOutcome::Ignored);
                }
                state.sequence = update.sequence;
                state.status = status;
                Ok(MirrorOutcome::Applied)
            }
        }
    }

    fn apply_device(
        &mut self,
        update: CompletionDeviceUpdate,
    ) -> Result<MirrorOutcome, ContractError> {
        self.check_epoch(update.device_epoch)?;
        let is_newer = self
            .device_sequence
            .is_none_or(|seen| update.sequence > seen);
        let is_stronger = health_severity(update.health) > health_severity(self.health);
        if !is_newer && !is_stronger {
            return Ok(MirrorOutcome::Ignored);
        }
        if is_newer {
            self.device_sequence = Some(update.sequence);
        }
        if !is_stronger {
            return Ok(MirrorOutcome::Applied);
        }
        self.health = update.health;
        if update.health == ProviderHealth::DeviceLost {
            for state in self.tokens.values_mut() {
                if !state.status.is_terminal() {
                    state.status = TokenMirrorStatus::DeviceLost;
                }
            }
        }
        Ok(MirrorOutcome::Applied)
    }

    fn check_epoch(&self, epoch: DeviceEpoch) -> Result<(), ContractError> {
        if epoch != self.device_epoch {
            return Err(ContractError::CompletionEpochMismatch {
                expected: self.device_epoch,
                actual: epoch,
            });
        }
        Ok(())
    }

    /// Converged observation for one token, or `None` when the mirror has no
    /// evidence about it. A lost device reports `DeviceLost` for every token of
    /// its epoch unless that token already has a terminal result.
    pub fn observation(&self, token: CompletionToken) -> Option<CompletionDisposition> {
        if token.device_epoch != self.device_epoch {
            return None;
        }
        let state = self.tokens.get(&token.submission_id);
        if let Some(state) = state {
            if state.status.is_terminal() {
                return Some(state.status.disposition(token));
            }
        }
        if self.health == ProviderHealth::DeviceLost {
            return Some(CompletionDisposition::DeviceLost { token: Some(token) });
        }
        state.map(|state| state.status.disposition(token))
    }

    /// Whether the token can no longer change state.
    pub fn is_terminal(&self, token: CompletionToken) -> bool {
        if token.device_epoch != self.device_epoch {
            return false;
        }
        if self.health == ProviderHealth::DeviceLost {
            return true;
        }
        self.tokens
            .get(&token.submission_id)
            .is_some_and(|state| state.status.is_terminal())
    }

    /// Failure summary for a token observed as `Failed`.
    pub fn failure(&self, token: CompletionToken) -> Option<&CompletionFailure> {
        if token.device_epoch != self.device_epoch {
            return None;
        }
        match &self.tokens.get(&token.submission_id)?.status {
            TokenMirrorStatus::Failed(failure) => Some(failure),
            _ => None,
        }
    }

    /// Whether the mirror has retirement evidence for the token. This follows
    /// [`disposition_retires_resources`] and additionally treats device loss as
    /// a teardown guarantee for every token of the epoch.
    pub fn retire_evidence(&self, token: CompletionToken) -> bool {
        if token.device_epoch != self.device_epoch {
            return false;
        }
        if self.health == ProviderHealth::DeviceLost {
            return true;
        }
        self.observation(token)
            .is_some_and(disposition_retires_resources)
    }

    /// Number of tokens the mirror has observed.
    pub fn known_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Number of observed tokens that are not terminal.
    pub fn outstanding(&self) -> usize {
        if self.health == ProviderHealth::DeviceLost {
            return 0;
        }
        self.tokens
            .values()
            .filter(|state| !state.status.is_terminal())
            .count()
    }

    /// Apply this mirror's current observation of `token` to a lease ledger.
    ///
    /// A device-lost status releases the whole ledger because device teardown is
    /// global; otherwise only this token's retirement evidence is applied.
    /// Unknown tokens leave the ledger unchanged.
    pub fn observe_into(
        &self,
        ledger: &mut LeaseLedger,
        token: CompletionToken,
    ) -> Result<LeaseObservation, ContractError> {
        if token.device_epoch != self.device_epoch {
            return Err(ContractError::CompletionEpochMismatch {
                expected: self.device_epoch,
                actual: token.device_epoch,
            });
        }
        if self.health == ProviderHealth::DeviceLost {
            ledger.device_lost();
            return Ok(LeaseObservation::Retired);
        }
        match self.observation(token) {
            Some(disposition) => ledger.observe(token, disposition),
            None => Ok(LeaseObservation::Pending),
        }
    }
}

fn status_from_update(update: CompletionUpdate) -> TokenMirrorStatus {
    match update {
        CompletionUpdate::Submitted => TokenMirrorStatus::Submitted,
        CompletionUpdate::CompletedVisible => TokenMirrorStatus::CompletedVisible,
        CompletionUpdate::Cancelled => TokenMirrorStatus::Cancelled,
        CompletionUpdate::Failed(failure) => TokenMirrorStatus::Failed(failure),
        CompletionUpdate::SubmittedUnknown => TokenMirrorStatus::SubmittedUnknown,
    }
}

const fn health_severity(health: ProviderHealth) -> u8 {
    match health {
        ProviderHealth::Usable => 0,
        ProviderHealth::Exhausted => 1,
        ProviderHealth::DeviceLost => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{
        AllocationId, BufferLease, LeaseId, LeaseReservation, ProviderErrorClass,
    };

    fn epoch() -> DeviceEpoch {
        DeviceEpoch::new(7)
    }

    fn token(submission: u64) -> CompletionToken {
        CompletionToken {
            device_epoch: epoch(),
            submission_id: SubmissionId::new(submission),
        }
    }

    fn seq(value: u64) -> CompletionSequence {
        CompletionSequence::new(value)
    }

    fn token_message(
        submission: u64,
        sequence: u64,
        update: CompletionUpdate,
    ) -> CompletionMessage {
        CompletionMessage::Token(CompletionTokenUpdate {
            token: token(submission),
            sequence: seq(sequence),
            update,
        })
    }

    fn device_message(sequence: u64, health: ProviderHealth) -> CompletionMessage {
        CompletionMessage::Device(CompletionDeviceUpdate {
            device_epoch: epoch(),
            sequence: seq(sequence),
            health,
        })
    }

    fn mirror() -> CompletionMirror {
        CompletionMirror::new(epoch()).expect("nonzero test epoch")
    }

    fn lease(lease_id: u64, allocation_id: u64) -> LeaseReservation {
        LeaseReservation {
            lease: BufferLease {
                lease_id: LeaseId::new(lease_id),
                allocation_id: AllocationId::new(allocation_id),
                owner_epoch: epoch(),
            },
            offset: 0,
            length: 16,
        }
    }

    #[derive(Default)]
    struct LoopbackChannel {
        pending: Vec<CompletionMessage>,
    }

    impl LoopbackChannel {
        fn send(&mut self, message: CompletionMessage) {
            self.pending.push(message);
        }

        fn replay<F>(&mut self, mirror: &mut CompletionMirror, schedule: F)
        where
            F: FnOnce(&mut Vec<CompletionMessage>),
        {
            let mut messages = std::mem::take(&mut self.pending);
            schedule(&mut messages);
            for message in messages {
                mirror.apply(message).expect("valid test message");
            }
        }
    }

    #[test]
    fn completion_sequence_advances_without_overflow() {
        assert_eq!(CompletionSequence::FIRST.get(), 1);
        assert_eq!(seq(3).next(), Some(seq(4)));
        assert_eq!(seq(u64::MAX).next(), None);
        assert!(seq(0).validate().is_err());
    }

    #[test]
    fn submitted_then_completed_converges_under_duplicates_and_reordering() {
        let mut mirror = mirror();
        let mut channel = LoopbackChannel::default();
        channel.send(token_message(1, 1, CompletionUpdate::Submitted));
        channel.send(token_message(1, 2, CompletionUpdate::Submitted));
        channel.send(token_message(1, 3, CompletionUpdate::CompletedVisible));
        channel.replay(&mut mirror, |messages| {
            let copy = messages.clone();
            messages.extend(copy);
            messages.reverse();
        });
        assert_eq!(
            mirror.observation(token(1)),
            Some(CompletionDisposition::CompletedVisible { token: token(1) })
        );
        assert_eq!(mirror.known_tokens(), 1);
        assert_eq!(mirror.outstanding(), 0);
        assert_eq!(mirror.token_sequence(token(1)), Some(seq(3)));
        assert_eq!(mirror.device_sequence(), None);
        assert!(mirror.is_terminal(token(1)));
        assert!(mirror.retire_evidence(token(1)));
    }

    #[test]
    fn first_terminal_wins_regardless_of_delivery_order() {
        let mut completed_first = mirror();
        assert_eq!(
            completed_first
                .apply(token_message(1, 1, CompletionUpdate::CompletedVisible))
                .unwrap(),
            MirrorOutcome::Applied
        );
        assert_eq!(
            completed_first
                .apply(token_message(1, 2, CompletionUpdate::Cancelled))
                .unwrap(),
            MirrorOutcome::Ignored
        );
        assert_eq!(
            completed_first.observation(token(1)),
            Some(CompletionDisposition::CompletedVisible { token: token(1) })
        );

        let mut cancelled_first = mirror();
        assert_eq!(
            cancelled_first
                .apply(token_message(1, 1, CompletionUpdate::Cancelled))
                .unwrap(),
            MirrorOutcome::Applied
        );
        assert_eq!(
            cancelled_first
                .apply(token_message(1, 2, CompletionUpdate::CompletedVisible))
                .unwrap(),
            MirrorOutcome::Ignored
        );
        assert_eq!(
            cancelled_first.observation(token(1)),
            Some(CompletionDisposition::Cancelled { token: token(1) })
        );
    }

    #[test]
    fn lower_sequence_terminal_wins_when_terminals_are_reordered() {
        let mut mirror = mirror();
        assert_eq!(
            mirror
                .apply(token_message(1, 5, CompletionUpdate::CompletedVisible))
                .unwrap(),
            MirrorOutcome::Applied
        );
        assert_eq!(
            mirror
                .apply(token_message(1, 3, CompletionUpdate::Cancelled))
                .unwrap(),
            MirrorOutcome::Applied
        );
        assert_eq!(
            mirror.observation(token(1)),
            Some(CompletionDisposition::Cancelled { token: token(1) })
        );
        assert_eq!(
            mirror
                .apply(token_message(1, 5, CompletionUpdate::CompletedVisible))
                .unwrap(),
            MirrorOutcome::Ignored
        );
    }

    #[test]
    fn terminal_supersedes_a_newer_heartbeat() {
        let mut mirror = mirror();
        assert_eq!(
            mirror
                .apply(token_message(1, 5, CompletionUpdate::Submitted))
                .unwrap(),
            MirrorOutcome::Applied
        );
        assert_eq!(
            mirror
                .apply(token_message(1, 3, CompletionUpdate::CompletedVisible))
                .unwrap(),
            MirrorOutcome::Applied
        );
        assert_eq!(
            mirror.observation(token(1)),
            Some(CompletionDisposition::CompletedVisible { token: token(1) })
        );
        assert_eq!(
            mirror
                .apply(token_message(1, 6, CompletionUpdate::Submitted))
                .unwrap(),
            MirrorOutcome::Ignored
        );
    }

    #[test]
    fn submitted_unknown_is_terminal_but_not_retirement_evidence() {
        let mut mirror = mirror();
        assert_eq!(
            mirror
                .apply(token_message(1, 1, CompletionUpdate::SubmittedUnknown))
                .unwrap(),
            MirrorOutcome::Applied
        );
        assert_eq!(
            mirror.observation(token(1)),
            Some(CompletionDisposition::SubmittedUnknown {
                token: Some(token(1))
            })
        );
        assert!(mirror.is_terminal(token(1)));
        assert!(!mirror.retire_evidence(token(1)));
        assert_eq!(
            mirror
                .apply(token_message(1, 2, CompletionUpdate::CompletedVisible))
                .unwrap(),
            MirrorOutcome::Ignored
        );
    }

    #[test]
    fn failed_carries_the_summary_and_rebuilds_provider_error() {
        let failure = CompletionFailure::new(
            ProviderPhase::Wait,
            ProviderErrorClass::Execute,
            "synthetic_failure",
        )
        .unwrap();
        let mut mirror = mirror();
        mirror
            .apply(token_message(
                1,
                1,
                CompletionUpdate::Failed(failure.clone()),
            ))
            .unwrap();
        assert_eq!(mirror.failure(token(1)), Some(&failure));
        assert_eq!(
            mirror.observation(token(1)),
            Some(CompletionDisposition::Failed {
                token: Some(token(1))
            })
        );
        assert!(mirror.retire_evidence(token(1)));

        let error = failure.to_provider_error(token(1)).unwrap();
        assert_eq!(error.phase, ProviderPhase::Wait);
        assert_eq!(error.class, ProviderErrorClass::Execute);
        assert_eq!(error.slug, "synthetic_failure");
        assert_eq!(error.retryability, Retryability::Unknown);
        assert_eq!(
            error.completion,
            CompletionDisposition::Failed {
                token: Some(token(1))
            }
        );

        let mut provider_error = ProviderError::new(
            ProviderPhase::Submit,
            ProviderErrorClass::Resource,
            "provider_unavailable",
        )
        .unwrap();
        provider_error.retryability = Retryability::RetryAfterRecreate;
        let summary = CompletionFailure::from_provider_error(&provider_error);
        assert_eq!(summary.slug, "provider_unavailable");
        assert_eq!(summary.retryability, Retryability::RetryAfterRecreate);
    }

    #[test]
    fn exhausted_health_keeps_in_flight_tokens_observable() {
        let mut mirror = mirror();
        assert_eq!(
            mirror
                .apply(device_message(1, ProviderHealth::Exhausted))
                .unwrap(),
            MirrorOutcome::Applied
        );
        assert_eq!(mirror.health(), ProviderHealth::Exhausted);
        assert_eq!(mirror.device_sequence(), Some(seq(1)));
        mirror
            .apply(token_message(1, 1, CompletionUpdate::Submitted))
            .unwrap();
        assert_eq!(mirror.outstanding(), 1);
        mirror
            .apply(token_message(1, 2, CompletionUpdate::CompletedVisible))
            .unwrap();
        assert_eq!(
            mirror.observation(token(1)),
            Some(CompletionDisposition::CompletedVisible { token: token(1) })
        );
        assert_eq!(mirror.outstanding(), 0);
        assert_eq!(
            mirror
                .apply(token_message(2, 1, CompletionUpdate::Submitted))
                .unwrap(),
            MirrorOutcome::Applied
        );
        assert_eq!(mirror.outstanding(), 1);
    }

    #[test]
    fn device_loss_marks_running_tokens_and_preserves_completed_results() {
        let mut mirror = mirror();
        mirror
            .apply(token_message(1, 1, CompletionUpdate::Submitted))
            .unwrap();
        mirror
            .apply(token_message(2, 1, CompletionUpdate::CompletedVisible))
            .unwrap();
        assert_eq!(
            mirror
                .apply(device_message(1, ProviderHealth::DeviceLost))
                .unwrap(),
            MirrorOutcome::Applied
        );
        assert_eq!(mirror.health(), ProviderHealth::DeviceLost);
        assert_eq!(
            mirror.observation(token(1)),
            Some(CompletionDisposition::DeviceLost {
                token: Some(token(1))
            })
        );
        assert_eq!(
            mirror.observation(token(2)),
            Some(CompletionDisposition::CompletedVisible { token: token(2) })
        );
        assert_eq!(
            mirror.observation(token(3)),
            Some(CompletionDisposition::DeviceLost {
                token: Some(token(3))
            })
        );
        assert!(mirror.retire_evidence(token(1)));
        assert!(mirror.retire_evidence(token(2)));
        assert!(mirror.retire_evidence(token(3)));
        assert_eq!(mirror.outstanding(), 0);
        assert_eq!(
            mirror
                .apply(token_message(1, 2, CompletionUpdate::CompletedVisible))
                .unwrap(),
            MirrorOutcome::Ignored
        );
        assert_eq!(
            mirror.observation(token(1)),
            Some(CompletionDisposition::DeviceLost {
                token: Some(token(1))
            })
        );
    }

    #[test]
    fn device_loss_wins_over_a_newer_exhaustion_message() {
        let mut mirror = mirror();
        assert_eq!(
            mirror
                .apply(device_message(2, ProviderHealth::Exhausted))
                .unwrap(),
            MirrorOutcome::Applied
        );
        assert_eq!(
            mirror
                .apply(device_message(1, ProviderHealth::DeviceLost))
                .unwrap(),
            MirrorOutcome::Applied
        );
        assert_eq!(mirror.health(), ProviderHealth::DeviceLost);
        assert_eq!(mirror.device_sequence(), Some(seq(2)));
        assert_eq!(
            mirror
                .apply(device_message(3, ProviderHealth::Usable))
                .unwrap(),
            MirrorOutcome::Applied
        );
        assert_eq!(mirror.health(), ProviderHealth::DeviceLost);
        assert_eq!(mirror.device_sequence(), Some(seq(3)));
        assert_eq!(
            mirror
                .apply(device_message(2, ProviderHealth::Exhausted))
                .unwrap(),
            MirrorOutcome::Ignored
        );
        assert_eq!(mirror.device_sequence(), Some(seq(3)));
    }

    #[test]
    fn rejects_invalid_identities_and_epochs() {
        assert!(CompletionMirror::new(DeviceEpoch::new(0)).is_err());
        let mut mirror = mirror();
        assert_eq!(
            mirror
                .apply(token_message(1, 0, CompletionUpdate::Submitted))
                .unwrap_err(),
            ContractError::InvalidIdentity("completion sequence")
        );
        let zero_submission = CompletionMessage::Token(CompletionTokenUpdate {
            token: CompletionToken {
                device_epoch: epoch(),
                submission_id: SubmissionId::new(0),
            },
            sequence: seq(1),
            update: CompletionUpdate::Submitted,
        });
        assert_eq!(
            mirror.apply(zero_submission).unwrap_err(),
            ContractError::InvalidIdentity("submission id")
        );
        let foreign = CompletionMessage::Token(CompletionTokenUpdate {
            token: CompletionToken {
                device_epoch: DeviceEpoch::new(8),
                submission_id: SubmissionId::new(1),
            },
            sequence: seq(1),
            update: CompletionUpdate::Submitted,
        });
        assert_eq!(
            mirror.apply(foreign).unwrap_err(),
            ContractError::CompletionEpochMismatch {
                expected: epoch(),
                actual: DeviceEpoch::new(8),
            }
        );
        assert_eq!(
            mirror
                .apply(device_message(0, ProviderHealth::Usable))
                .unwrap_err(),
            ContractError::InvalidIdentity("completion sequence")
        );
        let zero_epoch = CompletionMessage::Device(CompletionDeviceUpdate {
            device_epoch: DeviceEpoch::new(0),
            sequence: seq(1),
            health: ProviderHealth::Usable,
        });
        assert_eq!(
            mirror.apply(zero_epoch).unwrap_err(),
            ContractError::InvalidIdentity("completion device epoch")
        );
        assert!(
            CompletionFailure::new(ProviderPhase::Wait, ProviderErrorClass::Internal, " ").is_err()
        );
    }

    #[test]
    fn mirror_and_lease_ledger_compose() {
        let mut ledger = LeaseLedger::new();
        ledger.register(lease(1, 1)).unwrap();
        ledger.register(lease(2, 2)).unwrap();
        ledger.bind(LeaseId::new(1), token(1)).unwrap();
        ledger.bind(LeaseId::new(2), token(2)).unwrap();

        let mut mirror = mirror();
        mirror
            .apply(token_message(1, 1, CompletionUpdate::Submitted))
            .unwrap();
        mirror
            .apply(token_message(2, 1, CompletionUpdate::Submitted))
            .unwrap();
        mirror
            .apply(token_message(1, 2, CompletionUpdate::CompletedVisible))
            .unwrap();
        assert_eq!(
            mirror.observe_into(&mut ledger, token(1)).unwrap(),
            LeaseObservation::Retired
        );
        assert!(ledger.release_ready(LeaseId::new(1)));
        assert!(!ledger.release_ready(LeaseId::new(2)));
        assert_eq!(
            mirror.observe_into(&mut ledger, token(2)).unwrap(),
            LeaseObservation::Pending
        );
        assert!(!ledger.release_ready(LeaseId::new(2)));

        mirror
            .apply(device_message(1, ProviderHealth::DeviceLost))
            .unwrap();
        assert_eq!(
            mirror.observe_into(&mut ledger, token(2)).unwrap(),
            LeaseObservation::Retired
        );
        assert!(ledger.release_ready(LeaseId::new(2)));
        assert_eq!(ledger.release_all_ready().len(), 2);

        let foreign = CompletionToken {
            device_epoch: DeviceEpoch::new(8),
            submission_id: SubmissionId::new(1),
        };
        assert_eq!(
            mirror.observe_into(&mut ledger, foreign).unwrap_err(),
            ContractError::CompletionEpochMismatch {
                expected: epoch(),
                actual: DeviceEpoch::new(8),
            }
        );
    }
}
