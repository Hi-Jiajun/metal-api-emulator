//! Provider-side sender for the transport-independent completion stream.
//!
//! [`CompletionPublisher`] is the dual of
//! [`CompletionMirror`](super::CompletionMirror). A provider process owns the
//! authoritative completion state; the publisher assigns the monotonic
//! sequence for every notification and refuses transitions that would
//! contradict the state it already published. It performs no I/O: the caller
//! encodes and sends the returned message.
//!
//! A terminal update is final. Publishing the same terminal update again is an
//! idempotent replay and returns the identical message, so a transport may
//! retry a send without consuming a new sequence. Publishing a different
//! terminal update, or any non-terminal update after device loss, is refused.
//! Device health is monotonic `Usable < Exhausted < DeviceLost`; re-publishing
//! the current health returns the identical device message.

use super::{
    health_severity, CompletionDeviceUpdate, CompletionFailure, CompletionSequence,
    CompletionTokenUpdate, CompletionUpdate,
};
use crate::provider::{
    CompletionToken, ContractError, DeviceEpoch, ProviderError, ProviderHealth, SubmissionId,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq)]
struct TokenPublishState {
    last: CompletionTokenUpdate,
}

/// Sender-side dual of [`CompletionMirror`](super::CompletionMirror).
///
/// The publisher is scoped to the provider-supplied [`DeviceEpoch`] and
/// performs no I/O. A provider calls [`CompletionPublisher::publish`] or a
/// convenience method at the same point where it transitions the authoritative
/// in-process completion state, then hands the returned value to its transport.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionPublisher {
    device_epoch: DeviceEpoch,
    health: ProviderHealth,
    device_update: Option<CompletionDeviceUpdate>,
    tokens: BTreeMap<SubmissionId, TokenPublishState>,
}

impl CompletionPublisher {
    pub fn new(device_epoch: DeviceEpoch) -> Result<Self, ContractError> {
        if device_epoch.is_zero() {
            return Err(ContractError::InvalidIdentity("completion device epoch"));
        }
        Ok(Self {
            device_epoch,
            health: ProviderHealth::Usable,
            device_update: None,
            tokens: BTreeMap::new(),
        })
    }

    pub const fn device_epoch(&self) -> DeviceEpoch {
        self.device_epoch
    }

    pub const fn health(&self) -> ProviderHealth {
        self.health
    }

    pub const fn device_sequence(&self) -> Option<CompletionSequence> {
        match self.device_update {
            Some(update) => Some(update.sequence),
            None => None,
        }
    }

    pub fn token_sequence(&self, token: CompletionToken) -> Option<CompletionSequence> {
        if token.device_epoch != self.device_epoch {
            return None;
        }
        self.tokens
            .get(&token.submission_id)
            .map(|state| state.last.sequence)
    }

    /// Publish one token update and return the message to send.
    ///
    /// The same terminal update may be published repeatedly and returns the
    /// identical message. A different terminal, a non-terminal update after a
    /// terminal, or any non-terminal update after device loss is refused.
    pub fn publish(
        &mut self,
        token: CompletionToken,
        update: CompletionUpdate,
    ) -> Result<CompletionTokenUpdate, ContractError> {
        token.validate()?;
        self.check_epoch(token.device_epoch)?;
        if let Some(state) = self.tokens.get(&token.submission_id) {
            if state.last.update.is_terminal() {
                if state.last.update == update {
                    return Ok(state.last.clone());
                }
                return Err(ContractError::CompletionPublishAfterTerminal(token));
            }
        }
        if self.health == ProviderHealth::DeviceLost {
            return Err(ContractError::CompletionPublishAfterDeviceLost(token));
        }
        let sequence = match self.tokens.get(&token.submission_id) {
            Some(state) => state
                .last
                .sequence
                .next()
                .ok_or(ContractError::ArithmeticOverflow("completion sequence"))?,
            None => CompletionSequence::FIRST,
        };
        let message = CompletionTokenUpdate {
            token,
            sequence,
            update,
        };
        self.tokens.insert(
            token.submission_id,
            TokenPublishState {
                last: message.clone(),
            },
        );
        Ok(message)
    }

    pub fn submitted(
        &mut self,
        token: CompletionToken,
    ) -> Result<CompletionTokenUpdate, ContractError> {
        self.publish(token, CompletionUpdate::Submitted)
    }

    pub fn completed_visible(
        &mut self,
        token: CompletionToken,
    ) -> Result<CompletionTokenUpdate, ContractError> {
        self.publish(token, CompletionUpdate::CompletedVisible)
    }

    pub fn cancelled(
        &mut self,
        token: CompletionToken,
    ) -> Result<CompletionTokenUpdate, ContractError> {
        self.publish(token, CompletionUpdate::Cancelled)
    }

    pub fn failed(
        &mut self,
        token: CompletionToken,
        failure: CompletionFailure,
    ) -> Result<CompletionTokenUpdate, ContractError> {
        self.publish(token, CompletionUpdate::Failed(failure))
    }

    pub fn failed_from_provider_error(
        &mut self,
        token: CompletionToken,
        error: &ProviderError,
    ) -> Result<CompletionTokenUpdate, ContractError> {
        self.failed(token, CompletionFailure::from_provider_error(error))
    }

    pub fn submitted_unknown(
        &mut self,
        token: CompletionToken,
    ) -> Result<CompletionTokenUpdate, ContractError> {
        self.publish(token, CompletionUpdate::SubmittedUnknown)
    }

    /// Publish a device health transition and return the message to send.
    ///
    /// Health is monotonic. Re-publishing the current health returns the
    /// identical message; a weaker health is refused.
    pub fn publish_device(
        &mut self,
        health: ProviderHealth,
    ) -> Result<CompletionDeviceUpdate, ContractError> {
        if health_severity(health) < health_severity(self.health) {
            return Err(ContractError::CompletionHealthRegression {
                current: self.health,
                requested: health,
            });
        }
        if health == self.health {
            if let Some(update) = self.device_update {
                return Ok(update);
            }
        }
        let sequence = match self.device_update {
            Some(update) => update
                .sequence
                .next()
                .ok_or(ContractError::ArithmeticOverflow("completion sequence"))?,
            None => CompletionSequence::FIRST,
        };
        let update = CompletionDeviceUpdate {
            device_epoch: self.device_epoch,
            sequence,
            health,
        };
        self.health = health;
        self.device_update = Some(update);
        Ok(update)
    }

    pub fn usable(&mut self) -> Result<CompletionDeviceUpdate, ContractError> {
        self.publish_device(ProviderHealth::Usable)
    }

    pub fn exhausted(&mut self) -> Result<CompletionDeviceUpdate, ContractError> {
        self.publish_device(ProviderHealth::Exhausted)
    }

    pub fn device_lost(&mut self) -> Result<CompletionDeviceUpdate, ContractError> {
        self.publish_device(ProviderHealth::DeviceLost)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::wire::{CompletionMessage, CompletionMirror, LoopbackTransport};
    use crate::provider::{CompletionDisposition, ProviderErrorClass, ProviderPhase};

    fn epoch() -> DeviceEpoch {
        DeviceEpoch::new(7)
    }

    fn token(submission: u64) -> CompletionToken {
        CompletionToken {
            device_epoch: epoch(),
            submission_id: SubmissionId::new(submission),
        }
    }

    fn publisher() -> CompletionPublisher {
        CompletionPublisher::new(epoch()).expect("nonzero test epoch")
    }

    fn mirror() -> CompletionMirror {
        CompletionMirror::new(epoch()).expect("nonzero test epoch")
    }

    #[test]
    fn publisher_assigns_monotonic_sequences_per_token() {
        let mut publisher = publisher();
        assert_eq!(publisher.submitted(token(1)).unwrap().sequence.get(), 1);
        assert_eq!(publisher.submitted(token(1)).unwrap().sequence.get(), 2);
        assert_eq!(
            publisher
                .completed_visible(token(1))
                .unwrap()
                .sequence
                .get(),
            3
        );
        assert_eq!(publisher.submitted(token(2)).unwrap().sequence.get(), 1);
        assert_eq!(publisher.token_sequence(token(1)).unwrap().get(), 3);
        assert_eq!(publisher.token_sequence(token(2)).unwrap().get(), 1);
    }

    #[test]
    fn terminal_replay_is_idempotent_and_conflicts_are_refused() {
        let mut publisher = publisher();
        let first = publisher.completed_visible(token(1)).unwrap();
        let replay = publisher.completed_visible(token(1)).unwrap();
        assert_eq!(first, replay);
        assert_eq!(replay.sequence.get(), 1);
        assert_eq!(
            publisher.cancelled(token(1)).unwrap_err(),
            ContractError::CompletionPublishAfterTerminal(token(1))
        );
        assert_eq!(
            publisher.submitted(token(1)).unwrap_err(),
            ContractError::CompletionPublishAfterTerminal(token(1))
        );
    }

    #[test]
    fn device_health_is_monotonic_and_idempotent() {
        let mut publisher = publisher();
        let usable = publisher.usable().unwrap();
        assert_eq!(usable.sequence.get(), 1);
        assert_eq!(publisher.usable().unwrap(), usable);
        let exhausted = publisher.exhausted().unwrap();
        assert_eq!(exhausted.sequence.get(), 2);
        assert_eq!(publisher.exhausted().unwrap(), exhausted);
        let lost = publisher.device_lost().unwrap();
        assert_eq!(lost.sequence.get(), 3);
        assert_eq!(publisher.device_lost().unwrap(), lost);
        assert_eq!(
            publisher.exhausted().unwrap_err(),
            ContractError::CompletionHealthRegression {
                current: ProviderHealth::DeviceLost,
                requested: ProviderHealth::Exhausted,
            }
        );
    }

    #[test]
    fn device_loss_refuses_new_tokens_but_replays_terminal_updates() {
        let mut publisher = publisher();
        publisher.submitted(token(1)).unwrap();
        let completed = publisher.completed_visible(token(1)).unwrap();
        publisher.device_lost().unwrap();
        assert_eq!(
            publisher.submitted(token(2)).unwrap_err(),
            ContractError::CompletionPublishAfterDeviceLost(token(2))
        );
        assert_eq!(publisher.completed_visible(token(1)).unwrap(), completed);
    }

    #[test]
    fn publisher_and_mirror_round_trip() {
        let mut publisher = publisher();
        let mut transport = LoopbackTransport::new();
        let mut mirror = mirror();
        transport.send(publisher.submitted(token(1)).unwrap());
        transport.send(publisher.submitted(token(1)).unwrap());
        transport.send(publisher.completed_visible(token(1)).unwrap());
        transport.send(publisher.completed_visible(token(1)).unwrap());
        transport.send(publisher.exhausted().unwrap());
        transport.send(publisher.submitted(token(2)).unwrap());
        transport.send(publisher.completed_visible(token(2)).unwrap());
        transport.send(publisher.device_lost().unwrap());
        assert_eq!(transport.len(), 8);
        assert!(transport.deliver_all(&mut mirror).unwrap() > 0);
        assert_eq!(mirror.health(), ProviderHealth::DeviceLost);
        assert_eq!(
            mirror.observation(token(1)),
            Some(CompletionDisposition::CompletedVisible { token: token(1) })
        );
        assert_eq!(
            mirror.observation(token(2)),
            Some(CompletionDisposition::CompletedVisible { token: token(2) })
        );
        assert_eq!(mirror.outstanding(), 0);
        assert_eq!(mirror.device_sequence(), publisher.device_sequence());
        assert_eq!(
            mirror.token_sequence(token(1)),
            publisher.token_sequence(token(1))
        );
    }

    #[test]
    fn publisher_and_mirror_converge_under_reordering() {
        let mut publisher = publisher();
        let mut transport = LoopbackTransport::new();
        let mut mirror = mirror();
        transport.send(publisher.submitted(token(1)).unwrap());
        transport.send(publisher.submitted(token(1)).unwrap());
        transport.send(publisher.completed_visible(token(1)).unwrap());
        assert!(transport.deliver_reversed(&mut mirror).unwrap() > 0);
        assert_eq!(
            mirror.observation(token(1)),
            Some(CompletionDisposition::CompletedVisible { token: token(1) })
        );
        assert_eq!(
            mirror.token_sequence(token(1)),
            publisher.token_sequence(token(1))
        );
    }

    #[test]
    fn publisher_rejects_invalid_identities_and_epochs() {
        assert!(CompletionPublisher::new(DeviceEpoch::new(0)).is_err());
        let mut publisher = publisher();
        let zero_submission = CompletionToken {
            device_epoch: epoch(),
            submission_id: SubmissionId::new(0),
        };
        assert_eq!(
            publisher.submitted(zero_submission).unwrap_err(),
            ContractError::InvalidIdentity("submission id")
        );
        let foreign = CompletionToken {
            device_epoch: DeviceEpoch::new(8),
            submission_id: SubmissionId::new(1),
        };
        assert_eq!(
            publisher.submitted(foreign).unwrap_err(),
            ContractError::CompletionEpochMismatch {
                expected: epoch(),
                actual: DeviceEpoch::new(8),
            }
        );
    }

    #[test]
    fn publisher_builds_failure_summary_from_provider_error() {
        let error = ProviderError::new(
            ProviderPhase::Wait,
            ProviderErrorClass::Execute,
            "synthetic_failure",
        )
        .unwrap();
        let mut publisher = publisher();
        let message = publisher
            .failed_from_provider_error(token(1), &error)
            .unwrap();
        assert_eq!(
            message.update,
            CompletionUpdate::Failed(CompletionFailure::from_provider_error(&error))
        );
    }

    #[test]
    fn publisher_and_mirror_cross_a_thread_boundary() {
        let (sender, receiver) = std::sync::mpsc::channel::<CompletionMessage>();
        let handle = std::thread::spawn(move || {
            let mut publisher = publisher();
            sender
                .send(publisher.submitted(token(1)).unwrap().into())
                .unwrap();
            sender
                .send(publisher.completed_visible(token(1)).unwrap().into())
                .unwrap();
            sender
                .send(publisher.device_lost().unwrap().into())
                .unwrap();
        });
        let mut mirror = mirror();
        for message in receiver {
            mirror.apply(message).unwrap();
        }
        handle.join().unwrap();
        assert_eq!(mirror.health(), ProviderHealth::DeviceLost);
        assert_eq!(
            mirror.observation(token(1)),
            Some(CompletionDisposition::CompletedVisible { token: token(1) })
        );
    }
}
