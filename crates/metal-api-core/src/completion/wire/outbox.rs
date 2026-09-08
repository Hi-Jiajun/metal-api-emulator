//! Provider-side outbox binding a completion publisher to a sink.
//!
//! A provider owns GPU state and transitions an in-process
//! [`CompletionRecord`](crate::completion::CompletionRecord). The
//! [`CompletionOutbox`] mirrors those transitions onto the wire: the provider
//! calls [`CompletionOutbox::submitted`] at admission, installs the outbox as
//! the record's [`CompletionObserver`] so the first terminal transition is
//! published automatically, and calls [`CompletionOutbox::publish_device`]
//! when device health changes. It performs no I/O itself; the
//! [`CompletionSink`] chooses the transport.

use super::{CompletionFailure, CompletionMessage, CompletionPublisher, CompletionUpdate};
use crate::completion::{CompletionObserver, CompletionTerminal};
use crate::provider::{CompletionToken, ContractError, DeviceEpoch, ProviderHealth};
use std::sync::{Arc, Mutex};

/// Destination for published completion notifications.
///
/// The provider calls this from submit, completion-handler and cancellation
/// paths. An implementation must not block the provider for long: a socket or
/// pipe transport should queue the message and let a dedicated writer drain it.
/// Terminal notifications are idempotent in the publisher, so a transport may
/// retry a failed delivery.
pub trait CompletionSink: Send + Sync {
    /// Deliver one message.
    fn deliver(&self, message: CompletionMessage);
}

/// Provider-side publisher bound to a [`CompletionSink`].
///
/// The publisher state is authoritative and lives here, not in the sink, so a
/// rejected transition is reported to the caller and recorded in
/// [`CompletionOutbox::last_error`]. The observer path is best-effort: the
/// provider must not fail GPU work because a notification could not be
/// published.
pub struct CompletionOutbox {
    publisher: Mutex<CompletionPublisher>,
    sink: Arc<dyn CompletionSink>,
    last_error: Mutex<Option<ContractError>>,
}

impl CompletionOutbox {
    /// Create an outbox scoped to the provider device epoch.
    pub fn new(
        device_epoch: DeviceEpoch,
        sink: Arc<dyn CompletionSink>,
    ) -> Result<Self, ContractError> {
        Ok(Self {
            publisher: Mutex::new(CompletionPublisher::new(device_epoch)?),
            sink,
            last_error: Mutex::new(None),
        })
    }

    /// Device epoch every published message carries.
    pub fn device_epoch(&self) -> DeviceEpoch {
        self.lock_publisher().device_epoch()
    }

    /// Last health level published or observed.
    pub fn health(&self) -> ProviderHealth {
        self.lock_publisher().health()
    }

    /// Most recent publication rejection, if any.
    pub fn last_error(&self) -> Option<ContractError> {
        self.last_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Publish one token update and deliver it to the sink.
    pub fn publish(
        &self,
        token: CompletionToken,
        update: CompletionUpdate,
    ) -> Result<(), ContractError> {
        let message = match self.lock_publisher().publish(token, update) {
            Ok(update) => CompletionMessage::from(update),
            Err(error) => {
                self.record_error(error.clone());
                return Err(error);
            }
        };
        self.sink.deliver(message);
        Ok(())
    }

    /// Publish admission of `token`.
    pub fn submitted(&self, token: CompletionToken) -> Result<(), ContractError> {
        self.publish(token, CompletionUpdate::Submitted)
    }

    /// Publish a device health transition.
    pub fn publish_device(&self, health: ProviderHealth) -> Result<(), ContractError> {
        let message = match self.lock_publisher().publish_device(health) {
            Ok(update) => CompletionMessage::from(update),
            Err(error) => {
                self.record_error(error.clone());
                return Err(error);
            }
        };
        self.sink.deliver(message);
        Ok(())
    }

    /// Observer handle to install on a completion record.
    ///
    /// The returned handle shares this outbox, so admission and terminal
    /// transitions use one sequence space.
    pub fn observer(self: &Arc<Self>) -> Arc<dyn CompletionObserver> {
        let outbox: Arc<Self> = Arc::clone(self);
        let observer: Arc<dyn CompletionObserver> = outbox;
        observer
    }

    fn lock_publisher(&self) -> std::sync::MutexGuard<'_, CompletionPublisher> {
        self.publisher
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn record_error(&self, error: ContractError) {
        let mut slot = self
            .last_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *slot = Some(error);
    }
}

impl CompletionObserver for CompletionOutbox {
    fn observe_terminal(&self, token: CompletionToken, terminal: CompletionTerminal) {
        let update = match terminal {
            CompletionTerminal::CompletedVisible => CompletionUpdate::CompletedVisible,
            CompletionTerminal::Cancelled => CompletionUpdate::Cancelled,
            CompletionTerminal::SubmittedUnknown => CompletionUpdate::SubmittedUnknown,
            CompletionTerminal::Failed(error) => {
                CompletionUpdate::Failed(CompletionFailure::from_provider_error(&error))
            }
        };
        let _ = self.publish(token, update);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::wire::CompletionTokenUpdate;
    use crate::completion::CompletionRecord;
    use crate::provider::{
        CompletionDisposition, ProviderError, ProviderErrorClass, ProviderPhase, SubmissionId,
    };

    #[derive(Default)]
    struct RecordingSink {
        messages: Mutex<Vec<CompletionMessage>>,
    }

    impl CompletionSink for RecordingSink {
        fn deliver(&self, message: CompletionMessage) {
            self.messages.lock().expect("recording sink").push(message);
        }
    }

    fn token() -> CompletionToken {
        CompletionToken {
            device_epoch: DeviceEpoch::new(7),
            submission_id: SubmissionId::new(42),
        }
    }

    fn messages(sink: &Arc<RecordingSink>) -> Vec<CompletionMessage> {
        sink.messages.lock().expect("recording sink").clone()
    }

    fn token_update(message: &CompletionMessage) -> &CompletionTokenUpdate {
        match message {
            CompletionMessage::Token(update) => update,
            CompletionMessage::Device(update) => panic!("expected a token update, got {update:?}"),
        }
    }

    #[test]
    fn publishes_admission_then_the_first_terminal_transition() {
        let sink = Arc::new(RecordingSink::default());
        let outbox = Arc::new(CompletionOutbox::new(DeviceEpoch::new(7), sink.clone()).unwrap());
        outbox.submitted(token()).unwrap();

        let record = CompletionRecord::running_with_observer(token(), outbox.observer());
        record.complete(Vec::new());
        record.fail(
            ProviderError::new(ProviderPhase::Wait, ProviderErrorClass::Execute, "late").unwrap(),
        );
        record.cancel();

        let messages = messages(&sink);
        assert_eq!(messages.len(), 2);
        assert_eq!(token_update(&messages[0]).sequence.get(), 1);
        assert_eq!(
            token_update(&messages[0]).update,
            CompletionUpdate::Submitted
        );
        assert_eq!(token_update(&messages[1]).sequence.get(), 2);
        assert_eq!(
            token_update(&messages[1]).update,
            CompletionUpdate::CompletedVisible
        );
    }

    #[test]
    fn maps_a_failure_to_the_wire_summary() {
        let sink = Arc::new(RecordingSink::default());
        let outbox = Arc::new(CompletionOutbox::new(DeviceEpoch::new(7), sink.clone()).unwrap());
        let mut error = ProviderError::new(
            ProviderPhase::Wait,
            ProviderErrorClass::Execute,
            "metal_command_failed",
        )
        .unwrap();
        error.completion = CompletionDisposition::Failed {
            token: Some(token()),
        };

        let record = CompletionRecord::running_with_observer(token(), outbox.observer());
        record.fail(error);

        let messages = messages(&sink);
        assert_eq!(messages.len(), 1);
        match &token_update(&messages[0]).update {
            CompletionUpdate::Failed(failure) => {
                assert_eq!(failure.slug, "metal_command_failed");
                assert_eq!(failure.phase, ProviderPhase::Wait);
                assert_eq!(failure.class, ProviderErrorClass::Execute);
            }
            other => panic!("expected a failure update, got {other:?}"),
        }
    }

    #[test]
    fn maps_unknown_completion_and_cancellation() {
        let sink = Arc::new(RecordingSink::default());
        let outbox = Arc::new(CompletionOutbox::new(DeviceEpoch::new(7), sink.clone()).unwrap());

        let mut unknown = ProviderError::new(
            ProviderPhase::Wait,
            ProviderErrorClass::Execute,
            "completion_unknown",
        )
        .unwrap();
        unknown.completion = CompletionDisposition::SubmittedUnknown {
            token: Some(token()),
        };
        CompletionRecord::running_with_observer(token(), outbox.observer()).fail(unknown);

        let other = CompletionToken {
            device_epoch: DeviceEpoch::new(7),
            submission_id: SubmissionId::new(43),
        };
        CompletionRecord::running_with_observer(other, outbox.observer()).cancel();

        let messages = messages(&sink);
        assert_eq!(messages.len(), 2);
        assert_eq!(
            token_update(&messages[0]).update,
            CompletionUpdate::SubmittedUnknown
        );
        assert_eq!(
            token_update(&messages[1]).update,
            CompletionUpdate::Cancelled
        );
    }

    #[test]
    fn publishes_device_health_and_records_rejections() {
        let sink = Arc::new(RecordingSink::default());
        let outbox = Arc::new(CompletionOutbox::new(DeviceEpoch::new(7), sink.clone()).unwrap());
        outbox.publish_device(ProviderHealth::Exhausted).unwrap();
        outbox.publish_device(ProviderHealth::DeviceLost).unwrap();

        let messages = messages(&sink);
        assert_eq!(messages.len(), 2);
        match &messages[0] {
            CompletionMessage::Device(update) => {
                assert_eq!(update.health, ProviderHealth::Exhausted);
                assert_eq!(update.sequence.get(), 1);
            }
            other => panic!("expected a device update, got {other:?}"),
        }
        assert_eq!(outbox.health(), ProviderHealth::DeviceLost);

        let second = Arc::new(CompletionOutbox::new(DeviceEpoch::new(7), sink).unwrap());
        second.submitted(token()).unwrap();
        second
            .publish(token(), CompletionUpdate::CompletedVisible)
            .unwrap();
        let error = second
            .publish(token(), CompletionUpdate::Submitted)
            .unwrap_err();
        assert!(matches!(
            error,
            ContractError::CompletionPublishAfterTerminal(_)
        ));
        assert!(second.last_error().is_some());
    }

    #[test]
    fn device_loss_refuses_a_later_admission() {
        let sink = Arc::new(RecordingSink::default());
        let outbox = Arc::new(CompletionOutbox::new(DeviceEpoch::new(7), sink.clone()).unwrap());
        outbox.publish_device(ProviderHealth::DeviceLost).unwrap();
        let error = outbox.submitted(token()).unwrap_err();
        assert!(matches!(
            error,
            ContractError::CompletionPublishAfterDeviceLost(_)
        ));
    }
}
