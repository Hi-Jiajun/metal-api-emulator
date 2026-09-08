//! Owner-side receiver for the completion stream.
//!
//! [`CompletionReceiver`] pairs a [`CompletionTransport`] with the
//! transport-independent [`CompletionMirror`] from `metal-api-core`. It reads
//! one frame at a time, applies the decoded message to the mirror and exposes
//! the converged observation so an owner can retire leases or decide the next
//! wait. It performs no GPU work and never blocks on anything but the
//! underlying stream.

use crate::codec::CodecError;
use crate::transport::CompletionTransport;
use metal_api_core::completion::wire::{CompletionMessage, CompletionMirror, MirrorOutcome};
use metal_api_core::provider::{
    CompletionToken, ContractError, DeviceEpoch, LeaseLedger, LeaseObservation, ProviderHealth,
};
use std::fmt;
use std::io::{Read, Write};

/// Failure while receiving a completion notification.
///
/// A [`CodecError`] means the byte stream is unusable: framing is broken and
/// the connection should be re-established. A [`ContractError`] means a
/// well-framed message contradicted the mirror, for example a different
/// device epoch.
#[derive(Debug)]
pub enum EndpointError {
    Codec(CodecError),
    Contract(ContractError),
}

impl fmt::Display for EndpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(error) => write!(formatter, "completion receive failed: {error}"),
            Self::Contract(error) => {
                write!(
                    formatter,
                    "completion message rejected by the mirror: {error}"
                )
            }
        }
    }
}

impl std::error::Error for EndpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Codec(error) => Some(error),
            Self::Contract(error) => Some(error),
        }
    }
}

impl From<CodecError> for EndpointError {
    fn from(error: CodecError) -> Self {
        Self::Codec(error)
    }
}

impl From<ContractError> for EndpointError {
    fn from(error: ContractError) -> Self {
        Self::Contract(error)
    }
}

/// Owner-side mirror fed by a completion transport.
///
/// The receiver owns the transport so a dedicated thread can run
/// [`CompletionReceiver::recv`] in a loop while the owner observes the mirror
/// through a shared handle.
#[derive(Debug)]
pub struct CompletionReceiver<R, W> {
    transport: CompletionTransport<R, W>,
    mirror: CompletionMirror,
    applied: u64,
    ignored: u64,
}

impl<R, W> CompletionReceiver<R, W> {
    /// Wrap a transport and scope the mirror to the provider device epoch.
    pub fn new(
        transport: CompletionTransport<R, W>,
        device_epoch: DeviceEpoch,
    ) -> Result<Self, ContractError> {
        Ok(Self {
            transport,
            mirror: CompletionMirror::new(device_epoch)?,
            applied: 0,
            ignored: 0,
        })
    }

    /// Borrow the converged mirror.
    pub const fn mirror(&self) -> &CompletionMirror {
        &self.mirror
    }

    /// Borrow the underlying transport, for example to send a cancellation
    /// request or to set a read timeout.
    pub const fn transport(&self) -> &CompletionTransport<R, W> {
        &self.transport
    }

    /// Mutably borrow the underlying transport.
    pub fn transport_mut(&mut self) -> &mut CompletionTransport<R, W> {
        &mut self.transport
    }

    /// Number of messages that advanced the mirror.
    pub const fn applied(&self) -> u64 {
        self.applied
    }

    /// Number of duplicate or stale messages the mirror ignored.
    pub const fn ignored(&self) -> u64 {
        self.ignored
    }

    /// Current device health as converged by the mirror.
    pub const fn health(&self) -> ProviderHealth {
        self.mirror.health()
    }

    /// Apply the mirror's observation of `token` to a lease ledger.
    pub fn observe_into(
        &self,
        ledger: &mut LeaseLedger,
        token: CompletionToken,
    ) -> Result<LeaseObservation, ContractError> {
        self.mirror.observe_into(ledger, token)
    }

    /// Recover the transport and the mirror.
    pub fn into_parts(self) -> (CompletionTransport<R, W>, CompletionMirror) {
        (self.transport, self.mirror)
    }
}

impl<R: Read, W: Write> CompletionReceiver<R, W> {
    /// Read one frame, apply it and report how the mirror changed.
    ///
    /// A duplicate or stale notification returns [`MirrorOutcome::Ignored`]
    /// and is counted separately. A clean end of stream returns
    /// [`EndpointError::Codec`] wrapping [`CodecError::Eof`].
    pub fn recv(&mut self) -> Result<MirrorOutcome, EndpointError> {
        let message: CompletionMessage = self.transport.recv()?;
        let outcome = self.mirror.apply(message)?;
        match outcome {
            MirrorOutcome::Applied => self.applied += 1,
            MirrorOutcome::Ignored => self.ignored += 1,
        }
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CompletionCodec;
    use metal_api_core::completion::wire::{
        CompletionSequence, CompletionTokenUpdate, CompletionUpdate,
    };
    use metal_api_core::provider::{
        AllocationId, BufferLease, LeaseId, LeaseReservation, SubmissionId,
    };
    use std::io::Cursor;

    fn token() -> CompletionToken {
        CompletionToken {
            device_epoch: DeviceEpoch::new(7),
            submission_id: SubmissionId::new(42),
        }
    }

    fn submitted() -> CompletionMessage {
        CompletionMessage::Token(CompletionTokenUpdate {
            token: token(),
            sequence: CompletionSequence::new(1),
            update: CompletionUpdate::Submitted,
        })
    }

    #[test]
    fn applies_messages_and_counts_duplicates() {
        let frame = CompletionCodec::encode(&submitted()).unwrap();
        let mut receiver = CompletionReceiver::new(
            CompletionTransport::new(Cursor::new(frame.clone()), Cursor::new(Vec::new())),
            DeviceEpoch::new(7),
        )
        .unwrap();
        assert_eq!(receiver.recv().unwrap(), MirrorOutcome::Applied);
        assert_eq!(receiver.applied(), 1);
        assert_eq!(receiver.ignored(), 0);

        let mut duplicate = CompletionReceiver::new(
            CompletionTransport::new(
                Cursor::new([frame.clone(), frame].concat()),
                Cursor::new(Vec::new()),
            ),
            DeviceEpoch::new(7),
        )
        .unwrap();
        assert_eq!(duplicate.recv().unwrap(), MirrorOutcome::Applied);
        assert_eq!(duplicate.recv().unwrap(), MirrorOutcome::Ignored);
        assert_eq!(duplicate.applied(), 1);
        assert_eq!(duplicate.ignored(), 1);
    }

    #[test]
    fn rejects_a_message_from_another_device_epoch() {
        let frame = CompletionCodec::encode(&submitted()).unwrap();
        let mut receiver = CompletionReceiver::new(
            CompletionTransport::new(Cursor::new(frame), Cursor::new(Vec::new())),
            DeviceEpoch::new(8),
        )
        .unwrap();
        assert!(matches!(
            receiver.recv().unwrap_err(),
            EndpointError::Contract(ContractError::CompletionEpochMismatch { .. })
        ));
    }

    #[test]
    fn observes_into_a_lease_ledger() {
        let completed = CompletionMessage::Token(CompletionTokenUpdate {
            token: token(),
            sequence: CompletionSequence::new(2),
            update: CompletionUpdate::CompletedVisible,
        });
        let frame = CompletionCodec::encode(&completed).unwrap();
        let mut receiver = CompletionReceiver::new(
            CompletionTransport::new(Cursor::new(frame), Cursor::new(Vec::new())),
            DeviceEpoch::new(7),
        )
        .unwrap();
        let lease_id = LeaseId::new(3);
        let mut ledger = LeaseLedger::new();
        ledger
            .register(LeaseReservation {
                lease: BufferLease {
                    lease_id,
                    allocation_id: AllocationId::new(9),
                    owner_epoch: DeviceEpoch::new(7),
                },
                offset: 0,
                length: 64,
            })
            .unwrap();
        ledger.bind(lease_id, token()).unwrap();
        assert_eq!(ledger.outstanding(lease_id), Some(1));
        assert_eq!(receiver.recv().unwrap(), MirrorOutcome::Applied);
        assert_eq!(receiver.health(), ProviderHealth::Usable);
        assert_eq!(
            receiver.observe_into(&mut ledger, token()).unwrap(),
            LeaseObservation::Retired
        );
        assert_eq!(ledger.outstanding(lease_id), Some(0));
        assert!(ledger.release_ready(lease_id));
    }
}
