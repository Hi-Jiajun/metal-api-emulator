//! Owner-side receiver coverage: publisher -> wire -> mirror -> lease ledger.

use metal_api_core::completion::wire::{CompletionMessage, CompletionPublisher, MirrorOutcome};
use metal_api_core::provider::{
    AllocationId, BufferLease, CompletionDisposition, CompletionToken, ContractError, DeviceEpoch,
    LeaseId, LeaseLedger, LeaseObservation, LeaseReservation, ProviderHealth, SubmissionId,
};
use metal_api_ipc::receiver::{CompletionReceiver, EndpointError};
use metal_api_ipc::transport::CompletionTransport;
use metal_api_ipc::unix;
use std::process::{Command, Stdio};

fn token(id: u64) -> CompletionToken {
    CompletionToken {
        device_epoch: DeviceEpoch::new(7),
        submission_id: SubmissionId::new(id),
    }
}

fn register_lease(ledger: &mut LeaseLedger, lease_id: u64, tokens: &[CompletionToken]) {
    let lease_id = LeaseId::new(lease_id);
    ledger
        .register(LeaseReservation {
            lease: BufferLease {
                lease_id,
                allocation_id: AllocationId::new(lease_id.get() + 100),
                owner_epoch: DeviceEpoch::new(7),
            },
            offset: 0,
            length: 64,
        })
        .unwrap();
    for token in tokens {
        ledger.bind(lease_id, *token).unwrap();
    }
}

#[test]
fn receiver_converges_publisher_messages_and_retires_a_lease() {
    let (owner_transport, mut provider_transport) = unix::pair().unwrap();
    let mut receiver = CompletionReceiver::new(owner_transport, DeviceEpoch::new(7)).unwrap();
    let mut publisher = CompletionPublisher::new(DeviceEpoch::new(7)).unwrap();

    let message: CompletionMessage = publisher.submitted(token(42)).unwrap().into();
    provider_transport.send(&message).unwrap();
    provider_transport.flush().unwrap();
    assert_eq!(receiver.recv().unwrap(), MirrorOutcome::Applied);

    let message: CompletionMessage = publisher.completed_visible(token(42)).unwrap().into();
    provider_transport.send(&message).unwrap();
    provider_transport.flush().unwrap();
    assert_eq!(receiver.recv().unwrap(), MirrorOutcome::Applied);

    let mut ledger = LeaseLedger::new();
    register_lease(&mut ledger, 1, &[token(42)]);
    assert_eq!(ledger.outstanding(LeaseId::new(1)), Some(1));
    assert_eq!(
        receiver.mirror().observation(token(42)),
        Some(CompletionDisposition::CompletedVisible { token: token(42) })
    );
    assert_eq!(
        receiver.observe_into(&mut ledger, token(42)).unwrap(),
        LeaseObservation::Retired
    );
    assert_eq!(ledger.outstanding(LeaseId::new(1)), Some(0));
    assert!(ledger.release_ready(LeaseId::new(1)));
    assert_eq!(receiver.applied(), 2);
    assert_eq!(receiver.ignored(), 0);
}

#[test]
fn replayed_terminal_is_ignored_over_the_wire() {
    let (owner_transport, mut provider_transport) = unix::pair().unwrap();
    let mut receiver = CompletionReceiver::new(owner_transport, DeviceEpoch::new(7)).unwrap();
    let mut publisher = CompletionPublisher::new(DeviceEpoch::new(7)).unwrap();

    let first: CompletionMessage = publisher.completed_visible(token(42)).unwrap().into();
    let replay: CompletionMessage = publisher.completed_visible(token(42)).unwrap().into();
    assert_eq!(first, replay);

    for message in [&first, &replay] {
        provider_transport.send(message).unwrap();
        provider_transport.flush().unwrap();
    }
    assert_eq!(receiver.recv().unwrap(), MirrorOutcome::Applied);
    assert_eq!(receiver.recv().unwrap(), MirrorOutcome::Ignored);
    assert_eq!(receiver.applied(), 1);
    assert_eq!(receiver.ignored(), 1);
}

#[test]
fn device_lost_releases_every_bound_lease() {
    let (owner_transport, mut provider_transport) = unix::pair().unwrap();
    let mut receiver = CompletionReceiver::new(owner_transport, DeviceEpoch::new(7)).unwrap();
    let mut publisher = CompletionPublisher::new(DeviceEpoch::new(7)).unwrap();

    let mut ledger = LeaseLedger::new();
    register_lease(&mut ledger, 1, &[token(42), token(43)]);

    let message: CompletionMessage = publisher.device_lost().unwrap().into();
    provider_transport.send(&message).unwrap();
    provider_transport.flush().unwrap();
    assert_eq!(receiver.recv().unwrap(), MirrorOutcome::Applied);
    assert_eq!(receiver.health(), ProviderHealth::DeviceLost);
    assert_eq!(
        receiver.observe_into(&mut ledger, token(42)).unwrap(),
        LeaseObservation::Retired
    );
    assert!(ledger.is_device_lost());
    assert!(ledger.release_ready(LeaseId::new(1)));
    assert!(ledger.release(LeaseId::new(1)).is_some());
}

#[test]
fn rejects_a_message_from_another_device_epoch_over_the_wire() {
    let (owner_transport, mut provider_transport) = unix::pair().unwrap();
    let mut receiver = CompletionReceiver::new(owner_transport, DeviceEpoch::new(8)).unwrap();
    let mut publisher = CompletionPublisher::new(DeviceEpoch::new(7)).unwrap();

    let message: CompletionMessage = publisher.submitted(token(42)).unwrap().into();
    provider_transport.send(&message).unwrap();
    provider_transport.flush().unwrap();
    assert!(matches!(
        receiver.recv().unwrap_err(),
        EndpointError::Contract(ContractError::CompletionEpochMismatch { .. })
    ));
}

#[test]
fn receiver_round_trips_across_a_child_process() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_completion_echo"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn completion_echo helper");
    let stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let transport = CompletionTransport::new(stdout, stdin);
    let mut receiver = CompletionReceiver::new(transport, DeviceEpoch::new(7)).unwrap();
    let mut publisher = CompletionPublisher::new(DeviceEpoch::new(7)).unwrap();

    for update in [
        publisher.submitted(token(42)).unwrap(),
        publisher.completed_visible(token(42)).unwrap(),
    ] {
        let message: CompletionMessage = update.into();
        receiver.transport_mut().send(&message).unwrap();
        receiver.transport_mut().flush().unwrap();
        assert_eq!(receiver.recv().unwrap(), MirrorOutcome::Applied);
    }

    let (transport, mirror) = receiver.into_parts();
    drop(transport);
    assert!(child.wait().unwrap().success());
    assert_eq!(
        mirror.observation(token(42)),
        Some(CompletionDisposition::CompletedVisible { token: token(42) })
    );
}
