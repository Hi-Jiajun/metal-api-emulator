//! End-to-end provider outbox -> writer thread -> socket -> owner receiver.

use metal_api_core::completion::wire::{CompletionOutbox, MirrorOutcome};
use metal_api_core::completion::CompletionRecord;
use metal_api_core::provider::{
    AllocationId, BufferLease, CompletionDisposition, CompletionToken, DeviceEpoch, LeaseId,
    LeaseLedger, LeaseObservation, LeaseReservation, ProviderHealth, SubmissionId,
};
use metal_api_ipc::receiver::CompletionReceiver;
use metal_api_ipc::sender::spawn_writer;
use metal_api_ipc::unix;
use std::sync::Arc;

fn token() -> CompletionToken {
    CompletionToken {
        device_epoch: DeviceEpoch::new(7),
        submission_id: SubmissionId::new(42),
    }
}

fn lease_ledger() -> LeaseLedger {
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
    ledger
}

#[test]
fn outbox_writer_and_receiver_retire_a_lease() {
    let (owner_transport, provider_transport) = unix::pair().unwrap();
    let (sender, writer) = spawn_writer(provider_transport).unwrap();
    let mut receiver = CompletionReceiver::new(owner_transport, DeviceEpoch::new(7)).unwrap();

    let outbox = Arc::new(CompletionOutbox::new(DeviceEpoch::new(7), Arc::new(sender)).unwrap());
    outbox.submitted(token()).unwrap();
    let record = CompletionRecord::running_with_observer(token(), outbox.observer());
    record.complete(Vec::new());
    drop(record);
    drop(outbox);
    writer.join().unwrap().unwrap();

    assert_eq!(receiver.recv().unwrap(), MirrorOutcome::Applied);
    assert_eq!(receiver.recv().unwrap(), MirrorOutcome::Applied);
    assert_eq!(
        receiver.mirror().observation(token()),
        Some(CompletionDisposition::CompletedVisible { token: token() })
    );

    let mut ledger = lease_ledger();
    assert_eq!(ledger.outstanding(LeaseId::new(3)), Some(1));
    assert_eq!(
        receiver.observe_into(&mut ledger, token()).unwrap(),
        LeaseObservation::Retired
    );
    assert!(ledger.release_ready(LeaseId::new(3)));
    assert!(ledger.release(LeaseId::new(3)).is_some());
}

#[test]
fn device_loss_written_by_the_sender_releases_the_ledger() {
    let (owner_transport, provider_transport) = unix::pair().unwrap();
    let (sender, writer) = spawn_writer(provider_transport).unwrap();
    let mut receiver = CompletionReceiver::new(owner_transport, DeviceEpoch::new(7)).unwrap();

    let outbox = Arc::new(CompletionOutbox::new(DeviceEpoch::new(7), Arc::new(sender)).unwrap());
    outbox.publish_device(ProviderHealth::DeviceLost).unwrap();
    drop(outbox);
    writer.join().unwrap().unwrap();

    assert_eq!(receiver.recv().unwrap(), MirrorOutcome::Applied);
    assert_eq!(receiver.health(), ProviderHealth::DeviceLost);
    let mut ledger = lease_ledger();
    assert_eq!(
        receiver.observe_into(&mut ledger, token()).unwrap(),
        LeaseObservation::Retired
    );
    assert!(ledger.is_device_lost());
    assert!(ledger.release_ready(LeaseId::new(3)));
}
