//! Cross-process and socket-pair coverage for the completion transport.

use metal_api_core::completion::wire::{
    CompletionDeviceUpdate, CompletionFailure, CompletionMessage, CompletionSequence,
    CompletionTokenUpdate, CompletionUpdate,
};
use metal_api_core::provider::{
    CompletionToken, DeviceEpoch, ProviderErrorClass, ProviderHealth, ProviderPhase, SubmissionId,
};
use metal_api_ipc::codec::CodecError;
use metal_api_ipc::transport::CompletionTransport;
use metal_api_ipc::unix;
use std::process::{Command, Stdio};

fn token() -> CompletionToken {
    CompletionToken {
        device_epoch: DeviceEpoch::new(7),
        submission_id: SubmissionId::new(42),
    }
}

fn submitted(sequence: u64) -> CompletionMessage {
    CompletionMessage::Token(CompletionTokenUpdate {
        token: token(),
        sequence: CompletionSequence::new(sequence),
        update: CompletionUpdate::Submitted,
    })
}

fn failed(sequence: u64, slug: &str) -> CompletionMessage {
    CompletionMessage::Token(CompletionTokenUpdate {
        token: token(),
        sequence: CompletionSequence::new(sequence),
        update: CompletionUpdate::Failed(
            CompletionFailure::new(ProviderPhase::Wait, ProviderErrorClass::Resource, slug)
                .unwrap(),
        ),
    })
}

fn device_health(sequence: u64) -> CompletionMessage {
    CompletionMessage::Device(CompletionDeviceUpdate {
        device_epoch: DeviceEpoch::new(7),
        sequence: CompletionSequence::new(sequence),
        health: ProviderHealth::Exhausted,
    })
}

fn sample_messages() -> Vec<CompletionMessage> {
    vec![
        submitted(1),
        failed(2, "wait queue full"),
        device_health(1),
        CompletionMessage::Token(CompletionTokenUpdate {
            token: token(),
            sequence: CompletionSequence::new(3),
            update: CompletionUpdate::CompletedVisible,
        }),
    ]
}

#[test]
fn round_trips_over_a_unix_socket_pair() {
    let (mut owner, mut provider) = unix::pair().unwrap();
    let messages = sample_messages();
    let expected = messages.clone();
    let count = messages.len();

    let provider_thread = std::thread::spawn(move || {
        let mut echoed = Vec::new();
        for _ in 0..count {
            let message = provider.recv().unwrap();
            provider.send(&message).unwrap();
            provider.flush().unwrap();
            echoed.push(message);
        }
        echoed
    });

    for message in &messages {
        owner.send(message).unwrap();
    }
    owner.flush().unwrap();

    for message in &messages {
        assert_eq!(owner.recv().unwrap(), *message);
    }
    assert_eq!(owner.sent(), messages.len() as u64);
    assert_eq!(owner.received(), messages.len() as u64);
    assert_eq!(provider_thread.join().unwrap(), expected);
}

#[test]
fn round_trips_across_processes() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_completion_echo"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn completion_echo helper");
    let stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut transport = CompletionTransport::new(stdout, stdin);

    let messages = sample_messages();
    for message in &messages {
        transport.send(message).unwrap();
    }
    transport.flush().unwrap();

    let (reader, writer) = transport.into_inner();
    drop(writer);
    let mut receiver = CompletionTransport::new(reader, std::io::sink());
    for message in &messages {
        assert_eq!(receiver.recv().unwrap(), *message);
    }
    assert!(matches!(receiver.recv().unwrap_err(), CodecError::Eof));
    assert!(child.wait().unwrap().success());
}

#[test]
fn listener_accepts_a_client_and_reports_clean_eof() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "metal-api-ipc-{}-{unique}.sock",
        std::process::id()
    ));
    let listener = unix::UnixListenerTransport::bind(&path).unwrap();
    let message = submitted(1);

    let server_thread = std::thread::spawn(move || {
        let mut transport = listener.accept().unwrap();
        let received = transport.recv().unwrap();
        transport.send(&received).unwrap();
        transport.flush().unwrap();
        assert!(matches!(transport.recv().unwrap_err(), CodecError::Eof));
        received
    });

    let mut client = unix::connect(&path).unwrap();
    client.send(&message).unwrap();
    client.flush().unwrap();
    assert_eq!(client.recv().unwrap(), message);
    drop(client);
    assert_eq!(server_thread.join().unwrap(), message);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn reports_eof_and_rejects_a_bad_magic_from_a_stream() {
    let mut empty = CompletionTransport::new(std::io::empty(), std::io::sink());
    assert!(matches!(empty.recv().unwrap_err(), CodecError::Eof));

    let mut bad = CompletionTransport::new(
        std::io::Cursor::new(b"XXXX\x00\x00\x00\x01".to_vec()),
        std::io::sink(),
    );
    assert!(matches!(bad.recv().unwrap_err(), CodecError::BadMagic(_)));
}
