//! Provider-side sink that writes completion messages on a background thread.
//!
//! A provider must not block a completion handler or a submit path on a
//! socket write. [`spawn_writer`] moves the transport onto a dedicated thread
//! and returns a [`CompletionSender`], which implements the transport-agnostic
//! `metal_api_core::completion::wire::CompletionSink`. The provider outbox
//! calls `deliver` and returns immediately; the writer drains the queue in
//! order and flushes after every frame.
//!
//! If the transport fails, the writer stops and returns the codec error. The
//! owner observes end of stream, and the provider's publisher state can replay
//! terminal notifications on a new connection.

use crate::codec::CodecError;
use crate::transport::CompletionTransport;
use metal_api_core::completion::wire::{CompletionMessage, CompletionSink};
use std::io::{Read, Write};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};

/// Cloneable sink that queues messages for a [`CompletionWriter`].
///
/// `deliver` never blocks on I/O. If the writer has already stopped, the
/// message is dropped; the provider keeps its authoritative publisher state.
#[derive(Clone, Debug)]
pub struct CompletionSender {
    sender: Sender<CompletionMessage>,
}

impl CompletionSink for CompletionSender {
    fn deliver(&self, message: CompletionMessage) {
        let _ = self.sender.send(message);
    }
}

/// Handle to the background writer thread.
#[derive(Debug)]
pub struct CompletionWriter {
    handle: JoinHandle<Result<(), CodecError>>,
}

impl CompletionWriter {
    /// Wait for the writer to finish.
    ///
    /// The writer exits cleanly once every [`CompletionSender`] is dropped. A
    /// transport failure is returned as the inner error.
    pub fn join(self) -> thread::Result<Result<(), CodecError>> {
        self.handle.join()
    }
}

/// Spawn a writer thread for `transport` and return its sink.
///
/// The returned sink may be cloned and shared with a
/// `metal_api_core::completion::wire::CompletionOutbox`. Dropping every clone
/// ends the thread after it drains the queued messages.
pub fn spawn_writer<R, W>(
    mut transport: CompletionTransport<R, W>,
) -> std::io::Result<(CompletionSender, CompletionWriter)>
where
    R: Read + Send + 'static,
    W: Write + Send + 'static,
{
    let (sender, receiver) = mpsc::channel();
    let handle = thread::Builder::new()
        .name("metal-api-ipc-writer".into())
        .spawn(move || -> Result<(), CodecError> {
            while let Ok(message) = receiver.recv() {
                transport.send(&message)?;
                transport.flush()?;
            }
            Ok(())
        })?;
    Ok((CompletionSender { sender }, CompletionWriter { handle }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::completion::wire::{
        CompletionSequence, CompletionTokenUpdate, CompletionUpdate,
    };
    use metal_api_core::provider::{CompletionToken, DeviceEpoch, SubmissionId};
    use std::io;

    fn message() -> CompletionMessage {
        CompletionMessage::Token(CompletionTokenUpdate {
            token: CompletionToken {
                device_epoch: DeviceEpoch::new(7),
                submission_id: SubmissionId::new(42),
            },
            sequence: CompletionSequence::new(1),
            update: CompletionUpdate::Submitted,
        })
    }

    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn writer_stops_with_the_transport_error() {
        let transport = CompletionTransport::new(io::empty(), FailingWriter);
        let (sender, writer) = spawn_writer(transport).unwrap();
        sender.deliver(message());
        drop(sender);
        let error = writer.join().unwrap().unwrap_err();
        assert!(matches!(error, CodecError::Io(_)));
    }

    #[test]
    fn dropping_every_sender_ends_the_writer_cleanly() {
        let (sender, writer) =
            spawn_writer(CompletionTransport::new(io::empty(), io::sink())).unwrap();
        sender.deliver(message());
        drop(sender);
        writer.join().unwrap().unwrap();
    }
}
