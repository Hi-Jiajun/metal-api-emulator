//! Byte-stream wrapper around the completion codec.
//!
//! [`CompletionTransport`] pairs a [`Read`] source with a [`Write`] sink and
//! carries [`CompletionMessage`] values through
//! [`CompletionCodec`]. It adds no framing of
//! its own: one [`CompletionTransport::send`] writes exactly one frame and one
//! [`CompletionTransport::recv`] reads exactly one frame, so both ends of a
//! connection observe the same message order.
//!
//! The reader and writer are separate type parameters because the two halves
//! are often distinct handles, such as a child process's stdout and stdin. A
//! duplex socket is wrapped by the [`unix`] helpers, which clone
//! the stream into a reader and a writer.

use crate::codec::{CodecError, CompletionCodec};
use metal_api_core::completion::wire::CompletionMessage;
use std::io::{Read, Write};

/// Carries completion messages over a byte stream.
///
/// The type keeps the underlying halves accessible so a caller can set socket
/// options or recover them after the connection is closed. It never buffers
/// whole messages: each call maps to one codec frame.
#[derive(Debug)]
pub struct CompletionTransport<R, W> {
    reader: R,
    writer: W,
    sent: u64,
    received: u64,
}

impl<R: Read, W: Write> CompletionTransport<R, W> {
    /// Wrap a reader and a writer.
    pub const fn new(reader: R, writer: W) -> Self {
        Self {
            reader,
            writer,
            sent: 0,
            received: 0,
        }
    }

    /// Encode and write one frame.
    ///
    /// `write_all` is used, so a partial write is reported as an I/O error.
    /// The underlying writer is not flushed; a buffered or pipe writer should
    /// be followed by [`CompletionTransport::flush`].
    pub fn send(&mut self, message: &CompletionMessage) -> Result<(), CodecError> {
        CompletionCodec::write(&mut self.writer, message)?;
        self.sent += 1;
        Ok(())
    }

    /// Read and decode one frame.
    ///
    /// A clean end of stream before the first byte of a frame returns
    /// [`CodecError::Eof`]; a frame cut in half returns a truncation error.
    pub fn recv(&mut self) -> Result<CompletionMessage, CodecError> {
        let message = CompletionCodec::read(&mut self.reader)?;
        self.received += 1;
        Ok(message)
    }

    /// Flush the underlying writer.
    pub fn flush(&mut self) -> Result<(), CodecError> {
        self.writer.flush()?;
        Ok(())
    }

    /// Number of frames written by this transport.
    pub const fn sent(&self) -> u64 {
        self.sent
    }

    /// Number of frames read by this transport.
    pub const fn received(&self) -> u64 {
        self.received
    }

    /// Borrow the reader half.
    pub const fn reader(&self) -> &R {
        &self.reader
    }

    /// Borrow the writer half.
    pub const fn writer(&self) -> &W {
        &self.writer
    }

    /// Mutably borrow the reader half.
    pub fn reader_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    /// Mutably borrow the writer half.
    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.writer
    }

    /// Recover the reader and writer halves.
    pub fn into_inner(self) -> (R, W) {
        (self.reader, self.writer)
    }
}

#[cfg(unix)]
pub mod unix {
    //! Unix-domain socket helpers for [`CompletionTransport`].
    //!
    //! A `UnixStream` is a single full-duplex handle, while the transport wants
    //! a reader and a writer. [`pair`] and [`connect`] clone the stream into
    //! the two halves. [`UnixListenerTransport`] binds a listener and wraps
    //! each accepted connection the same way, which is enough for a provider
    //! process that owns one socket per client.

    use super::CompletionTransport;
    use std::io;
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::Path;
    use std::time::Duration;

    /// Transport over a Unix-domain socket.
    pub type UnixTransport = CompletionTransport<UnixStream, UnixStream>;

    /// Wrap one connected socket.
    pub fn from_stream(stream: UnixStream) -> io::Result<UnixTransport> {
        Ok(CompletionTransport::new(stream.try_clone()?, stream))
    }

    /// Connect to a listening socket.
    pub fn connect(path: impl AsRef<Path>) -> io::Result<UnixTransport> {
        from_stream(UnixStream::connect(path)?)
    }

    /// Create a connected socket pair. The two transports are independent
    /// peers; closing either one makes the other report end of stream.
    pub fn pair() -> io::Result<(UnixTransport, UnixTransport)> {
        let (first, second) = UnixStream::pair()?;
        Ok((from_stream(first)?, from_stream(second)?))
    }

    impl CompletionTransport<UnixStream, UnixStream> {
        /// Set the read timeout on the underlying socket.
        pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
            self.reader().set_read_timeout(timeout)
        }

        /// Set the write timeout on the underlying socket.
        pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
            self.writer().set_write_timeout(timeout)
        }

        /// Shut down both directions of the underlying socket.
        pub fn shutdown_both(&self) -> io::Result<()> {
            self.reader().shutdown(std::net::Shutdown::Both)
        }
    }

    /// Listening socket that accepts transports.
    #[derive(Debug)]
    pub struct UnixListenerTransport {
        listener: UnixListener,
    }

    impl UnixListenerTransport {
        /// Bind a listener to `path`.
        ///
        /// The caller owns the path and should remove the socket file when the
        /// listener is dropped.
        pub fn bind(path: impl AsRef<Path>) -> io::Result<Self> {
            Ok(Self {
                listener: UnixListener::bind(path)?,
            })
        }

        /// Wrap an existing listener.
        pub const fn from_listener(listener: UnixListener) -> Self {
            Self { listener }
        }

        /// Accept one connection.
        pub fn accept(&self) -> io::Result<UnixTransport> {
            let (stream, _) = self.listener.accept()?;
            from_stream(stream)
        }

        /// Borrow the underlying listener.
        pub const fn listener(&self) -> &UnixListener {
            &self.listener
        }

        /// Recover the underlying listener.
        pub fn into_listener(self) -> UnixListener {
            self.listener
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::completion::wire::{
        CompletionFailure, CompletionSequence, CompletionTokenUpdate, CompletionUpdate,
    };
    use metal_api_core::provider::{
        CompletionToken, DeviceEpoch, ProviderErrorClass, ProviderPhase, SubmissionId,
    };
    use std::io::Cursor;

    fn token() -> CompletionToken {
        CompletionToken {
            device_epoch: DeviceEpoch::new(7),
            submission_id: SubmissionId::new(42),
        }
    }

    fn message(sequence: u64, update: CompletionUpdate) -> CompletionMessage {
        CompletionMessage::Token(CompletionTokenUpdate {
            token: token(),
            sequence: CompletionSequence::new(sequence),
            update,
        })
    }

    #[test]
    fn round_trips_over_in_memory_halves() {
        let sent = message(1, CompletionUpdate::Submitted);
        let mut sender = CompletionTransport::new(Cursor::new(Vec::new()), Cursor::new(Vec::new()));
        sender.send(&sent).unwrap();
        assert_eq!(sender.sent(), 1);
        let (_, bytes) = sender.into_inner();

        let mut receiver =
            CompletionTransport::new(Cursor::new(bytes.into_inner()), Cursor::new(Vec::new()));
        assert_eq!(receiver.recv().unwrap(), sent);
        assert_eq!(receiver.received(), 1);
        assert!(matches!(receiver.recv().unwrap_err(), CodecError::Eof));
    }

    #[test]
    fn preserves_order_and_failure_detail() {
        let messages = [
            message(1, CompletionUpdate::Submitted),
            message(
                2,
                CompletionUpdate::Failed(
                    CompletionFailure::new(
                        ProviderPhase::Wait,
                        ProviderErrorClass::Resource,
                        "wait queue full",
                    )
                    .unwrap(),
                ),
            ),
            message(3, CompletionUpdate::CompletedVisible),
        ];

        let mut sender = CompletionTransport::new(Cursor::new(Vec::new()), Cursor::new(Vec::new()));
        for message in &messages {
            sender.send(message).unwrap();
        }
        let (_, bytes) = sender.into_inner();

        let mut receiver =
            CompletionTransport::new(Cursor::new(bytes.into_inner()), Cursor::new(Vec::new()));
        for message in &messages {
            assert_eq!(receiver.recv().unwrap(), *message);
        }
        assert!(matches!(receiver.recv().unwrap_err(), CodecError::Eof));
        assert_eq!(receiver.received(), messages.len() as u64);
    }
}
