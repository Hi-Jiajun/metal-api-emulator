//! Length-prefixed byte-stream transport for the neutral completion stream.
//!
//! `metal-api-core` defines transport-independent completion messages. This
//! crate carries those values over any [`std::io::Read`] + [`std::io::Write`]
//! stream with a small, versioned, dependency-free binary codec. It does not
//! own GPU resources, does not interpret provider state beyond decoding the
//! message, and deliberately has no external dependencies.
//!
//! The generic [`transport::CompletionTransport`] works over a Unix socket, a
//! TCP stream, a pipe pair or an in-memory buffer. The `unix` module (enabled
//! on Unix targets) adds a `UnixStream` pair helper and a listener wrapper for
//! the cross-process test and for a future owner/provider split.

pub mod codec;
pub mod transport;

#[cfg(unix)]
pub use transport::unix;
