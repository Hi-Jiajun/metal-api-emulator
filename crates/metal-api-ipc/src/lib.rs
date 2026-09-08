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
//!
//! [`receiver::CompletionReceiver`] is the owner-side half: it applies received
//! messages to a `metal_api_core::completion::wire::CompletionMirror` and can
//! retire leases through the mirror's converged observation.

pub mod codec;
pub mod receiver;
pub mod transport;

#[cfg(unix)]
pub use transport::unix;
