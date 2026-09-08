//! Length-prefixed byte-stream transport for the neutral completion stream.
//!
//! `metal-api-core` defines transport-independent completion messages. This
//! crate carries those values over any [`std::io::Read`] + [`std::io::Write`]
//! stream with a small, versioned binary codec. It does not own GPU resources,
//! does not interpret provider state beyond decoding the message, and depends
//! only on `metal-api-core` plus `libc` for the Unix shared-memory helpers.
//!
//! The generic [`transport::CompletionTransport`] works over a Unix socket, a
//! TCP stream, a pipe pair or an in-memory buffer. The `unix` module (enabled
//! on Unix targets) adds a `UnixStream` pair helper and a listener wrapper for
//! the cross-process test and for a future owner/provider split.
//!
//! [`receiver::CompletionReceiver`] is the owner-side half: it applies received
//! messages to a `metal_api_core::completion::wire::CompletionMirror` and can
//! retire leases through the mirror's converged observation.
//!
//! [`sender::spawn_writer`] is the provider-side half: it implements the core
//! `CompletionSink` over a background writer thread, so a provider outbox can
//! publish without blocking a completion handler.
//!
//! [`shared`] adds process-shared memory for no-copy provider leases: the
//! owner creates an anonymous mapping and passes its descriptor over a Unix
//! socket with `SCM_RIGHTS`, so the provider can import the same physical
//! pages instead of receiving a copy.

pub mod codec;
pub mod receiver;
pub mod sender;
pub mod transport;

#[cfg(unix)]
pub mod shared;

#[cfg(unix)]
pub use transport::unix;
