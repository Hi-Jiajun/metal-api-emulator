//! Owner-to-provider command channel.
//!
//! The completion stream carries provider-to-owner notifications. This module
//! adds the opposite direction: a versioned request/response channel an owner
//! uses to compile pipelines, submit traces, observe completions and read back
//! results from a provider that lives in another process. One connection is
//! strictly request/response, so a command never interleaves with a completion
//! frame.
//!
//! [`RemoteProvider`] implements the core [`ComputeProvider`] and
//! [`PipelineProvider`] traits over the channel. Provider implementations
//! serve it with [`serve_provider`]; the server validates every submission
//! with its own capabilities before calling the real provider, so a remote
//! owner cannot bypass admission.
//!
//! # Scope
//!
//! A frame is bounded by [`crate::command_codec::MAX_COMMAND_FRAME`]. When an
//! encoded request exceeds the sender's frame limit, the transport splits it
//! into chunk frames and the receiver reassembles it before decoding, so a
//! trace larger than one frame travels over the same connection (bounded by
//! [`crate::command_codec::MAX_CHUNKED_PAYLOAD`]). Lease-backed views carry
//! only their lease id.
//! Staged leases travel inside the frame; no-copy leases use
//! [`CommandRequest::ImportBorrowedLease`], which carries the reservation in
//! the frame and sends the owner mapping with `SCM_RIGHTS` immediately after
//! it. Descriptor requests need [`serve_provider_unix`] on the provider side
//! and [`RemoteProvider::import_borrowed_lease`] on the owner side.

use crate::codec::CodecError;
use crate::command_codec::{CommandCodec, MAX_CHUNKED_PAYLOAD, MAX_COMMAND_FRAME};
use metal_api_core::provider::{BorrowedLease, FieldValue, NoCopyLeaseImporter};
use metal_api_core::provider::{
    CompiledComputePipeline, CompletionDisposition, CompletionReadback, CompletionToken,
    ComputeProvider, ComputeTrace, DeviceEpoch, LeaseId, LeaseImporter, LeaseReservation,
    PipelineCompileRequest, PipelineProvider, ProviderCapabilities, ProviderError,
    ProviderErrorClass, ProviderHealth, ProviderPhase, ProviderSubmission, QueuePriority,
    ResourceTableSnapshot, Retryability, StagedLease, ValidatedComputeTrace,
};
use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Read, Write};
use std::sync::Mutex;
use std::time::Duration;

/// One owner request. The response is always a [`CommandResponse`] on the same
/// connection.
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum CommandRequest {
    /// Ask for the provider device epoch and capabilities.
    Capabilities,
    /// Ask for the provider's current health.
    Health,
    /// Compile one reviewed shader artifact on the provider.
    Compile { request: PipelineCompileRequest },
    /// Import staged owner bytes for one lease.
    ImportStagedLease { staged: StagedLease },
    /// Drop a staged lease import.
    ReleaseStagedLease { lease_id: LeaseId },
    /// Import a process-shared owner mapping as a no-copy lease.
    ///
    /// The frame carries the reservation; the owner sends exactly one
    /// descriptor over the same Unix socket immediately after the frame and
    /// the provider maps it before answering. Only [`serve_provider_unix`]
    /// can serve this request.
    ImportBorrowedLease { reservation: LeaseReservation },
    /// Drop a descriptor-backed no-copy lease import.
    ReleaseBorrowedLease { lease_id: LeaseId },
    /// Install an owner queue-priority marking on the provider's device
    /// queues.
    ///
    /// The frame carries the owner's marking, not a device table: the provider
    /// expands it to one tier per device queue (padding with
    /// `QueuePriority::Default`, truncating the rest) and answers with the
    /// table it installed. A connection that never sends this request leaves
    /// every queue at `QueuePriority::Default`, which is the scheduling
    /// behaviour the channel had before the tier table existed.
    ///
    /// One connection carries one exchange at a time, so a marking is ordered
    /// against the submits on the same connection: it steers every submission
    /// sent after its response, and nothing about a trace changes.
    SetQueuePriorities { tiers: Vec<QueuePriority> },
    /// Admit and submit one trace with its resource snapshot.
    Submit {
        trace: ComputeTrace,
        resources: ResourceTableSnapshot,
    },
    /// Observe a token for at most `timeout`.
    Wait {
        token: CompletionToken,
        timeout: Duration,
    },
    /// Read back host-visible bytes for a completed token.
    Readback { token: CompletionToken },
    /// Release observation of a token.
    Cancel { token: CompletionToken },
    /// Forget a compiled pipeline after its submissions retired.
    ReleasePipeline { pipeline: CompiledComputePipeline },
    /// Forget a completion slot.
    ReleaseCompletion { token: CompletionToken },
}

impl CommandRequest {
    /// Stable name used in protocol diagnostics.
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Capabilities => "capabilities",
            Self::Health => "health",
            Self::Compile { .. } => "compile",
            Self::ImportStagedLease { .. } => "import_staged_lease",
            Self::ReleaseStagedLease { .. } => "release_staged_lease",
            Self::ImportBorrowedLease { .. } => "import_borrowed_lease",
            Self::ReleaseBorrowedLease { .. } => "release_borrowed_lease",
            Self::SetQueuePriorities { .. } => "set_queue_priorities",
            Self::Submit { .. } => "submit",
            Self::Wait { .. } => "wait",
            Self::Readback { .. } => "readback",
            Self::Cancel { .. } => "cancel",
            Self::ReleasePipeline { .. } => "release_pipeline",
            Self::ReleaseCompletion { .. } => "release_completion",
        }
    }
}

/// One provider response. `Error` carries the exact [`ProviderError`] the
/// in-process provider returned, including fields, retryability and the
/// completion disposition.
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum CommandResponse {
    Capabilities {
        epoch: DeviceEpoch,
        capabilities: ProviderCapabilities,
    },
    Health {
        health: ProviderHealth,
    },
    Compiled {
        pipeline: CompiledComputePipeline,
    },
    Imported,
    Submitted {
        submission: ProviderSubmission,
    },
    Observed {
        disposition: CompletionDisposition,
    },
    Readback {
        readback: CompletionReadback,
    },
    /// The device queue table a [`CommandRequest::SetQueuePriorities`] marking
    /// installed, one entry per device queue.
    QueuePriorities {
        installed: Vec<QueuePriority>,
    },
    Released,
    Error {
        error: ProviderError,
    },
}

impl CommandResponse {
    /// Stable name used in protocol diagnostics.
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Capabilities { .. } => "capabilities",
            Self::Health { .. } => "health",
            Self::Compiled { .. } => "compiled",
            Self::Imported => "imported",
            Self::Submitted { .. } => "submitted",
            Self::Observed { .. } => "observed",
            Self::Readback { .. } => "readback",
            Self::QueuePriorities { .. } => "queue_priorities",
            Self::Released => "released",
            Self::Error { .. } => "error",
        }
    }
}

/// Failure on the command channel.
#[derive(Debug)]
pub enum CommandError {
    /// Framing, encoding or I/O failure.
    Codec(CodecError),
    /// The provider returned a structured refusal.
    Remote(ProviderError),
    /// The provider answered with a different operation.
    UnexpectedResponse {
        expected: &'static str,
        actual: &'static str,
    },
    /// A descriptor-carrying request arrived on a transport that cannot carry
    /// descriptors.
    DescriptorUnsupported { kind: &'static str },
}

impl fmt::Display for CommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(error) => write!(formatter, "command channel failed: {error}"),
            Self::Remote(error) => write!(formatter, "provider refused command: {error:?}"),
            Self::UnexpectedResponse { expected, actual } => write!(
                formatter,
                "command response mismatch: expected {expected}, received {actual}"
            ),
            Self::DescriptorUnsupported { kind } => write!(
                formatter,
                "command request {kind} needs a Unix descriptor transport"
            ),
        }
    }
}

impl std::error::Error for CommandError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Codec(error) => Some(error),
            Self::Remote(_)
            | Self::UnexpectedResponse { .. }
            | Self::DescriptorUnsupported { .. } => None,
        }
    }
}

impl From<CodecError> for CommandError {
    fn from(error: CodecError) -> Self {
        Self::Codec(error)
    }
}

/// Carries one request/response exchange over a byte stream.
#[derive(Debug)]
pub struct CommandTransport<R, W> {
    reader: R,
    writer: W,
    sent: u64,
    received: u64,
    max_frame: usize,
    next_transfer: u64,
    pending_chunk: Option<ChunkAssembly>,
}

#[derive(Debug)]
struct ChunkAssembly {
    transfer_id: u64,
    total: usize,
    bytes: Vec<u8>,
}

impl<R: Read, W: Write> CommandTransport<R, W> {
    /// Wrap a reader and a writer.
    pub const fn new(reader: R, writer: W) -> Self {
        Self {
            reader,
            writer,
            sent: 0,
            received: 0,
            max_frame: MAX_COMMAND_FRAME,
            next_transfer: 0,
            pending_chunk: None,
        }
    }

    /// Set the largest request payload sent as one frame before the transport
    /// splits it into chunk frames.
    ///
    /// Defaults to [`MAX_COMMAND_FRAME`]. A receiver accepts chunk frames
    /// regardless of its own value, so tests and constrained transports can
    /// lower it on the sending side only.
    pub fn set_max_frame(&mut self, max_frame: usize) {
        self.max_frame = max_frame.clamp(1, MAX_COMMAND_FRAME);
    }

    /// Largest request payload sent as one frame.
    pub const fn max_frame(&self) -> usize {
        self.max_frame
    }

    /// Write one named mapping after the current request frame.
    ///
    /// The name and length follow the frame bytes, so a server that just read
    /// an [`CommandRequest::ImportBorrowedLease`] can receive exactly this
    /// mapping before writing its response.
    pub fn send_mapping_name(&mut self, name: &str, len: usize) -> io::Result<()> {
        crate::shared::send_mapping(&mut self.writer, name, len)
    }

    /// Receive one named mapping sent by [`Self::send_mapping_name`].
    pub fn recv_mapping_name(&mut self) -> io::Result<(String, usize)> {
        crate::shared::recv_mapping(&mut self.reader)
    }

    /// Send one request and read its response.
    pub fn request(&mut self, request: &CommandRequest) -> Result<CommandResponse, CommandError> {
        self.request_with(request, |_| Ok(()))
    }

    /// Send one request, run `after_send` on the writer, then read the
    /// response.
    ///
    /// Descriptor-carrying requests use this to write the frame first and the
    /// `SCM_RIGHTS` control message second, so the provider reads the request
    /// before the descriptor it belongs to.
    pub fn request_with<F>(
        &mut self,
        request: &CommandRequest,
        after_send: F,
    ) -> Result<CommandResponse, CommandError>
    where
        F: FnOnce(&mut W) -> Result<(), CommandError>,
    {
        let payload = CommandCodec::encode_request_payload(request)?;
        self.send_payload(CommandCodec::request_frame_kind(), &payload)?;
        self.writer.flush().map_err(CodecError::Io)?;
        self.sent += 1;
        after_send(&mut self.writer)?;
        let payload = self.recv_payload(CommandCodec::response_frame_kind())?;
        let response = CommandCodec::decode_response_payload(&payload)?;
        self.received += 1;
        Ok(response)
    }

    /// Read one request. Used by the provider-side server loop.
    pub fn recv_request(&mut self) -> Result<CommandRequest, CommandError> {
        let payload = self.recv_payload(CommandCodec::request_frame_kind())?;
        let request = CommandCodec::decode_request_payload(&payload)?;
        self.received += 1;
        Ok(request)
    }

    /// Write one response. Used by the provider-side server loop.
    pub fn send_response(&mut self, response: &CommandResponse) -> Result<(), CommandError> {
        let payload = CommandCodec::encode_response_payload(response)?;
        self.send_payload(CommandCodec::response_frame_kind(), &payload)?;
        self.sent += 1;
        Ok(())
    }

    fn send_payload(&mut self, frame_kind: u8, payload: &[u8]) -> Result<(), CommandError> {
        if payload.len() <= self.max_frame {
            self.write_frame(frame_kind, payload)?;
            return Ok(());
        }
        let total =
            u64::try_from(payload.len()).map_err(|_| CodecError::ChunkedPayloadTooLarge {
                total: u64::MAX,
                maximum: MAX_CHUNKED_PAYLOAD,
            })?;
        if payload.len() > MAX_CHUNKED_PAYLOAD {
            return Err(CodecError::ChunkedPayloadTooLarge {
                total,
                maximum: MAX_CHUNKED_PAYLOAD,
            }
            .into());
        }
        let transfer_id = self.next_transfer;
        self.next_transfer = self.next_transfer.wrapping_add(1);
        let mut offset = 0usize;
        while offset < payload.len() {
            let end = (offset + self.max_frame).min(payload.len());
            CommandCodec::write_chunk_frame(
                &mut self.writer,
                transfer_id,
                offset as u64,
                total,
                &payload[offset..end],
            )?;
            offset = end;
        }
        Ok(())
    }

    fn write_frame(&mut self, frame_kind: u8, payload: &[u8]) -> Result<(), CommandError> {
        if frame_kind == CommandCodec::request_frame_kind() {
            CommandCodec::write_request_payload(&mut self.writer, payload)?;
        } else {
            CommandCodec::write_response_payload(&mut self.writer, payload)?;
        }
        Ok(())
    }

    fn recv_payload(&mut self, frame_kind: u8) -> Result<Vec<u8>, CommandError> {
        loop {
            let (kind, payload) = CommandCodec::read_raw_frame(&mut self.reader)?;
            if kind == frame_kind {
                if let Some(pending) = self.pending_chunk.take() {
                    return Err(CodecError::ChunkInterrupted {
                        received: pending.bytes.len() as u64,
                        declared: pending.total as u64,
                    }
                    .into());
                }
                return Ok(payload);
            }
            if kind != CommandCodec::chunk_frame_kind() {
                return Err(CodecError::UnknownFrameKind(kind).into());
            }
            let chunk = CommandCodec::decode_chunk_payload(&payload)?;
            let total =
                usize::try_from(chunk.total).map_err(|_| CodecError::ChunkedPayloadTooLarge {
                    total: chunk.total,
                    maximum: MAX_CHUNKED_PAYLOAD,
                })?;
            if total > MAX_CHUNKED_PAYLOAD {
                return Err(CodecError::ChunkedPayloadTooLarge {
                    total: chunk.total,
                    maximum: MAX_CHUNKED_PAYLOAD,
                }
                .into());
            }
            match &mut self.pending_chunk {
                None => {
                    if chunk.offset != 0 || chunk.bytes.is_empty() || total == 0 {
                        return Err(CodecError::ChunkSequence {
                            expected: 0,
                            actual: chunk.offset,
                        }
                        .into());
                    }
                    self.pending_chunk = Some(ChunkAssembly {
                        transfer_id: chunk.transfer_id,
                        total,
                        bytes: Vec::new(),
                    });
                }
                Some(pending) => {
                    if pending.transfer_id != chunk.transfer_id {
                        return Err(CodecError::ChunkSequence {
                            expected: pending.transfer_id,
                            actual: chunk.transfer_id,
                        }
                        .into());
                    }
                    if pending.total as u64 != chunk.total {
                        return Err(CodecError::ChunkTotalMismatch {
                            declared: pending.total as u64,
                            actual: chunk.total,
                        }
                        .into());
                    }
                    let expected = pending.bytes.len() as u64;
                    if chunk.offset != expected {
                        return Err(CodecError::ChunkSequence {
                            expected,
                            actual: chunk.offset,
                        }
                        .into());
                    }
                }
            }
            let end = chunk.offset + chunk.bytes.len() as u64;
            if end > chunk.total {
                return Err(CodecError::ChunkTotalMismatch {
                    declared: chunk.total,
                    actual: end,
                }
                .into());
            }
            let pending = self
                .pending_chunk
                .as_mut()
                .expect("pending chunk was set above");
            pending.bytes.extend_from_slice(&chunk.bytes);
            if end == chunk.total {
                let pending = self.pending_chunk.take().expect("completed chunk transfer");
                return Ok(pending.bytes);
            }
        }
    }

    /// Flush the underlying writer.
    pub fn flush(&mut self) -> Result<(), CommandError> {
        self.writer.flush().map_err(CodecError::Io)?;
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

    /// Recover the reader and writer halves.
    pub fn into_inner(self) -> (R, W) {
        (self.reader, self.writer)
    }
}

/// Client half of the command channel.
///
/// The transport is serialized because one connection carries one exchange at
/// a time. The owner may still observe the completion stream concurrently on a
/// separate connection; `wait` blocks only the command channel.
#[derive(Debug)]
pub struct RemoteProvider<R, W> {
    epoch: DeviceEpoch,
    capabilities: ProviderCapabilities,
    transport: Mutex<CommandTransport<R, W>>,
}

impl<R: Read, W: Write> RemoteProvider<R, W> {
    /// Negotiate capabilities and build the remote provider handle.
    pub fn connect(mut transport: CommandTransport<R, W>) -> Result<Self, CommandError> {
        let response = transport.request(&CommandRequest::Capabilities)?;
        match response {
            CommandResponse::Capabilities {
                epoch,
                capabilities,
            } => Ok(Self {
                epoch,
                capabilities,
                transport: Mutex::new(transport),
            }),
            CommandResponse::Error { error } => Err(CommandError::Remote(error)),
            other => Err(CommandError::UnexpectedResponse {
                expected: "capabilities",
                actual: other.kind(),
            }),
        }
    }

    fn exchange(
        &self,
        request: CommandRequest,
        phase: ProviderPhase,
    ) -> Result<CommandResponse, ProviderError> {
        self.exchange_with(request, phase, |_| Ok(()))
    }

    fn exchange_with<F>(
        &self,
        request: CommandRequest,
        phase: ProviderPhase,
        after_send: F,
    ) -> Result<CommandResponse, ProviderError>
    where
        F: FnOnce(&mut W) -> Result<(), CommandError>,
    {
        let mut transport = self
            .transport
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        transport
            .request_with(&request, after_send)
            .map_err(|error| transport_error(phase, error))
    }

    /// Import an owner named mapping as a no-copy lease over this command
    /// connection.
    ///
    /// The mapping name and length follow the request frame on the same byte
    /// stream, so any transport that carries the command frames can carry the
    /// mapping. Windows uses this path because it has no `SCM_RIGHTS`
    /// equivalent; Unix providers normally use
    /// [`RemoteProvider::import_borrowed_lease`] with a descriptor instead.
    /// The owner must keep the mapping alive until
    /// [`Self::release_borrowed_lease`] returns.
    pub fn import_named_borrowed_lease(
        &self,
        reservation: LeaseReservation,
        memory: &crate::shared::SharedMemory,
    ) -> Result<(), ProviderError> {
        let name = memory.mapping_name().ok_or_else(|| {
            let mut error = ProviderError::new(
                ProviderPhase::Resolve,
                ProviderErrorClass::Args,
                "shared_mapping_unnamed",
            )
            .expect("non-empty shared mapping slug");
            error.retryability = Retryability::Never;
            error.detail = Some("shared mapping has no exported name".into());
            error
        })?;
        let len = memory.len();
        match self.exchange_with(
            CommandRequest::ImportBorrowedLease { reservation },
            ProviderPhase::Resolve,
            |writer| {
                crate::shared::send_mapping(writer, name, len)
                    .map_err(|error| CommandError::Codec(CodecError::Io(error)))
            },
        )? {
            CommandResponse::Imported => Ok(()),
            CommandResponse::Error { error } => Err(error),
            other => Err(unexpected_response(
                ProviderPhase::Resolve,
                "imported",
                &other,
            )),
        }
    }

    /// Drop a descriptor- or name-backed no-copy lease import.
    pub fn release_borrowed_lease(&self, lease_id: LeaseId) -> Result<(), ProviderError> {
        match self.exchange(
            CommandRequest::ReleaseBorrowedLease { lease_id },
            ProviderPhase::Resolve,
        )? {
            CommandResponse::Released => Ok(()),
            CommandResponse::Error { error } => Err(error),
            other => Err(unexpected_response(
                ProviderPhase::Resolve,
                "released",
                &other,
            )),
        }
    }
}

impl<R: Read + Send, W: Write + Send> ComputeProvider for RemoteProvider<R, W> {
    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> ProviderHealth {
        match self.exchange(CommandRequest::Health, ProviderPhase::Resolve) {
            Ok(CommandResponse::Health { health }) => health,
            _ => ProviderHealth::Exhausted,
        }
    }

    /// Send the owner marking over the channel and return the provider's
    /// installed table.
    ///
    /// The owner cannot expand the marking itself: the queue count is a
    /// provider device property and the channel carries capabilities, not the
    /// queue table. The response is the only view of what the provider
    /// installed, so it is also the owner-side evidence that the marking
    /// reached the scheduler.
    fn set_queue_priorities(
        &self,
        tiers: &[QueuePriority],
    ) -> Result<Vec<QueuePriority>, ProviderError> {
        match self.exchange(
            CommandRequest::SetQueuePriorities {
                tiers: tiers.to_vec(),
            },
            ProviderPhase::Resolve,
        )? {
            CommandResponse::QueuePriorities { installed } => Ok(installed),
            CommandResponse::Error { error } => Err(error),
            other => Err(unexpected_response(
                ProviderPhase::Resolve,
                "queue_priorities",
                &other,
            )),
        }
    }

    fn submit(&self, trace: ValidatedComputeTrace) -> Result<ProviderSubmission, ProviderError> {
        let (trace, resources) = trace.into_parts();
        match self.exchange(
            CommandRequest::Submit { trace, resources },
            ProviderPhase::Submit,
        )? {
            CommandResponse::Submitted { submission } => Ok(submission),
            CommandResponse::Error { error } => Err(error),
            other => Err(unexpected_response(
                ProviderPhase::Submit,
                "submitted",
                &other,
            )),
        }
    }

    fn wait(
        &self,
        token: CompletionToken,
        timeout: Duration,
    ) -> Result<CompletionDisposition, ProviderError> {
        match self.exchange(CommandRequest::Wait { token, timeout }, ProviderPhase::Wait)? {
            CommandResponse::Observed { disposition } => Ok(disposition),
            CommandResponse::Error { error } => Err(error),
            other => Err(unexpected_response(ProviderPhase::Wait, "observed", &other)),
        }
    }

    fn cancel(&self, token: CompletionToken) -> Result<CompletionDisposition, ProviderError> {
        match self.exchange(CommandRequest::Cancel { token }, ProviderPhase::Wait)? {
            CommandResponse::Observed { disposition } => Ok(disposition),
            CommandResponse::Error { error } => Err(error),
            other => Err(unexpected_response(ProviderPhase::Wait, "observed", &other)),
        }
    }

    fn readback(&self, token: CompletionToken) -> Result<CompletionReadback, ProviderError> {
        match self.exchange(CommandRequest::Readback { token }, ProviderPhase::Readback)? {
            CommandResponse::Readback { readback } => Ok(readback),
            CommandResponse::Error { error } => Err(error),
            other => Err(unexpected_response(
                ProviderPhase::Readback,
                "readback",
                &other,
            )),
        }
    }
}

impl<R: Read + Send, W: Write + Send> LeaseImporter for RemoteProvider<R, W> {
    fn import_staged_lease(&self, staged: StagedLease) -> Result<(), ProviderError> {
        match self.exchange(
            CommandRequest::ImportStagedLease { staged },
            ProviderPhase::Resolve,
        )? {
            CommandResponse::Imported => Ok(()),
            CommandResponse::Error { error } => Err(error),
            other => Err(unexpected_response(
                ProviderPhase::Resolve,
                "imported",
                &other,
            )),
        }
    }

    fn release_staged_lease(&self, lease_id: LeaseId) -> Result<(), ProviderError> {
        match self.exchange(
            CommandRequest::ReleaseStagedLease { lease_id },
            ProviderPhase::Resolve,
        )? {
            CommandResponse::Released => Ok(()),
            CommandResponse::Error { error } => Err(error),
            other => Err(unexpected_response(
                ProviderPhase::Resolve,
                "released",
                &other,
            )),
        }
    }
}

impl<R: Read + Send, W: Write + Send> PipelineProvider for RemoteProvider<R, W> {
    fn device_epoch(&self) -> DeviceEpoch {
        self.epoch
    }

    fn compile(
        &self,
        request: PipelineCompileRequest,
    ) -> Result<CompiledComputePipeline, ProviderError> {
        match self.exchange(CommandRequest::Compile { request }, ProviderPhase::Compile)? {
            CommandResponse::Compiled { pipeline } => Ok(pipeline),
            CommandResponse::Error { error } => Err(error),
            other => Err(unexpected_response(
                ProviderPhase::Compile,
                "compiled",
                &other,
            )),
        }
    }

    fn release_pipeline(&self, pipeline: &CompiledComputePipeline) -> Result<(), ProviderError> {
        match self.exchange(
            CommandRequest::ReleasePipeline {
                pipeline: pipeline.clone(),
            },
            ProviderPhase::Resolve,
        )? {
            CommandResponse::Released => Ok(()),
            CommandResponse::Error { error } => Err(error),
            other => Err(unexpected_response(
                ProviderPhase::Resolve,
                "released",
                &other,
            )),
        }
    }

    fn release_completion(&self, token: CompletionToken) -> Result<(), ProviderError> {
        match self.exchange(
            CommandRequest::ReleaseCompletion { token },
            ProviderPhase::Wait,
        )? {
            CommandResponse::Released => Ok(()),
            CommandResponse::Error { error } => Err(error),
            other => Err(unexpected_response(ProviderPhase::Wait, "released", &other)),
        }
    }
}

#[cfg(unix)]
impl RemoteProvider<std::os::unix::net::UnixStream, std::os::unix::net::UnixStream> {
    /// Import an owner [`crate::shared::SharedMemory`] mapping as a no-copy
    /// lease over this command connection.
    ///
    /// The reservation must fit inside the mapped range. The descriptor
    /// travels over the same socket with `SCM_RIGHTS`; the provider maps the
    /// same physical pages and imports them through
    /// [`NoCopyLeaseImporter::import_borrowed_lease`]. The owner must keep the
    /// mapping alive until [`Self::release_borrowed_lease`] returns.
    pub fn import_borrowed_lease(
        &self,
        reservation: LeaseReservation,
        memory: &crate::shared::SharedMemory,
    ) -> Result<(), ProviderError> {
        match self.exchange_with(
            CommandRequest::ImportBorrowedLease { reservation },
            ProviderPhase::Resolve,
            |writer| {
                Ok(crate::shared::send_fd(writer, memory.descriptor()).map_err(CodecError::Io)?)
            },
        )? {
            CommandResponse::Imported => Ok(()),
            CommandResponse::Error { error } => Err(error),
            other => Err(unexpected_response(
                ProviderPhase::Resolve,
                "imported",
                &other,
            )),
        }
    }
}

fn transport_error(phase: ProviderPhase, error: CommandError) -> ProviderError {
    // A closed channel is a provider-availability failure: the remote handle
    // cannot serve new work and the owner must reconnect. Framing or contract
    // errors stay internal because they describe a protocol defect rather than
    // a missing provider.
    let channel_closed = matches!(
        &error,
        CommandError::Codec(CodecError::Eof | CodecError::Io(_))
    );
    let (class, slug, retryability) = if channel_closed {
        (
            ProviderErrorClass::Resource,
            "provider_unavailable",
            Retryability::RetryAfterRecreate,
        )
    } else {
        (
            ProviderErrorClass::Internal,
            "command_transport_failed",
            Retryability::Unknown,
        )
    };
    let mut provider_error =
        ProviderError::new(phase, class, slug).expect("non-empty command transport slug");
    provider_error.retryability = retryability;
    provider_error.detail = Some(error.to_string());
    provider_error
}

fn unexpected_response(
    phase: ProviderPhase,
    expected: &'static str,
    actual: &CommandResponse,
) -> ProviderError {
    let mut error = ProviderError::new(
        phase,
        ProviderErrorClass::Internal,
        "command_response_mismatch",
    )
    .expect("non-empty command response slug");
    error.retryability = Retryability::Unknown;
    error.detail = Some(format!("expected {expected}, received {}", actual.kind()));
    error
}

/// Combined provider endpoint the server requires.
///
/// A provider that advertises staged leases implements both
/// [`PipelineProvider`] and [`LeaseImporter`]; this trait lets the server hold
/// one trait object instead of a generic parameter.
pub trait ProviderEndpoint: PipelineProvider + LeaseImporter {}

impl<T: PipelineProvider + LeaseImporter> ProviderEndpoint for T {}

/// Provider-side loop. Serves requests until the owner closes the connection.
///
/// Submission requests are admitted with the provider's own capabilities
/// before [`ComputeProvider::submit`] is called, so remote owners cannot skip
/// the admission contract.
pub fn serve_provider<R: Read, W: Write>(
    provider: &dyn ProviderEndpoint,
    transport: &mut CommandTransport<R, W>,
) -> Result<(), CommandError> {
    loop {
        let request = match transport.recv_request() {
            Ok(request) => request,
            Err(CommandError::Codec(CodecError::Eof)) => return Ok(()),
            Err(error) => return Err(error),
        };
        if let CommandRequest::ImportBorrowedLease { .. }
        | CommandRequest::ReleaseBorrowedLease { .. } = request
        {
            return Err(CommandError::DescriptorUnsupported {
                kind: request.kind(),
            });
        }
        let response = handle_request(provider, request);
        transport.send_response(&response)?;
        transport.flush()?;
    }
}

fn handle_request<P>(provider: &P, request: CommandRequest) -> CommandResponse
where
    P: ProviderEndpoint + ?Sized,
{
    match request {
        CommandRequest::Capabilities => CommandResponse::Capabilities {
            epoch: provider.device_epoch(),
            capabilities: provider.capabilities(),
        },
        CommandRequest::Health => CommandResponse::Health {
            health: provider.health(),
        },
        CommandRequest::Compile { request } => match provider.compile(request) {
            Ok(pipeline) => CommandResponse::Compiled { pipeline },
            Err(error) => CommandResponse::Error { error },
        },
        CommandRequest::ImportStagedLease { staged } => {
            match provider.import_staged_lease(staged) {
                Ok(()) => CommandResponse::Imported,
                Err(error) => CommandResponse::Error { error },
            }
        }
        CommandRequest::ReleaseStagedLease { lease_id } => {
            match provider.release_staged_lease(lease_id) {
                Ok(()) => CommandResponse::Released,
                Err(error) => CommandResponse::Error { error },
            }
        }
        CommandRequest::ImportBorrowedLease { .. }
        | CommandRequest::ReleaseBorrowedLease { .. } => CommandResponse::Error {
            error: descriptor_error(
                "descriptor_transport_required",
                ProviderErrorClass::Internal,
            ),
        },
        CommandRequest::Submit { trace, resources } => {
            let admitted = match provider.capabilities().validate_trace(trace, resources) {
                Ok(admitted) => admitted,
                Err(error) => return CommandResponse::Error { error },
            };
            match provider.submit(admitted) {
                Ok(submission) => CommandResponse::Submitted { submission },
                Err(error) => CommandResponse::Error { error },
            }
        }
        CommandRequest::Wait { token, timeout } => match provider.wait(token, timeout) {
            Ok(disposition) => CommandResponse::Observed { disposition },
            Err(error) => CommandResponse::Error { error },
        },
        CommandRequest::Readback { token } => match provider.readback(token) {
            Ok(readback) => CommandResponse::Readback { readback },
            Err(error) => CommandResponse::Error { error },
        },
        CommandRequest::Cancel { token } => match provider.cancel(token) {
            Ok(disposition) => CommandResponse::Observed { disposition },
            Err(error) => CommandResponse::Error { error },
        },
        CommandRequest::ReleasePipeline { pipeline } => {
            match provider.release_pipeline(&pipeline) {
                Ok(()) => CommandResponse::Released,
                Err(error) => CommandResponse::Error { error },
            }
        }
        CommandRequest::ReleaseCompletion { token } => match provider.release_completion(token) {
            Ok(()) => CommandResponse::Released,
            Err(error) => CommandResponse::Error { error },
        },
        CommandRequest::SetQueuePriorities { tiers } => {
            match provider.set_queue_priorities(&tiers) {
                Ok(installed) => CommandResponse::QueuePriorities { installed },
                Err(error) => CommandResponse::Error { error },
            }
        }
    }
}

/// Provider endpoint that can import descriptor-backed no-copy leases.
#[cfg(unix)]
pub trait DescriptorProviderEndpoint: ProviderEndpoint + NoCopyLeaseImporter {}

#[cfg(unix)]
impl<T: ProviderEndpoint + NoCopyLeaseImporter> DescriptorProviderEndpoint for T {}

/// Provider endpoint that can import named no-copy mappings.
///
/// Any transport that carries the command frames can carry a named mapping
/// (the name and length follow the request frame), so this server works on
/// Unix and Windows. A Unix provider normally uses the descriptor variant
/// [`serve_provider_unix`], which avoids exposing a session-visible name.
pub trait MappedProviderEndpoint: ProviderEndpoint + NoCopyLeaseImporter {}

impl<T: ProviderEndpoint + NoCopyLeaseImporter> MappedProviderEndpoint for T {}

/// Provider-side loop for a Unix command connection that can carry
/// `SCM_RIGHTS` descriptors.
///
/// This serves every [`serve_provider`] request plus
/// [`CommandRequest::ImportBorrowedLease`] and
/// [`CommandRequest::ReleaseBorrowedLease`]. The server keeps each imported
/// mapping alive until the provider releases the lease, so the borrowed
/// pointer stays valid for the whole import.
#[cfg(unix)]
pub fn serve_provider_unix(
    provider: &dyn DescriptorProviderEndpoint,
    transport: &mut unix::UnixCommandTransport,
) -> Result<(), CommandError> {
    let mut mappings = BTreeMap::new();
    let result = loop {
        let request = match transport.recv_request() {
            Ok(request) => request,
            Err(CommandError::Codec(CodecError::Eof)) => break Ok(()),
            Err(error) => break Err(error),
        };
        let response = match request {
            CommandRequest::ImportBorrowedLease { reservation } => {
                import_borrowed_descriptor(provider, transport, &mut mappings, reservation)
            }
            CommandRequest::ReleaseBorrowedLease { lease_id } => {
                release_borrowed_mapping(provider, &mut mappings, lease_id)
            }
            request => handle_request(provider, request),
        };
        if let Err(error) = transport
            .send_response(&response)
            .and_then(|()| transport.flush())
        {
            break Err(error);
        }
    };
    close_borrowed_mappings(provider, &mut mappings);
    result
}

/// Provider-side loop for a command connection that carries named mappings.
///
/// The owner writes the mapping name and length after the
/// [`CommandRequest::ImportBorrowedLease`] frame and the server opens the
/// object with [`crate::shared::SharedMemory::from_name`]. This is the Windows
/// equivalent of [`serve_provider_unix`]'s `SCM_RIGHTS` path, and it works
/// over any [`CommandTransport`].
pub fn serve_provider_named<R: Read, W: Write>(
    provider: &dyn MappedProviderEndpoint,
    transport: &mut CommandTransport<R, W>,
) -> Result<(), CommandError> {
    let mut mappings = BTreeMap::new();
    let result = loop {
        let request = match transport.recv_request() {
            Ok(request) => request,
            Err(CommandError::Codec(CodecError::Eof)) => break Ok(()),
            Err(error) => break Err(error),
        };
        let response = match request {
            CommandRequest::ImportBorrowedLease { reservation } => {
                import_borrowed_named(provider, transport, &mut mappings, reservation)
            }
            CommandRequest::ReleaseBorrowedLease { lease_id } => {
                release_borrowed_mapping(provider, &mut mappings, lease_id)
            }
            request => handle_request(provider, request),
        };
        if let Err(error) = transport
            .send_response(&response)
            .and_then(|()| transport.flush())
        {
            break Err(error);
        }
    };
    close_borrowed_mappings(provider, &mut mappings);
    result
}

fn close_borrowed_mappings(
    provider: &dyn NoCopyLeaseImporter,
    mappings: &mut BTreeMap<LeaseId, crate::shared::SharedMemory>,
) {
    for (lease_id, mapping) in std::mem::take(mappings) {
        match provider.release_borrowed_lease(lease_id) {
            Ok(()) => drop(mapping),
            // A submission still retains the mapping, so the provider may
            // still read it. Leak the pages instead of unmapping under the
            // provider; the operating system reclaims them at process exit.
            Err(_) => std::mem::forget(mapping),
        }
    }
}

fn import_borrowed_mapping(
    provider: &dyn NoCopyLeaseImporter,
    mappings: &mut BTreeMap<LeaseId, crate::shared::SharedMemory>,
    reservation: LeaseReservation,
    mapping: crate::shared::SharedMemory,
) -> CommandResponse {
    let lease_id = reservation.lease.lease_id;
    let Ok(expected) = usize::try_from(reservation.length) else {
        return CommandResponse::Error {
            error: descriptor_error(
                "borrowed_lease_length_unsupported",
                ProviderErrorClass::Args,
            ),
        };
    };
    if mapping.len() < expected {
        return CommandResponse::Error {
            error: descriptor_error("borrowed_lease_length_mismatch", ProviderErrorClass::Args)
                .with_field("mapped", FieldValue::Unsigned(mapping.len() as u64))
                .with_field("reservation", FieldValue::Unsigned(reservation.length)),
        };
    }
    let borrowed = match BorrowedLease::new(reservation, mapping.as_ptr() as usize) {
        Ok(borrowed) => borrowed,
        Err(error) => {
            return CommandResponse::Error {
                error: descriptor_error("borrowed_lease_invalid", ProviderErrorClass::Args)
                    .with_detail(error.to_string()),
            }
        }
    };
    // SAFETY: `mapping` is kept in `mappings` until the provider releases the
    // lease, so the borrowed pointer stays valid and address-stable for the
    // whole import.
    if let Err(error) = unsafe { provider.import_borrowed_lease(borrowed) } {
        return CommandResponse::Error { error };
    }
    mappings.insert(lease_id, mapping);
    CommandResponse::Imported
}

fn release_borrowed_mapping(
    provider: &dyn NoCopyLeaseImporter,
    mappings: &mut BTreeMap<LeaseId, crate::shared::SharedMemory>,
    lease_id: LeaseId,
) -> CommandResponse {
    if !mappings.contains_key(&lease_id) {
        return CommandResponse::Error {
            error: descriptor_error("lease_not_imported", ProviderErrorClass::Args),
        };
    }
    match provider.release_borrowed_lease(lease_id) {
        Ok(()) => {
            mappings.remove(&lease_id);
            CommandResponse::Released
        }
        Err(error) => CommandResponse::Error { error },
    }
}

#[cfg(unix)]
fn import_borrowed_descriptor(
    provider: &dyn DescriptorProviderEndpoint,
    transport: &mut unix::UnixCommandTransport,
    mappings: &mut BTreeMap<LeaseId, crate::shared::SharedMemory>,
    reservation: LeaseReservation,
) -> CommandResponse {
    let lease_id = reservation.lease.lease_id;
    let descriptor = match transport.recv_descriptor() {
        Ok(descriptor) => descriptor,
        Err(error) => {
            return CommandResponse::Error {
                error: descriptor_error(
                    "borrowed_lease_descriptor_invalid",
                    ProviderErrorClass::Internal,
                )
                .with_detail(error.to_string()),
            }
        }
    };
    // Every import request carries a descriptor, including one that will be
    // rejected. Consume it first so a duplicate lease id cannot leave the
    // descriptor payload in the stream and desynchronize the next frame.
    if mappings.contains_key(&lease_id) {
        drop(descriptor);
        return CommandResponse::Error {
            error: descriptor_error("lease_already_imported", ProviderErrorClass::Args),
        };
    }
    let mapping = match crate::shared::SharedMemory::from_owned_fd(descriptor) {
        Ok(mapping) => mapping,
        Err(error) => {
            return CommandResponse::Error {
                error: descriptor_error(
                    "borrowed_lease_descriptor_invalid",
                    ProviderErrorClass::Internal,
                )
                .with_detail(error.to_string()),
            }
        }
    };
    import_borrowed_mapping(provider, mappings, reservation, mapping)
}

fn import_borrowed_named<R: Read, W: Write>(
    provider: &dyn MappedProviderEndpoint,
    transport: &mut CommandTransport<R, W>,
    mappings: &mut BTreeMap<LeaseId, crate::shared::SharedMemory>,
    reservation: LeaseReservation,
) -> CommandResponse {
    let lease_id = reservation.lease.lease_id;
    let (name, len) = match transport.recv_mapping_name() {
        Ok(mapping) => mapping,
        Err(error) => {
            return CommandResponse::Error {
                error: descriptor_error(
                    "borrowed_lease_mapping_invalid",
                    ProviderErrorClass::Internal,
                )
                .with_detail(error.to_string()),
            }
        }
    };
    // The name arrives for every import request, including one that will be
    // rejected, so the stream stays framed for the next request.
    if mappings.contains_key(&lease_id) {
        return CommandResponse::Error {
            error: descriptor_error("lease_already_imported", ProviderErrorClass::Args),
        };
    }
    let mapping = match crate::shared::SharedMemory::from_name(&name, len) {
        Ok(mapping) => mapping,
        Err(error) => {
            return CommandResponse::Error {
                error: descriptor_error(
                    "borrowed_lease_mapping_invalid",
                    ProviderErrorClass::Internal,
                )
                .with_detail(error.to_string()),
            }
        }
    };
    import_borrowed_mapping(provider, mappings, reservation, mapping)
}

fn descriptor_error(slug: &'static str, class: ProviderErrorClass) -> ProviderError {
    ProviderError::new(ProviderPhase::Resolve, class, slug).expect("static descriptor refusal slug")
}

#[cfg(unix)]
pub mod unix {
    //! Unix-domain socket helpers for [`CommandTransport`].

    use super::CommandTransport;
    use std::io;
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::Path;
    use std::time::Duration;

    /// Command transport over a Unix-domain socket.
    pub type UnixCommandTransport = CommandTransport<UnixStream, UnixStream>;

    /// Wrap one connected socket.
    pub fn from_stream(stream: UnixStream) -> io::Result<UnixCommandTransport> {
        Ok(CommandTransport::new(stream.try_clone()?, stream))
    }

    /// Connect to a listening socket.
    pub fn connect(path: impl AsRef<Path>) -> io::Result<UnixCommandTransport> {
        from_stream(UnixStream::connect(path)?)
    }

    /// Create a connected socket pair.
    pub fn pair() -> io::Result<(UnixCommandTransport, UnixCommandTransport)> {
        let (first, second) = UnixStream::pair()?;
        Ok((from_stream(first)?, from_stream(second)?))
    }

    impl CommandTransport<UnixStream, UnixStream> {
        /// Send one descriptor over the command socket with `SCM_RIGHTS`.
        ///
        /// The one-byte descriptor payload follows the current frame bytes,
        /// so a server that just read a request frame can receive exactly this
        /// descriptor before writing its response.
        pub fn send_descriptor(&self, descriptor: std::os::fd::BorrowedFd<'_>) -> io::Result<()> {
            crate::shared::send_fd(&self.writer, descriptor)
        }

        /// Receive one descriptor sent by [`Self::send_descriptor`].
        pub fn recv_descriptor(&self) -> io::Result<std::os::fd::OwnedFd> {
            crate::shared::recv_fd(&self.reader)
        }

        /// Set the read timeout on the underlying socket.
        pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
            self.reader.set_read_timeout(timeout)
        }

        /// Set the write timeout on the underlying socket.
        pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
            self.writer.set_write_timeout(timeout)
        }

        /// Shut down both directions of the underlying socket.
        pub fn shutdown_both(&self) -> io::Result<()> {
            self.reader.shutdown(std::net::Shutdown::Both)
        }
    }

    /// Listening socket that accepts command transports.
    #[derive(Debug)]
    pub struct UnixListenerCommandTransport {
        listener: UnixListener,
    }

    impl UnixListenerCommandTransport {
        /// Bind a listener to `path`.
        pub fn bind(path: impl AsRef<Path>) -> io::Result<Self> {
            Ok(Self {
                listener: UnixListener::bind(path)?,
            })
        }

        /// Accept one connection.
        pub fn accept(&self) -> io::Result<UnixCommandTransport> {
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

pub mod tcp {
    //! TCP helpers for [`CommandTransport`].
    //!
    //! A `TcpStream` is a single full-duplex handle, while the transport wants
    //! a reader and a writer, so [`from_stream`] clones it. This module is the
    //! portable command channel used by the Windows named-mapping path; it
    //! also lets the Linux tests exercise the same code.

    use super::CommandTransport;
    use std::io;
    use std::net::{TcpListener, TcpStream, ToSocketAddrs};
    use std::time::Duration;

    /// Command transport over a TCP connection.
    pub type TcpCommandTransport = CommandTransport<TcpStream, TcpStream>;

    /// Wrap one connected stream, disabling Nagle so the mapping handshake
    /// stays request/response sized.
    pub fn from_stream(stream: TcpStream) -> io::Result<TcpCommandTransport> {
        stream.set_nodelay(true)?;
        Ok(CommandTransport::new(stream.try_clone()?, stream))
    }

    /// Connect to a listening endpoint.
    pub fn connect(addr: impl ToSocketAddrs) -> io::Result<TcpCommandTransport> {
        from_stream(TcpStream::connect(addr)?)
    }

    impl CommandTransport<TcpStream, TcpStream> {
        /// Set the read timeout on the underlying socket.
        pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
            self.reader.set_read_timeout(timeout)
        }

        /// Set the write timeout on the underlying socket.
        pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
            self.writer.set_write_timeout(timeout)
        }

        /// Shut down both directions of the underlying socket.
        pub fn shutdown_both(&self) -> io::Result<()> {
            self.reader.shutdown(std::net::Shutdown::Both)
        }
    }

    /// Listening socket that accepts command transports.
    #[derive(Debug)]
    pub struct TcpListenerCommandTransport {
        listener: TcpListener,
    }

    impl TcpListenerCommandTransport {
        /// Bind a listener.
        pub fn bind(addr: impl ToSocketAddrs) -> io::Result<Self> {
            Ok(Self {
                listener: TcpListener::bind(addr)?,
            })
        }

        /// Address the listener is bound to.
        pub fn local_addr(&self) -> io::Result<std::net::SocketAddr> {
            self.listener.local_addr()
        }

        /// Accept one connection.
        pub fn accept(&self) -> io::Result<TcpCommandTransport> {
            let (stream, _) = self.listener.accept()?;
            from_stream(stream)
        }

        /// Borrow the underlying listener.
        pub const fn listener(&self) -> &TcpListener {
            &self.listener
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use super::serve_provider_unix;
    use super::{
        serve_provider, serve_provider_named, CommandRequest, CommandResponse, RemoteProvider,
    };
    use crate::codec::CodecError;
    use crate::command_codec::{
        CommandCodec, MAX_HEAP_PLACEMENTS, MAX_PRESENT_SENTINEL_BYTES, MAX_QUEUE_PRIORITIES,
        MAX_SUPPORTED_HEAP_STORAGE_MODES, MAX_SUPPORTED_INDIRECT_COMMANDS,
        MAX_SUPPORTED_PRESENT_MODES, MAX_TAGGED_TRACE_PASSES,
    };
    use metal_api_core::provider::{
        AcquirePolicy, AllocationId, AllocationRecord, AttachmentFormat, BufferAccess,
        BufferBindingContract, BufferLease, BufferSource, BufferView, BufferWriteback, ClearColor,
        CompiledComputePipeline, CompletionDisposition, CompletionPolicy, CompletionReadback,
        CompletionToken, ComputePass, ComputeProvider, ComputeTrace, DeviceEpoch, Dispatch,
        DispatchKind, DispatchType, FootprintProof, FunctionIdentity, HeapDescriptor, HeapId,
        HeapPayload, HeapPlacement, HeapResource, IndirectCommandBufferDescriptor,
        IndirectCommandDescriptor, IndirectCommandKind, IndirectCommandPayload,
        IndirectCommandRange, InitialState, LeaseId, LeaseImporter, LeaseReservation, LoadOp,
        OperationId, PipelineCompileRequest, PipelineContract, PipelineId, PipelineProvider,
        PresentDescriptor, PresentMode, PresentTarget, ProviderCapabilities, ProviderError,
        ProviderErrorClass, ProviderHealth, ProviderPhase, ProviderSubmission, QueuePriority,
        RenderAttachment, RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot,
        Retryability, SemanticDigest, ShaderSource, StagedLease, StorageMode, StoreOp,
        SubmissionId, TextureAccess, TextureFormat, TextureSource, TextureType, TextureView,
        TracePass, ValidatedComputeTrace, VertexLayout, ViewId, FULL_SCREEN_TRIANGLE_VERTICES,
        MAX_COLOR_ATTACHMENTS, MAX_PRESENT_IMAGE_COUNT, PROVIDER_SCHEMA_VERSION,
    };
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use metal_api_core::provider::{BorrowedLease, BorrowedLeaseRegistry, NoCopyLeaseImporter};

    fn token() -> CompletionToken {
        CompletionToken {
            device_epoch: DeviceEpoch::new(7),
            submission_id: SubmissionId::new(3),
        }
    }

    fn compile_request() -> PipelineCompileRequest {
        PipelineCompileRequest {
            entry_name: "copy_word".into(),
            logical_digest: SemanticDigest::new("fixture-v1", b"copy_word".to_vec()).unwrap(),
            source: ShaderSource::SanitizedLl("define void @copy_word() { ret void }".into()),
        }
    }

    fn pipeline(request: &PipelineCompileRequest) -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(7),
            pipeline_id: PipelineId::new(11),
            function: FunctionIdentity {
                logical_digest: request.logical_digest.clone(),
                entry_name: request.entry_name.clone(),
                source: request.source.kind(),
            },
            contract: PipelineContract {
                dispatch_kind: DispatchKind::ThreadsExact,
                required_local_size: None,
                fixed_grid: None,
                push_constant_offset: 0,
                push_constant_bytes: 0,
                buffer_bindings: vec![BufferBindingContract {
                    metal_binding: 0,
                    access: BufferAccess::Write,
                    footprint: FootprintProof::Static { max_bytes: 4 },
                }],
                shader_capabilities: vec!["buffer-write".into()],
                translator_revision: Some(
                    SemanticDigest::new("translator", b"v1".to_vec()).unwrap(),
                ),
            },
            // A compiled compute pipeline carries the compute half only; the
            // render half belongs to a render registration's entry.
            render: None,
        }
    }

    fn trace(pipeline: &CompiledComputePipeline) -> ComputeTrace {
        ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: pipeline.device_epoch,
            operation_id: OperationId::new(21),
            pipelines: vec![pipeline.clone()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![TracePass::Compute(ComputePass {
                pipeline: pipeline.pipeline_id,
                buffers: vec![BufferView {
                    view_id: ViewId::new(31),
                    metal_binding: 0,
                    allocation_id: AllocationId::new(41),
                    offset: 0,
                    length: 4,
                    access: BufferAccess::Write,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![1, 2, 3, 4]),
                }],
                dispatch: Dispatch {
                    kind: DispatchKind::ThreadsExact,
                    grid: [1, 1, 1],
                    threads_per_threadgroup: [1, 1, 1],
                },
                textures: Vec::new(),
            })],
            completion_policy: metal_api_core::provider::CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        }
    }

    fn resources() -> ResourceTableSnapshot {
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(41),
                owner_epoch: DeviceEpoch::new(7),
                size: 8,
            })
            .unwrap();
        resources
    }

    /// The bytes a pre-render build put on the wire for the fixture trace
    /// built by [`trace`]. Hand-copied from `encode_request` at commit
    /// `8ec26a7`, before `TracePass` existed.
    ///
    /// A compute-only trace must keep producing exactly these bytes: the
    /// render track is additive, so an owner that has nothing new to say must
    /// stay byte-for-byte compatible with a provider that predates it.
    const LEGACY_SUBMIT_FRAME: &[u8] = &[
        77, 67, 67, 49, 1, 0, 0, 1, 104, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 21,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0,
        0, 10, 102, 105, 120, 116, 117, 114, 101, 45, 118, 49, 0, 0, 0, 0, 0, 0, 0, 9, 99, 111,
        112, 121, 95, 119, 111, 114, 100, 0, 0, 0, 0, 0, 0, 0, 9, 99, 111, 112, 121, 95, 119, 111,
        114, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 98, 117, 102, 102,
        101, 114, 45, 119, 114, 105, 116, 101, 1, 0, 0, 0, 0, 0, 0, 0, 10, 116, 114, 97, 110, 115,
        108, 97, 116, 111, 114, 0, 0, 0, 0, 0, 0, 0, 2, 118, 49, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        4, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0,
        0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    fn render_attachment(width: u64, height: u64) -> RenderAttachment {
        RenderAttachment {
            view_id: ViewId::new(71),
            allocation_id: AllocationId::new(41),
            format: AttachmentFormat::Rgba8Unorm,
            width,
            height,
            load: LoadOp::Clear(ClearColor::new([0xfe, 0xfe, 0xfe, 0xfe])),
            store: StoreOp::Store,
        }
    }

    /// The render half a render registration hands the owner: the entry pair
    /// the render pass resolves against and the colour format both stages were
    /// compiled for.
    fn render_contract() -> RenderPipelineContract {
        RenderPipelineContract {
            vertex_entry: "full_screen_vertex".into(),
            fragment_entry: "solid_color_fragment".into(),
            color_format: AttachmentFormat::Rgba8Unorm,
            vertex_layout: VertexLayout::None,
        }
    }

    fn render_pass_descriptor(
        compiled: &CompiledComputePipeline,
        width: u64,
        height: u64,
    ) -> RenderPassDescriptor {
        RenderPassDescriptor {
            pipeline: compiled.pipeline_id,
            color_attachments: vec![render_attachment(width, height)],
            viewport: [0, 0, width as u32, height as u32],
            vertices: FULL_SCREEN_TRIANGLE_VERTICES,
            present: None,
        }
    }

    /// A trace whose only pass is a render pass.
    fn render_only_trace() -> ComputeTrace {
        let mut compiled = pipeline(&compile_request());
        compiled.render = Some(render_contract());
        ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: compiled.device_epoch,
            operation_id: OperationId::new(21),
            pipelines: vec![compiled.clone()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![TracePass::Render(render_pass_descriptor(&compiled, 2, 2))],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        }
    }

    /// A trace that carries both shapes, in order.
    fn mixed_trace() -> ComputeTrace {
        let mut compiled = pipeline(&compile_request());
        // The render entry names this very id, so the table entry has to carry
        // the render half the pass resolves against.
        compiled.render = Some(render_contract());
        let mut value = trace(&compiled);
        value
            .passes
            .push(TracePass::Render(render_pass_descriptor(&compiled, 2, 2)));
        value
    }

    #[test]
    fn compute_only_submit_keeps_its_pre_render_bytes() {
        let compiled = pipeline(&compile_request());
        let request = CommandRequest::Submit {
            trace: trace(&compiled),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(frame, LEGACY_SUBMIT_FRAME);
        // The pre-render payload tag is unchanged, so a decoder built before
        // the render track reads this frame exactly as it always did.
        assert_eq!(frame[9], 0x03);
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
    }

    #[test]
    fn render_pass_frames_use_a_new_tag_and_round_trip() {
        let request = CommandRequest::Submit {
            trace: mixed_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        // A new payload tag, not the compute submit tag: an older decoder
        // answers `UnknownCommandTag` instead of misreading the tagged list.
        assert_eq!(frame[9], 0x0f);
        assert_ne!(frame[9], 0x03);
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);

        // Both shapes travel as the same `CommandRequest::Submit`, so nothing
        // downstream needs a second variant.
        let decoded = CommandCodec::decode_request(&frame).unwrap();
        let CommandRequest::Submit { trace, .. } = &decoded else {
            panic!("a render submit decodes as a submit");
        };
        assert_eq!(trace.passes.len(), 2);
        assert!(trace.passes[0].as_compute().is_some());
        assert_eq!(
            trace.passes[1]
                .as_render()
                .map(|pass| pass.color_attachments.len()),
            Some(1)
        );

        // A render-only trace round-trips too.
        let request = CommandRequest::Submit {
            trace: render_only_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
    }

    #[test]
    fn render_frames_carry_the_pipeline_render_contract() {
        // The tagged layout carries the table entry's render half, so the trace
        // an owner sends and the trace a provider decodes describe the same
        // pipeline. Core admission on the provider side reads exactly this
        // field to compare the attachment with the pipeline the pass names.
        let request = CommandRequest::Submit {
            trace: mixed_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(frame[9], 0x0f);
        let decoded = CommandCodec::decode_request(&frame).unwrap();
        assert_eq!(decoded, request);
        let CommandRequest::Submit {
            trace: decoded_trace,
            ..
        } = &decoded
        else {
            panic!("a render submit decodes as a submit");
        };
        assert_eq!(
            decoded_trace.pipelines[0].render.as_ref(),
            Some(&render_contract()),
            "the render half has to survive the round trip"
        );

        // The pre-render layout has no room for the render half, so a
        // compute-only frame keeps producing the old bytes and decodes with
        // `None`: an owner with nothing to render sends no render metadata.
        let compute_only = CommandRequest::Submit {
            trace: trace(&pipeline(&compile_request())),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&compute_only).unwrap();
        assert_eq!(frame, LEGACY_SUBMIT_FRAME);
        let decoded = CommandCodec::decode_request(&frame).unwrap();
        assert_eq!(decoded, compute_only);
        let CommandRequest::Submit {
            trace: legacy_trace,
            ..
        } = &decoded
        else {
            panic!("a compute submit decodes as a submit");
        };
        assert!(
            legacy_trace.pipelines[0].render.is_none(),
            "a pre-render frame carries no render half"
        );
    }

    #[test]
    fn capability_frames_declare_render_bits_under_their_own_tag() {
        let legacy = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(7),
            capabilities: fake_capabilities(),
        };
        let frame = CommandCodec::encode_response(&legacy).unwrap();
        assert_eq!(frame[9], 0x01);
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), legacy);

        let mut capabilities = fake_capabilities();
        capabilities.supports_render_passes = true;
        capabilities.max_color_attachments = 1;
        capabilities.max_attachment_dimension = [2, 2];
        capabilities.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        let declaring = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(7),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&declaring).unwrap();
        assert_eq!(frame[9], 0x0a);
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), declaring);
    }

    #[test]
    fn render_trace_frames_refuse_truncation_oversize_and_unknown_kinds() {
        let request = CommandRequest::Submit {
            trace: render_only_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(frame[9], 0x0f);

        // Locate the single colour attachment's view id: every other identity
        // in this frame is a smaller number and the eight-byte big-endian 71
        // is unique. The fixed layout before it is attachment count (8),
        // pipeline id (8), pass kind (1) and pass count (8).
        let view_id = ViewId::new(71).get().to_be_bytes();
        let view_id_offset = frame
            .windows(view_id.len())
            .position(|window| window == view_id)
            .expect("the fixture attachment view id is on the wire");
        let attachment_count = view_id_offset - 8;
        let pass_kind = view_id_offset - 17;
        let pass_count = view_id_offset - 25;
        assert_eq!(frame[pass_kind], 0x01, "render pass kind tag");

        let mut unknown = frame.clone();
        unknown[pass_kind] = 0x7e;
        assert!(matches!(
            CommandCodec::decode_request(&unknown).unwrap_err(),
            CodecError::UnknownPassTag(0x7e)
        ));

        // The pipeline table is a tagged list too: its entry kind sits right
        // after the frame kind, the request tag, the schema version, the epoch,
        // the operation id and the entry count.
        let entry_kind = 4 + 1 + 4 + 1 + 2 + 8 + 8 + 8;
        assert_eq!(frame[entry_kind], 0x01, "render pipeline entry tag");
        let mut unknown_entry = frame.clone();
        unknown_entry[entry_kind] = 0x7e;
        assert!(matches!(
            CommandCodec::decode_request(&unknown_entry).unwrap_err(),
            CodecError::UnknownPipelineTag(0x7e)
        ));

        let mut oversize_attachments = frame.clone();
        oversize_attachments[attachment_count..attachment_count + 8]
            .copy_from_slice(&((MAX_COLOR_ATTACHMENTS + 1) as u64).to_be_bytes());
        assert!(matches!(
            CommandCodec::decode_request(&oversize_attachments).unwrap_err(),
            CodecError::ColorAttachmentCount { count, maximum }
                if count == MAX_COLOR_ATTACHMENTS + 1 && maximum == MAX_COLOR_ATTACHMENTS
        ));

        let mut oversize_passes = frame.clone();
        oversize_passes[pass_count..pass_count + 8]
            .copy_from_slice(&((MAX_TAGGED_TRACE_PASSES + 1) as u64).to_be_bytes());
        assert!(matches!(
            CommandCodec::decode_request(&oversize_passes).unwrap_err(),
            CodecError::TracePassCount { count, maximum }
                if count == MAX_TAGGED_TRACE_PASSES + 1 && maximum == MAX_TAGGED_TRACE_PASSES
        ));

        // A frame that declares its own (short) length must fail inside the
        // payload rather than decoding a partial trace.
        let payload = &frame[9..];
        let mut truncated = b"MCC1".to_vec();
        truncated.push(0x01);
        truncated.extend_from_slice(&((payload.len() - 8) as u32).to_be_bytes());
        truncated.extend_from_slice(&payload[..payload.len() - 8]);
        assert!(matches!(
            CommandCodec::decode_request(&truncated).unwrap_err(),
            CodecError::TruncatedPayload { .. }
        ));

        // The encoder refuses the same protocol bound instead of writing a
        // frame the decoder would reject.
        let mut trace = render_only_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("fixture trace carries a render pass");
        };
        pass.color_attachments.push(render_attachment(2, 2));
        assert!(matches!(
            CommandCodec::encode_request(&CommandRequest::Submit {
                trace,
                resources: resources(),
            })
            .unwrap_err(),
            CodecError::ColorAttachmentCount { count, maximum }
                if count == MAX_COLOR_ATTACHMENTS + 1 && maximum == MAX_COLOR_ATTACHMENTS
        ));
    }

    /// The bytes `encode_request` produced at commit `6ba8834` for the trace
    /// [`render_only_trace`] builds, i.e. before the present action existed.
    /// Captured by encoding the same fixture in a clean checkout of that commit.
    ///
    /// An offscreen render pass has to keep producing exactly these bytes: the
    /// present tag is additive, so a provider built before it keeps decoding
    /// every render frame it decoded before (`research/docs/24` §4.3).
    const LEGACY_RENDER_SUBMIT_FRAME: &[u8] = &[
        77, 67, 67, 49, 1, 0, 0, 1, 113, 15, 0, 2, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 21,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0,
        0, 0, 10, 102, 105, 120, 116, 117, 114, 101, 45, 118, 49, 0, 0, 0, 0, 0, 0, 0, 9, 99, 111,
        112, 121, 95, 119, 111, 114, 100, 0, 0, 0, 0, 0, 0, 0, 9, 99, 111, 112, 121, 95, 119, 111,
        114, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 98, 117, 102, 102,
        101, 114, 45, 119, 114, 105, 116, 101, 1, 0, 0, 0, 0, 0, 0, 0, 10, 116, 114, 97, 110, 115,
        108, 97, 116, 111, 114, 0, 0, 0, 0, 0, 0, 0, 2, 118, 49, 0, 0, 0, 0, 0, 0, 0, 18, 102, 117,
        108, 108, 95, 115, 99, 114, 101, 101, 110, 95, 118, 101, 114, 116, 101, 120, 0, 0, 0, 0, 0,
        0, 0, 20, 115, 111, 108, 105, 100, 95, 99, 111, 108, 111, 114, 95, 102, 114, 97, 103, 109,
        101, 110, 116, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0, 41, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
        0, 0, 0, 0, 0, 2, 0, 254, 254, 254, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2,
        0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 7, 0,
        0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// The bytes `encode_request` produces for the same fixture with its render
    /// pass handing its own attachment on (`research/docs/24` §4.1, shape one).
    ///
    /// This is the present half's hardcoded frame: the payload tag, the pass
    /// kind, the field order and the two optional discriminators are pinned, so
    /// a later change to the present section cannot pass by rewriting the
    /// fixture alongside itself.
    const PRESENT_SUBMIT_FRAME: &[u8] = &[
        77, 67, 67, 49, 1, 0, 0, 1, 173, 15, 0, 2, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 21,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0,
        0, 0, 10, 102, 105, 120, 116, 117, 114, 101, 45, 118, 49, 0, 0, 0, 0, 0, 0, 0, 9, 99, 111,
        112, 121, 95, 119, 111, 114, 100, 0, 0, 0, 0, 0, 0, 0, 9, 99, 111, 112, 121, 95, 119, 111,
        114, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 98, 117, 102, 102,
        101, 114, 45, 119, 114, 105, 116, 101, 1, 0, 0, 0, 0, 0, 0, 0, 10, 116, 114, 97, 110, 115,
        108, 97, 116, 111, 114, 0, 0, 0, 0, 0, 0, 0, 2, 118, 49, 0, 0, 0, 0, 0, 0, 0, 18, 102, 117,
        108, 108, 95, 115, 99, 114, 101, 101, 110, 95, 118, 101, 114, 116, 101, 120, 0, 0, 0, 0, 0,
        0, 0, 20, 115, 111, 108, 105, 100, 95, 99, 111, 108, 111, 114, 95, 102, 114, 97, 103, 109,
        101, 110, 116, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0, 41, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
        0, 0, 0, 0, 0, 2, 0, 254, 254, 254, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2,
        0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 71, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0,
        0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 64, 128, 192, 255, 0, 0, 0, 0,
        0, 0, 0, 71, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0,
        7, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// Byte offsets inside [`PRESENT_SUBMIT_FRAME`]'s present section, derived
    /// from the field order a decoder reads (`docs/24` §3.1). They are stated
    /// as offsets from the section's first byte so the mutation cases below
    /// change one field rather than a hand-counted position in the whole frame.
    const PRESENT_INITIAL_TAG: usize = 37;
    const PRESENT_SENTINEL_LENGTH: usize = 38;
    const PRESENT_MODE: usize = 58;
    const PRESENT_ACQUIRE: usize = 59;

    /// The fixture's render pass handing its own attachment on: the smallest
    /// trace that carries a present action (`docs/24` §3.5, shape one).
    fn presenting_trace() -> ComputeTrace {
        let mut trace = render_only_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.present = Some(PresentDescriptor {
            target: PresentTarget {
                allocation_id: AllocationId::new(41),
                view_id: ViewId::new(71),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                image_count: MAX_PRESENT_IMAGE_COUNT,
                initial: InitialState::Sentinel(vec![0x40, 0x80, 0xc0, 0xff]),
            },
            source: ViewId::new(71),
            mode: PresentMode::Fifo,
            acquire: AcquirePolicy::Blocking,
        });
        trace
    }

    /// A compute-only trace that also carries a heap payload: two disjoint
    /// buffer placements inside one `OwnedBytes` heap (`research/docs/25`
    /// §4.2).
    fn heap_trace() -> ComputeTrace {
        let compiled = pipeline(&compile_request());
        let mut trace = trace(&compiled);
        trace.heap = Some(Box::new(HeapPayload {
            descriptor: HeapDescriptor {
                size: 16,
                storage_mode: StorageMode::OwnedBytes,
                allows_aliasing: false,
            },
            placements: vec![
                HeapPlacement {
                    heap_id: HeapId::new(61),
                    offset: 0,
                    resource: HeapResource::Buffer { byte_size: 8 },
                },
                HeapPlacement {
                    heap_id: HeapId::new(61),
                    offset: 8,
                    resource: HeapResource::Buffer { byte_size: 8 },
                },
            ],
        }));
        trace
    }

    /// A compute-only trace that also carries an indirect-command payload: one
    /// non-indexed draw replayed from a fixed four-slot ICB (`research/docs/25`
    /// §4.3).
    fn icb_trace() -> ComputeTrace {
        let compiled = pipeline(&compile_request());
        let mut trace = trace(&compiled);
        trace.indirect = Some(Box::new(IndirectCommandPayload {
            buffer: IndirectCommandBufferDescriptor {
                max_commands: 4,
                kinds: vec![IndirectCommandKind::Draw],
            },
            command: IndirectCommandDescriptor::Draw {
                vertex_count: 3,
                instance_count: 1,
            },
            range: IndirectCommandRange { start: 0, count: 1 },
        }));
        trace
    }

    /// The bytes `encode_request` produces for [`heap_trace`]: the payload tag,
    /// the tagged pass layout and the heap/ICB tail are pinned, so a later
    /// change to the heap section cannot pass by rewriting the fixture
    /// alongside itself (`research/docs/25-heaps与ICB设计.md` §4.5).
    const HEAP_SUBMIT_FRAME: &[u8] = &[
        77, 67, 67, 49, 1, 0, 0, 1, 176, 16, 0, 2, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 21,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0,
        0, 0, 10, 102, 105, 120, 116, 117, 114, 101, 45, 118, 49, 0, 0, 0, 0, 0, 0, 0, 9, 99, 111,
        112, 121, 95, 119, 111, 114, 100, 0, 0, 0, 0, 0, 0, 0, 9, 99, 111, 112, 121, 95, 119, 111,
        114, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 98, 117, 102, 102,
        101, 114, 45, 119, 114, 105, 116, 101, 1, 0, 0, 0, 0, 0, 0, 0, 10, 116, 114, 97, 110, 115,
        108, 97, 116, 111, 114, 0, 0, 0, 0, 0, 0, 0, 2, 118, 49, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 4, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
        61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 61, 0, 0, 0, 0,
        0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 41,
        0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// The bytes `encode_request` produces for [`icb_trace`]: the same pinned
    /// tail policy as [`HEAP_SUBMIT_FRAME`], with the indirect-command payload
    /// in place of the heap payload.
    const ICB_SUBMIT_FRAME: &[u8] = &[
        77, 67, 67, 49, 1, 0, 0, 1, 138, 16, 0, 2, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 21,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0,
        0, 0, 10, 102, 105, 120, 116, 117, 114, 101, 45, 118, 49, 0, 0, 0, 0, 0, 0, 0, 9, 99, 111,
        112, 121, 95, 119, 111, 114, 100, 0, 0, 0, 0, 0, 0, 0, 9, 99, 111, 112, 121, 95, 119, 111,
        114, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 98, 117, 102, 102,
        101, 114, 45, 119, 114, 105, 116, 101, 1, 0, 0, 0, 0, 0, 0, 0, 10, 116, 114, 97, 110, 115,
        108, 97, 116, 111, 114, 0, 0, 0, 0, 0, 0, 0, 2, 118, 49, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 4, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 7,
        0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// The offset of the pass-kind byte in a frame whose only pass is the
    /// fixture render pass. Every other identity in the frame is a smaller
    /// number, so the first eight-byte big-endian 71 is the attachment's view
    /// id, and the attachment count, the pipeline id and the pass kind precede
    /// it.
    fn render_pass_kind_offset(frame: &[u8]) -> usize {
        let view_id = ViewId::new(71).get().to_be_bytes();
        frame
            .windows(view_id.len())
            .position(|window| window == view_id)
            .expect("the fixture attachment view id is on the wire")
            - 17
    }

    /// The offset of the present section in a presenting frame. Its first two
    /// identities are the target's allocation and view, a pair that appears
    /// nowhere else: the render attachment writes the same pair in the other
    /// order.
    fn present_section_offset(frame: &[u8]) -> usize {
        let mut marker = AllocationId::new(41).get().to_be_bytes().to_vec();
        marker.extend_from_slice(&ViewId::new(71).get().to_be_bytes());
        frame
            .windows(marker.len())
            .position(|window| window == marker)
            .expect("the present target identity pair is on the wire")
    }

    /// The byte offset of the first placement's resource kind inside a
    /// [`heap_trace`] frame. The heap id 61 appears nowhere else in the
    /// fixture, so its first occurrence locates the placement, and the kind
    /// sits right after the placement's heap id and offset.
    fn heap_resource_kind_offset(frame: &[u8]) -> usize {
        let marker = HeapId::new(61).get().to_be_bytes();
        let position = frame
            .windows(marker.len())
            .position(|window| window == marker)
            .expect("the heap placement identity is on the wire");
        position + marker.len() + 8
    }

    /// The byte offset of the indirect command kind inside an [`icb_trace`]
    /// frame. The command's vertex count (`3`) and instance count (`1`) form
    /// an eight-byte sequence that appears nowhere else in the fixture, so the
    /// kind is the byte that immediately precedes it.
    fn indirect_command_kind_offset(frame: &[u8]) -> usize {
        let mut marker = 3_u32.to_be_bytes().to_vec();
        marker.extend_from_slice(&1_u32.to_be_bytes());
        let position = frame
            .windows(marker.len())
            .position(|window| window == marker)
            .expect("the indirect command counts are on the wire");
        position - 1
    }

    #[test]
    fn render_frames_without_a_present_keep_their_pre_present_bytes() {
        let request = CommandRequest::Submit {
            trace: render_only_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        // Byte for byte what the render track published before the present
        // action existed: an offscreen pass is not a presenting pass, so it
        // must not pay for the present half (`docs/24` §4.3).
        assert_eq!(frame, LEGACY_RENDER_SUBMIT_FRAME);
        assert_eq!(frame[9], 0x0f);
        assert_eq!(frame[render_pass_kind_offset(&frame)], 0x01);
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);

        // And it decodes with no present action, which is what makes the
        // compute-only and offscreen paths keep the pre-present behaviour.
        let decoded = CommandCodec::decode_request(&frame).unwrap();
        let CommandRequest::Submit { trace, .. } = &decoded else {
            panic!("a render submit decodes as a submit");
        };
        assert!(trace.passes[0]
            .as_render()
            .expect("the fixture is a render pass")
            .present
            .is_none());
    }

    #[test]
    fn present_frames_use_their_own_pass_kind_and_round_trip() {
        let request = CommandRequest::Submit {
            trace: presenting_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        // The present half rides the submit tag shape one already uses, so no
        // new payload tag was needed (`docs/24` §4.1) and the completion, lease
        // and borrowed channels are untouched.
        assert_eq!(frame[9], 0x0f);
        assert_eq!(frame, PRESENT_SUBMIT_FRAME);
        assert_eq!(
            frame[render_pass_kind_offset(&frame)],
            0x02,
            "a presenting pass carries its own pass kind"
        );

        // Every field of the Step 1 value type survives the round trip,
        // including the two optional discriminators.
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        let decoded = CommandCodec::decode_request(&frame).unwrap();
        let CommandRequest::Submit { trace, .. } = &decoded else {
            panic!("a present submit decodes as a submit");
        };
        let present = trace.passes[0]
            .as_render()
            .expect("the fixture is a render pass")
            .present
            .as_ref()
            .expect("the fixture carries a present");
        assert_eq!(present.target.allocation_id, AllocationId::new(41));
        assert_eq!(present.target.view_id, ViewId::new(71));
        assert_eq!(present.target.image_count, MAX_PRESENT_IMAGE_COUNT);
        assert_eq!(
            present.target.initial,
            InitialState::Sentinel(vec![0x40, 0x80, 0xc0, 0xff])
        );
        assert_eq!(present.source, ViewId::new(71));
        assert_eq!(present.mode, PresentMode::Fifo);
        assert_eq!(present.acquire, AcquirePolicy::Blocking);

        // The two values this increment refuses but keeps expressible on the
        // wire travel too: a trace that asks for a deadline or for an undefined
        // target is decoded, and the refusal is core admission's job rather
        // than a decode-time downgrade (`docs/24` §3.1).
        let mut expressive = presenting_trace();
        let Some(TracePass::Render(pass)) = expressive.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        let carried = pass
            .present
            .as_mut()
            .expect("the fixture carries a present");
        carried.target.initial = InitialState::Undefined;
        carried.acquire = AcquirePolicy::Timeout(5);
        let request = CommandRequest::Submit {
            trace: expressive,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
    }

    #[test]
    fn present_frames_refuse_truncation_oversize_and_unknown_kinds() {
        let request = CommandRequest::Submit {
            trace: presenting_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        let present = present_section_offset(&frame);

        // An unknown pass kind is a tag the decoder refuses, so a frame that
        // claims a pass shape this build does not know cannot be read as a
        // render pass with a stray tail.
        let mut unknown_pass = frame.clone();
        let pass_kind = render_pass_kind_offset(&frame);
        unknown_pass[pass_kind] = 0x7e;
        assert!(matches!(
            CommandCodec::decode_request(&unknown_pass).unwrap_err(),
            CodecError::UnknownPassTag(0x7e)
        ));

        // The three closed values inside the present section each refuse an
        // unknown code instead of folding to a default.
        let mut unknown_initial = frame.clone();
        unknown_initial[present + PRESENT_INITIAL_TAG] = 0x02;
        assert!(matches!(
            CommandCodec::decode_request(&unknown_initial).unwrap_err(),
            CodecError::UnknownEnumValue {
                field: "present initial state",
                value: 0x02
            }
        ));
        let mut unknown_mode = frame.clone();
        unknown_mode[present + PRESENT_MODE] = 0x04;
        assert!(matches!(
            CommandCodec::decode_request(&unknown_mode).unwrap_err(),
            CodecError::UnknownEnumValue {
                field: "present mode",
                value: 0x04
            }
        ));
        let mut unknown_acquire = frame.clone();
        unknown_acquire[present + PRESENT_ACQUIRE] = 0x02;
        assert!(matches!(
            CommandCodec::decode_request(&unknown_acquire).unwrap_err(),
            CodecError::UnknownEnumValue {
                field: "present acquire policy",
                value: 0x02
            }
        ));

        // An oversized sentinel length is refused by the present bound before
        // the decoder treats it as the rest of the frame.
        let mut oversize = frame.clone();
        oversize[present + PRESENT_SENTINEL_LENGTH..present + PRESENT_SENTINEL_LENGTH + 8]
            .copy_from_slice(&((MAX_PRESENT_SENTINEL_BYTES + 1) as u64).to_be_bytes());
        assert!(matches!(
            CommandCodec::decode_request(&oversize).unwrap_err(),
            CodecError::PresentSentinelLength { length, maximum }
                if length == MAX_PRESENT_SENTINEL_BYTES + 1
                    && maximum == MAX_PRESENT_SENTINEL_BYTES
        ));

        // A frame that declares its own (short) length must fail inside the
        // payload rather than decoding a partial present.
        let payload = &frame[9..];
        let mut truncated = b"MCC1".to_vec();
        truncated.push(0x01);
        truncated.extend_from_slice(&((payload.len() - 8) as u32).to_be_bytes());
        truncated.extend_from_slice(&payload[..payload.len() - 8]);
        assert!(matches!(
            CommandCodec::decode_request(&truncated).unwrap_err(),
            CodecError::TruncatedPayload { .. }
        ));

        // The encoder refuses the same protocol bound instead of writing a
        // frame its own decoder would reject.
        let mut trace = presenting_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.present
            .as_mut()
            .expect("the fixture carries a present")
            .target
            .initial = InitialState::Sentinel(vec![0; MAX_PRESENT_SENTINEL_BYTES + 1]);
        assert!(matches!(
            CommandCodec::encode_request(&CommandRequest::Submit {
                trace,
                resources: resources(),
            })
            .unwrap_err(),
            CodecError::PresentSentinelLength { length, maximum }
                if length == MAX_PRESENT_SENTINEL_BYTES + 1
                    && maximum == MAX_PRESENT_SENTINEL_BYTES
        ));
    }

    #[test]
    fn heap_frames_use_their_own_tag_and_round_trip() {
        let request = CommandRequest::Submit {
            trace: heap_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        // The heap payload rides its own submit tag: an older decoder answers
        // `UnknownCommandTag` instead of misreading the tagged heap/ICB tail
        // (`research/docs/25-heaps与ICB设计.md` §4.5).
        assert_eq!(frame[9], 0x10);
        assert_eq!(frame, HEAP_SUBMIT_FRAME);

        // Every field of the Step 1 value types survives the round trip.
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        let decoded = CommandCodec::decode_request(&frame).unwrap();
        let CommandRequest::Submit { trace, .. } = &decoded else {
            panic!("a heap submit decodes as a submit");
        };
        let heap = trace.heap.as_ref().expect("the fixture carries a heap");
        assert_eq!(heap.descriptor.size, 16);
        assert_eq!(heap.descriptor.storage_mode, StorageMode::OwnedBytes);
        assert!(!heap.descriptor.allows_aliasing);
        assert_eq!(heap.placements.len(), 2);
        assert_eq!(heap.placements[0].heap_id, HeapId::new(61));
        assert_eq!(heap.placements[0].offset, 0);
        assert_eq!(
            heap.placements[0].resource,
            HeapResource::Buffer { byte_size: 8 }
        );
        assert_eq!(heap.placements[1].offset, 8);
    }

    #[test]
    fn icb_frames_use_their_own_tag_and_round_trip() {
        let request = CommandRequest::Submit {
            trace: icb_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(frame[9], 0x10);
        assert_eq!(frame, ICB_SUBMIT_FRAME);

        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        let decoded = CommandCodec::decode_request(&frame).unwrap();
        let CommandRequest::Submit { trace, .. } = &decoded else {
            panic!("an ICB submit decodes as a submit");
        };
        let indirect = trace.indirect.as_ref().expect("the fixture carries an ICB");
        assert_eq!(indirect.buffer.max_commands, 4);
        assert_eq!(indirect.buffer.kinds, vec![IndirectCommandKind::Draw]);
        assert_eq!(
            indirect.command,
            IndirectCommandDescriptor::Draw {
                vertex_count: 3,
                instance_count: 1
            }
        );
        assert_eq!(indirect.range, IndirectCommandRange { start: 0, count: 1 });
    }

    #[test]
    fn traces_without_heap_or_icb_keep_their_pre_heap_icb_bytes() {
        // A compute-only trace without a heap or ICB payload must not pay for
        // the heap/ICB tail: it keeps the exact bytes the pre-heap/ICB build
        // published (`docs/25-heaps与ICB设计.md` §4.5).
        let compiled = pipeline(&compile_request());
        let request = CommandRequest::Submit {
            trace: trace(&compiled),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(frame, LEGACY_SUBMIT_FRAME);
        assert_eq!(frame[9], 0x03);

        // And the offscreen render frame stays put too.
        let request = CommandRequest::Submit {
            trace: render_only_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(frame, LEGACY_RENDER_SUBMIT_FRAME);
        assert_eq!(frame[9], 0x0f);
    }

    #[test]
    fn heap_and_icb_frames_refuse_unknown_kinds_and_oversize() {
        // The heap resource kind is a closed two-value family: an unknown code
        // is a decoder refusal rather than a silent default.
        let frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: heap_trace(),
            resources: resources(),
        })
        .unwrap();
        let mut unknown_kind = frame.clone();
        unknown_kind[heap_resource_kind_offset(&frame)] = 0x7e;
        assert!(matches!(
            CommandCodec::decode_request(&unknown_kind).unwrap_err(),
            CodecError::UnknownEnumValue {
                field: "heap resource kind",
                value: 0x7e
            }
        ));

        // The indirect command kind is a closed three-value family, refused the
        // same way.
        let frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: icb_trace(),
            resources: resources(),
        })
        .unwrap();
        let mut unknown_kind = frame.clone();
        unknown_kind[indirect_command_kind_offset(&frame)] = 0x7e;
        assert!(matches!(
            CommandCodec::decode_request(&unknown_kind).unwrap_err(),
            CodecError::UnknownEnumValue {
                field: "indirect command kind",
                value: 0x7e
            }
        ));

        // The encoder refuses an oversized placement list before writing a
        // frame its own decoder would reject.
        let mut trace = heap_trace();
        let heap = trace.heap.as_mut().expect("the fixture carries a heap");
        heap.placements = (0..MAX_HEAP_PLACEMENTS + 1)
            .map(|index| HeapPlacement {
                heap_id: HeapId::new(61),
                offset: (index as u64) * 8,
                resource: HeapResource::Buffer { byte_size: 8 },
            })
            .collect();
        assert!(matches!(
            CommandCodec::encode_request(&CommandRequest::Submit {
                trace,
                resources: resources(),
            })
            .unwrap_err(),
            CodecError::HeapPlacementCount { count, maximum }
                if count == MAX_HEAP_PLACEMENTS + 1 && maximum == MAX_HEAP_PLACEMENTS
        ));

        // The encoder refuses an oversized indirect-command kind list too.
        let mut trace = icb_trace();
        let indirect = trace.indirect.as_mut().expect("the fixture carries an ICB");
        indirect.buffer.kinds = (0..MAX_SUPPORTED_INDIRECT_COMMANDS + 1)
            .map(|_| IndirectCommandKind::Draw)
            .collect();
        assert!(matches!(
            CommandCodec::encode_request(&CommandRequest::Submit {
                trace,
                resources: resources(),
            })
            .unwrap_err(),
            CodecError::IndirectCommandKindCount { count, maximum }
                if count == MAX_SUPPORTED_INDIRECT_COMMANDS + 1
                    && maximum == MAX_SUPPORTED_INDIRECT_COMMANDS
        ));
    }

    #[test]
    fn capability_frames_declare_present_bits_under_their_own_tag() {
        // A snapshot with no extended bits keeps the legacy capability bytes,
        // and a decoder that reads them treats it as present-refusing: that is
        // what such a provider was (`docs/24` §4.2).
        let legacy = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(7),
            capabilities: fake_capabilities(),
        };
        let frame = CommandCodec::encode_response(&legacy).unwrap();
        assert_eq!(frame[9], 0x01);
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), legacy);
        let CommandResponse::Capabilities { capabilities, .. } =
            CommandCodec::decode_response(&frame).unwrap()
        else {
            panic!("a capability response decodes as a capability response");
        };
        assert!(!capabilities.supports_presentation);
        assert_eq!(capabilities.max_present_targets, 0);
        assert!(capabilities.supported_present_modes.is_empty());
        assert_eq!(capabilities.max_present_image_count, 0);

        // A snapshot that declares presentation without any render bit still
        // needs the extended payload, because that is where the present bits
        // travel. This is the case the expanded `declares_render_support`
        // predicate exists for.
        let mut presenting = fake_capabilities();
        presenting.supports_presentation = true;
        presenting.max_present_targets = 1;
        presenting.supported_present_modes = vec![PresentMode::Fifo];
        presenting.max_present_image_count = MAX_PRESENT_IMAGE_COUNT;
        let declaring = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(7),
            capabilities: presenting,
        };
        let frame = CommandCodec::encode_response(&declaring).unwrap();
        assert_eq!(frame[9], 0x0a);
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), declaring);

        // Both halves travel in the same frame, so a snapshot that declares
        // render and present together round trips too.
        let mut both = fake_capabilities();
        both.supports_render_passes = true;
        both.max_color_attachments = 1;
        both.max_attachment_dimension = [2, 2];
        both.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        both.supports_presentation = true;
        both.max_present_targets = 1;
        both.supported_present_modes = vec![PresentMode::Fifo];
        both.max_present_image_count = MAX_PRESENT_IMAGE_COUNT;
        let declaring = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(7),
            capabilities: both,
        };
        let frame = CommandCodec::encode_response(&declaring).unwrap();
        assert_eq!(frame[9], 0x0a);
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), declaring);

        // The mode list carries a bound like the colour-format list does, so a
        // corrupt count cannot drive the decoder or the encoder.
        let mut oversize = fake_capabilities();
        oversize.supports_presentation = true;
        oversize.supported_present_modes = vec![PresentMode::Fifo; MAX_SUPPORTED_PRESENT_MODES + 1];
        assert!(matches!(
            CommandCodec::encode_response(&CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(7),
                capabilities: oversize,
            })
            .unwrap_err(),
            CodecError::PresentModeCount { count, maximum }
                if count == MAX_SUPPORTED_PRESENT_MODES + 1
                    && maximum == MAX_SUPPORTED_PRESENT_MODES
        ));
    }

    #[test]
    fn capability_frames_declare_heap_and_icb_bits_under_their_own_tag() {
        // A snapshot with no heap or ICB bits keeps the legacy capability
        // bytes, and a decoder that reads them treats it as heap/ICB-refusing:
        // that is what such a provider was (`docs/25` §4.1).
        let legacy = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(7),
            capabilities: fake_capabilities(),
        };
        let frame = CommandCodec::encode_response(&legacy).unwrap();
        assert_eq!(frame[9], 0x01);
        let CommandResponse::Capabilities { capabilities, .. } =
            CommandCodec::decode_response(&frame).unwrap()
        else {
            panic!("a capability response decodes as a capability response");
        };
        assert!(!capabilities.supports_heaps);
        assert_eq!(capabilities.max_heap_bytes, 0);
        assert!(capabilities.supported_heap_storage_modes.is_empty());
        assert!(!capabilities.supports_heap_aliasing);
        assert!(!capabilities.supports_indirect_command_buffers);
        assert_eq!(capabilities.max_indirect_commands, 0);
        assert!(capabilities.supported_indirect_commands.is_empty());

        // A snapshot that declares only heap bits still needs the extended
        // payload, because that is where the heap/ICB bits travel.
        let mut heaps = fake_capabilities();
        heaps.supports_heaps = true;
        heaps.max_heap_bytes = 64;
        heaps.supported_heap_storage_modes = vec![StorageMode::OwnedBytes];
        let declaring = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(7),
            capabilities: heaps,
        };
        let frame = CommandCodec::encode_response(&declaring).unwrap();
        assert_eq!(frame[9], 0x0a);
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), declaring);

        // A snapshot that declares only ICB bits behaves the same way.
        let mut icbs = fake_capabilities();
        icbs.supports_indirect_command_buffers = true;
        icbs.max_indirect_commands = 4;
        icbs.supported_indirect_commands = vec![IndirectCommandKind::Draw];
        let declaring = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(7),
            capabilities: icbs,
        };
        let frame = CommandCodec::encode_response(&declaring).unwrap();
        assert_eq!(frame[9], 0x0a);
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), declaring);

        // The heap storage-mode list carries a bound like the present-mode
        // list does, so a corrupt count cannot drive the encoder.
        let mut oversize = fake_capabilities();
        oversize.supports_heaps = true;
        oversize.supported_heap_storage_modes =
            vec![StorageMode::OwnedBytes; MAX_SUPPORTED_HEAP_STORAGE_MODES + 1];
        assert!(matches!(
            CommandCodec::encode_response(&CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(7),
                capabilities: oversize,
            })
            .unwrap_err(),
            CodecError::HeapStorageModeCount { count, maximum }
                if count == MAX_SUPPORTED_HEAP_STORAGE_MODES + 1
                    && maximum == MAX_SUPPORTED_HEAP_STORAGE_MODES
        ));

        // The indirect-command kind list carries the same guard.
        let mut oversize = fake_capabilities();
        oversize.supports_indirect_command_buffers = true;
        oversize.supported_indirect_commands =
            vec![IndirectCommandKind::Draw; MAX_SUPPORTED_INDIRECT_COMMANDS + 1];
        assert!(matches!(
            CommandCodec::encode_response(&CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(7),
                capabilities: oversize,
            })
            .unwrap_err(),
            CodecError::IndirectCommandKindCount { count, maximum }
                if count == MAX_SUPPORTED_INDIRECT_COMMANDS + 1
                    && maximum == MAX_SUPPORTED_INDIRECT_COMMANDS
        ));
    }

    #[test]
    fn round_trips_every_request_and_response() {
        let request = compile_request();
        let compiled = pipeline(&request);
        let trace = trace(&compiled);
        let requests = [
            CommandRequest::Capabilities,
            CommandRequest::Health,
            CommandRequest::Compile {
                request: request.clone(),
            },
            CommandRequest::ImportStagedLease {
                staged: StagedLease::new(
                    LeaseReservation {
                        lease: BufferLease {
                            lease_id: LeaseId::new(51),
                            allocation_id: AllocationId::new(41),
                            owner_epoch: DeviceEpoch::new(7),
                        },
                        offset: 0,
                        length: 4,
                    },
                    vec![1, 2, 3, 4],
                )
                .unwrap(),
            },
            CommandRequest::ReleaseStagedLease {
                lease_id: LeaseId::new(51),
            },
            CommandRequest::ImportBorrowedLease {
                reservation: LeaseReservation {
                    lease: BufferLease {
                        lease_id: LeaseId::new(52),
                        allocation_id: AllocationId::new(41),
                        owner_epoch: DeviceEpoch::new(7),
                    },
                    offset: 0,
                    length: 4096,
                },
            },
            CommandRequest::ReleaseBorrowedLease {
                lease_id: LeaseId::new(52),
            },
            CommandRequest::SetQueuePriorities {
                tiers: vec![
                    QueuePriority::High,
                    QueuePriority::Default,
                    QueuePriority::Low,
                ],
            },
            CommandRequest::Submit {
                trace: trace.clone(),
                resources: resources(),
            },
            CommandRequest::Wait {
                token: token(),
                timeout: Duration::from_millis(2500),
            },
            CommandRequest::Readback { token: token() },
            CommandRequest::Cancel { token: token() },
            CommandRequest::ReleasePipeline {
                pipeline: compiled.clone(),
            },
            CommandRequest::ReleaseCompletion { token: token() },
        ];
        for request in requests {
            let frame = CommandCodec::encode_request(&request).unwrap();
            assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        }

        // The wire format carries the field even though provider admission
        // refuses a non-None attribute stride.
        let mut strided = trace.clone();
        strided.passes[0]
            .as_compute_mut()
            .expect("compute trace entry")
            .buffers[0]
            .attribute_stride = Some(16);
        let request = CommandRequest::Submit {
            trace: strided,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);

        // The wire format carries texture bindings even though provider
        // admission refuses them until a provider executes them
        // (research/docs/16 §4.2).
        let mut textured = trace.clone();
        textured.passes[0]
            .as_compute_mut()
            .expect("compute trace entry")
            .textures
            .push(TextureView {
                view_id: ViewId::new(71),
                metal_binding: 0,
                allocation_id: AllocationId::new(41),
                texture_type: TextureType::D2,
                format: TextureFormat::R32Uint,
                width: 4,
                height: 4,
                depth: 1,
                array_length: 1,
                sample_count: 1,
                access: TextureAccess::Sampled,
                source: TextureSource::OwnedBytes(vec![0x5a; 64]),
            });
        let request = CommandRequest::Submit {
            trace: textured,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);

        let writeback = BufferWriteback {
            view_id: ViewId::new(31),
            allocation_id: AllocationId::new(41),
            offset: 0,
            bytes: vec![9, 8, 7, 6],
        };
        let mut error = ProviderError::new(
            ProviderPhase::Resolve,
            ProviderErrorClass::Capability,
            "storage_mode_unsupported",
        )
        .unwrap();
        error.retryability = Retryability::RetrySameTrace;
        error.detail = Some("detail".into());
        error.fields.insert(
            "lease".into(),
            metal_api_core::provider::FieldValue::Unsigned(9),
        );
        let responses = [
            CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(7),
                capabilities: ProviderCapabilities {
                    max_passes: 2,
                    supports_threads_exact: true,
                    supports_threadgroups: false,
                    supports_serial: true,
                    supports_concurrent: false,
                    max_local_size: [1, 1, 1],
                    max_invocations: 4,
                    max_group_count: [1, 1, 1],
                    max_storage_buffer_descriptors: 2,
                    max_buffer_range: 4096,
                    max_push_constant_bytes: 0,
                    alias_mode: metal_api_core::provider::AliasMode::Refused,
                    storage_modes: vec![metal_api_core::provider::StorageMode::OwnedBytes],
                    host_readback: true,
                    submit_only: false,
                    supports_render_passes: false,
                    max_color_attachments: 0,
                    max_attachment_dimension: [0, 0],
                    supported_color_formats: Vec::new(),
                    supports_presentation: false,
                    max_present_targets: 0,
                    supported_present_modes: Vec::new(),
                    max_present_image_count: 0,
                    supports_heaps: false,
                    max_heap_bytes: 0,
                    supported_heap_storage_modes: Vec::new(),
                    supports_heap_aliasing: false,
                    supports_indirect_command_buffers: false,
                    max_indirect_commands: 0,
                    supported_indirect_commands: Vec::new(),
                },
            },
            CommandResponse::Compiled {
                pipeline: compiled.clone(),
            },
            CommandResponse::Health {
                health: ProviderHealth::Usable,
            },
            CommandResponse::Imported,
            CommandResponse::Submitted {
                submission: ProviderSubmission {
                    completion: CompletionDisposition::Submitted { token: token() },
                    writebacks: vec![writeback.clone()],
                },
            },
            CommandResponse::Observed {
                disposition: CompletionDisposition::CompletedVisible { token: token() },
            },
            CommandResponse::Readback {
                readback: CompletionReadback {
                    completion: CompletionDisposition::CompletedVisible { token: token() },
                    writebacks: vec![writeback],
                },
            },
            CommandResponse::QueuePriorities {
                installed: vec![QueuePriority::High, QueuePriority::Low],
            },
            CommandResponse::Released,
            CommandResponse::Error { error },
        ];
        for response in responses {
            let frame = CommandCodec::encode_response(&response).unwrap();
            assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        }
    }

    /// The frames a pre-priority owner sends keep decoding — and keep their
    /// exact bytes — while a marking travels under a tag no older decoder knew.
    ///
    /// `MCC1` is length-delimited value by value, so the carrier for a queue
    /// marking is a new request/response tag instead of a new field inside an
    /// existing frame: an unmarked connection stays byte for byte the
    /// connection this channel had before the tier table, and a decoder that
    /// predates the tag answers `UnknownCommandTag` for a marked frame rather
    /// than misreading it. That is the whole version policy — no new magic, and
    /// no change to `PROVIDER_SCHEMA_VERSION`, because no trace value moved.
    #[test]
    fn legacy_frames_still_decode_byte_for_byte() {
        // Hand-written rather than produced by the encoder below: these are the
        // bytes an older build puts on the wire.
        let legacy_health = [b'M', b'C', b'C', b'1', 0x01, 0, 0, 0, 1, 0x09];
        let legacy_capabilities = [b'M', b'C', b'C', b'1', 0x01, 0, 0, 0, 1, 0x01];
        assert_eq!(
            CommandCodec::decode_request(&legacy_health).unwrap(),
            CommandRequest::Health
        );
        assert_eq!(
            CommandCodec::decode_request(&legacy_capabilities).unwrap(),
            CommandRequest::Capabilities
        );
        // The encoder still emits exactly those bytes, so a mixed-version pair
        // cannot disagree about a frame neither side changed.
        assert_eq!(
            CommandCodec::encode_request(&CommandRequest::Health).unwrap(),
            legacy_health
        );
        assert_eq!(
            CommandCodec::encode_request(&CommandRequest::Capabilities).unwrap(),
            legacy_capabilities
        );

        // The marking is a new payload tag in both directions: the request
        // extends the request range, the response uses the value after
        // `IMPORTED_RESPONSE`.
        let marking =
            CommandCodec::encode_request(&CommandRequest::SetQueuePriorities { tiers: Vec::new() })
                .unwrap();
        assert_eq!(marking[9], 0x0e);
        let installed = CommandCodec::encode_response(&CommandResponse::QueuePriorities {
            installed: Vec::new(),
        })
        .unwrap();
        assert_eq!(installed[9], 0x09);

        // An unknown request tag is refused loudly, which is the answer an
        // older decoder gives for the two new tags above.
        let unknown = [b'M', b'C', b'C', b'1', 0x01, 0, 0, 0, 1, 0x7e];
        assert!(matches!(
            CommandCodec::decode_request(&unknown).unwrap_err(),
            CodecError::UnknownCommandTag(0x7e)
        ));
    }

    #[test]
    fn queue_priority_frames_round_trip_and_refuse_unknown_input() {
        let request = CommandRequest::SetQueuePriorities {
            tiers: vec![
                QueuePriority::Low,
                QueuePriority::Default,
                QueuePriority::High,
            ],
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        let response = CommandResponse::QueuePriorities {
            installed: vec![QueuePriority::High, QueuePriority::Low],
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);

        // A rank this build does not know is refused instead of aging into a
        // tier the owner never asked for.
        let mut corrupted = CommandCodec::encode_request(&request).unwrap();
        *corrupted.last_mut().unwrap() = 3;
        assert!(matches!(
            CommandCodec::decode_request(&corrupted).unwrap_err(),
            CodecError::UnknownEnumValue {
                field: "queue priority",
                value: 3
            }
        ));

        // The protocol bound is enforced on both sides, so a corrupt count
        // cannot make the decoder allocate.
        let oversized = vec![QueuePriority::Default; MAX_QUEUE_PRIORITIES + 1];
        assert!(matches!(
            CommandCodec::encode_request(&CommandRequest::SetQueuePriorities {
                tiers: oversized
            })
            .unwrap_err(),
            CodecError::QueuePriorityCount { count, maximum }
                if count == MAX_QUEUE_PRIORITIES + 1 && maximum == MAX_QUEUE_PRIORITIES
        ));
        let mut payload = vec![0x0e];
        payload.extend_from_slice(&((MAX_QUEUE_PRIORITIES + 1) as u64).to_be_bytes());
        let mut frame = b"MCC1".to_vec();
        frame.push(0x01);
        frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(&payload);
        assert!(matches!(
            CommandCodec::decode_request(&frame).unwrap_err(),
            CodecError::QueuePriorityCount { count, maximum }
                if count == MAX_QUEUE_PRIORITIES + 1 && maximum == MAX_QUEUE_PRIORITIES
        ));
    }

    struct FakeProvider {
        epoch: DeviceEpoch,
        capabilities: ProviderCapabilities,
        submissions: Arc<AtomicU64>,
        imports: Arc<AtomicU64>,
        borrowed: Arc<BorrowedLeaseRegistry>,
    }

    impl ComputeProvider for FakeProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            self.capabilities.clone()
        }

        fn submit(
            &self,
            trace: ValidatedComputeTrace,
        ) -> Result<ProviderSubmission, ProviderError> {
            self.submissions.fetch_add(1, Ordering::SeqCst);
            let token = CompletionToken {
                device_epoch: trace.trace().device_epoch,
                submission_id: SubmissionId::new(1),
            };
            Ok(ProviderSubmission {
                completion: CompletionDisposition::Submitted { token },
                writebacks: Vec::new(),
            })
        }

        fn wait(
            &self,
            token: CompletionToken,
            _timeout: Duration,
        ) -> Result<CompletionDisposition, ProviderError> {
            Ok(CompletionDisposition::CompletedVisible { token })
        }

        fn readback(&self, token: CompletionToken) -> Result<CompletionReadback, ProviderError> {
            Ok(CompletionReadback {
                completion: CompletionDisposition::CompletedVisible { token },
                writebacks: vec![BufferWriteback {
                    view_id: ViewId::new(31),
                    allocation_id: AllocationId::new(41),
                    offset: 0,
                    bytes: vec![4, 3, 2, 1],
                }],
            })
        }
    }

    impl PipelineProvider for FakeProvider {
        fn device_epoch(&self) -> DeviceEpoch {
            self.epoch
        }

        fn compile(
            &self,
            request: PipelineCompileRequest,
        ) -> Result<CompiledComputePipeline, ProviderError> {
            Ok(pipeline(&request))
        }

        fn release_pipeline(
            &self,
            _pipeline: &CompiledComputePipeline,
        ) -> Result<(), ProviderError> {
            Ok(())
        }

        fn release_completion(&self, _token: CompletionToken) -> Result<(), ProviderError> {
            Ok(())
        }
    }

    impl LeaseImporter for FakeProvider {
        fn import_staged_lease(&self, _staged: StagedLease) -> Result<(), ProviderError> {
            self.imports.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn release_staged_lease(&self, _lease_id: LeaseId) -> Result<(), ProviderError> {
            Ok(())
        }
    }

    impl NoCopyLeaseImporter for FakeProvider {
        fn no_copy_alignment(&self) -> u64 {
            4096
        }

        unsafe fn import_borrowed_lease(
            &self,
            borrowed: BorrowedLease,
        ) -> Result<(), ProviderError> {
            self.borrowed.import(borrowed)
        }

        fn release_borrowed_lease(&self, lease_id: LeaseId) -> Result<(), ProviderError> {
            self.borrowed.release(lease_id)
        }
    }

    /// `FakeProvider` plus the one behaviour the marking carrier needs: it
    /// remembers the queue table an owner installed. Everything else is
    /// forwarded, so the channel test above still pins the rest of the
    /// protocol.
    struct TierRecorderProvider {
        inner: FakeProvider,
        installed: Arc<Mutex<Vec<QueuePriority>>>,
    }

    impl ComputeProvider for TierRecorderProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            self.inner.capabilities()
        }

        fn set_queue_priorities(
            &self,
            tiers: &[QueuePriority],
        ) -> Result<Vec<QueuePriority>, ProviderError> {
            // A device expands the marking with the core policy
            // (`queue_priorities_for_device`); this recorder only has to be
            // honest about what reached it and what it answers.
            let mut installed = self.installed.lock().expect("tier recorder table");
            installed.clear();
            installed.extend_from_slice(tiers);
            Ok(installed.clone())
        }

        fn submit(
            &self,
            trace: ValidatedComputeTrace,
        ) -> Result<ProviderSubmission, ProviderError> {
            self.inner.submit(trace)
        }

        fn wait(
            &self,
            token: CompletionToken,
            timeout: Duration,
        ) -> Result<CompletionDisposition, ProviderError> {
            self.inner.wait(token, timeout)
        }

        fn readback(&self, token: CompletionToken) -> Result<CompletionReadback, ProviderError> {
            self.inner.readback(token)
        }
    }

    impl PipelineProvider for TierRecorderProvider {
        fn device_epoch(&self) -> DeviceEpoch {
            self.inner.device_epoch()
        }

        fn compile(
            &self,
            request: PipelineCompileRequest,
        ) -> Result<CompiledComputePipeline, ProviderError> {
            self.inner.compile(request)
        }

        fn release_pipeline(
            &self,
            pipeline: &CompiledComputePipeline,
        ) -> Result<(), ProviderError> {
            self.inner.release_pipeline(pipeline)
        }

        fn release_completion(&self, token: CompletionToken) -> Result<(), ProviderError> {
            self.inner.release_completion(token)
        }
    }

    impl LeaseImporter for TierRecorderProvider {
        fn import_staged_lease(&self, staged: StagedLease) -> Result<(), ProviderError> {
            self.inner.import_staged_lease(staged)
        }

        fn release_staged_lease(&self, lease_id: LeaseId) -> Result<(), ProviderError> {
            self.inner.release_staged_lease(lease_id)
        }
    }

    fn fake_capabilities() -> ProviderCapabilities {
        ProviderCapabilities {
            max_passes: 1,
            supports_threads_exact: true,
            supports_threadgroups: false,
            supports_serial: true,
            supports_concurrent: false,
            max_local_size: [1, 1, 1],
            max_invocations: 1,
            max_group_count: [1, 1, 1],
            max_storage_buffer_descriptors: 1,
            max_buffer_range: 1024,
            max_push_constant_bytes: 0,
            alias_mode: metal_api_core::provider::AliasMode::Refused,
            storage_modes: vec![metal_api_core::provider::StorageMode::OwnedBytes],
            host_readback: true,
            submit_only: false,
            supports_render_passes: false,
            max_color_attachments: 0,
            max_attachment_dimension: [0, 0],
            supported_color_formats: Vec::new(),
            supports_presentation: false,
            max_present_targets: 0,
            supported_present_modes: Vec::new(),
            max_present_image_count: 0,
            supports_heaps: false,
            max_heap_bytes: 0,
            supported_heap_storage_modes: Vec::new(),
            supports_heap_aliasing: false,
            supports_indirect_command_buffers: false,
            max_indirect_commands: 0,
            supported_indirect_commands: Vec::new(),
        }
    }

    /// An owner marking crosses the channel, and an owner that never sends one
    /// leaves the provider without a table — i.e. every queue at
    /// `QueuePriority::Default`, the scheduling this channel had before the
    /// tier table existed.
    #[cfg(unix)]
    #[test]
    fn remote_provider_carries_a_queue_priority_marking() {
        let installed = Arc::new(Mutex::new(Vec::new()));
        let provider = TierRecorderProvider {
            inner: FakeProvider {
                epoch: DeviceEpoch::new(7),
                capabilities: fake_capabilities(),
                submissions: Arc::new(AtomicU64::new(0)),
                imports: Arc::new(AtomicU64::new(0)),
                borrowed: Arc::new(BorrowedLeaseRegistry::new()),
            },
            installed: Arc::clone(&installed),
        };
        let (client, mut server) = super::unix::pair().unwrap();
        let server_thread = std::thread::spawn(move || serve_provider(&provider, &mut server));

        let remote = RemoteProvider::connect(client).unwrap();
        assert!(installed.lock().unwrap().is_empty());

        // The response is the table the provider read, which is what an owner
        // compares against to confirm the marking reached the scheduler.
        let marking = vec![
            QueuePriority::High,
            QueuePriority::Default,
            QueuePriority::Low,
        ];
        assert_eq!(remote.set_queue_priorities(&marking).unwrap(), marking);
        assert_eq!(installed.lock().unwrap().as_slice(), marking.as_slice());

        // A later marking replaces the table instead of stacking on it, and the
        // empty marking returns the device to the all-`Default` table.
        let cleared = remote
            .set_queue_priorities(&[])
            .expect("an empty marking is installed");
        assert!(cleared.is_empty());
        assert!(installed.lock().unwrap().is_empty());

        // Priority traffic leaves the rest of the channel framed and usable.
        assert_eq!(remote.health(), ProviderHealth::Usable);
        drop(remote);
        server_thread.join().unwrap().unwrap();
    }

    /// A provider without a scheduler refuses a marking, both through the trait
    /// default and over the wire, and the refusal is an answer rather than a
    /// framing error.
    #[cfg(unix)]
    #[test]
    fn a_provider_without_queue_scheduling_refuses_a_marking() {
        let provider = FakeProvider {
            epoch: DeviceEpoch::new(7),
            capabilities: fake_capabilities(),
            submissions: Arc::new(AtomicU64::new(0)),
            imports: Arc::new(AtomicU64::new(0)),
            borrowed: Arc::new(BorrowedLeaseRegistry::new()),
        };
        // `FakeProvider` never overrides the method, so this is the core
        // default: a capability question, not a protocol error.
        let direct = ComputeProvider::set_queue_priorities(&provider, &[QueuePriority::High])
            .expect_err("a provider without a scheduler refuses a marking");
        assert_eq!(direct.phase, ProviderPhase::Resolve);
        assert_eq!(direct.class, ProviderErrorClass::Capability);
        assert_eq!(direct.slug, "queue_priorities_unsupported");

        let (client, mut server) = super::unix::pair().unwrap();
        let server_thread = std::thread::spawn(move || serve_provider(&provider, &mut server));
        let remote = RemoteProvider::connect(client).unwrap();
        let refused = remote
            .set_queue_priorities(&[QueuePriority::High])
            .expect_err("the refusal travels as an error response");
        assert_eq!(refused.slug, "queue_priorities_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(remote.health(), ProviderHealth::Usable);
        drop(remote);
        server_thread.join().unwrap().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn remote_provider_serves_compile_submit_wait_and_readback() {
        let submissions = Arc::new(AtomicU64::new(0));
        let imports = Arc::new(AtomicU64::new(0));
        let provider = FakeProvider {
            epoch: DeviceEpoch::new(7),
            capabilities: fake_capabilities(),
            submissions: Arc::clone(&submissions),
            imports: Arc::clone(&imports),
            borrowed: Arc::new(BorrowedLeaseRegistry::new()),
        };
        let (client, mut server) = super::unix::pair().unwrap();
        let server_thread = std::thread::spawn(move || serve_provider(&provider, &mut server));

        let remote = RemoteProvider::connect(client).unwrap();
        assert_eq!(remote.device_epoch(), DeviceEpoch::new(7));
        assert_eq!(remote.health(), ProviderHealth::Usable);
        assert_eq!(remote.capabilities(), fake_capabilities());
        let compiled = remote.compile(compile_request()).unwrap();
        remote
            .import_staged_lease(
                StagedLease::new(
                    LeaseReservation {
                        lease: BufferLease {
                            lease_id: LeaseId::new(51),
                            allocation_id: AllocationId::new(41),
                            owner_epoch: DeviceEpoch::new(7),
                        },
                        offset: 0,
                        length: 4,
                    },
                    vec![1, 2, 3, 4],
                )
                .unwrap(),
            )
            .unwrap();
        let trace = trace(&compiled);
        let admitted = remote
            .capabilities()
            .validate_trace(trace, resources())
            .unwrap();
        let submitted = remote.submit(admitted).unwrap();
        let token = submitted.completion.token().unwrap();
        assert_eq!(
            remote.wait(token, Duration::from_secs(1)).unwrap(),
            CompletionDisposition::CompletedVisible { token }
        );
        let readback = remote.readback(token).unwrap();
        assert_eq!(readback.writebacks.len(), 1);
        assert_eq!(readback.writebacks[0].bytes, vec![4, 3, 2, 1]);
        remote.release_completion(token).unwrap();
        remote.release_staged_lease(LeaseId::new(51)).unwrap();
        remote.release_pipeline(&compiled).unwrap();
        drop(remote);
        server_thread.join().unwrap().unwrap();
        assert_eq!(submissions.load(Ordering::SeqCst), 1);
        assert_eq!(imports.load(Ordering::SeqCst), 1);
    }

    #[cfg(unix)]
    #[test]
    fn remote_provider_reports_exhausted_after_the_command_channel_closes() {
        let (client, mut server) = super::unix::pair().unwrap();
        let server_thread = std::thread::spawn(move || {
            for _ in 0..2 {
                let response = match server.recv_request().unwrap() {
                    CommandRequest::Capabilities => CommandResponse::Capabilities {
                        epoch: DeviceEpoch::new(7),
                        capabilities: fake_capabilities(),
                    },
                    CommandRequest::Health => CommandResponse::Health {
                        health: ProviderHealth::Usable,
                    },
                    other => panic!("unexpected request {other:?}"),
                };
                server.send_response(&response).unwrap();
                server.flush().unwrap();
            }
        });

        let remote = RemoteProvider::connect(client).unwrap();
        assert_eq!(remote.health(), ProviderHealth::Usable);
        // The one-shot server closes the channel after the health reply, so the
        // next exchange observes EOF.
        server_thread.join().unwrap();
        // A closed command channel degrades to Exhausted rather than blocking
        // or claiming the device was lost: the owner must recreate the
        // provider before new work can be admitted.
        assert_eq!(remote.health(), ProviderHealth::Exhausted);
        let error = remote.compile(compile_request()).unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Resource);
        assert_eq!(error.slug, "provider_unavailable");
        assert_eq!(error.retryability, Retryability::RetryAfterRecreate);
    }

    #[cfg(unix)]
    #[test]
    fn remote_provider_imports_a_descriptor_backed_lease() {
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let provider = FakeProvider {
            epoch: DeviceEpoch::new(7),
            capabilities: fake_capabilities(),
            submissions: Arc::new(AtomicU64::new(0)),
            imports: Arc::new(AtomicU64::new(0)),
            borrowed: Arc::clone(&borrowed),
        };
        let (client, mut server) = super::unix::pair().unwrap();
        let server_thread = std::thread::spawn(move || serve_provider_unix(&provider, &mut server));

        let remote = RemoteProvider::connect(client).unwrap();
        let mut memory = crate::shared::SharedMemory::create(4096).unwrap();
        memory.as_mut_slice().fill(0x5a);
        let lease_id = LeaseId::new(77);
        let reservation = LeaseReservation {
            lease: BufferLease {
                lease_id,
                allocation_id: AllocationId::new(41),
                owner_epoch: DeviceEpoch::new(7),
            },
            offset: 0,
            length: 4096,
        };
        remote.import_borrowed_lease(reservation, &memory).unwrap();
        assert_eq!(borrowed.len(), 1);

        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(41),
                owner_epoch: DeviceEpoch::new(7),
                size: 4096,
            })
            .unwrap();
        resources.insert_lease(reservation).unwrap();
        let view = BufferView {
            view_id: ViewId::new(78),
            metal_binding: 0,
            allocation_id: AllocationId::new(41),
            offset: 0,
            length: 4,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::BorrowedNoCopy(lease_id),
        };
        let resolved = borrowed
            .view_pointer(lease_id, &view, DeviceEpoch::new(7), &resources)
            .unwrap();
        assert_eq!(resolved.len, 4);
        // SAFETY: the server keeps the mapping alive until release and the
        // owner mapping refers to the same physical pages.
        let observed = unsafe { std::slice::from_raw_parts(resolved.pointer as *const u8, 4) };
        assert_eq!(observed, &[0x5a; 4]);

        memory.as_mut_slice()[..4].copy_from_slice(&0x1234_5678_u32.to_le_bytes());
        // SAFETY: as above; the import is still live.
        let observed = unsafe { std::slice::from_raw_parts(resolved.pointer as *const u8, 4) };
        assert_eq!(observed, &0x1234_5678_u32.to_le_bytes());

        remote.release_borrowed_lease(lease_id).unwrap();
        assert_eq!(borrowed.len(), 0);
        let refused = remote.release_borrowed_lease(lease_id).unwrap_err();
        assert_eq!(refused.slug, "lease_not_imported");
        drop(remote);
        server_thread.join().unwrap().unwrap();
    }

    #[test]
    fn remote_provider_imports_a_named_borrowed_lease() {
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let provider = FakeProvider {
            epoch: DeviceEpoch::new(7),
            capabilities: fake_capabilities(),
            submissions: Arc::new(AtomicU64::new(0)),
            imports: Arc::new(AtomicU64::new(0)),
            borrowed: Arc::clone(&borrowed),
        };
        let (provider_reader, owner_writer) = std::io::pipe().unwrap();
        let (owner_reader, provider_writer) = std::io::pipe().unwrap();
        let mut server = super::CommandTransport::new(provider_reader, provider_writer);
        let server_thread =
            std::thread::spawn(move || serve_provider_named(&provider, &mut server));

        let client = super::CommandTransport::new(owner_reader, owner_writer);
        let remote = RemoteProvider::connect(client).unwrap();
        let (mut memory, _name) = crate::shared::SharedMemory::create_named(4096).unwrap();
        memory.as_mut_slice().fill(0x5a);
        let lease_id = LeaseId::new(78);
        let reservation = LeaseReservation {
            lease: BufferLease {
                lease_id,
                allocation_id: AllocationId::new(42),
                owner_epoch: DeviceEpoch::new(7),
            },
            offset: 0,
            length: 4096,
        };
        remote
            .import_named_borrowed_lease(reservation, &memory)
            .unwrap();
        assert_eq!(borrowed.len(), 1);
        // A duplicate import consumes its mapping name before the refusal, so
        // the connection stays framed for the next request.
        let duplicate = remote
            .import_named_borrowed_lease(reservation, &memory)
            .unwrap_err();
        assert_eq!(duplicate.slug, "lease_already_imported");
        assert_eq!(remote.health(), ProviderHealth::Usable);

        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(42),
                owner_epoch: DeviceEpoch::new(7),
                size: 4096,
            })
            .unwrap();
        resources.insert_lease(reservation).unwrap();
        let view = BufferView {
            view_id: ViewId::new(79),
            metal_binding: 0,
            allocation_id: AllocationId::new(42),
            offset: 0,
            length: 4,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::BorrowedNoCopy(lease_id),
        };
        let resolved = borrowed
            .view_pointer(lease_id, &view, DeviceEpoch::new(7), &resources)
            .unwrap();
        // SAFETY: the server keeps the mapping alive until release and the
        // owner mapping refers to the same physical pages.
        let observed = unsafe { std::slice::from_raw_parts(resolved.pointer as *const u8, 4) };
        assert_eq!(observed, &[0x5a; 4]);
        memory.as_mut_slice()[..4].copy_from_slice(&0x8765_4321_u32.to_le_bytes());
        // SAFETY: as above; the import is still live.
        let observed = unsafe { std::slice::from_raw_parts(resolved.pointer as *const u8, 4) };
        assert_eq!(observed, &0x8765_4321_u32.to_le_bytes());

        remote.release_borrowed_lease(lease_id).unwrap();
        assert_eq!(borrowed.len(), 0);
        drop(remote);
        server_thread.join().unwrap().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rejected_duplicate_borrowed_import_keeps_the_channel_in_sync() {
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let provider = FakeProvider {
            epoch: DeviceEpoch::new(7),
            capabilities: fake_capabilities(),
            submissions: Arc::new(AtomicU64::new(0)),
            imports: Arc::new(AtomicU64::new(0)),
            borrowed: Arc::clone(&borrowed),
        };
        let (client, mut server) = super::unix::pair().unwrap();
        let server_thread = std::thread::spawn(move || serve_provider_unix(&provider, &mut server));

        let remote = RemoteProvider::connect(client).unwrap();
        let lease_id = LeaseId::new(91);
        let reservation = LeaseReservation {
            lease: BufferLease {
                lease_id,
                allocation_id: AllocationId::new(42),
                owner_epoch: DeviceEpoch::new(7),
            },
            offset: 0,
            length: 4096,
        };

        let mut first = crate::shared::SharedMemory::create(4096).unwrap();
        first.as_mut_slice().fill(0x11);
        remote.import_borrowed_lease(reservation, &first).unwrap();

        let mut duplicate = crate::shared::SharedMemory::create(4096).unwrap();
        duplicate.as_mut_slice().fill(0x22);
        let error = remote
            .import_borrowed_lease(reservation, &duplicate)
            .unwrap_err();
        assert_eq!(error.slug, "lease_already_imported");

        // The rejected import still carried a descriptor. The server must
        // consume it, otherwise its one-byte payload desynchronizes the next
        // frame on this connection.
        remote.release_borrowed_lease(lease_id).unwrap();
        assert_eq!(borrowed.len(), 0);

        let mut third = crate::shared::SharedMemory::create(4096).unwrap();
        third.as_mut_slice().fill(0x33);
        remote.import_borrowed_lease(reservation, &third).unwrap();

        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(42),
                owner_epoch: DeviceEpoch::new(7),
                size: 4096,
            })
            .unwrap();
        resources.insert_lease(reservation).unwrap();
        let view = BufferView {
            view_id: ViewId::new(79),
            metal_binding: 0,
            allocation_id: AllocationId::new(42),
            offset: 0,
            length: 4,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::BorrowedNoCopy(lease_id),
        };
        let resolved = borrowed
            .view_pointer(lease_id, &view, DeviceEpoch::new(7), &resources)
            .unwrap();
        // SAFETY: the server keeps the mapping alive until release and the
        // owner mapping refers to the same physical pages.
        let observed = unsafe { std::slice::from_raw_parts(resolved.pointer as *const u8, 4) };
        assert_eq!(observed, &[0x33; 4]);

        remote.release_borrowed_lease(lease_id).unwrap();
        drop(remote);
        server_thread.join().unwrap().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn remote_provider_chunks_requests_above_the_frame_limit() {
        let submissions = Arc::new(AtomicU64::new(0));
        let imports = Arc::new(AtomicU64::new(0));
        let provider = FakeProvider {
            epoch: DeviceEpoch::new(7),
            capabilities: fake_capabilities(),
            submissions: Arc::clone(&submissions),
            imports: Arc::clone(&imports),
            borrowed: Arc::new(BorrowedLeaseRegistry::new()),
        };
        let (mut client, mut server) = super::unix::pair().unwrap();
        client.set_max_frame(32);
        server.set_max_frame(32);
        assert_eq!(client.max_frame(), 32);
        let server_thread = std::thread::spawn(move || serve_provider(&provider, &mut server));

        // Capabilities already exceeds 32 bytes, so connect and every request
        // below exercise chunk framing and reassembly.
        let remote = RemoteProvider::connect(client).unwrap();
        let compiled = remote.compile(compile_request()).unwrap();
        let trace = trace(&compiled);
        let admitted = remote
            .capabilities()
            .validate_trace(trace, resources())
            .unwrap();
        let submitted = remote.submit(admitted).unwrap();
        let token = submitted.completion.token().unwrap();
        assert_eq!(
            remote.wait(token, Duration::from_secs(1)).unwrap(),
            CompletionDisposition::CompletedVisible { token }
        );
        let readback = remote.readback(token).unwrap();
        assert_eq!(readback.writebacks.len(), 1);
        remote.release_completion(token).unwrap();
        remote.release_pipeline(&compiled).unwrap();
        drop(remote);
        server_thread.join().unwrap().unwrap();
        assert_eq!(submissions.load(Ordering::SeqCst), 1);
    }

    #[cfg(unix)]
    #[test]
    fn chunk_frames_reject_a_request_before_the_transfer_finishes() {
        let (mut client, mut server) = super::unix::pair().unwrap();
        let payload = CommandCodec::encode_request_payload(&CommandRequest::Compile {
            request: compile_request(),
        })
        .unwrap();
        assert!(payload.len() > 4);
        let half = payload.len() / 2;
        CommandCodec::write_chunk_frame(
            &mut client.writer,
            9,
            0,
            payload.len() as u64,
            &payload[..half],
        )
        .unwrap();
        CommandCodec::write_request(&mut client.writer, &CommandRequest::Health).unwrap();

        let error = server.recv_request().unwrap_err();
        assert!(matches!(
            error,
            super::CommandError::Codec(CodecError::ChunkInterrupted { .. })
        ));
    }
}
