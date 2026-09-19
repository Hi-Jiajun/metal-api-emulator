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
    /// The bytes a command frame spends before its payload: the magic, the
    /// frame kind and the payload's length. Tests that talk about "the payload"
    /// slice from here rather than repeating the number.
    const FRAME_HEADER: usize = 9;
    use metal_api_core::provider::{
        AcquirePolicy, AllocationId, AllocationRecord, AttachmentFormat, AttachmentLandingView,
        BlendAttachment, BlendFactor, BlendOperation, BufferAccess, BufferBindingContract,
        BufferLease, BufferSource, BufferView, BufferWriteback, ClearColor, ColorWriteMask,
        CompareFunction, CompiledComputePipeline, CompletionDisposition, CompletionPolicy,
        CompletionReadback, CompletionToken, ComputePass, ComputeProvider, ComputeTrace, CullMode,
        DepthFormat, DepthLoadOp, DepthResolveFilter, DepthStoreOp, DepthTest, DeviceEpoch,
        Dispatch, DispatchKind, DispatchType, FieldValue, FootprintProof, FunctionIdentity,
        GuestRun, HeapDescriptor, HeapId, HeapPayload, HeapPlacement, HeapResource,
        IndexBufferBinding, IndexFormat, IndirectCommandBufferDescriptor,
        IndirectCommandDescriptor, IndirectCommandKind, IndirectCommandPayload,
        IndirectCommandRange, InitialState, KeptFrame, KeptFrameLanding, LeaseId, LeaseImporter,
        LeaseReservation, LoadOp, MultisampleDepthResolve, MultisampleState,
        MultisampleStencilResolve, OperationId, PipelineCompileRequest, PipelineContract,
        PipelineId, PipelineProvider, PresentDescriptor, PresentMode, PresentTarget,
        ProviderCapabilities, ProviderError, ProviderErrorClass, ProviderHealth, ProviderPhase,
        ProviderSubmission, QueuePriority, RenderAttachment, RenderDepthAttachment,
        RenderDepthIdentity, RenderPassBlend, RenderPassCull, RenderPassDescriptor,
        RenderPipelineContract, RenderPipelineStage, RenderSamplerBinding, RenderStencilAttachment,
        RenderStencilIdentity, ResourceTableSnapshot, Retryability, SampleCount,
        SamplerAddressMode, SamplerFilter, SamplerPolicy, SemanticDigest, ShaderSource,
        StageBufferBinding, StageBufferView, StagedLease, StencilCompare, StencilFormat,
        StencilLoadOp, StencilOp, StencilResolveFilter, StencilTest, StorageMode, StoreOp,
        SubmissionId, TextureAccess, TextureBindingContract, TextureFormat, TextureSource,
        TextureType, TextureView, TracePass, ValidatedComputeTrace, VertexAttribute,
        VertexBufferLayout, VertexFormat, VertexLayout, ViewId, Winding,
        FULL_SCREEN_TRIANGLE_VERTICES, MAX_COLOR_ATTACHMENTS, MAX_PRESENT_IMAGE_COUNT,
        MAX_RENDER_SAMPLERS, MAX_RENDER_STAGE_BUFFERS, MAX_RENDER_STAGE_BUFFER_DECLARATIONS,
        MAX_RENDER_TEXTURES, MAX_VERTEX_BUFFERS, PROVIDER_SCHEMA_VERSION,
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
                texture_bindings: Vec::new(),
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
            stage_buffers: Vec::new(),
            vertex_entry: "full_screen_vertex".into(),
            fragment_entry: "solid_color_fragment".into(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::None,
            textures: Vec::new(),
        }
    }

    fn render_pass_descriptor(
        compiled: &CompiledComputePipeline,
        width: u64,
        height: u64,
    ) -> RenderPassDescriptor {
        RenderPassDescriptor {
            samplers: Vec::new(),
            stage_buffers: Vec::new(),
            blend: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
            cull: None,
            stencil: None,
            stencil_test: None,
            depth: None,
            depth_test: None,
            base_vertex: 0,
            pipeline: compiled.pipeline_id,
            color_attachments: vec![render_attachment(width, height)],
            viewport: [0, 0, width as u32, height as u32],
            scissor: None,
            vertices: FULL_SCREEN_TRIANGLE_VERTICES,
            vertex_buffers: Vec::new(),
            indices: None,
            instance_count: 1,
            textures: Vec::new(),
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

    /// The clear payload is one texel of the attachment's own format
    /// (`research/docs/23` §78), and the wire carries that width through the
    /// format byte written before the load operation — there is no length
    /// prefix, so a payload of another width cannot be framed at all.
    #[test]
    fn the_clear_payload_travels_at_the_attachment_formats_own_width() {
        let wide = ClearColor::from_bytes(&[0xf8, 0x3b, 0xf8, 0x3b, 0xf8, 0x3b, 0xf8, 0x3b])
            .expect("one eight-byte texel");
        let mut compiled = pipeline(&compile_request());
        compiled.render = Some(RenderPipelineContract {
            stage_buffers: Vec::new(),
            color_formats: vec![AttachmentFormat::Rgba16Float],
            textures: Vec::new(),
            ..render_contract()
        });
        let mut attachment = render_attachment(2, 2);
        attachment.format = AttachmentFormat::Rgba16Float;
        attachment.load = LoadOp::Clear(wide);
        let mut descriptor = render_pass_descriptor(&compiled, 2, 2);
        descriptor.color_attachments = vec![attachment];
        let trace = ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: compiled.device_epoch,
            operation_id: OperationId::new(22),
            pipelines: vec![compiled.clone()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![TracePass::Render(descriptor)],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        };
        let request = CommandRequest::Submit {
            trace,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        let decoded = CommandCodec::decode_request(&frame).unwrap();
        assert_eq!(decoded, request, "the eight-byte clear survives the frame");
        let CommandRequest::Submit { trace, .. } = &decoded else {
            panic!("a render submit decodes as a submit");
        };
        let pass = trace.passes[0]
            .as_render()
            .expect("the render pass decodes as a render pass");
        assert_eq!(pass.color_attachments[0].format.code(), 4);
        let LoadOp::Clear(clear) = pass.color_attachments[0].load else {
            panic!("the pass carries its clear");
        };
        assert_eq!(clear.as_bytes().len(), 8);
        assert_eq!(clear, wide, "the payload is the same value, byte for byte");

        // The narrow spelling for the wide format is refused *while encoding*,
        // by name: the receiver reads the payload at the format's width, so a
        // frame that carried four bytes here would have to desync on the next
        // field instead of failing closed.
        let mut narrow = request.clone();
        let CommandRequest::Submit { trace, .. } = &mut narrow else {
            panic!("the request is a submit");
        };
        let TracePass::Render(pass) = &mut trace.passes[0] else {
            panic!("the pass is a render pass");
        };
        pass.color_attachments[0].load = LoadOp::Clear(ClearColor::new([0xfe; 4]));
        let refused = CommandCodec::encode_request(&narrow)
            .expect_err("a four-byte clear cannot be framed for an eight-byte texel");
        eprintln!("refused: {refused:?}");
        assert!(matches!(
            refused,
            CodecError::ClearLength {
                format: 4,
                expected: 8,
                actual: 4,
            }
        ));
        assert_eq!(
            refused.to_string(),
            "attachment clear for format code 4 carries 4 bytes, but its texel is 8 bytes"
        );
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
        // The fixture already carries one attachment; four more cross the MRT
        // cap the encoder shares with the decoder.
        for _ in 0..MAX_COLOR_ATTACHMENTS {
            pass.color_attachments.push(render_attachment(2, 2));
        }
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

    /// The offset of the pipeline-table entry's kind tag. The frame kind (4),
    /// the request tag (1), the schema version (4), the epoch (1), the
    /// operation id (2), the entry count (8), the device epoch (8) and the
    /// pipeline id (8) precede it.
    fn pipeline_entry_kind_offset() -> usize {
        4 + 1 + 4 + 1 + 2 + 8 + 8 + 8
    }

    /// The offset of the byte behind the fixture's fragment entry name, i.e.
    /// where the entry's colour-format field starts. Located by its content so
    /// the assertion does not pin a position another increment could move.
    fn render_entry_format_offset(frame: &[u8]) -> usize {
        const FRAGMENT_ENTRY: &[u8] = b"solid_color_fragment";
        let position = frame
            .windows(FRAGMENT_ENTRY.len())
            .position(|window| window == FRAGMENT_ENTRY)
            .expect("the fixture fragment entry name is on the wire");
        position + FRAGMENT_ENTRY.len()
    }

    /// The render-only submit whose single pipeline entry declares `formats`
    /// in place of the fixture's one format.
    fn render_submit_with_formats(formats: Vec<AttachmentFormat>) -> CommandRequest {
        let mut trace = render_only_trace();
        let Some(entry) = trace.pipelines.first_mut() else {
            panic!("the fixture trace carries one pipeline entry");
        };
        let Some(contract) = entry.render.as_mut() else {
            panic!("the fixture entry carries a render half");
        };
        contract.color_formats = formats;
        CommandRequest::Submit {
            trace,
            resources: resources(),
        }
    }

    #[test]
    fn render_contract_encoding_refuses_empty_and_oversize_format_lists() {
        // No format describes no attachment at all, so it is refused rather
        // than written as a zero-length list a provider cannot resolve.
        assert!(matches!(
            CommandCodec::encode_request(&render_submit_with_formats(Vec::new())).unwrap_err(),
            CodecError::RenderPipelineFormatCount { count: 0, maximum }
                if maximum == MAX_COLOR_ATTACHMENTS
        ));

        // One past the MRT cap is refused with the count as asked for, exactly
        // like the render pass's own attachment cap.
        assert!(matches!(
            CommandCodec::encode_request(&render_submit_with_formats(vec![
                AttachmentFormat::Rgba8Unorm;
                MAX_COLOR_ATTACHMENTS + 1
            ]))
            .unwrap_err(),
            CodecError::RenderPipelineFormatCount { count, maximum }
                if count == MAX_COLOR_ATTACHMENTS + 1 && maximum == MAX_COLOR_ATTACHMENTS
        ));

        // The cap itself is the last shape the wire admits.
        CommandCodec::encode_request(&render_submit_with_formats(vec![
            AttachmentFormat::Rgba8Unorm;
            MAX_COLOR_ATTACHMENTS
        ]))
        .expect("the MRT cap is admitted");
    }

    #[test]
    fn single_format_render_contract_keeps_its_pre_mrt_bytes() {
        let request = render_submit_with_formats(vec![AttachmentFormat::Rgba8Unorm]);
        let frame = CommandCodec::encode_request(&request).unwrap();

        // The entry wears the pre-MRT tag, and that tag is followed by exactly
        // one format byte and then the vertex-layout tag: the MRT increment is
        // additive, so a one-attachment registration pays nothing for it.
        assert_eq!(
            frame[pipeline_entry_kind_offset()],
            0x01,
            "render entry tag"
        );
        let format_offset = render_entry_format_offset(&frame);
        assert_eq!(frame[format_offset], 0x02, "Rgba8Unorm follows the names");
        assert_eq!(
            frame[format_offset + 1],
            0x00,
            "the vertex layout follows the single format byte"
        );

        // And the whole frame is still the frozen pre-present frame for this
        // fixture, so nothing else in the entry moved either.
        assert_eq!(frame, LEGACY_RENDER_SUBMIT_FRAME);
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
    }

    #[test]
    fn multi_format_render_contract_round_trips_under_the_mrt_tag() {
        // A list, in location order, with a format a single-format contract
        // could not carry: the second entry is BGRA, the fourth repeats RGBA.
        let formats = vec![
            AttachmentFormat::Rgba8Unorm,
            AttachmentFormat::Bgra8Unorm,
            AttachmentFormat::R32Float,
            AttachmentFormat::Rgba8Unorm,
        ];
        let request = render_submit_with_formats(formats.clone());
        let frame = CommandCodec::encode_request(&request).unwrap();

        // The entry wears the MRT tag, and the format field is a big-endian
        // length prefix followed by one byte per format, in location order.
        assert_eq!(frame[pipeline_entry_kind_offset()], 0x02, "MRT entry tag");
        let list_offset = render_entry_format_offset(&frame);
        assert_eq!(
            &frame[list_offset..list_offset + 8],
            (formats.len() as u64).to_be_bytes(),
            "the list carries its own length"
        );
        assert_eq!(
            &frame[list_offset + 8..list_offset + 8 + formats.len()],
            [0x02, 0x03, 0x01, 0x02],
            "one format byte per location"
        );
        assert_eq!(
            frame[list_offset + 8 + formats.len()],
            0x00,
            "the vertex layout follows the list"
        );

        let decoded = CommandCodec::decode_request(&frame).unwrap();
        assert_eq!(decoded, request);
        let CommandRequest::Submit { trace, .. } = &decoded else {
            panic!("a render submit decodes as a submit");
        };
        let Some(contract) = trace
            .pipelines
            .first()
            .and_then(|entry| entry.render.as_ref())
        else {
            panic!("the decoded entry carries a render half");
        };
        assert_eq!(contract.color_formats, formats);
    }

    #[test]
    fn mrt_entries_refuse_format_lists_outside_the_admitted_band() {
        let request = render_submit_with_formats(vec![
            AttachmentFormat::Rgba8Unorm,
            AttachmentFormat::Bgra8Unorm,
        ]);
        let frame = CommandCodec::encode_request(&request).unwrap();
        let list_offset = render_entry_format_offset(&frame);

        // A zero-length list would leave the vertex layout to be read as a
        // format, so the decoder refuses it where it stands.
        let mut empty = frame.clone();
        empty[list_offset..list_offset + 8].copy_from_slice(&0_u64.to_be_bytes());
        assert!(matches!(
            CommandCodec::decode_request(&empty).unwrap_err(),
            CodecError::RenderPipelineFormatCount { count: 0, maximum }
                if maximum == MAX_COLOR_ATTACHMENTS
        ));

        // An over-long prefix is refused before a list of that length is
        // allocated or read.
        let mut oversize = frame.clone();
        oversize[list_offset..list_offset + 8]
            .copy_from_slice(&((MAX_COLOR_ATTACHMENTS + 1) as u64).to_be_bytes());
        assert!(matches!(
            CommandCodec::decode_request(&oversize).unwrap_err(),
            CodecError::RenderPipelineFormatCount { count, maximum }
                if count == MAX_COLOR_ATTACHMENTS + 1 && maximum == MAX_COLOR_ATTACHMENTS
        ));
    }

    /// The declaration block the v83 fixture writes: one count, one stage
    /// (`fragment` = 1), one `u32` index, one access (`Read` = 0) and one
    /// static footprint tag with its `u64` extent
    /// (`research/docs/23` §3.3, v83).
    const STAGE_BUFFER_DECLARATION_BLOCK: [u8; 16] = [
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x10,
    ];

    #[test]
    fn a_render_contract_with_stage_declarations_takes_its_own_pipeline_kind() {
        // The contract's declaration half travels under
        // `PIPELINE_KIND_RENDER_STAGE_BUFFERS` (`research/docs/23` §3.3, v83):
        // the entry writes its colour formats the MRT way and appends the
        // declaration block after the vertex layout.
        let plain = CommandCodec::encode_request(&render_submit_with_formats(vec![
            AttachmentFormat::Rgba8Unorm,
        ]))
        .unwrap();
        // The declaration half alone: the pass keeps the empty list, so the
        // frame differs from the plain one in the pipeline entry only.
        let mut trace = render_only_trace();
        trace.pipelines[0].render = Some(stage_buffer_render_contract());
        let request = CommandRequest::Submit {
            trace,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame,
            "the declaration frame re-encodes byte for byte"
        );
        assert_eq!(plain[pipeline_entry_kind_offset()], 0x01, "plain entry tag");
        assert_eq!(
            frame[pipeline_entry_kind_offset()],
            0x03,
            "declaration entry tag"
        );
        // The format field is the MRT-style list even for one format, because
        // the tag itself is the new shape.
        let list_offset = render_entry_format_offset(&frame);
        assert_eq!(
            &frame[list_offset..list_offset + 8],
            1_u64.to_be_bytes(),
            "the list carries its own length"
        );
        assert_eq!(frame[list_offset + 8], 0x02, "Rgba8Unorm follows the list");
        let at = frame
            .windows(STAGE_BUFFER_DECLARATION_BLOCK.len())
            .position(|window| window == STAGE_BUFFER_DECLARATION_BLOCK)
            .expect("the declaration block is on the wire");
        // The block is the whole difference plus the list's length prefix:
        // eight bytes of list length and sixteen bytes of declarations.
        assert_eq!(
            frame.len(),
            plain.len() + 8 + STAGE_BUFFER_DECLARATION_BLOCK.len()
        );
        eprintln!(
            "pipeline declaration frame: len={} plain={} kind={:#04x} block_at={at}",
            frame.len(),
            plain.len(),
            frame[pipeline_entry_kind_offset()]
        );
    }

    #[test]
    fn a_pipeline_stage_buffer_declaration_refuses_a_count_above_the_contract_cap() {
        // The encoder refuses the count before a single tuple is written.
        let mut contract = stage_buffer_render_contract();
        contract.stage_buffers = (0..=MAX_RENDER_STAGE_BUFFERS as u32)
            .map(|index| StageBufferBinding {
                stage: RenderPipelineStage::Vertex,
                index,
                access: BufferAccess::Read,
                footprint: FootprintProof::Static { max_bytes: 4 },
            })
            .collect();
        let mut trace = render_only_trace();
        trace.pipelines[0].render = Some(contract);
        let refused = CommandCodec::encode_request(&CommandRequest::Submit {
            trace,
            resources: resources(),
        })
        .unwrap_err();
        assert!(matches!(
            refused,
            CodecError::RenderStageBufferCount { stage: Some(RenderPipelineStage::Vertex), count, maximum }
                if count == MAX_RENDER_STAGE_BUFFERS + 1
                    && maximum == MAX_RENDER_STAGE_BUFFERS
        ));
        // The decoder refuses the same count by name instead of reading one
        // tuple more than the contract's ceiling states.
        let request = CommandRequest::Submit {
            trace: stage_buffer_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        let at = frame
            .windows(STAGE_BUFFER_DECLARATION_BLOCK.len())
            .position(|window| window == STAGE_BUFFER_DECLARATION_BLOCK)
            .expect("the declaration block is on the wire");
        let mut patched = frame.clone();
        // One above the pipeline-level list bound (`research/docs/23` §3.3,
        // §108; §117 E-SB2): the count byte is refused before a single tuple is
        // read, so the patched frame needs no matching block behind it. The
        // list bound is the pair's sum, so a ninth *stage* declaration is a
        // shape the wire carries and the contract refuses by stage instead
        // (`a_stage_buffer_stage_past_the_ceiling_is_refused_by_stage`).
        patched[at] = u8::try_from(MAX_RENDER_STAGE_BUFFER_DECLARATIONS + 1).unwrap();
        assert!(matches!(
            CommandCodec::decode_request(&patched).unwrap_err(),
            CodecError::RenderStageBufferCount { stage: None, count, maximum }
                if count == MAX_RENDER_STAGE_BUFFER_DECLARATIONS + 1
                    && maximum == MAX_RENDER_STAGE_BUFFER_DECLARATIONS
        ));
        eprintln!("pipeline declaration count refused: {refused}");
    }

    #[test]
    fn pipeline_entries_refuse_an_unknown_tag_next_to_the_stage_buffer_tag() {
        let request = render_submit_with_formats(vec![AttachmentFormat::Rgba8Unorm]);
        let frame = CommandCodec::encode_request(&request).unwrap();
        let mut unknown = frame.clone();
        // `0x07` is a kind no version of this walk assigns: compute `0x00`,
        // render `0x01`/`0x02`/`0x03`, the compute texture face `0x04`, the
        // render texture face `0x05` and the combined render face `0x06`
        // (`research/docs/23` §91, §3.3 v100).
        unknown[pipeline_entry_kind_offset()] = 0x07;
        assert!(matches!(
            CommandCodec::decode_request(&unknown).unwrap_err(),
            CodecError::UnknownPipelineTag(0x07)
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

    /// The reviewed quad's vertex stream: four `float32x2` positions, declared
    /// by the render pass itself (`research/docs/23` §3.6).
    fn vertex_stream_view() -> BufferView {
        BufferView {
            view_id: ViewId::new(41),
            metal_binding: 0,
            allocation_id: AllocationId::new(43),
            offset: 0,
            length: 32,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0; 32]),
        }
    }

    /// The reviewed quad's index buffer: six `uint16` indices.
    fn index_stream_view() -> BufferView {
        BufferView {
            view_id: ViewId::new(45),
            metal_binding: 0,
            allocation_id: AllocationId::new(47),
            offset: 0,
            length: 12,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0; 12]),
        }
    }

    /// The fixture's render contract extended with the reviewed quad's vertex
    /// layout (`research/docs/23` §3.3).
    fn vertex_input_render_contract() -> RenderPipelineContract {
        RenderPipelineContract {
            stage_buffers: Vec::new(),
            vertex_layout: VertexLayout::Buffers(vec![VertexBufferLayout {
                stride: 8,
                step: metal_api_core::provider::VertexStep::PerVertex,
                attributes: vec![VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32x2,
                }],
            }]),
            textures: Vec::new(),
            ..render_contract()
        }
    }

    /// A render trace whose pass binds a caller-held vertex stream and an index
    /// buffer, drawing the reviewed indexed quad.
    fn vertex_input_trace() -> ComputeTrace {
        let mut trace = render_only_trace();
        trace.pipelines[0].render = Some(vertex_input_render_contract());
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.vertices = 6;
        pass.vertex_buffers = vec![vertex_stream_view()];
        pass.indices = Some(IndexBufferBinding {
            view: index_stream_view(),
            format: IndexFormat::Uint16,
        });
        trace
    }

    /// One stage buffer entry for the v83 fixture: a fragment
    /// `[[buffer(0)]]` read of sixteen caller-held bytes
    /// (`research/docs/23` §3.3, v83).
    fn stage_buffer_view() -> StageBufferView {
        StageBufferView {
            stage: RenderPipelineStage::Fragment,
            view: BufferView {
                view_id: ViewId::new(49),
                metal_binding: 0,
                allocation_id: AllocationId::new(51),
                offset: 0,
                length: 16,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![
                    0x40, 0x80, 0xc0, 0xff, 0x40, 0x80, 0xc0, 0xff, 0x40, 0x80, 0xc0, 0xff, 0x40,
                    0x80, 0xc0, 0xff,
                ]),
            },
        }
    }

    /// The stage buffer entry's payload, spelled once so the tests assert on
    /// the same bytes the fixture carries.
    fn stage_buffer_payload() -> Vec<u8> {
        [0x40, 0x80, 0xc0, 0xff].repeat(4)
    }

    /// The fixture's render contract extended with the declaration that pairs
    /// with [`stage_buffer_view`] (`research/docs/23` §3.3, v83).
    fn stage_buffer_render_contract() -> RenderPipelineContract {
        RenderPipelineContract {
            stage_buffers: vec![StageBufferBinding {
                stage: RenderPipelineStage::Fragment,
                index: 0,
                access: BufferAccess::Read,
                footprint: FootprintProof::Static { max_bytes: 16 },
            }],
            textures: Vec::new(),
            ..render_contract()
        }
    }

    /// A render trace whose contract declares one stage buffer binding and
    /// whose pass fills it (`research/docs/23` §3.3, v83).
    fn stage_buffer_trace() -> ComputeTrace {
        let mut trace = render_only_trace();
        trace.pipelines[0].render = Some(stage_buffer_render_contract());
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.stage_buffers = vec![stage_buffer_view()];
        trace
    }

    /// The multisample fixture plus one stage buffer entry, so the frame
    /// carries a wide feature word *and* the v83 block
    /// (`research/docs/23` §3.3). The contract keeps the plain shape so the
    /// frame differs from [`multisample_trace`] in the pass alone.
    fn multisample_stage_buffer_trace() -> ComputeTrace {
        let mut trace = multisample_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.stage_buffers = vec![stage_buffer_view()];
        trace
    }

    /// A render trace that both samples one texture and reads one stage
    /// buffer, so the v70 and v83 blocks travel in one frame
    /// (`research/docs/23` §3.3). The contract keeps the plain shape so the
    /// frame differs from [`sampled_multisample_trace`] in the pass alone.
    fn sampled_stage_buffer_trace() -> ComputeTrace {
        let mut trace = sampled_multisample_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.stage_buffers = vec![stage_buffer_view()];
        trace
    }

    /// The five bytes that spell a v83 stage-buffer pass's fixed head: the
    /// tag, the wide feature word — here all zero, because the fixture's pass
    /// states no optional section — and the block's count and the entry's
    /// stage code (`fragment` = 1) (`research/docs/23` §3.3, v83).
    const STAGE_BUFFER_HEAD: [u8; 5] = [0x13, 0x00, 0x00, 0x01, 0x01];

    /// Where [`STAGE_BUFFER_HEAD`] sits in a frame, by position.
    fn stage_buffer_head_at(frame: &[u8]) -> usize {
        frame
            .windows(STAGE_BUFFER_HEAD.len())
            .position(|window| window == STAGE_BUFFER_HEAD)
            .expect("the frame carries the stage-buffer head")
    }

    #[test]
    fn a_stage_buffer_pass_takes_its_own_tag_and_round_trips() {
        let request = CommandRequest::Submit {
            trace: stage_buffer_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        // The round trip is byte exact in both directions: the decoded value
        // re-encodes to the same frame, so a relay that decodes and forwards
        // cannot perturb the shape.
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame
        );
        // The stage-buffer tag is followed by the same wide feature word the
        // wide tag carries and then by the block the word had no bit for: a
        // `u8` count, the entry's stage code and one full `BufferView`
        // (`research/docs/23` §3.3, v83).
        let position = stage_buffer_head_at(&frame);
        assert_eq!(
            &frame[position..position + STAGE_BUFFER_HEAD.len()],
            &STAGE_BUFFER_HEAD
        );
        eprintln!(
            "stage-buffer frame: len={} head={:02x?} at={position}",
            frame.len(),
            &frame[position..position + 5]
        );
        // The bytes the stage reads travel with the frame, so a provider can
        // fill the slot without a second declaration.
        let payload = stage_buffer_payload();
        let payload_at = frame
            .windows(payload.len())
            .position(|window| window == payload)
            .expect("the stage buffer's own bytes travel in the frame");
        eprintln!(
            "stage-buffer payload at={payload_at} bytes={:02x?}",
            &frame[payload_at..payload_at + payload.len()]
        );
    }

    #[test]
    fn a_stage_buffer_pass_is_the_pre_v83_frame_with_one_tag_and_one_block_more() {
        // The only difference between a pre-v83 wide frame and a v83
        // stage-buffer frame has to be the tag byte and the inserted block:
        // the wide word keeps its meanings, and every section after the block
        // is byte identical. This is the "old frames keep their bytes" rule
        // stated as a comparison instead of a golden vector.
        let plain_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: multisample_trace(),
            resources: resources(),
        })
        .unwrap();
        let staged_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: multisample_stage_buffer_trace(),
            resources: resources(),
        })
        .unwrap();
        // The nine-byte frame header carries the payload length, which the
        // block deliberately changes; every byte after it is what this test
        // compares.
        let plain = &plain_frame[9..];
        let staged = &staged_frame[9..];
        let position = plain
            .windows(3)
            .position(|window| window == [0x11, 0x20, 0x01])
            .expect("the plain fixture takes the wide tag");
        assert_eq!(&staged[..position], &plain[..position]);
        assert_eq!(staged[position], 0x13);
        assert_eq!(
            &staged[position + 1..position + 3],
            &plain[position + 1..position + 3],
            "the stage-buffer tag carries the same wide word"
        );
        assert_eq!(staged[position + 3], 0x01, "one stage buffer");
        let block_length = staged.len() - plain.len() - 1;
        assert_eq!(
            &staged[position + 4 + block_length..],
            &plain[position + 3..],
            "every section after the block is byte identical"
        );
        eprintln!(
            "pre-v83 len={} staged len={} block={} bytes tag_at={position}",
            plain_frame.len(),
            staged_frame.len(),
            block_length + 1
        );
    }

    #[test]
    fn a_sampled_stage_buffer_pass_carries_both_blocks() {
        let request = CommandRequest::Submit {
            trace: sampled_stage_buffer_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame,
            "the combined frame re-encodes byte for byte"
        );
        // The combined tag writes the sampled tag's payload first — the wide
        // word and the texture count — and appends the stage buffer block
        // after the texture block (`research/docs/23` §3.3, v83).
        assert!(
            frame
                .windows(4)
                .any(|window| window == [0x14, 0x20, 0x01, 0x01]),
            "the combined pass carries its own tag, the wide word and the texture count"
        );
        let texels: Vec<u8> = (0..4u8)
            .flat_map(|y| (0..4u8).flat_map(move |x| [x, y, x.wrapping_add(y), 0xff]))
            .collect();
        let texture_at = frame
            .windows(texels.len())
            .position(|window| window == texels)
            .expect("the sampled texture's bytes travel in the frame");
        let payload = stage_buffer_payload();
        let payload_at = frame
            .windows(payload.len())
            .position(|window| window == payload)
            .expect("the stage buffer's bytes travel in the frame");
        assert!(
            texture_at < payload_at,
            "the texture block precedes the stage buffer block"
        );
        // The combined frame shares the sampled frame's head — tag aside — and
        // every section after the stage block is byte identical, because both
        // tags walk the same wide word through the same section walker.
        let sampled_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: sampled_multisample_trace(),
            resources: resources(),
        })
        .unwrap();
        let sampled = &sampled_frame[9..];
        let combined = &frame[9..];
        let position = sampled
            .windows(3)
            .position(|window| window == [0x12, 0x20, 0x01])
            .expect("the sampled fixture takes the sampled tag");
        assert_eq!(&combined[..position], &sampled[..position]);
        assert_eq!(combined[position], 0x14);
        assert_eq!(
            &combined[position + 1..position + 3],
            &sampled[position + 1..position + 3]
        );
        assert_eq!(combined[position + 3], 0x01, "one sampled texture");
        // The stage block's payload is the last variable-length field before
        // the pass's own sections, so both frames' sections start where their
        // payloads end, and they are byte identical.
        let combined_sections = payload_at + payload.len() - 9;
        let sampled_sections = texture_at + texels.len() - 9;
        assert_eq!(
            &combined[combined_sections..],
            &sampled[sampled_sections..],
            "every section after the stage block is byte identical"
        );
        eprintln!(
            "combined len={} sampled len={} texels_at={texture_at} payload_at={payload_at}",
            frame.len(),
            sampled_frame.len()
        );
    }

    #[test]
    fn a_stage_buffer_pass_refuses_a_count_above_the_contract_cap() {
        let frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: stage_buffer_trace(),
            resources: resources(),
        })
        .unwrap();
        let position = stage_buffer_head_at(&frame);
        let mut patched = frame.clone();
        // One above the pass's list bound (`research/docs/23` §3.3, §108; §117
        // E-SB2): the count byte is refused before a single view is read.
        patched[position + 3] = u8::try_from(MAX_RENDER_STAGE_BUFFER_DECLARATIONS + 1).unwrap();
        assert!(matches!(
            CommandCodec::decode_request(&patched).unwrap_err(),
            CodecError::RenderStageBufferCount { stage: None, count, maximum }
                if count == MAX_RENDER_STAGE_BUFFER_DECLARATIONS + 1
                    && maximum == MAX_RENDER_STAGE_BUFFER_DECLARATIONS
        ));
        // The encoder refuses the same protocol bound instead of writing a
        // frame the decoder would reject.
        let mut trace = stage_buffer_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.stage_buffers = (0..=MAX_RENDER_STAGE_BUFFERS as u32)
            .map(|index| StageBufferView {
                stage: RenderPipelineStage::Fragment,
                view: BufferView {
                    view_id: ViewId::new(49 + u64::from(index)),
                    metal_binding: index,
                    ..stage_buffer_view().view
                },
            })
            .collect();
        let refused = CommandCodec::encode_request(&CommandRequest::Submit {
            trace,
            resources: resources(),
        })
        .unwrap_err();
        assert!(matches!(
            refused,
            CodecError::RenderStageBufferCount { stage: Some(RenderPipelineStage::Fragment), count, maximum }
                if count == MAX_RENDER_STAGE_BUFFERS + 1 && maximum == MAX_RENDER_STAGE_BUFFERS
        ));
        eprintln!("stage buffer count refusals: {refused}");
    }

    #[test]
    fn a_stage_buffer_pass_refuses_an_unknown_stage_code() {
        let frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: stage_buffer_trace(),
            resources: resources(),
        })
        .unwrap();
        let position = stage_buffer_head_at(&frame);
        let mut patched = frame.clone();
        // The stage byte follows the block's count: `0` is vertex, `1` is
        // fragment, and any other byte names no stage.
        patched[position + 4] = 0x02;
        let refused = CommandCodec::decode_request(&patched).unwrap_err();
        assert!(matches!(
            refused,
            CodecError::UnknownEnumValue {
                field: "render pipeline stage",
                value: 2,
            }
        ));
        eprintln!("unknown stage code refused: {refused}");
    }

    #[test]
    fn a_stage_buffer_pass_refuses_a_corrupt_view_by_name() {
        // The view keeps its own named refusals inside the block: a source tag
        // this version does not know is refused as `buffer source` instead of
        // being read as one of the three arms, exactly as a compute binding's
        // view is.
        let frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: stage_buffer_trace(),
            resources: resources(),
        })
        .unwrap();
        let payload = stage_buffer_payload();
        let payload_at = frame
            .windows(payload.len())
            .position(|window| window == payload)
            .expect("the stage buffer's own bytes travel in the frame");
        let mut patched = frame.clone();
        // The blob's eight-byte length precedes the payload, and the source
        // tag precedes that length. `3` is the guest-runs tag since E-TX6
        // (`research/docs/23` §74), so the probe states a tag that is still
        // unassigned: the byte after the arm the frame actually carries.
        patched[payload_at - 9] = 0x04;
        let refused = CommandCodec::decode_request(&patched).unwrap_err();
        assert!(matches!(
            refused,
            CodecError::UnknownEnumValue {
                field: "buffer source",
                value: 4,
            }
        ));
        eprintln!("corrupt stage buffer view refused: {refused}");
    }

    #[test]
    fn a_stage_buffer_pass_refuses_a_second_entry_that_is_not_there() {
        // The count is one byte, so a frame can claim more entries than it
        // wrote. The decoder reads the claimed entries and then the pass's own
        // fields from the wrong bytes, so the frame is refused inside the
        // payload instead of decoding a pass nobody wrote.
        let frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: stage_buffer_trace(),
            resources: resources(),
        })
        .unwrap();
        let position = stage_buffer_head_at(&frame);
        let mut patched = frame.clone();
        patched[position + 3] = 0x02;
        let refused = CommandCodec::decode_request(&patched).unwrap_err();
        eprintln!("over-claimed stage buffer count refused: {refused}");
        assert!(
            matches!(
                refused,
                CodecError::TruncatedPayload { .. }
                    | CodecError::UnknownEnumValue { .. }
                    | CodecError::TrailingPayload { .. }
            ),
            "a count above the written entries is refused inside the payload"
        );
    }

    #[test]
    fn vertex_input_frames_use_the_extended_render_tag_and_round_trip() {
        let request = CommandRequest::Submit {
            trace: vertex_input_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        // The frame tag stays the render one: the extended kind is a pass tag
        // inside the payload, so an older decoder that knows the render frame
        // still refuses the shape it cannot read instead of misreading it.
        assert_eq!(frame[9], 0x0f);
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);

        // The pass tag itself is the extended kind, immediately followed by a
        // feature byte whose vertex bit is set.
        let extended = frame
            .iter()
            .enumerate()
            .skip(10)
            .filter(|(_, byte)| **byte == 0x10)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert_eq!(
            extended.len(),
            1,
            "the fixture carries exactly one extended pass tag, got {extended:?}"
        );
        assert_eq!(frame[extended[0] + 1] & 0x01, 0x01);
        assert_eq!(
            frame[extended[0] + 1] & 0x02,
            0x00,
            "an offscreen pass carries no present bit"
        );

        // The feature byte is full as of v40, so the fail-closed probe is an
        // unknown *pass tag*: a decoder refuses the frame instead of reading
        // the payload as one of the kinds it knows.
        let mut patched = frame.clone();
        patched[extended[0]] = 0x7f;
        assert!(matches!(
            CommandCodec::decode_request(&patched),
            Err(CodecError::UnknownPassTag(0x7f))
        ));
    }

    /// The fixture's render contract extended with the reviewed instanced
    /// pair: the same position stream plus a `float32x4` tint that advances
    /// once per instance (`research/docs/23` §3.3, v31).
    fn instanced_render_contract() -> RenderPipelineContract {
        RenderPipelineContract {
            stage_buffers: Vec::new(),
            vertex_layout: VertexLayout::Buffers(vec![
                VertexBufferLayout {
                    stride: 8,
                    step: metal_api_core::provider::VertexStep::PerVertex,
                    attributes: vec![VertexAttribute {
                        location: 0,
                        offset: 0,
                        format: VertexFormat::Float32x2,
                    }],
                },
                VertexBufferLayout {
                    stride: 16,
                    step: metal_api_core::provider::VertexStep::PerInstance,
                    attributes: vec![VertexAttribute {
                        location: 1,
                        offset: 0,
                        format: VertexFormat::Float32x4,
                    }],
                },
            ]),
            textures: Vec::new(),
            ..render_contract()
        }
    }

    /// A render trace whose pass binds the reviewed pair and draws two
    /// instances.
    fn instanced_trace() -> ComputeTrace {
        let mut trace = render_only_trace();
        trace.pipelines[0].render = Some(instanced_render_contract());
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.vertices = 6;
        pass.instance_count = 2;
        pass.vertex_buffers = vec![
            vertex_stream_view(),
            BufferView {
                view_id: ViewId::new(49),
                metal_binding: 1,
                allocation_id: AllocationId::new(51),
                offset: 0,
                length: 32,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![0; 32]),
            },
        ];
        pass.indices = Some(IndexBufferBinding {
            view: index_stream_view(),
            format: IndexFormat::Uint16,
        });
        trace
    }

    #[test]
    fn instanced_frames_carry_the_instancing_feature_bit_and_round_trip() {
        let request = CommandRequest::Submit {
            trace: instanced_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(frame[9], 0x0f);
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        // The pass tag is the pair the encoder writes: `PASS_KIND_RENDER_EXT`
        // followed by the feature byte. Scanning for the pair instead of the
        // tag alone keeps a 0x10 inside a length field from being mistaken for
        // the tag.
        let extended = frame
            .windows(2)
            .enumerate()
            .skip(10)
            .filter(|(_, pair)| *pair == [0x10, 0x09])
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert_eq!(
            extended.len(),
            1,
            "the fixture carries exactly one instanced pass tag, got {extended:?}"
        );
        let extended_index = extended[0];
        assert_eq!(
            frame[extended_index + 1] & 0x09,
            0x09,
            "the instanced pass carries the vertex-input and instancing bits"
        );
        assert_eq!(
            frame[extended_index + 1] & 0x02,
            0x00,
            "an offscreen pass carries no present bit"
        );

        // A single-instance pass keeps the pre-v31 bytes: the instancing bit is
        // what makes the count travel, so a pass without it must not set one.
        let mut single = instanced_trace();
        let Some(TracePass::Render(pass)) = single.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.instance_count = 1;
        let single_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: single,
            resources: resources(),
        })
        .unwrap();
        let extended = single_frame
            .windows(2)
            .enumerate()
            .skip(10)
            .find(|(_, pair)| *pair == [0x10, 0x01])
            .map(|(index, _)| index)
            .expect("the single-instance pass keeps the pre-v31 extended tag");
        assert_eq!(
            single_frame[extended + 1] & 0x08,
            0x00,
            "a single-instance pass carries no instancing bit"
        );

        // The feature byte is full as of v40, so an unknown pass tag is the
        // fail-closed probe this fixture keeps.
        let mut patched = frame.clone();
        patched[extended_index] = 0x7e;
        assert!(matches!(
            CommandCodec::decode_request(&patched),
            Err(CodecError::UnknownPassTag(0x7e))
        ));
    }

    /// A render trace whose pass offsets the reviewed quad's indices by one
    /// (`research/docs/23` §3.3, v34).
    fn base_vertex_trace() -> ComputeTrace {
        let mut trace = vertex_input_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.base_vertex = 1;
        trace
    }

    #[test]
    fn a_base_vertex_pass_carries_its_own_feature_bit_and_round_trips() {
        let request = CommandRequest::Submit {
            trace: base_vertex_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(frame[9], 0x0f);
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        // The pass carries the vertex-input and base-vertex bits, and neither
        // the present nor the instancing one.
        let tag = frame
            .windows(2)
            .enumerate()
            .skip(10)
            .find(|(_, pair)| *pair == [0x10, 0x11])
            .map(|(index, _)| index)
            .expect("the base-vertex pass carries both feature bits");
        assert_eq!(frame[tag + 1] & 0x02, 0x00, "no present bit");
        assert_eq!(frame[tag + 1] & 0x08, 0x00, "no instancing bit");

        // A pass that offsets nothing keeps the pre-v34 bytes: the bit is what
        // makes the field travel.
        let mut plain = vertex_input_trace();
        let Some(TracePass::Render(pass)) = plain.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.base_vertex = 0;
        let plain_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: plain,
            resources: resources(),
        })
        .unwrap();
        let tag = plain_frame
            .windows(2)
            .enumerate()
            .skip(10)
            .find(|(_, pair)| *pair == [0x10, 0x01])
            .map(|(index, _)| index)
            .expect("the plain pass keeps the vertex-only feature byte");
        assert_eq!(plain_frame[tag + 1] & 0x10, 0x00, "no base-vertex bit");

        // The feature byte is full as of v40, so an unknown pass tag is the
        // fail-closed probe this fixture keeps.
        let mut patched = frame.clone();
        patched[tag] = 0x7d;
        assert!(matches!(
            CommandCodec::decode_request(&patched),
            Err(CodecError::UnknownPassTag(0x7d))
        ));
    }

    /// A render trace whose pass culls back faces with a counter-clockwise
    /// front (`research/docs/23` §3.3, v39).
    fn cull_trace() -> ComputeTrace {
        let mut trace = vertex_input_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.cull = Some(RenderPassCull {
            mode: CullMode::Back,
            winding: Winding::CounterClockwise,
        });
        trace
    }

    #[test]
    fn a_cull_pass_carries_its_own_feature_bit_and_round_trips() {
        let request = CommandRequest::Submit {
            trace: cull_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        let tag = frame
            .windows(2)
            .enumerate()
            .skip(10)
            .find(|(_, pair)| *pair == [0x10, 0x41])
            .map(|(index, _)| index)
            .expect("the culling pass carries the vertex-input and cull bits");

        // A pass that culls nothing keeps the pre-v39 bytes: the bit is what
        // makes the state travel.
        let plain = vertex_input_trace();
        let plain_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: plain,
            resources: resources(),
        })
        .unwrap();
        let plain_tag = plain_frame
            .windows(2)
            .enumerate()
            .skip(10)
            .find(|(_, pair)| *pair == [0x10, 0x01])
            .map(|(index, _)| index)
            .expect("the plain pass keeps the vertex-only feature byte");
        assert_eq!(plain_frame[plain_tag + 1] & 0x40, 0x00, "no cull bit");

        // No cull fixture at this layer: the section is exactly the two bytes
        // the pass states, and the state's own round trip above is what pins
        // them. An unknown *feature* bit stays a decoder refusal, which the
        // base-vertex fixture pins for the same kind of section.
        let _ = tag;
    }

    /// A render trace whose pass blends its single attachment with the
    /// reviewed factors (`research/docs/23` §3.3, v40).
    fn blend_trace() -> ComputeTrace {
        let mut trace = vertex_input_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.blend = Some(RenderPassBlend {
            attachments: vec![BlendAttachment {
                enabled: true,
                source_rgb: BlendFactor::SourceAlpha,
                destination_rgb: BlendFactor::OneMinusSourceAlpha,
                source_alpha: BlendFactor::SourceAlpha,
                destination_alpha: BlendFactor::OneMinusSourceAlpha,
                operation: BlendOperation::Add,
                alpha_operation: BlendOperation::Add,
                write_mask: ColorWriteMask::ALL,
            }],
        });
        trace
    }

    #[test]
    fn a_blend_pass_carries_its_own_feature_bit_and_round_trips() {
        let request = CommandRequest::Submit {
            trace: blend_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert!(
            frame.windows(2).any(|pair| pair == [0x10, 0x81]),
            "the blending pass carries the vertex-input and blend bits"
        );

        // A pass that blends nothing keeps the pre-v40 bytes.
        let plain = vertex_input_trace();
        let plain_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: plain,
            resources: resources(),
        })
        .unwrap();
        assert!(
            plain_frame.windows(2).any(|pair| pair == [0x10, 0x01]),
            "the plain pass keeps the vertex-only feature byte"
        );
        assert!(!plain_frame.windows(2).any(|pair| pair == [0x10, 0x81]));
    }

    /// The v40 blend section is five bytes per entry — the four factor codes
    /// and one operation — for an entry that blends with every channel written
    /// (`research/docs/23` §3.3, v40/v100). A pass that states one of the later
    /// fields has no position in those bytes, so the encoder refuses it by name
    /// rather than framing a state the receiver would read as the v40 shape.
    #[test]
    fn a_blend_state_the_v40_section_cannot_carry_is_refused_by_name() {
        fn disabled(attachment: &mut BlendAttachment) {
            attachment.enabled = false;
        }
        fn own_alpha_operation(attachment: &mut BlendAttachment) {
            attachment.alpha_operation = BlendOperation::Subtract;
        }
        fn masked(attachment: &mut BlendAttachment) {
            attachment.write_mask = ColorWriteMask::RED;
        }
        type Reshape = fn(&mut BlendAttachment);
        let cases: [(Reshape, &str); 3] = [
            (disabled, "blendingEnabled = false"),
            (own_alpha_operation, "an alpha operation of its own"),
            (masked, "a colour write mask"),
        ];
        for (reshape, field) in cases {
            let mut trace = blend_trace();
            let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
                panic!("the fixture is a render pass");
            };
            let blend = pass
                .blend
                .as_mut()
                .expect("the fixture states one blend entry");
            reshape(&mut blend.attachments[0]);
            let refused = CommandCodec::encode_request(&CommandRequest::Submit {
                trace,
                resources: resources(),
            })
            .expect_err("the v40 section cannot carry the state");
            eprintln!("refused: {refused}");
            assert_eq!(
                refused.to_string(),
                format!(
                    "colour attachment 0's blend state states {field}, which the v40 blend \
                     section cannot carry"
                )
            );
            // And the shape the section *can* carry still frames.
            let carried = CommandCodec::encode_request(&CommandRequest::Submit {
                trace: blend_trace(),
                resources: resources(),
            })
            .expect("the v40 shape frames");
            assert_eq!(
                CommandCodec::decode_request(&carried).unwrap(),
                CommandRequest::Submit {
                    trace: blend_trace(),
                    resources: resources(),
                }
            );
        }
    }

    /// A render trace whose pass opens the rail-owned depth attachment in the
    /// pre-v43 shape: no store action and no landing, which is what every frame
    /// published before v43 means (`research/docs/23` §3.3, v36).
    fn depth_trace() -> ComputeTrace {
        let mut trace = vertex_input_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.depth = Some(RenderDepthAttachment {
            format: DepthFormat::Depth32Float,
            width: 4,
            height: 4,
            load: DepthLoadOp::clear(1.0),
            store: None,
            identity: None,
        });
        pass.depth_test = Some(DepthTest {
            compare: CompareFunction::Less,
            write: true,
        });
        trace
    }

    /// The same pass with the v43 pair: the surface survives the pass and the
    /// trace names where its texels land (`research/docs/23` §3.3, v43).
    fn depth_store_trace() -> ComputeTrace {
        let mut trace = depth_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        let Some(depth) = pass.depth.as_mut() else {
            panic!("the fixture opens a depth attachment");
        };
        depth.store = Some(DepthStoreOp::Store);
        depth.identity = Some(RenderDepthIdentity {
            allocation_id: AllocationId::new(940),
            view_id: ViewId::new(950),
        });
        trace
    }

    #[test]
    fn a_depth_store_pass_takes_the_wide_feature_tag_and_round_trips() {
        let request = CommandRequest::Submit {
            trace: depth_store_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        // The wide tag is followed by the big-endian feature word: the narrow
        // byte it repeats — vertex input and depth, `0x21` — stays its *last*
        // byte, and the two bits the store action and the identity travel
        // under live in the byte before it, so `0x0321` reads as `03 21`.
        let wide = frame
            .windows(3)
            .enumerate()
            .skip(10)
            .find(|(_, window)| *window == [0x11, 0x03, 0x21])
            .map(|(index, _)| index)
            .expect("the storing pass takes the wide tag");
        assert_eq!(frame[wide], 0x11);

        // A pass that keeps nothing keeps the pre-v43 bytes exactly: the same
        // vertex-input and depth bits, under the narrow tag.
        let plain_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: depth_trace(),
            resources: resources(),
        })
        .unwrap();
        assert!(
            plain_frame.windows(2).any(|pair| pair == [0x10, 0x21]),
            "the discarded depth pass keeps the narrow tag and its feature byte"
        );
        assert!(
            !plain_frame.windows(2).any(|pair| pair == [0x11, 0x03]),
            "a pass with no wide section never takes the wide tag"
        );
    }

    #[test]
    fn wide_depth_features_without_a_depth_attachment_or_with_unknown_bits_are_refused() {
        let request = CommandRequest::Submit {
            trace: depth_store_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        let wide = frame
            .windows(3)
            .enumerate()
            .skip(10)
            .find(|(_, window)| *window == [0x11, 0x03, 0x21])
            .map(|(index, _)| index)
            .expect("the storing pass takes the wide tag");

        // `0x83` sets the stencil resolve bit (`0x8000`, v60) beside the
        // known depth ones. The wide word's high byte is now full, so no
        // unknown high bit exists to refuse: the stencil resolve bit names a
        // stencil surface the pass never opens, and the decoder refuses the
        // orphaned resolve exactly as it refuses the depth store and identity
        // bits (`research/docs/23` §3.3, v60).
        let mut unknown = frame.clone();
        // The wide word is big-endian, so the high byte is the first one after
        // the tag, beside the known depth ones (`0x0321`).
        unknown[wide + 1] = 0x83;
        let refused = CommandCodec::decode_request(&unknown);
        assert!(
            matches!(
                refused,
                Err(CodecError::DepthFeatureWithoutAttachment(0x8000))
            ),
            "a stencil resolve bit without a stencil block has to be refused, got {refused:?}"
        );

        // The store action and the identity describe the depth attachment: a
        // word that names them without the depth bit names a surface the pass
        // never opens, and the decoder refuses it before reading a section.
        let mut orphaned = frame;
        orphaned[wide + 2] = 0x01;
        assert!(
            matches!(
                CommandCodec::decode_request(&orphaned),
                Err(CodecError::DepthFeatureWithoutAttachment(0x0300))
            ),
            "a wide depth feature without a depth block has to be refused"
        );
    }

    #[test]
    fn a_depth_resolve_bit_without_a_depth_attachment_is_refused() {
        let request = CommandRequest::Submit {
            trace: multisample_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        let wide = frame
            .windows(3)
            .enumerate()
            .skip(10)
            .find(|(_, window)| *window == [0x11, 0x20, 0x01])
            .map(|(index, _)| index)
            .expect("the multisample pass takes the wide tag");

        // Set the depth-resolve bit (`0x4000`) beside the multisample bit,
        // with the low byte keeping the vertex-input bit alone: no depth
        // section exists, so the decoder refuses the orphaned resolve before
        // reading its filter, exactly as the depth store and identity bits are
        // refused (`research/docs/23` §3.3, v57).
        let mut orphaned = frame;
        orphaned[wide + 1] = 0x60;
        assert!(
            matches!(
                CommandCodec::decode_request(&orphaned),
                Err(CodecError::DepthFeatureWithoutAttachment(0x4000))
            ),
            "a depth resolve bit without a depth block has to be refused"
        );
    }

    /// A render trace whose pass opens a rail-owned stencil attachment and
    /// tests it (`research/docs/23` §3.3, v47): the state the reviewed
    /// increment's fixture uses, so the wire's byte shape is pinned by the same
    /// value the rails execute.
    fn stencil_trace() -> ComputeTrace {
        let mut trace = depth_store_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.stencil = Some(RenderStencilAttachment {
            format: StencilFormat::Stencil8,
            width: 4,
            height: 4,
            load: StencilLoadOp::clear(0),
            store: None,
            identity: None,
        });
        pass.stencil_test = Some(StencilTest {
            compare: StencilCompare::Equal,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::IncrementWrap,
            read_mask: 0xff,
            write_mask: 0xff,
            reference: 0,
        });
        trace
    }

    #[test]
    fn a_stencil_pass_takes_the_wide_tag_and_round_trips() {
        let request = CommandRequest::Submit {
            trace: stencil_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        // The wide word's low byte is still the narrow byte (vertex input and
        // depth, `0x21`); the high byte carries the store, identity and stencil
        // bits, so the pair reads `07 21`.
        assert!(
            frame
                .windows(3)
                .enumerate()
                .skip(10)
                .any(|(_, window)| window == [0x11, 0x07, 0x21]),
            "the stencil pass carries the third wide bit"
        );

        // A pass that opens no stencil attachment keeps the pre-v47 bytes: the
        // depth-store fixture's own word has no stencil bit.
        let plain = depth_store_trace();
        let plain_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: plain,
            resources: resources(),
        })
        .unwrap();
        assert!(plain_frame
            .windows(3)
            .any(|window| window == [0x11, 0x03, 0x21]));
    }

    /// A render trace whose pass states the four-sample raster, keeps its
    /// stencil surface and resolves it with a named filter
    /// (`research/docs/23` §3.3, v60).
    fn stencil_resolve_trace() -> ComputeTrace {
        let mut trace = stencil_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        {
            let stencil = pass
                .stencil
                .as_mut()
                .expect("the fixture opens a stencil attachment");
            stencil.store = Some(StoreOp::Store);
            stencil.identity = Some(RenderStencilIdentity {
                allocation_id: AllocationId::new(941),
                view_id: ViewId::new(951),
            });
        }
        pass.stencil_resolve = Some(MultisampleStencilResolve {
            filter: StencilResolveFilter::Sample0,
        });
        trace
    }

    #[test]
    fn a_stencil_resolve_pass_takes_the_final_wide_bit_and_round_trips() {
        let request = CommandRequest::Submit {
            trace: stencil_resolve_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        // The wide word's high byte carries the depth store/resource, stencil,
        // stencil store/resource, multisample and stencil resolve bits; with
        // the resolve bit (`0x80`) set the high byte is full
        // (`research/docs/23` §3.3, v60).
        assert!(
            frame
                .windows(3)
                .any(|window| window[0] == 0x11 && window[1] & 0x80 != 0 && window[2] == 0x21),
            "the stencil resolve pass carries the eighth and final wide bit"
        );

        // A pass that never resolves keeps the pre-v60 bytes: the same
        // fixture without the filter is refused by the contract's validate
        // (stored multisampled stencil without a resolve), but its frame has
        // no stencil resolve bit.
        let mut plain = stencil_resolve_trace();
        let Some(TracePass::Render(pass)) = plain.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.stencil_resolve = None;
        let plain_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: plain,
            resources: resources(),
        })
        .unwrap();
        assert!(!plain_frame
            .windows(3)
            .any(|window| window[0] == 0x11 && window[1] & 0x80 != 0));
    }

    #[test]
    fn stencil_resolve_frames_round_trip_every_admitted_filter() {
        // The v60 wire carries one filter code per pass, so each of the two
        // admitted filters has to survive its own round trip. The
        // depthResolvedSample filter names the sample the depth resolve
        // selected, so its fixture states a depth resolve beside it.
        for (filter, code) in [
            (StencilResolveFilter::Sample0, 0x00),
            (StencilResolveFilter::DepthResolvedSample, 0x01),
        ] {
            let mut trace = stencil_resolve_trace();
            let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
                panic!("the fixture is a render pass");
            };
            pass.stencil_resolve = Some(MultisampleStencilResolve { filter });
            if filter == StencilResolveFilter::DepthResolvedSample {
                pass.depth = Some(RenderDepthAttachment {
                    format: DepthFormat::Depth32Float,
                    width: 4,
                    height: 4,
                    load: DepthLoadOp::clear(1.0),
                    store: Some(DepthStoreOp::Store),
                    identity: Some(RenderDepthIdentity {
                        allocation_id: AllocationId::new(940),
                        view_id: ViewId::new(950),
                    }),
                });
                pass.depth_test = Some(DepthTest {
                    compare: CompareFunction::Less,
                    write: true,
                });
                pass.depth_resolve = Some(MultisampleDepthResolve {
                    filter: DepthResolveFilter::Min,
                });
            }
            let request = CommandRequest::Submit {
                trace,
                resources: resources(),
            };
            let frame = CommandCodec::encode_request(&request).unwrap();
            assert_eq!(
                CommandCodec::decode_request(&frame).unwrap(),
                request,
                "the {filter:?} filter survives the round trip"
            );
            // The filter byte follows the multisample count (and, for the
            // depthResolvedSample filter, the depth resolve's own filter
            // byte), so each admitted code travels in its own byte.
            assert!(frame.contains(&code), "the {filter:?} filter code travels");
        }
    }

    #[test]
    fn stencil_resolve_frames_refuse_unknown_filter_codes() {
        let request = CommandRequest::Submit {
            trace: stencil_resolve_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();

        // The filter code travels one byte after the multisample count, which
        // itself follows the stencil block, its store action and its
        // identity. The contract admits codes 0/1 only, so a third code is a
        // decoder refusal rather than a filter the caller did not ask for.
        let mut patched = frame.clone();
        // The stencil identity (allocation 941, view 951) travels as two
        // big-endian `u64`s; the ten-byte run ends that identity with the
        // four-sample count (`0x02`) followed by the Sample0 filter code
        // (`0x00`), so its last byte is the filter the decode would read.
        let filter = frame
            .windows(10)
            .enumerate()
            .skip(10)
            .find(|(_, window)| {
                *window == [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xB7, 0x02, 0x00]
            })
            .map(|(index, _)| index + 9)
            .expect("the stencil identity precedes the multisample count and the filter");
        patched[filter] = 0x03;
        assert!(matches!(
            CommandCodec::decode_request(&patched),
            Err(CodecError::UnknownEnumValue {
                field: "stencil resolve filter",
                value: 0x03,
            })
        ));
    }

    /// The one sampled texture a v70 render pass binds: a 4x4 `rgba8_unorm`
    /// surface whose sixteen texels are pairwise distinct
    /// (`research/docs/23` §3.3, v70).
    fn sampled_texture_view(binding: u32) -> TextureView {
        let mut bytes = Vec::with_capacity(64);
        for y in 0..4u8 {
            for x in 0..4u8 {
                bytes.extend_from_slice(&[x, y, x.wrapping_add(y), 0xff]);
            }
        }
        TextureView {
            view_id: ViewId::new(83),
            metal_binding: binding,
            allocation_id: AllocationId::new(53),
            texture_type: TextureType::D2,
            format: TextureFormat::Rgba8Unorm,
            width: 4,
            height: 4,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            access: TextureAccess::Sampled,
            source: TextureSource::OwnedBytes(bytes),
        }
    }

    /// A render trace whose pass states the four-sample raster *and* samples
    /// one texture, so both the wide word and the v70 block are present
    /// (`research/docs/23` §3.3, v51/v70).
    fn sampled_multisample_trace() -> ComputeTrace {
        let mut trace = multisample_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.textures = vec![sampled_texture_view(0)];
        trace
    }

    /// The guest-runs source arm (`research/docs/23` §74, E-TX6): the view's
    /// bytes are an ordered list of owner windows, and the frame carries the
    /// triples rather than the bytes they describe.
    ///
    /// The tag is new, so the three earlier source arms keep their own bytes —
    /// which is what makes this frame's size the owned arm's minus the byte
    /// payload plus one triple per run.
    #[test]
    fn a_guest_runs_source_round_trips_as_its_run_list() {
        let mut trace = multisample_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        let mut view = sampled_texture_view(0);
        let runs = vec![
            GuestRun {
                lease_id: LeaseId::new(21),
                offset: 0,
                length: 8,
            },
            GuestRun {
                lease_id: LeaseId::new(22),
                offset: 8,
                length: 8,
            },
        ];
        // The arm is a `BufferSource`, so the pass's own vertex input carries
        // it: one stream whose bytes are the two runs.
        let stream = BufferView {
            view_id: ViewId::new(9),
            metal_binding: 0,
            allocation_id: AllocationId::new(3),
            offset: 0,
            length: 16,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::GuestRuns(runs.clone()),
        };
        view.source = TextureSource::OwnedBytes(vec![0; 16]);
        pass.vertex_buffers = vec![stream.clone()];
        let mut resources = resources();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(3),
                owner_epoch: DeviceEpoch::new(7),
                size: 16,
            })
            .unwrap();
        for (lease_id, offset) in [(21_u64, 0_u64), (22, 8)] {
            resources
                .insert_lease(LeaseReservation {
                    lease: BufferLease {
                        lease_id: LeaseId::new(lease_id),
                        allocation_id: AllocationId::new(3),
                        owner_epoch: DeviceEpoch::new(7),
                    },
                    offset,
                    length: 8,
                })
                .unwrap();
        }
        let request = CommandRequest::Submit { trace, resources };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(
            CommandCodec::decode_request(&frame).unwrap(),
            request,
            "the run list is what travels"
        );
        // The listing itself is in the frame — one `(lease, offset, length)`
        // triple per run — and no run's bytes are.
        let mut expected = Vec::new();
        for run in &runs {
            // The codec's wide integers travel big-endian, like every other
            // identifier on this channel.
            expected.extend_from_slice(&run.lease_id.get().to_be_bytes());
            expected.extend_from_slice(&run.offset.to_be_bytes());
            expected.extend_from_slice(&run.length.to_be_bytes());
        }
        assert!(
            frame
                .windows(expected.len())
                .any(|window| window == expected.as_slice()),
            "the frame carries the run list"
        );
    }

    /// The trace-produced source arm (`research/docs/23` §110, E-TX3): the
    /// declaration names the trace's own production and carries no texel
    /// bytes, so the frame is the owned-bytes frame one source payload
    /// narrower.
    #[test]
    fn a_trace_view_source_round_trips_without_texel_bytes() {
        let mut trace = multisample_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        let mut view = sampled_texture_view(0);
        let texels = match view.source.clone() {
            TextureSource::OwnedBytes(bytes) => bytes,
            other => panic!("the fixture starts from owned bytes, got {other:?}"),
        };
        view.source = TextureSource::TraceView;
        pass.textures = vec![view.clone()];
        let request = CommandRequest::Submit {
            trace,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        // The arm's own name travels; the sixteen texels do not, because the
        // bytes do not exist before the trace runs.
        assert!(
            !frame.windows(texels.len()).any(|window| window == texels),
            "a trace-view declaration carries no texel bytes"
        );
    }

    /// The pass-entry snapshot source arm (`research/docs/23` §118, E-TX15):
    /// the declaration names the pass's own attachment and carries no texel
    /// bytes either, so the arm is one tag after the trace-view arm — and an
    /// older decoder reads that tag as an unknown texture source rather than as
    /// whatever payload follows it.
    #[test]
    fn a_pass_entry_snapshot_source_round_trips_without_texel_bytes() {
        let mut trace = multisample_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        let mut view = sampled_texture_view(0);
        let texels = match view.source.clone() {
            TextureSource::OwnedBytes(bytes) => bytes,
            other => panic!("the fixture starts from owned bytes, got {other:?}"),
        };
        view.source = TextureSource::PassEntrySnapshot;
        view.view_id = pass.color_attachments[0].view_id;
        view.allocation_id = pass.color_attachments[0].allocation_id;
        view.format = pass.color_attachments[0].format.as_texture_format();
        view.width = pass.color_attachments[0].width;
        view.height = pass.color_attachments[0].height;
        pass.color_attachments[0].load = LoadOp::Load;
        pass.textures = vec![view.clone()];
        let request = CommandRequest::Submit {
            trace: trace.clone(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert!(
            !frame.windows(texels.len()).any(|window| window == texels),
            "a pass-entry snapshot declaration carries no texel bytes"
        );
        // The arm is one tag, exactly as the trace-produced arm is: the two
        // frames differ in that one byte (`4` against `3`) and nowhere else, so
        // the new declaration adds no payload an older decoder could read as
        // its own.
        let mut trace_view = trace;
        let Some(TracePass::Render(pass)) = trace_view.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.textures[0].source = TextureSource::TraceView;
        let other = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: trace_view,
            resources: resources(),
        })
        .unwrap();
        assert_eq!(other.len(), frame.len(), "one tag either way");
        let differing = frame
            .iter()
            .zip(&other)
            .filter(|(left, right)| left != right)
            .map(|(left, right)| (*left, *right))
            .collect::<Vec<_>>();
        assert_eq!(differing, vec![(4_u8, 3_u8)]);
    }

    #[test]
    fn a_sampled_render_pass_takes_its_own_tag_and_round_trips() {
        let request = CommandRequest::Submit {
            trace: sampled_multisample_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        // The sampled tag is followed by the same wide feature word the wide
        // tag carries and then by the block the word had no bit for: a `u8`
        // count and one full `TextureView` (`research/docs/23` §3.3, v70).
        assert!(
            frame
                .windows(4)
                .any(|window| window == [0x12, 0x20, 0x01, 0x01]),
            "the sampled pass carries its own tag, the wide word and the texture count"
        );
        // The texel bytes travel with the frame, so a provider can build the
        // image without a second declaration.
        let texels: Vec<u8> = (0..4u8)
            .flat_map(|y| (0..4u8).flat_map(move |x| [x, y, x.wrapping_add(y), 0xff]))
            .collect();
        assert!(
            frame.windows(texels.len()).any(|window| window == texels),
            "the sampled texture's own bytes travel in the frame"
        );
    }

    #[test]
    fn a_sampled_pass_is_the_pre_v70_frame_with_one_tag_and_one_block_more() {
        // The only difference between a pre-v70 wide frame and a v70 sampled
        // frame has to be the tag byte and the inserted block: the wide word
        // keeps its meanings, and every section after the block is byte
        // identical. This is the "old frames keep their bytes" rule stated as
        // a comparison instead of a golden vector.
        let plain_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: multisample_trace(),
            resources: resources(),
        })
        .unwrap();
        let sampled_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: sampled_multisample_trace(),
            resources: resources(),
        })
        .unwrap();
        // The nine-byte frame header carries the payload length, which the
        // block deliberately changes; every byte after it is what this test
        // compares.
        let plain = &plain_frame[9..];
        let sampled = &sampled_frame[9..];
        let position = plain
            .windows(3)
            .position(|window| window == [0x11, 0x20, 0x01])
            .expect("the plain fixture takes the wide tag");
        assert_eq!(&sampled[..position], &plain[..position]);
        assert_eq!(sampled[position], 0x12);
        assert_eq!(
            &sampled[position + 1..position + 3],
            &plain[position + 1..position + 3],
            "the sampled tag carries the same wide word"
        );
        assert_eq!(sampled[position + 3], 0x01, "one sampled texture");
        let block_length = sampled.len() - plain.len() - 1;
        assert_eq!(
            &sampled[position + 4 + block_length..],
            &plain[position + 3..],
            "every section after the block is byte identical"
        );
    }

    #[test]
    fn a_sampled_pass_refuses_a_texture_count_above_the_contract_cap() {
        use metal_api_core::provider::MAX_RENDER_TEXTURES;

        let frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: sampled_multisample_trace(),
            resources: resources(),
        })
        .unwrap();
        let position = frame
            .windows(4)
            .position(|window| window == [0x12, 0x20, 0x01, 0x01])
            .expect("the sampled fixture carries its own tag");
        let mut patched = frame.clone();
        // One above the contract's own ceiling (`research/docs/23` §3.3,
        // v102): the count byte is refused before a single texture is read, so
        // the patched frame needs no matching block behind it.
        patched[position + 3] = u8::try_from(MAX_RENDER_TEXTURES + 1).unwrap();
        assert!(matches!(
            CommandCodec::decode_request(&patched),
            Err(CodecError::RenderTextureCount {
                count,
                maximum,
            }) if count == MAX_RENDER_TEXTURES + 1 && maximum == MAX_RENDER_TEXTURES
        ));
    }

    /// A render trace whose pass states the pass-wide four-sample raster
    /// (`research/docs/23` §3.3, v51).
    fn multisample_trace() -> ComputeTrace {
        let mut trace = vertex_input_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        trace
    }

    #[test]
    fn a_multisample_pass_takes_the_wide_tag_and_round_trips() {
        let request = CommandRequest::Submit {
            trace: multisample_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        // The wide word's low byte is still the narrow byte (vertex input,
        // `0x01`); the high byte carries the multisample bit, so the pair reads
        // `20 01` — the wide word travels big-endian (`research/docs/23` §3.3,
        // v51).
        assert!(
            frame.windows(3).any(|window| window == [0x11, 0x20, 0x01]),
            "the multisample pass carries the sixth wide bit"
        );

        // A pass that never states the raster keeps the pre-v51 bytes: the
        // same fixture without the state carries the narrow feature byte.
        let plain = vertex_input_trace();
        let plain_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: plain,
            resources: resources(),
        })
        .unwrap();
        assert!(plain_frame.windows(2).any(|pair| pair == [0x10, 0x01]));
        assert!(!plain_frame
            .windows(3)
            .any(|window| window == [0x11, 0x20, 0x01]));
    }

    /// A render trace whose pass states the four-sample raster, keeps its
    /// depth surface and resolves it with a named filter
    /// (`research/docs/23` §3.3, v57).
    fn depth_resolve_trace() -> ComputeTrace {
        let mut trace = multisample_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.depth = Some(RenderDepthAttachment {
            format: DepthFormat::Depth32Float,
            width: 4,
            height: 4,
            load: DepthLoadOp::clear(1.0),
            store: Some(DepthStoreOp::Store),
            identity: Some(RenderDepthIdentity {
                allocation_id: AllocationId::new(940),
                view_id: ViewId::new(950),
            }),
        });
        pass.depth_test = Some(DepthTest {
            compare: CompareFunction::Less,
            write: true,
        });
        pass.depth_resolve = Some(MultisampleDepthResolve {
            filter: DepthResolveFilter::Min,
        });
        trace
    }

    #[test]
    fn a_depth_resolve_pass_takes_the_wide_tag_and_round_trips() {
        let request = CommandRequest::Submit {
            trace: depth_resolve_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        // The wide word's low byte is still the narrow byte (vertex input and
        // depth, `0x21`); the high byte carries the depth store, depth
        // resource, multisample and depth resolve bits (`0x01 0x02 0x20 0x40`
        // = `0x63`), so the pair reads `63 21` — the wide word travels
        // big-endian (`research/docs/23` §3.3, v57).
        assert!(
            frame.windows(3).any(|window| window == [0x11, 0x63, 0x21]),
            "the depth resolve pass carries the seventh wide bit"
        );

        // A pass that never resolves keeps the pre-v57 bytes: the same
        // fixture without the filter is refused by the contract's validate
        // (stored multisampled depth without a resolve), but its frame is the
        // multisample plus depth-store word with no depth resolve bit.
        let mut plain = depth_resolve_trace();
        let Some(TracePass::Render(pass)) = plain.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.depth_resolve = None;
        let plain_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: plain,
            resources: resources(),
        })
        .unwrap();
        assert!(!plain_frame
            .windows(3)
            .any(|window| window == [0x11, 0x63, 0x21]));
    }

    #[test]
    fn depth_resolve_frames_round_trip_every_admitted_filter() {
        // The v57 wire carries one filter code per pass, so each of the three
        // admitted filters has to survive its own round trip — the first
        // fixture only pinned `Min` (`research/docs/23` §3.3, v57).
        for (filter, code) in [
            (DepthResolveFilter::Sample0, 0x00),
            (DepthResolveFilter::Min, 0x01),
            (DepthResolveFilter::Max, 0x02),
        ] {
            let mut trace = depth_resolve_trace();
            let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
                panic!("the fixture is a render pass");
            };
            pass.depth_resolve = Some(MultisampleDepthResolve { filter });
            let request = CommandRequest::Submit {
                trace,
                resources: resources(),
            };
            let frame = CommandCodec::encode_request(&request).unwrap();
            assert_eq!(
                CommandCodec::decode_request(&frame).unwrap(),
                request,
                "the {filter:?} filter survives the round trip"
            );
            // The filter byte follows the depth identity's view tail and the
            // four-sample count, so each admitted code travels in its own byte.
            let position = frame
                .windows(10)
                .enumerate()
                .skip(10)
                .find(|(_, window)| {
                    *window == [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xB6, 0x02, code]
                })
                .map(|(index, _)| index + 9)
                .unwrap_or_else(|| panic!("the frame carries the {filter:?} filter byte"));
            assert_eq!(frame[position], code);
        }
    }

    #[test]
    fn depth_resolve_frames_refuse_unknown_filter_codes() {
        let request = CommandRequest::Submit {
            trace: depth_resolve_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();

        // The filter code travels one byte after the multisample section,
        // which sits right after the depth block, its store action and its
        // identity. The contract admits codes 0/1/2 only, so a fourth code is
        // a decoder refusal rather than a filter the caller did not ask for.
        let mut patched = frame.clone();
        // The depth identity (allocation 940, view 950) travels as two
        // big-endian `u64`s; the ten-byte run ends that identity with the
        // four-sample count (`0x02`) followed by the Min filter code (`0x01`),
        // so its last byte is the filter the decode would read.
        let filter = frame
            .windows(10)
            .enumerate()
            .skip(10)
            .find(|(_, window)| {
                *window == [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xB6, 0x02, 0x01]
            })
            .map(|(index, _)| index + 9)
            .expect("the depth identity precedes the multisample count and the filter");
        patched[filter] = 0x07;
        assert!(matches!(
            CommandCodec::decode_request(&patched),
            Err(CodecError::UnknownEnumValue {
                field: "depth resolve filter",
                value: 0x07,
            })
        ));
    }

    #[test]
    fn a_pass_that_both_culls_and_blends_round_trips_in_the_declared_section_order() {
        // The encoder writes culling before blending, in the order the two
        // feature bits are declared (`research/docs/23` §3.3, v39/v40). No
        // fixture states both, so the decoder walked them in the other order
        // until v43 — a latent disagreement this round trip pins shut.
        let mut trace = cull_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.blend = Some(RenderPassBlend {
            attachments: vec![BlendAttachment {
                enabled: true,
                source_rgb: BlendFactor::SourceAlpha,
                destination_rgb: BlendFactor::OneMinusSourceAlpha,
                source_alpha: BlendFactor::SourceAlpha,
                destination_alpha: BlendFactor::OneMinusSourceAlpha,
                operation: BlendOperation::Add,
                alpha_operation: BlendOperation::Add,
                write_mask: ColorWriteMask::ALL,
            }],
        });
        let request = CommandRequest::Submit {
            trace,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert!(
            frame.windows(2).any(|pair| pair == [0x10, 0xC1]),
            "the pass carries the vertex-input, cull and blend bits"
        );
    }

    #[test]
    fn instancing_capability_bits_round_trip_and_extend_the_vertex_frame() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_passes = true;
        capabilities.max_color_attachments = 1;
        capabilities.max_attachment_dimension = [2, 2];
        capabilities.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        capabilities.max_vertex_buffers = MAX_VERTEX_BUFFERS as u32;
        capabilities.supported_vertex_formats = VertexFormat::ADMITTED.to_vec();
        capabilities.supported_index_formats = IndexFormat::ADMITTED.to_vec();
        assert!(!capabilities.declares_instancing_support());
        let vertex_only = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert_eq!(
            CommandCodec::decode_response(&vertex_only).unwrap(),
            CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(1),
                capabilities: capabilities.clone(),
            }
        );

        capabilities.supports_render_instancing = true;
        capabilities.max_render_instances = 4;
        assert!(capabilities.declares_instancing_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        // The instancing block is the extended payload's newest optional
        // section: one presence tag, one bool and one `u32`, so a snapshot that
        // declares the vertex-input bits alone keeps the shorter frame.
        assert_eq!(frame.len(), vertex_only.len() + 6);

        // The instancing block is positional: a snapshot that declares it
        // *without* the vertex-input bits writes the presence tag directly
        // after the heap/ICB half, and the decoder has to read it there rather
        // than as a vertex-input block.
        let mut only_instancing = fake_capabilities();
        only_instancing.supports_render_passes = true;
        only_instancing.max_color_attachments = 1;
        only_instancing.max_attachment_dimension = [2, 2];
        only_instancing.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        only_instancing.supports_render_instancing = true;
        only_instancing.max_render_instances = 4;
        let only_frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: only_instancing.clone(),
        })
        .unwrap();
        assert_eq!(
            CommandCodec::decode_response(&only_frame).unwrap(),
            CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(1),
                capabilities: only_instancing,
            }
        );

        // A decoder that predates the section refuses the tag rather than
        // reading it as another section's bytes.
        let mut patched = frame.clone();
        let tag = frame
            .iter()
            .rposition(|byte| *byte == 0x02)
            .expect("the instancing tail carries its presence tag");
        patched[tag] = 0x03;
        assert!(matches!(
            CommandCodec::decode_response(&patched),
            Err(CodecError::UnknownCapabilityTail(0x03))
        ));
    }

    #[test]
    fn multisample_capability_bits_round_trip_and_extend_the_instancing_frame() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_passes = true;
        capabilities.max_color_attachments = 1;
        capabilities.max_attachment_dimension = [2, 2];
        capabilities.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        assert!(!capabilities.declares_multisample_support());
        let plain = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert!(capabilities.declares_render_support());
        assert_eq!(
            CommandCodec::decode_response(&plain).unwrap(),
            CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(1),
                capabilities: capabilities.clone(),
            }
        );

        capabilities.supports_render_multisample = true;
        capabilities.max_render_sample_count = 4;
        assert!(capabilities.declares_multisample_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        // The multisample block is the extended payload's newest optional
        // section: one presence tag, one bool and one `u32`. A snapshot that
        // declares no multisample bit keeps the shorter frame; one that
        // declares only multisampling still writes the heap/ICB half the
        // decoder reads by position before the tag (31 bytes: two bools, a
        // `u64`, a `u32` and two length-prefixed empty lists), so the
        // difference is that half plus this block
        // (`research/docs/23` §3.3, v51).
        assert_eq!(frame.len(), plain.len() + 31 + 6);

        // The block is positional: a snapshot that declares it *without* the
        // vertex-input and instancing bits writes the presence tag directly
        // after the heap/ICB half, and the decoder has to read it there.
        let mut only_multisample = fake_capabilities();
        only_multisample.supports_render_passes = true;
        only_multisample.max_color_attachments = 1;
        only_multisample.max_attachment_dimension = [2, 2];
        only_multisample.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        only_multisample.supports_render_multisample = true;
        only_multisample.max_render_sample_count = 4;
        let only_frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: only_multisample.clone(),
        })
        .unwrap();
        assert_eq!(
            CommandCodec::decode_response(&only_frame).unwrap(),
            CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(1),
                capabilities: only_multisample,
            }
        );

        // A decoder that predates the section refuses the tag rather than
        // reading it as another section's bytes.
        let mut patched = frame.clone();
        let tag = frame
            .windows(6)
            .rposition(|window| window == [0x04, 0x01, 0x00, 0x00, 0x00, 0x04])
            .expect("the multisample tail carries its presence tag, its bool and its count");
        patched[tag] = 0x11;
        assert!(matches!(
            CommandCodec::decode_response(&patched),
            Err(CodecError::UnknownCapabilityTail(0x11))
        ));
    }

    #[test]
    fn depth_resolve_capability_bits_round_trip_and_extend_the_multisample_frame() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_passes = true;
        capabilities.max_color_attachments = 1;
        capabilities.max_attachment_dimension = [2, 2];
        capabilities.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        assert!(!capabilities.declares_depth_resolve_support());
        let plain = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert!(capabilities.declares_render_support());
        assert_eq!(
            CommandCodec::decode_response(&plain).unwrap(),
            CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(1),
                capabilities: capabilities.clone(),
            }
        );

        capabilities.supports_render_depth_resolve = true;
        capabilities.depth_resolve_modes = 0b101;
        assert!(capabilities.declares_depth_resolve_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        // The depth-resolve block is the extended payload's newest optional
        // section: one presence tag, one bool and one `u32` bitmask. A
        // snapshot that declares no depth-resolve bit keeps the shorter frame;
        // one that declares only depth resolving still writes the heap/ICB
        // half the decoder reads by position before the tag (31 bytes), so
        // the difference is that half plus this block
        // (`research/docs/23` §3.3, v57).
        assert_eq!(frame.len(), plain.len() + 31 + 6);

        // The block is positional: a snapshot that declares it *without* the
        // vertex-input, instancing and multisample bits writes the presence
        // tag directly after the heap/ICB half, and the decoder has to read
        // it there.
        let mut only_depth_resolve = fake_capabilities();
        only_depth_resolve.supports_render_passes = true;
        only_depth_resolve.max_color_attachments = 1;
        only_depth_resolve.max_attachment_dimension = [2, 2];
        only_depth_resolve.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        only_depth_resolve.supports_render_depth_resolve = true;
        only_depth_resolve.depth_resolve_modes = 0b101;
        let only_frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: only_depth_resolve.clone(),
        })
        .unwrap();
        assert_eq!(
            CommandCodec::decode_response(&only_frame).unwrap(),
            CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(1),
                capabilities: only_depth_resolve,
            }
        );

        // A decoder that predates the section refuses the tag rather than
        // reading it as another section's bytes.
        let mut patched = frame.clone();
        let tag = frame
            .windows(6)
            .rposition(|window| window == [0x08, 0x01, 0x00, 0x00, 0x00, 0x05])
            .expect("the depth-resolve tail carries its presence tag, its bool and its mask");
        patched[tag] = 0x11;
        assert!(matches!(
            CommandCodec::decode_response(&patched),
            Err(CodecError::UnknownCapabilityTail(0x11))
        ));
    }

    #[test]
    fn stencil_resolve_capability_bits_round_trip_and_extend_the_depth_resolve_frame() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_passes = true;
        capabilities.max_color_attachments = 1;
        capabilities.max_attachment_dimension = [2, 2];
        capabilities.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        capabilities.supports_render_depth_resolve = true;
        capabilities.depth_resolve_modes = 0b101;
        assert!(!capabilities.declares_stencil_resolve_support());
        let plain = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert!(capabilities.declares_render_support());
        assert_eq!(
            CommandCodec::decode_response(&plain).unwrap(),
            CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(1),
                capabilities: capabilities.clone(),
            }
        );

        capabilities.supports_render_stencil_resolve = true;
        capabilities.stencil_resolve_modes = 0b11;
        assert!(capabilities.declares_stencil_resolve_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        // The stencil-resolve block is the extended payload's newest optional
        // section: one presence tag, one bool and one `u32` bitmask. A
        // snapshot that declares no stencil-resolve bit keeps the shorter
        // frame; one that declares stencil resolving but nothing earlier still
        // writes the heap/ICB half the decoder reads by position before the
        // tag (31 bytes), so the difference is that half plus this block
        // (`research/docs/23` §3.3, v60).
        assert_eq!(frame.len(), plain.len() + 6);

        // The block is positional: a snapshot that declares it *without* the
        // vertex-input, instancing, multisample and depth-resolve bits writes
        // the presence tag directly after the heap/ICB half, and the decoder
        // has to read it there.
        let mut only_stencil_resolve = fake_capabilities();
        only_stencil_resolve.supports_render_passes = true;
        only_stencil_resolve.max_color_attachments = 1;
        only_stencil_resolve.max_attachment_dimension = [2, 2];
        only_stencil_resolve.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        only_stencil_resolve.supports_render_stencil_resolve = true;
        only_stencil_resolve.stencil_resolve_modes = 0b10;
        let only_frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: only_stencil_resolve.clone(),
        })
        .unwrap();
        assert_eq!(
            CommandCodec::decode_response(&only_frame).unwrap(),
            CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(1),
                capabilities: only_stencil_resolve,
            }
        );

        // A decoder that predates the section refuses a tag this version does
        // not know rather than reading it as another section's bytes.
        let mut patched = frame.clone();
        let tag = frame
            .windows(6)
            .rposition(|window| window == [0x10, 0x01, 0x00, 0x00, 0x00, 0x03])
            .expect("the stencil-resolve tail carries its presence tag, its bool and its mask");
        patched[tag] = 0x12;
        assert!(matches!(
            CommandCodec::decode_response(&patched),
            Err(CodecError::UnknownCapabilityTail(0x12))
        ));
    }

    #[test]
    fn render_texture_capability_bits_round_trip_and_extend_the_stencil_resolve_frame() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_passes = true;
        capabilities.max_color_attachments = 1;
        capabilities.max_attachment_dimension = [2, 2];
        capabilities.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        capabilities.supports_render_stencil_resolve = true;
        capabilities.stencil_resolve_modes = 0b11;
        assert!(!capabilities.declares_render_texture_support());
        let plain = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert_eq!(
            CommandCodec::decode_response(&plain).unwrap(),
            CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(1),
                capabilities: capabilities.clone(),
            }
        );

        capabilities.supports_render_texture_sampling = true;
        capabilities.max_render_textures = 1;
        capabilities.supported_render_texture_formats = vec![TextureFormat::Rgba8Unorm];
        assert!(capabilities.declares_render_texture_support());
        assert!(capabilities.declares_render_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        // The render-sampler block is the extended payload's newest optional
        // section: one presence tag, one bool, one `u32` binding cap, one
        // `u64` format count and one byte per format (`research/docs/23`
        // §3.3, v70).
        let block = [
            0x20, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x02,
        ];
        assert!(
            frame.windows(block.len()).any(|window| window == block),
            "the render-sampler tail carries its tag, its bool, its cap and its format list"
        );
        assert_eq!(frame.len(), plain.len() + block.len());

        // The v51 half guard, one section later: a snapshot that declares
        // *only* the render-sampler bits still writes the heap/ICB half the
        // decoder reads by position before the tag, so its declaration cannot
        // be dropped on the wire.
        let mut only_render_texture = fake_capabilities();
        only_render_texture.supports_render_passes = true;
        only_render_texture.max_color_attachments = 1;
        only_render_texture.max_attachment_dimension = [2, 2];
        only_render_texture.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        only_render_texture.supports_render_texture_sampling = true;
        only_render_texture.max_render_textures = 1;
        only_render_texture.supported_render_texture_formats =
            vec![TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Unorm];
        let only_frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: only_render_texture.clone(),
        })
        .unwrap();
        assert_eq!(
            CommandCodec::decode_response(&only_frame).unwrap(),
            CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(1),
                capabilities: only_render_texture,
            }
        );

        // A decoder that predates the section refuses a tag this version does
        // not know rather than reading it as another section's bytes. The
        // tail's presence tags are powers of two, so `0x7f` is a byte no
        // version of this walk can ever assign.
        let mut patched = frame.clone();
        let tag = frame
            .windows(block.len())
            .position(|window| window == block)
            .expect("the render-sampler tail carries its presence tag");
        patched[tag] = 0x7f;
        assert!(matches!(
            CommandCodec::decode_response(&patched),
            Err(CodecError::UnknownCapabilityTail(0x7f))
        ));
    }

    #[test]
    fn the_narrow_texture_format_codes_are_appended_and_read_back() {
        // The narrow lanes (`research/docs/23` §113): the wire's texture-format
        // code is the *last* byte of the render-sampler block, so a snapshot
        // declaring one format is one frame whose tail is that code. The five
        // codes before this increment keep their values — a renumbering would
        // read an old frame as another format — and the two narrow formats are
        // appended at 5 and 6.
        let codes = [
            (TextureFormat::R32Uint, 0u8),
            (TextureFormat::R32Float, 1),
            (TextureFormat::Rgba8Unorm, 2),
            (TextureFormat::Bgra8Unorm, 3),
            (TextureFormat::Rgba16Float, 4),
            (TextureFormat::R8Unorm, 5),
            (TextureFormat::R8G8Unorm, 6),
        ];
        for (format, code) in codes {
            let mut capabilities = fake_capabilities();
            capabilities.supports_render_passes = true;
            capabilities.max_color_attachments = 1;
            capabilities.max_attachment_dimension = [2, 2];
            capabilities.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
            capabilities.supports_render_texture_sampling = true;
            capabilities.max_render_textures = 1;
            capabilities.supported_render_texture_formats = vec![format];
            let response = CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(1),
                capabilities,
            };
            let frame = CommandCodec::encode_response(&response).unwrap();
            assert_eq!(frame.last(), Some(&code), "{format:?} travels as {code}");
            assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        }
    }

    #[test]
    fn a_legacy_decoder_refuses_the_narrow_codes_by_name() {
        // The other direction of the same append (`research/docs/23` §113): a
        // frame from a future version that names a code this one has never
        // assigned is refused whole rather than read as an older format. The
        // code sits in the render-sampler block's format list, so the patched
        // byte is the frame's last one.
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_passes = true;
        capabilities.max_color_attachments = 1;
        capabilities.max_attachment_dimension = [2, 2];
        capabilities.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        capabilities.supports_render_texture_sampling = true;
        capabilities.max_render_textures = 1;
        capabilities.supported_render_texture_formats = vec![TextureFormat::Rgba8Unorm];
        let frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        })
        .unwrap();
        // `7` is the code the 2026-09-19 widening assigned to `r16_float`
        // (census b10's `texture_shape` bucket), so the probe's first unknown
        // moves to `8` — the family's next unassigned code — exactly as it did
        // when the two narrow lanes took `5` and `6`.
        for unknown in [8u8, 0x80, 0xff] {
            let mut patched = frame.clone();
            *patched.last_mut().expect("the frame is non-empty") = unknown;
            assert!(matches!(
                CommandCodec::decode_response(&patched),
                Err(CodecError::UnknownEnumValue {
                    field: "texture format",
                    value,
                }) if value == unknown
            ));
        }
    }

    #[test]
    fn stage_buffer_capability_bits_round_trip_and_extend_the_render_texture_frame() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_passes = true;
        capabilities.max_color_attachments = 1;
        capabilities.max_attachment_dimension = [2, 2];
        capabilities.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        capabilities.supports_render_texture_sampling = true;
        capabilities.max_render_textures = 1;
        capabilities.supported_render_texture_formats = vec![TextureFormat::Rgba8Unorm];
        let plain = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert_eq!(
            CommandCodec::decode_response(&plain).unwrap(),
            CommandResponse::Capabilities {
                epoch: DeviceEpoch::new(1),
                capabilities: capabilities.clone(),
            }
        );

        capabilities.supports_render_stage_buffers = true;
        capabilities.max_render_stage_buffers = MAX_RENDER_STAGE_BUFFERS as u32;
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the capability frame re-encodes byte for byte"
        );
        // The stage-buffer block is the extended payload's newest optional
        // section: one presence tag, one bool and one `u32` binding cap
        // (`research/docs/23` §3.3, v83).
        let mut block = vec![0x40, 0x01];
        block.extend_from_slice(&(MAX_RENDER_STAGE_BUFFERS as u32).to_be_bytes());
        assert!(
            frame.windows(block.len()).any(|window| window == block),
            "the stage-buffer tail carries its tag, its bool and its cap"
        );
        assert_eq!(frame.len(), plain.len() + block.len());
        // A decoder that predates the section refuses a tag this version does
        // not know rather than reading it as another section's bytes. The
        // tail's presence tags are powers of two, so `0x7f` is a byte no
        // version of this walk can ever assign.
        let mut patched = frame.clone();
        let tag = frame
            .windows(block.len())
            .position(|window| window == block)
            .expect("the stage-buffer tail carries its presence tag");
        patched[tag] = 0x7f;
        assert!(matches!(
            CommandCodec::decode_response(&patched).unwrap_err(),
            CodecError::UnknownCapabilityTail(0x7f)
        ));
        eprintln!(
            "stage-buffer capability frame: len={} plain={} block={:02x?}",
            frame.len(),
            plain.len(),
            block
        );
    }

    #[test]
    fn an_only_stage_buffer_declaration_still_writes_the_extended_payload() {
        // The two stage buffer bits are part of the extended payload's own
        // question: a snapshot that declares one of them without any earlier
        // render bit still writes the extended frame, so the declaration
        // cannot be dropped on the wire (`research/docs/23` §3.3, v83).
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_stage_buffers = true;
        capabilities.max_render_stage_buffers = MAX_RENDER_STAGE_BUFFERS as u32;
        assert!(!capabilities.declares_render_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(frame[9], 0x0a, "the extended capability tag");
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the only-stage-buffer capability frame re-encodes byte for byte"
        );
        let mut block = vec![0x40, 0x01];
        block.extend_from_slice(&(MAX_RENDER_STAGE_BUFFERS as u32).to_be_bytes());
        assert!(
            frame.windows(block.len()).any(|window| window == block),
            "the stage-buffer tail carries its tag, its bool and its cap"
        );
        // A snapshot with both bits at their defaults keeps the legacy frame,
        // and the decoder reads the missing block as the two "cannot bind a
        // stage buffer" defaults provider admission refuses the shape with.
        let legacy = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: fake_capabilities(),
        })
        .unwrap();
        assert_eq!(legacy[9], 0x01, "the legacy capability tag");
        let decoded = match CommandCodec::decode_response(&legacy).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the legacy frame decodes to capabilities, got {other:?}"),
        };
        assert!(!decoded.supports_render_stage_buffers);
        assert_eq!(decoded.max_render_stage_buffers, 0);
        eprintln!(
            "only stage buffer bits: extended len={} legacy len={} flags={}",
            frame.len(),
            legacy.len(),
            decoded.max_render_stage_buffers
        );
    }

    /// The folded shape's bit travels in its own extended block
    /// (`research/docs/23` §3.3, E-TX9).
    ///
    /// The block is the capability tail's ninth section, so it cannot use a
    /// bit-flag tag the eight before it have all taken: it is introduced by the
    /// `0x00` escape and then the second family's own tag. The readings are the
    /// increment's three wire obligations — the round trip keeps every field,
    /// the bytes before the block are the frame the same snapshot writes
    /// without it (so no earlier block moved), and the frame re-encodes byte
    /// for byte.
    #[test]
    fn the_folded_shape_bit_travels_in_its_own_extended_block() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_passes = true;
        capabilities.max_color_attachments = 1;
        capabilities.max_attachment_dimension = [2, 2];
        capabilities.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        capabilities.supports_render_texture_sampling = true;
        capabilities.max_render_textures = 1;
        capabilities.supported_render_texture_formats = vec![TextureFormat::Rgba8Unorm];
        capabilities.supports_render_stage_buffers = true;
        capabilities.max_render_stage_buffers = MAX_RENDER_STAGE_BUFFERS as u32;
        // The frame the same snapshot writes with the bit at its default: the
        // new block's absence has to leave every byte before it exactly where
        // the previous increment put them.
        let without = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();

        capabilities.supports_render_stage_buffer_namespace_split = true;
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the capability frame re-encodes byte for byte"
        );
        // One escape byte, one tag from the second family and one bool.
        let block = [0x00, 0x01, 0x01];
        assert_eq!(
            &frame[frame.len() - block.len()..],
            &block,
            "the folded-shape block is the tail's last section"
        );
        assert_eq!(frame.len(), without.len() + block.len());
        // The frame header states the payload's length, so "the sections
        // before it keep their bytes" is a statement about the payload: the
        // longer frame's payload starts with the shorter frame's payload,
        // byte for byte.
        assert_eq!(
            &frame[FRAME_HEADER..frame.len() - block.len()],
            &without[FRAME_HEADER..],
            "the sections before it keep their bytes"
        );
        eprintln!(
            "folded-shape capability frame: len={} without={} block={block:02x?}",
            frame.len(),
            without.len()
        );
    }

    /// A frame that ends before the folded-shape block reads the bit as
    /// `false`, and a tag the walk does not know is refused rather than read as
    /// another section's bytes (`research/docs/23` §3.3, E-TX9).
    #[test]
    fn a_frame_without_the_folded_shape_block_reads_the_bit_as_false() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_stage_buffers = true;
        capabilities.max_render_stage_buffers = MAX_RENDER_STAGE_BUFFERS as u32;
        capabilities.supports_render_stage_buffer_namespace_split = true;
        let frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();

        // The pre-increment frame is this frame's payload without its three
        // trailing bytes, reframed: the walk finds nothing after the
        // compute-texture block (this snapshot declares none) and keeps the
        // bit's default.
        let mut prior_capabilities = capabilities.clone();
        prior_capabilities.supports_render_stage_buffer_namespace_split = false;
        let expected = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: prior_capabilities,
        };
        let prior = CommandCodec::encode_response(&expected).unwrap();
        assert_eq!(
            &prior[FRAME_HEADER..],
            &frame[FRAME_HEADER..frame.len() - 3],
            "the pre-increment payload is this payload without the block"
        );
        assert_eq!(CommandCodec::decode_response(&prior).unwrap(), expected);

        // The escape byte is the walk's own reserved value, and the second
        // family's tags are a closed set: an unknown one is a typed refusal,
        // not a silent read of the block's payload as another section.
        let mut unknown_escape = frame.clone();
        let escape_at = unknown_escape.len() - 3;
        unknown_escape[escape_at] = 0x04;
        assert!(matches!(
            CommandCodec::decode_response(&unknown_escape).unwrap_err(),
            CodecError::UnknownCapabilityTail(0x04)
        ));
        let mut unknown_tag = frame.clone();
        let tag_at = unknown_tag.len() - 2;
        unknown_tag[tag_at] = 0x7f;
        assert!(matches!(
            CommandCodec::decode_response(&unknown_tag).unwrap_err(),
            CodecError::UnknownCapabilityTail(0x7f)
        ));
    }

    /// The per-stage stage-buffer window travels in the escape family's
    /// *seventh* block (`research/docs/23` §3.3, §117 E-SB2).
    ///
    /// It is the family's first section that carries a number rather than a
    /// bool: the window is the provider's own answer — the contract's ceiling
    /// clamped by the device's `maxPerStageDescriptorStorageBuffers` — so a
    /// consumer gating a draw on this face reads the value instead of a bit.
    /// A frame that ends before the section reads the older and *stricter*
    /// rule (the list bound applies to the whole list), which is what every
    /// pre-E-SB2 snapshot meant by [`MAX_RENDER_STAGE_BUFFERS`], so the
    /// absent-section direction is fail-closed and the frames keep their bytes
    /// everywhere before it.
    #[test]
    fn the_per_stage_stage_buffer_window_travels_in_its_own_extended_block() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_stage_buffers = true;
        capabilities.max_render_stage_buffers = MAX_RENDER_STAGE_BUFFER_DECLARATIONS as u32;
        let without = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();

        capabilities.max_render_stage_buffers_per_stage = MAX_RENDER_STAGE_BUFFERS as u32;
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the capability frame re-encodes byte for byte"
        );
        // One escape byte, the family's seventh tag and one big-endian `u32`.
        let block = [0x00, 0x07, 0x00, 0x00, 0x00, 0x08];
        assert_eq!(
            &frame[frame.len() - block.len()..],
            &block,
            "the per-stage window's block is the tail's last section"
        );
        assert_eq!(frame.len(), without.len() + block.len());
        assert_eq!(
            &frame[FRAME_HEADER..frame.len() - block.len()],
            &without[FRAME_HEADER..],
            "the sections before it keep their bytes"
        );
        // The frame without the section is the pre-E-SB2 reading: the window
        // stays `0`, which is the "the list bound is the whole rule" arm, and
        // the value is what the section's payload carries rather than a second
        // presence flag.
        let decoded = CommandCodec::decode_response(&without).unwrap();
        let CommandResponse::Capabilities { capabilities, .. } = decoded else {
            panic!("the response is a capability snapshot");
        };
        assert_eq!(capabilities.max_render_stage_buffers_per_stage, 0);
        assert!(!capabilities.declares_render_stage_buffer_per_stage_ceiling());
        eprintln!(
            "per-stage stage-buffer window frame: len={} without={} block={block:02x?}",
            frame.len(),
            without.len()
        );
    }

    /// The gathered-extent shape's bit travels in the escape family's *second*
    /// block (`research/docs/23` §3.3, E-TX10).
    ///
    /// The block follows the folded-shape one and carries the family's next
    /// tag, so the frame is the pre-increment frame with one more `0x00 <tag>
    /// <bool>` section appended. The readings are the increment's wire
    /// obligations: the round trip keeps every field, the bytes before the
    /// block are the frame the same snapshot writes without it (so neither the
    /// folded-shape block nor anything before it moved), and the frame
    /// re-encodes byte for byte.
    #[test]
    fn the_gathered_extent_bit_travels_in_the_family_s_second_block() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_passes = true;
        capabilities.max_color_attachments = 1;
        capabilities.max_attachment_dimension = [2, 2];
        capabilities.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        capabilities.supports_render_texture_sampling = true;
        capabilities.max_render_textures = 1;
        capabilities.supported_render_texture_formats = vec![TextureFormat::Rgba8Unorm];
        capabilities.supports_render_stage_buffers = true;
        capabilities.max_render_stage_buffers = MAX_RENDER_STAGE_BUFFERS as u32;
        capabilities.supports_render_stage_buffer_namespace_split = true;
        // The frame the same snapshot writes with the gathered shape at its
        // default: the new block's absence has to leave every byte before it
        // exactly where the previous increment put them.
        let without = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();

        capabilities.supports_render_texture_gathered_extent = true;
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the capability frame re-encodes byte for byte"
        );
        // One escape byte, the family's next tag and one bool.
        let block = [0x00, 0x02, 0x01];
        assert_eq!(
            &frame[frame.len() - block.len()..],
            &block,
            "the gathered-extent block is the tail's last section"
        );
        assert_eq!(frame.len(), without.len() + block.len());
        assert_eq!(
            &frame[FRAME_HEADER..frame.len() - block.len()],
            &without[FRAME_HEADER..],
            "the sections before it keep their bytes"
        );
        // The folded-shape block is still exactly the section before it: a
        // decoder that stops after the family's first block (the E-TX9 walk)
        // leaves these three bytes unconsumed rather than reading the new
        // section as another block's payload.
        assert_eq!(
            &frame[frame.len() - 6..frame.len() - 3],
            &[0x00, 0x01, 0x01],
            "the folded-shape block keeps its place in front of the new one"
        );
        eprintln!(
            "gathered-extent capability frame: len={} without={} block={block:02x?}",
            frame.len(),
            without.len()
        );
    }

    /// A snapshot that declares *only* the gathered shape still writes the
    /// extended payload, and the three render-sampler fields keep their own
    /// readings beside it (`research/docs/23` §3.3, E-TX10).
    #[test]
    fn an_only_gathered_extent_declaration_still_writes_the_extended_payload() {
        let mut capabilities = fake_capabilities();
        assert!(!capabilities.declares_render_support());
        let prior = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert_eq!(prior[9], 0x01, "the default snapshot keeps the legacy tag");

        capabilities.supports_render_texture_gathered_extent = true;
        assert!(capabilities.declares_render_texture_gathered_extent_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(frame[9], 0x0a, "the extended capability tag");
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the only-gathered-shape capability frame re-encodes byte for byte"
        );
        // The declaration is the frame's last three bytes; the pre-increment
        // *code* wrote no frame with this shape at all (its tag guard answered
        // "legacy payload" and the declaration would have been dropped), which
        // is exactly the failure the bit's own guard exists to prevent. What a
        // decoder of the previous increment sees is the escape byte followed by
        // the family tag it does not know: a typed refusal rather than a
        // snapshot read as "the shape was not declared".
        assert_eq!(&frame[frame.len() - 3..], &[0x00, 0x02, 0x01]);
        let mut old_walk = frame.clone();
        old_walk.truncate(frame.len() - 3);
        assert_eq!(
            &old_walk[FRAME_HEADER..],
            &frame[FRAME_HEADER..frame.len() - 3],
            "the new block is the frame's only addition after the payload"
        );
        // The shape bit is not one of the three render-sampler fields: the
        // block they travel in stays unwritten, and the decoder reads their
        // defaults back.
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(!decoded.supports_render_texture_sampling);
        assert_eq!(decoded.max_render_textures, 0);
        assert!(decoded.supported_render_texture_formats.is_empty());
        assert!(decoded.supports_render_texture_gathered_extent);
        eprintln!(
            "only gathered extent: extended len={} prior len={}",
            frame.len(),
            prior.len()
        );
    }

    /// The frames the walk must refuse around the escape family
    /// (`research/docs/23` §3.3, E-TX10): a tag outside the family's closed
    /// set, a section the encoder cannot have written twice or out of order,
    /// and a byte that follows a family section without being the next
    /// section's escape.
    #[test]
    fn the_escape_family_refuses_tags_and_bytes_it_cannot_have_written() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_stage_buffers = true;
        capabilities.max_render_stage_buffers = MAX_RENDER_STAGE_BUFFERS as u32;
        capabilities.supports_render_stage_buffer_namespace_split = true;
        capabilities.supports_render_texture_gathered_extent = true;
        let frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert_eq!(&frame[frame.len() - 3..], &[0x00, 0x02, 0x01]);

        // A family tag the closed set does not name (`0x7f` is a byte no
        // version of the family assigns) is a typed refusal, not a payload
        // read as another section.
        let mut unknown_tag = frame.clone();
        let tag_at = unknown_tag.len() - 2;
        unknown_tag[tag_at] = 0x7f;
        assert!(matches!(
            CommandCodec::decode_response(&unknown_tag).unwrap_err(),
            CodecError::UnknownCapabilityTail(0x7f)
        ));

        // The family's sections are read in the one order the encoder writes
        // them: repeating the first tag is a section the encoder cannot have
        // placed there.
        let mut repeated = frame.clone();
        repeated[tag_at] = 0x01;
        assert!(matches!(
            CommandCodec::decode_response(&repeated).unwrap_err(),
            CodecError::UnknownCapabilityTail(0x01)
        ));

        // A byte after a family section that is not the next section's escape
        // would make the caller read a section that was never there, so the
        // walk refuses the byte instead of leaving its cursor on it.
        let mut stray_section = frame.clone();
        let escape_at = stray_section.len() - 3;
        stray_section[escape_at] = 0x40;
        assert!(matches!(
            CommandCodec::decode_response(&stray_section).unwrap_err(),
            CodecError::UnknownCapabilityTail(0x40)
        ));

        // The pre-increment frame is the payload without the new block: the
        // bit reads `false` and nothing else moved, which is the direction a
        // consumer of the previous increment sees.
        capabilities.supports_render_texture_gathered_extent = false;
        let prior = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        })
        .unwrap();
        assert_eq!(
            &prior[FRAME_HEADER..],
            &frame[FRAME_HEADER..frame.len() - 3],
            "the pre-increment payload is this payload without the block"
        );
        let decoded = match CommandCodec::decode_response(&prior) {
            Ok(CommandResponse::Capabilities { capabilities, .. }) => capabilities,
            other => panic!("the pre-increment frame decodes to capabilities, got {other:?}"),
        };
        assert!(decoded.supports_render_stage_buffer_namespace_split);
        assert!(!decoded.supports_render_texture_gathered_extent);
    }

    /// The superset vertex interface's bit travels in the escape family's
    /// *third* block (`research/docs/23` §3.3, E-TX11).
    ///
    /// The block follows the gathered-extent one and carries the family's next
    /// tag, so the frame is the pre-increment frame with one more
    /// `0x00 <tag> <bool>` section appended. The readings are the increment's
    /// wire obligations: the round trip keeps every field, the bytes before the
    /// block are the frame the same snapshot writes without it (so neither the
    /// gathered-extent block nor anything before it moved), and the frame
    /// re-encodes byte for byte.
    #[test]
    fn the_superset_vertex_interface_bit_travels_in_the_family_s_third_block() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_stage_buffers = true;
        capabilities.max_render_stage_buffers = MAX_RENDER_STAGE_BUFFERS as u32;
        capabilities.supports_render_stage_buffer_namespace_split = true;
        capabilities.supports_render_texture_gathered_extent = true;
        // The frame the same snapshot writes with the superset shape at its
        // default: the new block's absence has to leave every byte before it
        // exactly where the previous increment put them.
        let without = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert_eq!(&without[without.len() - 3..], &[0x00, 0x02, 0x01]);

        capabilities.supports_render_vertex_interface_superset = true;
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the capability frame re-encodes byte for byte"
        );
        // One escape byte, the family's third tag and one bool.
        let block = [0x00, 0x03, 0x01];
        assert_eq!(
            &frame[frame.len() - block.len()..],
            &block,
            "the superset interface's block is the tail's last section"
        );
        assert_eq!(frame.len(), without.len() + block.len());
        assert_eq!(
            &frame[FRAME_HEADER..frame.len() - block.len()],
            &without[FRAME_HEADER..],
            "the sections before it keep their bytes"
        );
        // The gathered-extent block is still exactly the section before it: a
        // decoder that stops after the family's second block (the E-TX10 walk)
        // leaves these three bytes unconsumed rather than reading the new
        // section as another block's payload.
        assert_eq!(
            &frame[frame.len() - 6..frame.len() - 3],
            &[0x00, 0x02, 0x01],
            "the gathered-extent block keeps its place in front of the new one"
        );
        eprintln!(
            "superset-interface capability frame: len={} without={} block={block:02x?}",
            frame.len(),
            without.len()
        );
    }

    /// A snapshot that declares *only* the superset vertex interface still
    /// writes the extended payload, and the three vertex-input fields keep
    /// their own readings beside it (`research/docs/23` §3.3, E-TX11).
    #[test]
    fn an_only_superset_interface_declaration_still_writes_the_extended_payload() {
        let mut capabilities = fake_capabilities();
        assert!(!capabilities.declares_render_support());
        let prior = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert_eq!(prior[9], 0x01, "the default snapshot keeps the legacy tag");

        capabilities.supports_render_vertex_interface_superset = true;
        assert!(capabilities.declares_render_vertex_interface_superset_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(frame[9], 0x0a, "the extended capability tag");
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the only-superset-interface capability frame re-encodes byte for byte"
        );
        // The declaration is the frame's last three bytes; the pre-increment
        // *code* wrote no frame with this shape at all (its tag guard answered
        // "legacy payload" and the declaration would have been dropped), which
        // is exactly the failure the bit's own guard exists to prevent. What a
        // decoder of the previous increment sees is the escape byte followed by
        // the family tag it does not know: a typed refusal rather than a
        // snapshot read as "the shape was not declared".
        assert_eq!(&frame[frame.len() - 3..], &[0x00, 0x03, 0x01]);
        let mut old_walk = frame.clone();
        old_walk.truncate(frame.len() - 3);
        assert_eq!(
            &old_walk[FRAME_HEADER..],
            &frame[FRAME_HEADER..frame.len() - 3],
            "the new block is the frame's only addition after the payload"
        );
        // The shape bit is not one of the three vertex-input fields: the block
        // they travel in stays unwritten, and the decoder reads their defaults
        // back.
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert_eq!(decoded.max_vertex_buffers, 0);
        assert!(decoded.supported_vertex_formats.is_empty());
        assert!(decoded.supported_index_formats.is_empty());
        assert!(decoded.supports_render_vertex_interface_superset);
        eprintln!(
            "only superset interface: extended len={} prior len={}",
            frame.len(),
            prior.len()
        );
    }

    /// The gathered extent's no-copy bit travels in the escape family's
    /// *fourth* block (`research/docs/23` §111, E-TX12).
    ///
    /// The block follows the superset interface's one and carries the family's
    /// next tag, so the frame is the pre-increment frame with one more
    /// `0x00 <tag> <bool>` section appended. The readings are the increment's
    /// wire obligations: the round trip keeps every field, the bytes before the
    /// block are the frame the same snapshot writes without it (so neither the
    /// superset interface's block nor anything before it moved), and the frame
    /// re-encodes byte for byte.
    #[test]
    fn the_no_copy_gathered_extent_bit_travels_in_the_family_s_fourth_block() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_stage_buffers = true;
        capabilities.max_render_stage_buffers = MAX_RENDER_STAGE_BUFFERS as u32;
        capabilities.supports_render_stage_buffer_namespace_split = true;
        capabilities.supports_render_texture_gathered_extent = true;
        capabilities.supports_render_vertex_interface_superset = true;
        // The frame the same snapshot writes with the no-copy arm at its
        // default: the new block's absence has to leave every byte before it
        // exactly where the previous increment put them.
        let without = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert_eq!(&without[without.len() - 3..], &[0x00, 0x03, 0x01]);

        capabilities.supports_render_texture_gathered_extent_no_copy = true;
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the capability frame re-encodes byte for byte"
        );
        // One escape byte, the family's fourth tag and one bool.
        let block = [0x00, 0x04, 0x01];
        assert_eq!(
            &frame[frame.len() - block.len()..],
            &block,
            "the no-copy gathered block is the tail's last section"
        );
        assert_eq!(frame.len(), without.len() + block.len());
        assert_eq!(
            &frame[FRAME_HEADER..frame.len() - block.len()],
            &without[FRAME_HEADER..],
            "the sections before it keep their bytes"
        );
        // The superset interface's block is still exactly the section before
        // it: a decoder that stops after the family's third block (the E-TX11
        // walk) leaves these three bytes unconsumed rather than reading the new
        // section as another block's payload.
        assert_eq!(
            &frame[frame.len() - 6..frame.len() - 3],
            &[0x00, 0x03, 0x01],
            "the superset interface's block keeps its place in front of the new one"
        );
        eprintln!(
            "no-copy gathered capability frame: len={} without={} block={block:02x?}",
            frame.len(),
            without.len()
        );
    }

    /// A snapshot that declares *only* the gathered extent's no-copy arm still
    /// writes the extended payload (`research/docs/23` §111, E-TX12).
    ///
    /// The three render-sampler fields and the other two shape bits keep their
    /// own readings beside it: the declaration travels in the frame's own
    /// tagged tail, so it must not be read as a statement about the sampling
    /// window or about the host-bytes arm.
    #[test]
    fn an_only_no_copy_gathered_declaration_still_writes_the_extended_payload() {
        let mut capabilities = fake_capabilities();
        assert!(!capabilities.declares_render_support());
        let prior = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert_eq!(prior[9], 0x01, "the default snapshot keeps the legacy tag");

        capabilities.supports_render_texture_gathered_extent_no_copy = true;
        assert!(capabilities.declares_render_texture_gathered_extent_no_copy_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(frame[9], 0x0a, "the extended capability tag");
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the only-no-copy capability frame re-encodes byte for byte"
        );
        // The declaration is the frame's last three bytes; the pre-increment
        // *code* wrote no frame with this shape at all (its tag guard answered
        // "legacy payload" and the declaration would have been dropped), which
        // is exactly the failure the bit's own guard exists to prevent. What a
        // decoder of the previous increment sees is the escape byte followed by
        // the family tag it does not know: a typed refusal rather than a
        // snapshot read as "the arm was not declared".
        assert_eq!(&frame[frame.len() - 3..], &[0x00, 0x04, 0x01]);
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(!decoded.supports_render_texture_sampling);
        assert_eq!(decoded.max_render_textures, 0);
        assert!(decoded.supported_render_texture_formats.is_empty());
        assert!(!decoded.supports_render_texture_gathered_extent);
        assert!(decoded.supports_render_texture_gathered_extent_no_copy);
        eprintln!(
            "only no-copy gathered: extended len={} prior len={}",
            frame.len(),
            prior.len()
        );
    }

    /// The landing-view arm's capability block is the tail family's fifth tag
    /// (`research/docs/23` §115 之后的增量，E-TX13).
    ///
    /// It follows the gathered extent's no-copy block, so the four sections
    /// before it keep their bytes; a decoder of the previous increment reads the
    /// escape byte followed by a family tag it does not know — a typed refusal
    /// rather than a snapshot silently read as "the arm was not declared".
    #[test]
    fn the_landing_view_block_is_the_tail_familys_fifth_tag() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_texture_gathered_extent_no_copy = true;
        let without = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert_eq!(&without[without.len() - 3..], &[0x00, 0x04, 0x01]);

        capabilities.supports_render_attachment_landing_view = true;
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the landing-view capability frame re-encodes byte for byte"
        );
        // One escape byte, the family's fifth tag and one bool.
        let block = [0x00, 0x05, 0x01];
        assert_eq!(
            &frame[frame.len() - block.len()..],
            &block,
            "the landing-view block is the tail's last section"
        );
        assert_eq!(frame.len(), without.len() + block.len());
        assert_eq!(
            &frame[FRAME_HEADER..frame.len() - block.len()],
            &without[FRAME_HEADER..],
            "the sections before it keep their bytes"
        );
        assert_eq!(
            &frame[frame.len() - 6..frame.len() - 3],
            &[0x00, 0x04, 0x01],
            "the no-copy gathered block keeps its place in front of the new one"
        );
    }

    /// A snapshot that declares *only* the landing-view arm still writes the
    /// extended payload (`research/docs/23` §115 之后的增量，E-TX13).
    ///
    /// The gathered-extent bits and the render-sampler fields keep their own
    /// readings beside it: the declaration travels in the frame's own tagged
    /// tail, so it must not be read as a statement about sampling.
    #[test]
    fn an_only_landing_view_declaration_still_writes_the_extended_payload() {
        let mut capabilities = fake_capabilities();
        assert!(!capabilities.declares_render_support());
        capabilities.supports_render_attachment_landing_view = true;
        assert!(capabilities.declares_render_attachment_landing_view_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(frame[9], 0x0a, "the extended capability tag");
        assert_eq!(&frame[frame.len() - 3..], &[0x00, 0x05, 0x01]);
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(decoded.supports_render_attachment_landing_view);
        assert!(!decoded.supports_render_texture_gathered_extent);
        assert!(!decoded.supports_render_texture_gathered_extent_no_copy);
        assert!(!decoded.supports_render_texture_sampling);
    }

    /// A runtime sampler block that states the coordinate axis (2026-09-19,
    /// census v43's `texture_state`) marks its header with the flag bit and
    /// appends one space byte per entry, and a decoder that predates the axis
    /// reads that header as a count above the contract's cap.
    #[test]
    fn a_pixel_coordinate_sampler_pass_marks_its_block_and_round_trips() {
        let mut trace = sampled_runtime_sampler_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.samplers = vec![RenderSamplerBinding::with_coordinates(
            0,
            SamplerPolicy {
                filter: SamplerFilter::LinearMipLinear,
                address: SamplerAddressMode::ClampToZero,
            },
            metal_api_core::provider::SamplerCoordinates::Pixel,
        )];
        let request = CommandRequest::Submit {
            trace,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame,
            "the pixel-coordinate sampler frame re-encodes byte for byte"
        );
        // The block is the normalized one's header with the flag bit set, one
        // coordinate-space byte appended, and nothing else moved: the entry's
        // four state bytes keep the codes the family published.
        const PIXEL_SAMPLER_BLOCK: [u8; 8] = [0x81, 0x00, 0x00, 0x00, 0x00, 0x05, 0x04, 0x01];
        assert!(
            frame
                .windows(PIXEL_SAMPLER_BLOCK.len())
                .any(|window| window == PIXEL_SAMPLER_BLOCK),
            "the flagged block is on the wire"
        );
        assert!(
            usize::from(PIXEL_SAMPLER_BLOCK[0]) > metal_api_core::provider::MAX_RENDER_SAMPLERS,
            "a decoder that predates the axis reads the flagged header as a count above the cap \
             and refuses the frame by name"
        );
    }

    /// The coordinate axis' capability block is the tail's second family's
    /// *eighth* tag, so it follows the per-stage stage-buffer window and
    /// touches nothing before it (2026-09-19, census v43's `texture_state`).
    ///
    /// A decoder of the previous increment reads the escape byte followed by a
    /// family tag it does not know — a typed refusal rather than a snapshot
    /// silently read as "the texel space was not declared".
    #[test]
    fn the_pixel_coordinate_sampler_block_is_the_tail_familys_eighth_tag() {
        let mut capabilities = fake_capabilities();
        // The frame has to be on the extended payload already, or the new bit
        // would change the payload's own form rather than only appending its
        // section: the landing-view bit is the oldest face that does that.
        capabilities.supports_render_attachment_landing_view = true;
        let without = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert!(without.ends_with(&[0x00, 0x05, 0x01]));
        assert!(!without.ends_with(&[0x00, 0x08, 0x01]));

        capabilities.supports_render_pixel_coordinate_sampler = true;
        assert!(capabilities.declares_render_pixel_coordinate_sampler_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the pixel-coordinate sampler capability frame re-encodes byte for byte"
        );
        let block = [0x00, 0x08, 0x01];
        assert_eq!(
            &frame[frame.len() - block.len()..],
            &block,
            "the new block is the tail's last section"
        );
        assert_eq!(frame.len(), without.len() + block.len());
        assert_eq!(
            &frame[FRAME_HEADER..frame.len() - block.len()],
            &without[FRAME_HEADER..],
            "the sections before it keep their bytes"
        );
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(decoded.supports_render_pixel_coordinate_sampler);
        // A frame that ends before the block reads the bit as the fail-closed
        // `false`, so a consumer keeps the census's refusal for the shape.
        let legacy = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: fake_capabilities(),
        })
        .unwrap();
        let decoded = match CommandCodec::decode_response(&legacy).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(!decoded.supports_render_pixel_coordinate_sampler);
    }

    /// The kept-frame landing block is the tail's second family's *sixth* tag,
    /// so it follows the landing-view block and touches nothing before it
    /// (`research/docs/23` §115 之后的增量，E-TX14/R4b).
    ///
    /// A decoder of the previous increment reads the escape byte followed by a
    /// family tag it does not know — a typed refusal rather than a snapshot
    /// silently read as "the entry was not declared".
    #[test]
    fn the_kept_frame_landing_block_is_the_tail_familys_sixth_tag() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_attachment_landing_view = true;
        let without = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert_eq!(&without[without.len() - 3..], &[0x00, 0x05, 0x01]);

        capabilities.supports_render_kept_frame_landing = true;
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the kept-frame landing capability frame re-encodes byte for byte"
        );
        let block = [0x00, 0x06, 0x01];
        assert_eq!(
            &frame[frame.len() - block.len()..],
            &block,
            "the kept-frame landing block is the tail's last section"
        );
        assert_eq!(frame.len(), without.len() + block.len());
        assert_eq!(
            &frame[FRAME_HEADER..frame.len() - block.len()],
            &without[FRAME_HEADER..],
            "the sections before it keep their bytes"
        );
        assert_eq!(
            &frame[frame.len() - 6..frame.len() - 3],
            &[0x00, 0x05, 0x01],
            "the landing-view block keeps its place in front of the new one"
        );
    }

    /// A snapshot that declares *only* the kept-frame landing entry still
    /// writes the extended payload (`research/docs/23` §115 之后的增量，
    /// E-TX14/R4b): the declaration travels in the frame's own tagged tail, so
    /// the bits beside it keep their own readings.
    #[test]
    fn an_only_kept_frame_landing_declaration_still_writes_the_extended_payload() {
        let mut capabilities = fake_capabilities();
        assert!(!capabilities.declares_render_support());
        capabilities.supports_render_kept_frame_landing = true;
        assert!(capabilities.declares_render_kept_frame_landing_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(frame[9], 0x0a, "the extended capability tag");
        assert_eq!(&frame[frame.len() - 3..], &[0x00, 0x06, 0x01]);
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(decoded.supports_render_kept_frame_landing);
        assert!(!decoded.supports_render_attachment_landing_view);
        assert!(!decoded.supports_render_texture_gathered_extent);
        assert!(!decoded.supports_render_texture_gathered_extent_no_copy);
        assert!(!decoded.supports_render_texture_sampling);

        // A frame that ends before the block reads the bit as the fail-closed
        // `false`, so a consumer refuses the entry rather than assuming it.
        let legacy = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: fake_capabilities(),
        })
        .unwrap();
        let decoded = match CommandCodec::decode_response(&legacy).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(!decoded.supports_render_kept_frame_landing);
    }

    /// The pass-entry snapshot block is the tail's second family's *eighth* tag,
    /// so it follows the per-stage stage-buffer window and touches nothing
    /// before it (`research/docs/23` §118, E-TX15). A decoder of the previous
    /// increment reads the escape byte followed by a family tag it does not
    /// know — a typed refusal rather than a snapshot silently read as some
    /// other section's payload.
    #[test]
    fn the_pass_entry_snapshot_block_is_the_tail_familys_ninth_tag() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_stage_buffers = true;
        capabilities.max_render_stage_buffers = 8;
        capabilities.max_render_stage_buffers_per_stage = 8;
        let without = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert_eq!(
            &without[without.len() - 6..],
            &[0x00, 0x07, 0x00, 0x00, 0x00, 0x08]
        );

        capabilities.supports_render_pass_entry_snapshot = true;
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the pass-entry snapshot capability frame re-encodes byte for byte"
        );
        let block = [0x00, 0x09, 0x01];
        assert_eq!(
            &frame[frame.len() - block.len()..],
            &block,
            "the pass-entry snapshot block is the tail's last section"
        );
        assert_eq!(frame.len(), without.len() + block.len());
        assert_eq!(
            &frame[FRAME_HEADER..frame.len() - block.len()],
            &without[FRAME_HEADER..],
            "the sections before it keep their bytes"
        );
    }

    /// A snapshot that declares *only* the pass-entry snapshot arm still writes
    /// the extended payload (`research/docs/23` §118, E-TX15): the declaration
    /// travels in the frame's own tagged tail, so the bits beside it keep their
    /// own readings, and a frame that ends before the block reads the arm as the
    /// fail-closed `false`.
    #[test]
    fn an_only_pass_entry_snapshot_declaration_still_writes_the_extended_payload() {
        let mut capabilities = fake_capabilities();
        assert!(!capabilities.declares_render_support());
        capabilities.supports_render_pass_entry_snapshot = true;
        assert!(capabilities.declares_render_pass_entry_snapshot_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(frame[9], 0x0a, "the extended capability tag");
        assert_eq!(&frame[frame.len() - 3..], &[0x00, 0x09, 0x01]);
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(decoded.supports_render_pass_entry_snapshot);
        assert!(!decoded.supports_render_kept_frame_landing);
        assert!(!decoded.supports_render_attachment_landing_view);
        assert!(!decoded.supports_render_texture_gathered_extent);
        assert!(!decoded.supports_render_texture_sampling);

        // A frame that ends before the block reads the bit as the fail-closed
        // `false`, so a consumer refuses the declaration rather than assuming
        // it.
        let legacy = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: fake_capabilities(),
        })
        .unwrap();
        let decoded = match CommandCodec::decode_response(&legacy).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(!decoded.supports_render_pass_entry_snapshot);
    }

    /// The layout-free vertex count above the milestone's three travels in the
    /// escape family's *eleventh* block (2026-09-19, census v45's `vertex_span`
    /// bucket).
    ///
    /// A decoder of the previous increment reads the escape byte followed by a
    /// family tag it does not know — a typed refusal rather than a snapshot
    /// silently read as "the widened count was not declared".
    #[test]
    fn the_layout_free_vertex_count_block_is_the_tail_familys_eleventh_tag() {
        let mut capabilities = fake_capabilities();
        // The frame has to be on the extended payload already, or the new bit
        // would change the payload's own form rather than only appending its
        // section: the landing-view bit is the oldest face that does that.
        capabilities.supports_render_attachment_landing_view = true;
        let without = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert!(without.ends_with(&[0x00, 0x05, 0x01]));
        assert!(!without.ends_with(&[0x00, 0x0b, 0x01]));

        capabilities.supports_render_vertex_count_above_triangle = true;
        assert!(capabilities.declares_render_vertex_count_above_triangle());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the layout-free vertex count frame re-encodes byte for byte"
        );
        let block = [0x00, 0x0b, 0x01];
        assert_eq!(
            &frame[frame.len() - block.len()..],
            &block,
            "the new block is the tail's last section"
        );
        assert_eq!(frame.len(), without.len() + block.len());
        assert_eq!(
            &frame[FRAME_HEADER..frame.len() - block.len()],
            &without[FRAME_HEADER..],
            "the sections before it keep their bytes"
        );
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(decoded.supports_render_vertex_count_above_triangle);
        // A frame that ends before the block reads the bit as the fail-closed
        // `false`, so a consumer keeps its own refusal by name for the shape.
        let legacy = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: fake_capabilities(),
        })
        .unwrap();
        let decoded = match CommandCodec::decode_response(&legacy).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(!decoded.supports_render_vertex_count_above_triangle);
    }

    /// A declaration whose *only* statement is the widened layout-free count
    /// still writes the extended payload: the block sits after the heap/ICB half
    /// the decoder reads by position before the family's escape, so a snapshot
    /// that never wrote that half would drop the declaration on the wire.
    #[test]
    fn an_only_layout_free_vertex_count_declaration_still_writes_the_extended_payload() {
        let mut capabilities = fake_capabilities();
        assert!(!capabilities.declares_render_support());
        capabilities.supports_render_vertex_count_above_triangle = true;
        assert!(capabilities.declares_render_vertex_count_above_triangle());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(frame[9], 0x0a, "the extended capability tag");
        assert_eq!(&frame[frame.len() - 3..], &[0x00, 0x0b, 0x01]);
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(decoded.supports_render_vertex_count_above_triangle);
        assert!(!decoded.supports_render_vertex_interface_superset);
        assert!(!decoded.max_render_texture_dimension_1d != 0);
        assert!(!decoded.supports_render_kept_frame_landing);
        assert!(!decoded.supports_heaps);
        assert!(!decoded.supports_indirect_command_buffers);
    }

    /// The superset fragment interface's block is the tail's second family's
    /// own last section (2026-09-20), written as `0x00 0x0e <bool>` after the
    /// layout-free vertex count.
    ///
    /// The reading is the same one every block in this family is pinned by:
    /// adding the declaration appends three bytes and leaves the frame before
    /// them untouched, a frame that ends before the block reads the bit as the
    /// fail-closed `false` (so a consumer keeps its own refusal by name for the
    /// shape), and a decoder that predates the tag refuses the frame rather
    /// than reading a value out of a family tag it does not know.
    #[test]
    fn the_superset_fragment_interface_block_is_the_tail_familys_last_tag() {
        let mut capabilities = fake_capabilities();
        // The frame has to be on the extended payload already, or the new bit
        // would change the payload's own form rather than only appending its
        // section: the landing-view bit is the oldest face that does that.
        capabilities.supports_render_attachment_landing_view = true;
        let without = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert!(without.ends_with(&[0x00, 0x05, 0x01]));
        assert!(!without.ends_with(&[0x00, 0x0e, 0x01]));
        let decoded = match CommandCodec::decode_response(&without).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(
            !decoded.supports_render_fragment_output_superset,
            "a frame that ends before the block reads the bit as the fail-closed default"
        );

        capabilities.supports_render_fragment_output_superset = true;
        assert!(capabilities.declares_render_fragment_output_superset_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the superset fragment interface frame re-encodes byte for byte"
        );
        let block = [0x00, 0x0e, 0x01];
        assert_eq!(
            &frame[frame.len() - block.len()..],
            &block,
            "the new block is the tail's last section"
        );
        assert_eq!(frame.len(), without.len() + block.len());
        assert_eq!(
            &frame[FRAME_HEADER..frame.len() - block.len()],
            &without[FRAME_HEADER..],
            "the sections before it keep their bytes"
        );
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(decoded.supports_render_fragment_output_superset);
        // The legacy payload cannot carry the block either, so a legacy frame
        // reads the same fail-closed default rather than a value beside it.
        let legacy = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: fake_capabilities(),
        })
        .unwrap();
        let decoded = match CommandCodec::decode_response(&legacy).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(!decoded.supports_render_fragment_output_superset);
    }

    /// The stage-buffer whole-binding block is the tail's escape family's next
    /// tag, `0x00 0x0c <bool>` (`research/docs/23` §3.3, E-SB3).
    ///
    /// The reading is that the section appends itself to a frame that was
    /// already on the extended payload, changes no earlier byte, and re-encodes
    /// after a decode; a frame that ends before it reads the bit as the
    /// fail-closed `false`, which is the consumer's own by-name refusal for the
    /// shape; and a family tag the walk does not know is still a typed refusal —
    /// a snapshot that sent `0x0d` (a section this increment does not define)
    /// would be refused rather than read as "the bit was not declared".
    #[test]
    fn the_whole_binding_block_is_the_tail_familys_next_tag() {
        let mut capabilities = fake_capabilities();
        // The frame has to be on the extended payload already, or the new bit
        // would change the payload's own form rather than only appending its
        // section: the stage-buffer pair and the landing-view bit are the two
        // faces that do that, and both are declared here so the only byte the
        // new bit can move is its own section.
        capabilities.supports_render_stage_buffers = true;
        capabilities.max_render_stage_buffers = 4;
        capabilities.supports_render_attachment_landing_view = true;
        let without = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();
        assert!(without
            .windows(3)
            .any(|window| window == [0x00, 0x05, 0x01]));
        assert!(!without.ends_with(&[0x00, 0x0c, 0x01]));

        capabilities.supports_render_stage_buffer_binding_range = true;
        assert!(capabilities.declares_render_stage_buffer_binding_range());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the whole-binding frame re-encodes byte for byte"
        );
        let block = [0x00, 0x0c, 0x01];
        assert_eq!(
            &frame[frame.len() - block.len()..],
            &block,
            "the new block is the tail's last section"
        );
        assert_eq!(frame.len(), without.len() + block.len());
        assert_eq!(
            &frame[FRAME_HEADER..frame.len() - block.len()],
            &without[FRAME_HEADER..],
            "the sections before it keep their bytes"
        );
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(decoded.supports_render_stage_buffer_binding_range);
        assert!(decoded.supports_render_attachment_landing_view);
        // A frame that ends before the block reads the bit as the fail-closed
        // `false`, so a consumer keeps its own refusal by name for the shape.
        let legacy = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: fake_capabilities(),
        })
        .unwrap();
        let decoded = match CommandCodec::decode_response(&legacy).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(!decoded.supports_render_stage_buffer_binding_range);

        // A family tag outside the closed set stays a typed refusal: the walk
        // must not read a section it does not define as "no declaration".
        // (0x7f rather than 0x0d: the three-dimensional window took 0x0d in the
        // same merge this test landed with, and the walk refuses a tag it does
        // not know, whichever number that is.)
        let mut frame_with_unknown = frame.clone();
        let last = frame_with_unknown.len() - 1;
        frame_with_unknown[last - 2] = 0x7f;
        let refused = CommandCodec::decode_response(&frame_with_unknown)
            .expect_err("an unknown family tag is refused");
        assert!(
            format!("{refused:?}").contains("UnknownCapabilityTail"),
            "the refusal is the tail's own typed arm: {refused:?}"
        );
    }

    /// A declaration whose *only* statement is the whole-binding arm still
    /// writes the extended payload (`research/docs/23` §3.3, E-SB3): the block
    /// sits after the heap/ICB half the decoder reads by position before the
    /// family's escape, so a snapshot that never wrote that half would drop the
    /// declaration on the wire.
    #[test]
    fn an_only_whole_binding_declaration_still_writes_the_extended_payload() {
        let mut capabilities = fake_capabilities();
        assert!(!capabilities.declares_render_support());
        capabilities.supports_render_stage_buffer_binding_range = true;
        assert!(capabilities.declares_render_stage_buffer_binding_range());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(frame[9], 0x0a, "the extended capability tag");
        assert_eq!(&frame[frame.len() - 3..], &[0x00, 0x0c, 0x01]);
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(decoded.supports_render_stage_buffer_binding_range);
        assert!(!decoded.supports_render_stage_buffers);
        assert!(!decoded.supports_render_vertex_count_above_triangle);
        assert!(!decoded.supports_heaps);
    }

    /// A declaration whose *only* statement is the superset fragment interface
    /// still writes the extended payload: the block sits after the heap/ICB half
    /// the decoder reads by position before the family's escape, so a snapshot
    /// that never wrote that half would drop the declaration on the wire.
    #[test]
    fn an_only_superset_fragment_interface_declaration_still_writes_the_extended_payload() {
        let mut capabilities = fake_capabilities();
        assert!(!capabilities.declares_render_support());
        capabilities.supports_render_fragment_output_superset = true;
        assert!(capabilities.declares_render_fragment_output_superset_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(frame[9], 0x0a, "the extended capability tag");
        assert_eq!(&frame[frame.len() - 3..], &[0x00, 0x0e, 0x01]);
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert!(decoded.supports_render_fragment_output_superset);
        assert!(!decoded.supports_render_vertex_interface_superset);
        assert!(!decoded.supports_render_vertex_count_above_triangle);
        assert!(!decoded.supports_render_kept_frame_landing);
        assert!(!decoded.supports_heaps);
        assert!(!decoded.supports_indirect_command_buffers);
    }

    /// The three-dimensional sampled window is the family's next tag and
    /// states an extent rather than a bit (2026-09-20, the `D3` sampled
    /// texture arm): a snapshot whose *only* statement is this window still
    /// writes the extended payload, the section is `0x00 0x0d` plus one
    /// big-endian `u64` — the one-dimensional window's own frame shape, for the
    /// reason `command_codec`'s constant states — and a snapshot that never
    /// spoke about the arm decodes as `0`, the fail-closed reading a consumer's
    /// own refusal by name is written against.
    #[test]
    fn an_only_three_dimensional_window_still_writes_the_extended_payload() {
        use metal_api_core::provider::MAX_RENDER_TEXTURE_DIMENSION_3D;

        let mut capabilities = fake_capabilities();
        assert!(!capabilities.declares_render_support());
        assert!(!capabilities.declares_render_texture_dimension_3d());
        capabilities.max_render_texture_dimension_3d = MAX_RENDER_TEXTURE_DIMENSION_3D;
        assert!(capabilities.declares_render_texture_dimension_3d());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(frame[9], 0x0a, "the extended capability tag");
        let mut section = vec![0x00, 0x0d];
        section.extend_from_slice(&MAX_RENDER_TEXTURE_DIMENSION_3D.to_be_bytes());
        assert_eq!(
            frame.windows(2).position(|pair| pair == [0x00, 0x0d]),
            Some(frame.len() - section.len()),
            "the window's section is the family's last one in this frame"
        );
        assert_eq!(
            &frame[frame.len() - section.len()..],
            &section[..],
            "the window is one presence tag, one family tag and one u64"
        );
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert_eq!(
            decoded.max_render_texture_dimension_3d,
            MAX_RENDER_TEXTURE_DIMENSION_3D
        );

        // The absent section reads zero: the same snapshot with the window back
        // at its default writes a shorter frame, and the decoder's answer for
        // the missing block is the fail-closed one.
        let mut bare = fake_capabilities();
        bare.supports_render_vertex_count_above_triangle = true;
        let frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: bare,
        })
        .unwrap();
        let decoded = match CommandCodec::decode_response(&frame).unwrap() {
            CommandResponse::Capabilities { capabilities, .. } => capabilities,
            other => panic!("the frame decodes to capabilities, got {other:?}"),
        };
        assert_eq!(decoded.max_render_texture_dimension_3d, 0);
    }

    /// A landing-only entry is a pass kind of its own: one tag, then the kept
    /// frame's identity and shape and the window's second declaration, in a
    /// fixed order (`research/docs/23` §115 之后的增量，E-TX14/R4b).
    ///
    /// The reading is that removing exactly that window from the frame gives the
    /// frame without the entry back, byte for byte: the entry adds its own tag
    /// and payload and moves nothing else.
    #[test]
    fn a_kept_frame_landing_is_one_tag_and_a_fixed_payload() {
        let entry = KeptFrameLanding {
            frame: KeptFrame {
                allocation_id: AllocationId::new(70),
                view_id: ViewId::new(71),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
            },
            landing: AttachmentLandingView {
                allocation_id: AllocationId::new(72),
                view_id: ViewId::new(73),
            },
        };
        let frame_for = |entry: Option<KeptFrameLanding>| {
            let mut trace = render_only_trace();
            if let Some(entry) = entry {
                trace.passes.push(TracePass::Landing(entry));
            }
            CommandCodec::encode_request(&CommandRequest::Submit {
                trace,
                resources: resources(),
            })
            .unwrap()
        };
        let without = frame_for(None);
        let with = frame_for(Some(entry));
        // `view_id`, `allocation_id`, `format`, `width`, `height`, the landing
        // view's `view_id` and its `allocation_id`, plus the tag.
        let mut payload = Vec::new();
        for id in [entry.frame.view_id.get(), entry.frame.allocation_id.get()] {
            payload.extend_from_slice(&id.to_be_bytes());
        }
        payload.push(entry.frame.format.code());
        for dimension in [entry.frame.width, entry.frame.height] {
            payload.extend_from_slice(&dimension.to_be_bytes());
        }
        for id in [
            entry.landing.view_id.get(),
            entry.landing.allocation_id.get(),
        ] {
            payload.extend_from_slice(&id.to_be_bytes());
        }
        let mut window = vec![0x19];
        window.extend_from_slice(&payload);
        assert_eq!(window.len(), 50, "the tag and its seven fields");
        assert_eq!(
            with.len(),
            without.len() + window.len(),
            "the entry adds its own bytes and nothing else"
        );
        let at = with
            .windows(window.len())
            .position(|candidate| candidate == window.as_slice())
            .expect("the entry's tag and payload are on the wire");
        // The entry is the last pass, so it is the last bytes before the
        // completion policy: the two frames agree on everything the entry does
        // not touch, and the one it adds is the entry itself.
        let CommandRequest::Submit {
            trace: with_trace, ..
        } = CommandCodec::decode_request(&with).unwrap()
        else {
            panic!("a render submit decodes as a submit");
        };
        let CommandRequest::Submit {
            trace: without_trace,
            ..
        } = CommandCodec::decode_request(&without).unwrap()
        else {
            panic!("a render submit decodes as a submit");
        };
        assert_eq!(with_trace.pipelines, without_trace.pipelines);
        assert_eq!(
            with_trace.encoder_dispatch_type,
            without_trace.encoder_dispatch_type
        );
        assert_eq!(
            with_trace.completion_policy,
            without_trace.completion_policy
        );
        assert_eq!(
            with_trace.passes.len(),
            without_trace.passes.len() + 1,
            "the entry is the one pass the frame adds"
        );
        assert_eq!(
            with_trace.passes[..without_trace.passes.len()],
            without_trace.passes[..],
            "every pass before the entry keeps its own bytes"
        );
        assert_eq!(
            with_trace.passes.last(),
            Some(&TracePass::Landing(entry)),
            "the entry round-trips through the codec"
        );
        // A tag this version does not know is refused by name rather than read
        // as the entry's payload or as the next pass: that is the answer an
        // older decoder gives this frame, and the reason the tag carries the
        // whole entry.
        let mut unknown = with.clone();
        unknown[at] = 0x1a;
        assert!(matches!(
            CommandCodec::decode_request(&unknown).unwrap_err(),
            CodecError::UnknownPassTag(0x1a)
        ));
        // The payload is fixed-length: a frame whose entry is cut short is
        // refused rather than read as a shorter pass, whichever layer names the
        // truncation (the frame's own length check or the walk's).
        let truncated = with[..with.len() - window.len() + 8].to_vec();
        assert!(
            CommandCodec::decode_request(&truncated).is_err(),
            "a truncated landing entry is not a shorter pass"
        );
    }

    /// A frame that ends before the superset block reads the bit as `false`,
    /// and the family's closed set now names three tags
    /// (`research/docs/23` §3.3, E-TX11).
    #[test]
    fn a_frame_without_the_superset_block_reads_the_bit_as_false() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_stage_buffers = true;
        capabilities.max_render_stage_buffers = MAX_RENDER_STAGE_BUFFERS as u32;
        capabilities.supports_render_stage_buffer_namespace_split = true;
        capabilities.supports_render_vertex_interface_superset = true;
        let frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: capabilities.clone(),
        })
        .unwrap();

        // The pre-increment frame is this frame's payload without its three
        // trailing bytes, reframed: the walk finds nothing after the
        // folded-shape block (this snapshot declares no gathered extent) and
        // keeps the bit's default.
        let mut prior_capabilities = capabilities.clone();
        prior_capabilities.supports_render_vertex_interface_superset = false;
        let expected = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: prior_capabilities,
        };
        let prior = CommandCodec::encode_response(&expected).unwrap();
        assert_eq!(
            &prior[FRAME_HEADER..],
            &frame[FRAME_HEADER..frame.len() - 3],
            "the pre-increment payload is this payload without the block"
        );
        assert_eq!(CommandCodec::decode_response(&prior).unwrap(), expected);

        // The family's tags are a closed set and `0x7f` is a tag no version of
        // the walk has assigned: a byte no version of the walk may read as a
        // section is a typed refusal. (`0x04` was this probe's value until
        // E-TX12 assigned it to the gathered extent's no-copy block, `0x05`
        // until E-TX13 assigned it to the attachment landing view, `0x06`
        // until E-TX14 assigned it to the kept-frame landing entry, `0x07` until
        // E-SB2 assigned it to the stage buffer per-stage window, `0x08` until
        // the texel space took it, `0x09` until E-TX15 assigned it to the
        // pass-entry snapshot arm, `0x0a` until the one-dimensional sampled
        // window took it, `0x0b` until the layout-free count above the
        // milestone's three vertices took it, `0x0c` until E-SB3 assigned it to
        // the stage-buffer whole-binding arm, and `0x0d` until the
        // three-dimensional sampled window took it — exactly the drift the
        // closed set exists to make visible. The probe now names a number the
        // family's sequence will not reach, so the next assignment does not
        // have to move it again.)
        let mut unknown_tag = frame.clone();
        let tag_at = unknown_tag.len() - 2;
        unknown_tag[tag_at] = 0x7f;
        assert!(matches!(
            CommandCodec::decode_response(&unknown_tag).unwrap_err(),
            CodecError::UnknownCapabilityTail(0x7f)
        ));
    }

    #[test]
    fn vertex_input_and_present_travel_as_independent_bits() {
        let mut trace = vertex_input_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.present = presenting_trace()
            .passes
            .first()
            .and_then(TracePass::as_render)
            .and_then(|pass| pass.present.clone());
        let request = CommandRequest::Submit {
            trace,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(frame[9], 0x0f);
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        let extended = frame
            .iter()
            .enumerate()
            .skip(10)
            .find(|(_, byte)| **byte == 0x10)
            .map(|(index, _)| index)
            .expect("the vertex-input pass carries the extended tag");
        assert_eq!(
            frame[extended + 1] & 0x03,
            0x03,
            "the pass carries both the vertex-input and present bits"
        );
    }

    #[test]
    fn vertex_input_capability_bits_round_trip_and_keep_the_legacy_frame_shape() {
        let mut capabilities = fake_capabilities();
        capabilities.supports_render_passes = true;
        capabilities.max_color_attachments = 1;
        capabilities.max_attachment_dimension = [2, 2];
        capabilities.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        capabilities.max_vertex_buffers = MAX_VERTEX_BUFFERS as u32;
        capabilities.supported_vertex_formats = VertexFormat::ADMITTED.to_vec();
        capabilities.supported_index_formats = IndexFormat::ADMITTED.to_vec();
        assert!(capabilities.declares_vertex_input_support());
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        // The vertex block is the extended payload's optional tail, so a
        // snapshot that declares none of the three bits keeps the exact bytes
        // its predecessors wrote; the existing legacy-frame pins cover that
        // half.
        let mut legacy = fake_capabilities();
        legacy.supports_render_passes = true;
        legacy.max_color_attachments = 1;
        legacy.max_attachment_dimension = [2, 2];
        legacy.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        legacy.max_vertex_buffers = 0;
        legacy.supported_vertex_formats = Vec::new();
        legacy.supported_index_formats = Vec::new();
        let legacy_frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: legacy,
        })
        .unwrap();
        assert!(legacy_frame.len() < frame.len());
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
    fn resident_load_and_store_take_their_own_tags_and_round_trip() {
        // R7 (`research/docs/23` §76): the provider-resident target is one more
        // tag in the field that already carries the attachment's load and store
        // decisions, so the frame keeps the pre-R7 order and a resident load
        // simply stops paying for the clear payload it does not carry.
        let sentinel = [0xfe_u8; 4];
        let clear_tape = LEGACY_RENDER_SUBMIT_FRAME
            .windows(sentinel.len())
            .position(|window| window == sentinel)
            .expect("the frozen frame carries the fixture's clear payload");
        assert_eq!(
            LEGACY_RENDER_SUBMIT_FRAME[clear_tape - 1],
            0x00,
            "the frozen frame's load tag is the `Clear` tag"
        );

        let mut trace = render_only_trace();
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the fixture ends in a render pass");
        };
        pass.color_attachments[0].load = LoadOp::Resident;
        pass.color_attachments[0].store = StoreOp::Resident;
        let request = CommandRequest::Submit {
            trace,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();

        // The two tags replace the four-byte clear payload, so the frame is
        // exactly four bytes shorter than the frozen pre-R7 shape and the sentinel
        // is gone from it.
        assert_eq!(
            frame.len() + sentinel.len(),
            LEGACY_RENDER_SUBMIT_FRAME.len()
        );
        assert!(
            !frame
                .windows(sentinel.len())
                .any(|window| window == sentinel),
            "a resident load carries no clear payload"
        );
        assert_eq!(
            frame[clear_tape - 1],
            0x03,
            "the resident load takes the tag after `Clear`/`Load`/`DontCare`"
        );

        let decoded = CommandCodec::decode_request(&frame).unwrap();
        assert_eq!(decoded, request);
        let CommandRequest::Submit { trace, .. } = &decoded else {
            panic!("a render submit decodes as a submit");
        };
        let pass = trace.passes[0]
            .as_render()
            .expect("the fixture is a render pass");
        assert_eq!(pass.color_attachments[0].load, LoadOp::Resident);
        assert_eq!(pass.color_attachments[0].store, StoreOp::Resident);

        // An unknown tag is a decoder error rather than a silent `Load` or
        // discard, so an older bridge refuses the frame it does not know.
        let mut unknown = frame.clone();
        unknown[clear_tape - 1] = 0x04;
        let error = CommandCodec::decode_request(&unknown)
            .expect_err("an unknown load tag is refused by name");
        assert!(
            matches!(
                error,
                CodecError::UnknownEnumValue {
                    field: "attachment load op",
                    value: 0x04,
                }
            ),
            "the refusal names the field and the tag: {error:?}"
        );
    }

    #[test]
    fn an_owner_window_store_is_one_tag_and_round_trips() {
        // E-TX8 (`research/docs/23` §114): the borrowed store's whole
        // declaration is the store tag — the window it lands in is the
        // attachment's own view declaration, which the frame already carries.
        // The reading is therefore a *byte* reading: the frame with the new arm
        // differs from the pre-E-TX8 store frame in exactly the one tag byte,
        // and no window, pointer or length is added anywhere.
        let sentinel = [0xfe_u8; 4];
        let clear_tape = LEGACY_RENDER_SUBMIT_FRAME
            .windows(sentinel.len())
            .position(|window| window == sentinel)
            .expect("the frozen frame carries the fixture's clear payload");

        let frame_for = |store: StoreOp| {
            let mut trace = render_only_trace();
            let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
                panic!("the fixture ends in a render pass");
            };
            pass.color_attachments[0].store = store;
            let request = CommandRequest::Submit {
                trace,
                resources: resources(),
            };
            CommandCodec::encode_request(&request).unwrap()
        };
        let stored = frame_for(StoreOp::Store);
        let borrowed = frame_for(StoreOp::Borrowed);
        assert_eq!(
            stored, LEGACY_RENDER_SUBMIT_FRAME,
            "the frozen frame states the pre-E-TX8 store arm"
        );
        assert_eq!(stored.len(), borrowed.len());
        let store_tag = clear_tape + sentinel.len();
        assert_eq!(stored[store_tag], 0x00, "the `Store` tag");
        assert_eq!(borrowed[store_tag], 0x03, "the owner-window store's tag");
        for (index, (left, right)) in stored.iter().zip(borrowed.iter()).enumerate() {
            if index == store_tag {
                continue;
            }
            assert_eq!(
                left, right,
                "the frame differs from the pre-E-TX8 bytes in the store tag alone (byte {index})"
            );
        }

        let decoded = CommandCodec::decode_request(&borrowed).unwrap();
        let CommandRequest::Submit { trace, .. } = &decoded else {
            panic!("a render submit decodes as a submit");
        };
        let pass = trace.passes[0]
            .as_render()
            .expect("the fixture is a render pass");
        assert_eq!(pass.color_attachments[0].store, StoreOp::Borrowed);

        // A tag no arm owns is a decoder error, so an older bridge refuses the
        // frame it does not know instead of reading it as a plain store. Tag
        // `0x04` is the landing-view arm's (`research/docs/23` §115 之后的增量，
        // E-TX13), so the next free tag is `0x05`.
        let mut unknown = borrowed.clone();
        unknown[store_tag] = 0x05;
        let error = CommandCodec::decode_request(&unknown)
            .expect_err("an unknown store tag is refused by name");
        assert!(
            matches!(
                error,
                CodecError::UnknownEnumValue {
                    field: "attachment store op",
                    value: 0x05,
                }
            ),
            "the refusal names the field and the tag: {error:?}"
        );
    }

    /// The landing-view store arm is one tag plus the second view's identity,
    /// and it is the only thing a pre-E-TX13 frame does not have
    /// (`research/docs/23` §115 之后的增量，E-TX13).
    #[test]
    fn a_landing_view_store_is_a_tag_plus_the_views_identity() {
        let sentinel = [0xfe_u8; 4];
        let clear_tape = LEGACY_RENDER_SUBMIT_FRAME
            .windows(sentinel.len())
            .position(|window| window == sentinel)
            .expect("the frozen frame carries the fixture's clear payload");
        let landing = AttachmentLandingView {
            allocation_id: AllocationId::new(77),
            view_id: ViewId::new(78),
        };
        let frame_for = |store: StoreOp| {
            let mut trace = render_only_trace();
            let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
                panic!("the fixture ends in a render pass");
            };
            pass.color_attachments[0].store = store;
            let request = CommandRequest::Submit {
                trace,
                resources: resources(),
            };
            CommandCodec::encode_request(&request).unwrap()
        };
        let stored = frame_for(StoreOp::Store);
        let landing_frame = frame_for(StoreOp::BorrowedLanding(landing));
        let store_tag = clear_tape + sentinel.len();
        assert_eq!(stored[store_tag], 0x00, "the `Store` tag");
        assert_eq!(
            landing_frame[store_tag], 0x04,
            "the landing-view store's tag"
        );
        assert_eq!(
            landing_frame.len(),
            stored.len() + 16,
            "the arm adds the second view's two ids and nothing else"
        );
        // The payload is the landing view's identity, in the same order the
        // attachment's own fields are written: view id first, then allocation.
        let payload = &landing_frame[store_tag + 1..store_tag + 17];
        assert_eq!(&payload[..8], &landing.view_id.get().to_be_bytes());
        assert_eq!(&payload[8..], &landing.allocation_id.get().to_be_bytes());
        // Everything before the arm and everything after it keeps its bytes:
        // the frame grows by the identity the arm carries and by nothing else.
        assert_eq!(
            &landing_frame[FRAME_HEADER..store_tag],
            &stored[FRAME_HEADER..store_tag],
            "the bytes in front of the store arm are the pre-E-TX13 ones"
        );
        assert_eq!(
            &landing_frame[store_tag + 17..],
            &stored[store_tag + 1..],
            "the bytes behind the store arm are the pre-E-TX13 ones"
        );
        let decoded = CommandCodec::decode_request(&landing_frame).unwrap();
        let CommandRequest::Submit { trace, .. } = &decoded else {
            panic!("a render submit decodes as a submit");
        };
        let pass = trace.passes[0]
            .as_render()
            .expect("the fixture is a render pass");
        assert_eq!(
            pass.color_attachments[0].store,
            StoreOp::BorrowedLanding(landing)
        );
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
    fn an_aliasing_heap_descriptor_round_trips_without_new_wire_bytes() {
        let mut trace = heap_trace();
        trace
            .heap
            .as_mut()
            .expect("the fixture carries a heap")
            .descriptor
            .allows_aliasing = true;
        let request = CommandRequest::Submit {
            trace,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        let decoded = CommandCodec::decode_request(&frame).unwrap();
        let CommandRequest::Submit { trace, .. } = &decoded else {
            panic!("an aliasing heap submit decodes as a submit");
        };
        let heap = trace.heap.as_ref().expect("the fixture carries a heap");
        assert!(heap.descriptor.allows_aliasing);
        // The aliasing flag is the existing descriptor bool: it rides the
        // same length-prefixed heap tail and changes no frame layout.
        assert_eq!(frame.len(), HEAP_SUBMIT_FRAME.len());
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
    fn capability_frames_without_heap_bits_keep_the_pre_heap_bytes() {
        // A provider that declares render and present bits but neither heap nor
        // ICB bits must keep the frame the pre-heap codec produced: the heap/ICB
        // section is an optional tail (`research/docs/25` §4.5), so this pin is
        // what stops it from being appended unconditionally.
        let mut both = fake_capabilities();
        both.supports_render_passes = true;
        both.max_color_attachments = 1;
        both.max_attachment_dimension = [2, 2];
        both.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        both.supports_presentation = true;
        both.max_present_targets = 1;
        both.supported_present_modes = vec![PresentMode::Fifo];
        both.max_present_image_count = MAX_PRESENT_IMAGE_COUNT;
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(7),
            capabilities: both,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(frame[9], 0x0a);
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        let hex = frame
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        // Captured from the pre-heap codec at `2ad57d1` and re-checked against
        // this encoder: the frames are byte-identical.
        assert_eq!(
            hex,
            "4d43433102000000950a00000000000000070000000101000100000000000000000100000000000000010000000000000001000000000000000100000000000000010000000000000001000000000000000100000001000000000000040000000000000000000000000001000100010000000100000000000000020000000000000002000000000000000102010000000100000000000000010200000001"
        );
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
                    supports_render_kept_frame_landing: false,
                    supports_render_pass_entry_snapshot: false,
                    max_render_texture_dimension_1d: 0,
                    max_render_texture_dimension_3d: 0,
                    supports_render_stage_buffers: false,
                    max_render_stage_buffers: 0,
                    max_render_stage_buffers_per_stage: 0,
                    supports_render_stage_buffer_namespace_split: false,
                    supports_render_stage_buffer_binding_range: false,
                    supports_render_fragment_output_superset: false,
                    supports_render_pixel_coordinate_sampler: false,
                    max_passes: 2,
                    supports_threads_exact: true,
                    supports_threadgroups: false,
                    supports_serial: true,
                    supports_concurrent: false,
                    max_local_size: [1, 1, 1],
                    max_invocations: 4,
                    max_group_count: [1, 1, 1],
                    max_storage_buffer_descriptors: 2,
                    supports_compute_texture_sampling: false,
                    max_compute_textures: 0,
                    supported_compute_texture_formats: Vec::new(),
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
                    max_vertex_buffers: 0,
                    supported_vertex_formats: Vec::new(),
                    supported_index_formats: Vec::new(),
                    supports_render_vertex_interface_superset: false,
                    supports_render_vertex_count_above_triangle: false,
                    supports_render_instancing: false,
                    max_render_instances: 0,
                    supports_render_multisample: false,
                    max_render_sample_count: 0,
                    supports_render_depth_resolve: false,
                    depth_resolve_modes: 0,
                    supports_render_stencil_resolve: false,
                    stencil_resolve_modes: 0,
                    supports_render_texture_sampling: false,
                    max_render_textures: 0,
                    supported_render_texture_formats: Vec::new(),
                    supports_render_texture_gathered_extent: false,
                    supports_render_texture_gathered_extent_no_copy: false,
                    supports_render_attachment_landing_view: false,
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
            supports_render_kept_frame_landing: false,
            supports_render_pass_entry_snapshot: false,
            max_render_texture_dimension_1d: 0,
            max_render_texture_dimension_3d: 0,
            supports_render_stage_buffers: false,
            max_render_stage_buffers: 0,
            max_render_stage_buffers_per_stage: 0,
            supports_render_stage_buffer_namespace_split: false,
            supports_render_stage_buffer_binding_range: false,
            supports_render_fragment_output_superset: false,
            supports_render_pixel_coordinate_sampler: false,
            max_passes: 1,
            supports_threads_exact: true,
            supports_threadgroups: false,
            supports_serial: true,
            supports_concurrent: false,
            max_local_size: [1, 1, 1],
            max_invocations: 1,
            max_group_count: [1, 1, 1],
            max_storage_buffer_descriptors: 1,
            supports_compute_texture_sampling: false,
            max_compute_textures: 0,
            supported_compute_texture_formats: Vec::new(),
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
            max_vertex_buffers: 0,
            supported_vertex_formats: Vec::new(),
            supported_index_formats: Vec::new(),
            supports_render_vertex_interface_superset: false,
            supports_render_vertex_count_above_triangle: false,
            supports_render_instancing: false,
            max_render_instances: 0,
            supports_render_multisample: false,
            max_render_sample_count: 0,
            supports_render_depth_resolve: false,
            depth_resolve_modes: 0,
            supports_render_stencil_resolve: false,
            stencil_resolve_modes: 0,
            supports_render_texture_sampling: false,
            max_render_textures: 0,
            supported_render_texture_formats: Vec::new(),
            supports_render_texture_gathered_extent: false,
            supports_render_texture_gathered_extent_no_copy: false,
            supports_render_attachment_landing_view: false,
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

    // -----------------------------------------------------------------
    // C1c: the compute texture declaration face on the wire
    // (`research/docs/23` §91).
    // -----------------------------------------------------------------

    /// The declaration block a contract with one [`TextureBindingContract`]
    /// writes: the count, then binding `0`, access sampled, type D2, format
    /// `R32Uint`, nearest filtering, clamp-to-edge addressing and a whole-view
    /// footprint (`research/docs/26` §21.3, C1).
    const R32UINT_DECLARATION_BLOCK: &[u8] = &[
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00,
    ];

    /// A compute registration whose contract declares the one sampled texture
    /// its kernel reads — the face C1 added to `PipelineContract` and C1c
    /// carries (`research/docs/26` §21.3).
    fn textured_pipeline() -> CompiledComputePipeline {
        let mut compiled = pipeline(&compile_request());
        compiled.contract.texture_bindings = vec![TextureBindingContract::sampled_r32uint(0)];
        compiled
    }

    /// The 4x4 `R32Uint` view [`textured_pipeline`] declares: the shape both
    /// rails that execute compute sampling read
    /// ([`TextureBindingContract::sampled_r32uint`]).
    fn sampled_r32uint_view() -> TextureView {
        TextureView {
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
        }
    }

    /// [`trace`] with the declared view bound at its own index, i.e. the trace
    /// `ComputePass::validate` admits against [`textured_pipeline`].
    fn textured_trace(pipeline: &CompiledComputePipeline) -> ComputeTrace {
        let mut value = trace(pipeline);
        let Some(TracePass::Compute(pass)) = value.passes.first_mut() else {
            panic!("the fixture is a compute pass");
        };
        pass.textures = vec![sampled_r32uint_view()];
        value
    }

    /// The compute texture bits the two rails publish, on top of the fixture
    /// snapshot (`research/docs/26` §21.3).
    fn compute_texture_capabilities() -> ProviderCapabilities {
        let mut capabilities = fake_capabilities();
        capabilities.supports_compute_texture_sampling = true;
        capabilities.max_compute_textures = 1;
        capabilities.supported_compute_texture_formats =
            vec![TextureFormat::R32Uint, TextureFormat::R32Float];
        capabilities
    }

    #[test]
    fn compute_texture_declarations_take_the_tagged_layout_and_round_trip() {
        let plain = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: trace(&pipeline(&compile_request())),
            resources: resources(),
        })
        .unwrap();
        assert_eq!(
            plain, LEGACY_SUBMIT_FRAME,
            "a compute-only trace that declares no texture keeps the pre-render bytes"
        );

        let request = CommandRequest::Submit {
            trace: textured_trace(&textured_pipeline()),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        // The pre-render layout has no section for the block, so the frame
        // takes the compute-texture submit tag and the tagged pipeline kind.
        assert_eq!(frame[9], 0x11);
        assert_eq!(frame[pipeline_entry_kind_offset()], 0x04);
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame,
            "the textured frame re-encodes byte for byte"
        );
        // The delta against the legacy shape is exactly the two kind bytes the
        // tagged layout adds and the declaration block: no other byte moves.
        assert!(
            frame
                .windows(R32UINT_DECLARATION_BLOCK.len())
                .any(|window| window == R32UINT_DECLARATION_BLOCK),
            "the declaration block travels in the frame"
        );
        // The same trace with the declaration list removed is the shape the
        // pre-C1c layout writes: the tag choice is the declarations' own
        // consequence, not the bound views'.
        let mut undeclared = textured_trace(&textured_pipeline());
        undeclared.pipelines[0].contract.texture_bindings.clear();
        let undeclared_frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: undeclared,
            resources: resources(),
        })
        .unwrap();
        assert_eq!(undeclared_frame[9], 0x03);
        assert_eq!(
            frame.len(),
            undeclared_frame.len() + 2 + R32UINT_DECLARATION_BLOCK.len()
        );

        // The decoded contract is the value admission pairs the pass against:
        // the declaration list survives the frame field for field.
        let CommandRequest::Submit { trace, .. } = CommandCodec::decode_request(&frame).unwrap()
        else {
            panic!("a textured submit decodes as a submit");
        };
        assert_eq!(
            trace.pipelines[0].contract.texture_bindings,
            vec![TextureBindingContract::sampled_r32uint(0)]
        );
        eprintln!(
            "compute texture submit: len={} undeclared={} plain={} tag={:#04x} kind={:#04x} \
             block={:02x?}",
            frame.len(),
            undeclared_frame.len(),
            plain.len(),
            frame[9],
            frame[pipeline_entry_kind_offset()],
            R32UINT_DECLARATION_BLOCK
        );
    }

    /// The sampler codes §109 appended (`research/docs/23` §109) travel the same
    /// block byte for byte: a declaration naming one of the widened filters and
    /// address modes re-encodes to the very frame it came from, and the two
    /// codes the first increment published keep their meanings.
    #[test]
    fn the_widened_sampler_names_travel_the_published_block() {
        let mut widened = textured_pipeline();
        widened.contract.texture_bindings[0].sampler =
            Some(metal_api_core::provider::SamplerPolicy {
                filter: metal_api_core::provider::SamplerFilter::LinearMipLinear,
                address: metal_api_core::provider::SamplerAddressMode::ClampToZero,
            });
        let request = CommandRequest::Submit {
            trace: textured_trace(&widened),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        // The published block with the two appended codes in the filter and
        // address bytes: finding it is what says the block kept its layout, and
        // its own bytes are the two codes §109 appended.
        let mut widened_block = R32UINT_DECLARATION_BLOCK.to_vec();
        widened_block[8] = 0x05;
        widened_block[9] = 0x04;
        let block = frame
            .windows(widened_block.len())
            .position(|window| window == widened_block)
            .expect("the declaration block is on the wire");
        eprintln!(
            "widened sampler block: filter={:#04x} address={:#04x}",
            frame[block + 8],
            frame[block + 9]
        );
        assert_eq!(frame[block + 8], 0x05, "linear min/mag + linear mip");
        assert_eq!(frame[block + 9], 0x04, "clamp-to-zero");
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame,
            "the widened declaration re-encodes byte for byte"
        );
    }

    #[test]
    fn the_compiled_response_carries_declarations_only_when_the_contract_has_them() {
        let plain = CommandResponse::Compiled {
            pipeline: pipeline(&compile_request()),
        };
        let plain_frame = CommandCodec::encode_response(&plain).unwrap();
        assert_eq!(plain_frame[9], 0x02);
        assert_eq!(CommandCodec::decode_response(&plain_frame).unwrap(), plain);

        let textured = CommandResponse::Compiled {
            pipeline: textured_pipeline(),
        };
        let frame = CommandCodec::encode_response(&textured).unwrap();
        assert_eq!(frame[9], 0x0b);
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), textured);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the compiled textured response re-encodes byte for byte"
        );
        assert_eq!(
            frame.len(),
            plain_frame.len() + R32UINT_DECLARATION_BLOCK.len()
        );
        // The owner learns the declaration list here, which is what lets it
        // build the trace above with the contract the provider reflected.
        let CommandResponse::Compiled { pipeline } = CommandCodec::decode_response(&frame).unwrap()
        else {
            panic!("a compiled response decodes as a compiled response");
        };
        assert_eq!(
            pipeline.contract.texture_bindings,
            vec![TextureBindingContract::sampled_r32uint(0)]
        );
        assert!(pipeline.render.is_none());
        eprintln!(
            "compiled textured pipeline: len={} plain={} tag={:#04x}",
            frame.len(),
            plain_frame.len(),
            frame[9]
        );
    }

    #[test]
    fn release_pipeline_carries_the_declarations_for_a_textured_registration() {
        let plain = CommandRequest::ReleasePipeline {
            pipeline: pipeline(&compile_request()),
        };
        let plain_frame = CommandCodec::encode_request(&plain).unwrap();
        assert_eq!(plain_frame[9], 0x07);
        assert_eq!(CommandCodec::decode_request(&plain_frame).unwrap(), plain);

        let textured = CommandRequest::ReleasePipeline {
            pipeline: textured_pipeline(),
        };
        let frame = CommandCodec::encode_request(&textured).unwrap();
        assert_eq!(frame[9], 0x12);
        // The native rail compares the released value against the registration
        // it holds, so the whole contract has to survive the frame.
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), textured);
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame,
            "the textured release re-encodes byte for byte"
        );
        assert_eq!(
            frame.len(),
            plain_frame.len() + R32UINT_DECLARATION_BLOCK.len()
        );
        eprintln!(
            "release textured pipeline: len={} plain={} tag={:#04x}",
            frame.len(),
            plain_frame.len(),
            frame[9]
        );
    }

    #[test]
    fn compute_texture_capability_bits_extend_the_stage_buffer_frame() {
        // The legacy frame every provider without a render or compute texture
        // bit still sends.
        let defaults = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: fake_capabilities(),
        };
        let default_frame = CommandCodec::encode_response(&defaults).unwrap();
        assert_eq!(default_frame[9], 0x01);
        assert_eq!(
            CommandCodec::decode_response(&default_frame).unwrap(),
            defaults
        );

        // The frame the two rails already sent: render bits, render sampling
        // and stage buffers.
        let mut prior = fake_capabilities();
        prior.supports_render_passes = true;
        prior.max_color_attachments = 1;
        prior.max_attachment_dimension = [2, 2];
        prior.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        prior.supports_render_texture_sampling = true;
        prior.max_render_textures = 1;
        prior.supported_render_texture_formats = vec![TextureFormat::Rgba8Unorm];
        prior.supports_render_stage_buffers = true;
        prior.max_render_stage_buffers = MAX_RENDER_STAGE_BUFFERS as u32;
        let prior_frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: prior.clone(),
        })
        .unwrap();

        let mut extended = prior.clone();
        extended.supports_compute_texture_sampling = true;
        extended.max_compute_textures = 1;
        extended.supported_compute_texture_formats =
            vec![TextureFormat::R32Uint, TextureFormat::R32Float];
        let response = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: extended,
        };
        let frame = CommandCodec::encode_response(&response).unwrap();
        assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        assert_eq!(
            CommandCodec::encode_response(&CommandCodec::decode_response(&frame).unwrap()).unwrap(),
            frame,
            "the capability frame re-encodes byte for byte"
        );
        // The block is the tail's newest section: one presence tag, one bool,
        // one `u32` binding cap and the admitted texture formats
        // (`research/docs/23` §91). Every integer is big-endian, the width
        // the frame header established.
        let block = [
            0x80, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
            0x00, 0x01,
        ];
        assert!(
            frame.windows(block.len()).any(|window| window == block),
            "the compute-texture tail carries its tag, its bool, its cap and its formats"
        );
        assert_eq!(frame.len(), prior_frame.len() + block.len());
        // The length header is the one byte run that has to move; every byte
        // the pre-C1c frame wrote after it is still where it was, with the
        // new block appended behind the stage-buffer half.
        assert_eq!(
            &frame[9..prior_frame.len()],
            &prior_frame[9..],
            "every byte the pre-C1c frame wrote stays where it was"
        );

        // A snapshot that declares only the three compute texture bits still
        // writes the heap/ICB half the decoder reads by position before the
        // tag, so the declaration cannot be dropped on the wire.
        let only_compute = CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: compute_texture_capabilities(),
        };
        let only_frame = CommandCodec::encode_response(&only_compute).unwrap();
        assert_eq!(only_frame[9], 0x0a);
        assert_eq!(
            CommandCodec::decode_response(&only_frame).unwrap(),
            only_compute
        );
        eprintln!(
            "compute texture capabilities: legacy={} render+stage={} extended={} only={} block={:02x?}",
            default_frame.len(),
            prior_frame.len(),
            frame.len(),
            only_frame.len(),
            block
        );
    }

    #[test]
    fn a_heap_bearing_trace_carries_its_pipeline_declarations_under_the_heap_tag() {
        // The tagged layout is chosen by the heap payload here, not by the
        // declarations, so this is the path where the two reasons have to
        // agree: the frame keeps the heap tag and its tail while the pipeline
        // entry takes the compute-texture kind (`research/docs/23` §91).
        let mut trace = heap_trace();
        trace.pipelines[0] = textured_pipeline();
        let Some(TracePass::Compute(pass)) = trace.passes.first_mut() else {
            panic!("the fixture's pass is a compute pass");
        };
        pass.textures = vec![sampled_r32uint_view()];
        let request = CommandRequest::Submit {
            trace,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(frame[9], 0x10);
        assert_eq!(frame[pipeline_entry_kind_offset()], 0x04);
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame,
            "the heap-bearing textured frame re-encodes byte for byte"
        );
        let CommandRequest::Submit { trace, .. } = CommandCodec::decode_request(&frame).unwrap()
        else {
            panic!("a heap-bearing submit decodes as a submit");
        };
        assert!(trace.heap.is_some(), "the heap tail still travels");
        assert_eq!(
            trace.pipelines[0].contract.texture_bindings,
            vec![TextureBindingContract::sampled_r32uint(0)]
        );
        eprintln!(
            "heap-bearing textured submit: len={} tag={:#04x} kind={:#04x}",
            frame.len(),
            frame[9],
            frame[pipeline_entry_kind_offset()]
        );
    }

    #[test]
    fn compute_texture_declarations_refuse_counts_and_unknown_codes_by_name() {
        // The encode side refuses a list above the shape's own cap before a
        // single tuple is written.
        let mut over = textured_pipeline();
        over.contract
            .texture_bindings
            .push(TextureBindingContract::sampled_r32uint(1));
        let refused = CommandCodec::encode_response(&CommandResponse::Compiled { pipeline: over })
            .unwrap_err();
        assert!(matches!(
            refused,
            CodecError::ComputeTextureCount {
                count: 2,
                maximum: 1
            }
        ));
        eprintln!("refused: {refused}");

        // An entry that carries a render half *and* a declaration list cannot
        // be framed at all: the render kinds' bytes are fixed, so the encoder
        // refuses the combination by name instead of writing a frame the
        // receiver would have to desync on.
        let mut combined = pipeline(&compile_request());
        combined.render = Some(render_contract());
        combined.contract.texture_bindings = vec![TextureBindingContract::sampled_r32uint(0)];
        let mut combined_trace = mixed_trace();
        combined_trace.pipelines[0] = combined;
        let refused = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: combined_trace,
            resources: resources(),
        })
        .unwrap_err();
        assert!(matches!(
            refused,
            CodecError::ComputeTextureDeclarationsWithRenderHalf { declarations: 1 }
        ));
        eprintln!("refused: {refused}");

        let frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: textured_trace(&textured_pipeline()),
            resources: resources(),
        })
        .unwrap();
        let block = frame
            .windows(R32UINT_DECLARATION_BLOCK.len())
            .position(|window| window == R32UINT_DECLARATION_BLOCK)
            .expect("the declaration block is on the wire");

        // The decode side reads the count as one byte and refuses it above the
        // same cap before sizing a `Vec`.
        let mut patched = frame.clone();
        patched[block] = 0x02;
        let refused = CommandCodec::decode_request(&patched).unwrap_err();
        assert!(matches!(
            refused,
            CodecError::ComputeTextureCount {
                count: 2,
                maximum: 1
            }
        ));
        eprintln!("refused: {refused}");

        // Every enum in a tuple keeps its named refusal: a guess here would
        // change which texels a remote read returns. The sampler codes §109
        // appended are *not* refused — they are the widened family's own
        // names, read back below — so the probes are the first code past the
        // appended range (`research/docs/23` §109).
        for (offset, field, code) in [
            (6, "texture type", 0x07),
            (8, "sampler filter", 0x06),
            (9, "sampler address", 0x05),
            (10, "texture footprint", 0x02),
        ] {
            let mut patched = frame.clone();
            patched[block + offset] = code;
            let refused = CommandCodec::decode_request(&patched).unwrap_err();
            assert!(
                matches!(
                    refused,
                    CodecError::UnknownEnumValue {
                        field: name,
                        value,
                    } if name == field && value == code
                ),
                "{field} code {code:#04x} is refused by name"
            );
            eprintln!("refused: {refused}");
        }

        // The capability snapshot's format list is bounded the same way.
        let mut wide = compute_texture_capabilities();
        wide.supported_compute_texture_formats = vec![TextureFormat::R32Uint; 9];
        let refused = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: wide,
        })
        .unwrap_err();
        assert!(matches!(
            refused,
            CodecError::ComputeTextureFormatCount {
                count: 9,
                maximum: 8
            }
        ));
        eprintln!("refused: {refused}");

        let capability_frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
            epoch: DeviceEpoch::new(1),
            capabilities: compute_texture_capabilities(),
        })
        .unwrap();
        let capability_block = [
            0x80, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
            0x00, 0x01,
        ];
        let capability_block = capability_frame
            .windows(capability_block.len())
            .position(|window| window == capability_block)
            .expect("the compute-texture block is on the wire");
        let mut patched = capability_frame.clone();
        // The `u64` format count follows the tag, the bool and the `u32` cap,
        // and the frame writes its integers big-endian.
        patched[capability_block + 6..capability_block + 14].copy_from_slice(&9u64.to_be_bytes());
        let refused = CommandCodec::decode_response(&patched).unwrap_err();
        assert!(matches!(
            refused,
            CodecError::ComputeTextureFormatCount {
                count: 9,
                maximum: 8
            }
        ));
        eprintln!("refused: {refused}");
    }

    #[test]
    fn a_capability_without_the_compute_texture_bits_refuses_the_decoded_trace_by_name() {
        let frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: textured_trace(&textured_pipeline()),
            resources: resources(),
        })
        .unwrap();
        let CommandRequest::Submit { trace, resources } =
            CommandCodec::decode_request(&frame).unwrap()
        else {
            panic!("a textured submit decodes as a submit");
        };
        // The decode side holds the provider's declaration list, which is what
        // admission pairs the pass against; a snapshot that never declared the
        // capability has to refuse the shape by name rather than execute it.
        let refusal = fake_capabilities().admit(&trace, &resources).unwrap_err();
        assert_eq!(refusal.slug, "compute_texture_input_unsupported");
        assert_eq!(refusal.fields.get("pass"), Some(&FieldValue::Unsigned(0)));
        assert_eq!(
            refusal.fields.get("textures"),
            Some(&FieldValue::Unsigned(1))
        );
        eprintln!(
            "refused: slug={} class={:?} fields={:?}",
            refusal.slug, refusal.class, refusal.fields
        );

        // A snapshot that declares the bit but no binding slot refuses the
        // count with its own name, one gate later.
        let mut narrow = fake_capabilities();
        narrow.supports_compute_texture_sampling = true;
        let refusal = narrow.admit(&trace, &resources).unwrap_err();
        assert_eq!(refusal.slug, "compute_texture_limit");
        eprintln!("refused: slug={} fields={:?}", refusal.slug, refusal.fields);
    }

    /// One runtime sampler state from the widened family (`research/docs/23`
    /// §109, v109): the two newest names — `LinearMipLinear` and
    /// `ClampToZero` — both travel as their own codes, so a frame built from
    /// this state is the one that has to survive the wire.
    fn widened_runtime_sampler(binding: u32) -> RenderSamplerBinding {
        RenderSamplerBinding {
            metal_binding: binding,
            policy: SamplerPolicy {
                filter: SamplerFilter::LinearMipLinear,
                address: SamplerAddressMode::ClampToZero,
            },
            coordinates: metal_api_core::provider::SamplerCoordinates::Normalized,
        }
    }

    /// A render trace whose pass states one runtime `[[sampler(n)]]` state and
    /// binds nothing else the newer tags carry: the shape the runtime-sampler
    /// kind `0x15` exists for (`research/docs/23` §3.3, v102).
    fn runtime_sampler_trace() -> ComputeTrace {
        let mut trace = render_only_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.samplers = vec![widened_runtime_sampler(0)];
        trace
    }

    /// The sampled fixture plus one runtime sampler state, so the v70 texture
    /// block and the runtime sampler block travel in one frame
    /// (`research/docs/23` §3.3, v70/v102). The contract keeps the plain
    /// shape, so the frame differs from [`sampled_multisample_trace`] in the
    /// pass alone.
    fn sampled_runtime_sampler_trace() -> ComputeTrace {
        let mut trace = sampled_multisample_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.samplers = vec![widened_runtime_sampler(0)];
        trace
    }

    /// The stage buffer fixture plus one runtime sampler state
    /// (`research/docs/23` §3.3, v83/v102): the shape the census's
    /// `texture_sampler_wire` lines are made of, stated on the wire.
    fn stage_buffer_runtime_sampler_trace() -> ComputeTrace {
        let mut trace = stage_buffer_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.samplers = vec![widened_runtime_sampler(0)];
        trace
    }

    /// The sampled *and* stage-buffer fixture plus one runtime sampler state,
    /// so all three blocks the newer tags carry travel in one frame
    /// (`research/docs/23` §3.3, v70/v83/v102).
    fn full_runtime_sampler_trace() -> ComputeTrace {
        let mut trace = sampled_stage_buffer_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.samplers = vec![widened_runtime_sampler(0)];
        trace
    }

    /// The four bytes that spell a runtime-sampler pass's fixed head: the tag,
    /// the wide feature word — all zero, because that fixture's pass states no
    /// optional section — and the block's count (`research/docs/23` §3.3,
    /// v102).
    const RUNTIME_SAMPLER_HEAD: [u8; 4] = [0x15, 0x00, 0x00, 0x01];

    /// Where [`RUNTIME_SAMPLER_HEAD`] sits in a frame, by position.
    fn runtime_sampler_head_at(frame: &[u8]) -> usize {
        frame
            .windows(RUNTIME_SAMPLER_HEAD.len())
            .position(|window| window == RUNTIME_SAMPLER_HEAD)
            .expect("the frame carries the runtime-sampler head")
    }

    /// The seven bytes one runtime sampler entry adds to a frame that already
    /// carries the tag: the block's count and the entry's index, filter and
    /// address (`research/docs/23` §3.3, v102).
    const RUNTIME_SAMPLER_BLOCK: [u8; 7] = [0x01, 0x00, 0x00, 0x00, 0x00, 0x05, 0x04];

    #[test]
    fn a_runtime_sampler_pass_takes_its_own_tag_and_round_trips() {
        let request = CommandRequest::Submit {
            trace: runtime_sampler_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame,
            "the runtime sampler frame re-encodes byte for byte"
        );
        // The tag, the wide word the pass's other sections did not touch, the
        // block's count and the entry's own bytes: the Metal index (0), the
        // filter code `0x05` (`LinearMipLinear`) and the address code `0x04`
        // (`ClampToZero`) — the two names §109 appended, so the widened family
        // is exactly what a decoded frame states back (`research/docs/23`
        // §3.3, v102/v109).
        let head = runtime_sampler_head_at(&frame);
        assert_eq!(
            &frame[head..head + 10],
            &[0x15, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x05, 0x04],
            "the sampler pass carries its own tag, the wide word and the entry"
        );
        eprintln!(
            "runtime sampler frame: len={} tag={:#04x} entry={:02x?}",
            frame.len(),
            frame[head],
            &frame[head + 4..head + 10]
        );
    }

    #[test]
    fn a_sampled_runtime_sampler_pass_keeps_both_halves_in_one_frame() {
        let request = CommandRequest::Submit {
            trace: sampled_runtime_sampler_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame,
            "the combined frame re-encodes byte for byte"
        );
        // The combined tag is the sampled tag's payload with the sampler block
        // appended (`research/docs/23` §3.3, v102): the tag changes, the wide
        // word and the texture block keep their bytes, the sampler block
        // follows the texels, and every section after it is the sampled
        // frame's.
        let sampled = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: sampled_multisample_trace(),
            resources: resources(),
        })
        .unwrap();
        let sampled_head = sampled
            .windows(3)
            .position(|window| window == [0x12, 0x20, 0x01])
            .expect("the sampled fixture takes the sampled tag");
        let texels: Vec<u8> = (0..4u8)
            .flat_map(|y| (0..4u8).flat_map(move |x| [x, y, x.wrapping_add(y), 0xff]))
            .collect();
        let texels_end = frame
            .windows(texels.len())
            .position(|window| window == texels)
            .expect("the sampled texture's own bytes travel in the frame")
            + texels.len();
        let block_at = frame
            .windows(RUNTIME_SAMPLER_BLOCK.len())
            .position(|window| window == RUNTIME_SAMPLER_BLOCK)
            .expect("the sampler block is on the wire");
        // The block's bytes are the count and the entry, and the encoder
        // appends them directly after the texture block's texels: the tag and
        // the wide word sit three bytes in front of them.
        assert_eq!(
            block_at, texels_end,
            "the sampler block follows the texels with nothing between"
        );
        // The tag sits where the sampled fixture's tag does: the two frames
        // are the same bytes up to it, because the pass, the pipeline table
        // and every section before them are unchanged.
        let head = sampled_head;
        assert_eq!(sampled[head], 0x12, "the sampled fixture's own tag");
        assert_eq!(frame[head], 0x16, "the combined tag");
        assert_eq!(
            &frame[head + 1..head + 3],
            &sampled[sampled_head + 1..sampled_head + 3],
            "the combined tag carries the same wide word"
        );
        assert_eq!(
            &frame[head + 3..texels_end],
            &sampled[sampled_head + 3..texels_end],
            "the texture block keeps its bytes"
        );
        assert_eq!(
            &frame[texels_end..texels_end + RUNTIME_SAMPLER_BLOCK.len()],
            &RUNTIME_SAMPLER_BLOCK,
            "the sampler block follows the texels"
        );
        assert_eq!(
            &frame[texels_end + RUNTIME_SAMPLER_BLOCK.len()..],
            &sampled[texels_end..],
            "every section after the sampler block is byte identical"
        );
        eprintln!(
            "sampled + runtime sampler: len={} sampled={} head={head}",
            frame.len(),
            sampled.len()
        );
    }

    #[test]
    fn a_stage_buffer_runtime_sampler_pass_keeps_both_blocks_in_one_frame() {
        let request = CommandRequest::Submit {
            trace: stage_buffer_runtime_sampler_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame,
            "the stage-buffer + sampler frame re-encodes byte for byte"
        );
        // The combined tag is the stage-buffer tag's payload with the sampler
        // block appended, so the block is the whole difference plus the tag
        // (`research/docs/23` §3.3, v83/v102).
        let stage_only = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: stage_buffer_trace(),
            resources: resources(),
        })
        .unwrap();
        // The tag, the empty wide word and the stage buffer block's count and
        // stage code are the combined frame's fixed head (`[0x13, …]` in the
        // stage-buffer-only frame).
        let head = frame
            .windows(5)
            .position(|window| window == [0x17, 0x00, 0x00, 0x01, 0x01])
            .expect("the combined frame carries the stage-buffer block behind its tag");
        assert_eq!(frame[head], 0x17, "the combined tag");
        assert_eq!(
            frame.len(),
            stage_only.len() + RUNTIME_SAMPLER_BLOCK.len(),
            "the sampler block is the whole addition"
        );
        assert!(
            frame
                .windows(RUNTIME_SAMPLER_BLOCK.len())
                .any(|window| window == RUNTIME_SAMPLER_BLOCK),
            "the sampler block travels in the same frame"
        );
        eprintln!(
            "stage buffer + runtime sampler: len={} stage_only={} head={head}",
            frame.len(),
            stage_only.len()
        );
    }

    #[test]
    fn a_full_render_pass_carries_all_three_blocks_in_one_frame() {
        let request = CommandRequest::Submit {
            trace: full_runtime_sampler_trace(),
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame,
            "the three-block frame re-encodes byte for byte"
        );
        // The combined tag's payload is the sampled + stage-buffer tag's with
        // the sampler block appended, so the block is the whole difference
        // (`research/docs/23` §3.3, v70/v83/v102).
        let sampled_stage_only = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: sampled_stage_buffer_trace(),
            resources: resources(),
        })
        .unwrap();
        assert_eq!(
            frame.len(),
            sampled_stage_only.len() + RUNTIME_SAMPLER_BLOCK.len(),
            "the sampler block is the whole addition"
        );
        // The three blocks are written in the one order the decoder reads:
        // textures, stage buffers, then the sampler states. Locating each by
        // its own bytes pins that order on the wire instead of on the encoder.
        let texels: Vec<u8> = (0..4u8)
            .flat_map(|y| (0..4u8).flat_map(move |x| [x, y, x.wrapping_add(y), 0xff]))
            .collect();
        let texels_at = frame
            .windows(texels.len())
            .position(|window| window == texels)
            .expect("the sampled texture's own bytes travel in the frame");
        let stage_payload = stage_buffer_payload();
        let stage_payload_at = frame
            .windows(stage_payload.len())
            .position(|window| window == stage_payload)
            .expect("the stage buffer's own bytes travel in the frame");
        let stage_payload_end = stage_payload_at + stage_payload.len();
        let sampler_at = frame
            .windows(RUNTIME_SAMPLER_BLOCK.len())
            .position(|window| window == RUNTIME_SAMPLER_BLOCK)
            .expect("the sampler block travels in the frame");
        assert!(
            texels_at < stage_payload_at && stage_payload_end <= sampler_at,
            "the blocks travel in the encoder's order: textures {texels_at}, \
             stage buffers {stage_payload_at}, sampler states {sampler_at}"
        );
        assert_eq!(
            sampler_at, stage_payload_end,
            "the sampler block follows the stage buffer block with nothing between"
        );
        eprintln!(
            "three blocks: len={} sampled_stage={} texels_at={texels_at} \
             stage_payload_at={stage_payload_at} sampler_at={sampler_at}",
            frame.len(),
            sampled_stage_only.len()
        );
    }

    #[test]
    fn a_runtime_sampler_pass_refuses_a_count_above_the_contract_cap() {
        let frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: runtime_sampler_trace(),
            resources: resources(),
        })
        .unwrap();
        let head = runtime_sampler_head_at(&frame);
        let mut patched = frame.clone();
        // One above the contract's own ceiling (`research/docs/23` §3.3,
        // v102): the count byte is refused before a single entry is read, so
        // the patched frame needs no matching entries behind it.
        patched[head + 3] = u8::try_from(MAX_RENDER_SAMPLERS + 1).unwrap();
        assert!(matches!(
            CommandCodec::decode_request(&patched).unwrap_err(),
            CodecError::RenderSamplerCount { count, maximum }
                if count == MAX_RENDER_SAMPLERS + 1 && maximum == MAX_RENDER_SAMPLERS
        ));
        // The encoder refuses the same protocol bound instead of writing a
        // frame the decoder would reject.
        let mut trace = runtime_sampler_trace();
        let Some(TracePass::Render(pass)) = trace.passes.first_mut() else {
            panic!("the fixture is a render pass");
        };
        pass.samplers = (0..=MAX_RENDER_SAMPLERS as u32)
            .map(widened_runtime_sampler)
            .collect();
        let refused = CommandCodec::encode_request(&CommandRequest::Submit {
            trace,
            resources: resources(),
        })
        .unwrap_err();
        assert!(matches!(
            refused,
            CodecError::RenderSamplerCount { count, maximum }
                if count == MAX_RENDER_SAMPLERS + 1 && maximum == MAX_RENDER_SAMPLERS
        ));
        eprintln!("runtime sampler count refusals: {refused}");
    }

    #[test]
    fn a_runtime_sampler_pass_refuses_an_unknown_state_by_name() {
        let frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: runtime_sampler_trace(),
            resources: resources(),
        })
        .unwrap();
        let head = runtime_sampler_head_at(&frame);
        // The entry's two state bytes follow the index: filter first, then
        // address. `0x06` names no filter and `0x05` no address, so each is
        // refused by name rather than folded onto a neighbouring state — a
        // state the decoder guessed would change which texels a remote read
        // returns (`research/docs/23` §3.3, v102/v109).
        let mut foreign_filter = frame.clone();
        foreign_filter[head + 8] = 0x06;
        let refused = CommandCodec::decode_request(&foreign_filter).unwrap_err();
        assert!(matches!(
            refused,
            CodecError::UnknownEnumValue {
                field: "sampler filter",
                value: 6,
            }
        ));
        let mut foreign_address = frame.clone();
        foreign_address[head + 9] = 0x05;
        let refused = CommandCodec::decode_request(&foreign_address).unwrap_err();
        assert!(matches!(
            refused,
            CodecError::UnknownEnumValue {
                field: "sampler address",
                value: 5,
            }
        ));
        eprintln!("foreign runtime sampler states refused: {refused}");
    }

    /// A render contract whose texture declarations state both sampler forms
    /// (`research/docs/23` §3.3, v100/v102): binding 3 samples through the
    /// runtime `[[sampler(0)]]` argument whose state the pass states, and
    /// binding 5 through the state the module's own AIR constexpr sampler
    /// carries.
    fn render_texture_contract() -> RenderPipelineContract {
        RenderPipelineContract {
            textures: vec![
                TextureBindingContract::sampled_runtime(3, TextureFormat::Rgba8Unorm, 0),
                TextureBindingContract::sampled(
                    5,
                    TextureFormat::Bgra8Unorm,
                    SamplerPolicy {
                        filter: SamplerFilter::LinearMipNearest,
                        address: SamplerAddressMode::MirrorRepeat,
                    },
                ),
            ],
            ..render_contract()
        }
    }

    /// The declaration block [`render_texture_contract`] writes: the count,
    /// then `(binding, access, type, format, form, payload, footprint)` per
    /// entry — the runtime form (`0x02`) with the Metal sampler index, then
    /// the static form (`0x01`) with its filter `0x04`
    /// (`LinearMipNearest`) and address `0x03` (`MirrorRepeat`)
    /// (`research/docs/23` §3.3, v102/v109).
    const RENDER_TEXTURE_DECLARATION_BLOCK: [u8; 25] = [
        0x02, // two declarations
        0x00, 0x00, 0x00, 0x03, // binding 3
        0x00, // Sampled
        0x02, // D2
        0x02, // Rgba8Unorm
        0x02, // the runtime form
        0x00, 0x00, 0x00, 0x00, // [[sampler(0)]]
        0x00, // WholeView
        0x00, 0x00, 0x00, 0x05, // binding 5
        0x00, // Sampled
        0x02, // D2
        0x03, // Bgra8Unorm
        0x01, // the static form
        0x04, // LinearMipNearest
        0x03, // MirrorRepeat
        0x00, // WholeView
    ];

    #[test]
    fn a_render_contract_with_texture_declarations_takes_its_own_pipeline_kind() {
        let plain = CommandCodec::encode_request(&render_submit_with_formats(vec![
            AttachmentFormat::Rgba8Unorm,
        ]))
        .unwrap();
        let mut trace = render_only_trace();
        trace.pipelines[0].render = Some(render_texture_contract());
        let request = CommandRequest::Submit {
            trace,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame,
            "the declaration frame re-encodes byte for byte"
        );
        // The tag is the texture face's own (`research/docs/23` §3.3, v100):
        // the entry writes its colour formats the MRT way and appends the
        // declaration block after the vertex layout, where both sampler forms
        // travel.
        assert_eq!(plain[pipeline_entry_kind_offset()], 0x01, "plain entry tag");
        assert_eq!(
            frame[pipeline_entry_kind_offset()],
            0x05,
            "texture declaration entry tag"
        );
        assert!(
            frame
                .windows(RENDER_TEXTURE_DECLARATION_BLOCK.len())
                .any(|window| window == RENDER_TEXTURE_DECLARATION_BLOCK),
            "both sampler forms travel in the declaration block"
        );
        let block_at = frame
            .windows(RENDER_TEXTURE_DECLARATION_BLOCK.len())
            .position(|window| window == RENDER_TEXTURE_DECLARATION_BLOCK)
            .expect("the declaration block is on the wire");
        assert_eq!(
            frame.len(),
            plain.len() + 8 + RENDER_TEXTURE_DECLARATION_BLOCK.len(),
            "the block and the list's length prefix are the whole difference"
        );
        eprintln!(
            "render texture declaration frame: len={} plain={} kind={:#04x} block_at={block_at}",
            frame.len(),
            plain.len(),
            frame[pipeline_entry_kind_offset()]
        );
    }

    #[test]
    fn a_render_contract_with_both_declaration_halves_takes_the_combined_kind() {
        // A contract may declare stage buffers and texture bindings at once;
        // the combined kind writes the stage buffer block first and the
        // texture declarations after it, in the one order the decoder reads
        // (`research/docs/23` §3.3, v83/v100).
        let mut contract = stage_buffer_render_contract();
        contract.textures = render_texture_contract().textures;
        let mut trace = render_only_trace();
        trace.pipelines[0].render = Some(contract);
        let request = CommandRequest::Submit {
            trace,
            resources: resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        assert_eq!(CommandCodec::decode_request(&frame).unwrap(), request);
        assert_eq!(
            CommandCodec::encode_request(&CommandCodec::decode_request(&frame).unwrap()).unwrap(),
            frame,
            "the combined declaration frame re-encodes byte for byte"
        );
        assert_eq!(
            frame[pipeline_entry_kind_offset()],
            0x06,
            "the combined declaration entry tag"
        );
        // The stage buffer declaration block is the v83 fixture's own — its
        // bytes do not depend on this tag — and it sits before the texture
        // declarations.
        let stage_at = frame
            .windows(STAGE_BUFFER_DECLARATION_BLOCK.len())
            .position(|window| window == STAGE_BUFFER_DECLARATION_BLOCK)
            .expect("the combined frame carries the stage buffer block");
        let textures_at = frame
            .windows(RENDER_TEXTURE_DECLARATION_BLOCK.len())
            .position(|window| window == RENDER_TEXTURE_DECLARATION_BLOCK)
            .expect("the combined frame carries the texture block");
        assert!(
            stage_at < textures_at,
            "the stage buffer block precedes the texture declarations: \
             {stage_at} < {textures_at}"
        );
        eprintln!(
            "combined declaration frame: len={} stage_at={stage_at} textures_at={textures_at}",
            frame.len()
        );
    }

    #[test]
    fn a_render_texture_declaration_block_refuses_a_count_above_the_contract_cap() {
        let frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: {
                let mut trace = render_only_trace();
                trace.pipelines[0].render = Some(render_texture_contract());
                trace
            },
            resources: resources(),
        })
        .unwrap();
        let block_at = frame
            .windows(RENDER_TEXTURE_DECLARATION_BLOCK.len())
            .position(|window| window == RENDER_TEXTURE_DECLARATION_BLOCK)
            .expect("the declaration block is on the wire");
        let mut patched = frame.clone();
        // One above the contract's own ceiling (`research/docs/23` §3.3,
        // v102): the count byte is refused before a single tuple is read.
        patched[block_at] = u8::try_from(MAX_RENDER_TEXTURES + 1).unwrap();
        assert!(matches!(
            CommandCodec::decode_request(&patched).unwrap_err(),
            CodecError::RenderTextureDeclarationCount { count, maximum }
                if count == MAX_RENDER_TEXTURES + 1 && maximum == MAX_RENDER_TEXTURES
        ));
        // The encoder refuses the same protocol bound instead of writing a
        // frame the decoder would reject.
        let mut contract = render_texture_contract();
        contract.textures = (0..=MAX_RENDER_TEXTURES as u32)
            .map(|binding| {
                TextureBindingContract::sampled_runtime(binding, TextureFormat::Rgba8Unorm, 0)
            })
            .collect();
        let mut trace = render_only_trace();
        trace.pipelines[0].render = Some(contract);
        let refused = CommandCodec::encode_request(&CommandRequest::Submit {
            trace,
            resources: resources(),
        })
        .unwrap_err();
        assert!(matches!(
            refused,
            CodecError::RenderTextureDeclarationCount { count, maximum }
                if count == MAX_RENDER_TEXTURES + 1 && maximum == MAX_RENDER_TEXTURES
        ));
        eprintln!("render texture declaration count refusals: {refused}");
    }

    #[test]
    fn a_render_texture_declaration_refuses_both_sampler_forms() {
        // The two forms are exclusive (`research/docs/23` §3.3, v102): the
        // block's form byte has no position that could mean "the module's own
        // state *and* the pass's argument", so the sender refuses the pair by
        // name rather than framing one of the two.
        let mut contract = render_texture_contract();
        contract.textures[0] = TextureBindingContract {
            sampler: Some(SamplerPolicy::reviewed_render_sampler()),
            ..contract.textures[0].clone()
        };
        let mut trace = render_only_trace();
        trace.pipelines[0].render = Some(contract);
        let refused = CommandCodec::encode_request(&CommandRequest::Submit {
            trace,
            resources: resources(),
        })
        .unwrap_err();
        assert!(matches!(
            refused,
            CodecError::RenderTextureSamplerFormUnsupported { binding: 3 }
        ));
        eprintln!("both sampler forms refused: {refused}");
    }

    #[test]
    fn a_render_texture_declaration_block_refuses_an_unknown_sampler_form() {
        let frame = CommandCodec::encode_request(&CommandRequest::Submit {
            trace: {
                let mut trace = render_only_trace();
                trace.pipelines[0].render = Some(render_texture_contract());
                trace
            },
            resources: resources(),
        })
        .unwrap();
        let block_at = frame
            .windows(RENDER_TEXTURE_DECLARATION_BLOCK.len())
            .position(|window| window == RENDER_TEXTURE_DECLARATION_BLOCK)
            .expect("the declaration block is on the wire");
        // The first entry's form byte is the eighth byte of the block: the
        // count byte and the entry's binding, access, type and format bytes
        // precede it.
        let mut patched = frame.clone();
        patched[block_at + 8] = 0x03;
        let refused = CommandCodec::decode_request(&patched).unwrap_err();
        assert!(matches!(
            refused,
            CodecError::UnknownEnumValue {
                field: "render texture sampler form",
                value: 3,
            }
        ));
        eprintln!("unknown render texture sampler form refused: {refused}");
    }

    /// The runtime-sampler shape both halves of the pairing state
    /// (`research/docs/23` §3.3, v102): the contract declares binding 0 as
    /// sampling through the runtime `[[sampler(0)]]` argument, and the pass
    /// binds both the texture view and the state that argument executes with.
    ///
    /// The attachment's view is declared the way a submission declares a
    /// landing — a one-thread compute pass that reads the same view — because
    /// admission resolves the render attachment against the trace's own
    /// declarations (`research/docs/23` §3.6), and the declaring pass's
    /// contract is narrowed to the read the declaration states.
    fn admitted_runtime_sampler_trace() -> ComputeTrace {
        let mut compiled = pipeline(&compile_request());
        compiled.contract.buffer_bindings[0].access = BufferAccess::Read;
        compiled.render = Some(RenderPipelineContract {
            textures: vec![TextureBindingContract::sampled_runtime(
                0,
                TextureFormat::Rgba8Unorm,
                0,
            )],
            ..render_contract()
        });
        let landing = BufferView {
            view_id: ViewId::new(71),
            metal_binding: 0,
            allocation_id: AllocationId::new(41),
            offset: 0,
            length: 16,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0; 16]),
        };
        let mut pass = render_pass_descriptor(&compiled, 2, 2);
        pass.textures = vec![sampled_texture_view(0)];
        pass.samplers = vec![widened_runtime_sampler(0)];
        ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: compiled.device_epoch,
            operation_id: OperationId::new(21),
            pipelines: vec![compiled.clone()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![
                TracePass::Compute(ComputePass {
                    pipeline: compiled.pipeline_id,
                    buffers: vec![landing],
                    textures: Vec::new(),
                    dispatch: Dispatch {
                        kind: DispatchKind::ThreadsExact,
                        grid: [1, 1, 1],
                        threads_per_threadgroup: [1, 1, 1],
                    },
                }),
                TracePass::Render(pass),
            ],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        }
    }

    /// The snapshot the runtime-sampler reading admits on: the render bits the
    /// fixture needs, the render-sampler bits (`research/docs/23` §3.3, v70)
    /// and the stage buffer bits whose shapes travel beside them (v83). The
    /// three pairs are the capability the decoded trace is handed to — the
    /// `supports_render_texture_sampling` bit is the one the runtime sampler
    /// face rides on, because a runtime `[[sampler(n)]]` argument is only
    /// meaningful for a texture a declaration pairs it with.
    fn runtime_sampler_capabilities() -> ProviderCapabilities {
        let mut capabilities = fake_capabilities();
        capabilities.max_passes = 8;
        capabilities.supports_render_passes = true;
        capabilities.max_color_attachments = 1;
        capabilities.max_attachment_dimension = [2, 2];
        capabilities.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        capabilities.supports_render_texture_sampling = true;
        capabilities.max_render_textures = MAX_RENDER_TEXTURES as u32;
        capabilities.supported_render_texture_formats = vec![TextureFormat::Rgba8Unorm];
        capabilities.supports_render_stage_buffers = true;
        capabilities.max_render_stage_buffers = MAX_RENDER_STAGE_BUFFERS as u32;
        capabilities
    }

    /// The allocations [`admitted_runtime_sampler_trace`] declares: the
    /// attachment's landing (sixteen bytes, the packed 2×2 `Rgba8Unorm`
    /// extent) and the sampled texture's own bytes (a 4×4 texel block).
    fn runtime_sampler_resources() -> ResourceTableSnapshot {
        let mut resources = ResourceTableSnapshot::new();
        for (allocation, size) in [(41_u64, 16_u64), (53, 64)] {
            resources
                .insert_allocation(AllocationRecord {
                    allocation_id: AllocationId::new(allocation),
                    owner_epoch: DeviceEpoch::new(7),
                    size,
                })
                .unwrap();
        }
        resources
    }

    /// Decode one submission frame into the trace and resource snapshot a
    /// provider process would see.
    fn carried_submission(frame: &[u8]) -> (ComputeTrace, ResourceTableSnapshot) {
        match CommandCodec::decode_request(frame).unwrap() {
            CommandRequest::Submit { trace, resources } => (trace, resources),
            other => panic!("the frame is a submission, got a {} request", other.kind()),
        }
    }

    /// The render entry of a fixture whose trace also carries the declaring
    /// compute pass (`research/docs/23` §3.6).
    fn declared_render_entry(trace: &mut ComputeTrace) -> &mut RenderPassDescriptor {
        trace
            .passes
            .iter_mut()
            .find_map(|pass| match pass {
                TracePass::Render(pass) => Some(pass),
                TracePass::Compute(_) => None,
                TracePass::Landing(_) => None,
            })
            .expect("the fixture carries a render pass")
    }

    /// The render entry of a decoded fixture.
    fn decoded_render_entry(trace: &ComputeTrace) -> &RenderPassDescriptor {
        trace
            .passes
            .iter()
            .find_map(|pass| match pass {
                TracePass::Render(pass) => Some(pass),
                TracePass::Compute(_) => None,
                TracePass::Landing(_) => None,
            })
            .expect("the fixture carries a render pass")
    }

    #[test]
    fn a_decoded_runtime_sampler_pass_admits_on_a_snapshot_that_declares_the_face() {
        // The reading this increment exists for (`research/docs/23` §3.3,
        // v102): the states the pass's `[[sampler(n)]]` arguments execute with
        // travel the frame, decode back into the pass, and the decoded trace
        // reaches admission with the pairing intact — instead of every
        // runtime-sampler frame decoding to an empty list.
        let request = CommandRequest::Submit {
            trace: admitted_runtime_sampler_trace(),
            resources: runtime_sampler_resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        let (trace, resources) = carried_submission(&frame);
        let pass = decoded_render_entry(&trace);
        assert_eq!(
            pass.samplers,
            vec![widened_runtime_sampler(0)],
            "the decoded pass states the very list the owner sent"
        );
        let Some(contract) = trace.pipelines[0].render.as_ref() else {
            panic!("the fixture pipeline carries a render half");
        };
        assert_eq!(
            contract.textures,
            vec![TextureBindingContract::sampled_runtime(
                0,
                TextureFormat::Rgba8Unorm,
                0
            )],
            "the decoded contract states the declaration the pass is paired against"
        );
        runtime_sampler_capabilities()
            .validate_trace(trace, resources)
            .expect("a snapshot that declares the render-sampler face admits the decoded pass");
        eprintln!(
            "decoded runtime sampler pass admitted: len={} samplers={:02x?}",
            frame.len(),
            RUNTIME_SAMPLER_BLOCK
        );
    }

    #[test]
    fn a_snapshot_without_the_render_sampler_capability_refuses_the_decoded_pass_by_name() {
        // Capability fail-closed (`research/docs/23` §3.3, v70): a snapshot
        // that never declared render-side sampling refuses the decoded pass by
        // name instead of executing it against descriptor bytes nobody
        // declared. The bit comes first, so this is the refusal a device
        // without the face answers however the frame was framed.
        let request = CommandRequest::Submit {
            trace: admitted_runtime_sampler_trace(),
            resources: runtime_sampler_resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        let (trace, resources) = carried_submission(&frame);
        let mut missing = runtime_sampler_capabilities();
        missing.supports_render_texture_sampling = false;
        let refusal = missing.validate_trace(trace, resources).unwrap_err();
        assert_eq!(refusal.slug, "render_texture_input_unsupported");
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        // The render entry is the trace's second pass: the declaring compute
        // pass that owns the attachment lands first (`research/docs/23` §3.6).
        assert_eq!(refusal.fields.get("pass"), Some(&FieldValue::Unsigned(1)));
        assert_eq!(
            refusal.fields.get("textures"),
            Some(&FieldValue::Unsigned(1))
        );
        eprintln!(
            "missing render-sampler capability refused: slug={} class={:?} fields={:?}",
            refusal.slug, refusal.class, refusal.fields
        );
    }

    #[test]
    fn a_frame_without_the_sampler_block_refuses_the_pairing_by_name() {
        // The shape a pre-v102 frame decodes to (`research/docs/23` §3.3,
        // v102): the contract still pairs the texture with the runtime
        // `[[sampler(0)]]` argument, but the pass carries no state — exactly
        // what dropping the block would produce — so the pairing refuses the
        // frame by name instead of executing it under a sampler nobody
        // stated.
        let mut dropped = admitted_runtime_sampler_trace();
        let pass = declared_render_entry(&mut dropped);
        pass.samplers = Vec::new();
        let request = CommandRequest::Submit {
            trace: dropped,
            resources: runtime_sampler_resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        let (trace, resources) = carried_submission(&frame);
        let pass = decoded_render_entry(&trace);
        assert!(
            pass.samplers.is_empty(),
            "the frame carries no sampler block"
        );
        let refusal = runtime_sampler_capabilities()
            .validate_trace(trace, resources)
            .unwrap_err();
        assert_eq!(refusal.slug, "render_runtime_sampler_missing");
        eprintln!(
            "dropped sampler states refused: slug={} detail={:?}",
            refusal.slug, refusal.detail
        );
    }

    #[test]
    fn a_pass_that_states_a_sampler_nothing_pairs_with_is_refused_by_name() {
        // The other direction of the same pairing (`research/docs/23` §3.3,
        // v102): a state nothing samples through would fill a descriptor slot
        // no declaration names, so it is refused by name too.
        let mut unpaired = admitted_runtime_sampler_trace();
        let Some(contract) = unpaired.pipelines[0].render.as_mut() else {
            panic!("the fixture pipeline carries a render half");
        };
        contract.textures = vec![TextureBindingContract::sampled(
            0,
            TextureFormat::Rgba8Unorm,
            SamplerPolicy::reviewed_render_sampler(),
        )];
        let request = CommandRequest::Submit {
            trace: unpaired,
            resources: runtime_sampler_resources(),
        };
        let frame = CommandCodec::encode_request(&request).unwrap();
        let (trace, resources) = carried_submission(&frame);
        let refusal = runtime_sampler_capabilities()
            .validate_trace(trace, resources)
            .unwrap_err();
        assert_eq!(refusal.slug, "render_runtime_sampler_unpaired");
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        eprintln!(
            "unpaired sampler state refused: slug={} fields={:?}",
            refusal.slug, refusal.fields
        );
    }
}
