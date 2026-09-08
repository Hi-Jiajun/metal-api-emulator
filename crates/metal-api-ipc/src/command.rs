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
use metal_api_core::provider::{
    CompiledComputePipeline, CompletionDisposition, CompletionReadback, CompletionToken,
    ComputeProvider, ComputeTrace, DeviceEpoch, LeaseId, LeaseImporter, LeaseReservation,
    PipelineCompileRequest, PipelineProvider, ProviderCapabilities, ProviderError,
    ProviderErrorClass, ProviderHealth, ProviderPhase, ProviderSubmission, ResourceTableSnapshot,
    Retryability, StagedLease, ValidatedComputeTrace,
};
use std::fmt;
use std::io::{Read, Write};
use std::sync::Mutex;
use std::time::Duration;

#[cfg(unix)]
use metal_api_core::provider::{BorrowedLease, FieldValue, NoCopyLeaseImporter};
#[cfg(unix)]
use std::collections::BTreeMap;

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

    /// Drop a descriptor-backed no-copy lease import.
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

fn transport_error(phase: ProviderPhase, error: CommandError) -> ProviderError {
    let mut provider_error = ProviderError::new(
        phase,
        ProviderErrorClass::Internal,
        "command_transport_failed",
    )
    .expect("non-empty command transport slug");
    provider_error.retryability = Retryability::Unknown;
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
    }
}

/// Provider endpoint that can import descriptor-backed no-copy leases.
#[cfg(unix)]
pub trait DescriptorProviderEndpoint: ProviderEndpoint + NoCopyLeaseImporter {}

#[cfg(unix)]
impl<T: ProviderEndpoint + NoCopyLeaseImporter> DescriptorProviderEndpoint for T {}

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
                release_borrowed_descriptor(provider, &mut mappings, lease_id)
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

#[cfg(unix)]
fn close_borrowed_mappings(
    provider: &dyn DescriptorProviderEndpoint,
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

#[cfg(unix)]
fn release_borrowed_descriptor(
    provider: &dyn DescriptorProviderEndpoint,
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

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use super::serve_provider_unix;
    use super::{serve_provider, CommandRequest, CommandResponse, RemoteProvider};
    #[cfg(unix)]
    use crate::codec::CodecError;
    use crate::command_codec::CommandCodec;
    use metal_api_core::provider::{
        AllocationId, AllocationRecord, BufferAccess, BufferBindingContract, BufferLease,
        BufferSource, BufferView, BufferWriteback, CompiledComputePipeline, CompletionDisposition,
        CompletionReadback, CompletionToken, ComputePass, ComputeProvider, ComputeTrace,
        DeviceEpoch, Dispatch, DispatchKind, DispatchType, FootprintProof, FunctionIdentity,
        LeaseId, LeaseImporter, LeaseReservation, OperationId, PipelineCompileRequest,
        PipelineContract, PipelineId, PipelineProvider, ProviderCapabilities, ProviderError,
        ProviderErrorClass, ProviderHealth, ProviderPhase, ProviderSubmission,
        ResourceTableSnapshot, Retryability, SemanticDigest, ShaderSource, StagedLease,
        SubmissionId, ValidatedComputeTrace, ViewId, PROVIDER_SCHEMA_VERSION,
    };
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[cfg(unix)]
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
        }
    }

    fn trace(pipeline: &CompiledComputePipeline) -> ComputeTrace {
        ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: pipeline.device_epoch,
            operation_id: OperationId::new(21),
            pipelines: vec![pipeline.clone()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![ComputePass {
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
            }],
            completion_policy: metal_api_core::provider::CompletionPolicy::HostReadback,
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
            CommandResponse::Released,
            CommandResponse::Error { error },
        ];
        for response in responses {
            let frame = CommandCodec::encode_response(&response).unwrap();
            assert_eq!(CommandCodec::decode_response(&frame).unwrap(), response);
        }
    }

    struct FakeProvider {
        epoch: DeviceEpoch,
        capabilities: ProviderCapabilities,
        submissions: Arc<AtomicU64>,
        imports: Arc<AtomicU64>,
        #[cfg(unix)]
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

    #[cfg(unix)]
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
        }
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
