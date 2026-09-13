//! Binary codec for the owner-to-provider command channel.
//!
//! One frame carries one [`CommandRequest`] or one [`CommandResponse`]. The
//! encoding is versioned by [`COMMAND_FRAME_MAGIC`] and every value is
//! length-delimited, so a decoder never trusts an untrusted length beyond the
//! frame itself. The codec is deliberately explicit: the core provider
//! contract stays free of serialization dependencies and the wire format
//! documents every field that crosses the process boundary.

use crate::codec::CodecError;
use crate::command::{CommandRequest, CommandResponse};
use metal_api_core::provider::{
    AffineAccess, AffineTerm, AliasMode, AllocationId, AllocationRecord, AttachmentFormat,
    BufferAccess, BufferBindingContract, BufferLease, BufferSource, BufferView, BufferWriteback,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy,
    CompletionReadback, CompletionToken, ComputePass, ComputeTrace, DeviceEpoch, Dispatch,
    DispatchKind, DispatchType, FieldValue, FootprintProof, FunctionIdentity, FunctionSource,
    LeaseId, LeaseReservation, LoadOp, OperationId, PipelineCompileRequest, PipelineContract,
    PipelineId, ProviderCapabilities, ProviderError, ProviderErrorClass, ProviderHealth,
    ProviderPhase, ProviderSubmission, QueuePriority, RenderAttachment, RenderPassDescriptor,
    RenderPipelineContract, ResourceTableSnapshot, Retryability, SemanticDigest, ShaderSource,
    StagedLease, StorageMode, StoreOp, SubmissionId, TextureAccess, TextureFormat, TextureSource,
    TextureType, TextureView, TracePass, VertexLayout, ViewId, MAX_COLOR_ATTACHMENTS,
};
use std::io::{Read, Write};

/// Protocol magic and version. A different byte sequence is refused.
pub const COMMAND_FRAME_MAGIC: [u8; 4] = *b"MCC1";
/// Maximum encoded payload, excluding the nine-byte frame header.
pub const MAX_COMMAND_FRAME: usize = 64 * 1024 * 1024;
/// Maximum total length of a request or response reassembled from chunk
/// frames.
pub const MAX_CHUNKED_PAYLOAD: usize = 256 * 1024 * 1024;

const REQUEST_FRAME: u8 = 0x01;
const RESPONSE_FRAME: u8 = 0x02;
const CHUNK_FRAME: u8 = 0x03;

const CAPABILITIES_REQUEST: u8 = 0x01;
const COMPILE_REQUEST: u8 = 0x02;
const SUBMIT_REQUEST: u8 = 0x03;
const WAIT_REQUEST: u8 = 0x04;
const READBACK_REQUEST: u8 = 0x05;
const CANCEL_REQUEST: u8 = 0x06;
const RELEASE_PIPELINE_REQUEST: u8 = 0x07;
const RELEASE_COMPLETION_REQUEST: u8 = 0x08;
const HEALTH_REQUEST: u8 = 0x09;
const IMPORT_STAGED_LEASE_REQUEST: u8 = 0x0a;
const RELEASE_STAGED_LEASE_REQUEST: u8 = 0x0b;
const IMPORT_BORROWED_LEASE_REQUEST: u8 = 0x0c;
const RELEASE_BORROWED_LEASE_REQUEST: u8 = 0x0d;
/// Owner queue-priority marking.
///
/// This tag is the reason the channel needed no version bump: it is additive,
/// so a decoder that predates it answers
/// [`CodecError::UnknownCommandTag`] for a marked frame while every frame an
/// older owner sends still decodes here unchanged (and therefore keeps the
/// provider's queues at `QueuePriority::Default`).
const SET_QUEUE_PRIORITIES_REQUEST: u8 = 0x0e;
/// Submit a trace whose pass list uses the tagged compute/render layout.
///
/// A compute-only trace keeps travelling under [`SUBMIT_REQUEST`] with the
/// exact pre-render pass bytes, so this tag is only ever emitted when a render
/// pass is present. An older decoder answers `UnknownCommandTag` for it
/// instead of misreading the tagged pass list, which is the same additive
/// policy the queue-priority tag uses.
const SUBMIT_RENDER_REQUEST: u8 = 0x0f;

const CAPABILITIES_RESPONSE: u8 = 0x01;
const COMPILED_RESPONSE: u8 = 0x02;
const SUBMITTED_RESPONSE: u8 = 0x03;
const OBSERVED_RESPONSE: u8 = 0x04;
const READBACK_RESPONSE: u8 = 0x05;
const RELEASED_RESPONSE: u8 = 0x06;
const HEALTH_RESPONSE: u8 = 0x07;
const IMPORTED_RESPONSE: u8 = 0x08;
/// The device queue table an owner marking installed.
const QUEUE_PRIORITIES_RESPONSE: u8 = 0x09;
/// Capabilities including the render bits the pre-render payload cannot carry.
///
/// Only a snapshot that declares at least one non-default render bit uses this
/// tag; every compute-only provider keeps sending the legacy
/// [`CAPABILITIES_RESPONSE`] bytes.
const RENDER_CAPABILITIES_RESPONSE: u8 = 0x0a;
const ERROR_RESPONSE: u8 = 0x7f;

/// Pass discriminator inside the tagged trace layout.
const PASS_KIND_COMPUTE: u8 = 0x00;
const PASS_KIND_RENDER: u8 = 0x01;

/// Pipeline-table discriminator inside the tagged trace layout.
///
/// The pipeline table carries one entry per registration, and the render half
/// of an entry is what lets core admission compare an attachment with the
/// pipeline the render pass names (review item I3, 2026-09-14). The
/// discriminator is a value rather than an optional length prefix so an
/// unknown tag is refused instead of being read as "no render half", exactly
/// like [`PASS_KIND_COMPUTE`]. Only the tagged layout writes it: a
/// compute-only trace keeps the pre-render pipeline bytes, so
/// `PIPELINE_KIND_COMPUTE` is what an entry in that layout decodes to.
const PIPELINE_KIND_COMPUTE: u8 = 0x00;
const PIPELINE_KIND_RENDER: u8 = 0x01;

/// Maximum number of passes one tagged trace frame may carry.
///
/// The legacy layout keeps its previous (payload-bounded) behaviour; this
/// bound exists so a corrupt tagged count cannot make the decoder allocate
/// before it has read a single pass.
pub const MAX_TAGGED_TRACE_PASSES: usize = 4096;

/// Maximum colour formats one capability snapshot may declare.
///
/// [`AttachmentFormat`] is a closed four-value family, so this bound can never
/// refuse a well-formed snapshot; it only stops a corrupt count from driving
/// the decoder.
pub const MAX_SUPPORTED_COLOR_FORMATS: usize = 16;

/// Maximum number of queue tiers one frame may carry.
///
/// The bound is a protocol limit, not a device limit: a marking describes the
/// owner's view of a device queue table, and the Vulkan provider caps a device
/// at eight queues. Refusing a longer frame keeps a corrupt length from making
/// the decoder allocate; a provider may still read a short marking, because the
/// expansion pads it (`metal_api_core::provider::queue_priorities_for_device`).
pub const MAX_QUEUE_PRIORITIES: usize = 64;

/// Stateless encoder/decoder for command frames.
pub struct CommandCodec;

/// One decoded chunk frame.
#[derive(Debug)]
pub(crate) struct ChunkPayload {
    pub(crate) transfer_id: u64,
    pub(crate) offset: u64,
    pub(crate) total: u64,
    pub(crate) bytes: Vec<u8>,
}

impl CommandCodec {
    /// Encode one complete request frame.
    pub fn encode_request(request: &CommandRequest) -> Result<Vec<u8>, CodecError> {
        frame(REQUEST_FRAME, Self::encode_request_payload(request)?)
    }

    /// Encode one request payload without the frame header.
    ///
    /// A transport that cannot fit the payload into [`MAX_COMMAND_FRAME`]
    /// splits it into chunk frames with `CommandCodec::write_chunk_frame`.
    pub fn encode_request_payload(request: &CommandRequest) -> Result<Vec<u8>, CodecError> {
        let mut encoder = Encoder::new();
        match request {
            CommandRequest::Capabilities => encoder.u8(CAPABILITIES_REQUEST),
            CommandRequest::Health => encoder.u8(HEALTH_REQUEST),
            CommandRequest::Compile { request } => {
                encoder.u8(COMPILE_REQUEST);
                put_compile_request(&mut encoder, request);
            }
            CommandRequest::ImportStagedLease { staged } => {
                encoder.u8(IMPORT_STAGED_LEASE_REQUEST);
                put_staged_lease(&mut encoder, staged);
            }
            CommandRequest::ReleaseStagedLease { lease_id } => {
                encoder.u8(RELEASE_STAGED_LEASE_REQUEST);
                encoder.u64(lease_id.get());
            }
            CommandRequest::ImportBorrowedLease { reservation } => {
                encoder.u8(IMPORT_BORROWED_LEASE_REQUEST);
                put_reservation(&mut encoder, reservation);
            }
            CommandRequest::ReleaseBorrowedLease { lease_id } => {
                encoder.u8(RELEASE_BORROWED_LEASE_REQUEST);
                encoder.u64(lease_id.get());
            }
            CommandRequest::SetQueuePriorities { tiers } => {
                encoder.u8(SET_QUEUE_PRIORITIES_REQUEST);
                put_queue_priorities(&mut encoder, tiers)?;
            }
            CommandRequest::Submit { trace, resources } => {
                encoder.u8(if trace.has_render_passes() {
                    SUBMIT_RENDER_REQUEST
                } else {
                    SUBMIT_REQUEST
                });
                put_trace(&mut encoder, trace)?;
                put_resources(&mut encoder, resources);
            }
            CommandRequest::Wait { token, timeout } => {
                encoder.u8(WAIT_REQUEST);
                put_token(&mut encoder, token);
                encoder.u64(u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX));
            }
            CommandRequest::Readback { token } => {
                encoder.u8(READBACK_REQUEST);
                put_token(&mut encoder, token);
            }
            CommandRequest::Cancel { token } => {
                encoder.u8(CANCEL_REQUEST);
                put_token(&mut encoder, token);
            }
            CommandRequest::ReleasePipeline { pipeline } => {
                encoder.u8(RELEASE_PIPELINE_REQUEST);
                put_pipeline(&mut encoder, pipeline);
            }
            CommandRequest::ReleaseCompletion { token } => {
                encoder.u8(RELEASE_COMPLETION_REQUEST);
                put_token(&mut encoder, token);
            }
        }
        Ok(encoder.bytes)
    }

    /// Decode one complete request frame.
    pub fn decode_request(frame: &[u8]) -> Result<CommandRequest, CodecError> {
        Self::decode_request_payload(unframe(frame, REQUEST_FRAME)?)
    }

    /// Decode one request payload without the frame header.
    pub fn decode_request_payload(payload: &[u8]) -> Result<CommandRequest, CodecError> {
        decode_request_payload(payload)
    }

    /// Encode one complete response frame.
    pub fn encode_response(response: &CommandResponse) -> Result<Vec<u8>, CodecError> {
        frame(RESPONSE_FRAME, Self::encode_response_payload(response)?)
    }

    /// Encode one response payload without the frame header.
    pub fn encode_response_payload(response: &CommandResponse) -> Result<Vec<u8>, CodecError> {
        let mut encoder = Encoder::new();
        match response {
            CommandResponse::Capabilities {
                epoch,
                capabilities,
            } => {
                if capabilities.declares_render_support() {
                    encoder.u8(RENDER_CAPABILITIES_RESPONSE);
                    put_epoch(&mut encoder, *epoch);
                    put_capabilities(&mut encoder, capabilities)?;
                } else {
                    encoder.u8(CAPABILITIES_RESPONSE);
                    put_epoch(&mut encoder, *epoch);
                    put_capabilities_legacy(&mut encoder, capabilities);
                }
            }
            CommandResponse::Health { health } => {
                encoder.u8(HEALTH_RESPONSE);
                put_health(&mut encoder, *health);
            }
            CommandResponse::Compiled { pipeline } => {
                encoder.u8(COMPILED_RESPONSE);
                put_pipeline(&mut encoder, pipeline);
            }
            CommandResponse::Imported => encoder.u8(IMPORTED_RESPONSE),
            CommandResponse::Submitted { submission } => {
                encoder.u8(SUBMITTED_RESPONSE);
                put_submission(&mut encoder, submission);
            }
            CommandResponse::Observed { disposition } => {
                encoder.u8(OBSERVED_RESPONSE);
                put_disposition(&mut encoder, *disposition);
            }
            CommandResponse::Readback { readback } => {
                encoder.u8(READBACK_RESPONSE);
                put_readback(&mut encoder, readback);
            }
            CommandResponse::QueuePriorities { installed } => {
                encoder.u8(QUEUE_PRIORITIES_RESPONSE);
                put_queue_priorities(&mut encoder, installed)?;
            }
            CommandResponse::Released => encoder.u8(RELEASED_RESPONSE),
            CommandResponse::Error { error } => {
                encoder.u8(ERROR_RESPONSE);
                put_error(&mut encoder, error);
            }
        }
        Ok(encoder.bytes)
    }

    /// Decode one complete response frame.
    pub fn decode_response(frame: &[u8]) -> Result<CommandResponse, CodecError> {
        Self::decode_response_payload(unframe(frame, RESPONSE_FRAME)?)
    }

    /// Decode one response payload without the frame header.
    pub fn decode_response_payload(payload: &[u8]) -> Result<CommandResponse, CodecError> {
        decode_response_payload(payload)
    }

    /// Write one request frame.
    pub fn write_request<W: Write>(
        writer: &mut W,
        request: &CommandRequest,
    ) -> Result<(), CodecError> {
        writer.write_all(&Self::encode_request(request)?)?;
        Ok(())
    }

    /// Read one request frame.
    pub fn read_request<R: Read>(reader: &mut R) -> Result<CommandRequest, CodecError> {
        let (kind, payload) = read_frame(reader)?;
        if kind != REQUEST_FRAME {
            return Err(CodecError::UnknownCommandTag(kind));
        }
        decode_request_payload(&payload)
    }

    /// Write one request payload as a single frame.
    pub(crate) fn write_request_payload<W: Write>(
        writer: &mut W,
        payload: &[u8],
    ) -> Result<(), CodecError> {
        write_framed(writer, REQUEST_FRAME, payload)
    }

    /// Write one response payload as a single frame.
    pub(crate) fn write_response_payload<W: Write>(
        writer: &mut W,
        payload: &[u8],
    ) -> Result<(), CodecError> {
        write_framed(writer, RESPONSE_FRAME, payload)
    }

    /// Write one chunk of a request payload.
    pub(crate) fn write_chunk_frame<W: Write>(
        writer: &mut W,
        transfer_id: u64,
        offset: u64,
        total: u64,
        bytes: &[u8],
    ) -> Result<(), CodecError> {
        let mut encoder = Encoder::new();
        encoder.u64(transfer_id);
        encoder.u64(offset);
        encoder.u64(total);
        encoder.blob(bytes);
        write_framed(writer, CHUNK_FRAME, &encoder.bytes)
    }

    /// Read one frame without interpreting its kind.
    pub(crate) fn read_raw_frame<R: Read>(reader: &mut R) -> Result<(u8, Vec<u8>), CodecError> {
        read_frame(reader)
    }

    /// Decode one chunk frame payload.
    pub(crate) fn decode_chunk_payload(payload: &[u8]) -> Result<ChunkPayload, CodecError> {
        let mut decoder = Decoder::new(payload);
        let transfer_id = decoder.u64()?;
        let offset = decoder.u64()?;
        let total = decoder.u64()?;
        let bytes = decoder.blob()?;
        decoder.finish()?;
        Ok(ChunkPayload {
            transfer_id,
            offset,
            total,
            bytes,
        })
    }

    /// Frame kind of a request frame.
    pub(crate) const fn request_frame_kind() -> u8 {
        REQUEST_FRAME
    }

    /// Frame kind of a response frame.
    pub(crate) const fn response_frame_kind() -> u8 {
        RESPONSE_FRAME
    }

    /// Frame kind of a chunk frame.
    pub(crate) const fn chunk_frame_kind() -> u8 {
        CHUNK_FRAME
    }

    /// Write one response frame.
    pub fn write_response<W: Write>(
        writer: &mut W,
        response: &CommandResponse,
    ) -> Result<(), CodecError> {
        writer.write_all(&Self::encode_response(response)?)?;
        Ok(())
    }

    /// Read one response frame.
    pub fn read_response<R: Read>(reader: &mut R) -> Result<CommandResponse, CodecError> {
        let (kind, payload) = read_frame(reader)?;
        if kind != RESPONSE_FRAME {
            return Err(CodecError::UnknownCommandTag(kind));
        }
        decode_response_payload(&payload)
    }
}

fn decode_request_payload(payload: &[u8]) -> Result<CommandRequest, CodecError> {
    let mut decoder = Decoder::new(payload);
    let tag = decoder.u8()?;
    let request = match tag {
        CAPABILITIES_REQUEST => CommandRequest::Capabilities,
        HEALTH_REQUEST => CommandRequest::Health,
        COMPILE_REQUEST => CommandRequest::Compile {
            request: get_compile_request(&mut decoder)?,
        },
        IMPORT_STAGED_LEASE_REQUEST => CommandRequest::ImportStagedLease {
            staged: get_staged_lease(&mut decoder)?,
        },
        RELEASE_STAGED_LEASE_REQUEST => CommandRequest::ReleaseStagedLease {
            lease_id: LeaseId::new(decoder.u64()?),
        },
        IMPORT_BORROWED_LEASE_REQUEST => CommandRequest::ImportBorrowedLease {
            reservation: get_reservation(&mut decoder)?,
        },
        RELEASE_BORROWED_LEASE_REQUEST => CommandRequest::ReleaseBorrowedLease {
            lease_id: LeaseId::new(decoder.u64()?),
        },
        SET_QUEUE_PRIORITIES_REQUEST => CommandRequest::SetQueuePriorities {
            tiers: get_queue_priorities(&mut decoder)?,
        },
        SUBMIT_REQUEST => CommandRequest::Submit {
            trace: get_trace(&mut decoder)?,
            resources: get_resources(&mut decoder)?,
        },
        SUBMIT_RENDER_REQUEST => CommandRequest::Submit {
            trace: get_trace_tagged(&mut decoder)?,
            resources: get_resources(&mut decoder)?,
        },
        WAIT_REQUEST => CommandRequest::Wait {
            token: get_token(&mut decoder)?,
            timeout: std::time::Duration::from_millis(decoder.u64()?),
        },
        READBACK_REQUEST => CommandRequest::Readback {
            token: get_token(&mut decoder)?,
        },
        CANCEL_REQUEST => CommandRequest::Cancel {
            token: get_token(&mut decoder)?,
        },
        RELEASE_PIPELINE_REQUEST => CommandRequest::ReleasePipeline {
            pipeline: get_pipeline(&mut decoder)?,
        },
        RELEASE_COMPLETION_REQUEST => CommandRequest::ReleaseCompletion {
            token: get_token(&mut decoder)?,
        },
        other => return Err(CodecError::UnknownCommandTag(other)),
    };
    decoder.finish()?;
    Ok(request)
}

fn decode_response_payload(payload: &[u8]) -> Result<CommandResponse, CodecError> {
    let mut decoder = Decoder::new(payload);
    let tag = decoder.u8()?;
    let response = match tag {
        CAPABILITIES_RESPONSE => {
            let epoch = get_epoch(&mut decoder)?;
            CommandResponse::Capabilities {
                epoch,
                capabilities: get_capabilities_legacy(&mut decoder)?,
            }
        }
        RENDER_CAPABILITIES_RESPONSE => CommandResponse::Capabilities {
            epoch: get_epoch(&mut decoder)?,
            capabilities: get_capabilities(&mut decoder)?,
        },
        HEALTH_RESPONSE => CommandResponse::Health {
            health: get_health(&mut decoder)?,
        },
        COMPILED_RESPONSE => CommandResponse::Compiled {
            pipeline: get_pipeline(&mut decoder)?,
        },
        IMPORTED_RESPONSE => CommandResponse::Imported,
        SUBMITTED_RESPONSE => CommandResponse::Submitted {
            submission: get_submission(&mut decoder)?,
        },
        OBSERVED_RESPONSE => CommandResponse::Observed {
            disposition: get_disposition(&mut decoder)?,
        },
        READBACK_RESPONSE => CommandResponse::Readback {
            readback: get_readback(&mut decoder)?,
        },
        QUEUE_PRIORITIES_RESPONSE => CommandResponse::QueuePriorities {
            installed: get_queue_priorities(&mut decoder)?,
        },
        RELEASED_RESPONSE => CommandResponse::Released,
        ERROR_RESPONSE => CommandResponse::Error {
            error: get_error(&mut decoder)?,
        },
        other => return Err(CodecError::UnknownCommandTag(other)),
    };
    decoder.finish()?;
    Ok(response)
}

fn reframe(kind: u8, payload: Vec<u8>) -> Vec<u8> {
    let mut frame = Vec::with_capacity(9 + payload.len());
    frame.extend_from_slice(&COMMAND_FRAME_MAGIC);
    frame.push(kind);
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);
    frame
}

fn frame(kind: u8, payload: Vec<u8>) -> Result<Vec<u8>, CodecError> {
    if payload.len() > MAX_COMMAND_FRAME {
        return Err(CodecError::FrameTooLarge {
            length: payload.len(),
            maximum: MAX_COMMAND_FRAME,
        });
    }
    Ok(reframe(kind, payload))
}

fn write_framed<W: Write>(writer: &mut W, kind: u8, payload: &[u8]) -> Result<(), CodecError> {
    if payload.len() > MAX_COMMAND_FRAME {
        return Err(CodecError::FrameTooLarge {
            length: payload.len(),
            maximum: MAX_COMMAND_FRAME,
        });
    }
    let mut header = [0u8; 9];
    header[..4].copy_from_slice(&COMMAND_FRAME_MAGIC);
    header[4] = kind;
    header[5..].copy_from_slice(&(payload.len() as u32).to_be_bytes());
    writer.write_all(&header).map_err(CodecError::Io)?;
    writer.write_all(payload).map_err(CodecError::Io)?;
    Ok(())
}

fn unframe(frame: &[u8], expected: u8) -> Result<&[u8], CodecError> {
    if frame.len() < 9 {
        return Err(CodecError::TruncatedFrame {
            expected: 9,
            actual: frame.len(),
        });
    }
    let mut magic = [0u8; 4];
    magic.copy_from_slice(&frame[..4]);
    if magic != COMMAND_FRAME_MAGIC {
        return Err(CodecError::BadMagic(magic));
    }
    let kind = frame[4];
    if kind != expected {
        return Err(CodecError::UnknownCommandTag(kind));
    }
    let length = u32::from_be_bytes([frame[5], frame[6], frame[7], frame[8]]) as usize;
    if length > MAX_COMMAND_FRAME {
        return Err(CodecError::FrameTooLarge {
            length,
            maximum: MAX_COMMAND_FRAME,
        });
    }
    if frame.len() != 9 + length {
        return Err(CodecError::TruncatedFrame {
            expected: 9 + length,
            actual: frame.len(),
        });
    }
    Ok(&frame[9..])
}

fn read_frame<R: Read>(reader: &mut R) -> Result<(u8, Vec<u8>), CodecError> {
    let mut header = [0u8; 9];
    read_header(reader, &mut header)?;
    let mut magic = [0u8; 4];
    magic.copy_from_slice(&header[..4]);
    if magic != COMMAND_FRAME_MAGIC {
        return Err(CodecError::BadMagic(magic));
    }
    let kind = header[4];
    let length = u32::from_be_bytes([header[5], header[6], header[7], header[8]]) as usize;
    if length > MAX_COMMAND_FRAME {
        return Err(CodecError::FrameTooLarge {
            length,
            maximum: MAX_COMMAND_FRAME,
        });
    }
    let mut payload = vec![0u8; length];
    read_payload(reader, &mut payload)?;
    Ok((kind, payload))
}

fn read_header<R: Read>(reader: &mut R, header: &mut [u8; 9]) -> Result<(), CodecError> {
    let mut read = 0;
    while read < header.len() {
        match reader.read(&mut header[read..]) {
            Ok(0) if read == 0 => return Err(CodecError::Eof),
            Ok(0) => {
                return Err(CodecError::TruncatedFrame {
                    expected: header.len(),
                    actual: read,
                })
            }
            Ok(count) => read += count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(CodecError::Io(error)),
        }
    }
    Ok(())
}

fn read_payload<R: Read>(reader: &mut R, payload: &mut [u8]) -> Result<(), CodecError> {
    let mut read = 0;
    while read < payload.len() {
        match reader.read(&mut payload[read..]) {
            Ok(0) => {
                return Err(CodecError::TruncatedFrame {
                    expected: payload.len(),
                    actual: read,
                })
            }
            Ok(count) => read += count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(CodecError::Io(error)),
        }
    }
    Ok(())
}

struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn blob(&mut self, value: &[u8]) {
        self.u64(value.len() as u64);
        self.bytes.extend_from_slice(value);
    }

    fn text(&mut self, value: &str) {
        self.blob(value.as_bytes());
    }

    fn opt_u64(&mut self, value: Option<u64>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.u64(value);
            }
            None => self.u8(0),
        }
    }

    fn opt_text(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.text(value);
            }
            None => self.u8(0),
        }
    }

    fn array3(&mut self, value: [u64; 3]) {
        for axis in value {
            self.u64(axis);
        }
    }

    fn opt_array3(&mut self, value: Option<[u64; 3]>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.array3(value);
            }
            None => self.u8(0),
        }
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], CodecError> {
        let remaining = self.remaining();
        if length > remaining {
            return Err(CodecError::TruncatedPayload {
                needed: length,
                remaining,
            });
        }
        let start = self.position;
        self.position += length;
        Ok(&self.bytes[start..self.position])
    }

    fn u8(&mut self) -> Result<u8, CodecError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, CodecError> {
        let mut bytes = [0u8; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_be_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, CodecError> {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_be_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, CodecError> {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_be_bytes(bytes))
    }

    fn i64(&mut self) -> Result<i64, CodecError> {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(i64::from_be_bytes(bytes))
    }

    fn bool(&mut self) -> Result<bool, CodecError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(CodecError::UnknownEnumValue {
                field: "boolean",
                value,
            }),
        }
    }

    fn blob(&mut self) -> Result<Vec<u8>, CodecError> {
        let length = usize::try_from(self.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: self.remaining(),
        })?;
        Ok(self.take(length)?.to_vec())
    }

    fn text(&mut self) -> Result<String, CodecError> {
        String::from_utf8(self.blob()?).map_err(CodecError::InvalidUtf8)
    }

    fn opt_u64(&mut self) -> Result<Option<u64>, CodecError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.u64()?)),
            value => Err(CodecError::UnknownEnumValue {
                field: "optional u64",
                value,
            }),
        }
    }

    fn opt_text(&mut self) -> Result<Option<String>, CodecError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.text()?)),
            value => Err(CodecError::UnknownEnumValue {
                field: "optional text",
                value,
            }),
        }
    }

    fn array3(&mut self) -> Result<[u64; 3], CodecError> {
        Ok([self.u64()?, self.u64()?, self.u64()?])
    }

    fn opt_array3(&mut self) -> Result<Option<[u64; 3]>, CodecError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.array3()?)),
            value => Err(CodecError::UnknownEnumValue {
                field: "optional u64 array",
                value,
            }),
        }
    }

    fn finish(self) -> Result<(), CodecError> {
        if self.position == self.bytes.len() {
            Ok(())
        } else {
            Err(CodecError::TrailingPayload {
                extra: self.bytes.len() - self.position,
            })
        }
    }
}

fn put_epoch(encoder: &mut Encoder, epoch: DeviceEpoch) {
    encoder.u64(epoch.get());
}

fn get_epoch(decoder: &mut Decoder<'_>) -> Result<DeviceEpoch, CodecError> {
    Ok(DeviceEpoch::new(decoder.u64()?))
}

fn put_token(encoder: &mut Encoder, token: &CompletionToken) {
    put_epoch(encoder, token.device_epoch);
    encoder.u64(token.submission_id.get());
}

fn get_token(decoder: &mut Decoder<'_>) -> Result<CompletionToken, CodecError> {
    Ok(CompletionToken {
        device_epoch: get_epoch(decoder)?,
        submission_id: SubmissionId::new(decoder.u64()?),
    })
}

/// One queue-priority table: a bounded count followed by one rank per tier.
///
/// The count is explicit so the decoder can refuse an oversized table before
/// reading it, and each tier travels as its scheduler rank
/// ([`QueuePriority::rank`]) rather than as a dense enum byte, so the wire value
/// is the same number the policy compares.
fn put_queue_priorities(encoder: &mut Encoder, tiers: &[QueuePriority]) -> Result<(), CodecError> {
    if tiers.len() > MAX_QUEUE_PRIORITIES {
        return Err(CodecError::QueuePriorityCount {
            count: tiers.len(),
            maximum: MAX_QUEUE_PRIORITIES,
        });
    }
    encoder.u64(tiers.len() as u64);
    for tier in tiers {
        encoder.u8(tier.rank());
    }
    Ok(())
}

fn get_queue_priorities(decoder: &mut Decoder<'_>) -> Result<Vec<QueuePriority>, CodecError> {
    let count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::QueuePriorityCount {
        count: usize::MAX,
        maximum: MAX_QUEUE_PRIORITIES,
    })?;
    if count > MAX_QUEUE_PRIORITIES {
        return Err(CodecError::QueuePriorityCount {
            count,
            maximum: MAX_QUEUE_PRIORITIES,
        });
    }
    let mut tiers = Vec::with_capacity(count);
    for _ in 0..count {
        let rank = decoder.u8()?;
        // A rank this build does not know is refused, never folded to
        // `Default`: the frame carries a tier the decoder cannot honour.
        let tier = QueuePriority::from_rank(rank).ok_or(CodecError::UnknownEnumValue {
            field: "queue priority",
            value: rank,
        })?;
        tiers.push(tier);
    }
    Ok(tiers)
}

fn put_digest(encoder: &mut Encoder, digest: &SemanticDigest) {
    encoder.text(digest.scheme());
    encoder.blob(digest.bytes());
}

fn get_digest(decoder: &mut Decoder<'_>) -> Result<SemanticDigest, CodecError> {
    Ok(SemanticDigest::new(decoder.text()?, decoder.blob()?)?)
}

fn put_function_source(encoder: &mut Encoder, source: FunctionSource) {
    encoder.u8(match source {
        FunctionSource::SanitizedLl => 0,
        FunctionSource::BinaryAir => 1,
        FunctionSource::MetalSource => 2,
        FunctionSource::Metallib => 3,
    });
}

fn get_function_source(decoder: &mut Decoder<'_>) -> Result<FunctionSource, CodecError> {
    match decoder.u8()? {
        0 => Ok(FunctionSource::SanitizedLl),
        1 => Ok(FunctionSource::BinaryAir),
        2 => Ok(FunctionSource::MetalSource),
        3 => Ok(FunctionSource::Metallib),
        value => Err(CodecError::UnknownEnumValue {
            field: "function source",
            value,
        }),
    }
}

fn put_shader_source(encoder: &mut Encoder, source: &ShaderSource) {
    match source {
        ShaderSource::SanitizedLl(source) => {
            encoder.u8(0);
            encoder.text(source);
        }
        ShaderSource::BinaryAir(source) => {
            encoder.u8(1);
            encoder.blob(source);
        }
        ShaderSource::MetalSource(source) => {
            encoder.u8(2);
            encoder.text(source);
        }
    }
}

fn get_shader_source(decoder: &mut Decoder<'_>) -> Result<ShaderSource, CodecError> {
    match decoder.u8()? {
        0 => Ok(ShaderSource::SanitizedLl(decoder.text()?)),
        1 => Ok(ShaderSource::BinaryAir(decoder.blob()?)),
        2 => Ok(ShaderSource::MetalSource(decoder.text()?)),
        value => Err(CodecError::UnknownEnumValue {
            field: "shader source",
            value,
        }),
    }
}

fn put_compile_request(encoder: &mut Encoder, request: &PipelineCompileRequest) {
    encoder.text(&request.entry_name);
    put_digest(encoder, &request.logical_digest);
    put_shader_source(encoder, &request.source);
}

fn get_compile_request(decoder: &mut Decoder<'_>) -> Result<PipelineCompileRequest, CodecError> {
    Ok(PipelineCompileRequest {
        entry_name: decoder.text()?,
        logical_digest: get_digest(decoder)?,
        source: get_shader_source(decoder)?,
    })
}

fn put_function_identity(encoder: &mut Encoder, function: &FunctionIdentity) {
    put_digest(encoder, &function.logical_digest);
    encoder.text(&function.entry_name);
    put_function_source(encoder, function.source);
}

fn get_function_identity(decoder: &mut Decoder<'_>) -> Result<FunctionIdentity, CodecError> {
    Ok(FunctionIdentity {
        logical_digest: get_digest(decoder)?,
        entry_name: decoder.text()?,
        source: get_function_source(decoder)?,
    })
}

fn put_dispatch_kind(encoder: &mut Encoder, kind: DispatchKind) {
    encoder.u8(match kind {
        DispatchKind::ThreadsExact => 0,
        DispatchKind::Threadgroups => 1,
    });
}

fn get_dispatch_kind(decoder: &mut Decoder<'_>) -> Result<DispatchKind, CodecError> {
    match decoder.u8()? {
        0 => Ok(DispatchKind::ThreadsExact),
        1 => Ok(DispatchKind::Threadgroups),
        value => Err(CodecError::UnknownEnumValue {
            field: "dispatch kind",
            value,
        }),
    }
}

fn put_access(encoder: &mut Encoder, access: BufferAccess) {
    encoder.u8(match access {
        BufferAccess::Read => 0,
        BufferAccess::Write => 1,
        BufferAccess::ReadWrite => 2,
        BufferAccess::Unused => 3,
    });
}

fn get_access(decoder: &mut Decoder<'_>) -> Result<BufferAccess, CodecError> {
    match decoder.u8()? {
        0 => Ok(BufferAccess::Read),
        1 => Ok(BufferAccess::Write),
        2 => Ok(BufferAccess::ReadWrite),
        3 => Ok(BufferAccess::Unused),
        value => Err(CodecError::UnknownEnumValue {
            field: "buffer access",
            value,
        }),
    }
}

fn put_footprint(encoder: &mut Encoder, footprint: &FootprintProof) {
    match footprint {
        FootprintProof::Static { max_bytes } => {
            encoder.u8(0);
            encoder.u64(*max_bytes);
        }
        FootprintProof::Affine { accesses } => {
            encoder.u8(1);
            encoder.u64(accesses.len() as u64);
            for access in accesses {
                encoder.u64(access.base_offset);
                encoder.u64(access.access_size);
                encoder.u64(access.terms.len() as u64);
                for term in &access.terms {
                    encoder.u8(term.axis);
                    encoder.u64(term.stride);
                }
            }
        }
        FootprintProof::Unbounded => encoder.u8(2),
    }
}

fn get_footprint(decoder: &mut Decoder<'_>) -> Result<FootprintProof, CodecError> {
    match decoder.u8()? {
        0 => Ok(FootprintProof::Static {
            max_bytes: decoder.u64()?,
        }),
        1 => {
            let count =
                usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
                    needed: usize::MAX,
                    remaining: decoder.remaining(),
                })?;
            let mut accesses = Vec::with_capacity(count.min(1024));
            for _ in 0..count {
                let base_offset = decoder.u64()?;
                let access_size = decoder.u64()?;
                let terms =
                    usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
                        needed: usize::MAX,
                        remaining: decoder.remaining(),
                    })?;
                let mut affine_terms = Vec::with_capacity(terms.min(1024));
                for _ in 0..terms {
                    affine_terms.push(AffineTerm {
                        axis: decoder.u8()?,
                        stride: decoder.u64()?,
                    });
                }
                accesses.push(AffineAccess {
                    base_offset,
                    access_size,
                    terms: affine_terms,
                });
            }
            Ok(FootprintProof::Affine { accesses })
        }
        2 => Ok(FootprintProof::Unbounded),
        value => Err(CodecError::UnknownEnumValue {
            field: "footprint proof",
            value,
        }),
    }
}

/// Encode the compute half of one pipeline-table entry.
///
/// The pre-render layout carries exactly these bytes, so this function stays
/// the legacy encoder for `Compiled`, `ReleasePipeline` and any compute-only
/// trace; the render half travels through [`put_pipeline_tagged`] only.
fn put_pipeline(encoder: &mut Encoder, pipeline: &CompiledComputePipeline) {
    put_epoch(encoder, pipeline.device_epoch);
    encoder.u64(pipeline.pipeline_id.get());
    put_function_identity(encoder, &pipeline.function);
    put_contract(encoder, &pipeline.contract);
}

/// Encode one pipeline-table entry of the tagged layout: the entry kind, then
/// the compute half, then the render half when the entry carries one.
fn put_pipeline_tagged(encoder: &mut Encoder, pipeline: &CompiledComputePipeline) {
    match &pipeline.render {
        Some(contract) => {
            encoder.u8(PIPELINE_KIND_RENDER);
            put_pipeline(encoder, pipeline);
            put_render_pipeline_contract(encoder, contract);
        }
        None => {
            encoder.u8(PIPELINE_KIND_COMPUTE);
            put_pipeline(encoder, pipeline);
        }
    }
}

fn put_render_pipeline_contract(encoder: &mut Encoder, contract: &RenderPipelineContract) {
    encoder.text(&contract.vertex_entry);
    encoder.text(&contract.fragment_entry);
    put_attachment_format(encoder, contract.color_format);
    encoder.u8(match contract.vertex_layout {
        VertexLayout::None => 0,
    });
}

/// Decode the compute half of one pipeline-table entry.
///
/// `render` is always `None`: the pre-render entry has no render half, and
/// [`get_pipeline_tagged`] fills it in for the layout that carries one.
fn get_pipeline(decoder: &mut Decoder<'_>) -> Result<CompiledComputePipeline, CodecError> {
    Ok(CompiledComputePipeline {
        device_epoch: get_epoch(decoder)?,
        pipeline_id: PipelineId::new(decoder.u64()?),
        function: get_function_identity(decoder)?,
        contract: get_contract(decoder)?,
        render: None,
    })
}

/// Decode one pipeline-table entry of the tagged layout.
fn get_pipeline_tagged(decoder: &mut Decoder<'_>) -> Result<CompiledComputePipeline, CodecError> {
    let kind = decoder.u8()?;
    let mut pipeline = get_pipeline(decoder)?;
    pipeline.render = match kind {
        PIPELINE_KIND_COMPUTE => None,
        PIPELINE_KIND_RENDER => Some(get_render_pipeline_contract(decoder)?),
        tag => return Err(CodecError::UnknownPipelineTag(tag)),
    };
    Ok(pipeline)
}

fn get_render_pipeline_contract(
    decoder: &mut Decoder<'_>,
) -> Result<RenderPipelineContract, CodecError> {
    Ok(RenderPipelineContract {
        vertex_entry: decoder.text()?,
        fragment_entry: decoder.text()?,
        color_format: get_attachment_format(decoder)?,
        vertex_layout: get_vertex_layout(decoder)?,
    })
}

fn get_vertex_layout(decoder: &mut Decoder<'_>) -> Result<VertexLayout, CodecError> {
    match decoder.u8()? {
        0 => Ok(VertexLayout::None),
        value => Err(CodecError::UnknownEnumValue {
            field: "vertex layout",
            value,
        }),
    }
}

fn put_contract(encoder: &mut Encoder, contract: &PipelineContract) {
    put_dispatch_kind(encoder, contract.dispatch_kind);
    encoder.opt_array3(contract.required_local_size);
    encoder.opt_array3(contract.fixed_grid);
    encoder.u32(contract.push_constant_offset);
    encoder.u32(contract.push_constant_bytes);
    encoder.u64(contract.buffer_bindings.len() as u64);
    for binding in &contract.buffer_bindings {
        encoder.u32(binding.metal_binding);
        put_access(encoder, binding.access);
        put_footprint(encoder, &binding.footprint);
    }
    encoder.u64(contract.shader_capabilities.len() as u64);
    for capability in &contract.shader_capabilities {
        encoder.text(capability);
    }
    match &contract.translator_revision {
        Some(digest) => {
            encoder.u8(1);
            put_digest(encoder, digest);
        }
        None => encoder.u8(0),
    }
}

fn get_contract(decoder: &mut Decoder<'_>) -> Result<PipelineContract, CodecError> {
    let dispatch_kind = get_dispatch_kind(decoder)?;
    let required_local_size = decoder.opt_array3()?;
    let fixed_grid = decoder.opt_array3()?;
    let push_constant_offset = decoder.u32()?;
    let push_constant_bytes = decoder.u32()?;
    let binding_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    let mut buffer_bindings = Vec::with_capacity(binding_count.min(1024));
    for _ in 0..binding_count {
        buffer_bindings.push(BufferBindingContract {
            metal_binding: decoder.u32()?,
            access: get_access(decoder)?,
            footprint: get_footprint(decoder)?,
        });
    }
    let capability_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    let mut shader_capabilities = Vec::with_capacity(capability_count.min(1024));
    for _ in 0..capability_count {
        shader_capabilities.push(decoder.text()?);
    }
    let translator_revision = match decoder.u8()? {
        0 => None,
        1 => Some(get_digest(decoder)?),
        value => {
            return Err(CodecError::UnknownEnumValue {
                field: "translator revision",
                value,
            })
        }
    };
    Ok(PipelineContract {
        dispatch_kind,
        required_local_size,
        fixed_grid,
        push_constant_offset,
        push_constant_bytes,
        buffer_bindings,
        shader_capabilities,
        translator_revision,
    })
}

fn put_view(encoder: &mut Encoder, view: &BufferView) {
    encoder.u64(view.view_id.get());
    encoder.u32(view.metal_binding);
    encoder.u64(view.allocation_id.get());
    encoder.u64(view.offset);
    encoder.u64(view.length);
    put_access(encoder, view.access);
    encoder.opt_u64(view.attribute_stride);
    match &view.source {
        BufferSource::OwnedBytes(bytes) => {
            encoder.u8(0);
            encoder.blob(bytes);
        }
        BufferSource::StagedLease(lease_id) => {
            encoder.u8(1);
            encoder.u64(lease_id.get());
        }
        BufferSource::BorrowedNoCopy(lease_id) => {
            encoder.u8(2);
            encoder.u64(lease_id.get());
        }
    }
}

fn get_view(decoder: &mut Decoder<'_>) -> Result<BufferView, CodecError> {
    let view_id = ViewId::new(decoder.u64()?);
    let metal_binding = decoder.u32()?;
    let allocation_id = AllocationId::new(decoder.u64()?);
    let offset = decoder.u64()?;
    let length = decoder.u64()?;
    let access = get_access(decoder)?;
    let attribute_stride = decoder.opt_u64()?;
    let source = match decoder.u8()? {
        0 => BufferSource::OwnedBytes(decoder.blob()?),
        1 => BufferSource::StagedLease(LeaseId::new(decoder.u64()?)),
        2 => BufferSource::BorrowedNoCopy(LeaseId::new(decoder.u64()?)),
        value => {
            return Err(CodecError::UnknownEnumValue {
                field: "buffer source",
                value,
            })
        }
    };
    Ok(BufferView {
        view_id,
        metal_binding,
        allocation_id,
        offset,
        length,
        access,
        attribute_stride,
        source,
    })
}

fn put_texture_type(encoder: &mut Encoder, texture_type: TextureType) {
    encoder.u8(match texture_type {
        TextureType::D1 => 0,
        TextureType::D1Array => 1,
        TextureType::D2 => 2,
        TextureType::D2Array => 3,
        TextureType::D2Multisample => 4,
        TextureType::D2MultisampleArray => 5,
        TextureType::D3 => 6,
    });
}

fn get_texture_type(decoder: &mut Decoder<'_>) -> Result<TextureType, CodecError> {
    match decoder.u8()? {
        0 => Ok(TextureType::D1),
        1 => Ok(TextureType::D1Array),
        2 => Ok(TextureType::D2),
        3 => Ok(TextureType::D2Array),
        4 => Ok(TextureType::D2Multisample),
        5 => Ok(TextureType::D2MultisampleArray),
        6 => Ok(TextureType::D3),
        value => Err(CodecError::UnknownEnumValue {
            field: "texture type",
            value,
        }),
    }
}

fn put_texture_format(encoder: &mut Encoder, format: TextureFormat) {
    encoder.u8(match format {
        TextureFormat::R32Uint => 0,
        TextureFormat::R32Float => 1,
        TextureFormat::Rgba8Unorm => 2,
        TextureFormat::Bgra8Unorm => 3,
    });
}

fn get_texture_format(decoder: &mut Decoder<'_>) -> Result<TextureFormat, CodecError> {
    match decoder.u8()? {
        0 => Ok(TextureFormat::R32Uint),
        1 => Ok(TextureFormat::R32Float),
        2 => Ok(TextureFormat::Rgba8Unorm),
        3 => Ok(TextureFormat::Bgra8Unorm),
        value => Err(CodecError::UnknownEnumValue {
            field: "texture format",
            value,
        }),
    }
}

fn put_texture_access(encoder: &mut Encoder, access: TextureAccess) {
    encoder.u8(match access {
        TextureAccess::Sampled => 0,
        TextureAccess::Storage => 1,
        TextureAccess::Unused => 2,
    });
}

fn get_texture_access(decoder: &mut Decoder<'_>) -> Result<TextureAccess, CodecError> {
    match decoder.u8()? {
        0 => Ok(TextureAccess::Sampled),
        1 => Ok(TextureAccess::Storage),
        2 => Ok(TextureAccess::Unused),
        value => Err(CodecError::UnknownEnumValue {
            field: "texture access",
            value,
        }),
    }
}

fn put_texture_source(encoder: &mut Encoder, source: &TextureSource) {
    match source {
        TextureSource::OwnedBytes(bytes) => {
            encoder.u8(0);
            encoder.blob(bytes);
        }
        TextureSource::StagedLease(lease_id) => {
            encoder.u8(1);
            encoder.u64(lease_id.get());
        }
        TextureSource::BorrowedNoCopy(lease_id) => {
            encoder.u8(2);
            encoder.u64(lease_id.get());
        }
    }
}

fn get_texture_source(decoder: &mut Decoder<'_>) -> Result<TextureSource, CodecError> {
    match decoder.u8()? {
        0 => Ok(TextureSource::OwnedBytes(decoder.blob()?)),
        1 => Ok(TextureSource::StagedLease(LeaseId::new(decoder.u64()?))),
        2 => Ok(TextureSource::BorrowedNoCopy(LeaseId::new(decoder.u64()?))),
        value => Err(CodecError::UnknownEnumValue {
            field: "texture source",
            value,
        }),
    }
}

fn put_texture(encoder: &mut Encoder, texture: &TextureView) {
    encoder.u64(texture.view_id.get());
    encoder.u32(texture.metal_binding);
    encoder.u64(texture.allocation_id.get());
    put_texture_type(encoder, texture.texture_type);
    put_texture_format(encoder, texture.format);
    encoder.u64(texture.width);
    encoder.u64(texture.height);
    encoder.u64(texture.depth);
    encoder.u64(texture.array_length);
    encoder.u64(texture.sample_count);
    put_texture_access(encoder, texture.access);
    put_texture_source(encoder, &texture.source);
}

fn get_texture(decoder: &mut Decoder<'_>) -> Result<TextureView, CodecError> {
    Ok(TextureView {
        view_id: ViewId::new(decoder.u64()?),
        metal_binding: decoder.u32()?,
        allocation_id: AllocationId::new(decoder.u64()?),
        texture_type: get_texture_type(decoder)?,
        format: get_texture_format(decoder)?,
        width: decoder.u64()?,
        height: decoder.u64()?,
        depth: decoder.u64()?,
        array_length: decoder.u64()?,
        sample_count: decoder.u64()?,
        access: get_texture_access(decoder)?,
        source: get_texture_source(decoder)?,
    })
}

fn put_dispatch(encoder: &mut Encoder, dispatch: &Dispatch) {
    put_dispatch_kind(encoder, dispatch.kind);
    encoder.array3(dispatch.grid);
    encoder.array3(dispatch.threads_per_threadgroup);
}

fn get_dispatch(decoder: &mut Decoder<'_>) -> Result<Dispatch, CodecError> {
    Ok(Dispatch {
        kind: get_dispatch_kind(decoder)?,
        grid: decoder.array3()?,
        threads_per_threadgroup: decoder.array3()?,
    })
}

fn put_dispatch_type(encoder: &mut Encoder, dispatch_type: DispatchType) {
    encoder.u8(match dispatch_type {
        DispatchType::Serial => 0,
        DispatchType::Concurrent => 1,
    });
}

fn get_dispatch_type(decoder: &mut Decoder<'_>) -> Result<DispatchType, CodecError> {
    match decoder.u8()? {
        0 => Ok(DispatchType::Serial),
        1 => Ok(DispatchType::Concurrent),
        value => Err(CodecError::UnknownEnumValue {
            field: "dispatch type",
            value,
        }),
    }
}

fn put_completion_policy(encoder: &mut Encoder, policy: CompletionPolicy) {
    encoder.u8(match policy {
        CompletionPolicy::HostReadback => 0,
        CompletionPolicy::SubmitOnly => 1,
    });
}

fn get_completion_policy(decoder: &mut Decoder<'_>) -> Result<CompletionPolicy, CodecError> {
    match decoder.u8()? {
        0 => Ok(CompletionPolicy::HostReadback),
        1 => Ok(CompletionPolicy::SubmitOnly),
        value => Err(CodecError::UnknownEnumValue {
            field: "completion policy",
            value,
        }),
    }
}

/// Encode one trace, choosing the pass layout from its own contents.
///
/// A compute-only trace writes the pre-render pass bytes; a render-bearing
/// trace tags every pass. The caller writes the matching payload tag, so
/// `SUBMIT_REQUEST` frames keep their exact legacy bytes and
/// `SUBMIT_RENDER_REQUEST` frames are self-describing.
fn put_trace(encoder: &mut Encoder, trace: &ComputeTrace) -> Result<(), CodecError> {
    let tagged = trace.has_render_passes();
    if tagged && trace.passes.len() > MAX_TAGGED_TRACE_PASSES {
        return Err(CodecError::TracePassCount {
            count: trace.passes.len(),
            maximum: MAX_TAGGED_TRACE_PASSES,
        });
    }
    encoder.u16(trace.schema_version);
    put_epoch(encoder, trace.device_epoch);
    encoder.u64(trace.operation_id.get());
    encoder.u64(trace.pipelines.len() as u64);
    for pipeline in &trace.pipelines {
        if tagged {
            put_pipeline_tagged(encoder, pipeline);
        } else {
            put_pipeline(encoder, pipeline);
        }
    }
    put_dispatch_type(encoder, trace.encoder_dispatch_type);
    encoder.u64(trace.passes.len() as u64);
    for pass in &trace.passes {
        match pass {
            TracePass::Compute(pass) => {
                if tagged {
                    encoder.u8(PASS_KIND_COMPUTE);
                }
                put_compute_pass(encoder, pass);
            }
            TracePass::Render(pass) => {
                if pass.color_attachments.len() > MAX_COLOR_ATTACHMENTS {
                    return Err(CodecError::ColorAttachmentCount {
                        count: pass.color_attachments.len(),
                        maximum: MAX_COLOR_ATTACHMENTS,
                    });
                }
                // `tagged` is true whenever a render entry exists, so the tag
                // below always belongs to the extended layout.
                encoder.u8(PASS_KIND_RENDER);
                put_render_pass(encoder, pass);
            }
        }
    }
    put_completion_policy(encoder, trace.completion_policy);
    Ok(())
}

fn put_compute_pass(encoder: &mut Encoder, pass: &ComputePass) {
    encoder.u64(pass.pipeline.get());
    encoder.u64(pass.buffers.len() as u64);
    for view in &pass.buffers {
        put_view(encoder, view);
    }
    encoder.u64(pass.textures.len() as u64);
    for texture in &pass.textures {
        put_texture(encoder, texture);
    }
    put_dispatch(encoder, &pass.dispatch);
}

fn put_render_pass(encoder: &mut Encoder, pass: &RenderPassDescriptor) {
    encoder.u64(pass.pipeline.get());
    encoder.u64(pass.color_attachments.len() as u64);
    for attachment in &pass.color_attachments {
        put_render_attachment(encoder, attachment);
    }
    for dimension in pass.viewport {
        encoder.u32(dimension);
    }
    encoder.u32(pass.vertices);
}

fn put_render_attachment(encoder: &mut Encoder, attachment: &RenderAttachment) {
    encoder.u64(attachment.view_id.get());
    encoder.u64(attachment.allocation_id.get());
    put_attachment_format(encoder, attachment.format);
    encoder.u64(attachment.width);
    encoder.u64(attachment.height);
    put_load_op(encoder, attachment.load);
    put_store_op(encoder, attachment.store);
}

fn put_attachment_format(encoder: &mut Encoder, format: AttachmentFormat) {
    encoder.u8(format.code());
}

fn put_load_op(encoder: &mut Encoder, load: LoadOp) {
    match load {
        LoadOp::Clear(color) => {
            encoder.u8(0);
            // Fixed four-byte payload: no length prefix, so the clear value
            // cannot be separated from its tag by a malformed length.
            encoder.bytes.extend_from_slice(&color.bytes);
        }
        LoadOp::Load => encoder.u8(1),
        LoadOp::DontCare => encoder.u8(2),
    }
}

fn put_store_op(encoder: &mut Encoder, store: StoreOp) {
    encoder.u8(match store {
        StoreOp::Store => 0,
        StoreOp::DontCare => 1,
    });
}

fn get_trace(decoder: &mut Decoder<'_>) -> Result<ComputeTrace, CodecError> {
    let schema_version = decoder.u16()?;
    let device_epoch = get_epoch(decoder)?;
    let operation_id = OperationId::new(decoder.u64()?);
    let pipeline_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    let mut pipelines = Vec::with_capacity(pipeline_count.min(1024));
    for _ in 0..pipeline_count {
        pipelines.push(get_pipeline(decoder)?);
    }
    let encoder_dispatch_type = get_dispatch_type(decoder)?;
    let pass_count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
        needed: usize::MAX,
        remaining: decoder.remaining(),
    })?;
    let mut passes = Vec::with_capacity(pass_count.min(1024));
    for _ in 0..pass_count {
        passes.push(TracePass::Compute(get_compute_pass(decoder)?));
    }
    Ok(ComputeTrace {
        schema_version,
        device_epoch,
        operation_id,
        pipelines,
        encoder_dispatch_type,
        passes,
        completion_policy: get_completion_policy(decoder)?,
    })
}

/// Decode a trace whose pass list is tagged. Only a `SUBMIT_RENDER_REQUEST`
/// frame uses this layout; a `SUBMIT_REQUEST` frame keeps decoding through
/// [`get_trace`] and stays compute-only.
fn get_trace_tagged(decoder: &mut Decoder<'_>) -> Result<ComputeTrace, CodecError> {
    let schema_version = decoder.u16()?;
    let device_epoch = get_epoch(decoder)?;
    let operation_id = OperationId::new(decoder.u64()?);
    let pipeline_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    let mut pipelines = Vec::with_capacity(pipeline_count.min(1024));
    for _ in 0..pipeline_count {
        pipelines.push(get_pipeline_tagged(decoder)?);
    }
    let encoder_dispatch_type = get_dispatch_type(decoder)?;
    let pass_count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
        needed: usize::MAX,
        remaining: decoder.remaining(),
    })?;
    if pass_count > MAX_TAGGED_TRACE_PASSES {
        return Err(CodecError::TracePassCount {
            count: pass_count,
            maximum: MAX_TAGGED_TRACE_PASSES,
        });
    }
    let mut passes = Vec::with_capacity(pass_count.min(1024));
    for _ in 0..pass_count {
        passes.push(match decoder.u8()? {
            PASS_KIND_COMPUTE => TracePass::Compute(get_compute_pass(decoder)?),
            PASS_KIND_RENDER => TracePass::Render(get_render_pass(decoder)?),
            tag => return Err(CodecError::UnknownPassTag(tag)),
        });
    }
    Ok(ComputeTrace {
        schema_version,
        device_epoch,
        operation_id,
        pipelines,
        encoder_dispatch_type,
        passes,
        completion_policy: get_completion_policy(decoder)?,
    })
}

fn get_compute_pass(decoder: &mut Decoder<'_>) -> Result<ComputePass, CodecError> {
    let pipeline = PipelineId::new(decoder.u64()?);
    let view_count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
        needed: usize::MAX,
        remaining: decoder.remaining(),
    })?;
    let mut buffers = Vec::with_capacity(view_count.min(1024));
    for _ in 0..view_count {
        buffers.push(get_view(decoder)?);
    }
    let texture_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    let mut textures = Vec::with_capacity(texture_count.min(1024));
    for _ in 0..texture_count {
        textures.push(get_texture(decoder)?);
    }
    Ok(ComputePass {
        pipeline,
        buffers,
        dispatch: get_dispatch(decoder)?,
        textures,
    })
}

fn get_render_pass(decoder: &mut Decoder<'_>) -> Result<RenderPassDescriptor, CodecError> {
    let pipeline = PipelineId::new(decoder.u64()?);
    let attachment_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    if attachment_count > MAX_COLOR_ATTACHMENTS {
        return Err(CodecError::ColorAttachmentCount {
            count: attachment_count,
            maximum: MAX_COLOR_ATTACHMENTS,
        });
    }
    let mut color_attachments = Vec::with_capacity(attachment_count);
    for _ in 0..attachment_count {
        color_attachments.push(get_render_attachment(decoder)?);
    }
    Ok(RenderPassDescriptor {
        pipeline,
        color_attachments,
        viewport: [
            decoder.u32()?,
            decoder.u32()?,
            decoder.u32()?,
            decoder.u32()?,
        ],
        vertices: decoder.u32()?,
    })
}

fn get_render_attachment(decoder: &mut Decoder<'_>) -> Result<RenderAttachment, CodecError> {
    let view_id = ViewId::new(decoder.u64()?);
    let allocation_id = AllocationId::new(decoder.u64()?);
    let format = get_attachment_format(decoder)?;
    let width = decoder.u64()?;
    let height = decoder.u64()?;
    let load = get_load_op(decoder)?;
    let store = get_store_op(decoder)?;
    Ok(RenderAttachment {
        view_id,
        allocation_id,
        format,
        width,
        height,
        load,
        store,
    })
}

fn get_attachment_format(decoder: &mut Decoder<'_>) -> Result<AttachmentFormat, CodecError> {
    let code = decoder.u8()?;
    AttachmentFormat::from_code(code).ok_or(CodecError::UnknownEnumValue {
        field: "attachment format",
        value: code,
    })
}

fn get_load_op(decoder: &mut Decoder<'_>) -> Result<LoadOp, CodecError> {
    match decoder.u8()? {
        0 => {
            let mut bytes = [0_u8; ClearColor::BYTES];
            bytes.copy_from_slice(decoder.take(ClearColor::BYTES)?);
            Ok(LoadOp::Clear(ClearColor::new(bytes)))
        }
        1 => Ok(LoadOp::Load),
        2 => Ok(LoadOp::DontCare),
        value => Err(CodecError::UnknownEnumValue {
            field: "attachment load op",
            value,
        }),
    }
}

fn get_store_op(decoder: &mut Decoder<'_>) -> Result<StoreOp, CodecError> {
    match decoder.u8()? {
        0 => Ok(StoreOp::Store),
        1 => Ok(StoreOp::DontCare),
        value => Err(CodecError::UnknownEnumValue {
            field: "attachment store op",
            value,
        }),
    }
}

fn put_resources(encoder: &mut Encoder, resources: &ResourceTableSnapshot) {
    let allocations: Vec<_> = resources.allocations().collect();
    encoder.u64(allocations.len() as u64);
    for allocation in &allocations {
        encoder.u64(allocation.allocation_id.get());
        put_epoch(encoder, allocation.owner_epoch);
        encoder.u64(allocation.size);
    }
    let leases: Vec<_> = resources.leases().collect();
    encoder.u64(leases.len() as u64);
    for reservation in &leases {
        put_reservation(encoder, reservation);
    }
}

fn get_resources(decoder: &mut Decoder<'_>) -> Result<ResourceTableSnapshot, CodecError> {
    let mut resources = ResourceTableSnapshot::new();
    let allocation_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    for _ in 0..allocation_count {
        resources.insert_allocation(AllocationRecord {
            allocation_id: AllocationId::new(decoder.u64()?),
            owner_epoch: get_epoch(decoder)?,
            size: decoder.u64()?,
        })?;
    }
    let lease_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    for _ in 0..lease_count {
        resources.insert_lease(get_reservation(decoder)?)?;
    }
    Ok(resources)
}

fn put_reservation(encoder: &mut Encoder, reservation: &LeaseReservation) {
    encoder.u64(reservation.lease.lease_id.get());
    encoder.u64(reservation.lease.allocation_id.get());
    put_epoch(encoder, reservation.lease.owner_epoch);
    encoder.u64(reservation.offset);
    encoder.u64(reservation.length);
}

fn get_reservation(decoder: &mut Decoder<'_>) -> Result<LeaseReservation, CodecError> {
    Ok(LeaseReservation {
        lease: BufferLease {
            lease_id: LeaseId::new(decoder.u64()?),
            allocation_id: AllocationId::new(decoder.u64()?),
            owner_epoch: get_epoch(decoder)?,
        },
        offset: decoder.u64()?,
        length: decoder.u64()?,
    })
}

fn put_staged_lease(encoder: &mut Encoder, staged: &StagedLease) {
    put_reservation(encoder, &staged.reservation);
    encoder.blob(&staged.bytes);
}

fn get_staged_lease(decoder: &mut Decoder<'_>) -> Result<StagedLease, CodecError> {
    Ok(StagedLease::new(
        get_reservation(decoder)?,
        decoder.blob()?,
    )?)
}

fn put_health(encoder: &mut Encoder, health: ProviderHealth) {
    encoder.u8(match health {
        ProviderHealth::Usable => 0,
        ProviderHealth::DeviceLost => 1,
        ProviderHealth::Exhausted => 2,
    });
}

fn get_health(decoder: &mut Decoder<'_>) -> Result<ProviderHealth, CodecError> {
    match decoder.u8()? {
        0 => Ok(ProviderHealth::Usable),
        1 => Ok(ProviderHealth::DeviceLost),
        2 => Ok(ProviderHealth::Exhausted),
        value => Err(CodecError::UnknownEnumValue {
            field: "provider health",
            value,
        }),
    }
}

fn put_writeback(encoder: &mut Encoder, writeback: &BufferWriteback) {
    encoder.u64(writeback.view_id.get());
    encoder.u64(writeback.allocation_id.get());
    encoder.u64(writeback.offset);
    encoder.blob(&writeback.bytes);
}

fn get_writeback(decoder: &mut Decoder<'_>) -> Result<BufferWriteback, CodecError> {
    Ok(BufferWriteback {
        view_id: ViewId::new(decoder.u64()?),
        allocation_id: AllocationId::new(decoder.u64()?),
        offset: decoder.u64()?,
        bytes: decoder.blob()?,
    })
}

fn put_disposition(encoder: &mut Encoder, disposition: CompletionDisposition) {
    match disposition {
        CompletionDisposition::NotSubmitted => encoder.u8(0),
        CompletionDisposition::Submitted { token } => {
            encoder.u8(1);
            put_token(encoder, &token);
        }
        CompletionDisposition::CompletedVisible { token } => {
            encoder.u8(2);
            put_token(encoder, &token);
        }
        CompletionDisposition::Cancelled { token } => {
            encoder.u8(3);
            put_token(encoder, &token);
        }
        CompletionDisposition::TimedOut { token } => {
            encoder.u8(4);
            put_token(encoder, &token);
        }
        CompletionDisposition::Failed { token } => {
            encoder.u8(5);
            put_optional_token(encoder, token);
        }
        CompletionDisposition::DeviceLost { token } => {
            encoder.u8(6);
            put_optional_token(encoder, token);
        }
        CompletionDisposition::SubmittedUnknown { token } => {
            encoder.u8(7);
            put_optional_token(encoder, token);
        }
    }
}

fn put_optional_token(encoder: &mut Encoder, token: Option<CompletionToken>) {
    match token {
        Some(token) => {
            encoder.u8(1);
            put_token(encoder, &token);
        }
        None => encoder.u8(0),
    }
}

fn get_optional_token(decoder: &mut Decoder<'_>) -> Result<Option<CompletionToken>, CodecError> {
    match decoder.u8()? {
        0 => Ok(None),
        1 => Ok(Some(get_token(decoder)?)),
        value => Err(CodecError::UnknownEnumValue {
            field: "optional completion token",
            value,
        }),
    }
}

fn get_disposition(decoder: &mut Decoder<'_>) -> Result<CompletionDisposition, CodecError> {
    Ok(match decoder.u8()? {
        0 => CompletionDisposition::NotSubmitted,
        1 => CompletionDisposition::Submitted {
            token: get_token(decoder)?,
        },
        2 => CompletionDisposition::CompletedVisible {
            token: get_token(decoder)?,
        },
        3 => CompletionDisposition::Cancelled {
            token: get_token(decoder)?,
        },
        4 => CompletionDisposition::TimedOut {
            token: get_token(decoder)?,
        },
        5 => CompletionDisposition::Failed {
            token: get_optional_token(decoder)?,
        },
        6 => CompletionDisposition::DeviceLost {
            token: get_optional_token(decoder)?,
        },
        7 => CompletionDisposition::SubmittedUnknown {
            token: get_optional_token(decoder)?,
        },
        value => {
            return Err(CodecError::UnknownEnumValue {
                field: "completion disposition",
                value,
            })
        }
    })
}

fn put_submission(encoder: &mut Encoder, submission: &ProviderSubmission) {
    put_disposition(encoder, submission.completion);
    encoder.u64(submission.writebacks.len() as u64);
    for writeback in &submission.writebacks {
        put_writeback(encoder, writeback);
    }
}

fn get_submission(decoder: &mut Decoder<'_>) -> Result<ProviderSubmission, CodecError> {
    let completion = get_disposition(decoder)?;
    let count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
        needed: usize::MAX,
        remaining: decoder.remaining(),
    })?;
    let mut writebacks = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        writebacks.push(get_writeback(decoder)?);
    }
    Ok(ProviderSubmission {
        completion,
        writebacks,
    })
}

fn put_readback(encoder: &mut Encoder, readback: &CompletionReadback) {
    put_disposition(encoder, readback.completion);
    encoder.u64(readback.writebacks.len() as u64);
    for writeback in &readback.writebacks {
        put_writeback(encoder, writeback);
    }
}

fn get_readback(decoder: &mut Decoder<'_>) -> Result<CompletionReadback, CodecError> {
    let completion = get_disposition(decoder)?;
    let count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
        needed: usize::MAX,
        remaining: decoder.remaining(),
    })?;
    let mut writebacks = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        writebacks.push(get_writeback(decoder)?);
    }
    Ok(CompletionReadback {
        completion,
        writebacks,
    })
}

fn put_phase(encoder: &mut Encoder, phase: ProviderPhase) {
    encoder.u8(match phase {
        ProviderPhase::Resolve => 0,
        ProviderPhase::Compile => 1,
        ProviderPhase::Encode => 2,
        ProviderPhase::Submit => 3,
        ProviderPhase::Wait => 4,
        ProviderPhase::Readback => 5,
    });
}

fn get_phase(decoder: &mut Decoder<'_>) -> Result<ProviderPhase, CodecError> {
    match decoder.u8()? {
        0 => Ok(ProviderPhase::Resolve),
        1 => Ok(ProviderPhase::Compile),
        2 => Ok(ProviderPhase::Encode),
        3 => Ok(ProviderPhase::Submit),
        4 => Ok(ProviderPhase::Wait),
        5 => Ok(ProviderPhase::Readback),
        value => Err(CodecError::UnknownEnumValue {
            field: "provider phase",
            value,
        }),
    }
}

fn put_class(encoder: &mut Encoder, class: ProviderErrorClass) {
    encoder.u8(match class {
        ProviderErrorClass::Args => 0,
        ProviderErrorClass::Capability => 1,
        ProviderErrorClass::Resource => 2,
        ProviderErrorClass::Compile => 3,
        ProviderErrorClass::Execute => 4,
        ProviderErrorClass::DeviceLost => 5,
        ProviderErrorClass::Internal => 6,
    });
}

fn get_class(decoder: &mut Decoder<'_>) -> Result<ProviderErrorClass, CodecError> {
    match decoder.u8()? {
        0 => Ok(ProviderErrorClass::Args),
        1 => Ok(ProviderErrorClass::Capability),
        2 => Ok(ProviderErrorClass::Resource),
        3 => Ok(ProviderErrorClass::Compile),
        4 => Ok(ProviderErrorClass::Execute),
        5 => Ok(ProviderErrorClass::DeviceLost),
        6 => Ok(ProviderErrorClass::Internal),
        value => Err(CodecError::UnknownEnumValue {
            field: "provider error class",
            value,
        }),
    }
}

fn put_retryability(encoder: &mut Encoder, retryability: Retryability) {
    encoder.u8(match retryability {
        Retryability::Never => 0,
        Retryability::RetrySameTrace => 1,
        Retryability::RetryAfterRecreate => 2,
        Retryability::Unknown => 3,
    });
}

fn get_retryability(decoder: &mut Decoder<'_>) -> Result<Retryability, CodecError> {
    match decoder.u8()? {
        0 => Ok(Retryability::Never),
        1 => Ok(Retryability::RetrySameTrace),
        2 => Ok(Retryability::RetryAfterRecreate),
        3 => Ok(Retryability::Unknown),
        value => Err(CodecError::UnknownEnumValue {
            field: "retryability",
            value,
        }),
    }
}

fn put_field_value(encoder: &mut Encoder, value: &FieldValue) {
    match value {
        FieldValue::Unsigned(value) => {
            encoder.u8(0);
            encoder.u64(*value);
        }
        FieldValue::Signed(value) => {
            encoder.u8(1);
            encoder.i64(*value);
        }
        FieldValue::Bool(value) => {
            encoder.u8(2);
            encoder.bool(*value);
        }
        FieldValue::Text(value) => {
            encoder.u8(3);
            encoder.text(value);
        }
    }
}

fn get_field_value(decoder: &mut Decoder<'_>) -> Result<FieldValue, CodecError> {
    match decoder.u8()? {
        0 => Ok(FieldValue::Unsigned(decoder.u64()?)),
        1 => Ok(FieldValue::Signed(decoder.i64()?)),
        2 => Ok(FieldValue::Bool(decoder.bool()?)),
        3 => Ok(FieldValue::Text(decoder.text()?)),
        value => Err(CodecError::UnknownEnumValue {
            field: "provider error field value",
            value,
        }),
    }
}

fn put_error(encoder: &mut Encoder, error: &ProviderError) {
    put_phase(encoder, error.phase);
    put_class(encoder, error.class);
    encoder.text(&error.slug);
    encoder.u64(error.fields.len() as u64);
    for (name, value) in &error.fields {
        encoder.text(name);
        put_field_value(encoder, value);
    }
    put_retryability(encoder, error.retryability);
    put_disposition(encoder, error.completion);
    encoder.opt_text(error.detail.as_deref());
}

fn get_error(decoder: &mut Decoder<'_>) -> Result<ProviderError, CodecError> {
    let phase = get_phase(decoder)?;
    let class = get_class(decoder)?;
    let slug = decoder.text()?;
    let mut error = ProviderError::new(phase, class, slug)?;
    let field_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    for _ in 0..field_count {
        error
            .fields
            .insert(decoder.text()?, get_field_value(decoder)?);
    }
    error.retryability = get_retryability(decoder)?;
    error.completion = get_disposition(decoder)?;
    error.detail = decoder.opt_text()?;
    Ok(error)
}

fn put_alias_mode(encoder: &mut Encoder, mode: AliasMode) {
    encoder.u8(match mode {
        AliasMode::Refused => 0,
        AliasMode::DistinctViews => 1,
        AliasMode::ExplicitPolicy => 2,
    });
}

fn get_alias_mode(decoder: &mut Decoder<'_>) -> Result<AliasMode, CodecError> {
    match decoder.u8()? {
        0 => Ok(AliasMode::Refused),
        1 => Ok(AliasMode::DistinctViews),
        2 => Ok(AliasMode::ExplicitPolicy),
        value => Err(CodecError::UnknownEnumValue {
            field: "alias mode",
            value,
        }),
    }
}

fn put_storage_mode(encoder: &mut Encoder, mode: StorageMode) {
    encoder.u8(match mode {
        StorageMode::OwnedBytes => 0,
        StorageMode::StagedLease => 1,
        StorageMode::BorrowedNoCopy => 2,
    });
}

fn get_storage_mode(decoder: &mut Decoder<'_>) -> Result<StorageMode, CodecError> {
    match decoder.u8()? {
        0 => Ok(StorageMode::OwnedBytes),
        1 => Ok(StorageMode::StagedLease),
        2 => Ok(StorageMode::BorrowedNoCopy),
        value => Err(CodecError::UnknownEnumValue {
            field: "storage mode",
            value,
        }),
    }
}

/// Encode a capability snapshot, including its render bits.
///
/// Only [`RENDER_CAPABILITIES_RESPONSE`] frames use this layout. A snapshot
/// whose render bits are all at their defaults is written by
/// [`put_capabilities_legacy`] under the legacy tag, which is what keeps both
/// current providers byte-identical on the wire.
fn put_capabilities(
    encoder: &mut Encoder,
    capabilities: &ProviderCapabilities,
) -> Result<(), CodecError> {
    put_capabilities_legacy(encoder, capabilities);
    if capabilities.supported_color_formats.len() > MAX_SUPPORTED_COLOR_FORMATS {
        return Err(CodecError::SupportedColorFormatCount {
            count: capabilities.supported_color_formats.len(),
            maximum: MAX_SUPPORTED_COLOR_FORMATS,
        });
    }
    encoder.bool(capabilities.supports_render_passes);
    encoder.u32(capabilities.max_color_attachments);
    encoder.u64(capabilities.max_attachment_dimension[0]);
    encoder.u64(capabilities.max_attachment_dimension[1]);
    encoder.u64(capabilities.supported_color_formats.len() as u64);
    for format in &capabilities.supported_color_formats {
        put_attachment_format(encoder, *format);
    }
    Ok(())
}

fn put_capabilities_legacy(encoder: &mut Encoder, capabilities: &ProviderCapabilities) {
    encoder.u32(capabilities.max_passes);
    encoder.bool(capabilities.supports_threads_exact);
    encoder.bool(capabilities.supports_threadgroups);
    encoder.bool(capabilities.supports_serial);
    encoder.bool(capabilities.supports_concurrent);
    encoder.array3(capabilities.max_local_size);
    encoder.u64(capabilities.max_invocations);
    encoder.array3(capabilities.max_group_count);
    encoder.u32(capabilities.max_storage_buffer_descriptors);
    encoder.u64(capabilities.max_buffer_range);
    encoder.u32(capabilities.max_push_constant_bytes);
    put_alias_mode(encoder, capabilities.alias_mode);
    encoder.u64(capabilities.storage_modes.len() as u64);
    for mode in &capabilities.storage_modes {
        put_storage_mode(encoder, *mode);
    }
    encoder.bool(capabilities.host_readback);
    encoder.bool(capabilities.submit_only);
}

/// Decode the pre-render capability payload and fill the render bits with the
/// defaults a provider that never declared them must be treated as having.
fn get_capabilities_legacy(decoder: &mut Decoder<'_>) -> Result<ProviderCapabilities, CodecError> {
    let max_passes = decoder.u32()?;
    let supports_threads_exact = decoder.bool()?;
    let supports_threadgroups = decoder.bool()?;
    let supports_serial = decoder.bool()?;
    let supports_concurrent = decoder.bool()?;
    let max_local_size = decoder.array3()?;
    let max_invocations = decoder.u64()?;
    let max_group_count = decoder.array3()?;
    let max_storage_buffer_descriptors = decoder.u32()?;
    let max_buffer_range = decoder.u64()?;
    let max_push_constant_bytes = decoder.u32()?;
    let alias_mode = get_alias_mode(decoder)?;
    let mode_count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
        needed: usize::MAX,
        remaining: decoder.remaining(),
    })?;
    let mut storage_modes = Vec::with_capacity(mode_count.min(1024));
    for _ in 0..mode_count {
        storage_modes.push(get_storage_mode(decoder)?);
    }
    Ok(ProviderCapabilities {
        max_passes,
        supports_threads_exact,
        supports_threadgroups,
        supports_serial,
        supports_concurrent,
        max_local_size,
        max_invocations,
        max_group_count,
        max_storage_buffer_descriptors,
        max_buffer_range,
        max_push_constant_bytes,
        alias_mode,
        storage_modes,
        host_readback: decoder.bool()?,
        submit_only: decoder.bool()?,
        supports_render_passes: false,
        max_color_attachments: 0,
        max_attachment_dimension: [0, 0],
        supported_color_formats: Vec::new(),
    })
}

fn get_capabilities(decoder: &mut Decoder<'_>) -> Result<ProviderCapabilities, CodecError> {
    let mut capabilities = get_capabilities_legacy(decoder)?;
    capabilities.supports_render_passes = decoder.bool()?;
    capabilities.max_color_attachments = decoder.u32()?;
    capabilities.max_attachment_dimension = [decoder.u64()?, decoder.u64()?];
    let format_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    if format_count > MAX_SUPPORTED_COLOR_FORMATS {
        return Err(CodecError::SupportedColorFormatCount {
            count: format_count,
            maximum: MAX_SUPPORTED_COLOR_FORMATS,
        });
    }
    let mut supported_color_formats = Vec::with_capacity(format_count);
    for _ in 0..format_count {
        supported_color_formats.push(get_attachment_format(decoder)?);
    }
    capabilities.supported_color_formats = supported_color_formats;
    Ok(capabilities)
}
