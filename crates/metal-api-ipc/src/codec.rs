//! Versioned binary codec for the neutral completion stream.
//!
//! The codec is deliberately small and dependency-free. It encodes exactly the
//! `metal-api-core` completion value family with a fixed magic/version so a
//! provider and an owner built from the same crate revision can exchange
//! notifications over a byte stream.
//!
//! Frame layout (all integers are big-endian):
//!
//! ```text
//! magic   4 bytes   b"MCW1"
//! length  4 bytes   payload length
//! payload ...
//! ```
//!
//! A frame is rejected before allocation when its declared length exceeds
//! [`MAX_FRAME_PAYLOAD`]. Enum values and tags are checked on decode, and the
//! decoded message must pass `CompletionMessage::validate`.

use metal_api_core::completion::wire::{
    CompletionDeviceUpdate, CompletionFailure, CompletionMessage, CompletionSequence,
    CompletionTokenUpdate, CompletionUpdate,
};
use metal_api_core::provider::{
    CompletionToken, ContractError, DeviceEpoch, ProviderErrorClass, ProviderHealth, ProviderPhase,
    Retryability, SubmissionId,
};
use std::fmt;
use std::io::{self, Read, Write};

/// Protocol magic and version. A different byte sequence is refused.
pub const FRAME_MAGIC: [u8; 4] = *b"MCW1";
/// Maximum encoded payload, excluding the eight-byte frame header.
pub const MAX_FRAME_PAYLOAD: usize = 1024 * 1024;
/// Maximum UTF-8 length of a `CompletionFailure` slug.
pub const MAX_FAILURE_SLUG: usize = 4096;

const TOKEN_MESSAGE_TAG: u8 = 0x01;
const DEVICE_MESSAGE_TAG: u8 = 0x02;

const SUBMITTED_TAG: u8 = 0x00;
const COMPLETED_VISIBLE_TAG: u8 = 0x01;
const CANCELLED_TAG: u8 = 0x02;
const FAILED_TAG: u8 = 0x03;
const SUBMITTED_UNKNOWN_TAG: u8 = 0x04;

/// Encoding or decoding failure.
///
/// I/O errors remain distinct from protocol errors so a transport can decide
/// whether to retry a connection. A clean end of stream is [`CodecError::Eof`];
/// a partial frame is [`CodecError::TruncatedFrame`].
#[derive(Debug)]
pub enum CodecError {
    Io(io::Error),
    Eof,
    BadMagic([u8; 4]),
    FrameTooLarge {
        length: usize,
        maximum: usize,
    },
    FailureSlugTooLong {
        length: usize,
        maximum: usize,
    },
    TruncatedFrame {
        expected: usize,
        actual: usize,
    },
    TruncatedPayload {
        needed: usize,
        remaining: usize,
    },
    TrailingPayload {
        extra: usize,
    },
    UnknownMessageTag(u8),
    UnknownUpdateTag(u8),
    UnknownCommandTag(u8),
    UnknownFrameKind(u8),
    ChunkSequence {
        expected: u64,
        actual: u64,
    },
    ChunkTotalMismatch {
        declared: u64,
        actual: u64,
    },
    ChunkInterrupted {
        received: u64,
        declared: u64,
    },
    ChunkedPayloadTooLarge {
        total: u64,
        maximum: usize,
    },
    QueuePriorityCount {
        count: usize,
        maximum: usize,
    },
    TracePassCount {
        count: usize,
        maximum: usize,
    },
    ColorAttachmentCount {
        count: usize,
        maximum: usize,
    },
    RenderPipelineFormatCount {
        count: usize,
        maximum: usize,
    },
    VertexBufferCount {
        count: usize,
        maximum: usize,
    },
    VertexAttributeCount {
        count: usize,
        maximum: usize,
    },
    SupportedColorFormatCount {
        count: usize,
        maximum: usize,
    },
    SupportedVertexFormatCount {
        count: usize,
        maximum: usize,
    },
    SupportedIndexFormatCount {
        count: usize,
        maximum: usize,
    },
    PresentModeCount {
        count: usize,
        maximum: usize,
    },
    PresentSentinelLength {
        length: usize,
        maximum: usize,
    },
    /// A render attachment's clear payload is not one texel of its own format
    /// (`research/docs/23` §78).
    ///
    /// The width is carried by the attachment's format rather than by a length
    /// prefix, so a payload of another width cannot be framed: the sender
    /// refuses it here instead of writing a frame the receiver would have to
    /// desync on.
    ClearLength {
        format: u8,
        expected: usize,
        actual: usize,
    },
    HeapPlacementCount {
        count: usize,
        maximum: usize,
    },
    HeapStorageModeCount {
        count: usize,
        maximum: usize,
    },
    IndirectCommandKindCount {
        count: usize,
        maximum: usize,
    },
    UnknownPassTag(u8),
    /// A feature bit an extended render pass set that this decoder does not
    /// know. The word is 16 bits wide since v43, when the narrow byte ran out
    /// of free bits; the low byte keeps the meanings it always had.
    UnknownRenderFeature(u16),
    /// A sampled-texture render pass declared more textures than the contract's
    /// own cap (`research/docs/23` §3.3, v70). The block's count is one byte,
    /// so this is the protocol's bound; the contract refuses anything above
    /// [`metal_api_core::provider::MAX_RENDER_TEXTURES`] before a frame is
    /// written.
    RenderTextureCount {
        count: usize,
        maximum: usize,
    },
    /// A capability snapshot declared more render-pass sampling formats than
    /// the protocol bound (`research/docs/23` §3.3, v70).
    RenderTextureFormatCount {
        count: usize,
        maximum: usize,
    },
    /// A runtime sampler list carried more entries than the contract's own cap
    /// (`research/docs/23` §3.3, v102). The block's count is one byte, so this
    /// is the protocol's bound; the contract refuses anything above
    /// [`metal_api_core::provider::MAX_RENDER_SAMPLERS`] before a frame is
    /// written.
    RenderSamplerCount {
        count: usize,
        maximum: usize,
    },
    /// A render texture declaration list carried more entries than the
    /// contract's own cap (`research/docs/23` §3.3, v100/v102). The block's
    /// count is one byte, so this is the protocol's bound; the contract
    /// refuses anything above
    /// [`metal_api_core::provider::MAX_RENDER_TEXTURES`] before a frame is
    /// written.
    RenderTextureDeclarationCount {
        count: usize,
        maximum: usize,
    },
    /// A render texture declaration stated both sampler forms at once
    /// (`research/docs/23` §3.3, v102).
    ///
    /// The two forms are exclusive: `sampler` is the state the module's own
    /// AIR constexpr sampler carries and `runtime_sampler` names the
    /// `[[sampler(n)]]` argument the pass states. The declaration block's form
    /// byte has no position that could mean both, so the sender refuses the
    /// pair by name instead of framing one of the two states.
    RenderTextureSamplerFormUnsupported {
        binding: u32,
    },
    /// A stage buffer list carried more entries than the contract's own cap
    /// (`research/docs/23` §3.3, v83). The block's count is one byte, so this
    /// is the protocol's bound; the contract refuses anything above
    /// [`metal_api_core::provider::MAX_RENDER_STAGE_BUFFERS`] before a frame
    /// is written.
    RenderStageBufferCount {
        count: usize,
        maximum: usize,
    },
    /// A compute texture declaration block carried more entries than the
    /// shape's own cap (`research/docs/23` §91). The block's count is one
    /// byte, so this is the protocol's bound; the codec refuses anything above
    /// [`metal_api_core::provider::MAX_COMPUTE_TEXTURES`] before a frame is
    /// written.
    ComputeTextureCount {
        count: usize,
        maximum: usize,
    },
    /// A capability snapshot declared more compute-side sampling formats than
    /// the protocol bound (`research/docs/23` §91).
    ComputeTextureFormatCount {
        count: usize,
        maximum: usize,
    },
    /// A pipeline-table entry carried a render half *and* a compute contract
    /// that declares texture bindings (`research/docs/23` §91).
    ///
    /// The render kinds' bodies are fixed — a decoder reads the render half
    /// exactly where the pre-v87 encoder wrote it — so there is no position
    /// the declaration block could take without a new kind per combination.
    /// The sender refuses the combination by name instead of writing a frame
    /// the receiver would have to desync on.
    ComputeTextureDeclarationsWithRenderHalf {
        declarations: usize,
    },
    /// A render pass stated blend state its v40 section cannot carry
    /// (`research/docs/23` §3.3, v100).
    ///
    /// The section carries one operation per entry, every channel written and
    /// blending enabled — the shape v40 published. A pass that states anything
    /// else (a disabled entry, a second operation for the alpha pair, a write
    /// mask) has no position in those bytes, so the sender refuses it by name
    /// instead of framing a state the receiver would read as blending with one
    /// operation and every channel written.
    RenderBlendStateUnsupported {
        location: usize,
        field: &'static str,
    },
    /// A wide feature that describes the depth attachment arrived without one.
    /// The store action and the identity are properties *of* the depth
    /// attachment, so either bit without the depth section names a surface the
    /// pass never opens (`research/docs/23` §3.3, v43).
    DepthFeatureWithoutAttachment(u16),
    UnknownCapabilityTail(u8),
    UnknownPipelineTag(u8),
    UnknownEnumValue {
        field: &'static str,
        value: u8,
    },
    InvalidUtf8(std::string::FromUtf8Error),
    Contract(ContractError),
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "completion codec I/O error: {error}"),
            Self::Eof => formatter.write_str("completion codec reached end of stream"),
            Self::BadMagic(magic) => write!(formatter, "completion codec bad magic {magic:?}"),
            Self::FrameTooLarge { length, maximum } => write!(
                formatter,
                "completion frame length {length} exceeds maximum {maximum}"
            ),
            Self::FailureSlugTooLong { length, maximum } => write!(
                formatter,
                "completion failure slug length {length} exceeds maximum {maximum}"
            ),
            Self::TruncatedFrame { expected, actual } => write!(
                formatter,
                "completion frame truncated: expected {expected} bytes, received {actual}"
            ),
            Self::TruncatedPayload { needed, remaining } => write!(
                formatter,
                "completion payload truncated: needed {needed} bytes, {remaining} remaining"
            ),
            Self::TrailingPayload { extra } => {
                write!(formatter, "completion payload has {extra} trailing bytes")
            }
            Self::UnknownMessageTag(tag) => {
                write!(formatter, "unknown completion message tag {tag:#04x}")
            }
            Self::UnknownUpdateTag(tag) => {
                write!(formatter, "unknown completion update tag {tag:#04x}")
            }
            Self::UnknownCommandTag(tag) => {
                write!(formatter, "unknown command frame tag {tag:#04x}")
            }
            Self::UnknownFrameKind(kind) => {
                write!(formatter, "unknown command frame kind {kind:#04x}")
            }
            Self::ChunkSequence { expected, actual } => write!(
                formatter,
                "command chunk sequence expected offset {expected}, received {actual}"
            ),
            Self::ChunkTotalMismatch { declared, actual } => write!(
                formatter,
                "command chunk total mismatch: declared {declared}, received {actual}"
            ),
            Self::ChunkInterrupted { received, declared } => write!(
                formatter,
                "command chunk transfer interrupted after {received} of {declared} bytes"
            ),
            Self::ChunkedPayloadTooLarge { total, maximum } => write!(
                formatter,
                "chunked command payload length {total} exceeds maximum {maximum}"
            ),
            Self::QueuePriorityCount { count, maximum } => write!(
                formatter,
                "queue priority table carries {count} tiers, maximum {maximum}"
            ),
            Self::TracePassCount { count, maximum } => write!(
                formatter,
                "tagged trace carries {count} passes, maximum {maximum}"
            ),
            Self::ColorAttachmentCount { count, maximum } => write!(
                formatter,
                "render pass carries {count} colour attachments, maximum {maximum}"
            ),
            Self::RenderPipelineFormatCount { count, maximum } => write!(
                formatter,
                "render pipeline contract carries {count} colour formats, maximum {maximum}"
            ),
            Self::VertexBufferCount { count, maximum } => write!(
                formatter,
                "render pass carries {count} vertex buffers, maximum {maximum}"
            ),
            Self::VertexAttributeCount { count, maximum } => write!(
                formatter,
                "vertex buffer layout carries {count} attributes, maximum {maximum}"
            ),
            Self::SupportedColorFormatCount { count, maximum } => write!(
                formatter,
                "capability snapshot names {count} colour formats, maximum {maximum}"
            ),
            Self::SupportedVertexFormatCount { count, maximum } => write!(
                formatter,
                "capability snapshot names {count} vertex formats, maximum {maximum}"
            ),
            Self::SupportedIndexFormatCount { count, maximum } => write!(
                formatter,
                "capability snapshot names {count} index formats, maximum {maximum}"
            ),
            Self::PresentSentinelLength { length, maximum } => write!(
                formatter,
                "present target sentinel carries {length} bytes, maximum {maximum}"
            ),
            Self::ClearLength {
                format,
                expected,
                actual,
            } => write!(
                formatter,
                "attachment clear for format code {format} carries {actual} bytes, but its texel \
                 is {expected} bytes"
            ),
            Self::HeapPlacementCount { count, maximum } => write!(
                formatter,
                "heap payload carries {count} placements, maximum {maximum}"
            ),
            Self::HeapStorageModeCount { count, maximum } => write!(
                formatter,
                "capability snapshot names {count} heap storage modes, maximum {maximum}"
            ),
            Self::IndirectCommandKindCount { count, maximum } => write!(
                formatter,
                "indirect command payload names {count} command kinds, maximum {maximum}"
            ),
            Self::PresentModeCount { count, maximum } => write!(
                formatter,
                "capability snapshot names {count} present modes, maximum {maximum}"
            ),
            Self::UnknownPassTag(tag) => {
                write!(formatter, "unknown trace pass tag {tag:#04x}")
            }
            Self::UnknownRenderFeature(features) => write!(
                formatter,
                "unknown extended render pass feature bits {features:#06x}"
            ),
            Self::RenderTextureCount { count, maximum } => write!(
                formatter,
                "render pass binds {count} sampled textures, maximum {maximum}"
            ),
            Self::RenderTextureFormatCount { count, maximum } => write!(
                formatter,
                "capability snapshot names {count} render texture formats, maximum {maximum}"
            ),
            Self::RenderSamplerCount { count, maximum } => write!(
                formatter,
                "runtime sampler list carries {count} bindings, maximum {maximum}"
            ),
            Self::RenderTextureDeclarationCount { count, maximum } => write!(
                formatter,
                "render texture declaration block carries {count} bindings, maximum {maximum}"
            ),
            Self::RenderTextureSamplerFormUnsupported { binding } => write!(
                formatter,
                "render texture declaration {binding} states a sampler state and a runtime \
                 sampler index, but the block carries exactly one of the two forms"
            ),
            Self::RenderStageBufferCount { count, maximum } => write!(
                formatter,
                "stage buffer list carries {count} bindings, maximum {maximum}"
            ),
            Self::ComputeTextureCount { count, maximum } => write!(
                formatter,
                "compute texture declaration block carries {count} bindings, maximum {maximum}"
            ),
            Self::ComputeTextureFormatCount { count, maximum } => write!(
                formatter,
                "capability snapshot names {count} compute texture formats, maximum {maximum}"
            ),
            Self::ComputeTextureDeclarationsWithRenderHalf { declarations } => write!(
                formatter,
                "a render pipeline entry carries {declarations} compute texture declarations, \
                 a block the render kinds cannot frame"
            ),
            Self::RenderBlendStateUnsupported { location, field } => write!(
                formatter,
                "colour attachment {location}'s blend state states {field}, which the v40 blend \
                 section cannot carry"
            ),
            Self::DepthFeatureWithoutAttachment(features) => write!(
                formatter,
                "extended render pass feature bits {features:#06x} describe a depth attachment \
                 the pass does not open"
            ),
            Self::UnknownCapabilityTail(tag) => {
                write!(formatter, "unknown capability payload tail tag {tag:#04x}")
            }
            Self::UnknownPipelineTag(tag) => {
                write!(formatter, "unknown pipeline table entry tag {tag:#04x}")
            }
            Self::UnknownEnumValue { field, value } => {
                write!(formatter, "unknown {field} value {value}")
            }
            Self::InvalidUtf8(error) => {
                write!(formatter, "completion failure slug is not UTF-8: {error}")
            }
            Self::Contract(error) => {
                write!(
                    formatter,
                    "completion message is structurally invalid: {error}"
                )
            }
        }
    }
}

impl std::error::Error for CodecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::InvalidUtf8(error) => Some(error),
            Self::Contract(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for CodecError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ContractError> for CodecError {
    fn from(error: ContractError) -> Self {
        Self::Contract(error)
    }
}

/// Stateless encoder/decoder for [`CompletionMessage`].
pub struct CompletionCodec;

impl CompletionCodec {
    /// Encode one complete frame.
    pub fn encode(message: &CompletionMessage) -> Result<Vec<u8>, CodecError> {
        let payload = encode_payload(message)?;
        if payload.len() > MAX_FRAME_PAYLOAD {
            return Err(CodecError::FrameTooLarge {
                length: payload.len(),
                maximum: MAX_FRAME_PAYLOAD,
            });
        }
        let mut frame = Vec::with_capacity(8 + payload.len());
        frame.extend_from_slice(&FRAME_MAGIC);
        frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(&payload);
        Ok(frame)
    }

    /// Decode one complete frame.
    pub fn decode(frame: &[u8]) -> Result<CompletionMessage, CodecError> {
        if frame.len() < 8 {
            return Err(CodecError::TruncatedFrame {
                expected: 8,
                actual: frame.len(),
            });
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&frame[..4]);
        if magic != FRAME_MAGIC {
            return Err(CodecError::BadMagic(magic));
        }
        let length = u32::from_be_bytes([frame[4], frame[5], frame[6], frame[7]]) as usize;
        if length > MAX_FRAME_PAYLOAD {
            return Err(CodecError::FrameTooLarge {
                length,
                maximum: MAX_FRAME_PAYLOAD,
            });
        }
        if frame.len() != 8 + length {
            return Err(CodecError::TruncatedFrame {
                expected: 8 + length,
                actual: frame.len(),
            });
        }
        decode_payload(&frame[8..])
    }

    /// Write one frame to a stream.
    pub fn write<W: Write>(writer: &mut W, message: &CompletionMessage) -> Result<(), CodecError> {
        let frame = Self::encode(message)?;
        writer.write_all(&frame)?;
        Ok(())
    }

    /// Read one frame from a stream. A clean end of stream before the first
    /// byte is [`CodecError::Eof`]; a partial frame is a truncation error.
    pub fn read<R: Read>(reader: &mut R) -> Result<CompletionMessage, CodecError> {
        let mut header = [0u8; 8];
        read_header(reader, &mut header)?;
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&header[..4]);
        if magic != FRAME_MAGIC {
            return Err(CodecError::BadMagic(magic));
        }
        let length = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
        if length > MAX_FRAME_PAYLOAD {
            return Err(CodecError::FrameTooLarge {
                length,
                maximum: MAX_FRAME_PAYLOAD,
            });
        }
        let mut payload = vec![0u8; length];
        read_payload(reader, &mut payload)?;
        decode_payload(&payload)
    }
}

fn encode_payload(message: &CompletionMessage) -> Result<Vec<u8>, CodecError> {
    let mut payload = Vec::new();
    match message {
        CompletionMessage::Token(update) => {
            payload.push(TOKEN_MESSAGE_TAG);
            put_u64(&mut payload, update.token.device_epoch.get());
            put_u64(&mut payload, update.token.submission_id.get());
            put_u64(&mut payload, update.sequence.get());
            encode_update(&mut payload, &update.update)?;
        }
        CompletionMessage::Device(update) => {
            payload.push(DEVICE_MESSAGE_TAG);
            put_u64(&mut payload, update.device_epoch.get());
            put_u64(&mut payload, update.sequence.get());
            payload.push(encode_health(update.health));
        }
    }
    Ok(payload)
}

fn encode_update(payload: &mut Vec<u8>, update: &CompletionUpdate) -> Result<(), CodecError> {
    match update {
        CompletionUpdate::Submitted => payload.push(SUBMITTED_TAG),
        CompletionUpdate::CompletedVisible => payload.push(COMPLETED_VISIBLE_TAG),
        CompletionUpdate::Cancelled => payload.push(CANCELLED_TAG),
        CompletionUpdate::Failed(failure) => {
            let slug = failure.slug.as_bytes();
            if slug.len() > MAX_FAILURE_SLUG {
                return Err(CodecError::FailureSlugTooLong {
                    length: slug.len(),
                    maximum: MAX_FAILURE_SLUG,
                });
            }
            payload.push(FAILED_TAG);
            payload.push(encode_phase(failure.phase));
            payload.push(encode_class(failure.class));
            payload.push(encode_retryability(failure.retryability));
            put_u32(payload, slug.len() as u32);
            payload.extend_from_slice(slug);
        }
        CompletionUpdate::SubmittedUnknown => payload.push(SUBMITTED_UNKNOWN_TAG),
    }
    Ok(())
}

fn decode_payload(payload: &[u8]) -> Result<CompletionMessage, CodecError> {
    let mut input = payload;
    let tag = take_u8(&mut input)?;
    let message = match tag {
        TOKEN_MESSAGE_TAG => {
            let device_epoch = DeviceEpoch::new(take_u64(&mut input)?);
            let submission_id = SubmissionId::new(take_u64(&mut input)?);
            let sequence = CompletionSequence::new(take_u64(&mut input)?);
            let update = decode_update(&mut input)?;
            CompletionMessage::Token(CompletionTokenUpdate {
                token: CompletionToken {
                    device_epoch,
                    submission_id,
                },
                sequence,
                update,
            })
        }
        DEVICE_MESSAGE_TAG => {
            let device_epoch = DeviceEpoch::new(take_u64(&mut input)?);
            let sequence = CompletionSequence::new(take_u64(&mut input)?);
            let health = decode_health(take_u8(&mut input)?)?;
            CompletionMessage::Device(CompletionDeviceUpdate {
                device_epoch,
                sequence,
                health,
            })
        }
        other => return Err(CodecError::UnknownMessageTag(other)),
    };
    if !input.is_empty() {
        return Err(CodecError::TrailingPayload { extra: input.len() });
    }
    message.validate()?;
    Ok(message)
}

fn decode_update(input: &mut &[u8]) -> Result<CompletionUpdate, CodecError> {
    match take_u8(input)? {
        SUBMITTED_TAG => Ok(CompletionUpdate::Submitted),
        COMPLETED_VISIBLE_TAG => Ok(CompletionUpdate::CompletedVisible),
        CANCELLED_TAG => Ok(CompletionUpdate::Cancelled),
        FAILED_TAG => {
            let phase = decode_phase(take_u8(input)?)?;
            let class = decode_class(take_u8(input)?)?;
            let retryability = decode_retryability(take_u8(input)?)?;
            let length = take_u32(input)? as usize;
            if length > MAX_FAILURE_SLUG {
                return Err(CodecError::FailureSlugTooLong {
                    length,
                    maximum: MAX_FAILURE_SLUG,
                });
            }
            let slug = String::from_utf8(take_slice(input, length)?.to_vec())
                .map_err(CodecError::InvalidUtf8)?;
            let mut failure = CompletionFailure::new(phase, class, slug)?;
            failure.retryability = retryability;
            Ok(CompletionUpdate::Failed(failure))
        }
        SUBMITTED_UNKNOWN_TAG => Ok(CompletionUpdate::SubmittedUnknown),
        other => Err(CodecError::UnknownUpdateTag(other)),
    }
}

fn read_header<R: Read>(reader: &mut R, header: &mut [u8; 8]) -> Result<(), CodecError> {
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
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
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
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(CodecError::Io(error)),
        }
    }
    Ok(())
}

fn put_u64(buffer: &mut Vec<u8>, value: u64) {
    buffer.extend_from_slice(&value.to_be_bytes());
}

fn put_u32(buffer: &mut Vec<u8>, value: u32) {
    buffer.extend_from_slice(&value.to_be_bytes());
}

fn take_u8(input: &mut &[u8]) -> Result<u8, CodecError> {
    let Some((first, rest)) = input.split_first() else {
        return Err(CodecError::TruncatedPayload {
            needed: 1,
            remaining: 0,
        });
    };
    *input = rest;
    Ok(*first)
}

fn take_u32(input: &mut &[u8]) -> Result<u32, CodecError> {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(take_slice(input, 4)?);
    Ok(u32::from_be_bytes(bytes))
}

fn take_u64(input: &mut &[u8]) -> Result<u64, CodecError> {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(take_slice(input, 8)?);
    Ok(u64::from_be_bytes(bytes))
}

fn take_slice<'a>(input: &mut &'a [u8], length: usize) -> Result<&'a [u8], CodecError> {
    if input.len() < length {
        return Err(CodecError::TruncatedPayload {
            needed: length,
            remaining: input.len(),
        });
    }
    let (value, rest) = input.split_at(length);
    *input = rest;
    Ok(value)
}

const fn encode_phase(phase: ProviderPhase) -> u8 {
    match phase {
        ProviderPhase::Resolve => 0,
        ProviderPhase::Compile => 1,
        ProviderPhase::Encode => 2,
        ProviderPhase::Submit => 3,
        ProviderPhase::Wait => 4,
        ProviderPhase::Readback => 5,
    }
}

fn decode_phase(value: u8) -> Result<ProviderPhase, CodecError> {
    match value {
        0 => Ok(ProviderPhase::Resolve),
        1 => Ok(ProviderPhase::Compile),
        2 => Ok(ProviderPhase::Encode),
        3 => Ok(ProviderPhase::Submit),
        4 => Ok(ProviderPhase::Wait),
        5 => Ok(ProviderPhase::Readback),
        other => Err(CodecError::UnknownEnumValue {
            field: "provider phase",
            value: other,
        }),
    }
}

const fn encode_class(class: ProviderErrorClass) -> u8 {
    match class {
        ProviderErrorClass::Args => 0,
        ProviderErrorClass::Capability => 1,
        ProviderErrorClass::Resource => 2,
        ProviderErrorClass::Compile => 3,
        ProviderErrorClass::Execute => 4,
        ProviderErrorClass::DeviceLost => 5,
        ProviderErrorClass::Internal => 6,
    }
}

fn decode_class(value: u8) -> Result<ProviderErrorClass, CodecError> {
    match value {
        0 => Ok(ProviderErrorClass::Args),
        1 => Ok(ProviderErrorClass::Capability),
        2 => Ok(ProviderErrorClass::Resource),
        3 => Ok(ProviderErrorClass::Compile),
        4 => Ok(ProviderErrorClass::Execute),
        5 => Ok(ProviderErrorClass::DeviceLost),
        6 => Ok(ProviderErrorClass::Internal),
        other => Err(CodecError::UnknownEnumValue {
            field: "provider error class",
            value: other,
        }),
    }
}

const fn encode_retryability(retryability: Retryability) -> u8 {
    match retryability {
        Retryability::Never => 0,
        Retryability::RetrySameTrace => 1,
        Retryability::RetryAfterRecreate => 2,
        Retryability::Unknown => 3,
    }
}

fn decode_retryability(value: u8) -> Result<Retryability, CodecError> {
    match value {
        0 => Ok(Retryability::Never),
        1 => Ok(Retryability::RetrySameTrace),
        2 => Ok(Retryability::RetryAfterRecreate),
        3 => Ok(Retryability::Unknown),
        other => Err(CodecError::UnknownEnumValue {
            field: "retryability",
            value: other,
        }),
    }
}

const fn encode_health(health: ProviderHealth) -> u8 {
    match health {
        ProviderHealth::Usable => 0,
        ProviderHealth::Exhausted => 1,
        ProviderHealth::DeviceLost => 2,
    }
}

fn decode_health(value: u8) -> Result<ProviderHealth, CodecError> {
    match value {
        0 => Ok(ProviderHealth::Usable),
        1 => Ok(ProviderHealth::Exhausted),
        2 => Ok(ProviderHealth::DeviceLost),
        other => Err(CodecError::UnknownEnumValue {
            field: "provider health",
            value: other,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn token() -> CompletionToken {
        CompletionToken {
            device_epoch: DeviceEpoch::new(7),
            submission_id: SubmissionId::new(42),
        }
    }

    fn failure() -> CompletionFailure {
        let mut failure = CompletionFailure::new(
            ProviderPhase::Wait,
            ProviderErrorClass::Execute,
            "synthetic_failure",
        )
        .unwrap();
        failure.retryability = Retryability::RetryAfterRecreate;
        failure
    }

    fn token_message(update: CompletionUpdate) -> CompletionMessage {
        CompletionMessage::Token(CompletionTokenUpdate {
            token: token(),
            sequence: CompletionSequence::new(3),
            update,
        })
    }

    fn device_message(health: ProviderHealth) -> CompletionMessage {
        CompletionMessage::Device(CompletionDeviceUpdate {
            device_epoch: DeviceEpoch::new(7),
            sequence: CompletionSequence::new(4),
            health,
        })
    }

    #[test]
    fn round_trips_every_message_shape() {
        for update in [
            CompletionUpdate::Submitted,
            CompletionUpdate::CompletedVisible,
            CompletionUpdate::Cancelled,
            CompletionUpdate::Failed(failure()),
            CompletionUpdate::SubmittedUnknown,
        ] {
            let message = token_message(update);
            let frame = CompletionCodec::encode(&message).unwrap();
            assert_eq!(CompletionCodec::decode(&frame).unwrap(), message);
        }
        for health in [
            ProviderHealth::Usable,
            ProviderHealth::Exhausted,
            ProviderHealth::DeviceLost,
        ] {
            let message = device_message(health);
            let frame = CompletionCodec::encode(&message).unwrap();
            assert_eq!(CompletionCodec::decode(&frame).unwrap(), message);
        }
    }

    #[test]
    fn writes_and_reads_frames_over_a_byte_stream() {
        let message = token_message(CompletionUpdate::CompletedVisible);
        let mut bytes = Vec::new();
        CompletionCodec::write(&mut bytes, &message).unwrap();
        let mut cursor = Cursor::new(bytes);
        assert_eq!(CompletionCodec::read(&mut cursor).unwrap(), message);
        assert!(matches!(
            CompletionCodec::read(&mut cursor).unwrap_err(),
            CodecError::Eof
        ));
    }

    #[test]
    fn rejects_bad_magic_and_truncated_frames() {
        let message = token_message(CompletionUpdate::Submitted);
        let mut frame = CompletionCodec::encode(&message).unwrap();
        frame[0] = b'X';
        assert!(matches!(
            CompletionCodec::decode(&frame).unwrap_err(),
            CodecError::BadMagic(_)
        ));
        let frame = CompletionCodec::encode(&message).unwrap();
        assert!(matches!(
            CompletionCodec::decode(&frame[..6]).unwrap_err(),
            CodecError::TruncatedFrame { .. }
        ));
        assert!(matches!(
            CompletionCodec::decode(&frame[..frame.len() - 1]).unwrap_err(),
            CodecError::TruncatedFrame { .. }
        ));
    }

    #[test]
    fn rejects_an_oversized_declared_length_before_allocating() {
        let mut frame = Vec::new();
        frame.extend_from_slice(&FRAME_MAGIC);
        frame.extend_from_slice(&((MAX_FRAME_PAYLOAD + 1) as u32).to_be_bytes());
        assert!(matches!(
            CompletionCodec::decode(&frame).unwrap_err(),
            CodecError::FrameTooLarge { .. }
        ));
    }

    #[test]
    fn rejects_unknown_tags_and_enum_values() {
        let mut payload = vec![0x7f];
        assert!(matches!(
            CompletionCodec::decode(&frame_with_payload(&payload)).unwrap_err(),
            CodecError::UnknownMessageTag(0x7f)
        ));
        payload.clear();
        payload.extend_from_slice(&[TOKEN_MESSAGE_TAG]);
        put_u64(&mut payload, 7);
        put_u64(&mut payload, 42);
        put_u64(&mut payload, 3);
        payload.push(0x7e);
        assert!(matches!(
            CompletionCodec::decode(&frame_with_payload(&payload)).unwrap_err(),
            CodecError::UnknownUpdateTag(0x7e)
        ));
        payload.clear();
        payload.extend_from_slice(&[DEVICE_MESSAGE_TAG]);
        put_u64(&mut payload, 7);
        put_u64(&mut payload, 1);
        payload.push(0x7d);
        assert!(matches!(
            CompletionCodec::decode(&frame_with_payload(&payload)).unwrap_err(),
            CodecError::UnknownEnumValue {
                field: "provider health",
                value: 0x7d
            }
        ));
    }

    #[test]
    fn rejects_trailing_payload_and_invalid_messages() {
        let mut payload = vec![DEVICE_MESSAGE_TAG];
        put_u64(&mut payload, 7);
        put_u64(&mut payload, 1);
        payload.push(encode_health(ProviderHealth::Usable));
        payload.push(0);
        assert!(matches!(
            CompletionCodec::decode(&frame_with_payload(&payload)).unwrap_err(),
            CodecError::TrailingPayload { extra: 1 }
        ));

        let invalid = CompletionMessage::Token(CompletionTokenUpdate {
            token: CompletionToken {
                device_epoch: DeviceEpoch::new(7),
                submission_id: SubmissionId::new(0),
            },
            sequence: CompletionSequence::new(1),
            update: CompletionUpdate::Submitted,
        });
        let frame = CompletionCodec::encode(&invalid).unwrap();
        assert!(matches!(
            CompletionCodec::decode(&frame).unwrap_err(),
            CodecError::Contract(ContractError::InvalidIdentity("submission id"))
        ));
    }

    #[test]
    fn rejects_an_oversized_failure_slug() {
        let message = CompletionMessage::Token(CompletionTokenUpdate {
            token: token(),
            sequence: CompletionSequence::new(1),
            update: CompletionUpdate::Failed(
                CompletionFailure::new(
                    ProviderPhase::Wait,
                    ProviderErrorClass::Internal,
                    "x".repeat(MAX_FAILURE_SLUG + 1),
                )
                .unwrap(),
            ),
        });
        assert!(matches!(
            CompletionCodec::encode(&message).unwrap_err(),
            CodecError::FailureSlugTooLong { .. }
        ));
    }

    fn frame_with_payload(payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::new();
        frame.extend_from_slice(&FRAME_MAGIC);
        frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(payload);
        frame
    }
}
