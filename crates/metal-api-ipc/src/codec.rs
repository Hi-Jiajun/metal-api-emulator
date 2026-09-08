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
    FrameTooLarge { length: usize, maximum: usize },
    FailureSlugTooLong { length: usize, maximum: usize },
    TruncatedFrame { expected: usize, actual: usize },
    TruncatedPayload { needed: usize, remaining: usize },
    TrailingPayload { extra: usize },
    UnknownMessageTag(u8),
    UnknownUpdateTag(u8),
    UnknownEnumValue { field: &'static str, value: u8 },
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
