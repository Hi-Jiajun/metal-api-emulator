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
//! A frame is bounded by [`crate::command_codec::MAX_COMMAND_FRAME`]; a trace
//! whose encoded owned bytes exceed that bound must use a chunked data channel
//! (not part of this version). Lease-backed views carry only their lease id,
//! so the owner and provider must import the backing through a separate
//! descriptor-passing channel before submitting a trace that references it.

use crate::codec::CodecError;
use crate::command_codec::CommandCodec;
use metal_api_core::provider::{
    CompiledComputePipeline, CompletionDisposition, CompletionReadback, CompletionToken,
    ComputeProvider, ComputeTrace, DeviceEpoch, PipelineCompileRequest, PipelineProvider,
    ProviderCapabilities, ProviderError, ProviderErrorClass, ProviderPhase, ProviderSubmission,
    ResourceTableSnapshot, Retryability, ValidatedComputeTrace,
};
use std::fmt;
use std::io::{Read, Write};
use std::sync::Mutex;
use std::time::Duration;

/// One owner request. The response is always a [`CommandResponse`] on the same
/// connection.
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum CommandRequest {
    /// Ask for the provider device epoch and capabilities.
    Capabilities,
    /// Compile one reviewed shader artifact on the provider.
    Compile { request: PipelineCompileRequest },
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
            Self::Compile { .. } => "compile",
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
    Compiled {
        pipeline: CompiledComputePipeline,
    },
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
            Self::Compiled { .. } => "compiled",
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
        }
    }
}

impl std::error::Error for CommandError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Codec(error) => Some(error),
            Self::Remote(_) | Self::UnexpectedResponse { .. } => None,
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
}

impl<R: Read, W: Write> CommandTransport<R, W> {
    /// Wrap a reader and a writer.
    pub const fn new(reader: R, writer: W) -> Self {
        Self {
            reader,
            writer,
            sent: 0,
            received: 0,
        }
    }

    /// Send one request and read its response.
    pub fn request(&mut self, request: &CommandRequest) -> Result<CommandResponse, CommandError> {
        CommandCodec::write_request(&mut self.writer, request)?;
        self.writer.flush().map_err(CodecError::Io)?;
        self.sent += 1;
        let response = CommandCodec::read_response(&mut self.reader)?;
        self.received += 1;
        Ok(response)
    }

    /// Read one request. Used by the provider-side server loop.
    pub fn recv_request(&mut self) -> Result<CommandRequest, CommandError> {
        let request = CommandCodec::read_request(&mut self.reader)?;
        self.received += 1;
        Ok(request)
    }

    /// Write one response. Used by the provider-side server loop.
    pub fn send_response(&mut self, response: &CommandResponse) -> Result<(), CommandError> {
        CommandCodec::write_response(&mut self.writer, response)?;
        self.sent += 1;
        Ok(())
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
        let mut transport = self
            .transport
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        transport
            .request(&request)
            .map_err(|error| transport_error(phase, error))
    }
}

impl<R: Read + Send, W: Write + Send> ComputeProvider for RemoteProvider<R, W> {
    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities.clone()
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

/// Provider-side loop. Serves requests until the owner closes the connection.
///
/// Submission requests are admitted with the provider's own capabilities
/// before [`ComputeProvider::submit`] is called, so remote owners cannot skip
/// the admission contract.
pub fn serve_provider<R: Read, W: Write>(
    provider: &dyn PipelineProvider,
    transport: &mut CommandTransport<R, W>,
) -> Result<(), CommandError> {
    loop {
        let request = match transport.recv_request() {
            Ok(request) => request,
            Err(CommandError::Codec(CodecError::Eof)) => return Ok(()),
            Err(error) => return Err(error),
        };
        let response = handle_request(provider, request);
        transport.send_response(&response)?;
        transport.flush()?;
    }
}

fn handle_request(provider: &dyn PipelineProvider, request: CommandRequest) -> CommandResponse {
    match request {
        CommandRequest::Capabilities => CommandResponse::Capabilities {
            epoch: provider.device_epoch(),
            capabilities: provider.capabilities(),
        },
        CommandRequest::Compile { request } => match provider.compile(request) {
            Ok(pipeline) => CommandResponse::Compiled { pipeline },
            Err(error) => CommandResponse::Error { error },
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
    use super::{serve_provider, CommandRequest, CommandResponse, RemoteProvider};
    use crate::command_codec::CommandCodec;
    use metal_api_core::provider::{
        AllocationId, AllocationRecord, BufferAccess, BufferBindingContract, BufferSource,
        BufferView, BufferWriteback, CompiledComputePipeline, CompletionDisposition,
        CompletionReadback, CompletionToken, ComputePass, ComputeProvider, ComputeTrace,
        DeviceEpoch, Dispatch, DispatchKind, DispatchType, FootprintProof, FunctionIdentity,
        OperationId, PipelineCompileRequest, PipelineContract, PipelineId, PipelineProvider,
        ProviderCapabilities, ProviderError, ProviderErrorClass, ProviderPhase, ProviderSubmission,
        ResourceTableSnapshot, Retryability, SemanticDigest, ShaderSource, SubmissionId,
        ValidatedComputeTrace, ViewId, PROVIDER_SCHEMA_VERSION,
    };
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

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
            CommandRequest::Compile {
                request: request.clone(),
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
        let provider = FakeProvider {
            epoch: DeviceEpoch::new(7),
            capabilities: fake_capabilities(),
            submissions: Arc::clone(&submissions),
        };
        let (client, mut server) = super::unix::pair().unwrap();
        let server_thread = std::thread::spawn(move || serve_provider(&provider, &mut server));

        let remote = RemoteProvider::connect(client).unwrap();
        assert_eq!(remote.device_epoch(), DeviceEpoch::new(7));
        assert_eq!(remote.capabilities(), fake_capabilities());
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
        assert_eq!(readback.writebacks[0].bytes, vec![4, 3, 2, 1]);
        remote.release_completion(token).unwrap();
        remote.release_pipeline(&compiled).unwrap();
        drop(remote);
        server_thread.join().unwrap().unwrap();
        assert_eq!(submissions.load(Ordering::SeqCst), 1);
    }
}
