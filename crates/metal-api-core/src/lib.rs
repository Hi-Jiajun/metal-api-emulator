//! Backend-neutral, source-level subset of Metal's compute object model.
//!
//! This crate owns API ordering and object lifetimes. A backend owns shader
//! translation and execution through [`ComputeExecutor`]. The first milestone
//! is synchronous on commit, but keeps Metal's explicit commit/wait boundary so
//! a later asynchronous executor does not need to change application code.

use std::any::Any;
use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

pub mod provider;
pub mod provider_api;

/// Opaque backend-owned compiled pipeline state.
pub type PipelineArtifact = Arc<dyn Any + Send + Sync>;

/// A three-dimensional Metal grid or threadgroup size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size([u32; 3]);

impl Size {
    pub fn new(width: u32, height: u32, depth: u32) -> Result<Self, ApiError> {
        let dimensions = [width, height, depth];
        if dimensions.contains(&0) {
            return Err(ApiError::ZeroSize);
        }
        Ok(Self(dimensions))
    }

    pub const fn dimensions(self) -> [u32; 3] {
        self.0
    }
}

/// A backend failure that remains visible at the API boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorError {
    message: String,
}

impl ExecutorError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl StdError for ExecutorError {}

/// Typed API/state-machine refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApiError {
    ZeroSize,
    EmptyLibrary,
    InvalidBinaryAir,
    MetallibUnsupported,
    EmptyFunctionName,
    EmptyBuffer,
    BufferOffsetOutOfBounds {
        offset: usize,
        length: usize,
    },
    AliasedBufferBindings {
        first: u32,
        second: u32,
    },
    ForeignPipeline,
    EncoderAlreadyOpen,
    EncoderNotEnded,
    EncoderAlreadyEnded,
    MissingPipeline,
    MissingDispatch,
    NoEncodedCommands,
    CommandBufferAlreadyCommitted,
    CommandBufferNotCommitted,
    CommandBufferNotCompleted,
    ExecutorPanicked,
    UnknownBufferUpdate(u32),
    BufferUpdateOutOfBounds {
        index: u32,
        offset: usize,
        update_len: usize,
        binding_len: usize,
    },
    StatePoisoned(&'static str),
    Executor(ExecutorError),
}

impl fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSize => formatter.write_str("grid dimensions must be non-zero"),
            Self::EmptyLibrary => formatter.write_str("AIR library source must not be empty"),
            Self::InvalidBinaryAir => formatter.write_str(
                "binary AIR must be one raw LLVM bitcode module or one offset-zero bitcode wrapper",
            ),
            Self::MetallibUnsupported => formatter.write_str(
                "MTLB containers require a function-name resolver and are not supported yet",
            ),
            Self::EmptyFunctionName => formatter.write_str("function name must not be empty"),
            Self::EmptyBuffer => formatter.write_str("buffer length must be non-zero"),
            Self::BufferOffsetOutOfBounds { offset, length } => {
                write!(formatter, "buffer offset {offset} exceeds length {length}")
            }
            Self::AliasedBufferBindings { first, second } => write!(
                formatter,
                "buffer alias identity across bindings {first} and {second} is not supported"
            ),
            Self::ForeignPipeline => {
                formatter.write_str("compute pipeline belongs to a different device executor")
            }
            Self::EncoderAlreadyOpen => formatter.write_str("a compute encoder is already open"),
            Self::EncoderNotEnded => {
                formatter.write_str("compute encoder was dropped without end_encoding")
            }
            Self::EncoderAlreadyEnded => formatter.write_str("compute encoder already ended"),
            Self::MissingPipeline => formatter.write_str("compute encoder has no pipeline"),
            Self::MissingDispatch => formatter.write_str("compute encoder has no dispatch"),
            Self::NoEncodedCommands => formatter.write_str("command buffer contains no commands"),
            Self::CommandBufferAlreadyCommitted => {
                formatter.write_str("command buffer was already committed")
            }
            Self::CommandBufferNotCommitted => {
                formatter.write_str("command buffer has not been committed")
            }
            Self::CommandBufferNotCompleted => {
                formatter.write_str("command buffer has not completed")
            }
            Self::ExecutorPanicked => formatter.write_str("compute executor panicked"),
            Self::UnknownBufferUpdate(index) => {
                write!(formatter, "executor updated unbound buffer index {index}")
            }
            Self::BufferUpdateOutOfBounds {
                index,
                offset,
                update_len,
                binding_len,
            } => write!(
                formatter,
                "executor update for buffer {index} at {offset} length {update_len} exceeds bound length {binding_len}"
            ),
            Self::StatePoisoned(owner) => write!(formatter, "{owner} state lock is poisoned"),
            Self::Executor(error) => write!(formatter, "executor: {error}"),
        }
    }
}

impl StdError for ApiError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Executor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ExecutorError> for ApiError {
    fn from(error: ExecutorError) -> Self {
        Self::Executor(error)
    }
}

/// Borrowed representation of one function's AIR source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AirSource<'a> {
    /// Already-sanitized textual LLVM IR.
    SanitizedLl(&'a str),
    /// One raw LLVM bitcode module or offset-zero LLVM bitcode wrapper.
    Binary(&'a [u8]),
}

#[derive(Clone)]
enum LibraryStorage {
    SanitizedLl(Arc<str>),
    Binary(Arc<[u8]>),
}

/// Immutable AIR library supplied by an application.
#[derive(Clone)]
pub struct Library {
    source: LibraryStorage,
}

impl Library {
    pub fn function(&self, name: impl Into<String>) -> Result<Function, ApiError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ApiError::EmptyFunctionName);
        }
        Ok(Function {
            source: self.source.clone(),
            name,
        })
    }
}

/// One named function in an AIR library.
#[derive(Clone)]
pub struct Function {
    source: LibraryStorage,
    name: String,
}

impl Function {
    pub fn air_source(&self) -> AirSource<'_> {
        match &self.source {
            LibraryStorage::SanitizedLl(source) => AirSource::SanitizedLl(source),
            LibraryStorage::Binary(source) => AirSource::Binary(source),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Snapshot of one buffer binding passed to an executor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferBinding {
    pub index: u32,
    pub bytes: Vec<u8>,
}

/// A bounded write returned by an executor, relative to a binding snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferUpdate {
    pub index: u32,
    pub offset: usize,
    pub bytes: Vec<u8>,
}

/// One fully encoded compute pass.
#[derive(Clone)]
pub struct ComputeSubmission {
    pub pipeline: PipelineArtifact,
    pub buffers: Vec<BufferBinding>,
    pub threads_per_grid: Size,
    pub threads_per_threadgroup: Size,
}

/// Backend contract consumed by the Metal-like object model.
pub trait ComputeExecutor: Send + Sync {
    fn new_compute_pipeline(&self, function: &Function) -> Result<PipelineArtifact, ExecutorError>;

    fn execute(&self, submission: ComputeSubmission) -> Result<Vec<BufferUpdate>, ExecutorError>;
}

/// Source-level Metal device backed by one executor.
#[derive(Clone)]
pub struct Device {
    executor: Arc<dyn ComputeExecutor>,
}

impl Device {
    pub fn new(executor: Arc<dyn ComputeExecutor>) -> Self {
        Self { executor }
    }

    pub fn new_library_with_air(&self, air: impl Into<Arc<str>>) -> Result<Library, ApiError> {
        let air = air.into();
        if air.trim().is_empty() {
            return Err(ApiError::EmptyLibrary);
        }
        Ok(Library {
            source: LibraryStorage::SanitizedLl(air),
        })
    }

    pub fn new_library_with_binary_air(
        &self,
        air: impl Into<Arc<[u8]>>,
    ) -> Result<Library, ApiError> {
        let air = air.into();
        validate_binary_air(&air)?;
        Ok(Library {
            source: LibraryStorage::Binary(air),
        })
    }

    pub fn new_compute_pipeline_state(
        &self,
        function: &Function,
    ) -> Result<ComputePipelineState, ApiError> {
        Ok(ComputePipelineState {
            artifact: self.executor.new_compute_pipeline(function)?,
            executor: Arc::clone(&self.executor),
        })
    }

    pub fn new_buffer(&self, length: usize) -> Result<Buffer, ApiError> {
        if length == 0 {
            return Err(ApiError::EmptyBuffer);
        }
        Ok(Buffer::from_bytes(vec![0; length]))
    }

    pub fn new_buffer_with_bytes(&self, bytes: impl Into<Vec<u8>>) -> Result<Buffer, ApiError> {
        let bytes = bytes.into();
        if bytes.is_empty() {
            return Err(ApiError::EmptyBuffer);
        }
        Ok(Buffer::from_bytes(bytes))
    }

    pub fn new_command_queue(&self) -> CommandQueue {
        CommandQueue {
            executor: Arc::clone(&self.executor),
        }
    }
}

fn validate_binary_air(air: &[u8]) -> Result<(), ApiError> {
    const RAW_BITCODE_MAGIC: [u8; 4] = [0x42, 0x43, 0xc0, 0xde];
    const WRAPPED_BITCODE_MAGIC: [u8; 4] = [0xde, 0xc0, 0x17, 0x0b];
    const WRAPPER_HEADER_LEN: usize = 0x14;

    if air.is_empty() {
        return Err(ApiError::EmptyLibrary);
    }
    let Some(magic) = air.get(..4) else {
        return Err(ApiError::InvalidBinaryAir);
    };
    if magic == b"MTLB" {
        return Err(ApiError::MetallibUnsupported);
    }
    if magic == RAW_BITCODE_MAGIC {
        return Ok(());
    }
    if magic != WRAPPED_BITCODE_MAGIC || air.len() < WRAPPER_HEADER_LEN {
        return Err(ApiError::InvalidBinaryAir);
    }
    let offset = u32::from_le_bytes(
        air[8..12]
            .try_into()
            .expect("wrapper header length checked"),
    ) as usize;
    let size = u32::from_le_bytes(
        air[12..16]
            .try_into()
            .expect("wrapper header length checked"),
    ) as usize;
    let valid = offset >= WRAPPER_HEADER_LEN
        && size != 0
        && offset.checked_add(size).is_some_and(|end| end == air.len());
    if !valid {
        return Err(ApiError::InvalidBinaryAir);
    }
    Ok(())
}

/// Opaque pipeline returned by a device.
#[derive(Clone)]
pub struct ComputePipelineState {
    artifact: PipelineArtifact,
    executor: Arc<dyn ComputeExecutor>,
}

/// Shared CPU-visible buffer used by the MVP.
#[derive(Clone)]
pub struct Buffer {
    bytes: Arc<Mutex<Vec<u8>>>,
}

impl Buffer {
    fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Arc::new(Mutex::new(bytes)),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, Vec<u8>>, ApiError> {
        self.bytes
            .lock()
            .map_err(|_| ApiError::StatePoisoned("buffer"))
    }

    fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.bytes, &other.bytes)
    }

    pub fn len(&self) -> Result<usize, ApiError> {
        Ok(self.lock()?.len())
    }

    pub fn is_empty(&self) -> Result<bool, ApiError> {
        Ok(self.lock()?.is_empty())
    }

    pub fn read(&self) -> Result<Vec<u8>, ApiError> {
        Ok(self.lock()?.clone())
    }

    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), ApiError> {
        let mut destination = self.lock()?;
        let end = offset
            .checked_add(bytes.len())
            .filter(|end| *end <= destination.len())
            .ok_or(ApiError::BufferOffsetOutOfBounds {
                offset,
                length: destination.len(),
            })?;
        destination[offset..end].copy_from_slice(bytes);
        Ok(())
    }
}

/// Queue which creates command buffers for one device/executor.
#[derive(Clone)]
pub struct CommandQueue {
    executor: Arc<dyn ComputeExecutor>,
}

impl CommandQueue {
    pub fn command_buffer(&self) -> CommandBuffer {
        CommandBuffer {
            executor: Arc::clone(&self.executor),
            shared: Arc::new(CommandBufferShared {
                inner: Mutex::new(CommandBufferInner {
                    passes: Vec::new(),
                    encoder_open: false,
                    recording_error: None,
                    status: CommandBufferStatus::Recording,
                    failure: None,
                }),
                completion: Condvar::new(),
            }),
        }
    }
}

#[derive(Clone)]
struct EncodedBinding {
    buffer: Buffer,
    offset: usize,
}

#[derive(Clone)]
struct ComputePass {
    pipeline: PipelineArtifact,
    buffers: BTreeMap<u32, EncodedBinding>,
    threads_per_grid: Size,
    threads_per_threadgroup: Size,
}

/// Observable command-buffer lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandBufferStatus {
    Recording,
    Committed,
    Completed,
    Failed,
}

struct CommandBufferInner {
    passes: Vec<ComputePass>,
    encoder_open: bool,
    recording_error: Option<ApiError>,
    status: CommandBufferStatus,
    failure: Option<ApiError>,
}

struct CommandBufferShared {
    inner: Mutex<CommandBufferInner>,
    completion: Condvar,
}

/// A single-use Metal-like command buffer.
pub struct CommandBuffer {
    executor: Arc<dyn ComputeExecutor>,
    shared: Arc<CommandBufferShared>,
}

impl CommandBuffer {
    fn lock(&self) -> Result<MutexGuard<'_, CommandBufferInner>, ApiError> {
        self.shared
            .inner
            .lock()
            .map_err(|_| ApiError::StatePoisoned("command buffer"))
    }

    pub fn status(&self) -> Result<CommandBufferStatus, ApiError> {
        Ok(self.lock()?.status)
    }

    pub fn compute_command_encoder(&self) -> Result<ComputeCommandEncoder, ApiError> {
        let mut inner = self.lock()?;
        if inner.status != CommandBufferStatus::Recording {
            return Err(ApiError::CommandBufferAlreadyCommitted);
        }
        if inner.encoder_open {
            return Err(ApiError::EncoderAlreadyOpen);
        }
        inner.encoder_open = true;
        Ok(ComputeCommandEncoder {
            shared: Arc::clone(&self.shared),
            executor: Arc::clone(&self.executor),
            pipeline: None,
            buffers: BTreeMap::new(),
            dispatch: None,
            ended: false,
        })
    }

    pub fn commit(&self) -> Result<(), ApiError> {
        let passes = {
            let mut inner = self.lock()?;
            if inner.status != CommandBufferStatus::Recording {
                return Err(ApiError::CommandBufferAlreadyCommitted);
            }
            if inner.encoder_open {
                return Err(ApiError::EncoderNotEnded);
            }
            if let Some(error) = inner.recording_error.clone() {
                inner.status = CommandBufferStatus::Failed;
                inner.failure = Some(error.clone());
                self.shared.completion.notify_all();
                return Err(error);
            }
            if inner.passes.is_empty() {
                return Err(ApiError::NoEncodedCommands);
            }
            inner.status = CommandBufferStatus::Committed;
            inner.passes.clone()
        };

        let result = match catch_unwind(AssertUnwindSafe(|| self.execute_passes(&passes))) {
            Ok(result) => result,
            Err(_) => Err(ApiError::ExecutorPanicked),
        };
        let mut inner = self.lock()?;
        match &result {
            Ok(()) => inner.status = CommandBufferStatus::Completed,
            Err(error) => {
                inner.status = CommandBufferStatus::Failed;
                inner.failure = Some(error.clone());
            }
        }
        self.shared.completion.notify_all();
        result
    }

    fn execute_passes(&self, passes: &[ComputePass]) -> Result<(), ApiError> {
        for pass in passes {
            let mut snapshots = Vec::with_capacity(pass.buffers.len());
            for (index, binding) in &pass.buffers {
                let bytes = binding.buffer.lock()?;
                if binding.offset > bytes.len() {
                    return Err(ApiError::BufferOffsetOutOfBounds {
                        offset: binding.offset,
                        length: bytes.len(),
                    });
                }
                snapshots.push(BufferBinding {
                    index: *index,
                    bytes: bytes[binding.offset..].to_vec(),
                });
            }
            let updates = self.executor.execute(ComputeSubmission {
                pipeline: Arc::clone(&pass.pipeline),
                buffers: snapshots,
                threads_per_grid: pass.threads_per_grid,
                threads_per_threadgroup: pass.threads_per_threadgroup,
            })?;

            let mut writes = Vec::with_capacity(updates.len());
            for update in updates {
                let binding = pass
                    .buffers
                    .get(&update.index)
                    .ok_or(ApiError::UnknownBufferUpdate(update.index))?;
                let binding_len = binding.buffer.len()?.saturating_sub(binding.offset);
                let update_end = update
                    .offset
                    .checked_add(update.bytes.len())
                    .filter(|end| *end <= binding_len)
                    .ok_or(ApiError::BufferUpdateOutOfBounds {
                        index: update.index,
                        offset: update.offset,
                        update_len: update.bytes.len(),
                        binding_len,
                    })?;
                debug_assert!(update_end <= binding_len);
                writes.push((binding.clone(), update));
            }
            for (binding, update) in writes {
                binding
                    .buffer
                    .write(binding.offset + update.offset, &update.bytes)?;
            }
        }
        Ok(())
    }

    pub fn wait_until_completed(&self) -> Result<(), ApiError> {
        let mut inner = self.lock()?;
        loop {
            match inner.status {
                CommandBufferStatus::Recording => return Err(ApiError::CommandBufferNotCommitted),
                CommandBufferStatus::Committed => {
                    inner = self
                        .shared
                        .completion
                        .wait(inner)
                        .map_err(|_| ApiError::StatePoisoned("command buffer"))?;
                }
                CommandBufferStatus::Completed => return Ok(()),
                CommandBufferStatus::Failed => {
                    return Err(inner
                        .failure
                        .clone()
                        .unwrap_or(ApiError::CommandBufferNotCompleted));
                }
            }
        }
    }
}

/// Mutable compute encoder sharing its command buffer's explicit recording state.
pub struct ComputeCommandEncoder {
    shared: Arc<CommandBufferShared>,
    executor: Arc<dyn ComputeExecutor>,
    pipeline: Option<PipelineArtifact>,
    buffers: BTreeMap<u32, EncodedBinding>,
    dispatch: Option<(Size, Size)>,
    ended: bool,
}

impl ComputeCommandEncoder {
    pub fn set_compute_pipeline_state(
        &mut self,
        pipeline: &ComputePipelineState,
    ) -> Result<(), ApiError> {
        self.ensure_open()?;
        if !Arc::ptr_eq(&self.executor, &pipeline.executor) {
            return Err(ApiError::ForeignPipeline);
        }
        self.pipeline = Some(Arc::clone(&pipeline.artifact));
        Ok(())
    }

    pub fn set_buffer(
        &mut self,
        index: u32,
        buffer: &Buffer,
        offset: usize,
    ) -> Result<(), ApiError> {
        self.ensure_open()?;
        let length = buffer.len()?;
        if offset > length {
            return Err(ApiError::BufferOffsetOutOfBounds { offset, length });
        }
        if let Some((first, _)) = self.buffers.iter().find(|(other_index, binding)| {
            **other_index != index && binding.buffer.shares_storage_with(buffer)
        }) {
            return Err(ApiError::AliasedBufferBindings {
                first: *first,
                second: index,
            });
        }
        self.buffers.insert(
            index,
            EncodedBinding {
                buffer: buffer.clone(),
                offset,
            },
        );
        Ok(())
    }

    pub fn dispatch_threads(
        &mut self,
        threads_per_grid: Size,
        threads_per_threadgroup: Size,
    ) -> Result<(), ApiError> {
        self.ensure_open()?;
        self.dispatch = Some((threads_per_grid, threads_per_threadgroup));
        Ok(())
    }

    pub fn end_encoding(mut self) -> Result<(), ApiError> {
        self.ensure_open()?;
        let result = match (&self.pipeline, self.dispatch) {
            (None, _) => Err(ApiError::MissingPipeline),
            (_, None) => Err(ApiError::MissingDispatch),
            (Some(pipeline), Some((threads_per_grid, threads_per_threadgroup))) => {
                let mut inner = self
                    .shared
                    .inner
                    .lock()
                    .map_err(|_| ApiError::StatePoisoned("command buffer"))?;
                if inner.status != CommandBufferStatus::Recording {
                    return Err(ApiError::CommandBufferAlreadyCommitted);
                }
                inner.passes.push(ComputePass {
                    pipeline: Arc::clone(pipeline),
                    buffers: self.buffers.clone(),
                    threads_per_grid,
                    threads_per_threadgroup,
                });
                Ok(())
            }
        };
        let mut inner = self
            .shared
            .inner
            .lock()
            .map_err(|_| ApiError::StatePoisoned("command buffer"))?;
        inner.encoder_open = false;
        if let Err(error) = &result {
            inner.recording_error = Some(error.clone());
        }
        self.ended = true;
        result
    }

    fn ensure_open(&self) -> Result<(), ApiError> {
        if self.ended {
            Err(ApiError::EncoderAlreadyEnded)
        } else {
            Ok(())
        }
    }
}

impl Drop for ComputeCommandEncoder {
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        if let Ok(mut inner) = self.shared.inner.lock() {
            inner.encoder_open = false;
            inner.recording_error = Some(ApiError::EncoderNotEnded);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CopyExecutor {
        bad_update: bool,
    }

    impl ComputeExecutor for CopyExecutor {
        fn new_compute_pipeline(
            &self,
            function: &Function,
        ) -> Result<PipelineArtifact, ExecutorError> {
            if function.name() != "copy_word" {
                return Err(ExecutorError::new("unknown function"));
            }
            Ok(Arc::new(function.name().to_string()))
        }

        fn execute(
            &self,
            submission: ComputeSubmission,
        ) -> Result<Vec<BufferUpdate>, ExecutorError> {
            let source = submission
                .buffers
                .iter()
                .find(|binding| binding.index == 0)
                .ok_or_else(|| ExecutorError::new("missing input"))?;
            if source.bytes.len() < 4 {
                return Err(ExecutorError::new("input is shorter than one word"));
            }
            let valid = BufferUpdate {
                index: 1,
                offset: 0,
                bytes: source.bytes[..4].to_vec(),
            };
            if self.bad_update {
                Ok(vec![
                    valid,
                    BufferUpdate {
                        index: 1,
                        offset: 4,
                        bytes: source.bytes[..4].to_vec(),
                    },
                ])
            } else {
                Ok(vec![valid])
            }
        }
    }

    struct PanicExecutor {
        entered: Arc<std::sync::Barrier>,
        release: Arc<std::sync::Barrier>,
    }

    impl ComputeExecutor for PanicExecutor {
        fn new_compute_pipeline(
            &self,
            _function: &Function,
        ) -> Result<PipelineArtifact, ExecutorError> {
            Ok(Arc::new(()))
        }

        fn execute(
            &self,
            _submission: ComputeSubmission,
        ) -> Result<Vec<BufferUpdate>, ExecutorError> {
            self.entered.wait();
            self.release.wait();
            panic!("synthetic executor panic")
        }
    }

    fn device(bad_update: bool) -> Device {
        Device::new(Arc::new(CopyExecutor { bad_update }))
    }

    fn pipeline(device: &Device) -> ComputePipelineState {
        let library = device
            .new_library_with_air("define void @copy_word() {}")
            .unwrap();
        let function = library.function("copy_word").unwrap();
        device.new_compute_pipeline_state(&function).unwrap()
    }

    #[test]
    fn copy_pass_updates_the_bound_output_after_commit() {
        let device = device(false);
        let pipeline = pipeline(&device);
        let input = device
            .new_buffer_with_bytes(0x6745_2301_u32.to_le_bytes())
            .unwrap();
        let output = device
            .new_buffer_with_bytes(0xabab_abab_u32.to_le_bytes())
            .unwrap();
        let queue = device.new_command_queue();
        let command = queue.command_buffer();
        {
            let mut encoder = command.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&pipeline).unwrap();
            encoder.set_buffer(0, &input, 0).unwrap();
            encoder.set_buffer(1, &output, 0).unwrap();
            encoder
                .dispatch_threads(Size::new(1, 1, 1).unwrap(), Size::new(1, 1, 1).unwrap())
                .unwrap();
            encoder.end_encoding().unwrap();
        }
        command.commit().unwrap();
        assert_eq!(command.status().unwrap(), CommandBufferStatus::Completed);
        command.wait_until_completed().unwrap();
        assert_eq!(output.read().unwrap(), 0x6745_2301_u32.to_le_bytes());
    }

    #[test]
    fn dropped_encoder_poisoning_is_reported_at_commit() {
        let device = device(false);
        let queue = device.new_command_queue();
        let command = queue.command_buffer();
        drop(command.compute_command_encoder().unwrap());
        assert_eq!(command.commit(), Err(ApiError::EncoderNotEnded));
        assert_eq!(
            command.wait_until_completed(),
            Err(ApiError::EncoderNotEnded)
        );
    }

    #[test]
    fn commit_rejects_an_open_encoder_without_consuming_the_command_buffer() {
        let device = device(false);
        let pipeline = pipeline(&device);
        let input = device.new_buffer_with_bytes([1, 0, 0, 0]).unwrap();
        let output = device.new_buffer(4).unwrap();
        let queue = device.new_command_queue();
        let command = queue.command_buffer();
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        encoder.set_buffer(0, &input, 0).unwrap();
        encoder.set_buffer(1, &output, 0).unwrap();
        encoder
            .dispatch_threads(Size::new(1, 1, 1).unwrap(), Size::new(1, 1, 1).unwrap())
            .unwrap();
        assert_eq!(command.commit(), Err(ApiError::EncoderNotEnded));
        assert_eq!(command.status().unwrap(), CommandBufferStatus::Recording);
        encoder.end_encoding().unwrap();
        command.commit().unwrap();
        assert_eq!(command.status().unwrap(), CommandBufferStatus::Completed);
    }

    #[test]
    fn a_command_buffer_is_single_use() {
        let device = device(false);
        let pipeline = pipeline(&device);
        let input = device.new_buffer_with_bytes([1, 0, 0, 0]).unwrap();
        let output = device.new_buffer(4).unwrap();
        let queue = device.new_command_queue();
        let command = queue.command_buffer();
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        encoder.set_buffer(0, &input, 0).unwrap();
        encoder.set_buffer(1, &output, 0).unwrap();
        encoder
            .dispatch_threads(Size::new(1, 1, 1).unwrap(), Size::new(1, 1, 1).unwrap())
            .unwrap();
        encoder.end_encoding().unwrap();
        command.commit().unwrap();
        assert_eq!(
            command.commit(),
            Err(ApiError::CommandBufferAlreadyCommitted)
        );
    }

    #[test]
    fn executor_cannot_write_past_a_bound_buffer() {
        let device = device(true);
        let pipeline = pipeline(&device);
        let input = device.new_buffer_with_bytes([1, 2, 3, 4]).unwrap();
        let output = device.new_buffer(4).unwrap();
        let queue = device.new_command_queue();
        let command = queue.command_buffer();
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        encoder.set_buffer(0, &input, 0).unwrap();
        encoder.set_buffer(1, &output, 0).unwrap();
        encoder
            .dispatch_threads(Size::new(1, 1, 1).unwrap(), Size::new(1, 1, 1).unwrap())
            .unwrap();
        encoder.end_encoding().unwrap();
        assert!(matches!(
            command.commit(),
            Err(ApiError::BufferUpdateOutOfBounds { index: 1, .. })
        ));
        assert_eq!(command.status().unwrap(), CommandBufferStatus::Failed);
        assert_eq!(output.read().unwrap(), [0, 0, 0, 0]);
    }

    #[test]
    fn encoder_refuses_missing_pipeline_and_out_of_bounds_offsets() {
        let device = device(false);
        let buffer = device.new_buffer(4).unwrap();
        let queue = device.new_command_queue();
        let command = queue.command_buffer();
        let mut encoder = command.compute_command_encoder().unwrap();
        assert_eq!(
            encoder.set_buffer(0, &buffer, 5),
            Err(ApiError::BufferOffsetOutOfBounds {
                offset: 5,
                length: 4
            })
        );
        encoder
            .dispatch_threads(Size::new(1, 1, 1).unwrap(), Size::new(1, 1, 1).unwrap())
            .unwrap();
        assert_eq!(encoder.end_encoding(), Err(ApiError::MissingPipeline));
    }

    #[test]
    fn encoder_refuses_a_pipeline_without_a_dispatch() {
        let device = device(false);
        let pipeline = pipeline(&device);
        let queue = device.new_command_queue();
        let command = queue.command_buffer();
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        assert_eq!(encoder.end_encoding(), Err(ApiError::MissingDispatch));
    }

    #[test]
    fn executor_panic_transitions_to_failed_and_wakes_waiters() {
        let entered = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::new(std::sync::Barrier::new(2));
        let device = Device::new(Arc::new(PanicExecutor {
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
        }));
        let pipeline = pipeline(&device);
        let input = device.new_buffer_with_bytes([1, 2, 3, 4]).unwrap();
        let queue = device.new_command_queue();
        let command = Arc::new(queue.command_buffer());
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        encoder.set_buffer(0, &input, 0).unwrap();
        encoder
            .dispatch_threads(Size::new(1, 1, 1).unwrap(), Size::new(1, 1, 1).unwrap())
            .unwrap();
        encoder.end_encoding().unwrap();

        let committer = {
            let command = Arc::clone(&command);
            std::thread::spawn(move || command.commit())
        };
        entered.wait();
        assert_eq!(command.status().unwrap(), CommandBufferStatus::Committed);
        let waiter = {
            let command = Arc::clone(&command);
            std::thread::spawn(move || command.wait_until_completed())
        };
        release.wait();
        assert_eq!(committer.join().unwrap(), Err(ApiError::ExecutorPanicked));
        assert_eq!(command.status().unwrap(), CommandBufferStatus::Failed);
        assert_eq!(waiter.join().unwrap(), Err(ApiError::ExecutorPanicked));
    }

    #[test]
    fn pipeline_and_queue_must_belong_to_the_same_executor() {
        let first = device(false);
        let second = device(false);
        let pipeline = pipeline(&first);
        let queue = second.new_command_queue();
        let command = queue.command_buffer();
        let mut encoder = command.compute_command_encoder().unwrap();
        assert_eq!(
            encoder.set_compute_pipeline_state(&pipeline),
            Err(ApiError::ForeignPipeline)
        );
    }

    #[test]
    fn binding_one_buffer_at_two_indices_is_an_explicit_refusal() {
        let device = device(false);
        let buffer = device.new_buffer(8).unwrap();
        let queue = device.new_command_queue();
        let command = queue.command_buffer();
        let mut encoder = command.compute_command_encoder().unwrap();
        encoder.set_buffer(0, &buffer, 0).unwrap();
        assert_eq!(
            encoder.set_buffer(1, &buffer, 4),
            Err(ApiError::AliasedBufferBindings {
                first: 0,
                second: 1
            })
        );
    }

    #[test]
    fn binary_air_is_byte_exact_and_container_formats_are_refused() {
        let device = device(false);
        let raw = vec![0x42, 0x43, 0xc0, 0xde, 0xff, 0x80, 0x00];
        let library = device.new_library_with_binary_air(raw.clone()).unwrap();
        let function = library.function("copy_word").unwrap();
        assert_eq!(function.air_source(), AirSource::Binary(&raw));

        assert!(matches!(
            device.new_library_with_binary_air(Vec::<u8>::new()),
            Err(ApiError::EmptyLibrary)
        ));
        assert!(matches!(
            device.new_library_with_binary_air(b"MTLB synthetic".as_slice()),
            Err(ApiError::MetallibUnsupported)
        ));
        assert!(matches!(
            device.new_library_with_binary_air([0xde, 0xc0, 0x17, 0x0b]),
            Err(ApiError::InvalidBinaryAir)
        ));
        assert!(matches!(
            device.new_library_with_binary_air([0xff, 0x80, 0x00, 0x01]),
            Err(ApiError::InvalidBinaryAir)
        ));
    }

    #[test]
    fn offset_zero_air_wrapper_is_preserved() {
        let device = device(false);
        let mut wrapper = vec![0_u8; 0x18];
        wrapper[0..4].copy_from_slice(&[0xde, 0xc0, 0x17, 0x0b]);
        wrapper[8..12].copy_from_slice(&0x14_u32.to_le_bytes());
        wrapper[12..16].copy_from_slice(&4_u32.to_le_bytes());
        wrapper[0x14..].copy_from_slice(&[0x42, 0x43, 0xc0, 0xde]);
        let library = device.new_library_with_binary_air(wrapper.clone()).unwrap();
        let function = library.function("copy_word").unwrap();
        assert_eq!(function.air_source(), AirSource::Binary(&wrapper));

        wrapper.push(0);
        assert!(matches!(
            device.new_library_with_binary_air(wrapper),
            Err(ApiError::InvalidBinaryAir)
        ));
    }
}
