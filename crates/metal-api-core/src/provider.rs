//! Backend-neutral values for the first Metal provider contract.
//!
//! This module deliberately contains no `ash`, `metal`, QEMU type, guest
//! pointer, or provider-owned handle. It is the value boundary shared by a
//! native Metal implementation and a Vulkan implementation. The older
//! [`crate::ComputeExecutor`] snapshot API remains separate and is kept for
//! compatibility with the first offline harness.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

/// Current version of the pure-value provider trace schema.
pub const PROVIDER_SCHEMA_VERSION: u16 = 1;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(u64);

        impl $name {
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            pub const fn get(self) -> u64 {
                self.0
            }
        }
    };
}

id_type!(DeviceEpoch);
id_type!(OperationId);
id_type!(AllocationId);
id_type!(ViewId);
id_type!(LeaseId);
id_type!(PipelineId);
id_type!(SubmissionId);

/// A semantic module digest. The scheme is carried instead of being silently
/// fixed to SHA-256; the normalization algorithm is still an open contract
/// decision and must be versioned by its caller.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SemanticDigest {
    scheme: String,
    bytes: Vec<u8>,
}

impl SemanticDigest {
    pub fn new(
        scheme: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<Self, ContractError> {
        let scheme = scheme.into();
        let bytes = bytes.into();
        if scheme.trim().is_empty() {
            return Err(ContractError::EmptyField("digest scheme"));
        }
        if bytes.is_empty() {
            return Err(ContractError::EmptyField("digest bytes"));
        }
        Ok(Self { scheme, bytes })
    }

    pub fn scheme(&self) -> &str {
        &self.scheme
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// The provider-side representation received before compilation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FunctionSource {
    SanitizedLl,
    BinaryAir,
    Metallib,
}

/// Logical identity used to align native and Vulkan parity cases.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FunctionIdentity {
    pub logical_digest: SemanticDigest,
    pub entry_name: String,
    pub source: FunctionSource,
}

impl FunctionIdentity {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.entry_name.trim().is_empty() {
            return Err(ContractError::EmptyField("function entry name"));
        }
        Ok(())
    }
}

/// Metal access resolved by neutral reflection/decoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferAccess {
    Read,
    Write,
    ReadWrite,
    Unused,
}

impl BufferAccess {
    pub const fn is_writable(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

/// Owner-issued capability for a staged or borrowed backing allocation.
///
/// A completion token is deliberately not stored here: one lease can be
/// borrowed by several passes or command buffers. The owner binds all active
/// reservations to their completion tokens and releases the lease only after
/// the final reservation reaches a terminal state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferLease {
    pub lease_id: LeaseId,
    pub allocation_id: AllocationId,
    pub owner_epoch: DeviceEpoch,
}

/// How the caller supplies a buffer's initial contents.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BufferSource {
    /// Offline/test input owned by the trace. The provider may copy it.
    OwnedBytes(Vec<u8>),
    /// Contents and lifetime are supplied by an owner-issued staged lease.
    StagedLease(LeaseId),
    /// Provider may use the owner's backing without copying; the lease remains
    /// valid until every associated completion reservation is terminal.
    BorrowedNoCopy(LeaseId),
}

impl BufferSource {
    pub const fn lease_id(&self) -> Option<LeaseId> {
        match self {
            Self::OwnedBytes(_) => None,
            Self::StagedLease(lease_id) | Self::BorrowedNoCopy(lease_id) => Some(*lease_id),
        }
    }
}

/// A logical Metal buffer view. Offsets, lengths, and writeback ranges are
/// always bytes, and use the wire's wide integer width until a provider does a
/// checked narrowing conversion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferView {
    pub view_id: ViewId,
    pub metal_binding: u32,
    pub allocation_id: AllocationId,
    pub offset: u64,
    pub length: u64,
    pub access: BufferAccess,
    pub attribute_stride: Option<u64>,
    pub source: BufferSource,
}

impl BufferView {
    pub fn validate_shape(&self) -> Result<u64, ContractError> {
        if self.length == 0 {
            return Err(ContractError::ZeroLength("buffer view"));
        }
        self.offset
            .checked_add(self.length)
            .ok_or(ContractError::ArithmeticOverflow("buffer view range"))
    }

    pub fn validate_against_lease(&self, lease: BufferLease) -> Result<(), ContractError> {
        self.validate_shape()?;
        if self.source.lease_id() != Some(lease.lease_id)
            || self.allocation_id != lease.allocation_id
        {
            return Err(ContractError::LeaseMismatch {
                view: self.view_id,
                lease: lease.lease_id,
            });
        }
        Ok(())
    }
}

/// The two direct Metal launch forms. `Threadgroups` is retained in the value
/// model for the future extension, but B0 providers may refuse it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DispatchKind {
    ThreadsExact,
    Threadgroups,
}

/// Metal's encoder/segment-level dispatch policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DispatchType {
    Serial,
    Concurrent,
}

/// One direct dispatch in canonical Metal units. For `ThreadsExact`, `grid`
/// is a thread count; for `Threadgroups`, it is a group count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dispatch {
    pub kind: DispatchKind,
    pub grid: [u64; 3],
    pub threads_per_threadgroup: [u64; 3],
}

impl Dispatch {
    pub fn validate(&self) -> Result<(), ContractError> {
        for (axis, value) in self.grid.into_iter().enumerate() {
            if value == 0 {
                return Err(ContractError::ZeroDimension {
                    field: "dispatch grid",
                    axis,
                });
            }
        }
        for (axis, value) in self.threads_per_threadgroup.into_iter().enumerate() {
            if value == 0 {
                return Err(ContractError::ZeroDimension {
                    field: "threads per threadgroup",
                    axis,
                });
            }
        }
        Ok(())
    }
}

/// Minimum reflected binding contract needed before encoding a buffer pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferBindingContract {
    pub metal_binding: u32,
    pub access: BufferAccess,
}

/// Provider-admission metadata for one translated pipeline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineContract {
    pub dispatch_kind: DispatchKind,
    pub required_local_size: [u64; 3],
    pub push_constant_bytes: u32,
    pub buffer_bindings: Vec<BufferBindingContract>,
    pub shader_capabilities: Vec<String>,
    pub translator_revision: Option<SemanticDigest>,
}

impl PipelineContract {
    pub fn validate(&self) -> Result<(), ContractError> {
        for (axis, value) in self.required_local_size.into_iter().enumerate() {
            if value == 0 {
                return Err(ContractError::ZeroDimension {
                    field: "required local size",
                    axis,
                });
            }
        }
        let mut bindings = BTreeMap::new();
        for binding in &self.buffer_bindings {
            if bindings.insert(binding.metal_binding, ()).is_some() {
                return Err(ContractError::DuplicateBinding(binding.metal_binding));
            }
        }
        Ok(())
    }
}

/// One ordered buffer-compute pass in a command buffer trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputePass {
    pub pipeline: PipelineId,
    pub buffers: Vec<BufferView>,
    pub dispatch: Dispatch,
}

impl ComputePass {
    pub fn validate(&self, pipeline_contract: &PipelineContract) -> Result<(), ContractError> {
        self.dispatch.validate()?;
        if self.dispatch.kind != pipeline_contract.dispatch_kind {
            return Err(ContractError::DispatchKindMismatch {
                expected: pipeline_contract.dispatch_kind,
                actual: self.dispatch.kind,
            });
        }
        let mut bindings = BTreeMap::new();
        let mut views = BTreeMap::new();
        for buffer in &self.buffers {
            buffer.validate_shape()?;
            if bindings.insert(buffer.metal_binding, ()).is_some() {
                return Err(ContractError::DuplicateBinding(buffer.metal_binding));
            }
            if views.insert(buffer.view_id, ()).is_some() {
                return Err(ContractError::DuplicateView(buffer.view_id));
            }
        }
        for reflected in &pipeline_contract.buffer_bindings {
            let Some(actual) = self
                .buffers
                .iter()
                .find(|buffer| buffer.metal_binding == reflected.metal_binding)
            else {
                return Err(ContractError::MissingBinding(reflected.metal_binding));
            };
            if actual.access != reflected.access {
                return Err(ContractError::AccessMismatch {
                    binding: reflected.metal_binding,
                    expected: reflected.access,
                    actual: actual.access,
                });
            }
        }
        for actual in &self.buffers {
            if !pipeline_contract
                .buffer_bindings
                .iter()
                .any(|reflected| reflected.metal_binding == actual.metal_binding)
            {
                return Err(ContractError::UnknownBinding(actual.metal_binding));
            }
        }
        Ok(())
    }
}

/// v0 output policy. Guest-page landing is intentionally not a v0 value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionPolicy {
    HostReadback,
    SubmitOnly,
}

/// The immutable value trace shared by both provider implementations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeTrace {
    pub schema_version: u16,
    pub device_epoch: DeviceEpoch,
    pub operation_id: OperationId,
    pub function: FunctionIdentity,
    pub pipeline_contract: PipelineContract,
    pub encoder_dispatch_type: DispatchType,
    pub passes: Vec<ComputePass>,
    pub completion_policy: CompletionPolicy,
}

impl ComputeTrace {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != PROVIDER_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchemaVersion(self.schema_version));
        }
        self.function.validate()?;
        self.pipeline_contract.validate()?;
        if self.passes.is_empty() {
            return Err(ContractError::EmptyTrace);
        }
        if self.encoder_dispatch_type != DispatchType::Serial {
            return Err(ContractError::UnsupportedDispatchType(
                self.encoder_dispatch_type,
            ));
        }
        let first_pipeline = self.passes[0].pipeline;
        for pass in &self.passes {
            if pass.pipeline != first_pipeline {
                return Err(ContractError::MixedPipelines);
            }
            pass.validate(&self.pipeline_contract)?;
        }
        Ok(())
    }
}

/// Opaque identity returned by a successful submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionToken {
    pub submission_id: SubmissionId,
}

/// Explicit completion observation. A timeout is non-terminal and may be
/// followed by another wait on the same token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionDisposition {
    NotSubmitted,
    Submitted { token: CompletionToken },
    CompletedVisible { token: CompletionToken },
    TimedOut { token: CompletionToken },
    Failed { token: Option<CompletionToken> },
    DeviceLost { token: Option<CompletionToken> },
    SubmittedUnknown { token: Option<CompletionToken> },
}

impl CompletionDisposition {
    pub const fn token(self) -> Option<CompletionToken> {
        match self {
            Self::NotSubmitted => None,
            Self::Submitted { token }
            | Self::CompletedVisible { token }
            | Self::TimedOut { token } => Some(token),
            Self::Failed { token }
            | Self::DeviceLost { token }
            | Self::SubmittedUnknown { token } => token,
        }
    }

    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::CompletedVisible { .. }
                | Self::Failed { .. }
                | Self::DeviceLost { .. }
                | Self::SubmittedUnknown { .. }
        )
    }
}

/// A deterministic provider writeback keyed by view and allocation identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferWriteback {
    pub view_id: ViewId,
    pub allocation_id: AllocationId,
    pub offset: u64,
    pub bytes: Vec<u8>,
}

impl BufferWriteback {
    pub fn end(&self) -> Result<u64, ContractError> {
        self.offset
            .checked_add(self.bytes.len() as u64)
            .ok_or(ContractError::ArithmeticOverflow("writeback range"))
    }
}

/// Capabilities are captured once for a provider device context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderCapabilities {
    pub supports_threads_exact: bool,
    pub supports_threadgroups: bool,
    pub supports_serial: bool,
    pub supports_concurrent: bool,
    pub max_local_size: [u64; 3],
    pub max_invocations: u64,
    pub max_group_count: [u64; 3],
    pub max_storage_buffer_descriptors: u32,
    pub max_buffer_range: u64,
    pub alias_mode: AliasMode,
    pub storage_modes: Vec<StorageMode>,
    pub host_readback: bool,
    pub submit_only: bool,
}

impl ProviderCapabilities {
    /// Admit a complete value trace without creating provider objects.
    ///
    /// Structural errors are reported as an `Args` refusal; selected-device
    /// limits and unsupported storage/completion modes are reported as
    /// `Capability` refusals. A provider implementation can perform the same
    /// checks immediately before encode, while keeping Vulkan/Metal handles
    /// out of the neutral contract.
    pub fn admit(&self, trace: &ComputeTrace) -> Result<(), ProviderError> {
        trace.validate().map_err(|error| {
            ProviderError::new(
                ProviderPhase::Resolve,
                ProviderErrorClass::Args,
                "trace_contract_invalid",
            )
            .expect("static provider refusal slug")
            .with_detail(error.to_string())
        })?;

        if !self.supports_serial || trace.encoder_dispatch_type != DispatchType::Serial {
            return Err(capability_error("dispatch_type_unsupported")
                .with_detail("B0 admits only serial encoder dispatch"));
        }
        if !self.supports_concurrent && trace.encoder_dispatch_type == DispatchType::Concurrent {
            return Err(capability_error("dispatch_type_unsupported"));
        }
        match trace.pipeline_contract.dispatch_kind {
            DispatchKind::ThreadsExact if !self.supports_threads_exact => {
                return Err(capability_error("dispatch_kind_unsupported"));
            }
            DispatchKind::Threadgroups if !self.supports_threadgroups => {
                return Err(capability_error("dispatch_kind_unsupported"));
            }
            _ => {}
        }
        match trace.completion_policy {
            CompletionPolicy::HostReadback if !self.host_readback => {
                return Err(capability_error("host_readback_unsupported"));
            }
            CompletionPolicy::SubmitOnly if !self.submit_only => {
                return Err(capability_error("submit_only_unsupported"));
            }
            _ => {}
        }

        for pass in &trace.passes {
            let local = pass.dispatch.threads_per_threadgroup;
            for (axis, requested) in local.into_iter().enumerate() {
                if requested > self.max_local_size[axis] {
                    return Err(capability_error("dispatch_local_size_limit")
                        .with_field("axis", FieldValue::Unsigned(axis as u64))
                        .with_field("requested", FieldValue::Unsigned(requested))
                        .with_field("maximum", FieldValue::Unsigned(self.max_local_size[axis])));
                }
            }
            let invocations = local
                .into_iter()
                .try_fold(1_u64, |total, value| total.checked_mul(value));
            let Some(invocations) = invocations else {
                return Err(capability_error("dispatch_invocation_overflow"));
            };
            if invocations > self.max_invocations {
                return Err(capability_error("dispatch_invocation_limit")
                    .with_field("requested", FieldValue::Unsigned(invocations))
                    .with_field("maximum", FieldValue::Unsigned(self.max_invocations)));
            }

            let groups = match pass.dispatch.kind {
                DispatchKind::Threadgroups => pass.dispatch.grid,
                DispatchKind::ThreadsExact => {
                    let mut groups = [0; 3];
                    for axis in 0..3 {
                        groups[axis] = ceil_div(pass.dispatch.grid[axis], local[axis])
                            .ok_or_else(|| capability_error("dispatch_group_count_overflow"))?;
                    }
                    groups
                }
            };
            for (axis, requested) in groups.into_iter().enumerate() {
                if requested > self.max_group_count[axis] {
                    return Err(capability_error("dispatch_group_count_limit")
                        .with_field("axis", FieldValue::Unsigned(axis as u64))
                        .with_field("requested", FieldValue::Unsigned(requested))
                        .with_field("maximum", FieldValue::Unsigned(self.max_group_count[axis])));
                }
            }

            if pass.buffers.len() > self.max_storage_buffer_descriptors as usize {
                return Err(capability_error("storage_buffer_descriptor_limit")
                    .with_field("requested", FieldValue::Unsigned(pass.buffers.len() as u64))
                    .with_field(
                        "maximum",
                        FieldValue::Unsigned(self.max_storage_buffer_descriptors as u64),
                    ));
            }
            let mut allocations = BTreeMap::new();
            for buffer in &pass.buffers {
                if buffer.length > self.max_buffer_range {
                    return Err(capability_error("storage_buffer_range_limit")
                        .with_field("binding", FieldValue::Unsigned(buffer.metal_binding as u64))
                        .with_field("requested", FieldValue::Unsigned(buffer.length))
                        .with_field("maximum", FieldValue::Unsigned(self.max_buffer_range)));
                }
                let storage_mode = match &buffer.source {
                    BufferSource::OwnedBytes(_) => StorageMode::OwnedBytes,
                    BufferSource::StagedLease(_) => StorageMode::StagedLease,
                    BufferSource::BorrowedNoCopy(_) => StorageMode::BorrowedNoCopy,
                };
                if !self.storage_modes.contains(&storage_mode) {
                    return Err(capability_error("storage_mode_unsupported")
                        .with_field("binding", FieldValue::Unsigned(buffer.metal_binding as u64)));
                }
                if self.alias_mode != AliasMode::DistinctViews
                    && allocations
                        .insert(buffer.allocation_id, buffer.view_id)
                        .is_some()
                {
                    return Err(capability_error("buffer_alias_unsupported")
                        .with_field("binding", FieldValue::Unsigned(buffer.metal_binding as u64)));
                }
            }
        }
        Ok(())
    }
}

fn ceil_div(value: u64, divisor: u64) -> Option<u64> {
    value
        .checked_add(divisor.checked_sub(1)?)?
        .checked_div(divisor)
}

fn capability_error(slug: &'static str) -> ProviderError {
    ProviderError::new(ProviderPhase::Resolve, ProviderErrorClass::Capability, slug)
        .expect("static provider refusal slug")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AliasMode {
    Refused,
    DistinctViews,
    ExplicitPolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageMode {
    OwnedBytes,
    StagedLease,
    BorrowedNoCopy,
}

/// Stable phase of a provider refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderPhase {
    Resolve,
    Compile,
    Encode,
    Submit,
    Wait,
    Readback,
}

/// Normalized refusal class. Provider-specific text is kept in `detail` only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderErrorClass {
    Args,
    Capability,
    Resource,
    Compile,
    Execute,
    DeviceLost,
    Internal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Retryability {
    Never,
    RetrySameTrace,
    RetryAfterRecreate,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FieldValue {
    Unsigned(u64),
    Signed(i64),
    Bool(bool),
    Text(String),
}

/// Structured error crossing the provider boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderError {
    pub phase: ProviderPhase,
    pub class: ProviderErrorClass,
    pub slug: String,
    pub fields: BTreeMap<String, FieldValue>,
    pub retryability: Retryability,
    pub completion: CompletionDisposition,
    pub detail: Option<String>,
}

impl ProviderError {
    pub fn new(
        phase: ProviderPhase,
        class: ProviderErrorClass,
        slug: impl Into<String>,
    ) -> Result<Self, ContractError> {
        let slug = slug.into();
        if slug.trim().is_empty() {
            return Err(ContractError::EmptyField("provider error slug"));
        }
        Ok(Self {
            phase,
            class,
            slug,
            fields: BTreeMap::new(),
            retryability: Retryability::Unknown,
            completion: CompletionDisposition::NotSubmitted,
            detail: None,
        })
    }

    pub fn with_field(mut self, key: impl Into<String>, value: FieldValue) -> Self {
        self.fields.insert(key.into(), value);
        self
    }

    pub fn with_completion(mut self, completion: CompletionDisposition) -> Self {
        self.completion = completion;
        self
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// Errors found before a provider has any opportunity to submit work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContractError {
    EmptyField(&'static str),
    ZeroLength(&'static str),
    ZeroDimension {
        field: &'static str,
        axis: usize,
    },
    ArithmeticOverflow(&'static str),
    LeaseMismatch {
        view: ViewId,
        lease: LeaseId,
    },
    DuplicateBinding(u32),
    MissingBinding(u32),
    UnknownBinding(u32),
    AccessMismatch {
        binding: u32,
        expected: BufferAccess,
        actual: BufferAccess,
    },
    DuplicateView(ViewId),
    EmptyTrace,
    UnsupportedSchemaVersion(u16),
    MixedPipelines,
    UnsupportedDispatchType(DispatchType),
    DispatchKindMismatch {
        expected: DispatchKind,
        actual: DispatchKind,
    },
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(formatter, "{field} must not be empty"),
            Self::ZeroLength(field) => write!(formatter, "{field} length must be non-zero"),
            Self::ZeroDimension { field, axis } => {
                write!(formatter, "{field} dimension {axis} must be non-zero")
            }
            Self::ArithmeticOverflow(field) => write!(formatter, "{field} overflows u64"),
            Self::LeaseMismatch { view, lease } => {
                write!(
                    formatter,
                    "view {:?} does not match lease {:?}",
                    view, lease
                )
            }
            Self::DuplicateBinding(binding) => {
                write!(formatter, "duplicate Metal binding {binding}")
            }
            Self::MissingBinding(binding) => write!(formatter, "missing Metal binding {binding}"),
            Self::UnknownBinding(binding) => write!(formatter, "unknown Metal binding {binding}"),
            Self::AccessMismatch {
                binding,
                expected,
                actual,
            } => write!(
                formatter,
                "Metal binding {binding} access mismatch: expected {expected:?}, received {actual:?}"
            ),
            Self::DuplicateView(view) => write!(formatter, "duplicate view {:?}", view),
            Self::EmptyTrace => formatter.write_str("compute trace must contain a pass"),
            Self::UnsupportedSchemaVersion(version) => {
                write!(formatter, "unsupported provider trace schema version {version}")
            }
            Self::MixedPipelines => {
                formatter.write_str("v0 compute trace cannot mix pipeline identities")
            }
            Self::UnsupportedDispatchType(dispatch_type) => {
                write!(
                    formatter,
                    "dispatch type {dispatch_type:?} is not supported by B0"
                )
            }
            Self::DispatchKindMismatch { expected, actual } => write!(
                formatter,
                "dispatch kind mismatch: expected {expected:?}, received {actual:?}"
            ),
        }
    }
}

impl Error for ContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest() -> SemanticDigest {
        SemanticDigest::new("test-v1", [7, 3, 1]).expect("non-empty digest")
    }

    fn buffer(view_id: u64, binding: u32) -> BufferView {
        BufferView {
            view_id: ViewId::new(view_id),
            metal_binding: binding,
            allocation_id: AllocationId::new(9),
            offset: 0,
            length: 4,
            access: BufferAccess::Write,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0; 4]),
        }
    }

    fn trace(passes: Vec<ComputePass>) -> ComputeTrace {
        ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(1),
            operation_id: OperationId::new(2),
            function: FunctionIdentity {
                logical_digest: digest(),
                entry_name: "copy_word".to_string(),
                source: FunctionSource::BinaryAir,
            },
            pipeline_contract: PipelineContract {
                dispatch_kind: DispatchKind::ThreadsExact,
                required_local_size: [1, 1, 1],
                push_constant_bytes: 0,
                buffer_bindings: vec![BufferBindingContract {
                    metal_binding: 0,
                    access: BufferAccess::Write,
                }],
                shader_capabilities: Vec::new(),
                translator_revision: None,
            },
            encoder_dispatch_type: DispatchType::Serial,
            passes,
            completion_policy: CompletionPolicy::HostReadback,
        }
    }

    fn pass(pipeline: u64, buffers: Vec<BufferView>) -> ComputePass {
        ComputePass {
            pipeline: PipelineId::new(pipeline),
            buffers,
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [10, 3, 1],
                threads_per_threadgroup: [8, 2, 1],
            },
        }
    }

    fn capabilities() -> ProviderCapabilities {
        ProviderCapabilities {
            supports_threads_exact: true,
            supports_threadgroups: false,
            supports_serial: true,
            supports_concurrent: false,
            max_local_size: [8, 8, 8],
            max_invocations: 64,
            max_group_count: [16, 16, 16],
            max_storage_buffer_descriptors: 4,
            max_buffer_range: 4096,
            alias_mode: AliasMode::Refused,
            storage_modes: vec![StorageMode::OwnedBytes],
            host_readback: true,
            submit_only: true,
        }
    }

    #[test]
    fn trace_accepts_ordered_passes_with_one_pipeline() {
        let result = trace(vec![
            pass(4, vec![buffer(1, 0)]),
            pass(4, vec![buffer(2, 0)]),
        ]);
        assert!(result.validate().is_ok());
    }

    #[test]
    fn trace_rejects_mixed_pipelines_and_duplicate_bindings() {
        let mixed = trace(vec![
            pass(4, vec![buffer(1, 0)]),
            pass(5, vec![buffer(2, 0)]),
        ]);
        assert_eq!(mixed.validate(), Err(ContractError::MixedPipelines));

        let duplicate = trace(vec![pass(4, vec![buffer(1, 0), buffer(2, 0)])]);
        assert_eq!(
            duplicate.validate(),
            Err(ContractError::DuplicateBinding(0))
        );
    }

    #[test]
    fn trace_rejects_missing_unknown_and_mismatched_access_bindings() {
        let missing = trace(vec![pass(4, Vec::new())]);
        assert_eq!(missing.validate(), Err(ContractError::MissingBinding(0)));

        let unknown = trace(vec![pass(4, vec![buffer(1, 0), buffer(2, 1)])]);
        assert_eq!(unknown.validate(), Err(ContractError::UnknownBinding(1)));

        let mut mismatched_trace = trace(vec![pass(4, vec![buffer(1, 0)])]);
        mismatched_trace.passes[0].buffers[0].access = BufferAccess::Read;
        assert_eq!(
            mismatched_trace.validate(),
            Err(ContractError::AccessMismatch {
                binding: 0,
                expected: BufferAccess::Write,
                actual: BufferAccess::Read,
            })
        );
    }

    #[test]
    fn trace_rejects_unknown_schema_version() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        value.schema_version = PROVIDER_SCHEMA_VERSION + 1;
        assert_eq!(
            value.validate(),
            Err(ContractError::UnsupportedSchemaVersion(
                PROVIDER_SCHEMA_VERSION + 1
            ))
        );
    }

    #[test]
    fn capabilities_admit_a_bounded_serial_trace() {
        let value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        assert!(capabilities().admit(&value).is_ok());
    }

    #[test]
    fn capabilities_report_limits_and_aliases_structurally() {
        let mut too_wide = trace(vec![pass(4, vec![buffer(1, 0)])]);
        too_wide.passes[0].dispatch.threads_per_threadgroup = [9, 1, 1];
        let error = capabilities().admit(&too_wide).unwrap_err();
        assert_eq!(error.slug, "dispatch_local_size_limit");
        assert_eq!(
            error.fields.get("requested"),
            Some(&FieldValue::Unsigned(9))
        );

        let mut alias = trace(vec![pass(
            4,
            vec![
                buffer(1, 0),
                BufferView {
                    metal_binding: 1,
                    ..buffer(2, 0)
                },
            ],
        )]);
        alias
            .pipeline_contract
            .buffer_bindings
            .push(BufferBindingContract {
                metal_binding: 1,
                access: BufferAccess::Write,
            });
        let error = capabilities().admit(&alias).unwrap_err();
        assert_eq!(error.slug, "buffer_alias_unsupported");
    }

    #[test]
    fn capabilities_refuse_a_future_dispatch_kind_without_guessing() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        value.pipeline_contract.dispatch_kind = DispatchKind::Threadgroups;
        value.passes[0].dispatch.kind = DispatchKind::Threadgroups;
        let error = capabilities().admit(&value).unwrap_err();
        assert_eq!(error.slug, "dispatch_kind_unsupported");
    }

    #[test]
    fn wide_ranges_are_checked_before_provider_narrowing() {
        let view = BufferView {
            offset: u64::MAX,
            length: 1,
            ..buffer(1, 0)
        };
        assert_eq!(
            view.validate_shape(),
            Err(ContractError::ArithmeticOverflow("buffer view range"))
        );

        let dispatch = Dispatch {
            kind: DispatchKind::ThreadsExact,
            grid: [u64::from(u32::MAX) + 1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
        };
        assert!(dispatch.validate().is_ok());
    }

    #[test]
    fn lease_identity_is_separate_from_view_identity() {
        let lease = BufferLease {
            lease_id: LeaseId::new(8),
            allocation_id: AllocationId::new(9),
            owner_epoch: DeviceEpoch::new(1),
        };
        let view = BufferView {
            view_id: ViewId::new(3),
            metal_binding: 0,
            allocation_id: AllocationId::new(9),
            offset: 4,
            length: 8,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::BorrowedNoCopy(LeaseId::new(8)),
        };
        assert!(view.validate_against_lease(lease).is_ok());
        assert_eq!(
            view.validate_against_lease(BufferLease {
                lease_id: LeaseId::new(7),
                ..lease
            }),
            Err(ContractError::LeaseMismatch {
                view: ViewId::new(3),
                lease: LeaseId::new(7),
            })
        );
    }

    #[test]
    fn completion_distinguishes_timeout_from_terminal_unknown() {
        let token = CompletionToken {
            submission_id: SubmissionId::new(12),
        };
        assert!(!CompletionDisposition::TimedOut { token }.is_terminal());
        assert!(CompletionDisposition::SubmittedUnknown { token: Some(token) }.is_terminal());
        assert_eq!(
            CompletionDisposition::TimedOut { token }.token(),
            Some(token)
        );
    }

    #[test]
    fn provider_error_fields_are_structured_and_ordered() {
        let error = ProviderError::new(
            ProviderPhase::Encode,
            ProviderErrorClass::Capability,
            "dispatch_dimension_overflow",
        )
        .unwrap()
        .with_field("maximum", FieldValue::Unsigned(u64::from(u32::MAX)))
        .with_field("dimension", FieldValue::Unsigned(0))
        .with_completion(CompletionDisposition::NotSubmitted);
        let keys = error.fields.keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys, ["dimension", "maximum"]);
        assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
    }
}
