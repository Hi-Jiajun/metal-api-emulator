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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

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

            pub const fn is_zero(self) -> bool {
                self.0 == 0
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

static NEXT_DEVICE_EPOCH: AtomicU64 = AtomicU64::new(1);

/// Allocate a nonzero identity shared by all provider implementations in this
/// process. Providers must use this allocator instead of choosing their own
/// epoch or maintaining a separate counter. Epochs are never reused, including
/// after a provider is dropped; serialized identities are not portable across
/// processes. Exhaustion refuses device creation without wrapping the counter.
pub fn allocate_device_epoch() -> Result<DeviceEpoch, ProviderError> {
    NEXT_DEVICE_EPOCH
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map(DeviceEpoch::new)
        .map_err(|_| {
            let mut error = ProviderError::new(
                ProviderPhase::Resolve,
                ProviderErrorClass::Internal,
                "device_epoch_exhausted",
            )
            .expect("non-empty epoch exhaustion slug");
            error.retryability = Retryability::Never;
            error
        })
}

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
    MetalSource,
    Metallib,
}

/// Backend input for the bounded compute compilation boundary. Providers may
/// refuse a source representation they do not support with a typed capability
/// error; accepting a representation does not imply support for all shaders.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShaderSource {
    SanitizedLl(String),
    BinaryAir(Vec<u8>),
    MetalSource(String),
}

impl ShaderSource {
    pub const fn kind(&self) -> FunctionSource {
        match self {
            Self::SanitizedLl(_) => FunctionSource::SanitizedLl,
            Self::BinaryAir(_) => FunctionSource::BinaryAir,
            Self::MetalSource(_) => FunctionSource::MetalSource,
        }
    }
}

/// Compile one entry point without exposing a backend artifact. The digest is
/// supplied by the caller to align parity fixtures; it is not a cache key or a
/// proof that differently represented source modules are equivalent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineCompileRequest {
    pub entry_name: String,
    pub logical_digest: SemanticDigest,
    pub source: ShaderSource,
}

impl PipelineCompileRequest {
    /// Check only common shape requirements. The backend validates binary
    /// containers, source syntax and its supported compute subset.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.entry_name.trim().is_empty() {
            return Err(ContractError::EmptyField("function entry name"));
        }
        let empty = match &self.source {
            ShaderSource::SanitizedLl(source) | ShaderSource::MetalSource(source) => {
                source.trim().is_empty()
            }
            ShaderSource::BinaryAir(source) => source.is_empty(),
        };
        if empty {
            return Err(ContractError::EmptyField("shader source"));
        }
        Ok(())
    }
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
/// reservations to their completion tokens and releases the backing only after
/// every reservation is known to be retired on the GPU. A terminal result such
/// as `SubmittedUnknown` is not evidence of retirement; it cannot release the
/// backing without a separate retirement or device teardown guarantee.
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
    /// valid until every associated GPU reservation is known to be retired.
    BorrowedNoCopy(LeaseId),
}

impl BufferSource {
    pub const fn lease_id(&self) -> Option<LeaseId> {
        match self {
            Self::OwnedBytes(_) => None,
            Self::StagedLease(lease_id) | Self::BorrowedNoCopy(lease_id) => Some(*lease_id),
        }
    }

    pub const fn kind(&self) -> BufferSourceKind {
        match self {
            Self::OwnedBytes(_) => BufferSourceKind::OwnedBytes,
            Self::StagedLease(_) => BufferSourceKind::StagedLease,
            Self::BorrowedNoCopy(_) => BufferSourceKind::BorrowedNoCopy,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferSourceKind {
    OwnedBytes,
    StagedLease,
    BorrowedNoCopy,
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
        if self.view_id.is_zero() {
            return Err(ContractError::InvalidIdentity("buffer view id"));
        }
        if self.allocation_id.is_zero() {
            return Err(ContractError::InvalidIdentity("allocation id"));
        }
        if self.attribute_stride.is_some() {
            return Err(ContractError::UnsupportedAttributeStride);
        }
        if self.length == 0 {
            return Err(ContractError::ZeroLength("buffer view"));
        }
        let end = self
            .offset
            .checked_add(self.length)
            .ok_or(ContractError::ArithmeticOverflow("buffer view range"))?;
        if let BufferSource::OwnedBytes(bytes) = &self.source {
            let actual = u64::try_from(bytes.len())
                .map_err(|_| ContractError::ArithmeticOverflow("owned buffer length"))?;
            if actual != self.length {
                return Err(ContractError::SourceLengthMismatch {
                    view: self.view_id,
                    expected: self.length,
                    actual,
                });
            }
        }
        Ok(end)
    }

    pub fn validate_against_lease(
        &self,
        lease: BufferLease,
        expected_epoch: DeviceEpoch,
    ) -> Result<(), ContractError> {
        self.validate_shape()?;
        if lease.lease_id.is_zero() || lease.allocation_id.is_zero() {
            return Err(ContractError::InvalidIdentity("lease identity"));
        }
        if expected_epoch.is_zero() {
            return Err(ContractError::InvalidIdentity("expected lease epoch"));
        }
        if lease.owner_epoch != expected_epoch {
            return Err(ContractError::LeaseEpochMismatch {
                lease: lease.lease_id,
                expected: expected_epoch,
                actual: lease.owner_epoch,
            });
        }
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferBindingContract {
    pub metal_binding: u32,
    pub access: BufferAccess,
    pub footprint: FootprintProof,
}

/// Normalized provider-admission proof for one buffer's reachable bytes.
///
/// One normalized affine byte access relative to a buffer view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AffineAccess {
    pub base_offset: u64,
    pub access_size: u64,
    pub terms: Vec<AffineTerm>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AffineTerm {
    pub axis: u8,
    pub stride: u64,
}

/// Normalized provider-admission proof for one buffer's reachable bytes.
///
/// `Affine` carries the bounded index expression so admission can evaluate it
/// against each pass's dispatch. It is still not a parity identity: native and
/// Vulkan providers may serialize the proof differently while producing the
/// same observable writes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FootprintProof {
    Static { max_bytes: u64 },
    Affine { accesses: Vec<AffineAccess> },
    Unbounded,
}

/// Provider-admission metadata for one translated pipeline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineContract {
    pub dispatch_kind: DispatchKind,
    /// `None` means the exact-thread provider chooses the local size per pass;
    /// `Some` is a fixed local-size contract that every pass must match.
    pub required_local_size: Option<[u64; 3]>,
    /// A fixed exact-thread grid, when the translated module baked one into
    /// its contract. `None` means the grid is selected at dispatch time.
    pub fixed_grid: Option<[u64; 3]>,
    /// Offset, in bytes, of the logical dispatch payload in the argument or
    /// push-constant area.
    pub push_constant_offset: u32,
    pub push_constant_bytes: u32,
    pub buffer_bindings: Vec<BufferBindingContract>,
    pub shader_capabilities: Vec<String>,
    pub translator_revision: Option<SemanticDigest>,
}

/// Neutral metadata for a pipeline registered by one provider context.
/// The provider retains the actual artifact and checks this metadata at submit.
/// This value neither owns the backend pipeline nor permits transfer between
/// provider contexts; release is explicit through [`PipelineProvider`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledComputePipeline {
    pub device_epoch: DeviceEpoch,
    pub pipeline_id: PipelineId,
    pub function: FunctionIdentity,
    pub contract: PipelineContract,
}

impl PipelineContract {
    pub fn validate(&self) -> Result<(), ContractError> {
        if let Some(local_size) = self.required_local_size {
            for (axis, value) in local_size.into_iter().enumerate() {
                if value == 0 {
                    return Err(ContractError::ZeroDimension {
                        field: "required local size",
                        axis,
                    });
                }
            }
        }
        if let Some(fixed_grid) = self.fixed_grid {
            if self.dispatch_kind != DispatchKind::ThreadsExact {
                return Err(ContractError::FixedGridRequiresExactDispatch);
            }
            for (axis, value) in fixed_grid.into_iter().enumerate() {
                if value == 0 {
                    return Err(ContractError::ZeroDimension {
                        field: "fixed dispatch grid",
                        axis,
                    });
                }
            }
        }
        if !self.push_constant_offset.is_multiple_of(4) {
            return Err(ContractError::MisalignedPushConstantOffset(
                self.push_constant_offset,
            ));
        }
        self.push_constant_offset
            .checked_add(self.push_constant_bytes)
            .ok_or(ContractError::ArithmeticOverflow("push constant range"))?;
        let mut bindings = BTreeMap::new();
        let mut previous_binding = None;
        for binding in &self.buffer_bindings {
            if previous_binding.is_some_and(|previous| previous > binding.metal_binding) {
                return Err(ContractError::NonCanonicalBindingOrder("pipeline contract"));
            }
            previous_binding = Some(binding.metal_binding);
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
        if self.pipeline.is_zero() {
            return Err(ContractError::InvalidIdentity("pipeline id"));
        }
        self.dispatch.validate()?;
        if self.dispatch.kind != pipeline_contract.dispatch_kind {
            return Err(ContractError::DispatchKindMismatch {
                expected: pipeline_contract.dispatch_kind,
                actual: self.dispatch.kind,
            });
        }
        let mut bindings = BTreeMap::new();
        let mut views = BTreeMap::new();
        let mut previous_binding = None;
        for buffer in &self.buffers {
            buffer.validate_shape()?;
            if previous_binding.is_some_and(|previous| previous > buffer.metal_binding) {
                return Err(ContractError::NonCanonicalBindingOrder("compute pass"));
            }
            previous_binding = Some(buffer.metal_binding);
            if bindings.insert(buffer.metal_binding, ()).is_some() {
                return Err(ContractError::DuplicateBinding(buffer.metal_binding));
            }
            if views.insert(buffer.view_id, ()).is_some() {
                return Err(ContractError::DuplicateView(buffer.view_id));
            }
        }
        if let Some(required_local_size) = pipeline_contract.required_local_size {
            let actual = self.dispatch.threads_per_threadgroup;
            if actual != required_local_size {
                return Err(ContractError::LocalSizeMismatch {
                    expected: required_local_size,
                    actual,
                });
            }
        }
        if let Some(fixed_grid) = pipeline_contract.fixed_grid {
            if self.dispatch.grid != fixed_grid {
                return Err(ContractError::GridMismatch {
                    expected: fixed_grid,
                    actual: self.dispatch.grid,
                });
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

/// A trace and resource snapshot that have passed all structural, capability,
/// range, lease, and alias checks. The fields are private so callers cannot
/// mutate the values between admission and a provider's encode step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidatedComputeTrace {
    trace: ComputeTrace,
    resources: ResourceTableSnapshot,
}

impl ValidatedComputeTrace {
    pub fn trace(&self) -> &ComputeTrace {
        &self.trace
    }

    pub fn resources(&self) -> &ResourceTableSnapshot {
        &self.resources
    }

    pub fn into_parts(self) -> (ComputeTrace, ResourceTableSnapshot) {
        (self.trace, self.resources)
    }
}

/// One allocation namespace record owned by neutral memory management.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationRecord {
    pub allocation_id: AllocationId,
    pub owner_epoch: DeviceEpoch,
    pub size: u64,
}

/// A lease reservation covering a range of one allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LeaseReservation {
    pub lease: BufferLease,
    pub offset: u64,
    pub length: u64,
}

impl LeaseReservation {
    pub fn end(&self) -> Result<u64, ContractError> {
        if self.length == 0 {
            return Err(ContractError::ZeroLength("lease reservation"));
        }
        self.offset
            .checked_add(self.length)
            .ok_or(ContractError::ArithmeticOverflow("lease reservation range"))
    }
}

/// Immutable resource namespace snapshot used during admission. It contains
/// identities and bounds only; it never owns guest pointers/provider handles,
/// and it does not implement retire/revoke or completion-time lease holding.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResourceTableSnapshot {
    allocations: BTreeMap<AllocationId, AllocationRecord>,
    leases: BTreeMap<LeaseId, LeaseReservation>,
}

impl ResourceTableSnapshot {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_allocation(&mut self, record: AllocationRecord) -> Result<(), ContractError> {
        if record.allocation_id.is_zero() {
            return Err(ContractError::InvalidIdentity("allocation id"));
        }
        if record.owner_epoch.is_zero() {
            return Err(ContractError::InvalidIdentity("allocation owner epoch"));
        }
        if record.size == 0 {
            return Err(ContractError::ZeroLength("allocation"));
        }
        if self.allocations.contains_key(&record.allocation_id) {
            return Err(ContractError::DuplicateAllocation(record.allocation_id));
        }
        self.allocations.insert(record.allocation_id, record);
        Ok(())
    }

    pub fn insert_lease(&mut self, reservation: LeaseReservation) -> Result<(), ContractError> {
        if reservation.lease.lease_id.is_zero() {
            return Err(ContractError::InvalidIdentity("lease id"));
        }
        if reservation.lease.allocation_id.is_zero() {
            return Err(ContractError::InvalidIdentity("lease allocation id"));
        }
        if reservation.lease.owner_epoch.is_zero() {
            return Err(ContractError::InvalidIdentity("lease owner epoch"));
        }
        let allocation = self
            .allocations
            .get(&reservation.lease.allocation_id)
            .ok_or(ContractError::UnknownAllocation(
                reservation.lease.allocation_id,
            ))?;
        if allocation.owner_epoch != reservation.lease.owner_epoch {
            return Err(ContractError::LeaseEpochMismatch {
                lease: reservation.lease.lease_id,
                expected: allocation.owner_epoch,
                actual: reservation.lease.owner_epoch,
            });
        }
        let end = reservation.end()?;
        if end > allocation.size {
            return Err(ContractError::LeaseRangeOutOfBounds {
                lease: reservation.lease.lease_id,
                end,
                allocation_size: allocation.size,
            });
        }
        if self.leases.contains_key(&reservation.lease.lease_id) {
            return Err(ContractError::DuplicateLease(reservation.lease.lease_id));
        }
        self.leases.insert(reservation.lease.lease_id, reservation);
        Ok(())
    }

    pub fn allocation(&self, allocation_id: AllocationId) -> Option<AllocationRecord> {
        self.allocations.get(&allocation_id).copied()
    }

    pub fn lease(&self, lease_id: LeaseId) -> Option<LeaseReservation> {
        self.leases.get(&lease_id).copied()
    }

    pub fn validate_trace(&self, trace: &ComputeTrace) -> Result<(), ContractError> {
        trace.validate()?;
        let mut views = BTreeMap::<
            ViewId,
            (
                AllocationId,
                u64,
                u64,
                BufferAccess,
                BufferSourceKind,
                Option<LeaseId>,
            ),
        >::new();
        let mut ranges = Vec::<(usize, AllocationId, ViewId, u64, u64, BufferAccess)>::new();
        for (pass_index, pass) in trace.passes.iter().enumerate() {
            for view in &pass.buffers {
                let end = view.validate_shape()?;
                let allocation = self
                    .allocations
                    .get(&view.allocation_id)
                    .ok_or(ContractError::UnknownAllocation(view.allocation_id))?;
                if allocation.owner_epoch != trace.device_epoch {
                    return Err(ContractError::AllocationEpochMismatch {
                        allocation: view.allocation_id,
                        expected: trace.device_epoch,
                        actual: allocation.owner_epoch,
                    });
                }
                if end > allocation.size {
                    return Err(ContractError::AllocationRangeOutOfBounds {
                        allocation: view.allocation_id,
                        end,
                        allocation_size: allocation.size,
                    });
                }
                for &(
                    other_pass,
                    other_allocation,
                    other_view,
                    other_start,
                    other_end,
                    other_access,
                ) in &ranges
                {
                    if other_allocation == view.allocation_id
                        && other_view != view.view_id
                        && other_access != BufferAccess::Unused
                        && view.access != BufferAccess::Unused
                        && (other_access.is_writable() || view.access.is_writable())
                        && view.offset < other_end
                        && other_start < end
                    {
                        return Err(ContractError::OverlappingWritableViews {
                            first: other_view,
                            second: view.view_id,
                            first_pass: other_pass,
                            second_pass: pass_index,
                        });
                    }
                }
                ranges.push((
                    pass_index,
                    view.allocation_id,
                    view.view_id,
                    view.offset,
                    end,
                    view.access,
                ));
                if let Some(lease_id) = view.source.lease_id() {
                    let reservation = self
                        .leases
                        .get(&lease_id)
                        .ok_or(ContractError::UnknownLease(lease_id))?;
                    if reservation.lease.allocation_id != view.allocation_id {
                        return Err(ContractError::LeaseMismatch {
                            view: view.view_id,
                            lease: lease_id,
                        });
                    }
                    if reservation.lease.owner_epoch != trace.device_epoch {
                        return Err(ContractError::LeaseEpochMismatch {
                            lease: lease_id,
                            expected: trace.device_epoch,
                            actual: reservation.lease.owner_epoch,
                        });
                    }
                    let lease_end = reservation.end()?;
                    if view.offset < reservation.offset || end > lease_end {
                        return Err(ContractError::LeaseRangeOutOfBounds {
                            lease: lease_id,
                            end,
                            allocation_size: lease_end,
                        });
                    }
                }
                let declaration = (
                    view.allocation_id,
                    view.offset,
                    view.length,
                    view.access,
                    view.source.kind(),
                    view.source.lease_id(),
                );
                if let Some(previous) = views.insert(view.view_id, declaration) {
                    if previous != declaration {
                        return Err(ContractError::ViewIdentityMismatch(view.view_id));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Explicit identities needed when the legacy snapshot API is converted into
/// a provider trace. The snapshot API only carries a Metal binding index; it
/// must not be treated as an allocation identity by inference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotBufferIdentity {
    pub metal_binding: u32,
    pub allocation_id: AllocationId,
    pub view_id: ViewId,
}

/// Caller-trusted metadata for the opaque pipeline carried by a legacy
/// snapshot submission. The core adapter cannot downcast or inspect that
/// artifact, so the caller owns the proof that these values came from it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SnapshotPipelineIdentity {
    pub pipeline_id: PipelineId,
    pub function: FunctionIdentity,
    pub pipeline_contract: PipelineContract,
}

/// Build a provider trace from the legacy owned-bytes snapshot contract.
///
/// This is a caller-trusted compatibility adapter, not a production provider
/// entry point. `ComputeSubmission::pipeline` is opaque and cannot be inspected
/// here; the caller must derive the supplied pipeline identity from that same
/// artifact. It deliberately refuses threadgroups, pseudo-aliases, and any
/// missing/extra identity record so future callers cannot silently lose
/// lifetime information.
pub fn trace_from_trusted_snapshot(
    submission: &crate::ComputeSubmission,
    device_epoch: DeviceEpoch,
    operation_id: OperationId,
    pipeline: SnapshotPipelineIdentity,
    identities: &[SnapshotBufferIdentity],
) -> Result<ComputeTrace, ContractError> {
    if pipeline.pipeline_contract.dispatch_kind != DispatchKind::ThreadsExact {
        return Err(ContractError::SnapshotDispatchUnsupported(
            pipeline.pipeline_contract.dispatch_kind,
        ));
    }
    let mut identity_by_binding = BTreeMap::new();
    let mut allocation_by_binding = BTreeMap::new();
    for identity in identities {
        if identity_by_binding
            .insert(identity.metal_binding, *identity)
            .is_some()
        {
            return Err(ContractError::DuplicateBinding(identity.metal_binding));
        }
        if allocation_by_binding
            .insert(identity.allocation_id, identity.metal_binding)
            .is_some()
        {
            return Err(ContractError::SnapshotAliasUnsupported(
                identity.allocation_id,
            ));
        }
    }
    let reflected = pipeline
        .pipeline_contract
        .buffer_bindings
        .iter()
        .map(|binding| (binding.metal_binding, binding.access))
        .collect::<BTreeMap<_, _>>();
    let mut buffers = Vec::with_capacity(submission.buffers.len());
    for binding in &submission.buffers {
        let identity = identity_by_binding
            .get(&binding.index)
            .ok_or(ContractError::MissingSnapshotIdentity(binding.index))?;
        let access = reflected
            .get(&binding.index)
            .copied()
            .ok_or(ContractError::UnknownBinding(binding.index))?;
        let length = u64::try_from(binding.bytes.len())
            .map_err(|_| ContractError::ArithmeticOverflow("snapshot buffer length"))?;
        buffers.push(BufferView {
            view_id: identity.view_id,
            metal_binding: binding.index,
            allocation_id: identity.allocation_id,
            offset: 0,
            length,
            access,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(binding.bytes.clone()),
        });
    }
    for identity in identities {
        if !submission
            .buffers
            .iter()
            .any(|binding| binding.index == identity.metal_binding)
        {
            return Err(ContractError::UnknownSnapshotIdentity(
                identity.metal_binding,
            ));
        }
    }
    let grid = submission.threads_per_grid.dimensions().map(u64::from);
    let local = submission
        .threads_per_threadgroup
        .dimensions()
        .map(u64::from);
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch,
        operation_id,
        function: pipeline.function,
        pipeline_contract: pipeline.pipeline_contract,
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers,
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid,
                threads_per_threadgroup: local,
            },
        }],
        completion_policy: CompletionPolicy::HostReadback,
    };
    trace.validate()?;
    Ok(trace)
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
        if self.device_epoch.is_zero() {
            return Err(ContractError::InvalidIdentity("device epoch"));
        }
        if self.operation_id.is_zero() {
            return Err(ContractError::InvalidIdentity("operation id"));
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

    pub fn validate_with_resources(
        &self,
        resources: &ResourceTableSnapshot,
    ) -> Result<(), ContractError> {
        resources.validate_trace(self)
    }
}

/// Opaque identity returned by a successful submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionToken {
    pub submission_id: SubmissionId,
    pub device_epoch: DeviceEpoch,
}

impl CompletionToken {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.submission_id.is_zero() {
            return Err(ContractError::InvalidIdentity("submission id"));
        }
        if self.device_epoch.is_zero() {
            return Err(ContractError::InvalidIdentity("completion device epoch"));
        }
        Ok(())
    }
}

/// Explicit completion observation. A timeout is non-terminal and may be
/// followed by another wait on the same token. Terminal observations describe
/// result availability, not permission to release GPU backing; in particular,
/// `SubmittedUnknown` does not establish GPU retirement.
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
    pub fn validate(self) -> Result<(), ContractError> {
        match self {
            Self::NotSubmitted => Ok(()),
            Self::Submitted { token }
            | Self::CompletedVisible { token }
            | Self::TimedOut { token } => token.validate(),
            Self::Failed { token }
            | Self::DeviceLost { token }
            | Self::SubmittedUnknown { token } => {
                token.as_ref().map_or(Ok(()), CompletionToken::validate)
            }
        }
    }

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

/// A deterministic provider writeback keyed by allocation and view identity.
/// `offset` is measured in bytes from the start of the allocation, like
/// [`BufferView::offset`], rather than from the start of the view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferWriteback {
    pub view_id: ViewId,
    pub allocation_id: AllocationId,
    pub offset: u64,
    pub bytes: Vec<u8>,
}

impl BufferWriteback {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.view_id.is_zero() {
            return Err(ContractError::InvalidIdentity("writeback view id"));
        }
        if self.allocation_id.is_zero() {
            return Err(ContractError::InvalidIdentity("writeback allocation id"));
        }
        if self.bytes.is_empty() {
            return Err(ContractError::ZeroLength("writeback"));
        }
        self.end().map(|_| ())
    }

    pub fn end(&self) -> Result<u64, ContractError> {
        let length = u64::try_from(self.bytes.len())
            .map_err(|_| ContractError::ArithmeticOverflow("writeback length"))?;
        self.offset
            .checked_add(length)
            .ok_or(ContractError::ArithmeticOverflow("writeback range"))
    }
}

/// Result returned by a provider after a validated trace is submitted.
///
/// A successful result must be `Submitted` or `CompletedVisible`; all failures
/// use [`ProviderError`]. Writebacks are CPU-visible and may only accompany
/// `CompletedVisible` with [`CompletionPolicy::HostReadback`]. They are ordered
/// by `(allocation_id, view_id)`. The current single-pass contract requires one
/// complete writeback for each writable view when host readback completes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderSubmission {
    pub completion: CompletionDisposition,
    pub writebacks: Vec<BufferWriteback>,
}

impl ProviderSubmission {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.completion.validate()?;
        if !matches!(
            self.completion,
            CompletionDisposition::Submitted { .. }
                | CompletionDisposition::CompletedVisible { .. }
        ) {
            return Err(ContractError::InvalidSubmissionCompletion(self.completion));
        }
        if !self.writebacks.is_empty()
            && !matches!(
                self.completion,
                CompletionDisposition::CompletedVisible { .. }
            )
        {
            return Err(ContractError::WritebackBeforeCompletion);
        }
        let mut views = BTreeMap::new();
        for writeback in &self.writebacks {
            writeback.validate()?;
            if views
                .insert((writeback.allocation_id, writeback.view_id), ())
                .is_some()
            {
                return Err(ContractError::DuplicateWriteback {
                    allocation: writeback.allocation_id,
                    view: writeback.view_id,
                });
            }
        }
        if self.writebacks.windows(2).any(|pair| {
            (pair[0].allocation_id, pair[0].view_id) > (pair[1].allocation_id, pair[1].view_id)
        }) {
            return Err(ContractError::NonCanonicalWritebackOrder);
        }
        Ok(())
    }

    /// Validate a result against the exact submitted trace. Multi-pass result
    /// semantics are not defined by this contract and are explicitly refused.
    pub fn validate_for_trace(&self, trace: &ComputeTrace) -> Result<(), ContractError> {
        trace.validate()?;
        self.validate()?;
        let [pass] = trace.passes.as_slice() else {
            return Err(ContractError::UnsupportedSubmissionPassCount(
                trace.passes.len(),
            ));
        };
        let token = self
            .completion
            .token()
            .ok_or(ContractError::InvalidSubmissionCompletion(self.completion))?;
        if token.device_epoch != trace.device_epoch {
            return Err(ContractError::CompletionEpochMismatch {
                expected: trace.device_epoch,
                actual: token.device_epoch,
            });
        }
        if !self.writebacks.is_empty() && trace.completion_policy != CompletionPolicy::HostReadback
        {
            return Err(ContractError::WritebackPolicyMismatch(
                trace.completion_policy,
            ));
        }
        for writeback in &self.writebacks {
            let view = pass
                .buffers
                .iter()
                .find(|view| {
                    view.allocation_id == writeback.allocation_id
                        && view.view_id == writeback.view_id
                })
                .ok_or(ContractError::UnknownWriteback {
                    allocation: writeback.allocation_id,
                    view: writeback.view_id,
                })?;
            if !view.access.is_writable() {
                return Err(ContractError::ReadOnlyWriteback(view.view_id));
            }
            let view_end = view.validate_shape()?;
            let end = writeback.end()?;
            if writeback.offset < view.offset || end > view_end {
                return Err(ContractError::WritebackRangeOutOfBounds {
                    view: view.view_id,
                    offset: writeback.offset,
                    end,
                    view_offset: view.offset,
                    view_end,
                });
            }
            if writeback.offset != view.offset || end != view_end {
                return Err(ContractError::IncompleteWriteback(view.view_id));
            }
        }
        if trace.completion_policy == CompletionPolicy::HostReadback
            && matches!(
                self.completion,
                CompletionDisposition::CompletedVisible { .. }
            )
        {
            for view in pass.buffers.iter().filter(|view| view.access.is_writable()) {
                if !self.writebacks.iter().any(|writeback| {
                    writeback.allocation_id == view.allocation_id
                        && writeback.view_id == view.view_id
                }) {
                    return Err(ContractError::MissingWriteback {
                        allocation: view.allocation_id,
                        view: view.view_id,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Backend-neutral execution boundary for a canonical Metal trace.
///
/// Implementations own compilation, provider handles, queue/encoder state and
/// completion retirement. They receive only an admitted value trace and must
/// keep provider-specific objects behind this trait. `wait` reports timeout as
/// a non-terminal disposition; provider failures use `ProviderError`.
pub trait ComputeProvider: Send + Sync {
    fn capabilities(&self) -> ProviderCapabilities;

    fn submit(&self, trace: ValidatedComputeTrace) -> Result<ProviderSubmission, ProviderError>;

    fn wait(
        &self,
        token: CompletionToken,
        timeout: Duration,
    ) -> Result<CompletionDisposition, ProviderError>;
}

/// Shared compilation and retirement boundary for a compute provider.
///
/// Backend objects remain in the implementing provider. Every new provider
/// context must obtain its epoch from [`allocate_device_epoch`], and every
/// submitted pipeline or completion must be checked against that context.
pub trait PipelineProvider: ComputeProvider {
    fn device_epoch(&self) -> DeviceEpoch;

    /// Validate the request and either register a compiled pipeline or return a
    /// typed refusal. Unsupported source kinds use a compile-phase capability
    /// error; malformed source uses an argument or compilation error.
    fn compile(
        &self,
        request: PipelineCompileRequest,
    ) -> Result<CompiledComputePipeline, ProviderError>;

    /// Stop accepting submissions for this pipeline. Implementations must
    /// verify both its epoch and registered identity/metadata before removing
    /// it, and reject stale or foreign values. Already submitted work retains
    /// its backing pipeline until the GPU is known to have retired it.
    fn release_pipeline(&self, pipeline: &CompiledComputePipeline) -> Result<(), ProviderError>;

    /// Release a retained completion record after verifying its epoch and
    /// submission identity. Forgetting a record is not evidence of GPU
    /// retirement and must not free resources still in use by submitted work.
    fn release_completion(&self, token: CompletionToken) -> Result<(), ProviderError>;
}

/// Capabilities are captured once for a provider device context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderCapabilities {
    /// Provider policy, not a physical-device limit. The current snapshot
    /// adapters expose one pass; a future command-buffer provider can raise it.
    pub max_passes: u32,
    pub supports_threads_exact: bool,
    pub supports_threadgroups: bool,
    pub supports_serial: bool,
    pub supports_concurrent: bool,
    pub max_local_size: [u64; 3],
    pub max_invocations: u64,
    pub max_group_count: [u64; 3],
    pub max_storage_buffer_descriptors: u32,
    pub max_buffer_range: u64,
    pub max_push_constant_bytes: u32,
    pub alias_mode: AliasMode,
    pub storage_modes: Vec<StorageMode>,
    pub host_readback: bool,
    pub submit_only: bool,
}

impl ProviderCapabilities {
    /// Freeze a trace and its resource snapshot after admission. The returned
    /// value is the hand-off object a future provider trait should consume.
    pub fn validate_trace(
        &self,
        trace: ComputeTrace,
        resources: ResourceTableSnapshot,
    ) -> Result<ValidatedComputeTrace, ProviderError> {
        self.admit(&trace, &resources)?;
        Ok(ValidatedComputeTrace { trace, resources })
    }

    /// Admit a complete value trace and its neutral resource namespace without
    /// creating provider objects.
    ///
    /// Structural errors are reported as an `Args` refusal; selected-device
    /// limits and unsupported storage/completion modes are reported as
    /// `Capability` refusals. A provider implementation can perform the same
    /// checks immediately before encode, while keeping Vulkan/Metal handles
    /// out of the neutral contract.
    pub fn admit(
        &self,
        trace: &ComputeTrace,
        resources: &ResourceTableSnapshot,
    ) -> Result<(), ProviderError> {
        trace.validate().map_err(contract_error_refusal)?;

        if trace.passes.len() > self.max_passes as usize {
            return Err(capability_error("pass_count_limit")
                .with_field("requested", FieldValue::Unsigned(trace.passes.len() as u64))
                .with_field("maximum", FieldValue::Unsigned(self.max_passes as u64)));
        }

        match trace.encoder_dispatch_type {
            DispatchType::Serial if !self.supports_serial => {
                return Err(capability_error("dispatch_type_unsupported"));
            }
            DispatchType::Concurrent if !self.supports_concurrent => {
                return Err(capability_error("dispatch_type_unsupported"));
            }
            _ => {}
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
        if trace.passes.len() != 1 {
            return Err(capability_error("multiple_passes_unsupported")
                .with_detail("B0 provider submission is one direct compute pass"));
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

        let mut allocations = BTreeMap::new();
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

            let push_end = trace
                .pipeline_contract
                .push_constant_offset
                .checked_add(trace.pipeline_contract.push_constant_bytes)
                .ok_or_else(|| capability_error("push_constant_range_overflow"))?;
            if push_end > self.max_push_constant_bytes {
                return Err(capability_error("push_constant_range_limit")
                    .with_field("requested", FieldValue::Unsigned(push_end as u64))
                    .with_field(
                        "maximum",
                        FieldValue::Unsigned(self.max_push_constant_bytes as u64),
                    ));
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
            for buffer in &pass.buffers {
                if buffer.length > self.max_buffer_range {
                    return Err(capability_error("storage_buffer_range_limit")
                        .with_field("binding", FieldValue::Unsigned(buffer.metal_binding as u64))
                        .with_field("requested", FieldValue::Unsigned(buffer.length))
                        .with_field("maximum", FieldValue::Unsigned(self.max_buffer_range)));
                }
                let reflected = trace
                    .pipeline_contract
                    .buffer_bindings
                    .iter()
                    .find(|binding| binding.metal_binding == buffer.metal_binding)
                    .expect("trace validation checked reflected bindings");
                match &reflected.footprint {
                    FootprintProof::Unbounded => {
                        return Err(capability_error("buffer_footprint_unbounded").with_field(
                            "binding",
                            FieldValue::Unsigned(buffer.metal_binding as u64),
                        ));
                    }
                    FootprintProof::Static { max_bytes } => {
                        if *max_bytes > buffer.length {
                            return Err(capability_error("buffer_footprint_exceeds_view")
                                .with_field(
                                    "binding",
                                    FieldValue::Unsigned(buffer.metal_binding as u64),
                                )
                                .with_field("required", FieldValue::Unsigned(*max_bytes))
                                .with_field("available", FieldValue::Unsigned(buffer.length)));
                        }
                    }
                    FootprintProof::Affine { accesses } => {
                        let required = affine_required_bytes(accesses, pass.dispatch.grid)?;
                        if required > buffer.length {
                            return Err(capability_error("buffer_footprint_exceeds_view")
                                .with_field(
                                    "binding",
                                    FieldValue::Unsigned(buffer.metal_binding as u64),
                                )
                                .with_field("required", FieldValue::Unsigned(required))
                                .with_field("available", FieldValue::Unsigned(buffer.length)));
                        }
                    }
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
                if let Some(previous_view) =
                    allocations.insert(buffer.allocation_id, buffer.view_id)
                {
                    if previous_view == buffer.view_id {
                        continue;
                    }
                    match self.alias_mode {
                        AliasMode::DistinctViews => {}
                        AliasMode::Refused => {
                            return Err(capability_error("buffer_alias_unsupported").with_field(
                                "binding",
                                FieldValue::Unsigned(buffer.metal_binding as u64),
                            ));
                        }
                        AliasMode::ExplicitPolicy => {
                            return Err(capability_error("buffer_alias_policy_required")
                                .with_field(
                                    "binding",
                                    FieldValue::Unsigned(buffer.metal_binding as u64),
                                ));
                        }
                    }
                }
            }
        }
        // Capability admission intentionally precedes backing/lease admission:
        // a malformed or unsupported dispatch must not be masked by a stale
        // resource handle, and the order matches the provider contract gates.
        resources
            .validate_trace(trace)
            .map_err(contract_error_refusal)?;
        Ok(())
    }
}

fn ceil_div(value: u64, divisor: u64) -> Option<u64> {
    value
        .checked_add(divisor.checked_sub(1)?)?
        .checked_div(divisor)
}

fn affine_required_bytes(accesses: &[AffineAccess], grid: [u64; 3]) -> Result<u64, ProviderError> {
    let mut required = 0_u64;
    for access in accesses {
        let mut end = access
            .base_offset
            .checked_add(access.access_size)
            .ok_or_else(|| capability_error("buffer_footprint_overflow"))?;
        for term in &access.terms {
            let maximum = grid
                .get(usize::from(term.axis))
                .copied()
                .and_then(|value| value.checked_sub(1))
                .ok_or_else(|| capability_error("buffer_footprint_axis_invalid"))?;
            end = end
                .checked_add(
                    maximum
                        .checked_mul(term.stride)
                        .ok_or_else(|| capability_error("buffer_footprint_overflow"))?,
                )
                .ok_or_else(|| capability_error("buffer_footprint_overflow"))?;
        }
        required = required.max(end);
    }
    Ok(required)
}

fn capability_error(slug: &'static str) -> ProviderError {
    ProviderError::new(ProviderPhase::Resolve, ProviderErrorClass::Capability, slug)
        .expect("static provider refusal slug")
}

fn contract_error_refusal(error: ContractError) -> ProviderError {
    let (class, slug) = match &error {
        ContractError::UnknownAllocation(_)
        | ContractError::DuplicateAllocation(_)
        | ContractError::AllocationEpochMismatch { .. }
        | ContractError::AllocationRangeOutOfBounds { .. }
        | ContractError::DuplicateLease(_)
        | ContractError::UnknownLease(_)
        | ContractError::LeaseEpochMismatch { .. }
        | ContractError::LeaseRangeOutOfBounds { .. }
        | ContractError::LeaseMismatch { .. }
        | ContractError::OverlappingWritableViews { .. }
        | ContractError::ViewIdentityMismatch(_)
        | ContractError::SnapshotAliasUnsupported(_) => {
            (ProviderErrorClass::Resource, "resource_contract_invalid")
        }
        ContractError::SourceLengthMismatch { .. } => {
            (ProviderErrorClass::Args, "buffer_source_length_mismatch")
        }
        _ => (ProviderErrorClass::Args, "trace_contract_invalid"),
    };
    ProviderError::new(ProviderPhase::Resolve, class, slug)
        .expect("static provider refusal slug")
        .with_detail(error.to_string())
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

/// Structural errors in provider input values or returned submission values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContractError {
    EmptyField(&'static str),
    InvalidIdentity(&'static str),
    ZeroLength(&'static str),
    ZeroDimension {
        field: &'static str,
        axis: usize,
    },
    ArithmeticOverflow(&'static str),
    MisalignedPushConstantOffset(u32),
    UnsupportedAttributeStride,
    SourceLengthMismatch {
        view: ViewId,
        expected: u64,
        actual: u64,
    },
    DuplicateAllocation(AllocationId),
    UnknownAllocation(AllocationId),
    AllocationEpochMismatch {
        allocation: AllocationId,
        expected: DeviceEpoch,
        actual: DeviceEpoch,
    },
    AllocationRangeOutOfBounds {
        allocation: AllocationId,
        end: u64,
        allocation_size: u64,
    },
    DuplicateLease(LeaseId),
    UnknownLease(LeaseId),
    LeaseEpochMismatch {
        lease: LeaseId,
        expected: DeviceEpoch,
        actual: DeviceEpoch,
    },
    LeaseRangeOutOfBounds {
        lease: LeaseId,
        end: u64,
        allocation_size: u64,
    },
    OverlappingWritableViews {
        first: ViewId,
        second: ViewId,
        first_pass: usize,
        second_pass: usize,
    },
    DuplicateWriteback {
        allocation: AllocationId,
        view: ViewId,
    },
    InvalidSubmissionCompletion(CompletionDisposition),
    WritebackBeforeCompletion,
    CompletionEpochMismatch {
        expected: DeviceEpoch,
        actual: DeviceEpoch,
    },
    UnsupportedSubmissionPassCount(usize),
    WritebackPolicyMismatch(CompletionPolicy),
    NonCanonicalWritebackOrder,
    UnknownWriteback {
        allocation: AllocationId,
        view: ViewId,
    },
    ReadOnlyWriteback(ViewId),
    WritebackRangeOutOfBounds {
        view: ViewId,
        offset: u64,
        end: u64,
        view_offset: u64,
        view_end: u64,
    },
    IncompleteWriteback(ViewId),
    MissingWriteback {
        allocation: AllocationId,
        view: ViewId,
    },
    ViewIdentityMismatch(ViewId),
    LeaseMismatch {
        view: ViewId,
        lease: LeaseId,
    },
    DuplicateBinding(u32),
    NonCanonicalBindingOrder(&'static str),
    MissingBinding(u32),
    UnknownBinding(u32),
    AccessMismatch {
        binding: u32,
        expected: BufferAccess,
        actual: BufferAccess,
    },
    LocalSizeMismatch {
        expected: [u64; 3],
        actual: [u64; 3],
    },
    GridMismatch {
        expected: [u64; 3],
        actual: [u64; 3],
    },
    FixedGridRequiresExactDispatch,
    DuplicateView(ViewId),
    EmptyTrace,
    UnsupportedSchemaVersion(u16),
    MixedPipelines,
    UnsupportedDispatchType(DispatchType),
    SnapshotDispatchUnsupported(DispatchKind),
    SnapshotAliasUnsupported(AllocationId),
    MissingSnapshotIdentity(u32),
    UnknownSnapshotIdentity(u32),
    DispatchKindMismatch {
        expected: DispatchKind,
        actual: DispatchKind,
    },
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(formatter, "{field} must not be empty"),
            Self::InvalidIdentity(identity) => write!(formatter, "{identity} must be non-zero"),
            Self::ZeroLength(field) => write!(formatter, "{field} length must be non-zero"),
            Self::ZeroDimension { field, axis } => {
                write!(formatter, "{field} dimension {axis} must be non-zero")
            }
            Self::ArithmeticOverflow(field) => write!(formatter, "{field} overflows u64"),
            Self::MisalignedPushConstantOffset(offset) => {
                write!(formatter, "push constant offset {offset} is not 4-byte aligned")
            }
            Self::UnsupportedAttributeStride => {
                formatter.write_str("attribute stride is outside the B0 buffer-compute subset")
            }
            Self::SourceLengthMismatch {
                view,
                expected,
                actual,
            } => write!(
                formatter,
                "view {:?} source length {actual} does not match declared length {expected}",
                view
            ),
            Self::DuplicateAllocation(allocation) => {
                write!(formatter, "duplicate allocation {:?}", allocation)
            }
            Self::UnknownAllocation(allocation) => {
                write!(formatter, "unknown allocation {:?}", allocation)
            }
            Self::AllocationEpochMismatch {
                allocation,
                expected,
                actual,
            } => write!(
                formatter,
                "allocation {:?} epoch mismatch: expected {:?}, received {:?}",
                allocation, expected, actual
            ),
            Self::AllocationRangeOutOfBounds {
                allocation,
                end,
                allocation_size,
            } => write!(
                formatter,
                "allocation {:?} range end {end} exceeds size {allocation_size}",
                allocation
            ),
            Self::DuplicateLease(lease) => write!(formatter, "duplicate lease {:?}", lease),
            Self::UnknownLease(lease) => write!(formatter, "unknown lease {:?}", lease),
            Self::LeaseEpochMismatch {
                lease,
                expected,
                actual,
            } => write!(
                formatter,
                "lease {:?} epoch mismatch: expected {:?}, received {:?}",
                lease, expected, actual
            ),
            Self::LeaseRangeOutOfBounds {
                lease,
                end,
                allocation_size,
            } => write!(
                formatter,
                "lease {:?} range end {end} exceeds reservation {allocation_size}",
                lease
            ),
            Self::OverlappingWritableViews {
                first,
                second,
                first_pass,
                second_pass,
            } => write!(
                formatter,
                "writable views {:?} (pass {first_pass}) and {:?} (pass {second_pass}) overlap without an alias policy",
                first, second
            ),
            Self::DuplicateWriteback { allocation, view } => write!(
                formatter,
                "duplicate writeback for allocation {:?}, view {:?}",
                allocation, view
            ),
            Self::InvalidSubmissionCompletion(completion) => write!(
                formatter,
                "successful submission requires Submitted or CompletedVisible, received {completion:?}"
            ),
            Self::WritebackBeforeCompletion => {
                formatter.write_str("writebacks require CompletedVisible")
            }
            Self::CompletionEpochMismatch { expected, actual } => write!(
                formatter,
                "completion epoch mismatch: expected {expected:?}, received {actual:?}"
            ),
            Self::UnsupportedSubmissionPassCount(count) => write!(
                formatter,
                "submission result validation requires one pass, received {count}"
            ),
            Self::WritebackPolicyMismatch(policy) => write!(
                formatter,
                "writebacks require HostReadback, received {policy:?}"
            ),
            Self::NonCanonicalWritebackOrder => {
                formatter.write_str("writebacks must be ordered by allocation and view identity")
            }
            Self::UnknownWriteback { allocation, view } => write!(
                formatter,
                "writeback for allocation {allocation:?}, view {view:?} has no matching bound view"
            ),
            Self::ReadOnlyWriteback(view) => write!(
                formatter,
                "writeback view {view:?} is not writable"
            ),
            Self::WritebackRangeOutOfBounds { view, offset, end, view_offset, view_end } => write!(
                formatter,
                "writeback view {view:?} range {offset}..{end} is outside view range {view_offset}..{view_end}"
            ),
            Self::IncompleteWriteback(view) => write!(
                formatter,
                "writeback must cover all of view {view:?}"
            ),
            Self::MissingWriteback { allocation, view } => write!(
                formatter,
                "completed host readback is missing allocation {allocation:?}, view {view:?}"
            ),
            Self::ViewIdentityMismatch(view) => {
                write!(formatter, "view {:?} changes resource declaration across passes", view)
            }
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
            Self::NonCanonicalBindingOrder(scope) => {
                write!(formatter, "{scope} bindings are not in canonical order")
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
            Self::LocalSizeMismatch { expected, actual } => write!(
                formatter,
                "local size mismatch: expected {expected:?}, received {actual:?}"
            ),
            Self::GridMismatch { expected, actual } => write!(
                formatter,
                "dispatch grid mismatch: expected {expected:?}, received {actual:?}"
            ),
            Self::FixedGridRequiresExactDispatch => {
                formatter.write_str("fixed grid requires exact-thread dispatch")
            }
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
            Self::SnapshotDispatchUnsupported(dispatch_kind) => write!(
                formatter,
                "snapshot adapter does not support dispatch kind {dispatch_kind:?}"
            ),
            Self::SnapshotAliasUnsupported(allocation) => write!(
                formatter,
                "snapshot adapter cannot represent aliased owned allocation {:?}",
                allocation
            ),
            Self::MissingSnapshotIdentity(binding) => {
                write!(formatter, "snapshot binding {binding} has no explicit resource identity")
            }
            Self::UnknownSnapshotIdentity(binding) => {
                write!(formatter, "snapshot identity {binding} has no matching buffer")
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

    fn compile_request(source: ShaderSource) -> PipelineCompileRequest {
        PipelineCompileRequest {
            entry_name: "copy_word".to_owned(),
            logical_digest: digest(),
            source,
        }
    }

    #[test]
    fn compilation_request_validates_common_shape_and_preserves_source_kind() {
        for (source, kind) in [
            (
                ShaderSource::SanitizedLl("define void @copy_word() {}".into()),
                FunctionSource::SanitizedLl,
            ),
            // Binary shape is deliberately delegated to the backend.
            (ShaderSource::BinaryAir(vec![1]), FunctionSource::BinaryAir),
            (
                ShaderSource::MetalSource("kernel void copy_word() {}".into()),
                FunctionSource::MetalSource,
            ),
        ] {
            assert_eq!(source.kind(), kind);
            assert_eq!(compile_request(source).validate(), Ok(()));
        }
        for source in [
            ShaderSource::SanitizedLl(" \n\t".into()),
            ShaderSource::BinaryAir(Vec::new()),
            ShaderSource::MetalSource(String::new()),
        ] {
            assert_eq!(
                compile_request(source).validate(),
                Err(ContractError::EmptyField("shader source"))
            );
        }
        let mut request = compile_request(ShaderSource::BinaryAir(vec![1]));
        request.entry_name = " \t".into();
        assert_eq!(
            request.validate(),
            Err(ContractError::EmptyField("function entry name"))
        );
    }

    #[test]
    fn device_epochs_are_unique_across_concurrent_context_creation() {
        let epochs = std::thread::scope(|scope| {
            let workers = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        (0..32)
                            .map(|_| allocate_device_epoch().unwrap())
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .flat_map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(epochs.iter().all(|epoch| !epoch.is_zero()));
        let unique = epochs.iter().collect::<std::collections::BTreeSet<_>>();
        assert_eq!(epochs.len(), unique.len());
    }

    struct CompileOnlyProvider {
        epoch: DeviceEpoch,
        source: FunctionSource,
        registered: std::sync::Mutex<Option<CompiledComputePipeline>>,
    }

    fn mock_refusal(class: ProviderErrorClass, slug: &str) -> ProviderError {
        ProviderError::new(ProviderPhase::Compile, class, slug).unwrap()
    }

    impl ComputeProvider for CompileOnlyProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            capabilities()
        }

        fn submit(&self, _: ValidatedComputeTrace) -> Result<ProviderSubmission, ProviderError> {
            Err(mock_refusal(ProviderErrorClass::Capability, "compile_only"))
        }

        fn wait(
            &self,
            _: CompletionToken,
            _: Duration,
        ) -> Result<CompletionDisposition, ProviderError> {
            Err(mock_refusal(ProviderErrorClass::Capability, "compile_only"))
        }
    }

    impl PipelineProvider for CompileOnlyProvider {
        fn device_epoch(&self) -> DeviceEpoch {
            self.epoch
        }

        fn compile(
            &self,
            request: PipelineCompileRequest,
        ) -> Result<CompiledComputePipeline, ProviderError> {
            request.validate().map_err(|error| {
                mock_refusal(ProviderErrorClass::Args, "invalid_compile_request")
                    .with_detail(error.to_string())
            })?;
            if request.source.kind() != self.source {
                return Err(mock_refusal(
                    ProviderErrorClass::Capability,
                    "unsupported_shader_source",
                ));
            }
            let metadata = CompiledComputePipeline {
                device_epoch: self.epoch,
                pipeline_id: PipelineId::new(1),
                function: FunctionIdentity {
                    logical_digest: request.logical_digest,
                    entry_name: request.entry_name,
                    source: request.source.kind(),
                },
                contract: trace(Vec::new()).pipeline_contract,
            };
            *self.registered.lock().unwrap() = Some(metadata.clone());
            Ok(metadata)
        }

        fn release_pipeline(
            &self,
            pipeline: &CompiledComputePipeline,
        ) -> Result<(), ProviderError> {
            let mut registered = self.registered.lock().unwrap();
            if registered.as_ref() != Some(pipeline) {
                return Err(mock_refusal(
                    ProviderErrorClass::Resource,
                    "unknown_pipeline",
                ));
            }
            *registered = None;
            Ok(())
        }

        fn release_completion(&self, _: CompletionToken) -> Result<(), ProviderError> {
            Err(mock_refusal(
                ProviderErrorClass::Resource,
                "unknown_completion",
            ))
        }
    }

    #[test]
    fn shared_pipeline_trait_keeps_backend_sources_and_contexts_distinct() {
        let providers: Vec<Box<dyn PipelineProvider>> =
            [FunctionSource::SanitizedLl, FunctionSource::MetalSource]
                .into_iter()
                .map(|source| {
                    Box::new(CompileOnlyProvider {
                        epoch: allocate_device_epoch().unwrap(),
                        source,
                        registered: std::sync::Mutex::new(None),
                    }) as Box<dyn PipelineProvider>
                })
                .collect();
        let sources = [
            ShaderSource::SanitizedLl("define void @copy_word() {}".into()),
            ShaderSource::MetalSource("kernel void copy_word() {}".into()),
        ];
        let pipelines = providers
            .iter()
            .zip(sources)
            .map(|(provider, source)| {
                let kind = source.kind();
                let metadata = provider.compile(compile_request(source)).unwrap();
                assert_eq!(metadata.device_epoch, provider.device_epoch());
                assert_eq!(metadata.function.source, kind);
                assert_eq!(metadata.function.logical_digest, digest());
                assert_eq!(metadata.function.entry_name, "copy_word");
                assert!(metadata.contract.validate().is_ok());
                metadata
            })
            .collect::<Vec<_>>();
        assert_ne!(pipelines[0].device_epoch, pipelines[1].device_epoch);
        assert_eq!(pipelines[0].pipeline_id, pipelines[1].pipeline_id);
        assert!(providers[1].release_pipeline(&pipelines[0]).is_err());
        let refusal = providers[0]
            .compile(compile_request(ShaderSource::MetalSource(
                "kernel x".into(),
            )))
            .unwrap_err();
        assert_eq!(refusal.phase, ProviderPhase::Compile);
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        for (provider, pipeline) in providers.iter().zip(&pipelines) {
            assert!(provider.release_pipeline(pipeline).is_ok());
            assert!(provider.release_pipeline(pipeline).is_err());
        }
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
                required_local_size: None,
                fixed_grid: None,
                push_constant_offset: 0,
                push_constant_bytes: 0,
                buffer_bindings: vec![BufferBindingContract {
                    metal_binding: 0,
                    access: BufferAccess::Write,
                    footprint: FootprintProof::Affine {
                        accesses: Vec::new(),
                    },
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
            max_passes: 1,
            supports_threads_exact: true,
            supports_threadgroups: false,
            supports_serial: true,
            supports_concurrent: false,
            max_local_size: [8, 8, 8],
            max_invocations: 64,
            max_group_count: [16, 16, 16],
            max_storage_buffer_descriptors: 4,
            max_buffer_range: 4096,
            max_push_constant_bytes: 128,
            alias_mode: AliasMode::Refused,
            storage_modes: vec![StorageMode::OwnedBytes],
            host_readback: true,
            submit_only: true,
        }
    }

    fn resources() -> ResourceTableSnapshot {
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(9),
                owner_epoch: DeviceEpoch::new(1),
                size: 4,
            })
            .unwrap();
        resources
    }

    fn leased_resources() -> ResourceTableSnapshot {
        let mut resources = resources();
        resources
            .insert_lease(LeaseReservation {
                lease: BufferLease {
                    lease_id: LeaseId::new(11),
                    allocation_id: AllocationId::new(9),
                    owner_epoch: DeviceEpoch::new(1),
                },
                offset: 1,
                length: 2,
            })
            .unwrap();
        resources
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
    fn resource_snapshot_checks_allocation_range_and_epoch() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        value.passes[0].buffers[0].offset = 2;
        let error = value.validate_with_resources(&resources()).unwrap_err();
        assert_eq!(
            error,
            ContractError::AllocationRangeOutOfBounds {
                allocation: AllocationId::new(9),
                end: 6,
                allocation_size: 4,
            }
        );

        let mut wrong_epoch = resources();
        wrong_epoch
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(10),
                owner_epoch: DeviceEpoch::new(2),
                size: 4,
            })
            .unwrap();
        value.passes[0].buffers[0].allocation_id = AllocationId::new(10);
        assert!(matches!(
            value.validate_with_resources(&wrong_epoch),
            Err(ContractError::AllocationEpochMismatch { .. })
        ));
    }

    #[test]
    fn resource_snapshot_checks_lease_range_epoch_and_cross_pass_identity() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        value.passes[0].buffers[0] = BufferView {
            view_id: ViewId::new(1),
            metal_binding: 0,
            allocation_id: AllocationId::new(9),
            offset: 1,
            length: 2,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::BorrowedNoCopy(LeaseId::new(11)),
        };
        value.pipeline_contract.buffer_bindings[0].access = BufferAccess::Read;
        assert!(value.validate_with_resources(&leased_resources()).is_ok());

        let mut out_of_lease = value.clone();
        out_of_lease.passes[0].buffers[0].offset = 0;
        assert!(matches!(
            out_of_lease.validate_with_resources(&leased_resources()),
            Err(ContractError::LeaseRangeOutOfBounds { .. })
        ));

        let mut switched = value;
        let mut second = switched.passes[0].clone();
        second.buffers[0].source = BufferSource::BorrowedNoCopy(LeaseId::new(12));
        switched.passes.push(second);
        let mut two_leases = leased_resources();
        two_leases
            .insert_lease(LeaseReservation {
                lease: BufferLease {
                    lease_id: LeaseId::new(12),
                    allocation_id: AllocationId::new(9),
                    owner_epoch: DeviceEpoch::new(1),
                },
                offset: 1,
                length: 2,
            })
            .unwrap();
        assert_eq!(
            switched.validate_with_resources(&two_leases),
            Err(ContractError::ViewIdentityMismatch(ViewId::new(1)))
        );
    }

    #[test]
    fn resource_snapshot_refuses_overlapping_writable_views_but_allows_read_only_overlap() {
        let mut value = trace(vec![
            pass(4, vec![buffer(1, 0)]),
            pass(4, vec![buffer(2, 0)]),
        ]);
        assert!(matches!(
            value.validate_with_resources(&resources()),
            Err(ContractError::OverlappingWritableViews {
                first: ViewId(1),
                second: ViewId(2),
                first_pass: 0,
                second_pass: 1,
            })
        ));

        value.pipeline_contract.buffer_bindings[0].access = BufferAccess::Read;
        value.passes[0].buffers[0].access = BufferAccess::Read;
        value.passes[1].buffers[0].access = BufferAccess::Read;
        assert!(value.validate_with_resources(&resources()).is_ok());
    }

    #[test]
    fn owned_snapshot_source_length_is_part_of_the_contract() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        value.passes[0].buffers[0].length = 3;
        assert_eq!(
            value.validate(),
            Err(ContractError::SourceLengthMismatch {
                view: ViewId::new(1),
                expected: 3,
                actual: 4,
            })
        );
    }

    #[test]
    fn duplicate_resource_insert_does_not_overwrite_the_snapshot() {
        let mut resources = ResourceTableSnapshot::new();
        let original = AllocationRecord {
            allocation_id: AllocationId::new(9),
            owner_epoch: DeviceEpoch::new(1),
            size: 4,
        };
        resources.insert_allocation(original).unwrap();
        assert_eq!(
            resources.insert_allocation(AllocationRecord {
                size: 8,
                ..original
            }),
            Err(ContractError::DuplicateAllocation(AllocationId::new(9)))
        );
        assert_eq!(resources.allocation(AllocationId::new(9)), Some(original));

        let reservation = LeaseReservation {
            lease: BufferLease {
                lease_id: LeaseId::new(11),
                allocation_id: AllocationId::new(9),
                owner_epoch: DeviceEpoch::new(1),
            },
            offset: 0,
            length: 2,
        };
        resources.insert_lease(reservation).unwrap();
        assert_eq!(
            resources.insert_lease(LeaseReservation {
                length: 1,
                ..reservation
            }),
            Err(ContractError::DuplicateLease(LeaseId::new(11)))
        );
        assert_eq!(resources.lease(LeaseId::new(11)), Some(reservation));
    }

    fn snapshot_submission() -> crate::ComputeSubmission {
        crate::ComputeSubmission {
            pipeline: std::sync::Arc::new(()),
            buffers: vec![crate::BufferBinding {
                index: 0,
                bytes: vec![1, 2, 3, 4],
            }],
            threads_per_grid: crate::Size::new(10, 3, 1).unwrap(),
            threads_per_threadgroup: crate::Size::new(8, 2, 1).unwrap(),
        }
    }

    fn snapshot_function() -> FunctionIdentity {
        FunctionIdentity {
            logical_digest: digest(),
            entry_name: "copy_word".to_string(),
            source: FunctionSource::BinaryAir,
        }
    }

    fn snapshot_pipeline(contract: PipelineContract) -> SnapshotPipelineIdentity {
        SnapshotPipelineIdentity {
            pipeline_id: PipelineId::new(5),
            function: snapshot_function(),
            pipeline_contract: contract,
        }
    }

    #[test]
    fn snapshot_adapter_preserves_explicit_resource_identity() {
        let submission = snapshot_submission();
        let contract = trace(Vec::new()).pipeline_contract;
        let value = trace_from_trusted_snapshot(
            &submission,
            DeviceEpoch::new(3),
            OperationId::new(4),
            snapshot_pipeline(contract),
            &[SnapshotBufferIdentity {
                metal_binding: 0,
                allocation_id: AllocationId::new(6),
                view_id: ViewId::new(7),
            }],
        )
        .unwrap();
        assert_eq!(value.passes.len(), 1);
        assert_eq!(value.passes[0].pipeline, PipelineId::new(5));
        assert_eq!(
            value.passes[0].buffers[0].allocation_id,
            AllocationId::new(6)
        );
        assert_eq!(value.passes[0].buffers[0].view_id, ViewId::new(7));
        assert_eq!(value.passes[0].dispatch.grid, [10, 3, 1]);
    }

    #[test]
    fn snapshot_adapter_refuses_identity_and_dispatch_shape_gaps() {
        let submission = snapshot_submission();
        let contract = trace(Vec::new()).pipeline_contract;
        let missing = trace_from_trusted_snapshot(
            &submission,
            DeviceEpoch::new(3),
            OperationId::new(4),
            snapshot_pipeline(contract.clone()),
            &[],
        )
        .unwrap_err();
        assert_eq!(missing, ContractError::MissingSnapshotIdentity(0));

        let extra = trace_from_trusted_snapshot(
            &submission,
            DeviceEpoch::new(3),
            OperationId::new(4),
            snapshot_pipeline(contract.clone()),
            &[
                SnapshotBufferIdentity {
                    metal_binding: 0,
                    allocation_id: AllocationId::new(6),
                    view_id: ViewId::new(7),
                },
                SnapshotBufferIdentity {
                    metal_binding: 9,
                    allocation_id: AllocationId::new(8),
                    view_id: ViewId::new(9),
                },
            ],
        )
        .unwrap_err();
        assert_eq!(extra, ContractError::UnknownSnapshotIdentity(9));

        let mut future = contract;
        future.dispatch_kind = DispatchKind::Threadgroups;
        let unsupported = trace_from_trusted_snapshot(
            &submission,
            DeviceEpoch::new(3),
            OperationId::new(4),
            snapshot_pipeline(future),
            &[SnapshotBufferIdentity {
                metal_binding: 0,
                allocation_id: AllocationId::new(6),
                view_id: ViewId::new(7),
            }],
        )
        .unwrap_err();
        assert_eq!(
            unsupported,
            ContractError::SnapshotDispatchUnsupported(DispatchKind::Threadgroups)
        );
    }

    #[test]
    fn snapshot_adapter_refuses_pseudo_aliases_from_independent_owned_bytes() {
        let mut submission = snapshot_submission();
        submission.buffers.push(crate::BufferBinding {
            index: 1,
            bytes: vec![5, 6, 7, 8],
        });
        let mut pipeline = snapshot_pipeline(trace(Vec::new()).pipeline_contract);
        pipeline
            .pipeline_contract
            .buffer_bindings
            .push(BufferBindingContract {
                metal_binding: 1,
                access: BufferAccess::Write,
                footprint: FootprintProof::Affine {
                    accesses: Vec::new(),
                },
            });
        let error = trace_from_trusted_snapshot(
            &submission,
            DeviceEpoch::new(3),
            OperationId::new(4),
            pipeline,
            &[
                SnapshotBufferIdentity {
                    metal_binding: 0,
                    allocation_id: AllocationId::new(6),
                    view_id: ViewId::new(7),
                },
                SnapshotBufferIdentity {
                    metal_binding: 1,
                    allocation_id: AllocationId::new(6),
                    view_id: ViewId::new(8),
                },
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ContractError::SnapshotAliasUnsupported(AllocationId::new(6))
        );
    }

    #[test]
    fn capabilities_admit_a_bounded_serial_trace() {
        let value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        assert!(capabilities().admit(&value, &resources()).is_ok());
    }

    #[test]
    fn capabilities_report_limits_and_aliases_structurally() {
        let mut too_wide = trace(vec![pass(4, vec![buffer(1, 0)])]);
        too_wide.passes[0].dispatch.threads_per_threadgroup = [9, 1, 1];
        let error = capabilities().admit(&too_wide, &resources()).unwrap_err();
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
                footprint: FootprintProof::Affine {
                    accesses: Vec::new(),
                },
            });
        let error = capabilities().admit(&alias, &resources()).unwrap_err();
        assert_eq!(error.slug, "buffer_alias_unsupported");
    }

    #[test]
    fn capabilities_refuse_a_future_dispatch_kind_without_guessing() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        value.pipeline_contract.dispatch_kind = DispatchKind::Threadgroups;
        value.passes[0].dispatch.kind = DispatchKind::Threadgroups;
        let error = capabilities().admit(&value, &resources()).unwrap_err();
        assert_eq!(error.slug, "dispatch_kind_unsupported");
    }

    #[test]
    fn capabilities_refuse_an_unbounded_writable_footprint() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        value.pipeline_contract.buffer_bindings[0].footprint = FootprintProof::Unbounded;
        let error = capabilities().admit(&value, &resources()).unwrap_err();
        assert_eq!(error.slug, "buffer_footprint_unbounded");
    }

    #[test]
    fn capabilities_enforce_pass_count_and_fixed_grid_contracts() {
        let value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        let mut one_pass = capabilities();
        one_pass.max_passes = 0;
        let error = one_pass.admit(&value, &resources()).unwrap_err();
        assert_eq!(error.slug, "pass_count_limit");

        let mut fixed = value.clone();
        fixed.pipeline_contract.fixed_grid = Some([10, 3, 1]);
        assert!(capabilities().admit(&fixed, &resources()).is_ok());
        fixed.passes[0].dispatch.grid = [9, 3, 1];
        let error = capabilities().admit(&fixed, &resources()).unwrap_err();
        assert_eq!(
            error.detail.as_deref(),
            Some("dispatch grid mismatch: expected [10, 3, 1], received [9, 3, 1]")
        );
    }

    #[test]
    fn capabilities_bound_affine_footprints_to_the_dispatch_grid() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        value.passes[0].buffers[0].length = 40;
        value.passes[0].buffers[0].source = BufferSource::OwnedBytes(vec![0; 40]);
        let mut backing = resources();
        backing
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(10),
                owner_epoch: DeviceEpoch::new(1),
                size: 40,
            })
            .unwrap();
        value.passes[0].buffers[0].allocation_id = AllocationId::new(10);
        value.pipeline_contract.buffer_bindings[0].footprint = FootprintProof::Affine {
            accesses: vec![AffineAccess {
                base_offset: 0,
                access_size: 4,
                terms: vec![AffineTerm { axis: 0, stride: 4 }],
            }],
        };
        assert!(capabilities().admit(&value, &backing).is_ok());
        value.passes[0].buffers[0].length = 36;
        value.passes[0].buffers[0].source = BufferSource::OwnedBytes(vec![0; 36]);
        let error = capabilities().admit(&value, &backing).unwrap_err();
        assert_eq!(error.slug, "buffer_footprint_exceeds_view");
        assert_eq!(
            error.fields.get("required"),
            Some(&FieldValue::Unsigned(40))
        );
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
        assert!(view
            .validate_against_lease(lease, DeviceEpoch::new(1))
            .is_ok());
        assert_eq!(
            view.validate_against_lease(
                BufferLease {
                    lease_id: LeaseId::new(7),
                    ..lease
                },
                DeviceEpoch::new(1),
            ),
            Err(ContractError::LeaseMismatch {
                view: ViewId::new(3),
                lease: LeaseId::new(7),
            })
        );
        assert!(matches!(
            view.validate_against_lease(lease, DeviceEpoch::new(2)),
            Err(ContractError::LeaseEpochMismatch { .. })
        ));
    }

    #[test]
    fn completion_distinguishes_timeout_from_terminal_unknown() {
        let token = CompletionToken {
            submission_id: SubmissionId::new(12),
            device_epoch: DeviceEpoch::new(1),
        };
        assert!(!CompletionDisposition::TimedOut { token }.is_terminal());
        assert!(CompletionDisposition::SubmittedUnknown { token: Some(token) }.is_terminal());
        assert_eq!(
            CompletionDisposition::TimedOut { token }.token(),
            Some(token)
        );
        assert!(CompletionDisposition::TimedOut { token }.validate().is_ok());
        assert!(CompletionDisposition::Submitted {
            token: CompletionToken {
                submission_id: SubmissionId::new(0),
                device_epoch: DeviceEpoch::new(1),
            }
        }
        .validate()
        .is_err());
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

    #[test]
    fn provider_submission_rejects_duplicate_or_empty_writebacks() {
        let writeback = BufferWriteback {
            view_id: ViewId::new(7),
            allocation_id: AllocationId::new(9),
            offset: 0,
            bytes: vec![1, 2, 3, 4],
        };
        let token = CompletionToken {
            submission_id: SubmissionId::new(12),
            device_epoch: DeviceEpoch::new(1),
        };
        let duplicate = ProviderSubmission {
            completion: CompletionDisposition::CompletedVisible { token },
            writebacks: vec![writeback.clone(), writeback.clone()],
        };
        assert!(matches!(
            duplicate.validate(),
            Err(ContractError::DuplicateWriteback { .. })
        ));

        let empty = ProviderSubmission {
            completion: CompletionDisposition::CompletedVisible { token },
            writebacks: vec![BufferWriteback {
                bytes: Vec::new(),
                ..writeback
            }],
        };
        assert_eq!(
            empty.validate(),
            Err(ContractError::ZeroLength("writeback"))
        );
    }

    fn completed_submission(trace: &ComputeTrace) -> ProviderSubmission {
        let mut writebacks = trace.passes[0]
            .buffers
            .iter()
            .filter(|view| view.access.is_writable())
            .map(|view| BufferWriteback {
                view_id: view.view_id,
                allocation_id: view.allocation_id,
                offset: view.offset,
                bytes: vec![0x5a; usize::try_from(view.length).unwrap()],
            })
            .collect::<Vec<_>>();
        writebacks.sort_by_key(|writeback| (writeback.allocation_id, writeback.view_id));
        ProviderSubmission {
            completion: CompletionDisposition::CompletedVisible {
                token: CompletionToken {
                    submission_id: SubmissionId::new(12),
                    device_epoch: trace.device_epoch,
                },
            },
            writebacks,
        }
    }

    #[test]
    fn submission_refuses_non_success_dispositions_and_unfinished_writebacks() {
        let trace = trace(vec![pass(4, vec![buffer(7, 0)])]);
        let mut submission = completed_submission(&trace);
        let token = submission.completion.token().unwrap();
        submission.completion = CompletionDisposition::Submitted { token };
        assert_eq!(
            submission.validate_for_trace(&trace),
            Err(ContractError::WritebackBeforeCompletion)
        );
        submission.writebacks.clear();
        submission.validate_for_trace(&trace).unwrap();
        for completion in [
            CompletionDisposition::NotSubmitted,
            CompletionDisposition::TimedOut { token },
            CompletionDisposition::Failed { token: Some(token) },
            CompletionDisposition::DeviceLost { token: Some(token) },
            CompletionDisposition::SubmittedUnknown { token: Some(token) },
        ] {
            submission.completion = completion;
            assert_eq!(
                submission.validate_for_trace(&trace),
                Err(ContractError::InvalidSubmissionCompletion(completion))
            );
        }
    }

    #[test]
    fn submission_requires_matching_epoch_and_single_pass() {
        let trace = trace(vec![pass(4, vec![buffer(7, 0)])]);
        let mut submission = completed_submission(&trace);
        submission.validate_for_trace(&trace).unwrap();
        submission.completion = CompletionDisposition::CompletedVisible {
            token: CompletionToken {
                device_epoch: DeviceEpoch::new(2),
                ..submission.completion.token().unwrap()
            },
        };
        assert_eq!(
            submission.validate_for_trace(&trace),
            Err(ContractError::CompletionEpochMismatch {
                expected: DeviceEpoch::new(1),
                actual: DeviceEpoch::new(2),
            })
        );

        let mut multi_pass = trace;
        multi_pass.passes.push(multi_pass.passes[0].clone());
        assert_eq!(
            completed_submission(&multi_pass).validate_for_trace(&multi_pass),
            Err(ContractError::UnsupportedSubmissionPassCount(2))
        );
    }

    #[test]
    fn submission_only_writes_back_bound_writable_identities() {
        let trace = trace(vec![pass(4, vec![buffer(7, 0)])]);
        for (allocation_id, view_id) in [
            (AllocationId::new(9), ViewId::new(8)),
            (AllocationId::new(10), ViewId::new(7)),
        ] {
            let mut submission = completed_submission(&trace);
            submission.writebacks[0].allocation_id = allocation_id;
            submission.writebacks[0].view_id = view_id;
            assert_eq!(
                submission.validate_for_trace(&trace),
                Err(ContractError::UnknownWriteback {
                    allocation: allocation_id,
                    view: view_id,
                })
            );
        }
        let submission = completed_submission(&trace);
        for access in [BufferAccess::Read, BufferAccess::Unused] {
            let mut read_only = trace.clone();
            read_only.passes[0].buffers[0].access = access;
            read_only.pipeline_contract.buffer_bindings[0].access = access;
            assert_eq!(
                submission.validate_for_trace(&read_only),
                Err(ContractError::ReadOnlyWriteback(ViewId::new(7)))
            );
        }
    }

    #[test]
    fn writeback_ranges_are_allocation_relative_and_cover_the_complete_view() {
        let mut view = buffer(7, 0);
        view.offset = 8;
        let trace = trace(vec![pass(4, vec![view])]);
        completed_submission(&trace)
            .validate_for_trace(&trace)
            .unwrap();
        for (offset, length) in [(0, 4), (7, 4), (9, 4), (8, 5)] {
            let mut submission = completed_submission(&trace);
            submission.writebacks[0].offset = offset;
            submission.writebacks[0].bytes.resize(length, 0);
            assert_eq!(
                submission.validate_for_trace(&trace),
                Err(ContractError::WritebackRangeOutOfBounds {
                    view: ViewId::new(7),
                    offset,
                    end: offset + u64::try_from(length).unwrap(),
                    view_offset: 8,
                    view_end: 12,
                })
            );
        }
        for (offset, length) in [(8, 3), (9, 3)] {
            let mut submission = completed_submission(&trace);
            submission.writebacks[0].offset = offset;
            submission.writebacks[0].bytes.resize(length, 0);
            assert_eq!(
                submission.validate_for_trace(&trace),
                Err(ContractError::IncompleteWriteback(ViewId::new(7)))
            );
        }
    }

    #[test]
    fn completed_host_readback_requires_every_writable_view_once_in_identity_order() {
        let first = buffer(7, 0);
        let mut second = buffer(6, 1);
        second.offset = 4;
        let mut trace = trace(vec![pass(4, vec![first, second])]);
        trace
            .pipeline_contract
            .buffer_bindings
            .push(BufferBindingContract {
                metal_binding: 1,
                ..trace.pipeline_contract.buffer_bindings[0].clone()
            });
        let submission = completed_submission(&trace);
        assert_eq!(submission.writebacks[0].view_id, ViewId::new(6));
        submission.validate_for_trace(&trace).unwrap();

        let mut missing = submission.clone();
        missing.writebacks.remove(0);
        assert_eq!(
            missing.validate_for_trace(&trace),
            Err(ContractError::MissingWriteback {
                allocation: AllocationId::new(9),
                view: ViewId::new(6),
            })
        );
        let mut duplicate = submission.clone();
        duplicate
            .writebacks
            .insert(0, duplicate.writebacks[0].clone());
        assert_eq!(
            duplicate.validate_for_trace(&trace),
            Err(ContractError::DuplicateWriteback {
                allocation: AllocationId::new(9),
                view: ViewId::new(6),
            })
        );
        let mut reversed = submission;
        reversed.writebacks.reverse();
        assert_eq!(
            reversed.validate_for_trace(&trace),
            Err(ContractError::NonCanonicalWritebackOrder)
        );
    }

    #[test]
    fn submit_only_never_returns_writebacks() {
        let mut trace = trace(vec![pass(4, vec![buffer(7, 0)])]);
        trace.completion_policy = CompletionPolicy::SubmitOnly;
        let mut submission = completed_submission(&trace);
        assert_eq!(
            submission.validate_for_trace(&trace),
            Err(ContractError::WritebackPolicyMismatch(
                CompletionPolicy::SubmitOnly
            ))
        );
        submission.writebacks.clear();
        submission.validate_for_trace(&trace).unwrap();
        submission.completion = CompletionDisposition::Submitted {
            token: submission.completion.token().unwrap(),
        };
        submission.validate_for_trace(&trace).unwrap();
    }
}
