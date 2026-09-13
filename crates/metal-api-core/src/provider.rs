//! Backend-neutral values for the first Metal provider contract.
//!
//! This module deliberately contains no `ash`, `metal`, QEMU type, guest
//! pointer, or provider-owned handle. It is the value boundary shared by a
//! native Metal implementation and a Vulkan implementation. The older
//! [`crate::ComputeExecutor`] snapshot API remains separate and is kept for
//! compatibility with the first offline harness.
//!
//! The render contract family ([`AttachmentFormat`], [`RenderAttachment`],
//! [`RenderPassDescriptor`]) is `research/docs/23-render-pipeline启动设计.md`
//! Step 1: types and validation only. Step 2 carries those passes through the
//! trace and the `MCC1` payload, and Step 3a adds the attachment-pipeline
//! sibling [`RenderPipelineContract`] as a value without wiring it to a
//! provider yet. Execution (admission, Vulkan render pass, native
//! `MTLRenderCommandEncoder`) is Step 3b+.
//! [`RenderPassDescriptor`]) is `research/docs/23-render-pipeline启动设计.md`.
//! Step 1 landed the types, Step 2 the trace carrier, the `MCC1` payload and
//! the capability bits, and Step 3c resolves a render attachment against the
//! trace's own resource table and the serial pool; executing a render pass
//! (Vulkan render pass, native `MTLRenderCommandEncoder`) is still open.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use crate::completion::{AbandonmentBudget, AbandonmentLedger, AbandonmentOutcome};

/// Current version of the pure-value provider trace schema.
pub const PROVIDER_SCHEMA_VERSION: u16 = 2;

/// Maximum distinct logical views collected before one command-buffer submit.
/// This shared bounded-resource policy applies to single-pass traces too.
pub const MAX_SERIAL_RESOURCES: usize = 64;

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

/// One owner-registered host address range a provider may import without
/// copying — for the VM line this is a guest RAM region whose host mapping is
/// stable for the registration's lifetime (`research/docs/19`). The type is
/// deliberately narrow: it proves the range's shape and identity so a lease can
/// be derived from it, and it does not claim anything about guest pages,
/// dirty tracking or invalidation, which stay owner-side.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostRegion {
    pub lease_id: LeaseId,
    pub owner_epoch: DeviceEpoch,
    pub host_pointer: usize,
    pub length: u64,
    /// Page size the owner registered the region with. Lease reservations cut
    /// from the region must be aligned to it.
    pub page_size: u64,
}

impl HostRegion {
    /// Validate the region's identity, pointer, alignment and extent. Refuses
    /// zero identities, null or unaligned pointers, zero or unaligned lengths,
    /// a page size that is not a power of two, and ranges that overflow.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.lease_id.is_zero() {
            return Err(ContractError::InvalidIdentity("host region lease id"));
        }
        if self.owner_epoch.is_zero() {
            return Err(ContractError::InvalidIdentity("host region owner epoch"));
        }
        if self.host_pointer == 0 {
            return Err(ContractError::NullHostPointer(self.lease_id));
        }
        if self.page_size == 0 || !self.page_size.is_power_of_two() {
            return Err(ContractError::InvalidHostRegionPageSize(self.page_size));
        }
        if self.length == 0 {
            return Err(ContractError::ZeroLength("host region"));
        }
        if !self.length.is_multiple_of(self.page_size) {
            return Err(ContractError::UnalignedHostRegion {
                field: "length",
                value: self.length,
                page_size: self.page_size,
            });
        }
        if !u64::try_from(self.host_pointer)
            .map(|pointer| pointer.is_multiple_of(self.page_size))
            .unwrap_or(false)
        {
            return Err(ContractError::UnalignedHostRegion {
                field: "pointer",
                value: u64::try_from(self.host_pointer).unwrap_or(u64::MAX),
                page_size: self.page_size,
            });
        }
        u64::try_from(self.host_pointer)
            .ok()
            .and_then(|pointer| pointer.checked_add(self.length))
            .ok_or(ContractError::ArithmeticOverflow("host region range"))?;
        Ok(())
    }

    /// Derive the borrowed backing for one page-aligned window of the region.
    /// The window is validated against the region before a lease is produced,
    /// so a provider never receives a pointer outside the registration.
    pub fn borrowed_window(
        &self,
        allocation_id: AllocationId,
        offset: u64,
        length: u64,
    ) -> Result<BorrowedLease, ContractError> {
        self.validate()?;
        if allocation_id.is_zero() {
            return Err(ContractError::InvalidIdentity("allocation id"));
        }
        if length == 0 {
            return Err(ContractError::ZeroLength("host region window"));
        }
        for (field, value) in [("offset", offset), ("length", length)] {
            if !value.is_multiple_of(self.page_size) {
                return Err(ContractError::UnalignedHostRegion {
                    field,
                    value,
                    page_size: self.page_size,
                });
            }
        }
        let end = offset
            .checked_add(length)
            .ok_or(ContractError::ArithmeticOverflow("host region window"))?;
        if end > self.length {
            return Err(ContractError::HostRegionWindowOutOfBounds {
                end,
                region_length: self.length,
            });
        }
        let pointer = u64::try_from(self.host_pointer)
            .map_err(|_| ContractError::ArithmeticOverflow("host region pointer"))?
            .checked_add(offset)
            .ok_or(ContractError::ArithmeticOverflow(
                "host region window pointer",
            ))?;
        let pointer = usize::try_from(pointer)
            .map_err(|_| ContractError::ArithmeticOverflow("host region window pointer"))?;
        BorrowedLease::new(
            LeaseReservation {
                lease: BufferLease {
                    lease_id: self.lease_id,
                    allocation_id,
                    owner_epoch: self.owner_epoch,
                },
                offset,
                length,
            },
            pointer,
        )
    }
}

/// Page-aligned dirty ranges one allocation accumulated from device
/// writebacks. The owner consumes this to flush guest pages; the type is pure,
/// so it can be produced from a submission's writebacks without a provider
/// callback (`research/docs/19` step 3).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirtySet {
    page_size: u64,
    ranges: Vec<(u64, u64)>,
}

impl DirtySet {
    /// A set over one page size; refuses zero and non-power-of-two sizes.
    pub fn new(page_size: u64) -> Result<Self, ContractError> {
        if page_size == 0 || !page_size.is_power_of_two() {
            return Err(ContractError::InvalidHostRegionPageSize(page_size));
        }
        Ok(Self {
            page_size,
            ranges: Vec::new(),
        })
    }

    pub fn page_size(&self) -> u64 {
        self.page_size
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Mark the pages covering one byte range. Adjacent or overlapping ranges
    /// coalesce, so the result is the canonical minimal cover.
    pub fn mark(&mut self, offset: u64, length: u64) -> Result<(), ContractError> {
        if length == 0 {
            return Ok(());
        }
        let end = offset
            .checked_add(length)
            .ok_or(ContractError::ArithmeticOverflow("dirty range"))?;
        let start_page = offset / self.page_size * self.page_size;
        let end_page = end
            .checked_add(self.page_size - 1)
            .ok_or(ContractError::ArithmeticOverflow("dirty range"))?
            / self.page_size
            * self.page_size;
        self.ranges.push((start_page, end_page));
        self.ranges.sort_unstable();
        let mut merged = Vec::with_capacity(self.ranges.len());
        for (start, end) in self.ranges.drain(..) {
            match merged.last_mut() {
                Some((_, last_end)) if start <= *last_end => {
                    if end > *last_end {
                        *last_end = end;
                    }
                }
                _ => merged.push((start, end)),
            }
        }
        self.ranges = merged;
        Ok(())
    }

    /// The canonical page-aligned ranges, ascending and disjoint.
    pub fn ranges(&self) -> &[(u64, u64)] {
        &self.ranges
    }

    /// Every writeback's written range, applied in order.
    pub fn mark_writebacks(&mut self, writebacks: &[BufferWriteback]) -> Result<(), ContractError> {
        for writeback in writebacks {
            let length = u64::try_from(writeback.bytes.len())
                .map_err(|_| ContractError::ArithmeticOverflow("writeback length"))?;
            self.mark(writeback.offset, length)?;
        }
        Ok(())
    }
}

/// One owner-registered guest window: a lease reservation plus the state that
/// decides when its host mapping may be reclaimed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestWindow {
    pub lease: LeaseId,
    pub allocation_id: AllocationId,
    pub offset: u64,
    pub length: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WindowState {
    Active,
    Retired,
}

/// Owner-side lifecycle for guest windows (`research/docs/19` step 4): a
/// window is active while any in-flight submission may touch it, becomes
/// retired when the lease observation says every reservation is known to be
/// retired, and only then may its host mapping be reclaimed. Reclamation is
/// explicit and returns the window, so the caller cannot silently drop the
/// backing of an active window.
#[derive(Debug, Default)]
pub struct GuestWindows {
    windows: BTreeMap<LeaseId, (GuestWindow, WindowState)>,
}

impl GuestWindows {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.windows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Register one window before its lease is imported.
    pub fn register(&mut self, window: GuestWindow) -> Result<(), ContractError> {
        if window.lease.is_zero() {
            return Err(ContractError::InvalidIdentity("guest window lease id"));
        }
        if window.allocation_id.is_zero() {
            return Err(ContractError::InvalidIdentity("guest window allocation id"));
        }
        if window.length == 0 {
            return Err(ContractError::ZeroLength("guest window"));
        }
        if self
            .windows
            .insert(window.lease, (window, WindowState::Active))
            .is_some()
        {
            return Err(ContractError::DuplicateLease(window.lease));
        }
        Ok(())
    }

    /// Record that the lease observation retired every reservation of this
    /// window. Retirement is idempotent; unknown windows are refused so a
    /// stray observation cannot mark another owner's window.
    pub fn retire(&mut self, lease: LeaseId) -> Result<(), ContractError> {
        let entry = self
            .windows
            .get_mut(&lease)
            .ok_or(ContractError::UnknownLease(lease))?;
        entry.1 = WindowState::Retired;
        Ok(())
    }

    /// Whether the window may be reclaimed now.
    pub fn is_reclaimable(&self, lease: LeaseId) -> bool {
        matches!(self.windows.get(&lease), Some((_, WindowState::Retired)))
    }

    /// Remove and return a retired window's registration. An active window is
    /// refused and left registered, which is the invariant that keeps a host
    /// mapping alive while anything may still touch it.
    pub fn reclaim(&mut self, lease: LeaseId) -> Result<GuestWindow, ContractError> {
        if !self.is_reclaimable(lease) {
            if self.windows.contains_key(&lease) {
                return Err(ContractError::GuestWindowStillActive(lease));
            }
            return Err(ContractError::UnknownLease(lease));
        }
        Ok(self
            .windows
            .remove(&lease)
            .expect("reclaimable window is registered")
            .0)
    }
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

/// Texture formats admitted by the first texture increment
/// (`research/docs/16` §4.1). The list is deliberately closed: a provider must
/// refuse an unlisted format instead of guessing a mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureFormat {
    R32Uint,
    R32Float,
    Rgba8Unorm,
    Bgra8Unorm,
}

impl TextureFormat {
    /// Tightly packed bytes one texel occupies in this format. Sampling and
    /// row padding are provider concerns; this is the byte extent the contract
    /// validates a source against.
    pub const fn bytes_per_texel(self) -> u64 {
        match self {
            Self::R32Uint | Self::R32Float => 4,
            Self::Rgba8Unorm | Self::Bgra8Unorm => 4,
        }
    }
}

/// Texture dimensionality admitted by the first texture increment. Array and
/// multisample variants carry their extra dimension explicitly so validation
/// can reject inconsistent combinations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureType {
    D1,
    D1Array,
    D2,
    D2Array,
    D2Multisample,
    D2MultisampleArray,
    D3,
}

impl TextureType {
    pub const fn is_array(self) -> bool {
        matches!(
            self,
            Self::D1Array | Self::D2Array | Self::D2MultisampleArray
        )
    }

    pub const fn is_multisample(self) -> bool {
        matches!(self, Self::D2Multisample | Self::D2MultisampleArray)
    }
}

/// How a shader uses one texture binding. The first increment admits read-only
/// sampling and refuses storage writeback (`research/docs/16` §4.1).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureAccess {
    Sampled,
    Storage,
    Unused,
}

impl TextureAccess {
    pub const fn is_writable(self) -> bool {
        matches!(self, Self::Storage)
    }
}

/// How the caller supplies a texture's initial contents. Mirrors
/// [`BufferSource`]; a lease covers the whole texture in the first increment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextureSource {
    /// Offline/test input owned by the trace. The provider may copy it.
    OwnedBytes(Vec<u8>),
    /// Contents and lifetime are supplied by an owner-issued staged lease.
    StagedLease(LeaseId),
    /// Provider may use the owner's backing without copying.
    BorrowedNoCopy(LeaseId),
}

impl TextureSource {
    pub const fn lease_id(&self) -> Option<LeaseId> {
        match self {
            Self::OwnedBytes(_) => None,
            Self::StagedLease(lease_id) | Self::BorrowedNoCopy(lease_id) => Some(*lease_id),
        }
    }
}

/// A logical Metal texture view. Dimensions are texels; the source length is
/// bytes and is validated against a tightly packed layout of
/// `width * height * depth * array_length * sample_count` texels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextureView {
    pub view_id: ViewId,
    pub metal_binding: u32,
    pub allocation_id: AllocationId,
    pub texture_type: TextureType,
    pub format: TextureFormat,
    pub width: u64,
    pub height: u64,
    pub depth: u64,
    pub array_length: u64,
    pub sample_count: u64,
    pub access: TextureAccess,
    pub source: TextureSource,
}

impl TextureView {
    /// Total tightly packed byte extent of one texture view.
    pub fn expected_bytes(&self) -> Result<u64, ContractError> {
        let texels = self
            .width
            .checked_mul(self.height)
            .and_then(|extent| extent.checked_mul(self.depth))
            .and_then(|extent| extent.checked_mul(self.array_length))
            .and_then(|extent| extent.checked_mul(self.sample_count))
            .ok_or(ContractError::ArithmeticOverflow("texture extent"))?;
        texels
            .checked_mul(self.format.bytes_per_texel())
            .ok_or(ContractError::ArithmeticOverflow("texture bytes"))
    }

    /// Structural validation only. It does not decide whether a provider
    /// supports the format or access; that is a capability refusal.
    pub fn validate_shape(&self) -> Result<(), ContractError> {
        if self.view_id.is_zero() {
            return Err(ContractError::InvalidIdentity("texture view id"));
        }
        if self.allocation_id.is_zero() {
            return Err(ContractError::InvalidIdentity("allocation id"));
        }
        for (axis, dimension) in [self.width, self.height, self.depth]
            .into_iter()
            .enumerate()
        {
            if dimension == 0 {
                return Err(ContractError::ZeroDimension {
                    field: "texture",
                    axis,
                });
            }
        }
        if self.array_length == 0 {
            return Err(ContractError::ZeroLength("texture array"));
        }
        if self.sample_count == 0 {
            return Err(ContractError::ZeroLength("texture sample count"));
        }
        if self.texture_type.is_multisample() {
            if self.sample_count < 2 {
                return Err(ContractError::TextureSampleCountMismatch {
                    texture_type: self.texture_type,
                    sample_count: self.sample_count,
                });
            }
        } else if self.sample_count != 1 {
            return Err(ContractError::TextureSampleCountMismatch {
                texture_type: self.texture_type,
                sample_count: self.sample_count,
            });
        }
        if !self.texture_type.is_array() && self.array_length != 1 {
            return Err(ContractError::TextureArrayLengthMismatch {
                texture_type: self.texture_type,
                array_length: self.array_length,
            });
        }
        let expected = self.expected_bytes()?;
        if let TextureSource::OwnedBytes(bytes) = &self.source {
            let actual = u64::try_from(bytes.len())
                .map_err(|_| ContractError::ArithmeticOverflow("owned texture length"))?;
            if actual != expected {
                return Err(ContractError::SourceLengthMismatch {
                    view: self.view_id,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
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
    /// The compute half of the registration, as before.
    pub contract: PipelineContract,
    /// The render half, when this registration compiles the two stages a
    /// [`RenderPassDescriptor`] names.
    ///
    /// Carrying it in the table entry is what lets core admission compare an
    /// attachment with the pipeline the pass names (review item I3,
    /// 2026-09-14). Before this field existed the trace could not say what a
    /// render pass would render with, so the `color_format` agreement was
    /// checked by each provider against its own registry — a lookup core cannot
    /// perform for a caller-supplied table.
    ///
    /// `None` means "this entry carries the compute half only", which is the
    /// pre-render entry every compute registration mints and the only shape the
    /// `SUBMIT_REQUEST` (`0x03`) wire layout can express. Both halves stay
    /// independent: a compute pass naming an entry that also renders is that
    /// provider's registry question, not a core rule (`docs/23` §4.1).
    pub render: Option<RenderPipelineContract>,
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
    /// Texture bindings for this pass. Both providers execute sampled texture
    /// bindings: the Vulkan upload places linear-image rows at the driver's
    /// `VkSubresourceLayout.rowPitch` and the native provider uses
    /// `MTLTexture`, and the v11/v12 suites run them over every rail
    /// (`research/docs/16` §4.5, §4.9).
    pub textures: Vec<TextureView>,
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
        // Texture bindings validated here; a provider that cannot execute them
        // refuses the trace during admission with its own capability error
        // (`research/docs/16` §4.6).
        for texture in &self.textures {
            texture.validate_shape()?;
            if views.contains_key(&texture.view_id) {
                return Err(ContractError::DuplicateView(texture.view_id));
            }
            // A texture and a buffer may share the Metal argument index (the
            // translator reports texture 0 / buffer 0), so only the Vulkan
            // descriptor namespace is checked for uniqueness here.
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

// ---------------------------------------------------------------------------
// Render contract, Step 1: value types and validation only.
//
// This block is `research/docs/23-render-pipeline启动设计.md` §3, Step 1. It
// adds the value types and their validation and nothing else: no execution, no
// wire-format change, no `ComputeTrace` wiring and no provider call site. The
// first increment is deliberately minimal — one colour attachment, one
// non-indexed draw, no dynamic state — and "not supported yet" is expressed by
// a missing field or by a validator refusal rather than by a default value, so
// a trace cannot imply state the execution step does not set (`docs/23` §3.2).
//
// Step 2 owns the execution wiring: admission and capability bits, the `MCC1`
// payload for a tagged pass union, the Vulkan `COLOR_ATTACHMENT_OPTIMAL` path
// with `vkCmdCopyImageToBuffer`, and the native `MTLRenderCommandEncoder`
// (`docs/23` §4.1, §6, §7.1). When that lands, keep this header pointing at the
// design doc rather than restating it.
// ---------------------------------------------------------------------------

/// Colour attachment formats expressible by the render contract.
///
/// The code values are deliberately the same as the existing `MCC1`
/// texture-format codes of `crates/metal-api-ipc/src/command_codec.rs`
/// (`R32Uint = 0`, `R32Float = 1`, `Rgba8Unorm = 2`, `Bgra8Unorm = 3`), so
/// Step 2 reuses that codec instead of maintaining a second mapping.
///
/// The list is closed and **contains no sRGB variant**: `docs/23` §7.4 defers
/// sRGB to the presentation track, and an sRGB attachment would silently
/// change the bytes the byte parity compares.
///
/// `Bgra8Unorm` and `R32Float` are admitted alongside the mandatory
/// `Rgba8Unorm` because `docs/23` §3.1 names both as first-increment
/// candidates and both already exist in [`TextureFormat`], so admitting them
/// costs no new format mapping. The first milestone only uses `Rgba8Unorm`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttachmentFormat {
    /// `VK_FORMAT_R8G8B8A8_UNORM` / `MTLPixelFormat::RGBA8Unorm`.
    Rgba8Unorm,
    /// `VK_FORMAT_B8G8R8A8_UNORM` / `MTLPixelFormat::BGRA8Unorm`.
    Bgra8Unorm,
    /// `VK_FORMAT_R32_SFLOAT` / `MTLPixelFormat::R32Float`.
    R32Float,
    /// `VK_FORMAT_R32_UINT` / `MTLPixelFormat::R32Uint`.
    ///
    /// Expressible for trace symmetry with [`TextureFormat`] and with the v11
    /// sampled-texture rail, but **refused** by the first render increment: its
    /// texels are integers, while [`LoadOp::Clear`] and the fragment output
    /// both carry colour bytes (`docs/23` §3.2, §7.4), so an integer
    /// attachment needs a value domain the first increment does not model. The
    /// code stays stable so Step 2 can enable it deliberately.
    R32Uint,
}

impl AttachmentFormat {
    /// Formats the first render increment admits as colour attachments. All
    /// three are 4-byte texel formats, which is what makes the fixed 4-byte
    /// [`ClearColor`] well formed.
    pub const ADMITTED: [Self; 3] = [Self::Rgba8Unorm, Self::Bgra8Unorm, Self::R32Float];

    /// Tightly packed bytes one texel occupies in this format. `rowPitch` and
    /// readback alignment stay provider concerns, exactly as they are for
    /// [`TextureFormat`].
    pub const fn bytes_per_texel(self) -> u64 {
        match self {
            Self::Rgba8Unorm | Self::Bgra8Unorm | Self::R32Float | Self::R32Uint => 4,
        }
    }

    /// Stable wire code, identical to the `MCC1` texture-format codes.
    pub const fn code(self) -> u8 {
        match self {
            Self::R32Uint => 0,
            Self::R32Float => 1,
            Self::Rgba8Unorm => 2,
            Self::Bgra8Unorm => 3,
        }
    }

    /// Inverse of [`AttachmentFormat::code`]. An unknown code is a decoder
    /// error, not a silent default.
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::R32Uint),
            1 => Some(Self::R32Float),
            2 => Some(Self::Rgba8Unorm),
            3 => Some(Self::Bgra8Unorm),
            _ => None,
        }
    }

    /// The matching [`TextureFormat`], so Step 2 can reuse the existing
    /// texture upload/readback plumbing instead of a parallel format path.
    pub const fn as_texture_format(self) -> TextureFormat {
        match self {
            Self::Rgba8Unorm => TextureFormat::Rgba8Unorm,
            Self::Bgra8Unorm => TextureFormat::Bgra8Unorm,
            Self::R32Float => TextureFormat::R32Float,
            Self::R32Uint => TextureFormat::R32Uint,
        }
    }

    /// Whether the first render increment admits this format as a colour
    /// attachment. See [`AttachmentFormat::R32Uint`] for the one refusal.
    pub const fn is_admitted_for_color_attachment(self) -> bool {
        matches!(self, Self::Rgba8Unorm | Self::Bgra8Unorm | Self::R32Float)
    }
}

/// The four tightly packed texel bytes a [`LoadOp::Clear`] writes into a colour
/// attachment.
///
/// The value is carried as bytes rather than as a `u32` or a float: `docs/23`
/// §3.5 fixes the byte-parity granularity at texel bytes and records that a
/// float clear such as `0.5` converts to `0x80` on Lavapipe but `0x7f` on
/// NVIDIA and dzn, so a float clear is not parity-stable. Bytes in memory order
/// also remove any endianness question from the contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClearColor {
    pub bytes: [u8; 4],
}

impl ClearColor {
    /// Bytes per clear value. Every admitted attachment format is exactly four
    /// bytes per texel ([`AttachmentFormat::bytes_per_texel`]); admitting a
    /// wider format requires widening this payload first.
    pub const BYTES: usize = 4;

    pub const fn new(bytes: [u8; 4]) -> Self {
        Self { bytes }
    }
}

/// How a colour attachment's contents are established when a render pass
/// begins (`research/docs/23` §3.1).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoadOp {
    /// Fill every texel with this colour before drawing. `docs/23` §1.3 uses a
    /// preset sentinel here so "the pass never ran" cannot pass the parity.
    Clear(ClearColor),
    /// Keep the attachment's previous contents.
    Load,
    /// Leave the previous contents undefined. Carried so the wire format has a
    /// fixed field set, but refused by the first render increment: the compared
    /// bytes would depend on state no earlier pass defined.
    DontCare,
}

/// How a colour attachment's contents are handed on after a render pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StoreOp {
    /// Store the pass's writes. This is what makes the attachment comparable.
    Store,
    /// Discard the pass's writes. Carried for the wire format's completeness
    /// and refused by the first render increment (`docs/23` §3.6): a discarded
    /// attachment must not be able to pass as "landed correctly".
    DontCare,
}

/// The colour attachments the first render increment admits. One, because
/// `docs/23` §3.3 defers multi-target rendering until `render_targets`
/// locations are mapped; the cap also stops a trace from smuggling an
/// attachment list past the wire format before that mapping exists. Step 2
/// grows the matching `ProviderCapabilities::max_color_attachments` field with
/// this value (`docs/23` §4.2).
pub const MAX_COLOR_ATTACHMENTS: usize = 1;

/// Vertices in the first milestone's single non-indexed draw: the full-screen
/// triangle generated from `vertex_id` (`research/docs/23` §1.2).
pub const FULL_SCREEN_TRIANGLE_VERTICES: u32 = 3;

/// One colour attachment of a render pass.
///
/// It references an existing resource by identity instead of embedding a
/// [`TextureView`]: `TextureView` carries a [`TextureAccess`] whose current
/// values are `Sampled`/`Storage`/`Unused` and whose codes are already pinned
/// by the `MCC1` codec, so adding a colour-attachment value there would change
/// the wire format (`docs/23` §2.1, §4.1). The attachment restates the shape
/// fields the first render increment needs instead.
///
/// Single-sample only: MSAA (`sample_count > 1`), depth/stencil and
/// array/cube attachments are all outside the first increment (`docs/23` §3.3)
/// and are expressed here by the absence of those fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderAttachment {
    pub view_id: ViewId,
    pub allocation_id: AllocationId,
    pub format: AttachmentFormat,
    pub width: u64,
    pub height: u64,
    pub load: LoadOp,
    pub store: StoreOp,
}

impl RenderAttachment {
    /// Tightly packed byte extent of the attachment, in the same texel-level
    /// unit the byte parity compares (`docs/23` §3.5). Provider row pitches and
    /// allocations are not part of the contract.
    pub fn expected_bytes(&self) -> Result<u64, ContractError> {
        self.width
            .checked_mul(self.height)
            .and_then(|texels| texels.checked_mul(self.format.bytes_per_texel()))
            .ok_or(ContractError::ArithmeticOverflow("attachment bytes"))
    }

    /// Structural validation only. Capability refusals (whether a device can
    /// render to this format at all) belong to admission in Step 2.
    pub fn validate_shape(&self) -> Result<(), ContractError> {
        if self.view_id.is_zero() {
            return Err(ContractError::InvalidIdentity("attachment view id"));
        }
        if self.allocation_id.is_zero() {
            return Err(ContractError::InvalidIdentity("attachment allocation id"));
        }
        for (axis, dimension) in [self.width, self.height].into_iter().enumerate() {
            if dimension == 0 {
                return Err(ContractError::ZeroDimension {
                    field: "attachment",
                    axis,
                });
            }
        }
        self.expected_bytes()?;
        if !self.format.is_admitted_for_color_attachment() {
            return Err(ContractError::UnsupportedAttachmentFormat(self.format));
        }
        if matches!(self.load, LoadOp::DontCare) {
            return Err(ContractError::UnsupportedAttachmentLoadOp(self.load));
        }
        if matches!(self.store, StoreOp::DontCare) {
            return Err(ContractError::UnsupportedAttachmentStoreOp(self.store));
        }
        Ok(())
    }
}

/// The first increment's render pass: one colour attachment, one non-indexed
/// draw, no dynamic state (`research/docs/23` §3.1).
///
/// Deliberately absent fields, i.e. the features `docs/23` §3.3 schedules
/// later: MSAA (no `sample_count`), depth/stencil attachments, MRT (the
/// attachment list is capped at [`MAX_COLOR_ATTACHMENTS`] until
/// `render_targets` locations are mapped), indexed and instanced draws (no
/// index or instance field) and dynamic state beyond the explicit viewport (no
/// scissor, blend, cull or winding).
///
/// This type is not referenced by [`ComputeTrace`] yet: Step 1 fixes the shape,
/// Step 2 makes `passes` a tagged union, extends the `MCC1` payload and teaches
/// the providers to execute it (`docs/23` §4.1, §6).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderPassDescriptor {
    /// The registered pipeline that supplies the vertex and fragment entries.
    pub pipeline: PipelineId,
    /// Colour attachments. The first increment admits exactly one.
    pub color_attachments: Vec<RenderAttachment>,
    /// `[origin_x, origin_y, width, height]`. The first increment accepts only
    /// the attachment-covering default `(0, 0, width, height)`: the viewport is
    /// explicit so a trace cannot imply viewport state the execution step does
    /// not set, and so a later dynamic-viewport extension is a deliberate
    /// change (`docs/23` §3.1).
    pub viewport: [u32; 4],
    /// Vertices of the single non-indexed draw. The first milestone draws
    /// [`FULL_SCREEN_TRIANGLE_VERTICES`].
    pub vertices: u32,
    /// The present action this pass hands its own attachment on to, or `None`
    /// for the offscreen-only pass.
    ///
    /// This is `research/docs/24` §3.5's shape one — present as the render
    /// pass's tail action — chosen by `docs/24` §4.1 and executed here: the
    /// descriptor hangs off the pass it renders, which is what makes §3.3's
    /// first ordering rule ("the target's writer completes before the present")
    /// structurally unbreakable rather than merely checked. A compute-only or
    /// offscreen trace leaves the field `None` and keeps the pre-present bytes
    /// exactly (`docs/24` §4.3).
    pub present: Option<PresentDescriptor>,
}

impl RenderPassDescriptor {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.pipeline.is_zero() {
            return Err(ContractError::InvalidIdentity("render pipeline id"));
        }
        if self.color_attachments.is_empty() {
            return Err(ContractError::EmptyAttachmentList);
        }
        if self.color_attachments.len() > MAX_COLOR_ATTACHMENTS {
            return Err(ContractError::AttachmentLimitExceeded {
                requested: self.color_attachments.len(),
                maximum: MAX_COLOR_ATTACHMENTS,
            });
        }
        for attachment in &self.color_attachments {
            attachment.validate_shape()?;
        }
        if self.vertices != FULL_SCREEN_TRIANGLE_VERTICES {
            return Err(ContractError::DrawVertexCountMismatch {
                expected: FULL_SCREEN_TRIANGLE_VERTICES,
                actual: self.vertices,
            });
        }
        let [origin_x, origin_y, width, height] = self.viewport;
        for (axis, dimension) in [width, height].into_iter().enumerate() {
            if dimension == 0 {
                return Err(ContractError::ZeroDimension {
                    field: "viewport",
                    axis: axis + 2,
                });
            }
        }
        if origin_x != 0 || origin_y != 0 {
            return Err(ContractError::ViewportOriginUnsupported {
                origin: [origin_x, origin_y],
            });
        }
        // The list is non-empty and capped at one, so the first attachment is
        // the only one to compare the viewport against.
        let attachment = &self.color_attachments[0];
        if u64::from(width) != attachment.width || u64::from(height) != attachment.height {
            return Err(ContractError::ViewportExtentMismatch {
                viewport: [width, height],
                attachment: [attachment.width, attachment.height],
            });
        }
        // The present action is validated by Step 1's own rules, not a second
        // copy of them: `validate_against` is the same entry point the
        // standalone Step 1 tests exercised, so a trace cannot reach execution
        // through a weaker gate than the one the contract already published.
        // `docs/24` §4.2 leaves the capability half (whether a snapshot can
        // present at all) to admission, which is why nothing here reads a
        // `ProviderCapabilities`.
        if let Some(present) = &self.present {
            present.validate_against(self)?;
        }
        Ok(())
    }
}

/// Which of a render pipeline's two compiled entry points a refusal is about.
///
/// Both entries live in one [`RenderPipelineContract`], so an entry-level
/// refusal has to name the stage that produced it: a bare entry name would not
/// say which of the two fields to fix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderPipelineStage {
    /// The stage that runs once per vertex (`vertex` in Metal Shading Language).
    Vertex,
    /// The stage that runs once per covered fragment (`fragment`).
    Fragment,
}

impl RenderPipelineStage {
    /// Lowercase spelling used by `Display` output and test failure messages.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }
}

/// How the vertex stage of a [`RenderPipelineContract`] receives its vertices.
///
/// `research/docs/23` §1.2 draws the first milestone's full-screen triangle
/// from `vertex_id` alone, so the pipeline binds no vertex buffer. The layout is
/// a value rather than an empty attribute list on purpose: Step 4 has to add a
/// vertex-buffer value beside [`VertexLayout::None`], and an empty `Vec` would
/// make "this pipeline has no vertex attributes" indistinguishable from
/// "vertex attributes are not decoded yet" in a wire format that is frozen
/// before that mapping exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VertexLayout {
    /// No vertex buffer is bound; positions come from the vertex index.
    None,
}

/// Provider-admission metadata for one registered render pipeline: the render
/// sibling of [`PipelineContract`].
///
/// Field set and refusals only. No trace, provider, capability default or
/// `MCC1` field references this value yet, so registering one changes no
/// behavior (`research/docs/23` §6 Step 3a). Step 3b pairs it with the two
/// translated artifacts and Step 4 executes it. [`PipelineContract`] stays the
/// compute-side contract and keeps its field semantics.
///
/// Entry organization: the first increment compiles one reviewed source module
/// twice, once per stage, the way [`CompiledComputePipeline`] pairs one
/// [`FunctionIdentity`] with one provider-owned [`crate::PipelineArtifact`]. The
/// two names therefore address two distinct stage functions of that module, so
/// duplicate entry names are refused instead of silently compiling one function
/// for both stages. Digest and source fields are deliberately absent here; Step
/// 3b's registration is where a [`FunctionIdentity`] is attached per stage,
/// exactly as the compute path does it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderPipelineContract {
    /// Entry point of the vertex-stage build of the module.
    pub vertex_entry: String,
    /// Entry point of the fragment-stage build of the module.
    pub fragment_entry: String,
    /// Colour-attachment format both stages were compiled against. One format,
    /// because the first increment admits one colour attachment
    /// ([`MAX_COLOR_ATTACHMENTS`]).
    pub color_format: AttachmentFormat,
    /// The vertex input shape the vertex stage was compiled for.
    pub vertex_layout: VertexLayout,
}

impl RenderPipelineContract {
    /// Structural validation of the contract alone. Whether a device can render
    /// to this format at all is admission's question, exactly as it is for
    /// [`RenderAttachment::validate_shape`].
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.vertex_entry.trim().is_empty() {
            return Err(ContractError::EmptyRenderPipelineEntry(
                RenderPipelineStage::Vertex,
            ));
        }
        if self.fragment_entry.trim().is_empty() {
            return Err(ContractError::EmptyRenderPipelineEntry(
                RenderPipelineStage::Fragment,
            ));
        }
        if self.vertex_entry == self.fragment_entry {
            // The fragment entry is the later of the two fields, so it is the
            // one that repeats the vertex entry.
            return Err(ContractError::DuplicateRenderPipelineEntry(
                RenderPipelineStage::Fragment,
            ));
        }
        if !self.color_format.is_admitted_for_color_attachment() {
            // The pipeline's format reuses the attachment's variant: an admitted
            // pipeline format is a subset of an admitted attachment format, so
            // one message and one capability slug stay correct for both sides
            // of the pair.
            return Err(ContractError::UnsupportedAttachmentFormat(
                self.color_format,
            ));
        }
        Ok(())
    }

    /// Cross-contract check against the pass that names this pipeline.
    ///
    /// The pass's own shape rules stay [`RenderPassDescriptor::validate`]'s job;
    /// this adds only the agreement the pass cannot check by itself, because it
    /// carries a [`PipelineId`] and not the pipeline's compiled format. An empty
    /// attachment list is reported as the pass's own
    /// [`ContractError::EmptyAttachmentList`] rather than as a vacuous
    /// agreement about formats that are not there.
    pub fn validate_against(&self, pass: &RenderPassDescriptor) -> Result<(), ContractError> {
        self.validate()?;
        if pass.color_attachments.is_empty() {
            return Err(ContractError::EmptyAttachmentList);
        }
        for attachment in &pass.color_attachments {
            if attachment.format != self.color_format {
                return Err(ContractError::RenderPipelineFormatMismatch {
                    pipeline: self.color_format,
                    attachment: attachment.format,
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Presentation contract, Step 1: value types and validation only.
//
// This block is `research/docs/24-presentation与swapchain设计.md` §3.5, Step 1.
// It adds `PresentMode`, `AcquirePolicy`, `InitialState` and `PresentTarget`
// with their structural validation and nothing else: no line-format field, no
// capability bit, no execution, no provider call site and no `ComputeTrace`
// wiring. Step 2 owns the wire shape (`docs/24` §4.1, §6 Step 2) and Step 3
// owns the Vulkan "readable swapchain equivalent" (`docs/24` §3.6).
//
// The first increment admits exactly one present target, one image and `Fifo`
// (`docs/24` §3.1). Every widening is expressed as "the field exists but the
// value is refused" rather than as a missing field, so a trace cannot imply
// state the execution step does not set — the rule `docs/23` §3.2 fixed for the
// render contract, applied to the present action.
// ---------------------------------------------------------------------------

/// How a finished present target is handed on (`research/docs/24` §3.1).
///
/// The variants are the four `VkPresentModeKHR` modes and carry their Vulkan
/// codes, because the first target is the Vulkan rail (`docs/24` §5.2) and
/// `docs/24` §7.1 records all four as available on this machine. Only
/// [`PresentMode::Fifo`] is admitted by the first increment: it is the one mode
/// Vulkan requires every implementation to support, so it needs no
/// driver-specific preference to be honest about, while picking another mode
/// would first need cross-driver parity evidence (`docs/24` §3.4) and, on the
/// Apple side, a different vocabulary entirely (`docs/24` §7.5).
///
/// The three non-FIFO variants sit beside `AttachmentFormat::R32Uint` and
/// `LoadOp::DontCare` as "expressible on the wire, refused by this increment":
/// they pin the code values Step 2 encodes while keeping the refusal a
/// deliberate, typed decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentMode {
    /// `VK_PRESENT_MODE_IMMEDIATE_KHR`. Expressible, refused by the first
    /// increment.
    Immediate,
    /// `VK_PRESENT_MODE_MAILBOX_KHR`. Expressible, refused by the first
    /// increment.
    Mailbox,
    /// `VK_PRESENT_MODE_FIFO_KHR`. The only admitted mode.
    Fifo,
    /// `VK_PRESENT_MODE_FIFO_RELAXED_KHR`. Expressible, refused by the first
    /// increment.
    FifoRelaxed,
}

impl PresentMode {
    /// Modes the first increment admits. One, so the trace cannot select a mode
    /// whose parity has not been argued for.
    pub const ADMITTED: [Self; 1] = [Self::Fifo];

    /// Stable wire code, identical to the `VkPresentModeKHR` values so Step 2
    /// reuses that mapping instead of maintaining a second one.
    pub const fn code(self) -> u8 {
        match self {
            Self::Immediate => 0,
            Self::Mailbox => 1,
            Self::Fifo => 2,
            Self::FifoRelaxed => 3,
        }
    }

    /// Inverse of [`PresentMode::code`]. An unknown code is a decoder error,
    /// not a silent default.
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Immediate),
            1 => Some(Self::Mailbox),
            2 => Some(Self::Fifo),
            3 => Some(Self::FifoRelaxed),
            _ => None,
        }
    }

    /// Whether the first increment admits this mode. See
    /// [`PresentMode::Fifo`] for the one admission.
    pub const fn is_admitted_for_present(self) -> bool {
        matches!(self, Self::Fifo)
    }
}

/// How a present target's ownership is acquired before a pass writes it
/// (`research/docs/24` §3.1).
///
/// The first increment admits only [`AcquirePolicy::Blocking`]. A bounded wait
/// is a deadline object, and `docs/13`'s deadline semantics have nothing to
/// hang on here, so `docs/24` §3.1 records the first increment as
/// "blocking only" and §3.4 defers acquire timeouts together with frame drops
/// and suboptimal rebuilds.
///
/// [`AcquirePolicy::Timeout`] carries the nanosecond budget such a deadline
/// would need. Keeping the value expressible is what lets the refusal name the
/// budget that has to wait, instead of the field being absent and the trace
/// silently losing its request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AcquirePolicy {
    /// Wait for the target without a deadline.
    Blocking,
    /// Wait at most this many nanoseconds. Expressible, refused by the first
    /// increment.
    Timeout(u64),
}

impl AcquirePolicy {
    /// Stable wire code for Step 2. `Blocking` stays `0` so a future decoder
    /// reads the first increment's only admitted value as the default.
    pub const fn code(self) -> u8 {
        match self {
            Self::Blocking => 0,
            Self::Timeout(_) => 1,
        }
    }

    /// The nanosecond budget, or `None` for an unbounded wait.
    pub const fn timeout_nanos(self) -> Option<u64> {
        match self {
            Self::Blocking => None,
            Self::Timeout(nanos) => Some(nanos),
        }
    }

    /// Whether the first increment admits this policy. See
    /// [`AcquirePolicy::Blocking`] for the one admission.
    pub const fn is_admitted_for_present(self) -> bool {
        matches!(self, Self::Blocking)
    }
}

/// What a present target holds before the pass that fills it runs
/// (`research/docs/24` §3.1).
///
/// The sentinel is what makes "the present never happened" falsifiable, exactly
/// as [`LoadOp::Clear`] does for the render track (`docs/23` §1.3): the bytes
/// are compared after `wait`, so a target still holding its sentinel cannot pass
/// as a present.
///
/// The sentinel carries its bytes as a `Vec<u8>` where the §3.5 sketch writes
/// `[u8; 4]`. The invariant the contract actually needs is "one tightly packed
/// texel", i.e. exactly `format.bytes_per_texel()` bytes; a length fixed at four
/// would make that agreement unreachable and therefore unverified, and the
/// sketch is explicitly field-level rather than compilable (`docs/24` §3.5).
/// The length rule is enforced by [`PresentTarget::validate_shape`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InitialState {
    /// Pre-fill the target with these bytes before the pass runs. `docs/24`
    /// §5.3 counts one `copy_in` for this state.
    Sentinel(Vec<u8>),
    /// Leave the target's previous contents undefined. `docs/24` §5.3 counts no
    /// `copy_in` for this state, because nothing has to be written before the
    /// pass.
    Undefined,
}

impl InitialState {
    /// The preset bytes, or `None` when the target declares no initial state.
    pub fn sentinel(&self) -> Option<&[u8]> {
        match self {
            Self::Sentinel(bytes) => Some(bytes),
            Self::Undefined => None,
        }
    }
}

/// Present targets the first increment admits in one pass: one.
///
/// `docs/24` §3.1 fixes "one target, one present, single buffering". The cap
/// also stops a trace from smuggling a longer target list past the wire format
/// before that mapping exists. Step 2 grows the matching
/// `ProviderCapabilities::max_present_targets` field with this value
/// (`docs/24` §4.2).
pub const MAX_PRESENT_TARGETS: usize = 1;

/// Images the first increment admits behind one present target: one.
///
/// `docs/24` §3.4 defers multi-buffering, because a rotating image needs a
/// state machine for "which image is in flight" that `docs/13`'s in-flight
/// reclamation has not settled yet. The field exists on
/// [`PresentTarget::image_count`] but only accepts this value, so a caller
/// asking for more is refused instead of silently getting single buffering
/// (`docs/24` §3.1).
pub const MAX_PRESENT_IMAGE_COUNT: u32 = 1;

/// One presentable target: the resource whose contents the present hands on
/// (`research/docs/24` §3.1, §3.5).
///
/// Like [`RenderAttachment`], it references an existing resource by identity and
/// restates the shape fields the increment needs; it does not embed a
/// [`TextureView`] (whose [`TextureAccess`] codes the `MCC1` codec has already
/// pinned) and it allocates nothing. `docs/24` §3.2 makes the caller reserve the
/// allocation under the existing lease semantics, so hazard tracking and
/// write-back merging need no new mechanism (`docs/14`).
///
/// Deliberately absent fields, i.e. the features `docs/24` §3.4 schedules
/// later: a surface handle, a resize/out-of-date path, a colour-space or sRGB
/// variant, and any per-image in-flight state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresentTarget {
    /// Allocation the target's bytes live in, in the same namespace as
    /// buffers, textures and render attachments.
    pub allocation_id: AllocationId,
    /// The view of that allocation the present hands on. It must be the view
    /// the pass renders into, which is what [`PresentDescriptor::source`]
    /// restates.
    pub view_id: ViewId,
    /// Format of the target. `docs/24` §3.2 inherits it from the source
    /// attachment rather than letting a target choose its own, because byte
    /// parity over two different formats would compare channel order and
    /// encoding rather than the pass's output.
    pub format: AttachmentFormat,
    /// Tightly packed texel width, in the same unit as
    /// [`RenderAttachment::width`].
    pub width: u64,
    /// Tightly packed texel height, in the same unit as
    /// [`RenderAttachment::height`].
    pub height: u64,
    /// Images behind this target. The first increment admits
    /// [`MAX_PRESENT_IMAGE_COUNT`].
    pub image_count: u32,
    /// What the target holds before the pass runs. See [`InitialState`].
    pub initial: InitialState,
}

impl PresentTarget {
    /// Tightly packed byte extent of the target, in the same texel-level unit
    /// the byte parity compares (`docs/23` §3.5). Provider row pitches, image
    /// layouts and allocation sizes are not part of the contract.
    pub fn expected_bytes(&self) -> Result<u64, ContractError> {
        self.width
            .checked_mul(self.height)
            .and_then(|texels| texels.checked_mul(self.format.bytes_per_texel()))
            .ok_or(ContractError::ArithmeticOverflow("present target bytes"))
    }

    /// Structural validation only. Whether a device can present to this format
    /// or extent at all, and whether the target can actually be acquired, are
    /// admission's questions in Step 2 (`docs/24` §4.2).
    pub fn validate_shape(&self) -> Result<(), ContractError> {
        if self.view_id.is_zero() {
            return Err(ContractError::InvalidIdentity("present target view id"));
        }
        if self.allocation_id.is_zero() {
            return Err(ContractError::InvalidIdentity(
                "present target allocation id",
            ));
        }
        for (axis, dimension) in [self.width, self.height].into_iter().enumerate() {
            if dimension == 0 {
                return Err(ContractError::ZeroDimension {
                    field: "present target",
                    axis,
                });
            }
        }
        self.expected_bytes()?;
        if !self.format.is_admitted_for_color_attachment() {
            // A present target reuses the render contract's admitted format set:
            // the target's format is inherited from a colour attachment
            // (`docs/24` §3.2), so the same capability narrowing and the same
            // refusal slug stay correct for both halves of the pair.
            return Err(ContractError::UnsupportedAttachmentFormat(self.format));
        }
        if self.image_count != MAX_PRESENT_IMAGE_COUNT {
            return Err(ContractError::PresentImageCountUnsupported {
                requested: self.image_count,
                maximum: MAX_PRESENT_IMAGE_COUNT,
            });
        }
        if let InitialState::Sentinel(bytes) = &self.initial {
            let expected = self.format.bytes_per_texel();
            let actual = bytes.len() as u64;
            if actual != expected {
                return Err(ContractError::PresentSentinelLengthMismatch {
                    format: self.format,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }
}

/// The first increment's present action on one render pass
/// (`research/docs/24` §3.5, shape one: `present` as the render pass's tail
/// action).
///
/// Shape one is what makes `docs/24` §3.3's first ordering rule — the target's
/// writer completes before the present — *structurally* unbreakable: the present
/// descriptor hangs off the render pass, so a trace cannot present a target no
/// pass rendered. `docs/24` §9 keeps that choice formally open until Step 2, so
/// this type is written to fit either shape: it names the pass's source view
/// explicitly instead of assuming its own position in a pass list, and
/// [`PresentDescriptor::validate_against`] is a pure function of the pass it is
/// asked about rather than a method on a pass that owns it.
///
/// This type is not referenced by [`ComputeTrace`] yet: Step 1 fixes the shape,
/// Step 2 makes the pass carry it and extends the `MCC1` payload (`docs/24`
/// §4.1).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresentDescriptor {
    /// The target the present hands on. See [`PresentTarget`].
    pub target: PresentTarget,
    /// View the pass renders into, which the target is handed on as. `docs/24`
    /// §3.1 requires it to be the same view as the pass's colour attachment, so
    /// the same lease and byte-range semantics carry over unchanged
    /// (`docs/14`); the agreement is checked by
    /// [`PresentDescriptor::validate_against`], which can only be asked once a
    /// pass is in hand.
    pub source: ViewId,
    /// Present mode. The first increment admits [`PresentMode::Fifo`] only.
    pub mode: PresentMode,
    /// How the target's ownership is acquired. The first increment admits
    /// [`AcquirePolicy::Blocking`] only.
    pub acquire: AcquirePolicy,
}

impl PresentDescriptor {
    /// Structural validation of the descriptor alone.
    ///
    /// The mode and the acquire policy are first-increment narrowings on a
    /// well-formed request, so they are stated as refusals here rather than left
    /// to each provider. The agreement with the pass the present hangs off is
    /// [`PresentDescriptor::validate_against`]'s job, because it needs a pass.
    pub fn validate(&self) -> Result<(), ContractError> {
        self.target.validate_shape()?;
        if !self.mode.is_admitted_for_present() {
            return Err(ContractError::PresentModeUnsupported(self.mode));
        }
        if !self.acquire.is_admitted_for_present() {
            return Err(ContractError::PresentAcquirePolicyUnsupported(self.acquire));
        }
        Ok(())
    }

    /// Cross-check against the pass this present hands on.
    ///
    /// Four rules, all from `docs/24` §3.1/§3.2: the source must *be* one of the
    /// pass's colour attachments, the target must restate that attachment's view
    /// identity, the target's allocation must be that attachment's allocation
    /// ("the same lease and byte-range semantics"), and the target's format and
    /// extent are inherited from it rather than chosen. Those are the only rules
    /// this adds; the pass's own shape rules stay
    /// [`RenderPassDescriptor::validate`]'s job, and an empty attachment list is
    /// reported as the pass's own [`ContractError::EmptyAttachmentList`] rather
    /// than as a vacuous agreement about an attachment that is not there — the
    /// same reporting [`RenderPipelineContract::validate_against`] chose.
    ///
    /// Deliberately **not** checked here: whether the target's [`InitialState`]
    /// agrees with the source attachment's [`LoadOp`]. `docs/24` states the
    /// format and extent inheritance (§3.2) but leaves the pre-pass contents to
    /// the count 口径 Step 4 settles (§5.3, §6), so inventing a rule here would
    /// bind the trace to a decision the design has not made.
    pub fn validate_against(&self, pass: &RenderPassDescriptor) -> Result<(), ContractError> {
        self.validate()?;
        if pass.color_attachments.is_empty() {
            return Err(ContractError::EmptyAttachmentList);
        }
        let Some(attachment) = pass
            .color_attachments
            .iter()
            .find(|attachment| attachment.view_id == self.source)
        else {
            return Err(ContractError::PresentSourceUnknown {
                source: self.source,
            });
        };
        if self.target.view_id != self.source {
            return Err(ContractError::PresentTargetViewMismatch {
                source: self.source,
                target: self.target.view_id,
            });
        }
        if self.target.allocation_id != attachment.allocation_id {
            return Err(ContractError::PresentTargetAllocationMismatch {
                source: self.source,
                target: self.target.allocation_id,
                attachment: attachment.allocation_id,
            });
        }
        if attachment.format != self.target.format {
            return Err(ContractError::PresentFormatMismatch {
                source: self.source,
                target: self.target.format,
                attachment: attachment.format,
            });
        }
        if attachment.width != self.target.width || attachment.height != self.target.height {
            return Err(ContractError::PresentExtentMismatch {
                source: self.source,
                target: [self.target.width, self.target.height],
                attachment: [attachment.width, attachment.height],
            });
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

/// One entry of a trace's ordered pass list.
///
/// `ComputeTrace` used to carry `Vec<ComputePass>` directly. The render track
/// needs a second pass shape, so the entry became a tagged union: the
/// discriminant is always present, and "no passes" is spelled by an empty
/// list, never by a combination of absent optionals. The `Compute` arm keeps
/// the exact value of the pre-render pass type, so every existing provider
/// path only has to unwrap a variant it already understands.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TracePass {
    /// One dispatch over the compute pipeline named by `ComputePass::pipeline`.
    Compute(ComputePass),
    /// One offscreen colour render pass (`research/docs/23` §3).
    Render(RenderPassDescriptor),
}

impl TracePass {
    /// The compute payload, or `None` for a render entry.
    pub fn as_compute(&self) -> Option<&ComputePass> {
        match self {
            Self::Compute(pass) => Some(pass),
            Self::Render(_) => None,
        }
    }

    /// Mutable access to the compute payload, or `None` for a render entry.
    pub fn as_compute_mut(&mut self) -> Option<&mut ComputePass> {
        match self {
            Self::Compute(pass) => Some(pass),
            Self::Render(_) => None,
        }
    }

    /// The render payload, or `None` for a compute entry.
    pub fn as_render(&self) -> Option<&RenderPassDescriptor> {
        match self {
            Self::Compute(_) => None,
            Self::Render(pass) => Some(pass),
        }
    }
}

impl From<ComputePass> for TracePass {
    fn from(pass: ComputePass) -> Self {
        Self::Compute(pass)
    }
}

impl From<RenderPassDescriptor> for TracePass {
    fn from(pass: RenderPassDescriptor) -> Self {
        Self::Render(pass)
    }
}

/// The immutable value trace shared by both provider implementations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeTrace {
    pub schema_version: u16,
    pub device_epoch: DeviceEpoch,
    pub operation_id: OperationId,
    /// Explicit metadata for every pipeline referenced by a pass. Each entry
    /// belongs to this trace's device epoch and must be used at least once.
    pub pipelines: Vec<CompiledComputePipeline>,
    pub encoder_dispatch_type: DispatchType,
    /// Ordered compute and render passes. `MCC1` keeps the pre-render byte
    /// layout for a compute-only list and tags every entry in the extended
    /// layout, so this field is the single source of pass order.
    pub passes: Vec<TracePass>,
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

    /// Iterate allocation records in identity order.
    pub fn allocations(&self) -> impl Iterator<Item = AllocationRecord> + '_ {
        self.allocations.values().copied()
    }

    /// Iterate lease reservations in identity order.
    pub fn leases(&self) -> impl Iterator<Item = LeaseReservation> + '_ {
        self.leases.values().copied()
    }

    pub fn validate_trace(&self, trace: &ComputeTrace) -> Result<(), ContractError> {
        trace.validate()?;
        let mut views =
            BTreeMap::<ViewId, (AllocationId, u64, u64, BufferSourceKind, Option<LeaseId>)>::new();
        let mut ranges = Vec::<(usize, AllocationId, ViewId, u64, u64, BufferAccess)>::new();
        for (pass_index, pass) in trace.passes.iter().enumerate() {
            // Attachments are not `BufferView`s; their range admission joins
            // the render execution step (`research/docs/23` §6 Step 4).
            let Some(pass) = pass.as_compute() else {
                continue;
            };
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

/// Completion-driven lifetime tracking for owner-issued buffer leases.
///
/// The neutral memory owner registers each lease reservation, binds the
/// completion token of every submission that references it, and releases the
/// backing only when every bound token has retirement evidence. Observation
/// alone is not enough: `Submitted`, `TimedOut`, `Cancelled` and
/// `SubmittedUnknown` do not prove the GPU stopped using the backing. A
/// provider that establishes retirement out of band can call
/// [`LeaseLedger::retire`]. Device loss is a teardown guarantee and releases
/// every lease.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LeaseLedger {
    leases: BTreeMap<LeaseId, LeaseState>,
    bindings: BTreeMap<TokenKey, BTreeSet<LeaseId>>,
    device_lost: bool,
}

type TokenKey = (DeviceEpoch, SubmissionId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LeaseState {
    reservation: LeaseReservation,
    outstanding: usize,
}

/// Observable state of one lease under a [`LeaseLedger`].
///
/// A lease with any outstanding token is still held. `Released` means either
/// every bound token retired or a device-loss teardown cleared the ledger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminalLeaseState {
    /// The number of bound tokens that have not retired. Non-zero means the
    /// backing is still held.
    Outstanding(usize),
    /// No token holds the lease; the owner may release the backing.
    Released,
}

/// Result of observing one completion token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeaseObservation {
    /// The observation does not prove GPU retirement; the lease stays held.
    Pending,
    /// The token is retired; a lease with no other outstanding token may now
    /// be released.
    Retired,
}

impl LeaseLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a lease reservation before binding submissions to it.
    pub fn register(&mut self, reservation: LeaseReservation) -> Result<(), ContractError> {
        let lease_id = reservation.lease.lease_id;
        if lease_id.is_zero() {
            return Err(ContractError::InvalidIdentity("lease id"));
        }
        if reservation.lease.allocation_id.is_zero() {
            return Err(ContractError::InvalidIdentity("lease allocation id"));
        }
        if reservation.lease.owner_epoch.is_zero() {
            return Err(ContractError::InvalidIdentity("lease owner epoch"));
        }
        reservation.end()?;
        if self.leases.contains_key(&lease_id) {
            return Err(ContractError::DuplicateLease(lease_id));
        }
        self.leases.insert(
            lease_id,
            LeaseState {
                reservation,
                outstanding: 0,
            },
        );
        Ok(())
    }

    /// Bind a submission token to a lease the submission references.
    ///
    /// Binding the same token twice to one lease is idempotent, so callers may
    /// walk a trace's views without deduplicating.
    pub fn bind(&mut self, lease_id: LeaseId, token: CompletionToken) -> Result<(), ContractError> {
        token.validate()?;
        let state = self
            .leases
            .get_mut(&lease_id)
            .ok_or(ContractError::UnknownLease(lease_id))?;
        let entry = self.bindings.entry(token_key(token)).or_default();
        if entry.insert(lease_id) {
            state.outstanding =
                state
                    .outstanding
                    .checked_add(1)
                    .ok_or(ContractError::ArithmeticOverflow(
                        "lease outstanding bindings",
                    ))?;
        }
        Ok(())
    }

    /// Record a completion observation for a bound token.
    ///
    /// The disposition must not carry a different token. Only retirement
    /// evidence (see [`disposition_retires_resources`]) releases the token;
    /// pending, cancelled, timed-out and unknown observations keep every bound
    /// lease held.
    pub fn observe(
        &mut self,
        token: CompletionToken,
        disposition: CompletionDisposition,
    ) -> Result<LeaseObservation, ContractError> {
        if let Some(observed) = disposition.token() {
            if observed != token {
                return Err(ContractError::InvalidSubmissionCompletion(disposition));
            }
        }
        if disposition_retires_resources(disposition) {
            self.retire(token);
            Ok(LeaseObservation::Retired)
        } else {
            Ok(LeaseObservation::Pending)
        }
    }

    /// Record out-of-band retirement for a token.
    ///
    /// Idempotent; an unknown token is ignored because observations may arrive
    /// in any order or after the lease was already released.
    pub fn retire(&mut self, token: CompletionToken) {
        let Some(lease_ids) = self.bindings.remove(&token_key(token)) else {
            return;
        };
        for lease_id in lease_ids {
            if let Some(state) = self.leases.get_mut(&lease_id) {
                state.outstanding = state.outstanding.saturating_sub(1);
            }
        }
    }

    /// Device teardown releases every lease regardless of outstanding tokens.
    pub fn device_lost(&mut self) {
        self.device_lost = true;
        self.bindings.clear();
        for state in self.leases.values_mut() {
            state.outstanding = 0;
        }
    }

    pub const fn is_device_lost(&self) -> bool {
        self.device_lost
    }

    pub fn contains(&self, lease_id: LeaseId) -> bool {
        self.leases.contains_key(&lease_id)
    }

    pub fn lease(&self, lease_id: LeaseId) -> Option<LeaseReservation> {
        self.leases.get(&lease_id).map(|state| state.reservation)
    }

    /// Number of bound tokens that have not retired, or `None` if unknown.
    pub fn outstanding(&self, lease_id: LeaseId) -> Option<usize> {
        self.leases.get(&lease_id).map(|state| state.outstanding)
    }

    /// Whether the owner may release the lease's backing.
    pub fn release_ready(&self, lease_id: LeaseId) -> bool {
        self.leases
            .get(&lease_id)
            .is_some_and(|state| self.device_lost || state.outstanding == 0)
    }

    /// Every tracked lease paired with its observable terminal state.
    ///
    /// The order is stable (lease id). A lease carrying outstanding tokens
    /// reports [`TerminalLeaseState::Outstanding`]; a lease whose tokens all
    /// retired, or every lease after device loss, reports
    /// [`TerminalLeaseState::Released`]. A released lease is removed, so it
    /// never appears twice and no dangling entry can hide here.
    pub fn leased(&self) -> Vec<(LeaseId, TerminalLeaseState)> {
        self.leases
            .iter()
            .map(|(lease_id, state)| {
                let state = if self.device_lost || state.outstanding == 0 {
                    TerminalLeaseState::Released
                } else {
                    TerminalLeaseState::Outstanding(state.outstanding)
                };
                (*lease_id, state)
            })
            .collect()
    }

    /// Remove and return a lease whose backing may be released.
    pub fn release(&mut self, lease_id: LeaseId) -> Option<LeaseReservation> {
        if !self.release_ready(lease_id) {
            return None;
        }
        self.leases.remove(&lease_id).map(|state| state.reservation)
    }

    /// Remove every lease whose backing may be released.
    pub fn release_all_ready(&mut self) -> Vec<LeaseReservation> {
        let ready: Vec<LeaseId> = self
            .leases
            .iter()
            .filter(|(_, state)| self.device_lost || state.outstanding == 0)
            .map(|(lease_id, _)| *lease_id)
            .collect();
        ready
            .into_iter()
            .filter_map(|lease_id| self.release(lease_id))
            .collect()
    }
}

fn token_key(token: CompletionToken) -> TokenKey {
    (token.device_epoch, token.submission_id)
}

/// Whether a completion observation proves the device no longer uses the
/// submission's resources.
///
/// `NotSubmitted` never reached the queue. `CompletedVisible` and `Failed` are
/// only published after terminal execution or after the fence/handler already
/// reported completion; providers use `SubmittedUnknown` for unknown queue or
/// wait state. `DeviceLost` is a teardown guarantee. `Submitted`, `TimedOut`,
/// `Cancelled` and `SubmittedUnknown` are not retirement evidence.
pub const fn disposition_retires_resources(disposition: CompletionDisposition) -> bool {
    matches!(
        disposition,
        CompletionDisposition::NotSubmitted
            | CompletionDisposition::CompletedVisible { .. }
            | CompletionDisposition::Failed { .. }
            | CompletionDisposition::DeviceLost { .. }
    )
}

/// Owner-issued lease backing staged into a provider.
///
/// `bytes` covers exactly the reservation window (`reservation.offset ..
/// reservation.offset + reservation.length`). A provider that advertises
/// [`StorageMode::StagedLease`] copies the bytes into provider-owned storage
/// before execution; the owner's [`LeaseLedger`] remains the authority for
/// when the owner backing may be dropped.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagedLease {
    pub reservation: LeaseReservation,
    pub bytes: Vec<u8>,
}

impl StagedLease {
    /// Build a staged lease and validate the reservation/length pairing.
    pub fn new(reservation: LeaseReservation, bytes: Vec<u8>) -> Result<Self, ContractError> {
        let staged = Self { reservation, bytes };
        staged.validate()?;
        Ok(staged)
    }

    /// Validate identities, the reservation range and the byte length.
    pub fn validate(&self) -> Result<(), ContractError> {
        let lease_id = self.reservation.lease.lease_id;
        if lease_id.is_zero() {
            return Err(ContractError::InvalidIdentity("staged lease id"));
        }
        if self.reservation.lease.allocation_id.is_zero() {
            return Err(ContractError::InvalidIdentity("staged lease allocation id"));
        }
        if self.reservation.lease.owner_epoch.is_zero() {
            return Err(ContractError::InvalidIdentity("staged lease owner epoch"));
        }
        self.reservation.end()?;
        let actual = u64::try_from(self.bytes.len())
            .map_err(|_| ContractError::ArithmeticOverflow("staged lease length"))?;
        if actual != self.reservation.length {
            return Err(ContractError::LeaseSourceLengthMismatch {
                lease: lease_id,
                expected: self.reservation.length,
                actual,
            });
        }
        Ok(())
    }

    /// Lease identity the bytes are staged for.
    pub const fn lease_id(&self) -> LeaseId {
        self.reservation.lease.lease_id
    }
}

/// Provider-side store of staged lease backing.
///
/// The registry owns copied bytes, not owner memory. Import refuses a duplicate
/// identity, and [`LeaseRegistry::view_bytes`] refuses a lease whose reservation
/// does not match the admitted resource snapshot.
#[derive(Debug, Default)]
pub struct LeaseRegistry {
    leases: Mutex<BTreeMap<LeaseId, StagedLease>>,
}

impl LeaseRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of imported leases.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether no lease is imported.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Stage the bytes for one lease. A duplicate identity is refused until
    /// [`LeaseRegistry::release`] drops the previous import.
    pub fn import(&self, staged: StagedLease) -> Result<(), ProviderError> {
        staged.validate().map_err(contract_error_refusal)?;
        let lease_id = staged.lease_id();
        let mut leases = self.lock();
        if leases.contains_key(&lease_id) {
            return Err(lease_error(
                "lease_already_imported",
                lease_id,
                ProviderErrorClass::Args,
            ));
        }
        leases.insert(lease_id, staged);
        Ok(())
    }

    /// Drop one imported lease.
    pub fn release(&self, lease_id: LeaseId) -> Result<(), ProviderError> {
        let mut leases = self.lock();
        if leases.remove(&lease_id).is_none() {
            return Err(lease_error(
                "lease_not_imported",
                lease_id,
                ProviderErrorClass::Args,
            ));
        }
        Ok(())
    }

    /// Resolve the view bytes for a staged lease.
    ///
    /// The admitted snapshot is authoritative: the staged reservation must
    /// match it exactly, and the view must fall inside it.
    pub fn view_bytes(
        &self,
        lease_id: LeaseId,
        view: &BufferView,
        device_epoch: DeviceEpoch,
        resources: &ResourceTableSnapshot,
    ) -> Result<Vec<u8>, ProviderError> {
        let leases = self.lock();
        let staged = leases
            .get(&lease_id)
            .ok_or_else(|| lease_error("lease_not_imported", lease_id, ProviderErrorClass::Args))?;
        let reservation = resources.lease(lease_id).ok_or_else(|| {
            lease_error("lease_not_admitted", lease_id, ProviderErrorClass::Resource)
        })?;
        if staged.reservation != reservation {
            return Err(lease_error(
                "lease_snapshot_mismatch",
                lease_id,
                ProviderErrorClass::Resource,
            ));
        }
        if reservation.lease.owner_epoch != device_epoch {
            return Err(lease_error(
                "lease_epoch_mismatch",
                lease_id,
                ProviderErrorClass::Resource,
            )
            .with_field("expected", FieldValue::Unsigned(device_epoch.get()))
            .with_field(
                "actual",
                FieldValue::Unsigned(reservation.lease.owner_epoch.get()),
            ));
        }
        let lease_end = reservation.end().map_err(contract_error_refusal)?;
        let view_end = view.offset.checked_add(view.length).ok_or_else(|| {
            contract_error_refusal(ContractError::ArithmeticOverflow("staged lease view range"))
        })?;
        let start = view.offset.checked_sub(reservation.offset).ok_or_else(|| {
            lease_error(
                "lease_range_out_of_bounds",
                lease_id,
                ProviderErrorClass::Resource,
            )
        })?;
        if view_end > lease_end {
            return Err(lease_error(
                "lease_range_out_of_bounds",
                lease_id,
                ProviderErrorClass::Resource,
            )
            .with_field("view_end", FieldValue::Unsigned(view_end))
            .with_field("lease_end", FieldValue::Unsigned(lease_end)));
        }
        let start = usize::try_from(start).map_err(|_| {
            contract_error_refusal(ContractError::ArithmeticOverflow("staged lease offset"))
        })?;
        let length = usize::try_from(view.length).map_err(|_| {
            contract_error_refusal(ContractError::ArithmeticOverflow(
                "staged lease view length",
            ))
        })?;
        let end = start.checked_add(length).ok_or_else(|| {
            contract_error_refusal(ContractError::ArithmeticOverflow("staged lease slice"))
        })?;
        staged
            .bytes
            .get(start..end)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| {
                lease_error(
                    "lease_range_out_of_bounds",
                    lease_id,
                    ProviderErrorClass::Resource,
                )
            })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<LeaseId, StagedLease>> {
        self.leases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Owner-issued no-copy lease backed by a host mapping in the provider's
/// address space.
///
/// The reservation window is `reservation.length` bytes starting at
/// `host_pointer`. The owner must keep that range valid, and at the same
/// address, until every submission that imported it has retirement evidence
/// and the provider has released the import. Unlike [`StagedLease`], the
/// provider never copies these bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BorrowedLease {
    pub reservation: LeaseReservation,
    pub host_pointer: usize,
}

impl BorrowedLease {
    /// Build a borrowed lease and validate its identity and pointer range.
    pub fn new(reservation: LeaseReservation, host_pointer: usize) -> Result<Self, ContractError> {
        let borrowed = Self {
            reservation,
            host_pointer,
        };
        borrowed.validate()?;
        Ok(borrowed)
    }

    /// Validate identities, the reservation range and the pointer range.
    pub fn validate(&self) -> Result<(), ContractError> {
        let lease_id = self.reservation.lease.lease_id;
        if lease_id.is_zero() {
            return Err(ContractError::InvalidIdentity("borrowed lease id"));
        }
        if self.reservation.lease.allocation_id.is_zero() {
            return Err(ContractError::InvalidIdentity(
                "borrowed lease allocation id",
            ));
        }
        if self.reservation.lease.owner_epoch.is_zero() {
            return Err(ContractError::InvalidIdentity("borrowed lease owner epoch"));
        }
        self.reservation.end()?;
        if self.host_pointer == 0 {
            return Err(ContractError::NullHostPointer(lease_id));
        }
        let length = usize::try_from(self.reservation.length)
            .map_err(|_| ContractError::ArithmeticOverflow("borrowed lease length"))?;
        self.host_pointer
            .checked_add(length)
            .ok_or(ContractError::ArithmeticOverflow("borrowed lease range"))?;
        Ok(())
    }

    /// Lease identity the pointer is borrowed for.
    pub const fn lease_id(&self) -> LeaseId {
        self.reservation.lease.lease_id
    }
}

/// Provider-side store of no-copy lease imports.
///
/// The registry tracks owner memory, not copies. [`BorrowedLeaseRegistry::retain`]
/// marks a submission that will read or write the owner mapping;
/// [`BorrowedLeaseRegistry::retire`] is called once the provider knows the GPU
/// can no longer use it. [`BorrowedLeaseRegistry::release`] refuses while any
/// retain is outstanding, so a provider cannot free an import under in-flight
/// work.
#[derive(Debug, Default)]
pub struct BorrowedLeaseRegistry {
    leases: Mutex<BTreeMap<LeaseId, BorrowedEntry>>,
}

#[derive(Debug)]
struct BorrowedEntry {
    lease: BorrowedLease,
    outstanding: u64,
}

/// A no-copy view resolved inside one imported reservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BorrowedView {
    pub pointer: usize,
    pub len: usize,
    /// Base of the imported reservation. Providers that map a whole
    /// reservation instead of the view window bind this pointer and apply
    /// [`BorrowedView::offset`].
    pub base_pointer: usize,
    /// Reservation length in bytes, the range covered by
    /// [`BorrowedView::base_pointer`].
    pub base_len: usize,
    /// Offset of [`BorrowedView::pointer`] from
    /// [`BorrowedView::base_pointer`].
    pub offset: usize,
    /// Valid bytes from `pointer` to the end of the imported reservation.
    pub capacity: usize,
}

impl BorrowedLeaseRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of imported leases.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether no lease is imported.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of outstanding retains, or `None` if the lease is unknown.
    pub fn outstanding(&self, lease_id: LeaseId) -> Option<u64> {
        self.lock().get(&lease_id).map(|entry| entry.outstanding)
    }

    /// Import one owner mapping. A duplicate identity is refused until
    /// [`BorrowedLeaseRegistry::release`] drops the previous import.
    pub fn import(&self, borrowed: BorrowedLease) -> Result<(), ProviderError> {
        borrowed.validate().map_err(contract_error_refusal)?;
        let lease_id = borrowed.lease_id();
        let mut leases = self.lock();
        if leases.contains_key(&lease_id) {
            return Err(lease_error(
                "lease_already_imported",
                lease_id,
                ProviderErrorClass::Args,
            ));
        }
        leases.insert(
            lease_id,
            BorrowedEntry {
                lease: borrowed,
                outstanding: 0,
            },
        );
        Ok(())
    }

    /// Retain one lease for a submission that will use the owner mapping.
    pub fn retain(&self, lease_id: LeaseId) -> Result<(), ProviderError> {
        let mut leases = self.lock();
        let entry = leases
            .get_mut(&lease_id)
            .ok_or_else(|| lease_error("lease_not_imported", lease_id, ProviderErrorClass::Args))?;
        entry.outstanding = entry.outstanding.checked_add(1).ok_or_else(|| {
            lease_error(
                "lease_retain_overflow",
                lease_id,
                ProviderErrorClass::Internal,
            )
        })?;
        Ok(())
    }

    /// Retain every listed lease, rolling back on the first refusal.
    pub fn retain_all(&self, lease_ids: &[LeaseId]) -> Result<(), ProviderError> {
        let mut retained = Vec::with_capacity(lease_ids.len());
        for &lease_id in lease_ids {
            if let Err(error) = self.retain(lease_id) {
                self.retire_all(&retained);
                return Err(error);
            }
            retained.push(lease_id);
        }
        Ok(())
    }

    /// Drop one retain. Idempotent for unknown leases because teardown may
    /// race with release.
    pub fn retire(&self, lease_id: LeaseId) {
        if let Some(entry) = self.lock().get_mut(&lease_id) {
            entry.outstanding = entry.outstanding.saturating_sub(1);
        }
    }

    /// Drop one retain for every listed lease.
    pub fn retire_all(&self, lease_ids: &[LeaseId]) {
        for &lease_id in lease_ids {
            self.retire(lease_id);
        }
    }

    /// Drop one imported lease. Refused while a retain is outstanding.
    pub fn release(&self, lease_id: LeaseId) -> Result<(), ProviderError> {
        let mut leases = self.lock();
        let entry = leases
            .get(&lease_id)
            .ok_or_else(|| lease_error("lease_not_imported", lease_id, ProviderErrorClass::Args))?;
        if entry.outstanding > 0 {
            return Err(
                lease_error("lease_in_use", lease_id, ProviderErrorClass::Resource)
                    .with_field("outstanding", FieldValue::Unsigned(entry.outstanding)),
            );
        }
        leases.remove(&lease_id);
        Ok(())
    }

    /// Resolve the view to owner memory without copying.
    ///
    /// The admitted snapshot is authoritative: the imported reservation must
    /// match it exactly, and the view must fall inside it.
    pub fn view_pointer(
        &self,
        lease_id: LeaseId,
        view: &BufferView,
        device_epoch: DeviceEpoch,
        resources: &ResourceTableSnapshot,
    ) -> Result<BorrowedView, ProviderError> {
        let leases = self.lock();
        let entry = leases
            .get(&lease_id)
            .ok_or_else(|| lease_error("lease_not_imported", lease_id, ProviderErrorClass::Args))?;
        let reservation = resources.lease(lease_id).ok_or_else(|| {
            lease_error("lease_not_admitted", lease_id, ProviderErrorClass::Resource)
        })?;
        if entry.lease.reservation != reservation {
            return Err(lease_error(
                "lease_snapshot_mismatch",
                lease_id,
                ProviderErrorClass::Resource,
            ));
        }
        if reservation.lease.owner_epoch != device_epoch {
            return Err(lease_error(
                "lease_epoch_mismatch",
                lease_id,
                ProviderErrorClass::Resource,
            )
            .with_field("expected", FieldValue::Unsigned(device_epoch.get()))
            .with_field(
                "actual",
                FieldValue::Unsigned(reservation.lease.owner_epoch.get()),
            ));
        }
        let lease_end = reservation.end().map_err(contract_error_refusal)?;
        let view_end = view.offset.checked_add(view.length).ok_or_else(|| {
            contract_error_refusal(ContractError::ArithmeticOverflow(
                "borrowed lease view range",
            ))
        })?;
        let start = view.offset.checked_sub(reservation.offset).ok_or_else(|| {
            lease_error(
                "lease_range_out_of_bounds",
                lease_id,
                ProviderErrorClass::Resource,
            )
        })?;
        if view_end > lease_end {
            return Err(lease_error(
                "lease_range_out_of_bounds",
                lease_id,
                ProviderErrorClass::Resource,
            )
            .with_field("view_end", FieldValue::Unsigned(view_end))
            .with_field("lease_end", FieldValue::Unsigned(lease_end)));
        }
        let start = usize::try_from(start).map_err(|_| {
            contract_error_refusal(ContractError::ArithmeticOverflow("borrowed lease offset"))
        })?;
        let len = usize::try_from(view.length).map_err(|_| {
            contract_error_refusal(ContractError::ArithmeticOverflow(
                "borrowed lease view length",
            ))
        })?;
        let pointer = entry.lease.host_pointer.checked_add(start).ok_or_else(|| {
            contract_error_refusal(ContractError::ArithmeticOverflow("borrowed lease pointer"))
        })?;
        let capacity = usize::try_from(lease_end - view.offset).map_err(|_| {
            contract_error_refusal(ContractError::ArithmeticOverflow("borrowed lease capacity"))
        })?;
        let base_len = usize::try_from(reservation.length).map_err(|_| {
            contract_error_refusal(ContractError::ArithmeticOverflow(
                "borrowed lease reservation length",
            ))
        })?;
        Ok(BorrowedView {
            pointer,
            len,
            base_pointer: entry.lease.host_pointer,
            base_len,
            offset: start,
            capacity,
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<LeaseId, BorrowedEntry>> {
        self.leases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn lease_error(slug: &'static str, lease_id: LeaseId, class: ProviderErrorClass) -> ProviderError {
    ProviderError::new(ProviderPhase::Resolve, class, slug)
        .expect("non-empty lease error slug")
        .with_field("lease", FieldValue::Unsigned(lease_id.get()))
}

/// Provider-side import of owner-issued lease backing.
///
/// A provider advertising [`StorageMode::StagedLease`] implements this trait.
/// Importing copies the bytes into provider-owned storage; the owner's
/// [`LeaseLedger`] remains the authority for when the owner backing may be
/// dropped.
pub trait LeaseImporter {
    /// Stage the bytes for `staged.reservation`.
    fn import_staged_lease(&self, staged: StagedLease) -> Result<(), ProviderError>;

    /// Drop the staged bytes for `lease_id`.
    fn release_staged_lease(&self, lease_id: LeaseId) -> Result<(), ProviderError>;
}

/// Provider-side import of owner-issued no-copy lease backing.
///
/// A provider advertising [`StorageMode::BorrowedNoCopy`] implements this
/// trait. The provider imports the owner mapping directly; it must not copy
/// the bytes, and it must keep every retain until the GPU can no longer use
/// the mapping.
pub trait NoCopyLeaseImporter {
    /// Required host-pointer alignment in bytes, or zero when the provider
    /// cannot import host memory.
    fn no_copy_alignment(&self) -> u64;

    /// Import `borrowed` without copying.
    ///
    /// # Safety
    ///
    /// `borrowed.host_pointer` must point to `borrowed.reservation.length`
    /// readable and writable bytes aligned to [`Self::no_copy_alignment`].
    /// Providers that map whole reservations, such as Metal no-copy buffers,
    /// additionally require `borrowed.reservation.length` to be a multiple of
    /// [`Self::no_copy_alignment`].
    /// The owner must keep that mapping valid and at the same address until
    /// every retained submission is retired and
    /// [`Self::release_borrowed_lease`] has returned.
    unsafe fn import_borrowed_lease(&self, borrowed: BorrowedLease) -> Result<(), ProviderError>;

    /// Drop one imported lease. Refused while a retain is outstanding.
    fn release_borrowed_lease(&self, lease_id: LeaseId) -> Result<(), ProviderError>;
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
        pipelines: vec![CompiledComputePipeline {
            device_epoch,
            pipeline_id: pipeline.pipeline_id,
            function: pipeline.function,
            contract: pipeline.pipeline_contract,
            // One owner-side submission carries a compute pipeline, and the
            // pre-render entry shape is the compute half only.
            render: None,
        }],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![TracePass::Compute(ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers,
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid,
                threads_per_threadgroup: local,
            },
            textures: Vec::new(),
        })],
        completion_policy: CompletionPolicy::HostReadback,
    };
    trace.validate()?;
    Ok(trace)
}

/// One view a trace declares in its own passes, reduced to the facts a render
/// attachment is admitted against.
///
/// A render attachment does not declare storage: it references an existing
/// resource by identity and restates the shape the first render increment draws
/// into (`RenderAttachment`'s documentation, `research/docs/23` §3.6).
/// Admission therefore resolves each attachment against the trace's own
/// resource table — the buffer and texture bindings of its compute passes — and
/// refuses an attachment that names a view the trace never declared, one whose
/// declaration covers a different allocation, or one whose declaration
/// describes a different extent. Resolving against that table is also what
/// keeps the first increment from opening a second resource-declaration
/// channel: allocation, epoch, lease and range admission stay the ones the
/// declaring pass already passed (`docs/23` §4.1).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeclaredView {
    Buffer {
        pass_index: usize,
        allocation_id: AllocationId,
        /// Offset of the view inside its allocation. The attachment's own byte
        /// range is this one, so two sibling views of one allocation can be
        /// compared without either of them naming the other
        /// (review item M3, 2026-09-14).
        offset: u64,
        length: u64,
        access: BufferAccess,
    },
    /// A 2D, single-sample, non-array view is the only texture shape the first
    /// increment can render into. Any other shape fails [`Self::covers`]
    /// instead of being reinterpreted as a flat extent.
    Texture {
        pass_index: usize,
        allocation_id: AllocationId,
        texture_type: TextureType,
        format: TextureFormat,
        width: u64,
        height: u64,
        depth: u64,
        array_length: u64,
        sample_count: u64,
        access: TextureAccess,
    },
}

impl DeclaredView {
    /// The trace entry that declared this view, so a refusal can point at it.
    const fn pass_index(self) -> usize {
        match self {
            Self::Buffer { pass_index, .. } | Self::Texture { pass_index, .. } => pass_index,
        }
    }

    const fn allocation_id(self) -> AllocationId {
        match self {
            Self::Buffer { allocation_id, .. } | Self::Texture { allocation_id, .. } => {
                allocation_id
            }
        }
    }

    /// The bytes this declaration occupies in its allocation.
    ///
    /// A buffer declaration states its own offset and length. A texture
    /// declaration states no offset ([`TextureView`] has none), so its packed
    /// extent is taken from the allocation's start — the same extent
    /// [`Self::extent_bytes`] reports. Both are compared with
    /// [`BufferRange::overlaps`], the comparator the in-flight hazard admission
    /// already uses (`research/docs/14` §3.2), so the render track does not
    /// grow a second notion of "these two uses share bytes".
    fn byte_range(self) -> BufferRange {
        match self {
            Self::Buffer {
                allocation_id,
                offset,
                length,
                ..
            } => BufferRange {
                allocation_id,
                offset,
                length,
            },
            Self::Texture { allocation_id, .. } => BufferRange {
                allocation_id,
                offset: 0,
                length: self.extent_bytes(),
            },
        }
    }

    /// Whether a compute pass writes this view.
    ///
    /// This is the first increment's whole hazard criterion for compute and
    /// render sharing bytes: a render pass stores its attachment into the
    /// referenced view, and a compute pass writing the same view leaves the two
    /// writers' order inexpressible in the trace, so the trace is refused
    /// rather than executed in an arbitrary order.
    ///
    /// A compute pass that only *reads* the view stays admissible, and the
    /// reason is the contract rather than an assumption: this increment fixes
    /// the execution order at "every compute pass, then every render pass", so a
    /// read *before* the store gets the bytes as they were and a read *after* it
    /// would not. The second shape is refused as
    /// [`ContractError::RenderPassOrderUnsupported`] instead of being silently
    /// reordered, which is what lets a provider execute the grouped order
    /// without changing what the trace means (review item I4, 2026-09-14;
    /// `research/docs/23` §3.6).
    const fn is_writable(self) -> bool {
        match self {
            Self::Buffer { access, .. } => access.is_writable(),
            Self::Texture { access, .. } => access.is_writable(),
        }
    }

    /// Tightly packed byte extent this declaration covers, in the same
    /// texel-level unit [`RenderAttachment::expected_bytes`] uses.
    ///
    /// Every factor is bounded by the declaration's own `validate_shape`, so
    /// the saturated arithmetic below cannot be reached with an overflow; it
    /// only exists to keep this helper free of a second error path.
    const fn extent_bytes(self) -> u64 {
        match self {
            Self::Buffer { length, .. } => length,
            Self::Texture {
                format,
                width,
                height,
                depth,
                array_length,
                sample_count,
                ..
            } => width
                .saturating_mul(height)
                .saturating_mul(depth)
                .saturating_mul(array_length)
                .saturating_mul(sample_count)
                .saturating_mul(format.bytes_per_texel()),
        }
    }

    /// Whether this declaration describes the shape the attachment restates, or
    /// the refusal that names the disagreement.
    ///
    /// A buffer declaration is compared by byte length, which is the unit the
    /// readback compares, so a refusal can name the two numbers. A texture
    /// declaration is compared by texel shape and format instead, because two
    /// byte-equal extents such as 4×1 and 2×2 are different render targets even
    /// though a flat byte count agrees: that refusal carries both shapes and
    /// both byte counts, so an operator can locate the field to fix
    /// (review item M4, 2026-09-14).
    fn admit_extent(
        self,
        pass_index: usize,
        attachment: &RenderAttachment,
    ) -> Result<(), ContractError> {
        match self {
            Self::Buffer { length, .. } => {
                let expected = attachment.expected_bytes()?;
                if expected == length {
                    Ok(())
                } else {
                    Err(ContractError::AttachmentExtentMismatch {
                        pass_index,
                        view: attachment.view_id,
                        expected,
                        declared: length,
                    })
                }
            }
            Self::Texture {
                texture_type,
                format,
                width,
                height,
                depth,
                array_length,
                sample_count,
                ..
            } => {
                let shape_agrees = texture_type == TextureType::D2
                    && depth == 1
                    && array_length == 1
                    && sample_count == 1
                    && width == attachment.width
                    && height == attachment.height
                    && format == attachment.format.as_texture_format();
                if shape_agrees {
                    return Ok(());
                }
                Err(ContractError::AttachmentTextureShapeMismatch {
                    pass_index,
                    view: attachment.view_id,
                    attachment_extent: [attachment.width, attachment.height],
                    attachment_format: attachment.format,
                    attachment_bytes: attachment.expected_bytes()?,
                    declared_type: texture_type,
                    declared_extent: [width, height],
                    declared_depth: depth,
                    declared_array_length: array_length,
                    declared_sample_count: sample_count,
                    declared_format: format,
                    declared_bytes: self.extent_bytes(),
                })
            }
        }
    }
}

impl ComputeTrace {
    /// The compute entries of `passes`, in trace order.
    ///
    /// Providers that cannot execute render passes refuse them during
    /// admission, so an admitted trace reaches a compute execution path with
    /// this iterator covering every entry.
    pub fn compute_passes(&self) -> impl Iterator<Item = &ComputePass> {
        self.passes.iter().filter_map(TracePass::as_compute)
    }

    /// The render entries of `passes`, in trace order.
    pub fn render_passes(&self) -> impl Iterator<Item = &RenderPassDescriptor> {
        self.passes.iter().filter_map(TracePass::as_render)
    }

    /// Every colour attachment in pass order, tagged with the index of the
    /// trace entry that carries it.
    ///
    /// Render admission and the serial pool both walk this list. A trace with
    /// no render entry yields nothing, so a compute-only trace keeps the
    /// pre-render pool and budget exactly (`research/docs/23` §3.6).
    pub fn attachments(&self) -> impl Iterator<Item = (usize, &RenderAttachment)> {
        self.passes
            .iter()
            .enumerate()
            .flat_map(|(pass_index, pass)| {
                pass.as_render().into_iter().flat_map(move |render| {
                    render
                        .color_attachments
                        .iter()
                        .map(move |attachment| (pass_index, attachment))
                })
            })
    }

    /// Whether this trace carries at least one render pass. This is the
    /// discriminator the `MCC1` encoder and render admission both read.
    pub fn has_render_passes(&self) -> bool {
        self.passes
            .iter()
            .any(|pass| matches!(pass, TracePass::Render(_)))
    }

    /// Every present action in pass order, tagged with the index of the trace
    /// entry that carries it (`research/docs/24` §3.5 shape one, where the
    /// present hangs off a render pass).
    ///
    /// Present admission, `MCC1` and the Step 3 execution step walk this list.
    /// A trace with no present action yields nothing, so the pre-present bytes,
    /// the serial pool and the render budget all keep their previous values
    /// (`docs/24` §4.3).
    pub fn present_actions(&self) -> impl Iterator<Item = (usize, &PresentDescriptor)> {
        self.passes
            .iter()
            .enumerate()
            .filter_map(|(pass_index, pass)| {
                pass.as_render()
                    .and_then(|render| render.present.as_ref())
                    .map(|present| (pass_index, present))
            })
    }

    /// Whether this trace carries at least one present action. A present action
    /// only exists on a render pass (shape one), so this is a narrowing of
    /// [`ComputeTrace::has_render_passes`] rather than an alternative to it.
    pub fn has_present_actions(&self) -> bool {
        self.present_actions().next().is_some()
    }

    /// Look up metadata without recursively validating the trace. Providers
    /// must still check this caller-supplied metadata against their registry.
    pub fn pipeline(&self, id: PipelineId) -> Result<&CompiledComputePipeline, ContractError> {
        self.pipelines
            .iter()
            .find(|pipeline| pipeline.pipeline_id == id)
            .ok_or(ContractError::UnknownPipeline(id))
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != PROVIDER_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchemaVersion(self.schema_version));
        }
        if self.passes.is_empty() {
            return Err(ContractError::EmptyTrace);
        }
        if self.device_epoch.is_zero() {
            return Err(ContractError::InvalidIdentity("device epoch"));
        }
        if self.operation_id.is_zero() {
            return Err(ContractError::InvalidIdentity("operation id"));
        }
        if self.pipelines.is_empty() {
            return Err(ContractError::EmptyPipelineTable);
        }
        let mut used = BTreeMap::new();
        for pipeline in &self.pipelines {
            if pipeline.pipeline_id.is_zero() {
                return Err(ContractError::InvalidIdentity("pipeline id"));
            }
            if pipeline.device_epoch != self.device_epoch {
                return Err(ContractError::PipelineEpochMismatch {
                    pipeline: pipeline.pipeline_id,
                    expected: self.device_epoch,
                    actual: pipeline.device_epoch,
                });
            }
            if used.insert(pipeline.pipeline_id, false).is_some() {
                return Err(ContractError::DuplicatePipeline(pipeline.pipeline_id));
            }
            pipeline.function.validate()?;
            pipeline.contract.validate()?;
            // The render half is validated wherever it is carried. Its
            // agreement with a render pass stays admission's job: a table entry
            // may carry one that only a compute pass names.
            if let Some(render) = &pipeline.render {
                render.validate()?;
            }
        }
        for pass in &self.passes {
            match pass {
                TracePass::Compute(pass) => {
                    if pass.pipeline.is_zero() {
                        return Err(ContractError::InvalidIdentity("pipeline id"));
                    }
                    let pipeline = self.pipeline(pass.pipeline)?;
                    pass.validate(&pipeline.contract)?;
                }
                TracePass::Render(pass) => {
                    // `RenderPassDescriptor::validate` repeats the identity
                    // check; the lookup additionally proves the referenced
                    // pipeline is in this trace's table and epoch.
                    pass.validate()?;
                    self.pipeline(pass.pipeline)?;
                }
            }
            let pipeline = match pass {
                TracePass::Compute(pass) => pass.pipeline,
                TracePass::Render(pass) => pass.pipeline,
            };
            *used
                .get_mut(&pipeline)
                .expect("pipeline lookup checked the metadata table") = true;
        }
        for (pipeline, was_used) in used {
            if !was_used {
                return Err(ContractError::UnusedPipeline(pipeline));
            }
        }
        Ok(())
    }

    /// Validate the serial resource-reuse subset supported by command-buffer
    /// providers. Each pass binds a subset of the complete logical view pool,
    /// and may select another registered pipeline or permute its bound views.
    /// Access follows each binding's reflection; view identity, allocation,
    /// range and source contents stay fixed. The repeated source bytes describe
    /// one initial upload. Views first bound by later passes are still collected
    /// and validated before submission, then uploaded with the whole pool;
    /// they do not introduce CPU uploads during execution. Writes from earlier
    /// passes remain visible to later passes, and readback happens after the
    /// final pass. General aliasing remains subject to resource admission.
    ///
    /// Render entries join the same walk (`research/docs/23` §3.6). Each
    /// attachment is resolved against the views the trace declares, its
    /// restated extent has to agree with the declaration it resolves to, and it
    /// must not race a compute pass that writes overlapping bytes of the same
    /// allocation. The trace's own order is part of that admission: because the
    /// increment executes every compute pass before every render pass, a compute
    /// pass that follows the store and binds the same bytes is refused too.
    /// Attachments are pooled by view identity, so the budget that bounds one
    /// serial submission covers them; a compute-only trace takes the pre-render
    /// path unchanged.
    pub fn validate_serial_buffer_reuse(&self) -> Result<(), ContractError> {
        self.validate()?;
        if self.passes.len() > 1 && self.encoder_dispatch_type != DispatchType::Serial {
            return Err(ContractError::ConcurrentPassesUnsupported);
        }
        let mut initial_buffers = BTreeMap::<ViewId, &BufferView>::new();
        // The views this trace declares, in first-use order. Buffer and texture
        // bindings share the view-id namespace, so one identity can hold
        // several declarations (a later pass may rebind the view as a texture,
        // for instance) and each of them has to agree with an attachment that
        // references it.
        let mut declared = BTreeMap::<ViewId, Vec<DeclaredView>>::new();
        for (pass_index, pass) in self.passes.iter().enumerate() {
            // Render entries name attachments by identity, not `BufferView`s;
            // they are resolved after the compute declarations below are known,
            // while the index still counts every trace entry.
            let Some(pass) = pass.as_compute() else {
                continue;
            };
            for texture in &pass.textures {
                declared
                    .entry(texture.view_id)
                    .or_default()
                    .push(DeclaredView::Texture {
                        pass_index,
                        allocation_id: texture.allocation_id,
                        texture_type: texture.texture_type,
                        format: texture.format,
                        width: texture.width,
                        height: texture.height,
                        depth: texture.depth,
                        array_length: texture.array_length,
                        sample_count: texture.sample_count,
                        access: texture.access,
                    });
            }
            for view in &pass.buffers {
                declared
                    .entry(view.view_id)
                    .or_default()
                    .push(DeclaredView::Buffer {
                        pass_index,
                        allocation_id: view.allocation_id,
                        offset: view.offset,
                        length: view.length,
                        access: view.access,
                    });
                if let Some(initial) = initial_buffers.get(&view.view_id) {
                    if view.allocation_id != initial.allocation_id
                        || view.offset != initial.offset
                        || view.length != initial.length
                        || view.source != initial.source
                    {
                        return Err(ContractError::SerialBufferRebinding { pass_index });
                    }
                } else {
                    initial_buffers.insert(view.view_id, view);
                    if initial_buffers.len() > MAX_SERIAL_RESOURCES {
                        return Err(ContractError::SerialResourceLimit {
                            requested: initial_buffers.len(),
                            maximum: MAX_SERIAL_RESOURCES,
                        });
                    }
                }
            }
        }
        // Render attachments spend the same budget as the declared views. A
        // buffer-declared target is counted by the loop above; a texture-declared
        // one is counted here, which is the only place a render pass can add a
        // pool key without a compute binding.
        let mut pool = initial_buffers
            .keys()
            .copied()
            .collect::<BTreeSet<ViewId>>();
        for (pass_index, attachment) in self.attachments() {
            let Some(declarations) = declared.get(&attachment.view_id) else {
                return Err(ContractError::AttachmentViewUnknown {
                    pass_index,
                    view: attachment.view_id,
                    allocation: attachment.allocation_id,
                });
            };
            for declaration in declarations {
                if declaration.allocation_id() != attachment.allocation_id {
                    return Err(ContractError::AttachmentViewAllocationMismatch {
                        pass_index,
                        view: attachment.view_id,
                        declared: declaration.allocation_id(),
                        referenced: attachment.allocation_id,
                    });
                }
                declaration.admit_extent(pass_index, attachment)?;
            }
            // The attachment's own byte range is the one its declaration
            // describes, and the hazard is any compute write of the same
            // allocation that overlaps it — whichever view that write goes
            // through. Identity equality would miss a sibling view and would
            // have to be re-derived for every shape, so the comparison reuses
            // the range comparator (`research/docs/14` §3.2, review item M3,
            // 2026-09-14).
            let ranges = declarations
                .iter()
                .map(|declaration| declaration.byte_range())
                .collect::<Vec<_>>();
            for (compute_view, others) in &declared {
                for other in others {
                    if !other.is_writable() {
                        continue;
                    }
                    if ranges
                        .iter()
                        .any(|range| range.overlaps(&other.byte_range()))
                    {
                        return Err(ContractError::AttachmentComputeConflict {
                            pass_index,
                            view: attachment.view_id,
                            compute_view: *compute_view,
                            compute_pass: other.pass_index(),
                        });
                    }
                }
            }
            // The read half of the same question, answered by the contract's own
            // execution order: every compute pass runs before every render pass,
            // so a compute pass *after* this render pass that binds overlapping
            // bytes would see bytes the trace's order does not give it. Writes
            // are already refused above, whichever side of the render pass they
            // are on.
            for (compute_view, others) in &declared {
                for other in others {
                    if other.pass_index() <= pass_index {
                        continue;
                    }
                    if ranges
                        .iter()
                        .any(|range| range.overlaps(&other.byte_range()))
                    {
                        return Err(ContractError::RenderPassOrderUnsupported {
                            pass_index,
                            compute_pass: other.pass_index(),
                            view: attachment.view_id,
                            compute_view: *compute_view,
                        });
                    }
                }
            }
            if pool.insert(attachment.view_id) && pool.len() > MAX_SERIAL_RESOURCES {
                return Err(ContractError::SerialResourceLimit {
                    requested: pool.len(),
                    maximum: MAX_SERIAL_RESOURCES,
                });
            }
        }
        Ok(())
    }

    /// The stable upload/readback pool for serial execution, in first-use
    /// order. Each view's access is the union of its uses across all passes,
    /// so a view written only by a later pass still requires final readback.
    /// `metal_binding` retains the first-use label, which may duplicate another
    /// resource's label and must never be used as a pool key. Providers identify
    /// resources by view identity and use each pass's bindings when encoding.
    ///
    /// A render pass stores its attachment into the referenced view, which is a
    /// write of that pooled view for this submission: the readback path matches
    /// writebacks against the pool, so an attachment landing in a buffer view
    /// makes that view writable even when no compute pass writes it
    /// (`research/docs/23` §3.6). Texture-backed targets add no buffer entry —
    /// the sampled-texture pool stays read-only and its attachment byte landing
    /// belongs to the render execution step.
    pub fn serial_resources(&self) -> Result<Vec<BufferView>, ContractError> {
        self.validate_serial_buffer_reuse()?;
        let mut resources = Vec::<BufferView>::new();
        let mut positions = BTreeMap::<ViewId, usize>::new();
        for pass in self.compute_passes() {
            for view in &pass.buffers {
                if let Some(&position) = positions.get(&view.view_id) {
                    let resource = &mut resources[position];
                    resource.access = match (resource.access, view.access) {
                        (BufferAccess::Unused, access) | (access, BufferAccess::Unused) => access,
                        (left, right) if left == right => left,
                        _ => BufferAccess::ReadWrite,
                    };
                } else {
                    positions.insert(view.view_id, resources.len());
                    resources.push(view.clone());
                }
            }
        }
        for (_, attachment) in self.attachments() {
            let Some(&position) = positions.get(&attachment.view_id) else {
                continue;
            };
            let resource = &mut resources[position];
            resource.access = match resource.access {
                BufferAccess::Unused => BufferAccess::Write,
                BufferAccess::Read => BufferAccess::ReadWrite,
                BufferAccess::Write | BufferAccess::ReadWrite => resource.access,
            };
        }
        Ok(resources)
    }

    /// The stable texture pool for serial execution, in first-use order. Each
    /// view's access is the union of its uses across all passes; the first use
    /// keeps its binding label. Mirrors [`Self::serial_resources`] for the
    /// sampled-texture increment (`research/docs/16` §4.6).
    pub fn serial_texture_resources(&self) -> Result<Vec<TextureView>, ContractError> {
        let mut resources = Vec::<TextureView>::new();
        let mut positions = BTreeMap::<ViewId, usize>::new();
        for pass in self.compute_passes() {
            for texture in &pass.textures {
                if let Some(&position) = positions.get(&texture.view_id) {
                    let resource = &mut resources[position];
                    resource.access = match (resource.access, texture.access) {
                        (TextureAccess::Unused, access) | (access, TextureAccess::Unused) => access,
                        (left, right) if left == right => left,
                        _ => TextureAccess::Storage,
                    };
                } else {
                    positions.insert(texture.view_id, resources.len());
                    resources.push(texture.clone());
                }
            }
        }
        Ok(resources)
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
/// `SubmittedUnknown` and `Cancelled` do not establish GPU retirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionDisposition {
    NotSubmitted,
    Submitted { token: CompletionToken },
    CompletedVisible { token: CompletionToken },
    Cancelled { token: CompletionToken },
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
            | Self::Cancelled { token }
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
            | Self::Cancelled { token }
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
                | Self::Cancelled { .. }
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
/// by `(allocation_id, view_id)`. A single pass or serial passes sharing the
/// precollected logical view pool require one complete final writeback for each
/// view written by any pass when host readback completes. Intermediate per-pass
/// outputs are not returned.
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
        validate_writeback_list(&self.writebacks)
    }

    /// Validate the final result against the exact submitted trace. Serial
    /// passes must reuse the initial logical view pool; each view written in
    /// any pass has one full writeback reflecting all passes.
    pub fn validate_for_trace(&self, trace: &ComputeTrace) -> Result<(), ContractError> {
        self.validate()?;
        validate_writebacks_for_trace(self.completion, &self.writebacks, trace)
    }
}

/// Host-visible writebacks retrieved after a submitted token is observed as
/// `CompletedVisible`.
///
/// Providers that complete inside `submit` return writebacks in
/// [`ProviderSubmission`] and do not need this type. A provider that returns
/// `Submitted` reports the same completion token from `wait` and then returns
/// the final writebacks here. Writebacks follow the same ordering, coverage and
/// exact-trace rules as [`ProviderSubmission`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionReadback {
    pub completion: CompletionDisposition,
    pub writebacks: Vec<BufferWriteback>,
}

impl CompletionReadback {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.completion.validate()?;
        if !matches!(
            self.completion,
            CompletionDisposition::CompletedVisible { .. }
        ) {
            return Err(ContractError::InvalidReadbackCompletion(self.completion));
        }
        validate_writeback_list(&self.writebacks)
    }

    /// Validate the readback against the exact submitted trace. Serial passes
    /// must reuse the initial logical view pool; each view written in any pass
    /// has one full writeback reflecting all passes.
    pub fn validate_for_trace(&self, trace: &ComputeTrace) -> Result<(), ContractError> {
        self.validate()?;
        validate_writebacks_for_trace(self.completion, &self.writebacks, trace)
    }
}

fn validate_writeback_list(writebacks: &[BufferWriteback]) -> Result<(), ContractError> {
    let mut views = BTreeMap::new();
    for writeback in writebacks {
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
    if writebacks.windows(2).any(|pair| {
        (pair[0].allocation_id, pair[0].view_id) > (pair[1].allocation_id, pair[1].view_id)
    }) {
        return Err(ContractError::NonCanonicalWritebackOrder);
    }
    Ok(())
}

fn validate_writebacks_for_trace(
    completion: CompletionDisposition,
    writebacks: &[BufferWriteback],
    trace: &ComputeTrace,
) -> Result<(), ContractError> {
    let resources = trace.serial_resources()?;
    let token = completion
        .token()
        .ok_or(ContractError::InvalidSubmissionCompletion(completion))?;
    if token.device_epoch != trace.device_epoch {
        return Err(ContractError::CompletionEpochMismatch {
            expected: trace.device_epoch,
            actual: token.device_epoch,
        });
    }
    if !writebacks.is_empty() && trace.completion_policy != CompletionPolicy::HostReadback {
        return Err(ContractError::WritebackPolicyMismatch(
            trace.completion_policy,
        ));
    }
    for writeback in writebacks {
        let view = resources
            .iter()
            .find(|view| {
                view.allocation_id == writeback.allocation_id && view.view_id == writeback.view_id
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
        && matches!(completion, CompletionDisposition::CompletedVisible { .. })
    {
        for view in resources.iter().filter(|view| view.access.is_writable()) {
            if !writebacks.iter().any(|writeback| {
                writeback.allocation_id == view.allocation_id && writeback.view_id == view.view_id
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

/// Backend-neutral execution boundary for a canonical Metal trace.
///
/// Implementations own compilation, provider handles, queue/encoder state and
/// completion retirement. They receive only an admitted value trace and must
/// keep provider-specific objects behind this trait. `wait` reports timeout as
/// a non-terminal disposition; provider failures use `ProviderError`.
pub trait ComputeProvider: Send + Sync {
    fn capabilities(&self) -> ProviderCapabilities;

    /// Install a host-side scheduling tier per device queue and report the
    /// table the provider actually installed.
    ///
    /// The argument is a marking, not a device table: an owner does not
    /// necessarily know how many queues the provider created (the command
    /// channel carries capabilities, not the queue count), so the provider
    /// expands it with [`queue_priorities_for_device`] — extra entries are
    /// ignored and missing ones stay [`QueuePriority::Default`]. The returned
    /// table therefore has exactly one entry per device queue, which is what an
    /// owner compares against to confirm the marking it sent.
    ///
    /// A tier only steers which device queue a submission is enqueued on. It
    /// never changes dependency order, lease reservation semantics or writeback
    /// contents, and a provider that received no marking keeps every queue at
    /// [`QueuePriority::Default`] — the pre-priority scheduling.
    /// A marking is scheduling state rather than device work: it neither
    /// admits a submission nor revives a provider that stopped admitting work.
    ///
    /// Providers without queue scheduling keep this refusal: a marking that
    /// cannot reach a scheduler is a capability question, not a protocol error.
    fn set_queue_priorities(
        &self,
        tiers: &[QueuePriority],
    ) -> Result<Vec<QueuePriority>, ProviderError> {
        let _ = tiers;
        Err(ProviderError::new(
            ProviderPhase::Resolve,
            ProviderErrorClass::Capability,
            "queue_priorities_unsupported",
        )
        .expect("non-empty provider error slug"))
    }

    /// Current provider health. Providers that can lose a device or exhaust a
    /// bounded abandonment budget must override this; callers must recreate a
    /// provider whose health is not [`ProviderHealth::Usable`].
    fn health(&self) -> ProviderHealth {
        ProviderHealth::Usable
    }

    fn submit(&self, trace: ValidatedComputeTrace) -> Result<ProviderSubmission, ProviderError>;

    fn wait(
        &self,
        token: CompletionToken,
        timeout: Duration,
    ) -> Result<CompletionDisposition, ProviderError>;

    /// Request cancellation of a deferred submission and release the
    /// provider's observation slot. Cancellation never un-submits device work:
    /// resources are reclaimed when the device retires the submission, or by
    /// the backend's completion handler.
    ///
    /// Returns the terminal disposition observed at the time of the request.
    /// `Cancelled` means the slot was still running and is now released from
    /// observation; a terminal disposition means cancellation lost the race
    /// and the original outcome is reported unchanged. After a successful
    /// cancellation, `wait` keeps reporting `Cancelled` until
    /// [`PipelineProvider::release_completion`] forgets the slot, and
    /// `readback` refuses because no result was observed.
    ///
    /// Providers without deferred work may leave the default refusal.
    fn cancel(&self, token: CompletionToken) -> Result<CompletionDisposition, ProviderError> {
        Err(ProviderError::new(
            ProviderPhase::Wait,
            ProviderErrorClass::Capability,
            "completion_cancel_unsupported",
        )
        .expect("non-empty provider error slug")
        .with_completion(CompletionDisposition::SubmittedUnknown { token: Some(token) }))
    }

    /// Return host-visible writebacks for a token already observed as
    /// `CompletedVisible`. The returned completion must repeat that token.
    ///
    /// Providers that finish inside `submit` return writebacks in
    /// [`ProviderSubmission`] and may leave the default refusal. A provider
    /// that returns `Submitted` must override this method: `wait` reports
    /// readiness, and `readback` supplies the final contents. The object API
    /// validates every readback against the exact submitted trace before any
    /// host byte changes.
    fn readback(&self, token: CompletionToken) -> Result<CompletionReadback, ProviderError> {
        Err(ProviderError::new(
            ProviderPhase::Wait,
            ProviderErrorClass::Capability,
            "completion_readback_unsupported",
        )
        .expect("non-empty provider error slug")
        .with_completion(CompletionDisposition::SubmittedUnknown { token: Some(token) }))
    }
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
    /// Provider policy, not a physical-device limit. Snapshot adapters may
    /// expose one pass; command-buffer providers can permit serial reuse.
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
    /// Whether this snapshot can execute the render pass shape of
    /// `research/docs/23`. Defaults to `false` everywhere: neither the Vulkan
    /// nor the native provider executes graphics work today, so a
    /// render-bearing trace is refused during admission instead of being
    /// silently downgraded.
    pub supports_render_passes: bool,
    /// Colour attachment slots the render track may address. `0` means the
    /// snapshot cannot render at all; the first render increment caps this at
    /// [`MAX_COLOR_ATTACHMENTS`].
    pub max_color_attachments: u32,
    /// Largest attachment extent `[width, height]` this snapshot admits.
    /// `[0, 0]` means no attachment is admissible.
    pub max_attachment_dimension: [u64; 2],
    /// Colour attachment formats this snapshot admits. Empty means none.
    /// Compared against [`AttachmentFormat`], which is the render contract's
    /// own format family; its wire codes are the `MCC1` texture-format codes,
    /// so no second mapping is needed (`docs/23` §3.1).
    pub supported_color_formats: Vec<AttachmentFormat>,
    /// Whether this snapshot can execute the present action of
    /// `research/docs/24`. Defaults to `false` everywhere: Step 2 publishes the
    /// contract and the refusals, while the Vulkan "readable swapchain
    /// equivalent" is `docs/24` §6 Step 3, so a present-bearing trace is
    /// refused during admission instead of being silently downgraded to an
    /// offscreen render (`docs/24` §4.2).
    ///
    /// The three fields below are the increments this bit narrows, declared in
    /// the same "the field exists but the value is refused" style
    /// [`PresentTarget::image_count`] uses. They stay at their defaults
    /// (`0`/empty/`0`) for a snapshot that cannot present, so a caller reading
    /// them without checking this bit cannot read a limit as an admission.
    pub supports_presentation: bool,
    /// Present targets this snapshot admits in one trace. `0` means the
    /// snapshot cannot present at all; the first increment admits
    /// [`MAX_PRESENT_TARGETS`].
    pub max_present_targets: u32,
    /// Present modes this snapshot admits. Empty means none; the first
    /// increment admits [`PresentMode::ADMITTED`] (`Fifo` only, `docs/24`
    /// §3.1). Compared by value rather than by wire code so the contract's own
    /// enum is the single vocabulary (`docs/24` §4.2).
    pub supported_present_modes: Vec<PresentMode>,
    /// Images behind one present target this snapshot admits. `0` means none;
    /// the first increment admits [`MAX_PRESENT_IMAGE_COUNT`], because
    /// multi-buffering needs the in-flight state machine `docs/24` §3.4
    /// schedules after this increment.
    pub max_present_image_count: u32,
}

impl ProviderCapabilities {
    /// Whether any extended bit differs from its default. `MCC1` uses this to
    /// keep a compute-only, non-presenting provider's capability frame at its
    /// exact legacy bytes and to carry the render and present bits only when
    /// they exist (`docs/24` §4.2).
    ///
    /// The present bits are part of the same question on purpose: a snapshot
    /// that declared presentation without declaring render would otherwise keep
    /// sending the legacy payload, and its present bits would be lost on the
    /// wire — the exact "declared a bit that travels as the old bytes" failure
    /// `docs/24` §4.2 calls out.
    pub fn declares_render_support(&self) -> bool {
        self.supports_render_passes
            || self.max_color_attachments != 0
            || self.max_attachment_dimension != [0, 0]
            || !self.supported_color_formats.is_empty()
            || self.declares_presentation_support()
    }

    /// Whether any present bit differs from its default.
    ///
    /// A snapshot with all four bits at their defaults is treated as unable to
    /// present: [`ProviderCapabilities::admit`] refuses any present-bearing
    /// trace it sees, and `MCC1` never has to carry the bits.
    pub fn declares_presentation_support(&self) -> bool {
        self.supports_presentation
            || self.max_present_targets != 0
            || !self.supported_present_modes.is_empty()
            || self.max_present_image_count != 0
    }

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

        // Render admission is the first capability gate and precedes every
        // reservation the caller performs after a successful `admit`: a
        // provider that cannot render refuses the whole trace here, and a
        // compute-only trace never enters the walk (`docs/23` §4.2).
        self.admit_render_passes(trace)?;

        // Present admission is the second gate and sits just as early
        // (`research/docs/24` §4.2): Step 2 publishes the contract and the
        // refusal, Step 3 owns execution, so a snapshot that cannot present
        // refuses a present-bearing trace before any resource action instead of
        // running its render half and dropping the present.
        self.admit_present_actions(trace)?;

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
        match trace.completion_policy {
            CompletionPolicy::HostReadback if !self.host_readback => {
                return Err(capability_error("host_readback_unsupported"));
            }
            CompletionPolicy::SubmitOnly if !self.submit_only => {
                return Err(capability_error("submit_only_unsupported"));
            }
            _ => {}
        }

        let mut allocations = BTreeMap::<AllocationId, BTreeMap<ViewId, BufferRange>>::new();
        for pass in trace.compute_passes() {
            let contract = &trace
                .pipeline(pass.pipeline)
                .map_err(contract_error_refusal)?
                .contract;
            match contract.dispatch_kind {
                DispatchKind::ThreadsExact if !self.supports_threads_exact => {
                    return Err(capability_error("dispatch_kind_unsupported"));
                }
                DispatchKind::Threadgroups if !self.supports_threadgroups => {
                    return Err(capability_error("dispatch_kind_unsupported"));
                }
                _ => {}
            }
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

            let push_end = contract
                .push_constant_offset
                .checked_add(contract.push_constant_bytes)
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
                let reflected = contract
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
                let range = BufferRange::new(buffer.allocation_id, buffer.offset, buffer.length);
                let views = allocations.entry(buffer.allocation_id).or_default();
                if views.contains_key(&buffer.view_id) {
                    // The same logical view may be bound by several passes; its
                    // identity and range were settled when it was first seen.
                    continue;
                }
                if !views.is_empty() {
                    match self.alias_mode {
                        AliasMode::DistinctViews => {
                            // Ranged aliasing: several views of one allocation
                            // are admitted only while their byte ranges stay
                            // disjoint. Overlap keeps the original refusal so a
                            // malformed trace cannot become an aliasing hazard.
                            if views.values().any(|other| other.overlaps(&range)) {
                                return Err(capability_error("buffer_alias_unsupported")
                                    .with_field(
                                        "binding",
                                        FieldValue::Unsigned(buffer.metal_binding as u64),
                                    )
                                    .with_field(
                                        "allocation",
                                        FieldValue::Unsigned(buffer.allocation_id.get()),
                                    ));
                            }
                        }
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
                views.insert(buffer.view_id, range);
            }
        }
        // Capability admission intentionally precedes backing/lease admission:
        // a malformed or unsupported dispatch must not be masked by a stale
        // resource handle, and the order matches the provider contract gates.
        trace
            .validate_serial_buffer_reuse()
            .map_err(contract_error_refusal)?;
        resources
            .validate_trace(trace)
            .map_err(contract_error_refusal)?;
        Ok(())
    }

    /// Render-track admission, run before the compute walk and before any
    /// resource reservation. A provider whose render bits stay at their
    /// defaults refuses every render-bearing trace with
    /// `render_passes_unsupported`; a compute-only trace never enters the loop
    /// body. When a provider does declare render support, the same walk
    /// enforces the attachment count, dimension and format bits and then the
    /// agreement between the pass and the pipeline it names.
    ///
    /// The order is deliberate (review items I3/I4, 2026-09-14): the
    /// capability bits a snapshot can answer on its own come first, so a
    /// provider that cannot render at all never reports a contract detail about
    /// work it would not execute. Only then is the named table entry read, and
    /// only then is its render half compared with the pass. That is the first
    /// gate that can check "this attachment format is the pipeline's own"
    /// without a provider registry.
    fn admit_render_passes(&self, trace: &ComputeTrace) -> Result<(), ProviderError> {
        let render_pass_count = trace.render_passes().count();
        if render_pass_count == 0 {
            return Ok(());
        }
        if !self.supports_render_passes {
            return Err(capability_error("render_passes_unsupported")
                .with_field("passes", FieldValue::Unsigned(render_pass_count as u64)));
        }
        for (pass_index, entry) in trace.passes.iter().enumerate() {
            let Some(pass) = entry.as_render() else {
                continue;
            };
            if pass.color_attachments.len() > self.max_color_attachments as usize {
                return Err(capability_error("color_attachment_limit")
                    .with_field(
                        "requested",
                        FieldValue::Unsigned(pass.color_attachments.len() as u64),
                    )
                    .with_field(
                        "maximum",
                        FieldValue::Unsigned(self.max_color_attachments as u64),
                    ));
            }
            for attachment in &pass.color_attachments {
                if !self.supported_color_formats.contains(&attachment.format) {
                    return Err(
                        capability_error("attachment_format_unsupported").with_field(
                            "format",
                            FieldValue::Unsigned(u64::from(attachment.format.code())),
                        ),
                    );
                }
                if attachment.width > self.max_attachment_dimension[0]
                    || attachment.height > self.max_attachment_dimension[1]
                {
                    return Err(capability_error("attachment_dimension_limit")
                        .with_field("width", FieldValue::Unsigned(attachment.width))
                        .with_field("height", FieldValue::Unsigned(attachment.height))
                        .with_field(
                            "maximum_width",
                            FieldValue::Unsigned(self.max_attachment_dimension[0]),
                        )
                        .with_field(
                            "maximum_height",
                            FieldValue::Unsigned(self.max_attachment_dimension[1]),
                        ));
                }
            }
            // The pass's own shape rules are already `trace.validate()`'s job,
            // which ran before this gate. What is added here is the agreement
            // only the table entry can answer: the pass names a
            // `PipelineId`, so an entry without a render half leaves the
            // comparison without a format, and an entry whose render half
            // disagrees with the attachment is refused as a caller-fixable
            // trace shape rather than left to a provider registry.
            let pipeline = trace
                .pipeline(pass.pipeline)
                .map_err(contract_error_refusal)?;
            let render = pipeline.render.as_ref().ok_or_else(|| {
                contract_error_refusal(ContractError::MissingRenderPipelineContract {
                    pass_index,
                    pipeline: pass.pipeline,
                })
            })?;
            render
                .validate_against(pass)
                .map_err(contract_error_refusal)?;
        }
        Ok(())
    }

    /// Present admission, the second capability gate (`research/docs/24` §4.2).
    ///
    /// The order repeats the render gate's deliberate choice: the bits a
    /// snapshot can answer on its own come first, so a provider whose present
    /// bits stay at their defaults refuses the whole trace with
    /// `present_targets_unsupported` and never reports a per-target detail
    /// about work it would not execute. A compute-only or offscreen-only trace
    /// has no present action, so it never enters the walk and keeps the
    /// pre-present admission exactly (`docs/24` §4.3).
    ///
    /// Only a snapshot that declares presentation reaches the target-count,
    /// mode and image-count bits. Their refusals are therefore statements about
    /// this snapshot's declared limits; the trace's own shape was already ruled
    /// on by `trace.validate()`, which runs Step 1's present rules through
    /// [`RenderPassDescriptor::validate`] before this gate.
    fn admit_present_actions(&self, trace: &ComputeTrace) -> Result<(), ProviderError> {
        let targets = trace.present_actions().count();
        if targets == 0 {
            return Ok(());
        }
        if !self.supports_presentation {
            return Err(capability_error("present_targets_unsupported")
                .with_field("targets", FieldValue::Unsigned(targets as u64)));
        }
        if targets > self.max_present_targets as usize {
            return Err(capability_error("present_target_limit")
                .with_field("requested", FieldValue::Unsigned(targets as u64))
                .with_field(
                    "maximum",
                    FieldValue::Unsigned(self.max_present_targets as u64),
                ));
        }
        for (pass_index, present) in trace.present_actions() {
            if !self.supported_present_modes.contains(&present.mode) {
                return Err(capability_error("present_mode_unsupported")
                    .with_field("mode", FieldValue::Unsigned(u64::from(present.mode.code())))
                    .with_field("pass", FieldValue::Unsigned(pass_index as u64)));
            }
            if present.target.image_count > self.max_present_image_count {
                return Err(capability_error("present_image_count_limit")
                    .with_field(
                        "requested",
                        FieldValue::Unsigned(u64::from(present.target.image_count)),
                    )
                    .with_field(
                        "maximum",
                        FieldValue::Unsigned(u64::from(self.max_present_image_count)),
                    ));
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
    use ContractError as E;

    // Exhaustive on purpose: adding a `ContractError` variant must force a
    // deliberate class/slug decision instead of silently becoming `Args`.
    let (class, slug) = match &error {
        // Resource lifecycle: identity, epoch, range and alias violations.
        E::UnknownAllocation(_)
        | E::DuplicateAllocation(_)
        | E::AllocationEpochMismatch { .. }
        | E::AllocationRangeOutOfBounds { .. }
        | E::DuplicateLease(_)
        | E::UnknownLease(_)
        | E::LeaseEpochMismatch { .. }
        | E::LeaseRangeOutOfBounds { .. }
        | E::LeaseMismatch { .. }
        | E::OverlappingWritableViews { .. }
        | E::ViewIdentityMismatch(_)
        | E::PipelineEpochMismatch { .. }
        | E::SnapshotAliasUnsupported(_) => {
            (ProviderErrorClass::Resource, "resource_contract_invalid")
        }
        E::CompletionEpochMismatch { .. } => {
            (ProviderErrorClass::Resource, "completion_epoch_mismatch")
        }
        // Caller-supplied source and pointer shapes.
        E::SourceLengthMismatch { .. } => {
            (ProviderErrorClass::Args, "buffer_source_length_mismatch")
        }
        E::TextureSampleCountMismatch { .. } | E::TextureArrayLengthMismatch { .. } => {
            (ProviderErrorClass::Args, "texture_shape_mismatch")
        }
        E::TextureBindingUnsupported(_) => (
            ProviderErrorClass::Capability,
            "texture_binding_unsupported",
        ),
        // Render contract, Step 1: the format, load/store, viewport-origin and
        // draw-shape rules are first-increment narrowings, so a well-formed
        // request for something wider is a capability refusal. A viewport that
        // disagrees with its own attachment is structural instead.
        E::UnsupportedAttachmentFormat(_) => (
            ProviderErrorClass::Capability,
            "attachment_format_unsupported",
        ),
        E::UnsupportedAttachmentLoadOp(_) => (
            ProviderErrorClass::Capability,
            "attachment_load_op_unsupported",
        ),
        E::UnsupportedAttachmentStoreOp(_) => (
            ProviderErrorClass::Capability,
            "attachment_store_op_unsupported",
        ),
        E::AttachmentLimitExceeded { .. } => (
            ProviderErrorClass::Capability,
            "attachment_count_unsupported",
        ),
        E::ViewportOriginUnsupported { .. } => (
            ProviderErrorClass::Capability,
            "viewport_origin_unsupported",
        ),
        E::DrawVertexCountMismatch { .. } => {
            (ProviderErrorClass::Capability, "draw_shape_unsupported")
        }
        E::ViewportExtentMismatch { .. } => (ProviderErrorClass::Args, "trace_contract_invalid"),
        // Presentation contract, Step 1. The three first-increment narrowings
        // are capability refusals for the same reason the render track's are:
        // the request is well formed and simply wider than this increment. The
        // sentinel length and the two inheritance disagreements are caller-
        // fixable trace shape instead, because they name two of the caller's own
        // fields that disagree rather than a feature the device lacks.
        E::PresentModeUnsupported(_) => {
            (ProviderErrorClass::Capability, "present_mode_unsupported")
        }
        E::PresentAcquirePolicyUnsupported(_) => (
            ProviderErrorClass::Capability,
            "present_acquire_policy_unsupported",
        ),
        E::PresentImageCountUnsupported { .. } => (
            ProviderErrorClass::Capability,
            "present_image_count_unsupported",
        ),
        E::PresentSentinelLengthMismatch { .. } => (
            ProviderErrorClass::Args,
            "present_sentinel_length_mismatch",
        ),
        E::PresentSourceUnknown { .. } => (ProviderErrorClass::Args, "present_source_unknown"),
        E::PresentTargetViewMismatch { .. } => (
            ProviderErrorClass::Args,
            "present_target_view_mismatch",
        ),
        E::PresentTargetAllocationMismatch { .. } => (
            ProviderErrorClass::Args,
            "present_target_allocation_mismatch",
        ),
        E::PresentFormatMismatch { .. } => (ProviderErrorClass::Args, "present_format_mismatch"),
        E::PresentExtentMismatch { .. } => (ProviderErrorClass::Args, "present_extent_mismatch"),
        // Render contract, Step 3c: an attachment that cannot be resolved in
        // the trace's own resource table names a resource the submission does
        // not have, which is the same class as an unknown allocation. A
        // resolvable attachment whose restated extent disagrees with its
        // declaration, or that races a compute writer, is a repairable trace
        // shape instead.
        E::AttachmentViewUnknown { .. } | E::AttachmentViewAllocationMismatch { .. } => (
            ProviderErrorClass::Resource,
            "attachment_allocation_unknown",
        ),
        // A byte-count disagreement and a texel-shape disagreement are the same
        // caller-fixable resource shape, so they keep one slug; the detail is
        // where the shape is spelled out (review item M4, 2026-09-14).
        E::AttachmentExtentMismatch { .. } | E::AttachmentTextureShapeMismatch { .. } => {
            (ProviderErrorClass::Args, "attachment_extent_mismatch")
        }
        E::AttachmentComputeConflict { .. } => {
            (ProviderErrorClass::Args, "attachment_resource_conflict")
        }
        // The trace is well formed; the shape is one this increment's fixed
        // compute-then-render execution order cannot honour, which is the same
        // class the provider walk reported before core owned the rule
        // (review item I4, 2026-09-14).
        E::RenderPassOrderUnsupported { .. } => {
            (ProviderErrorClass::Capability, "render_pass_order_unsupported")
        }
        E::LeaseSourceLengthMismatch { .. } => {
            (ProviderErrorClass::Args, "lease_source_length_mismatch")
        }
        E::NullHostPointer(_) => (ProviderErrorClass::Args, "borrowed_host_pointer_null"),
        E::GuestWindowStillActive(_) => (ProviderErrorClass::Resource, "guest_window_still_active"),
        E::InvalidHostRegionPageSize(_)
        | E::UnalignedHostRegion { .. }
        | E::HostRegionWindowOutOfBounds { .. } => {
            (ProviderErrorClass::Args, "host_region_invalid")
        }
        // Well-formed input asking for a feature outside the B0 subset.
        E::UnsupportedAttributeStride => (
            ProviderErrorClass::Capability,
            "buffer_attribute_stride_unsupported",
        ),
        E::UnsupportedSchemaVersion(_) => {
            (ProviderErrorClass::Capability, "trace_schema_unsupported")
        }
        E::UnsupportedDispatchType(_) => {
            (ProviderErrorClass::Capability, "dispatch_type_unsupported")
        }
        E::SnapshotDispatchUnsupported(_) => (
            ProviderErrorClass::Capability,
            "snapshot_dispatch_unsupported",
        ),
        // Completion/writeback protocol returned by the caller.
        E::DuplicateWriteback { .. }
        | E::InvalidSubmissionCompletion(_)
        | E::InvalidReadbackCompletion(_)
        | E::WritebackBeforeCompletion
        | E::WritebackPolicyMismatch(_)
        | E::NonCanonicalWritebackOrder
        | E::UnknownWriteback { .. }
        | E::ReadOnlyWriteback(_)
        | E::WritebackRangeOutOfBounds { .. }
        | E::IncompleteWriteback(_)
        | E::MissingWriteback { .. } => (ProviderErrorClass::Args, "writeback_contract_invalid"),
        // Provider-side completion publication invariants: a provider bug, not
        // caller input. The outbox returns `ContractError` directly, so they
        // share this converter for a stable class.
        E::CompletionPublishAfterTerminal(_)
        | E::CompletionPublishAfterDeviceLost(_)
        | E::CompletionHealthRegression { .. } => {
            (ProviderErrorClass::Internal, "completion_protocol_invalid")
        }
        // Structural trace errors: the caller can fix the trace.
        E::EmptyField(_)
        | E::InvalidIdentity(_)
        | E::ZeroLength(_)
        | E::ZeroDimension { .. }
        | E::ArithmeticOverflow(_)
        | E::MisalignedPushConstantOffset(_)
        | E::ConcurrentPassesUnsupported
        | E::SerialBufferRebinding { .. }
        | E::SerialResourceLimit { .. }
        | E::DuplicateBinding(_)
        | E::NonCanonicalBindingOrder(_)
        | E::MissingBinding(_)
        | E::UnknownBinding(_)
        | E::AccessMismatch { .. }
        | E::LocalSizeMismatch { .. }
        | E::GridMismatch { .. }
        | E::FixedGridRequiresExactDispatch
        | E::DuplicateView(_)
        | E::EmptyTrace
        | E::EmptyPipelineTable
        | E::UnknownPipeline(_)
        | E::DuplicatePipeline(_)
        | E::UnusedPipeline(_)
        | E::MissingSnapshotIdentity(_)
        | E::UnknownSnapshotIdentity(_)
        | E::EmptyAttachmentList
        // Render pipeline, Step 3a: the entry shape and the
        // pipeline/attachment agreement are structural, like the pass rules
        // above. The colour format is not listed here; it reuses
        // `UnsupportedAttachmentFormat`, which stays a capability refusal.
        | E::EmptyRenderPipelineEntry(_)
        | E::DuplicateRenderPipelineEntry(_)
        | E::RenderPipelineFormatMismatch { .. }
        | E::MissingRenderPipelineContract { .. }
        | E::DispatchKindMismatch { .. } => (ProviderErrorClass::Args, "trace_contract_invalid"),
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

/// One half-open byte range inside one allocation: `[offset, offset + length)`.
///
/// Ranges are the pure-value unit of provider-side hazard tracking proposed by
/// `research/docs/14`; nothing is wired to them yet. `length == 0` is an empty
/// range that conflicts with nothing. A range whose end overflows is treated as
/// conflicting so hazard admission fails closed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferRange {
    pub allocation_id: AllocationId,
    pub offset: u64,
    pub length: u64,
}

impl BufferRange {
    pub const fn new(allocation_id: AllocationId, offset: u64, length: u64) -> Self {
        Self {
            allocation_id,
            offset,
            length,
        }
    }

    /// Exclusive end offset; refuses overflow like [`LeaseReservation::end`].
    pub fn end(&self) -> Result<u64, ContractError> {
        self.offset
            .checked_add(self.length)
            .ok_or(ContractError::ArithmeticOverflow("buffer range"))
    }

    /// True when both ranges describe overlapping bytes of the same allocation.
    /// Empty ranges never overlap; an overflowing range conflicts (fail closed).
    pub fn overlaps(&self, other: &Self) -> bool {
        if self.allocation_id != other.allocation_id || self.length == 0 || other.length == 0 {
            return false;
        }
        match (self.end(), other.end()) {
            (Ok(left), Ok(right)) => self.offset < right && other.offset < left,
            _ => true,
        }
    }
}

/// Reads and writes of one submission expressed as allocation ranges. Hazard
/// admission compares two sets between in-flight submissions; the rule follows
/// `research/docs/14` §3.2: any overlap involving at least one write conflicts.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RangeSet {
    pub reads: Vec<BufferRange>,
    pub writes: Vec<BufferRange>,
}

impl RangeSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_read(mut self, range: BufferRange) -> Self {
        self.reads.push(range);
        self
    }

    pub fn with_write(mut self, range: BufferRange) -> Self {
        self.writes.push(range);
        self
    }

    /// True when any write of `self` overlaps a write or read of `other`, or
    /// any read of `self` overlaps a write of `other`.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        self.writes.iter().any(|write| {
            other.writes.iter().any(|other| write.overlaps(other))
                || other.reads.iter().any(|other| write.overlaps(other))
        }) || self
            .reads
            .iter()
            .any(|read| other.writes.iter().any(|write| read.overlaps(write)))
    }
}

/// Sort ranges into the canonical order from `research/docs/14` §3.4:
/// allocation, then offset, then length. Exact duplicates keep their relative
/// order.
pub fn sort_canonical(ranges: &mut [BufferRange]) {
    ranges.sort_by_key(|range| (range.allocation_id, range.offset, range.length));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageMode {
    OwnedBytes,
    StagedLease,
    BorrowedNoCopy,
}

/// Priority tier of one device queue as seen by the host scheduling policy.
///
/// This is a host-side scheduling hint layered on top of the existing
/// least-loaded queue choice, not a `VkDeviceQueueCreateInfo::pQueuePriorities`
/// value: Vulkan fixes queue priorities when the queue is created, so a provider
/// that wants to change them mid-flight has to express priority in its own
/// scheduler (`research/docs/21-队列优先级与公平性设计.md`).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QueuePriority {
    /// Background work. It still must make progress: see
    /// [`QueueSchedulingPolicy`].
    Low,
    /// The tier of every queue that was created without an explicit hint, and
    /// therefore the tier of every queue a provider creates today.
    #[default]
    Default,
    /// Latency-sensitive work.
    High,
}

impl QueuePriority {
    /// Tier rank used by the scheduler: `Low < Default < High`.
    pub const fn rank(self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Default => 1,
            Self::High => 2,
        }
    }

    /// Inverse of [`Self::rank`] for tier tables and wire carriers.
    ///
    /// A carrier that reads a rank this build does not know must refuse the
    /// value instead of falling back to a tier, otherwise a rank written by a
    /// newer peer would silently become `Default` and change the scheduling
    /// of a queue nobody asked to mark.
    pub const fn from_rank(rank: u8) -> Option<Self> {
        match rank {
            0 => Some(Self::Low),
            1 => Some(Self::Default),
            2 => Some(Self::High),
            _ => None,
        }
    }
}

/// Weighted, starvation-free scheduling policy for one provider's device
/// queues.
///
/// The policy is a rotation window of `low_weight + medium_weight +
/// high_weight` scheduling slots, walked highest tier first. Two properties
/// follow from the window alone, without any per-queue accounting:
///
/// - weighted shares: a tier that has a queue available claims its own slots,
///   so `Default` receives `medium_weight / window` of the slots in the long
///   run and `Low` receives `low_weight / window`;
/// - no starvation: every weight is clamped to at least one, so every tier
///   owns at least one slot per window, and the longest run of consecutive
///   `High` selections is exactly [`Self::high_priority_streak_limit`] — the
///   `high_weight` slots at the start of the window.
///
/// The default weights are `High` 4, `Default` 2, `Low` 1, so the streak limit
/// is 4 and a low-priority queue is selected at least once every 7 selections.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueSchedulingPolicy {
    low_weight: u32,
    medium_weight: u32,
    high_weight: u32,
}

impl QueueSchedulingPolicy {
    /// Weight of [`QueuePriority::Low`] in the default policy.
    pub const DEFAULT_LOW_WEIGHT: u32 = 1;
    /// Weight of [`QueuePriority::Default`] in the default policy.
    pub const DEFAULT_MEDIUM_WEIGHT: u32 = 2;
    /// Weight of [`QueuePriority::High`] in the default policy.
    pub const DEFAULT_HIGH_WEIGHT: u32 = 4;

    /// Build a weighted policy.
    ///
    /// A zero weight would starve its own tier, which contradicts the contract
    /// this type exists to express, so every weight is clamped up to one
    /// instead of being refused: the no-starvation invariant then holds for
    /// every value the caller can pass.
    pub const fn new(low_weight: u32, medium_weight: u32, high_weight: u32) -> Self {
        Self {
            low_weight: if low_weight == 0 { 1 } else { low_weight },
            medium_weight: if medium_weight == 0 { 1 } else { medium_weight },
            high_weight: if high_weight == 0 { 1 } else { high_weight },
        }
    }

    /// Weight of [`QueuePriority::Low`].
    pub const fn low_weight(self) -> u32 {
        self.low_weight
    }

    /// Weight of [`QueuePriority::Default`].
    pub const fn medium_weight(self) -> u32 {
        self.medium_weight
    }

    /// Weight of [`QueuePriority::High`].
    pub const fn high_weight(self) -> u32 {
        self.high_weight
    }

    /// Weight of `priority` in this policy.
    pub const fn weight(self, priority: QueuePriority) -> u32 {
        match priority {
            QueuePriority::Low => self.low_weight,
            QueuePriority::Default => self.medium_weight,
            QueuePriority::High => self.high_weight,
        }
    }

    /// Length of one rotation window in scheduling slots.
    pub const fn window(self) -> u64 {
        self.low_weight as u64 + self.medium_weight as u64 + self.high_weight as u64
    }

    /// Longest run of consecutive `High` selections the window allows before a
    /// lower tier is due.
    ///
    /// It is exactly the `High` weight, because the window places that tier's
    /// slots first. A caller that only needs the bound does not have to reason
    /// about the window at all.
    pub const fn high_priority_streak_limit(self) -> u32 {
        self.high_weight
    }

    /// Tier that slot `slot` of the rotation nominates.
    pub const fn nominated_priority(self, slot: u64) -> QueuePriority {
        let slot = slot % self.window();
        if slot < self.high_weight as u64 {
            QueuePriority::High
        } else if slot < (self.high_weight + self.medium_weight) as u64 {
            QueuePriority::Default
        } else {
            QueuePriority::Low
        }
    }
}

impl Default for QueueSchedulingPolicy {
    fn default() -> Self {
        Self::new(
            Self::DEFAULT_LOW_WEIGHT,
            Self::DEFAULT_MEDIUM_WEIGHT,
            Self::DEFAULT_HIGH_WEIGHT,
        )
    }
}

/// Tier distance used to resolve a nomination against the queues that are
/// actually available: the nominated tier wins, then the highest tier below it
/// (so a missing tier never hands its slots back to a greedier one), then the
/// closest tier above it.
fn tier_distance(tier: QueuePriority, nominated: QueuePriority) -> (u8, u8) {
    if tier == nominated {
        (0, 0)
    } else if tier.rank() < nominated.rank() {
        (1, nominated.rank() - tier.rank())
    } else {
        (2, tier.rank() - nominated.rank())
    }
}

/// Expand an owner queue-priority marking into one tier per device queue.
///
/// The marking is a hint, not a device table: an owner does not necessarily
/// know how many queues a provider created, and a marking that crossed a
/// process boundary carries the owner's view rather than the provider's. Extra
/// entries are ignored and missing entries stay [`QueuePriority::Default`],
/// which is exactly how [`select_queue_with_priority`] reads a short table. An
/// empty marking therefore produces the all-`Default` table a provider has
/// without any marking at all, so the default path keeps the pre-priority
/// scheduling byte for byte.
///
/// This is the one place a marking ages into a device table, so a
/// process-local marking and a marking that arrived over the command channel
/// cannot drift apart.
pub fn queue_priorities_for_device(
    queue_count: usize,
    marking: &[QueuePriority],
) -> Vec<QueuePriority> {
    let mut tiers = vec![QueuePriority::Default; queue_count];
    for (slot, tier) in tiers.iter_mut().zip(marking) {
        *slot = *tier;
    }
    tiers
}

/// Pick the device queue that should receive the next submission.
///
/// `in_flight[i]` is the number of submissions that have not retired on queue
/// `i`, so step one is the existing least-loaded rule: an idle queue always
/// beats a busy one, whatever their tiers. Among the least-loaded queues the
/// policy window decides — the nominated tier first, then a rotating choice
/// inside that tier — which is what makes the choice weighted and, because
/// every tier is nominated at least once per window, starvation-free.
///
/// `priorities` is read positionally. A missing entry defaults to
/// [`QueuePriority::Default`] and extra entries are ignored, so a caller cannot
/// make the scheduler panic by passing a differently sized slice.
/// The slice is the tier table a provider's submit path carries for its device
/// queues, so the same expression serves the default path (every queue at
/// [`QueuePriority::Default`], or no table at all) and a provider that installs
/// mixed tiers (`research/docs/21` §6 marks the queues with it).
/// `round_robin_start` is the caller's monotonic selection counter: it is both
/// the position inside the policy window and the tie-break cursor, so the
/// caller keeps advancing it by one per selection exactly as
/// `VulkanContext::select_queue` does. An empty queue list returns `0`, the
/// same answer the least-loaded rule gives today.
///
/// With a single tier the function reduces to the least-loaded rule with a
/// rotated tie-break, i.e. to the behaviour of `select_queue` in
/// `metal-api-vulkan`.
pub fn select_queue_with_priority(
    in_flight: &[usize],
    priorities: &[QueuePriority],
    round_robin_start: usize,
    policy: QueueSchedulingPolicy,
) -> usize {
    let len = in_flight.len();
    if len == 0 {
        return 0;
    }
    let start = round_robin_start % len;
    let priority_at = |index: usize| priorities.get(index).copied().unwrap_or_default();

    let min_load = in_flight.iter().copied().min().unwrap_or(0);
    let nominated = policy.nominated_priority(round_robin_start as u64);

    // The wanted tier is folded over the least-loaded queues only, so it is
    // always a tier that at least one candidate actually has.
    let mut wanted: Option<QueuePriority> = None;
    for (index, load) in in_flight.iter().enumerate() {
        if *load != min_load {
            continue;
        }
        let tier = priority_at(index);
        wanted = Some(match wanted {
            Some(current)
                if tier_distance(tier, nominated) >= tier_distance(current, nominated) =>
            {
                current
            }
            _ => tier,
        });
    }
    let wanted = wanted.expect("non-empty in-flight slice has a candidate queue");

    for step in 0..len {
        let index = (start + step) % len;
        if in_flight[index] == min_load && priority_at(index) == wanted {
            return index;
        }
    }

    // Unreachable: `wanted` is taken from a least-loaded queue, so the scan
    // above always returns. `start` keeps the function total and in range.
    start
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

/// Whether a provider can still admit new work.
///
/// `DeviceLost` and `Exhausted` are terminal for one provider instance: callers
/// must recreate the provider before retrying. `Exhausted` means the bounded
/// abandonment budget was reached, not that the device was observed as lost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderHealth {
    Usable,
    DeviceLost,
    Exhausted,
}

impl ProviderHealth {
    pub fn is_usable(self) -> bool {
        matches!(self, Self::Usable)
    }
}

/// Deterministic terminal state of one provider instance.
///
/// This is the admission authority the bounded-abandonment contract was
/// missing. [`AbandonmentBudget`] and [`AbandonmentLedger`] only report how
/// many unobservable submissions were tolerated; they do not decide what a new
/// submission must answer afterwards. `ProviderLifecycle` composes the budget
/// ledger with the [`LeaseLedger`] that owns lease retirement, so the two
/// documented terminal causes stay distinguishable end to end.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminalState {
    /// New work may still be admitted.
    Usable,
    /// The bounded abandonment budget is exhausted. The device was not
    /// necessarily observed as lost; abandoned resources stay retained until
    /// the process or provider instance goes away.
    Exhausted { submissions: u64, bytes: u64 },
    /// The device was observed as lost. Lease retirement is a teardown
    /// guarantee, so every lease is released regardless of outstanding tokens.
    DeviceLost,
}

/// Why a new submission was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminalRefusalReason {
    /// Bounded abandonment budget exhausted; the device may still be alive.
    AbandonmentBudget,
    /// The device is gone.
    DeviceLost,
}

/// Structured refusal of a new submission against a terminal provider instance.
///
/// Callers can branch on [`TerminalRefusal::reason`] or on the normalized
/// [`ProviderError`] fields instead of matching message strings. The refusal is
/// produced for every attempt while the provider stays terminal, so repeated
/// calls observe the same typed reason.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TerminalRefusal {
    reason: TerminalRefusalReason,
    state: TerminalState,
    /// Boxed so an `Err` from admission stays small; clippy's
    /// `result_large_err` lint flags the inline variant.
    error: Box<ProviderError>,
}

impl TerminalRefusal {
    /// The terminal cause that refused this submission.
    pub const fn reason(&self) -> TerminalRefusalReason {
        self.reason
    }

    /// The terminal state observed when the refusal was produced.
    pub const fn state(&self) -> TerminalState {
        self.state
    }

    /// Whether recreating the provider is the documented recovery.
    pub const fn requires_recreate(&self) -> bool {
        matches!(self.error.retryability, Retryability::RetryAfterRecreate)
    }

    /// Structured provider error with a stable class, slug, fields and
    /// retryability; callers may surface it directly across the provider
    /// boundary.
    pub const fn error(&self) -> &ProviderError {
        &self.error
    }

    /// Consume the refusal and return the structured provider error.
    pub fn into_error(self) -> ProviderError {
        *self.error
    }

    fn abandonment_budget(state: TerminalState) -> Self {
        let (submissions, bytes) = match state {
            TerminalState::Exhausted { submissions, bytes } => (submissions, bytes),
            _ => (0, 0),
        };
        let mut error = ProviderError::new(
            ProviderPhase::Resolve,
            ProviderErrorClass::Resource,
            "provider_unavailable",
        )
        .expect("static provider refusal slug")
        .with_field("terminal", FieldValue::Text("abandonment_budget".into()))
        .with_field("abandoned_submissions", FieldValue::Unsigned(submissions))
        .with_field("abandoned_bytes", FieldValue::Unsigned(bytes));
        error.retryability = Retryability::RetryAfterRecreate;
        Self {
            reason: TerminalRefusalReason::AbandonmentBudget,
            state,
            error: Box::new(error),
        }
    }

    fn device_lost(state: TerminalState) -> Self {
        let mut error = ProviderError::new(
            ProviderPhase::Resolve,
            ProviderErrorClass::DeviceLost,
            "device_lost",
        )
        .expect("static provider refusal slug")
        .with_field("terminal", FieldValue::Text("device_lost".into()));
        error.retryability = Retryability::RetryAfterRecreate;
        error.completion = CompletionDisposition::DeviceLost { token: None };
        Self {
            reason: TerminalRefusalReason::DeviceLost,
            state,
            error: Box::new(error),
        }
    }
}

/// Provider-scoped admission and terminal-state authority.
///
/// States are monotonic and never return to `Usable`. `Exhausted` and
/// `DeviceLost` are terminal for one instance, so callers must recreate the
/// provider before retrying; `health` on this lifecycle is therefore the
/// single source of truth for admission. `ProviderHealth::Exhausted` only
/// tells the caller *that* the budget ran out, not that the device was
/// observed as lost:
///
/// - `Exhausted` means abandoned submissions, not a lost device. A recreated
///   provider may reuse the same physical device.
/// - `DeviceLost` means the device is gone. Recreate the provider, and treat
///   the epoch change as the proof that prior GPU work can no longer retire.
///   Lease retirement is still a teardown guarantee, so
///   [`ProviderLifecycle::leases`] releases every lease even when no
///   completion was ever observed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderLifecycle {
    budget: AbandonmentBudget,
    ledger: AbandonmentLedger,
    leases: LeaseLedger,
    state: TerminalState,
    poisoned: bool,
}

impl ProviderLifecycle {
    /// Build a usable lifecycle for one provider instance.
    ///
    /// `max_submissions` is the abandoned count at which the budget is
    /// exhausted; a value of 1 fails closed on the first abandonment.
    pub fn new(max_submissions: u64, max_bytes: u64) -> Self {
        Self {
            budget: AbandonmentBudget::new(max_submissions, max_bytes),
            ledger: AbandonmentLedger::default(),
            leases: LeaseLedger::new(),
            state: TerminalState::Usable,
            poisoned: false,
        }
    }

    /// The admission budget this lifecycle enforces.
    pub const fn budget(&self) -> AbandonmentBudget {
        self.budget
    }

    /// Provider health for this instance. `Usable` is the only state that may
    /// admit new work.
    pub const fn health(&self) -> ProviderHealth {
        match self.state {
            TerminalState::Usable => ProviderHealth::Usable,
            TerminalState::Exhausted { .. } => ProviderHealth::Exhausted,
            TerminalState::DeviceLost => ProviderHealth::DeviceLost,
        }
    }

    /// The full terminal state, including the abandoned counters for
    /// `Exhausted`. Callers that only need admission use [`Self::health`].
    pub const fn state(&self) -> TerminalState {
        self.state
    }

    /// Whether this instance is still usable.
    pub const fn is_usable(&self) -> bool {
        matches!(self.state, TerminalState::Usable)
    }

    /// Whether termination came from an observed device loss. This is the
    /// query that separates `DeviceLost` from budget exhaustion; it is `false`
    /// both while usable and after budget exhaustion.
    pub const fn ended_by_device_loss(&self) -> bool {
        matches!(self.state, TerminalState::DeviceLost)
    }

    /// Whether termination came from the bounded abandonment budget.
    pub const fn ended_by_abandonment_budget(&self) -> bool {
        matches!(self.state, TerminalState::Exhausted { .. })
    }

    /// Admit one new submission.
    ///
    /// While usable this returns `Ok(())`. Once terminal it returns the same
    /// structured refusal on every call, so retries are deterministic and
    /// idempotent instead of string-matched.
    pub fn admit(&self) -> Result<(), TerminalRefusal> {
        match self.state {
            TerminalState::Usable => Ok(()),
            TerminalState::Exhausted { .. } => Err(TerminalRefusal::abandonment_budget(self.state)),
            TerminalState::DeviceLost => Err(TerminalRefusal::device_lost(self.state)),
        }
    }

    /// Record one submission whose completion can no longer be observed.
    ///
    /// The first abandonment that reaches the configured submission or byte
    /// limit makes the instance terminal. Recording after that point is a
    /// no-op for admission: the state stays `Exhausted` and the returned
    /// outcome stays `Exhausted`, so callers can repeat the call safely.
    pub fn record_abandonment(&mut self, bytes: u64) -> AbandonmentOutcome {
        if !matches!(self.state, TerminalState::Usable) {
            return AbandonmentOutcome::Exhausted;
        }
        let outcome = self.ledger.record(self.budget, bytes);
        if outcome == AbandonmentOutcome::Exhausted {
            self.poisoned = true;
            self.state = TerminalState::Exhausted {
                submissions: self.ledger.submissions(),
                bytes: self.ledger.bytes(),
            };
        }
        outcome
    }

    /// Mark the device as lost.
    ///
    /// This is the test-injection entry point that mirrors
    /// `inject_device_loss_for_test()`. It is terminal and idempotent, and it
    /// retires every lease because a lost device is a teardown guarantee.
    pub fn mark_device_lost(&mut self) {
        self.leases.device_lost();
        self.poisoned = true;
        self.state = TerminalState::DeviceLost;
    }

    /// Fail the instance closed for one submission whose completion can no
    /// longer be observed, without charging the abandonment ledger.
    ///
    /// [`ProviderLifecycle::record_abandonment`] is the entry point for a
    /// submission that is given up on: it counts the submission and its bytes
    /// before the budget decides. This is the entry point for transitions that
    /// only *classify* a submission the instance already gave up on — a queue
    /// that refused the submission, or a device-loss-class error observed
    /// after the work retired. Nothing is added to
    /// [`ProviderLifecycle::abandonment`], so the counters keep reporting real
    /// abandoned work, but the instance leaves `Usable` all the same: it stays
    /// terminal until it is recreated.
    ///
    /// The instance ends in [`TerminalState::Exhausted`], not `DeviceLost`,
    /// because no device loss was observed; admission therefore keeps
    /// answering with the `abandonment_budget` refusal. `DeviceLost` is
    /// terminal-ordered above `Exhausted`, so a state that already observed a
    /// device loss is never relabelled or downgraded.
    pub fn mark_unobservable_submission(&mut self) {
        if matches!(self.state, TerminalState::DeviceLost) {
            return;
        }
        self.poisoned = true;
        self.state = TerminalState::Exhausted {
            submissions: self.ledger.submissions(),
            bytes: self.ledger.bytes(),
        };
    }

    /// Whether the instance has been poisoned by exhaustion or device loss.
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Abandonment counters `(submissions, bytes)` recorded so far.
    pub const fn abandonment(&self) -> (u64, u64) {
        (self.ledger.submissions(), self.ledger.bytes())
    }

    /// Lease ledger for this lifecycle.
    pub fn leases(&self) -> &LeaseLedger {
        &self.leases
    }

    /// Mutable lease ledger for registration and binding.
    pub fn leases_mut(&mut self) -> &mut LeaseLedger {
        &mut self.leases
    }
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
    TextureSampleCountMismatch {
        texture_type: TextureType,
        sample_count: u64,
    },
    TextureArrayLengthMismatch {
        texture_type: TextureType,
        array_length: u64,
    },
    /// Retained for binding shapes that are not implemented yet. Admission no
    /// longer uses it: `ComputePass::validate` accepts texture bindings and both
    /// providers execute them, so the variant currently has **no construction
    /// point** — only the error-class/slug mapping and its `Display` arm remain.
    TextureBindingUnsupported(u32),
    // Render contract, Step 1 (`research/docs/23` §3.1). Structural refusals
    // come first, then the first-increment capability narrowings.
    EmptyAttachmentList,
    AttachmentLimitExceeded {
        requested: usize,
        maximum: usize,
    },
    UnsupportedAttachmentFormat(AttachmentFormat),
    UnsupportedAttachmentLoadOp(LoadOp),
    UnsupportedAttachmentStoreOp(StoreOp),
    ViewportOriginUnsupported {
        origin: [u32; 2],
    },
    ViewportExtentMismatch {
        viewport: [u32; 2],
        attachment: [u64; 2],
    },
    DrawVertexCountMismatch {
        expected: u32,
        actual: u32,
    },
    // Presentation contract, Step 1 (`research/docs/24` §3.1). The sentinel
    // and source-shape refusals are caller-fixable structure; the three
    // first-increment narrowings (image count, present mode, acquire policy) are
    // capability refusals on a well-formed request, the way the render track
    // already splits those two families.
    PresentSentinelLengthMismatch {
        format: AttachmentFormat,
        expected: u64,
        actual: u64,
    },
    /// The present names a view the pass it hangs off does not render into.
    ///
    /// Kept distinct from [`Self::AttachmentViewUnknown`]: that variant reports
    /// an attachment the *trace's resource table* does not declare, while this
    /// one reports a source the pass itself does not carry as an attachment.
    PresentSourceUnknown {
        source: ViewId,
    },
    /// The target names a view other than the source it presents.
    ///
    /// `docs/24` §3.1 requires `source` to be the same view as the pass's
    /// colour attachment and §3.2 makes the target a restatement of that
    /// resource's identity, so two different view identities in one present
    /// describe two resources where the first increment has exactly one.
    PresentTargetViewMismatch {
        source: ViewId,
        target: ViewId,
    },
    /// The target's allocation is not the source attachment's allocation.
    ///
    /// Same rule as [`Self::PresentTargetViewMismatch`], one level up: §3.1
    /// keeps "the same lease and byte-range semantics" for the pair, so the
    /// target cannot name a different allocation than the attachment it is
    /// handed on from.
    PresentTargetAllocationMismatch {
        source: ViewId,
        target: AllocationId,
        attachment: AllocationId,
    },
    PresentFormatMismatch {
        source: ViewId,
        target: AttachmentFormat,
        attachment: AttachmentFormat,
    },
    PresentExtentMismatch {
        source: ViewId,
        target: [u64; 2],
        attachment: [u64; 2],
    },
    PresentImageCountUnsupported {
        requested: u32,
        maximum: u32,
    },
    PresentModeUnsupported(PresentMode),
    PresentAcquirePolicyUnsupported(AcquirePolicy),
    // Render pipeline contract, Step 3a (`research/docs/23` §3.4). The entry
    // shape and the pipeline/attachment agreement are caller-fixable structure,
    // like the pass rules above; the colour format reuses
    // `UnsupportedAttachmentFormat`, whose refusal stays a capability narrowing
    // rather than becoming an argument error.
    EmptyRenderPipelineEntry(RenderPipelineStage),
    /// The stage whose entry repeats the other stage's entry name. Reported for
    /// the later of the two fields, i.e. always the fragment entry.
    DuplicateRenderPipelineEntry(RenderPipelineStage),
    RenderPipelineFormatMismatch {
        pipeline: AttachmentFormat,
        attachment: AttachmentFormat,
    },
    /// A render pass names a table entry that carries no render half, so the
    /// trace cannot say what the pass would render with.
    ///
    /// Kept distinct from [`Self::UnknownPipeline`]: the id exists and the
    /// entry is well formed, it simply has no `color_format` to agree with, and
    /// reusing the unknown-id variant would name a pipeline the trace does
    /// carry.
    MissingRenderPipelineContract {
        pass_index: usize,
        pipeline: PipelineId,
    },
    // Render contract, Step 3c (`research/docs/23` §3.6): an attachment is
    // resolved against the trace's own resource table, so these refusals name
    // both the attachment's trace entry and the declaration it disagreed with.
    AttachmentViewUnknown {
        pass_index: usize,
        view: ViewId,
        allocation: AllocationId,
    },
    AttachmentViewAllocationMismatch {
        pass_index: usize,
        view: ViewId,
        declared: AllocationId,
        referenced: AllocationId,
    },
    AttachmentExtentMismatch {
        pass_index: usize,
        view: ViewId,
        expected: u64,
        declared: u64,
    },
    /// A texture declaration does not describe the attachment's shape.
    ///
    /// Kept beside [`Self::AttachmentExtentMismatch`] rather than folded into
    /// it: a buffer declaration can only disagree about a byte count, while a
    /// texture can disagree about the texel shape or the format and still hold
    /// the same number of bytes. Reporting only the two byte counts there left
    /// `expected: 16, declared: 16` in the message, which names no field to fix
    /// (review item M4, 2026-09-14).
    AttachmentTextureShapeMismatch {
        pass_index: usize,
        view: ViewId,
        /// The attachment's own shape: `[width, height]` texels of
        /// `attachment_format`, i.e. `attachment_bytes` tightly packed bytes.
        attachment_extent: [u64; 2],
        attachment_format: AttachmentFormat,
        attachment_bytes: u64,
        /// The shape the trace's own declaration states.
        declared_type: TextureType,
        declared_extent: [u64; 2],
        declared_depth: u64,
        declared_array_length: u64,
        declared_sample_count: u64,
        declared_format: TextureFormat,
        declared_bytes: u64,
    },
    AttachmentComputeConflict {
        pass_index: usize,
        /// The attachment the render pass stores into.
        view: ViewId,
        /// The view the conflicting compute pass writes. It is `view` itself
        /// when the same identity is declared writable, and a sibling view of
        /// the same allocation when only the byte ranges overlap.
        compute_view: ViewId,
        compute_pass: usize,
    },
    /// A compute pass that follows a render pass binds bytes that render pass
    /// stored.
    ///
    /// The first increment executes every compute pass before every render pass,
    /// so this is the one shape whose meaning the execution order would change:
    /// the trace's own order defines post-render bytes for the later compute
    /// pass, and the grouped order would hand it pre-render ones
    /// (review item I4, 2026-09-14). Refusing it is what makes a trace mean one
    /// thing on every provider instead of leaving the rule to each of them.
    RenderPassOrderUnsupported {
        /// The render pass whose store the later compute pass overlaps.
        pass_index: usize,
        /// The compute pass that follows it.
        compute_pass: usize,
        /// The attachment the render pass stores into.
        view: ViewId,
        /// The view the later compute pass binds.
        compute_view: ViewId,
    },
    LeaseSourceLengthMismatch {
        lease: LeaseId,
        expected: u64,
        actual: u64,
    },
    NullHostPointer(LeaseId),
    GuestWindowStillActive(LeaseId),
    InvalidHostRegionPageSize(u64),
    UnalignedHostRegion {
        field: &'static str,
        value: u64,
        page_size: u64,
    },
    HostRegionWindowOutOfBounds {
        end: u64,
        region_length: u64,
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
    InvalidReadbackCompletion(CompletionDisposition),
    WritebackBeforeCompletion,
    CompletionEpochMismatch {
        expected: DeviceEpoch,
        actual: DeviceEpoch,
    },
    ConcurrentPassesUnsupported,
    SerialBufferRebinding {
        pass_index: usize,
    },
    SerialResourceLimit {
        requested: usize,
        maximum: usize,
    },
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
    EmptyPipelineTable,
    UnknownPipeline(PipelineId),
    DuplicatePipeline(PipelineId),
    UnusedPipeline(PipelineId),
    PipelineEpochMismatch {
        pipeline: PipelineId,
        expected: DeviceEpoch,
        actual: DeviceEpoch,
    },
    UnsupportedDispatchType(DispatchType),
    SnapshotDispatchUnsupported(DispatchKind),
    SnapshotAliasUnsupported(AllocationId),
    MissingSnapshotIdentity(u32),
    UnknownSnapshotIdentity(u32),
    DispatchKindMismatch {
        expected: DispatchKind,
        actual: DispatchKind,
    },
    CompletionPublishAfterTerminal(CompletionToken),
    CompletionPublishAfterDeviceLost(CompletionToken),
    CompletionHealthRegression {
        current: ProviderHealth,
        requested: ProviderHealth,
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
                write!(
                    formatter,
                    "push constant offset {offset} is not 4-byte aligned"
                )
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
            Self::TextureSampleCountMismatch {
                texture_type,
                sample_count,
            } => write!(
                formatter,
                "texture type {texture_type:?} does not admit sample count {sample_count}"
            ),
            Self::TextureArrayLengthMismatch {
                texture_type,
                array_length,
            } => write!(
                formatter,
                "texture type {texture_type:?} does not admit array length {array_length}"
            ),
            Self::TextureBindingUnsupported(binding) => write!(
                formatter,
                "texture binding {binding} is carried by the trace but not yet executed"
            ),
            Self::EmptyAttachmentList => {
                formatter.write_str("render pass needs at least one colour attachment")
            },
            Self::AttachmentLimitExceeded { requested, maximum } => write!(
                formatter,
                "render pass declares {requested} colour attachments, exceeding {maximum}"
            ),
            Self::UnsupportedAttachmentFormat(format) => write!(
                formatter,
                "attachment format {format:?} is outside the first render increment"
            ),
            Self::UnsupportedAttachmentLoadOp(load) => write!(
                formatter,
                "attachment load operation {load:?} is outside the first render increment"
            ),
            Self::UnsupportedAttachmentStoreOp(store) => write!(
                formatter,
                "attachment store operation {store:?} is outside the first render increment"
            ),
            Self::ViewportOriginUnsupported { origin } => write!(
                formatter,
                "viewport origin {origin:?} is outside the first render increment's (0, 0)"
            ),
            Self::ViewportExtentMismatch {
                viewport,
                attachment,
            } => write!(
                formatter,
                "viewport extent {viewport:?} does not cover attachment extent {attachment:?}"
            ),
            Self::DrawVertexCountMismatch { expected, actual } => write!(
                formatter,
                "draw vertex count mismatch: expected {expected}, received {actual}"
            ),
            Self::PresentSentinelLengthMismatch {
                format,
                expected,
                actual,
            } => write!(
                formatter,
                "present sentinel is {actual} bytes, but format {format:?} has {expected}-byte texels"
            ),
            Self::PresentSourceUnknown { source } => write!(
                formatter,
                "present source view {source:?} is not a colour attachment of the pass it presents"
            ),
            Self::PresentTargetViewMismatch { source, target } => write!(
                formatter,
                "present target view {target:?} is not the source view {source:?} it presents"
            ),
            Self::PresentTargetAllocationMismatch {
                source,
                target,
                attachment,
            } => write!(
                formatter,
                "present target allocation {target:?} is not the allocation {attachment:?} of source view {source:?}"
            ),
            Self::PresentFormatMismatch {
                source,
                target,
                attachment,
            } => write!(
                formatter,
                "present target format {target:?} does not match source attachment {source:?} format {attachment:?}"
            ),
            Self::PresentExtentMismatch {
                source,
                target,
                attachment,
            } => write!(
                formatter,
                "present target extent {target:?} does not match source attachment {source:?} extent {attachment:?}"
            ),
            Self::PresentImageCountUnsupported { requested, maximum } => write!(
                formatter,
                "present target declares {requested} images, exceeding {maximum}"
            ),
            Self::PresentModeUnsupported(mode) => write!(
                formatter,
                "present mode {mode:?} is outside the first presentation increment"
            ),
            Self::PresentAcquirePolicyUnsupported(policy) => write!(
                formatter,
                "present acquire policy {policy:?} is outside the first presentation increment"
            ),
            Self::EmptyRenderPipelineEntry(stage) => write!(
                formatter,
                "render pipeline {} entry name must not be empty",
                stage.name()
            ),
            Self::DuplicateRenderPipelineEntry(stage) => write!(
                formatter,
                "render pipeline {} entry repeats the other stage's entry name",
                stage.name()
            ),
            Self::RenderPipelineFormatMismatch {
                pipeline,
                attachment,
            } => write!(
                formatter,
                "render pipeline colour format {pipeline:?} does not match attachment format {attachment:?}"
            ),
            Self::MissingRenderPipelineContract {
                pass_index,
                pipeline,
            } => write!(
                formatter,
                "render pass {pass_index} names pipeline {pipeline:?}, but that entry carries no render contract to render with"
            ),
            Self::AttachmentViewUnknown {
                pass_index,
                view,
                allocation,
            } => write!(
                formatter,
                "render pass {pass_index} attachment view {view:?} (allocation {allocation:?}) is not declared by this trace"
            ),
            Self::AttachmentViewAllocationMismatch {
                pass_index,
                view,
                declared,
                referenced,
            } => write!(
                formatter,
                "render pass {pass_index} attachment view {view:?} references allocation {referenced:?}, but the trace declares it as {declared:?}"
            ),
            Self::AttachmentExtentMismatch {
                pass_index,
                view,
                expected,
                declared,
            } => write!(
                formatter,
                "render pass {pass_index} attachment view {view:?} covers {expected} bytes, but the trace declares {declared}"
            ),
            Self::AttachmentTextureShapeMismatch {
                pass_index,
                view,
                attachment_extent,
                attachment_format,
                attachment_bytes,
                declared_type,
                declared_extent,
                declared_depth,
                declared_array_length,
                declared_sample_count,
                declared_format,
                declared_bytes,
            } => write!(
                formatter,
                "render pass {pass_index} attachment view {view:?} is {width}×{height} {attachment_format:?} \
                 ({attachment_bytes} bytes), but the trace declares texture {declared_type:?} \
                 {declared_width}×{declared_height}×{declared_depth}, array {declared_array_length}, \
                 samples {declared_sample_count}, {declared_format:?} ({declared_bytes} bytes)",
                width = attachment_extent[0],
                height = attachment_extent[1],
                declared_width = declared_extent[0],
                declared_height = declared_extent[1],
            ),
            Self::AttachmentComputeConflict {
                pass_index,
                view,
                compute_view,
                compute_pass,
            } => write!(
                formatter,
                "render pass {pass_index} attachment view {view:?} shares bytes with compute pass {compute_pass}, which writes view {compute_view:?}"
            ),
            Self::RenderPassOrderUnsupported {
                pass_index,
                compute_pass,
                view,
                compute_view,
            } => write!(
                formatter,
                "render pass {pass_index} stores attachment view {view:?}, and compute pass {compute_pass} follows it while binding view {compute_view:?}: this increment runs every compute pass before every render pass, so the later read would see pre-render bytes"
            ),
            Self::LeaseSourceLengthMismatch {
                lease,
                expected,
                actual,
            } => write!(
                formatter,
                "lease {lease:?} staged length {actual} does not match reservation length {expected}"
            ),
            Self::NullHostPointer(lease) => {
                write!(formatter, "lease {lease:?} borrowed host pointer is null")
            }
            Self::GuestWindowStillActive(lease) => write!(
                formatter,
                "guest window {lease:?} is still active and cannot be reclaimed"
            ),
            Self::InvalidHostRegionPageSize(page_size) => write!(
                formatter,
                "host region page size {page_size} must be a nonzero power of two"
            ),
            Self::UnalignedHostRegion {
                field,
                value,
                page_size,
            } => write!(
                formatter,
                "host region {field} {value} is not aligned to {page_size}"
            ),
            Self::HostRegionWindowOutOfBounds {
                end,
                region_length,
            } => write!(
                formatter,
                "host region window ends at {end}, beyond region length {region_length}"
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
                "lease {:?} range end {end} exceeds allocation size {allocation_size}",
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
            Self::InvalidReadbackCompletion(completion) => write!(
                formatter,
                "completion readback requires CompletedVisible, received {completion:?}"
            ),
            Self::WritebackBeforeCompletion => {
                formatter.write_str("writebacks require CompletedVisible")
            }
            Self::CompletionEpochMismatch { expected, actual } => write!(
                formatter,
                "completion epoch mismatch: expected {expected:?}, received {actual:?}"
            ),
            Self::ConcurrentPassesUnsupported => {
                formatter.write_str("multiple compute passes require serial dispatch")
            }
            Self::SerialBufferRebinding { pass_index } => write!(
                formatter,
                "serial pass {pass_index} changes a previously declared view's allocation, range or source bytes"
            ),
            Self::SerialResourceLimit { requested, maximum } => write!(
                formatter,
                "serial resource pool contains {requested} distinct views, exceeding {maximum}"
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
            Self::ReadOnlyWriteback(view) => {
                write!(formatter, "writeback view {view:?} is not writable")
            }
            Self::WritebackRangeOutOfBounds {
                view,
                offset,
                end,
                view_offset,
                view_end,
            } => write!(
                formatter,
                "writeback view {view:?} range {offset}..{end} is outside view range {view_offset}..{view_end}"
            ),
            Self::IncompleteWriteback(view) => {
                write!(formatter, "writeback must cover all of view {view:?}")
            }
            Self::MissingWriteback { allocation, view } => write!(
                formatter,
                "completed host readback is missing allocation {allocation:?}, view {view:?}"
            ),
            Self::ViewIdentityMismatch(view) => {
                write!(
                    formatter,
                    "view {:?} changes resource declaration across passes",
                    view
                )
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
                write!(
                    formatter,
                    "unsupported provider trace schema version {version}"
                )
            }
            Self::EmptyPipelineTable => {
                formatter.write_str("compute trace must contain pipeline metadata")
            }
            Self::UnknownPipeline(pipeline) => write!(formatter, "unknown pipeline {pipeline:?}"),
            Self::DuplicatePipeline(pipeline) => {
                write!(formatter, "duplicate pipeline {pipeline:?}")
            }
            Self::UnusedPipeline(pipeline) => write!(formatter, "unused pipeline {pipeline:?}"),
            Self::PipelineEpochMismatch {
                pipeline,
                expected,
                actual,
            } => write!(
                formatter,
                "pipeline {pipeline:?} epoch mismatch: expected {expected:?}, received {actual:?}"
            ),
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
                write!(
                    formatter,
                    "snapshot binding {binding} has no explicit resource identity"
                )
            }
            Self::UnknownSnapshotIdentity(binding) => {
                write!(
                    formatter,
                    "snapshot identity {binding} has no matching buffer"
                )
            }
            Self::DispatchKindMismatch { expected, actual } => write!(
                formatter,
                "dispatch kind mismatch: expected {expected:?}, received {actual:?}"
            ),
            Self::CompletionPublishAfterTerminal(token) => {
                write!(formatter, "completion token {:?} is already terminal", token)
            }
            Self::CompletionPublishAfterDeviceLost(token) => write!(
                formatter,
                "completion token {:?} cannot be published after device loss",
                token
            ),
            Self::CompletionHealthRegression { current, requested } => write!(
                formatter,
                "completion health regression: current {current:?}, requested {requested:?}"
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

    struct TextureCase {
        texture_type: TextureType,
        format: TextureFormat,
        width: u64,
        height: u64,
        depth: u64,
        array_length: u64,
        sample_count: u64,
        bytes: Vec<u8>,
    }

    fn texture_view(case: TextureCase) -> TextureView {
        TextureView {
            view_id: ViewId::new(7),
            metal_binding: 3,
            allocation_id: AllocationId::new(11),
            texture_type: case.texture_type,
            format: case.format,
            width: case.width,
            height: case.height,
            depth: case.depth,
            array_length: case.array_length,
            sample_count: case.sample_count,
            access: TextureAccess::Sampled,
            source: TextureSource::OwnedBytes(case.bytes),
        }
    }

    #[test]
    fn texture_extent_is_the_tightly_packed_texel_count() {
        let view = texture_view(TextureCase {
            texture_type: TextureType::D2,
            format: TextureFormat::R32Uint,
            width: 4,
            height: 4,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            bytes: vec![0; 64],
        });
        assert_eq!(view.expected_bytes().expect("bounded extent"), 64);
        view.validate_shape().expect("4x4 R32Uint is valid");

        let array = texture_view(TextureCase {
            texture_type: TextureType::D2Array,
            format: TextureFormat::Rgba8Unorm,
            width: 2,
            height: 2,
            depth: 1,
            array_length: 3,
            sample_count: 1,
            bytes: vec![0; 48],
        });
        assert_eq!(array.expected_bytes().expect("bounded extent"), 48);
        array.validate_shape().expect("2x2x3 RGBA8 array is valid");
    }

    #[test]
    fn texture_source_length_must_match_the_declared_extent() {
        let short = texture_view(TextureCase {
            texture_type: TextureType::D2,
            format: TextureFormat::R32Uint,
            width: 4,
            height: 4,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            bytes: vec![0; 60],
        });
        assert!(matches!(
            short.validate_shape(),
            Err(ContractError::SourceLengthMismatch {
                expected: 64,
                actual: 60,
                ..
            })
        ));
    }

    #[test]
    fn texture_zero_dimensions_are_refused() {
        let empty = texture_view(TextureCase {
            texture_type: TextureType::D2,
            format: TextureFormat::R32Uint,
            width: 0,
            height: 4,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            bytes: Vec::new(),
        });
        assert!(matches!(
            empty.validate_shape(),
            Err(ContractError::ZeroDimension { axis: 0, .. })
        ));
    }

    #[test]
    fn texture_sample_count_must_match_the_texture_type() {
        let mismatched = texture_view(TextureCase {
            texture_type: TextureType::D2,
            format: TextureFormat::R32Uint,
            width: 2,
            height: 2,
            depth: 1,
            array_length: 1,
            sample_count: 4,
            bytes: vec![0; 64],
        });
        assert!(matches!(
            mismatched.validate_shape(),
            Err(ContractError::TextureSampleCountMismatch {
                sample_count: 4,
                ..
            })
        ));

        let multisample = texture_view(TextureCase {
            texture_type: TextureType::D2Multisample,
            format: TextureFormat::R32Uint,
            width: 2,
            height: 2,
            depth: 1,
            array_length: 1,
            sample_count: 4,
            bytes: vec![0; 64],
        });
        multisample
            .validate_shape()
            .expect("4x multisample R32Uint is valid");
    }

    #[test]
    fn texture_array_length_must_match_the_texture_type() {
        let mismatched = texture_view(TextureCase {
            texture_type: TextureType::D2,
            format: TextureFormat::R32Uint,
            width: 2,
            height: 2,
            depth: 1,
            array_length: 4,
            sample_count: 1,
            bytes: vec![0; 64],
        });
        assert!(matches!(
            mismatched.validate_shape(),
            Err(ContractError::TextureArrayLengthMismatch {
                array_length: 4,
                ..
            })
        ));
    }

    #[test]
    fn texture_extent_overflow_is_refused_not_wrapped() {
        let huge = texture_view(TextureCase {
            texture_type: TextureType::D2,
            format: TextureFormat::R32Uint,
            width: u64::MAX,
            height: u64::MAX,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            bytes: Vec::new(),
        });
        assert!(matches!(
            huge.expected_bytes(),
            Err(ContractError::ArithmeticOverflow("texture extent"))
        ));
    }

    fn unbound_pipeline_contract() -> PipelineContract {
        PipelineContract {
            dispatch_kind: DispatchKind::ThreadsExact,
            required_local_size: None,
            fixed_grid: None,
            push_constant_offset: 0,
            push_constant_bytes: 0,
            buffer_bindings: Vec::new(),
            shader_capabilities: Vec::new(),
            translator_revision: None,
        }
    }

    #[test]
    fn guest_windows_refuse_reclaim_before_retirement() {
        let mut windows = GuestWindows::new();
        let window = GuestWindow {
            lease: LeaseId::new(51),
            allocation_id: AllocationId::new(61),
            offset: 0,
            length: 0x2000,
        };
        windows.register(window).expect("register");
        assert_eq!(windows.len(), 1);
        assert!(!windows.is_reclaimable(window.lease));
        // Reclaiming an active window must fail and leave it registered.
        assert!(matches!(
            windows.reclaim(window.lease),
            Err(ContractError::GuestWindowStillActive(LeaseId(51)))
        ));
        assert_eq!(windows.len(), 1);

        windows.retire(window.lease).expect("retire");
        assert!(windows.is_reclaimable(window.lease));
        assert_eq!(windows.reclaim(window.lease).expect("reclaim"), window);
        assert!(windows.is_empty());

        // Unknown or malformed registrations are refused.
        assert!(matches!(
            windows.retire(LeaseId::new(99)),
            Err(ContractError::UnknownLease(LeaseId(99)))
        ));
        assert!(matches!(
            windows.register(GuestWindow {
                lease: LeaseId::new(0),
                allocation_id: AllocationId::new(61),
                offset: 0,
                length: 0x1000,
            }),
            Err(ContractError::InvalidIdentity("guest window lease id"))
        ));
        assert!(matches!(
            windows.register(GuestWindow {
                lease: LeaseId::new(52),
                allocation_id: AllocationId::new(61),
                offset: 0,
                length: 0,
            }),
            Err(ContractError::ZeroLength("guest window"))
        ));
        windows
            .register(GuestWindow {
                lease: LeaseId::new(53),
                allocation_id: AllocationId::new(61),
                offset: 0,
                length: 0x1000,
            })
            .expect("register");
        assert!(matches!(
            windows.register(GuestWindow {
                lease: LeaseId::new(53),
                allocation_id: AllocationId::new(62),
                offset: 0,
                length: 0x1000,
            }),
            Err(ContractError::DuplicateLease(LeaseId(53)))
        ));
        // Retirement is idempotent.
        windows.retire(LeaseId::new(53)).expect("retire");
        windows.retire(LeaseId::new(53)).expect("retire again");
    }

    #[test]
    fn dirty_set_marks_pages_and_coalesces_adjacent_ranges() {
        let mut dirty = DirtySet::new(0x1000).expect("page size");
        assert!(dirty.is_empty());
        dirty.mark(0x40, 4).expect("mark inside the first page");
        assert_eq!(dirty.ranges(), [(0, 0x1000)]);
        // A write at the start of the next page coalesces with the first.
        dirty.mark(0x1000, 8).expect("mark the second page");
        assert_eq!(dirty.ranges(), [(0, 0x2000)]);
        // A distant write stays separate.
        dirty.mark(0x4000, 0x1000).expect("mark a far page");
        assert_eq!(dirty.ranges(), [(0, 0x2000), (0x4000, 0x5000)]);
        // A bridging write merges both neighbours into one extent.
        dirty.mark(0x1800, 0x2800).expect("bridge the gap");
        assert_eq!(dirty.ranges(), [(0, 0x5000)]);
        // Zero-length marks are no-ops; a page size of zero is refused.
        let before = dirty.ranges().to_vec();
        dirty.mark(0x9000, 0).expect("zero length");
        assert_eq!(dirty.ranges(), before.as_slice());
        assert!(matches!(
            DirtySet::new(0x3000),
            Err(ContractError::InvalidHostRegionPageSize(0x3000))
        ));
    }

    #[test]
    fn dirty_set_derives_from_writebacks_in_order() {
        let mut dirty = DirtySet::new(0x1000).expect("page size");
        let writebacks = vec![
            BufferWriteback {
                view_id: ViewId::new(1),
                allocation_id: AllocationId::new(9),
                offset: 0x10,
                bytes: vec![1; 4],
            },
            BufferWriteback {
                view_id: ViewId::new(2),
                allocation_id: AllocationId::new(9),
                offset: 0x1010,
                bytes: vec![2; 4],
            },
        ];
        dirty.mark_writebacks(&writebacks).expect("mark");
        assert_eq!(dirty.ranges(), [(0, 0x2000)]);
    }

    fn host_region(pointer: usize, length: u64, page_size: u64) -> HostRegion {
        HostRegion {
            lease_id: LeaseId::new(41),
            owner_epoch: DeviceEpoch::new(7),
            host_pointer: pointer,
            length,
            page_size,
        }
    }

    #[test]
    fn host_region_validates_shape_and_derives_aligned_windows() {
        let region = host_region(0x1000, 0x4000, 0x1000);
        region.validate().expect("aligned registration is valid");

        let borrowed = region
            .borrowed_window(AllocationId::new(11), 0x1000, 0x2000)
            .expect("an aligned window inside the region derives a lease");
        assert_eq!(borrowed.host_pointer, 0x2000);
        assert_eq!(borrowed.reservation.offset, 0x1000);
        assert_eq!(borrowed.reservation.length, 0x2000);
        assert_eq!(borrowed.reservation.lease.lease_id, LeaseId::new(41));
        assert_eq!(
            borrowed.reservation.lease.allocation_id,
            AllocationId::new(11)
        );

        let page = region
            .borrowed_window(AllocationId::new(11), 0, 0x1000)
            .expect("the first page derives a lease");
        assert_eq!(page.host_pointer, 0x1000);
    }

    #[test]
    fn host_region_refuses_bad_shape_and_out_of_bounds_windows() {
        assert!(matches!(
            host_region(0x1800, 0x1000, 0x1000).validate(),
            Err(ContractError::UnalignedHostRegion {
                field: "pointer",
                ..
            })
        ));
        assert!(matches!(
            host_region(0x1000, 0x1500, 0x1000).validate(),
            Err(ContractError::UnalignedHostRegion {
                field: "length",
                ..
            })
        ));
        assert!(matches!(
            host_region(0x1000, 0x1000, 0x3000).validate(),
            Err(ContractError::InvalidHostRegionPageSize(0x3000))
        ));
        assert!(matches!(
            host_region(0, 0x1000, 0x1000).validate(),
            Err(ContractError::NullHostPointer(LeaseId(41)))
        ));

        let region = host_region(0x1000, 0x2000, 0x1000);
        assert!(matches!(
            region.borrowed_window(AllocationId::new(11), 0x1000, 0x2000),
            Err(ContractError::HostRegionWindowOutOfBounds {
                end: 0x3000,
                region_length: 0x2000
            })
        ));
        assert!(matches!(
            region.borrowed_window(AllocationId::new(11), 0x800, 0x1000),
            Err(ContractError::UnalignedHostRegion {
                field: "offset",
                ..
            })
        ));
        assert!(matches!(
            region.borrowed_window(AllocationId::new(0), 0, 0x1000),
            Err(ContractError::InvalidIdentity("allocation id"))
        ));
    }

    #[test]
    fn texture_bindings_validate_and_normalize_like_buffer_views() {
        let contract = unbound_pipeline_contract();
        let mut pass = ComputePass {
            pipeline: PipelineId::new(5),
            buffers: Vec::new(),
            textures: Vec::new(),
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
        };
        pass.validate(&contract)
            .expect("a pass without texture bindings validates");

        pass.textures.push(texture_view(TextureCase {
            texture_type: TextureType::D2,
            format: TextureFormat::R32Uint,
            width: 4,
            height: 4,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            bytes: vec![0; 64],
        }));
        pass.validate(&contract)
            .expect("a well-formed texture binding validates");

        // A malformed texture binding reports its shape error.
        pass.textures[0].width = 0;
        assert!(matches!(
            pass.validate(&contract),
            Err(ContractError::ZeroDimension { .. })
        ));
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
                contract: trace(Vec::new()).pipelines[0].contract.clone(),
                render: None,
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
            pipelines: vec![CompiledComputePipeline {
                device_epoch: DeviceEpoch::new(1),
                pipeline_id: PipelineId::new(4),
                function: FunctionIdentity {
                    logical_digest: digest(),
                    entry_name: "copy_word".to_string(),
                    source: FunctionSource::BinaryAir,
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
                        footprint: FootprintProof::Affine {
                            accesses: Vec::new(),
                        },
                    }],
                    shader_capabilities: Vec::new(),
                    translator_revision: None,
                },
                // A compute registration has no render half; a render-bearing
                // fixture declares one on the entry its render pass names.
                render: None,
            }],
            encoder_dispatch_type: DispatchType::Serial,
            passes: passes.into_iter().map(TracePass::Compute).collect(),
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
            textures: Vec::new(),
        }
    }

    fn compute_pass(trace: &ComputeTrace, index: usize) -> &ComputePass {
        trace.passes[index]
            .as_compute()
            .expect("test trace entry is a compute pass")
    }

    fn compute_pass_mut(trace: &mut ComputeTrace, index: usize) -> &mut ComputePass {
        trace.passes[index]
            .as_compute_mut()
            .expect("test trace entry is a compute pass")
    }

    fn compute_passes_mut(trace: &mut ComputeTrace) -> impl Iterator<Item = &mut ComputePass> {
        trace
            .passes
            .iter_mut()
            .filter_map(TracePass::as_compute_mut)
    }

    fn render_trace_pass(pipeline: u64, width: u64, height: u64) -> TracePass {
        TracePass::Render(RenderPassDescriptor {
            pipeline: PipelineId::new(pipeline),
            color_attachments: vec![RenderAttachment {
                view_id: ViewId::new(7),
                allocation_id: AllocationId::new(9),
                format: AttachmentFormat::Rgba8Unorm,
                width,
                height,
                load: LoadOp::Clear(ClearColor::new([0xfe; 4])),
                store: StoreOp::Store,
            }],
            viewport: [0, 0, width as u32, height as u32],
            vertices: FULL_SCREEN_TRIANGLE_VERTICES,
            present: None,
        })
    }

    /// A render-bearing trace must be refused by a snapshot that never
    /// declared render support, before the pass-count or resource walks, while
    /// the same compute-only trace keeps admitting. The second half proves the
    /// gate is the capability bit and not the trace shape.
    #[test]
    fn render_passes_are_refused_without_the_capability_bit() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        let resources = resources();
        assert!(capabilities().admit(&value, &resources).is_ok());

        value.passes.push(render_trace_pass(4, 2, 2));
        let refusal = capabilities().admit(&value, &resources).unwrap_err();
        assert_eq!(refusal.slug, "render_passes_unsupported");
        assert_eq!(refusal.phase, ProviderPhase::Resolve);
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        assert_eq!(refusal.fields.get("passes"), Some(&FieldValue::Unsigned(1)));
        // The freeze step refuses too: no `ValidatedComputeTrace` is produced
        // for a shape this snapshot cannot execute.
        assert_eq!(
            capabilities()
                .validate_trace(value.clone(), resources.clone())
                .unwrap_err()
                .slug,
            "render_passes_unsupported"
        );

        // Turning the bit on moves the refusal from the shape to the declared
        // attachment limits; the compute path is unchanged.
        let mut render = capabilities();
        render.max_passes = 8;
        render.supports_render_passes = true;
        render.max_color_attachments = 1;
        render.max_attachment_dimension = [2, 2];
        render.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        // The gate is the capability bit and not the trace shape: a
        // render-bearing trace whose attachment resolves against a declared
        // landing view is admitted, while `value` keeps failing the attachment
        // limits below.
        let landing = attachment_trace(landing_view(7, 9), attachment_into(7, 9));
        render.admit(&landing, &landing_resources()).unwrap();

        render.max_attachment_dimension = [1, 2];
        assert_eq!(
            render.admit(&value, &resources).unwrap_err().slug,
            "attachment_dimension_limit"
        );
        render.max_attachment_dimension = [2, 2];
        render.supported_color_formats = vec![AttachmentFormat::R32Float];
        assert_eq!(
            render.admit(&value, &resources).unwrap_err().slug,
            "attachment_format_unsupported"
        );
        render.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        render.max_color_attachments = 0;
        assert_eq!(
            render.admit(&value, &resources).unwrap_err().slug,
            "color_attachment_limit"
        );
    }

    // Render contract, review item I3 (2026-09-14). The render half of a
    // registration travels in the trace's pipeline table, so core admission
    // compares an attachment with the pipeline the pass names instead of
    // leaving that agreement to each provider's own registry lookup.

    /// Give the trace's table entry the render half the render pass names.
    fn declare_render_contract(value: &mut ComputeTrace) {
        value.pipelines[0].render = Some(render_pipeline_contract());
    }

    fn render_entry(value: &mut ComputeTrace) -> &mut RenderPassDescriptor {
        value
            .passes
            .iter_mut()
            .find_map(|pass| match pass {
                TracePass::Render(pass) => Some(pass),
                TracePass::Compute(_) => None,
            })
            .expect("the fixture carries a render pass")
    }

    #[test]
    fn render_admission_compares_the_attachment_with_the_named_pipeline() {
        // The control: the attachment carries the colour format the named
        // pipeline was compiled for, so core admission admits the trace.
        let mut agreeing = attachment_trace(landing_view(7, 9), attachment_into(7, 9));
        declare_render_contract(&mut agreeing);
        render_capabilities()
            .admit(&agreeing, &landing_resources())
            .expect("an attachment in the pipeline's own format is admitted");

        // The counterpart: another admitted colour format is still not the
        // format this pipeline's stages were compiled against, and the refusal
        // now comes from core admission rather than from a provider registry.
        let mut mismatched = agreeing.clone();
        render_entry(&mut mismatched).color_attachments[0].format = AttachmentFormat::Bgra8Unorm;
        let mut render = render_capabilities();
        render.supported_color_formats =
            vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Bgra8Unorm];
        let refusal = render
            .admit(&mismatched, &landing_resources())
            .expect_err("a pipeline cannot render into a format it was not compiled for");
        assert_eq!(refusal.slug, "trace_contract_invalid");
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.phase, ProviderPhase::Resolve);
        assert!(
            refusal
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("does not match attachment format")),
            "the refusal has to name both formats, got {:?}",
            refusal.detail
        );
    }

    #[test]
    fn render_admission_refuses_an_entry_that_carries_no_render_contract() {
        // A render pass names an id whose table entry only carries the compute
        // half: nothing in the trace says what the pass would render with, so
        // the trace is refused instead of falling back to a provider registry.
        let mut value = attachment_trace(landing_view(7, 9), attachment_into(7, 9));
        // The fixture builder declares the half; this case drops it again.
        value.pipelines[0].render = None;
        let expected = ContractError::MissingRenderPipelineContract {
            pass_index: 1,
            pipeline: PipelineId::new(4),
        };
        let refusal = render_capabilities()
            .admit(&value, &landing_resources())
            .unwrap_err();
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "trace_contract_invalid");
        let detail = expected.to_string();
        assert_eq!(refusal.detail.as_deref(), Some(detail.as_str()));
    }

    #[test]
    fn render_admission_keeps_the_unknown_pipeline_refusal() {
        // An id no table entry declares keeps its existing semantics: the
        // structural lookup refuses it before any render agreement is asked.
        let mut value = attachment_trace(landing_view(7, 9), attachment_into(7, 9));
        declare_render_contract(&mut value);
        render_entry(&mut value).pipeline = PipelineId::new(5);
        assert_eq!(
            value.validate(),
            Err(ContractError::UnknownPipeline(PipelineId::new(5)))
        );
        let refusal = render_capabilities()
            .admit(&value, &landing_resources())
            .unwrap_err();
        assert_eq!(refusal.slug, "trace_contract_invalid");
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert!(
            refusal
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("unknown pipeline")),
            "the refusal has to keep the unknown-pipeline detail, got {:?}",
            refusal.detail
        );
    }

    #[test]
    fn the_render_capability_gate_precedes_the_pipeline_agreement() {
        // This snapshot declares no render support at all, so a render-bearing
        // trace is refused by the capability bit even though its attachment
        // format also disagrees with the pipeline: the gate a provider can
        // answer without reading a pipeline has to stay first.
        let mut value = attachment_trace(landing_view(7, 9), attachment_into(7, 9));
        declare_render_contract(&mut value);
        render_entry(&mut value).color_attachments[0].format = AttachmentFormat::Bgra8Unorm;
        let refusal = capabilities().admit(&value, &resources()).unwrap_err();
        assert_eq!(refusal.slug, "render_passes_unsupported");
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
    }

    fn ping_pong_trace() -> ComputeTrace {
        let a = BufferView {
            allocation_id: AllocationId::new(10),
            access: BufferAccess::Read,
            source: BufferSource::OwnedBytes(vec![7; 4]),
            ..buffer(2, 0)
        };
        let b = buffer(1, 1);
        let mut value = trace(vec![
            pass(4, vec![a.clone(), b.clone()]),
            pass(
                4,
                vec![
                    BufferView {
                        metal_binding: 0,
                        access: BufferAccess::Read,
                        ..b
                    },
                    BufferView {
                        metal_binding: 1,
                        access: BufferAccess::Write,
                        ..a
                    },
                ],
            ),
        ]);
        let mut read_binding = value.pipelines[0].contract.buffer_bindings[0].clone();
        read_binding.access = BufferAccess::Read;
        let mut write_binding = value.pipelines[0].contract.buffer_bindings[0].clone();
        write_binding.metal_binding = 1;
        value.pipelines[0].contract.buffer_bindings = vec![read_binding, write_binding];
        value
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
            supports_render_passes: false,
            max_color_attachments: 0,
            max_attachment_dimension: [0, 0],
            supported_color_formats: Vec::new(),
            supports_presentation: false,
            max_present_targets: 0,
            supported_present_modes: Vec::new(),
            max_present_image_count: 0,
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
    fn serial_reuse_requires_stable_initial_view_identity_and_source_bytes() {
        let initial = buffer(1, 0);
        let value = trace(vec![pass(4, vec![initial.clone()]); 2]);
        value.validate_serial_buffer_reuse().unwrap();
        resources().validate_trace(&value).unwrap();

        let mut changed_bytes = initial.clone();
        changed_bytes.source = BufferSource::OwnedBytes(vec![1; 4]);
        let changed_length = BufferView {
            length: 3,
            source: BufferSource::OwnedBytes(vec![0; 3]),
            ..initial.clone()
        };
        for changed in [
            changed_bytes,
            changed_length,
            BufferView {
                allocation_id: AllocationId::new(10),
                ..initial.clone()
            },
            BufferView {
                offset: 1,
                ..initial.clone()
            },
            BufferView {
                source: BufferSource::StagedLease(LeaseId::new(11)),
                ..initial
            },
        ] {
            let mut rebound = value.clone();
            compute_pass_mut(&mut rebound, 1).buffers[0] = changed;
            assert_eq!(
                rebound.validate_serial_buffer_reuse(),
                Err(ContractError::SerialBufferRebinding { pass_index: 1 })
            );
        }

        let mut changed_access = value.clone();
        compute_pass_mut(&mut changed_access, 1).buffers[0].access = BufferAccess::Read;
        assert!(matches!(
            changed_access.validate_serial_buffer_reuse(),
            Err(ContractError::AccessMismatch { binding: 0, .. })
        ));

        let mut changed_pipeline = value;
        compute_pass_mut(&mut changed_pipeline, 1).pipeline = PipelineId::new(5);
        assert_eq!(
            changed_pipeline.validate_serial_buffer_reuse(),
            Err(ContractError::UnknownPipeline(PipelineId::new(5)))
        );
    }

    #[test]
    fn serial_reuse_accepts_binding_permutations_with_per_pass_access() {
        let value = ping_pong_trace();
        value.validate_serial_buffer_reuse().unwrap();
        let mut pool = resources();
        pool.insert_allocation(AllocationRecord {
            allocation_id: AllocationId::new(10),
            owner_epoch: value.device_epoch,
            size: 4,
        })
        .unwrap();
        pool.validate_trace(&value).unwrap();
        let mut provider = capabilities();
        provider.max_passes = 8;
        provider.admit(&value, &pool).unwrap();

        let expected = compute_pass(&value, 0)
            .buffers
            .iter()
            .cloned()
            .map(|view| BufferView {
                access: BufferAccess::ReadWrite,
                ..view
            })
            .collect::<Vec<_>>();
        assert_eq!(value.serial_resources().unwrap(), expected);

        // Permutation support does not relax the cross-view alias policy.
        let mut overlapping = value;
        for pass in compute_passes_mut(&mut overlapping) {
            for view in &mut pass.buffers {
                view.allocation_id = AllocationId::new(9);
            }
        }
        overlapping.validate_serial_buffer_reuse().unwrap();
        assert!(matches!(
            pool.validate_trace(&overlapping),
            Err(ContractError::OverlappingWritableViews { .. })
        ));
    }

    #[test]
    fn serial_resource_access_combines_each_views_uses() {
        for (first, second, expected) in [
            (BufferAccess::Unused, BufferAccess::Read, BufferAccess::Read),
            (
                BufferAccess::Write,
                BufferAccess::Unused,
                BufferAccess::Write,
            ),
            (
                BufferAccess::ReadWrite,
                BufferAccess::Unused,
                BufferAccess::ReadWrite,
            ),
            (BufferAccess::Read, BufferAccess::Read, BufferAccess::Read),
            (
                BufferAccess::Write,
                BufferAccess::Write,
                BufferAccess::Write,
            ),
            (
                BufferAccess::Unused,
                BufferAccess::Unused,
                BufferAccess::Unused,
            ),
        ] {
            let mut value = ping_pong_trace();
            value.pipelines[0].contract.buffer_bindings[0].access = first;
            value.pipelines[0].contract.buffer_bindings[1].access = second;
            for pass in compute_passes_mut(&mut value) {
                pass.buffers[0].access = first;
                pass.buffers[1].access = second;
            }
            let resources = value.serial_resources().unwrap();
            assert_eq!(resources[0].access, expected);
            assert_eq!(resources[1].access, expected);
        }
    }

    #[test]
    fn serial_permutations_refuse_source_reuploads_and_missing_pipeline_bindings() {
        let value = ping_pong_trace();
        let mut reupload = value.clone();
        compute_pass_mut(&mut reupload, 1).buffers[0].source = BufferSource::OwnedBytes(vec![7; 4]);
        assert_eq!(
            reupload.serial_resources(),
            Err(ContractError::SerialBufferRebinding { pass_index: 1 })
        );
        let mut missing = value.clone();
        compute_pass_mut(&mut missing, 1).buffers.pop();
        assert_eq!(
            missing.serial_resources(),
            Err(ContractError::MissingBinding(1))
        );
        let mut duplicate = value;
        compute_pass_mut(&mut duplicate, 1).buffers[1].view_id = ViewId::new(1);
        assert_eq!(
            duplicate.serial_resources(),
            Err(ContractError::DuplicateView(ViewId::new(1)))
        );
    }

    fn subset_trace() -> ComputeTrace {
        let mut value = ping_pong_trace();
        compute_pass_mut(&mut value, 1).buffers[1] = BufferView {
            allocation_id: AllocationId::new(11),
            source: BufferSource::OwnedBytes(vec![9; 4]),
            ..buffer(3, 1)
        };
        value
    }

    fn subset_resources() -> ResourceTableSnapshot {
        let mut pool = resources();
        for allocation_id in [10, 11] {
            pool.insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(allocation_id),
                owner_epoch: DeviceEpoch::new(1),
                size: 4,
            })
            .unwrap();
        }
        pool
    }

    #[test]
    fn serial_subsets_precollect_later_views_with_initial_bytes_in_first_use_order() {
        let value = subset_trace();
        let pool = value.serial_resources().unwrap();
        assert_eq!(
            pool.iter()
                .map(|view| view.view_id.get())
                .collect::<Vec<_>>(),
            vec![2, 1, 3]
        );
        assert_eq!(pool[0], compute_pass(&value, 0).buffers[0]);
        assert_eq!(pool[1].access, BufferAccess::ReadWrite);
        assert_eq!(pool[2], compute_pass(&value, 1).buffers[1]);
        assert_eq!(pool[2].source, BufferSource::OwnedBytes(vec![9; 4]));
        let mut provider = capabilities();
        provider.max_passes = 8;
        let admitted = provider
            .validate_trace(value.clone(), subset_resources())
            .unwrap();
        assert_eq!(admitted.trace(), &value);
    }

    #[test]
    fn serial_subsets_can_omit_views_when_later_pipeline_requires_fewer_bindings() {
        let mut value = ping_pong_trace();
        let mut read_only = value.pipelines[0].clone();
        read_only.pipeline_id = PipelineId::new(5);
        read_only.contract.buffer_bindings.truncate(1);
        value.pipelines.push(read_only);
        compute_pass_mut(&mut value, 1).pipeline = PipelineId::new(5);
        compute_pass_mut(&mut value, 1).buffers.truncate(1);
        let pool = value.serial_resources().unwrap();
        assert_eq!(pool.len(), 2);
        assert_eq!(pool[0].access, BufferAccess::Read);
        assert_eq!(pool[1].access, BufferAccess::ReadWrite);
        let mut provider = capabilities();
        provider.max_passes = 8;
        provider.admit(&value, &subset_resources()).unwrap();

        // Omitting A is legal for this pipeline; omitting its required B is not.
        compute_pass_mut(&mut value, 1).buffers.clear();
        assert_eq!(
            value.serial_resources(),
            Err(ContractError::MissingBinding(0))
        );
    }

    #[test]
    fn serial_subsets_refuse_changed_views_after_an_unbound_pass() {
        let mut value = subset_trace();
        value
            .passes
            .push(TracePass::Compute(compute_pass(&value, 0).clone()));
        value.validate_serial_buffer_reuse().unwrap();
        let initial = compute_pass(&value, 2).buffers[0].clone();
        for changed in [
            BufferView {
                source: BufferSource::OwnedBytes(vec![8; 4]),
                ..initial.clone()
            },
            BufferView {
                offset: 4,
                ..initial.clone()
            },
            BufferView {
                length: 3,
                source: BufferSource::OwnedBytes(vec![7; 3]),
                ..initial.clone()
            },
            BufferView {
                allocation_id: AllocationId::new(12),
                ..initial
            },
        ] {
            let mut invalid = value.clone();
            compute_pass_mut(&mut invalid, 2).buffers[0] = changed;
            assert_eq!(
                invalid.serial_resources(),
                Err(ContractError::SerialBufferRebinding { pass_index: 2 })
            );
        }
        // A view first introduced in pass 2 is also frozen before submission.
        value
            .passes
            .push(TracePass::Compute(compute_pass(&value, 1).clone()));
        compute_pass_mut(&mut value, 3).buffers[1].source = BufferSource::OwnedBytes(vec![10; 4]);
        assert_eq!(
            value.serial_resources(),
            Err(ContractError::SerialBufferRebinding { pass_index: 3 })
        );
    }

    #[test]
    fn serial_subsets_allow_duplicate_first_use_binding_labels_but_not_aliases() {
        let mut value = subset_trace();
        let pool = value.serial_resources().unwrap();
        assert_eq!(pool[1].metal_binding, pool[2].metal_binding);
        assert_ne!(pool[1].view_id, pool[2].view_id);
        subset_resources().validate_trace(&value).unwrap();

        // C may not overlap A's backing even though A is unbound in pass 2.
        compute_pass_mut(&mut value, 1).buffers[1].allocation_id = AllocationId::new(10);
        assert_eq!(
            subset_resources().validate_trace(&value),
            Err(ContractError::OverlappingWritableViews {
                first: ViewId::new(2),
                second: ViewId::new(3),
                first_pass: 0,
                second_pass: 1,
            })
        );
    }

    #[test]
    fn serial_resource_union_has_a_shared_64_view_limit_including_single_pass() {
        for pass_count in [1, 2] {
            for count in [MAX_SERIAL_RESOURCES, MAX_SERIAL_RESOURCES + 1] {
                let mut value = trace(Vec::new());
                let template = value.pipelines.pop().unwrap();
                let mut pool = ResourceTableSnapshot::new();
                let mut next_view = 1_u64;
                for pass_index in 0..pass_count {
                    let mut pipeline = template.clone();
                    pipeline.pipeline_id = PipelineId::new(4 + pass_index as u64);
                    let binding = pipeline.contract.buffer_bindings[0].clone();
                    pipeline.contract.buffer_bindings.clear();
                    let pass_views = if pass_index + 1 == pass_count {
                        count - (next_view as usize - 1)
                    } else {
                        MAX_SERIAL_RESOURCES / 2
                    };
                    let mut buffers = Vec::new();
                    for metal_binding in 0..pass_views as u32 {
                        let view = BufferView {
                            allocation_id: AllocationId::new(next_view),
                            ..buffer(next_view, metal_binding)
                        };
                        pool.insert_allocation(AllocationRecord {
                            allocation_id: view.allocation_id,
                            owner_epoch: value.device_epoch,
                            size: 4,
                        })
                        .unwrap();
                        pipeline
                            .contract
                            .buffer_bindings
                            .push(BufferBindingContract {
                                metal_binding,
                                ..binding.clone()
                            });
                        buffers.push(view);
                        next_view += 1;
                    }
                    value.passes.push(TracePass::Compute(pass(
                        pipeline.pipeline_id.get(),
                        buffers,
                    )));
                    value.pipelines.push(pipeline);
                }
                let mut provider = capabilities();
                provider.max_passes = 8;
                provider.max_storage_buffer_descriptors = 128;
                if count == MAX_SERIAL_RESOURCES {
                    assert_eq!(value.serial_resources().unwrap().len(), count);
                    provider.admit(&value, &pool).unwrap();
                } else {
                    let expected = ContractError::SerialResourceLimit {
                        requested: count,
                        maximum: MAX_SERIAL_RESOURCES,
                    };
                    assert_eq!(value.serial_resources(), Err(expected.clone()));
                    let error = provider.admit(&value, &pool).unwrap_err();
                    assert_eq!(error.class, ProviderErrorClass::Args);
                    assert_eq!(error.detail, Some(expected.to_string()));
                }
            }
        }
    }

    #[test]
    fn serial_subsets_require_later_allocation_records_before_submission() {
        let value = subset_trace();
        let mut incomplete_pool = resources();
        incomplete_pool
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(10),
                owner_epoch: value.device_epoch,
                size: 4,
            })
            .unwrap();
        let mut provider = capabilities();
        provider.max_passes = 8;
        let error = provider.admit(&value, &incomplete_pool).unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Resource);
        assert_eq!(
            error.detail,
            Some(ContractError::UnknownAllocation(AllocationId::new(11)).to_string())
        );
    }

    #[test]
    fn serial_reuse_preserves_single_pass_dispatch_and_validates_structure_first() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        value.encoder_dispatch_type = DispatchType::Concurrent;
        value.validate_serial_buffer_reuse().unwrap();
        value.passes.push(value.passes[0].clone());
        assert_eq!(
            value.validate_serial_buffer_reuse(),
            Err(ContractError::ConcurrentPassesUnsupported)
        );
        value.passes.clear();
        assert_eq!(
            value.validate_serial_buffer_reuse(),
            Err(ContractError::EmptyTrace)
        );
    }

    #[test]
    fn trace_rejects_unknown_pipelines_and_duplicate_bindings() {
        let mixed = trace(vec![
            pass(4, vec![buffer(1, 0)]),
            pass(5, vec![buffer(2, 0)]),
        ]);
        assert_eq!(
            mixed.validate(),
            Err(ContractError::UnknownPipeline(PipelineId::new(5)))
        );

        let duplicate = trace(vec![pass(4, vec![buffer(1, 0), buffer(2, 0)])]);
        assert_eq!(
            duplicate.validate(),
            Err(ContractError::DuplicateBinding(0))
        );
    }

    #[test]
    fn pipeline_table_requires_unique_used_metadata_from_the_trace_epoch() {
        let value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        assert_eq!(
            value.pipeline(PipelineId::new(4)).unwrap(),
            &value.pipelines[0]
        );

        let mut empty = value.clone();
        empty.pipelines.clear();
        assert_eq!(empty.validate(), Err(ContractError::EmptyPipelineTable));

        let mut zero = value.clone();
        zero.pipelines[0].pipeline_id = PipelineId::new(0);
        assert_eq!(
            zero.validate(),
            Err(ContractError::InvalidIdentity("pipeline id"))
        );

        let mut foreign = value.clone();
        foreign.pipelines[0].device_epoch = DeviceEpoch::new(2);
        assert_eq!(
            foreign.validate(),
            Err(ContractError::PipelineEpochMismatch {
                pipeline: PipelineId::new(4),
                expected: value.device_epoch,
                actual: DeviceEpoch::new(2),
            })
        );

        let mut duplicate = value.clone();
        duplicate.pipelines.push(duplicate.pipelines[0].clone());
        assert_eq!(
            duplicate.validate(),
            Err(ContractError::DuplicatePipeline(PipelineId::new(4)))
        );

        let mut unused = value;
        let mut second = unused.pipelines[0].clone();
        second.pipeline_id = PipelineId::new(5);
        unused.pipelines.push(second);
        assert_eq!(
            unused.validate(),
            Err(ContractError::UnusedPipeline(PipelineId::new(5)))
        );
        unused.pipelines[1].function.entry_name.clear();
        assert_eq!(
            unused.validate(),
            Err(ContractError::EmptyField("function entry name"))
        );
        unused.pipelines[1].function.entry_name = "other".into();
        unused.pipelines[1].contract.push_constant_offset = 1;
        assert_eq!(
            unused.validate(),
            Err(ContractError::MisalignedPushConstantOffset(1))
        );
    }

    fn mixed_trace() -> ComputeTrace {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)]); 2]);
        let mut second = value.pipelines[0].clone();
        second.pipeline_id = PipelineId::new(5);
        second.function.entry_name = "second_kernel".into();
        value.pipelines.push(second);
        compute_pass_mut(&mut value, 1).pipeline = PipelineId::new(5);
        value
    }

    #[test]
    fn mixed_pipelines_validate_each_pass_grid_local_size_and_binding_contract() {
        let mut value = mixed_trace();
        value.pipelines[0].contract.fixed_grid = Some([10, 3, 1]);
        value.pipelines[0].contract.required_local_size = Some([8, 2, 1]);
        value.pipelines[1].contract.fixed_grid = Some([3, 2, 1]);
        value.pipelines[1].contract.required_local_size = Some([2, 1, 1]);
        value.pipelines[1].contract.buffer_bindings[0].metal_binding = 7;
        value.pipelines[1].contract.buffer_bindings[0].access = BufferAccess::Read;
        compute_pass_mut(&mut value, 1).dispatch.grid = [3, 2, 1];
        compute_pass_mut(&mut value, 1)
            .dispatch
            .threads_per_threadgroup = [2, 1, 1];
        compute_pass_mut(&mut value, 1).buffers[0].metal_binding = 7;
        compute_pass_mut(&mut value, 1).buffers[0].access = BufferAccess::Read;
        value.validate_serial_buffer_reuse().unwrap();
        assert_eq!(
            value.serial_resources().unwrap()[0].access,
            BufferAccess::ReadWrite
        );

        let mut provider = capabilities();
        provider.max_passes = 8;
        provider.admit(&value, &resources()).unwrap();

        let mut wrong_grid = value.clone();
        compute_pass_mut(&mut wrong_grid, 1).dispatch.grid = compute_pass(&value, 0).dispatch.grid;
        assert!(matches!(
            wrong_grid.validate(),
            Err(ContractError::GridMismatch {
                expected: [3, 2, 1],
                ..
            })
        ));
        let mut wrong_local = value.clone();
        compute_pass_mut(&mut wrong_local, 1)
            .dispatch
            .threads_per_threadgroup = compute_pass(&value, 0).dispatch.threads_per_threadgroup;
        assert!(matches!(
            wrong_local.validate(),
            Err(ContractError::LocalSizeMismatch {
                expected: [2, 1, 1],
                ..
            })
        ));
        let mut wrong_access = value.clone();
        compute_pass_mut(&mut wrong_access, 1).buffers[0].access = BufferAccess::Write;
        assert!(matches!(
            wrong_access.validate(),
            Err(ContractError::AccessMismatch {
                binding: 7,
                expected: BufferAccess::Read,
                ..
            })
        ));
        value.encoder_dispatch_type = DispatchType::Concurrent;
        assert_eq!(
            value.validate_serial_buffer_reuse(),
            Err(ContractError::ConcurrentPassesUnsupported)
        );
    }

    #[test]
    fn capabilities_check_later_pipeline_dispatch_push_range_and_footprint() {
        let value = mixed_trace();
        let mut provider = capabilities();
        provider.max_passes = 8;
        provider.admit(&value, &resources()).unwrap();

        let mut dispatch = value.clone();
        dispatch.pipelines[1].contract.dispatch_kind = DispatchKind::Threadgroups;
        compute_pass_mut(&mut dispatch, 1).dispatch.kind = DispatchKind::Threadgroups;
        assert_eq!(
            provider.admit(&dispatch, &resources()).unwrap_err().slug,
            "dispatch_kind_unsupported"
        );

        let mut push = value.clone();
        push.pipelines[1].contract.push_constant_offset = 128;
        push.pipelines[1].contract.push_constant_bytes = 4;
        assert_eq!(
            provider.admit(&push, &resources()).unwrap_err().slug,
            "push_constant_range_limit"
        );

        for footprint in [
            FootprintProof::Static { max_bytes: 8 },
            FootprintProof::Affine {
                accesses: vec![AffineAccess {
                    base_offset: 4,
                    access_size: 4,
                    terms: Vec::new(),
                }],
            },
        ] {
            let mut too_wide = value.clone();
            too_wide.pipelines[1].contract.buffer_bindings[0].footprint = footprint;
            assert_eq!(
                provider.admit(&too_wide, &resources()).unwrap_err().slug,
                "buffer_footprint_exceeds_view"
            );
        }
    }

    #[test]
    fn trace_rejects_missing_unknown_and_mismatched_access_bindings() {
        let missing = trace(vec![pass(4, Vec::new())]);
        assert_eq!(missing.validate(), Err(ContractError::MissingBinding(0)));

        let unknown = trace(vec![pass(4, vec![buffer(1, 0), buffer(2, 1)])]);
        assert_eq!(unknown.validate(), Err(ContractError::UnknownBinding(1)));

        let mut mismatched_trace = trace(vec![pass(4, vec![buffer(1, 0)])]);
        compute_pass_mut(&mut mismatched_trace, 0).buffers[0].access = BufferAccess::Read;
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
        compute_pass_mut(&mut value, 0).buffers[0].offset = 2;
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
        compute_pass_mut(&mut value, 0).buffers[0].allocation_id = AllocationId::new(10);
        assert!(matches!(
            value.validate_with_resources(&wrong_epoch),
            Err(ContractError::AllocationEpochMismatch { .. })
        ));
    }

    #[test]
    fn resource_snapshot_checks_lease_range_epoch_and_cross_pass_identity() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        compute_pass_mut(&mut value, 0).buffers[0] = BufferView {
            view_id: ViewId::new(1),
            metal_binding: 0,
            allocation_id: AllocationId::new(9),
            offset: 1,
            length: 2,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::BorrowedNoCopy(LeaseId::new(11)),
        };
        value.pipelines[0].contract.buffer_bindings[0].access = BufferAccess::Read;
        assert!(value.validate_with_resources(&leased_resources()).is_ok());

        let mut out_of_lease = value.clone();
        compute_pass_mut(&mut out_of_lease, 0).buffers[0].offset = 0;
        assert!(matches!(
            out_of_lease.validate_with_resources(&leased_resources()),
            Err(ContractError::LeaseRangeOutOfBounds { .. })
        ));

        let mut switched = value;
        let mut second = switched.passes[0].clone();
        second
            .as_compute_mut()
            .expect("test trace entry is a compute pass")
            .buffers[0]
            .source = BufferSource::BorrowedNoCopy(LeaseId::new(12));
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

        value.pipelines[0].contract.buffer_bindings[0].access = BufferAccess::Read;
        compute_pass_mut(&mut value, 0).buffers[0].access = BufferAccess::Read;
        compute_pass_mut(&mut value, 1).buffers[0].access = BufferAccess::Read;
        assert!(value.validate_with_resources(&resources()).is_ok());
    }

    #[test]
    fn owned_snapshot_source_length_is_part_of_the_contract() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        compute_pass_mut(&mut value, 0).buffers[0].length = 3;
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
            textures: Vec::new(),
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
        let contract = trace(Vec::new()).pipelines[0].contract.clone();
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
        assert_eq!(value.pipelines.len(), 1);
        assert_eq!(value.pipelines[0].device_epoch, value.device_epoch);
        assert_eq!(value.pipelines[0].pipeline_id, PipelineId::new(5));
        assert_eq!(value.pipelines[0].function, snapshot_function());
        assert_eq!(compute_pass(&value, 0).pipeline, PipelineId::new(5));
        assert_eq!(
            compute_pass(&value, 0).buffers[0].allocation_id,
            AllocationId::new(6)
        );
        assert_eq!(compute_pass(&value, 0).buffers[0].view_id, ViewId::new(7));
        assert_eq!(compute_pass(&value, 0).dispatch.grid, [10, 3, 1]);
    }

    #[test]
    fn snapshot_adapter_refuses_identity_and_dispatch_shape_gaps() {
        let submission = snapshot_submission();
        let contract = trace(Vec::new()).pipelines[0].contract.clone();
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
        let mut pipeline = snapshot_pipeline(trace(Vec::new()).pipelines[0].contract.clone());
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
    fn capabilities_apply_provider_pass_limit_to_serial_reuse() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)]); 2]);
        assert_eq!(
            capabilities().admit(&value, &resources()).unwrap_err().slug,
            "pass_count_limit"
        );
        let mut multi_pass = capabilities();
        multi_pass.max_passes = 8;
        multi_pass.admit(&value, &resources()).unwrap();
        value.passes.resize(8, value.passes[0].clone());
        multi_pass.admit(&value, &resources()).unwrap();
        value.passes.push(value.passes[0].clone());
        assert_eq!(
            multi_pass.admit(&value, &resources()).unwrap_err().slug,
            "pass_count_limit"
        );
    }

    #[test]
    fn capabilities_check_dispatch_and_footprint_in_each_serial_pass() {
        let mut multi_pass = capabilities();
        multi_pass.max_passes = 8;
        let value = trace(vec![pass(4, vec![buffer(1, 0)]); 8]);
        for pass_index in 0..8 {
            for (local, grid, expected_slug) in [
                ([9, 1, 1], [10, 3, 1], "dispatch_local_size_limit"),
                ([8, 8, 2], [10, 3, 1], "dispatch_invocation_limit"),
                ([8, 2, 1], [129, 3, 1], "dispatch_group_count_limit"),
            ] {
                let mut invalid = value.clone();
                compute_pass_mut(&mut invalid, pass_index)
                    .dispatch
                    .threads_per_threadgroup = local;
                compute_pass_mut(&mut invalid, pass_index).dispatch.grid = grid;
                assert_eq!(
                    multi_pass.admit(&invalid, &resources()).unwrap_err().slug,
                    expected_slug
                );
            }

            let mut invalid = value.clone();
            invalid.pipelines[0].contract.buffer_bindings[0].footprint = FootprintProof::Affine {
                accesses: vec![AffineAccess {
                    base_offset: 0,
                    access_size: 4,
                    terms: vec![AffineTerm { axis: 0, stride: 4 }],
                }],
            };
            for pass in compute_passes_mut(&mut invalid) {
                pass.dispatch.grid = [1, 1, 1];
            }
            compute_pass_mut(&mut invalid, pass_index).dispatch.grid = [2, 1, 1];
            assert_eq!(
                multi_pass.admit(&invalid, &resources()).unwrap_err().slug,
                "buffer_footprint_exceeds_view"
            );
        }
    }

    #[test]
    fn capabilities_check_serial_reuse_before_resource_handles() {
        let mut multi_pass = capabilities();
        multi_pass.max_passes = 8;
        multi_pass.supports_concurrent = true;
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)]); 2]);
        value.encoder_dispatch_type = DispatchType::Concurrent;
        let error = multi_pass
            .admit(&value, &ResourceTableSnapshot::new())
            .unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Args);
        assert_eq!(
            error.detail,
            Some(ContractError::ConcurrentPassesUnsupported.to_string())
        );
        value.encoder_dispatch_type = DispatchType::Serial;
        compute_pass_mut(&mut value, 1).buffers[0].source = BufferSource::OwnedBytes(vec![1; 4]);
        let error = multi_pass
            .admit(&value, &ResourceTableSnapshot::new())
            .unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Args);
        assert_eq!(
            error.detail,
            Some(ContractError::SerialBufferRebinding { pass_index: 1 }.to_string())
        );
    }

    #[test]
    fn capabilities_report_limits_and_aliases_structurally() {
        let mut too_wide = trace(vec![pass(4, vec![buffer(1, 0)])]);
        compute_pass_mut(&mut too_wide, 0)
            .dispatch
            .threads_per_threadgroup = [9, 1, 1];
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
        alias.pipelines[0]
            .contract
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
    fn attribute_stride_is_a_structured_capability_refusal() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        compute_pass_mut(&mut value, 0).buffers[0].attribute_stride = Some(16);
        let error = capabilities().admit(&value, &resources()).unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.slug, "buffer_attribute_stride_unsupported");
        assert_eq!(
            error.detail,
            Some(ContractError::UnsupportedAttributeStride.to_string())
        );
    }

    #[test]
    fn schema_version_is_a_structured_capability_refusal() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        value.schema_version = PROVIDER_SCHEMA_VERSION + 1;
        let error = capabilities().admit(&value, &resources()).unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.slug, "trace_schema_unsupported");
        assert_eq!(
            error.detail,
            Some(ContractError::UnsupportedSchemaVersion(PROVIDER_SCHEMA_VERSION + 1).to_string())
        );
    }

    #[test]
    fn contract_error_refusal_classes_and_slugs_are_stable() {
        let cases = [
            (
                ContractError::PipelineEpochMismatch {
                    pipeline: PipelineId::new(4),
                    expected: DeviceEpoch::new(1),
                    actual: DeviceEpoch::new(2),
                },
                ProviderErrorClass::Resource,
                "resource_contract_invalid",
            ),
            (
                ContractError::CompletionEpochMismatch {
                    expected: DeviceEpoch::new(1),
                    actual: DeviceEpoch::new(2),
                },
                ProviderErrorClass::Resource,
                "completion_epoch_mismatch",
            ),
            (
                ContractError::NullHostPointer(LeaseId::new(1)),
                ProviderErrorClass::Args,
                "borrowed_host_pointer_null",
            ),
            (
                ContractError::MissingBinding(0),
                ProviderErrorClass::Args,
                "trace_contract_invalid",
            ),
            (
                ContractError::WritebackBeforeCompletion,
                ProviderErrorClass::Args,
                "writeback_contract_invalid",
            ),
            (
                ContractError::CompletionPublishAfterTerminal(CompletionToken {
                    submission_id: SubmissionId::new(1),
                    device_epoch: DeviceEpoch::new(1),
                }),
                ProviderErrorClass::Internal,
                "completion_protocol_invalid",
            ),
        ];
        for (error, class, slug) in cases {
            let refusal = contract_error_refusal(error);
            assert_eq!(refusal.class, class, "unexpected class for {slug}");
            assert_eq!(refusal.slug, slug);
        }
    }

    #[test]
    fn capabilities_refuse_a_future_dispatch_kind_without_guessing() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        value.pipelines[0].contract.dispatch_kind = DispatchKind::Threadgroups;
        compute_pass_mut(&mut value, 0).dispatch.kind = DispatchKind::Threadgroups;
        let error = capabilities().admit(&value, &resources()).unwrap_err();
        assert_eq!(error.slug, "dispatch_kind_unsupported");
    }

    #[test]
    fn capabilities_refuse_an_unbounded_writable_footprint() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        value.pipelines[0].contract.buffer_bindings[0].footprint = FootprintProof::Unbounded;
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
        fixed.pipelines[0].contract.fixed_grid = Some([10, 3, 1]);
        assert!(capabilities().admit(&fixed, &resources()).is_ok());
        compute_pass_mut(&mut fixed, 0).dispatch.grid = [9, 3, 1];
        let error = capabilities().admit(&fixed, &resources()).unwrap_err();
        assert_eq!(
            error.detail.as_deref(),
            Some("dispatch grid mismatch: expected [10, 3, 1], received [9, 3, 1]")
        );
    }

    #[test]
    fn capabilities_bound_affine_footprints_to_the_dispatch_grid() {
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        compute_pass_mut(&mut value, 0).buffers[0].length = 40;
        compute_pass_mut(&mut value, 0).buffers[0].source = BufferSource::OwnedBytes(vec![0; 40]);
        let mut backing = resources();
        backing
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(10),
                owner_epoch: DeviceEpoch::new(1),
                size: 40,
            })
            .unwrap();
        compute_pass_mut(&mut value, 0).buffers[0].allocation_id = AllocationId::new(10);
        value.pipelines[0].contract.buffer_bindings[0].footprint = FootprintProof::Affine {
            accesses: vec![AffineAccess {
                base_offset: 0,
                access_size: 4,
                terms: vec![AffineTerm { axis: 0, stride: 4 }],
            }],
        };
        assert!(capabilities().admit(&value, &backing).is_ok());
        compute_pass_mut(&mut value, 0).buffers[0].length = 36;
        compute_pass_mut(&mut value, 0).buffers[0].source = BufferSource::OwnedBytes(vec![0; 36]);
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

    fn lease_reservation(
        lease: u64,
        allocation: u64,
        offset: u64,
        length: u64,
    ) -> LeaseReservation {
        LeaseReservation {
            lease: BufferLease {
                lease_id: LeaseId::new(lease),
                allocation_id: AllocationId::new(allocation),
                owner_epoch: DeviceEpoch::new(1),
            },
            offset,
            length,
        }
    }

    fn lease_token(submission: u64) -> CompletionToken {
        CompletionToken {
            submission_id: SubmissionId::new(submission),
            device_epoch: DeviceEpoch::new(1),
        }
    }

    #[test]
    fn lease_ledger_releases_only_after_every_bound_token_retires() {
        let mut ledger = LeaseLedger::new();
        let lease = lease_reservation(1, 2, 0, 64);
        ledger.register(lease).unwrap();
        let first = lease_token(10);
        let second = lease_token(11);
        ledger.bind(lease.lease.lease_id, first).unwrap();
        ledger.bind(lease.lease.lease_id, second).unwrap();
        assert_eq!(ledger.outstanding(lease.lease.lease_id), Some(2));
        assert_eq!(
            ledger.observe(
                first,
                CompletionDisposition::CompletedVisible { token: first }
            ),
            Ok(LeaseObservation::Retired)
        );
        assert!(!ledger.release_ready(lease.lease.lease_id));
        assert_eq!(
            ledger.observe(
                second,
                CompletionDisposition::SubmittedUnknown {
                    token: Some(second)
                }
            ),
            Ok(LeaseObservation::Pending)
        );
        assert!(!ledger.release_ready(lease.lease.lease_id));
        assert_eq!(
            ledger.observe(
                second,
                CompletionDisposition::DeviceLost {
                    token: Some(second)
                }
            ),
            Ok(LeaseObservation::Retired)
        );
        assert!(ledger.release_ready(lease.lease.lease_id));
        assert_eq!(ledger.release(lease.lease.lease_id), Some(lease));
        assert!(!ledger.contains(lease.lease.lease_id));
    }

    #[test]
    fn lease_ledger_keeps_unknown_cancelled_and_timed_out_observations_pending() {
        for disposition in [
            CompletionDisposition::Submitted {
                token: lease_token(10),
            },
            CompletionDisposition::TimedOut {
                token: lease_token(10),
            },
            CompletionDisposition::Cancelled {
                token: lease_token(10),
            },
            CompletionDisposition::SubmittedUnknown {
                token: Some(lease_token(10)),
            },
        ] {
            let mut ledger = LeaseLedger::new();
            let lease = lease_reservation(1, 2, 0, 64);
            ledger.register(lease).unwrap();
            ledger.bind(lease.lease.lease_id, lease_token(10)).unwrap();
            assert_eq!(
                ledger.observe(lease_token(10), disposition),
                Ok(LeaseObservation::Pending),
                "disposition={disposition:?}"
            );
            assert!(!ledger.release_ready(lease.lease.lease_id));
            assert_eq!(ledger.release(lease.lease.lease_id), None);
            ledger.retire(lease_token(10));
            assert!(ledger.release_ready(lease.lease.lease_id));
        }
    }

    #[test]
    fn lease_ledger_shares_one_token_across_leases_and_is_idempotent() {
        let mut ledger = LeaseLedger::new();
        let first = lease_reservation(1, 2, 0, 64);
        let second = lease_reservation(3, 4, 8, 16);
        ledger.register(first).unwrap();
        ledger.register(second).unwrap();
        let token = lease_token(10);
        ledger.bind(first.lease.lease_id, token).unwrap();
        ledger.bind(first.lease.lease_id, token).unwrap();
        ledger.bind(second.lease.lease_id, token).unwrap();
        assert_eq!(ledger.outstanding(first.lease.lease_id), Some(1));
        assert_eq!(ledger.outstanding(second.lease.lease_id), Some(1));
        ledger.retire(token);
        ledger.retire(token);
        assert_eq!(ledger.outstanding(first.lease.lease_id), Some(0));
        assert_eq!(ledger.outstanding(second.lease.lease_id), Some(0));
        assert_eq!(ledger.release_all_ready().len(), 2);
        assert!(ledger.leases.is_empty());
    }

    #[test]
    fn lease_ledger_device_loss_releases_every_lease() {
        let mut ledger = LeaseLedger::new();
        let lease = lease_reservation(1, 2, 0, 64);
        ledger.register(lease).unwrap();
        ledger.bind(lease.lease.lease_id, lease_token(10)).unwrap();
        assert!(!ledger.is_device_lost());
        ledger.device_lost();
        assert!(ledger.is_device_lost());
        assert!(!ledger.release_ready(LeaseId::new(99)));
        assert!(ledger.release_ready(lease.lease.lease_id));
        assert_eq!(ledger.outstanding(lease.lease.lease_id), Some(0));
        assert_eq!(ledger.release(lease.lease.lease_id), Some(lease));
    }

    #[test]
    fn lease_ledger_refuses_duplicate_unknown_and_mismatched_bindings() {
        let mut ledger = LeaseLedger::new();
        let lease = lease_reservation(1, 2, 0, 64);
        ledger.register(lease).unwrap();
        assert_eq!(
            ledger.register(lease),
            Err(ContractError::DuplicateLease(lease.lease.lease_id))
        );
        assert_eq!(
            ledger.bind(LeaseId::new(9), lease_token(10)),
            Err(ContractError::UnknownLease(LeaseId::new(9)))
        );
        let other = lease_token(11);
        assert_eq!(
            ledger.observe(
                lease_token(10),
                CompletionDisposition::CompletedVisible { token: other }
            ),
            Err(ContractError::InvalidSubmissionCompletion(
                CompletionDisposition::CompletedVisible { token: other }
            ))
        );
        assert_eq!(
            ledger.register(lease_reservation(0, 2, 0, 64)),
            Err(ContractError::InvalidIdentity("lease id"))
        );
        assert_eq!(
            ledger.register(lease_reservation(5, 2, 0, 0)),
            Err(ContractError::ZeroLength("lease reservation"))
        );
    }

    #[test]
    fn provider_lifecycle_keeps_admitting_while_the_budget_is_intact() {
        let mut lifecycle = ProviderLifecycle::new(3, 4096);
        assert!(lifecycle.is_usable());
        assert_eq!(lifecycle.health(), ProviderHealth::Usable);
        assert_eq!(lifecycle.state(), TerminalState::Usable);
        assert!(lifecycle.admit().is_ok());
        assert_eq!(
            lifecycle.record_abandonment(16),
            AbandonmentOutcome::Admitted
        );
        assert!(lifecycle.is_usable());
        assert!(lifecycle.admit().is_ok());
        assert_eq!(
            lifecycle.record_abandonment(16),
            AbandonmentOutcome::Admitted
        );
        assert!(lifecycle.admit().is_ok());
        assert_eq!(lifecycle.abandonment(), (2, 32));
        assert_eq!(lifecycle.state(), TerminalState::Usable);
        assert!(!lifecycle.is_poisoned());
        assert!(!lifecycle.ended_by_abandonment_budget());
        assert!(!lifecycle.ended_by_device_loss());
    }

    #[test]
    fn provider_lifecycle_refuses_new_work_after_budget_exhaustion() {
        let mut lifecycle = ProviderLifecycle::new(2, 1024);
        assert_eq!(
            lifecycle.record_abandonment(64),
            AbandonmentOutcome::Admitted
        );
        assert!(lifecycle.admit().is_ok());
        assert_eq!(
            lifecycle.record_abandonment(64),
            AbandonmentOutcome::Exhausted
        );
        assert_eq!(
            lifecycle.state(),
            TerminalState::Exhausted {
                submissions: 2,
                bytes: 128
            }
        );
        assert_eq!(lifecycle.health(), ProviderHealth::Exhausted);
        assert!(lifecycle.is_poisoned());
        assert!(lifecycle.ended_by_abandonment_budget());
        assert!(!lifecycle.ended_by_device_loss());

        let refusal = lifecycle
            .admit()
            .expect_err("exhausted instance refuses work");
        assert_eq!(refusal.reason(), TerminalRefusalReason::AbandonmentBudget);
        assert!(refusal.requires_recreate());
        let error = refusal.error();
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(error.class, ProviderErrorClass::Resource);
        assert_eq!(error.slug, "provider_unavailable");
        assert_eq!(error.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(
            error.fields.get("terminal"),
            Some(&FieldValue::Text("abandonment_budget".into()))
        );
        assert_eq!(
            error.fields.get("abandoned_submissions"),
            Some(&FieldValue::Unsigned(2))
        );
        assert_eq!(
            error.fields.get("abandoned_bytes"),
            Some(&FieldValue::Unsigned(128))
        );
    }

    #[test]
    fn provider_lifecycle_exhaustion_and_refusals_are_idempotent() {
        let mut lifecycle = ProviderLifecycle::new(1, 4096);
        assert_eq!(
            lifecycle.record_abandonment(32),
            AbandonmentOutcome::Exhausted
        );
        let first = lifecycle
            .admit()
            .expect_err("exhausted instance refuses work");
        let counters = lifecycle.abandonment();
        assert_eq!(counters, (1, 32));
        for _ in 0..3 {
            // Repeating the abandonment cannot double count or change state.
            assert_eq!(
                lifecycle.record_abandonment(32),
                AbandonmentOutcome::Exhausted
            );
            assert_eq!(lifecycle.state(), first.state());
            assert_eq!(lifecycle.abandonment(), counters);
            assert_eq!(lifecycle.admit(), Err(first.clone()));
        }
        assert_eq!(lifecycle.health(), ProviderHealth::Exhausted);
    }

    #[test]
    fn provider_lifecycle_terminates_on_the_byte_budget_before_the_count_limit() {
        let mut lifecycle = ProviderLifecycle::new(16, 64);
        assert_eq!(
            lifecycle.record_abandonment(64),
            AbandonmentOutcome::Exhausted
        );
        assert_eq!(lifecycle.abandonment(), (1, 64));
        let refusal = lifecycle.admit().expect_err("byte budget refuses work");
        assert_eq!(
            refusal.error().fields.get("abandoned_submissions"),
            Some(&FieldValue::Unsigned(1))
        );
        assert_eq!(
            refusal.error().fields.get("abandoned_bytes"),
            Some(&FieldValue::Unsigned(64))
        );
        assert!(lifecycle.ended_by_abandonment_budget());
    }

    #[test]
    fn provider_lifecycle_device_loss_retires_leases_and_refuses_work() {
        let mut lifecycle = ProviderLifecycle::new(4, 4096);
        let first = lease_reservation(1, 2, 0, 64);
        let second = lease_reservation(3, 4, 8, 16);
        lifecycle.leases_mut().register(first).unwrap();
        lifecycle.leases_mut().register(second).unwrap();
        lifecycle
            .leases_mut()
            .bind(first.lease.lease_id, lease_token(10))
            .unwrap();
        lifecycle
            .leases_mut()
            .bind(second.lease.lease_id, lease_token(11))
            .unwrap();
        assert_eq!(
            lifecycle.leases().leased(),
            vec![
                (LeaseId::new(1), TerminalLeaseState::Outstanding(1)),
                (LeaseId::new(3), TerminalLeaseState::Outstanding(1)),
            ]
        );
        assert!(lifecycle.admit().is_ok());

        lifecycle.mark_device_lost();
        assert_eq!(lifecycle.health(), ProviderHealth::DeviceLost);
        assert_eq!(lifecycle.state(), TerminalState::DeviceLost);
        assert!(lifecycle.is_poisoned());
        assert!(lifecycle.ended_by_device_loss());
        assert!(!lifecycle.ended_by_abandonment_budget());
        assert!(lifecycle.leases().is_device_lost());
        // Every in-flight lease is observable in its terminal state: no
        // dangling entry and no lease withheld behind a missing completion.
        assert_eq!(
            lifecycle.leases().leased(),
            vec![
                (LeaseId::new(1), TerminalLeaseState::Released),
                (LeaseId::new(3), TerminalLeaseState::Released),
            ]
        );
        assert_eq!(lifecycle.leases_mut().release_all_ready().len(), 2);
        assert!(lifecycle.leases().leased().is_empty());

        let refusal = lifecycle.admit().expect_err("device loss refuses work");
        assert_eq!(refusal.reason(), TerminalRefusalReason::DeviceLost);
        assert!(refusal.requires_recreate());
        let error = refusal.error();
        assert_eq!(error.class, ProviderErrorClass::DeviceLost);
        assert_eq!(error.slug, "device_lost");
        assert_eq!(error.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(
            error.completion,
            CompletionDisposition::DeviceLost { token: None }
        );
        assert_eq!(
            error.fields.get("terminal"),
            Some(&FieldValue::Text("device_lost".into()))
        );
    }

    #[test]
    fn provider_lifecycle_seals_an_unobservable_submission_without_charging_it() {
        let mut lifecycle = ProviderLifecycle::new(4, 4096);
        lifecycle.mark_unobservable_submission();
        assert!(lifecycle.is_poisoned());
        assert!(lifecycle.ended_by_abandonment_budget());
        assert_eq!(lifecycle.health(), ProviderHealth::Exhausted);
        // The ledger stays honest: no submission was recorded as abandoned.
        assert_eq!(lifecycle.abandonment(), (0, 0));

        let refusal = lifecycle
            .admit()
            .expect_err("a sealed instance refuses work");
        assert_eq!(refusal.reason(), TerminalRefusalReason::AbandonmentBudget);
        assert!(refusal.requires_recreate());
        assert_eq!(
            refusal.error().fields.get("terminal"),
            Some(&FieldValue::Text("abandonment_budget".into()))
        );
        assert_eq!(
            refusal.error().fields.get("abandoned_submissions"),
            Some(&FieldValue::Unsigned(0))
        );

        // Sealing is idempotent, recording afterwards stays a no-op, and a
        // later observed device loss still wins over the abandonment cause.
        lifecycle.mark_unobservable_submission();
        assert_eq!(
            lifecycle.state(),
            TerminalState::Exhausted {
                submissions: 0,
                bytes: 0
            }
        );
        assert_eq!(
            lifecycle.record_abandonment(64),
            AbandonmentOutcome::Exhausted
        );
        assert_eq!(lifecycle.abandonment(), (0, 0));
        lifecycle.mark_device_lost();
        assert_eq!(lifecycle.state(), TerminalState::DeviceLost);
        assert_eq!(lifecycle.abandonment(), (0, 0));
        assert_eq!(
            lifecycle
                .admit()
                .expect_err("device loss refuses work")
                .reason(),
            TerminalRefusalReason::DeviceLost
        );
    }

    #[test]
    fn provider_lifecycle_device_loss_wins_over_a_later_budget_abandonment() {
        let mut lifecycle = ProviderLifecycle::new(2, 4096);
        lifecycle.mark_device_lost();
        // A device-lost provider no longer observes completions, so recording
        // an abandonment afterwards must not relabel the terminal cause.
        assert_eq!(
            lifecycle.record_abandonment(64),
            AbandonmentOutcome::Exhausted
        );
        assert_eq!(lifecycle.state(), TerminalState::DeviceLost);
        assert_eq!(lifecycle.health(), ProviderHealth::DeviceLost);
        assert_eq!(lifecycle.abandonment(), (0, 0));
        assert_eq!(
            lifecycle
                .admit()
                .expect_err("device loss refuses work")
                .reason(),
            TerminalRefusalReason::DeviceLost
        );
    }

    #[test]
    fn lease_ledger_reports_every_lease_terminal_state_without_duplicates() {
        let mut ledger = LeaseLedger::new();
        let first = lease_reservation(1, 2, 0, 64);
        let second = lease_reservation(3, 4, 8, 16);
        ledger.register(first).unwrap();
        ledger.register(second).unwrap();
        assert_eq!(
            ledger.leased(),
            vec![
                (LeaseId::new(1), TerminalLeaseState::Released),
                (LeaseId::new(3), TerminalLeaseState::Released),
            ]
        );
        ledger.bind(first.lease.lease_id, lease_token(10)).unwrap();
        ledger.bind(second.lease.lease_id, lease_token(11)).unwrap();
        assert_eq!(
            ledger.leased(),
            vec![
                (LeaseId::new(1), TerminalLeaseState::Outstanding(1)),
                (LeaseId::new(3), TerminalLeaseState::Outstanding(1)),
            ]
        );
        ledger.retire(lease_token(10));
        // The retired lease stays observable until the owner releases it.
        assert_eq!(
            ledger.leased(),
            vec![
                (LeaseId::new(1), TerminalLeaseState::Released),
                (LeaseId::new(3), TerminalLeaseState::Outstanding(1)),
            ]
        );
        // Repeating the retirement is idempotent and creates no second entry.
        ledger.retire(lease_token(10));
        assert_eq!(ledger.leased().len(), 2);
        let mut released = ledger.release_all_ready();
        released.sort_by_key(|reservation| reservation.lease.lease_id.get());
        assert_eq!(
            released
                .iter()
                .map(|reservation| reservation.lease.lease_id)
                .collect::<Vec<_>>(),
            vec![LeaseId::new(1)]
        );
        assert_eq!(
            ledger.leased(),
            vec![(LeaseId::new(3), TerminalLeaseState::Outstanding(1))]
        );
    }

    #[test]
    fn staged_lease_requires_exact_reservation_bytes() {
        let reservation = lease_reservation(1, 2, 0, 4);
        assert!(StagedLease::new(reservation, vec![7; 4]).is_ok());
        assert_eq!(
            StagedLease::new(reservation, vec![7; 3]),
            Err(ContractError::LeaseSourceLengthMismatch {
                lease: LeaseId::new(1),
                expected: 4,
                actual: 3,
            })
        );
        assert_eq!(
            StagedLease::new(lease_reservation(0, 2, 0, 4), vec![0; 4]),
            Err(ContractError::InvalidIdentity("staged lease id"))
        );
        assert_eq!(
            StagedLease::new(lease_reservation(1, 2, 0, 0), Vec::new()),
            Err(ContractError::ZeroLength("lease reservation"))
        );
    }

    #[test]
    fn lease_registry_imports_releases_and_resolves_view_bytes() {
        let registry = LeaseRegistry::new();
        assert!(registry.is_empty());
        let reservation = lease_reservation(1, 2, 8, 16);
        let bytes: Vec<u8> = (0..16).collect();
        registry
            .import(StagedLease::new(reservation, bytes.clone()).unwrap())
            .unwrap();
        assert_eq!(registry.len(), 1);
        assert_eq!(
            registry
                .import(StagedLease::new(reservation, bytes.clone()).unwrap())
                .unwrap_err()
                .slug,
            "lease_already_imported"
        );

        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(2),
                owner_epoch: DeviceEpoch::new(1),
                size: 64,
            })
            .unwrap();
        resources.insert_lease(reservation).unwrap();

        let mut view = buffer(1, 0);
        view.allocation_id = AllocationId::new(2);
        view.offset = 12;
        view.length = 4;
        view.source = BufferSource::StagedLease(LeaseId::new(1));
        assert_eq!(
            registry
                .view_bytes(LeaseId::new(1), &view, DeviceEpoch::new(1), &resources)
                .unwrap(),
            vec![4, 5, 6, 7]
        );

        let mut outside = view.clone();
        outside.offset = 24;
        outside.length = 4;
        assert_eq!(
            registry
                .view_bytes(LeaseId::new(1), &outside, DeviceEpoch::new(1), &resources)
                .unwrap_err()
                .slug,
            "lease_range_out_of_bounds"
        );

        let mut mismatched = ResourceTableSnapshot::new();
        mismatched
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(2),
                owner_epoch: DeviceEpoch::new(1),
                size: 64,
            })
            .unwrap();
        mismatched
            .insert_lease(lease_reservation(1, 2, 4, 16))
            .unwrap();
        assert_eq!(
            registry
                .view_bytes(LeaseId::new(1), &view, DeviceEpoch::new(1), &mismatched)
                .unwrap_err()
                .slug,
            "lease_snapshot_mismatch"
        );

        let mut unadmitted = ResourceTableSnapshot::new();
        unadmitted
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(2),
                owner_epoch: DeviceEpoch::new(1),
                size: 64,
            })
            .unwrap();
        assert_eq!(
            registry
                .view_bytes(LeaseId::new(1), &view, DeviceEpoch::new(1), &unadmitted)
                .unwrap_err()
                .slug,
            "lease_not_admitted"
        );

        assert_eq!(
            registry.release(LeaseId::new(9)).unwrap_err().slug,
            "lease_not_imported"
        );
        registry.release(LeaseId::new(1)).unwrap();
        assert!(registry.is_empty());
        assert_eq!(
            registry
                .view_bytes(LeaseId::new(1), &view, DeviceEpoch::new(1), &resources)
                .unwrap_err()
                .slug,
            "lease_not_imported"
        );
    }

    #[test]
    fn lease_registry_refuses_a_foreign_owner_epoch() {
        let registry = LeaseRegistry::new();
        let reservation = LeaseReservation {
            lease: BufferLease {
                lease_id: LeaseId::new(3),
                allocation_id: AllocationId::new(4),
                owner_epoch: DeviceEpoch::new(2),
            },
            offset: 0,
            length: 4,
        };
        registry
            .import(StagedLease::new(reservation, vec![0; 4]).unwrap())
            .unwrap();
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(4),
                owner_epoch: DeviceEpoch::new(2),
                size: 8,
            })
            .unwrap();
        resources.insert_lease(reservation).unwrap();
        let mut view = buffer(1, 0);
        view.allocation_id = AllocationId::new(4);
        view.length = 4;
        view.source = BufferSource::StagedLease(LeaseId::new(3));
        let error = registry
            .view_bytes(LeaseId::new(3), &view, DeviceEpoch::new(1), &resources)
            .unwrap_err();
        assert_eq!(error.slug, "lease_epoch_mismatch");
        assert_eq!(error.fields.get("expected"), Some(&FieldValue::Unsigned(1)));
        assert_eq!(error.fields.get("actual"), Some(&FieldValue::Unsigned(2)));
    }

    #[test]
    fn borrowed_lease_validates_identity_pointer_and_range() {
        let reservation = lease_reservation(1, 2, 8, 16);
        assert!(BorrowedLease::new(reservation, 0x4000).is_ok());
        assert_eq!(
            BorrowedLease::new(reservation, 0),
            Err(ContractError::NullHostPointer(LeaseId::new(1)))
        );
        assert_eq!(
            BorrowedLease::new(lease_reservation(0, 2, 8, 16), 0x4000),
            Err(ContractError::InvalidIdentity("borrowed lease id"))
        );
        assert_eq!(
            BorrowedLease::new(lease_reservation(1, 2, 8, 0), 0x4000),
            Err(ContractError::ZeroLength("lease reservation"))
        );
        assert_eq!(
            BorrowedLease::new(reservation, usize::MAX),
            Err(ContractError::ArithmeticOverflow("borrowed lease range"))
        );
    }

    #[test]
    fn borrowed_lease_registry_resolves_views_and_gates_release() {
        let registry = BorrowedLeaseRegistry::new();
        assert!(registry.is_empty());
        let reservation = lease_reservation(1, 2, 8, 16);
        let borrowed = BorrowedLease::new(reservation, 0x4000).unwrap();
        registry.import(borrowed).unwrap();
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.outstanding(LeaseId::new(1)), Some(0));
        assert_eq!(
            registry.import(borrowed).unwrap_err().slug,
            "lease_already_imported"
        );

        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(2),
                owner_epoch: DeviceEpoch::new(1),
                size: 64,
            })
            .unwrap();
        resources.insert_lease(reservation).unwrap();

        let mut view = buffer(1, 0);
        view.allocation_id = AllocationId::new(2);
        view.offset = 12;
        view.length = 4;
        view.source = BufferSource::BorrowedNoCopy(LeaseId::new(1));
        assert_eq!(
            registry
                .view_pointer(LeaseId::new(1), &view, DeviceEpoch::new(1), &resources)
                .unwrap(),
            BorrowedView {
                pointer: 0x4004,
                len: 4,
                base_pointer: 0x4000,
                base_len: 16,
                offset: 4,
                capacity: 12,
            }
        );

        let mut outside = view.clone();
        outside.offset = 24;
        assert_eq!(
            registry
                .view_pointer(LeaseId::new(1), &outside, DeviceEpoch::new(1), &resources)
                .unwrap_err()
                .slug,
            "lease_range_out_of_bounds"
        );
        assert_eq!(
            registry
                .view_pointer(LeaseId::new(1), &view, DeviceEpoch::new(2), &resources)
                .unwrap_err()
                .slug,
            "lease_epoch_mismatch"
        );

        registry.retain(LeaseId::new(1)).unwrap();
        assert_eq!(registry.outstanding(LeaseId::new(1)), Some(1));
        let in_use = registry.release(LeaseId::new(1)).unwrap_err();
        assert_eq!(in_use.slug, "lease_in_use");
        assert_eq!(
            in_use.fields.get("outstanding"),
            Some(&FieldValue::Unsigned(1))
        );
        registry.retain_all(&[LeaseId::new(1)]).unwrap();
        assert_eq!(registry.outstanding(LeaseId::new(1)), Some(2));
        registry.retire(LeaseId::new(1));
        registry.retire_all(&[LeaseId::new(1)]);
        assert_eq!(registry.outstanding(LeaseId::new(1)), Some(0));
        registry.release(LeaseId::new(1)).unwrap();
        assert!(registry.is_empty());
        assert_eq!(
            registry
                .view_pointer(LeaseId::new(1), &view, DeviceEpoch::new(1), &resources)
                .unwrap_err()
                .slug,
            "lease_not_imported"
        );
        assert_eq!(
            registry.retain(LeaseId::new(9)).unwrap_err().slug,
            "lease_not_imported"
        );
        assert_eq!(
            registry.release(LeaseId::new(9)).unwrap_err().slug,
            "lease_not_imported"
        );
    }

    #[test]
    fn retirement_evidence_matches_the_provider_contract() {
        let token = lease_token(10);
        for disposition in [
            CompletionDisposition::NotSubmitted,
            CompletionDisposition::CompletedVisible { token },
            CompletionDisposition::Failed { token: Some(token) },
            CompletionDisposition::DeviceLost { token: Some(token) },
        ] {
            assert!(
                disposition_retires_resources(disposition),
                "disposition={disposition:?}"
            );
        }
        for disposition in [
            CompletionDisposition::Submitted { token },
            CompletionDisposition::TimedOut { token },
            CompletionDisposition::Cancelled { token },
            CompletionDisposition::SubmittedUnknown { token: Some(token) },
        ] {
            assert!(
                !disposition_retires_resources(disposition),
                "disposition={disposition:?}"
            );
        }
    }

    #[test]
    fn completion_distinguishes_timeout_from_terminal_unknown() {
        let token = CompletionToken {
            submission_id: SubmissionId::new(12),
            device_epoch: DeviceEpoch::new(1),
        };
        assert!(!CompletionDisposition::TimedOut { token }.is_terminal());
        assert!(CompletionDisposition::Cancelled { token }.is_terminal());
        assert!(CompletionDisposition::SubmittedUnknown { token: Some(token) }.is_terminal());
        assert_eq!(
            CompletionDisposition::Cancelled { token }.token(),
            Some(token)
        );
        assert!(CompletionDisposition::Cancelled { token }
            .validate()
            .is_ok());
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
        let mut writebacks = compute_pass(trace, 0)
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
    fn submission_requires_matching_epoch_and_accepts_serial_reuse() {
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
        let final_result = completed_submission(&multi_pass);
        assert_eq!(final_result.writebacks.len(), 1);
        final_result.validate_for_trace(&multi_pass).unwrap();
        compute_pass_mut(&mut multi_pass, 1).buffers[0].view_id = ViewId::new(8);
        assert_eq!(
            final_result.validate_for_trace(&multi_pass),
            Err(ContractError::MissingWriteback {
                allocation: AllocationId::new(9),
                view: ViewId::new(8),
            })
        );
    }

    #[test]
    fn completion_readback_requires_visible_completion_and_exact_trace() {
        let trace = trace(vec![pass(4, vec![buffer(7, 0)])]);
        let submission = completed_submission(&trace);
        let readback = CompletionReadback {
            completion: submission.completion,
            writebacks: submission.writebacks,
        };
        readback.validate_for_trace(&trace).unwrap();

        let mut submitted = readback.clone();
        submitted.completion = CompletionDisposition::Submitted {
            token: submission.completion.token().unwrap(),
        };
        assert!(matches!(
            submitted.validate(),
            Err(ContractError::InvalidReadbackCompletion(
                CompletionDisposition::Submitted { .. }
            ))
        ));

        let mut incomplete = readback.clone();
        incomplete.writebacks.clear();
        assert_eq!(
            incomplete.validate_for_trace(&trace),
            Err(ContractError::MissingWriteback {
                allocation: AllocationId::new(9),
                view: ViewId::new(7),
            })
        );
    }

    #[test]
    fn serial_submission_requires_one_complete_final_writeback_per_writable_view() {
        let value = trace(vec![pass(4, vec![buffer(7, 0)]); 8]);
        let complete = completed_submission(&value);
        complete.validate_for_trace(&value).unwrap();
        let mut duplicate = complete.clone();
        duplicate.writebacks.push(duplicate.writebacks[0].clone());
        assert!(matches!(
            duplicate.validate_for_trace(&value),
            Err(ContractError::DuplicateWriteback { .. })
        ));
        let mut missing = complete.clone();
        missing.writebacks.clear();
        assert!(matches!(
            missing.validate_for_trace(&value),
            Err(ContractError::MissingWriteback { .. })
        ));
        let mut partial = complete;
        partial.writebacks[0].bytes.pop();
        assert_eq!(
            partial.validate_for_trace(&value),
            Err(ContractError::IncompleteWriteback(ViewId::new(7)))
        );
    }

    #[test]
    fn serial_submission_includes_views_written_only_in_later_passes() {
        let value = ping_pong_trace();
        // The old first-pass policy returns only B, which misses A's final data.
        let mut result = completed_submission(&value);
        assert_eq!(result.writebacks.len(), 1);
        assert_eq!(result.writebacks[0].view_id, ViewId::new(1));
        assert_eq!(
            result.validate_for_trace(&value),
            Err(ContractError::MissingWriteback {
                allocation: AllocationId::new(10),
                view: ViewId::new(2),
            })
        );
        result.writebacks.push(BufferWriteback {
            view_id: ViewId::new(2),
            allocation_id: AllocationId::new(10),
            offset: 0,
            bytes: vec![7; 4],
        });
        result.validate_for_trace(&value).unwrap();
        result.writebacks.remove(0);
        assert_eq!(
            result.validate_for_trace(&value),
            Err(ContractError::MissingWriteback {
                allocation: AllocationId::new(9),
                view: ViewId::new(1),
            })
        );
    }

    #[test]
    fn serial_submission_requires_final_writeback_for_a_view_first_bound_later() {
        let value = subset_trace();
        let mut result = completed_submission(&value);
        assert_eq!(result.writebacks.len(), 1);
        assert_eq!(
            result.validate_for_trace(&value),
            Err(ContractError::MissingWriteback {
                allocation: AllocationId::new(11),
                view: ViewId::new(3),
            })
        );
        result.writebacks.push(BufferWriteback {
            view_id: ViewId::new(3),
            allocation_id: AllocationId::new(11),
            offset: 0,
            bytes: vec![0x5a; 4],
        });
        result.validate_for_trace(&value).unwrap();
        result.writebacks.remove(0);
        assert_eq!(
            result.validate_for_trace(&value),
            Err(ContractError::MissingWriteback {
                allocation: AllocationId::new(9),
                view: ViewId::new(1),
            })
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
            compute_pass_mut(&mut read_only, 0).buffers[0].access = access;
            read_only.pipelines[0].contract.buffer_bindings[0].access = access;
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
        let second_binding = BufferBindingContract {
            metal_binding: 1,
            ..trace.pipelines[0].contract.buffer_bindings[0].clone()
        };
        trace.pipelines[0]
            .contract
            .buffer_bindings
            .push(second_binding);
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

    #[test]
    fn buffer_ranges_are_half_open_and_allocation_scoped() {
        let allocation = AllocationId::new(7);
        let other = AllocationId::new(8);
        let range = BufferRange::new(allocation, 10, 10);
        assert_eq!(range.end(), Ok(20));
        assert!(range.overlaps(&BufferRange::new(allocation, 0, 11)));
        assert!(range.overlaps(&BufferRange::new(allocation, 19, 1)));
        assert!(!range.overlaps(&BufferRange::new(allocation, 20, 4)));
        assert!(!range.overlaps(&BufferRange::new(allocation, 0, 10)));
        assert!(!range.overlaps(&BufferRange::new(other, 10, 10)));
    }

    #[test]
    fn empty_ranges_conflict_with_nothing_but_overflow_fails_closed() {
        let allocation = AllocationId::new(9);
        let empty = BufferRange::new(allocation, 4, 0);
        assert_eq!(empty.end(), Ok(4));
        assert!(!empty.overlaps(&BufferRange::new(allocation, 0, 8)));
        assert!(!BufferRange::new(allocation, 0, 8).overlaps(&empty));
        let overflowing = BufferRange::new(allocation, u64::MAX, 2);
        assert_eq!(
            overflowing.end(),
            Err(ContractError::ArithmeticOverflow("buffer range"))
        );
        assert!(overflowing.overlaps(&BufferRange::new(allocation, 0, 8)));
        assert!(BufferRange::new(allocation, 0, 8).overlaps(&overflowing));
    }

    #[test]
    fn range_set_conflicts_only_when_a_write_is_involved() {
        let allocation = AllocationId::new(3);
        let write = BufferRange::new(allocation, 16, 32);
        let read = BufferRange::new(allocation, 40, 8);
        let disjoint = BufferRange::new(allocation, 48, 8);

        let writer = RangeSet::new().with_write(write);
        let reader = RangeSet::new().with_read(read);
        let disjoint_reader = RangeSet::new().with_read(disjoint);
        let disjoint_writer = RangeSet::new().with_write(disjoint);

        assert!(writer.conflicts_with(&reader));
        assert!(reader.conflicts_with(&writer));
        assert!(writer.conflicts_with(&writer.clone()));
        assert!(!reader.conflicts_with(&RangeSet::new().with_read(read)));
        assert!(!reader.conflicts_with(&disjoint_reader));
        assert!(!writer.conflicts_with(&disjoint_writer));
        assert!(!RangeSet::new().conflicts_with(&writer));
    }

    #[test]
    fn canonical_order_sorts_ranges_by_allocation_offset_and_length() {
        let mut ranges = vec![
            BufferRange::new(AllocationId::new(2), 0, 4),
            BufferRange::new(AllocationId::new(1), 8, 4),
            BufferRange::new(AllocationId::new(1), 4, 8),
            BufferRange::new(AllocationId::new(1), 4, 4),
        ];
        sort_canonical(&mut ranges);
        let keys = ranges
            .iter()
            .map(|range| (range.allocation_id.get(), range.offset, range.length))
            .collect::<Vec<_>>();
        assert_eq!(keys, vec![(1, 4, 4), (1, 4, 8), (1, 8, 4), (2, 0, 4)]);
    }

    fn ranged_buffer(view_id: u64, offset: u64, length: u64) -> BufferView {
        BufferView {
            offset,
            length,
            source: BufferSource::OwnedBytes(vec![0; usize::try_from(length).unwrap()]),
            ..buffer(view_id, 0)
        }
    }

    fn sized_resources(size: u64) -> ResourceTableSnapshot {
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(9),
                owner_epoch: DeviceEpoch::new(1),
                size,
            })
            .unwrap();
        resources
    }

    fn ranged_capabilities(alias_mode: AliasMode) -> ProviderCapabilities {
        ProviderCapabilities {
            max_passes: 8,
            alias_mode,
            ..capabilities()
        }
    }

    #[test]
    fn distinct_view_ranges_are_admitted_only_while_disjoint() {
        let mut value = trace(vec![
            pass(4, vec![ranged_buffer(1, 0, 4)]),
            pass(4, vec![ranged_buffer(2, 4, 4)]),
        ]);
        let resources = sized_resources(16);
        let distinct = ranged_capabilities(AliasMode::DistinctViews);
        assert!(distinct.admit(&value, &resources).is_ok());

        // A refusing provider keeps refusing every second view of an allocation.
        assert_eq!(
            ranged_capabilities(AliasMode::Refused)
                .admit(&value, &resources)
                .unwrap_err()
                .slug,
            "buffer_alias_unsupported"
        );

        // Overlapping ranges keep the alias refusal even for read-read pairs,
        // which stay refused conservatively in the first version.
        compute_pass_mut(&mut value, 1).buffers[0] = ranged_buffer(2, 2, 4);
        assert_eq!(
            distinct.admit(&value, &resources).unwrap_err().slug,
            "buffer_alias_unsupported"
        );
        for pass in compute_passes_mut(&mut value) {
            pass.buffers[0].access = BufferAccess::Read;
        }
        value.pipelines[0].contract.buffer_bindings[0].access = BufferAccess::Read;
        assert_eq!(
            distinct.admit(&value, &resources).unwrap_err().slug,
            "buffer_alias_unsupported"
        );

        // The same logical view reused across passes is not an alias.
        compute_pass_mut(&mut value, 1).buffers[0] = ranged_buffer(1, 0, 4);
        for pass in compute_passes_mut(&mut value) {
            pass.buffers[0].access = BufferAccess::Read;
        }
        assert!(distinct.admit(&value, &resources).is_ok());
    }

    #[test]
    fn queue_priority_defaults_to_the_middle_tier() {
        assert_eq!(QueuePriority::default(), QueuePriority::Default);
        assert_eq!(QueuePriority::Low.rank(), 0);
        assert_eq!(QueuePriority::Default.rank(), 1);
        assert_eq!(QueuePriority::High.rank(), 2);
    }

    #[test]
    fn queue_priority_ranks_round_trip_and_refuse_unknown_values() {
        for tier in [
            QueuePriority::Low,
            QueuePriority::Default,
            QueuePriority::High,
        ] {
            assert_eq!(QueuePriority::from_rank(tier.rank()), Some(tier));
        }
        // A rank from a newer peer is refused instead of aging into `Default`:
        // guessing a tier would change a queue nobody marked.
        assert_eq!(QueuePriority::from_rank(3), None);
        assert_eq!(QueuePriority::from_rank(u8::MAX), None);
    }

    #[test]
    fn a_queue_priority_marking_ages_into_a_device_table() {
        // No marking: the provider keeps the all-`Default` table, i.e. the
        // scheduling behaviour that predates the tier table.
        assert_eq!(
            queue_priorities_for_device(3, &[]),
            vec![QueuePriority::Default; 3]
        );
        // A shorter marking is padded, a longer one is truncated, so the
        // expansion never depends on the owner knowing the queue count.
        assert_eq!(
            queue_priorities_for_device(3, &[QueuePriority::High]),
            vec![
                QueuePriority::High,
                QueuePriority::Default,
                QueuePriority::Default
            ]
        );
        assert_eq!(
            queue_priorities_for_device(
                2,
                &[QueuePriority::High, QueuePriority::Low, QueuePriority::High]
            ),
            vec![QueuePriority::High, QueuePriority::Low]
        );
        // The expansion is also the table `select_queue_with_priority` reads,
        // so a marking changes which queue a submit picks and nothing else:
        // both queues stay idle in every case below.
        let policy = QueueSchedulingPolicy::default();
        let marked = queue_priorities_for_device(2, &[QueuePriority::Low, QueuePriority::High]);
        let unmarked = queue_priorities_for_device(2, &[]);
        // Slot 0 of the window nominates `High`, so the marked queue wins where
        // the unmarked table would have kept its rotation on queue 0...
        assert_eq!(select_queue_with_priority(&[0, 0], &marked, 0, policy), 1);
        assert_eq!(select_queue_with_priority(&[0, 0], &unmarked, 0, policy), 0);
        // ...and slot 6 nominates `Low`, which only the unmarked queue matches.
        assert_eq!(select_queue_with_priority(&[0, 0], &marked, 6, policy), 0);
    }

    #[test]
    fn queue_scheduling_policy_window_and_streak_limit_are_explicit() {
        let policy = QueueSchedulingPolicy::default();
        assert_eq!(policy.low_weight(), 1);
        assert_eq!(policy.medium_weight(), 2);
        assert_eq!(policy.high_weight(), 4);
        assert_eq!(policy.weight(QueuePriority::Low), 1);
        assert_eq!(policy.weight(QueuePriority::Default), 2);
        assert_eq!(policy.weight(QueuePriority::High), 4);
        assert_eq!(policy.window(), 7);
        assert_eq!(policy.high_priority_streak_limit(), 4);

        // A zero weight is clamped to one, so no tier can own an empty window.
        let clamped = QueueSchedulingPolicy::new(0, 0, 0);
        assert_eq!(clamped.window(), 3);
        assert_eq!(clamped.nominated_priority(0), QueuePriority::High);
        assert_eq!(clamped.nominated_priority(1), QueuePriority::Default);
        assert_eq!(clamped.nominated_priority(2), QueuePriority::Low);
        // The window is periodic in the caller's monotonic counter.
        assert_eq!(clamped.nominated_priority(3), QueuePriority::High);
        assert_eq!(policy.nominated_priority(7), policy.nominated_priority(0));
    }

    #[test]
    fn priority_selection_prefers_idle_queues_before_higher_tiers() {
        let policy = QueueSchedulingPolicy::default();

        // A busy high-priority queue yields to an idle low-priority queue.
        assert_eq!(
            select_queue_with_priority(
                &[1, 0],
                &[QueuePriority::High, QueuePriority::Low],
                0,
                policy
            ),
            1
        );
        assert_eq!(
            select_queue_with_priority(
                &[0, 1],
                &[QueuePriority::Low, QueuePriority::High],
                2,
                policy
            ),
            0
        );
        // When every queue is busy the least-loaded one still wins.
        assert_eq!(
            select_queue_with_priority(
                &[3, 1],
                &[QueuePriority::High, QueuePriority::Low],
                0,
                policy
            ),
            1
        );
    }

    #[test]
    fn priority_selection_prefers_the_higher_tier_when_loads_tie() {
        let policy = QueueSchedulingPolicy::default();

        // Slot 0 of the window nominates `High`.
        assert_eq!(
            select_queue_with_priority(
                &[0, 0],
                &[QueuePriority::Low, QueuePriority::High],
                0,
                policy
            ),
            1
        );
        assert_eq!(
            select_queue_with_priority(
                &[0, 0, 0],
                &[
                    QueuePriority::Default,
                    QueuePriority::Low,
                    QueuePriority::High
                ],
                1,
                policy
            ),
            2
        );
        // Inside one tier the counter still rotates the ties.
        assert_eq!(
            select_queue_with_priority(
                &[0, 0],
                &[QueuePriority::High, QueuePriority::High],
                1,
                policy
            ),
            1
        );
    }

    #[test]
    fn priority_selection_yields_to_lower_tiers_without_starvation() {
        let policy = QueueSchedulingPolicy::default();
        let priorities = [QueuePriority::High, QueuePriority::Low];
        let picks: Vec<usize> = (0..14)
            .map(|cursor| select_queue_with_priority(&[0, 0], &priorities, cursor, policy))
            .collect();

        // Per seven-slot window the high tier takes four slots and the low tier
        // takes the rest, so over two windows: 8 high, 6 low.
        assert_eq!(picks.iter().filter(|pick| **pick == 0).count(), 8);
        assert_eq!(picks.iter().filter(|pick| **pick == 1).count(), 6);

        let mut longest_high_run = 0_usize;
        let mut run = 0_usize;
        for pick in &picks {
            if *pick == 0 {
                run += 1;
                longest_high_run = longest_high_run.max(run);
            } else {
                run = 0;
            }
        }
        assert_eq!(
            longest_high_run,
            policy.high_priority_streak_limit() as usize
        );

        // The low tier is never starved: it is selected in every window.
        for window in picks.chunks(policy.window() as usize) {
            assert!(window.contains(&1), "low tier starved in {window:?}");
        }
    }

    #[test]
    fn priority_selection_gives_each_present_tier_its_share_of_a_window() {
        let policy = QueueSchedulingPolicy::default();
        let priorities = [
            QueuePriority::High,
            QueuePriority::Default,
            QueuePriority::Low,
        ];
        let picks: Vec<usize> = (0..policy.window())
            .map(|cursor| {
                select_queue_with_priority(&[0, 0, 0], &priorities, cursor as usize, policy)
            })
            .collect();

        // Four high slots, two default slots, then the one low slot.
        assert_eq!(picks, vec![0, 0, 0, 0, 1, 1, 2]);
    }

    #[test]
    fn priority_selection_carries_the_submit_path_tier_table_over_ten_windows() {
        // The starting point of the real-device experiment in `research/docs/21`
        // §6: one high queue, one default queue and six low queues, with the
        // cursor advancing once per submission and every submission retired
        // before the next one is enqueued. `VulkanExecutor` reports exactly this
        // sequence through `queue_submission_counts()`.
        let policy = QueueSchedulingPolicy::default();
        let tiers = [
            QueuePriority::High,
            QueuePriority::Default,
            QueuePriority::Low,
            QueuePriority::Low,
            QueuePriority::Low,
            QueuePriority::Low,
            QueuePriority::Low,
            QueuePriority::Low,
        ];
        let in_flight = [0_usize; 8];
        let picks: Vec<usize> = (0..70)
            .map(|cursor| select_queue_with_priority(&in_flight, &tiers, cursor, policy))
            .collect();

        let share = |tier: QueuePriority| picks.iter().filter(|pick| tiers[**pick] == tier).count();
        assert_eq!(share(QueuePriority::High), 40);
        assert_eq!(share(QueuePriority::Default), 20);
        assert_eq!(share(QueuePriority::Low), 10);

        let mut longest_high_run = 0_usize;
        let mut run = 0_usize;
        for pick in &picks {
            if tiers[*pick] == QueuePriority::High {
                run += 1;
                longest_high_run = longest_high_run.max(run);
            } else {
                run = 0;
            }
        }
        assert_eq!(
            longest_high_run,
            policy.high_priority_streak_limit() as usize
        );
        for window in picks.chunks(policy.window() as usize) {
            assert!(
                window.iter().any(|pick| tiers[*pick] == QueuePriority::Low),
                "low tier starved in {window:?}"
            );
        }
    }

    #[test]
    fn priority_selection_rotates_the_cursor_within_one_tier() {
        let policy = QueueSchedulingPolicy::default();
        let picks: Vec<usize> = (0..6)
            .map(|cursor| {
                select_queue_with_priority(&[0, 0, 0], &[QueuePriority::Low; 3], cursor, policy)
            })
            .collect();

        assert_eq!(picks, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn priority_selection_is_panic_free_for_degenerate_inputs() {
        let policy = QueueSchedulingPolicy::default();

        // Empty input keeps the least-loaded rule's answer.
        assert_eq!(select_queue_with_priority(&[], &[], 5, policy), 0);
        // A single queue is selected whatever the cursor and tier say.
        assert_eq!(select_queue_with_priority(&[4], &[], 9, policy), 0);
        assert_eq!(
            select_queue_with_priority(&[0], &[QueuePriority::High, QueuePriority::Low], 3, policy),
            0
        );
        // A missing priority entry reads as `Default` and never panics.
        assert_eq!(
            select_queue_with_priority(&[0, 0], &[QueuePriority::Low], 0, policy),
            1
        );
    }

    // Render contract, Step 1 (`research/docs/23` §3.1). These tests pin the
    // value shape and the first-increment narrowings; they cannot pin
    // execution, because nothing executes a render pass yet.

    fn render_attachment(format: AttachmentFormat) -> RenderAttachment {
        RenderAttachment {
            view_id: ViewId::new(21),
            allocation_id: AllocationId::new(22),
            format,
            width: 2,
            height: 2,
            load: LoadOp::Clear(ClearColor::new([0x40, 0x80, 0xc0, 0xff])),
            store: StoreOp::Store,
        }
    }

    fn render_pass() -> RenderPassDescriptor {
        RenderPassDescriptor {
            pipeline: PipelineId::new(5),
            color_attachments: vec![render_attachment(AttachmentFormat::Rgba8Unorm)],
            viewport: [0, 0, 2, 2],
            vertices: FULL_SCREEN_TRIANGLE_VERTICES,
            present: None,
        }
    }

    #[test]
    fn render_pass_accepts_the_first_increment_shape() {
        let pass = render_pass();
        pass.validate()
            .expect("the first-increment render pass is well formed");
        assert_eq!(
            pass.color_attachments[0]
                .expected_bytes()
                .expect("bounded extent"),
            16
        );
        assert_eq!(MAX_COLOR_ATTACHMENTS, 1);
        assert_eq!(FULL_SCREEN_TRIANGLE_VERTICES, 3);

        // Every admitted format carries one 4-byte texel, which is what makes
        // the fixed 4-byte clear payload well formed.
        for format in AttachmentFormat::ADMITTED {
            assert!(format.is_admitted_for_color_attachment());
            assert_eq!(format.bytes_per_texel(), ClearColor::BYTES as u64);
            let mut adopted = pass.clone();
            adopted.color_attachments[0].format = format;
            adopted.validate().expect("an admitted format renders");
        }
    }

    #[test]
    fn render_pass_refuses_an_empty_attachment_list() {
        let mut pass = render_pass();
        pass.color_attachments.clear();
        assert_eq!(pass.validate(), Err(ContractError::EmptyAttachmentList));
        assert_eq!(
            contract_error_refusal(ContractError::EmptyAttachmentList).slug,
            "trace_contract_invalid"
        );
    }

    #[test]
    fn render_pass_refuses_a_zero_sized_viewport_or_attachment() {
        let mut pass = render_pass();
        pass.viewport = [0, 0, 0, 2];
        assert_eq!(
            pass.validate(),
            Err(ContractError::ZeroDimension {
                field: "viewport",
                axis: 2,
            })
        );
        pass.viewport = [0, 0, 2, 0];
        assert_eq!(
            pass.validate(),
            Err(ContractError::ZeroDimension {
                field: "viewport",
                axis: 3,
            })
        );

        let mut attachment = render_attachment(AttachmentFormat::Rgba8Unorm);
        attachment.width = 0;
        assert_eq!(
            attachment.validate_shape(),
            Err(ContractError::ZeroDimension {
                field: "attachment",
                axis: 0,
            })
        );
        attachment = render_attachment(AttachmentFormat::Rgba8Unorm);
        attachment.height = 0;
        assert_eq!(
            attachment.validate_shape(),
            Err(ContractError::ZeroDimension {
                field: "attachment",
                axis: 1,
            })
        );
    }

    #[test]
    fn render_pass_refuses_an_unsupported_attachment_format() {
        // R32Uint is expressible but outside the first render increment.
        assert!(!AttachmentFormat::R32Uint.is_admitted_for_color_attachment());
        let mut pass = render_pass();
        pass.color_attachments[0].format = AttachmentFormat::R32Uint;
        assert_eq!(
            pass.validate(),
            Err(ContractError::UnsupportedAttachmentFormat(
                AttachmentFormat::R32Uint
            ))
        );
        let refusal = contract_error_refusal(ContractError::UnsupportedAttachmentFormat(
            AttachmentFormat::R32Uint,
        ));
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        assert_eq!(refusal.slug, "attachment_format_unsupported");
    }

    #[test]
    fn render_pass_refuses_more_attachments_than_the_device_admits() {
        let mut pass = render_pass();
        pass.color_attachments
            .push(render_attachment(AttachmentFormat::Bgra8Unorm));
        assert_eq!(
            pass.validate(),
            Err(ContractError::AttachmentLimitExceeded {
                requested: 2,
                maximum: MAX_COLOR_ATTACHMENTS,
            })
        );
        let refusal = contract_error_refusal(ContractError::AttachmentLimitExceeded {
            requested: 2,
            maximum: MAX_COLOR_ATTACHMENTS,
        });
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        assert_eq!(refusal.slug, "attachment_count_unsupported");
    }

    #[test]
    fn attachment_format_codes_round_trip_like_the_mcc1_texture_codes() {
        let cases = [
            (
                AttachmentFormat::R32Uint,
                0_u8,
                TextureFormat::R32Uint,
                false,
            ),
            (AttachmentFormat::R32Float, 1, TextureFormat::R32Float, true),
            (
                AttachmentFormat::Rgba8Unorm,
                2,
                TextureFormat::Rgba8Unorm,
                true,
            ),
            (
                AttachmentFormat::Bgra8Unorm,
                3,
                TextureFormat::Bgra8Unorm,
                true,
            ),
        ];
        for (format, code, texture_format, admitted) in cases {
            assert_eq!(format.code(), code);
            assert_eq!(AttachmentFormat::from_code(code), Some(format));
            assert_eq!(format.as_texture_format(), texture_format);
            assert_eq!(
                format.bytes_per_texel(),
                texture_format.bytes_per_texel(),
                "the attachment bridge must not change a texel's size"
            );
            assert_eq!(format.is_admitted_for_color_attachment(), admitted);
        }
        assert_eq!(AttachmentFormat::from_code(4), None);
        assert_eq!(AttachmentFormat::ADMITTED.len(), 3);
    }

    #[test]
    fn render_pass_refuses_undefined_load_and_store_operations() {
        let mut pass = render_pass();
        pass.color_attachments[0].load = LoadOp::DontCare;
        assert_eq!(
            pass.validate(),
            Err(ContractError::UnsupportedAttachmentLoadOp(LoadOp::DontCare))
        );
        pass.color_attachments[0].load = LoadOp::Clear(ClearColor::new([0, 0, 0, 0]));
        pass.color_attachments[0].store = StoreOp::DontCare;
        assert_eq!(
            pass.validate(),
            Err(ContractError::UnsupportedAttachmentStoreOp(
                StoreOp::DontCare
            ))
        );

        // `Load` is admitted: the compared bytes then depend only on earlier
        // passes, which the hazard rules already order.
        pass.color_attachments[0].store = StoreOp::Store;
        pass.color_attachments[0].load = LoadOp::Load;
        pass.validate().expect("Load + Store is admitted");
        assert_eq!(
            contract_error_refusal(ContractError::UnsupportedAttachmentStoreOp(
                StoreOp::DontCare
            ))
            .slug,
            "attachment_store_op_unsupported"
        );
        assert_eq!(
            contract_error_refusal(ContractError::UnsupportedAttachmentLoadOp(LoadOp::DontCare))
                .slug,
            "attachment_load_op_unsupported"
        );
    }

    #[test]
    fn render_pass_refuses_a_viewport_that_is_not_the_attachment_extent() {
        let mut pass = render_pass();
        pass.viewport = [1, 0, 2, 2];
        assert_eq!(
            pass.validate(),
            Err(ContractError::ViewportOriginUnsupported { origin: [1, 0] })
        );
        pass.viewport = [0, 0, 4, 2];
        assert_eq!(
            pass.validate(),
            Err(ContractError::ViewportExtentMismatch {
                viewport: [4, 2],
                attachment: [2, 2],
            })
        );
        assert_eq!(
            contract_error_refusal(ContractError::ViewportOriginUnsupported { origin: [1, 0] })
                .slug,
            "viewport_origin_unsupported"
        );
        assert_eq!(
            contract_error_refusal(ContractError::ViewportExtentMismatch {
                viewport: [4, 2],
                attachment: [2, 2],
            })
            .slug,
            "trace_contract_invalid"
        );
    }

    #[test]
    fn render_pass_refuses_a_draw_that_is_not_the_full_screen_triangle() {
        let mut pass = render_pass();
        pass.vertices = 6;
        assert_eq!(
            pass.validate(),
            Err(ContractError::DrawVertexCountMismatch {
                expected: FULL_SCREEN_TRIANGLE_VERTICES,
                actual: 6,
            })
        );
        pass.vertices = 0;
        assert_eq!(
            pass.validate(),
            Err(ContractError::DrawVertexCountMismatch {
                expected: FULL_SCREEN_TRIANGLE_VERTICES,
                actual: 0,
            })
        );
        assert_eq!(
            contract_error_refusal(ContractError::DrawVertexCountMismatch {
                expected: FULL_SCREEN_TRIANGLE_VERTICES,
                actual: 6,
            })
            .slug,
            "draw_shape_unsupported"
        );
    }

    #[test]
    fn render_attachment_refuses_zero_identities_before_any_format() {
        let attachment = render_attachment(AttachmentFormat::Rgba8Unorm);

        let mut zero_view = attachment;
        zero_view.view_id = ViewId::new(0);
        assert_eq!(
            zero_view.validate_shape(),
            Err(ContractError::InvalidIdentity("attachment view id"))
        );

        let mut zero_allocation = attachment;
        zero_allocation.allocation_id = AllocationId::new(0);
        assert_eq!(
            zero_allocation.validate_shape(),
            Err(ContractError::InvalidIdentity("attachment allocation id"))
        );

        // A zero pipeline identity is refused before any attachment is read.
        let mut pass = render_pass();
        pass.pipeline = PipelineId::new(0);
        assert_eq!(
            pass.validate(),
            Err(ContractError::InvalidIdentity("render pipeline id"))
        );
    }

    #[test]
    fn render_attachment_extent_overflow_is_refused_not_wrapped() {
        let mut attachment = render_attachment(AttachmentFormat::Rgba8Unorm);
        attachment.width = u64::MAX;
        attachment.height = u64::MAX;
        assert_eq!(
            attachment.expected_bytes(),
            Err(ContractError::ArithmeticOverflow("attachment bytes"))
        );
        assert_eq!(
            attachment.validate_shape(),
            Err(ContractError::ArithmeticOverflow("attachment bytes"))
        );
    }

    // Render pipeline contract, Step 3a (`research/docs/23` §3.4, §6). The value
    // is referenced by no trace and no provider yet, so these tests pin the field
    // set and the refusals; execution stays unpinned.

    fn render_pipeline_contract() -> RenderPipelineContract {
        RenderPipelineContract {
            vertex_entry: "full_screen_vertex".to_owned(),
            fragment_entry: "solid_color_fragment".to_owned(),
            color_format: AttachmentFormat::Rgba8Unorm,
            vertex_layout: VertexLayout::None,
        }
    }

    #[test]
    fn render_pipeline_contract_accepts_the_first_increment_shape() {
        let contract = render_pipeline_contract();
        contract
            .validate()
            .expect("the first-increment render pipeline is well formed");
        assert!(
            matches!(contract.vertex_layout, VertexLayout::None),
            "the first increment fixes one explicitly empty vertex layout"
        );
        assert_eq!(RenderPipelineStage::Vertex.name(), "vertex");
        assert_eq!(RenderPipelineStage::Fragment.name(), "fragment");
        assert_ne!(
            contract.vertex_entry, contract.fragment_entry,
            "the two stage entries address two compiled stage functions"
        );

        // Every admitted attachment format is admitted on the pipeline side too,
        // so the two contracts cannot drift apart on the format set.
        for format in AttachmentFormat::ADMITTED {
            let mut admitted = contract.clone();
            admitted.color_format = format;
            admitted.validate().expect("an admitted format compiles");
        }
    }

    // Render contract, Step 3c (`research/docs/23` §3.6): attachment resource
    // admission and the serial pool. An attachment references a view the trace
    // declares, so these tests resolve identities and extents; they cannot pin
    // execution, because neither provider renders yet.

    /// A render attachment that lands in `view_id`/`allocation_id`. The 2×2
    /// `Rgba8Unorm` shape is 16 tightly packed bytes, which is the extent the
    /// landing views below declare.
    fn attachment_into(view_id: u64, allocation_id: u64) -> RenderAttachment {
        RenderAttachment {
            view_id: ViewId::new(view_id),
            allocation_id: AllocationId::new(allocation_id),
            format: AttachmentFormat::Rgba8Unorm,
            width: 2,
            height: 2,
            load: LoadOp::Clear(ClearColor::new([0xfe, 0xfe, 0xfe, 0xfe])),
            store: StoreOp::Store,
        }
    }

    fn render_pass_into(attachment: RenderAttachment) -> TracePass {
        TracePass::Render(RenderPassDescriptor {
            pipeline: PipelineId::new(4),
            viewport: [0, 0, attachment.width as u32, attachment.height as u32],
            color_attachments: vec![attachment],
            vertices: FULL_SCREEN_TRIANGLE_VERTICES,
            present: None,
        })
    }

    /// The buffer-shaped landing view a render pass stores 16 attachment bytes
    /// into, read-only from the compute side.
    fn landing_view(view_id: u64, allocation_id: u64) -> BufferView {
        BufferView {
            view_id: ViewId::new(view_id),
            metal_binding: 0,
            allocation_id: AllocationId::new(allocation_id),
            offset: 0,
            length: 16,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0; 16]),
        }
    }

    /// A compute pass declaring one buffer view plus a render pass whose
    /// attachment references the same view identity and allocation.
    fn attachment_trace(view: BufferView, attachment: RenderAttachment) -> ComputeTrace {
        let access = view.access;
        let mut value = trace(vec![pass(4, vec![view])]);
        // `ComputePass::validate` compares each binding against its reflection,
        // so the pipeline contract carries the landing view's access.
        value.pipelines[0].contract.buffer_bindings[0].access = access;
        // The render pass names entry 4, so the fixture carries that entry's
        // render half the way a registered render pipeline's metadata does.
        declare_render_contract(&mut value);
        value.passes.push(render_pass_into(attachment));
        value
    }

    /// A compute pass declaring one sampled texture view plus a render pass
    /// whose attachment lands in it.
    fn texture_attachment_trace(
        texture: TextureView,
        attachment: RenderAttachment,
    ) -> ComputeTrace {
        let mut value = trace(Vec::new());
        value.pipelines[0].contract.buffer_bindings.clear();
        value.passes.push(TracePass::Compute(ComputePass {
            pipeline: PipelineId::new(4),
            buffers: Vec::new(),
            textures: vec![texture],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
        }));
        declare_render_contract(&mut value);
        value.passes.push(render_pass_into(attachment));
        value
    }

    fn two_by_two_texture(format: TextureFormat) -> TextureView {
        texture_view(TextureCase {
            texture_type: TextureType::D2,
            format,
            width: 2,
            height: 2,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            bytes: vec![0; 4 * format.bytes_per_texel() as usize],
        })
    }

    /// `count` compute passes, each declaring one distinct 2×2 view. With
    /// `attach`, every declaration is also the target of a render pass.
    fn many_attachment_targets(count: usize, attach: bool) -> ComputeTrace {
        let mut value = trace(Vec::new());
        let template = value.pipelines.pop().expect("template pipeline");
        for index in 1..=count as u64 {
            let mut pipeline = template.clone();
            pipeline.pipeline_id = PipelineId::new(100 + index);
            pipeline.contract.buffer_bindings.clear();
            let pipeline_id = pipeline.pipeline_id;
            let mut texture = two_by_two_texture(TextureFormat::Rgba8Unorm);
            texture.view_id = ViewId::new(index);
            texture.allocation_id = AllocationId::new(index);
            value.passes.push(TracePass::Compute(ComputePass {
                pipeline: pipeline_id,
                buffers: Vec::new(),
                textures: vec![texture],
                dispatch: Dispatch {
                    kind: DispatchKind::ThreadsExact,
                    grid: [1, 1, 1],
                    threads_per_threadgroup: [1, 1, 1],
                },
            }));
            if attach {
                // The render entry names this pass's own pipeline, so the entry
                // has to carry the render half the render pass resolves with.
                pipeline.render = Some(render_pipeline_contract());
            }
            value.pipelines.push(pipeline);
            if attach {
                // The attachment's render entry names a registered pipeline too.
                let mut entry = render_pass_into(attachment_into(index, index));
                match &mut entry {
                    TracePass::Render(descriptor) => descriptor.pipeline = pipeline_id,
                    TracePass::Compute(_) => unreachable!("built as a render entry"),
                }
                value.passes.push(entry);
            }
        }
        value
    }

    /// The snapshot that backs a 16-byte landing view in allocation 9.
    fn landing_resources() -> ResourceTableSnapshot {
        let mut pool = ResourceTableSnapshot::new();
        pool.insert_allocation(AllocationRecord {
            allocation_id: AllocationId::new(9),
            owner_epoch: DeviceEpoch::new(1),
            size: 16,
        })
        .unwrap();
        pool
    }

    fn render_capabilities() -> ProviderCapabilities {
        let mut provider = capabilities();
        provider.max_passes = 8;
        provider.supports_render_passes = true;
        provider.max_color_attachments = 1;
        provider.max_attachment_dimension = [2, 2];
        provider.supported_color_formats = vec![AttachmentFormat::Rgba8Unorm];
        provider
    }

    fn admitted_completion() -> CompletionDisposition {
        CompletionDisposition::CompletedVisible {
            token: CompletionToken {
                device_epoch: DeviceEpoch::new(1),
                submission_id: SubmissionId::new(3),
            },
        }
    }

    #[test]
    fn render_pipeline_contract_refuses_an_empty_entry_name() {
        let mut contract = render_pipeline_contract();
        contract.vertex_entry.clear();
        assert_eq!(
            contract.validate(),
            Err(ContractError::EmptyRenderPipelineEntry(
                RenderPipelineStage::Vertex
            ))
        );

        // Whitespace is empty too, matching the trim rule the compute side
        // already applies to entry names.
        let mut contract = render_pipeline_contract();
        contract.fragment_entry = "   ".to_owned();
        assert_eq!(
            contract.validate(),
            Err(ContractError::EmptyRenderPipelineEntry(
                RenderPipelineStage::Fragment
            ))
        );

        let refusal = contract_error_refusal(ContractError::EmptyRenderPipelineEntry(
            RenderPipelineStage::Vertex,
        ));
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "trace_contract_invalid");
        assert!(
            refusal
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("vertex entry")),
            "the refusal detail has to name the stage to fix, got {:?}",
            refusal.detail
        );
    }

    #[test]
    fn attachment_views_resolve_against_the_trace_resource_table() {
        let value = attachment_trace(landing_view(7, 9), attachment_into(7, 9));
        value.validate_serial_buffer_reuse().unwrap();
        assert_eq!(
            value
                .attachments()
                .map(|(pass_index, attachment)| (pass_index, attachment.view_id))
                .collect::<Vec<_>>(),
            vec![(1, ViewId::new(7))]
        );
        render_capabilities()
            .admit(&value, &landing_resources())
            .unwrap();

        // The pool records the attachment's store as well: the landing view is
        // read by the compute pass and written by the render pass.
        let pool = value.serial_resources().unwrap();
        assert_eq!(pool.len(), 1);
        assert_eq!(pool[0].view_id, ViewId::new(7));
        assert_eq!(pool[0].access, BufferAccess::ReadWrite);

        // A texture-declared target resolves on the same terms when it is the
        // 2D, single-sample, non-array shape that carries the attachment's
        // format.
        let mut texture = two_by_two_texture(TextureFormat::Rgba8Unorm);
        texture.allocation_id = AllocationId::new(11);
        let target = texture_attachment_trace(texture, attachment_into(7, 11));
        target.validate_serial_buffer_reuse().unwrap();

        // A compute-only trace has no attachment to resolve, so it reaches its
        // pre-render pool unchanged.
        let compute_only = trace(vec![pass(4, vec![buffer(1, 0)])]);
        assert_eq!(compute_only.attachments().count(), 0);
        assert_eq!(
            compute_only.serial_resources().unwrap()[0].access,
            BufferAccess::Write
        );
    }

    #[test]
    fn render_pipeline_contract_refuses_one_entry_name_for_both_stages() {
        let mut contract = render_pipeline_contract();
        let shared = contract.vertex_entry.clone();
        contract.fragment_entry = shared;
        assert_eq!(
            contract.validate(),
            Err(ContractError::DuplicateRenderPipelineEntry(
                RenderPipelineStage::Fragment
            ))
        );
        // Duplication is checked before the format, so a contract that is wrong
        // twice reports the entry shape a caller can see without a device.
        contract.color_format = AttachmentFormat::R32Uint;
        assert_eq!(
            contract.validate(),
            Err(ContractError::DuplicateRenderPipelineEntry(
                RenderPipelineStage::Fragment
            ))
        );

        let refusal = contract_error_refusal(ContractError::DuplicateRenderPipelineEntry(
            RenderPipelineStage::Fragment,
        ));
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "trace_contract_invalid");
    }

    #[test]
    fn render_pipeline_contract_refuses_an_unsupported_color_format() {
        // R32Uint is expressible but outside the first render increment, and the
        // pipeline refuses it with the same variant and slug as the attachment.
        let mut contract = render_pipeline_contract();
        contract.color_format = AttachmentFormat::R32Uint;
        assert_eq!(
            contract.validate(),
            Err(ContractError::UnsupportedAttachmentFormat(
                AttachmentFormat::R32Uint
            ))
        );
        let refusal = contract_error_refusal(ContractError::UnsupportedAttachmentFormat(
            AttachmentFormat::R32Uint,
        ));
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        assert_eq!(refusal.slug, "attachment_format_unsupported");

        // The pipeline's own format is checked before it is compared against the
        // pass, so the capability narrowing is not reported as a mismatch.
        let mut pass = render_pass();
        pass.color_attachments[0].format = AttachmentFormat::Rgba8Unorm;
        assert_eq!(
            contract.validate_against(&pass),
            Err(ContractError::UnsupportedAttachmentFormat(
                AttachmentFormat::R32Uint
            ))
        );
    }

    #[test]
    fn attachment_views_must_be_declared_by_the_trace() {
        // Nothing declares view 7: the attachment names a resource this
        // submission does not have.
        let mut value = trace(vec![pass(4, vec![buffer(1, 0)])]);
        value.passes.push(render_trace_pass(4, 2, 2));
        let expected = ContractError::AttachmentViewUnknown {
            pass_index: 1,
            view: ViewId::new(7),
            allocation: AllocationId::new(9),
        };
        assert_eq!(value.validate_serial_buffer_reuse(), Err(expected.clone()));
        let refusal = contract_error_refusal(expected.clone());
        assert_eq!(refusal.class, ProviderErrorClass::Resource);
        assert_eq!(refusal.slug, "attachment_allocation_unknown");
        assert_eq!(refusal.detail, Some(expected.to_string()));

        // A render-only trace declares no view at all, so the first increment
        // refuses it instead of opening a second resource-declaration channel.
        let mut render_only = trace(vec![pass(4, vec![buffer(1, 0)])]);
        render_only.passes.clear();
        render_only.passes.push(render_trace_pass(4, 2, 2));
        assert_eq!(
            render_only.validate_serial_buffer_reuse(),
            Err(ContractError::AttachmentViewUnknown {
                pass_index: 0,
                view: ViewId::new(7),
                allocation: AllocationId::new(9),
            })
        );

        // A declared view whose allocation is not the one the attachment names
        // is the same class of refusal: the (view, allocation) pair has to
        // exist, and only the identity pair does.
        let mismatched = attachment_trace(landing_view(7, 9), attachment_into(7, 10));
        let expected = ContractError::AttachmentViewAllocationMismatch {
            pass_index: 1,
            view: ViewId::new(7),
            declared: AllocationId::new(9),
            referenced: AllocationId::new(10),
        };
        assert_eq!(
            mismatched.validate_serial_buffer_reuse(),
            Err(expected.clone())
        );
        assert_eq!(
            contract_error_refusal(expected).slug,
            "attachment_allocation_unknown"
        );
    }

    #[test]
    fn attachment_extent_must_match_the_declared_view() {
        // A buffer declaration is compared by byte length: 2×2 `Rgba8Unorm`
        // issues 16 bytes, the declaration holds 8.
        let mut short = landing_view(7, 9);
        short.length = 8;
        short.source = BufferSource::OwnedBytes(vec![0; 8]);
        let value = attachment_trace(short, attachment_into(7, 9));
        let expected = ContractError::AttachmentExtentMismatch {
            pass_index: 1,
            view: ViewId::new(7),
            expected: 16,
            declared: 8,
        };
        assert_eq!(value.validate_serial_buffer_reuse(), Err(expected.clone()));
        let refusal = contract_error_refusal(expected);
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "attachment_extent_mismatch");

        // A texture declaration is compared by texel shape and format as well.
        // 4×1 is the same byte count as 2×2 and still a different target, so the
        // refusal has to name both shapes: two equal byte counts cannot locate
        // the mistake (review item M4, 2026-09-14).
        let mut wide = two_by_two_texture(TextureFormat::Rgba8Unorm);
        wide.width = 4;
        wide.height = 1;
        wide.allocation_id = AllocationId::new(11);
        assert_eq!(wide.expected_bytes().unwrap(), 16);
        let shape = texture_attachment_trace(wide, attachment_into(7, 11));
        let expected = ContractError::AttachmentTextureShapeMismatch {
            pass_index: 1,
            view: ViewId::new(7),
            attachment_extent: [2, 2],
            attachment_format: AttachmentFormat::Rgba8Unorm,
            attachment_bytes: 16,
            declared_type: TextureType::D2,
            declared_extent: [4, 1],
            declared_depth: 1,
            declared_array_length: 1,
            declared_sample_count: 1,
            declared_format: TextureFormat::Rgba8Unorm,
            declared_bytes: 16,
        };
        assert_eq!(shape.validate_serial_buffer_reuse(), Err(expected.clone()));
        let refusal = contract_error_refusal(expected.clone());
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "attachment_extent_mismatch");
        let detail = refusal
            .detail
            .clone()
            .expect("the refusal carries its detail");
        for fragment in ["2×2", "4×1", "Rgba8Unorm", "16 bytes"] {
            assert!(
                detail.contains(fragment),
                "the refusal has to name {fragment:?} to be locatable, got {detail:?}"
            );
        }
        let expected_detail = expected.to_string();
        assert_eq!(refusal.detail.as_deref(), Some(expected_detail.as_str()));

        // A format mismatch lands on the same rule, even when the byte extents
        // agree: `R32Float` and `Rgba8Unorm` both occupy 4 bytes per texel, so
        // only the format comparison catches this one.
        let mut float = two_by_two_texture(TextureFormat::R32Float);
        float.allocation_id = AllocationId::new(11);
        let format = texture_attachment_trace(float, attachment_into(7, 11));
        let mut expected_format = expected;
        let ContractError::AttachmentTextureShapeMismatch {
            declared_extent,
            declared_format,
            ..
        } = &mut expected_format
        else {
            panic!("the fixture expects the texture shape refusal");
        };
        *declared_extent = [2, 2];
        *declared_format = TextureFormat::R32Float;
        assert_eq!(format.validate_serial_buffer_reuse(), Err(expected_format));
    }

    #[test]
    fn render_pipeline_contract_refuses_a_pass_with_a_different_attachment_format() {
        let contract = render_pipeline_contract();
        let mut pass = render_pass();
        pass.color_attachments[0].format = AttachmentFormat::Bgra8Unorm;
        assert_eq!(
            contract.validate_against(&pass),
            Err(ContractError::RenderPipelineFormatMismatch {
                pipeline: AttachmentFormat::Rgba8Unorm,
                attachment: AttachmentFormat::Bgra8Unorm,
            })
        );

        let refusal = contract_error_refusal(ContractError::RenderPipelineFormatMismatch {
            pipeline: AttachmentFormat::Rgba8Unorm,
            attachment: AttachmentFormat::Bgra8Unorm,
        });
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "trace_contract_invalid");
        assert!(
            refusal
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("does not match attachment format")),
            "the refusal detail has to name both formats, got {:?}",
            refusal.detail
        );
    }

    #[test]
    fn attachment_targets_must_not_race_a_compute_writer() {
        // A compute pass that writes the target and a render pass that stores
        // into it leave the two writers' order inexpressible.
        let mut writing = landing_view(7, 9);
        writing.access = BufferAccess::Write;
        let value = attachment_trace(writing, attachment_into(7, 9));
        let expected = ContractError::AttachmentComputeConflict {
            pass_index: 1,
            view: ViewId::new(7),
            compute_view: ViewId::new(7),
            compute_pass: 0,
        };
        assert_eq!(value.validate_serial_buffer_reuse(), Err(expected.clone()));
        let refusal = contract_error_refusal(expected);
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "attachment_resource_conflict");

        // A storage texture binding is the texture-side spelling of the same
        // hazard.
        let mut storage = two_by_two_texture(TextureFormat::Rgba8Unorm);
        storage.access = TextureAccess::Storage;
        storage.allocation_id = AllocationId::new(11);
        assert_eq!(
            texture_attachment_trace(storage, attachment_into(7, 11))
                .validate_serial_buffer_reuse(),
            Err(ContractError::AttachmentComputeConflict {
                pass_index: 1,
                view: ViewId::new(7),
                compute_view: ViewId::new(7),
                compute_pass: 0,
            })
        );

        // Two render passes storing the same target in order are admitted: the
        // second pass's load/store sees the first pass's bytes, so serial order
        // is the whole ordering statement the trace needs.
        let mut ordered = attachment_trace(landing_view(7, 9), attachment_into(7, 9));
        ordered.passes.push(render_pass_into(attachment_into(7, 9)));
        ordered.validate_serial_buffer_reuse().unwrap();
        assert_eq!(
            ordered.serial_resources().unwrap()[0].access,
            BufferAccess::ReadWrite
        );
    }

    // Review item M3 (2026-09-14): whether an attachment races a compute writer
    // is a byte-range question inside one allocation, not a view-identity
    // question. Two sibling views of one allocation can overlap without sharing
    // an identity, and the same allocation can carry neighbouring views that
    // never meet.

    /// A trace whose render attachment lands in `landing` while one compute pass
    /// writes `writer`; both views name the same allocation.
    fn sibling_writer_trace(
        landing: BufferView,
        writer: BufferView,
        attachment: RenderAttachment,
    ) -> ComputeTrace {
        let mut landing = landing;
        landing.metal_binding = 0;
        let mut writer = writer;
        writer.metal_binding = 1;
        let accesses = [landing.access, writer.access];
        let mut value = trace(vec![pass(4, vec![landing, writer])]);
        // `ComputePass::validate` compares each binding against its reflection,
        // so the pipeline contract carries both views' accesses.
        value.pipelines[0].contract.buffer_bindings = accesses
            .into_iter()
            .enumerate()
            .map(|(index, access)| BufferBindingContract {
                metal_binding: index as u32,
                access,
                footprint: FootprintProof::Affine {
                    accesses: Vec::new(),
                },
            })
            .collect();
        declare_render_contract(&mut value);
        value.passes.push(render_pass_into(attachment));
        value
    }

    #[test]
    fn attachment_conflicts_follow_the_allocation_byte_range() {
        // Views 7 and 8 are siblings of allocation 9: 7 is the attachment's own
        // landing view at [0, 16), 8 is written by compute at [8, 24). They
        // share bytes without sharing an identity, so this is exactly the
        // write/write hazard the same-view rule catches.
        let mut overlapping = landing_view(8, 9);
        overlapping.offset = 8;
        overlapping.access = BufferAccess::Write;
        let value = sibling_writer_trace(landing_view(7, 9), overlapping, attachment_into(7, 9));
        let expected = ContractError::AttachmentComputeConflict {
            pass_index: 1,
            view: ViewId::new(7),
            compute_view: ViewId::new(8),
            compute_pass: 0,
        };
        assert_eq!(value.validate_serial_buffer_reuse(), Err(expected.clone()));
        let refusal = contract_error_refusal(expected);
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "attachment_resource_conflict");

        // The control: a sibling view of the same allocation that shares no byte
        // with the attachment is admitted. View identity is not what makes the
        // hazard, so an identity-only rule is wrong in both directions.
        let mut disjoint = landing_view(8, 9);
        disjoint.offset = 16;
        disjoint.length = 8;
        disjoint.access = BufferAccess::Write;
        disjoint.source = BufferSource::OwnedBytes(vec![0; 8]);
        let allowed = sibling_writer_trace(landing_view(7, 9), disjoint, attachment_into(7, 9));
        allowed.validate_serial_buffer_reuse().unwrap();

        // The neighbours are admitted end to end as well, on a snapshot whose
        // allocation holds both byte ranges.
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(9),
                owner_epoch: DeviceEpoch::new(1),
                size: 24,
            })
            .unwrap();
        let mut render = render_capabilities();
        // Two sibling views of one allocation need the alias mode that admits
        // them; what this assertion is about is the byte ranges, not aliasing.
        render.alias_mode = AliasMode::DistinctViews;
        render.admit(&allowed, &resources).unwrap();
    }

    #[test]
    fn the_contract_fixes_compute_before_render() {
        // The increment executes every compute pass before every render pass, so
        // a compute pass that *follows* a render store of bytes it binds would
        // observe pre-render bytes where the trace's own order defines
        // post-render ones. That order is now part of the contract core
        // admission enforces, rather than an assumption only a provider walk
        // states (review item I4, 2026-09-14).
        let mut value = attachment_trace(landing_view(7, 9), attachment_into(7, 9));
        value
            .passes
            .push(TracePass::Compute(pass(4, vec![landing_view(7, 9)])));
        let expected = ContractError::RenderPassOrderUnsupported {
            pass_index: 1,
            compute_pass: 2,
            view: ViewId::new(7),
            compute_view: ViewId::new(7),
        };
        assert_eq!(value.validate_serial_buffer_reuse(), Err(expected.clone()));
        let refusal = contract_error_refusal(expected);
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        assert_eq!(refusal.phase, ProviderPhase::Resolve);
        assert_eq!(refusal.slug, "render_pass_order_unsupported");

        // The legal direction is the one the milestone case uses: the compute pass
        // that reads the attachment's bytes comes *before* the render store, so
        // serial order gives it the bytes the trace asked for.
        let legal = attachment_trace(landing_view(7, 9), attachment_into(7, 9));
        legal.validate_serial_buffer_reuse().unwrap();

        // A compute pass whose bytes merely neighbour the attachment's range is
        // not reordered work: the same byte-range rule that governs the
        // write/write pair governs the read half too.
        let mut neighbour = landing_view(8, 9);
        neighbour.offset = 16;
        neighbour.length = 8;
        neighbour.source = BufferSource::OwnedBytes(vec![0; 8]);
        let mut value = attachment_trace(landing_view(7, 9), attachment_into(7, 9));
        value
            .passes
            .push(TracePass::Compute(pass(4, vec![neighbour])));
        value.validate_serial_buffer_reuse().unwrap();
    }

    #[test]
    fn render_pipeline_contract_accepts_a_pass_with_the_compiled_format() {
        // The control for the mismatch above: the pass the pipeline was compiled
        // for is accepted, for every admitted format.
        for format in AttachmentFormat::ADMITTED {
            let mut contract = render_pipeline_contract();
            contract.color_format = format;
            let mut pass = render_pass();
            pass.color_attachments[0].format = format;
            contract
                .validate_against(&pass)
                .expect("a pipeline compiles for the attachment it renders into");
            // The pass keeps its own shape rules; agreement does not replace
            // them.
            pass.validate().expect("the pass is well formed on its own");
        }
    }

    #[test]
    fn render_pipeline_contract_refuses_to_agree_with_an_empty_attachment_list() {
        let contract = render_pipeline_contract();
        let mut pass = render_pass();
        pass.color_attachments.clear();
        assert_eq!(
            contract.validate_against(&pass),
            Err(ContractError::EmptyAttachmentList)
        );
    }

    #[test]
    fn attachment_targets_share_the_serial_resource_budget() {
        // 64 texture-backed targets resolve and stay inside the pre-render
        // budget; the 65th is refused with the existing bounded-resource
        // refusal, because the pool keeps every attachment target live for the
        // whole serial submission.
        let value = many_attachment_targets(MAX_SERIAL_RESOURCES, true);
        value.validate_serial_buffer_reuse().unwrap();
        assert!(value.serial_resources().unwrap().is_empty());

        let over = many_attachment_targets(MAX_SERIAL_RESOURCES + 1, true);
        let expected = ContractError::SerialResourceLimit {
            requested: MAX_SERIAL_RESOURCES + 1,
            maximum: MAX_SERIAL_RESOURCES,
        };
        assert_eq!(over.validate_serial_buffer_reuse(), Err(expected.clone()));
        let refusal = contract_error_refusal(expected);
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        // The budget keeps the refusal identity it already had for buffers; an
        // attachment target spends that budget, it does not introduce a second.
        assert_eq!(refusal.slug, "trace_contract_invalid");

        // Control: texture views were never part of the pre-render buffer
        // budget, so the same 65 declarations without attachments still
        // validate. Attachments are what bring them into the budget.
        many_attachment_targets(MAX_SERIAL_RESOURCES + 1, false)
            .validate_serial_buffer_reuse()
            .unwrap();
    }

    #[test]
    fn attachment_stores_make_the_target_a_writeback_subject() {
        let value = attachment_trace(landing_view(7, 9), attachment_into(7, 9));
        let admitted = render_capabilities()
            .validate_trace(value, landing_resources())
            .unwrap();
        let admitted_trace = admitted.trace();

        // The pool reports the landing view as writable, so a visible
        // completion owes a writeback for it: the render track's bytes land
        // through the readback the compute path already uses.
        assert_eq!(
            validate_writebacks_for_trace(admitted_completion(), &[], admitted_trace),
            Err(ContractError::MissingWriteback {
                allocation: AllocationId::new(9),
                view: ViewId::new(7),
            })
        );
        let landed = BufferWriteback {
            view_id: ViewId::new(7),
            allocation_id: AllocationId::new(9),
            offset: 0,
            bytes: [0x40, 0x80, 0xc0, 0xff].repeat(4),
        };
        validate_writebacks_for_trace(admitted_completion(), &[landed], admitted_trace).unwrap();

        // Control: a compute-only trace whose view stays read-only keeps the
        // pre-render pool access, so nothing about its readback changed.
        let mut read_only = trace(vec![pass(4, vec![buffer(1, 0)])]);
        read_only.pipelines[0].contract.buffer_bindings[0].access = BufferAccess::Read;
        compute_pass_mut(&mut read_only, 0).buffers[0].access = BufferAccess::Read;
        assert_eq!(
            read_only.serial_resources().unwrap()[0].access,
            BufferAccess::Read
        );
    }

    // Presentation contract, Step 1 (`research/docs/24` §3.1). These tests pin
    // the value shape and the first-increment narrowings. Nothing encodes or
    // executes a present action yet, so there is no execution to pin; the
    // control cases exist to prove each refusal is the rule it claims to be and
    // not a side effect of some other field.

    fn present_target(format: AttachmentFormat) -> PresentTarget {
        PresentTarget {
            allocation_id: AllocationId::new(22),
            view_id: ViewId::new(21),
            format,
            width: 2,
            height: 2,
            image_count: MAX_PRESENT_IMAGE_COUNT,
            initial: InitialState::Sentinel(vec![0x40, 0x80, 0xc0, 0xff]),
        }
    }

    fn present_descriptor() -> PresentDescriptor {
        PresentDescriptor {
            target: present_target(AttachmentFormat::Rgba8Unorm),
            source: ViewId::new(21),
            mode: PresentMode::Fifo,
            acquire: AcquirePolicy::Blocking,
        }
    }

    #[test]
    fn present_descriptor_accepts_the_first_increment_shape() {
        let present = present_descriptor();
        present
            .validate_against(&render_pass())
            .expect("the first-increment present is well formed");
        assert_eq!(present.target.expected_bytes().expect("bounded extent"), 16);
        assert_eq!(MAX_PRESENT_TARGETS, 1);
        assert_eq!(MAX_PRESENT_IMAGE_COUNT, 1);
        assert_eq!(PresentMode::ADMITTED, [PresentMode::Fifo]);
        assert_eq!(AcquirePolicy::Blocking.timeout_nanos(), None);
        assert_eq!(AcquirePolicy::Blocking.code(), 0);

        // Every admitted format is presentable with a one-texel sentinel, which
        // is what `docs/24` §3.2's "inherit the format" rule leaves as the only
        // admitted target colour space.
        for format in AttachmentFormat::ADMITTED {
            let mut adopted = present.clone();
            adopted.target.format = format;
            let mut pass = render_pass();
            pass.color_attachments[0].format = format;
            adopted
                .validate_against(&pass)
                .expect("an admitted format presents");
            assert_eq!(adopted.target.format.bytes_per_texel(), 4);
        }

        // A target that declares no pre-pass contents is expressible and
        // admitted (`docs/24` §5.3 gives it a zero `copy_in` count); only the
        // sentinel carries a length that has to agree with the format.
        let mut undefined = present.clone();
        undefined.target.initial = InitialState::Undefined;
        undefined.target.validate_shape().unwrap();
        assert_eq!(undefined.target.initial.sentinel(), None);

        // Wire codes are pinned so Step 2 encodes the values this step tested.
        for (mode, code) in [
            (PresentMode::Immediate, 0),
            (PresentMode::Mailbox, 1),
            (PresentMode::Fifo, 2),
            (PresentMode::FifoRelaxed, 3),
        ] {
            assert_eq!(mode.code(), code);
            assert_eq!(PresentMode::from_code(code), Some(mode));
        }
        assert_eq!(PresentMode::from_code(4), None);
    }

    #[test]
    fn present_refuses_a_mode_other_than_fifo() {
        for mode in [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
        ] {
            assert!(!mode.is_admitted_for_present());
            let mut present = present_descriptor();
            present.mode = mode;
            assert_eq!(
                present.validate_against(&render_pass()),
                Err(ContractError::PresentModeUnsupported(mode))
            );
        }
        let refusal =
            contract_error_refusal(ContractError::PresentModeUnsupported(PresentMode::Mailbox));
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        assert_eq!(refusal.phase, ProviderPhase::Resolve);
        assert_eq!(refusal.slug, "present_mode_unsupported");
    }

    #[test]
    fn present_refuses_an_acquire_policy_with_a_deadline() {
        let mut present = present_descriptor();
        present.acquire = AcquirePolicy::Timeout(1_000_000);
        assert!(!present.acquire.is_admitted_for_present());
        assert_eq!(present.acquire.code(), 1);
        assert_eq!(present.acquire.timeout_nanos(), Some(1_000_000));
        let expected =
            ContractError::PresentAcquirePolicyUnsupported(AcquirePolicy::Timeout(1_000_000));
        assert_eq!(
            present.validate_against(&render_pass()),
            Err(expected.clone())
        );
        let refusal = contract_error_refusal(expected);
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        assert_eq!(refusal.slug, "present_acquire_policy_unsupported");
    }

    #[test]
    fn present_refuses_a_target_that_is_not_single_buffered() {
        for requested in [0, 2] {
            let mut present = present_descriptor();
            present.target.image_count = requested;
            let expected = ContractError::PresentImageCountUnsupported {
                requested,
                maximum: MAX_PRESENT_IMAGE_COUNT,
            };
            assert_eq!(
                present.validate_against(&render_pass()),
                Err(expected.clone())
            );
            let refusal = contract_error_refusal(expected);
            assert_eq!(refusal.class, ProviderErrorClass::Capability);
            assert_eq!(refusal.slug, "present_image_count_unsupported");
        }
    }

    #[test]
    fn present_refuses_a_target_format_that_disagrees_with_its_source() {
        let mut present = present_descriptor();
        present.target.format = AttachmentFormat::Bgra8Unorm;
        let expected = ContractError::PresentFormatMismatch {
            source: ViewId::new(21),
            target: AttachmentFormat::Bgra8Unorm,
            attachment: AttachmentFormat::Rgba8Unorm,
        };
        assert_eq!(
            present.validate_against(&render_pass()),
            Err(expected.clone())
        );
        let refusal = contract_error_refusal(expected);
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "present_format_mismatch");

        // Control: the format is inherited, not fixed. A pass rendering into the
        // target's own format is the shape that passes.
        let mut pass = render_pass();
        pass.color_attachments[0].format = AttachmentFormat::Bgra8Unorm;
        present.validate_against(&pass).unwrap();
    }

    #[test]
    fn present_refuses_a_target_extent_that_disagrees_with_its_source() {
        let mut present = present_descriptor();
        present.target.width = 4;
        let expected = ContractError::PresentExtentMismatch {
            source: ViewId::new(21),
            target: [4, 2],
            attachment: [2, 2],
        };
        assert_eq!(
            present.validate_against(&render_pass()),
            Err(expected.clone())
        );
        let refusal = contract_error_refusal(expected);
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "present_extent_mismatch");

        // Control: a target with no source to disagree with fails on the
        // pass's own empty-attachment rule, not on a vacuous extent agreement.
        let mut empty = render_pass();
        empty.color_attachments.clear();
        assert_eq!(
            present_descriptor().validate_against(&empty),
            Err(ContractError::EmptyAttachmentList)
        );
    }

    #[test]
    fn present_refuses_a_sentinel_whose_length_does_not_fit_the_format() {
        for length in [0usize, 3, 5] {
            let mut present = present_descriptor();
            present.target.initial = InitialState::Sentinel(vec![0x11; length]);
            let expected = ContractError::PresentSentinelLengthMismatch {
                format: AttachmentFormat::Rgba8Unorm,
                expected: 4,
                actual: length as u64,
            };
            assert_eq!(
                present.validate_against(&render_pass()),
                Err(expected.clone())
            );
            let refusal = contract_error_refusal(expected);
            assert_eq!(refusal.class, ProviderErrorClass::Args);
            assert_eq!(refusal.slug, "present_sentinel_length_mismatch");
        }

        // Boundary: exactly one tightly packed texel is the admitted length, for
        // every admitted format. The target's own format is what the sentinel is
        // measured against, not the source's.
        for format in AttachmentFormat::ADMITTED {
            let mut present = present_descriptor();
            present.target.format = format;
            present.target.initial =
                InitialState::Sentinel(vec![0x11; format.bytes_per_texel() as usize]);
            assert_eq!(
                present.target.initial.sentinel().map(|bytes| bytes.len()),
                Some(4)
            );
            let mut pass = render_pass();
            pass.color_attachments[0].format = format;
            present.validate_against(&pass).unwrap();
        }
    }

    #[test]
    fn present_refuses_a_source_that_the_pass_does_not_render_into() {
        let mut present = present_descriptor();
        present.source = ViewId::new(99);
        let expected = ContractError::PresentSourceUnknown {
            source: ViewId::new(99),
        };
        assert_eq!(
            present.validate_against(&render_pass()),
            Err(expected.clone())
        );
        let refusal = contract_error_refusal(expected);
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "present_source_unknown");

        // Control: only the `source` field changed, so the refusal is about the
        // source and not about the target's own identity.
        present.source = ViewId::new(21);
        present
            .validate_against(&render_pass())
            .expect("the pass's own view is the admitted source");
    }

    #[test]
    fn present_refuses_a_target_without_identity_extent_or_bounded_bytes() {
        let mut present = present_descriptor();
        present.target.view_id = ViewId::new(0);
        assert_eq!(
            present.target.validate_shape(),
            Err(ContractError::InvalidIdentity("present target view id"))
        );

        let mut present = present_descriptor();
        present.target.allocation_id = AllocationId::new(0);
        assert_eq!(
            present.target.validate_shape(),
            Err(ContractError::InvalidIdentity(
                "present target allocation id"
            ))
        );

        let mut zero_width = present_descriptor();
        zero_width.target.width = 0;
        assert_eq!(
            zero_width.target.validate_shape(),
            Err(ContractError::ZeroDimension {
                field: "present target",
                axis: 0,
            })
        );
        let mut zero_height = present_descriptor();
        zero_height.target.height = 0;
        let expected = ContractError::ZeroDimension {
            field: "present target",
            axis: 1,
        };
        assert_eq!(zero_height.target.validate_shape(), Err(expected.clone()));
        let refusal = contract_error_refusal(expected);
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "trace_contract_invalid");

        // The byte extent fails closed rather than wrapping, exactly as the
        // attachment's does.
        let mut present = present_descriptor();
        present.target.width = u64::MAX;
        assert_eq!(
            present.target.expected_bytes(),
            Err(ContractError::ArithmeticOverflow("present target bytes"))
        );
        assert_eq!(
            present.target.validate_shape(),
            Err(ContractError::ArithmeticOverflow("present target bytes"))
        );
    }

    #[test]
    fn present_refuses_a_target_format_outside_the_render_increment() {
        // `R32Uint` is expressible for texture symmetry but is not a colour
        // attachment, so it cannot be an inherited present target either.
        let mut present = present_descriptor();
        present.target.format = AttachmentFormat::R32Uint;
        let expected = ContractError::UnsupportedAttachmentFormat(AttachmentFormat::R32Uint);
        assert_eq!(present.validate(), Err(expected.clone()));
        let refusal = contract_error_refusal(expected);
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        assert_eq!(refusal.slug, "attachment_format_unsupported");
    }

    #[test]
    fn present_source_selects_one_of_several_attachments() {
        // Two attachments stand in for the multi-target pass a later increment
        // admits; the present must compare against the one its `source` names,
        // not against `color_attachments[0]`.
        let mut pass = render_pass();
        let mut second = render_attachment(AttachmentFormat::Bgra8Unorm);
        second.view_id = ViewId::new(31);
        second.allocation_id = AllocationId::new(32);
        pass.color_attachments.push(second);
        // The second attachment is 2x2 like the first, so the extent rule stays
        // out of the way of the source-selection rule under test.
        assert_eq!(pass.color_attachments.len(), 2);

        let mut present = present_descriptor();
        present.source = ViewId::new(31);
        present.target.view_id = ViewId::new(31);
        present.target.allocation_id = AllocationId::new(32);
        present.target.format = AttachmentFormat::Bgra8Unorm;
        present
            .validate_against(&pass)
            .expect("the named attachment is the one that must agree");

        // Same source, different target format: the refusal names the *second*
        // attachment's format, which is what makes this the control that proves
        // source selection rather than an accident of list order.
        present.target.format = AttachmentFormat::Rgba8Unorm;
        let expected = ContractError::PresentFormatMismatch {
            source: ViewId::new(31),
            target: AttachmentFormat::Rgba8Unorm,
            attachment: AttachmentFormat::Bgra8Unorm,
        };
        assert_eq!(present.validate_against(&pass), Err(expected));

        // The allocation disagreement is reported against the same named
        // attachment, so a target cannot borrow the first attachment's
        // allocation either.
        present.target.format = AttachmentFormat::Bgra8Unorm;
        present.target.allocation_id = AllocationId::new(22);
        let expected = ContractError::PresentTargetAllocationMismatch {
            source: ViewId::new(31),
            target: AllocationId::new(22),
            attachment: AllocationId::new(32),
        };
        assert_eq!(present.validate_against(&pass), Err(expected));

        // An extent disagreement is reported against the same named attachment,
        // so a wide target cannot pass by matching the first attachment instead.
        present.target.allocation_id = AllocationId::new(32);
        present.target.height = 4;
        let expected = ContractError::PresentExtentMismatch {
            source: ViewId::new(31),
            target: [2, 4],
            attachment: [2, 2],
        };
        assert_eq!(present.validate_against(&pass), Err(expected));

        // And the first attachment is still presentable on its own terms, so the
        // three refusals above are about the *named* attachment rather than
        // about the second one being unusable.
        let mut first = present_descriptor();
        first.source = ViewId::new(21);
        first.validate_against(&pass).unwrap();
    }

    #[test]
    fn present_refuses_a_target_that_restates_another_view_or_allocation() {
        // The target must restate the source attachment's identity; a view or an
        // allocation that names a different resource is a trace that describes
        // two resources where the first increment has one.
        let mut present = present_descriptor();
        present.target.view_id = ViewId::new(31);
        let expected = ContractError::PresentTargetViewMismatch {
            source: ViewId::new(21),
            target: ViewId::new(31),
        };
        assert_eq!(
            present.validate_against(&render_pass()),
            Err(expected.clone())
        );
        let refusal = contract_error_refusal(expected);
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "present_target_view_mismatch");

        let mut present = present_descriptor();
        present.target.allocation_id = AllocationId::new(32);
        let expected = ContractError::PresentTargetAllocationMismatch {
            source: ViewId::new(21),
            target: AllocationId::new(32),
            attachment: AllocationId::new(22),
        };
        assert_eq!(
            present.validate_against(&render_pass()),
            Err(expected.clone())
        );
        let refusal = contract_error_refusal(expected);
        assert_eq!(refusal.class, ProviderErrorClass::Args);
        assert_eq!(refusal.slug, "present_target_allocation_mismatch");

        // Control: the pass's own attachment identity is the one that agrees, so
        // both refusals are about the target's restatement and not about the
        // identity being unwritable.
        present_descriptor()
            .validate_against(&render_pass())
            .unwrap();
    }

    // Presentation contract, Step 2 (`research/docs/24` §4.1, §4.2): the render
    // pass carries the present action and admission gains the present gate.
    // Execution stays Step 3, so nothing here acquires or presents anything —
    // the tests pin the shape, the reuse of Step 1's refusals, and the
    // capability refusals that keep an unexecutable present from being run as an
    // offscreen render.

    /// The present action that hands `attachment` on, built from the Step 1
    /// fixture so the target restates the attachment's identity, format and
    /// extent exactly as `docs/24` §3.2 requires.
    fn present_for(attachment: &RenderAttachment) -> PresentDescriptor {
        let mut present = present_descriptor();
        present.target.view_id = attachment.view_id;
        present.target.allocation_id = attachment.allocation_id;
        present.target.format = attachment.format;
        present.target.width = attachment.width;
        present.target.height = attachment.height;
        present.source = attachment.view_id;
        present
    }

    /// The offscreen render trace the presenting trace is built from: the same
    /// two passes, with the render pass handing nothing on.
    fn offscreen_render_trace() -> ComputeTrace {
        attachment_trace(landing_view(7, 9), attachment_into(7, 9))
    }

    /// The smallest trace that needs the present bits: an admitted offscreen
    /// render trace whose render pass hands its own attachment on.
    fn presenting_trace() -> ComputeTrace {
        let mut value = offscreen_render_trace();
        let Some(TracePass::Render(pass)) = value.passes.last_mut() else {
            panic!("the fixture ends in a render pass");
        };
        pass.present = Some(present_for(&pass.color_attachments[0]));
        value
    }

    /// A snapshot that declares the four present bits (`docs/24` §4.2) on top of
    /// the render bits the fixture already declares.
    fn presenting_capabilities() -> ProviderCapabilities {
        let mut provider = render_capabilities();
        provider.supports_presentation = true;
        provider.max_present_targets = MAX_PRESENT_TARGETS as u32;
        provider.supported_present_modes = PresentMode::ADMITTED.to_vec();
        provider.max_present_image_count = MAX_PRESENT_IMAGE_COUNT;
        provider
    }

    #[test]
    fn present_capability_bits_default_to_unsupported() {
        // Every snapshot that predates this step keeps its exact behaviour: the
        // present bits stay at "cannot present", so no present action is
        // admitted and `MCC1` keeps sending the legacy capability bytes.
        let default = capabilities();
        assert!(!default.supports_presentation);
        assert_eq!(default.max_present_targets, 0);
        assert!(default.supported_present_modes.is_empty());
        assert_eq!(default.max_present_image_count, 0);
        assert!(!default.declares_presentation_support());
        assert!(!default.declares_render_support());

        // A render-declaring snapshot that never declared presentation keeps the
        // two halves from disagreeing silently: it renders and does not present.
        let render_only = render_capabilities();
        assert!(render_only.declares_render_support());
        assert!(!render_only.declares_presentation_support());

        // The present bits travel in the same extended payload as the render
        // bits, so a snapshot that declared presentation on its own still has to
        // ask for that payload (`docs/24` §4.2). This is the case that would
        // otherwise be written to the wire as the legacy bytes.
        let mut presenting_only = capabilities();
        presenting_only.supports_presentation = true;
        assert!(presenting_only.declares_presentation_support());
        assert!(presenting_only.declares_render_support());
    }

    #[test]
    fn present_trace_is_refused_before_any_resource_action_without_the_bit() {
        let trace = presenting_trace();
        let resources = landing_resources();

        // Control: the same trace without the present action admits, and the
        // present-bit snapshot admits the very same trace. The pair is what
        // proves the gate is the capability bit and not the trace shape.
        offscreen_render_trace()
            .validate()
            .expect("the offscreen fixture is well formed");
        render_capabilities()
            .admit(&offscreen_render_trace(), &resources)
            .expect("an offscreen render trace needs no present bit");
        presenting_capabilities()
            .admit(&trace, &resources)
            .expect("a declared present bit admits the presenting trace");

        // The refusal is a capability refusal, and it arrives with an empty
        // resource namespace: if the gate sat anywhere after the resource walk,
        // this call would report the missing allocation instead of the present
        // bit, which is exactly the "before any resource action" ordering
        // `docs/24` §4.2 asks for.
        let refusal = render_capabilities()
            .admit(&trace, &ResourceTableSnapshot::new())
            .unwrap_err();
        assert_eq!(refusal.slug, "present_targets_unsupported");
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        assert_eq!(refusal.phase, ProviderPhase::Resolve);
        assert_eq!(
            refusal.fields.get("targets"),
            Some(&FieldValue::Unsigned(1)),
            "the refusal has to carry the target count it refused"
        );

        // The trace's own contract is well formed, so the refusal above is the
        // capability gate rather than a shape error wearing a capability slug.
        trace
            .validate()
            .expect("the presenting trace is well formed");
        assert_eq!(
            render_capabilities()
                .admit(&trace, &resources)
                .unwrap_err()
                .slug,
            "present_targets_unsupported"
        );
    }

    #[test]
    fn present_capability_narrows_the_declared_modes_targets_and_images() {
        let trace = presenting_trace();
        let resources = landing_resources();

        // A snapshot that declares the bit but no mode refuses the trace's mode
        // by wire code, naming the pass that carried it.
        let mut no_modes = presenting_capabilities();
        no_modes.supported_present_modes.clear();
        let refusal = no_modes.admit(&trace, &resources).unwrap_err();
        assert_eq!(refusal.slug, "present_mode_unsupported");
        assert_eq!(
            refusal.fields.get("mode"),
            Some(&FieldValue::Unsigned(u64::from(PresentMode::Fifo.code())))
        );
        assert_eq!(
            refusal.fields.get("pass"),
            Some(&FieldValue::Unsigned(1)),
            "the refusal names the pass index that carried the present"
        );

        // The target count is a separate increment: the bit alone does not admit
        // an unbounded number of targets.
        let mut no_targets = presenting_capabilities();
        no_targets.max_present_targets = 0;
        let refusal = no_targets.admit(&trace, &resources).unwrap_err();
        assert_eq!(refusal.slug, "present_target_limit");
        assert_eq!(
            refusal.fields.get("maximum"),
            Some(&FieldValue::Unsigned(0))
        );

        // And the image count is the multi-buffering bound `docs/24` §3.4
        // defers, refused by the same declared-limits style.
        let mut no_images = presenting_capabilities();
        no_images.max_present_image_count = 0;
        let refusal = no_images.admit(&trace, &resources).unwrap_err();
        assert_eq!(refusal.slug, "present_image_count_limit");
        assert_eq!(
            refusal.fields.get("requested"),
            Some(&FieldValue::Unsigned(u64::from(MAX_PRESENT_IMAGE_COUNT)))
        );
    }

    #[test]
    fn render_pass_carries_its_present_and_reuses_the_step_one_refusals() {
        let mut value = presenting_trace();
        value
            .validate()
            .expect("the presenting trace is well formed");

        // The present is reachable through the trace and points at the render
        // pass's own attachment, which is what shape one buys: a present cannot
        // be attached to a pass that did not render its source.
        let actions = value.present_actions().collect::<Vec<_>>();
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].0, 1, "the render entry carries the present");
        assert_eq!(actions[0].1.source, ViewId::new(7));
        assert!(value.has_present_actions());

        // A source that the pass does not render into is refused by Step 1's own
        // rule, reached through the pass rather than restated here.
        let Some(TracePass::Render(pass)) = value.passes.last_mut() else {
            panic!("the fixture ends in a render pass");
        };
        pass.present
            .as_mut()
            .expect("the fixture carries a present")
            .source = ViewId::new(99);
        assert_eq!(
            value.validate(),
            Err(ContractError::PresentSourceUnknown {
                source: ViewId::new(99)
            })
        );
        assert_eq!(
            contract_error_refusal(ContractError::PresentSourceUnknown {
                source: ViewId::new(99)
            })
            .slug,
            "present_source_unknown"
        );

        // Control: the offscreen trace has no present action at all, so the
        // pre-present paths see exactly the trace they saw before this step.
        let offscreen = offscreen_render_trace();
        assert!(!offscreen.has_present_actions());
        assert_eq!(offscreen.present_actions().count(), 0);
        offscreen.validate().unwrap();
    }
}
