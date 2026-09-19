//! Minimal Vulkan executor for the source-level Metal compute facade.
//!
//! The first milestone accepts only ordinary Metal buffer arguments. It uses
//! metal2vulkan reflection as the descriptor contract and its exact-thread plan
//! as the dispatch contract; unsupported resources fail before any Vulkan work
//! is submitted.

use crate::readback_rect::ReadbackFallback;
use ash::ext::{device_fault, external_memory_host};
use ash::khr::shader_float_controls2;
use ash::{vk, Device as AshDevice, Entry, Instance};
use metal2vulkan::passes::{Stage, TransformOptions};
use metal2vulkan::reflect::{
    BufferExtent, BufferFootprint, BufferIndexSource, DescriptorLayout, KernelDispatch,
    KernelDispatchPlan, ResourceAccess, ResourceKind, ShaderReflection, ShaderStage,
    KERNEL_LOCAL_SIZE_SPEC_IDS,
};
use metal_api_core::completion::AbandonmentOutcome;
use metal_api_core::provider::{
    BorrowedLeaseRegistry, CompletionDisposition, FieldValue, LeaseId, PipelineContract,
    ProviderCapabilities, ProviderError, ProviderErrorClass, ProviderHealth, ProviderLifecycle,
    ProviderPhase, QueuePriority, QueueSchedulingPolicy, Retryability, SemanticDigest,
    TerminalRefusal, MAX_SERIAL_RESOURCES,
};
use metal_api_core::{
    AirSource, BufferBinding, BufferUpdate, ComputeExecutor, ComputeSubmission, ExecutorError,
    Function, PipelineArtifact,
};
use spirv::{BuiltIn, Capability, Decoration, Op};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, CString};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

mod compute_provider;
mod phase_profile;
mod provider;
mod readback_rect;
mod render;

pub use compute_provider::{
    CompiledComputePipeline, HeapPlacementObservation, IcbReplayObservation, RenderPipelineRequest,
    ResidentTargetRetirement, TranslatedRenderPipelineRequest, VulkanComputeProvider,
    PRESENT_TARGET_BUDGET, RESIDENT_TARGET_BUDGET,
};
pub use render::RenderStage;
pub use render::STAGE_BUFFER_NAMESPACE_SET;

/// The canonical descriptor layout for a folded pair of render stages
/// (`research/docs/23` §3.3, E-TX9).
///
/// One translated render stage's `[[buffer(n)]]` arguments land in the set this
/// layout names; every other resource class keeps the translator's own bands.
/// It is the arrangement a caller reaches for when two rendered stages each
/// read a `[[buffer(n)]]` argument with the same `n`: Metal's
/// `setVertexBuffer(_:offset:index:)` and `setFragmentBuffer(_:offset:index:)`
/// name independent index spaces, so translating both stages under the
/// translator's default layout folds the two descriptors onto one
/// `(set, binding)` and the rail refuses the pass by name
/// (`render_stage_buffer_layout_unsupported`).
///
/// The layout belongs to the *vertex* half of the pair and moves it to
/// [`STAGE_BUFFER_NAMESPACE_SET`] — the set the reviewed stage-buffer pair
/// already binds its vertex stage at, and still inside the rail's own pipeline
/// layout. The fragment half keeps the translator's default, so the rail's
/// fragment-only image path (which pins set 0) is untouched.
///
/// The caller picks the stage it translates with this layout; the rail reads
/// the slot back out of the returned reflection, so the layout is the module's
/// own Vulkan ABI rather than a second declaration that could disagree with it.
/// The shape the rail can execute is advertised as
/// [`metal_api_core::provider::ProviderCapabilities::supports_render_stage_buffer_namespace_split`].
pub fn stage_buffer_namespace_layout() -> DescriptorLayout {
    DescriptorLayout {
        set: STAGE_BUFFER_NAMESPACE_SET,
        ..DescriptorLayout::default()
    }
}

const FENCE_TIMEOUT_NS: u64 = 20_000_000_000;
const MAX_SERIAL_DISPATCHES: usize = 8;
static SCRATCH_SERIAL: AtomicU64 = AtomicU64::new(0);

/// The SPIR-V extension the translator emits beside `FloatControls2`.
///
/// `metal2vulkan` `43c46ac` decorates a floating-point result that withholds a
/// fast-math permission with `FPFastMathMode`, and the decoration demands
/// `OpCapability FloatControls2` together with this `OpExtension` name.
const SPV_KHR_FLOAT_CONTROLS2: &str = "SPV_KHR_float_controls2";

type EnqueueProbe = Arc<dyn Fn(usize) + Send + Sync>;

fn failure(message: impl Into<String>) -> ExecutorError {
    ExecutorError::new(message)
}

/// The device features the SPIR-V capability gate admits.
///
/// The gate used to be device-independent: the reviewed subset named
/// capabilities every admitted device has to enable (`Shader`, `ImageQuery`,
/// the `shaderInt8`/`shaderInt64` pair and the two sampled-image shapes), so one
/// whitelist could answer for all of them. `FloatControls2` is the first
/// capability a translation can demand that a device may or may not have, so it
/// is answered by the device that will execute the module: the provider derives
/// this policy from the selected device and hands it to both translation entry
/// points and to the render registration gate. A device without the feature
/// keeps the fail-closed phase-1 answer, byte for byte.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpirvFeaturePolicy {
    float_controls2: bool,
}

impl SpirvFeaturePolicy {
    /// The phase-1 subset: the capabilities every admitted device enables.
    pub const PHASE1: Self = Self {
        float_controls2: false,
    };

    /// Admit (or keep refusing) `FloatControls2` with `SPV_KHR_float_controls2`.
    pub const fn with_float_controls2(mut self, admitted: bool) -> Self {
        self.float_controls2 = admitted;
        self
    }

    /// Whether the gate admits `FloatControls2` + `SPV_KHR_float_controls2`.
    pub const fn float_controls2(self) -> bool {
        self.float_controls2
    }
}

/// What the selected device reported about `VK_KHR_shader_float_controls2`.
///
/// Two facts, kept apart because they are asked separately and can disagree:
/// the extension name is enumerated from the device's extension list, and
/// `shaderFloatControls2` is the feature query that answers whether the
/// capability is actually available. Only the conjunction — the pair the device
/// create info enables — admits the SPIR-V capability, so a device that
/// advertises the name with the bit off stays refused. The two readings are also
/// the evidence a host rail records for a device it cannot create here (R8).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FloatControls2Support {
    extension: bool,
    feature: bool,
}

impl FloatControls2Support {
    /// Whether the device enumerated `VK_KHR_shader_float_controls2` by name.
    pub const fn extension_present(self) -> bool {
        self.extension
    }

    /// Whether `VkPhysicalDeviceShaderFloatControls2FeaturesKHR` reports the
    /// `shaderFloatControls2` feature on.
    pub const fn feature_reported(self) -> bool {
        self.feature
    }

    /// Whether the device was created with the extension and feature enabled.
    pub const fn enabled(self) -> bool {
        self.extension && self.feature
    }

    /// The gate policy this support answers with.
    pub const fn policy(self) -> SpirvFeaturePolicy {
        SpirvFeaturePolicy::PHASE1.with_float_controls2(self.enabled())
    }
}

/// Admit one submission against a provider lifecycle and return the refusal
/// the provider boundary reports.
///
/// This is the whole mapping between the two crates. The core
/// [`ProviderLifecycle`] owns the terminal state and spells its refusal once
/// (`provider_unavailable` or `device_lost`, the `terminal` field, the
/// abandoned counters and `RetryAfterRecreate`); unwrapping it here keeps the
/// Vulkan side from re-encoding a slug, a field or a retryability that could
/// then drift from the contract. Nothing matches on message text.
fn terminal_refusal(lifecycle: &ProviderLifecycle) -> Result<(), ProviderError> {
    lifecycle.admit().map_err(TerminalRefusal::into_error)
}

/// Device queues created per selected queue family.
///
/// Four queues are enough to demonstrate independent in-flight work without
/// over-subscribing drivers whose family reports many queues. A family with a
/// single queue (Lavapipe) keeps the previous single-queue behaviour.
const MAX_QUEUES_PER_FAMILY: usize = 4;

/// Total device queues created across the primary and dedicated compute
/// families. Compute-only queues can overlap with graphics work on drivers
/// that expose a separate family, so the scheduler may use up to two families.
const MAX_DEVICE_QUEUES: usize = 8;

/// Abandonment budget of one Vulkan device.
///
/// The direct executor and the provider share one context, so they share one
/// bound: the first submission whose completion can no longer be observed ends
/// the instance, which then has to be recreated (`docs/PROVIDER-B1.md` §7).
const ABANDONMENT_BUDGET_SUBMISSIONS: u64 = 1;

/// Byte bound of the same budget. Bytes are accounted but never refunded,
/// because abandoned device memory cannot be returned safely.
const ABANDONMENT_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

/// A driver boundary whose answer a test may substitute.
///
/// Production reaches both points through `VulkanContext::submit_commands` and
/// `VulkanContext::wait_for_fence`, and both answer a
/// `VK_ERROR_DEVICE_LOST` the same way: the loss goes through the core
/// `ProviderLifecycle`. CI cannot make a live driver lose its device, so the
/// substitution replaces the driver's answer at exactly one of those two
/// boundaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceLossPoint {
    /// `vkQueueSubmit` answers `VK_ERROR_DEVICE_LOST`.
    Submit,
    /// `vkWaitForFences` answers `VK_ERROR_DEVICE_LOST`.
    Wait,
}

/// One `VkDeviceFaultAddressInfoEXT` record.
///
/// [`address_type`](Self::address_type) keeps the raw
/// `VK_DEVICE_FAULT_ADDRESS_TYPE_*` value; the provider error spells the same
/// enum as a name (`READ_INVALID`, `WRITE_INVALID`, ...) so a caller can read
/// the fault without matching Vulkan enums.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceFaultAddress {
    pub address_type: i32,
    pub reported_address: u64,
    pub address_precision: u64,
}

/// Diagnostic snapshot of one `VK_EXT_device_fault` query.
///
/// The extension is optional and the record is evidence, never a gate: a
/// device that does not advertise it answers
/// `extension_present == false` with no addresses, and a device that does but
/// refuses the query answers the same way. Neither case changes the loss
/// itself, which the provider reports through the core lifecycle regardless.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceFaultSnapshot {
    /// Whether the physical device advertised `VK_EXT_device_fault`.
    pub extension_present: bool,
    /// Driver-supplied fault description, when the driver wrote one.
    pub description: Option<String>,
    /// Addresses the driver reported, in the order it returned them.
    pub addresses: Vec<DeviceFaultAddress>,
    /// Number of vendor fault records the driver reported.
    pub vendor_info_count: u32,
    /// Size in bytes of the vendor fault binary the driver can return.
    pub vendor_binary_size: u64,
}

impl DeviceFaultSnapshot {
    /// The record for a device that could not answer the fault query.
    ///
    /// `advertised` is the physical device's answer to "does the extension
    /// exist at all", so the bit survives a missing entry point: a device that
    /// offers `VK_EXT_device_fault` but never hands the query over answers this
    /// shape unchanged, and the evidence still tells "the extension is absent"
    /// from "the query could not run". Description and counts stay empty until
    /// the driver writes them.
    pub(crate) fn unavailable(advertised: bool) -> Self {
        Self {
            extension_present: advertised,
            description: None,
            addresses: Vec::new(),
            vendor_info_count: 0,
            vendor_binary_size: 0,
        }
    }

    /// The structured fields a device-loss error carries for this record.
    ///
    /// Addresses are capped at [`DEVICE_FAULT_ADDRESS_FIELDS`]; the count
    /// field always reports how many the driver returned, so a truncated
    /// listing is never mistaken for a complete one.
    pub(crate) fn evidence_fields(&self) -> Vec<(String, FieldValue)> {
        let mut fields = Vec::with_capacity(5 + 3 * DEVICE_FAULT_ADDRESS_FIELDS);
        fields.push((
            "device_fault_extension".to_owned(),
            FieldValue::Bool(self.extension_present),
        ));
        if let Some(description) = &self.description {
            fields.push((
                "device_fault_description".to_owned(),
                FieldValue::Text(description.clone()),
            ));
        }
        fields.push((
            "device_fault_addresses".to_owned(),
            FieldValue::Unsigned(self.addresses.len() as u64),
        ));
        for (index, address) in self
            .addresses
            .iter()
            .take(DEVICE_FAULT_ADDRESS_FIELDS)
            .enumerate()
        {
            fields.push((
                format!("device_fault_address_type_{index}"),
                FieldValue::Text(device_fault_address_type_name(address.address_type).to_owned()),
            ));
            fields.push((
                format!("device_fault_address_{index}"),
                FieldValue::Unsigned(address.reported_address),
            ));
            fields.push((
                format!("device_fault_address_precision_{index}"),
                FieldValue::Unsigned(address.address_precision),
            ));
        }
        fields.push((
            "device_fault_vendor_infos".to_owned(),
            FieldValue::Unsigned(u64::from(self.vendor_info_count)),
        ));
        fields.push((
            "device_fault_vendor_binary_size".to_owned(),
            FieldValue::Unsigned(self.vendor_binary_size),
        ));
        fields
    }
}

/// Upper bound of fault addresses copied into one provider error.
const DEVICE_FAULT_ADDRESS_FIELDS: usize = 4;

/// The one `VK_EXT_device_fault` entry point, named the way ash's generated
/// loader names it.
const GET_DEVICE_FAULT_INFO_EXT: &CStr = c"vkGetDeviceFaultInfoEXT";

/// The `vkGetDeviceFaultInfoEXT` entry point the created device hands over.
///
/// `device_fault::Device::new` cannot answer this question on its own: with
/// ash's `loaded` feature a missing entry point becomes a callable stub that
/// panics from inside an `extern "system"` frame — a panic the process cannot
/// unwind — so the probe asks `vkGetDeviceProcAddr` the same question the
/// generated loader asks, with the same command name.
fn device_fault_entry_point(instance: &Instance, device: &AshDevice) -> vk::PFN_vkVoidFunction {
    unsafe { instance.get_device_proc_addr(device.handle(), GET_DEVICE_FAULT_INFO_EXT.as_ptr()) }
}

/// Whether the fault loader may be built for a created device.
///
/// Enumerating a device extension is not enabling it: the spec lets
/// `vkGetDeviceProcAddr` answer NULL for a device-level command whose extension
/// the device was not created with, and the Windows RTX 5060 ICD does exactly
/// that (2026-09-18). Driving the query there aborts the process, so the loader
/// is only built when the device really hands the entry point over; a device
/// that advertised the extension but cannot is still reported as advertised,
/// with no addresses ([`DeviceFaultSnapshot::unavailable`]).
fn device_fault_loader_ready(advertised: bool, entry_point: vk::PFN_vkVoidFunction) -> bool {
    advertised && entry_point.is_some()
}

/// Vulkan name of one `VkDeviceFaultAddressTypeEXT` value.
fn device_fault_address_type_name(address_type: i32) -> &'static str {
    match vk::DeviceFaultAddressTypeEXT::from_raw(address_type) {
        vk::DeviceFaultAddressTypeEXT::NONE => "NONE",
        vk::DeviceFaultAddressTypeEXT::READ_INVALID => "READ_INVALID",
        vk::DeviceFaultAddressTypeEXT::WRITE_INVALID => "WRITE_INVALID",
        vk::DeviceFaultAddressTypeEXT::EXECUTE_INVALID => "EXECUTE_INVALID",
        vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_UNKNOWN => "INSTRUCTION_POINTER_UNKNOWN",
        vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_INVALID => "INSTRUCTION_POINTER_INVALID",
        vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_FAULT => "INSTRUCTION_POINTER_FAULT",
        _ => "UNKNOWN",
    }
}

/// Vulkan name of one raw result, for the `vk_result` evidence field.
fn vk_result_name(result: vk::Result) -> String {
    match result {
        vk::Result::SUCCESS => "VK_SUCCESS".to_owned(),
        vk::Result::NOT_READY => "VK_NOT_READY".to_owned(),
        vk::Result::TIMEOUT => "VK_TIMEOUT".to_owned(),
        vk::Result::ERROR_OUT_OF_HOST_MEMORY => "VK_ERROR_OUT_OF_HOST_MEMORY".to_owned(),
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => "VK_ERROR_OUT_OF_DEVICE_MEMORY".to_owned(),
        vk::Result::ERROR_DEVICE_LOST => "VK_ERROR_DEVICE_LOST".to_owned(),
        vk::Result::ERROR_UNKNOWN => "VK_ERROR_UNKNOWN".to_owned(),
        other => format!("VK_RESULT_{}", other.as_raw()),
    }
}

/// Attach a `VK_EXT_device_fault` record to a device-loss error.
pub(crate) fn with_device_fault_evidence(
    mut error: ProviderError,
    fault: &DeviceFaultSnapshot,
) -> ProviderError {
    for (key, value) in fault.evidence_fields() {
        error = error.with_field(key, value);
    }
    error
}

/// Structured provider error for a `VK_ERROR_DEVICE_LOST` observed at a driver
/// boundary.
///
/// Every rail that enqueues through `VulkanContext::submit_commands` or waits
/// through `VulkanContext::wait_for_fence` reports a loss through here,
/// so no boundary can answer a lost device as an ordinary execution failure:
/// the core lifecycle is marked lost (leases included) and the error carries
/// the raw result plus the `VK_EXT_device_fault` record. `slug` stays the
/// boundary's own, so a caller can still tell which step reported the loss.
pub(crate) fn device_loss_refusal(
    context: &VulkanContext,
    phase: ProviderPhase,
    slug: &'static str,
    step: &str,
) -> ProviderError {
    let result = vk::Result::ERROR_DEVICE_LOST;
    let error = ExecutionFailure::vulkan(result, format!("{step}: {result}")).into_provider(
        phase,
        ProviderErrorClass::Execute,
        slug,
        CompletionDisposition::DeviceLost { token: None },
    );
    with_device_fault_evidence(error, &context.observe_device_loss())
}

/// Native Vulkan implementation of the Phase 1 compute subset.
/// Cumulative render-readback regions of the written-rect increment
/// (`docs/WRITTEN-RECT-READBACK.md` §2).
///
/// Every stored colour attachment the render half publishes is read back one of
/// two ways: through the rectangle this pass can have written — in which case
/// the rest of the frame is the seed the rail already holds host-side — or
/// whole, exactly as the pre-increment rail did. The counters are cumulative
/// over the executor's life and always on, so a round can report the split
/// beside the phase profile and an e2e oracle can tell which arm a shape took
/// without turning the profile on.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReadbackRegionCounts {
    /// Stored attachments read back through their written rectangle.
    pub rect_attachments: usize,
    /// Bytes those readbacks copied out of the device mapping.
    pub rect_bytes: usize,
    /// Bytes the same attachments' whole extents occupy: what the
    /// whole-attachment readback would have copied instead.
    pub rect_extent_bytes: usize,
    /// Stored attachments read back whole.
    pub full_attachments: usize,
    /// Bytes those whole readbacks copied.
    pub full_bytes: usize,
    /// Whole readbacks the `METAL_API_VULKAN_FULL_READBACK` control switch asked
    /// for.
    pub switch_attachments: usize,
    /// Whole readbacks of shapes whose seed this rail does not hold host-side
    /// (a multisampled raster, a resident load, an undefined load).
    pub shape_attachments: usize,
    /// Whole readbacks whose declared viewport or scissor the rail cannot prove
    /// (empty, or reaching outside the attachment's extent).
    pub bounds_attachments: usize,
    /// Whole readbacks whose written rectangle covers the whole attachment, so
    /// narrowing it saves nothing.
    pub whole_attachments: usize,
}

pub struct VulkanExecutor {
    context: Arc<VulkanContext>,
}

impl VulkanExecutor {
    pub fn new() -> Result<Arc<Self>, ExecutorError> {
        Ok(Arc::new(Self {
            context: Arc::new(VulkanContext::new()?),
        }))
    }

    pub fn device_name(&self) -> &str {
        &self.context.device_name
    }

    /// What this device reported about `VK_KHR_shader_float_controls2` (R8).
    ///
    /// The two readings — extension name and `shaderFloatControls2` — are the
    /// capability snapshot a host rail records, and
    /// [`Self::spirv_feature_policy`] is the gate answer derived from them. A
    /// device that reports neither keeps the phase-1 subset.
    pub fn float_controls2_support(&self) -> FloatControls2Support {
        self.context.float_controls2_support()
    }

    /// Whether this device was created with `samplerMirrorClampToEdge`
    /// enabled (`research/docs/23` §109).
    ///
    /// The one device fact a family address mode needs: a rail creates
    /// `MIRROR_CLAMP_TO_EDGE` samplers only when this answers `true`, so the
    /// reading a host records for a device it cannot create here is the same
    /// bit the creation refused on.
    pub fn supports_sampler_mirror_clamp_to_edge(&self) -> bool {
        self.context.sampler_mirror_clamp_to_edge()
    }

    /// The SPIR-V capability policy this device answers with (R8).
    ///
    /// [`TranslatedComputePipeline::translate_with_policy`] and
    /// [`TranslatedRenderStage::translate_with_policy`] take it, and the
    /// provider's own translation paths pass it, so a module a caller translated
    /// against this device is the module its gate validates.
    pub fn spirv_feature_policy(&self) -> SpirvFeaturePolicy {
        self.context.spirv_feature_policy()
    }

    /// Report the selected Vulkan device as neutral provider capabilities.
    ///
    /// The snapshot executor exposes owned host bytes and synchronous
    /// readback. `VulkanComputeProvider` adds staged and, when
    /// `VK_EXT_external_memory_host` is present, borrowed no-copy leases.
    pub fn provider_capabilities(&self) -> ProviderCapabilities {
        let mut capabilities = provider::capabilities_from_limits(&self.context.properties.limits);
        // The two depth-resolve bits are the device's own answer, overlaid on
        // the limits-derived snapshot: the mask carries exactly the admitted
        // filters the device reports, and the capability bit is the mask's
        // non-empty form, so a device without any admitted filter keeps both
        // bits fail-closed (`research/docs/23` §3.3, v57).
        capabilities.depth_resolve_modes =
            provider::depth_resolve_mode_mask(self.context.depth_resolve_modes);
        capabilities.supports_render_depth_resolve = capabilities.depth_resolve_modes != 0;
        // The two stencil-resolve bits are the device's own answer, overlaid
        // the same way the depth pair is: the mask carries exactly the
        // admitted filters the device reports, and the capability bit is the
        // mask's non-empty form (`research/docs/23` §3.3, v60). Vulkan has no
        // stencil mode for Metal's depthResolvedSample, so only the Sample0
        // bit can ever appear.
        capabilities.stencil_resolve_modes =
            provider::stencil_resolve_mode_mask(self.context.stencil_resolve_modes);
        capabilities.supports_render_stencil_resolve = capabilities.stencil_resolve_modes != 0;
        capabilities
    }

    /// The reviewed 2/4/8 sample counts the device's framebuffer admits, as
    /// the contract-code bitmask `render.rs` derives from the device limits:
    /// bit `i` = `SampleCount` code `i` (`research/docs/23` §3.3, v61).
    ///
    /// The capture runner reads the device-gated sample-count cases against
    /// this mask, because the snapshot's single ceiling cannot say "8x yes,
    /// 2x no" — Lavapipe is exactly that device.
    pub fn render_sample_count_mask(&self) -> u32 {
        crate::render::limits_render_sample_count_mask(&self.context.properties.limits)
    }

    /// Simulate a confirmed device loss for lifecycle tests.
    ///
    /// CI cannot produce a deterministic `VK_ERROR_DEVICE_LOST`, so this hook
    /// marks the shared context lost: `health` becomes `DeviceLost`, new work
    /// is refused with `RetryAfterRecreate`, and still-submitted resources are
    /// destroyed instead of retained. It does not replace a real device-loss
    /// run and must not be used outside tests.
    #[doc(hidden)]
    pub fn inject_device_loss_for_test(&self) {
        self.context.mark_device_lost();
    }

    /// Substitute the next driver answer at one queue boundary with
    /// `VK_ERROR_DEVICE_LOST`.
    ///
    /// Unlike [`Self::inject_device_loss_for_test`], which marks the lifecycle
    /// directly, this hook does not touch the lifecycle at all: the context
    /// reaches `DeviceLost` only through the path a real driver answer takes,
    /// so the test observes the provider's reaction (`vk::Result` evidence,
    /// `VK_EXT_device_fault` query, terminal transition, lease retirement and
    /// the refusal on later submissions) rather than its own setup. Tests only.
    #[doc(hidden)]
    pub fn inject_driver_device_loss_for_test(&self, point: DeviceLossPoint) {
        self.context.arm_driver_loss_injection(point);
    }

    /// Diagnostic `VK_EXT_device_fault` record of the last observed device
    /// loss.
    ///
    /// `None` until a loss is observed. A device without the extension still
    /// answers a snapshot, with `extension_present == false`.
    #[doc(hidden)]
    pub fn last_device_fault(&self) -> Option<DeviceFaultSnapshot> {
        self.context.last_device_fault()
    }

    /// Number of device queues created across the primary and dedicated
    /// compute families.
    pub fn queue_count(&self) -> usize {
        self.context.queue_count()
    }

    /// Host-side scheduling tier installed on each device queue.
    ///
    /// The tier is a provider scheduling attribute, not a
    /// `VkDeviceQueueCreateInfo::pQueuePriorities` value: Vulkan fixes queue
    /// priorities at device creation, so the provider expresses priority in its
    /// own scheduler (`research/docs/21` §2). Every queue starts at
    /// [`QueuePriority::Default`].
    pub fn queue_priorities(&self) -> Vec<QueuePriority> {
        self.context.queue_priorities()
    }

    /// Mark each device queue with a host-side scheduling tier.
    ///
    /// `tiers[i]` is the tier of device queue `i`, so the slice must have
    /// exactly [`Self::queue_count`] entries: a differently sized table is
    /// refused rather than padded, which keeps "queue `i` has tier
    /// `tiers[i]`" true for every later submission. The table is read on each
    /// queue selection, and with every queue at [`QueuePriority::Default`] that
    /// selection is the previous least-loaded rule. Installing a tier only
    /// changes which device queue carries the submission: dependency order,
    /// reservations and writeback results are unaffected.
    pub fn set_queue_priorities(&self, tiers: &[QueuePriority]) -> Result<(), ExecutorError> {
        self.context.set_queue_priorities(tiers).map_err(failure)
    }

    /// Number of distinct device queue families used by the scheduler.
    #[doc(hidden)]
    pub fn queue_family_count(&self) -> usize {
        self.context.queue_family_count()
    }

    /// Cumulative device-buffer copy-in / copy-out operations. Smoke tests use
    /// it to prove that several views of one allocation share one copy.
    #[doc(hidden)]
    pub fn buffer_copy_counts(&self) -> (usize, usize) {
        self.context.buffer_copy_counts()
    }

    /// Cumulative device-buffer copy-in / copy-out bytes. Smoke tests use it
    /// to prove that a view that cannot read copies nothing in
    /// (`research/docs/15` step 4).
    #[doc(hidden)]
    pub fn buffer_copy_bytes(&self) -> (usize, usize) {
        self.context.buffer_copy_bytes()
    }

    /// Cumulative present acquire / present completions of the presentation
    /// rail. Smoke tests use it to prove a presenting case reports one of each.
    #[doc(hidden)]
    pub fn present_counts(&self) -> (usize, usize) {
        self.context.present_counts()
    }

    /// Cumulative render-readback regions: how many stored attachments were
    /// read back through their written rectangle, how many bytes that cost, and
    /// how many took the whole-extent path with the reason
    /// (`docs/WRITTEN-RECT-READBACK.md` §2).
    ///
    /// The counters are always on and cumulative, so a comparison needs the
    /// difference between two readings — which is what
    /// `METAL_API_VULKAN_FULL_READBACK` lets one process observe: the same
    /// shape runs twice, once per arm.
    #[doc(hidden)]
    pub fn readback_region_counts(&self) -> ReadbackRegionCounts {
        self.context.readback_regions()
    }

    /// Successful submissions recorded per device queue.
    #[doc(hidden)]
    pub fn queue_submission_counts(&self) -> Vec<usize> {
        self.context.queue_submission_counts()
    }

    /// Cumulative queue selections the scheduler made, one per device queue.
    ///
    /// This is the allocation observation the policy exposes without a
    /// test-only probe: every submit path reports its selected queue here,
    /// even when the driver then refuses the submission. It pairs with
    /// [`Self::queue_submission_counts`], which counts only confirmed
    /// submissions, and [`Self::queue_completion_counts`], which counts
    /// retirements (`research/docs/21` §6).
    #[doc(hidden)]
    pub fn queue_enqueue_counts(&self) -> Vec<usize> {
        self.context.queue_enqueue_counts()
    }

    /// Cumulative queue retirements, one per device queue.
    ///
    /// Every confirmed submission eventually retires, so on a healthy device
    /// this converges to [`Self::queue_submission_counts`]; the gap between the
    /// two surfaces is the work the queue still holds in flight
    /// (`research/docs/21` §6).
    #[doc(hidden)]
    pub fn queue_completion_counts(&self) -> Vec<usize> {
        self.context.queue_completion_counts()
    }

    /// Install a probe called with the selected queue index while that queue's
    /// host enqueue lock is held. Smoke tests use it to prove that independent
    /// queues enqueue concurrently; production callers leave it unset.
    #[doc(hidden)]
    pub fn set_enqueue_probe_for_test(&self, probe: EnqueueProbe) {
        if let Ok(mut slot) = self.context.enqueue_probe.lock() {
            *slot = Some(probe);
        }
    }

    /// Remove a probe installed by [`Self::set_enqueue_probe_for_test`].
    #[doc(hidden)]
    pub fn clear_enqueue_probe_for_test(&self) {
        if let Ok(mut slot) = self.context.enqueue_probe.lock() {
            *slot = None;
        }
    }
}

pub(crate) struct VulkanPipelineArtifact {
    context: Arc<VulkanContext>,
    translated: TranslatedComputePipeline,
}

/// A validated AIR-to-SPIR-V compute pipeline with no Vulkan device objects.
///
/// This is the shared translation boundary used by both the standalone
/// executor and adapters targeting an existing Vulkan engine. Constructing it
/// runs metal2vulkan and validates the deliberately narrow buffer-compute
/// contract, but does not create an instance, device, queue, or pipeline.
pub struct TranslatedComputePipeline {
    spv: Vec<u8>,
    reflection: ShaderReflection,
}

impl TranslatedComputePipeline {
    /// Translate under the phase-1 capability subset.
    ///
    /// The subset every admitted device enables is the fail-closed default: a
    /// caller holding the device hands its own answer to
    /// [`Self::translate_with_policy`] instead.
    pub fn translate(function: &Function) -> Result<Self, ExecutorError> {
        Self::translate_with_policy(function, SpirvFeaturePolicy::PHASE1)
    }

    /// Translate under one device's capability policy.
    ///
    /// The policy is the device's own answer (`SpirvFeaturePolicy`), and it
    /// only ever opens the capabilities that device enabled: every other shape
    /// of the module is checked exactly as [`Self::translate`] checks it.
    pub fn translate_with_policy(
        function: &Function,
        policy: SpirvFeaturePolicy,
    ) -> Result<Self, ExecutorError> {
        let options = TransformOptions {
            kernel_local_size: [1, 1, 1],
            kernel_dispatch: Some(KernelDispatch::safe_default()),
            ..TransformOptions::default()
        };
        let scratch = ScratchDir::new()?;
        let translated = match function.air_source() {
            AirSource::SanitizedLl(source) => metal2vulkan::translate_sanitized_native_reflected(
                source,
                Stage::Kernel,
                scratch.path(),
                options,
            ),
            AirSource::Binary(source) => {
                let input = scratch.path().join("input.air");
                std::fs::write(&input, source).map_err(|error| {
                    failure(format!(
                        "write binary AIR scratch {}: {error}",
                        input.display()
                    ))
                })?;
                let input = input
                    .to_str()
                    .ok_or_else(|| failure("binary AIR scratch path is not valid UTF-8"))?;
                metal2vulkan::translate_reflected_with_options(
                    input,
                    Stage::Kernel,
                    scratch.path(),
                    options,
                )
            }
        };
        let (spv, reflection) = translated
            .map_err(|error| failure(format!("translate {}: {error}", function.name())))?;
        validate_spirv_capabilities(&spv, policy)?;
        validate_pipeline_reflection(function.name(), &reflection)?;
        Ok(Self { spv, reflection })
    }

    pub fn spirv(&self) -> &[u8] {
        &self.spv
    }

    pub fn reflection(&self) -> &ShaderReflection {
        &self.reflection
    }

    pub fn validate_buffers(
        &self,
        buffers: &[BufferBinding],
        threads_per_grid: [u32; 3],
    ) -> Result<(), ExecutorError> {
        let widths = buffers
            .iter()
            .map(|binding| (binding.index, binding.bytes.len()))
            .collect::<Vec<_>>();
        self.validate_binding_widths(&widths, threads_per_grid)
    }

    /// Validate `(Metal binding index, byte length)` pairs. No-copy bindings
    /// have a host pointer instead of owned bytes, so width is the only
    /// property shared with `BufferBinding`.
    pub(crate) fn validate_binding_widths(
        &self,
        widths: &[(u32, usize)],
        threads_per_grid: [u32; 3],
    ) -> Result<(), ExecutorError> {
        validate_bound_buffers(&self.reflection, widths, threads_per_grid)
    }

    pub fn validate_threadgroup(&self, local_size: [u32; 3]) -> Result<(), ExecutorError> {
        if let Some(maximum) = self.reflection.max_work_group_size {
            let total = local_size
                .into_iter()
                .try_fold(1_u32, u32::checked_mul)
                .ok_or_else(|| failure("threadgroup invocation count overflows u32"))?;
            if total > maximum {
                return Err(failure(format!(
                    "threadgroup has {total} invocations but AIR permits at most {maximum}"
                )));
            }
        }
        Ok(())
    }

    /// Map this translated kernel to the neutral provider contract. Vulkan
    /// descriptor locations and exact-thread regions stay implementation-only.
    pub fn provider_contract(
        &self,
        translator_revision: Option<SemanticDigest>,
    ) -> Result<PipelineContract, ExecutorError> {
        provider::pipeline_contract(&self.reflection, translator_revision)
    }
}

/// One translated render stage: the SPIR-V module and the reflection of the AIR
/// it came from.
///
/// [`TranslatedComputePipeline`] is this value's compute sibling. Both run the
/// same translator over the same narrow input set (sanitized LLVM IR or one raw
/// or offset-zero-wrapped AIR module), but they answer different questions: a
/// compute pipeline owns the dispatch contract, while a render stage is one half
/// of a graphics pipeline, so it is checked against a host's
/// [`RenderPipelineContract`](metal_api_core::provider::RenderPipelineContract)
/// when the pair is registered
/// ([`VulkanComputeProvider::register_translated_render_pipeline`]).
///
/// The stage the caller names is the one the translator is asked for
/// ([`RenderStage`]), not the one the AIR happens to declare: a stage whose
/// reflection reports a different one is refused here, at translation, instead
/// of reaching a registration that would describe it wrongly.
pub struct TranslatedRenderStage {
    pub(crate) stage: RenderStage,
    pub(crate) spirv: Vec<u8>,
    pub(crate) reflection: ShaderReflection,
}

impl TranslatedRenderStage {
    /// Translate one render stage from `function`'s AIR.
    ///
    /// `stage` names the stage to translate; `function` carries the module's
    /// AIR and the entry name the pipeline table reports. The returned module is
    /// validated for the SPIR-V capabilities this rail enables, exactly as the
    /// compute path is, so a module this provider could not create is refused
    /// before it reaches a registration.
    ///
    /// Translate one render stage under the phase-1 capability subset.
    ///
    /// The subset every admitted device enables is the fail-closed default: a
    /// caller holding the device hands its own answer to
    /// [`Self::translate_with_policy`] instead.
    pub fn translate(stage: RenderStage, function: &Function) -> Result<Self, ExecutorError> {
        Self::translate_with_policy(stage, function, SpirvFeaturePolicy::PHASE1)
    }

    /// Translate one render stage under one device's capability policy.
    ///
    /// The policy is the device's own answer
    /// ([`VulkanExecutor::spirv_feature_policy`] /
    /// [`VulkanComputeProvider::spirv_feature_policy`]), and the provider
    /// re-asks it at registration, so a module translated against another
    /// device's policy is refused where the pipeline would be minted rather
    /// than where the module is decoded.
    ///
    /// The descriptor layout is the translator's default — every Metal
    /// resource in set 0. A caller that wants the two stages' buffer
    /// namespaces to stay apart hands its own layout to
    /// [`Self::translate_with_policy_and_layout`].
    pub fn translate_with_policy(
        stage: RenderStage,
        function: &Function,
        policy: SpirvFeaturePolicy,
    ) -> Result<Self, ExecutorError> {
        Self::translate_with_policy_and_layout(stage, function, policy, DescriptorLayout::default())
    }

    /// Translate one render stage under one device's capability policy and one
    /// descriptor layout.
    ///
    /// The layout is the Vulkan ABI the module is emitted against: it decides
    /// which set and binding every `[[buffer(n)]]` argument lands in, and the
    /// rail reads that slot back out of the returned reflection when it binds
    /// the pass's stage buffers (`research/docs/23` §3.3, v84). The translator's
    /// default puts every Metal resource in set 0, which is all one stage
    /// needs; a caller whose two stages read `[[buffer(N)]]` arguments wants
    /// the reviewed stage-buffer pair's arrangement instead — the vertex
    /// stage's buffers in set 1, the fragment stage's in set 2 — because the
    /// two stages' Metal buffer index spaces are independent
    /// (`setVertexBuffer(_:offset:index:)` and
    /// `setFragmentBuffer(_:offset:index:)`), so one set cannot carry both
    /// without one stage's slot overwriting the other's.
    pub fn translate_with_policy_and_layout(
        stage: RenderStage,
        function: &Function,
        policy: SpirvFeaturePolicy,
        descriptor_layout: DescriptorLayout,
    ) -> Result<Self, ExecutorError> {
        // The kernel options do not apply to a graphics stage: `TransformOptions`
        // defaults carry the API's own defaults (amplification 1, no sampled
        // raster count, no specialized sampler), and the render rail supplies
        // its pipeline state itself. The descriptor layout is the one option
        // that does apply — it is the module's own Vulkan ABI, not pipeline
        // state.
        let options = TransformOptions::default()
            .with_descriptor_layout(descriptor_layout)
            .map_err(|error| {
                failure(format!(
                    "translate {} {}: descriptor layout: {error}",
                    stage.name(),
                    function.name()
                ))
            })?;
        let scratch = ScratchDir::new()?;
        let translated = match function.air_source() {
            AirSource::SanitizedLl(source) => metal2vulkan::translate_sanitized_native_reflected(
                source,
                stage.translator_stage(),
                scratch.path(),
                options,
            ),
            AirSource::Binary(source) => {
                let input = scratch.path().join("input.air");
                std::fs::write(&input, source).map_err(|error| {
                    failure(format!(
                        "write binary AIR scratch {}: {error}",
                        input.display()
                    ))
                })?;
                let input = input
                    .to_str()
                    .ok_or_else(|| failure("binary AIR scratch path is not valid UTF-8"))?;
                metal2vulkan::translate_reflected_with_options(
                    input,
                    stage.translator_stage(),
                    scratch.path(),
                    options,
                )
            }
        };
        let (spirv, reflection) = translated.map_err(|error| {
            failure(format!(
                "translate {} {}: {error}",
                stage.name(),
                function.name()
            ))
        })?;
        validate_spirv_capabilities(&spirv, policy)?;
        // Metal's clip space is +y up and Vulkan's is +y down, so a translated
        // vertex module that writes its position unchanged would rasterize a
        // vertically mirrored frame. The reviewed `render_spv/*.vert.spvasm`
        // modules negate the position's y by hand (v38, `research/docs/23`
        // §32); a module that came out of the translator gets the same
        // alignment here (`research/docs/23` §40). Only the translated vertex
        // arm passes through this function, so the hand-written modules and
        // the native (macOS Metal) rail keep their own conventions.
        let spirv = if stage == RenderStage::Vertex {
            negate_position_y(&spirv)?
        } else {
            spirv
        };
        if reflection.stage != stage.reflected_stage() {
            return Err(failure(format!(
                "translate {} {}: the reflection reports stage {:?}",
                stage.name(),
                function.name(),
                reflection.stage
            )));
        }
        Ok(Self {
            stage,
            spirv,
            reflection,
        })
    }

    /// The stage this module was translated for.
    pub const fn stage(&self) -> RenderStage {
        self.stage
    }

    /// The translated SPIR-V module.
    pub fn spirv(&self) -> &[u8] {
        &self.spirv
    }

    /// The reflection of the AIR the module was translated from.
    pub fn reflection(&self) -> &ShaderReflection {
        &self.reflection
    }
}

impl ComputeExecutor for VulkanExecutor {
    fn new_compute_pipeline(&self, function: &Function) -> Result<PipelineArtifact, ExecutorError> {
        self.context.ensure_usable()?;
        let translated = TranslatedComputePipeline::translate_with_policy(
            function,
            self.context.spirv_feature_policy(),
        )?;
        self.context.ensure_usable()?;
        Ok(Arc::new(VulkanPipelineArtifact {
            context: Arc::clone(&self.context),
            translated,
        }))
    }

    fn execute(&self, submission: ComputeSubmission) -> Result<Vec<BufferUpdate>, ExecutorError> {
        self.context.ensure_usable()?;
        let artifact = Arc::downcast::<VulkanPipelineArtifact>(Arc::clone(&submission.pipeline))
            .map_err(|_| failure("pipeline artifact is not a Vulkan compute pipeline"))?;
        if !Arc::ptr_eq(&artifact.context, &self.context) {
            return Err(failure(
                "pipeline artifact belongs to another Vulkan device",
            ));
        }
        // The synchronous path selects through the same priority/fairness
        // policy as the deferred object path (`research/docs/21` §4). A
        // single-queue device still answers zero, so this changes nothing on a
        // one-queue family like Lavapipe.
        let queue_index = self.context.pick_queue();
        let _execution = self
            .context
            .lock_queue(queue_index)
            .map_err(|_| failure("Vulkan queue lock is poisoned"))?;
        self.context.ensure_usable()?;
        execute_submission(&self.context, artifact, submission, queue_index)
    }
}

/// Pre-priority queue choice: the least-loaded queue, breaking ties from
/// `round_robin_start`.
///
/// This is the implementation the live path used before the core policy was
/// wired in, kept as the oracle for the equivalence test below
/// (`queue_priority_policy_reduces_to_select_queue_on_one_tier`). The live path
/// is [`VulkanContext::pick_queue`] and always goes through the core policy.
#[cfg(test)]
fn select_queue(in_flight: &[usize], round_robin_start: usize) -> usize {
    if in_flight.is_empty() {
        return 0;
    }
    let start = round_robin_start % in_flight.len();
    let mut best = start;
    let mut best_load = in_flight[start];
    for step in 1..in_flight.len() {
        let index = (start + step) % in_flight.len();
        if in_flight[index] < best_load {
            best = index;
            best_load = in_flight[index];
        }
    }
    best
}

/// Queue choice for one submission: the whole live view of the core policy.
///
/// `cursor` is the provider's monotonic selection counter, not a value already
/// folded by the queue count: the policy takes the tie-break cursor from
/// `cursor % queues` itself, and folding the counter first would alias the
/// window phase with the queue count (8 queues, a 7-slot window). The tests
/// below call this function with the same cursor the live path passes.
fn select_queue_for_submission(
    in_flight: &[usize],
    tiers: &[QueuePriority],
    cursor: usize,
) -> usize {
    metal_api_core::provider::select_queue_with_priority(
        in_flight,
        tiers,
        cursor,
        QueueSchedulingPolicy::default(),
    )
}

pub(crate) struct VulkanContext {
    entry: ManuallyDrop<Entry>,
    instance: Instance,
    /// The physical device `device` was created from. Kept so the render rail
    /// can query format and queue-family properties without a second
    /// enumeration (`research/docs/23` §6 Step 3b).
    physical: vk::PhysicalDevice,
    device: AshDevice,
    external_memory_host: Option<ExternalMemoryHost>,
    /// `VK_EXT_device_fault` entry points, loaded only when the device
    /// advertises the extension *and* hands the query over.
    device_fault: Option<device_fault::Device>,
    /// Whether the physical device advertised `VK_EXT_device_fault`, with or
    /// without a usable entry point.
    device_fault_advertised: bool,
    /// Diagnostic record of the last observed device loss, if any.
    device_fault_record: Mutex<Option<DeviceFaultSnapshot>>,
    /// Test-only substitution of the next driver answer at one queue boundary.
    driver_loss_injection: Mutex<Option<DeviceLossPoint>>,
    queue_families: Vec<u32>,
    queues: Vec<vk::Queue>,
    next_queue: AtomicUsize,
    queue_submissions: Vec<AtomicUsize>,
    queue_in_flight: Vec<AtomicUsize>,
    /// Cumulative queue selections the scheduler made, one per device queue.
    ///
    /// Unlike `queue_submissions`, which counts submissions a device queue
    /// confirmed, this counts every selection the enqueue paths reported —
    /// the scheduler's allocation, whether or not the driver then accepted the
    /// submission. Together the two surfaces let an observer tell "the policy
    /// never selected this queue" from "this queue was selected but the
    /// submission was refused" (`research/docs/21` §6). Production callers read
    /// it through [`VulkanExecutor::queue_enqueue_counts`]; no probe
    /// installation is required.
    queue_enqueue_counts: Vec<AtomicUsize>,
    /// Cumulative queue retirements, one per device queue.
    ///
    /// Every confirmed submission eventually retires, so on a healthy device
    /// this converges to `queue_submissions`; a widening gap is the in-flight
    /// work the queue still holds (`research/docs/21` §6).
    queue_completion_counts: Vec<AtomicUsize>,
    properties: vk::PhysicalDeviceProperties,
    /// The depth resolve modes the device reports through
    /// `VK_KHR_depth_stencil_resolve` (`research/docs/23` §3.3, v57). The
    /// capability snapshot below maps them onto the contract's closed filter
    /// family; the raw flags stay here so the render rail's admission can
    /// answer the same per-filter question the snapshot answered.
    depth_resolve_modes: vk::ResolveModeFlags,
    /// The stencil resolve modes the device reports through
    /// `VK_KHR_depth_stencil_resolve` (`research/docs/23` §3.3, v60). The
    /// capability snapshot below maps them onto the contract's closed filter
    /// family; the raw flags stay here so the render rail's admission can
    /// answer the same per-filter question the snapshot answered.
    stencil_resolve_modes: vk::ResolveModeFlags,
    /// Whether the device lets the depth and stencil resolve modes differ
    /// (`research/docs/23` §3.3, v60): `independent_resolve` is the raw
    /// property the render rail reads for the combined surface a stored pair
    /// opens (`research/docs/23` §3.3, v60/v70): a device that reports `false`
    /// requires both resolve modes to agree, so a pass whose two faces resolve
    /// through two different filters is refused by name instead of being
    /// submitted as a subpass description the device rejects. The v60 fixtures
    /// resolve both faces through the same filter, which is why the reviewed
    /// shape never reaches that branch.
    independent_resolve: bool,
    /// Whether the device lets one resolve mode be `NONE` while the other is
    /// not (`research/docs/23` §3.3, v60): a stencil-only resolve states
    /// `depthResolveMode = NONE`, which a device that reports `false` refuses.
    independent_resolve_none: bool,
    memory: vk::PhysicalDeviceMemoryProperties,
    device_name: String,
    /// What the device reported about `VK_KHR_shader_float_controls2`, and
    /// whether the feature was enabled at device creation (R8). The SPIR-V
    /// capability gate reads it as [`FloatControls2Support::policy`], so the
    /// snapshot and the gate are the same pair of readings.
    float_controls2: FloatControls2Support,
    /// Whether the device was created with `samplerMirrorClampToEdge` enabled
    /// (`research/docs/23` §109).
    ///
    /// `VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE` is only valid on a device
    /// that enabled this feature — the extension being core from Vulkan 1.2
    /// does not turn the bit on — so the rail records the device's own answer
    /// and creates that mode only when it was enabled, instead of asking for a
    /// mode the device was never told about. Every other address mode the
    /// family names is Vulkan 1.0 core with no feature of its own.
    sampler_mirror_clamp_to_edge: bool,
    queue_locks: Vec<Mutex<()>>,
    enqueue_probe: Mutex<Option<EnqueueProbe>>,
    /// The single admission and terminal-state authority for this device.
    ///
    /// Admission, health and the abandonment counters all come from this
    /// `metal_api_core` lifecycle, so the executor cannot report one state
    /// while refusing on another. The lifecycle owns the budget, the ledger
    /// and the spelling of a terminal refusal; this crate never keeps a second
    /// copy of that state.
    lifecycle: Mutex<ProviderLifecycle>,
    /// Host-side scheduling tier of each device queue, read by every queue
    /// selection. A provider-owned attribute, not a
    /// `VkDeviceQueueCreateInfo::pQueuePriorities` value (`research/docs/21`
    /// §2); every queue starts at [`QueuePriority::Default`], which is what
    /// keeps the pre-priority choice the default.
    queue_priorities: Mutex<Vec<QueuePriority>>,
    /// Device-buffer copy-in and copy-out operations. One of each per touched
    /// allocation, not per view: several views of one allocation share one
    /// device buffer (`research/docs/15` §3.3).
    buffer_uploads: AtomicUsize,
    buffer_readbacks: AtomicUsize,
    /// Bytes actually copied in or out. A write-only view copies nothing in,
    /// which is the observable form of the footprint-bounded transfer
    /// (`research/docs/15` step 4).
    buffer_upload_bytes: AtomicUsize,
    buffer_readback_bytes: AtomicUsize,
    /// The render half's readback regions (`docs/WRITTEN-RECT-READBACK.md`
    /// §2): every stored attachment either copies only its written rectangle
    /// out of the device or reads back whole, and this is the count of each,
    /// the bytes each cost, and — for the whole-extent arm — which fact sent it
    /// there. The counters are process-wide and always on; they are what a
    /// round reports beside the phase profile, and what the e2e oracle reads to
    /// tell a trimmed readback from a fallback.
    readback_rect_attachments: AtomicUsize,
    readback_rect_bytes: AtomicUsize,
    readback_rect_extent_bytes: AtomicUsize,
    readback_full_attachments: AtomicUsize,
    readback_full_bytes: AtomicUsize,
    readback_fallback_switch: AtomicUsize,
    readback_fallback_shape: AtomicUsize,
    readback_fallback_bounds: AtomicUsize,
    readback_whole: AtomicUsize,
    /// Presentation completions of the first present increment
    /// (`research/docs/24` §3.3, §5.3): one acquire when the provider takes
    /// ownership of a present target for a pass, one present when the target's
    /// terminal transition and readback complete. Both count per present
    /// action, so a case with one present reports `1/1`.
    present_acquires: AtomicUsize,
    present_presents: AtomicUsize,
}

/// Loaded `VK_EXT_external_memory_host` entry points and the alignment the
/// device requires for imported host pointers.
struct ExternalMemoryHost {
    device: external_memory_host::Device,
    min_alignment: u64,
}

impl VulkanContext {
    fn new() -> Result<Self, ExecutorError> {
        let entry = unsafe { Entry::load() }
            .map_err(|error| failure(format!("load Vulkan loader: {error}")))?;
        let application_name = CString::new("metal-api-emulator").expect("static application name");
        let application = vk::ApplicationInfo::default()
            .application_name(&application_name)
            .application_version(1)
            .engine_name(&application_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_3);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&application);
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(|error| failure(format!("create Vulkan instance: {error}")))?;

        let selection = select_physical_device(&instance);
        let (physical, queue_family, shader_int8) = match selection {
            Ok(selection) => selection,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(error);
            }
        };
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let primary_queues = queue_family_properties
            .get(queue_family as usize)
            .map(|family| family.queue_count as usize)
            .unwrap_or(1)
            .clamp(1, MAX_QUEUES_PER_FAMILY);
        let mut family_plans = vec![(queue_family, primary_queues)];
        let dedicated_compute = queue_family_properties
            .iter()
            .enumerate()
            .filter(|(index, family)| {
                *index as u32 != queue_family
                    && family.queue_count > 0
                    && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .max_by_key(|(_, family)| family.queue_count)
            .map(|(index, family)| {
                (
                    index as u32,
                    (family.queue_count as usize).clamp(1, MAX_QUEUES_PER_FAMILY),
                )
            });
        if let Some(plan) = dedicated_compute {
            family_plans.push(plan);
        }
        let priority_sets = family_plans
            .iter()
            .map(|(_, count)| vec![1.0_f32; *count])
            .collect::<Vec<_>>();
        let queue_infos = family_plans
            .iter()
            .zip(&priority_sets)
            .map(|((family, count), priorities)| {
                let mut info = vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*family)
                    .queue_priorities(priorities);
                info.queue_count = *count as u32;
                info
            })
            .collect::<Vec<_>>();
        let extensions = match unsafe { instance.enumerate_device_extension_properties(physical) } {
            Ok(extensions) => extensions,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(failure(format!(
                    "enumerate Vulkan device extensions: {error}"
                )));
            }
        };
        let has_external_memory_host = extensions.iter().any(|extension| {
            extension
                .extension_name_as_c_str()
                .is_ok_and(|name| name == external_memory_host::NAME)
        });
        let has_device_fault = extensions.iter().any(|extension| {
            extension
                .extension_name_as_c_str()
                .is_ok_and(|name| name == device_fault::NAME)
        });
        // `VK_KHR_shader_float_controls2` (R8): the extension name answers
        // "can the device be asked for the feature at all", and the feature
        // query answers "does it have it". The query is a valid
        // `VkPhysicalDeviceFeatures2` chain whether or not the extension is
        // supported — an unsupported device reports the bit off — so both
        // readings can be recorded before the device exists.
        let float_controls2_extension = extensions.iter().any(|extension| {
            extension
                .extension_name_as_c_str()
                .is_ok_and(|name| name == shader_float_controls2::NAME)
        });
        let mut float_controls2_features =
            vk::PhysicalDeviceShaderFloatControls2FeaturesKHR::default();
        let mut float_controls2_query =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut float_controls2_features);
        unsafe { instance.get_physical_device_features2(physical, &mut float_controls2_query) };
        let float_controls2 = FloatControls2Support {
            extension: float_controls2_extension,
            feature: float_controls2_features.shader_float_controls2 == vk::TRUE,
        };
        let mut enabled_extensions = Vec::new();
        if has_external_memory_host {
            enabled_extensions.push(external_memory_host::NAME.as_ptr());
        }
        // The name is only requested when the device advertised it *and*
        // reported the feature: enabling an absent extension is a device
        // creation error, and enabling one whose bit is off would put the
        // module gate and the device out of step.
        if float_controls2.enabled() {
            enabled_extensions.push(shader_float_controls2::NAME.as_ptr());
        }
        // `samplerMirrorClampToEdge` (`VK_KHR_sampler_mirror_clamp_to_edge`,
        // core from Vulkan 1.2) is the one feature an address mode the family
        // names needs (`research/docs/23` §109). The query is a valid
        // `VkPhysicalDeviceFeatures2` chain on every 1.3 device, and the bit is
        // enabled iff the device reported it: enabling a feature the device
        // does not have is a device-creation error, and the rail's refusal for
        // the mode reads this same bit.
        let mut mirror_clamp_features = vk::PhysicalDeviceVulkan12Features::default();
        let mut mirror_clamp_query =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut mirror_clamp_features);
        unsafe { instance.get_physical_device_features2(physical, &mut mirror_clamp_query) };
        let sampler_mirror_clamp_to_edge =
            mirror_clamp_features.sampler_mirror_clamp_to_edge == vk::TRUE;
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default().maintenance4(true);
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_int8(shader_int8)
            .sampler_mirror_clamp_to_edge(sampler_mirror_clamp_to_edge);
        let physical_features = vk::PhysicalDeviceFeatures::default().shader_int64(true);
        let mut float_controls2_enable =
            vk::PhysicalDeviceShaderFloatControls2FeaturesKHR::default()
                .shader_float_controls2(float_controls2.enabled());
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enabled_extensions)
            .enabled_features(&physical_features)
            .push_next(&mut vulkan13)
            .push_next(&mut vulkan12)
            .push_next(&mut float_controls2_enable);
        let device = match unsafe { instance.create_device(physical, &device_info, None) } {
            Ok(device) => device,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(failure(format!("create Vulkan device: {error}")));
            }
        };
        let mut queues = Vec::new();
        let mut queue_families = Vec::new();
        for (family, count) in &family_plans {
            for index in 0..*count {
                queues.push(unsafe { device.get_device_queue(*family, index as u32) });
                queue_families.push(*family);
            }
        }
        let queue_count = queues.len();
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        // The depth resolve modes the device reports (`research/docs/23` §3.3,
        // v57): `VK_KHR_depth_stencil_resolve` is core from 1.2 and the
        // selected device is already 1.3, so the query always runs and a
        // device that reports no admitted filter simply leaves the flags at
        // zero, which the capability snapshot maps to "cannot resolve".
        let mut depth_stencil_resolve = vk::PhysicalDeviceDepthStencilResolveProperties::default();
        let mut resolve_properties =
            vk::PhysicalDeviceProperties2::default().push_next(&mut depth_stencil_resolve);
        unsafe { instance.get_physical_device_properties2(physical, &mut resolve_properties) };
        let depth_resolve_modes = depth_stencil_resolve.supported_depth_resolve_modes;
        let stencil_resolve_modes = depth_stencil_resolve.supported_stencil_resolve_modes;
        let independent_resolve = depth_stencil_resolve.independent_resolve != 0;
        let independent_resolve_none = depth_stencil_resolve.independent_resolve_none != 0;
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let external_memory_host = has_external_memory_host.then(|| {
            let mut host_properties = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
            let mut properties2 =
                vk::PhysicalDeviceProperties2::default().push_next(&mut host_properties);
            unsafe { instance.get_physical_device_properties2(physical, &mut properties2) };
            ExternalMemoryHost {
                device: external_memory_host::Device::new(&instance, &device),
                min_alignment: host_properties.min_imported_host_pointer_alignment,
            }
        });
        // The fault query is diagnostic evidence, so the entry points are only
        // loaded for a device that advertises the extension *and* hands the
        // query over; neither an absent extension nor one whose entry point the
        // ICD refuses to resolve may fail device creation, and a refusal has to
        // degrade to `DeviceFaultSnapshot::unavailable(false)` instead of reaching
        // ash's panicking stub.
        let device_fault = device_fault_loader_ready(
            has_device_fault,
            device_fault_entry_point(&instance, &device),
        )
        .then(|| device_fault::Device::new(&instance, &device));

        Ok(Self {
            entry: ManuallyDrop::new(entry),
            instance,
            physical,
            device,
            external_memory_host,
            device_fault,
            device_fault_advertised: has_device_fault,
            device_fault_record: Mutex::new(None),
            driver_loss_injection: Mutex::new(None),
            queue_families,
            queues,
            next_queue: AtomicUsize::new(0),
            queue_submissions: (0..queue_count).map(|_| AtomicUsize::new(0)).collect(),
            buffer_uploads: AtomicUsize::new(0),
            buffer_readbacks: AtomicUsize::new(0),
            buffer_upload_bytes: AtomicUsize::new(0),
            buffer_readback_bytes: AtomicUsize::new(0),
            readback_rect_attachments: AtomicUsize::new(0),
            readback_rect_bytes: AtomicUsize::new(0),
            readback_rect_extent_bytes: AtomicUsize::new(0),
            readback_full_attachments: AtomicUsize::new(0),
            readback_full_bytes: AtomicUsize::new(0),
            readback_fallback_switch: AtomicUsize::new(0),
            readback_fallback_shape: AtomicUsize::new(0),
            readback_fallback_bounds: AtomicUsize::new(0),
            readback_whole: AtomicUsize::new(0),
            present_acquires: AtomicUsize::new(0),
            present_presents: AtomicUsize::new(0),
            queue_in_flight: (0..queue_count).map(|_| AtomicUsize::new(0)).collect(),
            queue_enqueue_counts: (0..queue_count).map(|_| AtomicUsize::new(0)).collect(),
            queue_completion_counts: (0..queue_count).map(|_| AtomicUsize::new(0)).collect(),
            properties,
            depth_resolve_modes,
            stencil_resolve_modes,
            independent_resolve,
            independent_resolve_none,
            memory,
            device_name,
            float_controls2,
            sampler_mirror_clamp_to_edge,
            queue_locks: (0..queue_count).map(|_| Mutex::new(())).collect(),
            enqueue_probe: Mutex::new(None),
            lifecycle: Mutex::new(ProviderLifecycle::new(
                ABANDONMENT_BUDGET_SUBMISSIONS,
                ABANDONMENT_BUDGET_BYTES,
            )),
            queue_priorities: Mutex::new(vec![QueuePriority::Default; queue_count]),
        })
    }

    /// Lock the lifecycle for one transition or query.
    ///
    /// Terminal states are monotonic and admission never unwinds through this
    /// mutex, so a poison left by an unrelated panic still protects the last
    /// admitted state; recovering the guard keeps refusal available instead of
    /// turning one panic into a second, unrelated failure.
    fn lock_lifecycle(&self) -> MutexGuard<'_, ProviderLifecycle> {
        self.lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The selected device's own limits, for the render rail's attachment
    /// window (R1b, `research/docs/23` §70).
    ///
    /// The declared window is this device's `maxFramebuffer{Width,Height}`
    /// clamped by the reviewed ceiling, and the rail re-asks both halves at
    /// execution; reading the same `VkPhysicalDeviceLimits` is what keeps the
    /// snapshot and the rail's answer from drifting apart.
    pub(crate) fn physical_device_limits(&self) -> &vk::PhysicalDeviceLimits {
        &self.properties.limits
    }

    /// The contract's admitted depth-resolve filter mask for this device
    /// (`research/docs/23` §3.3, v57). The render rail's admission reads the
    /// same mask the capability snapshot published, so a directly-constructed
    /// request is refused with the same per-filter question the snapshot
    /// answered.
    pub(crate) fn admitted_depth_resolve_modes(&self) -> u32 {
        provider::depth_resolve_mode_mask(self.depth_resolve_modes)
    }

    /// The contract's admitted stencil-resolve filter mask for this device
    /// (`research/docs/23` §3.3, v60). The render rail's admission reads the
    /// same mask the capability snapshot published, so a directly-constructed
    /// request is refused with the same per-filter question the snapshot
    /// answered.
    pub(crate) fn admitted_stencil_resolve_modes(&self) -> u32 {
        provider::stencil_resolve_mode_mask(self.stencil_resolve_modes)
    }

    /// What this device reported about `VK_KHR_shader_float_controls2` (R8).
    pub(crate) const fn float_controls2_support(&self) -> FloatControls2Support {
        self.float_controls2
    }

    /// Whether this device was created with `samplerMirrorClampToEdge` enabled
    /// (`research/docs/23` §109).
    pub(crate) const fn sampler_mirror_clamp_to_edge(&self) -> bool {
        self.sampler_mirror_clamp_to_edge
    }

    /// The SPIR-V capability policy this device answers with (R8).
    ///
    /// Both translation entry points and the render registration gate read this
    /// one value, so the module a rail admits and the device that will execute
    /// it cannot drift apart.
    pub(crate) const fn spirv_feature_policy(&self) -> SpirvFeaturePolicy {
        self.float_controls2.policy()
    }

    /// Admit one new submission against the lifecycle.
    ///
    /// The core refusal is returned unchanged, so the slug, class, phase,
    /// `terminal` field, abandoned counters and retryability the provider
    /// boundary reports are exactly the contract's.
    pub(crate) fn admit(&self) -> Result<(), ProviderError> {
        terminal_refusal(&self.lock_lifecycle())
    }

    /// Provider health, straight from the lifecycle.
    pub(crate) fn health(&self) -> ProviderHealth {
        self.lock_lifecycle().health()
    }

    /// Refuse new work on a terminal context for the direct `ComputeExecutor`
    /// API, which can only carry a message.
    ///
    /// The structured refusal is not lost — it is what [`VulkanContext::admit`]
    /// returns — and the message is derived from it instead of from a second
    /// state check, so both APIs always agree on why work was refused.
    fn ensure_usable(&self) -> Result<(), ExecutorError> {
        self.admit().map_err(|error| {
            failure(format!(
                "Vulkan executor is unavailable: {} ({:?})",
                error.slug, error.retryability
            ))
        })
    }

    /// Device queue for the next independent submission.
    ///
    /// The choice is the core priority/fairness policy (`research/docs/21` §4)
    /// applied to the tiers installed by [`Self::set_queue_priorities`]: an idle
    /// queue still beats a busy one whatever the tiers, and among equally loaded
    /// queues the rotation window nominates a tier, breaking ties from the
    /// selection cursor. Every queue a provider creates starts at
    /// [`QueuePriority::Default`], where the policy is the previous least-loaded
    /// rule for every load vector — the equivalence test in this module pins
    /// that. A single-queue family always returns zero, preserving the previous
    /// behaviour on devices such as Lavapipe.
    pub(crate) fn pick_queue(&self) -> usize {
        let len = self.queues.len();
        if len == 0 {
            return 0;
        }
        // Both the window phase and the tie-break cursor come from the
        // monotonic selection counter. Folding it by the queue count *before*
        // the policy would alias the two: with 8 queues and a 7-slot window,
        // cursors 7 and 8 would nominate the same slot and let the high tier
        // run longer than its weight.
        let cursor = self.next_queue.fetch_add(1, Ordering::Relaxed);
        let mut loads = [0_usize; MAX_DEVICE_QUEUES];
        for (slot, counter) in loads.iter_mut().zip(&self.queue_in_flight) {
            *slot = counter.load(Ordering::Relaxed);
        }
        let tiers = self
            .queue_priorities
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // A table shorter than the device is not an error here: the core policy
        // reads missing entries as `Default`. `set_queue_priorities` refuses
        // that shape, so it only happens if a device grows queues later.
        let ranked = tiers.len().min(len);
        select_queue_for_submission(&loads[..len], &tiers[..ranked], cursor)
    }

    /// Install the scheduling tier of every device queue.
    ///
    /// The table is read on each queue selection, so the slice must describe
    /// the device exactly: a differently sized table is refused instead of
    /// silently padded.
    pub(crate) fn set_queue_priorities(&self, tiers: &[QueuePriority]) -> Result<(), &'static str> {
        let mut installed = self
            .queue_priorities
            .lock()
            .map_err(|_| "Vulkan queue priority table is poisoned")?;
        if tiers.len() != installed.len() {
            return Err("Vulkan queue priority table size mismatch");
        }
        installed.copy_from_slice(tiers);
        Ok(())
    }

    /// Snapshot of the tiers currently installed on the device queues.
    pub(crate) fn queue_priorities(&self) -> Vec<QueuePriority> {
        self.queue_priorities
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Report one enqueue on the selected device queue to the installed probe.
    ///
    /// Every submit path calls this while it holds that queue's host enqueue
    /// lock, so the probe observes the same sequence the device does, whichever
    /// path (synchronous or deferred) enqueued the command buffer. Production
    /// callers leave the probe unset, and the cumulative selection still lands
    /// in `queue_enqueue_counts` so the allocation stays observable without a
    /// probe.
    pub(crate) fn notify_enqueue(&self, index: usize) {
        if let Some(counter) = self.queue_enqueue_counts.get(index) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(probe) = self
            .enqueue_probe
            .lock()
            .ok()
            .and_then(|probe| probe.clone())
        {
            probe(index);
        }
    }

    /// Lock the host-side enqueue section of one device queue.
    ///
    /// Vulkan requires host access to a `VkQueue` to be externally
    /// synchronized. Per-queue locks let independent queues enqueue
    /// concurrently while submissions to the same queue stay serialized.
    pub(crate) fn lock_queue(&self, index: usize) -> Result<MutexGuard<'_, ()>, &'static str> {
        self.queue_locks
            .get(index)
            .ok_or("Vulkan queue index out of range")?
            .lock()
            .map_err(|_| "Vulkan queue lock is poisoned")
    }

    pub(crate) fn record_queue_submission(&self, index: usize) {
        if let Some(counter) = self.queue_submissions.get(index) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(counter) = self.queue_in_flight.get(index) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn record_queue_retirement(&self, index: usize) {
        if let Some(counter) = self.queue_in_flight.get(index) {
            let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(1)
            });
        }
        if let Some(counter) = self.queue_completion_counts.get(index) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn queue_count(&self) -> usize {
        self.queues.len()
    }

    pub(crate) fn queue_family_count(&self) -> usize {
        let mut families = self.queue_families.clone();
        families.sort_unstable();
        families.dedup();
        families.len()
    }

    pub(crate) fn record_buffer_upload(&self) {
        self.buffer_uploads.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_buffer_upload_bytes(&self, bytes: usize) {
        self.buffer_upload_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(crate) fn record_buffer_readback(&self) {
        self.buffer_readbacks.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_buffer_readback_bytes(&self, bytes: usize) {
        self.buffer_readback_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record one stored attachment read back through its written rectangle:
    /// the bytes that left the device mapping, and the bytes the attachment's
    /// whole extent occupies (what the whole-attachment readback would have
    /// copied instead).
    pub(crate) fn record_readback_rect(&self, bytes: usize, extent_bytes: usize) {
        self.readback_rect_attachments
            .fetch_add(1, Ordering::Relaxed);
        self.readback_rect_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.readback_rect_extent_bytes
            .fetch_add(extent_bytes, Ordering::Relaxed);
    }

    /// Record one stored attachment read back whole, whatever sent it there.
    pub(crate) fn record_readback_full(&self, bytes: usize) {
        self.readback_full_attachments
            .fetch_add(1, Ordering::Relaxed);
        self.readback_full_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record *why* one attachment read back whole: the control switch, a shape
    /// whose seed this rail does not hold host-side, a declared rect the rail
    /// cannot prove, or a rectangle that covers the whole attachment anyway.
    pub(crate) fn record_readback_fallback(&self, bucket: ReadbackFallback) {
        match bucket {
            ReadbackFallback::Switch => &self.readback_fallback_switch,
            ReadbackFallback::Shape => &self.readback_fallback_shape,
            ReadbackFallback::Bounds => &self.readback_fallback_bounds,
            ReadbackFallback::Whole => &self.readback_whole,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// The render half's cumulative readback regions
    /// (`docs/WRITTEN-RECT-READBACK.md` §2).
    pub(crate) fn readback_regions(&self) -> ReadbackRegionCounts {
        ReadbackRegionCounts {
            rect_attachments: self.readback_rect_attachments.load(Ordering::Relaxed),
            rect_bytes: self.readback_rect_bytes.load(Ordering::Relaxed),
            rect_extent_bytes: self.readback_rect_extent_bytes.load(Ordering::Relaxed),
            full_attachments: self.readback_full_attachments.load(Ordering::Relaxed),
            full_bytes: self.readback_full_bytes.load(Ordering::Relaxed),
            switch_attachments: self.readback_fallback_switch.load(Ordering::Relaxed),
            shape_attachments: self.readback_fallback_shape.load(Ordering::Relaxed),
            bounds_attachments: self.readback_fallback_bounds.load(Ordering::Relaxed),
            whole_attachments: self.readback_whole.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn buffer_copy_counts(&self) -> (usize, usize) {
        (
            self.buffer_uploads.load(Ordering::Relaxed),
            self.buffer_readbacks.load(Ordering::Relaxed),
        )
    }

    /// Record one acquire of a present target. Counted once per present action,
    /// before the pass that renders into the target runs (`docs/24` §3.6).
    pub(crate) fn record_present_acquire(&self) {
        self.present_acquires.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one completed present. Counted once per present action, after the
    /// target's terminal transition and readback have landed (`docs/24` §3.6).
    pub(crate) fn record_present(&self) {
        self.present_presents.fetch_add(1, Ordering::Relaxed);
    }

    /// Cumulative (acquire, present) completions of the presentation rail.
    pub(crate) fn present_counts(&self) -> (usize, usize) {
        (
            self.present_acquires.load(Ordering::Relaxed),
            self.present_presents.load(Ordering::Relaxed),
        )
    }

    pub(crate) fn buffer_copy_bytes(&self) -> (usize, usize) {
        (
            self.buffer_upload_bytes.load(Ordering::Relaxed),
            self.buffer_readback_bytes.load(Ordering::Relaxed),
        )
    }

    pub(crate) fn queue_submission_counts(&self) -> Vec<usize> {
        self.queue_submissions
            .iter()
            .map(|counter| counter.load(Ordering::Relaxed))
            .collect()
    }

    /// Cumulative queue selections per device queue, the scheduler's
    /// allocation before any driver answer (`research/docs/21` §6).
    pub(crate) fn queue_enqueue_counts(&self) -> Vec<usize> {
        self.queue_enqueue_counts
            .iter()
            .map(|counter| counter.load(Ordering::Relaxed))
            .collect()
    }

    /// Cumulative queue retirements per device queue (`research/docs/21` §6).
    pub(crate) fn queue_completion_counts(&self) -> Vec<usize> {
        self.queue_completion_counts
            .iter()
            .map(|counter| counter.load(Ordering::Relaxed))
            .collect()
    }

    /// Record one submission whose completion can no longer be observed.
    ///
    /// The lifecycle charges the budget, so the returned outcome is what tells
    /// a caller whether it may keep the context in service or whether this
    /// abandonment already ended it.
    pub(crate) fn record_abandonment(&self, bytes: u64) -> AbandonmentOutcome {
        self.lock_lifecycle().record_abandonment(bytes)
    }

    /// Report `(abandoned submissions, abandoned bytes)` recorded so far.
    pub(crate) fn abandonment_stats(&self) -> (u64, u64) {
        self.lock_lifecycle().abandonment()
    }

    /// Fail the context closed for a submission whose completion can no longer
    /// be observed, without charging the budget.
    ///
    /// The queue-refused and post-retirement classification paths report here:
    /// the submission is not abandoned GPU work (so the counters stay honest),
    /// but the context stops trusting the device and refuses new work.
    pub(crate) fn mark_unobservable_submission(&self) {
        self.lock_lifecycle().mark_unobservable_submission();
    }

    pub(crate) fn mark_device_lost(&self) {
        self.lock_lifecycle().mark_device_lost();
    }

    /// Diagnostic `VK_EXT_device_fault` record of the last observed loss.
    ///
    /// `None` until a loss is observed on this context.
    pub(crate) fn last_device_fault(&self) -> Option<DeviceFaultSnapshot> {
        self.device_fault_record
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Query `VK_EXT_device_fault` for the fault the driver last reported.
    ///
    /// The record is diagnostic evidence and never a gate. A device that does
    /// not advertise the extension answers
    /// [`DeviceFaultSnapshot::unavailable`]; a device that advertises it but
    /// never handed its entry point over (the Windows RTX 5060 ICD,
    /// 2026-09-18), and a driver that hands it over but refuses the query, both
    /// answer the same shape with `extension_present == true` and no addresses.
    /// Refusing a loss report because the diagnostics were unavailable would
    /// throw away the loss itself.
    fn query_device_fault(&self) -> DeviceFaultSnapshot {
        // The advertised bit survives a missing entry point: the field answers
        // "did the device offer the extension", not "did the query run".
        let mut snapshot = DeviceFaultSnapshot::unavailable(self.device_fault_advertised);
        let Some(loader) = self.device_fault.as_ref() else {
            return snapshot;
        };
        // The driver reports the counts it wants to write first; the second
        // call fills the arrays sized from that answer. `vendorBinarySize` is
        // recorded but the binary itself is never requested: this is a
        // diagnostic snapshot, not a crash dump.
        let entry = loader.fp().get_device_fault_info_ext;
        let mut counts = vk::DeviceFaultCountsEXT::default();
        if unsafe { entry(loader.device(), &mut counts, std::ptr::null_mut()) }
            != vk::Result::SUCCESS
        {
            return snapshot;
        }
        let mut addresses =
            vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
        let mut vendor_infos =
            vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
        let mut info = vk::DeviceFaultInfoEXT::<'_> {
            p_address_infos: addresses.as_mut_ptr(),
            p_vendor_infos: vendor_infos.as_mut_ptr(),
            ..Default::default()
        };
        counts.address_info_count = addresses.len() as u32;
        counts.vendor_info_count = vendor_infos.len() as u32;
        if unsafe { entry(loader.device(), &mut counts, &mut info) } != vk::Result::SUCCESS {
            return snapshot;
        }
        snapshot.description = info
            .description_as_c_str()
            .ok()
            .map(|description| description.to_string_lossy().into_owned())
            .filter(|description| !description.is_empty());
        snapshot.addresses = addresses
            .iter()
            .take(counts.address_info_count as usize)
            .map(|address| DeviceFaultAddress {
                address_type: address.address_type.as_raw(),
                reported_address: address.reported_address,
                address_precision: address.address_precision,
            })
            .collect();
        snapshot.vendor_info_count = counts.vendor_info_count;
        snapshot.vendor_binary_size = counts.vendor_binary_size;
        snapshot
    }

    /// Record one device loss observed at a driver boundary.
    ///
    /// This is the only bridge from a raw `VK_ERROR_DEVICE_LOST` into the core
    /// lifecycle: the terminal state comes from
    /// [`ProviderLifecycle::mark_device_lost`], so the instance stops admitting
    /// work through the documented `device_lost` refusal and every lease in the
    /// lifecycle ledger retires as a teardown guarantee. The returned record is
    /// the evidence the caller attaches to its error.
    pub(crate) fn observe_device_loss(&self) -> DeviceFaultSnapshot {
        let fault = self.query_device_fault();
        if let Ok(mut record) = self.device_fault_record.lock() {
            *record = Some(fault.clone());
        }
        self.mark_device_lost();
        fault
    }

    /// Arm the test-only driver-answer substitution at one queue boundary.
    fn arm_driver_loss_injection(&self, point: DeviceLossPoint) {
        if let Ok(mut slot) = self.driver_loss_injection.lock() {
            *slot = Some(point);
        }
    }

    /// Consume the substitution armed for `point`, if any.
    fn take_driver_loss_injection(&self, point: DeviceLossPoint) -> bool {
        let Ok(mut slot) = self.driver_loss_injection.lock() else {
            return false;
        };
        if *slot == Some(point) {
            *slot = None;
            true
        } else {
            false
        }
    }

    /// `vkQueueSubmit` on one selected device queue.
    ///
    /// Every submission in this crate enqueues here, so the substitution sits
    /// exactly where the driver's answer would arrive. An armed substitution
    /// skips the call instead of overwriting a successful one: a device that
    /// answers `VK_ERROR_DEVICE_LOST` has not executed the submission, and a
    /// test that enqueued it anyway would leave real work in flight on a device
    /// the provider is about to treat as gone.
    pub(crate) fn submit_commands(
        &self,
        queue_index: usize,
        submits: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> Result<(), vk::Result> {
        if self.take_driver_loss_injection(DeviceLossPoint::Submit) {
            return Err(vk::Result::ERROR_DEVICE_LOST);
        }
        let queue = *self
            .queues
            .get(queue_index)
            .ok_or(vk::Result::ERROR_UNKNOWN)?;
        unsafe { self.device.queue_submit(queue, submits, fence) }
    }

    /// `vkWaitForFences` on one completion fence.
    ///
    /// A driver answer of `VK_ERROR_DEVICE_LOST` says nothing about whether the
    /// fence was reached, so an armed substitution may only replace an answer
    /// that observed completion: the real wait runs first, and a wait that
    /// timed out keeps the substitution armed and reports the timeout it really
    /// received. That is what lets the provider destroy the handles of a
    /// waiting submission without leaving in-flight work behind.
    pub(crate) fn wait_for_fence(
        &self,
        fence: vk::Fence,
        timeout_ns: u64,
    ) -> Result<(), vk::Result> {
        let wait = unsafe { self.device.wait_for_fences(&[fence], true, timeout_ns) };
        if wait.is_ok() && self.take_driver_loss_injection(DeviceLossPoint::Wait) {
            return Err(vk::Result::ERROR_DEVICE_LOST);
        }
        wait
    }

    fn abandon(self: &Arc<Self>, resources: ExecutionResources) {
        let _ = self.record_abandonment(resources.owned_bytes());
        // A queue that never told us whether it accepted the submission leaves
        // the context terminal even when the budget still has room: the
        // submission below is leaked rather than retired, and no caller can
        // prove the device state afterwards.
        self.mark_unobservable_submission();
        // The queue may still access every handle in `resources`. Keep both it
        // and one context reference alive until process exit; destroying either
        // after a host timeout would violate Vulkan object lifetime rules.
        std::mem::forget(resources);
    }

    fn memory_type(
        &self,
        bits: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Result<u32, ExecutorError> {
        (0..self.memory.memory_type_count)
            .find(|index| {
                bits & (1 << index) != 0
                    && self.memory.memory_types[*index as usize]
                        .property_flags
                        .contains(required)
            })
            .ok_or_else(|| {
                failure(format!(
                    "no Vulkan memory type satisfies flags {:#x} for mask {bits:#x}",
                    required.as_raw()
                ))
            })
    }

    /// Alignment required for `VK_EXT_external_memory_host` imports, or zero
    /// when the extension is unavailable.
    pub(crate) fn external_memory_host_alignment(&self) -> u64 {
        self.external_memory_host
            .as_ref()
            .map_or(0, |host| host.min_alignment)
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        let lifecycle = self.lock_lifecycle();
        let abandoned = lifecycle.abandonment().0 > 0;
        let device_lost = lifecycle.ended_by_device_loss();
        drop(lifecycle);
        if abandoned && !device_lost {
            // A recorded abandonment may still be executing on the queue.
            // Timeout paths leak an extra Arc, so this arm is defensive rather
            // than expected. Never unload the Vulkan loader under pending work.
            // A lost device is the other case: nothing can be observed
            // anymore, so the device objects are destroyed without waiting.
            return;
        }
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
            ManuallyDrop::drop(&mut self.entry);
        }
    }
}

fn select_physical_device(
    instance: &Instance,
) -> Result<(vk::PhysicalDevice, u32, bool), ExecutorError> {
    let physicals = unsafe { instance.enumerate_physical_devices() }
        .map_err(|error| failure(format!("enumerate Vulkan physical devices: {error}")))?;
    physicals
        .into_iter()
        .filter_map(|physical| {
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            if properties.api_version < vk::API_VERSION_1_3 {
                return None;
            }
            let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan13);
            unsafe { instance.get_physical_device_features2(physical, &mut features) };
            let shader_int64 = features.features.shader_int64 == vk::TRUE;
            let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut features12 = vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan12);
            unsafe { instance.get_physical_device_features2(physical, &mut features12) };
            if vulkan13.maintenance4 != vk::TRUE {
                return None;
            }
            // The reviewed texture fixtures return an i8 status beside the
            // texel, so the shader declares Int8. Vulkan exposes that through
            // the shaderInt8 feature; a device without it cannot execute the
            // same SPIR-V.
            if vulkan12.shader_int8 != vk::TRUE {
                return None;
            }
            if !shader_int64 {
                return None;
            }
            let queues = unsafe { instance.get_physical_device_queue_family_properties(physical) };
            queues
                .iter()
                .enumerate()
                .filter(|(_, queue)| {
                    queue.queue_count > 0 && queue.queue_flags.contains(vk::QueueFlags::COMPUTE)
                })
                .map(|(index, queue)| {
                    let type_score = match properties.device_type {
                        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
                        vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
                        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                        vk::PhysicalDeviceType::CPU => 1,
                        _ => 0,
                    };
                    let queue_score =
                        u32::from(queue.queue_flags.contains(vk::QueueFlags::GRAPHICS));
                    ((type_score, queue_score), physical, index as u32)
                })
                .max_by_key(|(score, _, _)| *score)
        })
        .max_by_key(|(score, _, _)| *score)
        .map(|(_, physical, family)| (physical, family, true))
        .ok_or_else(|| {
            failure(
                "no Vulkan 1.3 physical device with maintenance4, shaderInt8 and a compute queue",
            )
        })
}

struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    fn new() -> Result<Self, ExecutorError> {
        loop {
            let serial = SCRATCH_SERIAL.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("metal-api-vulkan-{}-{serial}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(failure(format!(
                        "create translation scratch {}: {error}",
                        path.display()
                    )))
                }
            }
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn execute_submission(
    context: &Arc<VulkanContext>,
    artifact: Arc<VulkanPipelineArtifact>,
    submission: ComputeSubmission,
    queue_index: usize,
) -> Result<Vec<BufferUpdate>, ExecutorError> {
    execute_submission_with_status(context, artifact, submission, queue_index)
        .map_err(|error| failure(error.detail.unwrap_or(error.slug)))
}

/// Execute while preserving the phase and queue disposition for provider callers.
/// The caller owns serialization: it holds the host enqueue lock of the queue
/// it selected with `VulkanContext::pick_queue` and passes that index, so the
/// device enqueue and the lock always describe the same queue. It supplies its
/// token after observing the result.
pub(crate) fn execute_submission_with_status(
    context: &Arc<VulkanContext>,
    artifact: Arc<VulkanPipelineArtifact>,
    submission: ComputeSubmission,
    queue_index: usize,
) -> Result<Vec<BufferUpdate>, ProviderError> {
    let dispatch = (
        submission.threads_per_grid.dimensions(),
        submission.threads_per_threadgroup.dimensions(),
    );
    execute_serial_submission_with_status(context, artifact, submission, &[dispatch], queue_index)
}

/// One ordered dispatch, mapping Metal binding indices to uploaded pool keys.
/// A Metal argument index is not unique across resource kinds: the translator
/// reports a sampled texture and a buffer under the same index, so the pool key
/// carries the kind as well (`research/docs/16` §4.4).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum PoolKind {
    Buffer,
    Texture,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct PoolKey {
    pub kind: PoolKind,
    pub index: u32,
}

impl PoolKey {
    pub(crate) const fn buffer(index: u32) -> Self {
        Self {
            kind: PoolKind::Buffer,
            index,
        }
    }
}

/// One Metal binding: the argument index plus the pool resource it names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Binding {
    pub metal_index: u32,
    pub key: PoolKey,
    /// The binding width the planner validates: the pool window in bytes for a
    /// buffer, the tightly packed extent for a texture.
    pub width: usize,
}

/// One ordered dispatch, mapping Metal binding indices to uploaded pool keys.
#[derive(Clone, Debug)]
pub(crate) struct BoundDispatch {
    pub grid: [u32; 3],
    pub local: [u32; 3],
    pub bindings: Vec<Binding>,
}

/// Execute one to eight ordered dispatches with one pipeline and buffer set.
/// Tuples contain (grid, local size); the first must match the submission sizes.
/// `queue_index` is the queue the caller selected and locked.
pub(crate) fn execute_serial_submission_with_status(
    context: &Arc<VulkanContext>,
    artifact: Arc<VulkanPipelineArtifact>,
    submission: ComputeSubmission,
    dispatches: &[([u32; 3], [u32; 3])],
    queue_index: usize,
) -> Result<Vec<BufferUpdate>, ProviderError> {
    validate_serial_dispatches(
        (
            submission.threads_per_grid.dimensions(),
            submission.threads_per_threadgroup.dimensions(),
        ),
        dispatches,
    )?;
    let textures = submission.textures.clone();
    let bound = identity_dispatches(&submission.buffers, &textures, dispatches);
    execute_rebound_submission_with_status(
        context,
        artifact,
        submission.buffers,
        &bound,
        &textures,
        queue_index,
    )
}

/// Execute one pipeline against a selected subset of uploaded buffers per pass.
/// The caller owns serialization. All passes are validated before creating
/// request resources, and share one upload, command buffer, fence, and readback.
/// Updates identify pool keys and include each buffer writable in any pass once.
pub(crate) fn execute_rebound_submission_with_status(
    context: &Arc<VulkanContext>,
    artifact: Arc<VulkanPipelineArtifact>,
    buffers: Vec<BufferBinding>,
    dispatches: &[BoundDispatch],
    textures: &[metal_api_core::provider::TextureView],
    queue_index: usize,
) -> Result<Vec<BufferUpdate>, ProviderError> {
    let artifacts = vec![artifact; dispatches.len()];
    execute_pipeline_sequence_with_status(
        context,
        &artifacts,
        buffers,
        dispatches,
        textures,
        queue_index,
    )
}

/// Execute one to eight ordered pipeline dispatches over one uploaded pool.
/// Each pass owns its shader, descriptor layout, specialization, and binding
/// map, while the sequence shares one command buffer, fence, and final readback.
pub(crate) fn execute_pipeline_sequence_with_status(
    context: &Arc<VulkanContext>,
    artifacts: &[Arc<VulkanPipelineArtifact>],
    buffers: Vec<BufferBinding>,
    dispatches: &[BoundDispatch],
    textures: &[metal_api_core::provider::TextureView],
    queue_index: usize,
) -> Result<Vec<BufferUpdate>, ProviderError> {
    refuse_executor_storage_landings(textures)?;
    let buffers = buffers
        .into_iter()
        .map(PoolBinding::Owned)
        .collect::<Vec<_>>();
    buffer_updates(execute_pool_sequence_with_status(
        context,
        artifacts,
        &buffers,
        dispatches,
        SequenceTail {
            borrowed: None,
            textures,
            indirect_dispatch: None,
        },
        queue_index,
    )?)
}

/// The standalone executor's `execute` contract returns buffer-keyed updates,
/// so a storage image landing has no channel there (`research/docs/26` §21.4,
/// C2). The provider path owns the writeback channel and executes the shape, so
/// this refusal is about the boundary and not about the device: a caller that
/// hands a writable texture to the executor is refused by name instead of
/// being handed a trace whose texels silently never leave.
pub(crate) fn refuse_executor_storage_landings(
    textures: &[metal_api_core::provider::TextureView],
) -> Result<(), ProviderError> {
    let Some(texture) = textures.iter().find(|texture| texture.access.is_writable()) else {
        return Ok(());
    };
    Err(ProviderError::new(
        ProviderPhase::Resolve,
        ProviderErrorClass::Capability,
        "compute_storage_image_executor_unsupported",
    )
    .expect("non-empty provider error slug")
    .with_field(
        "binding",
        FieldValue::Unsigned(u64::from(texture.metal_binding)),
    )
    .with_field("access", FieldValue::Text(format!("{:?}", texture.access)))
    .with_detail(
        "the executor API returns buffer updates only; submit the trace through a provider, \
         which publishes storage image landings on the writeback channel",
    ))
}

/// Narrow a pool sequence's landings onto the executor's buffer-keyed update
/// channel. Callers run [`refuse_executor_storage_landings`] first, so a
/// texture landing here is a programming error rather than a request shape.
fn buffer_updates(updates: Vec<LandingUpdate>) -> Result<Vec<BufferUpdate>, ProviderError> {
    let mut buffer_updates = Vec::with_capacity(updates.len());
    for update in updates {
        match update.target {
            LandingTarget::Buffer(index) => buffer_updates.push(BufferUpdate {
                index,
                offset: update.offset,
                bytes: update.bytes,
            }),
            LandingTarget::Texture(index) => {
                return Err(ProviderError::new(
                    ProviderPhase::Resolve,
                    ProviderErrorClass::Internal,
                    "compute_storage_image_executor_unsupported",
                )
                .expect("non-empty provider error slug")
                .with_field("texture", FieldValue::Unsigned(u64::from(index)))
                .with_detail(
                    "a storage image landing reached the executor's buffer-keyed update channel",
                ));
            }
        }
    }
    Ok(buffer_updates)
}

/// The trailing inputs a pool-sequence submission carries beyond the core
/// pipeline/buffer/dispatch arguments. Grouped so the executor entry points
/// stay under the project's seven-argument ceiling while the provider still
/// passes borrowed leases, sampled textures, and the optional indirect
/// dispatch as one unit.
pub(crate) struct SequenceTail<'a> {
    pub borrowed: Option<(Arc<BorrowedLeaseRegistry>, Vec<LeaseId>)>,
    pub textures: &'a [metal_api_core::provider::TextureView],
    pub indirect_dispatch: Option<[u32; 3]>,
}

/// Execute a pool whose bindings are either provider-owned copies or owner
/// host mappings imported without copying. `borrowed` carries the registry and
/// the leases retained for this submission; their Drop retires every retain.
/// `queue_index` is the queue the caller selected with
/// `VulkanContext::pick_queue` and locked; the device enqueue uses that index,
/// so the lock and the enqueue can never describe different queues.
pub(crate) fn execute_pool_sequence_with_status(
    context: &Arc<VulkanContext>,
    artifacts: &[Arc<VulkanPipelineArtifact>],
    buffers: &[PoolBinding],
    dispatches: &[BoundDispatch],
    tail: SequenceTail<'_>,
    queue_index: usize,
) -> Result<Vec<LandingUpdate>, ProviderError> {
    for artifact in artifacts {
        if !Arc::ptr_eq(context, &artifact.context) {
            return Err(dispatch_args_error(failure(
                "pipeline artifact belongs to another Vulkan device",
            )));
        }
    }
    let result =
        execute_submission_stages(context, artifacts, buffers, dispatches, tail, queue_index);
    if result
        .as_ref()
        .is_err_and(|error| error.class == ProviderErrorClass::DeviceLost)
    {
        // Every `DeviceLost`-class error is produced where a driver answer was
        // observed, and that site already routed the loss through the core
        // lifecycle. This is the fail-closed net: a loss-class error must never
        // leave the instance admitting work, whatever path produced it.
        context.mark_device_lost();
    }
    result
}

fn execute_submission_stages(
    context: &Arc<VulkanContext>,
    artifacts: &[Arc<VulkanPipelineArtifact>],
    buffers: &[PoolBinding],
    dispatches: &[BoundDispatch],
    tail: SequenceTail<'_>,
    queue_index: usize,
) -> Result<Vec<LandingUpdate>, ProviderError> {
    let mut pending =
        PendingExecution::submit(context, queue_index, artifacts, buffers, dispatches, tail)?;
    if !pending.wait(FENCE_TIMEOUT_NS)? {
        context.mark_unobservable_submission();
        return Err(ExecutionFailure::vulkan(
            vk::Result::TIMEOUT,
            "compute completion timed out after 20 seconds",
        )
        .into_provider(
            ProviderPhase::Wait,
            ProviderErrorClass::Execute,
            "vulkan-wait",
            CompletionDisposition::SubmittedUnknown { token: None },
        ));
    }
    // The readback is the second half of the CPU↔GPU round trip and the one
    // half a reader most easily mistakes for device latency: it is host-mapped
    // copies, so it is bytes and threads, not the queue.
    let _read_updates = crate::phase_profile::Bar::enter(crate::phase_profile::Phase::ReadUpdates);
    pending.read_updates()
}

/// The resource one landing update belongs to (`research/docs/26` §21.4, C2).
///
/// A buffer landing is keyed by its pool key — the shape the standalone
/// executor's `BufferUpdate` channel already carries. A storage image's bytes
/// belong to a texture view rather than to any Metal buffer index, so the
/// update keeps the texture's own key and the provider maps it onto the
/// writeback channel it owns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LandingTarget {
    /// A buffer pool key (the position of the pooled view).
    Buffer(u32),
    /// The Metal texture index of a storage image.
    Texture(u32),
}

/// One complete resource landing a pool sequence produced: the whole tightly
/// packed extent of a writable buffer view or storage image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LandingUpdate {
    pub target: LandingTarget,
    pub offset: usize,
    pub bytes: Vec<u8>,
}

/// A recorded and queue-submitted sequence whose completion fence is pending.
///
/// `submit` performs planning, resource creation, recording and `queue_submit`;
/// the caller must hold the selected queue's host lock. `wait` observes the
/// device fence and may run on any thread without that lock, so an asynchronous
/// provider no longer needs a worker per submission. Dropping a still-pending
/// value poisons the context and retains every in-flight handle until process
/// exit, matching `ExecutionResources`' unknown-retirement policy.
pub(crate) struct PendingExecution {
    resources: ExecutionResources,
    writable_pool_keys: BTreeSet<u32>,
}

impl PendingExecution {
    pub(crate) fn submit(
        context: &Arc<VulkanContext>,
        queue_index: usize,
        artifacts: &[Arc<VulkanPipelineArtifact>],
        buffers: &[PoolBinding],
        dispatches: &[BoundDispatch],
        tail: SequenceTail<'_>,
    ) -> Result<Self, ProviderError> {
        let mut resources = ExecutionResources::new(Arc::clone(context));
        resources.set_borrowed_leases(tail.borrowed);
        let translated = artifacts
            .iter()
            .map(|artifact| &artifact.translated)
            .collect::<Vec<_>>();
        let planned =
            plan_pipeline_sequence(&translated, buffers, &context.properties.limits, dispatches)?;
        let plans = &planned.plans;
        if let Some(threadgroups) = tail.indirect_dispatch {
            // One indirect command replays exactly one full-workgroup region:
            // a single `vkCmdDispatchIndirect` launches one workgroup count at
            // one local size, so a plan split into partial-tail regions (or
            // spread over several compute passes) has no faithful indirect
            // equivalent. The encoded threadgroups must also equal the planned
            // count, otherwise the footprint proof would describe a different
            // launch than the one the rail replays.
            let [plan] = plans.as_slice() else {
                return Err(indirect_command_refusal(
                    "an indirect dispatch replays exactly one compute pass",
                ));
            };
            let [region] = plan.regions.as_slice() else {
                return Err(indirect_command_refusal(
                    "an indirect dispatch replays a single full-workgroup region",
                ));
            };
            if region.group_count != threadgroups {
                return Err(indirect_command_refusal(format!(
                    "indirect dispatch threadgroups {threadgroups:?} disagree with the planned group count {:?}",
                    region.group_count
                )));
            }
        }
        if plans.iter().all(|plan| plan.regions.is_empty()) {
            return Ok(Self {
                resources,
                writable_pool_keys: BTreeSet::new(),
            });
        }
        // The submission profile's three device-side bars. They are disjoint
        // (`resource_build` ends where `record` starts, and `record` where
        // `queue_submit` does), so their sum against the enclosing `total` bar
        // is an identity a reader can check — see `crate::phase_profile`. All
        // three resolve to `None` after one relaxed load when the profile is
        // off, and none of them reads a clock in that case.
        let _build = crate::phase_profile::Bar::enter(crate::phase_profile::Phase::ResourceBuild);
        resources
            .create_pipeline_objects(&translated, plans)
            .map_err(|error| {
                error.into_provider(
                    ProviderPhase::Compile,
                    ProviderErrorClass::Compile,
                    "vulkan-pipeline-create",
                    CompletionDisposition::NotSubmitted,
                )
            })?;
        let encode_error = |error: ExecutionFailure| {
            error.into_provider(
                ProviderPhase::Encode,
                ProviderErrorClass::Resource,
                "vulkan-encode",
                CompletionDisposition::NotSubmitted,
            )
        };
        resources.create_buffers(buffers).map_err(encode_error)?;
        resources
            .create_textures(tail.textures, dispatches)
            .map_err(encode_error)?;
        resources
            .create_static_samplers(&translated)
            .map_err(encode_error)?;
        resources
            .create_descriptors(&translated, dispatches)
            .map_err(encode_error)?;
        if let Some(threadgroups) = tail.indirect_dispatch {
            resources
                .create_indirect_dispatch(threadgroups)
                .map_err(encode_error)?;
        }
        drop(_build);
        let _record = crate::phase_profile::Bar::enter(crate::phase_profile::Phase::Record);
        resources
            .record(&translated, plans, queue_index)
            .map_err(encode_error)?;
        drop(_record);
        let _queue_submit =
            crate::phase_profile::Bar::enter(crate::phase_profile::Phase::QueueSubmit);
        if let Err(failure) = resources.submit(queue_index) {
            // The provider error is built while `resources` is still owned, so
            // the fault record observed at the driver boundary is attached
            // before an abandonment hands the resources to the context.
            let abandon = failure.is_pending() && !failure.is_device_lost();
            let error = resources.attach_device_loss_fault(failure.into_provider());
            // `ExecutionResources::submit` already routed a driver-reported
            // device loss through the core lifecycle and marked its own
            // handles for destruction. Only an unobservable submission that is
            // *not* a loss is abandoned, which retains its handles.
            if abandon {
                context.abandon(resources);
            }
            return Err(error);
        }
        drop(_queue_submit);
        Ok(Self {
            resources,
            writable_pool_keys: planned.writable_pool_keys,
        })
    }

    /// Wait for the completion fence. `Ok(true)` means the queue retired the
    /// work; `Ok(false)` means the timeout elapsed and the caller may retry.
    pub(crate) fn wait(&mut self, timeout_ns: u64) -> Result<bool, ProviderError> {
        if !self.resources.submitted {
            // No fence behind this submission: the only wait that is exactly
            // free, and the profile counts it as its own population rather than
            // letting it dilute the driver-call reading below.
            crate::phase_profile::note_fence_wait_skipped();
            self.resources.completed = true;
            return Ok(true);
        }
        match self.resources.wait(timeout_ns) {
            Ok(retired) => Ok(retired),
            Err(failure) => Err(self
                .resources
                .attach_device_loss_fault(failure.into_provider())),
        }
    }

    pub(crate) fn read_updates(&self) -> Result<Vec<LandingUpdate>, ProviderError> {
        self.resources
            .read_updates(&self.writable_pool_keys)
            .map_err(ExecutionFailure::into_readback_provider)
    }

    pub(crate) fn owned_bytes(&self) -> u64 {
        self.resources.owned_bytes()
    }

    pub(crate) fn mark_device_lost(&mut self) {
        self.resources.mark_device_lost();
    }

    pub(crate) fn retain_after_budgeted_abandon(&mut self) {
        self.resources.retain_after_budgeted_abandon();
    }
}

fn dispatch_args_error(error: ExecutorError) -> ProviderError {
    ExecutionFailure::from(error).into_provider(
        ProviderPhase::Resolve,
        ProviderErrorClass::Args,
        "vulkan-dispatch-args",
        CompletionDisposition::NotSubmitted,
    )
}

/// The capability refusal the compute rail publishes when an indirect dispatch
/// cannot be faithfully replayed (`research/docs/25` §4.3). It mirrors the
/// render rail's `icb_command_unsupported` so both rails name the same slug for
/// the same "well-formed but wider than this increment" class.
fn indirect_command_refusal(detail: impl Into<String>) -> ProviderError {
    let mut error = ProviderError::new(
        ProviderPhase::Resolve,
        ProviderErrorClass::Capability,
        "icb_command_unsupported",
    )
    .expect("static provider refusal slug");
    error.retryability = Retryability::Never;
    error.with_detail(detail.into())
}

fn validate_serial_dispatches(
    first_dispatch: ([u32; 3], [u32; 3]),
    dispatches: &[([u32; 3], [u32; 3])],
) -> Result<(), ProviderError> {
    validate_dispatch_count(dispatches.len())?;
    if dispatches[0] != first_dispatch {
        return Err(dispatch_args_error(failure(
            "first serial dispatch sizes differ from the submission sizes",
        )));
    }
    Ok(())
}

fn validate_dispatch_count(count: usize) -> Result<(), ProviderError> {
    if !(1..=MAX_SERIAL_DISPATCHES).contains(&count) {
        return Err(dispatch_args_error(failure(format!(
            "serial submission requires 1..={MAX_SERIAL_DISPATCHES} dispatches, got {count}",
        ))));
    }
    Ok(())
}

fn identity_dispatches<T: PoolWidth>(
    buffers: &[T],
    textures: &[metal_api_core::provider::TextureView],
    dispatches: &[([u32; 3], [u32; 3])],
) -> Vec<BoundDispatch> {
    // Textures share the Metal argument index space with buffers (a fixture
    // reports texture 0 and buffer 0), so their pool keys are offset into a
    // separate range and the kind is carried explicitly.
    let texture_base = u32::try_from(buffers.len()).unwrap_or(u32::MAX);
    dispatches
        .iter()
        .map(|&(grid, local)| BoundDispatch {
            grid,
            local,
            bindings: buffers
                .iter()
                .map(|buffer| Binding {
                    metal_index: buffer.pool_index(),
                    key: PoolKey::buffer(buffer.pool_index()),
                    width: buffer.pool_len(),
                })
                .chain(textures.iter().map(|texture| Binding {
                    metal_index: texture.metal_binding,
                    key: PoolKey {
                        kind: PoolKind::Texture,
                        index: texture_base + texture.metal_binding,
                    },
                    width: usize::try_from(texture.expected_bytes().unwrap_or(0)).unwrap_or(0),
                }))
                .collect(),
        })
        .collect()
}

#[cfg(test)]
fn plan_serial_submission<T: PoolWidth>(
    translated: &TranslatedComputePipeline,
    buffers: &[T],
    limits: &vk::PhysicalDeviceLimits,
    first_dispatch: ([u32; 3], [u32; 3]),
    dispatches: &[([u32; 3], [u32; 3])],
) -> Result<Vec<KernelDispatchPlan>, ProviderError> {
    validate_serial_dispatches(first_dispatch, dispatches)?;
    let bound = identity_dispatches(buffers, &[], dispatches);
    Ok(plan_rebound_submission(translated, buffers, limits, &bound)?.plans)
}

#[derive(Debug)]
struct ReboundSubmissionPlan {
    plans: Vec<KernelDispatchPlan>,
    writable_pool_keys: BTreeSet<u32>,
}

#[cfg(test)]
fn plan_rebound_submission<T: PoolWidth>(
    translated: &TranslatedComputePipeline,
    buffers: &[T],
    limits: &vk::PhysicalDeviceLimits,
    dispatches: &[BoundDispatch],
) -> Result<ReboundSubmissionPlan, ProviderError> {
    let translated = vec![translated; dispatches.len()];
    plan_pipeline_sequence(&translated, buffers, limits, dispatches)
}

/// Pure preflight: no request-specific Vulkan objects exist until this returns.
/// Each pass uniquely maps its reflected Metal slots into the shared pool.
/// Every uploaded resource must be used by at least one pass in the sequence.
fn plan_pipeline_sequence<T: PoolWidth>(
    translated: &[&TranslatedComputePipeline],
    buffers: &[T],
    limits: &vk::PhysicalDeviceLimits,
    dispatches: &[BoundDispatch],
) -> Result<ReboundSubmissionPlan, ProviderError> {
    let resolve_capability = |error| {
        ExecutionFailure::from(error).into_provider(
            ProviderPhase::Resolve,
            ProviderErrorClass::Capability,
            "vulkan-dispatch-capability",
            CompletionDisposition::NotSubmitted,
        )
    };
    validate_dispatch_count(dispatches.len())?;
    if translated.len() != dispatches.len() {
        return Err(dispatch_args_error(failure(
            "pipeline artifact count must match dispatch count",
        )));
    }
    if buffers.len() > MAX_SERIAL_RESOURCES {
        return Err(resolve_capability(failure(format!(
            "buffer pool exceeds serial resource limit {MAX_SERIAL_RESOURCES}",
        ))));
    }
    let mut pool = BTreeMap::new();
    for buffer in buffers {
        if pool.insert(buffer.pool_index(), buffer).is_some() {
            return Err(dispatch_args_error(failure(format!(
                "buffer pool key {} occurs more than once",
                buffer.pool_index()
            ))));
        }
    }
    let mut plans = Vec::with_capacity(dispatches.len());
    let mut used_pool_keys = BTreeSet::new();
    let mut writable_pool_keys = BTreeSet::new();
    let mut descriptor_count = 0_u32;
    for (translated, dispatch) in translated.iter().zip(dispatches) {
        let reflection = translated.reflection();
        let reflected_contract = reflection.kernel_dispatch.ok_or_else(|| {
            resolve_capability(failure("translated kernel has no dispatch contract"))
        })?;
        if !matches!(reflected_contract, KernelDispatch::ThreadsDynamic { .. }) {
            return Err(resolve_capability(failure(format!(
                "translated kernel returned unexpected dispatch contract {reflected_contract:?}"
            ))));
        }
        let mut pass_pool_keys = BTreeSet::new();
        let mut widths = Vec::with_capacity(dispatch.bindings.len());
        for binding in &dispatch.bindings {
            if binding.key.kind == PoolKind::Buffer && !pool.contains_key(&binding.key.index) {
                return Err(dispatch_args_error(failure(format!(
                    "unknown buffer pool key {}",
                    binding.key.index
                ))));
            }
            if !pass_pool_keys.insert(binding.key) {
                return Err(dispatch_args_error(failure(format!(
                    "pool key {binding:?} is bound more than once in one pass",
                ))));
            }
            // Buffer width validation speaks about buffer bindings only; a
            // texture binding shares the Metal index space and is checked by
            // `create_textures` instead.
            if binding.key.kind == PoolKind::Buffer {
                widths.push((binding.metal_index, binding.width));
            }
        }
        validate_local_size(limits, dispatch.local).map_err(resolve_capability)?;
        translated
            .validate_binding_widths(&widths, dispatch.grid)
            .map_err(dispatch_args_error)?;
        used_pool_keys.extend(pass_pool_keys);
        for binding in &dispatch.bindings {
            let reflected = reflection
                .bindings
                .iter()
                .find(|reflected| reflected.metal_index == binding.metal_index)
                .expect("validated reflected binding");
            if binding.key.kind == PoolKind::Buffer
                && !matches!(
                    reflected.access,
                    Some(ResourceAccess::Unused | ResourceAccess::ReadOnly)
                )
            {
                writable_pool_keys.insert(binding.key.index);
            }
        }
        translated
            .validate_threadgroup(dispatch.local)
            .map_err(resolve_capability)?;
        let plan = reflected_contract
            .plan(dispatch.local, Some(dispatch.grid))
            .map_err(|error| {
                dispatch_args_error(failure(format!("plan exact dispatch: {error}")))
            })?;
        validate_dispatch_plan(limits, reflected_contract, &plan).map_err(resolve_capability)?;
        validate_descriptor_limits(limits, reflection).map_err(resolve_capability)?;
        // This sum sizes the pool; descriptor device limits apply to each pass.
        descriptor_count = u32::try_from(reflection.bindings.len())
            .ok()
            .and_then(|count| descriptor_count.checked_add(count))
            .ok_or_else(|| resolve_capability(failure("descriptor pool count overflows u32")))?;
        plans.push(plan);
    }
    let used_buffers = used_pool_keys
        .iter()
        .filter(|key| key.kind == PoolKind::Buffer)
        .count();
    if used_buffers != pool.len() {
        return Err(dispatch_args_error(failure(
            "every uploaded buffer pool resource must be bound in at least one pass",
        )));
    }
    for buffer in buffers {
        validate_storage_buffer_size(limits, buffer.pool_index(), buffer.pool_len())
            .map_err(resolve_capability)?;
    }
    Ok(ReboundSubmissionPlan {
        plans,
        writable_pool_keys,
    })
}

fn validate_local_size(
    limits: &vk::PhysicalDeviceLimits,
    local: [u32; 3],
) -> Result<(), ExecutorError> {
    if local.contains(&0) {
        return Err(failure("threadgroup dimensions must be nonzero"));
    }
    for (dimension, &size) in local.iter().enumerate() {
        if size > limits.max_compute_work_group_size[dimension] {
            return Err(failure(format!(
                "threadgroup dimension {dimension}={} exceeds Vulkan limit {}",
                size, limits.max_compute_work_group_size[dimension]
            )));
        }
    }
    let invocations = local
        .into_iter()
        .try_fold(1_u32, u32::checked_mul)
        .ok_or_else(|| failure("threadgroup invocation count overflows u32"))?;
    if invocations > limits.max_compute_work_group_invocations {
        return Err(failure(format!(
            "threadgroup has {invocations} invocations but Vulkan permits {}",
            limits.max_compute_work_group_invocations
        )));
    }
    Ok(())
}

fn validate_dispatch_plan(
    limits: &vk::PhysicalDeviceLimits,
    contract: KernelDispatch,
    plan: &KernelDispatchPlan,
) -> Result<(), ExecutorError> {
    let range = contract
        .push_constant_range()
        .ok_or_else(|| failure("exact dispatch has no push-constant range"))?;
    let end = range
        .offset
        .checked_add(range.size)
        .ok_or_else(|| failure("dispatch push-constant range overflows u32"))?;
    if end > limits.max_push_constants_size {
        return Err(failure(format!(
            "dispatch push constants end at {end}, beyond Vulkan limit {}",
            limits.max_push_constants_size
        )));
    }
    for region in &plan.regions {
        validate_local_size(limits, region.local_size)?;
        for (dimension, &count) in region.group_count.iter().enumerate() {
            if count > limits.max_compute_work_group_count[dimension] {
                return Err(failure(format!(
                    "dispatch group count dimension {dimension}={} exceeds Vulkan limit {}",
                    count, limits.max_compute_work_group_count[dimension]
                )));
            }
        }
    }
    Ok(())
}

fn validate_descriptor_limits(
    limits: &vk::PhysicalDeviceLimits,
    reflection: &ShaderReflection,
) -> Result<(), ExecutorError> {
    let buffer_count = u32::try_from(
        reflection
            .bindings
            .iter()
            .filter(|binding| binding.kind == ResourceKind::Buffer)
            .count(),
    )
    .map_err(|_| failure("reflected buffer count overflows u32"))?;
    // Sampled textures bind as combined image samplers: one sampled image and
    // one sampler per binding.
    let sampled_count = u32::try_from(
        reflection
            .bindings
            .iter()
            .filter(|binding| matches!(binding.kind, ResourceKind::Texture))
            .count(),
    )
    .map_err(|_| failure("reflected texture count overflows u32"))?;
    // Writable storage images bind as `STORAGE_IMAGE` descriptors
    // (`research/docs/26` §21.4, C2), a third descriptor class with its own
    // per-stage and per-set limits.
    let storage_image_count = u32::try_from(
        reflection
            .bindings
            .iter()
            .filter(|binding| matches!(binding.kind, ResourceKind::StorageImage))
            .count(),
    )
    .map_err(|_| failure("reflected storage image count overflows u32"))?;
    // AIR static samplers and runtime `[[sampler(n)]]` bindings each consume
    // one sampler descriptor (`research/docs/26` §21.3, C1b).
    let sampler_count = u32::try_from(
        reflection
            .bindings
            .iter()
            .filter(|binding| {
                matches!(
                    binding.kind,
                    ResourceKind::StaticSampler | ResourceKind::Sampler
                )
            })
            .count(),
    )
    .map_err(|_| failure("reflected sampler count overflows u32"))?;
    if limits.max_bound_descriptor_sets == 0
        || buffer_count > limits.max_per_stage_descriptor_storage_buffers
        || buffer_count > limits.max_descriptor_set_storage_buffers
        || sampled_count > limits.max_per_stage_descriptor_sampled_images
        || sampled_count > limits.max_descriptor_set_sampled_images
        || storage_image_count > limits.max_per_stage_descriptor_storage_images
        || storage_image_count > limits.max_descriptor_set_storage_images
        || sampled_count.saturating_add(sampler_count) > limits.max_per_stage_descriptor_samplers
        || sampled_count.saturating_add(sampler_count) > limits.max_descriptor_set_samplers
        || buffer_count
            .saturating_add(sampled_count)
            .saturating_add(storage_image_count)
            > limits.max_per_stage_resources
    {
        return Err(failure(format!(
            "{buffer_count} storage buffers, {sampled_count} sampled textures, {storage_image_count} storage images and {sampler_count} samplers exceed Vulkan descriptor limits per-stage-buffers={} per-set-buffers={} per-stage-images={} per-stage-samplers={} all-resources={} bound-sets={}",
            limits.max_per_stage_descriptor_storage_buffers,
            limits.max_descriptor_set_storage_buffers,
            limits.max_per_stage_descriptor_sampled_images,
            limits.max_per_stage_descriptor_samplers,
            limits.max_per_stage_resources,
            limits.max_bound_descriptor_sets
        )));
    }
    Ok(())
}

/// Descriptor type one reflected binding needs. Sampled textures use a
/// combined image sampler because the translator synthesizes the sampler and
/// the provider supplies one per sampled image (`research/docs/16` §4.3).
/// An AIR-embedded constexpr sampler (`ResourceKind::StaticSampler`) and a
/// runtime `[[sampler(n)]]` (`ResourceKind::Sampler`) each own a `VkSampler`
/// the provider creates from the reflected state and binds at the binding the
/// reflection names (`research/docs/26` §21.3, C1b). A writable storage image
/// (`ResourceKind::StorageImage`) binds as a plain `STORAGE_IMAGE` with no
/// sampler at all (§21.4, C2).
fn descriptor_type_for_binding(
    binding: &metal2vulkan::reflect::ResourceBinding,
) -> vk::DescriptorType {
    match binding.kind {
        ResourceKind::Texture => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ResourceKind::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        ResourceKind::StaticSampler | ResourceKind::Sampler => vk::DescriptorType::SAMPLER,
        _ => vk::DescriptorType::STORAGE_BUFFER,
    }
}

/// Map one AIR-embedded constexpr sampler state onto the contract's closed
/// sampler family (`research/docs/26` §21.3, C1b).
///
/// The family is the contract's own ([`SamplerPolicy`]): six filters —
/// `{nearest, linear}` minification/magnification, each with the mip filter
/// `{not-mipmapped, nearest, linear}` — crossed with five address modes
/// (`clamp-to-edge`, `repeat`, `mirror-clamp-to-edge`, `mirror-repeat`,
/// `clamp-to-zero`), under normalized coordinates, one address mode on the two
/// axes a 2D view reads, no comparison function, no reduction mode and no
/// anisotropy.
/// `research/docs/23` §109 widened the two first-increment names to this list.
///
/// The **third** addressing axis (`r` in AIR, `addressModeW` in Vulkan) is read
/// on no view this family samples (2026-09-19, R44): every sampled surface the
/// render family admits is one single-sample, non-arrayed 2D view
/// (`render_texture_shape_unsupported` refuses every other reflected shape, and
/// the request side's bind gate refuses every other view), and a 2D sample
/// carries no third coordinate — Vulkan applies `addressModeW` to the r
/// coordinate of a 3D view, while an array's third coordinate *selects* a layer
/// rather than being addressed. A state whose two addressing axes agree is
/// therefore inside the family whatever its `r` says, and the rail spells its
/// own fixed value for that axis on the create-info (the same mode, see
/// [`sampler_create_info`]) instead of carrying a field no sample can reach.
/// The request side folds the same axis under the same proof (R44's
/// `..._state_address_w_folded`, counted rather than dropped silently).
/// Anything else is refused here rather than approximated, because a
/// substituted sampler changes which texels a sample returns without changing
/// the module the state came from.
///
/// Two states the AIR vocabulary can name stay outside the family by name:
///
/// - `bicubic`, whose four taps the family cannot state; and
/// - `clampToBorderColor`, whose border colour is a state of its own
///   (`MTLSamplerBorderColor`) that the family does not name — the family's
///   only border is `clampToZero`'s zero.
///
/// The translator's own vocabulary has no `mirrorClampToEdge`, so that family
/// name is reached through a request that states it (the render pass's
/// `[[sampler(n)]]` state) rather than through an AIR decode; §109's readings
/// execute it on the rail and name the gap.
///
/// [`SamplerPolicy`]: metal_api_core::provider::SamplerPolicy
pub(crate) fn static_sampler_policy(
    state: &metal2vulkan::reflect::StaticSamplerState,
) -> Result<metal_api_core::provider::SamplerPolicy, ExecutorError> {
    use metal2vulkan::reflect::{
        SamplerAddressMode as AirAddress, SamplerCompareFunction, SamplerCoordinates,
        SamplerFilter as AirFilter, SamplerMipFilter as AirMipFilter, SamplerReduction,
    };

    let filter = |filter: AirFilter, mip: AirMipFilter| match (filter, mip) {
        (AirFilter::Nearest, AirMipFilter::None) => {
            Ok(metal_api_core::provider::SamplerFilter::Nearest)
        }
        (AirFilter::Linear, AirMipFilter::None) => {
            Ok(metal_api_core::provider::SamplerFilter::Linear)
        }
        (AirFilter::Nearest, AirMipFilter::Nearest) => {
            Ok(metal_api_core::provider::SamplerFilter::NearestMipNearest)
        }
        (AirFilter::Nearest, AirMipFilter::Linear) => {
            Ok(metal_api_core::provider::SamplerFilter::NearestMipLinear)
        }
        (AirFilter::Linear, AirMipFilter::Nearest) => {
            Ok(metal_api_core::provider::SamplerFilter::LinearMipNearest)
        }
        (AirFilter::Linear, AirMipFilter::Linear) => {
            Ok(metal_api_core::provider::SamplerFilter::LinearMipLinear)
        }
        (AirFilter::Bicubic, _) => Err(failure(
            "the canonical sampler family has no bicubic filter; refusing instead of substituting",
        )),
    };
    let address = |address: AirAddress| match address {
        AirAddress::ClampToEdge => Ok(metal_api_core::provider::SamplerAddressMode::ClampToEdge),
        AirAddress::Repeat => Ok(metal_api_core::provider::SamplerAddressMode::Repeat),
        AirAddress::MirroredRepeat => {
            Ok(metal_api_core::provider::SamplerAddressMode::MirrorRepeat)
        }
        AirAddress::ClampToZero => Ok(metal_api_core::provider::SamplerAddressMode::ClampToZero),
        AirAddress::ClampToBorder => Err(failure(
            "the canonical sampler family states no border colour, so clampToBorderColor is \
             refused by name instead of substituting a sampler that answers another colour",
        )),
    };
    if state.min_filter != state.mag_filter {
        return Err(failure(format!(
            "an AIR sampler whose min ({:?}) and mag ({:?}) filters differ is outside the reviewed family",
            state.min_filter, state.mag_filter
        )));
    }
    // The two axes a 2D view reads state one mode; the third is not read at
    // all (see the note above `sampler_create_info` for the proof and for the
    // fold's own reading on the request side, R44).
    if state.address_mode_s != state.address_mode_t {
        return Err(failure(format!(
            "an AIR sampler whose two addressing axes disagree (s {:?}, t {:?}; r {:?} is not \
             read by any view this family samples) is outside the reviewed family",
            state.address_mode_s, state.address_mode_t, state.address_mode_r
        )));
    }
    if state.coordinates != SamplerCoordinates::Normalized {
        return Err(failure(
            "an AIR sampler with pixel coordinates is outside the reviewed family",
        ));
    }
    if state.compare_function != SamplerCompareFunction::Never {
        return Err(failure(format!(
            "an AIR sampler with compare function {:?} is outside the reviewed family",
            state.compare_function
        )));
    }
    if state.reduction != SamplerReduction::WeightedAverage {
        return Err(failure(format!(
            "an AIR sampler with {:?} reduction is outside the reviewed family",
            state.reduction
        )));
    }
    if state.max_anisotropy != 1 {
        return Err(failure(format!(
            "an AIR sampler with anisotropy {} is outside the reviewed family",
            state.max_anisotropy
        )));
    }
    Ok(metal_api_core::provider::SamplerPolicy {
        filter: filter(state.min_filter, state.mip_filter)?,
        address: address(state.address_mode_s)?,
    })
}

/// The Vulkan create-info for one contract sampler policy.
///
/// Shared by both rails that create a `VkSampler` from a contract policy: the
/// compute narrow class's AIR-embedded state and the render sampler's
/// declaration (`research/docs/23` §3.3, v100), plus the render sampler's
/// widened state family (`research/docs/23` §109).
///
/// The policy's two fields are the states that can move a sample on the
/// canonical one-mip views, and every name maps onto the `VkSamplerCreateInfo`
/// field that decides it:
///
/// | contract | Vulkan |
/// |---|---|
/// | `Nearest` / `Linear` (min and mag) | `magFilter` / `minFilter` |
/// | `…MipNearest` / `…MipLinear` | `mipmapMode` |
/// | `ClampToEdge` / `Repeat` / `MirrorClampToEdge` / `MirrorRepeat` | `addressMode{U,V,W}` |
/// | `ClampToZero` | `addressMode{U,V,W}` = `CLAMP_TO_BORDER` + `borderColor` = transparent black |
///
/// The fields Metal's other sampler state would name are stated here as the
/// family's own fixed values: `minLod`/`maxLod` pinned to `0..=0` (every
/// canonical view carries one mip level, so level zero is the only reachable
/// level and a mip filter can only name the mode it is selected under), no
/// comparison, `unnormalizedCoordinates` false and no anisotropy.
///
/// `addressModeW` is the family's own copy of the mode `addressMode{U,V}`
/// carries: the render family samples one single-sample, non-arrayed 2D view
/// per binding, and a 2D sample reads no third coordinate for a `W` mode to
/// address ([`static_sampler_policy`] carries the whole proof). The create-info
/// still spells the field — a `VkSamplerCreateInfo` with three modes and one
/// value cannot leave one of them to a default — and spelling it with the U/V
/// mode is what makes the folded state the request side admits (R44) execute
/// the same sample program the equal-axes state always did.
pub(crate) fn sampler_create_info(
    policy: metal_api_core::provider::SamplerPolicy,
) -> vk::SamplerCreateInfo<'static> {
    use metal_api_core::provider::{SamplerAddressMode, SamplerFilter};

    let filter = if policy.filter.is_linear() {
        vk::Filter::LINEAR
    } else {
        vk::Filter::NEAREST
    };
    let mip = match policy.filter {
        SamplerFilter::Nearest
        | SamplerFilter::Linear
        | SamplerFilter::NearestMipNearest
        | SamplerFilter::LinearMipNearest => vk::SamplerMipmapMode::NEAREST,
        SamplerFilter::NearestMipLinear | SamplerFilter::LinearMipLinear => {
            vk::SamplerMipmapMode::LINEAR
        }
    };
    let address = match policy.address {
        SamplerAddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        SamplerAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        SamplerAddressMode::MirrorClampToEdge => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
        SamplerAddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        // Metal's `clampToZero` is Vulkan's border mode with the
        // transparent-black border: the built-in border colours need no
        // extension and the family states no colour of its own.
        SamplerAddressMode::ClampToZero => vk::SamplerAddressMode::CLAMP_TO_BORDER,
    };
    vk::SamplerCreateInfo::default()
        .mag_filter(filter)
        .min_filter(filter)
        .mipmap_mode(mip)
        .address_mode_u(address)
        .address_mode_v(address)
        .address_mode_w(address)
        .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
        .min_lod(0.0)
        .max_lod(0.0)
}

/// The device feature one contract address mode needs before a sampler may be
/// created with it, or `None` when the mode needs none.
///
/// `VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE` is only valid on a device
/// created with `samplerMirrorClampToEdge` enabled (`VK_KHR_sampler_mirror_clamp_to_edge`'s
/// promoted feature; asking for the mode without it is undefined behaviour, not
/// an error the driver returns). The other four modes are Vulkan 1.0 core with
/// no feature of their own. The name travels in the refusal, so an observer can
/// tell "this device never reported the feature" from "the mode is outside the
/// family".
pub(crate) const fn sampler_address_mode_feature(
    address: metal_api_core::provider::SamplerAddressMode,
) -> Option<&'static str> {
    match address {
        metal_api_core::provider::SamplerAddressMode::MirrorClampToEdge => {
            Some("samplerMirrorClampToEdge")
        }
        metal_api_core::provider::SamplerAddressMode::ClampToEdge
        | metal_api_core::provider::SamplerAddressMode::Repeat
        | metal_api_core::provider::SamplerAddressMode::MirrorRepeat
        | metal_api_core::provider::SamplerAddressMode::ClampToZero => None,
    }
}

fn validate_storage_buffer_size(
    limits: &vk::PhysicalDeviceLimits,
    index: u32,
    len: usize,
) -> Result<(), ExecutorError> {
    let size =
        u64::try_from(len).map_err(|_| failure(format!("buffer {index} length overflows u64")))?;
    if size > u64::from(limits.max_storage_buffer_range) {
        return Err(failure(format!(
            "buffer {index} length {size} exceeds maxStorageBufferRange {}",
            limits.max_storage_buffer_range
        )));
    }
    Ok(())
}

/// Read one SPIR-V literal string operand (`OpExtension`, `OpSource`).
///
/// The operand is a null-terminated UTF-8 string packed into words, so the
/// decode is the little-endian byte run up to the first NUL. `None` is a
/// malformed operand, which the callers treat as a refusal rather than as an
/// extension the module may carry.
fn spirv_literal_string(words: &[u32]) -> Option<String> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let end = bytes.iter().position(|byte| *byte == 0)?;
    std::str::from_utf8(&bytes[..end]).ok().map(str::to_owned)
}

/// Negate the `y` of every write to a `BuiltIn Position` output variable.
///
/// Metal's NDC is +y up and Vulkan's is +y down. The reviewed hand-written
/// vertex modules carry an `OpFNegate` on the position's `y` for exactly this
/// reason (`research/docs/23` §32, v38); a module that came out of the
/// translator writes the position exactly as its AIR states it, so the
/// alignment has to be put back here (`research/docs/23` §40).
///
/// For every `OpStore` into such a variable the rewrite inserts one
/// `OpCompositeExtract`/`OpFNegate`/`OpCompositeInsert` triple and stores the
/// rebuilt vector instead:
///
/// ```text
/// %y  = OpCompositeExtract %float %position 1
/// %ny = OpFNegate %float %y
/// %p  = OpCompositeInsert %v4float %ny %position 1
/// OpStore %gl_Position %p
/// ```
///
/// This is a rewrite rather than a `TransformOptions` switch because the
/// pinned translator exposes no such option. It stays inside the translated
/// vertex arm: the hand-written modules never pass through it, and the native
/// (macOS Metal) rail never sees it. A vertex module with no position store is
/// refused rather than silently left unaligned.
fn negate_position_y(spirv: &[u8]) -> Result<Vec<u8>, ExecutorError> {
    if !spirv.len().is_multiple_of(4) {
        return Err(failure("translated SPIR-V is not word aligned"));
    }
    let words = spirv
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    if words.len() < 5 {
        return Err(failure("translated SPIR-V has an invalid header"));
    }
    // Pass 1: the shape the rewrite needs — which variables are `Position`
    // outputs, and the pointer/vector/float types their stores carry.
    let mut position_variables = BTreeSet::new();
    let mut pointer_pointee = BTreeMap::new();
    let mut vector_components = BTreeMap::new();
    let mut float_widths = BTreeMap::new();
    let mut variable_types = BTreeMap::new();
    let mut cursor = 5;
    while cursor < words.len() {
        let header = words[cursor];
        let word_count = (header >> 16) as usize;
        let opcode = header & 0xffff;
        let end = cursor
            .checked_add(word_count)
            .filter(|end| word_count != 0 && *end <= words.len())
            .ok_or_else(|| {
                failure(format!(
                    "translated SPIR-V has a malformed instruction at word {cursor}"
                ))
            })?;
        if opcode == Op::Decorate as u32 && word_count == 4 {
            if words[cursor + 2] == Decoration::BuiltIn as u32
                && words[cursor + 3] == BuiltIn::Position as u32
            {
                position_variables.insert(words[cursor + 1]);
            }
        } else if opcode == Op::TypePointer as u32 && word_count == 4 {
            pointer_pointee.insert(words[cursor + 1], words[cursor + 3]);
        } else if opcode == Op::TypeVector as u32 && word_count == 4 {
            vector_components.insert(words[cursor + 1], words[cursor + 2]);
        } else if opcode == Op::TypeFloat as u32 && word_count == 3 {
            float_widths.insert(words[cursor + 1], words[cursor + 2]);
        } else if opcode == Op::Variable as u32 && word_count >= 4 {
            variable_types.insert(words[cursor + 2], words[cursor + 1]);
        }
        cursor = end;
    }
    if position_variables.is_empty() {
        return Err(failure(
            "translated vertex SPIR-V has no BuiltIn Position output",
        ));
    }
    // The position pointer's pointee is a float vector; the rewrite needs the
    // component type for the extract/negate and the vector type for the
    // insert.
    let mut position_types = BTreeMap::new();
    for variable in &position_variables {
        let pointer = variable_types.get(variable).ok_or_else(|| {
            failure("translated vertex SPIR-V decorates a non-variable with BuiltIn Position")
        })?;
        let vector = pointer_pointee.get(pointer).ok_or_else(|| {
            failure(
                "translated vertex SPIR-V writes BuiltIn Position through an unknown pointer type",
            )
        })?;
        let component = vector_components.get(vector).ok_or_else(|| {
            failure("translated vertex SPIR-V writes BuiltIn Position to a non-vector type")
        })?;
        if float_widths.get(component) != Some(&32) {
            return Err(failure(
                "translated vertex SPIR-V writes BuiltIn Position to a non-float32 type",
            ));
        }
        position_types.insert(*variable, (*vector, *component));
    }
    // Pass 2: insert the negation in front of every store into one of them.
    let mut bound = words[3];
    let mut rewritten = Vec::with_capacity(words.len() + 16);
    rewritten.extend_from_slice(&words[..5]);
    let mut stores = 0usize;
    let mut cursor = 5;
    while cursor < words.len() {
        let word_count = (words[cursor] >> 16) as usize;
        let opcode = words[cursor] & 0xffff;
        // Pass 1 checked every instruction's length.
        let end = cursor + word_count;
        let position = if opcode == Op::Store as u32 && word_count >= 3 {
            position_types.get(&words[cursor + 1]).copied()
        } else {
            None
        };
        match position {
            Some((vector, component)) => {
                let value = words[cursor + 2];
                let y = bound;
                let negated = bound + 1;
                let inserted = bound + 2;
                bound += 3;
                rewritten.push((5 << 16) | Op::CompositeExtract as u32);
                rewritten.extend_from_slice(&[component, y, value, 1]);
                rewritten.push((4 << 16) | Op::FNegate as u32);
                rewritten.extend_from_slice(&[component, negated, y]);
                rewritten.push((6 << 16) | Op::CompositeInsert as u32);
                rewritten.extend_from_slice(&[vector, inserted, negated, value, 1]);
                rewritten.push(words[cursor]);
                rewritten.extend_from_slice(&[words[cursor + 1], inserted]);
                if word_count > 3 {
                    rewritten.extend_from_slice(&words[cursor + 3..end]);
                }
                stores += 1;
            }
            None => rewritten.extend_from_slice(&words[cursor..end]),
        }
        cursor = end;
    }
    if stores == 0 {
        return Err(failure(
            "translated vertex SPIR-V never stores to its BuiltIn Position output",
        ));
    }
    rewritten[3] = bound;
    Ok(rewritten
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<u8>>())
}

/// The device-independent shape checks plus `policy`'s capability subset.
///
/// `policy` is the device's own answer ([`SpirvFeaturePolicy`]); the phase-1
/// capabilities stay unconditional, `FloatControls2` is admitted exactly when
/// the policy carries it, and every other capability or `OpExtension` name
/// keeps the refusal it had before the gate grew a device-derived arm.
fn validate_spirv_capabilities(
    spv: &[u8],
    policy: SpirvFeaturePolicy,
) -> Result<(), ExecutorError> {
    if !spv.len().is_multiple_of(4) {
        return Err(failure("translated SPIR-V is not word aligned"));
    }
    let words = spv
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(failure("translated SPIR-V has an invalid header"));
    }
    let mut cursor = 5;
    while cursor < words.len() {
        let header = words[cursor];
        let word_count = (header >> 16) as usize;
        let opcode = header & 0xffff;
        let end = cursor
            .checked_add(word_count)
            .filter(|end| word_count != 0 && *end <= words.len())
            .ok_or_else(|| {
                failure(format!(
                    "translated SPIR-V has a malformed instruction at word {cursor}"
                ))
            })?;
        if opcode == Op::Capability as u32 {
            if word_count != 2 {
                return Err(failure("SPIR-V OpCapability has invalid length"));
            }
            let capability = words[cursor + 1];
            // Vulkan 1.0 core capabilities: Shader is required by every
            // module, ImageQuery by texture size/level queries, and
            // Sampled1D/SampledBuffer cover the linear and buffer texture
            // shapes the reviewed fixtures use. `FloatControls2` rides the
            // device's own policy: the translator demands it for a float
            // result that withholds a fast-math permission, and only a device
            // that enabled `VK_KHR_shader_float_controls2` admits it.
            // Everything else stays refused until a capability gate admits a
            // provider feature.
            if !matches!(
                capability,
                value if value == Capability::Shader as u32
                    || value == Capability::ImageQuery as u32
                    // The reviewed texture fixtures return an i8 status
                    // beside the texel; the device enables shaderInt8.
                    || value == Capability::Int8 as u32
                    // Index arithmetic in the reviewed fixtures widens to
                    // i64; the device enables shaderInt64.
                    || value == Capability::Int64 as u32
                    || value == Capability::Sampled1D as u32
                    || value == Capability::SampledBuffer as u32
                    || (policy.float_controls2()
                        && value == Capability::FloatControls2 as u32)
            ) {
                return Err(failure(format!(
                    "SPIR-V capability {capability} requires a Vulkan feature outside the Phase 1 subset"
                )));
            }
        } else if opcode == Op::Extension as u32 {
            // The one extension the gate can admit is the name that belongs to
            // the capability above; it is admitted on the same policy bit, and
            // a module naming any other extension stays refused whatever the
            // policy says.
            let extension = spirv_literal_string(&words[cursor + 1..end]);
            if !(policy.float_controls2() && extension.as_deref() == Some(SPV_KHR_FLOAT_CONTROLS2))
            {
                return Err(failure(
                    "SPIR-V extensions are outside the Phase 1 feature subset",
                ));
            }
        }
        cursor = end;
    }
    Ok(())
}

/// Create a `VkImage` and bind allocated memory of a requested property class.
///
/// The sampled-texture rail and the render-attachment rail need the same four
/// steps (create image, read its memory requirements, pick a satisfying type,
/// allocate and bind); only the `VkImageCreateInfo` and the property class
/// differ. Keeping the sequence in one place means a new image-backed resource
/// cannot forget the memory-type check or leave a half-created image behind on
/// failure (`research/docs/23` §6 Step 3b).
///
/// On failure the image and memory created so far are destroyed here, so a
/// caller only has to clean up what it created before this call.
fn allocate_image_backing(
    context: &VulkanContext,
    info: &vk::ImageCreateInfo,
    properties: vk::MemoryPropertyFlags,
    what: &str,
) -> Result<(vk::Image, vk::DeviceMemory, vk::MemoryRequirements), ExecutionFailure> {
    let image = unsafe { context.device.create_image(info, None) }.map_err(|error| {
        ExecutionFailure::vulkan(error, format!("create {what} image: {error}"))
    })?;
    let requirements = unsafe { context.device.get_image_memory_requirements(image) };
    let memory_type = match context.memory_type(requirements.memory_type_bits, properties) {
        Ok(index) => index,
        Err(error) => {
            unsafe { context.device.destroy_image(image, None) };
            return Err(error.into());
        }
    };
    let allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    let memory = match unsafe { context.device.allocate_memory(&allocation, None) } {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { context.device.destroy_image(image, None) };
            return Err(ExecutionFailure::vulkan(
                error,
                format!("allocate {what} memory: {error}"),
            ));
        }
    };
    if let Err(error) = unsafe { context.device.bind_image_memory(image, memory, 0) } {
        unsafe {
            context.device.destroy_image(image, None);
            context.device.free_memory(memory, None);
        }
        return Err(ExecutionFailure::vulkan(
            error,
            format!("bind {what} memory: {error}"),
        ));
    }
    Ok((image, memory, requirements))
}

/// Create the single-mip, single-layer 2D colour view over `image`.
///
/// Shared by the sampled-texture rail and the render-attachment rail; the
/// aspect mask is `COLOR` for both because neither admits a depth/stencil or
/// plane-disjoint format (`research/docs/23` §3.3).
fn create_depth_image_view(
    context: &VulkanContext,
    image: vk::Image,
    format: vk::Format,
    what: &str,
) -> Result<vk::ImageView, ExecutionFailure> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe { context.device.create_image_view(&info, None) }
        .map_err(|error| ExecutionFailure::vulkan(error, format!("create {what} view: {error}")))
}

/// The stencil sibling of [`create_depth_image_view`]: the same 2D,
/// single-mip, single-layer view over the image's **stencil** aspect, which is
/// what a `stencil8` attachment's image needs (`research/docs/23` §3.3, v47).
fn create_stencil_image_view(
    context: &VulkanContext,
    image: vk::Image,
    format: vk::Format,
    what: &str,
) -> Result<vk::ImageView, ExecutionFailure> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::STENCIL,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe { context.device.create_image_view(&info, None) }
        .map_err(|error| ExecutionFailure::vulkan(error, format!("create {what} view: {error}")))
}

/// The combined view of a `D32_SFLOAT_S8_UINT` attachment: one 2D,
/// single-mip, single-layer view over **both** the depth and the stencil
/// aspect, which is what a combined depth-stencil attachment names in the
/// render pass (`research/docs/23` §3.3, v60). The two aspect-only views above
/// serve the single-surface shapes; the combined shape binds this one view to
/// the one attachment both faces share.
fn create_depth_stencil_image_view(
    context: &VulkanContext,
    image: vk::Image,
    format: vk::Format,
    what: &str,
) -> Result<vk::ImageView, ExecutionFailure> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe { context.device.create_image_view(&info, None) }
        .map_err(|error| ExecutionFailure::vulkan(error, format!("create {what} view: {error}")))
}

fn create_color_image_view(
    context: &VulkanContext,
    image: vk::Image,
    format: vk::Format,
    what: &str,
) -> Result<vk::ImageView, ExecutionFailure> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe { context.device.create_image_view(&info, None) }
        .map_err(|error| ExecutionFailure::vulkan(error, format!("create {what} view: {error}")))
}

fn validate_pipeline_reflection(
    requested_entry: &str,
    reflection: &ShaderReflection,
) -> Result<(), ExecutorError> {
    if reflection.stage != ShaderStage::Kernel {
        return Err(failure("pipeline reflection is not a compute kernel"));
    }
    if reflection.entry_point.as_deref() != Some(requested_entry) {
        return Err(failure(format!(
            "requested entry {:?}, translated entry is {:?}",
            requested_entry, reflection.entry_point
        )));
    }
    if reflection.descriptor_layout.set != 0 {
        return Err(failure(format!(
            "Phase 1 requires descriptor set 0, got {}",
            reflection.descriptor_layout.set
        )));
    }
    if reflection.bindings.is_empty() {
        return Err(failure(
            "Phase 1 requires at least one reflected Metal buffer",
        ));
    }
    if !reflection.argument_buffer_fields.is_empty()
        || !reflection.vertex_attributes.is_empty()
        || !reflection.varyings.is_empty()
        || !reflection.render_targets.is_empty()
        || !reflection.depth_members.is_empty()
        || reflection.depth_qualifier.is_some()
        || !reflection.stencil_members.is_empty()
        || reflection.vertex_builtins.is_some()
        || reflection.tessellation.is_some()
        || !reflection.imageblock_layouts.is_empty()
        || !reflection.implicit_imageblock_attachments.is_empty()
        || reflection.fragment_imageblock.is_some()
        || !reflection.runtime_sampler_specializations.is_empty()
        || !reflection.runtime_storage_image_specializations.is_empty()
        || !reflection.function_constants.is_empty()
    {
        return Err(failure(
            "pipeline uses Metal resources or specialization state outside the Phase 1 buffer-compute subset",
        ));
    }
    let mut descriptor_bindings = BTreeSet::new();
    for binding in &reflection.bindings {
        // The reviewed compute face: Metal buffers, sampled textures, writable
        // storage images, and the AIR-embedded constexpr samplers this rail
        // executes (`research/docs/26` §21.3–§21.4, C1b/C2). Everything else —
        // runtime `[[sampler(n)]]`, color inputs, imageblocks, texture arrays —
        // stays outside the subset and is refused by name.
        if !matches!(
            binding.kind,
            ResourceKind::Buffer
                | ResourceKind::Texture
                | ResourceKind::StorageImage
                | ResourceKind::StaticSampler
        ) {
            return Err(failure(format!(
                "the compute texture subset supports Metal buffers, sampled textures, writable storage images and AIR static samplers, not {:?}",
                binding.kind
            )));
        }
        // A texture and a buffer may share the Metal argument index; the
        // Vulkan descriptor binding is the unique key (checked below).
        let what = match binding.kind {
            ResourceKind::Buffer => "buffer",
            ResourceKind::Texture => "texture",
            ResourceKind::StorageImage => "storage image",
            _ => "static sampler",
        };
        let descriptor = binding.descriptor.ok_or_else(|| {
            failure(format!(
                "Metal {what} {} has no Vulkan descriptor",
                binding.metal_index
            ))
        })?;
        if descriptor.set != 0 || descriptor.count != 1 {
            return Err(failure(format!(
                "Metal {what} {} uses unsupported descriptor set={} count={}",
                binding.metal_index, descriptor.set, descriptor.count
            )));
        }
        if !descriptor_bindings.insert(descriptor.binding) {
            return Err(failure(format!(
                "duplicate Vulkan descriptor binding {}",
                descriptor.binding
            )));
        }
        if binding.kind == ResourceKind::Texture {
            if binding.access != Some(ResourceAccess::Sampled) {
                return Err(failure(format!(
                    "Metal texture {} is not a sampled read ({:?})",
                    binding.metal_index, binding.access
                )));
            }
            if binding.texture_shape.is_none() {
                return Err(failure(format!(
                    "Metal texture {} has no reflected shape",
                    binding.metal_index
                )));
            }
            continue;
        }
        if binding.kind == ResourceKind::StorageImage {
            // A storage image is the writable sibling of the sampled face
            // (`research/docs/26` §21.4, C2). The shape is checked where the
            // contract is derived (`map_storage_image_binding`), so this gate
            // only states what the execution path needs: one writable
            // classification and a reflected shape to address it with.
            if binding.access != Some(ResourceAccess::Storage) {
                return Err(failure(format!(
                    "Metal storage image {} is not a writable storage classification ({:?})",
                    binding.metal_index, binding.access
                )));
            }
            if binding.texture_shape.is_none() {
                return Err(failure(format!(
                    "Metal storage image {} has no reflected shape",
                    binding.metal_index
                )));
            }
            continue;
        }
        if binding.kind == ResourceKind::StaticSampler {
            // The state itself is checked where the pipeline contract is
            // derived (`static_sampler_policy`); here only the descriptor
            // location is checked, exactly as for the two resource kinds
            // above.
            continue;
        }
        if binding.extent.is_none() {
            return Err(failure(format!(
                "Metal buffer {} does not have a reflected extent",
                binding.metal_index
            )));
        }
        let footprint = binding.footprint.as_ref().ok_or_else(|| {
            failure(format!(
                "Metal buffer {} has no executable access footprint",
                binding.metal_index
            ))
        })?;
        if binding.access.is_none() {
            return Err(failure(format!(
                "Metal buffer {} has no access classification",
                binding.metal_index
            )));
        }
        if footprint.has_unbounded_access {
            return Err(failure(format!(
                "Metal buffer {} has data-dependent or unbounded access outside the Phase 1 subset",
                binding.metal_index
            )));
        }
        for access in &footprint.strided_accesses {
            for term in &access.terms {
                if !matches!(
                    term.source,
                    BufferIndexSource::GlobalInvocationIdX
                        | BufferIndexSource::GlobalInvocationIdY
                        | BufferIndexSource::GlobalInvocationIdZ
                ) {
                    return Err(failure(format!(
                        "Metal buffer {} uses unsupported {:?} indexed access",
                        binding.metal_index, term.source
                    )));
                }
            }
        }
    }
    if !matches!(
        reflection.kernel_dispatch,
        Some(KernelDispatch::ThreadsDynamic { .. })
    ) {
        return Err(failure(
            "pipeline did not reflect an exact dynamic dispatch",
        ));
    }
    Ok(())
}

fn validate_bound_buffers(
    reflection: &ShaderReflection,
    buffers: &[(u32, usize)],
    grid: [u32; 3],
) -> Result<(), ExecutorError> {
    // Buffer indices only. Textures and storage images share the Metal argument
    // index space with buffers (`research/docs/26` §21.3–§21.4): a storage
    // image alone at index 0 is not a buffer slot the caller failed to bind, and
    // the checks below answer buffer questions only.
    let metal_indices = reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::Buffer)
        .map(|binding| binding.metal_index)
        .collect::<BTreeSet<_>>();
    let mut supplied = BTreeSet::new();
    for &(index, len) in buffers {
        if len == 0 {
            return Err(failure(format!("buffer {index} has an empty bound range")));
        }
        if !supplied.insert(index) {
            return Err(failure(format!("buffer {index} is bound more than once")));
        }
        let reflected = reflection
            .bindings
            .iter()
            .find(|candidate| {
                candidate.metal_index == index && candidate.kind == ResourceKind::Buffer
            })
            .ok_or_else(|| failure(format!("buffer {index} is not reflected")))?;
        let mut required = u64::from(reflected.declared_size.unwrap_or(0));
        if let Some(BufferExtent::Object { bytes }) = reflected.extent {
            required = required.max(u64::from(bytes));
        }
        let footprint = reflected
            .footprint
            .as_ref()
            .expect("pipeline validation requires a footprint");
        for range in &footprint.static_ranges {
            let end = range
                .offset
                .checked_add(range.size)
                .ok_or_else(|| failure(format!("buffer {index} footprint overflows u64")))?;
            required = required.max(end);
        }
        required = required.max(
            strided_footprint_reach(footprint, grid)
                .map_err(|error| failure(format!("buffer {index} {error}")))?,
        );
        let supplied_len = u64::try_from(len)
            .map_err(|_| failure(format!("buffer {index} length overflows u64")))?;
        ensure_buffer_reach(index, supplied_len, required)?;
    }
    if supplied != metal_indices {
        return Err(failure(format!(
            "bound Metal buffer indices {supplied:?} do not match reflection {metal_indices:?}"
        )));
    }
    Ok(())
}

fn ensure_buffer_reach(index: u32, supplied_len: u64, required: u64) -> Result<(), ExecutorError> {
    if supplied_len < required {
        return Err(failure(format!(
            "buffer {index} length {supplied_len} is shorter than reflected reach {required}"
        )));
    }
    Ok(())
}

fn strided_footprint_reach(
    footprint: &BufferFootprint,
    grid: [u32; 3],
) -> Result<u64, &'static str> {
    let mut required = 0_u64;
    for access in &footprint.strided_accesses {
        let mut end = access
            .base_offset
            .checked_add(access.access_size)
            .ok_or("strided footprint overflows u64")?;
        for term in &access.terms {
            let dimension = match term.source {
                BufferIndexSource::GlobalInvocationIdX => 0,
                BufferIndexSource::GlobalInvocationIdY => 1,
                BufferIndexSource::GlobalInvocationIdZ => 2,
                _ => return Err("uses an unsupported index source"),
            };
            let maximum = u64::from(
                grid[dimension]
                    .checked_sub(1)
                    .ok_or("uses an empty dispatch dimension")?,
            );
            let contribution = maximum
                .checked_mul(term.stride)
                .ok_or("strided footprint overflows u64")?;
            end = end
                .checked_add(contribution)
                .ok_or("strided footprint overflows u64")?;
        }
        required = required.max(end);
    }
    Ok(required)
}

/// One execution buffer, either copied into provider memory or imported from
/// an owner host mapping.
#[derive(Debug)]
pub(crate) enum PoolBinding {
    Owned(BufferBinding),
    /// One device buffer per allocation, shared by every owned view of it.
    ///
    /// `index` stays the pool key (the view), so the pool key space, the
    /// planner, the writable-key set and the writeback mapping are unchanged.
    /// `allocation` identifies the shared device buffer, and the view is its
    /// `[offset, offset + length)` window. Only the first entry of an allocation
    /// carries `bytes`; the image is uploaded once. `research/docs/15` §3.
    SharedOwned {
        index: u32,
        allocation: u64,
        offset: usize,
        length: usize,
        /// Whether the view can read. A write-only view uploads nothing.
        access: metal_api_core::provider::BufferAccess,
        bytes: Vec<u8>,
    },
    /// One device buffer bound at a placement offset inside a heap slab.
    ///
    /// `index`/`allocation`/`offset`/`length` keep the same meaning as
    /// [`Self::SharedOwned`]: the pool key is the view, the allocation
    /// identifies the buffer, and the view is its `[offset, offset + length)`
    /// window. `heap_offset` is the buffer's base inside the slab and
    /// `heap_size` is the total slab byte size; both come straight from the
    /// trace's heap payload (`research/docs/25-heaps与ICB设计.md` §6 Step 3).
    /// `allocation_size` is the allocation's full byte size from its
    /// `AllocationRecord`: the device buffer spans exactly this many bytes so
    /// the buffer extent and the placement's `byte_size` stay equal even when
    /// a view addresses only a prefix of the allocation.
    HeapOwned {
        index: u32,
        allocation: u64,
        offset: usize,
        length: usize,
        access: metal_api_core::provider::BufferAccess,
        bytes: Vec<u8>,
        allocation_size: usize,
        heap_offset: usize,
        heap_size: usize,
    },
    Imported {
        index: u32,
        pointer: usize,
        len: usize,
        capacity: usize,
    },
}

impl PoolBinding {
    pub(crate) fn index(&self) -> u32 {
        match self {
            Self::Owned(binding) => binding.index,
            Self::SharedOwned { index, .. }
            | Self::HeapOwned { index, .. }
            | Self::Imported { index, .. } => *index,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Owned(binding) => binding.bytes.len(),
            // The reflected binding width is the view, not the shared backing.
            Self::SharedOwned { length, .. } | Self::HeapOwned { length, .. } => *length,
            Self::Imported { len, .. } => *len,
        }
    }
}

/// Where one pool key lives inside a device buffer: the shared backing plus the
/// view's window inside it. A per-view binding is the same shape with offset
/// zero and the whole buffer as its window.
struct ViewWindow {
    buffer_key: u64,
    offset: usize,
    length: usize,
}

/// Width view shared by owned bindings and no-copy pool bindings.
trait PoolWidth {
    fn pool_index(&self) -> u32;
    fn pool_len(&self) -> usize;
}

impl PoolWidth for BufferBinding {
    fn pool_index(&self) -> u32 {
        self.index
    }

    fn pool_len(&self) -> usize {
        self.bytes.len()
    }
}

impl PoolWidth for PoolBinding {
    fn pool_index(&self) -> u32 {
        self.index()
    }

    fn pool_len(&self) -> usize {
        self.len()
    }
}

struct GpuBuffer {
    index: u64,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Byte offset of `buffer` within `memory`. Owned and imported backings
    /// bind at zero; a heap-placed buffer is bound at its placement offset
    /// inside the shared slab, so `buffer` and `memory` no longer share a
    /// base address.
    bind_offset: usize,
    len: usize,
    /// Borrowed mappings are the owner's pointer; owned mappings are the
    /// device mapping made at creation time and kept for sparse uploads.
    host_pointer: Option<usize>,
    mapping: Option<usize>,
    /// Byte ranges already copied into this backing, keyed by their view
    /// offset. A write-only view copies nothing in and leaves no entry.
    uploaded_ranges: BTreeMap<usize, usize>,
}

/// One texture owned by an execution. A sampled texture is a host-visible
/// linear `VkImage` and the sampler the provider supplies for it, because the
/// translator synthesizes the sampler (`research/docs/16` §4.3); a writable
/// storage image carries no sampler and travels through a
/// [`GpuStorageImage`] transfer buffer instead (`research/docs/26` §21.4, C2).
/// Only D2 single-sample R32 textures are admitted on either face.
struct GpuTexture {
    index: u64,
    /// The descriptor writer resolves the texture through the pass binding
    /// map, so the pool key is stored with the image.
    pool_key: PoolKey,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// `vk::Sampler::null()` for a storage image, which binds as a plain
    /// `STORAGE_IMAGE` descriptor.
    sampler: vk::Sampler,
    /// The transfer backing of a storage image; `None` for a sampled texture.
    storage: Option<GpuStorageImage>,
}

/// The host-visible transfer buffer one storage image's bytes travel through
/// (`research/docs/26` §21.4, C2). It holds the view's initial contents before
/// the first dispatch and the landed texels after the last one, both tightly
/// packed, and stays mapped for the execution's lifetime.
struct GpuStorageImage {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapping: usize,
    len: usize,
    /// The extent the copy regions cover: the view's own texel extent.
    extent: vk::Extent3D,
}

/// One `VkSampler` created for an AIR-embedded constexpr sampler binding
/// (`research/docs/26` §21.3, C1b). The module's own state is the create-info,
/// so the sampler a pass executes with is the one its SPIR-V was lowered
/// against.
struct GpuStaticSampler {
    /// The descriptor binding the reflection named for this sampler.
    binding: u32,
    sampler: vk::Sampler,
}

/// A pass owns every object derived from its shader's reflection. Keeping this
/// ownership separate prevents using one shader's layout for a later shader.
struct PipelineObjects {
    context: Arc<VulkanContext>,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader: vk::ShaderModule,
    pipelines: BTreeMap<[u32; 3], vk::Pipeline>,
}

struct ExecutionResources {
    context: Arc<VulkanContext>,
    pipeline_objects: Vec<PipelineObjects>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    queue_index: usize,
    submitted: bool,
    completed: bool,
    device_lost: bool,
    /// `VK_EXT_device_fault` record of the loss this submission observed.
    device_loss_fault: Option<DeviceFaultSnapshot>,
    leak_is_budgeted: bool,
    buffers: Vec<GpuBuffer>,
    /// The single heap slab backing every heap-placed buffer in this
    /// submission. Allocated once, mapped once and freed once at `Drop`;
    /// heap-placed `GpuBuffer`s carry `memory == null()` so their `Drop` pass
    /// never double-frees the slab (`research/docs/25-heaps与ICB设计.md` §6
    /// Step 3).
    heap_memory: Option<vk::DeviceMemory>,
    /// Byte size of `heap_memory`, for the abandonment budget.
    heap_bytes: Option<u64>,
    /// Sampled textures addressed by their Metal argument index.
    textures: Vec<GpuTexture>,
    /// Samplers created for the translated modules' AIR-embedded constexpr
    /// samplers, addressed by their reflected descriptor binding
    /// (`research/docs/26` §21.3, C1b).
    static_samplers: Vec<GpuStaticSampler>,
    /// Pool key to its window in `buffers`. `research/docs/15` §3.
    /// Pool key (kind plus index) to its window in `buffers` or `textures`.
    view_windows: BTreeMap<PoolKey, ViewWindow>,
    /// The host-visible `INDIRECT_BUFFER` an indirect dispatch replays from.
    /// Null when this submission dispatches directly (`research/docs/25` §6
    /// Step 4).
    indirect_buffer: vk::Buffer,
    indirect_memory: vk::DeviceMemory,
    borrowed: Option<(Arc<BorrowedLeaseRegistry>, Vec<LeaseId>)>,
}

/// How `ExecutionResources::drop` must treat still-submitted handles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResourceDropPolicy {
    Destroy,
    Retain,
}

fn resource_drop_policy(submitted: bool, completed: bool, device_lost: bool) -> ResourceDropPolicy {
    if submitted && !completed && !device_lost {
        ResourceDropPolicy::Retain
    } else {
        ResourceDropPolicy::Destroy
    }
}

/// One teardown step for an interrupted heap-slab bind.
///
/// Vulkan requires every `VkBuffer` bound to a `VkDeviceMemory` be destroyed
/// before that memory is freed, so the order below is load-bearing.
#[derive(Debug, Eq, PartialEq)]
enum HeapSlabCleanup {
    Destroy(vk::Buffer),
    Free,
}

/// Compute the teardown order for a heap slab whose `failing` buffer could not
/// be bound: the failing buffer, then every buffer already bound by earlier
/// iterations, and only then `Free`. Extracted as a pure function so a unit
/// test can pin "free last" without a device.
fn heap_slab_bind_failure_cleanup(
    failing: vk::Buffer,
    already_bound: &[vk::Buffer],
) -> Vec<HeapSlabCleanup> {
    let mut steps = Vec::with_capacity(already_bound.len() + 2);
    steps.push(HeapSlabCleanup::Destroy(failing));
    steps.extend(already_bound.iter().copied().map(HeapSlabCleanup::Destroy));
    steps.push(HeapSlabCleanup::Free);
    steps
}

struct ExecutionFailure {
    result: Option<vk::Result>,
    detail: String,
}

impl ExecutionFailure {
    fn vulkan(result: vk::Result, detail: impl Into<String>) -> Self {
        Self {
            result: Some(result),
            detail: detail.into(),
        }
    }

    fn into_provider(
        self,
        phase: ProviderPhase,
        class: ProviderErrorClass,
        slug: &'static str,
        completion: CompletionDisposition,
    ) -> ProviderError {
        let device_lost = self.result == Some(vk::Result::ERROR_DEVICE_LOST);
        let class = if device_lost {
            ProviderErrorClass::DeviceLost
        } else {
            class
        };
        let completion = if device_lost
            && matches!(completion, CompletionDisposition::SubmittedUnknown { .. })
        {
            CompletionDisposition::DeviceLost { token: None }
        } else {
            completion
        };
        let mut error = ProviderError::new(phase, class, slug)
            .expect("static Vulkan error slug")
            .with_completion(completion)
            .with_detail(self.detail);
        if device_lost {
            // The driver's own answer is the evidence for a loss, and the
            // documented recovery is the same one the core refusal spells:
            // recreate the provider. Other failures keep the caller's class and
            // retryability and carry the driver text in `detail` only.
            error.retryability = Retryability::RetryAfterRecreate;
            error = error
                .with_field(
                    "vk_result",
                    FieldValue::Text(vk_result_name(vk::Result::ERROR_DEVICE_LOST)),
                )
                .with_field(
                    "vk_result_raw",
                    FieldValue::Signed(i64::from(vk::Result::ERROR_DEVICE_LOST.as_raw())),
                );
        }
        error
    }

    fn into_readback_provider(self) -> ProviderError {
        self.into_provider(
            ProviderPhase::Readback,
            ProviderErrorClass::Execute,
            "vulkan-readback",
            CompletionDisposition::Failed { token: None },
        )
    }
}

impl From<ExecutorError> for ExecutionFailure {
    fn from(error: ExecutorError) -> Self {
        Self {
            result: None,
            detail: error.to_string(),
        }
    }
}

enum SubmissionFailure {
    /// Nothing reached the queue, so ordinary RAII cleanup is valid.
    Safe {
        phase: ProviderPhase,
        error: ExecutionFailure,
    },
    /// Queue acceptance or completion is unknown. Handles must stay
    /// alive until process exit or an out-of-band reaper proves completion.
    Pending {
        phase: ProviderPhase,
        error: ExecutionFailure,
    },
}

impl SubmissionFailure {
    fn from_queue_submit(error: ExecutionFailure) -> Self {
        // Vulkan guarantees an unsuccessful allocation leaves referenced
        // resources unaffected. Other failures do not prove queue rejection.
        if matches!(
            error.result,
            Some(vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
        ) {
            Self::Safe {
                phase: ProviderPhase::Submit,
                error,
            }
        } else {
            Self::Pending {
                phase: ProviderPhase::Submit,
                error,
            }
        }
    }

    fn is_pending(&self) -> bool {
        matches!(self, Self::Pending { .. })
    }

    fn is_device_lost(&self) -> bool {
        let error = match self {
            Self::Safe { error, .. } | Self::Pending { error, .. } => error,
        };
        error.result == Some(vk::Result::ERROR_DEVICE_LOST)
    }

    fn into_provider(self) -> ProviderError {
        match self {
            Self::Safe { phase, error } => error.into_provider(
                phase,
                ProviderErrorClass::Execute,
                match phase {
                    ProviderPhase::Encode => "vulkan-fence-create",
                    _ => "vulkan-queue-submit",
                },
                CompletionDisposition::NotSubmitted,
            ),
            Self::Pending { phase, error } => error.into_provider(
                phase,
                ProviderErrorClass::Execute,
                match phase {
                    ProviderPhase::Submit => "vulkan-queue-submit",
                    _ => "vulkan-wait",
                },
                CompletionDisposition::SubmittedUnknown { token: None },
            ),
        }
    }
}

impl PipelineObjects {
    fn new(context: Arc<VulkanContext>) -> Self {
        Self {
            context,
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            shader: vk::ShaderModule::null(),
            pipelines: BTreeMap::new(),
        }
    }

    fn create(
        &mut self,
        spv: &[u8],
        reflection: &ShaderReflection,
        plans: &[KernelDispatchPlan],
    ) -> Result<(), ExecutionFailure> {
        if !spv.len().is_multiple_of(4) {
            return Err(failure("translated SPIR-V is not word aligned").into());
        }
        let words = spv
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&words);
        self.shader = unsafe { self.context.device.create_shader_module(&shader_info, None) }
            .map_err(|error| {
                ExecutionFailure::vulkan(error, format!("create shader module: {error}"))
            })?;

        let mut layout_bindings = reflection
            .bindings
            .iter()
            .map(|binding| {
                let descriptor = binding.descriptor.expect("validated descriptor");
                vk::DescriptorSetLayoutBinding::default()
                    .binding(descriptor.binding)
                    .descriptor_type(descriptor_type_for_binding(binding))
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect::<Vec<_>>();
        layout_bindings.sort_by_key(|binding| binding.binding);
        let set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        self.set_layout = unsafe {
            self.context
                .device
                .create_descriptor_set_layout(&set_layout_info, None)
        }
        .map_err(|error| {
            ExecutionFailure::vulkan(error, format!("create descriptor-set layout: {error}"))
        })?;

        let set_layouts = [self.set_layout];
        let contract = reflection
            .kernel_dispatch
            .expect("validated kernel dispatch");
        let range = contract
            .push_constant_range()
            .expect("validated exact dispatch range");
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(range.offset)
            .size(range.size)];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        self.pipeline_layout = unsafe {
            self.context
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .map_err(|error| {
            ExecutionFailure::vulkan(error, format!("create pipeline layout: {error}"))
        })?;

        for region in plans.iter().flat_map(|plan| &plan.regions) {
            if self.pipelines.contains_key(&region.local_size) {
                continue;
            }
            let pipeline = self.create_compute_pipeline(region.local_size)?;
            self.pipelines.insert(region.local_size, pipeline);
        }
        Ok(())
    }

    fn create_compute_pipeline(
        &self,
        local_size: [u32; 3],
    ) -> Result<vk::Pipeline, ExecutionFailure> {
        let main = CString::new("main").expect("static entry name");
        let entries: [vk::SpecializationMapEntry; 3] =
            std::array::from_fn(|index| vk::SpecializationMapEntry {
                constant_id: KERNEL_LOCAL_SIZE_SPEC_IDS[index],
                offset: index as u32 * 4,
                size: 4,
            });
        let data = local_size
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        if std::env::var_os("METAL_API_DEBUG_DISPATCH").is_some() {
            eprintln!(
                "PIPELINE local={local_size:?} spec_ids={KERNEL_LOCAL_SIZE_SPEC_IDS:?} data={data:?}"
            );
        }
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&data);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.shader)
            .name(&main)
            .specialization_info(&specialization);
        let info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.pipeline_layout)];
        match unsafe {
            self.context
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &info, None)
        } {
            Ok(pipelines) => Ok(pipelines[0]),
            Err((partial, error)) => {
                for pipeline in partial {
                    unsafe { self.context.device.destroy_pipeline(pipeline, None) };
                }
                Err(ExecutionFailure::vulkan(
                    error,
                    format!("create compute pipeline: {error}"),
                ))
            }
        }
    }
}

impl Drop for PipelineObjects {
    fn drop(&mut self) {
        unsafe {
            for pipeline in self.pipelines.values().copied() {
                self.context.device.destroy_pipeline(pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.context
                    .device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.set_layout != vk::DescriptorSetLayout::null() {
                self.context
                    .device
                    .destroy_descriptor_set_layout(self.set_layout, None);
            }
            if self.shader != vk::ShaderModule::null() {
                self.context.device.destroy_shader_module(self.shader, None);
            }
        }
    }
}

impl ExecutionResources {
    fn new(context: Arc<VulkanContext>) -> Self {
        Self {
            context,
            pipeline_objects: Vec::new(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: Vec::new(),
            command_pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            queue_index: 0,
            submitted: false,
            completed: false,
            device_lost: false,
            device_loss_fault: None,
            leak_is_budgeted: false,
            buffers: Vec::new(),
            heap_memory: None,
            heap_bytes: None,
            textures: Vec::new(),
            static_samplers: Vec::new(),
            view_windows: BTreeMap::new(),
            indirect_buffer: vk::Buffer::null(),
            indirect_memory: vk::DeviceMemory::null(),
            borrowed: None,
        }
    }

    fn owned_bytes(&self) -> u64 {
        let heap_bytes = self.heap_bytes.unwrap_or(0);
        let buffers = self.buffers.iter().fold(0_u64, |total, buffer| {
            if buffer.host_pointer.is_some() || buffer.memory == vk::DeviceMemory::null() {
                return total;
            }
            total.saturating_add(u64::try_from(buffer.len).unwrap_or(u64::MAX))
        });
        heap_bytes.saturating_add(buffers)
    }

    fn set_borrowed_leases(
        &mut self,
        borrowed: Option<(Arc<BorrowedLeaseRegistry>, Vec<LeaseId>)>,
    ) {
        self.borrowed = borrowed;
    }

    fn mark_device_lost(&mut self) {
        self.device_lost = true;
    }

    /// Attach the fault record of this submission's device loss, if it
    /// observed one, and hand the provider error on unchanged otherwise.
    fn attach_device_loss_fault(&mut self, error: ProviderError) -> ProviderError {
        match self.device_loss_fault.take() {
            Some(fault) => with_device_fault_evidence(error, &fault),
            None => error,
        }
    }

    fn retain_after_budgeted_abandon(&mut self) {
        self.leak_is_budgeted = true;
    }

    fn create_pipeline_objects(
        &mut self,
        translated: &[&TranslatedComputePipeline],
        plans: &[KernelDispatchPlan],
    ) -> Result<(), ExecutionFailure> {
        for (translated, plan) in translated.iter().zip(plans) {
            let mut objects = PipelineObjects::new(Arc::clone(&self.context));
            objects.create(
                translated.spirv(),
                translated.reflection(),
                std::slice::from_ref(plan),
            )?;
            self.pipeline_objects.push(objects);
        }
        Ok(())
    }

    fn create_buffers(&mut self, bindings: &[PoolBinding]) -> Result<(), ExecutionFailure> {
        // Every shared backing must be sized before any of them is created:
        // the first view of an allocation may not be its largest end offset.
        let mut shared_sizes = BTreeMap::<u64, usize>::new();
        // Heap placements are one buffer per allocation inside a single slab:
        // `heap_sizes` is each allocation's device buffer size (its largest
        // view end), `heap_offsets` its binding offset inside the slab, and
        // `heap_slab_size` the slab's total byte size from the trace payload.
        let mut heap_sizes = BTreeMap::<u64, usize>::new();
        let mut heap_offsets = BTreeMap::<u64, usize>::new();
        let mut heap_slab_size: Option<usize> = None;
        for supplied in bindings {
            match supplied {
                PoolBinding::SharedOwned {
                    allocation,
                    offset,
                    length,
                    ..
                } => {
                    let end = offset.checked_add(*length).ok_or_else(|| {
                        failure(format!(
                            "shared buffer {allocation} view range overflows usize"
                        ))
                    })?;
                    shared_sizes
                        .entry(*allocation)
                        .and_modify(|size| *size = (*size).max(end))
                        .or_insert(end);
                }
                PoolBinding::HeapOwned {
                    allocation,
                    offset,
                    length,
                    allocation_size,
                    heap_offset,
                    heap_size,
                    ..
                } => {
                    let end = offset.checked_add(*length).ok_or_else(|| {
                        failure(format!(
                            "heap buffer {allocation} view range overflows usize"
                        ))
                    })?;
                    // The device buffer spans the allocation's full byte size
                    // (`allocation_size`), not the largest view end, so the
                    // buffer extent and the placement's `byte_size` agree
                    // (`research/docs/25` §6 Step 3). Every view window is
                    // already bounded by the allocation, but the check stays
                    // here to keep the invariant local.
                    if end > *allocation_size {
                        return Err(failure(format!(
                            "heap buffer {allocation} view ends at {end}, beyond its {allocation_size}-byte allocation"
                        ))
                        .into());
                    }
                    match heap_sizes.get(allocation) {
                        Some(existing) if *existing != *allocation_size => {
                            return Err(failure(format!(
                                "heap allocation {allocation} has conflicting sizes {existing} and {allocation_size}"
                            ))
                            .into());
                        }
                        _ => {
                            heap_sizes.insert(*allocation, *allocation_size);
                        }
                    }
                    match heap_offsets.get(allocation) {
                        Some(existing) if *existing != *heap_offset => {
                            return Err(failure(format!(
                                "heap allocation {allocation} has conflicting offsets {existing} and {heap_offset}"
                            ))
                            .into());
                        }
                        _ => {
                            heap_offsets.insert(*allocation, *heap_offset);
                        }
                    }
                    match heap_slab_size {
                        Some(existing) if existing != *heap_size => {
                            return Err(failure(format!(
                                "heap slab size disagrees across placements: {existing} and {heap_size}"
                            ))
                            .into());
                        }
                        _ => heap_slab_size = Some(*heap_size),
                    }
                }
                _ => {}
            }
        }
        if let Some(slab_size) = heap_slab_size {
            self.create_heap_buffers(&heap_sizes, &heap_offsets, slab_size)?;
        }
        for supplied in bindings {
            match supplied {
                PoolBinding::Owned(binding) => {
                    let index = binding.index;
                    let length = binding.bytes.len();
                    self.create_owned_buffer(binding)?;
                    self.register_view(PoolKey::buffer(index), u64::from(index), 0, length)?;
                }
                PoolBinding::SharedOwned {
                    index,
                    allocation,
                    offset,
                    length,
                    access,
                    bytes,
                } => {
                    // A shared backing is created once per allocation and
                    // covers the largest end offset of its views. Only the
                    // view's own bytes are ever copied in, at the view's own
                    // offset: a write-only view carries no snapshot and its
                    // upload region stays undefined, which the footprint
                    // proof allows because nothing reads outside a view's own
                    // accesses (`research/docs/15` step 4).
                    if bytes.len() != *length {
                        return Err(failure(format!(
                            "shared buffer {allocation} view has {} bytes, expected {length}",
                            bytes.len()
                        ))
                        .into());
                    }
                    if self
                        .buffers
                        .iter()
                        .all(|buffer| buffer.index != *allocation)
                    {
                        let size = *shared_sizes
                            .get(allocation)
                            .expect("shared view sizing pass covered every allocation");
                        self.create_owned_backing(*allocation, size, &[])?;
                    }
                    let gpu = self
                        .buffers
                        .iter_mut()
                        .find(|buffer| buffer.index == *allocation)
                        .expect("shared backing was just created");
                    let upload = if *access == metal_api_core::provider::BufferAccess::Write {
                        &bytes[..0]
                    } else {
                        bytes.as_slice()
                    };
                    if !upload.is_empty() {
                        if gpu.uploaded_ranges.contains_key(offset) {
                            return Err(failure(format!(
                                "shared buffer {allocation} already uploaded bytes at offset {offset}"
                            ))
                            .into());
                        }
                        let mapping = gpu
                            .host_pointer
                            .unwrap_or_else(|| gpu.mapping.expect("created backing is mapped"));
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                upload.as_ptr(),
                                (mapping as *mut u8).add(*offset),
                                upload.len(),
                            );
                        }
                        gpu.uploaded_ranges.insert(*offset, upload.len());
                        self.context.record_buffer_upload_bytes(upload.len());
                    }
                    self.register_view(PoolKey::buffer(*index), *allocation, *offset, *length)?;
                }
                PoolBinding::HeapOwned {
                    index,
                    allocation,
                    offset,
                    length,
                    access,
                    bytes,
                    ..
                } => {
                    // The slab and the allocation's buffer already exist;
                    // only the view's own bytes are uploaded at its window
                    // inside the buffer (`research/docs/25` §6 Step 3). A
                    // write-only view uploads nothing.
                    if bytes.len() != *length {
                        return Err(failure(format!(
                            "heap buffer {allocation} view has {} bytes, expected {length}",
                            bytes.len()
                        ))
                        .into());
                    }
                    let gpu = self
                        .buffers
                        .iter_mut()
                        .find(|buffer| buffer.index == *allocation)
                        .expect("heap buffer was just created");
                    let upload = if *access == metal_api_core::provider::BufferAccess::Write {
                        &bytes[..0]
                    } else {
                        bytes.as_slice()
                    };
                    if !upload.is_empty() {
                        if gpu.uploaded_ranges.contains_key(offset) {
                            return Err(failure(format!(
                                "heap buffer {allocation} already uploaded bytes at offset {offset}"
                            ))
                            .into());
                        }
                        let mapping = gpu
                            .host_pointer
                            .unwrap_or_else(|| gpu.mapping.expect("heap slab is mapped"));
                        let host_offset =
                            gpu.bind_offset.checked_add(*offset).ok_or_else(|| {
                                failure(format!(
                                    "heap buffer {allocation} host offset overflows usize"
                                ))
                            })?;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                upload.as_ptr(),
                                (mapping as *mut u8).add(host_offset),
                                upload.len(),
                            );
                        }
                        gpu.uploaded_ranges.insert(*offset, upload.len());
                        self.context.record_buffer_upload_bytes(upload.len());
                    }
                    self.register_view(PoolKey::buffer(*index), *allocation, *offset, *length)?;
                }
                PoolBinding::Imported {
                    index,
                    pointer,
                    len,
                    capacity,
                } => {
                    let length = *len;
                    self.import_host_buffer(*index, *pointer, *len, *capacity)?;
                    self.register_view(PoolKey::buffer(*index), u64::from(*index), 0, length)?;
                }
            }
        }
        self.buffers.sort_by_key(|buffer| buffer.index);
        Ok(())
    }

    /// Create one heap slab and one `VkBuffer` per heap placement, bound at
    /// the placement offset inside the slab (`research/docs/25-heaps与ICB设计.md`
    /// §6 Step 3). The slab is a single `VkDeviceMemory` allocation selected
    /// from the intersection of every buffer's `VkMemoryRequirements`, mapped
    /// once for upload/readback, and freed once at `Drop`.
    fn create_heap_buffers(
        &mut self,
        heap_sizes: &BTreeMap<u64, usize>,
        heap_offsets: &BTreeMap<u64, usize>,
        slab_size: usize,
    ) -> Result<(), ExecutionFailure> {
        struct PendingHeapBuffer {
            allocation: u64,
            buffer: vk::Buffer,
            requirements: vk::MemoryRequirements,
            offset: usize,
            size: usize,
        }

        // Buffers are created first so the slab memory type can be chosen from
        // the intersection of every buffer's requirements instead of assuming
        // one representative allocation is representative.
        let mut pending = Vec::<PendingHeapBuffer>::new();
        for (allocation, size) in heap_sizes {
            let offset = *heap_offsets
                .get(allocation)
                .expect("heap offset pass covered every allocation");
            let size_u64 =
                u64::try_from(*size).map_err(|_| failure("heap buffer size overflows u64"))?;
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size_u64)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = unsafe { self.context.device.create_buffer(&buffer_info, None) }.map_err(
                |error| {
                    ExecutionFailure::vulkan(
                        error,
                        format!("create heap buffer {allocation}: {error}"),
                    )
                },
            )?;
            let requirements =
                unsafe { self.context.device.get_buffer_memory_requirements(buffer) };
            let alignment = usize::try_from(requirements.alignment).unwrap_or(usize::MAX);
            if alignment == 0 || !offset.is_multiple_of(alignment) {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(failure(format!(
                    "heap buffer {allocation} offset {offset} is not a multiple of its {alignment}-byte alignment"
                ))
                .into());
            }
            pending.push(PendingHeapBuffer {
                allocation: *allocation,
                buffer,
                requirements,
                offset,
                size: *size,
            });
        }

        let mut type_bits = pending
            .first()
            .map_or(0, |head| head.requirements.memory_type_bits);
        for head in &pending[1..] {
            type_bits &= head.requirements.memory_type_bits;
        }
        let memory_type = match self.context.memory_type(
            type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(index) => index,
            Err(error) => {
                for head in &pending {
                    unsafe { self.context.device.destroy_buffer(head.buffer, None) };
                }
                return Err(error.into());
            }
        };
        let slab_size =
            u64::try_from(slab_size).map_err(|_| failure("heap slab size overflows u64"))?;
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(slab_size)
            .memory_type_index(memory_type);
        let memory = match unsafe { self.context.device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(error) => {
                for head in &pending {
                    unsafe { self.context.device.destroy_buffer(head.buffer, None) };
                }
                return Err(ExecutionFailure::vulkan(
                    error,
                    format!("allocate heap slab memory: {error}"),
                ));
            }
        };
        let mapped = match unsafe {
            self.context
                .device
                .map_memory(memory, 0, slab_size, vk::MemoryMapFlags::empty())
        } {
            Ok(mapped) => mapped,
            Err(error) => {
                for head in &pending {
                    unsafe { self.context.device.destroy_buffer(head.buffer, None) };
                }
                unsafe { self.context.device.free_memory(memory, None) };
                return Err(ExecutionFailure::vulkan(
                    error,
                    format!("map heap slab: {error}"),
                ));
            }
        };
        self.heap_memory = Some(memory);
        self.heap_bytes = Some(slab_size);
        let mut bound = Vec::<GpuBuffer>::with_capacity(pending.len());
        let mut bound_buffers = Vec::<vk::Buffer>::with_capacity(pending.len());
        for head in pending {
            if let Err(error) = unsafe {
                self.context
                    .device
                    .bind_buffer_memory(head.buffer, memory, head.offset as u64)
            } {
                // Vulkan requires every buffer bound to this slab to be
                // destroyed before the slab memory is freed. The earlier
                // iterations' buffers are not in `self.buffers` yet, so the
                // helper destroys them here, together with the buffer whose
                // bind just failed, before freeing the slab.
                for step in heap_slab_bind_failure_cleanup(head.buffer, &bound_buffers) {
                    match step {
                        HeapSlabCleanup::Destroy(buffer) => unsafe {
                            self.context.device.destroy_buffer(buffer, None)
                        },
                        HeapSlabCleanup::Free => unsafe {
                            self.context.device.free_memory(memory, None)
                        },
                    }
                }
                self.heap_memory = None;
                self.heap_bytes = None;
                return Err(ExecutionFailure::vulkan(
                    error,
                    format!(
                        "bind heap buffer {} at offset {}: {error}",
                        head.allocation, head.offset
                    ),
                ));
            }
            self.context.record_buffer_upload();
            bound_buffers.push(head.buffer);
            bound.push(GpuBuffer {
                index: head.allocation,
                buffer: head.buffer,
                memory: vk::DeviceMemory::null(),
                bind_offset: head.offset,
                len: head.size,
                host_pointer: None,
                mapping: Some(mapped as usize),
                uploaded_ranges: BTreeMap::new(),
            });
        }
        self.buffers.extend(bound);
        Ok(())
    }

    /// Upload the submission's textures. Two shapes are admitted: a D2,
    /// single-sample R32Uint/R32Float sampled texture with an owned byte source
    /// (host-visible linear image plus the sampler its binding executes with),
    /// and the writable storage image of `research/docs/26` §21.4 (C2), whose
    /// bytes travel through a transfer buffer because a `STORAGE_IMAGE` needs
    /// the optimal tiling. The provider creates the image, its view and, for a
    /// sampled binding, the sampler the translator expects; the descriptor
    /// write happens in `create_descriptors`.
    fn create_textures(
        &mut self,
        textures: &[metal_api_core::provider::TextureView],
        dispatches: &[BoundDispatch],
    ) -> Result<(), ExecutionFailure> {
        use metal_api_core::provider::{TextureAccess, TextureFormat, TextureSource, TextureType};
        for texture in textures {
            texture
                .validate_shape()
                .map_err(|error| failure(format!("texture {}: {error}", texture.metal_binding)))?;
            let byte_length = texture
                .expected_bytes()
                .map_err(|error| failure(format!("texture {}: {error}", texture.metal_binding)))?;
            if texture.access == TextureAccess::Storage {
                self.create_storage_texture(texture, byte_length, dispatches)?;
                continue;
            }
            let vk_format = match texture.format {
                TextureFormat::R32Uint => vk::Format::R32_UINT,
                TextureFormat::R32Float => vk::Format::R32_SFLOAT,
                other => {
                    return Err(failure(format!(
                        "texture {} needs a D2 single-sample R32Uint or R32Float sampled texture, not {other:?}",
                        texture.metal_binding
                    ))
                    .into())
                }
            };
            if texture.texture_type != TextureType::D2
                || texture.sample_count != 1
                || texture.depth != 1
                || texture.array_length != 1
                || texture.access != TextureAccess::Sampled
            {
                return Err(failure(format!(
                    "texture {} needs a D2 single-sample R32Uint or R32Float sampled texture",
                    texture.metal_binding
                ))
                .into());
            }
            let TextureSource::OwnedBytes(bytes) = &texture.source else {
                return Err(failure(format!(
                    "texture {} needs an owned byte source in the first increment",
                    texture.metal_binding
                ))
                .into());
            };
            let index = u64::from(texture.metal_binding);
            if self.textures.iter().any(|existing| existing.index == index) {
                return Err(failure(format!(
                    "texture {} occurs more than once",
                    texture.metal_binding
                ))
                .into());
            }
            let extent = vk::Extent3D {
                width: u32::try_from(texture.width)
                    .map_err(|_| failure("texture width overflows u32"))?,
                height: u32::try_from(texture.height)
                    .map_err(|_| failure("texture height overflows u32"))?,
                depth: 1,
            };
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk_format)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::LINEAR)
                .usage(vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::PREINITIALIZED);
            let (image, memory, requirements) = allocate_image_backing(
                &self.context,
                &image_info,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                "texture",
            )?;
            let mapped = match unsafe {
                self.context.device.map_memory(
                    memory,
                    0,
                    requirements.size,
                    vk::MemoryMapFlags::empty(),
                )
            } {
                Ok(mapped) => mapped,
                Err(error) => {
                    unsafe {
                        self.context.device.destroy_image(image, None);
                        self.context.device.free_memory(memory, None);
                    }
                    return Err(ExecutionFailure::vulkan(
                        error,
                        format!("map texture memory: {error}"),
                    ));
                }
            };
            // The owned bytes are tightly packed `width * 4` byte rows, but a
            // linear image's rows are only *at least* that far apart: the
            // driver chooses `VkSubresourceLayout.rowPitch`, and Lavapipe
            // returns 64 bytes for a 4x4 R32Uint image whose rows hold 16.
            // Writing row `r` at `r * width * 4` then lands every row after the
            // first in bytes the driver never reads, so `texture.read(x, y)`
            // reports 0 for every V != 0 texel while V == 0 still looks right.
            // Ask the driver for the layout instead of inferring the stride
            // from the extent.
            let tight_row_bytes = usize::try_from(texture.width)
                .ok()
                .and_then(|width| width.checked_mul(4))
                .ok_or_else(|| failure("texture row pitch overflows usize"))?;
            let (base_offset, row_pitch, depth_pitch) = if texture.height == 1 && texture.depth == 1
            {
                // A single row (down to a single texel) carries no row distance
                // to get wrong, so its tightly packed copy stays byte-for-byte
                // the previous behaviour and skips the layout query.
                (0, tight_row_bytes, 0)
            } else {
                let subresource = vk::ImageSubresource {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    array_layer: 0,
                };
                let layout = unsafe {
                    self.context
                        .device
                        .get_image_subresource_layout(image, subresource)
                };
                (
                    usize::try_from(layout.offset)
                        .map_err(|_| failure("texture row offset overflows usize"))?,
                    usize::try_from(layout.row_pitch)
                        .map_err(|_| failure("texture device row pitch overflows usize"))?,
                    // Only D2 depth-1 images are admitted today, so the depth
                    // pitch stays zero; consuming it anyway keeps a future
                    // depth > 1 shape from silently assuming tight slices.
                    usize::try_from(layout.depth_pitch)
                        .map_err(|_| failure("texture device depth pitch overflows usize"))?,
                )
            };
            let rows_per_slice = usize::try_from(texture.height)
                .map_err(|_| failure("texture height overflows usize"))?;
            for (row, chunk) in bytes.chunks(tight_row_bytes).enumerate() {
                let destination = base_offset
                    + (row / rows_per_slice) * depth_pitch
                    + (row % rows_per_slice) * row_pitch;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        chunk.as_ptr(),
                        (mapped.cast::<u8>()).add(destination),
                        chunk.len(),
                    );
                }
            }
            unsafe { self.context.device.unmap_memory(memory) };
            let view = match create_color_image_view(&self.context, image, vk_format, "texture") {
                Ok(view) => view,
                Err(error) => {
                    unsafe {
                        self.context.device.destroy_image(image, None);
                        self.context.device.free_memory(memory, None);
                    }
                    return Err(error);
                }
            };
            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            let sampler = match unsafe { self.context.device.create_sampler(&sampler_info, None) } {
                Ok(sampler) => sampler,
                Err(error) => {
                    unsafe {
                        self.context.device.destroy_image_view(view, None);
                        self.context.device.destroy_image(image, None);
                        self.context.device.free_memory(memory, None);
                    }
                    return Err(ExecutionFailure::vulkan(
                        error,
                        format!("create texture sampler: {error}"),
                    ));
                }
            };
            self.context.record_buffer_upload();
            self.context.record_buffer_upload_bytes(bytes.len());
            // The descriptor writer resolves a sampled texture through the
            // pass binding map, so the Metal argument index needs a pool key.
            let pool_key = dispatches
                .iter()
                .flat_map(|dispatch| dispatch.bindings.iter())
                .find(|binding| {
                    binding.metal_index == texture.metal_binding
                        && binding.key.kind == PoolKind::Texture
                })
                .map(|binding| binding.key)
                .ok_or_else(|| {
                    failure(format!(
                        "Metal texture {} is not bound in any dispatch",
                        texture.metal_binding
                    ))
                })?;
            self.register_view(
                pool_key,
                index,
                0,
                usize::try_from(byte_length).map_err(|_| {
                    failure(format!(
                        "texture {} length overflows usize",
                        texture.metal_binding
                    ))
                })?,
            )?;
            self.textures.push(GpuTexture {
                index,
                pool_key,
                image,
                memory,
                view,
                sampler,
                storage: None,
            });
        }
        Ok(())
    }

    /// Create one writable storage image and the transfer buffer its bytes
    /// travel through (`research/docs/26` §21.4, C2).
    ///
    /// A `STORAGE_IMAGE` is only guaranteed on optimal tiling, so the image
    /// cannot be the host-visible linear one a sampled texture is. The bytes
    /// therefore enter through `vkCmdCopyBufferToImage` before the first
    /// dispatch and leave through `vkCmdCopyImageToBuffer` after the last one,
    /// both recorded by [`Self::record`]; the transfer buffer stays mapped for
    /// the execution's lifetime, exactly as an owned buffer backing does.
    /// `byte_length` is the view's tightly packed extent, which is also the
    /// extent the writeback channel requires.
    fn create_storage_texture(
        &mut self,
        texture: &metal_api_core::provider::TextureView,
        byte_length: u64,
        dispatches: &[BoundDispatch],
    ) -> Result<(), ExecutionFailure> {
        use metal_api_core::provider::{TextureAccess, TextureFormat, TextureSource, TextureType};

        // One format: the R32f storage image `texture2d<float, write>` and
        // `texture2d<float, read_write>` lower to, and the one the contract
        // derivation admits (`map_storage_image_binding`). A second format
        // enters through a fixture that proves its landing, exactly as the
        // sampled face's list does.
        let vk_format = match texture.format {
            TextureFormat::R32Float => vk::Format::R32_SFLOAT,
            other => {
                return Err(failure(format!(
                    "storage image {} needs the reviewed D2 single-sample R32Float shape, not {other:?}",
                    texture.metal_binding
                ))
                .into())
            }
        };
        if texture.texture_type != TextureType::D2
            || texture.sample_count != 1
            || texture.depth != 1
            || texture.array_length != 1
            || texture.access != TextureAccess::Storage
        {
            return Err(failure(format!(
                "storage image {} needs the reviewed D2 single-sample R32Float shape",
                texture.metal_binding
            ))
            .into());
        }
        let TextureSource::OwnedBytes(bytes) = &texture.source else {
            return Err(failure(format!(
                "storage image {} needs an owned byte source in this increment",
                texture.metal_binding
            ))
            .into());
        };
        let index = u64::from(texture.metal_binding);
        if self.textures.iter().any(|existing| existing.index == index) {
            return Err(failure(format!(
                "texture {} occurs more than once",
                texture.metal_binding
            ))
            .into());
        }
        // The first device fact the storage face asks: this format has to be
        // usable as a storage image on the tiling the rail creates. A device
        // that answers otherwise is refused by name instead of being handed a
        // descriptor it cannot execute.
        let features = unsafe {
            self.context
                .instance
                .get_physical_device_format_properties(self.context.physical, vk_format)
        };
        if !features
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
        {
            return Err(failure(format!(
                "device reports no optimal-tiling storage image support for {:?}; refusing instead of binding a descriptor it cannot execute",
                texture.format
            ))
            .into());
        }
        let extent = vk::Extent3D {
            width: u32::try_from(texture.width)
                .map_err(|_| failure("storage image width overflows u32"))?,
            height: u32::try_from(texture.height)
                .map_err(|_| failure("storage image height overflows u32"))?,
            depth: 1,
        };
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let (image, memory, _requirements) = allocate_image_backing(
            &self.context,
            &image_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            "storage image",
        )?;
        let view = match create_color_image_view(&self.context, image, vk_format, "storage image") {
            Ok(view) => view,
            Err(error) => {
                unsafe {
                    self.context.device.destroy_image(image, None);
                    self.context.device.free_memory(memory, None);
                }
                return Err(error);
            }
        };
        // The transfer buffer: host-visible, tightly packed, and large enough
        // for the view's whole extent. It is created with the bytes already in
        // it, so the record path only has to copy device-to-device.
        let buffer_info = vk::BufferCreateInfo::default()
            .size(byte_length)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = match unsafe { self.context.device.create_buffer(&buffer_info, None) } {
            Ok(buffer) => buffer,
            Err(error) => {
                unsafe {
                    self.context.device.destroy_image_view(view, None);
                    self.context.device.destroy_image(image, None);
                    self.context.device.free_memory(memory, None);
                }
                return Err(ExecutionFailure::vulkan(
                    error,
                    format!("create storage image transfer buffer: {error}"),
                ));
            }
        };
        let requirements = unsafe { self.context.device.get_buffer_memory_requirements(buffer) };
        let memory_type = match self.context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(index) => index,
            Err(error) => {
                unsafe {
                    self.context.device.destroy_buffer(buffer, None);
                    self.context.device.destroy_image_view(view, None);
                    self.context.device.destroy_image(image, None);
                    self.context.device.free_memory(memory, None);
                }
                return Err(error.into());
            }
        };
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let buffer_memory = match unsafe { self.context.device.allocate_memory(&allocation, None) }
        {
            Ok(memory) => memory,
            Err(error) => {
                unsafe {
                    self.context.device.destroy_buffer(buffer, None);
                    self.context.device.destroy_image_view(view, None);
                    self.context.device.destroy_image(image, None);
                    self.context.device.free_memory(memory, None);
                }
                return Err(ExecutionFailure::vulkan(
                    error,
                    format!("allocate storage image transfer memory: {error}"),
                ));
            }
        };
        let cleanup = |context: &VulkanContext| unsafe {
            context.device.destroy_buffer(buffer, None);
            context.device.free_memory(buffer_memory, None);
            context.device.destroy_image_view(view, None);
            context.device.destroy_image(image, None);
            context.device.free_memory(memory, None);
        };
        if let Err(error) = unsafe {
            self.context
                .device
                .bind_buffer_memory(buffer, buffer_memory, 0)
        } {
            cleanup(&self.context);
            return Err(ExecutionFailure::vulkan(
                error,
                format!("bind storage image transfer memory: {error}"),
            ));
        }
        let mapped = match unsafe {
            self.context.device.map_memory(
                buffer_memory,
                0,
                requirements.size,
                vk::MemoryMapFlags::empty(),
            )
        } {
            Ok(mapped) => mapped,
            Err(error) => {
                cleanup(&self.context);
                return Err(ExecutionFailure::vulkan(
                    error,
                    format!("map storage image transfer memory: {error}"),
                ));
            }
        };
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
        }
        self.context.record_buffer_upload();
        self.context.record_buffer_upload_bytes(bytes.len());
        let pool_key = match dispatches
            .iter()
            .flat_map(|dispatch| dispatch.bindings.iter())
            .find(|binding| {
                binding.metal_index == texture.metal_binding
                    && binding.key.kind == PoolKind::Texture
            })
            .map(|binding| binding.key)
        {
            Some(pool_key) => pool_key,
            None => {
                cleanup(&self.context);
                return Err(failure(format!(
                    "Metal storage image {} is not bound in any dispatch",
                    texture.metal_binding
                ))
                .into());
            }
        };
        let length = usize::try_from(byte_length).map_err(|_| {
            failure(format!(
                "storage image {} length overflows usize",
                texture.metal_binding
            ))
        })?;
        if let Err(error) = self.register_view(pool_key, index, 0, length) {
            cleanup(&self.context);
            return Err(error);
        }
        self.textures.push(GpuTexture {
            index,
            pool_key,
            image,
            memory,
            view,
            sampler: vk::Sampler::null(),
            storage: Some(GpuStorageImage {
                buffer,
                memory: buffer_memory,
                mapping: mapped as usize,
                len: length,
                extent,
            }),
        });
        Ok(())
    }

    /// Record where one pool key lives inside a device buffer.
    fn register_view(
        &mut self,
        pool_key: PoolKey,
        buffer_key: u64,
        offset: usize,
        length: usize,
    ) -> Result<(), ExecutionFailure> {
        if self
            .view_windows
            .insert(
                pool_key,
                ViewWindow {
                    buffer_key,
                    offset,
                    length,
                },
            )
            .is_some()
        {
            return Err(failure(format!("pool key {pool_key:?} occurs more than once")).into());
        }
        Ok(())
    }

    /// The device buffer a window names.
    fn gpu_buffer(&self, key: u64) -> &GpuBuffer {
        self.buffers
            .iter()
            .find(|buffer| buffer.index == key)
            .expect("validated GPU buffer pool key")
    }

    /// The window of one pool key.
    fn view_window(&self, pool_key: PoolKey) -> &ViewWindow {
        self.view_windows
            .get(&pool_key)
            .expect("validated GPU buffer view window")
    }

    fn create_owned_buffer(&mut self, supplied: &BufferBinding) -> Result<(), ExecutionFailure> {
        self.create_owned_backing(
            u64::from(supplied.index),
            supplied.bytes.len(),
            &supplied.bytes,
        )
    }

    /// One host-visible device buffer, uploaded once from `bytes`. A shared view
    /// names it by its allocation identity instead of by its pool key.
    fn create_owned_backing(
        &mut self,
        index: u64,
        size: usize,
        upload: &[u8],
    ) -> Result<(), ExecutionFailure> {
        // The device buffer spans every byte any of the allocation's views can
        // address; the bytes actually copied in cover only the views that can
        // read (`research/docs/15` step 4).
        let size = u64::try_from(size)
            .map_err(|_| failure(format!("buffer {index} length overflows u64")))?;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer =
            unsafe { self.context.device.create_buffer(&buffer_info, None) }.map_err(|error| {
                ExecutionFailure::vulkan(error, format!("create buffer {}: {error}", index))
            })?;
        let requirements = unsafe { self.context.device.get_buffer_memory_requirements(buffer) };
        let memory_type = match self.context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(index) => index,
            Err(error) => {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(error.into());
            }
        };
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = match unsafe { self.context.device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(ExecutionFailure::vulkan(
                    error,
                    format!("allocate buffer {} memory: {error}", index),
                ));
            }
        };
        if let Err(error) = unsafe { self.context.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.context.device.destroy_buffer(buffer, None);
                self.context.device.free_memory(memory, None);
            }
            return Err(ExecutionFailure::vulkan(
                error,
                format!("bind buffer {} memory: {error}", index),
            ));
        }
        let mapped = match unsafe {
            self.context
                .device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        } {
            Ok(mapped) => mapped,
            Err(error) => {
                unsafe {
                    self.context.device.destroy_buffer(buffer, None);
                    self.context.device.free_memory(memory, None);
                }
                return Err(ExecutionFailure::vulkan(
                    error,
                    format!("map buffer {}: {error}", index),
                ));
            }
        };
        unsafe {
            if !upload.is_empty() {
                std::ptr::copy_nonoverlapping(upload.as_ptr(), mapped.cast::<u8>(), upload.len());
            }
        }
        self.context.record_buffer_upload();
        self.context.record_buffer_upload_bytes(upload.len());
        self.buffers.push(GpuBuffer {
            index,
            buffer,
            memory,
            bind_offset: 0,
            len: usize::try_from(size).unwrap_or(usize::MAX),
            host_pointer: None,
            mapping: Some(mapped as usize),
            uploaded_ranges: BTreeMap::new(),
        });
        Ok(())
    }

    /// Import the owner's host mapping for `pointer` without copying. The
    /// buffer covers `len` bytes; the imported allocation may be larger, and
    /// `capacity` is the number of valid bytes the owner reserved after
    /// `pointer`.
    fn import_host_buffer(
        &mut self,
        index: u32,
        pointer: usize,
        len: usize,
        capacity: usize,
    ) -> Result<(), ExecutionFailure> {
        let Some(host) = self.context.external_memory_host.as_ref() else {
            return Err(failure(format!(
                "buffer {index} needs VK_EXT_external_memory_host, which is unavailable"
            ))
            .into());
        };
        let size = u64::try_from(len)
            .map_err(|_| failure(format!("buffer {index} length overflows u64")))?;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer =
            unsafe { self.context.device.create_buffer(&buffer_info, None) }.map_err(|error| {
                ExecutionFailure::vulkan(error, format!("create buffer {index}: {error}"))
            })?;
        let requirements = unsafe { self.context.device.get_buffer_memory_requirements(buffer) };
        let capacity = u64::try_from(capacity).unwrap_or(u64::MAX);
        if requirements.size > capacity {
            unsafe { self.context.device.destroy_buffer(buffer, None) };
            return Err(failure(format!(
                "buffer {index} needs {} imported bytes but the lease reserves {capacity}",
                requirements.size
            ))
            .into());
        }
        let mut properties = vk::MemoryHostPointerPropertiesEXT::default();
        let result = unsafe {
            (host.device.fp().get_memory_host_pointer_properties_ext)(
                host.device.device(),
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                pointer as *const std::ffi::c_void,
                &mut properties,
            )
        };
        if result != vk::Result::SUCCESS {
            unsafe { self.context.device.destroy_buffer(buffer, None) };
            return Err(ExecutionFailure::vulkan(
                result,
                format!("query buffer {index} host pointer: {result}"),
            ));
        }
        let memory_type = match self.context.memory_type(
            requirements.memory_type_bits & properties.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(index) => index,
            Err(error) => {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(error.into());
            }
        };
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
            .host_pointer(pointer as *mut std::ffi::c_void);
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut import);
        let memory = match unsafe { self.context.device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(ExecutionFailure::vulkan(
                    error,
                    format!("import buffer {index} host memory: {error}"),
                ));
            }
        };
        if let Err(error) = unsafe { self.context.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.context.device.destroy_buffer(buffer, None);
                self.context.device.free_memory(memory, None);
            }
            return Err(ExecutionFailure::vulkan(
                error,
                format!("bind buffer {index} imported memory: {error}"),
            ));
        }
        self.buffers.push(GpuBuffer {
            index: u64::from(index),
            buffer,
            memory,
            bind_offset: 0,
            len,
            host_pointer: Some(pointer),
            mapping: None,
            uploaded_ranges: BTreeMap::new(),
        });
        Ok(())
    }

    /// Create one `VkSampler` per translated module's AIR-embedded constexpr
    /// sampler (`research/docs/26` §21.3, C1b).
    ///
    /// The create-info comes from the module's own decoded state
    /// ([`static_sampler_policy`]), never from a provider default: a rail that
    /// substituted nearest/clamp here would change which texels
    /// `air.sample_texture_*` returns without changing the request. A runtime
    /// `[[sampler(n)]]` binding carries no AIR state and no request surface
    /// this contract defines, so it is refused by name. A linear state also
    /// requires the device to report linear filtering for the rail's sampled
    /// float format; a device that cannot filter it is refused by name rather
    /// than silently sampled nearest.
    fn create_static_samplers(
        &mut self,
        translated: &[&TranslatedComputePipeline],
    ) -> Result<(), ExecutionFailure> {
        for pipeline in translated {
            for binding in &pipeline.reflection().bindings {
                if !matches!(
                    binding.kind,
                    ResourceKind::StaticSampler | ResourceKind::Sampler
                ) {
                    continue;
                }
                let descriptor = binding
                    .descriptor
                    .ok_or_else(|| failure("sampler binding has no descriptor location"))?;
                if binding.kind == ResourceKind::Sampler {
                    return Err(failure(format!(
                        "runtime [[sampler({})]] is not part of the reviewed compute contract; \
                         refusing instead of binding an unstated state",
                        binding.metal_index
                    ))
                    .into());
                }
                let state = binding.static_sampler.ok_or_else(|| {
                    failure(format!(
                        "AIR static sampler at descriptor {} carries no decoded state",
                        descriptor.binding
                    ))
                })?;
                let policy = static_sampler_policy(&state)?;
                // The device question an address mode asks before the sampler
                // is created (`research/docs/23` §109): the family's
                // `mirrorClampToEdge` is the one mode whose Vulkan feature is
                // not core-and-enabled-by-default, so a device that never
                // reported it is refused by name instead of being handed a
                // mode it was never told about.
                if let Some(feature) = sampler_address_mode_feature(policy.address) {
                    if !self.context.sampler_mirror_clamp_to_edge() {
                        return Err(failure(format!(
                            "the device did not report the {feature} feature, which the {:?} \
                             address mode needs; refusing the mode by name instead of creating \
                             a sampler the device was never told about",
                            policy.address
                        ))
                        .into());
                    }
                }
                if policy.filter.is_linear() {
                    let features = unsafe {
                        self.context.instance.get_physical_device_format_properties(
                            self.context.physical,
                            vk::Format::R32_SFLOAT,
                        )
                    };
                    // The reviewed textures are linear-tiled images
                    // (`create_textures`), so the device fact this checks is
                    // the linear-tiling feature bit.
                    let filter_features = features.linear_tiling_features;
                    if !filter_features
                        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
                    {
                        return Err(failure(format!(
                            "device reports no linear filtering for the sampled float format \
                             (filter={:?} address={:?}); refusing instead of sampling nearest",
                            policy.filter, policy.address
                        ))
                        .into());
                    }
                }
                let info = sampler_create_info(policy);
                let sampler = unsafe { self.context.device.create_sampler(&info, None) }.map_err(
                    |error| {
                        ExecutionFailure::vulkan(error, format!("create static sampler: {error}"))
                    },
                )?;
                self.static_samplers.push(GpuStaticSampler {
                    binding: descriptor.binding,
                    sampler,
                });
            }
        }
        Ok(())
    }

    fn create_descriptors(
        &mut self,
        translated: &[&TranslatedComputePipeline],
        dispatches: &[BoundDispatch],
    ) -> Result<(), ExecutionFailure> {
        let pass_count = u32::try_from(dispatches.len())
            .map_err(|_| failure("descriptor set count overflows u32"))?;
        let mut storage_buffer_count = 0_u32;
        let mut sampled_image_count = 0_u32;
        let mut storage_image_count = 0_u32;
        let mut sampler_count = 0_u32;
        for pipeline in translated {
            for binding in &pipeline.reflection().bindings {
                let counter = match descriptor_type_for_binding(binding) {
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER => &mut sampled_image_count,
                    vk::DescriptorType::STORAGE_IMAGE => &mut storage_image_count,
                    vk::DescriptorType::SAMPLER => &mut sampler_count,
                    _ => &mut storage_buffer_count,
                };
                *counter = counter
                    .checked_add(1)
                    .ok_or_else(|| failure("descriptor pool count overflows u32"))?;
            }
        }
        let mut sizes = Vec::with_capacity(4);
        if storage_buffer_count > 0 {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: storage_buffer_count,
            });
        }
        if sampled_image_count > 0 {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: sampled_image_count,
            });
        }
        if storage_image_count > 0 {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: storage_image_count,
            });
        }
        if sampler_count > 0 {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: sampler_count,
            });
        }
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(pass_count)
            .pool_sizes(&sizes);
        self.descriptor_pool =
            unsafe { self.context.device.create_descriptor_pool(&pool_info, None) }.map_err(
                |error| ExecutionFailure::vulkan(error, format!("create descriptor pool: {error}")),
            )?;
        let layouts = self
            .pipeline_objects
            .iter()
            .map(|objects| objects.set_layout)
            .collect::<Vec<_>>();
        let allocation = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        self.descriptor_sets = unsafe { self.context.device.allocate_descriptor_sets(&allocation) }
            .map_err(|error| {
                ExecutionFailure::vulkan(error, format!("allocate descriptor sets: {error}"))
            })?;

        // Each recorded pass owns a distinct immutable set. Updating a single
        // shared set here would make every dispatch observe the last mapping.
        for ((dispatch, &set), translated) in
            dispatches.iter().zip(&self.descriptor_sets).zip(translated)
        {
            let reflection = translated.reflection();
            let mut writes = Vec::with_capacity(reflection.bindings.len());
            let mut buffer_infos = Vec::with_capacity(reflection.bindings.len());
            let mut image_infos = Vec::with_capacity(reflection.bindings.len());
            let mut storage_infos = Vec::with_capacity(reflection.bindings.len());
            let mut sampler_infos = Vec::with_capacity(reflection.bindings.len());
            for binding in &reflection.bindings {
                if matches!(
                    binding.kind,
                    ResourceKind::StaticSampler | ResourceKind::Sampler
                ) {
                    let descriptor = binding.descriptor.expect("validated descriptor");
                    let created = self
                        .static_samplers
                        .iter()
                        .find(|sampler| sampler.binding == descriptor.binding)
                        .ok_or_else(|| {
                            failure(format!(
                                "no sampler was created for descriptor {}",
                                descriptor.binding
                            ))
                        })?;
                    sampler_infos.push(vk::DescriptorImageInfo::default().sampler(created.sampler));
                    writes.push(vk::WriteDescriptorSet::default());
                    continue;
                }
                if binding.kind == ResourceKind::Texture {
                    let pool_key = dispatch
                        .bindings
                        .iter()
                        .find(|candidate| {
                            candidate.metal_index == binding.metal_index
                                && candidate.key.kind == PoolKind::Texture
                        })
                        .map(|candidate| candidate.key)
                        .ok_or_else(|| {
                            failure(format!(
                                "pass binds no texture at Metal index {}; bindings={:?}",
                                binding.metal_index, dispatch.bindings
                            ))
                        })?;
                    let texture = self
                        .textures
                        .iter()
                        .find(|texture| texture.pool_key == pool_key)
                        .ok_or_else(|| {
                            failure(format!(
                                "Metal texture {} was not uploaded for this submission",
                                binding.metal_index
                            ))
                        })?;
                    let info = vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_view(texture.view)
                        .sampler(texture.sampler);
                    image_infos.push(info);
                    writes.push(vk::WriteDescriptorSet::default());
                    continue;
                }
                if binding.kind == ResourceKind::StorageImage {
                    // A storage image is written by the module itself, so the
                    // descriptor carries the image view in `GENERAL` and no
                    // sampler (`research/docs/26` §21.4, C2). The layout is the
                    // one the record path transitions to before the first
                    // dispatch binds it.
                    let pool_key = dispatch
                        .bindings
                        .iter()
                        .find(|candidate| {
                            candidate.metal_index == binding.metal_index
                                && candidate.key.kind == PoolKind::Texture
                        })
                        .map(|candidate| candidate.key)
                        .ok_or_else(|| {
                            failure(format!(
                                "pass binds no storage image at Metal index {}; bindings={:?}",
                                binding.metal_index, dispatch.bindings
                            ))
                        })?;
                    let texture = self
                        .textures
                        .iter()
                        .find(|texture| texture.pool_key == pool_key)
                        .ok_or_else(|| {
                            failure(format!(
                                "Metal storage image {} was not uploaded for this submission",
                                binding.metal_index
                            ))
                        })?;
                    let info = vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_view(texture.view);
                    storage_infos.push(info);
                    writes.push(vk::WriteDescriptorSet::default());
                    continue;
                }
                let pool_key = dispatch
                    .bindings
                    .iter()
                    .find(|candidate| {
                        candidate.metal_index == binding.metal_index
                            && candidate.key.kind == PoolKind::Buffer
                    })
                    .expect("validated pass binding")
                    .key;
                let window = self.view_window(pool_key);
                let gpu = self.gpu_buffer(window.buffer_key);
                let info = vk::DescriptorBufferInfo::default()
                    .buffer(gpu.buffer)
                    .offset(window.offset as u64)
                    .range(window.length as u64);
                buffer_infos.push(info);
                writes.push(vk::WriteDescriptorSet::default());
            }
            // Attach the collected infos once all pushes are done: the
            // descriptor writes borrow the vectors, so the slices stay valid
            // until `update_descriptor_sets` returns.
            let mut image_cursor = 0;
            let mut storage_cursor = 0;
            let mut buffer_cursor = 0;
            let mut sampler_cursor = 0;
            let writes = writes
                .into_iter()
                .zip(&reflection.bindings)
                .map(|(write, binding)| {
                    let descriptor = binding.descriptor.expect("validated descriptor");
                    let write = write.dst_set(set).dst_binding(descriptor.binding);
                    if matches!(
                        binding.kind,
                        ResourceKind::StaticSampler | ResourceKind::Sampler
                    ) {
                        let write = write
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(std::slice::from_ref(&sampler_infos[sampler_cursor]));
                        sampler_cursor += 1;
                        write
                    } else if binding.kind == ResourceKind::Texture {
                        let write = write
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(std::slice::from_ref(&image_infos[image_cursor]));
                        image_cursor += 1;
                        write
                    } else if binding.kind == ResourceKind::StorageImage {
                        let write = write
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(std::slice::from_ref(&storage_infos[storage_cursor]));
                        storage_cursor += 1;
                        write
                    } else {
                        let write = write
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(&buffer_infos[buffer_cursor]));
                        buffer_cursor += 1;
                        write
                    }
                })
                .collect::<Vec<_>>();
            unsafe { self.context.device.update_descriptor_sets(&writes, &[]) };
        }
        Ok(())
    }

    /// Encode one `VkDispatchIndirectCommand` into a host-visible
    /// `INDIRECT_BUFFER` the compute rail replays with `vkCmdDispatchIndirect`
    /// (`research/docs/25` §6 Step 4). The command is written by the CPU once
    /// and unmapped, mirroring `render::create_indirect_draw`: this is the ICB
    /// *equivalent*, not a `VK_EXT_device_generated_commands` device command.
    fn create_indirect_dispatch(&mut self, threadgroups: [u32; 3]) -> Result<(), ExecutionFailure> {
        let command = vk::DispatchIndirectCommand {
            x: threadgroups[0],
            y: threadgroups[1],
            z: threadgroups[2],
        };
        let byte_length = std::mem::size_of::<vk::DispatchIndirectCommand>() as u64;
        let info = vk::BufferCreateInfo::default()
            .size(byte_length)
            .usage(vk::BufferUsageFlags::INDIRECT_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer =
            unsafe { self.context.device.create_buffer(&info, None) }.map_err(|error| {
                ExecutionFailure::vulkan(error, format!("create indirect buffer: {error}"))
            })?;
        let requirements = unsafe { self.context.device.get_buffer_memory_requirements(buffer) };
        let memory_type = match self.context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(index) => index,
            Err(error) => {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(error.into());
            }
        };
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = match unsafe { self.context.device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(ExecutionFailure::vulkan(
                    error,
                    format!("allocate indirect memory: {error}"),
                ));
            }
        };
        if let Err(error) = unsafe { self.context.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.context.device.destroy_buffer(buffer, None);
                self.context.device.free_memory(memory, None);
            }
            return Err(ExecutionFailure::vulkan(
                error,
                format!("bind indirect memory: {error}"),
            ));
        }
        let mapping = match unsafe {
            self.context.device.map_memory(
                memory,
                0,
                requirements.size,
                vk::MemoryMapFlags::empty(),
            )
        } {
            Ok(mapping) => mapping,
            Err(error) => {
                unsafe {
                    self.context.device.destroy_buffer(buffer, None);
                    self.context.device.free_memory(memory, None);
                }
                return Err(ExecutionFailure::vulkan(
                    error,
                    format!("map indirect memory: {error}"),
                ));
            }
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &command as *const vk::DispatchIndirectCommand as *const u8,
                mapping.cast::<u8>(),
                byte_length as usize,
            );
            self.context.device.unmap_memory(memory);
        }
        self.indirect_buffer = buffer;
        self.indirect_memory = memory;
        Ok(())
    }

    fn record(
        &mut self,
        translated: &[&TranslatedComputePipeline],
        plans: &[KernelDispatchPlan],
        queue_index: usize,
    ) -> Result<(), ExecutionFailure> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.context.queue_families[queue_index])
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        self.command_pool = unsafe { self.context.device.create_command_pool(&pool_info, None) }
            .map_err(|error| {
                ExecutionFailure::vulkan(error, format!("create command pool: {error}"))
            })?;
        let allocation = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        self.command = unsafe { self.context.device.allocate_command_buffers(&allocation) }
            .map_err(|error| {
                ExecutionFailure::vulkan(error, format!("allocate command buffer: {error}"))
            })?[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.context
                .device
                .begin_command_buffer(self.command, &begin)
                .map_err(|error| {
                    ExecutionFailure::vulkan(error, format!("begin command buffer: {error}"))
                })?;
            // Host-visible linear images are uploaded in PREINITIALIZED and the
            // sampled descriptor binds them in GENERAL, so the first transition
            // needs only the new layout, not an access scope. A storage image
            // is not in that state: it is created UNDEFINED and its own upload
            // chain below transitions it to GENERAL (`research/docs/26` §21.4,
            // C2), so it is skipped here.
            for texture in &self.textures {
                if texture.storage.is_some() {
                    continue;
                }
                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::PREINITIALIZED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(texture.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                self.context.device.cmd_pipeline_barrier(
                    self.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
            // Each storage image lands its initial contents before the first
            // dispatch: UNDEFINED -> TRANSFER_DST_OPTIMAL, copy the whole
            // tightly packed view from its transfer buffer, then
            // TRANSFER_DST_OPTIMAL -> GENERAL, which is the layout the storage
            // descriptor states and the only one a shader may read or write
            // through (`research/docs/26` §21.4, C2). The copy covers whole
            // texels of a single-mip, single-layer 2D image, so the buffer's
            // rows are the image's rows with no padding to state.
            for texture in &self.textures {
                let Some(storage) = &texture.storage else {
                    continue;
                };
                let subresource = vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let acquire = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(texture.image)
                    .subresource_range(range);
                self.context.device.cmd_pipeline_barrier(
                    self.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[acquire],
                );
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(subresource)
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(storage.extent);
                self.context.device.cmd_copy_buffer_to_image(
                    self.command,
                    storage.buffer,
                    texture.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
                let release = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(texture.image)
                    .subresource_range(range);
                self.context.device.cmd_pipeline_barrier(
                    self.command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[release],
                );
            }
            for (pass_index, plan) in plans.iter().enumerate() {
                let objects = &self.pipeline_objects[pass_index];
                let reflection = translated[pass_index].reflection();
                let offset = reflection
                    .kernel_dispatch
                    .expect("validated kernel dispatch")
                    .push_constant_range()
                    .expect("validated exact range")
                    .offset;
                self.context.device.cmd_bind_descriptor_sets(
                    self.command,
                    vk::PipelineBindPoint::COMPUTE,
                    objects.pipeline_layout,
                    reflection.descriptor_layout.set,
                    &[self.descriptor_sets[pass_index]],
                    &[],
                );
                if pass_index != 0 {
                    // Order all earlier compute accesses and make their writes
                    // visible to the next pass's reads and writes (RAW/WAW).
                    // The execution dependency also covers WAR hazards.
                    // Khronos legacy compute-to-compute synchronization:
                    // https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples-(Legacy-synchronization-APIs)
                    let barriers = [vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )];
                    self.context.device.cmd_pipeline_barrier(
                        self.command,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &barriers,
                        &[],
                        &[],
                    );
                }
                for region in &plan.regions {
                    let pipeline = objects.pipelines[&region.local_size];
                    self.context.device.cmd_bind_pipeline(
                        self.command,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline,
                    );
                    let words = plan.push_constants(*region);
                    let bytes = words
                        .into_iter()
                        .flat_map(u32::to_ne_bytes)
                        .collect::<Vec<_>>();
                    self.context.device.cmd_push_constants(
                        self.command,
                        objects.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        offset,
                        &bytes,
                    );
                    if std::env::var_os("METAL_API_DEBUG_DISPATCH").is_some() {
                        eprintln!(
                            "DISPATCH region local={:?} groups={:?} threads_base={:?}",
                            region.local_size, region.group_count, region.thread_base
                        );
                    }
                    if self.indirect_buffer == vk::Buffer::null() {
                        self.context.device.cmd_dispatch(
                            self.command,
                            region.group_count[0],
                            region.group_count[1],
                            region.group_count[2],
                        );
                    } else {
                        // The indirect replay reads the workgroup count the CPU
                        // encoded above from the host-visible `INDIRECT_BUFFER`.
                        self.context.device.cmd_dispatch_indirect(
                            self.command,
                            self.indirect_buffer,
                            0,
                        );
                    }
                }
            }
            // Every storage image copies its landed texels back after the last
            // dispatch: GENERAL -> TRANSFER_SRC_OPTIMAL, one
            // `vkCmdCopyImageToBuffer` of the whole tightly packed view, then a
            // TRANSFER -> HOST barrier so the mapped transfer buffer's bytes
            // are the ones the copy wrote (`research/docs/26` §21.4, C2). The
            // transfer buffer is the same one the initial contents rode in on,
            // so a landing is one buffer per image and no second allocation.
            for texture in &self.textures {
                let Some(storage) = &texture.storage else {
                    continue;
                };
                let subresource = vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let release = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(texture.image)
                    .subresource_range(range);
                self.context.device.cmd_pipeline_barrier(
                    self.command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[release],
                );
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(subresource)
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(storage.extent);
                self.context.device.cmd_copy_image_to_buffer(
                    self.command,
                    texture.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    storage.buffer,
                    &[region],
                );
                let available = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(storage.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                self.context.device.cmd_pipeline_barrier(
                    self.command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[available],
                    &[],
                );
            }
            // Fence retirement establishes execution completion; this barrier
            // makes compute writes available to coherent host readback.
            let readback_barriers = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            self.context.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &readback_barriers,
                &[],
                &[],
            );
            self.context
                .device
                .end_command_buffer(self.command)
                .map_err(|error| {
                    ExecutionFailure::vulkan(error, format!("end command buffer: {error}"))
                })?;
        }
        Ok(())
    }

    fn submit(&mut self, queue_index: usize) -> Result<(), SubmissionFailure> {
        self.queue_index = queue_index;
        self.context.notify_enqueue(queue_index);
        self.fence = unsafe {
            self.context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|error| SubmissionFailure::Safe {
            phase: ProviderPhase::Encode,
            error: ExecutionFailure::vulkan(error, format!("create completion fence: {error}")),
        })?;
        let commands = [self.command];
        let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
        if let Err(result) = self
            .context
            .submit_commands(queue_index, &submits, self.fence)
        {
            let failure = SubmissionFailure::from_queue_submit(ExecutionFailure::vulkan(
                result,
                format!("submit compute command buffer: {result}"),
            ));
            if failure.is_device_lost() {
                // A driver-reported loss is a terminal device event, not an
                // execution failure: the queue never confirmed the submission,
                // and the device is gone, so these handles are destroyed
                // instead of retained (`resource_drop_policy`).
                self.mark_device_lost();
                self.device_loss_fault = Some(self.context.observe_device_loss());
            } else {
                // The queue did not confirm the submission, so nothing about
                // this context may be reused: the outcome of the fence is
                // unknown.
                self.context.mark_unobservable_submission();
            }
            self.submitted = failure.is_pending();
            return Err(failure);
        }
        self.context.record_queue_submission(queue_index);
        self.submitted = true;
        Ok(())
    }

    fn wait(&mut self, timeout_ns: u64) -> Result<bool, SubmissionFailure> {
        // The one fence wait this submission performs, classified by how long
        // the driver call itself took: a wait that returns immediately means the
        // queue had already retired the work, and those microseconds are driver
        // overhead rather than device latency.
        let mut _fence_wait =
            crate::phase_profile::Bar::enter_fence_wait(crate::phase_profile::Phase::FenceWait);
        let wait = self.context.wait_for_fence(self.fence, timeout_ns);
        match wait {
            Ok(()) => {
                if !self.completed {
                    self.completed = true;
                    self.context.record_queue_retirement(self.queue_index);
                }
                Ok(true)
            }
            Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => {
                if let Some(bar) = _fence_wait.as_mut() {
                    bar.mark_timed_out();
                }
                Ok(false)
            }
            Err(result) if result == vk::Result::ERROR_DEVICE_LOST => {
                // The loss is observed while waiting, so the submission
                // reached the queue and its handles are still unknowns: the
                // resources stay submitted and are destroyed by the device
                // loss, and the context stops admitting work through the core
                // lifecycle.
                self.mark_device_lost();
                self.device_loss_fault = Some(self.context.observe_device_loss());
                Err(SubmissionFailure::Pending {
                    phase: ProviderPhase::Wait,
                    error: ExecutionFailure::vulkan(
                        result,
                        format!("wait for compute completion failed: {result}"),
                    ),
                })
            }
            Err(result) => {
                self.context.mark_unobservable_submission();
                Err(SubmissionFailure::Pending {
                    phase: ProviderPhase::Wait,
                    error: ExecutionFailure::vulkan(
                        result,
                        format!("wait for compute completion failed: {result}"),
                    ),
                })
            }
        }
    }

    fn read_updates(
        &self,
        writable_pool_keys: &BTreeSet<u32>,
    ) -> Result<Vec<LandingUpdate>, ExecutionFailure> {
        let mut updates = Vec::new();
        // One readback operation per distinct device buffer, then one slice per
        // writable view: a shared backing is read once even when several of its
        // views are writable.
        let mut read_buffers = BTreeSet::<u64>::new();
        for &pool_key in writable_pool_keys {
            let window = self.view_window(PoolKey::buffer(pool_key));
            let gpu = self.gpu_buffer(window.buffer_key);
            // Owned backings were mapped once at creation and are unmapped only
            // when the execution resources are destroyed; imported backings are
            // the owner's own mapping.
            let mapping = gpu
                .host_pointer
                .or(gpu.mapping)
                .ok_or_else(|| failure(format!("buffer {} is not mapped", gpu.index)))?;
            if read_buffers.insert(window.buffer_key) {
                self.context.record_buffer_readback();
            }
            let end = window.offset + window.length;
            if end > gpu.len {
                return Err(failure(format!(
                    "buffer pool key {pool_key} window ends at {end}, beyond its backing"
                ))
                .into());
            }
            let host_offset = gpu.bind_offset.checked_add(window.offset).ok_or_else(|| {
                failure(format!(
                    "buffer pool key {pool_key} host offset overflows usize"
                ))
            })?;
            let bytes = unsafe {
                std::slice::from_raw_parts((mapping as *const u8).add(host_offset), window.length)
            };
            let bytes = bytes.to_vec();
            if !bytes.is_empty() {
                self.context.record_buffer_readback_bytes(bytes.len());
            }
            updates.push(LandingUpdate {
                target: LandingTarget::Buffer(pool_key),
                offset: 0,
                bytes,
            });
        }
        // One readback per storage image, taken from the transfer buffer the
        // record path copied the landed texels into (`research/docs/26` §21.4,
        // C2). The bytes are the view's whole tightly packed extent, which is
        // exactly the landing the writeback channel requires, and the mapping
        // was made at creation and stays valid until this execution's
        // resources are destroyed.
        for texture in &self.textures {
            let Some(storage) = &texture.storage else {
                continue;
            };
            self.context.record_buffer_readback();
            let bytes = unsafe {
                std::slice::from_raw_parts(storage.mapping as *const u8, storage.len).to_vec()
            };
            if !bytes.is_empty() {
                self.context.record_buffer_readback_bytes(bytes.len());
            }
            updates.push(LandingUpdate {
                target: LandingTarget::Texture(
                    u32::try_from(texture.index)
                        .map_err(|_| failure("storage image index overflows u32"))?,
                ),
                offset: 0,
                bytes,
            });
        }
        updates.sort_by_key(|update| match update.target {
            LandingTarget::Buffer(key) => (0_u8, key),
            LandingTarget::Texture(index) => (1_u8, index),
        });
        Ok(updates)
    }
}

impl Drop for ExecutionResources {
    fn drop(&mut self) {
        // A retained in-flight submission may still read or write borrowed
        // owner memory, so only a destroying drop retires its retains.
        if resource_drop_policy(self.submitted, self.completed, self.device_lost)
            == ResourceDropPolicy::Retain
        {
            if !self.leak_is_budgeted {
                // Retaining an unretirable submission is exactly what the
                // bounded abandonment budget accounts for: recording it ends
                // the context in the same transition that counts it.
                self.context.record_abandonment(self.owned_bytes());
            }
            // A panic between queue submission and the explicit wait outcome
            // cannot unwind into destruction of in-flight handles. Raw Vulkan
            // handles below are intentionally left live, and this strong
            // context reference keeps the loader/device live until process exit.
            let _ = Arc::into_raw(Arc::clone(&self.context));
            // Rust still drops fields after this Drop returns. Explicitly retain
            // child RAII owners as well as the raw execution handles.
            for objects in self.pipeline_objects.drain(..) {
                std::mem::forget(objects);
            }
            return;
        }
        if self.submitted && !self.completed {
            self.context.record_queue_retirement(self.queue_index);
        }
        if let Some((registry, lease_ids)) = self.borrowed.take() {
            registry.retire_all(&lease_ids);
        }
        unsafe {
            if self.fence != vk::Fence::null() {
                self.context.device.destroy_fence(self.fence, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.context
                    .device
                    .destroy_command_pool(self.command_pool, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.context
                    .device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            self.pipeline_objects.clear();
            for buffer in &self.buffers {
                self.context.device.destroy_buffer(buffer.buffer, None);
                if buffer.memory != vk::DeviceMemory::null() {
                    self.context.device.free_memory(buffer.memory, None);
                }
            }
            if let Some(memory) = self.heap_memory {
                self.context.device.free_memory(memory, None);
            }
            for texture in &self.textures {
                if texture.sampler != vk::Sampler::null() {
                    self.context.device.destroy_sampler(texture.sampler, None);
                }
                self.context.device.destroy_image_view(texture.view, None);
                self.context.device.destroy_image(texture.image, None);
                self.context.device.free_memory(texture.memory, None);
                if let Some(storage) = &texture.storage {
                    self.context.device.unmap_memory(storage.memory);
                    self.context.device.destroy_buffer(storage.buffer, None);
                    self.context.device.free_memory(storage.memory, None);
                }
            }
            for sampler in &self.static_samplers {
                self.context.device.destroy_sampler(sampler.sampler, None);
            }
            // The indirect buffer is unbound by construction (its memory is
            // freed right after), so destroy before free, matching the render
            // rail's `create_indirect_draw` teardown.
            if self.indirect_buffer != vk::Buffer::null() {
                self.context
                    .device
                    .destroy_buffer(self.indirect_buffer, None);
            }
            if self.indirect_memory != vk::DeviceMemory::null() {
                self.context.device.free_memory(self.indirect_memory, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use metal_api_core::provider::{FieldValue, Retryability};

    #[test]
    fn heap_bind_failure_frees_memory_after_every_buffer_is_destroyed() {
        let failing = vk::Buffer::from_raw(3);
        let bound = [vk::Buffer::from_raw(1), vk::Buffer::from_raw(2)];
        let steps = heap_slab_bind_failure_cleanup(failing, &bound);
        // The slab memory may only be freed after every buffer bound to it has
        // been destroyed; `Free` must therefore be the final step.
        assert_eq!(
            steps,
            vec![
                HeapSlabCleanup::Destroy(vk::Buffer::from_raw(3)),
                HeapSlabCleanup::Destroy(vk::Buffer::from_raw(1)),
                HeapSlabCleanup::Destroy(vk::Buffer::from_raw(2)),
                HeapSlabCleanup::Free,
            ]
        );
        assert!(matches!(steps.last(), Some(HeapSlabCleanup::Free)));
    }

    /// Test-only helper: turn (metal_index, pool index) pairs into bindings,
    /// taking each binding's width from the pool it names.
    fn test_bindings(pairs: &[(u32, u32)], pool: &[BufferBinding]) -> Vec<Binding> {
        pairs
            .iter()
            .map(|&(metal_index, index)| Binding {
                metal_index,
                key: PoolKey::buffer(index),
                width: pool
                    .iter()
                    .find(|buffer| buffer.index == index)
                    .map(|buffer| buffer.bytes.len())
                    .unwrap_or(0),
            })
            .collect()
    }

    fn test_pool_width(pool: &[BufferBinding], index: u32) -> usize {
        pool.iter()
            .find(|buffer| buffer.index == index)
            .map(|buffer| buffer.bytes.len())
            .unwrap_or(0)
    }

    use metal2vulkan::meta::{KernMeta, KernRole};
    use metal2vulkan::reflect::{BufferStrideTerm, BufferStridedAccess};

    #[test]
    fn texture_fixture_translates_and_reflects_a_sampled_binding() {
        let source =
            include_str!("../../../examples/metal-smoke/shaders/kernel_read_texture_2d.ll");
        let options = TransformOptions {
            kernel_local_size: [1, 1, 1],
            kernel_dispatch: Some(KernelDispatch::safe_default()),
            ..TransformOptions::default()
        };
        let scratch = ScratchDir::new().expect("scratch directory");
        let (spirv, reflection) = metal2vulkan::translate_sanitized_native_reflected(
            source,
            Stage::Kernel,
            &scratch.path,
            options,
        )
        .expect("the texture fixture translates");
        assert!(!spirv.is_empty());
        let texture = reflection
            .bindings
            .iter()
            .find(|binding| binding.texture_shape.is_some())
            .expect("the fixture reflects a texture binding");
        assert_eq!(texture.access, Some(ResourceAccess::Sampled));
    }

    #[test]
    fn texture_fixture_creates_a_compute_pipeline_on_the_selected_device() {
        let executor = match VulkanExecutor::new() {
            Ok(executor) => executor,
            Err(error) => {
                eprintln!("SKIP: no Vulkan device: {error}");
                return;
            }
        };
        let device = metal_api_core::Device::new(executor);
        let library = device
            .new_library_with_air(include_str!(
                "../../../examples/metal-smoke/shaders/kernel_read_texture_2d.ll"
            ))
            .expect("the fixture library loads");
        let function = library
            .function("read_texture_2d")
            .expect("the fixture entry exists");
        // The pipeline creates once reflection admits a sampled texture: the
        // descriptor-set layout carries a combined image sampler for it. The
        // execution path still has to create the image, sampler and descriptor
        // write (`research/docs/16` §4.3).
        device
            .new_compute_pipeline_state(&function)
            .expect("a texture-reading pipeline creates its descriptor layout");
    }

    #[test]
    fn object_api_binds_and_executes_a_sampled_texture() {
        use metal_api_core::provider::TextureFormat;
        let executor = match VulkanExecutor::new() {
            Ok(executor) => executor,
            Err(error) => {
                eprintln!("SKIP: no Vulkan device: {error}");
                return;
            }
        };
        let provider =
            crate::VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider");
        let device = metal_api_core::provider_api::Device::new(Arc::new(provider));
        let pipeline = device
            .compile_pipeline(metal_api_core::provider::PipelineCompileRequest {
                entry_name: "read_texture_2d".to_owned(),
                logical_digest: metal_api_core::provider::SemanticDigest::new(
                    "metal-smoke-fixture-v1",
                    b"object_sampled_texture".to_vec(),
                )
                .expect("digest"),
                source: metal_api_core::provider::ShaderSource::SanitizedLl(
                    include_str!("../../../examples/metal-smoke/shaders/kernel_read_texture_2d.ll")
                        .to_owned(),
                ),
            })
            .expect("pipeline");
        let mut texels = Vec::with_capacity(64);
        for value in 0..16_u32 {
            texels.extend_from_slice(&value.to_le_bytes());
        }
        let texture = device
            .new_texture_with_bytes(TextureFormat::R32Uint, 4, 4, texels)
            .expect("texture object");
        let output = device
            .new_buffer_with_bytes(vec![0_u8; 64])
            .expect("output buffer");
        let queue = device.new_command_queue();
        let command = queue.command_buffer();
        {
            let mut encoder = command.compute_command_encoder().expect("encoder");
            encoder
                .set_compute_pipeline_state(&pipeline)
                .expect("pipeline state");
            encoder.set_texture(0, &texture).expect("texture binding");
            encoder
                .set_buffer(0, &output.view(0, 64).unwrap())
                .expect("buffer binding");
            encoder
                .dispatch_threads(
                    metal_api_core::Size::new(1, 1, 1).unwrap(),
                    metal_api_core::Size::new(1, 1, 1).unwrap(),
                )
                .expect("dispatch");
            encoder.end_encoding().expect("end encoding");
        }
        command.commit().expect("commit");
        command.wait_until_completed().expect("completion");
        let observed = output.read().expect("readback");
        assert_eq!(observed[..4], 0_u32.to_le_bytes());
    }

    /// The object API now owns a render command encoder, so one command buffer
    /// can run the declaring compute pass and then render (and present) into
    /// the attachment buffer the compute pass declared. This is the object-rail
    /// sibling of the trace rail's render+present round trip.
    #[test]
    fn object_api_executes_render_and_present_on_the_selected_device() {
        use metal_api_core::provider::{
            AttachmentFormat, PipelineCompileRequest, RenderPipelineContract, SemanticDigest,
            ShaderSource, VertexLayout,
        };
        use metal_api_core::provider_api::{self as objects, RenderAttachmentLoad};

        let executor = match VulkanExecutor::new() {
            Ok(executor) => executor,
            Err(error) => {
                eprintln!("SKIP: no Vulkan device: {error}");
                return;
            }
        };
        let provider = Arc::new(
            crate::VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider"),
        );
        let device = objects::Device::new(
            Arc::clone(&provider) as Arc<dyn metal_api_core::provider::PipelineProvider>
        );

        let copy = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "copy_word".to_owned(),
                logical_digest: SemanticDigest::new(
                    "metal-smoke-fixture-v1",
                    b"object_render_declaring".to_vec(),
                )
                .expect("digest"),
                source: ShaderSource::SanitizedLl(
                    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll")
                        .to_owned(),
                ),
            })
            .expect("compute pipeline");
        let render_metadata = provider
            .register_render_pipeline(RenderPipelineRequest {
                contract: RenderPipelineContract {
                    stage_buffers: Vec::new(),
                    vertex_entry: "vertex_main".to_owned(),
                    fragment_entry: crate::render::SOLID_FRAGMENT_ENTRY.to_owned(),
                    color_formats: vec![AttachmentFormat::Rgba8Unorm],
                    vertex_layout: VertexLayout::None,
                    textures: Vec::new(),
                },
                vertex_spirv: include_bytes!("render_spv/fullscreen_triangle.vert.spv").to_vec(),
                fragment_spirv: crate::render::solid_fragment_spirv(&[
                    AttachmentFormat::Rgba8Unorm,
                ])
                .expect("reviewed fragment stage")
                .to_vec(),
                logical_digest: SemanticDigest::new(
                    "metal-smoke-fixture-v1",
                    b"object_render_pipeline".to_vec(),
                )
                .expect("digest"),
            })
            .expect("render pipeline registration");
        let render = device
            .render_pipeline(&render_metadata)
            .expect("render handle");

        let attachment = device
            .new_buffer_with_bytes(vec![0xfe; 16])
            .expect("attachment buffer");
        let attachment_view = attachment.view(0, 16).expect("attachment view");
        let output = device
            .new_buffer_with_bytes(vec![0xff; 4])
            .expect("output buffer");
        let output_view = output.view(0, 4).expect("output view");

        let queue = device.new_command_queue();
        let command = queue.command_buffer();
        {
            let mut encoder = command.compute_command_encoder().expect("compute encoder");
            encoder
                .set_compute_pipeline_state(&copy)
                .expect("compute pipeline");
            encoder
                .set_buffer(0, &attachment_view)
                .expect("attachment binding");
            encoder.set_buffer(1, &output_view).expect("output binding");
            encoder
                .dispatch_threads(
                    metal_api_core::Size::new(1, 1, 1).unwrap(),
                    metal_api_core::Size::new(1, 1, 1).unwrap(),
                )
                .expect("dispatch");
            encoder.end_encoding().expect("end compute");
        }
        {
            let mut encoder = command.render_command_encoder().expect("render encoder");
            encoder
                .set_render_pipeline_state(&render)
                .expect("render pipeline");
            encoder
                .draw_render_pass(
                    &attachment_view,
                    AttachmentFormat::Rgba8Unorm,
                    2,
                    2,
                    RenderAttachmentLoad::Clear([0xfe; 4]),
                    Some(objects::PresentInitial::Sentinel([0xef; 4])),
                )
                .expect("render pass");
            encoder.end_encoding().expect("end render");
        }
        command.commit().expect("commit");
        command.wait_until_completed().expect("completion");
        assert_eq!(
            attachment.read().expect("attachment readback"),
            [0x40, 0x80, 0xc0, 0xff].repeat(4),
            "the rendered attachment reads back the solid unorm8 colour"
        );
        assert_eq!(
            provider.present_counts(),
            (1, 1),
            "one present target is acquired and presented once"
        );
    }

    /// In deferred mode the present tail executes during `submit`, so a
    /// cancelled present-bearing command abandons the writeback landing but
    /// keeps the submit-time present action (counters and target layout).
    #[test]
    fn cancelling_an_async_present_command_keeps_the_submit_time_present() {
        use metal_api_core::provider::{
            AttachmentFormat, CompletionDisposition, PipelineCompileRequest,
            RenderPipelineContract, SemanticDigest, ShaderSource, VertexLayout,
        };
        use metal_api_core::provider_api::{self as objects, RenderAttachmentLoad};
        use metal_api_core::CommandBufferStatus;

        let executor = match VulkanExecutor::new() {
            Ok(executor) => executor,
            Err(error) => {
                eprintln!("SKIP: no Vulkan device: {error}");
                return;
            }
        };
        let provider = Arc::new(
            crate::VulkanComputeProvider::with_executor(Arc::clone(&executor))
                .expect("provider")
                .with_async_execution(true),
        );
        let device = objects::Device::new(provider.clone());

        let copy = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "copy_word".to_owned(),
                logical_digest: SemanticDigest::new(
                    "metal-smoke-fixture-v1",
                    b"object_render_cancel_declaring".to_vec(),
                )
                .expect("digest"),
                source: ShaderSource::SanitizedLl(
                    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll")
                        .to_owned(),
                ),
            })
            .expect("compute pipeline");
        let render_metadata = provider
            .register_render_pipeline(RenderPipelineRequest {
                contract: RenderPipelineContract {
                    stage_buffers: Vec::new(),
                    vertex_entry: "vertex_main".to_owned(),
                    fragment_entry: crate::render::SOLID_FRAGMENT_ENTRY.to_owned(),
                    color_formats: vec![AttachmentFormat::Rgba8Unorm],
                    vertex_layout: VertexLayout::None,
                    textures: Vec::new(),
                },
                vertex_spirv: include_bytes!("render_spv/fullscreen_triangle.vert.spv").to_vec(),
                fragment_spirv: crate::render::solid_fragment_spirv(&[
                    AttachmentFormat::Rgba8Unorm,
                ])
                .expect("reviewed fragment stage")
                .to_vec(),
                logical_digest: SemanticDigest::new(
                    "metal-smoke-fixture-v1",
                    b"object_render_cancel_pipeline".to_vec(),
                )
                .expect("digest"),
            })
            .expect("render pipeline registration");
        let render = device
            .render_pipeline(&render_metadata)
            .expect("render handle");

        let attachment = device
            .new_buffer_with_bytes(vec![0xfe; 16])
            .expect("attachment buffer");
        let attachment_view = attachment.view(0, 16).expect("attachment view");
        let output = device
            .new_buffer_with_bytes(vec![0xff; 4])
            .expect("output buffer");
        let output_view = output.view(0, 4).expect("output view");

        let queue = device.new_command_queue();
        let command = queue.command_buffer();
        {
            let mut encoder = command.compute_command_encoder().expect("compute encoder");
            encoder
                .set_compute_pipeline_state(&copy)
                .expect("compute pipeline");
            encoder
                .set_buffer(0, &attachment_view)
                .expect("attachment binding");
            encoder.set_buffer(1, &output_view).expect("output binding");
            encoder
                .dispatch_threads(
                    metal_api_core::Size::new(1, 1, 1).unwrap(),
                    metal_api_core::Size::new(1, 1, 1).unwrap(),
                )
                .expect("dispatch");
            encoder.end_encoding().expect("end compute");
        }
        {
            let mut encoder = command.render_command_encoder().expect("render encoder");
            encoder
                .set_render_pipeline_state(&render)
                .expect("render pipeline");
            encoder
                .draw_render_pass(
                    &attachment_view,
                    AttachmentFormat::Rgba8Unorm,
                    2,
                    2,
                    RenderAttachmentLoad::Clear([0xfe; 4]),
                    Some(objects::PresentInitial::Sentinel([0xef; 4])),
                )
                .expect("render pass");
            encoder.end_encoding().expect("end render");
        }
        command.commit().expect("commit");
        assert_eq!(
            command.status().unwrap(),
            CommandBufferStatus::Committed,
            "the deferred command stays Committed until its results land"
        );
        // The present action ran during `submit`, before any observation.
        assert_eq!(
            provider.present_counts(),
            (1, 1),
            "the present tail advances its counters at submit time"
        );
        command.cancel().expect("cancel");
        assert_eq!(
            command.status().unwrap(),
            CommandBufferStatus::Failed,
            "cancel turns the command Failed"
        );
        assert!(matches!(
            command.wait_until_completed(),
            Err(objects::Error::CompletionUnavailable(
                CompletionDisposition::Cancelled { .. }
            ))
        ));
        // Cancel does not roll the present action back, and neither the render
        // writeback nor the compute writeback landed.
        assert_eq!(
            provider.present_counts(),
            (1, 1),
            "cancel abandons observation without rolling the present back"
        );
        assert_eq!(
            attachment.read().expect("attachment readback"),
            vec![0xfe; 16],
            "the render writeback is not landed by a cancelled command"
        );
        assert_eq!(
            output.read().expect("output readback"),
            vec![0xff; 4],
            "the compute writeback is not landed by a cancelled command"
        );
    }

    #[test]
    fn texture_fixture_executes_a_texel_read_on_the_selected_device() {
        use metal_api_core::provider::{
            AllocationId, TextureAccess, TextureFormat, TextureSource, TextureType, TextureView,
            ViewId,
        };
        let executor = match VulkanExecutor::new() {
            Ok(executor) => executor,
            Err(error) => {
                eprintln!("SKIP: no Vulkan device: {error}");
                return;
            }
        };
        let device = metal_api_core::Device::new(
            Arc::clone(&executor) as Arc<dyn metal_api_core::ComputeExecutor>
        );
        let library = device
            .new_library_with_air(include_str!(
                "../../../examples/metal-smoke/shaders/kernel_read_texture_2d.ll"
            ))
            .expect("the fixture library loads");
        let function = library
            .function("read_texture_2d")
            .expect("the fixture entry exists");
        let pipeline = executor
            .new_compute_pipeline(&function)
            .expect("pipeline creates");
        // 4x4 R32Uint texels 0..15; thread 0 reads texel (0, 0).
        let mut texels = Vec::with_capacity(64);
        for value in 0..16_u32 {
            texels.extend_from_slice(&value.to_le_bytes());
        }
        let texture = TextureView {
            view_id: ViewId::new(900),
            metal_binding: 0,
            allocation_id: AllocationId::new(901),
            texture_type: TextureType::D2,
            format: TextureFormat::R32Uint,
            width: 4,
            height: 4,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            access: TextureAccess::Sampled,
            source: TextureSource::OwnedBytes(texels),
        };
        let submission = metal_api_core::ComputeSubmission {
            pipeline,
            buffers: vec![metal_api_core::BufferBinding {
                index: 0,
                bytes: vec![0_u8; 64],
            }],
            textures: vec![texture],
            threads_per_grid: metal_api_core::Size::new(1, 1, 1).unwrap(),
            threads_per_threadgroup: metal_api_core::Size::new(1, 1, 1).unwrap(),
        };
        let updates = executor.execute(submission).expect("texture read executes");
        assert_eq!(updates.len(), 1);
        // The fixture declares a 64-byte output buffer; a 1x1 dispatch writes
        // only its first word.
        assert_eq!(updates[0].bytes.len(), 64);
        assert_eq!(updates[0].bytes[..4], 0_u32.to_le_bytes());
    }

    fn serial_fixture() -> (
        TranslatedComputePipeline,
        Vec<BufferBinding>,
        vk::PhysicalDeviceLimits,
    ) {
        let meta = KernMeta {
            roles: vec![(0, KernRole::Buffer(0))],
            max_work_group_size: Some(32),
            ..KernMeta::default()
        };
        let mut reflection = ShaderReflection::from_kernel(&meta, Some("serial"), [1, 1, 1]);
        reflection.kernel_dispatch = Some(KernelDispatch::safe_default());
        reflection.bindings[0].footprint = Some(BufferFootprint {
            static_ranges: Vec::new(),
            strided_accesses: vec![BufferStridedAccess {
                base_offset: 0,
                access_size: 4,
                terms: vec![
                    BufferStrideTerm {
                        source: BufferIndexSource::GlobalInvocationIdX,
                        stride: 4,
                    },
                    BufferStrideTerm {
                        source: BufferIndexSource::GlobalInvocationIdY,
                        stride: 40,
                    },
                    BufferStrideTerm {
                        source: BufferIndexSource::GlobalInvocationIdZ,
                        stride: 120,
                    },
                ],
            }],
            has_unbounded_access: false,
        });
        let limits = vk::PhysicalDeviceLimits {
            max_compute_work_group_size: [128, 128, 64],
            max_compute_work_group_invocations: 128,
            max_compute_work_group_count: [65535; 3],
            max_push_constants_size: 128,
            max_bound_descriptor_sets: 4,
            max_per_stage_descriptor_storage_buffers: 8,
            max_descriptor_set_storage_buffers: 8,
            max_per_stage_resources: 8,
            max_storage_buffer_range: 4096,
            ..vk::PhysicalDeviceLimits::default()
        };
        (
            TranslatedComputePipeline {
                spv: Vec::new(),
                reflection,
            },
            vec![BufferBinding {
                index: 0,
                bytes: vec![0; 240],
            }],
            limits,
        )
    }

    fn rebound_fixture() -> (
        TranslatedComputePipeline,
        Vec<BufferBinding>,
        vk::PhysicalDeviceLimits,
    ) {
        let (mut translated, mut buffers, limits) = serial_fixture();
        translated.reflection.bindings[0].access = Some(ResourceAccess::ReadOnly);
        let mut output = translated.reflection.bindings[0].clone();
        output.metal_index = 1;
        output.descriptor.as_mut().unwrap().binding = 1;
        output.access = Some(ResourceAccess::WriteOnly);
        translated.reflection.bindings.push(output);
        buffers.push(BufferBinding {
            index: 1,
            bytes: vec![0; 240],
        });
        (translated, buffers, limits)
    }

    fn ping_pong_dispatches(pool: &[BufferBinding]) -> Vec<BoundDispatch> {
        vec![
            BoundDispatch {
                grid: [10, 3, 2],
                local: [8, 2, 1],
                bindings: vec![
                    Binding {
                        metal_index: 0,
                        key: PoolKey::buffer(0),
                        width: test_pool_width(pool, 0),
                    },
                    Binding {
                        metal_index: 1,
                        key: PoolKey::buffer(1),
                        width: test_pool_width(pool, 1),
                    },
                ],
            },
            BoundDispatch {
                grid: [7, 2, 1],
                local: [4, 1, 1],
                bindings: vec![
                    Binding {
                        metal_index: 1,
                        key: PoolKey::buffer(0),
                        width: test_pool_width(pool, 0),
                    },
                    Binding {
                        metal_index: 0,
                        key: PoolKey::buffer(1),
                        width: test_pool_width(pool, 1),
                    },
                ],
            },
        ]
    }

    fn alternate_pipeline_fixture() -> TranslatedComputePipeline {
        let meta = KernMeta {
            roles: vec![(0, KernRole::Buffer(3)), (1, KernRole::Buffer(7))],
            max_work_group_size: Some(16),
            ..KernMeta::default()
        };
        let mut reflection = ShaderReflection::from_kernel(&meta, Some("alternate"), [1; 3]);
        reflection.kernel_dispatch = Some(KernelDispatch::ThreadsDynamic { offset: 16 });
        let (first, _, _) = rebound_fixture();
        for (index, binding) in reflection.bindings.iter_mut().enumerate() {
            binding.footprint = first.reflection.bindings[index].footprint.clone();
            // Deliberately reverse access roles while retaining the same pool
            // mapping, so readback must consult each pipeline's reflection.
            binding.access = Some(if index == 0 {
                ResourceAccess::WriteOnly
            } else {
                ResourceAccess::ReadOnly
            });
        }
        validate_pipeline_reflection("alternate", &reflection).unwrap();
        TranslatedComputePipeline {
            spv: Vec::new(),
            reflection,
        }
    }

    fn mixed_pipeline_dispatches(pool: &[BufferBinding]) -> Vec<BoundDispatch> {
        vec![
            BoundDispatch {
                grid: [10, 3, 2],
                local: [8, 2, 1],
                bindings: vec![
                    Binding {
                        metal_index: 0,
                        key: PoolKey::buffer(0),
                        width: test_pool_width(pool, 0),
                    },
                    Binding {
                        metal_index: 1,
                        key: PoolKey::buffer(1),
                        width: test_pool_width(pool, 1),
                    },
                ],
            },
            BoundDispatch {
                grid: [7, 2, 1],
                local: [4, 1, 1],
                bindings: vec![
                    Binding {
                        metal_index: 3,
                        key: PoolKey::buffer(0),
                        width: test_pool_width(pool, 0),
                    },
                    Binding {
                        metal_index: 7,
                        key: PoolKey::buffer(1),
                        width: test_pool_width(pool, 1),
                    },
                ],
            },
        ]
    }

    #[test]
    fn mixed_preflight_uses_each_shader_binding_layout_and_write_access() {
        let (first, buffers, limits) = rebound_fixture();
        let second = alternate_pipeline_fixture();
        assert_ne!(
            first.reflection.bindings[0].descriptor,
            second.reflection.bindings[0].descriptor
        );
        let dispatches = mixed_pipeline_dispatches(&buffers);
        let planned =
            plan_pipeline_sequence(&[&first, &second], &buffers, &limits, &dispatches).unwrap();
        assert_eq!(planned.writable_pool_keys, BTreeSet::from([0, 1]));
        assert_eq!(planned.plans.len(), 2);
        for (plan, dispatch) in planned.plans.iter().zip(dispatches) {
            assert_eq!(plan.push_constants(plan.regions[0])[..3], dispatch.grid);
        }
    }

    fn subset_pipeline_fixture() -> (
        TranslatedComputePipeline,
        TranslatedComputePipeline,
        Vec<BufferBinding>,
        vk::PhysicalDeviceLimits,
        Vec<BoundDispatch>,
    ) {
        let (mut first, mut buffers, limits) = rebound_fixture();
        let mut scalar = first.reflection.bindings[0].clone();
        scalar.metal_index = 9;
        scalar.descriptor.as_mut().unwrap().binding = 9;
        scalar.footprint.as_mut().unwrap().strided_accesses[0]
            .terms
            .clear();
        first.reflection.bindings.push(scalar);
        validate_pipeline_reflection("serial", &first.reflection).unwrap();
        buffers[0].index = 11;
        buffers[1].index = 19;
        buffers.push(BufferBinding {
            index: 23,
            bytes: vec![0; 4],
        });
        buffers.push(BufferBinding {
            index: 29,
            bytes: vec![0; 240],
        });
        let dispatches = vec![
            BoundDispatch {
                grid: [10, 3, 2],
                local: [8, 2, 1],
                bindings: vec![
                    Binding {
                        metal_index: 0,
                        key: PoolKey::buffer(11),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 11)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                    Binding {
                        metal_index: 1,
                        key: PoolKey::buffer(19),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 19)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                    Binding {
                        metal_index: 9,
                        key: PoolKey::buffer(23),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 23)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                ],
            },
            BoundDispatch {
                grid: [7, 2, 1],
                local: [4, 1, 1],
                bindings: vec![
                    Binding {
                        metal_index: 3,
                        key: PoolKey::buffer(19),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 19)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                    Binding {
                        metal_index: 7,
                        key: PoolKey::buffer(29),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 29)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                ],
            },
            BoundDispatch {
                grid: [10, 3, 2],
                local: [8, 2, 1],
                bindings: vec![
                    Binding {
                        metal_index: 0,
                        key: PoolKey::buffer(19),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 19)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                    Binding {
                        metal_index: 1,
                        key: PoolKey::buffer(29),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 29)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                    Binding {
                        metal_index: 9,
                        key: PoolKey::buffer(23),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 23)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                ],
            },
        ];
        (
            first,
            alternate_pipeline_fixture(),
            buffers,
            limits,
            dispatches,
        )
    }

    #[test]
    fn subset_preflight_allows_three_two_three_slots_and_later_pool_resources() {
        let (first, second, buffers, limits, dispatches) = subset_pipeline_fixture();
        let planned =
            plan_pipeline_sequence(&[&first, &second, &first], &buffers, &limits, &dispatches)
                .unwrap();
        // Resource 29 appears only in later passes and takes slot 1 from 19.
        // The scalar pool resource stays read-only and is absent in pass two.
        assert_eq!(planned.writable_pool_keys, BTreeSet::from([19, 29]));
        assert_eq!(planned.plans.len(), 3);
        for (plan, dispatch) in planned.plans.iter().zip(dispatches) {
            assert_eq!(plan.push_constants(plan.regions[0])[..3], dispatch.grid);
        }
    }

    #[test]
    fn subset_preflight_rejects_unused_pool_and_invalid_per_pass_maps() {
        let (first, second, mut buffers, limits, dispatches) = subset_pipeline_fixture();
        buffers.push(BufferBinding {
            index: 31,
            bytes: vec![0; 240],
        });
        let error =
            plan_pipeline_sequence(&[&first, &second, &first], &buffers, &limits, &dispatches)
                .unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Args);
        assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
        assert!(error.detail.unwrap().contains("at least one pass"));
        buffers.pop();

        for (bindings, detail) in [
            (
                test_bindings(&[(3, 19), (7, 31)], &buffers),
                "unknown buffer pool key 31",
            ),
            (
                test_bindings(&[(3, 19), (7, 19)], &buffers),
                "bound more than once in one pass",
            ),
            (
                test_bindings(&[(3, 19)], &buffers),
                "do not match reflection",
            ),
            (
                test_bindings(&[(3, 19), (3, 29)], &buffers),
                "buffer 3 is bound more than once",
            ),
            (
                test_bindings(&[(3, 19), (7, 29), (9, 11)], &buffers),
                "buffer 9 is not reflected",
            ),
        ] {
            let mut invalid_dispatches = dispatches.clone();
            invalid_dispatches[1].bindings = bindings;
            let error = plan_pipeline_sequence(
                &[&first, &second, &first],
                &buffers,
                &limits,
                &invalid_dispatches,
            )
            .unwrap_err();
            assert_eq!(error.class, ProviderErrorClass::Args);
            assert_eq!(error.phase, ProviderPhase::Resolve);
            assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
            assert!(error.detail.unwrap().contains(detail));
        }
    }

    #[test]
    fn subset_preflight_checks_scalar_rebound_to_later_array_footprint() {
        let (first, second, buffers, limits, mut dispatches) = subset_pipeline_fixture();
        // The first pipeline reads pool 23 as a scalar, but the next pipeline
        // reads its input as an array. The later pass must reject that mapping.
        plan_pipeline_sequence(&[&first], &buffers[..3], &limits, &dispatches[..1]).unwrap();
        dispatches[1].bindings[1] = Binding {
            metal_index: 7,
            key: PoolKey::buffer(23),
            width: buffers[..]
                .iter()
                .find(|buffer| buffer.index == 23)
                .map_or(0, |buffer| buffer.bytes.len()),
        };
        let error =
            plan_pipeline_sequence(&[&first, &second, &first], &buffers, &limits, &dispatches)
                .unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Args);
        assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
        assert!(error
            .detail
            .unwrap()
            .contains("buffer 7 length 4 is shorter than reflected reach 68"));
    }

    #[test]
    fn subset_preflight_bounds_total_resources_separately_from_per_pass_descriptors() {
        let (mut translated, _, limits) = serial_fixture();
        let template = translated.reflection.bindings[0].clone();
        translated.reflection.bindings = (0..8)
            .map(|index| {
                let mut binding = template.clone();
                binding.metal_index = index;
                binding.descriptor.as_mut().unwrap().binding = index;
                binding
            })
            .collect();
        let mut buffers = (0..MAX_SERIAL_RESOURCES)
            .map(|index| BufferBinding {
                index: index as u32,
                bytes: vec![0; 4],
            })
            .collect::<Vec<_>>();
        let dispatches = (0..8)
            .map(|pass| BoundDispatch {
                grid: [1; 3],
                local: [1; 3],
                bindings: (0..8)
                    .map(|index| Binding {
                        metal_index: index,
                        key: PoolKey::buffer(pass * 8 + index),
                        width: test_pool_width(&buffers, pass * 8 + index),
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let translated = vec![&translated; 8];
        assert_eq!(
            plan_pipeline_sequence(&translated, &buffers, &limits, &dispatches)
                .unwrap()
                .plans
                .len(),
            8
        );
        buffers.push(BufferBinding {
            index: MAX_SERIAL_RESOURCES as u32,
            bytes: vec![0; 4],
        });
        let error =
            plan_pipeline_sequence(&translated, &buffers, &limits, &dispatches).unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
        assert!(error.detail.unwrap().contains("serial resource limit 64"));
    }

    #[test]
    fn mixed_preflight_rejects_missing_or_extra_pipeline_artifacts() {
        let (first, buffers, limits) = rebound_fixture();
        let second = alternate_pipeline_fixture();
        let dispatches = mixed_pipeline_dispatches(&buffers);
        for translated in [vec![], vec![&first], vec![&first, &second, &first]] {
            let error =
                plan_pipeline_sequence(&translated, &buffers, &limits, &dispatches).unwrap_err();
            assert_eq!(error.phase, ProviderPhase::Resolve);
            assert_eq!(error.class, ProviderErrorClass::Args);
            assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
            assert!(error.detail.unwrap().contains("artifact count"));
        }
    }

    #[test]
    fn mixed_preflight_checks_later_shader_footprint_before_any_execution() {
        let (first, buffers, limits) = rebound_fixture();
        let mut second = alternate_pipeline_fixture();
        second.reflection.bindings[0]
            .footprint
            .as_mut()
            .unwrap()
            .strided_accesses[0]
            .base_offset = 240;
        let dispatches = mixed_pipeline_dispatches(&buffers);
        plan_pipeline_sequence(&[&first], &buffers, &limits, &dispatches[..1]).unwrap();
        let error =
            plan_pipeline_sequence(&[&first, &second], &buffers, &limits, &dispatches).unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Args);
        assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
        assert!(error.detail.unwrap().contains("buffer 3"));
    }

    #[test]
    fn mixed_preflight_checks_later_shader_push_range_and_air_threadgroup_limit() {
        let (first, buffers, limits) = rebound_fixture();
        let second = alternate_pipeline_fixture();
        let dispatches = mixed_pipeline_dispatches(&buffers);
        let small_push_limits = vk::PhysicalDeviceLimits {
            max_push_constants_size: 48,
            ..limits
        };
        plan_pipeline_sequence(&[&first], &buffers, &small_push_limits, &dispatches[..1]).unwrap();
        let error = plan_pipeline_sequence(
            &[&first, &second],
            &buffers,
            &small_push_limits,
            &dispatches,
        )
        .unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
        assert!(error.detail.unwrap().contains("push constants end at 64"));

        let mut dispatches = dispatches;
        dispatches[1].local = [8, 4, 1];
        let error =
            plan_pipeline_sequence(&[&first, &second], &buffers, &limits, &dispatches).unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
        assert!(error.detail.unwrap().contains("AIR permits at most 16"));
    }

    #[test]
    fn rebound_preflight_accepts_ping_pong_and_collects_all_written_pool_keys() {
        let (translated, buffers, limits) = rebound_fixture();
        let dispatches = ping_pong_dispatches(&buffers);
        let first =
            plan_rebound_submission(&translated, &buffers, &limits, &dispatches[..1]).unwrap();
        assert_eq!(first.writable_pool_keys, BTreeSet::from([1]));
        let both = plan_rebound_submission(&translated, &buffers, &limits, &dispatches).unwrap();
        // Pool 0 is read-only initially and writable later; both resources need
        // one final update even though their binding roles change between passes.
        assert_eq!(both.writable_pool_keys, BTreeSet::from([0, 1]));
        assert_eq!(both.plans.len(), 2);
        for (plan, dispatch) in both.plans.iter().zip(&dispatches) {
            assert_eq!(plan.push_constants(plan.regions[0])[..3], dispatch.grid);
        }
    }

    #[test]
    fn rebound_preflight_separates_pool_keys_from_metal_binding_indices() {
        let (translated, mut buffers, limits) = rebound_fixture();
        buffers[0].index = 11;
        buffers[1].index = 19;
        let dispatches = [
            BoundDispatch {
                grid: [10, 3, 2],
                local: [8, 2, 1],
                bindings: vec![
                    Binding {
                        metal_index: 0,
                        key: PoolKey::buffer(11),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 11)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                    Binding {
                        metal_index: 1,
                        key: PoolKey::buffer(19),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 19)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                ],
            },
            BoundDispatch {
                grid: [10, 3, 2],
                local: [8, 2, 1],
                bindings: vec![
                    Binding {
                        metal_index: 0,
                        key: PoolKey::buffer(19),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 19)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                    Binding {
                        metal_index: 1,
                        key: PoolKey::buffer(11),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 11)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                ],
            },
        ];
        let planned = plan_rebound_submission(&translated, &buffers, &limits, &dispatches).unwrap();
        assert_eq!(planned.writable_pool_keys, BTreeSet::from([11, 19]));
    }

    #[test]
    fn rebound_preflight_rejects_ambiguous_or_incomplete_resource_maps() {
        let (translated, buffers, limits) = rebound_fixture();
        for bindings in [
            // Duplicate pool use.
            test_bindings(&[(0, 0), (1, 0)], &buffers),
            // Unknown pool key.
            test_bindings(&[(0, 0), (1, 2)], &buffers),
            // Missing pool resource and Metal slot.
            test_bindings(&[(0, 0)], &buffers),
            // Duplicate Metal slot.
            test_bindings(&[(0, 0), (0, 1)], &buffers),
            // Unknown Metal slot.
            test_bindings(&[(0, 0), (2, 1)], &buffers),
        ] {
            let mut dispatches = ping_pong_dispatches(&buffers);
            dispatches[1].bindings = bindings;
            let error =
                plan_rebound_submission(&translated, &buffers, &limits, &dispatches).unwrap_err();
            assert_eq!(error.class, ProviderErrorClass::Args);
            assert_eq!(error.phase, ProviderPhase::Resolve);
            assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
        }
        let duplicate_pool = vec![buffers[0].clone(), buffers[0].clone()];
        let error = plan_rebound_submission(
            &translated,
            &duplicate_pool,
            &limits,
            &ping_pong_dispatches(&buffers),
        )
        .unwrap_err();
        assert!(error
            .detail
            .as_deref()
            .unwrap()
            .contains("pool key 0 occurs more than once"));
        let error = plan_rebound_submission(
            &translated,
            &buffers[..1],
            &limits,
            &ping_pong_dispatches(&buffers),
        )
        .unwrap_err();
        assert!(error
            .detail
            .as_deref()
            .unwrap()
            .contains("unknown buffer pool key 1"));
    }

    #[test]
    fn rebound_preflight_checks_the_later_slot_footprint_against_its_mapped_buffer() {
        let (mut translated, mut buffers, limits) = rebound_fixture();
        buffers[0].bytes.truncate(4);
        buffers[1].bytes.truncate(8);
        translated.reflection.bindings[1]
            .footprint
            .as_mut()
            .unwrap()
            .strided_accesses[0]
            .access_size = 8;
        let dispatches = [
            BoundDispatch {
                grid: [1; 3],
                local: [1; 3],
                bindings: vec![
                    Binding {
                        metal_index: 0,
                        key: PoolKey::buffer(0),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 0)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                    Binding {
                        metal_index: 1,
                        key: PoolKey::buffer(1),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 1)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                ],
            },
            BoundDispatch {
                grid: [1; 3],
                local: [1; 3],
                bindings: vec![
                    Binding {
                        metal_index: 0,
                        key: PoolKey::buffer(1),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 1)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                    Binding {
                        metal_index: 1,
                        key: PoolKey::buffer(0),
                        width: buffers[..]
                            .iter()
                            .find(|buffer| buffer.index == 0)
                            .map_or(0, |buffer| buffer.bytes.len()),
                    },
                ],
            },
        ];
        plan_rebound_submission(&translated, &buffers, &limits, &dispatches[..1]).unwrap();
        let error =
            plan_rebound_submission(&translated, &buffers, &limits, &dispatches).unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Args);
        assert!(error.detail.as_deref().unwrap().contains("buffer 1"));
        assert!(error
            .detail
            .as_deref()
            .unwrap()
            .contains("reflected reach 8"));
        assert_eq!(error.completion, CompletionDisposition::NotSubmitted);

        let (translated, buffers, limits) = rebound_fixture();
        let mut dispatches = ping_pong_dispatches(&buffers);
        dispatches[1].grid = [11, 3, 2];
        let error =
            plan_rebound_submission(&translated, &buffers, &limits, &dispatches).unwrap_err();
        assert!(error
            .detail
            .as_deref()
            .unwrap()
            .contains("reflected reach 244"));
    }

    #[test]
    fn serial_preflight_preserves_each_dispatch_grid_and_tail_specialization() {
        let (translated, buffers, limits) = serial_fixture();
        let dispatches = [([10, 3, 2], [8, 2, 1]), ([7, 2, 1], [4, 1, 1])];
        let plans =
            plan_serial_submission(&translated, &buffers, &limits, dispatches[0], &dispatches)
                .unwrap();
        assert_eq!(plans.len(), 2);
        for (plan, (grid, _)) in plans.iter().zip(dispatches) {
            let launched: u32 = plan
                .regions
                .iter()
                .map(|region| {
                    region.local_size.into_iter().product::<u32>()
                        * region.group_count.into_iter().product::<u32>()
                })
                .sum();
            assert_eq!(launched, grid.into_iter().product::<u32>());
            assert_eq!(plan.push_constants(plan.regions[0])[..3], grid);
        }
        let specializations = plans
            .iter()
            .flat_map(|plan| &plan.regions)
            .map(|region| region.local_size)
            .collect::<BTreeSet<_>>();
        assert!(specializations.contains(&[8, 2, 1]));
        assert!(specializations.contains(&[2, 1, 1]));
        assert!(specializations.contains(&[4, 1, 1]));
        assert!(specializations.contains(&[3, 1, 1]));
    }

    #[test]
    fn serial_preflight_bounds_pass_count_and_rejects_ambiguous_first_sizes() {
        let (translated, buffers, limits) = serial_fixture();
        let first = ([10, 3, 2], [8, 2, 1]);
        assert_eq!(
            plan_serial_submission(&translated, &buffers, &limits, first, &[first; 8])
                .unwrap()
                .len(),
            8
        );
        for dispatches in [Vec::new(), vec![first; 9], vec![([1; 3], [1; 3])]] {
            let error = plan_serial_submission(&translated, &buffers, &limits, first, &dispatches)
                .unwrap_err();
            assert_eq!(error.class, ProviderErrorClass::Args);
            assert_eq!(error.phase, ProviderPhase::Resolve);
            assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
        }
    }

    #[test]
    fn serial_preflight_checks_later_buffer_reach_and_threadgroup_limits() {
        let (translated, buffers, limits) = serial_fixture();
        let first = ([10, 3, 2], [8, 2, 1]);
        let dispatches = [first, ([11, 3, 2], [8, 2, 1])];
        let error =
            plan_serial_submission(&translated, &buffers, &limits, first, &dispatches).unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Args);
        assert!(error
            .detail
            .as_deref()
            .unwrap()
            .contains("reflected reach 244"));
        assert_eq!(error.completion, CompletionDisposition::NotSubmitted);

        for local in [[0, 1, 1], [129, 1, 1], [8, 8, 1]] {
            let dispatches = [first, ([10, 3, 2], local)];
            let error = plan_serial_submission(&translated, &buffers, &limits, first, &dispatches)
                .unwrap_err();
            assert_eq!(error.class, ProviderErrorClass::Capability);
            assert_eq!(error.phase, ProviderPhase::Resolve);
            assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
        }
    }

    #[test]
    fn serial_preflight_checks_later_dispatch_count_and_shared_resource_limits() {
        let (translated, buffers, limits) = serial_fixture();
        let first = ([10, 3, 2], [8, 2, 1]);
        let dispatches = [first, ([10, 3, 2], [1, 2, 1])];
        let small_grid_limits = vk::PhysicalDeviceLimits {
            max_compute_work_group_count: [2, 65535, 65535],
            ..limits
        };
        let error = plan_serial_submission(
            &translated,
            &buffers,
            &small_grid_limits,
            first,
            &dispatches,
        )
        .unwrap_err();
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert!(error
            .detail
            .as_deref()
            .unwrap()
            .contains("group count dimension 0=10"));
        assert_eq!(error.completion, CompletionDisposition::NotSubmitted);

        for reduced in [
            vk::PhysicalDeviceLimits {
                max_descriptor_set_storage_buffers: 0,
                ..limits
            },
            vk::PhysicalDeviceLimits {
                max_storage_buffer_range: 239,
                ..limits
            },
            vk::PhysicalDeviceLimits {
                max_push_constants_size: 47,
                ..limits
            },
        ] {
            let error = plan_serial_submission(&translated, &buffers, &reduced, first, &[first])
                .unwrap_err();
            assert_eq!(error.class, ProviderErrorClass::Capability);
            assert_eq!(error.phase, ProviderPhase::Resolve);
            assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
        }
    }

    #[test]
    fn failures_before_queue_acceptance_are_not_submitted() {
        for phase in [ProviderPhase::Encode, ProviderPhase::Submit] {
            let failure = SubmissionFailure::Safe {
                phase,
                error: ExecutionFailure::vulkan(
                    vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                    "host allocation failed",
                ),
            };
            assert!(!failure.is_pending());
            let error = failure.into_provider();
            assert_eq!(error.phase, phase);
            assert_eq!(error.class, ProviderErrorClass::Execute);
            assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
            assert_eq!(error.detail.as_deref(), Some("host allocation failed"));
        }
    }

    #[test]
    fn queue_submit_only_allocation_errors_guarantee_safe_rejection() {
        for result in [
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        ] {
            let failure = SubmissionFailure::from_queue_submit(ExecutionFailure::vulkan(
                result,
                "queue allocation failed",
            ));
            assert!(!failure.is_pending());
            let error = failure.into_provider();
            assert_eq!(error.phase, ProviderPhase::Submit);
            assert_eq!(error.completion, CompletionDisposition::NotSubmitted);
        }

        let unknown = SubmissionFailure::from_queue_submit(ExecutionFailure::vulkan(
            vk::Result::ERROR_UNKNOWN,
            "queue outcome unknown",
        ));
        assert!(unknown.is_pending());
        let error = unknown.into_provider();
        assert_eq!(error.phase, ProviderPhase::Submit);
        assert_eq!(error.class, ProviderErrorClass::Execute);
        assert_eq!(
            error.completion,
            CompletionDisposition::SubmittedUnknown { token: None }
        );

        let lost = SubmissionFailure::from_queue_submit(ExecutionFailure::vulkan(
            vk::Result::ERROR_DEVICE_LOST,
            "queue device lost",
        ));
        assert!(lost.is_pending());
        let error = lost.into_provider();
        assert_eq!(error.phase, ProviderPhase::Submit);
        assert_eq!(error.class, ProviderErrorClass::DeviceLost);
        assert_eq!(
            error.completion,
            CompletionDisposition::DeviceLost { token: None }
        );
    }

    #[test]
    fn wait_failures_preserve_unknown_completion_and_pending_resources() {
        for result in [vk::Result::TIMEOUT, vk::Result::ERROR_OUT_OF_HOST_MEMORY] {
            let failure = SubmissionFailure::Pending {
                phase: ProviderPhase::Wait,
                error: ExecutionFailure::vulkan(result, "wait did not establish completion"),
            };
            assert!(failure.is_pending());
            let error = failure.into_provider();
            assert_eq!(error.phase, ProviderPhase::Wait);
            assert_eq!(error.class, ProviderErrorClass::Execute);
            assert_eq!(
                error.completion,
                CompletionDisposition::SubmittedUnknown { token: None }
            );
        }
    }

    #[test]
    fn device_loss_is_classified_from_vulkan_result() {
        assert!(
            SubmissionFailure::from_queue_submit(ExecutionFailure::vulkan(
                vk::Result::ERROR_DEVICE_LOST,
                "queue device lost",
            ))
            .is_device_lost()
        );
        assert!(
            !SubmissionFailure::from_queue_submit(ExecutionFailure::vulkan(
                vk::Result::ERROR_UNKNOWN,
                "queue outcome unknown",
            ))
            .is_device_lost()
        );
        let error = SubmissionFailure::Pending {
            phase: ProviderPhase::Wait,
            error: ExecutionFailure::vulkan(
                vk::Result::ERROR_DEVICE_LOST,
                "arbitrary driver detail",
            ),
        }
        .into_provider();
        assert_eq!(error.phase, ProviderPhase::Wait);
        assert_eq!(error.class, ProviderErrorClass::DeviceLost);
        assert_eq!(
            error.completion,
            CompletionDisposition::DeviceLost { token: None }
        );

        let misleading_detail = SubmissionFailure::Pending {
            phase: ProviderPhase::Wait,
            error: ExecutionFailure::vulkan(
                vk::Result::TIMEOUT,
                "ERROR_DEVICE_LOST appears only in diagnostics",
            ),
        }
        .into_provider();
        assert_eq!(misleading_detail.class, ProviderErrorClass::Execute);
        assert_eq!(
            misleading_detail.completion,
            CompletionDisposition::SubmittedUnknown { token: None }
        );
    }

    #[test]
    fn only_unobserved_live_work_retains_vulkan_handles() {
        assert_eq!(
            resource_drop_policy(true, false, false),
            ResourceDropPolicy::Retain
        );
        assert_eq!(
            resource_drop_policy(true, false, true),
            ResourceDropPolicy::Destroy
        );
        assert_eq!(
            resource_drop_policy(true, true, false),
            ResourceDropPolicy::Destroy
        );
        assert_eq!(
            resource_drop_policy(false, false, false),
            ResourceDropPolicy::Destroy
        );
    }

    /// Build a context for the lifecycle tests, following the skip pattern the
    /// other device-backed tests use.
    fn lifecycle_context() -> Option<Arc<VulkanContext>> {
        match VulkanContext::new() {
            Ok(context) => Some(Arc::new(context)),
            Err(error) => {
                eprintln!("SKIP: no Vulkan device: {error}");
                None
            }
        }
    }

    /// Health, admission and the refusal fields have to describe one state.
    ///
    /// This is the single-source check: a context that reports `Usable` admits,
    /// and a context that reports a terminal health returns the refusal naming
    /// that same terminal cause.
    fn assert_health_and_refusal_agree(context: &VulkanContext) {
        let health = context.health();
        match context.admit() {
            Ok(()) => assert_eq!(health, ProviderHealth::Usable),
            Err(error) => {
                assert!(
                    !health.is_usable(),
                    "usable context refused work: {error:?}"
                );
                let expected = match health {
                    ProviderHealth::DeviceLost => "device_lost",
                    ProviderHealth::Exhausted => "abandonment_budget",
                    ProviderHealth::Usable => unreachable!("handled by the match above"),
                };
                assert_eq!(error.phase, ProviderPhase::Resolve);
                assert_eq!(error.retryability, Retryability::RetryAfterRecreate);
                assert_eq!(
                    error.fields.get("terminal"),
                    Some(&FieldValue::Text(expected.to_owned())),
                    "refusal field disagrees with the reported health: {error:?}"
                );
            }
        }
    }

    #[test]
    fn exhausted_lifecycle_refuses_new_work_idempotently() {
        let Some(context) = lifecycle_context() else {
            return;
        };
        assert_eq!(context.health(), ProviderHealth::Usable);
        assert!(context.admit().is_ok());
        assert_eq!(context.abandonment_stats(), (0, 0));

        // One unretirable submission is the whole Vulkan budget, so the first
        // abandonment is what ends this instance.
        assert_eq!(
            context.record_abandonment(4096),
            AbandonmentOutcome::Exhausted
        );
        assert_eq!(context.abandonment_stats(), (1, 4096));
        assert_eq!(context.health(), ProviderHealth::Exhausted);

        let refusal = context
            .admit()
            .expect_err("exhausted context admitted work");
        assert_eq!(refusal.class, ProviderErrorClass::Resource);
        assert_eq!(refusal.slug, "provider_unavailable");
        assert_eq!(refusal.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(refusal.completion, CompletionDisposition::NotSubmitted);
        assert_eq!(
            refusal.fields.get("terminal"),
            Some(&FieldValue::Text("abandonment_budget".into()))
        );
        assert_eq!(
            refusal.fields.get("abandoned_submissions"),
            Some(&FieldValue::Unsigned(1))
        );
        assert_eq!(
            refusal.fields.get("abandoned_bytes"),
            Some(&FieldValue::Unsigned(4096))
        );

        // Retries answer the same structured refusal and neither re-charge the
        // budget nor drift into another reason.
        for _ in 0..3 {
            assert_eq!(
                context
                    .admit()
                    .expect_err("exhausted context admitted work"),
                refusal
            );
            assert_eq!(
                context.record_abandonment(4096),
                AbandonmentOutcome::Exhausted
            );
            assert_eq!(context.abandonment_stats(), (1, 4096));
        }
        assert_eq!(context.health(), ProviderHealth::Exhausted);
    }

    #[test]
    fn device_loss_and_budget_exhaustion_stay_distinguishable() {
        let Some(lost_context) = lifecycle_context() else {
            return;
        };
        let Some(exhausted_context) = lifecycle_context() else {
            return;
        };
        assert_eq!(
            exhausted_context.record_abandonment(64),
            AbandonmentOutcome::Exhausted
        );
        lost_context.mark_device_lost();

        assert_eq!(lost_context.health(), ProviderHealth::DeviceLost);
        assert_eq!(exhausted_context.health(), ProviderHealth::Exhausted);

        let lost = lost_context.admit().expect_err("lost device admitted work");
        let exhausted = exhausted_context
            .admit()
            .expect_err("exhausted context admitted work");
        assert_eq!(lost.class, ProviderErrorClass::DeviceLost);
        assert_eq!(lost.slug, "device_lost");
        assert_eq!(lost.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(
            lost.completion,
            CompletionDisposition::DeviceLost { token: None }
        );
        assert_eq!(
            lost.fields.get("terminal"),
            Some(&FieldValue::Text("device_lost".into()))
        );
        assert_eq!(exhausted.class, ProviderErrorClass::Resource);
        assert_eq!(exhausted.slug, "provider_unavailable");
        assert_eq!(exhausted.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(exhausted.completion, CompletionDisposition::NotSubmitted);
        assert_eq!(
            exhausted.fields.get("terminal"),
            Some(&FieldValue::Text("abandonment_budget".into()))
        );
        assert_ne!(lost.slug, exhausted.slug);

        // An observed device loss outranks an exhausted budget, so the cause is
        // upgraded instead of being masked by the earlier abandonment.
        exhausted_context.mark_device_lost();
        assert_eq!(exhausted_context.health(), ProviderHealth::DeviceLost);
        assert_eq!(
            exhausted_context
                .admit()
                .expect_err("lost device admitted work")
                .slug,
            "device_lost"
        );
    }

    #[test]
    fn health_admission_and_refusals_follow_one_lifecycle() {
        let Some(context) = lifecycle_context() else {
            return;
        };
        // Control: the normal path still admits and still executes. The
        // `copy_word` fixture writes buffer 1 from buffer 0, so an exact
        // writeback proves admission, submission and readback all ran.
        assert_health_and_refusal_agree(&context);
        let executor = Arc::new(VulkanExecutor {
            context: Arc::clone(&context),
        });
        let device = metal_api_core::Device::new(
            Arc::clone(&executor) as Arc<dyn metal_api_core::ComputeExecutor>
        );
        let library = device
            .new_library_with_air(include_str!(
                "../../../examples/metal-smoke/shaders/kernel_copy_word.ll"
            ))
            .expect("the fixture library loads");
        let function = library
            .function("copy_word")
            .expect("the fixture entry exists");
        let pipeline = executor
            .new_compute_pipeline(&function)
            .expect("pipeline creates");
        let submission = metal_api_core::ComputeSubmission {
            pipeline,
            buffers: vec![
                metal_api_core::BufferBinding {
                    index: 0,
                    bytes: 0x6745_2301_u32.to_le_bytes().to_vec(),
                },
                metal_api_core::BufferBinding {
                    index: 1,
                    bytes: vec![0_u8; 4],
                },
            ],
            textures: Vec::new(),
            threads_per_grid: metal_api_core::Size::new(1, 1, 1).expect("grid size"),
            threads_per_threadgroup: metal_api_core::Size::new(1, 1, 1).expect("local size"),
        };
        let updates = executor.execute(submission).expect("the copy executes");
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].index, 1);
        assert_eq!(updates[0].bytes, 0x6745_2301_u32.to_le_bytes());
        assert_eq!(context.health(), ProviderHealth::Usable);
        assert_eq!(context.abandonment_stats(), (0, 0));
        assert_health_and_refusal_agree(&context);

        // Every terminal transition keeps the same agreement, including the
        // sealed cause that records no abandoned work.
        let Some(sealed) = lifecycle_context() else {
            return;
        };
        sealed.mark_unobservable_submission();
        assert_eq!(sealed.health(), ProviderHealth::Exhausted);
        assert_eq!(
            sealed.abandonment_stats(),
            (0, 0),
            "a queue-refused submission is not abandoned GPU work"
        );
        assert_health_and_refusal_agree(&sealed);

        let Some(exhausted) = lifecycle_context() else {
            return;
        };
        exhausted.record_abandonment(1024);
        assert_health_and_refusal_agree(&exhausted);

        let Some(lost) = lifecycle_context() else {
            return;
        };
        lost.mark_device_lost();
        assert_health_and_refusal_agree(&lost);
    }

    #[test]
    fn readback_failure_is_failed_after_queue_retirement() {
        let error = ExecutionFailure::vulkan(vk::Result::ERROR_MEMORY_MAP_FAILED, "readback map")
            .into_readback_provider();
        assert_eq!(error.phase, ProviderPhase::Readback);
        assert_eq!(error.class, ProviderErrorClass::Execute);
        assert_eq!(
            error.completion,
            CompletionDisposition::Failed { token: None }
        );

        let lost = ExecutionFailure::vulkan(vk::Result::ERROR_DEVICE_LOST, "readback map")
            .into_readback_provider();
        assert_eq!(lost.class, ProviderErrorClass::DeviceLost);
        assert_eq!(
            lost.completion,
            CompletionDisposition::Failed { token: None }
        );
    }

    fn spirv_bytes(instructions: &[&[u32]]) -> Vec<u8> {
        let mut words = vec![0x0723_0203, 0x0001_0400, 0, 1, 0];
        for instruction in instructions {
            words.extend_from_slice(instruction);
        }
        words.into_iter().flat_map(u32::to_le_bytes).collect()
    }

    /// The minimal translated-vertex shape the y rewrite works on: one output
    /// variable, optionally decorated `BuiltIn Position`, optionally stored to.
    /// `store` and `decorate` are separate so the refusal paths can be built.
    fn minimal_position_module(store: bool, decorate: bool) -> Vec<u8> {
        let mut instructions: Vec<Vec<u32>> = vec![
            vec![(2 << 16) | Op::Capability as u32, Capability::Shader as u32],
            vec![(3 << 16) | Op::MemoryModel as u32, 0, 1],
            vec![
                (4 << 16) | Op::Decorate as u32,
                4,
                Decoration::BuiltIn as u32,
                if decorate {
                    BuiltIn::Position as u32
                } else {
                    BuiltIn::VertexIndex as u32
                },
            ],
            vec![(3 << 16) | Op::TypeFloat as u32, 1, 32],
            vec![(4 << 16) | Op::TypeVector as u32, 2, 1, 4],
            vec![
                (4 << 16) | Op::TypePointer as u32,
                3,
                spirv::StorageClass::Output as u32,
                2,
            ],
            vec![
                (4 << 16) | Op::Variable as u32,
                3,
                4,
                spirv::StorageClass::Output as u32,
            ],
            vec![(2 << 16) | Op::TypeVoid as u32, 5],
            vec![(3 << 16) | Op::TypeFunction as u32, 6, 5],
            vec![
                (5 << 16) | Op::Function as u32,
                5,
                7,
                spirv::FunctionControl::NONE.bits(),
                6,
            ],
            vec![(2 << 16) | Op::Label as u32, 8],
            vec![(3 << 16) | Op::Undef as u32, 2, 9],
        ];
        if store {
            instructions.push(vec![(3 << 16) | Op::Store as u32, 4, 9]);
        }
        instructions.push(vec![(1 << 16) | Op::Return as u32]);
        instructions.push(vec![(1 << 16) | Op::FunctionEnd as u32]);
        let mut words = vec![0x0723_0203, 0x0001_0400, 0, 10, 0];
        for instruction in &instructions {
            words.extend_from_slice(instruction);
        }
        words.into_iter().flat_map(u32::to_le_bytes).collect()
    }

    #[test]
    fn negate_position_y_rewrites_the_position_store_and_refuses_unusable_modules() {
        let module = minimal_position_module(true, true);
        let rewritten = negate_position_y(&module).expect("the position store rewrites");
        // The extract, the negate and the insert, and nothing else.
        assert_eq!(rewritten.len(), module.len() + 15 * 4);
        let words = rewritten
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        assert_eq!(words[3], 13, "three new ids extend the bound");
        let mut cursor = 5;
        let mut store = None;
        while cursor < words.len() {
            if words[cursor] & 0xffff == Op::Store as u32 {
                store = Some(cursor);
            }
            cursor += (words[cursor] >> 16) as usize;
        }
        let store = store.expect("the store stays");
        let insert = store - 6;
        let negate = insert - 4;
        let extract = negate - 5;
        assert_eq!(words[extract], (5 << 16) | Op::CompositeExtract as u32);
        assert_eq!(words[negate], (4 << 16) | Op::FNegate as u32);
        assert_eq!(words[insert], (6 << 16) | Op::CompositeInsert as u32);
        assert_eq!(words[extract + 3], 9, "the extract reads the stored value");
        assert_eq!(words[extract + 4], 1, "the extract reads the y component");
        assert_eq!(words[negate + 3], words[extract + 2]);
        assert_eq!(words[insert + 3], words[negate + 2]);
        assert_eq!(words[insert + 5], 1, "the insert replaces the y component");
        assert_eq!(words[store + 1], 4, "the store still targets the position");
        assert_eq!(words[store + 2], words[insert + 2]);

        let no_store = minimal_position_module(false, true);
        let error =
            negate_position_y(&no_store).expect_err("a position output without a store is refused");
        assert!(error.message().contains("never stores"), "{error}");

        let no_position = minimal_position_module(true, false);
        let error =
            negate_position_y(&no_position).expect_err("a module without a position is refused");
        assert!(error.message().contains("no BuiltIn Position"), "{error}");
    }

    #[test]
    fn phase_one_spirv_feature_gate_accepts_reviewed_capabilities_and_rejects_optional_ones() {
        let shader = [
            (2_u32 << 16) | Op::Capability as u32,
            Capability::Shader as u32,
        ];
        assert!(
            validate_spirv_capabilities(&spirv_bytes(&[&shader]), SpirvFeaturePolicy::PHASE1)
                .is_ok()
        );

        // The reviewed texture fixtures need these core capabilities, and the
        // device enables the matching shaderInt8/shaderInt64 features.
        for capability in [
            Capability::Int8,
            Capability::Int64,
            Capability::ImageQuery,
            Capability::Sampled1D,
            Capability::SampledBuffer,
        ] {
            let instruction = [(2_u32 << 16) | Op::Capability as u32, capability as u32];
            assert!(
                validate_spirv_capabilities(
                    &spirv_bytes(&[&shader, &instruction]),
                    SpirvFeaturePolicy::PHASE1
                )
                .is_ok(),
                "{capability:?} is inside the reviewed subset"
            );
        }

        // A capability outside the reviewed subset (Float64 has no matching
        // enabled feature) stays refused.
        let float64 = [
            (2_u32 << 16) | Op::Capability as u32,
            Capability::Float64 as u32,
        ];
        let error = validate_spirv_capabilities(
            &spirv_bytes(&[&shader, &float64]),
            SpirvFeaturePolicy::PHASE1,
        )
        .unwrap_err();
        assert!(error.message().contains("outside the Phase 1 subset"));

        let extension = [(2_u32 << 16) | Op::Extension as u32, 0];
        let error = validate_spirv_capabilities(
            &spirv_bytes(&[&shader, &extension]),
            SpirvFeaturePolicy::PHASE1,
        )
        .unwrap_err();
        assert!(error.message().contains("extensions"));
    }

    /// The extension instruction's operand is a NUL-terminated literal string
    /// packed into words, so the decoder is what a policy-based admission has to
    /// read; the helper is exercised through the gate itself above.
    fn extension_instruction(name: &str) -> Vec<u32> {
        let mut bytes = name.as_bytes().to_vec();
        bytes.push(0);
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        let words = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        let mut instruction = vec![((1 + words.len()) as u32) << 16 | Op::Extension as u32];
        instruction.extend_from_slice(&words);
        instruction
    }

    /// R8: `FloatControls2` is the first capability a *device* answers.
    ///
    /// Under the phase-1 policy the capability and the extension name it rides
    /// in on keep the refusals they had before the gate grew a device arm — the
    /// same sentences, so a device without the feature answers exactly as it did
    /// before. Under a policy that admits the feature both are let through, and
    /// a module naming *another* extension is still refused: the device arm opens
    /// one capability and one extension name, never "extensions".
    #[test]
    fn float_controls2_rides_the_device_policy_and_nothing_else_does() {
        let shader = [
            (2_u32 << 16) | Op::Capability as u32,
            Capability::Shader as u32,
        ];
        let float_controls2 = [
            (2_u32 << 16) | Op::Capability as u32,
            Capability::FloatControls2 as u32,
        ];
        let extension = extension_instruction(SPV_KHR_FLOAT_CONTROLS2);
        let other_extension = extension_instruction("SPV_EXT_descriptor_indexing");

        // The capability number the decline text carries, pinned here because it
        // is what the render rail's decline recorded on the real device.
        assert_eq!(Capability::FloatControls2 as u32, 6029);

        let phase1 = validate_spirv_capabilities(
            &spirv_bytes(&[&shader, &float_controls2]),
            SpirvFeaturePolicy::PHASE1,
        )
        .unwrap_err();
        assert!(
            phase1.message().contains("capability 6029"),
            "the refusal keeps the capability number: {}",
            phase1.message()
        );
        let phase1_extension = validate_spirv_capabilities(
            &spirv_bytes(&[&shader, &extension]),
            SpirvFeaturePolicy::PHASE1,
        )
        .unwrap_err();
        assert_eq!(
            phase1_extension.message(),
            "SPIR-V extensions are outside the Phase 1 feature subset"
        );

        let admitted = SpirvFeaturePolicy::PHASE1.with_float_controls2(true);
        assert!(
            validate_spirv_capabilities(&spirv_bytes(&[&shader, &float_controls2]), admitted)
                .is_ok()
        );
        assert!(
            validate_spirv_capabilities(&spirv_bytes(&[&shader, &extension]), admitted).is_ok()
        );
        // Both together, in the order the translator emits them.
        assert!(validate_spirv_capabilities(
            &spirv_bytes(&[&shader, &float_controls2, &extension]),
            admitted
        )
        .is_ok());
        let other =
            validate_spirv_capabilities(&spirv_bytes(&[&shader, &other_extension]), admitted)
                .unwrap_err();
        assert_eq!(
            other.message(),
            "SPIR-V extensions are outside the Phase 1 feature subset"
        );
        // A capability the device did not answer for stays refused whatever the
        // float-controls bit says.
        let float64 = [
            (2_u32 << 16) | Op::Capability as u32,
            Capability::Float64 as u32,
        ];
        let still_refused =
            validate_spirv_capabilities(&spirv_bytes(&[&shader, &float64]), admitted).unwrap_err();
        assert!(still_refused.message().contains("capability 10"));
    }

    /// The widened family states every field it names on the sampler the rail
    /// creates (`research/docs/23` §109): six filter names x five address
    /// modes, each one asserted against the `VkSamplerCreateInfo` field that
    /// decides it, plus the fields the family pins (normalized coordinates, no
    /// comparison, no anisotropy, LOD range `0..=0`).
    #[test]
    fn the_sampler_family_states_every_field_on_the_create_info() {
        use metal_api_core::provider::{SamplerAddressMode, SamplerFilter, SamplerPolicy};

        for (filter, linear, mipmapped, mip) in [
            (
                SamplerFilter::Nearest,
                false,
                false,
                vk::SamplerMipmapMode::NEAREST,
            ),
            (
                SamplerFilter::Linear,
                true,
                false,
                vk::SamplerMipmapMode::NEAREST,
            ),
            (
                SamplerFilter::NearestMipNearest,
                false,
                true,
                vk::SamplerMipmapMode::NEAREST,
            ),
            (
                SamplerFilter::NearestMipLinear,
                false,
                true,
                vk::SamplerMipmapMode::LINEAR,
            ),
            (
                SamplerFilter::LinearMipNearest,
                true,
                true,
                vk::SamplerMipmapMode::NEAREST,
            ),
            (
                SamplerFilter::LinearMipLinear,
                true,
                true,
                vk::SamplerMipmapMode::LINEAR,
            ),
        ] {
            assert_eq!(filter.is_linear(), linear, "{filter:?}");
            assert_eq!(filter.is_mipmapped(), mipmapped, "{filter:?}");
            for (address, expected, feature) in [
                (
                    SamplerAddressMode::ClampToEdge,
                    vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    None,
                ),
                (
                    SamplerAddressMode::Repeat,
                    vk::SamplerAddressMode::REPEAT,
                    None,
                ),
                (
                    SamplerAddressMode::MirrorClampToEdge,
                    vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
                    Some("samplerMirrorClampToEdge"),
                ),
                (
                    SamplerAddressMode::MirrorRepeat,
                    vk::SamplerAddressMode::MIRRORED_REPEAT,
                    None,
                ),
                (
                    SamplerAddressMode::ClampToZero,
                    vk::SamplerAddressMode::CLAMP_TO_BORDER,
                    None,
                ),
            ] {
                let info = sampler_create_info(SamplerPolicy { filter, address });
                let vk_filter = if linear {
                    vk::Filter::LINEAR
                } else {
                    vk::Filter::NEAREST
                };
                let where_ = format!("{filter:?} {address:?}");
                assert!(info.mag_filter == vk_filter, "mag filter: {where_}");
                assert!(info.min_filter == vk_filter, "min filter: {where_}");
                assert!(info.mipmap_mode == mip, "mipmap mode: {where_}");
                assert!(info.address_mode_u == expected, "address u: {where_}");
                assert!(info.address_mode_v == expected, "address v: {where_}");
                assert!(info.address_mode_w == expected, "address w: {where_}");
                assert!(
                    info.border_color == vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
                    "border colour: {where_}"
                );
                assert_eq!(info.min_lod, 0.0);
                assert_eq!(info.max_lod, 0.0);
                assert!(info.unnormalized_coordinates == vk::FALSE);
                assert!(info.compare_enable == vk::FALSE);
                assert!(info.anisotropy_enable == vk::FALSE);
                assert_eq!(sampler_address_mode_feature(address), feature);
            }
        }
    }

    /// The census's own state reaches the family (`research/docs/23` §109):
    /// linear minification and magnification with a linear mip filter and
    /// mirrored-repeat addressing maps onto the two names §109 appended — and
    /// the states the family cannot state keep their named refusals instead of
    /// being approximated.
    #[test]
    fn the_mapping_admits_the_census_state_and_refuses_the_rest() {
        use metal2vulkan::reflect::{
            SamplerAddressMode as AirAddress, SamplerBorderColor, SamplerCompareFunction,
            SamplerCoordinates, SamplerFilter as AirFilter, SamplerMipFilter, SamplerReduction,
            StaticSamplerState,
        };
        use metal_api_core::provider::{SamplerAddressMode, SamplerFilter};

        let state = |filter: AirFilter, mip: SamplerMipFilter, address: AirAddress| {
            StaticSamplerState {
                min_filter: filter,
                mag_filter: filter,
                mip_filter: mip,
                address_mode_s: address,
                address_mode_t: address,
                address_mode_r: address,
                coordinates: SamplerCoordinates::Normalized,
                compare_function: SamplerCompareFunction::Never,
                max_anisotropy: 1,
                // Metal's own default maximum rather than zero: the family
                // states one mip level, so a maximum cannot exclude level zero
                // and the ordinary state has to stay admitted.
                lod_min_clamp: 0.0,
                lod_max_clamp: 65504.0,
                border_color: SamplerBorderColor::TransparentBlack,
                reduction: SamplerReduction::WeightedAverage,
                lod_bias: 0.0,
                raw_words: [0; 2],
            }
        };

        let census = static_sampler_policy(&state(
            AirFilter::Linear,
            SamplerMipFilter::Linear,
            AirAddress::MirroredRepeat,
        ))
        .expect("linear min/mag with a linear mip filter and mirror-repeat is inside the family");
        eprintln!("census-shaped state maps onto {census:?}");
        assert_eq!(census.filter, SamplerFilter::LinearMipLinear);
        assert_eq!(census.address, SamplerAddressMode::MirrorRepeat);

        let zero = static_sampler_policy(&state(
            AirFilter::Nearest,
            SamplerMipFilter::Nearest,
            AirAddress::ClampToZero,
        ))
        .expect("clamp-to-zero is inside the family");
        assert_eq!(zero.filter, SamplerFilter::NearestMipNearest);
        assert_eq!(zero.address, SamplerAddressMode::ClampToZero);

        // The third addressing axis is not read by any view this family samples
        // (2026-09-19, R44): `s` and `t` agree, `r` is the Metal default, and
        // the state is admitted as the mode the two read axes state.
        let mut third_axis = state(
            AirFilter::Nearest,
            SamplerMipFilter::None,
            AirAddress::ClampToZero,
        );
        third_axis.address_mode_r = AirAddress::ClampToEdge;
        let folded = static_sampler_policy(&third_axis)
            .expect("the third addressing axis is not read by a 2D sample");
        eprintln!("third-axis state folds onto {folded:?}");
        assert_eq!(folded.filter, SamplerFilter::Nearest);
        assert_eq!(folded.address, SamplerAddressMode::ClampToZero);

        // The two axes a 2D view *does* read still have to agree, and the
        // refusal names the rule rather than the third axis.
        let mut split = state(
            AirFilter::Nearest,
            SamplerMipFilter::None,
            AirAddress::ClampToZero,
        );
        split.address_mode_t = AirAddress::Repeat;
        let split = static_sampler_policy(&split)
            .expect_err("the two axes a 2D view reads have to state one mode");
        eprintln!("axes that address disagree: {}", split.message());
        assert!(split.message().contains("two addressing axes disagree"));

        let border = static_sampler_policy(&state(
            AirFilter::Linear,
            SamplerMipFilter::None,
            AirAddress::ClampToBorder,
        ))
        .expect_err("clamp-to-border needs a border colour the family does not name");
        eprintln!("clamp-to-border refused: {}", border.message());
        assert!(border.message().contains("states no border colour"));

        let bicubic = static_sampler_policy(&state(
            AirFilter::Bicubic,
            SamplerMipFilter::None,
            AirAddress::ClampToEdge,
        ))
        .expect_err("bicubic is not a filter the family names");
        eprintln!("bicubic refused: {}", bicubic.message());
        assert!(bicubic.message().contains("no bicubic filter"));
    }

    /// The support struct is the pair of readings, and the policy is the
    /// conjunction: a device that advertises the name with the bit off, or
    /// carries the bit without the name, keeps the phase-1 answer.
    #[test]
    fn float_controls2_support_admits_only_the_conjunction() {
        assert_eq!(
            FloatControls2Support::default().policy(),
            SpirvFeaturePolicy::PHASE1
        );
        for support in [
            FloatControls2Support {
                extension: true,
                feature: false,
            },
            FloatControls2Support {
                extension: false,
                feature: true,
            },
            FloatControls2Support::default(),
        ] {
            assert!(!support.enabled());
            assert!(!support.policy().float_controls2());
        }
        let both = FloatControls2Support {
            extension: true,
            feature: true,
        };
        assert!(both.extension_present());
        assert!(both.feature_reported());
        assert!(both.enabled());
        assert!(both.policy().float_controls2());
    }

    #[test]
    fn exact_tail_plan_covers_thirty_threads_in_four_regions() {
        use metal2vulkan::reflect::KernelDispatchRegion;

        let contract = KernelDispatch::safe_default();
        let plan = contract.plan([8, 2, 1], Some([10, 3, 1])).unwrap();
        assert_eq!(plan.threadgroups_per_grid, [2, 2, 1]);
        let expected = vec![
            KernelDispatchRegion {
                local_size: [8, 2, 1],
                group_count: [1, 1, 1],
                thread_base: [0, 0, 0],
                threadgroup_base: [0, 0, 0],
            },
            KernelDispatchRegion {
                local_size: [2, 2, 1],
                group_count: [1, 1, 1],
                thread_base: [8, 0, 0],
                threadgroup_base: [1, 0, 0],
            },
            KernelDispatchRegion {
                local_size: [8, 1, 1],
                group_count: [1, 1, 1],
                thread_base: [0, 2, 0],
                threadgroup_base: [0, 1, 0],
            },
            KernelDispatchRegion {
                local_size: [2, 1, 1],
                group_count: [1, 1, 1],
                thread_base: [8, 2, 0],
                threadgroup_base: [1, 1, 0],
            },
        ];
        assert_eq!(plan.regions, expected);
        assert_eq!(
            plan.regions
                .iter()
                .map(|region| plan.push_constants(*region))
                .collect::<Vec<_>>(),
            vec![
                [10, 3, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1],
                [10, 3, 1, 8, 0, 0, 1, 0, 0, 2, 2, 1],
                [10, 3, 1, 0, 2, 0, 0, 1, 0, 2, 2, 1],
                [10, 3, 1, 8, 2, 0, 1, 1, 0, 2, 2, 1],
            ]
        );
        let launched = plan
            .regions
            .iter()
            .map(|region| {
                region.local_size.into_iter().product::<u32>()
                    * region.group_count.into_iter().product::<u32>()
            })
            .sum::<u32>();
        assert_eq!(launched, 30);
    }

    #[test]
    fn global_id_strides_are_bounded_by_the_exact_thread_grid() {
        let footprint = BufferFootprint {
            static_ranges: Vec::new(),
            strided_accesses: vec![BufferStridedAccess {
                base_offset: 0,
                access_size: 4,
                terms: vec![
                    BufferStrideTerm {
                        source: BufferIndexSource::GlobalInvocationIdX,
                        stride: 4,
                    },
                    BufferStrideTerm {
                        source: BufferIndexSource::GlobalInvocationIdY,
                        stride: 40,
                    },
                ],
            }],
            has_unbounded_access: false,
        };
        assert_eq!(strided_footprint_reach(&footprint, [10, 3, 1]), Ok(120));
        let error = ensure_buffer_reach(0, 116, 120).unwrap_err();
        assert_eq!(
            error.message(),
            "buffer 0 length 116 is shorter than reflected reach 120"
        );
        assert!(ensure_buffer_reach(0, 120, 120).is_ok());
    }

    #[test]
    fn unsupported_or_overflowing_index_strides_are_refused() {
        let unsupported = BufferFootprint {
            static_ranges: Vec::new(),
            strided_accesses: vec![BufferStridedAccess {
                base_offset: 0,
                access_size: 4,
                terms: vec![BufferStrideTerm {
                    source: BufferIndexSource::LocalInvocationIndex,
                    stride: 4,
                }],
            }],
            has_unbounded_access: false,
        };
        assert_eq!(
            strided_footprint_reach(&unsupported, [10, 3, 1]),
            Err("uses an unsupported index source")
        );

        let overflowing = BufferFootprint {
            static_ranges: Vec::new(),
            strided_accesses: vec![BufferStridedAccess {
                base_offset: u64::MAX,
                access_size: 1,
                terms: Vec::new(),
            }],
            has_unbounded_access: false,
        };
        assert_eq!(
            strided_footprint_reach(&overflowing, [1, 1, 1]),
            Err("strided footprint overflows u64")
        );
    }

    #[test]
    fn read_only_buffers_are_not_returned_as_updates() {
        assert!(matches!(
            Some(ResourceAccess::ReadOnly),
            Some(ResourceAccess::Unused | ResourceAccess::ReadOnly)
        ));
        assert!(!matches!(
            Some(ResourceAccess::WriteOnly),
            Some(ResourceAccess::Unused | ResourceAccess::ReadOnly)
        ));
    }

    #[test]
    fn queue_selection_prefers_idle_queues_and_keeps_round_robin_ties() {
        assert_eq!(select_queue(&[0, 0, 0, 0], 0), 0);
        assert_eq!(select_queue(&[0, 0, 0, 0], 2), 2);
        assert_eq!(select_queue(&[3, 1, 2, 0], 0), 3);
        assert_eq!(select_queue(&[1, 1, 0, 1], 2), 2);
        assert_eq!(select_queue(&[2, 2, 2, 1], 3), 3);
        assert_eq!(select_queue(&[7], 5), 0);
        assert_eq!(select_queue(&[], 0), 0);
    }

    #[test]
    fn queue_priority_policy_reduces_to_select_queue_on_one_tier() {
        use metal_api_core::provider::QueuePriority;

        let probes = 3_usize;
        for len in 0..=3_usize {
            for encoded in 0..probes.pow(len as u32) {
                let mut loads = vec![0_usize; len];
                let mut rest = encoded;
                for load in &mut loads {
                    *load = rest % probes;
                    rest /= probes;
                }
                let priorities = vec![QueuePriority::Default; len];
                // The cursor range crosses the 7-slot window boundary on
                // purpose: the default path must stay the least-loaded rule for
                // every cursor, not only for the first window.
                for cursor in 0..32_usize {
                    assert_eq!(
                        select_queue_for_submission(&loads, &priorities, cursor),
                        select_queue(&loads, cursor),
                        "loads {loads:?} cursor {cursor} must keep the least-loaded rule"
                    );
                }
            }
        }
    }

    #[test]
    fn queue_priority_policy_separates_tiers_when_loads_tie() {
        use metal_api_core::provider::{QueuePriority, QueueSchedulingPolicy};

        let policy = QueueSchedulingPolicy::default();
        let loads = [0_usize, 0];
        let priorities = [QueuePriority::Low, QueuePriority::High];
        // Window slot 0 nominates the high tier, so the high queue wins even
        // though both queues are idle.
        assert_eq!(select_queue_for_submission(&loads, &priorities, 0), 1);
        // The last slot of the window belongs to the low tier, so the idle high
        // queue yields instead of running again.
        let low_slot = (policy.window() - 1) as usize;
        assert_eq!(
            select_queue_for_submission(&loads, &priorities, low_slot),
            0
        );
    }

    /// The `research/docs/21` §6 queue shape: one high queue, one default queue
    /// and six low queues, the marking the RTX 5060 experiment installs.
    fn rtx_5060_queue_tiers() -> Vec<QueuePriority> {
        use metal_api_core::provider::QueuePriority;

        vec![
            QueuePriority::High,
            QueuePriority::Default,
            QueuePriority::Low,
            QueuePriority::Low,
            QueuePriority::Low,
            QueuePriority::Low,
            QueuePriority::Low,
            QueuePriority::Low,
        ]
    }

    /// Violation labels of one observed tier sequence under the
    /// `research/docs/21` §6 window contract: the weighted shares, the
    /// high-tier streak bound, and one low tier per window. The fair policy
    /// sequence reports an empty list; a sequence that violates the contract
    /// is exactly the unfair allocation the observation surfaces
    /// (`queue_enqueue_counts` / `queue_submission_counts`) expose.
    fn queue_priority_window_violations(tiers_seen: &[QueuePriority]) -> Vec<&'static str> {
        use metal_api_core::provider::{QueuePriority, QueueSchedulingPolicy};

        let policy = QueueSchedulingPolicy::default();
        let window = usize::try_from(policy.window()).expect("window fits usize");
        assert_eq!(window, 7);
        assert_eq!(
            tiers_seen.len(),
            70,
            "the contract is stated over 10 windows"
        );
        let mut violations = Vec::new();
        let count = |tier: QueuePriority| tiers_seen.iter().filter(|seen| **seen == tier).count();
        let windows = tiers_seen.len() / window;
        let expected_high = policy.high_weight() as usize * windows;
        let expected_default = policy.medium_weight() as usize * windows;
        if count(QueuePriority::High) != expected_high
            || count(QueuePriority::Default) != expected_default
        {
            violations.push("tier_shares_off");
        }

        let mut streak = 0_usize;
        let mut longest = 0_usize;
        for tier in tiers_seen {
            streak = if *tier == QueuePriority::High {
                streak + 1
            } else {
                0
            };
            longest = longest.max(streak);
        }
        if longest > policy.high_priority_streak_limit() as usize {
            violations.push("high_streak_exceeds_weight");
        }

        for start in (0..tiers_seen.len() - window + 1).step_by(window) {
            let lows = tiers_seen[start..start + window]
                .iter()
                .filter(|seen| **seen == QueuePriority::Low)
                .count();
            if lows == 0 {
                violations.push("low_tier_starved_in_window");
            }
        }
        violations
    }

    /// Assert the window contract on an observed tier sequence: no violation
    /// label, and the fair §6 shape saturates the streak bound exactly.
    fn assert_queue_priority_window(tiers_seen: &[QueuePriority]) {
        use metal_api_core::provider::{QueuePriority, QueueSchedulingPolicy};

        let violations = queue_priority_window_violations(tiers_seen);
        assert!(
            violations.is_empty(),
            "the observed tier sequence violates the window contract: {violations:?}"
        );

        let policy = QueueSchedulingPolicy::default();
        let mut streak = 0_usize;
        let mut longest = 0_usize;
        for tier in tiers_seen {
            streak = if *tier == QueuePriority::High {
                streak + 1
            } else {
                0
            };
            longest = longest.max(streak);
        }
        assert_eq!(
            u32::try_from(longest).expect("streak fits u32"),
            policy.high_priority_streak_limit(),
            "the fair sequence saturates the high-tier streak bound"
        );
    }

    #[test]
    fn queue_priority_window_spreads_the_submission_tiers_over_the_rtx_5060_shape() {
        let tiers = rtx_5060_queue_tiers();
        // Every submission retires before the next one is enqueued, so the tier
        // table is the only state the policy sees. The cursor is the monotonic
        // selection counter, exactly as `VulkanContext::pick_queue` advances it.
        let loads = vec![0_usize; tiers.len()];
        let tiers_seen: Vec<QueuePriority> = (0..70_usize)
            .map(|cursor| tiers[select_queue_for_submission(&loads, &tiers, cursor)])
            .collect();
        assert_queue_priority_window(&tiers_seen);
    }

    #[test]
    fn queue_observation_surfaces_flag_an_unfair_allocation() {
        // The fair policy sequence — exactly the tier sequence the per-queue
        // observation counters and the enqueue probe report for ten retired
        // windows on the §6 queue shape — satisfies the contract.
        let tiers = rtx_5060_queue_tiers();
        let loads = vec![0_usize; tiers.len()];
        let fair: Vec<QueuePriority> = (0..70_usize)
            .map(|cursor| tiers[select_queue_for_submission(&loads, &tiers, cursor)])
            .collect();
        assert!(
            queue_priority_window_violations(&fair).is_empty(),
            "the fair policy sequence must satisfy the window contract"
        );

        // A scheduler pinned to the first (high) queue: both the streak bound
        // and the low-tier slot are violated, so the observation surfaces that
        // carry the sequence expose the unfairness instead of hiding it.
        let pinned = vec![QueuePriority::High; 70];
        let violations = queue_priority_window_violations(&pinned);
        assert!(violations.contains(&"high_streak_exceeds_weight"));
        assert!(violations.contains(&"low_tier_starved_in_window"));

        // A bursty allocation that lands the right tier totals (40/20/10) but
        // front-loads every low slot: the tier-share check alone cannot tell
        // the difference, and the window checks still flag it.
        let mut bursty = vec![QueuePriority::Low; 10];
        bursty.extend((0..40).map(|_| QueuePriority::High));
        bursty.extend((0..20).map(|_| QueuePriority::Default));
        let violations = queue_priority_window_violations(&bursty);
        assert!(violations.contains(&"high_streak_exceeds_weight"));
        assert!(violations.contains(&"low_tier_starved_in_window"));
        assert!(
            !violations.contains(&"tier_shares_off"),
            "a share-correct sequence must not be flagged for its totals"
        );
    }

    #[test]
    fn queue_priority_tiers_never_override_the_least_loaded_rule() {
        // The high queue is busy while six low queues are idle: window slot 0
        // nominates the high tier, and the high queue still must not jump ahead
        // of an idle queue (`research/docs/21` §3 invariant 1).
        let tiers = rtx_5060_queue_tiers();
        let mut loads = vec![0_usize; tiers.len()];
        loads[0] = 7;
        for cursor in 0..14_usize {
            let picked = select_queue_for_submission(&loads, &tiers, cursor);
            assert_ne!(picked, 0, "cursor {cursor} picked the busy high queue");
            assert_eq!(loads[picked], 0, "cursor {cursor} picked a busy queue");
        }
    }

    #[test]
    fn queue_priority_table_reaches_the_async_object_submit_path() {
        use metal_api_core::provider::{PipelineCompileRequest, ShaderSource, TextureFormat};

        let executor = match VulkanExecutor::new() {
            Ok(executor) => executor,
            Err(error) => {
                eprintln!("SKIP: no Vulkan device: {error}");
                return;
            }
        };
        let queues = executor.queue_count();
        assert!(queues >= 1, "a selected device exposes at least one queue");
        // The §6 marking, truncated to the device: Lavapipe exposes one queue
        // and therefore keeps the degenerate one-tier table.
        let installed: Vec<QueuePriority> =
            rtx_5060_queue_tiers().into_iter().take(queues).collect();
        executor
            .set_queue_priorities(&installed)
            .expect("a table with one entry per queue is accepted");
        assert_eq!(executor.queue_priorities(), installed);
        assert!(
            executor
                .set_queue_priorities(&installed[..queues - 1])
                .is_err(),
            "a table that does not describe the device is refused"
        );

        let observed = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&observed);
        executor.set_enqueue_probe_for_test(Arc::new(move |queue| {
            if let Ok(mut sequence) = sink.lock() {
                sequence.push(queue);
            }
        }));

        // The asynchronous object path commits one command buffer per probe
        // call, so its sequence is the observation surface for the window
        // contract below. The synchronous paths select through the same policy
        // (`synchronous_paths_select_queues_through_the_priority_policy`).
        let provider = crate::VulkanComputeProvider::with_executor(Arc::clone(&executor))
            .expect("provider")
            .with_async_execution(true);
        let device = metal_api_core::provider_api::Device::new(Arc::new(provider));
        let pipeline = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "read_texture_2d".to_owned(),
                logical_digest: metal_api_core::provider::SemanticDigest::new(
                    "metal-smoke-fixture-v1",
                    b"queue_priority_table".to_vec(),
                )
                .expect("digest"),
                source: ShaderSource::SanitizedLl(
                    include_str!("../../../examples/metal-smoke/shaders/kernel_read_texture_2d.ll")
                        .to_owned(),
                ),
            })
            .expect("pipeline");
        let mut texels = Vec::with_capacity(64);
        for value in 0..16_u32 {
            texels.extend_from_slice(&value.to_le_bytes());
        }
        let texture = device
            .new_texture_with_bytes(TextureFormat::R32Uint, 4, 4, texels)
            .expect("texture object");
        let output = device
            .new_buffer_with_bytes(vec![0_u8; 64])
            .expect("output buffer");
        let queue = device.new_command_queue();
        let submissions = 70_usize;
        for _ in 0..submissions {
            let command = queue.command_buffer();
            {
                let mut encoder = command.compute_command_encoder().expect("encoder");
                encoder
                    .set_compute_pipeline_state(&pipeline)
                    .expect("pipeline state");
                encoder.set_texture(0, &texture).expect("texture binding");
                encoder
                    .set_buffer(0, &output.view(0, 64).unwrap())
                    .expect("buffer binding");
                encoder
                    .dispatch_threads(
                        metal_api_core::Size::new(1, 1, 1).unwrap(),
                        metal_api_core::Size::new(1, 1, 1).unwrap(),
                    )
                    .expect("dispatch");
                encoder.end_encoding().expect("end encoding");
            }
            command.commit().expect("commit");
            command.wait_until_completed().expect("completion");
        }
        executor.clear_enqueue_probe_for_test();
        // Priority only steers queue selection: the landing bytes are the same
        // ones the synchronous path produces.
        assert_eq!(output.read().expect("readback")[..4], 0_u32.to_le_bytes());

        let sequence = observed.lock().expect("probe sequence").clone();
        assert_eq!(sequence.len(), submissions, "one probe call per commit");
        assert!(sequence.iter().all(|index| *index < queues));
        let counts = executor.queue_submission_counts();
        assert_eq!(counts.len(), queues);
        assert_eq!(counts.iter().sum::<usize>(), submissions);
        for (index, count) in counts.iter().enumerate() {
            assert_eq!(
                *count,
                sequence.iter().filter(|picked| **picked == index).count(),
                "queue {index} counts disagree with the enqueue probe"
            );
        }
        // The production observation surfaces agree with the probe on every
        // queue: selections are counted at enqueue time and retirements at
        // completion time, so the scheduler's allocation stays queryable
        // without the test-only probe (`research/docs/21` §6 observation).
        let enqueues = executor.queue_enqueue_counts();
        let completions = executor.queue_completion_counts();
        assert_eq!(enqueues.len(), queues);
        assert_eq!(completions.len(), queues);
        assert_eq!(enqueues.iter().sum::<usize>(), submissions);
        assert_eq!(completions.iter().sum::<usize>(), submissions);
        for (index, count) in counts.iter().enumerate() {
            assert_eq!(
                enqueues[index],
                sequence.iter().filter(|picked| **picked == index).count(),
                "queue {index} enqueue counts disagree with the enqueue probe"
            );
            assert_eq!(
                completions[index], *count,
                "queue {index} completion counts disagree with the submission counts"
            );
        }
        if queues < 7 || installed.iter().collect::<BTreeSet<_>>().len() < 3 {
            eprintln!(
                "SKIP queue priority window: queues={queues} tiers={installed:?} \
                 submissions={submissions}"
            );
            return;
        }
        let tiers_seen: Vec<QueuePriority> =
            sequence.iter().map(|index| installed[*index]).collect();
        assert_queue_priority_window(&tiers_seen);
    }

    #[test]
    fn provider_queue_priority_marking_installs_on_the_device_table() {
        use metal_api_core::provider::ComputeProvider;

        let executor = match VulkanExecutor::new() {
            Ok(executor) => executor,
            Err(error) => {
                eprintln!("SKIP: no Vulkan device: {error}");
                return;
            }
        };
        let queues = executor.queue_count();
        assert!(queues >= 1, "a selected device exposes at least one queue");
        let provider =
            crate::VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider");

        // A marking longer than the device is truncated and a shorter one is
        // padded, so the owner never has to know the queue count.
        let marking = rtx_5060_queue_tiers();
        let installed = provider
            .set_queue_priorities(&marking)
            .expect("a marking expands to the device table");
        let expected: Vec<QueuePriority> = marking.into_iter().take(queues).collect();
        assert_eq!(installed, expected);
        assert_eq!(installed.len(), queues);
        // The response is the table the scheduler actually reads.
        assert_eq!(executor.queue_priorities(), installed);

        // The empty marking is the all-`Default` table, i.e. exactly the
        // scheduling a connection that never sends the request keeps.
        let cleared = provider
            .set_queue_priorities(&[])
            .expect("an empty marking clears the table");
        assert_eq!(cleared, vec![QueuePriority::Default; queues]);
        assert_eq!(executor.queue_priorities(), cleared);
    }

    #[test]
    fn synchronous_paths_select_queues_through_the_priority_policy() {
        use metal_api_core::provider::{ComputeProvider, PipelineCompileRequest, ShaderSource};

        let executor = match VulkanExecutor::new() {
            Ok(executor) => executor,
            Err(error) => {
                eprintln!("SKIP: no Vulkan device: {error}");
                return;
            }
        };
        let queues = executor.queue_count();
        // The §6 marking, truncated to the device: Lavapipe exposes one queue
        // and therefore keeps the degenerate one-tier table.
        let installed: Vec<QueuePriority> =
            rtx_5060_queue_tiers().into_iter().take(queues).collect();
        executor
            .set_queue_priorities(&installed)
            .expect("a table with one entry per queue is accepted");

        let observed = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&observed);
        executor.set_enqueue_probe_for_test(Arc::new(move |queue| {
            if let Ok(mut sequence) = sink.lock() {
                sequence.push(queue);
            }
        }));

        // Path one: the standalone `ComputeExecutor` entry point, which used to
        // be pinned to queue 0.
        let device = metal_api_core::Device::new(
            Arc::clone(&executor) as Arc<dyn metal_api_core::ComputeExecutor>
        );
        let library = device
            .new_library_with_air(include_str!(
                "../../../examples/metal-smoke/shaders/kernel_read_texture_2d.ll"
            ))
            .expect("the fixture library loads");
        let function = library
            .function("read_texture_2d")
            .expect("the fixture entry exists");
        let pipeline = executor
            .new_compute_pipeline(&function)
            .expect("pipeline creates");
        let mut texels = Vec::with_capacity(64);
        for value in 0..16_u32 {
            texels.extend_from_slice(&value.to_le_bytes());
        }
        let submission = metal_api_core::ComputeSubmission {
            pipeline,
            buffers: vec![metal_api_core::BufferBinding {
                index: 0,
                bytes: vec![0_u8; 64],
            }],
            textures: vec![metal_api_core::provider::TextureView {
                view_id: metal_api_core::provider::ViewId::new(910),
                metal_binding: 0,
                allocation_id: metal_api_core::provider::AllocationId::new(911),
                texture_type: metal_api_core::provider::TextureType::D2,
                format: metal_api_core::provider::TextureFormat::R32Uint,
                width: 4,
                height: 4,
                depth: 1,
                array_length: 1,
                sample_count: 1,
                access: metal_api_core::provider::TextureAccess::Sampled,
                source: metal_api_core::provider::TextureSource::OwnedBytes(texels),
            }],
            threads_per_grid: metal_api_core::Size::new(1, 1, 1).unwrap(),
            threads_per_threadgroup: metal_api_core::Size::new(1, 1, 1).unwrap(),
        };
        let updates = executor.execute(submission).expect("texture read executes");
        assert_eq!(updates[0].bytes[..4], 0_u32.to_le_bytes());

        // Path two: the synchronous provider, which used to call
        // `execute_on_context` on queue 0.
        let provider =
            crate::VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider");
        assert_eq!(
            provider
                .set_queue_priorities(&installed)
                .expect("the device-sized marking installs unchanged"),
            installed
        );
        let objects = metal_api_core::provider_api::Device::new(Arc::new(provider));
        let pipeline = objects
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "read_texture_2d".to_owned(),
                logical_digest: metal_api_core::provider::SemanticDigest::new(
                    "metal-smoke-fixture-v1",
                    b"synchronous_queue_selection".to_vec(),
                )
                .expect("digest"),
                source: ShaderSource::SanitizedLl(
                    include_str!("../../../examples/metal-smoke/shaders/kernel_read_texture_2d.ll")
                        .to_owned(),
                ),
            })
            .expect("pipeline");
        let mut texels = Vec::with_capacity(64);
        for value in 0..16_u32 {
            texels.extend_from_slice(&value.to_le_bytes());
        }
        let texture = objects
            .new_texture_with_bytes(
                metal_api_core::provider::TextureFormat::R32Uint,
                4,
                4,
                texels,
            )
            .expect("texture object");
        let output = objects
            .new_buffer_with_bytes(vec![0_u8; 64])
            .expect("output buffer");
        let queue = objects.new_command_queue();
        let command = queue.command_buffer();
        {
            let mut encoder = command.compute_command_encoder().expect("encoder");
            encoder
                .set_compute_pipeline_state(&pipeline)
                .expect("pipeline state");
            encoder.set_texture(0, &texture).expect("texture binding");
            encoder
                .set_buffer(0, &output.view(0, 64).unwrap())
                .expect("buffer binding");
            encoder
                .dispatch_threads(
                    metal_api_core::Size::new(1, 1, 1).unwrap(),
                    metal_api_core::Size::new(1, 1, 1).unwrap(),
                )
                .expect("dispatch");
            encoder.end_encoding().expect("end encoding");
        }
        command.commit().expect("commit");
        command.wait_until_completed().expect("completion");
        assert_eq!(output.read().expect("readback")[..4], 0_u32.to_le_bytes());
        executor.clear_enqueue_probe_for_test();

        // Both paths reported exactly one selection, and each one is the answer
        // the core policy gives for that cursor with every queue idle.
        let loads = vec![0_usize; queues];
        let sequence = observed.lock().expect("probe sequence").clone();
        assert_eq!(
            sequence.len(),
            2,
            "one probe call per synchronous submit: {sequence:?}"
        );
        for (cursor, picked) in sequence.iter().enumerate() {
            assert_eq!(
                *picked,
                select_queue_for_submission(&loads, &installed, cursor),
                "synchronous selection {cursor} did not go through the queue policy"
            );
        }
        // The marking is scheduler state: two submissions neither consume it
        // nor end the context that admits them.
        assert_eq!(executor.queue_priorities(), installed);
    }

    // -------------------------------------------------------------------
    // Real device-loss path: the driver's answer at a queue boundary.
    // -------------------------------------------------------------------

    use metal_api_core::provider::{
        AllocationId, AllocationRecord, BufferLease, BufferSource, BufferView, CompletionPolicy,
        CompletionToken, ComputePass, ComputeProvider, ComputeTrace, Dispatch, DispatchKind,
        DispatchType, LeaseReservation, OperationId, ResourceTableSnapshot, SubmissionId,
        TerminalLeaseState, TerminalState, TracePass, ValidatedComputeTrace, ViewId,
        PROVIDER_SCHEMA_VERSION,
    };

    /// The fault-address-name mapping the evidence field reports.
    #[test]
    fn device_fault_evidence_names_every_reported_address() {
        let snapshot = DeviceFaultSnapshot {
            extension_present: true,
            description: Some("page fault".to_owned()),
            addresses: vec![
                DeviceFaultAddress {
                    address_type: vk::DeviceFaultAddressTypeEXT::READ_INVALID.as_raw(),
                    reported_address: 0x1000,
                    address_precision: 0x40,
                },
                DeviceFaultAddress {
                    address_type: vk::DeviceFaultAddressTypeEXT::WRITE_INVALID.as_raw(),
                    reported_address: 0x2000,
                    address_precision: 0x10,
                },
                DeviceFaultAddress {
                    address_type: vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_FAULT.as_raw(),
                    reported_address: 0x3000,
                    address_precision: 4,
                },
                DeviceFaultAddress {
                    address_type: 99,
                    reported_address: 0x4000,
                    address_precision: 1,
                },
                // Beyond the reported cap: counted, not listed.
                DeviceFaultAddress {
                    address_type: vk::DeviceFaultAddressTypeEXT::NONE.as_raw(),
                    reported_address: 0x5000,
                    address_precision: 1,
                },
            ],
            vendor_info_count: 2,
            vendor_binary_size: 4096,
        };
        let fields: BTreeMap<String, FieldValue> = snapshot.evidence_fields().into_iter().collect();
        assert_eq!(
            fields.get("device_fault_extension"),
            Some(&FieldValue::Bool(true))
        );
        assert_eq!(
            fields.get("device_fault_description"),
            Some(&FieldValue::Text("page fault".to_owned()))
        );
        assert_eq!(
            fields.get("device_fault_addresses"),
            Some(&FieldValue::Unsigned(5))
        );
        assert_eq!(
            fields.get("device_fault_address_type_0"),
            Some(&FieldValue::Text("READ_INVALID".to_owned()))
        );
        assert_eq!(
            fields.get("device_fault_address_0"),
            Some(&FieldValue::Unsigned(0x1000))
        );
        assert_eq!(
            fields.get("device_fault_address_precision_0"),
            Some(&FieldValue::Unsigned(0x40))
        );
        assert_eq!(
            fields.get("device_fault_address_type_1"),
            Some(&FieldValue::Text("WRITE_INVALID".to_owned()))
        );
        assert_eq!(
            fields.get("device_fault_address_type_2"),
            Some(&FieldValue::Text("INSTRUCTION_POINTER_FAULT".to_owned()))
        );
        assert_eq!(
            fields.get("device_fault_address_type_3"),
            Some(&FieldValue::Text("UNKNOWN".to_owned()))
        );
        assert!(
            !fields.contains_key("device_fault_address_type_4"),
            "the listing is capped at {DEVICE_FAULT_ADDRESS_FIELDS} addresses"
        );
        assert_eq!(
            fields.get("device_fault_vendor_infos"),
            Some(&FieldValue::Unsigned(2))
        );
        assert_eq!(
            fields.get("device_fault_vendor_binary_size"),
            Some(&FieldValue::Unsigned(4096))
        );

        // An unavailable record is evidence too, and it never claims a fault.
        let unavailable: BTreeMap<String, FieldValue> = DeviceFaultSnapshot::unavailable(false)
            .evidence_fields()
            .into_iter()
            .collect();
        assert_eq!(
            unavailable.get("device_fault_extension"),
            Some(&FieldValue::Bool(false))
        );
        assert_eq!(
            unavailable.get("device_fault_addresses"),
            Some(&FieldValue::Unsigned(0))
        );
        assert!(!unavailable.contains_key("device_fault_description"));
    }

    /// A device that advertises `VK_EXT_device_fault` but never hands over
    /// `vkGetDeviceFaultInfoEXT` must not be driven.
    ///
    /// The Windows RTX 5060 ICD answers NULL for the entry point of a device
    /// extension it was not created with (2026-09-18), and ash's `loaded` stub
    /// turns a call on that answer into a panic the process cannot unwind, so
    /// the loader has to stay off and the record has to say "advertised, no
    /// addresses" instead of aborting the run.
    #[test]
    fn the_fault_loader_needs_an_entry_point_the_device_hands_over() {
        unsafe extern "system" fn stub() {}
        // The Windows answer: advertised, but `vkGetDeviceProcAddr` is empty.
        assert!(!device_fault_loader_ready(true, None));
        // A permissive ICD — the Linux loaders hand the pointer over even for
        // an extension the device was not created with — keeps the diagnostic.
        assert!(device_fault_loader_ready(true, Some(stub)));
        // Nothing is loaded for a device that never advertised it.
        assert!(!device_fault_loader_ready(false, Some(stub)));

        // The advertised-but-no-entry-point record the Windows ICD produces.
        let advertised: BTreeMap<String, FieldValue> = DeviceFaultSnapshot::unavailable(true)
            .evidence_fields()
            .into_iter()
            .collect();
        assert_eq!(
            advertised.get("device_fault_extension"),
            Some(&FieldValue::Bool(true))
        );
        assert_eq!(
            advertised.get("device_fault_addresses"),
            Some(&FieldValue::Unsigned(0))
        );
        assert!(!advertised.contains_key("device_fault_description"));
    }

    /// Only a loss carries the raw `vk::Result`: the field means "the driver
    /// answered this", so an ordinary failure must not claim it.
    #[test]
    fn device_loss_error_carries_the_raw_vk_result_and_the_documented_recovery() {
        let lost = ExecutionFailure::vulkan(vk::Result::ERROR_DEVICE_LOST, "queue device lost")
            .into_provider(
                ProviderPhase::Submit,
                ProviderErrorClass::Execute,
                "vulkan-queue-submit",
                CompletionDisposition::SubmittedUnknown { token: None },
            );
        assert_eq!(lost.class, ProviderErrorClass::DeviceLost);
        assert_eq!(lost.slug, "vulkan-queue-submit");
        assert_eq!(lost.phase, ProviderPhase::Submit);
        assert_eq!(lost.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(
            lost.completion,
            CompletionDisposition::DeviceLost { token: None }
        );
        assert_eq!(
            lost.fields.get("vk_result"),
            Some(&FieldValue::Text("VK_ERROR_DEVICE_LOST".to_owned()))
        );
        assert_eq!(
            lost.fields.get("vk_result_raw"),
            Some(&FieldValue::Signed(i64::from(
                vk::Result::ERROR_DEVICE_LOST.as_raw()
            )))
        );
        assert_eq!(lost.detail.as_deref(), Some("queue device lost"));

        let ordinary = ExecutionFailure::vulkan(vk::Result::ERROR_UNKNOWN, "queue unknown")
            .into_provider(
                ProviderPhase::Submit,
                ProviderErrorClass::Execute,
                "vulkan-queue-submit",
                CompletionDisposition::SubmittedUnknown { token: None },
            );
        assert_eq!(ordinary.class, ProviderErrorClass::Execute);
        assert_eq!(ordinary.retryability, Retryability::Unknown);
        assert!(!ordinary.fields.contains_key("vk_result"));
        assert!(!ordinary.fields.contains_key("vk_result_raw"));
    }

    /// A provider over a fresh device, or `None` when the box has none.
    fn device_loss_executor() -> Option<Arc<VulkanExecutor>> {
        match VulkanExecutor::new() {
            Ok(executor) => Some(executor),
            Err(error) => {
                eprintln!("SKIP: no Vulkan device: {error}");
                None
            }
        }
    }

    /// A one-dispatch owner-readback trace over the shared `copy_word` fixture.
    ///
    /// The fixture reads binding 0 and writes binding 1, so a writeback proves
    /// the device executed the submission that carried the fence.
    fn copy_word_trace(
        provider: &VulkanComputeProvider,
        executor: &Arc<VulkanExecutor>,
    ) -> (ComputeTrace, ResourceTableSnapshot) {
        let device = metal_api_core::Device::new(
            Arc::clone(executor) as Arc<dyn metal_api_core::ComputeExecutor>
        );
        let library = device
            .new_library_with_air(include_str!(
                "../../../examples/metal-smoke/shaders/kernel_copy_word.ll"
            ))
            .expect("the fixture library loads");
        let function = library
            .function("copy_word")
            .expect("the fixture entry exists");
        let pipeline = provider
            .compile_pipeline(
                &function,
                SemanticDigest::new("metal-smoke-fixture-v1", b"driver_device_loss".to_vec())
                    .expect("digest"),
            )
            .expect("the fixture pipeline compiles");
        let mut buffers = Vec::new();
        for (index, offset, bytes) in [
            (0_u32, 8_u64, 0x6745_2301_u32.to_le_bytes().to_vec()),
            (1_u32, 16_u64, vec![0_u8; 4]),
        ] {
            let access = pipeline
                .contract
                .buffer_bindings
                .iter()
                .find(|binding| binding.metal_binding == index)
                .expect("fixture binding is reflected")
                .access;
            buffers.push(BufferView {
                view_id: ViewId::new(200 + u64::from(index)),
                metal_binding: index,
                allocation_id: AllocationId::new(100 + u64::from(index)),
                offset,
                length: u64::try_from(bytes.len()).expect("fixture byte length"),
                access,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(bytes),
            });
        }
        let trace = ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: pipeline.device_epoch,
            operation_id: OperationId::new(77),
            pipelines: vec![pipeline.clone()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![TracePass::Compute(ComputePass {
                pipeline: pipeline.pipeline_id,
                buffers,
                textures: Vec::new(),
                dispatch: Dispatch {
                    kind: DispatchKind::ThreadsExact,
                    grid: [1, 1, 1],
                    threads_per_threadgroup: [1, 1, 1],
                },
            })],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        };
        let mut resources = ResourceTableSnapshot::new();
        for view in &trace.passes[0]
            .as_compute()
            .expect("fixture pass is a compute pass")
            .buffers
        {
            resources
                .insert_allocation(AllocationRecord {
                    allocation_id: view.allocation_id,
                    owner_epoch: trace.device_epoch,
                    size: view.offset + view.length + 8,
                })
                .expect("fixture allocation registers");
        }
        (trace, resources)
    }

    fn admitted_trace(
        provider: &VulkanComputeProvider,
        trace: &ComputeTrace,
        resources: &ResourceTableSnapshot,
    ) -> ValidatedComputeTrace {
        provider
            .capabilities()
            .validate_trace(trace.clone(), resources.clone())
            .expect("the fixture trace admits")
    }

    /// A loss reported by `vkQueueSubmit` or `vkWaitForFences`.
    ///
    /// Both boundaries answer the same way: the raw result arrives as a
    /// structured `device_lost`, the core lifecycle reports `DeviceLost`, every
    /// in-flight lease retires, later submissions are refused with the same
    /// typed reason, and only a recreated provider resumes work.
    #[test]
    fn driver_reported_submit_loss_is_terminal_and_keeps_vk_result_evidence() {
        assert_driver_loss_is_terminal(DeviceLossPoint::Submit);
    }

    #[test]
    fn driver_reported_wait_loss_is_terminal_and_keeps_vk_result_evidence() {
        assert_driver_loss_is_terminal(DeviceLossPoint::Wait);
    }

    fn assert_driver_loss_is_terminal(point: DeviceLossPoint) {
        let Some(executor) = device_loss_executor() else {
            return;
        };
        let provider =
            VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider creates");
        let (trace, resources) = copy_word_trace(&provider, &executor);
        let context = Arc::clone(&executor.context);
        assert_eq!(context.health(), ProviderHealth::Usable);

        // The owner mapping an in-flight submission reads. The lifecycle
        // ledger is the teardown authority for it (`research/docs/13` §5): a
        // device loss releases every lease even when no completion was ever
        // observed.
        let lease_id = LeaseId::new(7);
        let lease_token = CompletionToken {
            submission_id: SubmissionId::new(1),
            device_epoch: provider.device_epoch(),
        };
        {
            let mut lifecycle = context.lock_lifecycle();
            lifecycle
                .leases_mut()
                .register(LeaseReservation {
                    lease: BufferLease {
                        lease_id,
                        allocation_id: AllocationId::new(100),
                        owner_epoch: provider.device_epoch(),
                    },
                    offset: 0,
                    length: 4,
                })
                .expect("the in-flight lease registers");
            lifecycle
                .leases_mut()
                .bind(lease_id, lease_token)
                .expect("the submission token binds to the lease");
        }
        assert_eq!(
            context.lock_lifecycle().leases().leased(),
            vec![(lease_id, TerminalLeaseState::Outstanding(1))],
            "the lease is held while the submission is in flight"
        );

        executor.inject_driver_device_loss_for_test(point);
        let error = provider
            .submit(admitted_trace(&provider, &trace, &resources))
            .expect_err("the substituted driver answer refuses the submission");

        // 1. The first error is structured and carries the driver's own answer.
        assert_eq!(error.class, ProviderErrorClass::DeviceLost);
        assert_eq!(error.retryability, Retryability::RetryAfterRecreate);
        let (expected_phase, expected_slug, expected_enqueues) = match point {
            DeviceLossPoint::Submit => (ProviderPhase::Submit, "vulkan-queue-submit", 0),
            DeviceLossPoint::Wait => (ProviderPhase::Wait, "vulkan-wait", 1),
        };
        assert_eq!(error.phase, expected_phase);
        assert_eq!(error.slug, expected_slug);
        assert_eq!(
            error.fields.get("vk_result"),
            Some(&FieldValue::Text("VK_ERROR_DEVICE_LOST".to_owned()))
        );
        assert_eq!(
            error.fields.get("vk_result_raw"),
            Some(&FieldValue::Signed(i64::from(
                vk::Result::ERROR_DEVICE_LOST.as_raw()
            )))
        );
        let CompletionDisposition::DeviceLost { token: Some(token) } = error.completion else {
            panic!("the loss error lost its submission token: {error:?}");
        };
        let observed: usize = context.queue_submission_counts().iter().sum();
        assert_eq!(
            observed, expected_enqueues,
            "a submit-time loss never reaches the driver; a wait-time loss does"
        );
        // The device-fault snapshot is evidence, and its absence is not an
        // error: a device without `VK_EXT_device_fault` answers an empty record.
        let fault = executor
            .last_device_fault()
            .expect("the loss recorded a fault snapshot");
        assert_eq!(
            error.fields.get("device_fault_extension"),
            Some(&FieldValue::Bool(fault.extension_present))
        );
        assert_eq!(
            error.fields.get("device_fault_addresses"),
            Some(&FieldValue::Unsigned(fault.addresses.len() as u64))
        );
        if !fault.extension_present {
            assert!(fault.addresses.is_empty());
            assert!(fault.description.is_none());
        }

        // 2. Health and the lease ledger come from the core lifecycle.
        assert_eq!(context.health(), ProviderHealth::DeviceLost);
        assert_eq!(context.lock_lifecycle().state(), TerminalState::DeviceLost);
        let lifecycle = context.lock_lifecycle();
        assert!(lifecycle.leases().is_device_lost());
        assert_eq!(
            lifecycle.leases().leased(),
            vec![(lease_id, TerminalLeaseState::Released)],
            "a lost device is a teardown guarantee for in-flight leases"
        );
        assert!(lifecycle.leases().release_ready(lease_id));
        drop(lifecycle);

        // 3. Later submissions answer the same typed refusal, and the refusal
        // itself is idempotent: two calls compare equal, token included.
        let first = context.admit().expect_err("a lost context admits no work");
        let second = context.admit().expect_err("a lost context admits no work");
        assert_eq!(first, second);
        assert_eq!(first.slug, "device_lost");
        assert_eq!(first.class, ProviderErrorClass::DeviceLost);
        assert_eq!(first.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(
            first.fields.get("terminal"),
            Some(&FieldValue::Text("device_lost".to_owned()))
        );
        for _ in 0..2 {
            let refused = provider
                .submit(admitted_trace(&provider, &trace, &resources))
                .expect_err("a lost provider admits no work");
            assert_eq!(refused.slug, "device_lost");
            assert_eq!(refused.class, ProviderErrorClass::DeviceLost);
            assert_eq!(
                refused.fields.get("terminal"),
                Some(&FieldValue::Text("device_lost".to_owned()))
            );
        }

        // 4. The lost instance is not reusable, and recreation is the repair.
        assert_eq!(provider.health(), ProviderHealth::DeviceLost);
        assert_eq!(context.abandonment_stats(), (0, 0));
        let Some(recovered_executor) = device_loss_executor() else {
            return;
        };
        let recovered = VulkanComputeProvider::with_executor(Arc::clone(&recovered_executor))
            .expect("the recreated provider");
        let (trace, resources) = copy_word_trace(&recovered, &recovered_executor);
        let submission = recovered
            .submit(admitted_trace(&recovered, &trace, &resources))
            .expect("a recreated provider admits work");
        assert_eq!(recovered.health(), ProviderHealth::Usable);
        let [writeback] = submission.writebacks.as_slice() else {
            panic!("the recovered submission needs exactly one writeback");
        };
        assert_eq!(writeback.bytes, 0x6745_2301_u32.to_le_bytes());
        assert!(matches!(
            submission.completion,
            CompletionDisposition::CompletedVisible { .. }
        ));
        eprintln!(
            "PASS driver_device_loss point={point:?} phase={expected_phase:?} slug={expected_slug} \
             vk_result=VK_ERROR_DEVICE_LOST raw={} enqueues={observed} health={:?} \
             leases=Released fault_extension={} fault_addresses={} token={} recovered=exact",
            vk::Result::ERROR_DEVICE_LOST.as_raw(),
            provider.health(),
            fault.extension_present,
            fault.addresses.len(),
            token.submission_id.get(),
        );
    }

    /// In-place rebuild after a confirmed device loss.
    ///
    /// A `VK_ERROR_DEVICE_LOST` from `vkQueueSubmit` terminates the provider;
    /// [`VulkanComputeProvider::rebuild_after_device_loss`] then swaps in a
    /// brand-new device and advances the epoch, and a re-registration of the
    /// same function resubmits the same logical trace with an exact byte
    /// readback. The old epoch's token stays refused — including one whose
    /// `SubmissionId` is numerically identical to the new submission's — so
    /// the `(epoch, submission)` identity cannot be silently reused across the
    /// rebuild.
    #[test]
    fn rebuild_after_device_loss_readmits_the_same_trace_on_a_fresh_device() {
        let Some(executor) = device_loss_executor() else {
            return;
        };
        let provider =
            VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider creates");
        let (trace, resources) = copy_word_trace(&provider, &executor);
        let old_epoch = provider.device_epoch();
        assert_eq!(provider.health(), ProviderHealth::Usable);

        // 1. The rebuild entry refuses every non-loss state, fail-closed.
        let refused = provider
            .rebuild_after_device_loss()
            .expect_err("a usable provider refuses the rebuild entry");
        assert_eq!(refused.slug, "rebuild_requires_device_loss");
        assert_eq!(
            refused.fields.get("terminal"),
            Some(&FieldValue::Text("usable".to_owned()))
        );

        // 2. The substituted driver answer fails the submission through the
        //    real loss path, and the lifecycle terminates on `DeviceLost`.
        executor.inject_driver_device_loss_for_test(DeviceLossPoint::Submit);
        let error = provider
            .submit(admitted_trace(&provider, &trace, &resources))
            .expect_err("the substituted driver answer refuses the submission");
        assert_eq!(error.class, ProviderErrorClass::DeviceLost);
        let CompletionDisposition::DeviceLost {
            token: Some(old_token),
        } = error.completion
        else {
            panic!("the loss error lost its submission token: {error:?}");
        };
        assert_eq!(provider.health(), ProviderHealth::DeviceLost);
        assert_eq!(
            executor.context.lock_lifecycle().state(),
            TerminalState::DeviceLost
        );

        // 3. Rebuild in place: fresh device, advanced epoch, usable again.
        provider
            .rebuild_after_device_loss()
            .expect("the lost provider rebuilds in place");
        let new_epoch = provider.device_epoch();
        assert!(
            new_epoch.get() > old_epoch.get(),
            "a rebuilt device is a new epoch: {old_epoch:?} -> {new_epoch:?}"
        );
        assert_eq!(provider.health(), ProviderHealth::Usable);
        // The old executor still names the dead device: the provider stopped
        // sharing it the moment the rebuild installed the fresh owner.
        assert_eq!(
            executor.context.lock_lifecycle().state(),
            TerminalState::DeviceLost
        );

        // 4. Old-epoch identities are not re-admitted...
        let refused = provider
            .release_completion(old_token)
            .expect_err("an old-epoch completion token must not be reused");
        assert_eq!(refused.slug, "device_epoch_mismatch");
        assert_eq!(
            refused.fields.get("expected"),
            Some(&FieldValue::Unsigned(new_epoch.get()))
        );
        assert_eq!(
            refused.fields.get("actual"),
            Some(&FieldValue::Unsigned(old_epoch.get()))
        );
        let refused = provider
            .submit(admitted_trace(&provider, &trace, &resources))
            .expect_err("an old-epoch trace must not be re-admitted");
        assert_eq!(refused.slug, "device_epoch_mismatch");

        // 5. ...but the same logical trace, re-registered on the new epoch,
        //    executes to the exact expected bytes.
        let (trace, resources) = copy_word_trace(&provider, &executor);
        let submission = provider
            .submit(admitted_trace(&provider, &trace, &resources))
            .expect("the rebuilt provider admits the same logical trace");
        assert_eq!(provider.health(), ProviderHealth::Usable);
        let [writeback] = submission.writebacks.as_slice() else {
            panic!("the rebuilt submission needs exactly one writeback");
        };
        assert_eq!(writeback.bytes, 0x6745_2301_u32.to_le_bytes());
        let CompletionDisposition::CompletedVisible { token: new_token } = submission.completion
        else {
            panic!(
                "the rebuilt submission completed visibly: {:?}",
                submission.completion
            );
        };
        assert_eq!(new_token.device_epoch, new_epoch);
        // Submission ids stay monotonic across the rebuild, so replaying the
        // new submission's id with the old epoch must still be refused: the
        // `(epoch, submission)` identity is what separates the two devices,
        // not the submission counter.
        let stale_same_id = CompletionToken {
            submission_id: new_token.submission_id,
            device_epoch: old_epoch,
        };
        let refused = provider
            .release_completion(stale_same_id)
            .expect_err("a same-id token from the old epoch must not be reused");
        assert_eq!(refused.slug, "device_epoch_mismatch");
        eprintln!(
            "PASS rebuild_after_device_loss old_epoch={} new_epoch={} \
             old_token_refused=device_epoch_mismatch submission={} readback=exact",
            old_epoch.get(),
            new_epoch.get(),
            new_token.submission_id.get(),
        );
    }
}
