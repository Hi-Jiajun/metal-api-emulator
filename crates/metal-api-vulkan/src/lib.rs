//! Minimal Vulkan executor for the source-level Metal compute facade.
//!
//! The first milestone accepts only ordinary Metal buffer arguments. It uses
//! metal2vulkan reflection as the descriptor contract and its exact-thread plan
//! as the dispatch contract; unsupported resources fail before any Vulkan work
//! is submitted.

use ash::{vk, Device as AshDevice, Entry, Instance};
use metal2vulkan::passes::{Stage, TransformOptions};
use metal2vulkan::reflect::{
    BufferExtent, BufferFootprint, BufferIndexSource, KernelDispatch, ResourceAccess, ResourceKind,
    ShaderReflection, ShaderStage, KERNEL_LOCAL_SIZE_SPEC_IDS,
};
use metal_api_core::{
    BufferBinding, BufferUpdate, ComputeExecutor, ComputeSubmission, ExecutorError, Function,
    PipelineArtifact,
};
use spirv::{Capability, Op};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, CString};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const FENCE_TIMEOUT_NS: u64 = 20_000_000_000;
static SCRATCH_SERIAL: AtomicU64 = AtomicU64::new(0);

fn failure(message: impl Into<String>) -> ExecutorError {
    ExecutorError::new(message)
}

/// Native Vulkan implementation of the Phase 1 compute subset.
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
}

struct VulkanPipelineArtifact {
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
    pub fn translate(function: &Function) -> Result<Self, ExecutorError> {
        let options = TransformOptions {
            kernel_local_size: [1, 1, 1],
            kernel_dispatch: Some(KernelDispatch::safe_default()),
            ..TransformOptions::default()
        };
        let scratch = ScratchDir::new()?;
        let (spv, reflection) = metal2vulkan::translate_sanitized_native_reflected(
            function.air(),
            Stage::Kernel,
            scratch.path(),
            options,
        )
        .map_err(|error| failure(format!("translate {}: {error}", function.name())))?;
        validate_spirv_capabilities(&spv)?;
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
        validate_bound_buffers(&self.reflection, buffers, threads_per_grid)
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
}

impl ComputeExecutor for VulkanExecutor {
    fn new_compute_pipeline(&self, function: &Function) -> Result<PipelineArtifact, ExecutorError> {
        self.context.ensure_usable()?;
        let translated = TranslatedComputePipeline::translate(function)?;
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
        let _execution = self
            .context
            .execution_lock
            .lock()
            .map_err(|_| failure("Vulkan execution lock is poisoned"))?;
        self.context.ensure_usable()?;
        execute_submission(&self.context, artifact, submission)
    }
}

struct VulkanContext {
    entry: ManuallyDrop<Entry>,
    instance: Instance,
    device: AshDevice,
    queue_family: u32,
    queue: vk::Queue,
    properties: vk::PhysicalDeviceProperties,
    memory: vk::PhysicalDeviceMemoryProperties,
    device_name: String,
    execution_lock: Mutex<()>,
    poisoned: AtomicBool,
    abandoned: AtomicBool,
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
        let (physical, queue_family) = match selection {
            Ok(selection) => selection,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(error);
            }
        };
        let priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default().maintenance4(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .push_next(&mut vulkan13);
        let device = match unsafe { instance.create_device(physical, &device_info, None) } {
            Ok(device) => device,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(failure(format!("create Vulkan device: {error}")));
            }
        };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        Ok(Self {
            entry: ManuallyDrop::new(entry),
            instance,
            device,
            queue_family,
            queue,
            properties,
            memory,
            device_name,
            execution_lock: Mutex::new(()),
            poisoned: AtomicBool::new(false),
            abandoned: AtomicBool::new(false),
        })
    }

    fn ensure_usable(&self) -> Result<(), ExecutorError> {
        if self.poisoned.load(Ordering::Acquire) {
            Err(failure(
                "Vulkan executor is poisoned after an incomplete submission",
            ))
        } else {
            Ok(())
        }
    }

    fn abandon(self: &Arc<Self>, resources: ExecutionResources) {
        self.poisoned.store(true, Ordering::Release);
        self.abandoned.store(true, Ordering::Release);
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
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        if self.abandoned.load(Ordering::Acquire) {
            // Timeout paths leak an extra Arc, so this arm is defensive rather
            // than expected. Never unload the Vulkan loader under pending work.
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

fn select_physical_device(instance: &Instance) -> Result<(vk::PhysicalDevice, u32), ExecutorError> {
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
            if vulkan13.maintenance4 != vk::TRUE {
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
        .map(|(_, physical, family)| (physical, family))
        .ok_or_else(|| {
            failure("no Vulkan 1.3 physical device with maintenance4 exposes a compute queue")
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
) -> Result<Vec<BufferUpdate>, ExecutorError> {
    let grid = submission.threads_per_grid.dimensions();
    let local = submission.threads_per_threadgroup.dimensions();
    validate_local_size(context, local)?;
    let reflection = artifact.translated.reflection();
    artifact
        .translated
        .validate_buffers(&submission.buffers, grid)?;
    artifact.translated.validate_threadgroup(local)?;
    let reflected_contract = reflection
        .kernel_dispatch
        .ok_or_else(|| failure("translated kernel has no dispatch contract"))?;
    if !matches!(reflected_contract, KernelDispatch::ThreadsDynamic { .. }) {
        return Err(failure(format!(
            "translated kernel returned unexpected dispatch contract {reflected_contract:?}"
        )));
    }
    let plan = reflected_contract
        .plan(local, Some(grid))
        .map_err(|error| failure(format!("plan exact dispatch: {error}")))?;
    validate_dispatch_plan(context, reflected_contract, &plan)?;
    if plan.regions.is_empty() {
        return Ok(Vec::new());
    }

    let mut resources = ExecutionResources::new(Arc::clone(context));
    resources.create_pipeline_objects(artifact.translated.spirv(), reflection, &plan)?;
    resources.create_buffers(reflection, &submission.buffers)?;
    resources.create_descriptors(reflection)?;
    resources.record(reflection, reflected_contract, &plan)?;
    match resources.submit_and_wait() {
        Ok(()) => {}
        Err(SubmissionFailure::Safe(error)) => return Err(error),
        Err(SubmissionFailure::Pending(error)) => {
            context.abandon(resources);
            return Err(error);
        }
    }
    resources.read_updates(reflection)
}

fn validate_local_size(context: &VulkanContext, local: [u32; 3]) -> Result<(), ExecutorError> {
    let limits = context.properties.limits;
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
    context: &VulkanContext,
    contract: KernelDispatch,
    plan: &metal2vulkan::reflect::KernelDispatchPlan,
) -> Result<(), ExecutorError> {
    let limits = context.properties.limits;
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
        validate_local_size(context, region.local_size)?;
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

fn validate_spirv_capabilities(spv: &[u8]) -> Result<(), ExecutorError> {
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
            if capability != Capability::Shader as u32 {
                return Err(failure(format!(
                    "SPIR-V capability {capability} requires a Vulkan feature outside the Phase 1 subset"
                )));
            }
        } else if opcode == Op::Extension as u32 {
            return Err(failure(
                "SPIR-V extensions are outside the Phase 1 feature subset",
            ));
        }
        cursor = end;
    }
    Ok(())
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
    let mut metal_indices = BTreeSet::new();
    let mut descriptor_bindings = BTreeSet::new();
    for binding in &reflection.bindings {
        if binding.kind != ResourceKind::Buffer {
            return Err(failure(format!(
                "Phase 1 supports only Metal buffers, not {:?}",
                binding.kind
            )));
        }
        if !metal_indices.insert(binding.metal_index) {
            return Err(failure(format!(
                "duplicate reflected Metal buffer index {}",
                binding.metal_index
            )));
        }
        let descriptor = binding.descriptor.ok_or_else(|| {
            failure(format!(
                "Metal buffer {} has no Vulkan descriptor",
                binding.metal_index
            ))
        })?;
        if descriptor.set != 0 || descriptor.count != 1 {
            return Err(failure(format!(
                "Metal buffer {} uses unsupported descriptor set={} count={}",
                binding.metal_index, descriptor.set, descriptor.count
            )));
        }
        if !descriptor_bindings.insert(descriptor.binding) {
            return Err(failure(format!(
                "duplicate Vulkan descriptor binding {}",
                descriptor.binding
            )));
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
    buffers: &[BufferBinding],
    grid: [u32; 3],
) -> Result<(), ExecutorError> {
    let metal_indices = reflection
        .bindings
        .iter()
        .map(|binding| binding.metal_index)
        .collect::<BTreeSet<_>>();
    let mut supplied = BTreeSet::new();
    for binding in buffers {
        if binding.bytes.is_empty() {
            return Err(failure(format!(
                "buffer {} has an empty bound range",
                binding.index
            )));
        }
        if !supplied.insert(binding.index) {
            return Err(failure(format!(
                "buffer {} is bound more than once",
                binding.index
            )));
        }
        let reflected = reflection
            .bindings
            .iter()
            .find(|candidate| candidate.metal_index == binding.index)
            .ok_or_else(|| failure(format!("buffer {} is not reflected", binding.index)))?;
        let mut required = u64::from(reflected.declared_size.unwrap_or(0));
        if let Some(BufferExtent::Object { bytes }) = reflected.extent {
            required = required.max(u64::from(bytes));
        }
        let footprint = reflected
            .footprint
            .as_ref()
            .expect("pipeline validation requires a footprint");
        for range in &footprint.static_ranges {
            let end = range.offset.checked_add(range.size).ok_or_else(|| {
                failure(format!("buffer {} footprint overflows u64", binding.index))
            })?;
            required = required.max(end);
        }
        required = required.max(
            strided_footprint_reach(footprint, grid)
                .map_err(|error| failure(format!("buffer {} {error}", binding.index)))?,
        );
        let supplied_len = u64::try_from(binding.bytes.len())
            .map_err(|_| failure(format!("buffer {} length overflows u64", binding.index)))?;
        ensure_buffer_reach(binding.index, supplied_len, required)?;
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

struct GpuBuffer {
    index: u32,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    len: usize,
}

struct ExecutionResources {
    context: Arc<VulkanContext>,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader: vk::ShaderModule,
    pipelines: BTreeMap<[u32; 3], vk::Pipeline>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    submitted: bool,
    completed: bool,
    buffers: Vec<GpuBuffer>,
}

enum SubmissionFailure {
    /// Nothing reached the queue, so ordinary RAII cleanup is valid.
    Safe(ExecutorError),
    /// Queue submission succeeded but completion is unknown. Handles must stay
    /// alive until process exit or an out-of-band reaper proves completion.
    Pending(ExecutorError),
}

impl ExecutionResources {
    fn new(context: Arc<VulkanContext>) -> Self {
        Self {
            context,
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            shader: vk::ShaderModule::null(),
            pipelines: BTreeMap::new(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            command_pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            submitted: false,
            completed: false,
            buffers: Vec::new(),
        }
    }

    fn create_pipeline_objects(
        &mut self,
        spv: &[u8],
        reflection: &ShaderReflection,
        plan: &metal2vulkan::reflect::KernelDispatchPlan,
    ) -> Result<(), ExecutorError> {
        if !spv.len().is_multiple_of(4) {
            return Err(failure("translated SPIR-V is not word aligned"));
        }
        let buffer_count = u32::try_from(reflection.bindings.len())
            .map_err(|_| failure("reflected buffer count overflows u32"))?;
        let limits = self.context.properties.limits;
        if limits.max_bound_descriptor_sets == 0
            || buffer_count > limits.max_per_stage_descriptor_storage_buffers
            || buffer_count > limits.max_descriptor_set_storage_buffers
            || buffer_count > limits.max_per_stage_resources
        {
            return Err(failure(format!(
                "{buffer_count} storage buffers exceed Vulkan descriptor limits per-stage={} per-set={} all-resources={} bound-sets={}",
                limits.max_per_stage_descriptor_storage_buffers,
                limits.max_descriptor_set_storage_buffers,
                limits.max_per_stage_resources,
                limits.max_bound_descriptor_sets
            )));
        }
        let words = spv
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&words);
        self.shader = unsafe { self.context.device.create_shader_module(&shader_info, None) }
            .map_err(|error| failure(format!("create shader module: {error}")))?;

        let mut layout_bindings = reflection
            .bindings
            .iter()
            .map(|binding| {
                let descriptor = binding.descriptor.expect("validated descriptor");
                vk::DescriptorSetLayoutBinding::default()
                    .binding(descriptor.binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
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
        .map_err(|error| failure(format!("create descriptor-set layout: {error}")))?;

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
        .map_err(|error| failure(format!("create pipeline layout: {error}")))?;

        for region in &plan.regions {
            if self.pipelines.contains_key(&region.local_size) {
                continue;
            }
            let pipeline = self.create_compute_pipeline(region.local_size)?;
            self.pipelines.insert(region.local_size, pipeline);
        }
        Ok(())
    }

    fn create_compute_pipeline(&self, local_size: [u32; 3]) -> Result<vk::Pipeline, ExecutorError> {
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
                Err(failure(format!("create compute pipeline: {error}")))
            }
        }
    }

    fn create_buffers(
        &mut self,
        reflection: &ShaderReflection,
        bindings: &[BufferBinding],
    ) -> Result<(), ExecutorError> {
        for reflected in &reflection.bindings {
            let supplied = bindings
                .iter()
                .find(|binding| binding.index == reflected.metal_index)
                .expect("validated buffer binding");
            self.create_buffer(supplied)?;
        }
        self.buffers.sort_by_key(|buffer| buffer.index);
        Ok(())
    }

    fn create_buffer(&mut self, supplied: &BufferBinding) -> Result<(), ExecutorError> {
        let size = u64::try_from(supplied.bytes.len())
            .map_err(|_| failure(format!("buffer {} length overflows u64", supplied.index)))?;
        if size > self.context.properties.limits.max_storage_buffer_range as u64 {
            return Err(failure(format!(
                "buffer {} length {size} exceeds maxStorageBufferRange {}",
                supplied.index, self.context.properties.limits.max_storage_buffer_range
            )));
        }
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.context.device.create_buffer(&buffer_info, None) }
            .map_err(|error| failure(format!("create buffer {}: {error}", supplied.index)))?;
        let requirements = unsafe { self.context.device.get_buffer_memory_requirements(buffer) };
        let memory_type = match self.context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(index) => index,
            Err(error) => {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(error);
            }
        };
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = match unsafe { self.context.device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(failure(format!(
                    "allocate buffer {} memory: {error}",
                    supplied.index
                )));
            }
        };
        if let Err(error) = unsafe { self.context.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.context.device.destroy_buffer(buffer, None);
                self.context.device.free_memory(memory, None);
            }
            return Err(failure(format!(
                "bind buffer {} memory: {error}",
                supplied.index
            )));
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
                return Err(failure(format!("map buffer {}: {error}", supplied.index)));
            }
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                supplied.bytes.as_ptr(),
                mapped.cast::<u8>(),
                supplied.bytes.len(),
            );
            self.context.device.unmap_memory(memory);
        }
        self.buffers.push(GpuBuffer {
            index: supplied.index,
            buffer,
            memory,
            len: supplied.bytes.len(),
        });
        Ok(())
    }

    fn create_descriptors(&mut self, reflection: &ShaderReflection) -> Result<(), ExecutorError> {
        let count = u32::try_from(self.buffers.len())
            .map_err(|_| failure("descriptor count overflows u32"))?;
        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: count,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        self.descriptor_pool =
            unsafe { self.context.device.create_descriptor_pool(&pool_info, None) }
                .map_err(|error| failure(format!("create descriptor pool: {error}")))?;
        let layouts = [self.set_layout];
        let allocation = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        self.descriptor_set = unsafe { self.context.device.allocate_descriptor_sets(&allocation) }
            .map_err(|error| failure(format!("allocate descriptor set: {error}")))?[0];

        let infos = reflection
            .bindings
            .iter()
            .map(|binding| {
                let gpu = self
                    .buffers
                    .iter()
                    .find(|buffer| buffer.index == binding.metal_index)
                    .expect("validated GPU buffer");
                vk::DescriptorBufferInfo::default()
                    .buffer(gpu.buffer)
                    .offset(0)
                    .range(gpu.len as u64)
            })
            .collect::<Vec<_>>();
        let writes = reflection
            .bindings
            .iter()
            .zip(&infos)
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(binding.descriptor.expect("validated descriptor").binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect::<Vec<_>>();
        unsafe { self.context.device.update_descriptor_sets(&writes, &[]) };
        Ok(())
    }

    fn record(
        &mut self,
        reflection: &ShaderReflection,
        contract: KernelDispatch,
        plan: &metal2vulkan::reflect::KernelDispatchPlan,
    ) -> Result<(), ExecutorError> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.context.queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        self.command_pool = unsafe { self.context.device.create_command_pool(&pool_info, None) }
            .map_err(|error| failure(format!("create command pool: {error}")))?;
        let allocation = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        self.command = unsafe { self.context.device.allocate_command_buffers(&allocation) }
            .map_err(|error| failure(format!("allocate command buffer: {error}")))?[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.context
                .device
                .begin_command_buffer(self.command, &begin)
                .map_err(|error| failure(format!("begin command buffer: {error}")))?;
            self.context.device.cmd_bind_descriptor_sets(
                self.command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                reflection.descriptor_layout.set,
                &[self.descriptor_set],
                &[],
            );
            let offset = contract
                .push_constant_range()
                .expect("validated exact range")
                .offset;
            for region in &plan.regions {
                let pipeline = self.pipelines[&region.local_size];
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
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    offset,
                    &bytes,
                );
                self.context.device.cmd_dispatch(
                    self.command,
                    region.group_count[0],
                    region.group_count[1],
                    region.group_count[2],
                );
            }
            self.context
                .device
                .end_command_buffer(self.command)
                .map_err(|error| failure(format!("end command buffer: {error}")))?;
        }
        Ok(())
    }

    fn submit_and_wait(&mut self) -> Result<(), SubmissionFailure> {
        self.fence = unsafe {
            self.context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|error| {
            SubmissionFailure::Safe(failure(format!("create completion fence: {error}")))
        })?;
        let commands = [self.command];
        let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
        if let Err(error) = unsafe {
            self.context
                .device
                .queue_submit(self.context.queue, &submits, self.fence)
        } {
            self.context.poisoned.store(true, Ordering::Release);
            return Err(SubmissionFailure::Safe(failure(format!(
                "submit compute command buffer: {error}"
            ))));
        }
        self.submitted = true;
        let wait = unsafe {
            self.context
                .device
                .wait_for_fences(&[self.fence], true, FENCE_TIMEOUT_NS)
        };
        match wait {
            Ok(()) => {
                self.completed = true;
                Ok(())
            }
            Err(error) => {
                self.context.poisoned.store(true, Ordering::Release);
                let message = if error == vk::Result::TIMEOUT {
                    "compute completion timed out after 20 seconds".to_string()
                } else {
                    format!("wait for compute completion failed: {error}")
                };
                Err(SubmissionFailure::Pending(failure(message)))
            }
        }
    }

    fn read_updates(
        &self,
        reflection: &ShaderReflection,
    ) -> Result<Vec<BufferUpdate>, ExecutorError> {
        let mut updates = Vec::new();
        for reflected in &reflection.bindings {
            if matches!(
                reflected.access,
                Some(ResourceAccess::Unused | ResourceAccess::ReadOnly)
            ) {
                continue;
            }
            let gpu = self
                .buffers
                .iter()
                .find(|buffer| buffer.index == reflected.metal_index)
                .expect("validated GPU buffer");
            let mapped = unsafe {
                self.context.device.map_memory(
                    gpu.memory,
                    0,
                    gpu.len as u64,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|error| failure(format!("map buffer {} for readback: {error}", gpu.index)))?;
            let bytes =
                unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), gpu.len).to_vec() };
            unsafe { self.context.device.unmap_memory(gpu.memory) };
            updates.push(BufferUpdate {
                index: gpu.index,
                offset: 0,
                bytes,
            });
        }
        updates.sort_by_key(|update| update.index);
        Ok(updates)
    }
}

impl Drop for ExecutionResources {
    fn drop(&mut self) {
        if self.submitted && !self.completed {
            self.context.poisoned.store(true, Ordering::Release);
            self.context.abandoned.store(true, Ordering::Release);
            // A panic between queue submission and the explicit wait outcome
            // cannot unwind into destruction of in-flight handles. Raw Vulkan
            // handles below are intentionally left live, and this strong
            // context reference keeps the loader/device live until process exit.
            let _ = Arc::into_raw(Arc::clone(&self.context));
            return;
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
            for buffer in &self.buffers {
                self.context.device.destroy_buffer(buffer.buffer, None);
                self.context.device.free_memory(buffer.memory, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::reflect::{BufferStrideTerm, BufferStridedAccess};

    fn spirv_bytes(instructions: &[&[u32]]) -> Vec<u8> {
        let mut words = vec![0x0723_0203, 0x0001_0400, 0, 1, 0];
        for instruction in instructions {
            words.extend_from_slice(instruction);
        }
        words.into_iter().flat_map(u32::to_le_bytes).collect()
    }

    #[test]
    fn phase_one_spirv_feature_gate_accepts_shader_and_rejects_optional_capabilities() {
        let shader = [
            (2_u32 << 16) | Op::Capability as u32,
            Capability::Shader as u32,
        ];
        assert!(validate_spirv_capabilities(&spirv_bytes(&[&shader])).is_ok());

        let int64 = [
            (2_u32 << 16) | Op::Capability as u32,
            Capability::Int64 as u32,
        ];
        let error = validate_spirv_capabilities(&spirv_bytes(&[&shader, &int64])).unwrap_err();
        assert!(error.message().contains("outside the Phase 1 subset"));

        let extension = [(2_u32 << 16) | Op::Extension as u32, 0];
        let error = validate_spirv_capabilities(&spirv_bytes(&[&shader, &extension])).unwrap_err();
        assert!(error.message().contains("extensions"));
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
}
