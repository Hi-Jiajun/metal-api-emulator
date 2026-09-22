//! The device's own answers, as a parseable line block in the boot log.
//
//! # Why this exists
//
// Capability truth comes from three layers: what the driver declares (the
// Vulkan queries), what our implementation's window states, and the measured
// quirk. The first layer was scattered across capability frames, census
// summaries and one-off probe rounds, so "this (GPU, driver) is like *this*"
// was not a file anyone could archive, diff or reuse. It is now: the same
// canonical table the standalone probe reads
// (`tools/device-profile/probe/`, whose `canonical` module this one mirrors)
// is available from a real boot.
//
// # The switch
//
// `METAL_API_VULKAN_DEVICE_PROFILE` — **off unless a truthy word turns it on**
// (`1`, `on`, `ON`, `true`, `yes`; anything else, including unset, is off).
// Off is one relaxed load in the device's own construction and nothing else:
// no clock read, no allocation, no driver call. On emits one `DEVICE_PROFILE`
// line block to the process's stderr — the boot log a round already captures —
// which `tools/device-profile/from-log.py` turns back into the same JSON the
// probe writes, so a boot's own reading can be diffed against the probe's.
//
// # The line shape
//
// One line is a kind and its `k=v` clauses:
//
// ```text
// DEVICE_PROFILE <kind> <key>=<value> <key>=<value> …
// ```
//
// A value that is empty or holds a space is double-quoted (`deviceName` does),
// and a `"`, newline or carriage return inside a value is replaced by `_`, so
// the block survives the log reader that splits lines on spaces without
// inventing an escape grammar. The kinds are `begin`, `identity`, `limit`,
// `feature`, `format`, `shape`, `carrier`, `queue_family`, `memory_heap`,
// `memory_type` and `end`; `from-log.py` is the reader and names them all.
//
// # What it costs when it is on
//
// The shape table really creates and destroys one
// [`vk::ImageCreateInfo`] per cell — the RTF round is why a feature word is
// not enough to answer "can this device make the object"
// (`evidence/rtx-texture-format-*-2026-09-21/`). That is a few hundred
// create/destroy pairs at device startup, which is why the switch is off by
// default and why a *counting* or *timing* round must not have it on: the
// objects are real, and a census that counts device objects would count them.
// Nothing is left behind: every image is destroyed in the same call, and the
// carrier probe allocates and frees its own memory.

use ash::vk;
use std::fmt::Write as _;
use std::sync::OnceLock;

/// The schema tag both this block and the probe's JSON carry.
pub(crate) const SCHEMA: &str = "metal-api-device-profile/1";

/// Every line this module writes starts with this word.
const PREFIX: &str = "DEVICE_PROFILE";

/// Whether the profile dump is armed, read once from the process environment.
///
/// Off is the default and means one relaxed load: nothing below runs, no
/// driver call is made, and the device the product builds is byte-for-byte the
/// device it built before this module existed.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_VULKAN_DEVICE_PROFILE")
                .ok()
                .as_deref(),
        )
    })
}

/// The switch's own reading, spelled exactly as the crate's other
/// default-off instruments spell it: a truthy word arms, everything else —
/// including unset — leaves the dump off.
fn parse_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1" | "on" | "ON" | "true" | "yes")
    )
}

/// Write the profile block once per process, if the switch is armed.
///
/// Once per *process*, not per device: a rebuild after a device loss builds a
/// second context, and a second block would say the same thing about the same
/// driver.
pub(crate) fn dump_once(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    device: &ash::Device,
    properties: &vk::PhysicalDeviceProperties,
    memory: &vk::PhysicalDeviceMemoryProperties,
) {
    if !enabled() {
        return;
    }
    static EMITTED: OnceLock<()> = OnceLock::new();
    if EMITTED.set(()).is_err() {
        return;
    }
    for line in lines(instance, physical, device, properties, memory) {
        eprintln!("{line}");
    }
}

fn lines(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    device: &ash::Device,
    properties: &vk::PhysicalDeviceProperties,
    memory: &vk::PhysicalDeviceMemoryProperties,
) -> Vec<String> {
    let mut out = Vec::new();
    let api_version = properties.api_version;
    let has_1_1 = api_version >= vk::API_VERSION_1_1;
    let has_1_2 = api_version >= vk::API_VERSION_1_2;

    out.push(
        Clause::new("begin")
            .num("schema_version", 1)
            .text("schema", SCHEMA)
            .num("observed_at_epoch_seconds", epoch_seconds())
            .text("source", "product")
            .text("os", std::env::consts::OS)
            .text("arch", std::env::consts::ARCH)
            .text(
                "icd",
                &std::env::var("VK_ICD_FILENAMES").unwrap_or_else(|_| "-".to_owned()),
            )
            .done(),
    );

    // `VK_KHR_driver_properties` and the UUIDs sit in the `pNext` chain of the
    // properties query: `driverName` is the only place the driver calls itself
    // what it is, and the device UUID is what a profile of the same card
    // across two driver installs holds on to.
    let mut ids = vk::PhysicalDeviceIDProperties::default();
    let mut driver = vk::PhysicalDeviceDriverProperties::default();
    if has_1_1 {
        let mut query = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut ids)
            .push_next(&mut driver);
        unsafe { instance.get_physical_device_properties2(physical, &mut query) };
    }
    let driver_info = if has_1_1 {
        c_chars(&driver.driver_info)
    } else {
        String::new()
    };
    let vulkan_driver_version = version_string(properties.driver_version);
    let named_driver_version = if driver_info.is_empty() {
        vulkan_driver_version.clone()
    } else {
        driver_info.clone()
    };
    out.push(
        Clause::new("identity")
            .text("device_name", &c_chars(&properties.device_name))
            .num("vendor_id", u64::from(properties.vendor_id))
            .num("device_id", u64::from(properties.device_id))
            .num("device_type", raw32(properties.device_type.as_raw()))
            .text("device_type_name", device_type_name(properties.device_type))
            .text("api_version", &version_string(api_version))
            .num("api_version_raw", u64::from(api_version))
            .text("driver_version", &named_driver_version)
            .text("driver_version_vulkan", &vulkan_driver_version)
            .num("driver_version_raw", u64::from(properties.driver_version))
            .num(
                "driver_id",
                raw32(if has_1_1 {
                    driver.driver_id.as_raw()
                } else {
                    0
                }),
            )
            .text(
                "driver_id_name",
                driver_id_name(if has_1_1 {
                    driver.driver_id.as_raw()
                } else {
                    0
                }),
            )
            .text(
                "driver_name",
                &if has_1_1 {
                    c_chars(&driver.driver_name)
                } else {
                    String::new()
                },
            )
            .text("driver_info", &driver_info)
            .text(
                "driver_conformance_version",
                &if has_1_1 {
                    format!(
                        "{}.{}.{}.{}",
                        driver.conformance_version.major,
                        driver.conformance_version.minor,
                        driver.conformance_version.subminor,
                        driver.conformance_version.patch
                    )
                } else {
                    String::new()
                },
            )
            .text("device_uuid", &hex(&ids.device_uuid))
            .text("driver_uuid", &hex(&ids.driver_uuid))
            .text("device_luid", &hex(&ids.device_luid))
            .bool(
                "device_luid_valid",
                has_1_1 && ids.device_luid_valid == vk::TRUE,
            )
            .num(
                "device_node_mask",
                if has_1_1 {
                    u64::from(ids.device_node_mask)
                } else {
                    0
                },
            )
            .text("pipeline_cache_uuid", &hex(&properties.pipeline_cache_uuid))
            .done(),
    );

    let extensions =
        unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    let robustness2 = extensions.iter().any(|extension| {
        extension
            .extension_name_as_c_str()
            .map(|name| name.to_bytes() == b"VK_EXT_robustness2")
            .unwrap_or(false)
    });
    features(instance, physical, has_1_2, robustness2, &mut out);
    limits(&properties.limits, &mut out);

    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    for (index, family) in families.iter().enumerate() {
        out.push(
            Clause::new("queue_family")
                .num("index", index as u64)
                .num("queue_count", u64::from(family.queue_count))
                .text("flags", &queue_flag_names(family.queue_flags))
                .num("flags_raw", u64::from(family.queue_flags.as_raw()))
                .num(
                    "timestamp_valid_bits",
                    u64::from(family.timestamp_valid_bits),
                )
                .text(
                    "granularity",
                    &format!(
                        "{}x{}x{}",
                        family.min_image_transfer_granularity.width,
                        family.min_image_transfer_granularity.height,
                        family.min_image_transfer_granularity.depth
                    ),
                )
                .done(),
        );
    }

    let heap_count = usize::try_from(memory.memory_heap_count).unwrap_or(0);
    for (index, heap) in memory.memory_heaps[..heap_count.min(memory.memory_heaps.len())]
        .iter()
        .enumerate()
    {
        out.push(
            Clause::new("memory_heap")
                .num("index", index as u64)
                .num("size_bytes", heap.size)
                .text("flags", &memory_heap_flag_names(heap.flags))
                .num("flags_raw", u64::from(heap.flags.as_raw()))
                .done(),
        );
    }
    let type_count = usize::try_from(memory.memory_type_count).unwrap_or(0);
    for (index, memory_type) in memory.memory_types[..type_count.min(memory.memory_types.len())]
        .iter()
        .enumerate()
    {
        out.push(
            Clause::new("memory_type")
                .num("index", index as u64)
                .num("heap_index", u64::from(memory_type.heap_index))
                .text(
                    "property_flags",
                    &memory_property_flag_names(memory_type.property_flags),
                )
                .num(
                    "property_flags_raw",
                    u64::from(memory_type.property_flags.as_raw()),
                )
                .done(),
        );
    }

    for format in canonical::formats() {
        let words =
            unsafe { instance.get_physical_device_format_properties(physical, format.format) };
        out.push(
            Clause::new("format")
                .text("name", format.name)
                .num("raw", raw32(format.format.as_raw()))
                .text(
                    "linear_features",
                    &format!("{:#x}", words.linear_tiling_features.as_raw()),
                )
                .text(
                    "linear_feature_names",
                    &format_flag_names(words.linear_tiling_features),
                )
                .text(
                    "optimal_features",
                    &format!("{:#x}", words.optimal_tiling_features.as_raw()),
                )
                .text(
                    "optimal_feature_names",
                    &format_flag_names(words.optimal_tiling_features),
                )
                .text(
                    "buffer_features",
                    &format!("{:#x}", words.buffer_features.as_raw()),
                )
                .done(),
        );
    }

    for shape in canonical::shapes() {
        let answer = unsafe {
            instance.get_physical_device_image_format_properties(
                physical,
                shape.format,
                shape.image_type,
                shape.tiling,
                shape.usage,
                vk::ImageCreateFlags::empty(),
            )
        };
        let mut row = Clause::new("shape")
            .text("format", shape.format_name)
            .text("type", shape.type_name)
            .text("extent", shape.extent_name)
            .text("tiling", shape.tiling_name)
            .text("usage", shape.usage_name)
            .num("usage_raw", u64::from(shape.usage.as_raw()));
        row = match &answer {
            Ok(properties) => row
                .text("ifp", "OK")
                .text("ifp_max_extent", &extent_name(properties.max_extent))
                .num("ifp_max_mip_levels", u64::from(properties.max_mip_levels))
                .num(
                    "ifp_max_array_layers",
                    u64::from(properties.max_array_layers),
                )
                .text(
                    "ifp_sample_counts",
                    &format!("{:#x}", properties.sample_counts.as_raw()),
                )
                .num("ifp_max_resource_size", properties.max_resource_size),
            Err(error) => row.text("ifp", &format!("{error:?}")),
        };
        let info = image_create_info(&shape);
        let mut carrier_line = None;
        row = match unsafe { device.create_image(&info, None) } {
            Ok(image) => {
                unsafe { device.destroy_image(image, None) };
                row.text("create", "OK")
            }
            Err(error) => {
                let row = row.text("create", &format!("{error:?}"));
                if shape.tiling == vk::ImageTiling::LINEAR {
                    carrier_line = Some(carrier(instance, device, physical, &shape).done());
                }
                row
            }
        };
        out.push(row.done());
        if let Some(line) = carrier_line {
            out.push(line);
        }
    }

    out.push(
        Clause::new("end")
            .num("shapes", canonical::shapes().len() as u64)
            .done(),
    );
    out
}

/// The canonical feature set, read through one `VkPhysicalDeviceFeatures2`
/// chain: the core 1.0 block, the 1.1/1.2 structs that carry the 16-bit and
/// 8-bit shader features, and `VK_EXT_robustness2` when the device has it.
fn features(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    has_1_2: bool,
    robustness2: bool,
    out: &mut Vec<String>,
) {
    let mut query = vk::PhysicalDeviceFeatures2::default();
    let mut vulkan11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut robust = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
    if has_1_2 {
        query = query.push_next(&mut vulkan12).push_next(&mut vulkan11);
    }
    if robustness2 {
        query = query.push_next(&mut robust);
    }
    let head = unsafe {
        instance.get_physical_device_features2(physical, &mut query);
        query.features
    };
    let mut feature = |name: &str, value: bool| {
        out.push(
            Clause::new("feature")
                .text("name", name)
                .bool("value", value)
                .done(),
        )
    };
    feature("robustBufferAccess", head.robust_buffer_access == vk::TRUE);
    feature(
        "fullDrawIndexUint32",
        head.full_draw_index_uint32 == vk::TRUE,
    );
    feature("imageCubeArray", head.image_cube_array == vk::TRUE);
    feature("independentBlend", head.independent_blend == vk::TRUE);
    feature("geometryShader", head.geometry_shader == vk::TRUE);
    feature("tessellationShader", head.tessellation_shader == vk::TRUE);
    feature("sampleRateShading", head.sample_rate_shading == vk::TRUE);
    feature("dualSrcBlend", head.dual_src_blend == vk::TRUE);
    feature("logicOp", head.logic_op == vk::TRUE);
    feature("multiDrawIndirect", head.multi_draw_indirect == vk::TRUE);
    feature(
        "drawIndirectFirstInstance",
        head.draw_indirect_first_instance == vk::TRUE,
    );
    feature("depthClamp", head.depth_clamp == vk::TRUE);
    feature("depthBiasClamp", head.depth_bias_clamp == vk::TRUE);
    feature("fillModeNonSolid", head.fill_mode_non_solid == vk::TRUE);
    feature("wideLines", head.wide_lines == vk::TRUE);
    feature("largePoints", head.large_points == vk::TRUE);
    feature("alphaToOne", head.alpha_to_one == vk::TRUE);
    feature("multiViewport", head.multi_viewport == vk::TRUE);
    feature("samplerAnisotropy", head.sampler_anisotropy == vk::TRUE);
    feature(
        "textureCompressionBC",
        head.texture_compression_bc == vk::TRUE,
    );
    feature(
        "occlusionQueryPrecise",
        head.occlusion_query_precise == vk::TRUE,
    );
    feature(
        "pipelineStatisticsQuery",
        head.pipeline_statistics_query == vk::TRUE,
    );
    feature(
        "vertexPipelineStoresAndAtomics",
        head.vertex_pipeline_stores_and_atomics == vk::TRUE,
    );
    feature(
        "fragmentStoresAndAtomics",
        head.fragment_stores_and_atomics == vk::TRUE,
    );
    feature(
        "shaderImageGatherExtended",
        head.shader_image_gather_extended == vk::TRUE,
    );
    feature(
        "shaderStorageImageExtendedFormats",
        head.shader_storage_image_extended_formats == vk::TRUE,
    );
    feature(
        "shaderStorageImageMultisample",
        head.shader_storage_image_multisample == vk::TRUE,
    );
    feature(
        "shaderStorageImageReadWithoutFormat",
        head.shader_storage_image_read_without_format == vk::TRUE,
    );
    feature(
        "shaderStorageImageWriteWithoutFormat",
        head.shader_storage_image_write_without_format == vk::TRUE,
    );
    feature(
        "shaderUniformBufferArrayDynamicIndexing",
        head.shader_uniform_buffer_array_dynamic_indexing == vk::TRUE,
    );
    feature(
        "shaderSampledImageArrayDynamicIndexing",
        head.shader_sampled_image_array_dynamic_indexing == vk::TRUE,
    );
    feature(
        "shaderStorageBufferArrayDynamicIndexing",
        head.shader_storage_buffer_array_dynamic_indexing == vk::TRUE,
    );
    feature(
        "shaderStorageImageArrayDynamicIndexing",
        head.shader_storage_image_array_dynamic_indexing == vk::TRUE,
    );
    feature("shaderClipDistance", head.shader_clip_distance == vk::TRUE);
    feature("shaderCullDistance", head.shader_cull_distance == vk::TRUE);
    feature("shaderFloat64", head.shader_float64 == vk::TRUE);
    feature("shaderInt64", head.shader_int64 == vk::TRUE);
    feature("shaderInt16", head.shader_int16 == vk::TRUE);
    feature(
        "shaderResourceResidency",
        head.shader_resource_residency == vk::TRUE,
    );
    if has_1_2 {
        feature(
            "storageBuffer16BitAccess",
            vulkan11.storage_buffer16_bit_access == vk::TRUE,
        );
        feature(
            "uniformAndStorageBuffer16BitAccess",
            vulkan11.uniform_and_storage_buffer16_bit_access == vk::TRUE,
        );
        feature(
            "storagePushConstant16",
            vulkan11.storage_push_constant16 == vk::TRUE,
        );
        feature(
            "storageInputOutput16",
            vulkan11.storage_input_output16 == vk::TRUE,
        );
        feature("multiview", vulkan11.multiview == vk::TRUE);
        feature(
            "shaderDrawParameters",
            vulkan11.shader_draw_parameters == vk::TRUE,
        );
        feature("shaderFloat16", vulkan12.shader_float16 == vk::TRUE);
        feature("shaderInt8", vulkan12.shader_int8 == vk::TRUE);
        feature(
            "storageBuffer8BitAccess",
            vulkan12.storage_buffer8_bit_access == vk::TRUE,
        );
        feature(
            "uniformAndStorageBuffer8BitAccess",
            vulkan12.uniform_and_storage_buffer8_bit_access == vk::TRUE,
        );
        feature(
            "storagePushConstant8",
            vulkan12.storage_push_constant8 == vk::TRUE,
        );
        feature(
            "scalarBlockLayout",
            vulkan12.scalar_block_layout == vk::TRUE,
        );
        feature("timelineSemaphore", vulkan12.timeline_semaphore == vk::TRUE);
        feature("hostQueryReset", vulkan12.host_query_reset == vk::TRUE);
        feature(
            "bufferDeviceAddress",
            vulkan12.buffer_device_address == vk::TRUE,
        );
        feature(
            "descriptorIndexing",
            vulkan12.descriptor_indexing == vk::TRUE,
        );
        feature(
            "samplerMirrorClampToEdge",
            vulkan12.sampler_mirror_clamp_to_edge == vk::TRUE,
        );
        feature(
            "drawIndirectCount",
            vulkan12.draw_indirect_count == vk::TRUE,
        );
    }
    if robustness2 {
        feature(
            "robustBufferAccess2",
            robust.robust_buffer_access2 == vk::TRUE,
        );
        feature(
            "robustImageAccess2",
            robust.robust_image_access2 == vk::TRUE,
        );
        feature("nullDescriptor", robust.null_descriptor == vk::TRUE);
    }
}

/// The limit subset a fix decision is read from — the same list the probe
/// prints, so the two artefacts diff field by field.
fn limits(limits: &vk::PhysicalDeviceLimits, out: &mut Vec<String>) {
    let mut limit = |name: &str, value: Scene| {
        // The kind travels with the value: `from-log.py` has to rebuild the
        // probe's JSON, and "2147483647" is a number while "0xf" is a string
        // and "1x2x3" is a vector. A reader that guessed would be wrong the
        // first time a sample-count mask landed in a numeric field.
        let kind = value.kind();
        out.push(
            Clause::new("limit")
                .text("name", name)
                .text("kind", kind)
                .value("value", value)
                .done(),
        )
    };
    limit(
        "maxComputeWorkGroupCount",
        Scene::Vector(vec![
            u64::from(limits.max_compute_work_group_count[0]),
            u64::from(limits.max_compute_work_group_count[1]),
            u64::from(limits.max_compute_work_group_count[2]),
        ]),
    );
    limit(
        "maxComputeWorkGroupSize",
        Scene::Vector(vec![
            u64::from(limits.max_compute_work_group_size[0]),
            u64::from(limits.max_compute_work_group_size[1]),
            u64::from(limits.max_compute_work_group_size[2]),
        ]),
    );
    limit(
        "maxComputeWorkGroupInvocations",
        Scene::Number(u64::from(limits.max_compute_work_group_invocations)),
    );
    limit(
        "maxComputeSharedMemorySize",
        Scene::Number(u64::from(limits.max_compute_shared_memory_size)),
    );
    limit(
        "maxPerStageDescriptorSamplers",
        Scene::Number(u64::from(limits.max_per_stage_descriptor_samplers)),
    );
    limit(
        "maxPerStageDescriptorUniformBuffers",
        Scene::Number(u64::from(limits.max_per_stage_descriptor_uniform_buffers)),
    );
    limit(
        "maxPerStageDescriptorStorageBuffers",
        Scene::Number(u64::from(limits.max_per_stage_descriptor_storage_buffers)),
    );
    limit(
        "maxPerStageDescriptorSampledImages",
        Scene::Number(u64::from(limits.max_per_stage_descriptor_sampled_images)),
    );
    limit(
        "maxPerStageDescriptorStorageImages",
        Scene::Number(u64::from(limits.max_per_stage_descriptor_storage_images)),
    );
    limit(
        "maxPerStageDescriptorInputAttachments",
        Scene::Number(u64::from(limits.max_per_stage_descriptor_input_attachments)),
    );
    limit(
        "maxPerStageResources",
        Scene::Number(u64::from(limits.max_per_stage_resources)),
    );
    limit(
        "maxDescriptorSetSamplers",
        Scene::Number(u64::from(limits.max_descriptor_set_samplers)),
    );
    limit(
        "maxDescriptorSetUniformBuffers",
        Scene::Number(u64::from(limits.max_descriptor_set_uniform_buffers)),
    );
    limit(
        "maxDescriptorSetStorageBuffers",
        Scene::Number(u64::from(limits.max_descriptor_set_storage_buffers)),
    );
    limit(
        "maxDescriptorSetSampledImages",
        Scene::Number(u64::from(limits.max_descriptor_set_sampled_images)),
    );
    limit(
        "maxDescriptorSetStorageImages",
        Scene::Number(u64::from(limits.max_descriptor_set_storage_images)),
    );
    limit(
        "maxBoundDescriptorSets",
        Scene::Number(u64::from(limits.max_bound_descriptor_sets)),
    );
    limit(
        "maxUniformBufferRange",
        Scene::Number(u64::from(limits.max_uniform_buffer_range)),
    );
    limit(
        "maxStorageBufferRange",
        Scene::Number(u64::from(limits.max_storage_buffer_range)),
    );
    limit(
        "minUniformBufferOffsetAlignment",
        Scene::Number(limits.min_uniform_buffer_offset_alignment),
    );
    limit(
        "minStorageBufferOffsetAlignment",
        Scene::Number(limits.min_storage_buffer_offset_alignment),
    );
    limit(
        "maxPushConstantsSize",
        Scene::Number(u64::from(limits.max_push_constants_size)),
    );
    limit(
        "framebufferColorSampleCounts",
        Scene::Text(format!(
            "{:#x}",
            limits.framebuffer_color_sample_counts.as_raw()
        )),
    );
    limit(
        "framebufferDepthSampleCounts",
        Scene::Text(format!(
            "{:#x}",
            limits.framebuffer_depth_sample_counts.as_raw()
        )),
    );
    limit(
        "sampledImageColorSampleCounts",
        Scene::Text(format!(
            "{:#x}",
            limits.sampled_image_color_sample_counts.as_raw()
        )),
    );
    limit(
        "sampledImageIntegerSampleCounts",
        Scene::Text(format!(
            "{:#x}",
            limits.sampled_image_integer_sample_counts.as_raw()
        )),
    );
    limit(
        "sampledImageDepthSampleCounts",
        Scene::Text(format!(
            "{:#x}",
            limits.sampled_image_depth_sample_counts.as_raw()
        )),
    );
    limit(
        "maxColorAttachments",
        Scene::Number(u64::from(limits.max_color_attachments)),
    );
    limit(
        "maxFramebufferWidth",
        Scene::Number(u64::from(limits.max_framebuffer_width)),
    );
    limit(
        "maxFramebufferHeight",
        Scene::Number(u64::from(limits.max_framebuffer_height)),
    );
    limit(
        "maxFramebufferLayers",
        Scene::Number(u64::from(limits.max_framebuffer_layers)),
    );
    limit(
        "maxImageDimension1D",
        Scene::Number(u64::from(limits.max_image_dimension1_d)),
    );
    limit(
        "maxImageDimension2D",
        Scene::Number(u64::from(limits.max_image_dimension2_d)),
    );
    limit(
        "maxImageDimension3D",
        Scene::Number(u64::from(limits.max_image_dimension3_d)),
    );
    limit(
        "maxImageDimensionCube",
        Scene::Number(u64::from(limits.max_image_dimension_cube)),
    );
    limit(
        "maxImageArrayLayers",
        Scene::Number(u64::from(limits.max_image_array_layers)),
    );
    limit(
        "maxTexelBufferElements",
        Scene::Number(u64::from(limits.max_texel_buffer_elements)),
    );
    limit(
        "maxTexelOffset",
        Scene::Signed(i64::from(limits.max_texel_offset)),
    );
    limit(
        "minTexelOffset",
        Scene::Signed(i64::from(limits.min_texel_offset)),
    );
    limit(
        "maxSamplerLodBias",
        Scene::Float(f64::from(limits.max_sampler_lod_bias)),
    );
    limit(
        "maxSamplerAnisotropy",
        Scene::Float(f64::from(limits.max_sampler_anisotropy)),
    );
    limit(
        "maxViewports",
        Scene::Number(u64::from(limits.max_viewports)),
    );
    limit(
        "maxViewportDimensions",
        Scene::Vector(vec![
            u64::from(limits.max_viewport_dimensions[0]),
            u64::from(limits.max_viewport_dimensions[1]),
        ]),
    );
    limit(
        "viewportBoundsRange",
        Scene::FloatVector(vec![
            f64::from(limits.viewport_bounds_range[0]),
            f64::from(limits.viewport_bounds_range[1]),
        ]),
    );
    limit(
        "maxDrawIndirectCount",
        Scene::Number(u64::from(limits.max_draw_indirect_count)),
    );
    limit(
        "maxVertexInputAttributes",
        Scene::Number(u64::from(limits.max_vertex_input_attributes)),
    );
    limit(
        "maxVertexInputBindings",
        Scene::Number(u64::from(limits.max_vertex_input_bindings)),
    );
    limit(
        "maxVertexInputAttributeOffset",
        Scene::Number(u64::from(limits.max_vertex_input_attribute_offset)),
    );
    limit(
        "maxVertexInputBindingStride",
        Scene::Number(u64::from(limits.max_vertex_input_binding_stride)),
    );
    limit(
        "maxVertexOutputComponents",
        Scene::Number(u64::from(limits.max_vertex_output_components)),
    );
    limit(
        "maxFragmentInputComponents",
        Scene::Number(u64::from(limits.max_fragment_input_components)),
    );
    limit(
        "maxFragmentOutputAttachments",
        Scene::Number(u64::from(limits.max_fragment_output_attachments)),
    );
    limit(
        "maxFragmentCombinedOutputResources",
        Scene::Number(u64::from(limits.max_fragment_combined_output_resources)),
    );
    limit(
        "maxMemoryAllocationCount",
        Scene::Number(u64::from(limits.max_memory_allocation_count)),
    );
    limit(
        "bufferImageGranularity",
        Scene::Number(limits.buffer_image_granularity),
    );
    limit(
        "nonCoherentAtomSize",
        Scene::Number(limits.non_coherent_atom_size),
    );
    limit(
        "optimalBufferCopyOffsetAlignment",
        Scene::Number(limits.optimal_buffer_copy_offset_alignment),
    );
    limit(
        "optimalBufferCopyRowPitchAlignment",
        Scene::Number(limits.optimal_buffer_copy_row_pitch_alignment),
    );
    limit(
        "timestampComputeAndGraphics",
        Scene::Bool(limits.timestamp_compute_and_graphics == vk::TRUE),
    );
    limit(
        "timestampPeriod",
        Scene::Float(f64::from(limits.timestamp_period)),
    );
}

/// The rail's own `VkImageCreateInfo` for one shape: one mip, one layer, one
/// sample, exclusive sharing, and the initial layout tiling and usage allow.
fn image_create_info(shape: &canonical::Shape) -> vk::ImageCreateInfo<'static> {
    // A host-visible image is created `PREINITIALIZED` so the rail can write
    // through the returned layout without a barrier; a color attachment may
    // not start there, so that lane states `UNDEFINED`.
    let preinitialized = shape.tiling == vk::ImageTiling::LINEAR
        && !shape.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT);
    vk::ImageCreateInfo::default()
        .image_type(shape.image_type)
        .format(shape.format)
        .extent(shape.extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(shape.tiling)
        .usage(shape.usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(if preinitialized {
            vk::ImageLayout::PREINITIALIZED
        } else {
            vk::ImageLayout::UNDEFINED
        })
}

/// The carrier the rail's fallback states for a refused `LINEAR` shape: an
/// `OPTIMAL`, device-local `TRANSFER_DST` image filled by one
/// `vkCmdCopyBufferToImage`. Create, allocate and bind are each recorded: the
/// fallback is worthless if only the create half succeeds.
fn carrier(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    shape: &canonical::Shape,
) -> Clause {
    let info = vk::ImageCreateInfo::default()
        .image_type(shape.image_type)
        .format(shape.format)
        .extent(shape.extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::from_raw(
            vk::ImageUsageFlags::SAMPLED.as_raw() | vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
        ))
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let mut clause = Clause::new("carrier")
        .text("format", shape.format_name)
        .text("type", shape.type_name)
        .text("extent", shape.extent_name)
        .text("tiling", shape.tiling_name)
        .text("usage", shape.usage_name);
    let image = match unsafe { device.create_image(&info, None) } {
        Ok(image) => image,
        Err(error) => return clause.text("create", &format!("{error:?}")),
    };
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
    let type_count = usize::try_from(memory_properties.memory_type_count).unwrap_or(0);
    let device_local = (0..type_count).find(|index| {
        requirements.memory_type_bits & (1 << index) != 0
            && memory_properties.memory_types[*index]
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
    });
    let Some(device_local) = device_local else {
        unsafe { device.destroy_image(image, None) };
        return clause.text("create", "OK").text("memory_type", "-");
    };
    clause = clause
        .text("create", "OK")
        .num("memory_type", device_local as u64);
    let allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(u32::try_from(device_local).unwrap_or(0));
    let memory = match unsafe { device.allocate_memory(&allocation, None) } {
        Ok(memory) => memory,
        Err(error) => return clause.text("allocate", &format!("{error:?}")),
    };
    let bound = unsafe { device.bind_image_memory(image, memory, 0) };
    unsafe {
        device.destroy_image(image, None);
        device.free_memory(memory, None);
    }
    clause.text("allocate", "OK").text(
        "bind",
        match bound {
            Ok(()) => "OK",
            Err(_) => "ERROR_UNKNOWN",
        },
    )
}

/// One line under construction: the `DEVICE_PROFILE <kind>` head and its
/// `k=v` clauses, in the order the caller states them.
struct Clause {
    out: String,
}

/// One limit's value, so the `limit` lines carry numbers, booleans, floats and
/// the two- and three-element vectors the probe's JSON carries.
enum Scene {
    Number(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    Vector(Vec<u64>),
    FloatVector(Vec<f64>),
}

impl Scene {
    /// The word this value's `kind=` clause carries, so the log reader can
    /// rebuild the probe's JSON without guessing at a string's type.
    fn kind(&self) -> &'static str {
        match self {
            Scene::Number(_) => "number",
            Scene::Signed(_) => "signed",
            Scene::Float(_) => "float",
            Scene::Bool(_) => "bool",
            Scene::Text(_) => "text",
            Scene::Vector(_) => "vector",
            Scene::FloatVector(_) => "float_vector",
        }
    }
}

impl Clause {
    fn new(kind: &str) -> Self {
        let mut out = String::with_capacity(96);
        out.push_str(PREFIX);
        out.push(' ');
        out.push_str(kind);
        Self { out }
    }

    fn num(mut self, key: &str, value: u64) -> Self {
        let _ = write!(self.out, " {key}={value}");
        self
    }

    fn bool(mut self, key: &str, value: bool) -> Self {
        let _ = write!(self.out, " {key}={value}");
        self
    }

    /// A value that may hold a space: quoted, with the three characters the
    /// line reader cannot carry neutralized.
    fn text(mut self, key: &str, value: &str) -> Self {
        let sanitized: String = value
            .chars()
            .map(|character| match character {
                '"' | '\n' | '\r' => '_',
                other => other,
            })
            .collect();
        if sanitized.is_empty() || sanitized.contains(' ') {
            let _ = write!(self.out, " {key}=\"{sanitized}\"");
        } else {
            let _ = write!(self.out, " {key}={sanitized}");
        }
        self
    }

    fn value(mut self, key: &str, value: Scene) -> Self {
        match value {
            Scene::Number(number) => {
                let _ = write!(self.out, " {key}={number}");
            }
            Scene::Signed(number) => {
                let _ = write!(self.out, " {key}={number}");
            }
            Scene::Float(number) => {
                let _ = write!(self.out, " {key}={number:.6}");
            }
            Scene::Bool(flag) => {
                let _ = write!(self.out, " {key}={flag}");
            }
            Scene::Text(text) => {
                let _ = write!(self.out, " {key}={text}");
            }
            Scene::Vector(values) => {
                let joined = values
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join("x");
                let _ = write!(self.out, " {key}={joined}");
            }
            Scene::FloatVector(values) => {
                let joined = values
                    .iter()
                    .map(|number| format!("{number:.6}"))
                    .collect::<Vec<_>>()
                    .join("x");
                let _ = write!(self.out, " {key}={joined}");
            }
        }
        self
    }

    fn done(self) -> String {
        self.out
    }
}

fn epoch_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

fn extent_name(extent: vk::Extent3D) -> String {
    format!("{}x{}x{}", extent.width, extent.height, extent.depth)
}

fn c_chars(value: &[std::os::raw::c_char]) -> String {
    unsafe { std::ffi::CStr::from_ptr(value.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn version_string(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version)
    )
}

/// A Vulkan `int32_t` enumeration value as an unsigned reading.
fn raw32(value: i32) -> u64 {
    u32::try_from(value).map(u64::from).unwrap_or(0)
}

fn device_type_name(device_type: vk::PhysicalDeviceType) -> &'static str {
    match device_type {
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated-gpu",
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete-gpu",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual-gpu",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    }
}

/// The driver's own name for itself, where the enumeration has one.
fn driver_id_name(driver_id: i32) -> &'static str {
    match driver_id {
        value if value == vk::DriverId::AMD_PROPRIETARY.as_raw() => "AMD_PROPRIETARY",
        value if value == vk::DriverId::AMD_OPEN_SOURCE.as_raw() => "AMD_OPEN_SOURCE",
        value if value == vk::DriverId::MESA_RADV.as_raw() => "MESA_RADV",
        value if value == vk::DriverId::NVIDIA_PROPRIETARY.as_raw() => "NVIDIA_PROPRIETARY",
        value if value == vk::DriverId::INTEL_PROPRIETARY_WINDOWS.as_raw() => {
            "INTEL_PROPRIETARY_WINDOWS"
        }
        value if value == vk::DriverId::INTEL_OPEN_SOURCE_MESA.as_raw() => "INTEL_OPEN_SOURCE_MESA",
        value if value == vk::DriverId::MESA_LLVMPIPE.as_raw() => "MESA_LLVMPIPE",
        value if value == vk::DriverId::MESA_TURNIP.as_raw() => "MESA_TURNIP",
        value if value == vk::DriverId::MESA_DOZEN.as_raw() => "MESA_DOZEN",
        value if value == vk::DriverId::MESA_VENUS.as_raw() => "MESA_VENUS",
        value if value == vk::DriverId::MESA_NVK.as_raw() => "MESA_NVK",
        value if value == vk::DriverId::MESA_PANVK.as_raw() => "MESA_PANVK",
        value if value == vk::DriverId::GOOGLE_SWIFTSHADER.as_raw() => "GOOGLE_SWIFTSHADER",
        value if value == vk::DriverId::MOLTENVK.as_raw() => "MOLTENVK",
        _ => "OTHER",
    }
}

fn queue_flag_names(flags: vk::QueueFlags) -> String {
    let mut names = Vec::new();
    if flags.contains(vk::QueueFlags::GRAPHICS) {
        names.push("GRAPHICS");
    }
    if flags.contains(vk::QueueFlags::COMPUTE) {
        names.push("COMPUTE");
    }
    if flags.contains(vk::QueueFlags::TRANSFER) {
        names.push("TRANSFER");
    }
    if flags.contains(vk::QueueFlags::SPARSE_BINDING) {
        names.push("SPARSE_BINDING");
    }
    if flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR) {
        names.push("VIDEO_DECODE");
    }
    if flags.contains(vk::QueueFlags::VIDEO_ENCODE_KHR) {
        names.push("VIDEO_ENCODE");
    }
    joined(names)
}

fn memory_heap_flag_names(flags: vk::MemoryHeapFlags) -> String {
    let mut names = Vec::new();
    if flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
        names.push("DEVICE_LOCAL");
    }
    if flags.contains(vk::MemoryHeapFlags::MULTI_INSTANCE) {
        names.push("MULTI_INSTANCE");
    }
    joined(names)
}

fn memory_property_flag_names(flags: vk::MemoryPropertyFlags) -> String {
    let mut names = Vec::new();
    if flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
        names.push("DEVICE_LOCAL");
    }
    if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
        names.push("HOST_VISIBLE");
    }
    if flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
        names.push("HOST_COHERENT");
    }
    if flags.contains(vk::MemoryPropertyFlags::HOST_CACHED) {
        names.push("HOST_CACHED");
    }
    if flags.contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED) {
        names.push("LAZILY_ALLOCATED");
    }
    if flags.contains(vk::MemoryPropertyFlags::PROTECTED) {
        names.push("PROTECTED");
    }
    joined(names)
}

fn format_flag_names(flags: vk::FormatFeatureFlags) -> String {
    let candidates = [
        (vk::FormatFeatureFlags::SAMPLED_IMAGE, "SAMPLED_IMAGE"),
        (vk::FormatFeatureFlags::STORAGE_IMAGE, "STORAGE_IMAGE"),
        (
            vk::FormatFeatureFlags::STORAGE_IMAGE_ATOMIC,
            "STORAGE_IMAGE_ATOMIC",
        ),
        (
            vk::FormatFeatureFlags::UNIFORM_TEXEL_BUFFER,
            "UNIFORM_TEXEL_BUFFER",
        ),
        (
            vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER,
            "STORAGE_TEXEL_BUFFER",
        ),
        (vk::FormatFeatureFlags::VERTEX_BUFFER, "VERTEX_BUFFER"),
        (vk::FormatFeatureFlags::COLOR_ATTACHMENT, "COLOR_ATTACHMENT"),
        (
            vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND,
            "COLOR_ATTACHMENT_BLEND",
        ),
        (
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            "DEPTH_STENCIL_ATTACHMENT",
        ),
        (vk::FormatFeatureFlags::TRANSFER_SRC, "TRANSFER_SRC"),
        (vk::FormatFeatureFlags::TRANSFER_DST, "TRANSFER_DST"),
        (
            vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
            "SAMPLED_IMAGE_FILTER_LINEAR",
        ),
        (
            vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_MINMAX,
            "SAMPLED_IMAGE_FILTER_MINMAX",
        ),
        (vk::FormatFeatureFlags::BLIT_SRC, "BLIT_SRC"),
        (vk::FormatFeatureFlags::BLIT_DST, "BLIT_DST"),
    ];
    let mut names = Vec::new();
    for (flag, name) in candidates {
        if flags.contains(flag) {
            names.push(name);
        }
    }
    joined(names)
}

fn joined(names: Vec<&'static str>) -> String {
    if names.is_empty() {
        "-".to_owned()
    } else {
        names.join("|")
    }
}

/// The canonical shape table: what a "device profile" is a profile *of*.
///
/// Mirrors `tools/device-profile/probe/src/canonical.rs` verbatim in
/// behaviour — the probe's JSON and this block have to be diffable against
/// each other, which is only true while both sides expand the same list.
/// Change one, change the other.
mod canonical {
    use ash::vk;

    pub(super) struct FormatName {
        pub(super) name: &'static str,
        pub(super) format: vk::Format,
    }

    pub(super) struct ExtentName {
        pub(super) name: &'static str,
        pub(super) image_type: vk::ImageType,
        pub(super) extent: vk::Extent3D,
    }

    pub(super) struct UsageName {
        pub(super) name: &'static str,
        pub(super) usage: vk::ImageUsageFlags,
        /// Vulkan states `COLOR_ATTACHMENT` for two-dimensional images only.
        pub(super) two_dimensional_only: bool,
    }

    pub(super) struct Shape {
        pub(super) format_name: &'static str,
        pub(super) format: vk::Format,
        pub(super) type_name: &'static str,
        pub(super) image_type: vk::ImageType,
        pub(super) extent_name: &'static str,
        pub(super) extent: vk::Extent3D,
        pub(super) tiling_name: &'static str,
        pub(super) tiling: vk::ImageTiling,
        pub(super) usage_name: &'static str,
        pub(super) usage: vk::ImageUsageFlags,
    }

    pub(super) fn formats() -> Vec<FormatName> {
        vec![
            FormatName {
                name: "B8G8R8A8_UNORM",
                format: vk::Format::B8G8R8A8_UNORM,
            },
            FormatName {
                name: "B8G8R8A8_SRGB",
                format: vk::Format::B8G8R8A8_SRGB,
            },
            FormatName {
                name: "R8G8B8A8_UNORM",
                format: vk::Format::R8G8B8A8_UNORM,
            },
            FormatName {
                name: "R8G8B8A8_SRGB",
                format: vk::Format::R8G8B8A8_SRGB,
            },
            FormatName {
                name: "R16G16B16A16_SFLOAT",
                format: vk::Format::R16G16B16A16_SFLOAT,
            },
            FormatName {
                name: "R32_SFLOAT",
                format: vk::Format::R32_SFLOAT,
            },
            FormatName {
                name: "R32_UINT",
                format: vk::Format::R32_UINT,
            },
            FormatName {
                name: "R16_SFLOAT",
                format: vk::Format::R16_SFLOAT,
            },
            FormatName {
                name: "R8_UNORM",
                format: vk::Format::R8_UNORM,
            },
        ]
    }

    pub(super) fn extents() -> Vec<ExtentName> {
        vec![
            ExtentName {
                name: "1920x1080x1",
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D {
                    width: 1920,
                    height: 1080,
                    depth: 1,
                },
            },
            ExtentName {
                name: "1024x1024x1",
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D {
                    width: 1024,
                    height: 1024,
                    depth: 1,
                },
            },
            ExtentName {
                name: "4x4x1",
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D {
                    width: 4,
                    height: 4,
                    depth: 1,
                },
            },
            ExtentName {
                name: "64x64x8",
                image_type: vk::ImageType::TYPE_3D,
                extent: vk::Extent3D {
                    width: 64,
                    height: 64,
                    depth: 8,
                },
            },
            ExtentName {
                name: "16384x1x1",
                image_type: vk::ImageType::TYPE_1D,
                extent: vk::Extent3D {
                    width: 16384,
                    height: 1,
                    depth: 1,
                },
            },
        ]
    }

    pub(super) fn usages() -> Vec<UsageName> {
        vec![
            UsageName {
                name: "SAMPLED",
                usage: vk::ImageUsageFlags::SAMPLED,
                two_dimensional_only: false,
            },
            UsageName {
                name: "SAMPLED+TRANSFER_DST",
                usage: vk::ImageUsageFlags::from_raw(
                    vk::ImageUsageFlags::SAMPLED.as_raw()
                        | vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
                ),
                two_dimensional_only: false,
            },
            UsageName {
                name: "COLOR_ATTACHMENT+SAMPLED",
                usage: vk::ImageUsageFlags::from_raw(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw()
                        | vk::ImageUsageFlags::SAMPLED.as_raw(),
                ),
                two_dimensional_only: true,
            },
            UsageName {
                name: "STORAGE",
                usage: vk::ImageUsageFlags::STORAGE,
                two_dimensional_only: false,
            },
        ]
    }

    /// The whole truth table, expanded in the probe's reading order: extent,
    /// then usage, then tiling, then format.
    pub(super) fn shapes() -> Vec<Shape> {
        let formats = formats();
        let mut shapes = Vec::new();
        for extent in extents() {
            for usage in usages() {
                if usage.two_dimensional_only && extent.image_type != vk::ImageType::TYPE_2D {
                    continue;
                }
                for (tiling_name, tiling) in [
                    ("LINEAR", vk::ImageTiling::LINEAR),
                    ("OPTIMAL", vk::ImageTiling::OPTIMAL),
                ] {
                    for format in &formats {
                        shapes.push(Shape {
                            format_name: format.name,
                            format: format.format,
                            type_name: image_type_name(extent.image_type),
                            image_type: extent.image_type,
                            extent_name: extent.name,
                            extent: extent.extent,
                            tiling_name,
                            tiling,
                            usage_name: usage.name,
                            usage: usage.usage,
                        });
                    }
                }
            }
        }
        shapes
    }

    fn image_type_name(image_type: vk::ImageType) -> &'static str {
        match image_type {
            vk::ImageType::TYPE_1D => "1d",
            vk::ImageType::TYPE_2D => "2d",
            vk::ImageType::TYPE_3D => "3d",
            _ => "other",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_enabled, Clause, SCHEMA};

    /// The switch is off for unset and for every word that is not one of the
    /// five truthy ones, and on for each of those in the spelling a round's
    /// launcher uses. The default matters more than the arming: unset must
    /// leave the product exactly as it was.
    #[test]
    fn the_switch_is_off_unless_a_truthy_word_arms_it() {
        assert!(!parse_enabled(None));
        assert!(!parse_enabled(Some("")));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("off")));
        assert!(!parse_enabled(Some("OFF")));
        assert!(!parse_enabled(Some("false")));
        assert!(!parse_enabled(Some("no")));
        assert!(!parse_enabled(Some("forced")));
        assert!(parse_enabled(Some("1")));
        assert!(parse_enabled(Some("on")));
        assert!(parse_enabled(Some("ON")));
        assert!(parse_enabled(Some("true")));
        assert!(parse_enabled(Some("yes")));
        assert!(parse_enabled(Some(" on ")));
    }

    /// A value with a space is quoted, and the three characters a line reader
    /// cannot carry are neutralized rather than escaped: `from-log.py` splits
    /// on spaces and reads quotes, so a value may not invent an escape.
    #[test]
    fn a_value_with_a_space_is_quoted_and_neutralized() {
        let line = Clause::new("identity")
            .text("device_name", "NVIDIA GeForce RTX 5060")
            .text("driver_name", "NVIDIA")
            .text("driver_info", "616.92")
            .text("quoted", "a\"b\nc")
            .done();
        assert_eq!(
            line,
            "DEVICE_PROFILE identity device_name=\"NVIDIA GeForce RTX 5060\" \
             driver_name=NVIDIA driver_info=616.92 quoted=a_b_c"
        );
    }

    /// The block's own head is what a log reader keys on, and the schema line
    /// is the one the converter checks before it trusts any other line.
    #[test]
    fn the_block_names_its_schema() {
        let line = Clause::new("begin").text("schema", SCHEMA).done();
        assert!(line.starts_with("DEVICE_PROFILE begin "));
        assert!(line.contains("schema=metal-api-device-profile/1"));
    }
}
