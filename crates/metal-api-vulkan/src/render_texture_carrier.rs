//! The carrier an uploaded render texture's bytes travel in when the device
//! does not admit the host-visible one (2026-09-21, the IT3 black window).
//!
//! # What this is for
//!
//! [`crate::render`]'s `create_render_textures` uploads one sampled declaration
//! per pass into an image it creates itself. The carrier that image states is
//! the host-visible one — `LINEAR` tiling over a `HOST_VISIBLE |
//! HOST_COHERENT` allocation the rail writes through the driver's own
//! `VkSubresourceLayout` — unless the declaration belongs to one of the three
//! device-copied arms (the owner's no-copy window, the pass-entry snapshot and
//! the volume).
//!
//! Vulkan promises the linear carrier for the two-dimensional shape only: no
//! three-dimensional format has to support `LINEAR` tiling, and neither does
//! any one-dimensional one. The volume arm already paid for that lesson on an
//! RTX 5060 (`vkCreateImage`: "Requested format is not supported on this
//! device", the reading its own comment records), and the same device refuses
//! the **one-dimensional** shape for every format the rail's window names: the
//! shape-by-shape probe archived beside the IT3 evidence
//! (`evidence/rtx-texture-format-<tip>-2026-09-21/`) reads
//! `create=ERROR_FORMAT_NOT_SUPPORTED` for each `1d-16384x1` lane while every
//! two-dimensional shape of the same formats, usages and extents creates.
//!
//! That refusal is the IT3 black window. The video pipeline's LUT declaration
//! is a `16384x1` `R32_SFLOAT` `TYPE_1D` view; its one `vkCreateImage` was
//! refused, the pass that would have drawn the video's next frame was skipped
//! by name (`draws_skipped_after_engine_refusal`), and the window kept
//! presenting the restaled black backing for the rest of the run.
//!
//! # What the fallback does
//!
//! One question, asked of the device before any object exists for the
//! declaration: `vkGetPhysicalDeviceImageFormatProperties` for the shape the
//! rail is about to state (`format`, `image_type`, `LINEAR`, the declaration's
//! usage), held against the `maxExtent` it answers. A shape the device does not
//! admit takes the volume's carrier instead — an `OPTIMAL`, device-local
//! `TRANSFER_DST` image filled by one `vkCmdCopyBufferToImage` from a
//! host-visible staging buffer — which is the same two-step shape the volume
//! and pass-entry snapshot arms already use, and one every conformant device
//! supports for a sampled format it admits at all.
//!
//! The question is asked per declaration rather than remembered: the render
//! rail's other device questions (`admit_color_attachment`,
//! `format_supports_color_attachment_samples`) are asked per pass too, and a
//! cache would have to key on the whole shape to stay honest about `maxExtent`.
//!
//! # The switch and its reading
//!
//! `METAL_API_VULKAN_RENDER_TEXTURE_LINEAR_FALLBACK` (`1`, `on`, `ON`, `true`,
//! `yes`) arms it; **unset is on**, because the increment was flipped once its
//! device round had read it — the RTX 5060 round's own transcript shows the
//! `1d`/`3d` + `LINEAR` shapes refused (`create=ERROR_FORMAT_NOT_SUPPORTED`)
//! while every two-dimensional shape of the same formats, usages and extents
//! creates, and the armed arm answers all six of the round's directed cases
//! where the control arm fails the one-dimensional LUT by exit code. The
//! control words `0`/`off`/`false`/`no` restore the pre-fix path byte for byte:
//! the question below is not asked at all, no carrier moves, and every
//! declaration is created exactly as it was.

use ash::vk;
use metal_api_core::provider::{FieldValue, ProviderError};
use std::sync::OnceLock;

/// Whether the fallback is armed, read once from the process environment.
pub(crate) fn enabled_from_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_VULKAN_RENDER_TEXTURE_LINEAR_FALLBACK")
                .ok()
                .as_deref(),
        )
    })
}

/// The switch's own reading: **on unless the variable turns it off**. The
/// fallback was flipped on once its device round had priced it (the RTX 5060
/// transcript: the `1d`/`3d` + `LINEAR` shapes are refused, every 2D shape of
/// the same formats creates; the armed arm passes the directed 1D LUT that the
/// control arm fails, and its 141 captures are byte-identical to the base
/// round's). The control words keep the pre-fix carrier reachable.
fn parse_enabled(value: Option<&str>) -> bool {
    !matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// The device's own answer for the shape one declaration would state as a
/// host-visible `LINEAR` image.
///
/// Every part is a reading rather than a summary: the query's own result, the
/// declaration's extent held against the `maxExtent` the query answered, and
/// the format's two tiling feature words. A refusal about the same shape
/// attaches all of them, so the failure names what the device said instead of
/// leaving the shape to be inferred from the request.
#[derive(Clone, Copy)]
pub(crate) struct LinearAdmission {
    /// `vkGetPhysicalDeviceImageFormatProperties`'s own answer.
    answer: vk::Result,
    /// Whether the declaration's extent fits the `maxExtent` that answer
    /// carried. A query that refused carries none, so this is `false` beside
    /// it.
    extent_admitted: bool,
    /// The format's `LINEAR` features carry `SAMPLED_IMAGE`.
    sampled_linear: bool,
    /// The format's `LINEAR` feature word.
    linear_features: vk::FormatFeatureFlags,
    /// The format's `OPTIMAL` feature word, which the fallback's carrier reads.
    optimal_features: vk::FormatFeatureFlags,
}

impl LinearAdmission {
    /// Whether the host-visible carrier stays legal for this shape.
    pub(crate) fn admits_linear_carrier(self) -> bool {
        self.answer == vk::Result::SUCCESS && self.extent_admitted && self.sampled_linear
    }

    /// The fields this answer adds to a refusal about the same shape: the
    /// device's own words, so a reader of the fail log can tell "the driver
    /// never promised this" from "the rail asked for the wrong thing".
    pub(crate) fn attach(self, error: ProviderError) -> ProviderError {
        error
            .with_field(
                "device_linear_answer",
                FieldValue::Text(format!("{:?}", self.answer)),
            )
            .with_field(
                "device_linear_sampled",
                FieldValue::Bool(self.sampled_linear),
            )
            .with_field(
                "device_linear_features",
                FieldValue::Unsigned(u64::from(self.linear_features.as_raw())),
            )
            .with_field(
                "device_optimal_features",
                FieldValue::Unsigned(u64::from(self.optimal_features.as_raw())),
            )
            .with_field(
                "device_extent_admitted",
                FieldValue::Bool(self.extent_admitted),
            )
    }

    /// The same answer as one `k=v` clause, for the sentence a fail line's
    /// detail carries: the boundary that logs this refusal splits its line on
    /// spaces, so no value here holds one.
    pub(crate) fn reading(self) -> String {
        format!(
            "linear_answer={:?} sampled_linear={} linear_features={:#x} \
             optimal_features={:#x} extent_admitted={}",
            self.answer,
            self.sampled_linear,
            self.linear_features.as_raw(),
            self.optimal_features.as_raw(),
            self.extent_admitted,
        )
    }
}

/// Ask the device the one question the host-visible carrier depends on,
/// through the crate's own context so the instance and the physical device are
/// the selected ones.
pub(crate) fn ask_linear_admission(
    context: &crate::VulkanContext,
    format: vk::Format,
    image_type: vk::ImageType,
    extent: vk::Extent3D,
    usage: vk::ImageUsageFlags,
) -> LinearAdmission {
    let features = unsafe {
        context
            .instance
            .get_physical_device_format_properties(context.physical, format)
    };
    let answer = unsafe {
        context
            .instance
            .get_physical_device_image_format_properties(
                context.physical,
                format,
                image_type,
                vk::ImageTiling::LINEAR,
                usage,
                vk::ImageCreateFlags::empty(),
            )
    };
    let extent_admitted = answer.as_ref().is_ok_and(|properties| {
        extent.width <= properties.max_extent.width
            && extent.height <= properties.max_extent.height
            && extent.depth <= properties.max_extent.depth
    });
    LinearAdmission {
        answer: answer
            .map(|_| vk::Result::SUCCESS)
            .unwrap_or_else(|error| error),
        extent_admitted,
        sampled_linear: features
            .linear_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE),
        linear_features: features.linear_tiling_features,
        optimal_features: features.optimal_tiling_features,
    }
}

/// Whether one **uploaded** declaration has to take the device-copied carrier:
/// the switch is armed *and* the device does not admit the host-visible shape
/// the rail would otherwise state for it.
///
/// The usage is the one the host-visible arm states — `SAMPLED` alone; a
/// device-copied arm is the only one that adds `TRANSFER_DST`. The switch is
/// read before the question, so the off arm asks the device nothing at all.
pub(crate) fn uploaded_shape_takes_device_copy(
    context: &crate::VulkanContext,
    format: vk::Format,
    image_type: vk::ImageType,
    extent: vk::Extent3D,
) -> bool {
    enabled_from_env()
        && !ask_linear_admission(
            context,
            format,
            image_type,
            extent,
            vk::ImageUsageFlags::SAMPLED,
        )
        .admits_linear_carrier()
}

#[cfg(test)]
mod tests {
    use super::{parse_enabled, LinearAdmission};
    use ash::vk;

    /// The switch is **on** for unset and for everything that is not one of
    /// the four control words, and off for each of them in the case the
    /// process environment hands them over.
    #[test]
    fn the_switch_is_on_unless_a_control_word_turns_it_off() {
        assert!(parse_enabled(None));
        assert!(parse_enabled(Some("")));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("off")));
        assert!(!parse_enabled(Some("no")));
        assert!(!parse_enabled(Some("false")));
        assert!(!parse_enabled(Some("OFF")));
        assert!(parse_enabled(Some("force")));
        assert!(parse_enabled(Some("1")));
        assert!(parse_enabled(Some("on")));
        assert!(parse_enabled(Some("ON")));
        assert!(parse_enabled(Some("true")));
        assert!(parse_enabled(Some("Yes")));
        assert!(parse_enabled(Some(" on ")));
    }

    /// The carrier is admitted only when all three readings say so: the query
    /// answered, the extent fits what it answered, and the format's linear
    /// features carry a sampled image. Each one alone is a device that never
    /// promised the shape.
    #[test]
    fn the_carrier_needs_all_three_readings() {
        let admitted = LinearAdmission {
            answer: vk::Result::SUCCESS,
            extent_admitted: true,
            sampled_linear: true,
            linear_features: vk::FormatFeatureFlags::SAMPLED_IMAGE,
            optimal_features: vk::FormatFeatureFlags::SAMPLED_IMAGE,
        };
        assert!(admitted.admits_linear_carrier());
        let refused = LinearAdmission {
            answer: vk::Result::ERROR_FORMAT_NOT_SUPPORTED,
            ..admitted
        };
        assert!(!refused.admits_linear_carrier());
        let too_wide = LinearAdmission {
            extent_admitted: false,
            ..admitted
        };
        assert!(!too_wide.admits_linear_carrier());
        let no_sampled_bit = LinearAdmission {
            sampled_linear: false,
            ..admitted
        };
        assert!(!no_sampled_bit.admits_linear_carrier());
    }
}
