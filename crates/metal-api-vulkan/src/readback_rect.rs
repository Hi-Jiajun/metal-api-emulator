//! The rectangle one render pass can have written, and the readback it saves.
//!
//! The render half used to hand back every stored attachment whole: the
//! `vkCmdCopyImageToBuffer` that stages an attachment copied its full
//! `width * height * texel` extent and the host then read that many bytes out of
//! the staging buffer's mapping. The phase profile put that one copy at 50.3% of
//! the whole `submit` call — `render_readback` = 3.86 ms/draw, which is a
//! 1920x1080 RGBA8 surface read at 2.15 GB/s (`docs/SUBMIT-PHASE-PROFILE.md`,
//! `evidence/submit-split-e3af9e0-2026-09-19`) — while the guest's own shapes
//! write a few thousand texels of that surface per record.
//!
//! The rule this module states is the narrow one that can be *proved*
//! (`docs/WRITTEN-RECT-READBACK.md`):
//!
//! * a single-sample pass writes only inside `viewport ∩ scissor ∩ extent`. The
//!   pass's render area is the attachment's whole extent, the viewport rect is
//!   what NDC's clip volume maps to, and the scissor test is what the fragment
//!   stage's writes pass through, so a texel outside that intersection receives
//!   no fragment at all (`research/docs/23` §3.1, v100 / v29);
//! * the texels outside it keep the bytes the *image* was seeded with. A
//!   `LoadOp::Load` image is seeded from the attachment's own declaration and a
//!   `LoadOp::Clear` image is filled with the clear's payload, so both are bytes
//!   this rail already holds host-side;
//!
//! so a readback that copies only the intersection out of the device and rebuilds
//! the rest of the frame from the seed publishes exactly the frame the whole
//! readback publishes, byte for byte — at both destinations the readback serves:
//! the writeback channel and an owner-window landing.
//!
//! Every shape whose seed this rail does *not* hold host-side, and every shape
//! that writes outside the intersection, reads back whole exactly as before and
//! is counted as a fallback ([`RectRefusal`], `VulkanContext::readback_regions`).

use metal_api_core::provider::LoadOp;

/// A rectangle of one attachment's texel grid, in attachment coordinates.
///
/// `width` or `height` may be zero: the intersection of a declared viewport and
/// a declared scissor is empty when the two are disjoint, and a draw inside an
/// empty intersection writes nothing — which is a *proof* rather than a gap, so
/// the shape still takes the trimmed path (with a zero-byte copy).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WrittenRect {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl WrittenRect {
    /// The rectangle covering one attachment's whole extent.
    pub(crate) const fn whole(extent: [u32; 2]) -> Self {
        Self {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        }
    }

    /// Whether the rectangle covers no texel at all.
    pub(crate) const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Whether the rectangle is the attachment's whole extent, which is the
    /// shape whose readback saves nothing.
    pub(crate) const fn is_whole(&self, extent: [u32; 2]) -> bool {
        self.x == 0 && self.y == 0 && self.width == extent[0] && self.height == extent[1]
    }

    /// The number of texels the rectangle covers.
    pub(crate) fn texels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// The bytes a tightly packed copy of the rectangle occupies.
    pub(crate) fn byte_length(&self, texel_bytes: u64) -> Option<u64> {
        self.texels().checked_mul(texel_bytes)
    }
}

/// Why one attachment's readback cannot be narrowed to its written rectangle.
///
/// Each arm is a fact about the pass, not a severity: the answer to every one of
/// them is the same whole-attachment readback, and they are kept apart so the
/// profile can say *which* shape the guest's traffic falls back on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RectRefusal {
    /// The pass rasterizes more than one sample per texel. A multisampled
    /// attachment's observable bytes are the *resolve target's*, and the
    /// subpass resolve writes every texel of it rather than only the ones the
    /// draw covered (`research/docs/23` §3.3, v51).
    Multisample,
    /// The load operation's contents are not bytes this rail holds: a
    /// `LoadOp::Resident` attachment keeps the provider image's own bytes, and a
    /// `LoadOp::DontCare` attachment's pre-pass contents are undefined.
    LoadOp,
    /// A `LoadOp::Load` attachment with no resolved previous contents. The rail
    /// refuses that shape before any device object exists; the arm is the
    /// decision's own restatement of it.
    NoSource,
    /// The declared viewport is empty or reaches outside the attachment's
    /// extent, so "inside the viewport" is not a rectangle fully inside the
    /// raster.
    Viewport,
    /// The declared scissor is empty or reaches outside the attachment's
    /// extent. The core contract refuses both (`ScissorOutOfBounds`); a
    /// directly-constructed request that states one falls back here.
    Scissor,
}

impl RectRefusal {
    /// The counter this refusal is banked in.
    pub(crate) const fn fallback(self) -> ReadbackFallback {
        match self {
            Self::Viewport | Self::Scissor => ReadbackFallback::Bounds,
            Self::Multisample | Self::LoadOp | Self::NoSource => ReadbackFallback::Shape,
        }
    }
}

/// Why one stored attachment read back whole
/// (`VulkanExecutor::readback_region_counts`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadbackFallback {
    /// `METAL_API_VULKAN_FULL_READBACK` asked for the pre-increment path.
    Switch,
    /// The shape's seed is not host state this rail holds.
    Shape,
    /// The declared viewport or scissor is not a rectangle the rail can prove.
    Bounds,
    /// The written rectangle covers the whole attachment.
    Whole,
}

/// The rectangle one single-sample pass can have written, or the reason this
/// rail cannot prove the pass stayed inside one.
///
/// `extent` is the attachment's own texel grid, `viewport` and `scissor` are the
/// pass's declared rects in framebuffer coordinates (`[x, y, width, height]`),
/// `multisampled` is whether the pass rasterizes more than one sample per texel,
/// `load` is the attachment's load operation and `has_source` whether a `Load`
/// attachment's previous contents were resolved.
///
/// The answer is a *superset* of what the pass may have written: a rectangle
/// that is too large costs bytes, one that is too small would publish stale
/// texels, so every shape this function cannot state exactly is answered with
/// [`RectRefusal`] rather than with a guess.
pub(crate) fn written_rect(
    extent: [u32; 2],
    viewport: [u32; 4],
    scissor: Option<[u32; 4]>,
    multisampled: bool,
    load: LoadOp,
    has_source: bool,
) -> Result<WrittenRect, RectRefusal> {
    if multisampled {
        return Err(RectRefusal::Multisample);
    }
    match load {
        // Both seeds are host state this rail holds: the clear's own payload and
        // the previous contents the attachment's declaration resolved to.
        LoadOp::Clear(_) => {}
        LoadOp::Load if has_source => {}
        LoadOp::Load => return Err(RectRefusal::NoSource),
        LoadOp::Resident | LoadOp::DontCare => return Err(RectRefusal::LoadOp),
    }
    let viewport = rect_of(viewport);
    if !inside_the_extent(viewport, extent) {
        return Err(RectRefusal::Viewport);
    }
    let scissor = match scissor {
        Some(scissor) => {
            let scissor = rect_of(scissor);
            if !inside_the_extent(scissor, extent) {
                return Err(RectRefusal::Scissor);
            }
            scissor
        }
        // A pass that declares no scissor tests against the whole render area,
        // which is the attachment's own extent (`research/docs/23` §3.3, v29).
        None => WrittenRect::whole(extent),
    };
    Ok(intersection(viewport, scissor))
}

/// Build the base a `LoadOp::Clear` attachment's trimmed readback starts from:
/// one clear payload repeated over the attachment's whole extent, which is what
/// the render pass's clear writes into every texel of its render area.
///
/// `clear` is the clear's payload — `AttachmentFormat::bytes_per_texel()` bytes
/// in the format's memory order — and `texels` is the attachment's extent in
/// texels. The repetition is done through a kilobyte-wide pattern rather than
/// texel by texel, because the payload can be one byte wide and the frames this
/// serves are megabytes.
pub(crate) fn clear_frame(
    texels: u64,
    texel_bytes: usize,
    clear: &[u8],
) -> Result<Vec<u8>, PatchRefusal> {
    if texel_bytes == 0 || clear.len() != texel_bytes {
        return Err(PatchRefusal::Shape);
    }
    let length = texels
        .checked_mul(texel_bytes as u64)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(PatchRefusal::Shape)?;
    // A multiple of the texel width, so every chunk of the frame below is a
    // whole number of texels of the same payload.
    const PATTERN: usize = 4096;
    let mut pattern = Vec::with_capacity(PATTERN);
    while pattern.len() < PATTERN {
        pattern.extend_from_slice(clear);
    }
    let mut frame = Vec::with_capacity(length);
    while frame.len() < length {
        let remaining = length - frame.len();
        let take = remaining.min(pattern.len());
        frame.extend_from_slice(&pattern[..take]);
    }
    Ok(frame)
}

/// Write one trimmed rectangle's device bytes into the frame's own texel grid.
///
/// `frame` is the attachment's whole tightly packed extent, `rect_bytes` is the
/// same rectangle as `vkCmdCopyImageToBuffer` leaves it — tightly packed rows of
/// `rect.width` texels, which is what a copy region with no row stride produces
/// (`VkBufferImageCopy::bufferRowLength = 0`) — and `texel_bytes` is the
/// attachment format's texel width.
///
/// A `rect` that is empty copies nothing, which is the shape of a draw whose
/// scissor and viewport do not intersect.
pub(crate) fn patch_rect(
    frame: &mut [u8],
    extent: [u32; 2],
    rect: WrittenRect,
    texel_bytes: usize,
    rect_bytes: &[u8],
) -> Result<(), PatchRefusal> {
    if texel_bytes == 0 {
        return Err(PatchRefusal::Shape);
    }
    let row_bytes = usize::try_from(u64::from(extent[0]) * texel_bytes as u64)
        .map_err(|_| PatchRefusal::Shape)?;
    let frame_bytes = row_bytes
        .checked_mul(extent[1] as usize)
        .ok_or(PatchRefusal::Shape)?;
    if frame.len() != frame_bytes {
        return Err(PatchRefusal::Frame);
    }
    let rect_row_bytes = usize::try_from(u64::from(rect.width) * texel_bytes as u64)
        .map_err(|_| PatchRefusal::Shape)?;
    let expected = rect_row_bytes
        .checked_mul(rect.height as usize)
        .ok_or(PatchRefusal::Shape)?;
    if rect_bytes.len() != expected {
        return Err(PatchRefusal::Rect);
    }
    if rect.is_empty() {
        return Ok(());
    }
    let x_offset =
        usize::try_from(u64::from(rect.x) * texel_bytes as u64).map_err(|_| PatchRefusal::Shape)?;
    let row_start =
        usize::try_from(u64::from(rect.y) * row_bytes as u64).map_err(|_| PatchRefusal::Shape)?;
    for row in 0..rect.height as usize {
        let Some(frame_row) = frame.get_mut(row_start + row * row_bytes..) else {
            return Err(PatchRefusal::Rect);
        };
        let Some(destination) = frame_row.get_mut(x_offset..x_offset + rect_row_bytes) else {
            return Err(PatchRefusal::Rect);
        };
        let source = &rect_bytes[row * rect_row_bytes..(row + 1) * rect_row_bytes];
        destination.copy_from_slice(source);
    }
    Ok(())
}

/// Why a frame could not be assembled from its seed and its rectangle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PatchRefusal {
    /// The clear payload, the texel width or one of the extents is not a shape
    /// the caller could have got from an admitted attachment.
    Shape,
    /// The frame is not the attachment's own tightly packed extent.
    Frame,
    /// The rectangle's bytes are not the rectangle's own tightly packed extent.
    Rect,
}

impl PatchRefusal {
    pub(crate) const fn detail(self) -> &'static str {
        match self {
            Self::Shape => {
                "the clear payload and the texel width have to state one contained texel"
            }
            Self::Frame => {
                "the frame a trimmed readback patches has to be the attachment's own \
                            tightly packed byte extent"
            }
            Self::Rect => {
                "the copy-out bytes of a trimmed readback have to be the written \
                           rectangle's own tightly packed rows"
            }
        }
    }
}

/// Whether `METAL_API_VULKAN_FULL_READBACK` asks for the pre-increment behaviour.
///
/// Off by default: the switch is the control arm a round can run the *old* path
/// with, so the two rails' frames can be compared byte for byte from the same
/// process (`docs/WRITTEN-RECT-READBACK.md` §4). It is read once, exactly as the
/// phase profile's own switch is.
pub(crate) fn full_readback_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("METAL_API_VULKAN_FULL_READBACK")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("1" | "on" | "ON" | "true" | "yes")
        )
    })
}

/// The rectangle a declared `[x, y, width, height]` states.
const fn rect_of(rect: [u32; 4]) -> WrittenRect {
    WrittenRect {
        x: rect[0],
        y: rect[1],
        width: rect[2],
        height: rect[3],
    }
}

/// Whether `rect` is a non-empty rectangle fully inside `extent`.
///
/// The core contract holds both the viewport and the scissor to the pass's
/// raster (`ViewportOutsideAttachment`, `ScissorOutOfBounds`), so this is the
/// decision's own second line: a directly-constructed request that reaches
/// outside is answered with the whole-attachment readback rather than with a
/// rectangle that would clip the pass's writes.
const fn inside_the_extent(rect: WrittenRect, extent: [u32; 2]) -> bool {
    if rect.width == 0 || rect.height == 0 {
        return false;
    }
    match rect.x.checked_add(rect.width) {
        Some(right) if right <= extent[0] => {}
        _ => return false,
    }
    match rect.y.checked_add(rect.height) {
        Some(bottom) if bottom <= extent[1] => {}
        _ => return false,
    }
    true
}

/// The overlap of two rectangles, which may be empty.
fn intersection(a: WrittenRect, b: WrittenRect) -> WrittenRect {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let right = (a.x + a.width).min(b.x + b.width);
    let bottom = (a.y + a.height).min(b.y + b.height);
    WrittenRect {
        x,
        y,
        width: right.saturating_sub(x),
        height: bottom.saturating_sub(y),
    }
}

/// A clear payload no admitted attachment can state, used by the tests below to
/// spell "a load op whose bytes this rail does not hold".
#[cfg(test)]
fn test_clear() -> metal_api_core::provider::ClearColor {
    metal_api_core::provider::ClearColor::new([0x40, 0x80, 0xc0, 0xff])
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXTENT: [u32; 2] = [8, 8];
    const FULL: [u32; 4] = [0, 0, 8, 8];

    fn load_with_source(
        viewport: [u32; 4],
        scissor: Option<[u32; 4]>,
    ) -> Result<WrittenRect, RectRefusal> {
        written_rect(EXTENT, viewport, scissor, false, LoadOp::Load, true)
    }

    #[test]
    fn the_whole_extent_is_the_rect_of_a_covering_pass() {
        let rect = load_with_source(FULL, None).expect("a covering pass is a rectangle");
        assert_eq!(rect, WrittenRect::whole(EXTENT));
        assert!(rect.is_whole(EXTENT));
        assert_eq!(rect.byte_length(4), Some(256));
    }

    #[test]
    fn a_partial_scissor_narrows_the_readback_to_itself() {
        let rect = load_with_source(FULL, Some([3, 2, 2, 2])).expect("a scissor inside the raster");
        assert_eq!(
            rect,
            WrittenRect {
                x: 3,
                y: 2,
                width: 2,
                height: 2
            }
        );
        assert_eq!(rect.byte_length(4), Some(16));
    }

    #[test]
    fn a_partial_viewport_narrows_the_readback_to_itself() {
        let rect = load_with_source([1, 4, 3, 2], None).expect("a viewport inside the raster");
        assert_eq!(
            rect,
            WrittenRect {
                x: 1,
                y: 4,
                width: 3,
                height: 2
            }
        );
    }

    #[test]
    fn the_narrower_of_two_declared_rects_wins() {
        // The viewport says columns 0..4 and the scissor says columns 2..6: the
        // pass writes only the overlap, which is the superset the readback has
        // to carry.
        let rect = load_with_source([0, 0, 4, 4], Some([2, 2, 4, 4])).expect("an overlap");
        assert_eq!(
            rect,
            WrittenRect {
                x: 2,
                y: 2,
                width: 2,
                height: 2
            }
        );
    }

    #[test]
    fn a_clear_load_is_a_rectangle_and_an_undefined_one_is_not() {
        let clear = written_rect(
            EXTENT,
            FULL,
            None,
            false,
            LoadOp::Clear(test_clear()),
            false,
        )
        .expect("a clear's own payload is the seed this rail holds");
        assert_eq!(clear, WrittenRect::whole(EXTENT));
        assert_eq!(
            written_rect(EXTENT, FULL, None, false, LoadOp::Resident, false),
            Err(RectRefusal::LoadOp)
        );
        assert_eq!(
            written_rect(EXTENT, FULL, None, false, LoadOp::DontCare, false),
            Err(RectRefusal::LoadOp)
        );
        assert_eq!(
            written_rect(EXTENT, FULL, None, false, LoadOp::Load, false),
            Err(RectRefusal::NoSource)
        );
    }

    #[test]
    fn a_multisampled_pass_reads_back_whole() {
        // The resolve writes every texel of the resolve target, so the draw's
        // own rectangle says nothing about what the readback has to carry.
        assert_eq!(
            written_rect(EXTENT, FULL, Some([0, 0, 1, 1]), true, LoadOp::Load, true),
            Err(RectRefusal::Multisample)
        );
    }

    #[test]
    fn a_viewport_outside_the_extent_reads_back_whole() {
        assert_eq!(
            load_with_source([7, 0, 2, 2], None),
            Err(RectRefusal::Viewport)
        );
        assert_eq!(
            load_with_source([0, 0, u32::MAX, u32::MAX], None),
            Err(RectRefusal::Viewport)
        );
        assert_eq!(
            load_with_source([0, 0, 0, 4], None),
            Err(RectRefusal::Viewport)
        );
        assert_eq!(RectRefusal::Viewport.fallback(), ReadbackFallback::Bounds);
    }

    #[test]
    fn a_scissor_outside_the_extent_reads_back_whole() {
        assert_eq!(
            load_with_source(FULL, Some([0, 0, 9, 1])),
            Err(RectRefusal::Scissor)
        );
        assert_eq!(
            load_with_source(FULL, Some([8, 8, 1, 1])),
            Err(RectRefusal::Scissor)
        );
        assert_eq!(
            load_with_source(FULL, Some([0, 0, 4, 0])),
            Err(RectRefusal::Scissor)
        );
        assert_eq!(RectRefusal::Scissor.fallback(), ReadbackFallback::Bounds);
    }

    #[test]
    fn a_disjoint_pair_is_an_empty_rect_and_writes_nothing() {
        // The scissor and the viewport are both inside the raster, but they do
        // not overlap: no fragment can pass both tests.
        let rect =
            load_with_source([0, 0, 2, 2], Some([4, 4, 2, 2])).expect("both rects are legal");
        assert!(rect.is_empty());
        assert_eq!(rect.byte_length(4), Some(0));
    }

    #[test]
    fn the_frame_is_rebuilt_from_the_seed_and_patched_where_the_pass_wrote() {
        // Three rows of four texels, four bytes each: the middle two texels of
        // the last row were written.
        let extent = [4, 3];
        let seed = [0x11_u8; 48];
        let mut frame = seed.to_vec();
        let rect = WrittenRect {
            x: 1,
            y: 2,
            width: 2,
            height: 1,
        };
        // Two texels of the row the copy-out carried, tightly packed.
        let rect_bytes = [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x01, 0x02];
        patch_rect(&mut frame, extent, rect, 4, &rect_bytes).expect("a well-formed rectangle");
        assert_eq!(&frame[..16], &seed[..16], "row 0 is untouched");
        assert_eq!(&frame[16..32], &seed[16..32], "row 1 is untouched");
        assert_eq!(
            &frame[32..36],
            &seed[32..36],
            "the texel left of the rectangle keeps the seed"
        );
        assert_eq!(&frame[36..44], &rect_bytes, "the rectangle's own texels");
        assert_eq!(
            &frame[44..48],
            &seed[44..48],
            "the texel right of the rectangle keeps the seed"
        );
    }

    #[test]
    fn an_empty_rectangle_patches_nothing() {
        let extent = [2, 2];
        let mut frame = vec![0x7f_u8; 16];
        let seed = frame.clone();
        let empty = WrittenRect {
            x: 1,
            y: 1,
            width: 0,
            height: 0,
        };
        patch_rect(&mut frame, extent, empty, 4, &[]).expect("nothing to copy");
        assert_eq!(frame, seed);
    }

    #[test]
    fn a_clear_fill_repeats_one_payload_and_refuses_a_mismatch() {
        let clear = [0x40, 0x80, 0xc0, 0xff];
        assert_eq!(clear_frame(6, 4, &clear), Ok(clear.repeat(6)));
        // A one-byte payload widens like any other, and a frame longer than the
        // pattern is filled to its exact length rather than to a pattern
        // boundary.
        assert_eq!(
            clear_frame(3, 1, &[0xab]).expect("one byte per texel"),
            vec![0xab; 3]
        );
        assert_eq!(
            clear_frame(2000, 4, &clear).expect("eight kilobytes of texels"),
            clear.repeat(2000)
        );
        assert_eq!(clear_frame(6, 4, &clear[..2]), Err(PatchRefusal::Shape));
        assert_eq!(clear_frame(6, 3, &clear), Err(PatchRefusal::Shape));
        assert_eq!(clear_frame(6, 0, &[]), Err(PatchRefusal::Shape));
        assert_eq!(
            clear_frame(u64::MAX, 8, &[0_u8; 8]),
            Err(PatchRefusal::Shape),
            "an extent no usize can hold is refused rather than truncated"
        );
    }

    #[test]
    fn a_patch_whose_shapes_disagree_is_refused_rather_than_truncated() {
        let extent = [4, 2];
        let rect = WrittenRect {
            x: 0,
            y: 0,
            width: 2,
            height: 2,
        };
        let mut frame = vec![0_u8; 32];
        assert_eq!(
            patch_rect(&mut frame, extent, rect, 4, &[0_u8; 8]),
            Err(PatchRefusal::Rect),
            "eight bytes are one row of the rectangle, not both"
        );
        let mut short = vec![0_u8; 4];
        assert_eq!(
            patch_rect(&mut short, extent, rect, 4, &[0_u8; 16]),
            Err(PatchRefusal::Frame)
        );
        assert_eq!(
            patch_rect(&mut frame, extent, rect, 0, &[]),
            Err(PatchRefusal::Shape)
        );
    }
}
