//! Default-off phase timing for one provider submission.
//!
//! `provider.submit` is one call that returns only after the device work it
//! submitted has retired and its host-visible bytes have been read back (the
//! reims rail's `FrameSpan::ProvSubmit` measures exactly that call). That bar is
//! where the guest's draw cost lives, but it is a *total*: planning, resource
//! creation, recording, `vkQueueSubmit`, `vkWaitForFences` and the host-mapped
//! readback are all inside it, and none of them can be told apart from outside
//! this crate.
//!
//! This module splits that total into disjoint bars, driven by
//! `METAL_API_VULKAN_PHASE_PROFILE`:
//!
//! * unset (the default) or any value other than a truthy one: off. Every call
//!   site is a `let _bar = Bar::enter(Phase::X)` that resolves to `None` after
//!   one relaxed load — **no clock read, no lock, no allocation**.
//! * `1` / `on` / `true` / `yes`: on. Each bar reads the monotonic clock twice
//!   and adds to a thread-local table; every
//!   `METAL_API_VULKAN_PHASE_PROFILE_EVERY` (default 256) completed submissions
//!   that thread writes one line to **stderr** (the QEMU process's own stderr,
//!   i.e. the round's boot log):
//!
//!   ```text
//!   PHASE submit n=256 total_us=... admit_us=... plan_us=... pool_us=...
//!   resource_build_us=... record_us=... queue_submit_us=... fence_wait_us=...
//!   fence_wait_idle_n=... fence_wait_idle_us=... fence_wait_blocked_n=...
//!   fence_wait_blocked_us=... fence_wait_timeout_n=... read_updates_us=...
//!   render_total_us=... render_setup_us=... render_record_us=...
//!   render_submit_us=... render_wait_us=... (with render_wait's own
//!   idle/blocked/timeout fields) render_readback_us=... writebacks_us=...
//!   settle_us=... fence_wait_skipped_n=... plan_settle_us=... render_us=...
//!   readback_rect_n=... readback_rect_bytes=... readback_rect_extent_bytes=...
//!   readback_full_n=... readback_full_bytes=... readback_switch_n=...
//!   readback_shape_n=... readback_bounds_n=... readback_whole_n=...
//!   setup_admits_us=... setup_attachments_us=... setup_depth_stencil_us=...
//!   setup_render_pass_us=... setup_textures_us=... setup_stage_buffers_us=...
//!   setup_pipeline_us=... setup_readbacks_us=... setup_inputs_us=...
//!   setup_command_pool_us=... texture_backing_us=... texture_upload_us=...
//!   texture_view_us=... texture_sampler_us=... texture_import_us=...
//!   texture_descriptor_us=...
//!   reuse_hit_n=... reuse_miss_n=...
//!   reuse_mismatch_n=... reuse_unkeyed_n=... reuse_disabled_n=...
//!   pool_hit_n=... pool_miss_n=... pool_disabled_n=... pool_return_n=...
//!   pool_drop_n=...
//!   import_hit_n=... import_miss_n=... import_disabled_n=... import_return_n=...
//!   import_drop_n=...
//!   render_offscreen_n=... render_present_n=...
//!   readback_rect_us=... readback_full_us=... readback_seed_us=...
//!   readback_surfaces_us=... readback_shape_us=... readback_named_us=...
//!   landing_lookup_us=... landing_windows_us=... landing_stage_us=...
//!   landing_record_us=... landing_wait_us=... landing_fetch_us=...
//!   landing_write_us=... landing_release_us=... landing_named_us=...
//!   landing_n=... landing_bytes=...
//!   ```
//!
//! and, beside the disjoint fields, the aggregate readings the nested splits
//! imply: `render_us` (the five `render_*` children), `render_residual_us`
//! (their eight siblings that divide what those five leave unnamed) and
//! `texture_named_us` (the six nested regions inside `setup_textures`). The
//! aggregates are printed beside the fields they aggregate rather than added to
//! them, exactly as `plan_settle_us` already was.
//!
//! Fields are **sums over the line's own window** (`n` submissions), not means,
//! so a reader can add lines together and divide by the summed `n` without
//! weighting error. µs fields carry three decimals, i.e. nanosecond resolution.
//!
//! The `setup_*` fields are the one nested split: they divide `render_setup`
//! itself into the regions a fix would move (the admissions the rail re-runs,
//! the attachment and depth images, the render pass and its framebuffers, the
//! sampled textures, the stage buffers, the pipeline, the readback
//! destinations, the caller-held inputs and the command pool). `render_setup`
//! stays their enclosing bar, so `sum(setup_*) <= render_setup_us` and the
//! difference is the seam between those regions — the plan of the whole setup,
//! charged to the bar that encloses them rather than to one of them.
//!
//! Two further nested splits answer the two questions the first split left
//! open, and neither is part of the disjoint sum:
//!
//! * `texture_view_us`, `texture_sampler_us`, `texture_import_us` and
//!   `texture_descriptor_us` divide what `setup_textures` spent *beside*
//!   `texture_backing_us` and `texture_upload_us` — the image view a pooled
//!   backing did not carry, the samplers, the per-declaration host-pointer
//!   imports of owner windows, and the sampled set's layout, pool, set and
//!   writes. `backing + upload + view + sampler + import + descriptor <=
//!   setup_textures_us`.
//! * `render_resolve_us`, `render_present_us`, `render_publish_us`,
//!   `render_landing_us`, `render_prepare_us`, `render_retain_us`,
//!   `render_land_owner_us` and `render_teardown_us` divide the render half's
//!   residual — what `render_total` cost minus its five children: the outer
//!   loop's resolution and publication around each pass, a landing-only plan
//!   entry, the present rail's own pass, the offscreen rail entry's admissions
//!   and affine index resolution, the input retains, the owner-window landing
//!   that follows a pass, and the pass objects' teardown. `sum(render children)
//!   + sum(render residual) <= render_total_us`.
//!
//! The `readback_*` fields are the one exception in *unit*, not in window: they
//! count the stored attachments this window's submissions published — how many
//! `vkCmdCopyImageToBuffer` stages carried only the pass's written rectangle
//! (`readback_rect_n`) and how many bytes the host then read
//! (`readback_rect_bytes`, versus `readback_rect_extent_bytes`, what those
//! attachments' whole extents occupy), how many carried the whole attachment
//! (`readback_full_n` / `readback_full_bytes`) and which fact sent them there
//! (`readback_switch_n` for the `METAL_API_VULKAN_FULL_READBACK` control arm,
//! `readback_shape_n` for a shape whose seed this rail does not hold,
//! `readback_bounds_n` for a declared rect it cannot prove,
//! `readback_whole_n` for a rectangle that covered the whole attachment anyway).
//! They are window sums like every other field, so the same add-and-divide rule
//! answers "bytes read back per submission" (`docs/WRITTEN-RECT-READBACK.md`).
//!
//! Two more nested splits answer the two bars the first three rounds selected
//! as the next cut, and neither is part of the disjoint sum:
//!
//! * `readback_rect_us`, `readback_full_us` and `readback_surfaces_us` divide
//!   `render_readback` itself — the trimmed arm, the whole-extent arm and the
//!   depth/stencil/stage-buffer copy-outs — with `readback_seed_us` nested
//!   inside the trimmed arm (the rebuild that arm pays and the whole-extent arm
//!   does not). `readback_shape_us` is the decision `plan_readback_regions`
//!   takes, which happens inside `setup_readbacks` rather than inside the
//!   readback bar: the arm is chosen before any device object exists.
//! * `landing_lookup_us`, `landing_windows_us`, `landing_stage_us`,
//!   `landing_record_us`, `landing_wait_us`, `landing_fetch_us`,
//!   `landing_write_us` and `landing_release_us` divide `render_landing` — one
//!   landing-only entry's identity bookkeeping, its window resolution, its
//!   staging objects, the copy's recording and submission, its fence wait, the
//!   host fetch of the copied frame, the write into the owner's pages and the
//!   release of those objects. The printed `landing_named_us` is their sum, and
//!   `landing_n` / `landing_bytes` count what they were spent on.
//!
//! The accumulator is **thread-local**, and a line is emitted by the thread that
//! filled its own window. That is what makes each line self-consistent: with one
//! process-wide table, two threads submitting at once would land their bars in
//! the same window, and the line's `n`, its fields and its enclosing `total`
//! would describe different populations.
//!
//! The bars are deliberately disjoint: `enter`-style nesting would charge the
//! child's time to the parent as well, and then "how much of the total is the
//! readback" would have no answer. `total` encloses the call and `render_total`
//! encloses the render half; every other field is a region inside one of them,
//! so `sum(fields) <= total` and `sum(render children) <= render_total` are
//! identities a reader can check, and the difference is the seam between the
//! bars (function calls, `Arc` clones, the queue lock) — the residual, not a
//! missing bar.
//!
//! A phase may be charged at more than one disjoint region: `settle` covers the
//! completion bookkeeping of both the synchronous and the deferred arm, and
//! `writebacks` covers the mapping and the contract validation of the merged
//! result. Every field is still a sum of disjoint time.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::readback_rect::ReadbackFallback;

/// One bar of the submission profile.
///
/// The discriminant is the slot index, so `Phase::QueueSubmit as usize` and
/// [`PHASE_NAMES`] have to stay in step; `phase_names_cover_every_phase` pins
/// that down the way the draw rail pins its own table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Phase {
    /// The enclosing bar: one whole `submit` call, success or refusal.
    Total,
    /// Epoch check, capability admission and the indirect-command resolve.
    Admit,
    /// Everything that decides *what* will run: the render plan, the heap
    /// placement plan, the registered pipelines, the serial resource pool and
    /// the per-pass dispatch list.
    Plan,
    /// The submission's own inputs: device bindings for every pooled view,
    /// including the owned-byte copies, the staged-lease resolution and the
    /// gathered guest runs, plus the borrow retains that keep them alive.
    Pool,
    /// Device objects for this submission: pipeline objects, buffers, textures,
    /// static samplers, descriptor sets and the indirect buffer.
    ResourceBuild,
    /// Command-buffer recording.
    Record,
    /// Completion fence creation and `vkQueueSubmit`.
    QueueSubmit,
    /// `vkWaitForFences` on the submission's completion fence, reported with an
    /// idle/blocked split — see [`Local::note_fence_wait`].
    FenceWait,
    /// The host-mapped readback of every writable view's bytes.
    ReadUpdates,
    /// The enclosing bar for the render half executed inside the same `submit`
    /// (render/present passes and their own submissions). Like [`Phase::Total`]
    /// it encloses children, so it is an aggregate a reader checks the others
    /// against rather than a bar to add to them.
    RenderTotal,
    /// Everything an offscreen render pass does before it records: the
    /// attachment/pipeline resolution, the device objects, and the readback
    /// destination buffers the pass will copy out into.
    RenderSetup,
    /// Command-buffer recording for one render pass, including the
    /// `vkCmdCopyImageToBuffer` that stages the pass's stored attachments.
    RenderRecord,
    /// The render half's completion fence and `vkQueueSubmit`.
    RenderSubmit,
    /// The render half's `vkWaitForFences`. This is where the pass's own
    /// copy-out executes, so it is device time as much as it is latency.
    RenderWait,
    /// The render half's host-visible readback: the mapped copy of every stored
    /// attachment, the depth/stencil surfaces and the writable stage buffers.
    RenderReadback,
    /// Writeback mapping and the contract validation of the merged result.
    Writebacks,
    /// Completion bookkeeping: the terminal observation, its record insert and
    /// the health synchronisation.
    Settle,
    /// Inside `render_setup`: the admissions the render rail re-runs on the
    /// request — attachment count, the all-discarded rule, the format and
    /// sample-count gates the device is asked about — before the first device
    /// object exists.
    SetupAdmits,
    /// Inside `render_setup`: the colour attachment images (or the resident
    /// targets a pass borrows instead), their backings and views.
    SetupAttachments,
    /// Inside `render_setup`: the depth, stencil or combined depth-stencil
    /// surface, its resolve target and the readback destinations those faces
    /// land in.
    SetupDepthStencil,
    /// Inside `render_setup`: the render pass, the seed render pass a
    /// multisampled `Load` runs first, and both framebuffers.
    SetupRenderPass,
    /// Inside `render_setup`: the sampled textures — their images, uploads,
    /// samplers — and the descriptor set layout, pool and set that bind them.
    SetupTextures,
    /// Inside `render_setup`: the stage buffers, their sets and the layouts
    /// those sets occupy in the pipeline layout.
    SetupStageBuffers,
    /// Inside `render_setup`: the two shader modules, the pipeline layout and
    /// the graphics pipeline itself.
    SetupPipeline,
    /// Inside `render_setup`: the readback plan and one host-visible
    /// destination buffer per stored colour attachment.
    SetupReadbacks,
    /// Inside `render_setup`: the caller-held vertex and index streams, the
    /// previous-byte buffers a `Load` uploads from, and the indirect-command
    /// buffers.
    SetupInputs,
    /// Inside `render_setup`: the command pool and the command buffer
    /// allocated from it.
    SetupCommandPool,
    /// Inside `setup_textures`: the backing one sampled declaration's image
    /// needs — `vkCreateImage`, `vkAllocateMemory` and `vkBindImageMemory`, or
    /// nothing at all when `crate::render_texture_pool` hands one back.
    ///
    /// This bar and [`Phase::TextureUpload`] are *nested* inside
    /// [`Phase::SetupTextures`] rather than beside it: they divide that bar's
    /// own region, so they must never be added to the disjoint sum a reading
    /// checks (`docs/TEXTURE-BACKING-POOL.md`).
    TextureBacking,
    /// Inside `setup_textures`: the texels' own trip into the backing — the
    /// byte arms' `vkMapMemory`, copy and `vkUnmapMemory`. The no-copy and
    /// pass-entry-snapshot arms write nothing here, because their bytes never
    /// exist on the host.
    TextureUpload,
    /// Inside `setup_textures`: the sampled declaration's own image view, when
    /// the backing pool did not hand one over (`vkCreateImageView`).
    TextureView,
    /// Inside `setup_textures`: the sampler one declaration's slot states
    /// (`vkCreateSampler`). A texel-fetch slot creates none and charges nothing.
    TextureSampler,
    /// Inside `setup_textures`: the import of an owner's window as a
    /// host-pointer buffer (`docs/23` §75, R5c) — the no-copy arm's staging
    /// buffer and memory, allocated and bound per declaration per pass.
    TextureImport,
    /// Inside `setup_textures`: the sampled set's own layout, pool, set and
    /// descriptor writes — the state the fragment module reads the textures
    /// through, rebuilt for every pass.
    TextureDescriptor,
    /// Inside the render half's residual (`render_total` minus the five
    /// `render_*` children): the outer loop's per-entry resolution before the
    /// rail is called — the attachment/landing/resident declarations looked up
    /// against the trace's view list, the present target, the executor lock and
    /// the produced-bytes context.
    ///
    /// This bar and the six below it divide what the render half's own split
    /// did not name, so they are *nested* inside [`Phase::RenderTotal`] beside
    /// its five children: `sum(render children) + sum(render residual) <=
    /// render_total_us`, and the difference is the seam that remains.
    RenderResolve,
    /// Inside the render half's residual: the present rail's own pass
    /// (`execute_present_render`) — its request, its objects, its recording,
    /// its readback and its teardown. Present has its own setup and readback
    /// shape rather than the offscreen one the five children were placed for.
    RenderPresent,
    /// Inside the render half's residual: the outer loop's per-entry
    /// publication after the rail returned — the writeback pushes, the resident
    /// and re-kept identities, and the stage-buffer landings.
    RenderPublish,
    /// Inside the render half's residual: one landing-only plan entry
    /// (`land_kept_frame_entry`), which runs in the plan's order instead of a
    /// pass.
    RenderLanding,
    /// Inside the render half's residual: the offscreen rail entry's own work
    /// before the pass executor — the extent/contract admissions the rail
    /// re-runs on the request, the stage-pair validation and the affine index
    /// resolution (which may read an owner's window through the lease channel).
    RenderPrepare,
    /// Inside the render half's residual: the input retains an offscreen pass
    /// takes before its first import (`RenderInputRetains::retain`).
    RenderRetain,
    /// Inside the render half's residual: the owner-window landing that follows
    /// a successful offscreen pass (`land_owner_windows`) — the pass's texels
    /// copied into the guest pages the store named.
    RenderLandOwner,
    /// Inside the render half's residual: the pass objects' teardown
    /// (`OffscreenObjects::drop`) once the fence has proven the device done
    /// with them — the images, views, samplers, descriptor pools, framebuffers,
    /// render passes, readback buffers, fences and command pools one pass held.
    RenderTeardown,
    /// Inside `render_readback`: one stored attachment's *trimmed* frame — the
    /// mapped rectangle copied to the host and the seed rebuilt around it. The
    /// arm the written-rectangle increment exists for
    /// (`docs/WRITTEN-RECT-READBACK.md`).
    ReadbackRect,
    /// Inside `render_readback`: one stored attachment's *whole* extent copied
    /// out of its mapping, the pre-increment shape of the readback
    /// (`docs/WRITTEN-RECT-READBACK.md` §1).
    ReadbackFull,
    /// Inside `readback_rect`: the rebuild itself (`rebuild_frame`) — the seed
    /// the uncovered texels come from (a clear's repeated payload, or the bytes
    /// a `Load` resolved to) and the patch that pins the copied rectangle back
    /// into it.
    ReadbackSeed,
    /// Inside `render_readback`: the depth, stencil and writable stage-buffer
    /// copy-outs, which follow the colour attachments through the same
    /// channel (`research/docs/23` §3.3).
    ReadbackSurfaces,
    /// The readback *decision* itself (`plan_readback_regions`): which shapes
    /// keep the written-rectangle arm and which fall back to the whole
    /// attachment. This one is nested inside `setup_readbacks`, not inside
    /// `render_readback`, because the decision is taken before any device
    /// object exists (`docs/WRITTEN-RECT-READBACK.md` §2).
    ReadbackShape,
    /// Inside `render_landing`: the identity bookkeeping before the copy — the
    /// kept frame's target looked up in the resident registry and the landing
    /// view's declaration found in the trace's own view list.
    LandingLookup,
    /// Inside `render_landing`: the owner windows the frame will be written
    /// into, resolved against the lease channel (the single borrowed window or
    /// the ordered guest-run list, and the extent check both arms take).
    LandingWindows,
    /// Inside `render_landing`: the landing's own staging objects — the
    /// host-visible `TRANSFER_DST` buffer and its memory, the mapping, the
    /// command pool, the command buffer and the completion fence.
    LandingStage,
    /// Inside `render_landing`: recording the image→buffer copy (its barrier,
    /// its region and the availability barrier behind it) and submitting it.
    LandingRecord,
    /// Inside `render_landing`: `vkWaitForFences` for the landing copy.
    LandingWait,
    /// Inside `render_landing`: the host fetch of the copied frame — the
    /// mapping read into the `Vec` the owner's windows are written from. This
    /// is the part a landing shares with the readback channel.
    LandingFetch,
    /// Inside `render_landing`: the write into the owner's live pages
    /// (`AttachmentLanding::land`), which is what the copy exists for.
    LandingWrite,
    /// Inside `render_landing`: releasing the staging objects — the fence, the
    /// command pool, the mapping, the buffer and its memory.
    LandingRelease,
}

const PHASE_COUNT: usize = Phase::LandingRelease as usize + 1;

/// The printed field name of each phase, in slot order.
const PHASE_NAMES: [&str; PHASE_COUNT] = [
    "total",
    "admit",
    "plan",
    "pool",
    "resource_build",
    "record",
    "queue_submit",
    "fence_wait",
    "read_updates",
    "render_total",
    "render_setup",
    "render_record",
    "render_submit",
    "render_wait",
    "render_readback",
    "writebacks",
    "settle",
    "setup_admits",
    "setup_attachments",
    "setup_depth_stencil",
    "setup_render_pass",
    "setup_textures",
    "setup_stage_buffers",
    "setup_pipeline",
    "setup_readbacks",
    "setup_inputs",
    "setup_command_pool",
    "texture_backing",
    "texture_upload",
    "texture_view",
    "texture_sampler",
    "texture_import",
    "texture_descriptor",
    "render_resolve",
    "render_present",
    "render_publish",
    "render_landing",
    "render_prepare",
    "render_retain",
    "render_land_owner",
    "render_teardown",
    "readback_rect",
    "readback_full",
    "readback_seed",
    "readback_surfaces",
    "readback_shape",
    "landing_lookup",
    "landing_windows",
    "landing_stage",
    "landing_record",
    "landing_wait",
    "landing_fetch",
    "landing_write",
    "landing_release",
];

/// The slots the printed `plan_settle_us` field aggregates: the CPU-only matter
/// around the device work — what will run (`plan`), onto what (`pool`), and what
/// its completion left to do (`settle`).
const PLAN_SETTLE_SLOTS: [usize; 3] = [
    Phase::Plan as usize,
    Phase::Pool as usize,
    Phase::Settle as usize,
];

/// The slots the printed `render_us` field aggregates: the disjoint regions of
/// the render half. `render_total` is their enclosing bar and is printed beside
/// them rather than added to them.
const RENDER_SLOTS: [usize; 5] = [
    Phase::RenderSetup as usize,
    Phase::RenderRecord as usize,
    Phase::RenderSubmit as usize,
    Phase::RenderWait as usize,
    Phase::RenderReadback as usize,
];

/// The slots that divide what [`RENDER_SLOTS`] leaves unnamed — the render
/// half's residual. Like the `setup_*` fields they are a nested split rather
/// than a second disjoint set: `sum(RENDER_SLOTS) + sum(RENDER_RESIDUAL_SLOTS)
/// <= render_total`, and the difference is the seam between the bars.
///
/// The list is a reading aid rather than a printed field; a tool that checks
/// the identity reads it from here.
const RENDER_RESIDUAL_SLOTS: [usize; 8] = [
    Phase::RenderResolve as usize,
    Phase::RenderPresent as usize,
    Phase::RenderPublish as usize,
    Phase::RenderLanding as usize,
    Phase::RenderPrepare as usize,
    Phase::RenderRetain as usize,
    Phase::RenderLandOwner as usize,
    Phase::RenderTeardown as usize,
];

/// The nested split of `setup_textures`, beside the two upload/backing bars it
/// already had: `sum(setup_textures children) <= setup_textures_us`.
const TEXTURE_SLOTS: [usize; 6] = [
    Phase::TextureBacking as usize,
    Phase::TextureUpload as usize,
    Phase::TextureView as usize,
    Phase::TextureSampler as usize,
    Phase::TextureImport as usize,
    Phase::TextureDescriptor as usize,
];

/// The slots inside `render_readback` that divide the frames it publishes: the
/// trimmed arm, the whole-extent arm and the depth/stencil/stage-buffer
/// copy-outs. `readback_seed` is deliberately not a member — it is nested
/// inside the trimmed arm, so adding it would count the same microseconds
/// twice. The printed `readback_named_us` is this set's sum.
const READBACK_ARM_SLOTS: [usize; 3] = [
    Phase::ReadbackRect as usize,
    Phase::ReadbackFull as usize,
    Phase::ReadbackSurfaces as usize,
];

/// The slots inside `render_landing` that divide one landing-only entry: the
/// identity bookkeeping, the window resolution, the staging objects, the
/// recording and submission, the fence wait, the host fetch, the write into the
/// owner's pages and the release of the staging objects. Every slot is a region
/// of its own — none encloses another — so the set's sum, printed as
/// `landing_named_us`, is bounded by `render_landing_us` and the difference is
/// the seam between them.
const LANDING_SLOTS: [usize; 8] = [
    Phase::LandingLookup as usize,
    Phase::LandingWindows as usize,
    Phase::LandingStage as usize,
    Phase::LandingRecord as usize,
    Phase::LandingWait as usize,
    Phase::LandingFetch as usize,
    Phase::LandingWrite as usize,
    Phase::LandingRelease as usize,
];

/// The bars that are a fence wait, and therefore carry the idle/blocked split.
const WAIT_SLOTS: [usize; 2] = [Phase::FenceWait as usize, Phase::RenderWait as usize];

/// Submissions per emitted line, and therefore the `n=` field.
const EVERY_DEFAULT: u64 = 256;

/// A fence wait that returns inside this many nanoseconds is counted as an
/// immediate return rather than as device latency.
///
/// `vkWaitForFences` on a signaled fence costs a driver call (single-digit
/// microseconds here); a fence the queue has not retired cannot return before
/// the device work has run. The two populations are far apart, so the threshold
/// classifies them rather than splitting one population — and both buckets are
/// printed, so a reader who disagrees with the cut can still see the raw sums
/// (and move it with `METAL_API_VULKAN_PHASE_IDLE_NS`).
const IDLE_NS_DEFAULT: u64 = 100_000;

/// The readback region one stored attachment was published through
/// (`docs/WRITTEN-RECT-READBACK.md` §2), as an emitted line counts it.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ReadbackRegion {
    /// Only the pass's written rectangle left the device. `bytes` is what the
    /// host copied out of the mapping and `extent_bytes` what the attachment's
    /// whole extent occupies — the cost the same attachment had before the
    /// increment.
    Rect { bytes: u64, extent_bytes: u64 },
    /// The whole attachment left the device.
    Full { bytes: u64 },
    /// The whole attachment left the device because this rail could not prove a
    /// narrower rectangle: the control switch, the shape's seed, the declared
    /// bounds, or a rectangle that covered the whole attachment anyway.
    Fallback(ReadbackFallback),
}

/// One window's readback regions, in the same units the counter line prints.
#[derive(Default)]
struct ReadbackCounts {
    rect_n: u64,
    rect_bytes: u64,
    rect_extent_bytes: u64,
    full_n: u64,
    full_bytes: u64,
    switch_n: u64,
    shape_n: u64,
    bounds_n: u64,
    whole_n: u64,
}

impl ReadbackCounts {
    fn note(&mut self, region: ReadbackRegion) {
        match region {
            ReadbackRegion::Rect {
                bytes,
                extent_bytes,
            } => {
                self.rect_n += 1;
                self.rect_bytes += bytes;
                self.rect_extent_bytes += extent_bytes;
            }
            ReadbackRegion::Full { bytes } => {
                self.full_n += 1;
                self.full_bytes += bytes;
            }
            ReadbackRegion::Fallback(fallback) => match fallback {
                ReadbackFallback::Switch => self.switch_n += 1,
                ReadbackFallback::Shape => self.shape_n += 1,
                ReadbackFallback::Bounds => self.bounds_n += 1,
                ReadbackFallback::Whole => self.whole_n += 1,
            },
        }
    }
}

/// Count one stored attachment's readback region for the emitting thread's
/// window. A no-op — one relaxed load — while the profile is off, exactly as a
/// bar is.
#[inline]
pub(crate) fn note_readback(region: ReadbackRegion) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| local.borrow_mut().readback.note(region));
}

/// Count one landed kept frame for the emitting thread's window, by the bytes
/// the owner's pages received.
///
/// The landing's bars are a per-entry reading; a round that wants "what does
/// one landing cost" and "how big is one landing" divides them by this count
/// and this byte sum, exactly as the `readback_*` counters do for a stored
/// attachment.
#[inline]
pub(crate) fn note_landing(bytes: u64) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        local.landing_n += 1;
        local.landing_bytes += bytes;
    });
}

/// Count one offscreen pass's use of the shape-decided render objects
/// (`crate::render_setup_reuse`) for the emitting thread's window.
///
/// The five outcomes partition every pass that reaches the mechanism: it was
/// served from the cache, it built its objects and cached them, a digest
/// collision was refused by the comparison, it could not state a key at all,
/// or the switch was off. A round that reads `reuse_hit_n=0` can tell which of
/// the other four it is looking at, which is the whole reason they are counted
/// apart rather than as one "not a hit".
#[inline]
pub(crate) fn note_reuse(outcome: crate::render_setup_reuse::Outcome) {
    if !enabled() {
        return;
    }
    use crate::render_setup_reuse::Outcome;
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        match outcome {
            Outcome::Hit => local.reuse_hit_n += 1,
            Outcome::Miss => local.reuse_miss_n += 1,
            Outcome::Mismatch => local.reuse_mismatch_n += 1,
            Outcome::Unkeyed => local.reuse_unkeyed_n += 1,
            Outcome::Disabled => local.reuse_disabled_n += 1,
        }
    });
}

/// Count one sampled declaration's use of the pooled backing
/// (`crate::render_texture_pool`) for the emitting thread's window.
///
/// The four outcomes partition every declaration that reaches the mechanism:
/// the pool held a backing of its shape and handed it over, it held none and
/// the declaration built its own, the switch was off, or a pass handed a
/// backing back and the pool destroyed it instead of holding it (the switch
/// went off mid-pass, or the shape does not fit the cap). A round that reads
/// `pool_hit_n=0` can tell which of the others it is looking at.
#[inline]
pub(crate) fn note_texture_pool(outcome: crate::render_texture_pool::PoolOutcome) {
    if !enabled() {
        return;
    }
    use crate::render_texture_pool::PoolOutcome;
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        match outcome {
            PoolOutcome::Hit => local.pool_hit_n += 1,
            PoolOutcome::Miss => local.pool_miss_n += 1,
            PoolOutcome::Disabled => local.pool_disabled_n += 1,
            PoolOutcome::Returned => local.pool_return_n += 1,
            PoolOutcome::Dropped => local.pool_drop_n += 1,
        }
    });
}

/// Count one sampled declaration's use of the pooled owner-window import
/// (`crate::render_import_pool`) for the emitting thread's window.
///
/// The five outcomes partition every declaration that reaches the mechanism:
/// the pool held this window's import and handed it over, it held none and the
/// declaration imported the range itself, the switch was off, or a completed
/// pass handed an import back and the pool kept it (or destroyed it instead).
#[inline]
pub(crate) fn note_import_pool(outcome: crate::render_import_pool::ImportOutcome) {
    if !enabled() {
        return;
    }
    use crate::render_import_pool::ImportOutcome;
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        match outcome {
            ImportOutcome::Hit => local.import_hit_n += 1,
            ImportOutcome::Miss => local.import_miss_n += 1,
            ImportOutcome::Disabled => local.import_disabled_n += 1,
            ImportOutcome::Returned => local.import_return_n += 1,
            ImportOutcome::Dropped => local.import_drop_n += 1,
        }
    });
}

/// Which shape one executed render pass had: the offscreen rail the five
/// `render_*` children were placed for, or the present rail, whose own setup,
/// recording and readback are the residual's [`Phase::RenderPresent`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderShape {
    Offscreen,
    Present,
}

/// Count one executed render pass for the emitting thread's window, by shape.
///
/// A round needs this beside the bars: the two shapes are not one population,
/// so "what does a pass cost" is only readable when the reader can see how many
/// of each the window carried.
#[inline]
pub(crate) fn note_render_shape(shape: RenderShape) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        match shape {
            RenderShape::Offscreen => local.render_offscreen_n += 1,
            RenderShape::Present => local.render_present_n += 1,
        }
    });
}

/// One thread's window of the profile.
struct Local {
    ns: [u64; PHASE_COUNT],
    calls: [u64; PHASE_COUNT],
    wait_idle_ns: [u64; PHASE_COUNT],
    wait_idle_calls: [u64; PHASE_COUNT],
    wait_blocked_ns: [u64; PHASE_COUNT],
    wait_blocked_calls: [u64; PHASE_COUNT],
    wait_timeout_calls: [u64; PHASE_COUNT],
    /// Waits with no fence to wait for (`PendingExecution::wait`'s `!submitted`
    /// early return): the only waits that are exactly free.
    fence_skipped_calls: u64,
    /// `total` bars closed since the last emitted line.
    window: u64,
    /// The stored attachments this window's submissions read back, by region.
    readback: ReadbackCounts,
    /// The offscreen passes this window's submissions ran, by reuse outcome.
    reuse_hit_n: u64,
    reuse_miss_n: u64,
    reuse_mismatch_n: u64,
    reuse_unkeyed_n: u64,
    reuse_disabled_n: u64,
    /// The sampled declarations this window's passes made, by what
    /// `crate::render_texture_pool` answered.
    pool_hit_n: u64,
    pool_miss_n: u64,
    pool_disabled_n: u64,
    pool_return_n: u64,
    pool_drop_n: u64,
    /// The sampled declarations this window's passes made, by what
    /// `crate::render_import_pool` answered.
    import_hit_n: u64,
    import_miss_n: u64,
    import_disabled_n: u64,
    import_return_n: u64,
    import_drop_n: u64,
    /// The render passes this window's submissions executed, by shape. The two
    /// do not share a cost shape, so a bar reading has to name its population.
    render_offscreen_n: u64,
    render_present_n: u64,
    /// The kept frames this window's landing-only entries delivered, and the
    /// bytes the owner's pages received.
    landing_n: u64,
    landing_bytes: u64,
    /// The readback staging buffers this window's passes allocated, by which
    /// memory type the selection took.
    staging_cached_n: u64,
    staging_plain_n: u64,
}

/// An empty window, spelled out because the slot tables are longer than the
/// largest array `Default` is derived over (32): every table is zeroed here on
/// purpose, so a phase whose reading must not be inherited has a slot to be
/// zero in.
impl Default for Local {
    fn default() -> Self {
        Self {
            ns: [0; PHASE_COUNT],
            calls: [0; PHASE_COUNT],
            wait_idle_ns: [0; PHASE_COUNT],
            wait_idle_calls: [0; PHASE_COUNT],
            wait_blocked_ns: [0; PHASE_COUNT],
            wait_blocked_calls: [0; PHASE_COUNT],
            wait_timeout_calls: [0; PHASE_COUNT],
            fence_skipped_calls: 0,
            window: 0,
            readback: ReadbackCounts::default(),
            reuse_hit_n: 0,
            reuse_miss_n: 0,
            reuse_mismatch_n: 0,
            reuse_unkeyed_n: 0,
            reuse_disabled_n: 0,
            pool_hit_n: 0,
            pool_miss_n: 0,
            pool_disabled_n: 0,
            pool_return_n: 0,
            pool_drop_n: 0,
            import_hit_n: 0,
            import_miss_n: 0,
            import_disabled_n: 0,
            import_return_n: 0,
            import_drop_n: 0,
            render_offscreen_n: 0,
            render_present_n: 0,
            landing_n: 0,
            landing_bytes: 0,
            staging_cached_n: 0,
            staging_plain_n: 0,
        }
    }
}

impl Local {
    #[inline]
    fn charge(&mut self, phase: Phase, ns: u64) {
        let slot = phase as usize;
        self.ns[slot] += ns;
        self.calls[slot] += 1;
    }

    /// Bank one `vkWaitForFences`, classified by how long the call itself took.
    ///
    /// The classification is the answer to "how much of the wait is the device
    /// and how much is the driver call": a wait that returns immediately means
    /// the queue had already retired the work, and the microseconds that remain
    /// here buy no GPU time.
    #[inline]
    fn note_fence_wait(&mut self, phase: Phase, ns: u64, timed_out: bool) {
        self.charge(phase, ns);
        let slot = phase as usize;
        if ns < idle_ns() {
            self.wait_idle_ns[slot] += ns;
            self.wait_idle_calls[slot] += 1;
        } else {
            self.wait_blocked_ns[slot] += ns;
            self.wait_blocked_calls[slot] += 1;
        }
        if timed_out {
            self.wait_timeout_calls[slot] += 1;
        }
    }

    /// One submission closed. Emits the window's line when it is full.
    ///
    /// Banked from the enclosing bar's `Drop`, so the submission the line is
    /// closed by is already inside it: children drop before their parent, and
    /// counting at entry would publish a line whose `total` was missing the
    /// submission whose fields it reports.
    fn note_total(&mut self, ns: u64) {
        self.charge(Phase::Total, ns);
        self.window += 1;
        if self.window < every() {
            return;
        }
        let n = self.window;
        let mut fields = String::with_capacity(600);
        let mut plan_settle_ns = 0u64;
        let mut render_ns = 0u64;
        let mut render_residual_ns = 0u64;
        let mut texture_named_ns = 0u64;
        let mut readback_named_ns = 0u64;
        let mut landing_named_ns = 0u64;
        for (slot, name) in PHASE_NAMES.iter().enumerate() {
            let ns = std::mem::take(&mut self.ns[slot]);
            self.calls[slot] = 0;
            if PLAN_SETTLE_SLOTS.contains(&slot) {
                plan_settle_ns += ns;
            }
            if RENDER_SLOTS.contains(&slot) {
                render_ns += ns;
            }
            if RENDER_RESIDUAL_SLOTS.contains(&slot) {
                render_residual_ns += ns;
            }
            if TEXTURE_SLOTS.contains(&slot) {
                texture_named_ns += ns;
            }
            if READBACK_ARM_SLOTS.contains(&slot) {
                readback_named_ns += ns;
            }
            if LANDING_SLOTS.contains(&slot) {
                landing_named_ns += ns;
            }
            if WAIT_SLOTS.contains(&slot) {
                let idle_calls = std::mem::take(&mut self.wait_idle_calls[slot]);
                let idle_us = micros(std::mem::take(&mut self.wait_idle_ns[slot]));
                let blocked_calls = std::mem::take(&mut self.wait_blocked_calls[slot]);
                let blocked_us = micros(std::mem::take(&mut self.wait_blocked_ns[slot]));
                let timed_out = std::mem::take(&mut self.wait_timeout_calls[slot]);
                fields.push_str(&format!(
                    " {name}_us={:.3} {name}_idle_n={idle_calls} {name}_idle_us={idle_us:.3} \
                     {name}_blocked_n={blocked_calls} {name}_blocked_us={blocked_us:.3} \
                     {name}_timeout_n={timed_out}",
                    micros(ns)
                ));
                continue;
            }
            fields.push_str(&format!(" {name}_us={:.3}", micros(ns)));
        }
        let skipped = std::mem::take(&mut self.fence_skipped_calls);
        let readback = std::mem::take(&mut self.readback);
        let reuse_hit_n = std::mem::take(&mut self.reuse_hit_n);
        let reuse_miss_n = std::mem::take(&mut self.reuse_miss_n);
        let reuse_mismatch_n = std::mem::take(&mut self.reuse_mismatch_n);
        let reuse_unkeyed_n = std::mem::take(&mut self.reuse_unkeyed_n);
        let reuse_disabled_n = std::mem::take(&mut self.reuse_disabled_n);
        let pool_hit_n = std::mem::take(&mut self.pool_hit_n);
        let pool_miss_n = std::mem::take(&mut self.pool_miss_n);
        let pool_disabled_n = std::mem::take(&mut self.pool_disabled_n);
        let pool_return_n = std::mem::take(&mut self.pool_return_n);
        let pool_drop_n = std::mem::take(&mut self.pool_drop_n);
        let import_hit_n = std::mem::take(&mut self.import_hit_n);
        let import_miss_n = std::mem::take(&mut self.import_miss_n);
        let import_disabled_n = std::mem::take(&mut self.import_disabled_n);
        let import_return_n = std::mem::take(&mut self.import_return_n);
        let import_drop_n = std::mem::take(&mut self.import_drop_n);
        let render_offscreen_n = std::mem::take(&mut self.render_offscreen_n);
        let render_present_n = std::mem::take(&mut self.render_present_n);
        let landing_n = std::mem::take(&mut self.landing_n);
        let landing_bytes = std::mem::take(&mut self.landing_bytes);
        let staging_cached_n = std::mem::take(&mut self.staging_cached_n);
        let staging_plain_n = std::mem::take(&mut self.staging_plain_n);
        self.window = 0;
        let plan_settle_us = micros(plan_settle_ns);
        let render_us = micros(render_ns);
        let render_residual_us = micros(render_residual_ns);
        let texture_named_us = micros(texture_named_ns);
        let readback_named_us = micros(readback_named_ns);
        let landing_named_us = micros(landing_named_ns);
        eprintln!(
            "PHASE submit n={n}{fields} fence_wait_skipped_n={skipped} \
             plan_settle_us={plan_settle_us:.3} render_us={render_us:.3} \
             render_residual_us={render_residual_us:.3} \
             texture_named_us={texture_named_us:.3} \
             readback_named_us={readback_named_us:.3} \
             landing_named_us={landing_named_us:.3} \
             readback_rect_n={} readback_rect_bytes={} readback_rect_extent_bytes={} \
             readback_full_n={} readback_full_bytes={} readback_switch_n={} \
             readback_shape_n={} readback_bounds_n={} readback_whole_n={} \
             reuse_hit_n={reuse_hit_n} reuse_miss_n={reuse_miss_n} \
             reuse_mismatch_n={reuse_mismatch_n} reuse_unkeyed_n={reuse_unkeyed_n} \
             reuse_disabled_n={reuse_disabled_n} pool_hit_n={pool_hit_n} \
             pool_miss_n={pool_miss_n} pool_disabled_n={pool_disabled_n} \
             pool_return_n={pool_return_n} pool_drop_n={pool_drop_n} \
             import_hit_n={import_hit_n} import_miss_n={import_miss_n} \
             import_disabled_n={import_disabled_n} import_return_n={import_return_n} \
             import_drop_n={import_drop_n} \
             render_offscreen_n={render_offscreen_n} \
             render_present_n={render_present_n} \
             landing_n={landing_n} landing_bytes={landing_bytes} \
             staging_cached_n={staging_cached_n} staging_plain_n={staging_plain_n}",
            readback.rect_n,
            readback.rect_bytes,
            readback.rect_extent_bytes,
            readback.full_n,
            readback.full_bytes,
            readback.switch_n,
            readback.shape_n,
            readback.bounds_n,
            readback.whole_n,
        );
    }
}

thread_local! {
    /// One window per submitting thread. A thread that never submits never
    /// touches this, and a thread that does gets a table only its own bars
    /// write — which is what lets a line's `n` and its fields be the same
    /// population.
    static LOCAL: RefCell<Local> = RefCell::new(Local::default());
}

#[inline]
fn micros(ns: u64) -> f64 {
    ns as f64 / 1_000.0
}

/// Count one readback staging buffer's memory selection
/// (`crate::readback_memory`) for the emitting thread's window.
///
/// The two outcomes partition every staging allocation: it took the device's
/// host-cached type, or it fell back to the first host-visible one (no cached
/// type, or the switch's control arm). A round that reads a small
/// `staging_cached_n` can tell which of the two it is looking at from the
/// `STAGING readback memory` line the process prints once.
#[inline]
pub(crate) fn note_staging_memory(cached: bool) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        if cached {
            local.staging_cached_n += 1;
        } else {
            local.staging_plain_n += 1;
        }
    });
}

/// Whether the profile is on, read once from the process environment.
#[inline]
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_VULKAN_PHASE_PROFILE")
                .ok()
                .as_deref(),
        )
    })
}

fn every() -> u64 {
    static EVERY: OnceLock<u64> = OnceLock::new();
    *EVERY.get_or_init(|| {
        parse_env_u64("METAL_API_VULKAN_PHASE_PROFILE_EVERY")
            .filter(|every| *every > 0)
            .unwrap_or(EVERY_DEFAULT)
    })
}

fn idle_ns() -> u64 {
    static IDLE: OnceLock<u64> = OnceLock::new();
    *IDLE.get_or_init(|| parse_env_u64("METAL_API_VULKAN_PHASE_IDLE_NS").unwrap_or(IDLE_NS_DEFAULT))
}

/// The enable word, read exactly as a round's launcher spells it.
fn parse_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1" | "on" | "ON" | "true" | "yes")
    )
}

fn parse_env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse().ok()
}

/// A running bar, banked when it drops.
///
/// `None` when the profile is off, which keeps every call site a
/// `let _bar = Bar::enter(..)` with no branch in it: the switch is read once
/// here rather than at each bracket.
pub(crate) struct Bar {
    slot: Phase,
    started: Instant,
}

impl Bar {
    #[inline]
    pub(crate) fn enter(phase: Phase) -> Option<Self> {
        if !enabled() {
            return None;
        }
        Some(Self {
            slot: phase,
            started: Instant::now(),
        })
    }

    #[inline]
    pub(crate) fn enter_fence_wait(phase: Phase) -> Option<FenceWaitBar> {
        if !enabled() {
            return None;
        }
        Some(FenceWaitBar {
            phase,
            started: Instant::now(),
            timed_out: false,
        })
    }
}

impl Drop for Bar {
    #[inline]
    fn drop(&mut self) {
        let phase = self.slot;
        let ns = elapsed_ns(self.started.elapsed());
        LOCAL.with(|local| {
            let mut local = local.borrow_mut();
            if phase == Phase::Total {
                local.note_total(ns);
            } else {
                local.charge(phase, ns);
            }
        });
    }
}

/// The `vkWaitForFences` bar: same shape as [`Bar`], but it reports the idle /
/// blocked split instead of a plain charge.
pub(crate) struct FenceWaitBar {
    phase: Phase,
    started: Instant,
    timed_out: bool,
}

impl Drop for FenceWaitBar {
    #[inline]
    fn drop(&mut self) {
        let ns = elapsed_ns(self.started.elapsed());
        let phase = self.phase;
        let timed_out = self.timed_out;
        LOCAL.with(|local| local.borrow_mut().note_fence_wait(phase, ns, timed_out));
    }
}

impl FenceWaitBar {
    /// The driver answered with nothing yet (`TIMEOUT` / `NOT_READY`), so the
    /// caller may retry. Counted apart from the retired waits because a retry
    /// loop is a different finding from a slow queue.
    #[inline]
    pub(crate) fn mark_timed_out(&mut self) {
        self.timed_out = true;
    }
}

/// A wait with no fence behind it: exactly free, and exactly countable.
#[inline]
pub(crate) fn note_fence_wait_skipped() {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| local.borrow_mut().fence_skipped_calls += 1);
}

#[inline]
fn elapsed_ns(elapsed: Duration) -> u64 {
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The printed table and the phase enum must agree slot for slot: a missing
    /// name is lost time and a shifted name is time charged to a neighbour,
    /// which reads like an answer and is worse than no reading at all.
    #[test]
    fn phase_names_cover_every_phase() {
        assert_eq!(PHASE_NAMES.len(), PHASE_COUNT);
        assert_eq!(PHASE_NAMES[Phase::Total as usize], "total");
        assert_eq!(PHASE_NAMES[Phase::FenceWait as usize], "fence_wait");
        for (slot, name) in PHASE_NAMES.iter().enumerate() {
            assert!(!name.is_empty(), "slot {slot} has no field name");
            assert!(
                PHASE_NAMES
                    .iter()
                    .take(slot)
                    .all(|previous| previous != name),
                "duplicate field name {name}"
            );
        }
        assert!(PLAN_SETTLE_SLOTS.iter().all(|slot| *slot < PHASE_COUNT));
    }

    /// The two nested splits name regions their enclosing bar states, and no
    /// slot of either is the enclosing bar or a sibling of it: a slip here
    /// would make the identity a reader checks (`children + residual <=
    /// render_total`, `texture children <= setup_textures`) untrue without
    /// saying so.
    #[test]
    fn the_nested_splits_stay_inside_their_enclosing_bar() {
        assert_eq!(PHASE_NAMES[Phase::RenderResolve as usize], "render_resolve");
        assert_eq!(
            PHASE_NAMES[Phase::RenderTeardown as usize],
            "render_teardown"
        );
        assert_eq!(
            PHASE_NAMES[Phase::TextureDescriptor as usize],
            "texture_descriptor"
        );
        for slot in RENDER_RESIDUAL_SLOTS {
            assert_ne!(slot, Phase::RenderTotal as usize);
            assert!(!RENDER_SLOTS.contains(&slot));
        }
        for slot in TEXTURE_SLOTS {
            assert_ne!(slot, Phase::SetupTextures as usize);
        }
        // Both splits are named apart from each other as well: a slot in both
        // would be charged to two regions that a reader would then add.
        for slot in RENDER_RESIDUAL_SLOTS {
            assert!(!TEXTURE_SLOTS.contains(&slot));
        }
        // The readback and landing splits (the sp7 round) follow the same two
        // rules, and the trimmed arm's seed rebuild is deliberately *not* a
        // member of the arm set: it runs inside the trimmed arm, so adding it
        // there would count those microseconds twice.
        for slot in READBACK_ARM_SLOTS {
            assert_ne!(slot, Phase::RenderReadback as usize);
            assert_ne!(slot, Phase::ReadbackSeed as usize);
            assert!(!RENDER_SLOTS.contains(&slot));
        }
        for slot in LANDING_SLOTS {
            assert_ne!(slot, Phase::RenderLanding as usize);
            assert!(!RENDER_RESIDUAL_SLOTS.contains(&slot));
            assert!(!RENDER_SLOTS.contains(&slot));
            assert!(!READBACK_ARM_SLOTS.contains(&slot));
        }
        assert_eq!(
            PHASE_NAMES[Phase::LandingLookup as usize],
            "landing_lookup",
            "the landing split's first slot is the entry's own lookup"
        );
        assert_eq!(
            PHASE_NAMES[Phase::ReadbackRect as usize],
            "readback_rect",
            "the readback split's first slot is the trimmed arm"
        );
    }

    /// Off by default: a process without the variable must not read a clock or
    /// reach the accumulator from any call site.
    #[test]
    fn the_profile_is_off_unless_the_variable_says_otherwise() {
        for value in [
            None,
            Some(""),
            Some("0"),
            Some("off"),
            Some("no"),
            Some("2"),
        ] {
            assert!(
                !parse_enabled(value),
                "{value:?} must not enable the profile"
            );
        }
        for value in ["1", "on", "ON", "true", "yes", " 1 "] {
            assert!(
                parse_enabled(Some(value)),
                "{value} must enable the profile"
            );
        }
    }

    /// The buckets have to partition the wait: their sum is the field, or a
    /// reader cannot tell device latency from driver-call overhead.
    #[test]
    fn the_fence_wait_buckets_partition_the_wait() {
        let mut local = Local::default();
        local.note_fence_wait(Phase::FenceWait, 1_000, false);
        local.note_fence_wait(Phase::FenceWait, IDLE_NS_DEFAULT, false);
        local.note_fence_wait(Phase::FenceWait, 5_000_000, true);
        local.note_fence_wait(Phase::RenderWait, 7, false);
        let slot = Phase::FenceWait as usize;
        let total = local.ns[Phase::FenceWait as usize];
        assert_eq!(
            total,
            local.wait_idle_ns[slot] + local.wait_blocked_ns[slot]
        );
        assert_eq!(total, 1_000 + IDLE_NS_DEFAULT + 5_000_000);
        assert_eq!(local.wait_idle_calls[slot], 1);
        assert_eq!(local.wait_blocked_calls[slot], 2);
        assert_eq!(local.wait_timeout_calls[slot], 1);
        assert_eq!(local.calls[slot], 3);
        // Each wait bar keeps its own buckets: a render wait must not land in
        // the submission-level wait's idle reading.
        let render = Phase::RenderWait as usize;
        assert_eq!(local.ns[render], 7);
        assert_eq!(local.wait_idle_calls[render], 1);
        assert_eq!(local.calls[render], 1);
    }

    /// A window drains on the `total` bar that fills it, and a drained window
    /// starts empty: a line whose counters kept the previous window's time
    /// would double a phase in the table a reader is summing.
    #[test]
    fn a_window_closes_on_the_total_bar_that_fills_it() {
        let mut local = Local::default();
        let window = every();
        for _ in 0..window {
            local.charge(Phase::Plan, 10);
            local.note_total(100);
        }
        assert_eq!(local.window, 0, "the window must have been drained");
        assert_eq!(local.ns[Phase::Total as usize], 0);
        assert_eq!(local.ns[Phase::Plan as usize], 0);
        local.charge(Phase::Plan, 7);
        assert_eq!(local.ns[Phase::Plan as usize], 7);
    }
}
