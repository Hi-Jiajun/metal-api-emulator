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
//!   plan_resources_us=...
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
//!   buffer_hit_n=... buffer_miss_n=... buffer_disabled_n=... buffer_return_n=...
//!   buffer_drop_n=...
//!   compute_buffer_hit_n=... compute_buffer_miss_n=...
//!   compute_buffer_disabled_n=... compute_buffer_return_n=...
//!   compute_buffer_drop_n=...
//!   compute_pipeline_hit_n=... compute_pipeline_miss_n=...
//!   compute_pipeline_mismatch_n=... compute_pipeline_disabled_n=...
//!   compute_pipeline_return_n=... compute_pipeline_drop_n=...
//!   render_offscreen_n=... render_present_n=...
//!   render_batch_n=... render_batch_passes=...
//!   render_batch_passes_1=... render_batch_passes_2=...
//!   render_batch_passes_3_4=... render_batch_passes_5_8=...
//!   render_batch_passes_gt8=...
//!   readback_rect_us=... readback_full_us=... readback_seed_us=...
//!   readback_surfaces_us=... readback_shape_us=... readback_named_us=...
//!   landing_lookup_us=... landing_windows_us=... landing_stage_us=...
//!   landing_record_us=... landing_wait_us=... landing_fetch_us=...
//!   landing_write_us=... landing_release_us=... landing_named_us=...
//!   landing_n=... landing_bytes=...
//!   teardown_sync_us=... teardown_pipeline_us=... teardown_passes_us=...
//!   teardown_textures_us=... teardown_descriptors_us=...
//!   teardown_depth_stencil_us=... teardown_attachments_us=...
//!   teardown_readbacks_us=... teardown_buffers_us=... teardown_previous_us=...
//!   teardown_named_us=... render_release_reuse_us=... render_release_pool_us=...
//!   render_release_import_us=... render_release_uploads_us=... render_retire_us=...
//!   td_image_n=... td_view_n=... td_sampler_n=... td_buffer_n=... td_memory_n=...
//!   rb_pipeline_us=... rb_pipeline_n=... rb_buffer_us=... rb_buffer_n=...
//!   rb_image_us=... rb_image_n=... rb_view_us=... rb_view_n=...
//!   rb_sampler_us=... rb_sampler_n=... rb_descriptor_us=... rb_descriptor_n=...
//!   rb_indirect_us=... rb_indirect_n=... rb_named_us=... submit_teardown_us=...
//!   submit_td_sync_us=... submit_td_sync_n=... submit_td_pipeline_us=...
//!   submit_td_pipeline_n=... submit_td_buffers_us=... submit_td_buffers_n=...
//!   submit_td_textures_us=... submit_td_textures_n=... submit_td_retains_us=...
//!   submit_td_retains_n=... submit_td_named_us=... submit_lock_us=...
//!   submit_bookkeep_us=... submit_merge_us=... submit_validate_us=...
//!   submit_validate_derive_us=... submit_validate_check_us=...
//!   submit_validate_release_us=... submit_validate_named_us=...
//!   submit_release_us=... submit_release_bindings_us=...
//!   submit_release_views_us=... submit_release_pool_us=...
//!   submit_release_plan_us=...
//!   submit_release_named_us=... submit_seam_us=...
//!   wait_submit_n=... wait_render_n=... wait_landing_n=...
//!   wait_present_n=... wait_queue_n=... wait_timeline_n=... wait_ahead_sum=...
//!   wait_ahead_n=...
//!   ```
//!
//! and, beside the disjoint fields, the aggregate readings the nested splits
//! imply: `render_us` (the five `render_*` children), `render_residual_us`
//! (their thirteen siblings that divide what those five leave unnamed),
//! `texture_named_us` (the six nested regions inside `setup_textures`) and
//! `teardown_named_us` (the ten nested regions inside `render_teardown`). The
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
//!   that follows a pass, the pass objects' teardown, the four pooled
//!   hand-backs and the retains' release. `sum(render children)
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
//! The fourth round splits the bar the first one left untouched — the pass
//! objects' teardown — and the residual regions beside it:
//!
//! * `teardown_sync_us`, `teardown_pipeline_us`, `teardown_passes_us`,
//!   `teardown_textures_us`, `teardown_descriptors_us`,
//!   `teardown_depth_stencil_us`, `teardown_attachments_us`,
//!   `teardown_readbacks_us`, `teardown_buffers_us` and
//!   `teardown_previous_us` divide `render_teardown` itself — the two sync
//!   objects, the pipeline-shaped objects the shape cache did not take, the
//!   render pass and framebuffers, the sampled declarations' own objects, the
//!   descriptor pools and layouts, the depth/stencil surfaces, the colour
//!   attachments, the readback destinations, the remaining buffers and the
//!   previous-byte staging buffers — in the order `OffscreenObjects::drop`
//!   works through them. The printed `teardown_named_us` is their sum, so
//!   `sum(teardown children) <= render_teardown_us` and the difference is the
//!   seam between the groups.
//! * `render_release_reuse_us`, `render_release_pool_us`,
//!   `render_release_import_us`, `render_release_uploads_us` and
//!   `render_retire_us` are siblings of
//!   `render_teardown` inside the render half's residual: the three pooled
//!   returns (`OffscreenObjects::release_reusable` /
//!   `release_pooled_textures` / `release_imported_windows`, which the teardown
//!   bar does *not* enclose because they run before it) and the input retains'
//!   release. They are named because "give it back" is only free if a reading
//!   says so, and because a pool's eviction destroys objects inside the return.
//! * the five `td_*_n` counters are the population behind those bars: how many
//!   images, image views, samplers, buffers and memories the window's teardowns
//!   really destroyed (a null handle charges no count). A family whose count is
//!   zero is a region the pools already emptied.
//!
//! The fifth round splits what the first four left unnamed, and it is two cuts
//! rather than one:
//!
//! * `rb_pipeline_us`, `rb_buffer_us`, `rb_image_us`, `rb_view_us`,
//!   `rb_sampler_us`, `rb_descriptor_us` and `rb_indirect_us` divide
//!   `resource_build` by the object families its create sites name — one bar
//!   per object rather than per family, so the `rb_*_n` count printed beside
//!   each is that family's population in the window and its microseconds can be
//!   read per object. The printed `rb_named_us` is their sum, so
//!   `sum(rb_*) <= resource_build_us` and the difference is the validation,
//!   sizing and pool-key registration the families do not own.
//! * `submit_teardown_us` names the compute half's own teardown
//!   (`ExecutionResources::drop`), which the first four cuts left inside the
//!   enclosing `total` — the counterpart of the render half's `render_teardown`
//!   and, before this cut, the largest unnamed region of a submission.
//!   `submit_td_sync_us`, `submit_td_pipeline_us`, `submit_td_buffers_us`,
//!   `submit_td_textures_us` and `submit_td_retains_us` divide it in the order
//!   the drop works through them, with `submit_td_named_us` as their printed
//!   sum and the same `_n` counts beside them.
//!
//! The remaining seam — the executor and queue locks, the per-submission plan
//! between `pool` and `resource_build`, the merge of the two halves'
//! writebacks, the terminal contract validation and the host-side release the
//! call's own tail pays — is `submit_lock_us`, `submit_bookkeep_us`,
//! `submit_merge_us`, `submit_validate_us` and `submit_release_us`, printed
//! with `submit_seam_us` as their sum. They are disjoint from the disjoint sum
//! rather than part of it: `sum(disjoint) + sum(seam) <= total_us`, and what
//! remains is the function-call boundary between them.
//!
//! The sixth cut splits the two regions of that seam a reading can act on, and
//! both are nested rather than disjoint:
//!
//! * `submit_release_us` names the tail of the call: the values `total` was
//!   declared before, dropped after the last numbered bar — the pooled
//!   bindings' own copies of the views' bytes, the serial resource pool and the
//!   texture views, the dispatch list, the heap plan, the render plan and the
//!   pipeline artifacts. `submit_release_bindings_us`,
//!   `submit_release_views_us` and `submit_release_plan_us` divide it, with
//!   `submit_release_named_us` as their printed sum, so
//!   `sum(submit_release_*) <= submit_release_us`. The bar resolves to `None`
//!   when the profile is off, and the tail values then drop exactly where they
//!   dropped before the cut.
//! * `submit_validate_derive_us` and `submit_validate_check_us` divide
//!   `submit_validate` — the two pool derivations the terminal validation takes
//!   for itself (`ComputeTrace::serial_resources` and
//!   `serial_texture_resources`) and the walk that reads them, with
//!   `submit_validate_named_us` as their printed sum. The two halves are the
//!   reading a cut of that bar needs: the derivations are what the submission
//!   has already paid for once (`plan` derived the resource pool, `pool` the
//!   texture views) and the walk is the contract check the call exists for. The
//!   `g3dprobe` round measured the derivations a cut would remove at ≈4.9 µs a
//!   submission (0.2 %) and declined to widen the contract for them
//!   (`docs/COMPUTE-PIPELINE-REUSE.md` §6), so this bar is read and not cut.
//!
//! The same round answers "what is a wait waiting on": `wait_submit_n`,
//! `wait_render_n`, `wait_landing_n` and `wait_present_n` count every fence the
//! window waited on by whose completion it is, `wait_queue_n` and
//! `wait_timeline_n` are the two kinds this provider does not have (they stay
//! zero, which is the evidence for "binary completion fences only"), and
//! `wait_ahead_sum` / `wait_ahead_n` read how much *earlier* work the waited
//! queue still held when the wait began — the difference between waiting for
//! one's own submission on an idle queue and waiting behind others.
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
    /// Inside `plan`: the derivation of the submission's own serial resource
    /// pool. It is one call — `ComputeTrace::serial_resources`, which owns the
    /// table and copies every view's declared bytes, or
    /// `ComputeTrace::serial_resources_ref`, which lends the trace's own
    /// declarations and copies nothing — and it is the region the seventh
    /// cut's switch moves (`crate::serial_resources_borrow`).
    PlanResources,
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
    /// Inside `render_teardown`: the pass's two synchronisation objects — the
    /// completion fence and the command pool the command buffer was allocated
    /// from (`vkDestroyFence` / `vkDestroyCommandPool`, the latter releasing
    /// the buffer with it).
    ///
    /// This bar and the nine below it divide `render_teardown` itself, so they
    /// are *nested* inside [`Phase::RenderTeardown`] rather than beside it:
    /// `sum(teardown children) <= render_teardown_us`, and the difference is
    /// the seam between the groups of `OffscreenObjects::drop`.
    TeardownSync,
    /// Inside `render_teardown`: the pipeline-shaped objects the shape cache
    /// did not take — the graphics pipeline, its layout and the two shader
    /// modules. A pass whose key the cache answered leaves these null and
    /// charges nothing here (`crate::render_setup_reuse`).
    TeardownPipeline,
    /// Inside `render_teardown`: the render pass, the seed render pass a
    /// multisampled `Load` runs first, and both framebuffers.
    TeardownPasses,
    /// Inside `render_teardown`: the sampled declarations' own samplers, views,
    /// images and memories — what neither the texture backing pool nor a
    /// texel-fetch slot's missing sampler left behind
    /// (`crate::render_texture_pool`), beside a no-copy arm's imported window
    /// when the import pool did not take it back.
    TeardownTextures,
    /// Inside `render_teardown`: the sampler set's descriptor pool and layout,
    /// the stage-buffer sets' own pools and layouts, and the empty layouts the
    /// pipeline layout's unused slots hold.
    TeardownDescriptors,
    /// Inside `render_teardown`: the rail-owned depth and stencil surfaces with
    /// their resolve targets (`research/docs/23` §3.3, v36/v47/v60).
    TeardownDepthStencil,
    /// Inside `render_teardown`: the colour attachment images (or the resolve
    /// targets they own), their views and their memories.
    TeardownAttachments,
    /// Inside `render_teardown`: the readback destinations — one unmap, one
    /// buffer and one memory per stored colour attachment.
    TeardownReadbacks,
    /// Inside `render_teardown`: every remaining buffer with its memory — the
    /// stage buffers, the indirect and index buffers, and the caller-held
    /// vertex streams.
    TeardownBuffers,
    /// Inside `render_teardown`: the previous-byte staging buffers a `Load`
    /// uploaded from, which are destroyed in their own loop after the buffers
    /// above.
    TeardownPrevious,
    /// Inside the render half's residual, beside [`Phase::RenderRetain`]:
    /// handing the pipeline-shaped objects back to the shape cache once the
    /// fence has retired them (`OffscreenObjects::release_reusable`).
    RenderReleaseReuse,
    /// Inside the render half's residual: handing the sampled textures' pooled
    /// backing back (`OffscreenObjects::release_pooled_textures`) — which also
    /// pays whatever the pool's own eviction destroys.
    RenderReleasePool,
    /// Inside the render half's residual: handing the owner-window imports
    /// back (`OffscreenObjects::release_imported_windows`).
    RenderReleaseImport,
    /// Inside the render half's residual: handing the rail-owned host-visible
    /// upload buffers back (`OffscreenObjects::release_pooled_uploads`) — the
    /// previous bytes, vertex streams, index and indirect buffers and stage
    /// buffers one pass built or took, which the fourth cut pools
    /// (`crate::render_buffer_pool`).
    RenderReleaseUploads,
    /// Inside the render half's residual: the input retains the pass took
    /// before its first import being released once the fence has proven the
    /// device done with the owner's pages (`RenderInputRetains::retire`).
    RenderRetire,
    /// Inside `render_record`: the loop that issues the draws **after** the
    /// head of a render pass that carries an ordered list of draws
    /// (`research/docs/23` §3.3, G3-B/B-2) — one binding and one `vkCmdDraw*`
    /// per draw, through the same `record_draw_binding` the single-draw arm
    /// uses.
    ///
    /// The bar is default-**off** in the sense that matters: it is entered only
    /// when the phase profile is on *and* the pass actually carries draws
    /// beyond its head, so a single-draw pass — every pass written before the
    /// arm, and every pass of a trace that never assembles a list — charges
    /// nothing here and its `render_record_us` is unchanged. It is a *nested*
    /// bar inside `render_record`, not a sibling: adding it to the disjoint sum
    /// would count the same microseconds twice.
    RenderDrawsLoop,
    /// Inside `resource_build`: the pipeline-shaped objects one compute
    /// pipeline needs (`PipelineObjects::create`) — the two shader modules, the
    /// pipeline layout and the pipeline. Charged once per pipeline, so the
    /// printed `rb_pipeline_n` is how many pipelines one submission built.
    ///
    /// This bar and the six below it divide `resource_build` itself, so they
    /// are *nested* inside [`Phase::ResourceBuild`] rather than beside it:
    /// `sum(rb_*) <= resource_build_us`, and the difference is the bookkeeping
    /// the named families do not own — the argument validation, the shared/heap
    /// sizing passes and the pool-key registrations that run between the device
    /// calls. The families are the ones the create sites actually name; a
    /// framebuffer or a render pass is not among them because the compute half
    /// builds neither (the render rail's are `setup_render_pass` and
    /// `teardown_passes`).
    RbPipeline,
    /// Inside `resource_build`: one host-visible device buffer and the memory
    /// bound to it, upload included (`create_owned_backing`, the heap slab, the
    /// indirect replay's buffer pair). Charged once per buffer.
    RbBuffer,
    /// Inside `resource_build`: one sampled or storage image with its memory
    /// and, for the sampled arm, the texels' own trip into it
    /// (`allocate_image_backing`). Charged once per image.
    RbImage,
    /// Inside `resource_build`: one image view
    /// (`create_color_image_view` / `create_depth_image_view`).
    RbView,
    /// Inside `resource_build`: one `vkCreateSampler` — a declaration's own
    /// sampler or a pipeline's static sampler.
    RbSampler,
    /// Inside `resource_build`: one pass's descriptor set — the layout, the
    /// pool, the set and the writes that fill it (`create_descriptors`).
    RbDescriptor,
    /// Inside `resource_build`: one indirect replay's command buffer and its
    /// memory (`create_indirect_dispatch`). A direct dispatch builds none and
    /// charges nothing here.
    RbIndirect,
    /// The compute half's own teardown (`ExecutionResources::drop` under the
    /// destroying policy): the fence, the pools, the pipeline objects, the
    /// buffers, the images, the samplers and the borrowed retains a submission
    /// built or took, destroyed once the fence has proven the device done with
    /// them.
    ///
    /// It is a *disjoint* bar of the submission rather than a child of
    /// `resource_build`: it runs at the end of the submission, after the
    /// readback, and it is the compute half's counterpart of the render half's
    /// `render_teardown`. Placing the two halves' teardowns in one reading is
    /// what makes "what does one submission's teardown cost" answerable.
    ///
    /// Like the render half's, this bar emits one [`Phase::Total`] window per
    /// measured submission, so a line whose `n` counts submissions counts
    /// teardowns with them.
    SubmitTeardown,
    /// Inside `submit_teardown`: the submission's own synchronisation and pool
    /// objects — the completion fence, the command pool and the descriptor
    /// pool.
    SubmitTdSync,
    /// Inside `submit_teardown`: the pipeline-shaped objects
    /// (`PipelineObjects::drop`: the pipeline, its layout and the two shader
    /// modules).
    SubmitTdPipeline,
    /// Inside `submit_teardown`: every buffer with its memory — the owned and
    /// shared backings, the heap slab's buffers, the storage images' transfer
    /// buffers and the indirect replay's pair.
    SubmitTdBuffers,
    /// Inside `submit_teardown`: the sampled and storage declarations' own
    /// samplers, views, images and memories.
    SubmitTdTextures,
    /// Inside `submit_teardown`: retiring the borrowed leases the submission's
    /// guest-run gathers took, which is what lets the owner's pages go.
    SubmitTdRetains,
    /// The seam between the submission's disjoint bars on the compute side: the
    /// executor lock, the queue pick, the queue lock and the arena admission a
    /// submission takes before its resources can be built.
    SubmitLock,
    /// The seam between `pool` and `resource_build`: the per-submission plan —
    /// the translated-artifact list and `plan_pipeline_sequence` — taken inside
    /// `PendingExecution::submit` before the first device object exists.
    SubmitBookkeep,
    /// The seam after the render half: merging the two halves' writebacks into
    /// one keyed map (`BTreeMap::insert` per written view).
    SubmitMerge,
    /// The seam after the merge: `ProviderSubmission::validate_for_trace`, the
    /// terminal contract validation of the merged writeback list against the
    /// exact submitted trace. Before this cut it was charged to `writebacks`
    /// only because that bar's guard happened to be alive across it.
    SubmitValidate,
    /// The sixth cut's first region: the host-side release of everything the
    /// submission built or took, which runs **after** [`Phase::Settle`] has
    /// been charged and is therefore inside the enclosing `total` and outside
    /// every other bar.
    ///
    /// The order is forced by the language rather than chosen: `total` is the
    /// first binding in `submit`, so it is the last to drop, and the values
    /// declared after it — the pooled bindings' byte copies, the resource pool,
    /// the texture views, the plan values and the pipeline artifacts — are
    /// dropped *after* the `settle` guard that was declared last. That tail is
    /// a real per-submission cost (it frees everything the call allocated) and
    /// before this cut nothing named it: the fifth cut's seam residual, read as
    /// `total_us` minus the disjoint bars, carried it without a name.
    ///
    /// It is a *nested* bar of the seam rather than a member of the disjoint
    /// sum: entering it is gated on the profile being on (`Bar::enter` resolves
    /// to `None` and the tail values drop where they always did), and its three
    /// children below divide it the way the release itself is grouped.
    SubmitRelease,
    /// Inside `submit_release`: the pooled bindings — one device binding per
    /// pooled view with its own copy of the view's bytes — dropped as one
    /// vector.
    SubmitReleaseBindings,
    /// Inside `submit_release`: the submission's serial resource pool and its
    /// texture views, the two derived tables the plan and the validation read.
    SubmitReleaseViews,
    /// Inside `submit_release_views`: the pool's *own* table — what `plan`
    /// derived — dropped on its own so the two halves of that child can be read
    /// apart. Off it is the first of the two copies a submission pays for its
    /// declared bytes; on it is a table of references to the trace's
    /// declarations (`crate::serial_resources_borrow`).
    SubmitReleasePool,
    /// Inside `submit_release`: the plan values the call held for its own tail —
    /// the per-pass dispatch list, the heap placement plan, the render plan and
    /// the pipeline artifacts the plan selected.
    SubmitReleasePlan,
    /// Inside `submit_validate`: the two pool derivations the terminal
    /// validation takes for itself — `ComputeTrace::serial_resources` (which
    /// re-validates the trace and walks every pass's buffer declarations) and
    /// `ComputeTrace::serial_texture_resources`.
    ///
    /// This bar and [`Phase::SubmitValidateCheck`] divide `submit_validate`,
    /// which the fifth cut named as the largest region of the seam: the
    /// derivations are the half a cut can remove (the submission already holds
    /// both tables — `plan` derived the pool and `pool` derived the texture
    /// views), and the check is the half a cut cannot, because it is the
    /// contract check the call exists for.
    SubmitValidateDerive,
    /// Inside `submit_validate`: the walk that checks the merged writeback list
    /// against the derived resources and the exact submitted trace.
    SubmitValidateCheck,
    /// Inside `submit_validate`: the release of the two tables the validation
    /// derived for itself, which the block used to pay as an unnamed seam
    /// (`sp16` read it at 43.0 µs/submission). Off it is the third copy's
    /// `free`; on the tables are lent, so it is the release of a table of
    /// references (`crate::serial_resources_borrow`).
    SubmitValidateRelease,
}

const PHASE_COUNT: usize = Phase::SubmitValidateRelease as usize + 1;

/// The printed field name of each phase, in slot order.
const PHASE_NAMES: [&str; PHASE_COUNT] = [
    "total",
    "admit",
    "plan",
    "plan_resources",
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
    "teardown_sync",
    "teardown_pipeline",
    "teardown_passes",
    "teardown_textures",
    "teardown_descriptors",
    "teardown_depth_stencil",
    "teardown_attachments",
    "teardown_readbacks",
    "teardown_buffers",
    "teardown_previous",
    "render_release_reuse",
    "render_release_pool",
    "render_release_import",
    "render_release_uploads",
    "render_retire",
    "render_draws_loop",
    "rb_pipeline",
    "rb_buffer",
    "rb_image",
    "rb_view",
    "rb_sampler",
    "rb_descriptor",
    "rb_indirect",
    "submit_teardown",
    "submit_td_sync",
    "submit_td_pipeline",
    "submit_td_buffers",
    "submit_td_textures",
    "submit_td_retains",
    "submit_lock",
    "submit_bookkeep",
    "submit_merge",
    "submit_validate",
    "submit_release",
    "submit_release_bindings",
    "submit_release_views",
    "submit_release_pool",
    "submit_release_plan",
    "submit_validate_derive",
    "submit_validate_check",
    "submit_validate_release",
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
const RENDER_RESIDUAL_SLOTS: [usize; 13] = [
    Phase::RenderResolve as usize,
    Phase::RenderPresent as usize,
    Phase::RenderPublish as usize,
    Phase::RenderLanding as usize,
    Phase::RenderPrepare as usize,
    Phase::RenderRetain as usize,
    Phase::RenderLandOwner as usize,
    Phase::RenderTeardown as usize,
    Phase::RenderReleaseReuse as usize,
    Phase::RenderReleasePool as usize,
    Phase::RenderReleaseImport as usize,
    Phase::RenderReleaseUploads as usize,
    Phase::RenderRetire as usize,
];

/// The nested split of `render_teardown`, added by the fourth cut: the ten
/// regions `OffscreenObjects::drop` works through, in the order it works
/// through them. `render_teardown` stays their enclosing bar, so
/// `sum(teardown children) <= render_teardown_us` and the difference is the
/// seam between the groups. The printed `teardown_named_us` is this set's sum.
const TEARDOWN_SLOTS: [usize; 10] = [
    Phase::TeardownSync as usize,
    Phase::TeardownPipeline as usize,
    Phase::TeardownPasses as usize,
    Phase::TeardownTextures as usize,
    Phase::TeardownDescriptors as usize,
    Phase::TeardownDepthStencil as usize,
    Phase::TeardownAttachments as usize,
    Phase::TeardownReadbacks as usize,
    Phase::TeardownBuffers as usize,
    Phase::TeardownPrevious as usize,
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

/// The nested split of `resource_build`, added by the fifth cut: the object
/// families the create sites name, one bar per object. `resource_build` stays
/// their enclosing bar, so `sum(RESOURCE_BUILD_SLOTS) <= resource_build_us` and
/// the difference is the validation, sizing and registration the families do
/// not own. The printed `rb_named_us` is this set's sum.
///
/// Each member is entered once per object rather than once per family, so the
/// call count the line prints beside it (`rb_*_n`) is that family's population
/// in the window — the reading a pooling decision needs.
const RESOURCE_BUILD_SLOTS: [usize; 7] = [
    Phase::RbPipeline as usize,
    Phase::RbBuffer as usize,
    Phase::RbImage as usize,
    Phase::RbView as usize,
    Phase::RbSampler as usize,
    Phase::RbDescriptor as usize,
    Phase::RbIndirect as usize,
];

/// The nested split of `submit_teardown`, the compute half's counterpart of the
/// render half's `teardown_*` cut: the sync objects, the pipeline-shaped
/// objects, the buffers, the textures and the borrowed retains, in the order
/// `ExecutionResources::drop` works through them. `submit_teardown` stays their
/// enclosing bar, so `sum(SUBMIT_TEARDOWN_SLOTS) <= submit_teardown_us`. The
/// printed `submit_td_named_us` is this set's sum.
const SUBMIT_TEARDOWN_SLOTS: [usize; 5] = [
    Phase::SubmitTdSync as usize,
    Phase::SubmitTdPipeline as usize,
    Phase::SubmitTdBuffers as usize,
    Phase::SubmitTdTextures as usize,
    Phase::SubmitTdRetains as usize,
];

/// The named regions of the seam the disjoint bars leave — the enclosing
/// `total`'s own residual. They are disjoint from each other and from every bar
/// of the disjoint sum, so `sum(SUBMIT_SEAM_SLOTS) <= total_us -
/// sum(disjoint bars)`, and the difference is the function-call boundary that
/// remains. The printed `submit_seam_us` is this set's sum.
///
/// The sixth cut added `submit_release` to the set. It is the region the fifth
/// cut's residual was largest in and the one no earlier cut could name: the
/// tail of the call, after the last numbered bar, where the language drops the
/// values `total` was declared before.
const SUBMIT_SEAM_SLOTS: [usize; 6] = [
    Phase::SubmitTeardown as usize,
    Phase::SubmitRelease as usize,
    Phase::SubmitLock as usize,
    Phase::SubmitBookkeep as usize,
    Phase::SubmitMerge as usize,
    Phase::SubmitValidate as usize,
];

/// The nested split of `submit_release`: the pooled bindings' byte copies, the
/// resource pool and texture views, and the plan values, in the order the
/// call's own tail drops them. `submit_release` stays their enclosing bar, so
/// `sum(SUBMIT_RELEASE_SLOTS) <= submit_release_us`; the printed
/// `submit_release_named_us` is this set's sum.
const SUBMIT_RELEASE_SLOTS: [usize; 3] = [
    Phase::SubmitReleaseBindings as usize,
    Phase::SubmitReleaseViews as usize,
    Phase::SubmitReleasePlan as usize,
];

/// The nested split of `submit_release_views`: the pool's own table, dropped
/// apart from the texture views that share its parent. The parent stays the
/// enclosing bar, so `submit_release_pool_us <= submit_release_views_us` and the
/// pool's share is read as a share rather than added to the release's three
/// children.
///
/// The set is read by the tests rather than by the printer: a one-member nested
/// split needs no printed sum of its own — the bar is the field — but the rule
/// that it is nested and not a fourth sibling is what keeps
/// `submit_release_named_us` a sum of the release's children, so it is held
/// there.
#[cfg(test)]
const SUBMIT_RELEASE_VIEWS_SLOTS: [usize; 1] = [Phase::SubmitReleasePool as usize];

/// The nested split of `submit_validate`: the two pool derivations the
/// validation takes, the walk that reads them, and the release of the tables
/// the validation derived for itself. Like the other splits it is not part of
/// the disjoint sum, and `submit_validate` stays the enclosing bar:
/// `sum(SUBMIT_VALIDATE_SLOTS) <= submit_validate_us`.
///
/// The seventh cut added the release: before it, that region was the seam this
/// bar carried (`submit_validate_us` minus the two halves), which the `sp16`
/// round read at 43.0 µs/submission — the third copy of a submission's declared
/// bytes being freed. Naming it makes the split whole on both of the seventh
/// cut's arms instead of leaving a difference a reader has to interpret.
const SUBMIT_VALIDATE_SLOTS: [usize; 3] = [
    Phase::SubmitValidateDerive as usize,
    Phase::SubmitValidateCheck as usize,
    Phase::SubmitValidateRelease as usize,
];

/// The teardown groups whose *call count* is printed beside their
/// microseconds: each is entered once per region it names — per destroyed
/// object inside the three object loops and once per submission for the two
/// whole-group regions — so the count is that group's population in the window.
/// The `resource_build` families take their counts from
/// [`note_build_object`] instead, because their bars are whole-call regions and
/// a region count is not an object count.
const COUNTED_SLOTS: [usize; 5] = [
    Phase::SubmitTdSync as usize,
    Phase::SubmitTdPipeline as usize,
    Phase::SubmitTdBuffers as usize,
    Phase::SubmitTdTextures as usize,
    Phase::SubmitTdRetains as usize,
];

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

/// The object a wait blocked on.
///
/// A wait's idle/blocked split says how much of it was device latency, but not
/// what the wait was behind. On this provider every wait is `vkWaitForFences`
/// on one **binary completion fence**, so the honest answer to "is the wait for
/// a submission count, a queue or a timeline semaphore" is "none of them, it is
/// a fence" — and that answer is only checkable if the populations are counted
/// apart. The four kinds below partition every fence wait the provider performs
/// (`wait_queue_n` and `wait_timeline_n` are printed beside them and stay zero
/// for as long as that is true).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
// The shared `Fence` suffix is the reading, not a naming accident: every object
// this provider waits on is a completion fence, and the prefix is what tells
// the four populations apart.
#[allow(clippy::enum_variant_names)]
pub(crate) enum WaitObject {
    /// The compute submission's own completion fence
    /// (`PendingExecution::wait`), inside the `fence_wait` bar.
    SubmitFence,
    /// The render half's completion fence (`OffscreenObjects::submit_and_wait`),
    /// inside the `render_wait` bar — where a pass's copy-out executes.
    RenderFence,
    /// A kept-frame landing's completion fence, inside the `landing_wait` bar.
    LandingFence,
    /// The present rail's sentinel fence, whose wait is inside the present
    /// entry's own residual region.
    PresentFence,
}

const WAIT_OBJECT_COUNT: usize = WaitObject::PresentFence as usize + 1;

/// The printed field-name stem of each wait object, in slot order.
const WAIT_OBJECT_NAMES: [&str; WAIT_OBJECT_COUNT] =
    ["wait_submit", "wait_render", "wait_landing", "wait_present"];

impl WaitObject {
    #[inline]
    fn slot(self) -> usize {
        self as usize
    }
}

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

/// Count one host-visible upload buffer's use of the pooled pair
/// (`crate::render_buffer_pool`) for the emitting thread's window.
///
/// The five outcomes partition every creation that reaches the mechanism: the
/// pool held a buffer of this shape and handed it over, it held none and the
/// creation built its own, the switch was off, or a completed pass handed a
/// buffer back and the pool kept it (or destroyed it instead).
#[inline]
pub(crate) fn note_buffer_pool(outcome: crate::render_buffer_pool::UploadOutcome) {
    if !enabled() {
        return;
    }
    use crate::render_buffer_pool::UploadOutcome;
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        match outcome {
            UploadOutcome::Hit => local.buffer_hit_n += 1,
            UploadOutcome::Miss => local.buffer_miss_n += 1,
            UploadOutcome::Disabled => local.buffer_disabled_n += 1,
            UploadOutcome::Returned => local.buffer_return_n += 1,
            UploadOutcome::Dropped => local.buffer_drop_n += 1,
        }
    });
}

/// Count one compute-side host-visible upload buffer's use of the pooled pair
/// (`crate::compute_buffer_pool`) for the emitting thread's window.
///
/// The five outcomes partition every creation that reaches the mechanism: the
/// pool held a buffer of this shape and handed it over, it held none and the
/// creation built its own, the switch was off, or a submission whose fence was
/// observed handed a buffer back and the pool kept it (or destroyed it
/// instead). They are separate from the render rail's `buffer_*_n` because the
/// two rails have their own switches: a round has to be able to see the compute
/// half's reuse on its own.
#[inline]
pub(crate) fn note_compute_buffer_pool(outcome: crate::compute_buffer_pool::ComputeBufferOutcome) {
    if !enabled() {
        return;
    }
    use crate::compute_buffer_pool::ComputeBufferOutcome;
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        match outcome {
            ComputeBufferOutcome::Hit => local.compute_buffer_hit_n += 1,
            ComputeBufferOutcome::Miss => local.compute_buffer_miss_n += 1,
            ComputeBufferOutcome::Disabled => local.compute_buffer_disabled_n += 1,
            ComputeBufferOutcome::Returned => local.compute_buffer_return_n += 1,
            ComputeBufferOutcome::Dropped => local.compute_buffer_drop_n += 1,
        }
    });
}

/// Count one compute creation's use of the shape-decided pipeline table
/// (`crate::compute_pipeline_reuse`) for the emitting thread's window.
///
/// The six outcomes partition every creation that reaches the mechanism and
/// every hand-back a completed submission makes: the table held objects of this
/// shape and handed them over, it held none and the creation built its own, an
/// entry shared the digest but not the shape, the switch was off, or a
/// submission whose fence was observed handed its objects back and the table
/// kept them (or destroyed them instead). They are separate from the render
/// rail's `reuse_*_n` because the two rails have their own switches: a round
/// has to be able to see the compute half's reuse on its own.
#[inline]
pub(crate) fn note_compute_pipeline_reuse(
    outcome: crate::compute_pipeline_reuse::ComputePipelineOutcome,
) {
    if !enabled() {
        return;
    }
    use crate::compute_pipeline_reuse::ComputePipelineOutcome;
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        match outcome {
            ComputePipelineOutcome::Hit => local.compute_pipeline_hit_n += 1,
            ComputePipelineOutcome::Miss => local.compute_pipeline_miss_n += 1,
            ComputePipelineOutcome::Mismatch => local.compute_pipeline_mismatch_n += 1,
            ComputePipelineOutcome::Disabled => local.compute_pipeline_disabled_n += 1,
            ComputePipelineOutcome::Returned => local.compute_pipeline_return_n += 1,
            ComputePipelineOutcome::Dropped => local.compute_pipeline_drop_n += 1,
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

/// Count one executed render *batch* and the passes it carried
/// (`REIMS_VGPU_RENDER_BATCH`).
///
/// A batch is one submission scope: one `vkQueueSubmit`, one fence and one
/// wait, carrying `passes` recorded render passes. The reading a round wants
/// beside `render_offscreen_n` is the divisor — `render_offscreen_n /
/// render_batch_n` is the mean batch length, and the band counters say which
/// lengths the window's population was made of (the same bands
/// `runtime/exec/report.rs` uses for `stream_draws_*`, so the two rails'
/// readings are comparable).
///
/// Every counted batch is one fence and one wait: `wait_render_n` divided by
/// `render_batch_n` is therefore the waits per batch, and it falls below the
/// per-pass count exactly to the degree this instrument's population is
/// batched.
#[inline]
pub(crate) fn note_render_batch(passes: u64) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        local.render_batch_n += 1;
        local.render_batch_passes += passes;
        match passes {
            0 | 1 => local.render_batch_passes_1 += 1,
            2 => local.render_batch_passes_2 += 1,
            3..=4 => local.render_batch_passes_3_4 += 1,
            5..=8 => local.render_batch_passes_5_8 += 1,
            _ => local.render_batch_passes_gt8 += 1,
        }
    });
}

/// Count one trace the run rail stated: a plan of two or more render passes
/// (`REIMS_VGPU_RENDER_BATCH`), and how many passes it carried.
///
/// This is the population the three readings beside [`note_render_batch`]
/// divide, and it is what makes `render_batch_n == 0` readable: a round whose
/// run rail assembled runs while this counter is zero never handed the provider
/// a trace with two passes in it, which is a fact about the *owner's* assembly
/// rather than about a predicate here.
#[inline]
pub(crate) fn note_render_batch_trace(passes: u64) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        local.render_batch_traces_n += 1;
        local.render_batch_traces_passes += passes;
    });
}

/// Count one run this provider opened — a pass whose frame stays in the
/// identity's image, so the pass after it may load it inside the same scope.
///
/// An opened run of one member executes the per-pass path, so this counter is
/// larger than [`note_render_batch`]'s by exactly the openings no successor
/// continued.
#[inline]
pub(crate) fn note_render_batch_open() {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| local.borrow_mut().render_batch_open_n += 1);
}

/// Count one pass in a stated run's trace that could not open a run at all, and
/// whether the fact that stopped it was its own frame
/// (`frame_not_kept`: the pass's frame does not stay in the identity's image).
#[inline]
pub(crate) fn note_render_batch_refusal(frame_not_kept: bool) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        local.render_batch_refused_n += 1;
        if frame_not_kept {
            local.render_batch_refused_frame_n += 1;
        }
    });
}

/// Count one pass that did not continue the run open before it, and whether the
/// fact that stopped it was its own load (`load`: the pass does not open from
/// the image its predecessor kept).
#[inline]
pub(crate) fn note_render_batch_break(load: bool) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        local.render_batch_broken_n += 1;
        if load {
            local.render_batch_broken_load_n += 1;
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
    /// Every fence wait the window performed, by what it waited on. The four
    /// partition the waits; `wait_queue_calls` and `wait_timeline_calls` are the
    /// two kinds this provider does not have, printed so that "it waits on
    /// binary fences only" is a reading rather than a claim.
    wait_object_calls: [u64; WAIT_OBJECT_COUNT],
    wait_queue_calls: u64,
    wait_timeline_calls: u64,
    /// How much earlier work was still in flight on the waited queue when the
    /// wait began: the sum over the window's waits of every *other* submission
    /// the queue had not retired yet, and how many of those waits began with at
    /// least one. Divide the sum by the waits to get the mean depth — the
    /// difference between waiting for one's own work on an idle queue and
    /// waiting behind earlier submissions.
    wait_ahead_sum: u64,
    wait_ahead_calls: u64,
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
    /// The host-visible upload buffers this window's passes created or handed
    /// back, by what `crate::render_buffer_pool` answered.
    buffer_hit_n: u64,
    buffer_miss_n: u64,
    buffer_disabled_n: u64,
    buffer_return_n: u64,
    buffer_drop_n: u64,
    /// The compute half's own host-visible upload buffers this window's
    /// submissions created or handed back, by what
    /// `crate::compute_buffer_pool` answered. A separate group from
    /// `buffer_*_n` because the two rails carry their own switch.
    compute_buffer_hit_n: u64,
    compute_buffer_miss_n: u64,
    compute_buffer_disabled_n: u64,
    compute_buffer_return_n: u64,
    compute_buffer_drop_n: u64,
    /// The compute half's own shape-decided pipeline groups this window's
    /// submissions took, built or handed back, by what
    /// `crate::compute_pipeline_reuse` answered. A separate group from
    /// `reuse_*_n` because the two rails carry their own switch, and separate
    /// from `rb_pipeline_n` because a group that was handed over built nothing.
    compute_pipeline_hit_n: u64,
    compute_pipeline_miss_n: u64,
    compute_pipeline_mismatch_n: u64,
    compute_pipeline_disabled_n: u64,
    compute_pipeline_return_n: u64,
    compute_pipeline_drop_n: u64,
    /// The render passes this window's submissions executed, by shape. The two
    /// do not share a cost shape, so a bar reading has to name its population.
    render_offscreen_n: u64,
    render_present_n: u64,
    /// The render batches this window's submissions executed, the passes they
    /// carried, and the batch-length bands those passes were grouped in
    /// (`REIMS_VGPU_RENDER_BATCH`).
    render_batch_n: u64,
    render_batch_passes: u64,
    render_batch_passes_1: u64,
    render_batch_passes_2: u64,
    render_batch_passes_3_4: u64,
    render_batch_passes_5_8: u64,
    render_batch_passes_gt8: u64,
    /// The *traces* the run rail stated (`render_batch_traces_n` plans of two or
    /// more render passes, carrying `render_batch_traces_passes` passes), and
    /// what became of their runs: how many opened (`render_batch_open_n`), how
    /// many passes could not open one at all (`render_batch_refused_n`, of which
    /// `_frame_n` did not keep their frame in the identity's image), and how many
    /// passes did not continue an open run (`render_batch_broken_n`, of which
    /// `_load_n` did not open from that image).
    ///
    /// The three divide the same population and their sum is a reading, not a
    /// taxonomy: a trace whose second pass does not continue the first counts one
    /// open and one break, and only `render_batch_n` above counts a run that was
    /// actually carried as one submission scope (a run of one member executes
    /// the per-pass path by construction, so it is not counted there).
    render_batch_traces_n: u64,
    render_batch_traces_passes: u64,
    render_batch_open_n: u64,
    render_batch_refused_n: u64,
    render_batch_refused_frame_n: u64,
    render_batch_broken_n: u64,
    render_batch_broken_load_n: u64,
    /// The kept frames this window's landing-only entries delivered, and the
    /// bytes the owner's pages received.
    landing_n: u64,
    landing_bytes: u64,
    /// The readback staging buffers this window's passes allocated, by which
    /// memory type the selection took.
    staging_cached_n: u64,
    staging_plain_n: u64,
    /// The bytes the window's submissions moved into their pooled bindings,
    /// split by which way they arrived: a copy the submission made for itself
    /// (the trace's own snapshot bytes with the sixth cut's mechanism off) or a
    /// borrow of the table it already holds (with the mechanism on). The other
    /// two binding sources — a staged lease's copy and a gathered run list —
    /// are always owned and are counted in neither
    /// (`crate::submit_binding_borrow`).
    binding_copy_calls: u64,
    binding_copy_bytes: u64,
    binding_borrow_calls: u64,
    binding_borrow_bytes: u64,
    /// The declared bytes the window's *pool derivations* moved, split the same
    /// way: a copy the derivation made for itself (`ComputeTrace::serial_resources`,
    /// the pre-cut path) or a loan of the trace's own declarations
    /// (`ComputeTrace::serial_resources_ref`). One submission derives its pool
    /// twice — once in `plan` and once in `submit_validate` — so the two
    /// counters read two derivations' worth per submission on either arm
    /// (`crate::serial_resources_borrow`).
    resource_copy_views: u64,
    resource_copy_bytes: u64,
    resource_borrow_views: u64,
    resource_borrow_bytes: u64,
    /// The device objects the window's teardowns actually destroyed, by family.
    /// They are the population behind the `teardown_*` bars: a bar's
    /// microseconds divided by its own family's count is one object's cost, and
    /// a family whose count is zero is a region the pools already emptied.
    td_image_n: u64,
    td_view_n: u64,
    td_sampler_n: u64,
    td_buffer_n: u64,
    td_memory_n: u64,
    /// The device objects this window's `resource_build` regions created, by
    /// family — the population behind the `rb_*` bars
    /// ([`BuildObject`]).
    build_objects: [u64; BUILD_OBJECT_COUNT],
    /// How many enclosing `total` bars this thread has open right now. The
    /// compute half's teardown is the one region that can run *outside* a
    /// submission — a deferred object API retires its resources from `wait` —
    /// and a bar charged there would land in a window whose `total` never
    /// contained it, which is exactly what the identity a reader checks must
    /// not allow. The depth is therefore what
    /// [`Bar::enter_in_submission`] reads to decide whether to charge at all.
    depth: u32,
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
            wait_object_calls: [0; WAIT_OBJECT_COUNT],
            wait_queue_calls: 0,
            wait_timeline_calls: 0,
            wait_ahead_sum: 0,
            wait_ahead_calls: 0,
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
            buffer_hit_n: 0,
            buffer_miss_n: 0,
            buffer_disabled_n: 0,
            buffer_return_n: 0,
            buffer_drop_n: 0,
            compute_buffer_hit_n: 0,
            compute_buffer_miss_n: 0,
            compute_buffer_disabled_n: 0,
            compute_buffer_return_n: 0,
            compute_buffer_drop_n: 0,
            compute_pipeline_hit_n: 0,
            compute_pipeline_miss_n: 0,
            compute_pipeline_mismatch_n: 0,
            compute_pipeline_disabled_n: 0,
            compute_pipeline_return_n: 0,
            compute_pipeline_drop_n: 0,
            render_offscreen_n: 0,
            render_present_n: 0,
            render_batch_n: 0,
            render_batch_passes: 0,
            render_batch_passes_1: 0,
            render_batch_passes_2: 0,
            render_batch_passes_3_4: 0,
            render_batch_passes_5_8: 0,
            render_batch_passes_gt8: 0,
            render_batch_traces_n: 0,
            render_batch_traces_passes: 0,
            render_batch_open_n: 0,
            render_batch_refused_n: 0,
            render_batch_refused_frame_n: 0,
            render_batch_broken_n: 0,
            render_batch_broken_load_n: 0,
            landing_n: 0,
            landing_bytes: 0,
            staging_cached_n: 0,
            staging_plain_n: 0,
            binding_copy_calls: 0,
            binding_copy_bytes: 0,
            binding_borrow_calls: 0,
            binding_borrow_bytes: 0,
            resource_copy_views: 0,
            resource_copy_bytes: 0,
            resource_borrow_views: 0,
            resource_borrow_bytes: 0,
            td_image_n: 0,
            td_view_n: 0,
            td_sampler_n: 0,
            td_buffer_n: 0,
            td_memory_n: 0,
            build_objects: [0; BUILD_OBJECT_COUNT],
            depth: 0,
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
    fn note_fence_wait(
        &mut self,
        phase: Phase,
        ns: u64,
        timed_out: bool,
        object: WaitObject,
        ahead: usize,
    ) {
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
        self.note_wait_object(object, ahead);
    }

    /// Bank the *identity* of one fence wait that is timed by another bar.
    ///
    /// A landing's wait is its own region (`landing_wait`) and the present
    /// rail's sentinel is inside the present entry's residual, so their
    /// microseconds must not also land in `fence_wait`; their populations still
    /// belong in the window's answer to "what does this submission wait on",
    /// which is what this counts.
    #[inline]
    fn note_wait_object(&mut self, object: WaitObject, ahead: usize) {
        self.wait_object_calls[object.slot()] += 1;
        self.wait_ahead_sum += ahead as u64;
        if ahead > 0 {
            self.wait_ahead_calls += 1;
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
        let mut teardown_named_ns = 0u64;
        let mut rb_named_ns = 0u64;
        let mut submit_td_named_ns = 0u64;
        let mut submit_seam_ns = 0u64;
        let mut submit_release_named_ns = 0u64;
        let mut submit_validate_named_ns = 0u64;
        for (slot, name) in PHASE_NAMES.iter().enumerate() {
            let ns = std::mem::take(&mut self.ns[slot]);
            let calls = std::mem::replace(&mut self.calls[slot], 0);
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
            if TEARDOWN_SLOTS.contains(&slot) {
                teardown_named_ns += ns;
            }
            if RESOURCE_BUILD_SLOTS.contains(&slot) {
                rb_named_ns += ns;
            }
            if SUBMIT_TEARDOWN_SLOTS.contains(&slot) {
                submit_td_named_ns += ns;
            }
            if SUBMIT_SEAM_SLOTS.contains(&slot) {
                submit_seam_ns += ns;
            }
            if SUBMIT_RELEASE_SLOTS.contains(&slot) {
                submit_release_named_ns += ns;
            }
            if SUBMIT_VALIDATE_SLOTS.contains(&slot) {
                submit_validate_named_ns += ns;
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
            // The family and teardown bars are entered once per object, so the
            // count beside them is that family's population rather than the
            // window's submission count.
            if COUNTED_SLOTS.contains(&slot) {
                fields.push_str(&format!(" {name}_us={:.3} {name}_n={calls}", micros(ns)));
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
        let buffer_hit_n = std::mem::take(&mut self.buffer_hit_n);
        let buffer_miss_n = std::mem::take(&mut self.buffer_miss_n);
        let buffer_disabled_n = std::mem::take(&mut self.buffer_disabled_n);
        let buffer_return_n = std::mem::take(&mut self.buffer_return_n);
        let buffer_drop_n = std::mem::take(&mut self.buffer_drop_n);
        let compute_buffer_hit_n = std::mem::take(&mut self.compute_buffer_hit_n);
        let compute_buffer_miss_n = std::mem::take(&mut self.compute_buffer_miss_n);
        let compute_buffer_disabled_n = std::mem::take(&mut self.compute_buffer_disabled_n);
        let compute_buffer_return_n = std::mem::take(&mut self.compute_buffer_return_n);
        let compute_buffer_drop_n = std::mem::take(&mut self.compute_buffer_drop_n);
        let compute_pipeline_hit_n = std::mem::take(&mut self.compute_pipeline_hit_n);
        let compute_pipeline_miss_n = std::mem::take(&mut self.compute_pipeline_miss_n);
        let compute_pipeline_mismatch_n = std::mem::take(&mut self.compute_pipeline_mismatch_n);
        let compute_pipeline_disabled_n = std::mem::take(&mut self.compute_pipeline_disabled_n);
        let compute_pipeline_return_n = std::mem::take(&mut self.compute_pipeline_return_n);
        let compute_pipeline_drop_n = std::mem::take(&mut self.compute_pipeline_drop_n);
        let render_offscreen_n = std::mem::take(&mut self.render_offscreen_n);
        let render_present_n = std::mem::take(&mut self.render_present_n);
        let render_batch_n = std::mem::take(&mut self.render_batch_n);
        let render_batch_passes = std::mem::take(&mut self.render_batch_passes);
        let render_batch_passes_1 = std::mem::take(&mut self.render_batch_passes_1);
        let render_batch_passes_2 = std::mem::take(&mut self.render_batch_passes_2);
        let render_batch_passes_3_4 = std::mem::take(&mut self.render_batch_passes_3_4);
        let render_batch_passes_5_8 = std::mem::take(&mut self.render_batch_passes_5_8);
        let render_batch_passes_gt8 = std::mem::take(&mut self.render_batch_passes_gt8);
        let render_batch_traces_n = std::mem::take(&mut self.render_batch_traces_n);
        let render_batch_traces_passes = std::mem::take(&mut self.render_batch_traces_passes);
        let render_batch_open_n = std::mem::take(&mut self.render_batch_open_n);
        let render_batch_refused_n = std::mem::take(&mut self.render_batch_refused_n);
        let render_batch_refused_frame_n = std::mem::take(&mut self.render_batch_refused_frame_n);
        let render_batch_broken_n = std::mem::take(&mut self.render_batch_broken_n);
        let render_batch_broken_load_n = std::mem::take(&mut self.render_batch_broken_load_n);
        let landing_n = std::mem::take(&mut self.landing_n);
        let landing_bytes = std::mem::take(&mut self.landing_bytes);
        let staging_cached_n = std::mem::take(&mut self.staging_cached_n);
        let staging_plain_n = std::mem::take(&mut self.staging_plain_n);
        let binding_copy_calls = std::mem::take(&mut self.binding_copy_calls);
        let binding_copy_bytes = std::mem::take(&mut self.binding_copy_bytes);
        let binding_borrow_calls = std::mem::take(&mut self.binding_borrow_calls);
        let binding_borrow_bytes = std::mem::take(&mut self.binding_borrow_bytes);
        let resource_copy_views = std::mem::take(&mut self.resource_copy_views);
        let resource_copy_bytes = std::mem::take(&mut self.resource_copy_bytes);
        let resource_borrow_views = std::mem::take(&mut self.resource_borrow_views);
        let resource_borrow_bytes = std::mem::take(&mut self.resource_borrow_bytes);
        let td_image_n = std::mem::take(&mut self.td_image_n);
        let td_view_n = std::mem::take(&mut self.td_view_n);
        let td_sampler_n = std::mem::take(&mut self.td_sampler_n);
        let td_buffer_n = std::mem::take(&mut self.td_buffer_n);
        let td_memory_n = std::mem::take(&mut self.td_memory_n);
        let wait_object_calls = std::mem::take(&mut self.wait_object_calls);
        let wait_queue_calls = std::mem::take(&mut self.wait_queue_calls);
        let wait_timeline_calls = std::mem::take(&mut self.wait_timeline_calls);
        let wait_ahead_sum = std::mem::take(&mut self.wait_ahead_sum);
        let wait_ahead_calls = std::mem::take(&mut self.wait_ahead_calls);
        let build_objects = std::mem::take(&mut self.build_objects);
        self.window = 0;
        let plan_settle_us = micros(plan_settle_ns);
        let render_us = micros(render_ns);
        let render_residual_us = micros(render_residual_ns);
        let texture_named_us = micros(texture_named_ns);
        let readback_named_us = micros(readback_named_ns);
        let landing_named_us = micros(landing_named_ns);
        let teardown_named_us = micros(teardown_named_ns);
        let rb_named_us = micros(rb_named_ns);
        let submit_td_named_us = micros(submit_td_named_ns);
        let submit_seam_us = micros(submit_seam_ns);
        let submit_release_named_us = micros(submit_release_named_ns);
        let submit_validate_named_us = micros(submit_validate_named_ns);
        let mut wait_fields = String::with_capacity(200);
        for (object_slot, object_name) in WAIT_OBJECT_NAMES.iter().enumerate() {
            wait_fields.push_str(&format!(
                " {object_name}_n={}",
                wait_object_calls[object_slot]
            ));
        }
        wait_fields.push_str(&format!(
            " wait_queue_n={wait_queue_calls} wait_timeline_n={wait_timeline_calls} \
             wait_ahead_sum={wait_ahead_sum} wait_ahead_n={wait_ahead_calls}"
        ));
        let mut build_fields = String::with_capacity(200);
        for (object_slot, object_name) in BUILD_OBJECT_NAMES.iter().enumerate() {
            build_fields.push_str(&format!(" {object_name}_n={}", build_objects[object_slot]));
        }
        eprintln!(
            "PHASE submit n={n}{fields} fence_wait_skipped_n={skipped} \
             plan_settle_us={plan_settle_us:.3} render_us={render_us:.3} \
             render_residual_us={render_residual_us:.3} \
             texture_named_us={texture_named_us:.3} \
             readback_named_us={readback_named_us:.3} \
             landing_named_us={landing_named_us:.3} \
             teardown_named_us={teardown_named_us:.3} \
             rb_named_us={rb_named_us:.3} submit_td_named_us={submit_td_named_us:.3} \
             submit_seam_us={submit_seam_us:.3} \
             submit_release_named_us={submit_release_named_us:.3} \
             submit_validate_named_us={submit_validate_named_us:.3}{build_fields}{wait_fields} \
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
             buffer_hit_n={buffer_hit_n} buffer_miss_n={buffer_miss_n} \
             buffer_disabled_n={buffer_disabled_n} buffer_return_n={buffer_return_n} \
             buffer_drop_n={buffer_drop_n} \
             compute_buffer_hit_n={compute_buffer_hit_n} \
             compute_buffer_miss_n={compute_buffer_miss_n} \
             compute_buffer_disabled_n={compute_buffer_disabled_n} \
             compute_buffer_return_n={compute_buffer_return_n} \
             compute_buffer_drop_n={compute_buffer_drop_n} \
             compute_pipeline_hit_n={compute_pipeline_hit_n} \
             compute_pipeline_miss_n={compute_pipeline_miss_n} \
             compute_pipeline_mismatch_n={compute_pipeline_mismatch_n} \
             compute_pipeline_disabled_n={compute_pipeline_disabled_n} \
             compute_pipeline_return_n={compute_pipeline_return_n} \
             compute_pipeline_drop_n={compute_pipeline_drop_n} \
             render_offscreen_n={render_offscreen_n} \
             render_present_n={render_present_n} \
             render_batch_n={render_batch_n} \
             render_batch_passes={render_batch_passes} \
             render_batch_passes_1={render_batch_passes_1} \
             render_batch_passes_2={render_batch_passes_2} \
             render_batch_passes_3_4={render_batch_passes_3_4} \
             render_batch_passes_5_8={render_batch_passes_5_8} \
             render_batch_passes_gt8={render_batch_passes_gt8} \
             render_batch_traces_n={render_batch_traces_n} \
             render_batch_traces_passes={render_batch_traces_passes} \
             render_batch_open_n={render_batch_open_n} \
             render_batch_refused_n={render_batch_refused_n} \
             render_batch_refused_frame_n={render_batch_refused_frame_n} \
             render_batch_broken_n={render_batch_broken_n} \
             render_batch_broken_load_n={render_batch_broken_load_n} \
             landing_n={landing_n} landing_bytes={landing_bytes} \
             staging_cached_n={staging_cached_n} staging_plain_n={staging_plain_n} \
             submit_binding_copies_n={binding_copy_calls} \
             submit_binding_copies_bytes={binding_copy_bytes} \
             submit_binding_borrows_n={binding_borrow_calls} \
             submit_binding_borrows_bytes={binding_borrow_bytes} \
             submit_resource_copies_n={resource_copy_views} \
             submit_resource_copies_bytes={resource_copy_bytes} \
             submit_resource_borrows_n={resource_borrow_views} \
             submit_resource_borrows_bytes={resource_borrow_bytes} \
             td_image_n={td_image_n} td_view_n={td_view_n} td_sampler_n={td_sampler_n} \
             td_buffer_n={td_buffer_n} td_memory_n={td_memory_n}",
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

/// One pooled binding's bytes the submission *copied* for itself: the trace's
/// own snapshot bytes, cloned into the binding's own vector
/// (`crate::submit_binding_borrow`). The other two binding sources are always
/// owned and are not counted here.
#[inline]
pub(crate) fn note_binding_copy(bytes: u64) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        local.binding_copy_calls += 1;
        local.binding_copy_bytes += bytes;
    });
}

/// The same bytes the sixth cut's mechanism *borrowed* from the table the
/// submission already holds. A round reads the pair of counters to say how many
/// bytes the mechanism took off `pool` and the release.
#[inline]
pub(crate) fn note_binding_borrow(bytes: u64) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        local.binding_borrow_calls += 1;
        local.binding_borrow_bytes += bytes;
    });
}

/// The declared bytes one **pool derivation** copied for itself: the views
/// `ComputeTrace::serial_resources` cloned, by count and by byte
/// (`crate::serial_resources_borrow`). One submission derives its pool twice —
/// in `plan` and in `submit_validate` — so a submission that declares N bytes
/// reads `2N` here on the pre-cut path.
#[inline]
pub(crate) fn note_resource_copy(views: u64, bytes: u64) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        local.resource_copy_views += views;
        local.resource_copy_bytes += bytes;
    });
}

/// The same declared bytes the seventh cut's *lending* derivation borrowed:
/// `ComputeTrace::serial_resources_ref` copies no view, so the bytes it hands
/// over are the trace's own. Read beside [`note_resource_copy`], the pair says
/// which arm ran and how much of the declaration the derivation moved.
#[inline]
pub(crate) fn note_resource_borrow(views: u64, bytes: u64) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        local.resource_borrow_views += views;
        local.resource_borrow_bytes += bytes;
    });
}

/// Whether the profile is on, for a call site that would otherwise *compute* a
/// reading's own inputs before handing them to a `note_*`.
///
/// The byte counters are the one kind of reading whose inputs cost something to
/// collect (a walk over the pool), so the site that collects them asks this
/// first: with the profile off it is the same one relaxed load every other
/// instrumented site pays, and nothing else is touched.
#[inline]
pub(crate) fn counting() -> bool {
    enabled()
}

/// One device object a `resource_build` family created.
///
/// The bars the fifth cut adds to `resource_build` are *regions* — a whole
/// create call for the pipeline objects, the buffers, the descriptor sets and
/// the indirect replay, and one object's own backing for the images, views and
/// samplers of the sampled declarations — so a region count is not an object
/// count. These are the object counts, taken at the same sites the bars are
/// (a null handle counts nothing, exactly as the teardown census does), and
/// they are what a per-object cost is read with:
///
/// ```text
/// per object = rb_<family>_us / rb_<family>_n
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BuildObject {
    /// A pipeline-shaped object group: the pipeline, its layout and the two
    /// shader modules.
    Pipeline,
    /// One `vkCreateBuffer`, whatever it carries (a pooled view's backing, a
    /// heap slab's placement, a storage image's transfer buffer, an indirect
    /// replay's command buffer, an imported host window).
    Buffer,
    /// One `vkCreateImage` (a sampled declaration's linear image or a storage
    /// declaration's optimal-tiling one).
    Image,
    /// One `vkCreateImageView`.
    View,
    /// One `vkCreateSampler` — a declaration's own or a pipeline's static one.
    Sampler,
    /// One descriptor set allocated (`vkAllocateDescriptorSets`), which is one
    /// recorded pass's own immutable set.
    Descriptor,
    /// One indirect replay's command buffer.
    Indirect,
    /// One `vkAllocateMemory`, counted beside whichever family the backing it
    /// belongs to was charged to. A round that wants the allocation's own share
    /// of a family divides that family's bar by this count.
    Memory,
}

const BUILD_OBJECT_COUNT: usize = BuildObject::Memory as usize + 1;

/// The printed field-name stem of each build object, in slot order.
const BUILD_OBJECT_NAMES: [&str; BUILD_OBJECT_COUNT] = [
    "rb_pipeline",
    "rb_buffer",
    "rb_image",
    "rb_view",
    "rb_sampler",
    "rb_descriptor",
    "rb_indirect",
    "rb_memory",
];

/// One device object a teardown really destroyed (the handle was not null, so
/// the count is the population the `teardown_*` bars were spent on — not the
/// slots the drop walked past).
///
/// The families are the ones the printed counters name: an image (a colour
/// attachment, a depth or stencil surface, its resolve target or a sampled
/// declaration's own image), an image view, a sampler, a buffer (a readback
/// destination, a stage buffer, an indirect or index buffer, a caller-held
/// stream or a previous-byte staging buffer) and a device memory freed beside
/// one of them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TeardownObject {
    Image,
    View,
    Sampler,
    Buffer,
    Memory,
}

#[inline]
pub(crate) fn note_teardown_object(object: TeardownObject) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        match object {
            TeardownObject::Image => local.td_image_n += 1,
            TeardownObject::View => local.td_view_n += 1,
            TeardownObject::Sampler => local.td_sampler_n += 1,
            TeardownObject::Buffer => local.td_buffer_n += 1,
            TeardownObject::Memory => local.td_memory_n += 1,
        }
    });
}

/// Count one device object a `resource_build` region created ([`BuildObject`]).
///
/// Counted at the creation site rather than derived from a bar's call count:
/// the `rb_*` bars are whole-call regions for the pipeline objects, the
/// buffers, the descriptor sets and the indirect replay, so only an explicit
/// count can say how many objects those regions made. A round divides the
/// family's bar by this to get one object's cost, which is the reading a
/// pooling decision is made from.
#[inline]
pub(crate) fn note_build_object(object: BuildObject) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| local.borrow_mut().build_objects[object as usize] += 1);
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
        if phase == Phase::Total {
            LOCAL.with(|local| local.borrow_mut().depth += 1);
        }
        Some(Self {
            slot: phase,
            started: Instant::now(),
        })
    }

    /// Enter a bar that is only charged while a submission's enclosing `total`
    /// bar is open: a deferred object API retires a submission's resources from
    /// `wait`, outside any `total`, and charging that region to the window of
    /// the call that happened to retire it would make `sum(fields) <= total`
    /// untrue without saying so. Resolving to `None` outside a submission keeps
    /// the window's fields a partition of its own `total`.
    #[inline]
    pub(crate) fn enter_in_submission(phase: Phase) -> Option<Self> {
        if !enabled() {
            return None;
        }
        if LOCAL.with(|local| local.borrow().depth) == 0 {
            return None;
        }
        Some(Self {
            slot: phase,
            started: Instant::now(),
        })
    }

    #[inline]
    pub(crate) fn enter_fence_wait(
        phase: Phase,
        object: WaitObject,
        ahead: usize,
    ) -> Option<FenceWaitBar> {
        if !enabled() {
            return None;
        }
        Some(FenceWaitBar {
            phase,
            object,
            ahead,
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
                // The enclosing bar is the one that closes the depth the
                // teardown bar's own guard reads; children have already
                // dropped, so this is the last thing in the submission.
                local.depth = local.depth.saturating_sub(1);
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
    object: WaitObject,
    /// How many *other* submissions the waited queue had not retired when this
    /// wait began: zero is a wait for one's own work on an idle queue.
    ahead: usize,
    started: Instant,
    timed_out: bool,
}

impl Drop for FenceWaitBar {
    #[inline]
    fn drop(&mut self) {
        let ns = elapsed_ns(self.started.elapsed());
        let phase = self.phase;
        let object = self.object;
        let ahead = self.ahead;
        let timed_out = self.timed_out;
        LOCAL.with(|local| {
            local
                .borrow_mut()
                .note_fence_wait(phase, ns, timed_out, object, ahead)
        });
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

/// Count one fence wait whose microseconds belong to another bar.
///
/// Called where the wait's own region is already named (`landing_wait`, the
/// present rail's sentinel) so that the window's wait-object census covers
/// every fence the provider waits on while no microsecond is counted twice.
#[inline]
pub(crate) fn note_fence_wait_object(object: WaitObject, ahead: usize) {
    if !enabled() {
        return;
    }
    LOCAL.with(|local| local.borrow_mut().note_wait_object(object, ahead));
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
        // The teardown split (the fourth round) is nested inside a residual
        // slot rather than beside it, so its members must be neither the
        // enclosing bar nor any sibling the disjoint sum or another split
        // already claims — a slot in two lists would be counted twice by a
        // reader who summed them.
        for slot in TEARDOWN_SLOTS {
            assert_ne!(slot, Phase::RenderTeardown as usize);
            assert!(!RENDER_SLOTS.contains(&slot));
            assert!(!TEXTURE_SLOTS.contains(&slot));
            assert!(!READBACK_ARM_SLOTS.contains(&slot));
            assert!(!LANDING_SLOTS.contains(&slot));
            assert!(!RENDER_RESIDUAL_SLOTS.contains(&slot));
        }
        assert_eq!(PHASE_NAMES[Phase::TeardownSync as usize], "teardown_sync");
        assert_eq!(
            PHASE_NAMES[Phase::TeardownPrevious as usize],
            "teardown_previous",
            "the teardown split's last slot is the previous-byte buffers"
        );
        // The three pooled returns and the retain release are residual
        // siblings of the teardown bar, not children of it: a slot in both
        // sets would count the same microseconds twice.
        for phase in [
            Phase::RenderReleaseReuse,
            Phase::RenderReleasePool,
            Phase::RenderReleaseImport,
            Phase::RenderRetire,
        ] {
            assert!(RENDER_RESIDUAL_SLOTS.contains(&(phase as usize)));
            assert!(!TEARDOWN_SLOTS.contains(&(phase as usize)));
            assert_ne!(phase, Phase::RenderTeardown);
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
        local.note_fence_wait(Phase::FenceWait, 1_000, false, WaitObject::SubmitFence, 0);
        local.note_fence_wait(
            Phase::FenceWait,
            IDLE_NS_DEFAULT,
            false,
            WaitObject::SubmitFence,
            2,
        );
        local.note_fence_wait(
            Phase::FenceWait,
            5_000_000,
            true,
            WaitObject::SubmitFence,
            0,
        );
        local.note_fence_wait(Phase::RenderWait, 7, false, WaitObject::RenderFence, 1);
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
        // The wait-object census partitions those same waits by what they
        // blocked on, and the depth reading separates "my own work on an idle
        // queue" from "behind earlier submissions".
        assert_eq!(local.wait_object_calls[WaitObject::SubmitFence.slot()], 3);
        assert_eq!(local.wait_object_calls[WaitObject::RenderFence.slot()], 1);
        assert_eq!(local.wait_object_calls[WaitObject::LandingFence.slot()], 0);
        assert_eq!(local.wait_object_calls[WaitObject::PresentFence.slot()], 0);
        assert_eq!(local.wait_ahead_sum, 3);
        assert_eq!(local.wait_ahead_calls, 2);
        assert_eq!(
            local.wait_queue_calls + local.wait_timeline_calls,
            0,
            "this provider waits on binary fences only"
        );
    }

    /// The fifth cut's two nested splits stay nested, and the seam slots stay
    /// outside the disjoint sum: a family bar inside the disjoint sum would
    /// make `sum(fields) <= total` untrue, and a seam slot inside
    /// `resource_build` or `submit_teardown` would be counted twice by a reader
    /// who summed both.
    #[test]
    fn the_fifth_cut_stays_inside_what_it_divides() {
        for slot in RESOURCE_BUILD_SLOTS {
            assert_ne!(slot, Phase::ResourceBuild as usize);
            assert!(!SUBMIT_SEAM_SLOTS.contains(&slot));
            assert!(!SUBMIT_TEARDOWN_SLOTS.contains(&slot));
            assert!(
                !COUNTED_SLOTS.contains(&slot),
                "a family bar's call count is a region count, not an object count"
            );
        }
        // Each family bar has an object counter of the same name, so a round
        // divides the field by the count that sits beside it.
        assert_eq!(BUILD_OBJECT_NAMES.len(), BUILD_OBJECT_COUNT);
        assert_eq!(BUILD_OBJECT_NAMES.len(), RESOURCE_BUILD_SLOTS.len() + 1);
        for (object_slot, name) in BUILD_OBJECT_NAMES.iter().enumerate() {
            if *name == "rb_memory" {
                continue;
            }
            assert_eq!(
                PHASE_NAMES[RESOURCE_BUILD_SLOTS[object_slot]], *name,
                "the object census and the family bars must line up slot for slot"
            );
        }
        for slot in SUBMIT_TEARDOWN_SLOTS {
            assert_ne!(slot, Phase::SubmitTeardown as usize);
            assert!(SUBMIT_SEAM_SLOTS.contains(&(Phase::SubmitTeardown as usize)));
            assert!(!RENDER_SLOTS.contains(&slot));
            assert!(!RENDER_RESIDUAL_SLOTS.contains(&slot));
            assert!(!TEARDOWN_SLOTS.contains(&slot));
            assert!(!RESOURCE_BUILD_SLOTS.contains(&slot));
        }
        // `render_teardown` and `submit_teardown` are two halves of one
        // reading, so neither may appear in the other's nested set.
        assert!(!TEARDOWN_SLOTS.contains(&(Phase::SubmitTeardown as usize)));
        for slot in SUBMIT_SEAM_SLOTS {
            assert!(!RENDER_SLOTS.contains(&slot));
            assert!(!RENDER_RESIDUAL_SLOTS.contains(&slot));
            assert!(!RESOURCE_BUILD_SLOTS.contains(&slot));
            assert!(!TEARDOWN_SLOTS.contains(&slot));
            assert!(!LANDING_SLOTS.contains(&slot));
            assert!(!COUNTED_SLOTS.contains(&slot));
        }
        assert_eq!(PHASE_NAMES[Phase::RbPipeline as usize], "rb_pipeline");
        assert_eq!(
            PHASE_NAMES[Phase::SubmitValidate as usize],
            "submit_validate",
            "the seam split's last slot is the terminal contract validation"
        );
        assert_eq!(
            PHASE_NAMES[Phase::SubmitTeardown as usize],
            "submit_teardown"
        );
        assert_eq!(WAIT_OBJECT_NAMES.len(), WAIT_OBJECT_COUNT);
    }

    /// The sixth cut's two splits divide regions the fifth cut named, and
    /// neither may be read as part of the disjoint sum: the release is inside
    /// the enclosing `total` *after* every numbered bar, and the validation's
    /// two halves are inside `submit_validate`. The seam's own set names the
    /// release but not its children, so `submit_seam_us` never counts the tail
    /// twice.
    #[test]
    fn the_sixth_cut_stays_inside_what_it_divides() {
        for slot in SUBMIT_RELEASE_SLOTS {
            assert_ne!(slot, Phase::SubmitRelease as usize);
            assert!(!SUBMIT_VALIDATE_SLOTS.contains(&slot));
            assert!(
                !SUBMIT_SEAM_SLOTS.contains(&slot),
                "a child of the release is not a seam bar of its own"
            );
            assert!(!SUBMIT_TEARDOWN_SLOTS.contains(&slot));
            assert!(!RESOURCE_BUILD_SLOTS.contains(&slot));
            assert!(!COUNTED_SLOTS.contains(&slot));
        }
        for slot in SUBMIT_VALIDATE_SLOTS {
            assert_ne!(slot, Phase::SubmitValidate as usize);
            assert!(!SUBMIT_RELEASE_SLOTS.contains(&slot));
            assert!(
                !SUBMIT_SEAM_SLOTS.contains(&slot),
                "a child of the validation is not a seam bar of its own"
            );
            assert!(!COUNTED_SLOTS.contains(&slot));
        }
        assert!(SUBMIT_SEAM_SLOTS.contains(&(Phase::SubmitRelease as usize)));
        assert!(SUBMIT_SEAM_SLOTS.contains(&(Phase::SubmitValidate as usize)));
        assert_eq!(PHASE_NAMES[Phase::SubmitRelease as usize], "submit_release");
        assert_eq!(
            PHASE_NAMES[Phase::SubmitValidateDerive as usize],
            "submit_validate_derive"
        );
        assert_eq!(
            PHASE_NAMES[Phase::SubmitValidateCheck as usize],
            "submit_validate_check"
        );
        // The release has exactly three children, and the validation has exactly
        // three regions (the seventh cut named the release that used to be its
        // seam): a set that missed one would read as a seam rather than as an
        // unsplit region.
        assert_eq!(SUBMIT_RELEASE_SLOTS.len(), 3);
        assert_eq!(SUBMIT_VALIDATE_SLOTS.len(), 3);
        // The seventh cut's pool bar is a *nested* child of `submit_release_views`
        // rather than a fourth sibling: a set that held it would make
        // `submit_release_named_us` count the pool twice, once in its parent and
        // once on its own.
        assert!(!SUBMIT_RELEASE_SLOTS.contains(&(Phase::SubmitReleasePool as usize)));
        assert_eq!(
            SUBMIT_RELEASE_VIEWS_SLOTS,
            [Phase::SubmitReleasePool as usize]
        );
        assert_ne!(
            Phase::SubmitReleasePool as usize,
            Phase::SubmitReleaseViews as usize
        );
        assert_eq!(
            PHASE_NAMES[Phase::SubmitReleasePool as usize],
            "submit_release_pool"
        );
        assert_eq!(
            PHASE_NAMES[Phase::SubmitValidateRelease as usize],
            "submit_validate_release"
        );
        assert_eq!(PHASE_NAMES[Phase::PlanResources as usize], "plan_resources");
        // The seventh cut's three regions are nested exactly like the sixth
        // cut's: one inside `plan`, one inside `submit_validate` and one inside
        // `submit_release_views`, and none of them is a seam bar or a member of
        // the disjoint sum.
        for slot in [
            Phase::PlanResources as usize,
            Phase::SubmitValidateRelease as usize,
            Phase::SubmitReleasePool as usize,
        ] {
            assert!(
                !SUBMIT_SEAM_SLOTS.contains(&slot),
                "a nested region is not a seam bar of its own"
            );
            assert!(!COUNTED_SLOTS.contains(&slot));
            assert!(!RESOURCE_BUILD_SLOTS.contains(&slot));
            assert!(!SUBMIT_TEARDOWN_SLOTS.contains(&slot));
            assert!(!RENDER_SLOTS.contains(&slot));
        }
        assert!(SUBMIT_RELEASE_VIEWS_SLOTS.contains(&(Phase::SubmitReleasePool as usize)));
        assert!(SUBMIT_VALIDATE_SLOTS.contains(&(Phase::SubmitValidateRelease as usize)));
        assert!(PLAN_SETTLE_SLOTS.contains(&(Phase::Plan as usize)));
        // The release's three children are disjoint from the nested pool child
        // and from each other, so a reader adds the three and reads the fourth
        // as a share of `submit_release_views`.
        for slot in SUBMIT_RELEASE_VIEWS_SLOTS {
            assert!(!SUBMIT_RELEASE_SLOTS.contains(&slot));
            assert!(!SUBMIT_VALIDATE_SLOTS.contains(&slot));
        }
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
