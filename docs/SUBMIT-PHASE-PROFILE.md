# The submission phase profile

`VulkanComputeProvider::submit` is one call that does not return until the work
it submitted has retired on the device and its host-visible bytes have been read
back. On the guest desktop the reims rail measures exactly that call
(`FrameSpan::ProvSubmit`), which makes it the largest single bar of a draw — and
also an opaque one: the plan, the resource creation, the recording, the
`vkQueueSubmit`, the `vkWaitForFences` and the host-mapped readback all happen
inside it.

This crate can split that call into named bars. The switch is
`METAL_API_VULKAN_PHASE_PROFILE`:

| value | effect |
|---|---|
| unset (default), `0`, `off`, `no`, anything else | off |
| `1`, `on`, `ON`, `true`, `yes` | on |

Off is the default and costs one relaxed load per bracket: every call site is
`let _bar = Bar::enter(Phase::X)`, which resolves to `None` without reading a
clock, taking a lock or allocating. With the profile off the emitted frames and
the executed commands are byte for byte what they were before this module
existed; the conformance suites are the evidence for that, because they compare
captured bytes rather than timings.

A second switch, `METAL_API_VULKAN_SUBMIT_SAMPLES`, prints **one line per
completed submission** instead of one line per window — the distribution rather
than the mean. It is described under [One line per
submission](#one-line-per-submission); the two switches are read apart, so
either one works without the other.

## The line

With the profile on, each thread that submits prints one line to **stderr** every
`METAL_API_VULKAN_PHASE_PROFILE_EVERY` (default 256) completed submissions —
never one line per draw:

```text
PHASE submit n=256 total_us=... admit_us=... plan_us=... pool_us=... resource_build_us=... record_us=... queue_submit_us=... fence_wait_us=... fence_wait_idle_n=... fence_wait_idle_us=... fence_wait_blocked_n=... fence_wait_blocked_us=... fence_wait_timeout_n=... read_updates_us=... render_total_us=... render_setup_us=... render_record_us=... render_submit_us=... render_wait_us=... render_wait_idle_n=... render_wait_idle_us=... render_wait_blocked_n=... render_wait_blocked_us=... render_wait_timeout_n=... render_readback_us=... writebacks_us=... settle_us=... fence_wait_skipped_n=... plan_settle_us=... render_us=...

PHASE submit n=256 ... readback_rect_n=... readback_rect_bytes=...
readback_rect_extent_bytes=... readback_full_n=... readback_full_bytes=...
readback_switch_n=... readback_shape_n=... readback_bounds_n=...
readback_whole_n=... setup_admits_us=... setup_attachments_us=...
setup_depth_stencil_us=... setup_render_pass_us=... setup_textures_us=...
setup_stage_buffers_us=... setup_pipeline_us=... setup_readbacks_us=...
setup_inputs_us=... setup_command_pool_us=... reuse_hit_n=... reuse_miss_n=...
reuse_mismatch_n=... reuse_unkeyed_n=... reuse_disabled_n=... texture_backing_us=...
texture_upload_us=... texture_view_us=... texture_sampler_us=...
texture_import_us=... texture_descriptor_us=... render_resolve_us=...
render_present_us=... render_publish_us=... render_landing_us=...
render_prepare_us=... render_retain_us=... render_land_owner_us=...
render_teardown_us=... render_residual_us=... texture_named_us=...
pool_hit_n=... pool_miss_n=... pool_disabled_n=... pool_return_n=...
pool_drop_n=... import_hit_n=... import_miss_n=... import_disabled_n=...
import_return_n=... import_drop_n=... buffer_hit_n=... buffer_miss_n=...
buffer_disabled_n=... buffer_return_n=... buffer_drop_n=...
compute_buffer_hit_n=... compute_buffer_miss_n=... compute_buffer_disabled_n=...
compute_buffer_return_n=... compute_buffer_drop_n=...
compute_pipeline_hit_n=... compute_pipeline_miss_n=... compute_pipeline_mismatch_n=...
compute_pipeline_disabled_n=... compute_pipeline_return_n=... compute_pipeline_drop_n=...
render_offscreen_n=... render_present_n=...
readback_rect_us=... readback_full_us=... readback_seed_us=...
readback_surfaces_us=... readback_shape_us=... readback_named_us=...
landing_lookup_us=... landing_windows_us=... landing_stage_us=...
landing_record_us=... landing_wait_us=... landing_fetch_us=...
landing_write_us=... landing_release_us=... landing_named_us=...
landing_n=... landing_bytes=... staging_cached_n=... staging_plain_n=...
teardown_sync_us=... teardown_pipeline_us=... teardown_passes_us=...
teardown_textures_us=... teardown_descriptors_us=...
teardown_depth_stencil_us=... teardown_attachments_us=...
teardown_readbacks_us=... teardown_buffers_us=... teardown_previous_us=...
teardown_named_us=... render_release_reuse_us=... render_release_pool_us=...
render_release_import_us=... render_retire_us=...
td_image_n=... td_view_n=... td_sampler_n=... td_buffer_n=... td_memory_n=...
td_image_us=... td_view_us=... td_buffer_us=... td_memory_us=...
td_object_named_us=... td_unreturned_n=... td_unreturned_<kind>_n=...
buf_miss_<kind>_n=... buf_return_<kind>_n=... buf_miss_empty_n=...
buf_miss_occupied_n=... buf_miss_held_sum=... buf_miss_after_evict_n=...
pool_miss_empty_n=... pool_miss_occupied_n=... pool_miss_held_sum=...
pool_miss_after_evict_n=... pool_miss_same_extent_n=...
pool_miss_same_format_n=... pool_miss_same_image_type_n=...
pool_miss_same_view_type_n=... pool_miss_same_device_copy_n=...
render_release_draws_us=...
rb_pipeline_us=... rb_buffer_us=... rb_image_us=... rb_view_us=...
rb_sampler_us=... rb_descriptor_us=... rb_indirect_us=... rb_named_us=...
submit_teardown_us=... submit_td_sync_us=... submit_td_sync_n=...
submit_td_pipeline_us=... submit_td_pipeline_n=... submit_td_buffers_us=...
submit_td_buffers_n=... submit_td_textures_us=... submit_td_textures_n=...
submit_td_retains_us=... submit_td_retains_n=... submit_td_named_us=...
submit_lock_us=... submit_bookkeep_us=... submit_merge_us=...
submit_validate_us=... submit_validate_derive_us=... submit_validate_check_us=...
submit_validate_release_us=... submit_validate_named_us=...
submit_release_us=... submit_release_bindings_us=... submit_release_views_us=...
submit_release_pool_us=... submit_release_plan_us=... submit_release_named_us=...
plan_resources_us=... staging_window_us=... submit_seam_us=...
submit_resource_copies_n=... submit_resource_copies_bytes=...
submit_resource_borrows_n=... submit_resource_borrows_bytes=...
staging_window_copies_n=... staging_window_copies_bytes=...
staging_window_shares_n=... staging_window_shares_bytes=...
rb_pipeline_n=... rb_buffer_n=... rb_image_n=... rb_view_n=... rb_sampler_n=...
rb_descriptor_n=... rb_indirect_n=... rb_memory_n=...
wait_submit_n=... wait_render_n=... wait_landing_n=... wait_present_n=...
wait_queue_n=... wait_timeline_n=... wait_ahead_sum=... wait_ahead_n=...
```

Every µs field is a **sum over that line's own window**, not a mean, with three
decimals (nanosecond resolution). Adding lines and dividing by the summed `n=`
gives the round's mean without a weighting error, and it makes the line's own
identity checkable:

```text
admit + plan + pool + resource_build + record + queue_submit + fence_wait
      + read_updates + render_setup + render_record + render_submit
      + render_wait + render_readback + writebacks + settle
      + submit_teardown + submit_lock + submit_bookkeep
      + submit_merge + submit_validate  <=  total
```

The difference is the seam between the bars — plain function calls, `Arc`
clones, the function-call boundary — and is reported as the residual rather than
hidden in one of the fields. The five `submit_*` seam fields above name most of
that residual and their sum is printed as `submit_seam_us`; what is left after
them is the boundary itself. `total` and `render_total` are the enclosing bars;
the others are disjoint regions, so no field to be summed can contain another. The render
half has its own identity, `render_setup + render_record + render_submit +
render_wait + render_readback <= render_total`, whose difference is the work of
a render path this split does not name (the present and indirect-replay paths
have their own setup and readback).

| field | region |
|---|---|
| `total` | the whole `submit` call, success or refusal |
| `admit` | epoch check, capability admission, indirect-command resolve |
| `plan` | render plan, heap placement, registered pipelines, serial pool, dispatch list |
| `pool` | device bindings for the pooled views: owned-byte copies, staged-lease resolution, guest-run gathers, borrow retains |
| `resource_build` | pipeline objects, buffers, textures, static samplers, descriptor sets, indirect buffer |
| `record` | command-buffer recording |
| `queue_submit` | completion fence creation and `vkQueueSubmit` |
| `fence_wait` | `vkWaitForFences`, split idle/blocked (below) |
| `read_updates` | host-mapped readback of every writable view's bytes |
| `render_total` | the enclosing bar for the render/present half executed inside the same `submit` |
| `render_setup` | an offscreen pass's resolution, device objects and readback destinations, before recording |
| `render_record` | the render pass's command recording, `vkCmdCopyImageToBuffer` included |
| `render_submit` | the render half's queue lock, completion fence and `vkQueueSubmit` |
| `render_wait` | the render half's `vkWaitForFences` — where the pass's copy-out runs (idle/blocked split) |
| `render_readback` | the render half's host-visible readback: stored attachments, depth/stencil, writable stage buffers |
| `writebacks` | writeback mapping and the contract validation of the merged result |
| `settle` | terminal observation, completion-record insert, health synchronisation |
| `plan_settle` | the aggregate `plan + pool + settle` |
| `render` | the aggregate `render_setup + render_record + render_submit + render_wait + render_readback` |
| `setup_admits` | inside `render_setup`: the admissions the rail re-runs on the request |
| `setup_attachments` | inside `render_setup`: the colour attachments' images, or the resident targets a pass borrows |
| `setup_depth_stencil` | inside `render_setup`: the depth/stencil surface, its resolve target and the readbacks that face lands in |
| `setup_render_pass` | inside `render_setup`: the render pass, the seed render pass and both framebuffers |
| `setup_textures` | inside `render_setup`: the sampled textures, their uploads, samplers and descriptor set |
| `setup_stage_buffers` | inside `render_setup`: the stage buffers, their sets and the layouts those sets occupy |
| `setup_pipeline` | inside `render_setup`: the two shader modules, the pipeline layout and the graphics pipeline |
| `setup_readbacks` | inside `render_setup`: the readback plan and the stored attachments' destinations |
| `setup_inputs` | inside `render_setup`: the caller-held streams, the previous-byte buffers and the indirect commands |
| `setup_command_pool` | inside `render_setup`: the command pool and its command buffer |
| `texture_backing` | inside `setup_textures`: one sampled declaration's backing (or nothing, when the pool hands one back) |
| `texture_upload` | inside `setup_textures`: the texels' own trip into that backing |
| `texture_view` | inside `setup_textures`: the image view a pooled backing did not carry |
| `texture_sampler` | inside `setup_textures`: the samplers one declaration's slots state |
| `texture_import` | inside `setup_textures`: the host-pointer import of an owner's no-copy window |
| `texture_descriptor` | inside `setup_textures`: the sampled set's layout, pool, set and writes |
| `render_resolve` | inside the render half's residual: the outer loop's per-entry resolution before the rail is called |
| `render_present` | inside the render half's residual: the present rail's own pass |
| `render_publish` | inside the render half's residual: the outer loop's per-entry publication after the rail returned |
| `render_landing` | inside the render half's residual: one landing-only plan entry |
| `render_prepare` | inside the render half's residual: the offscreen rail entry's admissions and request resolution |
| `render_retain` | inside the render half's residual: the input retains a pass takes before its first import |
| `render_land_owner` | inside the render half's residual: the owner-window landing that follows a successful pass |
| `render_teardown` | inside the render half's residual: the pass objects' destruction once the fence proved the device done |
| `render_release_reuse` | inside the render half's residual, beside `render_teardown`: handing the pipeline-shaped objects back to the shape cache |
| `render_release_pool` | inside the render half's residual: handing the sampled textures' pooled backing back, eviction included |
| `render_release_import` | inside the render half's residual: handing the owner-window imports back |
| `render_retire` | inside the render half's residual: releasing the input retains the pass took |
| `render_release_draws` | inside the render half's residual: a draw list's **per-draw** objects handing back (`crate::draw_object_release`) — the four families `render_release_reuse` to `render_release_uploads` hand back for the pass's own set. Never entered with the switch off |
| `teardown_sync` | inside `render_teardown`: the completion fence and the command pool |
| `teardown_pipeline` | inside `render_teardown`: the pipeline, its layout and the two shader modules the shape cache did not take |
| `teardown_passes` | inside `render_teardown`: the render pass, the seed render pass and both framebuffers |
| `teardown_textures` | inside `render_teardown`: the sampled declarations' own samplers, views, images, memories and imported windows |
| `teardown_descriptors` | inside `render_teardown`: the descriptor pools, set layouts and the empty layouts of unused set positions |
| `teardown_depth_stencil` | inside `render_teardown`: the depth and stencil surfaces with their resolve targets |
| `teardown_attachments` | inside `render_teardown`: the colour attachments' images, views, memories and resolve targets |
| `teardown_readbacks` | inside `render_teardown`: the readback destinations — one unmap, buffer and memory per stored attachment |
| `teardown_buffers` | inside `render_teardown`: the stage buffers, the indirect and index buffers and the caller-held vertex streams |
| `teardown_previous` | inside `render_teardown`: the previous-byte staging buffers a `Load` uploaded from |
| `td_image` / `td_view` / `td_buffer` / `td_memory` | inside the ten `teardown_*` groups, **once per object**: the single `vkDestroyImage` / `vkDestroyImageView` / `vkDestroyBuffer` / `vkFreeMemory` call that object pays. Read with the matching `td_*_n` (`td_memory_us ÷ td_memory_n` is one allocation's cost) |
| `readback_rect` | inside `render_readback`: one stored attachment's trimmed frame — the written rectangle copied out of the mapping and the seed rebuilt around it |
| `readback_full` | inside `render_readback`: one stored attachment's whole extent copied out of its mapping |
| `readback_seed` | inside `readback_rect`: the rebuild itself (the seed and the patch), which only the trimmed arm pays |
| `readback_surfaces` | inside `render_readback`: the depth, stencil and writable stage-buffer copy-outs |
| `readback_shape` | inside `setup_readbacks` (not inside `render_readback`): the decision itself, taken before any device object exists. This is a *time*; the `readback_shape_n` counter next to the `readback_*` counts is the number of attachments that decision sent to the whole-extent arm |
| `landing_lookup` | inside `render_landing`: the kept frame's target and the landing view's own declaration |
| `landing_windows` | inside `render_landing`: the owner windows the frame lands in, resolved against the lease channel |
| `landing_stage` | inside `render_landing`: the staging buffer, its memory and mapping, the command pool, the command buffer and the fence |
| `landing_record` | inside `render_landing`: the copy's recording and its submission |
| `landing_wait` | inside `render_landing`: `vkWaitForFences` for the landing copy |
| `landing_fetch` | inside `render_landing`: the host read of the copied frame — the part a landing shares with the readback channel |
| `landing_write` | inside `render_landing`: the write into the owner's live pages |
| `landing_release` | inside `render_landing`: destroying the fence, the command pool, the mapping, the buffer and its memory |
| `rb_pipeline` | inside `resource_build`: the pipeline-shaped objects one compute pipeline needs — `create_pipeline_objects`, one bar per compute pass, enclosing the module, the descriptor-set layout, the pipeline layout and the pipelines. The bar covers the shape table's lookup too, on either arm; `rb_pipeline_n` counts only the groups the window **built**, so a group the table handed over is a smaller bar and a population that `compute_pipeline_hit_n` names (`docs/COMPUTE-PIPELINE-REUSE.md`) |
| `rb_buffer` | inside `resource_build`: every device buffer the submission builds with its memory bound and uploaded — `create_buffers` as one region (owned backings, the heap slab's placements, the imported host windows, the storage image's transfer buffer). A creation of a shape the compute pool holds takes the pool's pair instead of building one (`docs/COMPUTE-BUFFER-POOL.md`), which is exactly what `compute_buffer_hit_n` counts; the family's own `rb_buffer_n` counts the buffers the window really built, so a hit arm's region is smaller than its population |
| `rb_image` | inside `resource_build`: one sampled or storage image's backing — `allocate_image_backing` plus, for the sampled arm, the texels' trip into it (`create_textures` / `create_storage_texture`, one bar per image) |
| `rb_view` | inside `resource_build`: one image view (`create_color_image_view` in the compute texture paths, one bar per view) |
| `rb_sampler` | inside `resource_build`: the static samplers of the translated modules (`create_static_samplers`) and each sampled declaration's own sampler, one bar per sampler |
| `rb_descriptor` | inside `resource_build`: one pass's immutable descriptor set — the pool, the set layouts, the allocation and the writes (`create_descriptors`, one region per submission that builds sets) |
| `rb_indirect` | inside `resource_build`: the indirect replay's command buffer and the memory bound to it (`create_indirect_dispatch`; a direct dispatch builds none) |
| `submit_teardown` | the compute half's own teardown (`ExecutionResources::drop`) — the fence, the pools, the pipeline objects, the buffers, the images, the samplers and the borrowed retains, destroyed once the fence proved the device done with them. The counterpart of the render half's `render_teardown`, and charged only while the submission's own `total` bar is open |
| `submit_td_sync` / `submit_td_pipeline` / `submit_td_buffers` / `submit_td_textures` / `submit_td_retains` | inside `submit_teardown`: the computation fence, the command pool and the descriptor pool; the pipeline-shaped objects; every buffer with its memory; the sampled and storage declarations' samplers, views, images and memories; and retiring the borrowed leases the submission's gathers took — in the order the drop works through them. The pipeline group's per-group region is either the destroy the fresh path always stated or the hand-back of a shape-decided group (`compute_pipeline_return_n`, `compute_pipeline_drop_n`, `docs/COMPUTE-PIPELINE-REUSE.md`), and the buffer group's per-buffer region the same choice for a pooled pair (`compute_buffer_return_n`, `compute_buffer_drop_n`), so a window's entry in either region names which of the two it was |
| `submit_lock` | the submission's executor lock, queue pick, queue lock and arena admission, before the compute half's first bar |
| `submit_bookkeep` | between `pool` and `resource_build`: the translated-artifact list and `plan_pipeline_sequence` |
| `submit_merge` | between the halves: the keyed merge of the compute and render writebacks |
| `submit_validate` | the terminal `ProviderSubmission::validate_for_trace` of the merged writeback list |

The ten `setup_*` fields are the one nested split in the line: they divide
`render_setup` itself, so `sum(setup_*) <= render_setup_us` and the difference is
the seam between those regions — the plan of the whole setup, which stays
charged to the bar that encloses them. A round that reads them can say *which*
part of a pass's assembly a change moved, which is what the render-setup reuse
increment (`docs/RENDER-SETUP-REUSE.md`) needed and what the bar alone could not
answer.

Two further nested splits divide what the first ones left unnamed, and neither
is part of the disjoint sum either:

* the six `texture_*` fields divide `setup_textures` itself:
  `backing + upload + view + sampler + import + descriptor <= setup_textures`,
  and the printed `texture_named_us` is their sum. A round that reads them can
  tell a texture path that builds device objects from one that imports an
  owner's window or writes descriptors — the reading that selected the second
  cut (`docs/RENDER-IMPORT-POOL.md`), where `texture_import_us` was 1 117.7 of
  `setup_textures`'s 1 141.1 µs/submit and the other five came to 5.6 µs.
* the twelve `render_*` residual fields divide what `render_total` cost minus
  its five children — the outer loop around each pass, a landing-only entry, the
  present rail, the offscreen rail entry's own admissions, the retains, the
  owner-window landing, the pass teardown, the three pooled hand-backs and the
  retains' release:
  `sum(render children) + sum(render residual) <= render_total`, with the
  printed `render_residual_us` as the residual's own sum. The seam that remains
  is the function-call boundary between them, and the sp4 round read it as
  32.0 µs/submit out of a 1 769.0 µs/submit residual.
* the ten `teardown_*` fields divide `render_teardown` itself — the pass's two
  synchronisation objects, the pipeline-shaped objects the shape cache did not
  take, the render pass and framebuffers, the sampled declarations' own objects,
  the descriptor state, the depth/stencil surfaces, the colour attachments, the
  readback destinations, the remaining buffers and the previous-byte staging
  buffers, in the order `OffscreenObjects::drop` works through them — with
  `teardown_named_us` as their printed sum, so
  `sum(teardown children) <= render_teardown_us`. They are the fourth cut's
  subject: `render_teardown` was the largest unnamed bar the third cut left
  (494.2 of 2 697.3 µs/submit), and the five `td_*_n` counters beside them are
  its population — how many images, image views, samplers, buffers and memories
  the window's teardowns really destroyed, so a bar's microseconds can be
  divided by what they were spent on and a region the pools already emptied
  reads as a zero count rather than as a free one.
* the readback's four arms divide `render_readback` itself: the trimmed arm
  (`readback_rect`, with `readback_seed` nested inside it — the rebuild only that
  arm pays), the whole-extent arm (`readback_full`) and the depth, stencil and
  writable stage-buffer copy-outs (`readback_surfaces`). `readback_shape` is the
  decision that picks the arm, and it is nested inside `setup_readbacks` rather
  than inside the readback bar because it is taken before any device object
  exists; `readback_named_us` is the printed sum of the three arms. The sp7
  round read 1 816.6 of `render_readback`'s 1 845.1 µs/submit in the whole-extent
  arm, at 169 MB/s, and 27.7 in the trimmed one.
* the eight `landing_*` fields divide one landing-only entry (`render_landing`):
  its identity lookup, its window resolution, its staging objects, the copy's
  recording and submission, its fence wait, the host fetch of the copied frame,
  the write into the owner's pages and the release of those objects, with
  `landing_named_us` as their printed sum and `landing_n` / `landing_bytes` as
  what they were spent on. The sp7 round read 49 206.8 of a landing's
  51 939.2 µs in `landing_fetch` — 168 MB/s, the same rate the whole-extent
  readback arm paid, which is what named the memory both mappings point at as
  the third cut's subject (`docs/READBACK-MEMORY.md`).

The fifth cut's two splits and its seam are read like the ones above them. The
seven `rb_*` fields divide `resource_build` by the object families the create
sites name, with `rb_named_us` printed as their sum, so `sum(rb_*) <=
resource_build_us` — the difference is the argument validation, the shared and
heap sizing passes and the pool-key registrations the families do not own. The
population behind each family is the `rb_*_n` counter printed beside the family
in the same line (`rb_memory_n` counts every `vkAllocateMemory`, whichever
family's backing it belongs to), and the two together are the reading a pooling
decision is made from:

```text
per object = rb_<family>_us / rb_<family>_n
```

A family bar is a **region**, not an object: the pipeline, buffer and descriptor
bars cover their whole create call, so their call count is a submission count
and only the `rb_*_n` counter says how many objects the region made. Those three
are also the families where "the same shape again" is the poolable thing, which
is why the count is printed beside the microseconds rather than left to be
divided by `n`.

`submit_teardown` is the compute half's counterpart of `render_teardown`, and it
is a disjoint bar of the submission rather than a child of `resource_build`
because it runs at the end of the call, after the readback. Before this cut it
ran inside the enclosing `total` and inside no other field, so it was invisible
to every reading above. Its five children divide it in the order
`ExecutionResources::drop` walks, with `submit_td_named_us` as their sum and the
same `_n` counts beside the three object groups. Those counts are populations
with one caveat each: the pipeline and texture groups are entered once per
destroyed object, the sync and retain groups once per submission, and the buffer
group once per buffer **plus** once for the heap slab's own memory and once for
the indirect replay's pair — so `submit_td_buffers_n` is an upper bound on the
destroyed buffers by at most two entries per submission. A round that wants one
buffer's cost reads it from the build side (`rb_buffer_us / rb_buffer_n`) or
subtracts the per-submission entries: the g3a round read 85 825 buffer-group
entries against 43 329 buffers built and 42 496 submissions, i.e. one buffer
created and destroyed per submission. With the compute pool on
(`docs/COMPUTE-BUFFER-POOL.md`) the same entries are no longer all destroys — a
submission whose fence was observed hands its pair back (and unmaps it) inside
the very same region — so the region's microseconds fall while its `_n`
population does not, which is what the increment's own A/B reads.
One boundary is worth stating:
a deferred object API retires its resources from `wait`, outside any submission,
and those microseconds belong to no submission's window — the teardown bars
therefore resolve to nothing when no `total` bar is open, so a window's fields
stay a partition of its own `total` in both the synchronous and the deferred
arm. The reclamation of the rail's own arm happens inside `submit`, and that is
where these bars read.

`submit_lock`, `submit_bookkeep`, `submit_merge` and `submit_validate` name the
rest of the seam the disjoint bars leave, and `submit_seam_us` is the sum of
those four with `submit_teardown` and `submit_release`. `submit_bookkeep` is deliberately *not* part
of `plan`: it is the part of a submission that happens after the pool is
resolved and before the first device object exists — the pipeline plan and the
translated-artifact table — which is exactly the region a cached plan would
remove from a submission while a pool would not.

The sixth cut adds the seam's last unnamed region and splits the largest bar
that was already named:

* `submit_release` is the **tail of the call**: `total` is the first binding in
  `submit`, so it is the last to drop, and the values declared after it — the
  pooled bindings with their own copies of the views' bytes, the serial
  resource pool, the texture views, the per-pass dispatch list, the heap plan,
  the render plan and the pipeline artifacts — drop *after* the `settle` guard
  that was declared last. Those microseconds are inside `total` and inside no
  other field, which is why the fifth cut's seam read as `total_us` minus the
  disjoint bars and the sp13 round could only call the remainder unnamed: the
  sp16 round read that remainder at 170.6 µs/submission before this cut and
  2.7 µs after it. The three children divide it —
  `submit_release_bindings_us` (the pooled bindings),
  `submit_release_views_us` (the resource pool and the texture views) and
  `submit_release_plan_us` (the dispatch list, the heap plan, the render plan
  and the artifacts) — with `submit_release_named_us` as their printed sum. The
  bar is entered only while the profile is on, and the branch that drops the
  tail values with it is skipped when the profile is off, so those values drop
  exactly where they dropped before the cut.
* `submit_validate_derive_us` and `submit_validate_check_us` divide
  `submit_validate`: the two pool derivations the terminal validation takes for
  itself (`ComputeTrace::serial_resources`, which re-validates the trace and
  walks every compute pass's declaration list, and
  `serial_texture_resources`) and the walk that reads them, with
  `submit_validate_named_us` as their sum. The sp16 round read the derivation at
  185.4 of the bar's 229.1 µs/submission — but that bar's mean is a host-stall
  tail rather than per-submission work (`docs/COMPUTE-PIPELINE-REUSE.md` §6
  measured the derivations a cut would remove at ≈4.9 µs, 0.2 % of a
  submission), which is why this cut reads the split and does not cut it.

The seventh cut names the same batch of bytes three more times, and moves two
of the three:

* `plan_resources_us` is the one call inside `plan` that derives the
  submission's serial resource pool: `ComputeTrace::serial_resources`, which
  owns the table and clones every view with its declared bytes, or — with the
  seventh cut's mechanism on (`METAL_API_VULKAN_SUBMIT_RESOURCE_BORROW`, off by
  default) — `ComputeTrace::serial_resources_ref`, which returns the trace's own
  declarations lent. `plan` stays the enclosing bar, so
  `plan_resources_us <= plan_us`.
* `submit_validate_release_us` is the third child of `submit_validate`: the
  release of the two tables the validation derived for itself, which before the
  cut was the unnamed remainder of that bar. The sp16 round read that remainder
  at 43.0 µs/submission while its siblings summed to 186.1 of the bar's 229.1 —
  the free of a second copy of the submission's declared bytes.
* `submit_release_pool_us` is the second child of `submit_release_views`: the
  pool's *own* table (what `plan` derived), dropped on its own rather than with
  the texture views beside it.

`submit_resource_copies_n` / `_bytes` and `submit_resource_borrows_n` / `_bytes`
count the declared bytes a **pool derivation** moved — the views whose source is
the trace's own snapshot, and their total length. One submission derives its
pool twice (once in `plan`, once in `submit_validate`), so the pre-cut path
reports two copies of a submission's declared bytes and the cut's arm reports
the same bytes lent twice and no copies. The other two sources (`StagedLease`,
`GuestRuns`) carry no bytes of their own in the pool and are counted in neither.
The pair is the mechanism's own reading, and
`docs/SUBMIT-RESOURCE-BORROW.md` carries the A/B that prices it.

`submit_binding_copies_n` / `submit_binding_copies_bytes` and
`submit_binding_borrows_n` / `submit_binding_borrows_bytes` say how the window's
submissions filled their pooled bindings with the *trace's own* snapshot bytes:
as a copy the submission made for itself, or — with the sixth cut's mechanism on
(`METAL_API_VULKAN_SUBMIT_BINDING_BORROW`, off by default) — as a borrow of the
serial resource pool that already holds them. The other two binding sources are
always owned and are counted in neither: a staged lease's bytes are copied out
of the staging registry's lock and a gathered run list is built by the gather
itself. The pair is the mechanism's own reading, and
`docs/SUBMIT-BINDING-BORROW.md` carries the A/B that prices it.

The eighth cut names the fourth source of a binding's bytes, the one the three
cuts before it left unnamed:

* `staging_window_us` is the region a **staged lease**'s window is resolved in —
  `LeaseRegistry::view_bytes` / `::texture_bytes` on the compute half's `pool` and
  on the render half's input resolution (the render texture walk takes
  `::texture_bytes`). It is *nested*: the bar it divides stays its parent, so it
  is never added to the disjoint sum. Off the cut's mechanism the region is the
  registry's one `Vec` clone per resolved window; on it the same region hands the
  binding a handle on the registry's own bytes and clones nothing
  (`crate::staging_borrow`).
* `staging_window_copies_n` / `staging_window_copies_bytes` count the windows the
  registry copied for the window's submissions, and their total length: one entry
  per resolved staged view or texture, and the bytes that entry cloned. With the
  eighth cut's mechanism on (`METAL_API_VULKAN_STAGING_BORROW`, **off** by
  default) this pair falls to zero and
  `staging_window_shares_n` / `_bytes` count the same windows and the same bytes
  lent by handle instead — the reading the cut is ranked by, and the third copy
  of the owner bytes the sixth and seventh cuts took off the submission.

`staging_cached_n` and `staging_plain_n` count the readback staging buffers a
window's submissions allocated, by which memory type the selection took
(`crate::readback_memory`): the first for a buffer backed by the device's
host-cached type, the second for one backed by the first host-visible type the
device states — a device with no cached type, or the mechanism's own control
arm. The process also prints the type it chose once, as
`STAGING readback memory type_index=... flags=... cached=...`, while the profile
is on.

The `reuse_*` fields are counts, not times, and they partition every offscreen
pass that reached the shape cache: `reuse_hit_n` passes were served the shader
modules, pipeline layout and pipeline a pass of the same shape built before,
`reuse_miss_n` built their own and cached them, `reuse_mismatch_n` were refused
by the full comparison after a digest collision, `reuse_unkeyed_n` could not
state an exact key at all (and so were never looked up or cached), and
`reuse_disabled_n` ran with `METAL_API_VULKAN_RENDER_SETUP_CACHE=0`. A reading
with `reuse_hit_n=0` is only meaningful beside the other four.

The `pool_*` fields are the same kind of reading for the pooled sampled-texture
backing (`docs/TEXTURE-BACKING-POOL.md`) and the `import_*` fields for the
pooled owner-window import (`docs/RENDER-IMPORT-POOL.md`): `hit`/`miss`/
`disabled` partition a declaration's `take`, `return`/`drop` partition a
completed pass's hand-back, and the five can be added to the same `n=`.
The `buffer_*` fields are that reading for the render half's own host-visible
upload buffers (`docs/RENDER-BUFFER-POOL.md`) and the `compute_buffer_*` fields
are the compute half's counterpart (`docs/COMPUTE-BUFFER-POOL.md`): a separate
group because the two rails carry their own switch and a round has to be able to
read one half's reuse without the other's numbers standing in for it. On the
compute side one `take` is one device buffer the submission builds or takes —
the submission's own staged bytes, its shared backing, and the indirect
replay's twelve-byte command — and one `return` is one pair handed back after
the submission's fence was observed; the five partition the same population
`rb_buffer_n` counts on the build side.
The `compute_pipeline_*` fields are the same reading one level up
(`docs/COMPUTE-PIPELINE-REUSE.md`): the compute half's *shape-decided* objects —
the shader module, the descriptor-set layout, the pipeline layout and the
compute pipelines — which a submission of the same shape takes instead of
rebuilding. `hit`/`miss`/`mismatch`/`disabled` partition every creation,
`return`/`drop` every hand-back of a submission whose fence was observed, and
the six are a separate group from the render half's `reuse_*_n` because the two
rails carry their own switch. A served creation is not also a build, so
`rb_pipeline_n` — which counts the groups a window really built — falls with
`compute_pipeline_hit_n` rising; the two are two readings of one population and
must be read together.
`render_offscreen_n` and `render_present_n` count the render passes a window's
submissions executed, by shape: the five `render_*` children divide the
offscreen executor, so a reading of them beside a present count would otherwise
hide that the two shapes are different populations (the sp4 round read
`render_offscreen_n=1.000` and `render_present_n=0.000` per submission).

`plan_settle` is printed as one field because it is the answer to a question
about the *rail* rather than about the device: of the CPU time a submission
costs, how much is deciding and bookkeeping (cacheable, incremental) as opposed
to touching the device (not). It is the sum of three disjoint fields above, not
a fourth region, so it must not be added to them. `render` is the same kind of
reading for the other half; `render_total` is its enclosing bar, and the
difference between the two is what a render path this split does not name cost.

## Waiting that is not waiting

`fence_wait_us` alone cannot answer "how much of this is the device". Three
populations are therefore counted apart — and `render_wait` carries the same
three fields of its own (`render_wait_idle_n`, `render_wait_blocked_n`,
`render_wait_timeout_n`), because a render pass's copy-out runs inside its wait:

| field | meaning |
|---|---|
| `fence_wait_idle_n` / `_us` | waits the driver answered inside `METAL_API_VULKAN_PHASE_IDLE_NS` (default 100 000 ns): the queue had already retired the work, so the microseconds are driver-call overhead |
| `fence_wait_blocked_n` / `_us` | waits that actually blocked on the device |
| `fence_wait_skipped_n` | waits with no fence behind them (`PendingExecution::wait`'s `!submitted` early return): exactly free |
| `fence_wait_timeout_n` / `render_wait_timeout_n` | waits the driver answered `TIMEOUT`/`NOT_READY`, i.e. the caller may retry |

The two time buckets partition their own wait field by construction
(`idle_us + blocked_us == fence_wait_us`, and the same for `render_wait`), and
the idle/blocked cut is a displayed number rather than a hidden one: move it with
`METAL_API_VULKAN_PHASE_IDLE_NS` if a round's own distribution disagrees with it.
The cut is worth reading as a measured distribution rather than as a law: in the
first real round the `immediate` population still averaged ~85 µs per call
against ~406 µs for the blocked ones — a driver call is not free on this box,
which is why the default sits at 100 µs rather than at 1 µs, and why the raw
bucket sums are printed beside the counts.

The same cut says *what* a wait was waiting on, which the idle/blocked split
cannot: a wait that blocks because the queue is deep and one that blocks because
its own work is long are the same number there. Four kinds partition every fence
the provider waits on, and two more are printed to make "none of them" a reading
rather than a claim:

| field | meaning |
|---|---|
| `wait_submit_n` | waits on the compute submission's own completion fence (inside `fence_wait`) |
| `wait_render_n` | waits on a render pass's completion fence (inside `render_wait`) |
| `wait_landing_n` | waits on a kept-frame landing's fence (inside `landing_wait`) |
| `wait_present_n` | waits on the present rail's sentinel fence |
| `wait_queue_n` | waits on a queue rather than on one submission's completion — **always zero here**, and printed so a round can see that |
| `wait_timeline_n` | waits on a `VkSemaphore` timeline — **always zero here**, for the same reason |
| `wait_ahead_sum` / `wait_ahead_n` | the sum over the window's waits of how many *other* submissions the waited queue still held when the wait began, and how many of those waits began with at least one. Divide by the waits for the mean depth: `0` is a wait for one's own work on an idle queue, `n > 0` is a wait behind earlier submissions |

This provider's only wait is `vkWaitForFences` on a binary completion fence, and
the depth is read from the queue's own in-flight counter at the moment the wait
starts — it is a snapshot of what stood ahead, not a difference of two totals.

The accumulator is thread-local because a line has to describe one population.
A process-wide table would put two submitting threads' bars in the same window,
and then a line's `n`, its fields and its `total` would disagree — the one thing
the identity above is supposed to catch.

## One line per submission

`PHASE submit` is a mean over `METAL_API_VULKAN_PHASE_PROFILE_EVERY` submissions,
and two rounds were read wrong from exactly that shape: most of a two-digit
millisecond mean came from what the boot's first seconds did *once*, and a
distribution with two modes (a one-off translation beside a steady state) has no
mean that describes either mode. `METAL_API_VULKAN_SUBMIT_SAMPLES` adds the
missing instrument — the same truth table as the profile switch, and the two are
read apart:

| `METAL_API_VULKAN_PHASE_PROFILE` | `METAL_API_VULKAN_SUBMIT_SAMPLES` | what is printed |
|---|---|---|
| off | off | nothing (the default) |
| on | off | the window line, as above |
| off | on | one `SUBMIT_SAMPLE` line per submission |
| on | on | both |

```text
SUBMIT_SAMPLE lane=0 n=1 t_ms=0.000 total_us=... admit_us=... plan_us=...
pool_us=... resource_build_us=... record_us=... queue_submit_us=...
fence_wait_us=... read_updates_us=... writebacks_us=... settle_us=...
render_total_us=... submit_release_us=... submit_seam_us=...
submit_validate_us=... landing_us=... landing_n=... passes=...
```

Every field is one of the regions or aggregates the window line already prints,
so the two read the same way — with three differences of *shape* rather than of
meaning:

* the value of each field is **this submission's own** time, taken as the
  running table minus the snapshot the previous line left for that thread, so a
  reader can take the distribution instead of a mean. No second clock is read;
* `n` counts submissions **on that thread** and `t_ms` is a monotonic millisecond
  offset from that thread's first sample, so the lanes of a multi-threaded round
  can be put back in order without trusting the interleaving of the lines as
  they were written;
* `passes` is the number of executed render passes the submission carried, out
  of the same `render_offscreen_n` / `render_present_n` population the window
  line prints — the denominator a per-pass cost needs.

`submit_release_us`, `submit_seam_us`, `submit_validate_us` and `landing_us` are
aggregates of nested regions, exactly as on the window line: they are printed
beside the single slots rather than added to them, so a reader who wants to
decompose a submission uses the single slots and treats these four as "what this
family cost".

A window still closes when `every()` submissions have filled it — a window is a
window — and it empties the table the samples are taken against, so the
snapshots move with that drain. Without that, the one submission after every
drain would report the whole window's remainder as its own time; that is the
pose this switch is read in, with both switches on.

### What the submission was made of

A distribution of microseconds says which bar the slow submissions spent their
time in. It does not say what those submissions **were**, and the reading this
line exists for is the one where the two are needed together: a tail whose top
5% carries 40.6% of the total (`fs1`) is only actionable once its own shape is
known — a bigger declaration, a wider readback, a first sight of a pipeline,
more objects torn down. Two tables are therefore appended to every sample line,
both read through the same snapshot subtraction the bars use:

* **`CHILD_BARS`** — the 21 regions the named parents are made of, each charged
  inside a parent the window line already prints, each carrying the window
  line's own name for that phase (`<phase>_us`), so the two lines read side by
  side without a translation table:

  ```text
  plan_resources_us admit_epoch_us admit_capabilities_us
  submit_validate_derive_us submit_validate_check_us submit_validate_release_us
  render_setup_us render_readback_us readback_rect_us readback_full_us
  render_resolve_us render_teardown_us teardown_buffers_us
  teardown_readbacks_us teardown_attachments_us teardown_textures_us
  submit_teardown_us
  td_image_us td_view_us td_buffer_us td_memory_us
  ```

  They are *nested* inside their parents exactly as their parents are nested
  inside `total`: a reader compares them with the parent, not with each other.
  They keep their **own** snapshot table, because a child can also be a member
  of a parent's aggregate (`submit_validate_derive_us` is in
  `SUBMIT_VALIDATE_SLOTS`, `submit_teardown_us` is in `SUBMIT_SEAM_SLOTS`) and
  one table is spent by whoever reads it first.

* **`SHAPE_SOURCES`** — 50 counters the window line already prints, at one
  submission's resolution:

  ```text
  offscreen_n present_n batch_n batch_passes
  reuse_hit_n reuse_miss_n reuse_unkeyed_n
  pool_hit_n pool_miss_n buf_hit_n buf_miss_n
  cp_hit_n cp_miss_n cb_hit_n cb_miss_n
  views_n views_bytes vcopies_n vcopies_bytes
  wshare_n wshare_bytes
  rb_rect_n rb_rect_bytes rb_full_n rb_full_bytes rb_shape_n rb_bounds_n rb_whole_n
  landing_bytes
  td_buffer_n td_view_n td_image_n td_memory_n
  staging_cached_n staging_plain_n
  buf_miss_vertex_n buf_miss_input_index_n buf_miss_stage_n
  buf_miss_previous_n buf_miss_volume_n buf_miss_index_n buf_miss_indirect_n
  td_unreturned_n buf_miss_empty_n buf_miss_occupied_n
  pool_miss_empty_n pool_miss_occupied_n
  pool_miss_same_extent_n pool_miss_same_format_n
  ```

  The last two rows are the teardown round's own ("why is this object not in
  the pool"): the per-kind misses name *which declaration* missed, `td_unreturned_n`
  counts the objects that were **destroyed while still carrying their pool
  key** (a pair that could have been handed back and was not), and the two
  counts beside it separate "the pool held nothing" from "it held shapes this
  one differs from" (`docs/DRAW-OBJECT-RELEASE.md` reads them for the cut the
  round pointed at).

* **`BUF_MISS_KEY` / `POOL_MISS_KEY`** — the miss *shape*, capped at
  `MISS_KEY_LINES` (16) lines each per window, printed only while
  `METAL_API_VULKAN_PHASE_PROFILE` is on. They carry the value no counter can:
  a missed upload's creation-site name, byte length and usage flags, and a
  missed backing's image type, format, extent, view type and carrier arm, each
  beside the census of what the pool held instead (how many entries, how many
  agreed on each axis, how many were evicted since the previous ask).

  The families answer the shape questions in the order they were asked: which
  passes the submission ran (`offscreen_n` / `present_n`, the batch bands),
  which pipeline family it decided (`reuse_*`, `cp_*`) and which pools it took
  from (`pool_*`, `buf_*`, `cb_*`), how big its declaration was (`views_n` /
  `views_bytes` — the bytes its two pool derivations moved, so one submission
  reads two derivations' worth), how wide its readback was and by which decision
  (`rb_rect_*` / `rb_full_*` / `rb_shape_n` / `rb_bounds_n` / `rb_whole_n`), and
  how many device objects its teardown destroyed (`td_*_n`).

  These are counters, not bars: they say how many, not how long. A round divides
  a bar's microseconds by the counter behind it (a teardown's microseconds by
  its own family's count, a readback's by its own regions) to get the cost of
  one object, and reads the *distribution* of the counters across the tail to
  say what the tail was made of.

Neither table adds a charging site: they are the counters and slots that already
exist, read once more per submission while the switch is on. With the switch off
nothing reads them at all.

One boundary worth naming: the census's `mid` (the rail's mapping id) is not on
this line, because the provider's own namespaces are `(allocation_id, view_id)`
and `gva`, not the rail's mapping ids. "Which surface was this" is answered here
by the pass and readback counters, and on the rail's side by its own
`linux_render_provider … mid=…` lines.

## Aligning the line with the rail's own bars

The reims rail's `frame_span` line and this line measure the same call from two
sides, and they line up field for field:

* `frame_span`'s `prov_submit_us_mean` (per frame) × `frames` is the wall time
  the rail spent inside `provider.submit`; this line's `total_us` summed over a
  round is the same wall time, counted by the provider. Divide either by the
  number of submissions (`n=` here, `provider_draws` in the census there) and
  the two readings should agree to within the rail's own bracket.
* `frame_span`'s `prov_plan_us` / `prov_gate_us` / `prov_trace_us` are the
  *rail's* CPU work before the call; this line's `plan` and `pool` are the
  provider's own. They are adjacent, not the same time: `plan_settle_us` here
  does not include anything the rail charged to `prov_*`.
* `frame_span`'s `prov_settle_us` is the retirement chain and the writeback of
  the record's writable views (the rail's `plan.settle`); this line's `settle`
  is the provider's completion bookkeeping. Same word, different sides of the
  call — a comparison of the two is a reading of the seam, not of one number.
* `provider_held_chain_middle` / `render_provider_resident_store` decide how much
  of the frame never leaves the provider's images; when those are non-zero, the
  readback this line reports is the *unrelayed* remainder, and a drop in
  `read_updates_us` with the same draw count is that relay working.

Two caveats when reading a line: the first window of a round includes any
submissions made while the guest was still settling, and a window whose `n` is
smaller than the configured every-count is the tail of a round rather than a
quiet window.

## What one statement is made of

The phase line above prices the *call*. It cannot say what the owner→provider
statement that call carries is made of, because the statement is encoded on the
rail's side of the wire and its sections are a property of the bytes rather
than of the provider's time. That reading has a line of its own: the rail's
`frame_span`, where the encoder's own offsets are printed beside the frame they
were read from.

The sections are measured inside `metal_api_ipc::statement`, at the same
`CommandCodec::encode_request_payload` that writes them, so a field is a
function of the bytes that were written and cannot disagree with the frame it
is reported beside. **Off by default**: unset, the switch
`METAL_API_IPC_STATEMENT_ACCOUNTING` costs one relaxed load per encoded payload
and changes no byte of any frame (two unit tests pin that — the payload is
identical with the account on and off, and a priced frame decodes to the request
it was written from). The owner rail arms it when its own frame profile is on
(`REIMS_VGPU_FRAME_PROFILE`), so a round sets the switch it already sets.

Six positional fields tile the payload, in the order the encoder writes it:

| field | section |
|---|---|
| `stmt_tag_bytes` | the request tag that selects the submission's shape |
| `stmt_trace_bytes` | the trace header: schema (2), device epoch (8), operation id (8), pipeline count (8) |
| `stmt_pipeline_bytes` | the pipeline table behind that count |
| `stmt_pass_bytes` | the dispatch type, the pass count and every pass |
| `stmt_resources_bytes` | the allocation and lease-reservation table |
| `stmt_tail_bytes` | the completion policy and any heap/ICB tail |

so that a window's line closes on itself:

```text
stmt_total_bytes == stmt_tag_bytes + stmt_trace_bytes + stmt_pipeline_bytes
                  + stmt_pass_bytes + stmt_resources_bytes + stmt_tail_bytes
wire_bytes       == stmt_total_bytes + 9 * stmt_total_n
```

`stmt_total_n` is the count the other statement fields are divided by — the
statements the window's frames were built from — and the `wire_bytes` and
`wire_frames` fields beside it are the same frames counted whole, nine-byte
frame header included.

The views are **not** a seventh tile, because a view sits inside whichever pass,
stage-buffer block or vertex input states it. They are roll-ups within
`stmt_pass_bytes`, each with its own count:

| field | meaning |
|---|---|
| `stmt_views_n` / `stmt_views_bytes` | every `BufferView`, and the bytes it took from its `view_id` to the end of its source |
| `stmt_view_payload_bytes` | the part of that which was the view's own `OwnedBytes` payload — the batch of bytes the provider then copies, uploads and releases, and the reading the statement-economy change is judged against |
| `stmt_view_declared_bytes` | the sum of those views' declared `length`s: a zero-filled payload is its view's `length` by construction, so this is the extent a zero-fill declaration may stand in for |
| `stmt_textures_n` / `stmt_textures_bytes` / `stmt_texture_payload_bytes` | the same three readings for `TextureView`s |

Like every other field on that line, these are **per-frame means over the
window's `frames`**, differenced at the same present the wire counters are, so a
statement that straddles a report boundary stays whole and the sections and the
frame they came from are never split across two windows.

### The zero-fill arm's own reading

Statement economy W2-A (`openspec/changes/render-statement-economy` §3) adds one
arm to `BufferSource`: `ZeroFill { length }`, a declaration whose bytes are its
own `length` in zeros and which the statement does **not** carry. The render
rail's two producers of zero frames — a storing attachment's `Clear` /
`Resident` arm and every in-flight production's declaration — state it when
`REIMS_VGPU_ZERO_FILL_DECL=on` (default off); the provider materializes the same
zeros at the view's own window, so no device-visible byte moves and the frame the
pass lands is the frame it landed before.

A round reads the cut off the pair of roll-ups above, which is why both are
printed:

| reading | arm off (shipped) | arm on |
|---|---|---|
| `stmt_view_payload_bytes` | the extent's bytes travel | that many fewer |
| `stmt_view_declared_bytes` | the same extents | the same |

beside the E side's `zero_fill_bytes_n`, the admission census' own slot for the
bytes a declaration states rather than carries (`METAL_API_CORE_ADMIT_PROFILE=1`;
it trades places with `owned_bytes_n` in the same window). Those three fields are
the mechanism reading; a frame-interval difference of the same size as two
identical arms' spread is noise, and is read as such.

## Boundaries

The profile reports wall time inside one call. It does not say which draw paid
it (the census's per-second `store_routes` does), it does not cover the rail's
own CPU work outside the call, and it does not cover the presentation path or
the engine rail. A reading from this line is a reading of the canonical
provider's submission cost on the machine that produced it — not a claim about
any other device, and not complete Metal conformance evidence.

`render` is one bucket: a render (or present, or indirect) pass submitted inside
the same call has its own record/submit/wait folded into it. Split it the same
way if a round ever shows it large.
