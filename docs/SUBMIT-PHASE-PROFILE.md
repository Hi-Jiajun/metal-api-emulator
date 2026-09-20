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
import_return_n=... import_drop_n=... render_offscreen_n=... render_present_n=...
readback_rect_us=... readback_full_us=... readback_seed_us=...
readback_surfaces_us=... readback_shape_us=... readback_named_us=...
landing_lookup_us=... landing_windows_us=... landing_stage_us=...
landing_record_us=... landing_wait_us=... landing_fetch_us=...
landing_write_us=... landing_release_us=... landing_named_us=...
landing_n=... landing_bytes=... staging_cached_n=... staging_plain_n=...
```

Every µs field is a **sum over that line's own window**, not a mean, with three
decimals (nanosecond resolution). Adding lines and dividing by the summed `n=`
gives the round's mean without a weighting error, and it makes the line's own
identity checkable:

```text
admit + plan + pool + resource_build + record + queue_submit + fence_wait
      + read_updates + render_setup + render_record + render_submit
      + render_wait + render_readback + writebacks + settle  <=  total
```

The difference is the seam between the bars — plain function calls, `Arc`
clones, the queue lock — and is reported as the residual rather than hidden in
one of the fields. `total` and `render_total` are the enclosing bars; the others
are disjoint regions, so no field to be summed can contain another. The render
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
* the eight `render_*` residual fields divide what `render_total` cost minus its
  five children — the outer loop around each pass, a landing-only entry, the
  present rail, the offscreen rail entry's own admissions, the retains, the
  owner-window landing and the pass teardown:
  `sum(render children) + sum(render residual) <= render_total`, with the
  printed `render_residual_us` as the residual's own sum. The seam that remains
  is the function-call boundary between them, and the sp4 round read it as
  32.0 µs/submit out of a 1 769.0 µs/submit residual.
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

The accumulator is thread-local because a line has to describe one population.
A process-wide table would put two submitting threads' bars in the same window,
and then a line's `n`, its fields and its `total` would disagree — the one thing
the identity above is supposed to catch.

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
