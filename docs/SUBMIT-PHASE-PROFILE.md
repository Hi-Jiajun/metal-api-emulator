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
reuse_mismatch_n=... reuse_unkeyed_n=... reuse_disabled_n=...
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

The ten `setup_*` fields are the one nested split in the line: they divide
`render_setup` itself, so `sum(setup_*) <= render_setup_us` and the difference is
the seam between those regions — the plan of the whole setup, which stays
charged to the bar that encloses them. A round that reads them can say *which*
part of a pass's assembly a change moved, which is what the render-setup reuse
increment (`docs/RENDER-SETUP-REUSE.md`) needed and what the bar alone could not
answer.

The `reuse_*` fields are counts, not times, and they partition every offscreen
pass that reached the shape cache: `reuse_hit_n` passes were served the shader
modules, pipeline layout and pipeline a pass of the same shape built before,
`reuse_miss_n` built their own and cached them, `reuse_mismatch_n` were refused
by the full comparison after a digest collision, `reuse_unkeyed_n` could not
state an exact key at all (and so were never looked up or cached), and
`reuse_disabled_n` ran with `METAL_API_VULKAN_RENDER_SETUP_CACHE=0`. A reading
with `reuse_hit_n=0` is only meaningful beside the other four.

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
