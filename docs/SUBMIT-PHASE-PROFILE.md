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
PHASE submit n=256 total_us=... admit_us=... plan_us=... pool_us=... resource_build_us=... record_us=... queue_submit_us=... fence_wait_us=... fence_wait_idle_n=... fence_wait_idle_us=... fence_wait_blocked_n=... fence_wait_blocked_us=... read_updates_us=... render_us=... writebacks_us=... settle_us=... fence_wait_skipped_n=... fence_wait_timeout_n=... plan_settle_us=...
```

Every µs field is a **sum over that line's own window**, not a mean, with three
decimals (nanosecond resolution). Adding lines and dividing by the summed `n=`
gives the round's mean without a weighting error, and it makes the line's own
identity checkable:

```text
admit + plan + pool + resource_build + record + queue_submit + fence_wait
      + read_updates + render + writebacks + settle  <=  total
```

The difference is the seam between the bars — plain function calls, `Arc`
clones, the queue lock — and is reported as the residual rather than hidden in
one of the fields. `total` is the only enclosing bar: the others are disjoint
regions, so no field can contain another.

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
| `render` | the render/present half executed inside the same `submit` |
| `writebacks` | writeback mapping and the contract validation of the merged result |
| `settle` | terminal observation, completion-record insert, health synchronisation |
| `plan_settle` | the aggregate `plan + pool + settle` |

`plan_settle` is printed as one field because it is the answer to a question
about the *rail* rather than about the device: of the CPU time a submission
costs, how much is deciding and bookkeeping (cacheable, incremental) as opposed
to touching the device (not). It is the sum of three disjoint fields above, not
a fourth region, so it must not be added to them.

## Waiting that is not waiting

`fence_wait_us` alone cannot answer "how much of this is the device". Three
populations are therefore counted apart:

| field | meaning |
|---|---|
| `fence_wait_idle_n` / `_us` | waits the driver answered inside `METAL_API_VULKAN_PHASE_IDLE_NS` (default 100 000 ns): the queue had already retired the work, so the microseconds are driver-call overhead |
| `fence_wait_blocked_n` / `_us` | waits that actually blocked on the device |
| `fence_wait_skipped_n` | waits with no fence behind them (`PendingExecution::wait`'s `!submitted` early return): exactly free |
| `fence_wait_timeout_n` | waits the driver answered `TIMEOUT`/`NOT_READY`, i.e. the caller may retry |

The two time buckets partition `fence_wait_us` by construction
(`idle_us + blocked_us == fence_wait_us`), and the idle/blocked cut is a
displayed number rather than a hidden one: move it with
`METAL_API_VULKAN_PHASE_IDLE_NS` if a round's own distribution disagrees with it.

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
