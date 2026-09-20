# The memory a readback staging buffer is backed by

This increment changes one thing: which memory type a host-readback staging
buffer is allocated from. It is the third cut on the submission-cost line, and
it is the one the sp7 round's own reading selected - both of the two bars that
round split were paying the same rate, and the rate turned out to be the memory
rather than the copy.

## 1. The reading that selected it

The phase profile's new bars split `render_readback` and `render_landing` into
their arms and regions. The sp7 round (reims `65db73f` x provider `1555e79`,
300 s, 125 windows, one submitting thread) read:

* `readback_full` (whole-extent stored attachments): 307 328 B/submit in
  1 816.6 us/submit - 169 MB/s.
* `readback_rect` (trimmed attachments): 1 282 B/submit in 27.7 us/submit - the
  seed rebuild, not a read of any size.
* `readback_surfaces` (depth, stencil, writable stage buffers):
  0.1 us/submit.
* `landing_fetch` (one kept frame's host read): 8 294 400 B per landing in
  49 206.8 us/landing - 168 MB/s.
* `landing_wait` (the same landing's device copy): 8 294 400 B per landing in
  1 679.3 us/landing - 4.9 GB/s.

Two regions, one rate, at sizes 27x apart, and no fixed part: the smaller read
is 8.29/0.82 of the larger (4.87 ms per whole-extent attachment, 49.2 ms per
landing), so the cost is per byte and not per call, per object or per copy. The
device's own writes into the same buffers are three orders of magnitude faster
(`landing_wait`), which is what makes this a property of the host's read of the
mapping.

Both mappings came from the same place: `allocate_host_readback` asked for
`HOST_VISIBLE | HOST_COHERENT`, and `VulkanContext::memory_type` answers with
the first type that satisfies the flags. On this device that is the
host-visible window over device-local memory (type index 1, flags `0x6`):
uncached reads across the bus, which is where 168 MB/s comes from. The line the
process prints once per round states exactly this, in both arms - see section 4.

## 2. The change

`crates/metal-api-vulkan/src/readback_memory.rs` chooses the type, preferring
`HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` and falling back to
`HOST_VISIBLE | HOST_COHERENT` - exactly the pair the rail asked for before.

Why cached: every byte of these buffers is read by the host through the mapping
the allocation returns, and none of them is written by the host. The device's
write is the copy that fills it, and PCIe writes are posted whether the
destination is cached or not; `landing_wait` above is the evidence that the
write side is not the cost.

Why still coherent: the host read happens after the fence, and this rail never
invalidates a mapped range. `HOST_COHERENT` is what keeps the read correct
without an invalidate, so it stays in both flag sets; the preference only ever
adds `HOST_CACHED`.

Why the fallback matters: a device that states no cached host-visible type gets
the type it always got, with the same failure sentence when it states no
host-visible type at all. The change is a preference, never a requirement.

Every staging buffer the render rail reads goes through the helper: the colour,
depth and stencil readback destinations, the writable stage buffers' copy-out
destinations and the kept-frame landing's own destination. The other
`HOST_VISIBLE | HOST_COHERENT` buffers in this crate are write-direction (the
host fills them and the device reads them - the indirect commands, the storage
image transfer buffers, the render inputs), where the first host-visible type is
the right answer and the same preference would be a pessimization; they are
deliberately not changed.

## 3. The switch

`METAL_API_VULKAN_CACHED_READBACK=0` (also `off`, `no`, `false`) drops the
preference: the selection is exactly what it was before this module existed.
That is the control arm the A/B states, and it is what makes the round a
comparison of one decision rather than of two builds.

The phase line counts both outcomes per window (`staging_cached_n` /
`staging_plain_n`), so a round that reads a small `staging_cached_n` can tell a
device with no cached type from one whose mechanism is off. The process also
prints the type it chose once on stderr while the profile is on:

    STAGING readback memory type_index=3 flags=0xe cached=1

## 4. What the arms read

Same exe bytes in all three arms (the sha256 is in the round's evidence), one
launcher line apart.

| reading | sp8 default | sp9 control | sp8b default |
|---|---|---|---|
| `total` (us/submit) | 2 697.3 | 5 580.7 | 2 667.7 |
| `render_readback` (us/submit) | 70.5 | 1 848.8 | 67.6 |
| - `readback_full` | 41.6 | 1 821.1 | 41.8 |
| - `readback_rect` (untouched) | 28.3 | 26.9 | 25.1 |
| - `readback_surfaces` (untouched) | 0.10 | 0.10 | 0.10 |
| `render_landing` (us/submit) | 55.9 | 992.7 | 52.6 |
| - `landing_fetch` (us/landing) | 1 000.8 | 49 180.9 | 904.2 |
| - `landing_wait` (us/landing) | 893.0 | 1 672.1 | 756.7 |
| - `landing_stage` / `_write` / `_release` (us/landing) | 192.5 / 290.1 / 198.4 | 183.9 / 295.2 / 272.6 | 198.1 / 295.1 / 194.9 |
| `render_wait` (us/submit) | 271.0 | 406.1 | 260.8 |
| `render_teardown` (us/submit) | 494.2 | 532.9 | 500.4 |
| `staging_cached_n` / `staging_plain_n` (per submit) | 0.501 / 0.000 | 0.000 / 0.500 | 0.506 / 0.000 |
| `STAGING readback memory` line | `type_index=3 flags=0xe cached=1` | `type_index=2 flags=0x6 cached=0` | `type_index=3 flags=0xe cached=1` |
| rail's own `prov_submit_us_mean` (us/draw) | 2 745 | 5 619 | 2 713 |
| rail's own `host per draw` (us) | 4 244 | 6 989 | 4 193 |

Reading the table:

* The control arm reproduces the pre-cut profile: the sp7 round (before this
  cut, different exe, 125 windows) read 5 619.8 us/submit total, 1 845.1 for
  `render_readback` and 988.5 for `render_landing`, all within 0.7% of sp9's
  5 580.7 / 1 848.8 / 992.7. The mechanism's own counter says why: the control's
  staging buffers take type index 2 (flags `0x6` - write-combined host memory),
  which is the type the old flag pair selected.
* The two increment arms agree with each other (total 2 697.3 vs 2 667.7, 1.1%;
  `readback_full` 41.6 vs 41.8; `landing_fetch` 1 000.8 vs 904.2 us/landing).
* The two regions this cut never touches do not move: `readback_rect`
  (28.3 / 26.9 / 25.1 us/submit - it is the seed rebuild, which reads owned
  bytes) and `readback_surfaces` (0.10 / 0.10 / 0.10).
* Two more bars move *with* the mechanism rather than beside it, and in the
  helpful direction: the device's own writes into a staging buffer cost less in
  the cached type (`landing_wait` 893.0 against 1 672.1 us/landing for the same
  8.29 MB copy, and `render_wait` 271.0 against 406.1 us/submit), and so does
  destroying it (`landing_release`, `render_teardown`).
* The work is the same in all three arms: per submission the round read
  `readback_full_n` 0.373 / 0.373 / 0.370, `landing_n` 0.019 / 0.019 / 0.020 and
  0.501 / 0.500 / 0.506 staging allocations, so the same number of buffers was
  selected in every arm.

What the guest saw: the same 300 s produced 50 432 / 32 512 / 51 456
submissions (197 / 127 / 201 windows of the phase line) - that is the point of
the cut, the guest got 55-58% more submissions through the same dwell - the
rail's own `prov_submit_us_mean` fell from 5 619 to 2 745 / 2 713 us/draw
(-51.2% / -51.7%), and the provider's own whole-call reading fell by 51.7% /
52.2%.

## 5. What this does not claim

It does not say a cached mapping is faster for every device: the fallback keeps
the old answer, and the round's own numbers are the claim, on this device.

It does not change any byte the guest sees. The staging buffer's contents are
copied into the owner's pages by the same pointer copy as before, and the device
copy that fills it writes the same texels; the landing and readback e2e suites
are run in both switch positions for exactly that reason.

It is not a readback of fewer bytes: the shapes, the rectangles and the seed
rules are the previous increment's, untouched.
