# Metal API emulator experiment

Experimental source-level Metal compute objects backed by Vulkan, for fast
host-side iteration without booting a VM. The long-term proposal is to share
one Metal semantic path between native Metal and a Windows Vulkan provider.
This repository is an independent project: the provider and contract design
here live outside both upstream repositories and nothing in this repository
has been merged into them. (The fork's Windows/WHPX host port is a separate
contribution that *was* merged upstream — reims-vgpu PR #57.) The same path
also drives a live macOS guest through a fork of reims-vgpu — see
[Live VM path](#live-vm-path-windows--whpx--reims-vgpu) below.

The working application path is:

```text
Device -> Library/Function -> ComputePipelineState
       -> CommandQueue -> CommandBuffer -> ComputeCommandEncoder
       -> Buffer bindings -> dispatchThreads -> commit/wait -> readback
```

`metal-api-core` owns these Rust objects and their command state machine.
`metal-api-vulkan` translates AIR through the pinned metal2vulkan revision and
executes the buffer-compute subset. An [optional reims integration](integration/reims/README.md)
runs the same fixtures against the reims Vulkan engine in a separate workspace.

## Live VM path (Windows + WHPX + reims-vgpu)

The same canonical path also runs against a live macOS guest. On Windows 11,
QEMU (11.1.x, WHPX acceleration) boots a macOS Ventura image whose GPU work is
served by a fork of [reims-vgpu](https://github.com/steelbrain/reims-vgpu): the
guest's Metal render and compute records are carried into this repository's
provider and executed by the Vulkan engine. A census harness relinks QEMU
against the provider's static library, boots the guest for a fixed dwell and
reads the class answers per drain window, so every claim below is a reading
rather than an impression.

Readings from a 2026-09-19 round (desktop and login scenes):

- About 96% of the guest's draw records are answered by the canonical provider
  (37,550 of 39,087 records in round `v37`); the remainder are refused by name
  and stay on the fork's self-contained engine.
- The largest single host span is `provider.submit`. A per-pass written-rect
  readback cut host time per draw from 7,886 µs to 6,669 µs, and the bytes read
  back from the device by about 26%, in a same-binary A/B run.
- The present-to-present frame profile reports intervals of roughly 1.1 s under
  the census workload, with the host draw span accounting for about 81% of each
  interval; the open work is the landing/publishing path and per-pass setup.

Boundaries: this is an independent experiment. The provider and contract design
here are not part of either upstream repository; the fork is a local adapter on
top of the upstream Windows host port (reims-vgpu PR #57), and the census path
is not an upstream-approved architecture. No guest images or Apple binaries are
distributed. QEMU and reims-vgpu keep their own license terms; see
[NOTICE.md](NOTICE.md).

## Current status

- Working: synchronous buffer compute, exact-thread dispatch (including tail
  regions), textual LLVM IR, raw AIR bitcode and offset-zero bitcode wrappers.
- Validation: API ordering, foreign pipeline rejection, bounded buffer access,
  device limits and CPU-visible readback. Multiple bindings of the same Buffer
  are detected and refused.
- Experimental provider: `VulkanComputeProvider` implements `ComputeProvider`
  for up to eight serial exact-thread passes using an explicit pipeline table
  and up to 64 stable owned views, followed by one host readback.
  Each pass binds its own resource subset; all resources upload before execution.
  It registers pipelines, revalidates each trace against its actual device and
  artifact, and returns checked allocation-relative writebacks.
- Completion: both providers default to synchronous `submit`/`wait`. With
  `with_async_execution(true)`, `submit` returns `Submitted` and
  `wait`/`readback` retrieve the final writebacks: Vulkan records and submits
  the owned-byte request on the calling thread under the shared queue lock and
  waits on the completion fence in `wait`, while native Metal registers an
  `MTLCommandBuffer` completion handler that retains the command resources
  until readback. A configurable observation deadline (20 seconds by default,
  `with_observation_deadline`) publishes `SubmittedUnknown`; the timed-out
  submission goes to the Vulkan retirement thread or the native completion
  handler, so the provider keeps accepting new work while its abandonment
  budget remains intact. A provider-scoped `AbandonmentBudget` bounds how many
  unobservable submissions it tolerates; once exhausted, `health()` reports
  `Exhausted` and new work is refused with `provider_unavailable` and
  `RetryAfterRecreate`. A confirmed device loss reports `DeviceLost`,
  destroys the lost device's handles and refuses new work until the provider
  is recreated; Vulkan derives it from `VK_ERROR_DEVICE_LOST`, while native
  Metal derives it from `MTLCommandBufferError::DeviceRemoved` (code 11).
  `ComputeProvider::cancel` and
  `CommandBuffer::cancel` release a pending observation without claiming device
  retirement.
- Shared provider API: compilation, pipeline metadata and release now use
  `PipelineProvider`. The Rust native Metal backend accepts six exact
  reviewed MSL fixtures and shares the optional deferred completion mode.
  Its v1-v8 execution passed
  [five-path CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/34223294821).
- Shared object API: experimental `metal_api_core::provider_api` records
  pipeline/buffer objects into one complete trace and submits it once. Host
  buffer writes land only after complete output validation. See
  [PROVIDER-OBJECTS.md](docs/PROVIDER-OBJECTS.md) for lifecycle and validation.
- Cross-process completion: `metal_api_core::completion::wire` defines a
  transport-independent `CompletionMessage` stream and a receiver-side
  `CompletionMirror`. Per-stream sequences make duplicate, reordered and
  coalesced notifications safe; the first terminal observation wins,
  `Exhausted` keeps in-flight tokens observable, and `DeviceLost` is terminal
  teardown evidence. `CompletionPublisher` is the sender-side dual: it assigns
  the per-stream sequences, makes terminal replay idempotent and refuses
  contradictory transitions. `LoopbackTransport` is an in-memory test and
  diagnostic transport. The mirror composes with `LeaseLedger` through
  `observe_into`; `metal-api-ipc` carries the stream over `MCW1` frames and
  providers publish through the epoch-checked outbox. Guest memory, display and
  VM behavior remain future work.
- Cloud validation: commit `489b489c11b43afbf821295d9d5fe8d9303e1e79` passed
  the full v1-v7 five-path comparison: Swift native, Vulkan direct, Rust Metal
  provider, Vulkan object API and Rust Metal object API. The archived evidence
  contains 26 cases per path; this verifies the bounded object API on the
  reviewed fixtures, not general Metal conformance.
- Binary AIR encoding: commit `d8e42bf3fe3668c8e516af9fe5544db9316d97ca`
  passed the full v1-v8 five-path comparison in
  [CI run 34223294821](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/34223294821).
  V8 reuses the v7 cases with raw and Apple-wrapped bitcode; native rails keep
  using MSL. The archived evidence contains 29 cases per path and does not
  establish general Metal conformance. The Windows Vulkan provider ran the
  same v1-v8 direct, object and async-object rails on an NVIDIA GeForce
  RTX 5060 with matching host-visible writebacks.
- Ranged aliasing: both providers declare `AliasMode::DistinctViews`, so two
  disjoint views of one allocation are admitted and executed while any overlap
  (including read-read) stays refused. The [v10 suite](conformance/SUITE-V10.md)
  binds two disjoint views of one 16-byte allocation through the reviewed
  `copy_word` program, and the comparator compares that allocation as a single
  extent. Disjoint ranges cannot exchange data through the allocation because
  each view's footprint proof bounds its accesses inside its own half-open
  range.
- Ranged reservations: the object API reserves the byte ranges a command
  actually touches, and a reservation conflicts only when the ranges overlap
  and at least one side writes. Two command buffers that touch disjoint ranges
  of one allocation are therefore both accepted and stay in flight together,
  which `provider-smoke` checks on a real device. The host bytes are guarded
  only while the trace snapshots them: every view copies its bytes into the
  trace, so the guard is released before `submit` and a parked submission no
  longer holds the allocation against disjoint CPU access. Conflicting CPU
  access stays excluded for the whole commit-to-completion window by the
  reservations themselves, and `lock_unreserved` re-checks them under the
  guard.
- One device buffer per allocation is implemented in both providers: the
  Vulkan backing and the native `MTLBuffer` are sized from the allocation's
  views and created once, each view binds its own `(offset, length)` window,
  and the Vulkan provider copies in only the bytes a view can read
  (write-only views copy nothing in). `provider-smoke` asserts the exact
  operation and byte counts, and the comparator enforces the allocation-level
  copy contract on every suite. See `research/docs/15` for the design and
  `conformance/SUITE-V10.md` for the disjoint-view suite.
- Sampled textures execute on both providers: the reviewed
  `read_texture_2d` fixture is a copy of one R32Uint texel, and suite-v11
  runs it through all five rails (Swift native, Vulkan direct, Rust Metal
  provider, Vulkan objects, Rust Metal objects) in CI. The object API
  declares textures with `new_texture_with_bytes` and binds them with
  `set_texture`; see `conformance/SUITE-V10.md`'s successor state in
  `research/docs/18`. suite-v12 reads every cell of that image under two
  dispatch shapes over the same five rails. The multi-invocation failures that
  first looked like a translator defect were a provider bug: the Vulkan upload
  assumed tightly packed rows and ignored `VkSubresourceLayout.rowPitch`, which
  `15a6be3` fixed. The old attribution and its retraction are recorded in
  `research/docs/17-纹理多invocation归因修正记录.md`, which is a correction
  record rather than an upstream issue draft.
- Offscreen rendering runs on all five rails: `suite-v13` declares
  one `rgba8_unorm` 2x2 colour attachment, a two-stage graphics pipeline draws
  the reviewed fullscreen triangle so that every texel reads `40 80 c0 ff`,
  and the comparator observes the image through the same allocation writeback
  channel as every other case. The contract and trace values landed in
  `cd5bced`/`b14f496`, the
  Vulkan executor in `2e64eff`/`0319da1`, the observation in `e581562`, and
  `8430446` added the suite to CI. CI run `34776215859` reports five-rail
  parity for it — Swift native oracle, Vulkan trace, Rust Metal provider,
  Vulkan objects and Rust Metal objects all agree byte for byte; since both
  object rails gained a render command encoder (`ea0eb25` for Vulkan,
  `8793b7a` for native), all five backends execute the pass (each object rail
  in both its direct and async shapes) — and
  the attachment observation travels the same writeback channel as every other
  case. The native flip was earned, not assumed: CI run `34774478149` ran
  `native-oracle --render-selftest` on an Apple Paravirtual device and read
  `40 80 c0 ff` four times back, and the suite path afterwards captured the same
  bytes through the Rust native provider. See `conformance/RENDER-CAPTURE.md`.
- Surfaceless presentation runs on the provider rails: `suite-v14` adds a present
  action to the same 2x2 render case — a provider-owned, sentinel-preset target
  that is acquired once, rendered into, made host-readable and reported as
  `present: {"acquire": 1, "present": 1}` through the same writeback channel as
  every other observation. The contract and trace values landed in
  `6ba8834`/`e4f451f`/`541be0c`, the Vulkan executor in `8fdd61b` (with the
  target-layout serialization and colour-write scope fixes in `c45f001`), the
  comparator rules in `725a557`, the native equivalent and its Apple self-test in
  `5f09fa3`/`7c528f2`, and the v14 suite plus CI wiring in `1f8adc1`/`49e68a2`.
  CI run `34782615760` is green: the Rust native provider executes the presenting
  case on an Apple Paravirtual device, Lavapipe runs it on the Vulkan rail,
  `--present-selftest` passes on the Swift oracle, and compare-captures reports
  five-rail parity for v14; the RTX 5060 run is archived in
  `evidence/windows-rtx5060-v14-1f8adc1-2026-09-14/`. Since `ea0eb25`/`341b334`
  the Vulkan object rail executes the same render and present tail through its
  own render command encoder, and the native object rail follows (`8793b7a`);
  the object and object-async shapes on both backends report
  `present: {"acquire": 1, "present": 1}` (Vulkan CI run `34784255291`), with the RTX
  5060 v13/v14 captures archived in
  `evidence/windows-rtx5060-v13v14-objects-341b334-2026-09-14/`. This is a *readable-target
  equivalent*: it does not create a `VkSurfaceKHR`, a swapchain or a window, and
  it models neither multi-buffering nor vsync (`research/docs/24` §3.6).
- The heap and indirect-command tracks have their first committed suite:
  `suite-v15` carries one heap case (two allocations bound into one
  `VkDeviceMemory` slab at offsets 0 and 256), one indirect-dispatch case and one
  indirect-draw render case, each reporting the provider's own placement/replay
  record next to the same writeback-byte comparison every other case uses. CI run
  `34826165116` is green — Lavapipe runs all three Vulkan shapes, the macOS rails
  validate and capture the suite, compare-captures reports five-rail parity — and
  the RTX 5060 run is archived in
  `evidence/windows-rtx5060-v15-suite-87cd4bc-2026-09-14/`. The fixture's marker
  names only `vulkan`: the native rails do not execute heap/indirect cases yet
  (`research/docs/25` Step 7), and indexed draws and heap aliasing remain outside
  the increment.
- The render track's first generalisation is committed as `suite-v16`:
  `quad_indexed_clear_2x2` draws a caller-held `float32x2` vertex stream through a
  caller-held `uint16` index buffer (`vkCmdBindVertexBuffers` +
  `vkCmdBindIndexBuffer` + `vkCmdDrawIndexed`), with the pipeline's vertex input
  state built from the contract's `VertexLayout::Buffers`. Render inputs carry
  their own bytes (`Vec<BufferView>`), so a trace with no compute pass can still
  declare them. The rail proves the footprint before touching the device (stream
  covers `stride × vertices`, index view covers `count × width`, every index names
  a vertex the stream holds), and `render_e2e`'s collapsed-stream case shows the
  draw reads the caller's bytes rather than `vertex_id`. CI run `34863349995` and
  the RTX 5060 capture
  (`evidence/windows-rtx5060-v16-4a926a0-2026-09-14/`) cover the Vulkan rail; the
  fixture's marker names only `vulkan` until the object binding surface and the
  native `MTLVertexDescriptor` path land (`research/docs/23` §10). MRT,
  `LoadOp::Load`, `StoreOp::DontCare`, more attachment formats, depth/stencil,
  instancing and dynamic state remain outside the increment.
- Guest memory has its owner-side contract: `HostRegion` registers a host
  address range and derives page-aligned borrowed windows,
  `provider-smoke` imports such a window without copying and observes the
  device write in place on both drivers, `DirtySet` accumulates the pages a
  submission wrote, and `GuestWindows` refuses to reclaim a window until its
  lease is retired. See `research/docs/19`.
- Open design work: device-loss reclamation beyond the bounded abandonment
  budget (the `ProviderLifecycle` admission/refusal and lease-retirement
  contract exists, both providers admit and report health through it, and a
  real `VK_ERROR_DEVICE_LOST` from `submit`, `wait` or the render submission now
  routes to the device-loss terminal state instead of a generic execution
  failure — what is still missing is a *reproducible* real loss: the
  deterministic teardown evidence remains the injected test loss), general
  native shader admission and CPU uploads
  during command-buffer execution, guest-memory lifecycle wiring beyond the
  owner-side contract (the owner-side `HostRegion`/`DirtySet`/`GuestWindows`
  pieces exist, and `research/docs/20` designs the reims-side projection and
  registration with no call site in reims yet), and deriving a scheduling tier
  from the guest (the priority policy, the per-queue tiers, and the owner
  marking that carries them to a remote provider over the command channel now
  exist, while nothing maps an `MTLCommandQueue` to a tier yet). Resource
  snapshots do not hold live guest pages.
- Not implemented: render features beyond the one-additional-suite increment
  (vertex buffers and MRT, `LoadOp::Load`, `StoreOp::DontCare`, formats beyond
  `rgba8_unorm` inside a suite), sampler and
  texture generalisation beyond the sampled fixture, presentation beyond the
  surfaceless readable-target equivalent (real surfaces and swapchains,
  multi-buffering, present modes other than FIFO, vsync, suboptimal handling),
  heap and indirect execution on the native rails and through the object API,
  indexed draws and heap aliasing (`research/docs/25` Steps 6-7), general MTLB
  function-name resolution, Windows MSL compilation, arbitrary AIR/MSL
  compilation and reflection, and production reims integration (Gate 2/3). This
  is not a Metal.framework ABI implementation.

A [native Metal capture harness](conformance/README.md) is prepared for two
shared fixtures, with a Vulkan JSON capture runner and comparator. The Swift
runner captured the two v1 cases on the cloud Apple Paravirtual device and
matched Vulkan results in [CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/33973870668).
The [eight-case v2 matrix](conformance/SUITE-V2.md) expands offsets, dispatch
boundaries and sparse read/write bindings. Its Swift/Vulkan native run also
[passed](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/33974824176). The Swift
oracle remains separate from the new native Rust provider. The
[shared-provider increment](docs/SHARED-PROVIDERS.md) adds a third comparison
path that passed v1/v2 in CI. The [serial-pass extension](conformance/SERIAL-PASSES.md)
adds one-upload/multiple-dispatch sequences and passed
[three-way CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/33980765113).
The [buffer-rebinding extension](conformance/BUFFER-REBINDING.md) also passed
[three-way v4 CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/33981697514).
The new [mixed-pipeline extension](conformance/MIXED-PIPELINES.md) supports
switching compute shaders inside a single command buffer and passed
[three-way v5 CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/33982552902).
The new [pipeline-layout suite](conformance/PIPELINE-LAYOUTS.md) changes binding
numbers and access roles between programs and passed
[three-way v6 CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/33983234385).
The [resource-subset extension](conformance/RESOURCE-SUBSETS.md) allows a pass to
bind only the resources it needs, including views first used by later passes.
It passed [three-way v7 CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/34010989175).
The [binary-AIR encoding suite](conformance/suite-v8.json) reuses the v7 cases
with raw and Apple-wrapped LLVM bitcode and passed
[five-path v8 CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/34223294821).
The new [provider object API](docs/PROVIDER-OBJECTS.md) runs these existing suites
through Device/Buffer/CommandBuffer/Encoder objects; its native cloud execution
and five-path comparison passed in run
[34011824447](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/34011824447).
Existing evidence does not establish general Metal conformance.

The goal is the host provider used by reims and source-level test programs;
loading arbitrary macOS Objective-C/Swift binaries on Windows is outside this
project's current scope. See the [collaboration draft](UPSTREAM-DISCUSSION-DRAFT.md).

## Build and test the standalone workspace

Install Rust, a C linker and Git. The current preparation is tested with Rust
1.96.0; the manifests retain the previous 1.87 minimum, which has not yet been
separately verified. A first build downloads Cargo dependencies, including
metal2vulkan at `43c46ac8a24adf1a6e872b8a52c706ec9614fad0`.

```sh
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked --no-deps -- -D warnings
cargo fmt --all -- --check
```

These commands require no sibling worktrees, reims checkout, GPU or VM.

For live smoke, install a Vulkan 1.3 loader/ICD with maintenance4, `llvm-as`,
`llvm-dis` and `spirv-val`. Put the tools on PATH or set `METAL_API_LLVM_AS`,
`METAL2VULKAN_LLVM_DIS` and `METAL2VULKAN_SPIRV_VAL` to their executable paths.

```sh
cargo run --locked -p metal-smoke -- --executor standalone
cargo run --locked -p metal-smoke --bin provider-smoke
```

On Linux, select Lavapipe explicitly when needed by setting `VK_ICD_FILENAMES`
to the installed `lvp_icd*.json` file under `/usr/share/vulkan/icd.d/`.

On Windows, use the Rust GNU target with MSYS2 MinGW-w64 GCC, LLVM and
SPIRV-Tools available on PATH, plus the GPU vendor's Vulkan driver:

```powershell
cargo build --locked --release --target x86_64-pc-windows-gnu -p metal-smoke
.\run-smoke.ps1 -Runner .\target\x86_64-pc-windows-gnu\release\metal-smoke.exe
```

The PowerShell wrapper accepts tool paths from the environment above and also
recognizes the conventional `C:\msys64\mingw64\bin` installation. To run the
optional engine comparison after building it, pass `-ReimsRunner` with the path
to `reims-smoke.exe`. Both executables run in separate processes.
Pass `-ProviderSmoke` to run `provider-smoke.exe` (the cross-platform provider
suite) and `-CaptureMatrix` to write the v1-v10 direct, object and
async-object captures under `target\windows-captures`.

`provider-smoke.exe` also runs on Windows. The Unix-domain-socket IPC and
descriptor cases are skipped there because Windows has no `SCM_RIGHTS`
equivalent, but the two-process command channel runs over TCP with a named
section mapping: the owner creates it with `CreateFileMappingW`, the child
opens it with `OpenFileMappingW` and imports it as a borrowed no-copy lease.
The RTX 5060 run exercised `VK_EXT_external_memory_host` host-pointer import
with the owner mapping observing the GPU write in place for both the
single-process and the named two-process cases.

The optional `reims-smoke.exe` also cross-compiles and runs on the same host.
It reported the reims-vgpu persistent Vulkan engine and passed the copy,
raw/wrapped AIR and indexed-boundary checks on the RTX 5060.

The suite checks:

```text
PASS copy_word output=0x67452301
PASS binary_air_copy_word encoding=raw output=0x67452301
PASS binary_air_copy_word encoding=wrapper output=0x67452301
PASS indexed_boundary_dispatch words=30 regions=4
```

`provider-smoke` runs the same four cases through canonical traces and compares
against the original snapshot executor on the same Vulkan device. It checks
nonzero view offsets, immutable pipeline metadata, owner epochs, completion
tokens and explicit registry release, plus the asynchronous completion path:
an outbox stream, a Unix-socket completion hop and a two-process owner/provider
case where the child owns the Vulkan device and the parent retires a lease from
the mirrored terminal alone. The same two processes also exercise the
owner-to-provider `MCC1` request/response channel: the parent compiles,
submits, waits, reads back and releases on the child, imports a staged lease,
and passes a shared mapping descriptor with `SCM_RIGHTS` so the child imports
the same pages as a `BorrowedNoCopy` lease and writes through them in place. A
second two-process case uses TCP and a named section instead of `SCM_RIGHTS`,
so the same owner/provider command channel runs on Windows. It also imports an
owner-issued staged lease, executes a view from the copied
window and retires the lease through the owner ledger, then imports an aligned
owner mapping without copying
(`VK_EXT_external_memory_host`) and proves live reads and in-place GPU writes
before release. The command connection lowers its frame limit to 1 KiB in this
case, so every request travels as chunk frames and the child reassembles it
before decoding. The object-API queue case commits two command buffers with a
shared buffer in commit order: the second commit blocks on the first command's
reservation, so the destination can only observe the first command's write
when the queue preserves order. A second object-API case commits two command
buffers with disjoint buffers: the second commit must return while the first
command is still pending, so independent submissions stay in flight at once.
A third case records two dispatches in one command buffer with a data
dependency, so the inter-pass compute barrier must make the first pass's write
visible to the second. The executor creates up to four queues in the selected
family plus up to four in a dedicated compute-only family when the device
exposes one, and the async provider picks the least-loaded queue, breaking ties
with a round-robin cursor. Each queue has its own host enqueue lock, so
independent queues can submit concurrently while submissions to one queue stay
serialized; a probe-backed case commits one command buffer per queue and
requires every queue lock to be held at once. Lavapipe reports
`queues=1 families=1` and skips the concurrency case, while the Windows RTX
5060 reports `queues=8 families=2 distinct=8`.
A final case injects a simulated device loss
into a dedicated executor: `health` must report `DeviceLost`, new compilation
must be refused with `device_lost`/`RetryAfterRecreate`, and a freshly created
executor/provider pair must resume work with an exact writeback. The object
API's `Device::health()` must expose the same lost and recovered states. The
injection exercises the lifecycle state machine; it is not a real GPU device
loss. This is a provider/legacy Vulkan comparison, not a native Metal oracle.

The indexed case launches a 10x3 grid with an 8x2 nominal threadgroup, exercises
a barrier and checks all 30 output words. Its source and reference output are
attributed in [NOTICE.md](NOTICE.md). Generated AIR stays temporary.

## Evidence and limits

Earlier local checkpoints ran these four checks on Linux/Lavapipe and
Windows/RTX 5060 with both Vulkan executors. Those runs validate the narrow
snapshot executor path. They do not demonstrate native Metal parity.
The 2026-09-08 Windows RTX 5060 run additionally exercised the cross-platform
provider smoke, including borrowed no-copy host-pointer import, and the v1-v8
capture rails.
Publication-preparation checks are recorded in [docs/VALIDATION.md](docs/VALIDATION.md).
The subsequent provider implementation is described in
[docs/PROVIDER-B1.md](docs/PROVIDER-B1.md).

The standalone fence wait has a 20-second bound. Reims uses its explicit
synchronous retirement entry. Neither interface provides an end-to-end
initialization/compilation/lock deadline. Guest memory, display and VM behavior
remain outside this offline test boundary.

## License

LGPL-3.0-or-later, copyright (C) 2026 Jiajun Liang. See [LICENSE](LICENSE),
[COPYING](COPYING) and [NOTICE.md](NOTICE.md) for the source attribution and
license texts.
