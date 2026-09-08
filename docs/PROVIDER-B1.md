# B1: Vulkan provider

`VulkanComputeProvider` implements the core `ComputeProvider` trait for a
single serial exact-thread pass, owned byte views and complete host readback.
The existing `ComputeExecutor` remains available as an independent application
entry point. Both paths share Vulkan execution machinery and the queue lock.
The provider defaults to synchronous execution and can be switched to deferred
completion with `with_async_execution(true)`. The completion slot is the shared
`metal_api_core::completion::CompletionRecord`, so the native Metal provider
uses the same wait/readback semantics.

## Invocation and ownership

1. Create a Vulkan executor, then `VulkanComputeProvider::with_executor` to use
   its device. A provider receives a unique process-local device epoch.
2. Call `compile_pipeline(&Function, logical_digest)`. The provider retains the
   translated artifact and returns `CompiledComputePipeline` metadata: epoch,
   pipeline ID, function identity and reflection-derived contract. The logical
   digest is caller-issued fixture metadata, not a content-verified digest or
   artifact-cache key. Pipeline IDs always designate separately registered
   artifacts. Compilation is currently a Vulkan-provider method, not yet a
   cross-provider compilation trait.
3. Build a `ComputeTrace` with that metadata and a `ResourceTableSnapshot`.
   `capabilities().validate_trace(...)` produces the immutable input.
4. `submit` rechecks the receiving device's capabilities and registered artifact,
   including the exact function identity and reflected contract. It refuses
   stale epochs, unknown pipelines, forged reflection, unsupported storage and
   narrowing overflow before creating request-specific Vulkan objects.
5. In the default synchronous mode, a successful `submit` returns
   `CompletedVisible` and canonical writebacks sorted by
   `(allocation_id, view_id)`. Offsets are allocation-relative; only writable
   views are returned, each exactly once at its complete declared size.
6. In async mode, `submit` records and submits the owned-byte request on the
   calling thread under the shared queue lock and returns `Submitted` without
   waiting. `wait` waits on the device completion fence with its
   caller-supplied timeout and then performs readback; `readback` returns the
   same canonical writebacks after completion. A configurable observation
   deadline (20 seconds by default, `with_observation_deadline`) reports
   `vulkan-completion-unknown` with `SubmittedUnknown`; the timed-out
   submission goes to the same retirement thread, so the executor keeps
   accepting new work while its abandonment budget remains intact. `cancel` releases the
   observation slot and hands the pending submission to retirement, reporting
   `Cancelled`; `release_completion` does the same for a still-pending
   submission. `release_pipeline` removes a registry entry. In-progress
   submissions keep their own artifact reference. These calls do not release
   abandoned GPU work.
7. Bounded abandonment: a fence that never signals (or a completion handler
   that never fires) is recorded against a provider-scoped
   `AbandonmentBudget`. Vulkan defaults to one abandoned submission and 64 MiB;
   native Metal defaults to eight submissions and 64 MiB. The builder
   `with_abandonment_budget` raises or lowers the bound. When the bound is
   reached, `health()` reports `Exhausted` and new compile/submit calls are
   refused with `provider_unavailable` and `Retryability::RetryAfterRecreate`.
   A confirmed `VK_ERROR_DEVICE_LOST`, or a native
   `MTLCommandBufferError::DeviceRemoved` (code 11) observed on a terminal
   command buffer, takes the `DeviceLost` path instead: `health()` reports
   `DeviceLost`, the lost device's handles are destroyed rather than leaked,
   and new work is refused with the provider's device-loss slug (`device_lost`
   on Vulkan, `metal_device_removed` on native Metal) and
   `Retryability::RetryAfterRecreate`. A device-loss failure does not consume
   the abandonment budget.

The synchronous mode keeps the direct trace rail and existing captures
unchanged. The async mode is used by the object-API capture path with
`provider-capture --api objects --async`; CI runs v1-v8 this way on Lavapipe.
Async mode still serializes submission with the same queue lock, so it overlaps
host work and completion observation rather than concurrent GPU execution.
There is no end-to-end deadline on compilation, locks, initialization or
submit. The synchronous fence wait keeps a fixed 20-second bound; a configured
observation deadline reports unknown completion and retires the submission
when its fence signals. Resources whose fence never signals are retained to
process exit, but the abandonment budget bounds how many such submissions a
provider will tolerate before it fails closed. Live guest leases, multi-pass
ordering, general MTLB resolution and native Metal remain unimplemented. The
core `LeaseLedger` defines completion-driven lease release, and
`metal_api_core::completion::wire` defines the transport-independent
notification stream and its sender-side publisher that will carry those tokens
between processes; provider-side guest/no-copy lease import and the actual IPC
transport remain unimplemented.

## Execution failures and visibility

Failures retain Resolve/Compile/Encode/Submit/Wait/Readback phase information.
Vulkan return codes and the native `NSError` code determine device-loss
classification; diagnostic strings are not parsed. Native Metal maps
`MTLCommandBufferError::DeviceRemoved` (code 11) to `DeviceLost` on both the
synchronous and completion-handler paths; every other terminal command-buffer
error keeps the `metal_command_failed` path and marks the provider
`Exhausted`. The Vulkan and native slugs remain distinct (`device_lost` versus
`metal_device_removed`), so this is not yet a shared error vocabulary.

- Before queue submission: `NotSubmitted`.
- Queue OOM failures: `NotSubmitted`, because Vulkan guarantees referenced
  resource state is unaffected.
- Other queue failures and fence-wait failures: completion is unknown; handles
  remain retained. Device loss marks the shared executor unusable.
- Readback failure after fence completion: `Failed`, with a token, without
  claiming CPU-visible results.

The queue failure treatment follows the
[vkQueueSubmit failure contract](https://docs.vulkan.org/refpages/latest/refpages/source/vkQueueSubmit.html).
The shared execution path now includes a compute-shader-write to host-read
memory barrier before fence completion, following the
[Khronos host readback example](https://docs.vulkan.org/guide/latest/synchronization_examples.html#_cpu_read_back_of_data_written_by_a_compute_shader).
HOST_COHERENT memory is still required. Unknown terminal results are not proof
that backing storage can be reclaimed; this implementation deliberately keeps
its existing abandonment policy.

## Reproduction

```sh
cargo test --workspace --locked
cargo run --locked -p metal-smoke --bin provider-smoke
```

The provider runner shares one Vulkan device with the old snapshot executor,
compiles each path independently and compares both against the fixture golden.
It covers textual/raw/wrapped copies, the indexed 10x3 boundary/barrier case,
nonzero view offsets, forged pipeline metadata, isolation between two providers
on one executor, unknown completion tokens, use after registry release,
zero-deadline reclamation, explicit cancellation with a recorded outbox stream,
single-process completion over a Unix socket and a two-process owner/provider
completion hop. The runner spawns itself with `--completion-child`: the child
owns the Vulkan device and publishes admission and `CompletedVisible` through
the real outbox and writer thread, while the parent owns only the listener,
mirror and lease ledger. This is Vulkan-provider versus Vulkan-executor
regression coverage, not native Metal parity.

## Verification of this local increment

- 2026-09-08 two-process completion owner/provider smoke: `provider-smoke`
  gained a `--completion-child` mode. The parent binds a Unix listener, spawns
  the child, learns the child's device epoch and submission identity from its
  handshake and retires a lease from the mirrored terminal transition; the
  child owns the Vulkan device and publishes through `CompletionOutbox` and
  `metal-api-ipc::sender::spawn_writer`. Lavapipe reports
  `provider_completion_ipc_process owner=parent provider=child transport=unix
  outbox=Submitted,CompletedVisible lease=retired`. 221 Rust tests and 115
  Python tests passed.
- 2026-09-08 provider-side completion publisher: `CompletionPublisher` is the
  sender-side dual of `CompletionMirror`; it assigns monotonic per-token and
  device-health sequences, returns the identical message for an idempotent
  terminal replay, and refuses a conflicting terminal or any non-terminal
  update after device loss. `LoopbackTransport` is an in-memory test and
  diagnostic transport. Nine new core tests cover per-token sequences,
  idempotent terminal replay, monotonic health, device-loss refusal,
  publisher/mirror round trip under reordering, identity validation and a
  cross-thread `mpsc` boundary. 186 Rust tests passed (core 127, native 9,
  Vulkan 36, capture 14) and 115 Python tests passed.
- 2026-09-08 cross-process completion wire contract:
  `metal-api-core::completion::wire` adds `CompletionMessage`,
  `CompletionMirror`, `CompletionSequence` and `CompletionFailure` with 12
  unit tests covering duplicate/reordered delivery, first-terminal-wins,
  lower-sequence terminal convergence, `Exhausted`/`DeviceLost` health
  propagation, identity validation and `LeaseLedger` composition. 177 Rust
  tests passed (core 118, native 9, Vulkan 36, capture 14) and 115 Python tests
  passed.
- 2026-09-08 completion-driven lease ledger: `LeaseLedger` and
  `disposition_retires_resources` were added to `metal-api-core::provider`
  with six unit tests covering multi-token retirement, idempotent binding,
  shared tokens, device loss, unknown/cancelled/timeout retention and
  malformed registrations. 165 Rust tests passed (core 106, native 9,
  Vulkan 36, capture 14) and 115 Python tests passed; Lavapipe v8 direct and
  async-object captures matched. The ledger is not yet wired into provider
  buffer import.
- 2026-09-08 Windows RTX 5060 v8 validation: Windows GNU debug binaries built
  from `7e67a0e` ran `provider-smoke` and the v1-v8 direct, object and
  async-object captures on an NVIDIA GeForce RTX 5060. All 12 smoke checks and
  all 29 cases per rail passed, including the v8 raw/wrapped AIR encoding.
  Binary SHA-256 values are recorded in
  `evidence/windows-rtx5060-v8-7e67a0e-2026-09-08/manifest.md`. This is a
  Vulkan-provider validation on Windows, not native Metal parity.
- 2026-09-08 native device-removal increment: 159 Rust tests passed (core 100,
  native 9, Vulkan 36, capture 14) and 115 Python tests passed. Native Metal
  classifies `MTLCommandBufferError::DeviceRemoved` (code 11) as `DeviceLost`
  on both the synchronous and completion-handler paths and refuses new work
  with `metal_device_removed`/`RetryAfterRecreate`. The object API now covers
  submit-time `DeviceLost` (observed token preserved) and `Exhausted`
  (`NotSubmitted`) failures that leave the command `Failed` with no host
  bytes changed. Linux/Lavapipe ran the v8 direct and async-object captures
  with unchanged host-visible writebacks. Formatting, Clippy with
  `-D warnings`, rustdoc and the macOS cross-target check passed. No real
  device-removal error was injected.
- 2026-09-08 binary-AIR encoding increment: 151 Rust tests passed (core 96,
  native 7, Vulkan 34, capture 14) and 115 Python tests passed. `suite-v8.json`
  reuses the v7 cases with raw and Apple-wrapped bitcode; Lavapipe and the
  Apple Paravirtual device passed the five-path v1-v8 comparison in
  [CI run 34223294821](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/34223294821).
  Formatting, Clippy with `-D warnings` and rustdoc passed.
- 2026-09-08 observation-deadline increment: 147 Rust tests passed (core 92,
  native 7, Vulkan 34, capture 14) and 113 Python tests passed. The shared
  `ObservationDeadline` clamps caller timeouts to the remaining observation
  window and both providers use it with a configurable limit (20 seconds by
  default). Linux/Lavapipe v1-v7 async-object captures passed with unchanged
  host-visible writebacks. Formatting, Clippy with `-D warnings` and rustdoc
  passed.
- 2026-09-08 Vulkan device-fence increment: 145 Rust tests passed (core 90,
  native 7, Vulkan 34, capture 14) and 113 Python tests passed. Async submit
  records and submits on the calling thread, and `wait` observes the device
  completion fence without a per-submission worker. Linux/Lavapipe ran v1-v7
  direct, object and async-object captures (26 cases per rail) with unchanged
  host-visible writebacks. Formatting, Clippy with `-D warnings` and rustdoc
  passed.
- 2026-09-08 native async increment: 145 Rust tests passed (core 90, native 7,
  Vulkan 34, capture 14) and 113 Python tests passed. The shared completion
  record moved to `metal_api_core::completion` with four tests; the native
  provider's deferred mode is covered by the macOS object capture. Linux/Lavapipe ran
  `provider-capture --api objects --async` for v1-v7 (26 cases) with the same
  host-visible writebacks and allocations as the suite; the default synchronous
  direct/object captures were unchanged. Formatting, Clippy with `-D warnings`
  and rustdoc passed.
- 58 unit tests passed: 41 core and 17 Vulkan. New cases include result identity,
  writable-view coverage, premature writeback, range/order validation, bounded
  ID generation, narrowing and execution-failure classification.
- The optional reims integration's 3 adapter tests also passed after the
  shared smoke library change; its 3 pre-existing dead-code warnings remain.
- Formatting, workspace Clippy with `-D warnings` and rustdoc with
  `RUSTDOCFLAGS=-D warnings` passed on Rust 1.96.0.
- Linux/Lavapipe: all four provider comparison cases and rejection/release
  checks passed. The original standalone smoke also passed its four cases.
- Windows GNU debug binaries linked. Running `provider-smoke.exe` directly via
  Windows PowerShell reported NVIDIA GeForce RTX 5060 and passed all four
  comparison cases and rejection/release checks. No VM or deployment-file
  replacement was needed. This does not validate the PowerShell script-loading
  issue recorded for the earlier publication candidate.
- Windows provider executable SHA-256:
  `e8b61a45a64e44850fe048af52420dc2d93fa14feea9d571dd51bf0bf03da661`.
- Device-loss/timeout mappings are unit-tested; no real GPU device loss or
  timeout was injected. No Vulkan validation layer was available locally.
- Initial publication CI at `9d7c007` passed. The workflow now includes
  `provider-smoke`, but this increment has not been pushed or run in remote CI.

No native Metal provider, production reims routing, guest-memory lifecycle or
VM/display result is claimed for this step.
