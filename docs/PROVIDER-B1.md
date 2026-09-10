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
   narrowing overflow before creating request-specific Vulkan objects. A
   non-`None` `BufferView::attribute_stride` is refused during admission with
   the structured `buffer_attribute_stride_unsupported` capability slug; an
   unsupported trace schema version is `trace_schema_unsupported` (capability),
   and pipeline/allocation/lease epoch mismatches share the
   `resource_contract_invalid` resource class. The refusal table matches every
   `ContractError` variant exhaustively, so a new variant cannot silently fall
   back to the generic `trace_contract_invalid` args slug.
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

   `provider-smoke` now covers this lifecycle on a dedicated Vulkan executor
   with a simulated injection: after `VulkanExecutor::inject_device_loss_for_test`,
   `health()` must report `DeviceLost`, the next `compile_pipeline` must be
   refused with `device_lost` and `RetryAfterRecreate`, and a newly created
   executor/provider pair must complete a fresh `copy_word` submission with an
   exact writeback. The object API's `Device::health()` must expose the lost
   and recovered states on the same executors. The hook exists because CI
   cannot produce a deterministic `VK_ERROR_DEVICE_LOST`; the case validates
   the state machine, not a real device loss.

The synchronous mode keeps the direct trace rail and existing captures
unchanged. The async mode is used by the object-API capture path with
`provider-capture --api objects --async`; CI runs v1-v9 this way on Lavapipe.
`provider-smoke` also commits two object-API command buffers with a shared
buffer dependency: the second commit blocks on the first command's reservation
and only then submits, so the final destination proves commit-order execution
on the real device. This checks host-side reservation ordering, not concurrent
GPU execution. A second case commits two command buffers with disjoint buffers
and asserts that the second commit returns while the first command is still
pending, so independent submissions stay in flight at once. A third case
records two dispatches in one command buffer with a data dependency and checks
that the inter-pass compute barrier makes the first pass's write visible to the
second. The executor creates up to four queues in the selected family (clamped
to the family's queue count) plus up to four in a dedicated compute-only family
when the device exposes one, and the async provider picks the least-loaded
queue, breaking ties with a round-robin cursor. Each queue has its own host
enqueue lock, so independent queues can submit concurrently while submissions
to one queue stay serialized; a probe-backed case commits one command buffer per
queue and requires every queue lock to be held at once. Lavapipe reports
`queues=1 families=1` and skips the concurrency case; the Windows RTX 5060
reports `queues=8 families=2 distinct=8`.
There is no end-to-end deadline on compilation, locks, initialization or
submit. The synchronous fence wait keeps a fixed 20-second bound; a configured
observation deadline reports unknown completion and retires the submission
when its fence signals. Resources whose fence never signals are retained to
process exit, but the abandonment budget bounds how many such submissions a
provider will tolerate before it fails closed. Live guest leases and general
MTLB resolution remain unimplemented. The core `LeaseLedger` defines
completion-driven lease release, and `metal_api_core::completion::wire` defines
the transport-independent notification stream and its sender-side publisher
that carry those tokens between processes; `metal-api-ipc` carries the stream
and the owner-to-provider `MCC1` command channel, and the native provider maps
borrowed reservations with `newBufferWithBytesNoCopy:`. Provider-side
guest-memory import remains unimplemented.

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
single-process completion over a Unix socket, a staged lease import where the
owner supplies the reservation window and the provider executes and retires it,
a no-copy host-memory import that proves live reads and in-place GPU writes,
a two-process owner/provider completion hop, and an owner-driven command
channel where the parent compiles, submits, waits, reads back and releases on
the child and imports both a staged lease and a descriptor-backed no-copy
lease. The runner spawns itself with `--completion-child`: the child owns the
Vulkan device and publishes admission and `CompletedVisible` through the real
outbox and writer thread, while the parent owns only the listener, mirror and
lease ledger. `--command-child` serves the same provider over `MCC1`; the
parent owns no provider and drives every operation remotely. This is
Vulkan-provider versus Vulkan-executor regression coverage, not native Metal
parity.

## Verification of this local increment

- 2026-09-08 remote-provider channel-failure contract: a `RemoteProvider` whose
  command channel closes reports `ProviderHealth::Exhausted` from `health()`
  instead of blocking or claiming `DeviceLost`, and its operations fail with
  the structured `provider_unavailable` resource error and
  `Retryability::RetryAfterRecreate`; the owner must recreate the provider
  before new work is admitted. Framing and contract errors remain
  `command_transport_failed` with unknown retryability because they describe a
  protocol defect rather than a missing provider. A local unnamed mapping is
  now rejected with the args-class `shared_mapping_unnamed` instead of an I/O
  error. The new IPC test
  `remote_provider_reports_exhausted_after_the_command_channel_closes` uses a
  one-shot server that answers `Capabilities` and `Health` then closes,
  observes EOF on the next exchange and pins the `Usable` to `Exhausted`
  transition plus the structured operation refusal. The owner half
  `run_remote_provider_disconnect_process` repeats the contract across a real
  process boundary: it kills the command child after a `Usable` health check
  and reports
  `PASS provider_disconnect_process transport=unix health=Exhausted refusal=provider_unavailable retry=RetryAfterRecreate killed=true`.
- 2026-09-08 owner-to-provider command channel: `metal-api-ipc::command` adds a
  versioned `MCC1` request/response channel with `RemoteProvider`,
  `serve_provider` and `serve_provider_unix`. The owner remotely compiles,
  submits, waits, reads back, cancels and releases; the provider re-admits every
  submission with its own capabilities before calling `submit`.
  `ImportStagedLease`/`ReleaseStagedLease` and `health` travel over the channel,
  and `ImportBorrowedLease` carries the reservation in the frame plus the owner
  mapping with `SCM_RIGHTS`; the Unix server maps it, keeps it alive until
  release and imports it through `NoCopyLeaseImporter`. Requests larger than
  the sender's frame limit are split into chunk frames and reassembled before
  decoding, bounded by `MAX_CHUNKED_REQUEST` (256 MiB); the command smoke
  lowers the limit to 1 KiB so every request takes that path. Lavapipe reports
  `provider_command_process ... commands=health,compile,import_lease,import_borrowed,submit,wait,readback,release completion=mirrored writeback=exact lease=retired,refused borrowed=retired,in_place`.
- 2026-09-08 Vulkan no-copy lease import: `metal_api_core::provider` gains
  `BorrowedLease`, `BorrowedLeaseRegistry` and the `NoCopyLeaseImporter` trait.
  The Vulkan provider advertises `StorageMode::BorrowedNoCopy` only when the
  device exposes `VK_EXT_external_memory_host`, imports the owner's aligned host
  mapping with `VkImportMemoryHostPointerInfoEXT`, and reads and writes it in
  place. Per-submission retains gate release (`lease_in_use`); a destroying
  execution drop retires them, while a retained in-flight submission keeps them
  held. Two new core tests cover pointer/range validation and retain/release
  resolution. Lavapipe reports `provider_borrowed_lease lease=98
  alignment=4096 copy_in=live copy_out=in_place retired=true
  refusal=lease_not_imported`.
- 2026-09-08 native staged lease import: `NativeMetalProvider` implements
  `LeaseImporter`, advertises `StorageMode::StagedLease` and resolves every
  admitted view through `LeaseRegistry` before `newBufferWithBytes:`. A new
  macOS test imports an 8-byte window, executes the copied view, retires the
  lease through `LeaseLedger` and refuses it after release.
- 2026-09-08 Vulkan staged lease import: `metal_api_core::provider` gains
  `StagedLease`, `LeaseRegistry` and the `LeaseImporter` trait. The Vulkan
  provider advertises `StorageMode::StagedLease`, copies the owner's reservation
  window into provider-owned storage at submit time, slices each view from the
  reservation, and refuses unimported, snapshot-mismatched or out-of-range
  leases; `BorrowedNoCopy` remains typed-refused. Three new core tests cover
  exact-length validation, import/release/duplicate resolution and foreign
  epoch refusal. 224 Rust tests and 115 Python tests passed; Lavapipe reports
  `provider_staged_lease lease=97 writeback=exact retired=true
  refusal=lease_not_imported`.
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
- 2026-09-08 Windows rail restored and host-pointer import validated: the
  Windows GNU build had been broken since the command-channel descriptor work
  (`descriptor_error` was Unix-gated). `e891102` makes it unconditional and
  gates the Unix-domain-socket smoke cases with `#[cfg(unix)]`.
  `provider-smoke.exe` on the RTX 5060 now runs the cross-platform suite,
  including `provider_staged_lease` and
  `provider_borrowed_lease lease=98 alignment=4096 copy_in=live
  copy_out=in_place`, the first real-device check of
  `VK_EXT_external_memory_host` host-pointer import; all 24 v1-v8
  direct/object/async-object captures still match. Binary SHA-256 values are
  recorded in `evidence/windows-rtx5060-e891102-2026-09-08/manifest.md`. The
  Unix command-channel/descriptor cases are skipped on Windows because there
  is no `SCM_RIGHTS` equivalent. The optional reims adapter also cross-compiled
  and ran on the RTX 5060 (`reims-smoke.exe`, `PASS suite executor=reims`).
- 2026-09-08 Windows named-mapping command channel: `metal-api-ipc` can create
  named mappings (`shm_open` on Unix, `CreateFileMappingW` on Windows) and
  carry the mapping name and length after an `ImportBorrowedLease` frame, so
  the owner/provider command channel no longer needs `SCM_RIGHTS` for no-copy
  leases. `provider-smoke` runs a two-process TCP case where the RTX 5060
  child opens the owner's named section, imports it through
  `VK_EXT_external_memory_host` and writes through the owner's pages; a
  duplicate import is refused after consuming its mapping name and the
  connection stays framed. CI run `34243286265` passed all five jobs with 241
  Rust tests (core 145, ipc 26, receiver 5, sender 2, transport 4, native 9,
  Vulkan 36, capture 14), 115 Python tests, the Lavapipe v1-v8 direct, object
  and async-object captures and the macOS/reims jobs. The RTX 5060 run is
  archived in `evidence/windows-named-8439f7e-2026-09-08/` and the CI run in
  `evidence/windows-named-8439f7e-2026-09-08/run-34243286265/`.
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
- Device-loss/timeout mappings are unit-tested and `provider-smoke` injects a
  simulated device loss to check the health/refusal/recreate lifecycle; no real
  GPU device loss or timeout was injected. No Vulkan validation layer was
  available locally.
- Initial publication CI at `9d7c007` passed. The workflow now includes
  `provider-smoke`, but this increment has not been pushed or run in remote CI.

No native Metal provider, production reims routing, guest-memory lifecycle or
VM/display result is claimed for this step.
