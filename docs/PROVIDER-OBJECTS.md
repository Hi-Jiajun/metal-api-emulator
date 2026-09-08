# Shared provider object API

`metal_api_core::provider_api` connects source-level compute objects to the
shared `PipelineProvider` boundary. Previously the application object model
used `ComputeExecutor` and executed each pass with its own CPU snapshot and
readback. The new entry point records the entire command buffer into one
canonical trace, then calls the selected native Metal or Vulkan provider once.
The existing `ComputeExecutor` entry point remains available.

The objects are experimental Rust APIs, not Metal.framework ABI objects.
Submission is split at Metal's commit/wait boundary. A provider that finishes
inside `submit` completes at commit; a provider that returns `Submitted` leaves
the command pending until `wait_until_completed` observes completion, retrieves
`readback` and lands validated writebacks. This increment does not add a
background worker, general shader support, guest memory, rendering or
production reims routing.

## Recording and execution

1. Create `Device::new(Arc<dyn PipelineProvider>)` and compile a
   `PipelineCompileRequest` into an opaque `Pipeline` object.
2. Create CPU buffers with `new_buffer_with_bytes`, then explicit
   `buffer.view(offset, length)` objects. Each view owns an identity; clone the
   same view to bind it again in another pass. The current provider subset
   allows only one view of each allocation within a command buffer.
3. Create a command queue, command buffer and compute encoder. Bind a pipeline
   and its views. Each `dispatch_threads` records the current pipeline and
   bindings immediately; later changes do not overwrite earlier dispatches.
   `clear_buffers` removes the previous binding table when switching layouts.
4. End encoding and commit. The API reserves the complete resource union at
   commit time, snapshots its bytes, validates all passes, then makes one
   provider submission.
5. A `CompletedVisible` submission with a completely validated final output
   lands writes and sets status to `Completed` at commit. A `Submitted`
   submission stores its token, exact trace and reservations; the first
   `wait_until_completed` retries non-terminal timeouts, calls `readback`,
   validates the result against that trace and only then lands writes and sets
   `Completed`.

All bindings derive access from the selected pipeline metadata. Device owners
are opaque shared identities, so a foreign pipeline or view is refused even
when another wrapper refers to the same underlying provider. Allocation, view
and operation identities are generated internally. Limits remain at most eight
serial passes and 64 views, additionally constrained by provider capabilities.

Initial CPU bytes are read at commit time. The API reserves every participating
buffer in allocation order and keeps those reservations until completion,
validation and landing finish. Concurrent CPU reads/writes and overlapping
commands wait for that boundary. Commands sharing buffers acquire reservations
in the same order. Every readback and destination range is checked before the
first host write, so an invalid later result cannot leave an earlier buffer
partially updated.

Provider implementations must not synchronously reenter these same object
buffers during `submit`, `wait` or `readback`: the caller holds their
reservations at that time.

## Lifetimes and failure

Recorded dispatches retain their pipeline and buffer/view objects. Dropping an
application's pipeline handle after encoding does not release its registry
entry while the command still references it. The last pipeline owner releases
the provider entry. Command lifetime owns the completion observation.

Open or abandoned encoders, missing bindings, unsupported ranges and foreign
objects are refused. A committed command is single-use. Provider failures or
panics terminate its state and wake waiters without applying partial output.
An asynchronous `Submitted` response is sufficient only when the provider
implements `readback`; otherwise the command fails at wait with a structured
capability error and no host bytes change. Dropping a pending command releases
its reservations and completion record without claiming that unknown GPU work
retired; the backend's retention policy still governs GPU storage.

## Capture and comparison

No shader or suite is added. `provider-capture --api objects` reruns the same
26 cases from v1-v7, including multiple dispatches on one encoder, pipeline
changes, changed slot counts and buffers first used later. The capture maps
opaque object allocation/view IDs to fixture labels only after the object API
has validated the actual provider result. Allocation reports read the actual
object buffers after landing.

| Path | Report backend | Allocation observation |
|---|---|---|
| Independent Swift collector | native-metal | Full GPU buffer readback |
| Vulkan direct trace | vulkan | Host writeback landing |
| Rust Metal direct trace | native-metal-provider | Host writeback landing |
| Vulkan object API | vulkan-objects | Host writeback landing |
| Rust Metal object API | native-metal-provider-objects | Host writeback landing |

```sh
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --api objects --suite conformance/suite-v7.json --output vulkan-objects.json
# Deferred Vulkan completion (same report backend and host-visible results):
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --api objects --async --suite conformance/suite-v7.json \
  --output vulkan-objects-async.json
# On macOS:
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --api objects --backend native-metal-provider \
  --suite conformance/suite-v7.json --output metal-objects.json
python3 conformance/compare.py --suite conformance/suite-v7.json \
  --native swift-metal.json --vulkan vulkan-trace.json \
  --metal-provider metal-trace.json --vulkan-objects vulkan-objects.json \
  --metal-objects metal-objects.json
```

Both object paths preserve the provider observation boundary: host guard bytes
do not observe GPU memory outside an uploaded view. The comparator requires
the exact backend identity for each CLI input; direct trace captures cannot
stand in for object captures.

## Verification checkpoint

- 144 Rust tests passed: core 86, native 7, Vulkan 37, capture 14. The eight
  new core tests cover asynchronous `Submitted` finalization, readback
  validation, non-terminal wait retries, terminal failure/unknown completion,
  unsupported readback, reservation blocking and overlapping async commands.
  Three new Vulkan tests cover the shared completion record's timeout,
  waiter-wakeup, readback and failure paths.
  Earlier tests cover single submission, recording snapshots, commit-time
  bytes, foreign ownership, limits, aliases, atomic failure, panic/waiter
  recovery, concurrent commands and pipeline/completion retirement.
- 113 Python tests passed. Object captures have distinct required identities
  and must satisfy the same full writeback and allocation checks as trace
  captures. Synthetic reports are comparator tests only.
- `VulkanComputeProvider::with_async_execution(true)` now returns `Submitted`,
  runs the prepared owned-byte request on one worker under the shared queue
  lock, and serves `wait`/`readback` from a shared completion record. The
  default synchronous mode, direct trace rail and all five comparison paths are
  unchanged. CI runs `provider-capture --api objects --async` for v1-v7 on
  Lavapipe; the native provider still completes inside `submit`.
- All 26 v1-v7 object cases passed on Linux/Lavapipe and Windows/RTX 5060.
  Results agree per allocation/view with fresh Linux direct-trace captures and
  the archived Swift/Rust Metal reports from run 34010989175.
- All existing suite and shader files remain byte-for-byte unchanged.
  Formatting, Clippy, rustdoc, Windows GNU build and macOS ARM64 Rust
  typecheck/Clippy passed. The pre-existing `block` dependency still emits its
  future Rust compatibility notice for the macOS target.

Native object-API GPU execution and all-five comparison passed in CI run
`34011824447` for commit `489b489c11b43afbf821295d9d5fe8d9303e1e79`.
The downloaded evidence was revalidated locally: 7 suites, 35 reports and 26
cases per path; an independent canonical comparison found every case's
completion, writebacks and allocation bytes equal across all five paths. Raw
JSON ordering and backend/device metadata may differ. No new Metal shader
support is claimed.
