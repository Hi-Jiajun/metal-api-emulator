# Shared compilation and native Rust provider

The `PipelineProvider` interface extends `ComputeProvider` with device identity,
compilation and explicit pipeline/completion release. `CompiledComputePipeline`
metadata now lives in `metal-api-core`. Both providers use the core's single
process-local epoch allocator, so separately created native and Vulkan contexts
cannot accidentally receive the same epoch. Epochs and pipeline IDs are not
portable handles across processes.

The caller supplies `PipelineCompileRequest { entry_name, logical_digest, source }`.
`ShaderSource` tags textual AIR/LLVM IR, binary AIR or MSL source. Providers refuse
unsupported input representations. The digest remains caller-supplied fixture
identity; it is not proof that two independently compiled modules are equal.

- Vulkan accepts textual LLVM IR and single-module binary AIR through its
  existing translator/validator. The prior concrete `compile_pipeline` helper
  remains available for source compatibility.
- `metal-api-native::NativeMetalProvider` accepts only the exact three MSL
  sources paired with the conformance fixtures. It uses their reviewed static
  or affine access proofs and fixed grids rather than trusting arbitrary MSL
  to carry a caller-declared footprint. This restriction must remain until
  general native reflection/footprint admission is implemented.

The native backend uses one serialized Metal device/queue owner, fresh shared
buffers, a retained pipeline and a synchronous completion boundary by default,
with an optional `MTLCommandBuffer` completion handler for deferred completion.
It returns allocation-relative writable view data validated against the same
core trace contract. A timeout observation retires the submission while the
context stays usable until the abandonment budget is exhausted; unknown GPU
resources remain retained rather than being freed prematurely.
It supports no-copy host reservations, serial multi-pass commands and
asynchronous cancellation; guest mappings, general buffer aliasing and
production reims routing remain outside this subset.

## One capture runner

`provider-capture` selects a provider once, then compiles, submits, waits and
releases resources through `dyn PipelineProvider`. Provider-specific Metal or
Vulkan objects are absent from the shared case execution path.

```sh
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --backend vulkan --suite conformance/suite-v2.json --output vulkan.json
# macOS only:
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --backend native-metal-provider --suite conformance/suite-v2.json --output rust-metal.json
```

The existing Swift oracle remains an independent program and is not replaced.
The comparator can now check three actual captures:

```sh
python3 conformance/compare.py --suite conformance/suite-v2.json \
  --native swift-metal.json --vulkan vulkan.json --metal-provider rust-metal.json
```

The new report backend is `native-metal-provider`, with allocation observation
`host-writeback-landing`. Like the Vulkan provider, it reports view writes that
are landed into initialized host allocations. The Swift `native-metal` oracle
continues to report full Metal buffer readback, including actual GPU guard and
read-only observations. Reports from the two native paths are not interchangeable.

## Completion-driven lease lifetime

`metal_api_core::provider::LeaseLedger` tracks owner-issued buffer leases across
submissions. The owner registers each `LeaseReservation`, binds the completion
token of every submission that references the lease, and may release the
backing only when every bound token has retirement evidence. `NotSubmitted`,
`CompletedVisible`, `Failed` and `DeviceLost` retire a token; `Submitted`,
`TimedOut`, `Cancelled` and `SubmittedUnknown` keep the lease held. A provider
that establishes retirement out of band can call `retire`, and `device_lost`
releases every registered lease as a teardown guarantee. Binding is
idempotent, and one token may cover several leases.

This is a core lifetime contract. The Vulkan provider also imports owner-issued
`StagedLease` backing: `LeaseImporter::import_staged_lease` copies the
reservation window into provider-owned storage, `StorageMode::StagedLease` is
advertised in its capabilities, and a view is resolved from the imported bytes
at submit time. When the device exposes `VK_EXT_external_memory_host`, the same
provider also advertises `StorageMode::BorrowedNoCopy`: `NoCopyLeaseImporter`
imports an aligned owner host mapping with `VkImportMemoryHostPointerInfoEXT`,
`BorrowedLeaseRegistry` tracks per-submission retains, and GPU writes land in
the owner mapping in place. A retained in-flight submission keeps the import
held, and release is refused with `lease_in_use` until a destroying execution
drop retires it. The native provider also imports `StagedLease` windows: it
advertises `StorageMode::StagedLease`, resolves the admitted window through the
same registry and uploads the copied bytes. It also maps aligned borrowed
reservations with `newBufferWithBytesNoCopy:` and imports them as
`BorrowedNoCopy`; guest memory import remains future work.

## Cross-process completion notifications

`metal_api_core::completion::wire` is the transport-independent half of a
cross-process completion protocol. The provider process publishes
`CompletionMessage` values; the owner process applies them to a
`CompletionMirror` that converges to the same terminal semantics as the
in-process `CompletionRecord`. Every per-token and device-health stream carries
a monotonic `CompletionSequence`, so duplicate, reordered and coalesced
notifications are safe to replay. A token's first terminal observation wins;
when two conflicting terminals arrive in different orders, the lower sequence
wins. `Exhausted` refuses new admissions but keeps in-flight tokens observable,
while `DeviceLost` marks every non-terminal token and is teardown evidence.
`TimedOut` is owner-local and never appears on the wire, and `CompletedVisible`
means the owner may request the readback over the separate data channel.

`CompletionMirror::observe_into` applies the converged observation to a
`LeaseLedger`, so the owner can release a lease only after a terminal with
retirement evidence or device loss. `CompletionPublisher` is the sender-side
dual: it assigns the per-stream sequences, returns the identical message for an
idempotent terminal replay, and refuses a different terminal, a non-terminal
after a terminal, or a token update after device loss. `LoopbackTransport` is
an in-memory test and diagnostic transport that preserves order; a real pipe,
socket, shared ring or RPC layer chooses its own encoding and error type. The
wire types have no serialization dependency. `metal-api-ipc` carries them over
any `Read` + `Write` stream with a versioned `MCW1` frame codec:
`CompletionReceiver` drives the owner mirror and lease ledger,
`sender::spawn_writer` runs the provider-side writer thread, and
`provider-smoke` proves a two-process split where the child owns the Vulkan
device and publishes through a Unix socket while the parent retires a lease
from the mirror alone. `metal-api-ipc::command` adds the opposite direction: a
versioned `MCC1` request/response channel where the owner compiles, submits,
waits, reads back, cancels and releases on a provider in another process, and
imports staged leases plus descriptor-backed no-copy leases over the same
connection, and chunk frames carry requests larger than one frame. A rejected
duplicate import still consumes its descriptor so the connection stays framed.
dma-buf is a deliberate non-goal for the Windows rail (Linux-only kernel
object); host-pointer import and the copy rails are the fallbacks. Guest memory
and the production guest/display path remain future work.

The Unix-domain-socket IPC, command-channel and shared-memory descriptor cases
are `#[cfg(unix)]`. The Windows build skips them and runs the cross-platform
Vulkan suite, including the single-process staged and borrowed no-copy imports;
the borrowed import was validated on an RTX 5060 with
`VK_EXT_external_memory_host` (`copy_out=in_place`). A Windows shared-memory
handle transport is not implemented, so the owner/provider command channel
remains Unix-only for borrowed leases.

The macOS workflow builds/tests the native crate before running the Swift GPU
probe. Only after successful eligible Swift captures does it run Rust-native
captures; any Rust native execution or comparison failure fails the job. The
comparison job then checks both v1 and v2 with all three captured results.

## Verification boundary

This local increment prepares a Rust native backend and shared interface.
Linux/Windows validation exercises Vulkan and the platform refusal. macOS ARM64
cross-checking can verify native Rust types, but not link the Apple frameworks
or execute GPU work. Actual macOS build, native backend execution and three-way
comparison require the updated cloud workflow. Prior Swift/Vulkan v1/v2 results
remain valid historical evidence and do not count as a Rust-native run.

## Local verification record (2026-09-06)

- Linux tests: 70 passed (44 core, 5 native contract/platform tests, 17 Vulkan,
  4 capture-runner tests); 49 Python report/orchestration tests passed.
- Workspace formatting, warnings-denied Clippy and rustdoc passed.
- macOS ARM64 target: all Rust native/capture targets type-check and native
  Clippy passes with warnings denied. No Apple SDK/framework link or GPU
  execution was performed by those cross-target checks.
- Linux/Lavapipe and Windows/RTX 5060: the common-interface Vulkan v2 capture
  matches the previously observed Swift native v2 report. Provider smoke also
  passed foreign-owner release, altered metadata and unsupported MSL-source
  refusals. This reuses the unchanged fixture identities, not an old Vulkan
  executable.
- Windows GNU capture/smoke executables linked; the optional reims workspace
  checked with its locked dependencies after the facade dependency update.
- Dependency limitation: metal-rs 0.33's transitive block 0.1.6 emits a Rust
  future-incompatibility advisory on the macOS target. It is not a warning
  introduced or suppressed in this implementation. objc 0.2's legacy macro
  feature name is explicitly declared to rustc's check-cfg lint in Cargo.toml.
- A new macOS CI run still needs to establish framework linkage, actual Rust
  native GPU execution and agreement with both Swift and Vulkan.

The former concrete Vulkan `release_pipeline(PipelineId)` API is replaced by
`release_pipeline(&CompiledComputePipeline)`, matching the shared trait and
requiring epoch/metadata verification. This is an intentional API change in
this experimental repository. Existing capture/smoke callers were updated.

## Follow-up: bounded serial passes

The [serial-pass contract](../conformance/SERIAL-PASSES.md) introduced up to
eight serial dispatches, uploaded once and read back once. Subsequent
[rebinding](../conformance/BUFFER-REBINDING.md),
[mixed-pipeline](../conformance/MIXED-PIPELINES.md) and
[per-program layout](../conformance/PIPELINE-LAYOUTS.md) extensions added view
permutations and per-pass pipeline selection. The Rust trace envelope uses
schema 2 with an explicit pipeline table; every pass resolves its own contract
and registry entry. V1-v6 passed
[three-way native/Vulkan CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/33983234385).

The [resource-subset extension](../conformance/RESOURCE-SUBSETS.md) admits up to
64 unique views across the trace. Each pass binds the views needed by its
selected pipeline. Later first use is supported; repeated view identities
retain their original allocation, range and initial bytes. The complete union
is validated and uploaded before encoding. Readback covers every view written
by any pass. Both providers offer an optional async mode: Vulkan defers
completion to a device fence observed by `wait`, while native Metal uses an
`MTLCommandBuffer` completion handler. Both still serialize submission;
concurrent GPU execution, mid-execution CPU uploads and general aliasing remain
outside this provider subset. A provider-scoped `AbandonmentBudget` bounds how
many submissions may lose their completion observation before the provider
fails closed with `provider_unavailable`/`RetryAfterRecreate`; a confirmed
device loss instead reports `DeviceLost`, destroys the lost device's handles
and refuses new work with `RetryAfterRecreate`; Vulkan derives it from
`VK_ERROR_DEVICE_LOST`, while native Metal derives it from
`MTLCommandBufferError::DeviceRemoved` (code 11) in the command buffer error
object. V1-v7 passed
[three-way native/Vulkan CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/34010989175).
The v8 binary-AIR encoding suite reuses those cases and passed the five-path
v1-v8 comparison in
[CI run 34223294821](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/34223294821).

The subsequent [provider object API](PROVIDER-OBJECTS.md) connects application
objects to this trace boundary. Its native execution and five-path v1-v7
validation passed in [CI run 34011824447](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/34011824447).
