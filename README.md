# Metal API emulator experiment

Experimental source-level Metal compute objects backed by Vulkan, for fast
host-side iteration without booting a VM. The long-term proposal is to share
one Metal semantic path between native Metal and a Windows Vulkan provider.
This repository is an independent prototype; upstream has not adopted it.

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
  `observe_into`; the actual IPC transport and provider wiring remain future
  work.
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
- Open design work: device-loss reclamation beyond the bounded abandonment
  budget, the cross-process completion transport and provider wiring (the core
  wire publisher and mirror now exist), provider-side lease import (the core
  `LeaseLedger` release contract now exists), general native shader admission,
  CPU uploads during command-buffer execution and aliases.
  Resource snapshots do not hold live guest pages.
- Not implemented: general MTLB function-name resolution, Windows MSL compilation,
  textures, rendering, presentation, heaps, ICBs or production
  reims integration. This is not a Metal.framework ABI implementation.

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
tokens and explicit registry release. This is a provider/legacy Vulkan
comparison, not a native Metal oracle.

The indexed case launches a 10x3 grid with an 8x2 nominal threadgroup, exercises
a barrier and checks all 30 output words. Its source and reference output are
attributed in [NOTICE.md](NOTICE.md). Generated AIR stays temporary.

## Evidence and limits

Earlier local checkpoints ran these four checks on Linux/Lavapipe and
Windows/RTX 5060 with both Vulkan executors. Those runs validate the narrow
snapshot executor path. They do not demonstrate native Metal parity.
Publication-preparation checks are recorded in [docs/VALIDATION.md](docs/VALIDATION.md).
The subsequent provider implementation is described in
[docs/PROVIDER-B1.md](docs/PROVIDER-B1.md).

The standalone fence wait has a 20-second bound. Reims uses its explicit
synchronous retirement entry. Neither interface provides an end-to-end
initialization/compilation/lock deadline. Guest memory, display and VM behavior
remain outside this offline test boundary.

## License

LGPL-3.0-or-later; see [LICENSE](LICENSE), [COPYING](COPYING) and
[NOTICE.md](NOTICE.md) for the source attribution and license texts.
