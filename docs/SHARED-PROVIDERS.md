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
buffers, a retained pipeline and a synchronous completion boundary. It returns
allocation-relative writable view data validated against the same core trace
contract. Failed or timed-out submitted work invalidates the native context;
unknown GPU resources remain retained rather than being freed prematurely.
It does not add no-copy guest mappings, general buffer aliasing, multi-pass
commands, asynchronous cancellation or production reims routing.

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

The current providers extend this single-pass checkpoint to at most eight
serial dispatches on the same pipeline/buffer table, uploaded once and read
back once. Only per-pass grid/local dimensions vary. See
[the serial-pass contract](../conformance/SERIAL-PASSES.md). General rebinding,
mixed pipelines and asynchronous command buffers remain unimplemented.

The subsequent [rebinding increment](../conformance/BUFFER-REBINDING.md) allows
per-pass permutations of the same initialized views. New views, altered source
bytes, ranges and mixed pipelines remain refused. Native v4 verification is
pending independently from earlier serial-pass results.

The [mixed-pipeline extension](../conformance/MIXED-PIPELINES.md) changes the
Rust trace envelope to schema 2 with an explicit pipeline table; every pass
resolves its own contract and registry entry. Existing-view pool rules remain.
This new v5 increment still awaits native cloud validation.
