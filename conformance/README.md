# Native Metal reference capture

This directory contains the native Metal/Vulkan comparison harness for bounded
buffer-compute cases. The native runner is a standalone Swift program using
Metal; it is not a Rust `ComputeProvider` implementation. Its source is prepared
for Apple silicon macOS 11+. The initial capture version at `a57c985` compiled
and validated the shared suite in macOS CI. The subsequent probe run at
`8579380` captured both v1 cases on an Apple Paravirtual device and successfully
compared them to Vulkan. The current [eight-case v2 extension](SUITE-V2.md) is
also passed Swift/Vulkan comparison in
[CI run 33974824176](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/33974824176).

## Shared input

Both runners load the exact bytes of `suite.json`. It names two cases:

- `copy_word`: one invocation copying a 4-byte input, with nonzero buffer offsets.
- `indexed_boundary`: a 10x3 grid, nominal 8x2 threadgroups, a barrier and 30 indexed
  output words. The last groups test nonuniform threadgroup sizes.

Each case includes binding/view/allocation identity, offsets, full backing
sizes, initial bytes, expected writebacks and an LLVM/MSL source pair with
SHA-256 hashes. Both runners verify the source hashes and admit only the two
reviewed source identities and dispatch shapes. The pairing is a manually
reviewed semantic fixture, not evidence that MSL and AIR bytes are identical.
The [v2 suite](SUITE-V2.md) adds input/offset variants, more group boundaries,
and an owned 3D read/write fixture while keeping v1 unchanged. Changing a
source requires updating and reviewing both runners' source pins.
Git attributes keep hashed files at LF line endings on Windows too.

Only the successful completion of all cases produces a report. Existing output
paths are refused. Keep real captures in the ignored `conformance/captures/`
directory; generated binaries and reports are not committed.

## Windows or Linux: capture Vulkan

From the repository root, with the usual LLVM/SPIR-V and Vulkan tools installed:

```sh
mkdir -p conformance/captures
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --suite conformance/suite.json --output conformance/captures/vulkan.json
python3 conformance/compare.py --suite conformance/suite.json \
  --check conformance/captures/vulkan.json
```

For software Vulkan on Linux, set `VK_ICD_FILENAMES` to the installed Lavapipe
ICD JSON. On Windows, run the compiled `provider-capture.exe` with the same
arguments; `python` may replace `python3`. The root README lists tool path
environment variables. Avoid PowerShell redirection for capture transport;
`--output` writes UTF-8 bytes directly.

## Apple silicon Mac: capture native Metal

Install Xcode or its command-line developer tools. From the same revision of
this repository, compile the Swift runner and validate inputs without a GPU:

```sh
mkdir -p conformance/captures
xcrun swiftc -swift-version 5 -warnings-as-errors \
  -framework Foundation -framework Metal -framework CoreGraphics -framework CryptoKit \
  conformance/NativeOracle.swift -o conformance/native-oracle
conformance/native-oracle --suite conformance/suite.json --validate-suite
```

Query device eligibility without submitting GPU commands:

```sh
conformance/native-oracle --probe
```

Then run the actual reference capture:

```sh
conformance/native-oracle --suite conformance/suite.json \
  --output conformance/captures/native-metal.json
python3 conformance/compare.py --suite conformance/suite.json \
  --check conformance/captures/native-metal.json
```

The runner requires a unified-memory Apple GPU with nonuniform threadgroups;
unsupported devices fail instead of rounding up the dispatch. It uses shared
buffers, binds the actual offsets, dispatches the original grid/local values,
waits for a completion handler and checks command-buffer status before readback.
A 20-second completion timeout exits without a report; it does not cancel GPU
work or produce an expected-output substitute. Native shader compilation may
fail separately from Swift compilation.

The CoreGraphics link is needed for a command-line program obtaining a
[default Metal device](https://developer.apple.com/documentation/metal/mtlcreatesystemdefaultdevice%28%29).
Execution uses [exact-thread dispatch](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreads%28_%3Athreadsperthreadgroup%3A%29)
and [retained command-buffer resources](https://developer.apple.com/documentation/metal/mtlcommandbuffer/retainedreferences).

## Compare real captures

Transfer the native report to the machine holding the Vulkan report, keeping
both reports and `suite.json` from the same revision:

```sh
python3 conformance/compare.py --suite conformance/suite.json \
  --native conformance/captures/native-metal.json \
  --vulkan conformance/captures/vulkan.json
```

The comparator refuses stale suite hashes, wrong backend roles, incomplete or
duplicate cases, incomplete results, incorrect writeback identity/order/bytes,
and inconsistent full allocation contents. A `--check` pass is one capture
check, not cross-backend parity. The tests use explicitly synthetic reports to
test these refusals; no synthetic report is saved as native evidence.

## What the reports prove

Reports contain `schema_version`, suite name/hash, backend, device/platform,
`allocation_observation` and per-case completion/writebacks/allocations.

- Native `allocation_observation = gpu-buffer-readback`: bytes come from the
  complete MTLBuffer. The runner checks actual read-only buffers and guard bytes
  around the writable view after GPU completion.
- Vulkan `allocation_observation = host-writeback-landing`: the provider returns
  only writable view bytes. The runner applies those to initialized host
  allocations. Guard and read-only checks therefore verify the API writeback
  and host landing behavior, not direct GPU-side canary observations.

Passing comparison establishes agreement of the supplied host-visible results
for these two fixtures. It does not authenticate a report's hardware origin,
prove LLVM/MSL binary equivalence, test all refusal semantics or establish full
Metal conformance. Native execution is only claimed once a real capture is
collected with its device, OS and repository revision recorded.

## Validation available so far

On Windows/RTX 5060 and Linux/Lavapipe, the new Vulkan capture runs both cases
and its reports match the fixture's expected bytes. Python comparator tests
cover positive and negative report checks. Local Rust tests, formatting,
Clippy and Windows cross-build checks cover the Vulkan runner.

The initial macOS compile-only run at `a57c985` passed. The updated workflow
retains that compile/input validation and probes Metal device eligibility.
On an eligible device it captures real native output, validates it and passes
it to a separate cross-backend comparison job. It uses the existing standard
`macos-15` runner; no paid runner is configured.

If the runner has no eligible GPU, its `status.json` says `unavailable`, no
native result is generated and the comparison job is skipped. A green workflow
with that status does **not** establish native Metal parity. If an eligible
device fails compilation, execution, timeout or result checks, the job fails;
those errors are not downgraded to unavailable. Probe output, command logs and
status are preserved in `native-evidence`, and the Linux report is preserved
in `vulkan-capture` workflow artifacts.

For a collaborator's machine, the same orchestration can require a GPU:

```sh
python3 conformance/run_native.py --oracle conformance/native-oracle \
  --suite conformance/suite.json --output-dir conformance/captures/native-run \
  --require-metal
```

The output directory must be new. `status.json` distinguishes `captured`,
`unavailable` and `failed`, and the wrapper enforces process timeouts in addition
to the capture tool's GPU fence timeout. `captured` means a validated native
report exists; cross-backend comparison is a separate check. The status may
also carry `--source-revision` for local capture provenance.

The initial probe workflow completed real native v1 capture and comparison in
[CI run 33973870668](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/33973870668).
The v2 extension also passed real native comparison in
[CI run 33974824176](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/33974824176).
The new Rust-native provider path still needs its own remote run. No Swift compiler, macOS SDK or Apple GPU is available locally.
Python orchestration tests use simulated commands and synthetic reports only.

## Shared Rust-native interface (new local increment)

The capture runner can also select `--backend native-metal-provider` on macOS.
This uses the same Rust `PipelineProvider` interface as Vulkan and reports
host writeback landing, while the independent Swift runner retains full GPU
buffer observations. See [the shared-provider contract](../docs/SHARED-PROVIDERS.md).
The updated workflow compares all three reports after real capture; the new
Rust native implementation is not considered validated by earlier Swift runs.

## Serial-pass extension

[suite-v3.json](suite-v3.json) adds three ordered read-modify-write cases with
2, 3 and 8 dispatches, one upload and one final completion. The earlier suites
remain unchanged. See [SERIAL-PASSES.md](SERIAL-PASSES.md) for the resource reuse
contract, expected results and pending native verification.

## Existing-view rebinding

[suite-v4.json](suite-v4.json) exercises serial buffer-role swaps with one
initial upload. The final result includes views written only by later passes.
See [BUFFER-REBINDING.md](BUFFER-REBINDING.md) for its exact admission rules and
validation limits. New v4 native comparison is pending at this local checkpoint.

## Pipeline-table extension

[suite-v5.json](suite-v5.json) alternates two different compute shaders in a
single serial submission, keeping the initialized view pool. The Rust trace
contract is now schema 2 with per-pipeline metadata; report JSON remains
schema 1. See [MIXED-PIPELINES.md](MIXED-PIPELINES.md) for boundaries and local
results. New native v5 GPU verification is pending independently of earlier
published three-way suites.

## Per-program layouts

[suite-v6.json](suite-v6.json) switches between different binding numbers,
access modes and slot orders over the fixed resource pool. See
[PIPELINE-LAYOUTS.md](PIPELINE-LAYOUTS.md) for the exact layouts and local
results. This new native v6 coverage is pending its own cloud run.
