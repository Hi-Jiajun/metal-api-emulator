# Native Metal reference capture

Current state: the harness covers the compute/alias/texture/command-buffer
suites `suite.json` through `suite-v18.json`. The render-bearing suites are
`suite-v13.json` (offscreen 2x2 colour attachment), `suite-v14.json` (the same
attachment as a surfaceless present target, with `research/docs/24`'s
acquire/present counts), `suite-v16.json` (the indexed vertex-input quad),
`suite-v17.json` (a loading pass) and `suite-v18.json` (one draw writing two
colour locations). The render and present observation rules live in
[RENDER-CAPTURE.md](RENDER-CAPTURE.md) §6-§7 and the MRT rules in §10; the
per-suite rules are exercised by
`python3 -m unittest discover -s conformance`. The v1/v2 description below is
the historical starting point and is kept for the suite's own record.

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

The same job also runs the offscreen render self-test
([RENDER-CAPTURE.md](RENDER-CAPTURE.md) §5) whenever the oracle's probe calls
the device eligible, and prints `render selftest: SKIP (no Metal device)`
instead of failing when it does not. Only a log carrying `4080c0ff` four times
and `render_selftest: PASS` is evidence that the reviewed render path executed
on Apple hardware; a SKIP is an infrastructure outcome, not a native result.

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
results. Native v6 passed [three-way CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/33983234385).

## Resource subsets

[suite-v7.json](suite-v7.json) varies the resources and their count between
passes. Later passes first use additional buffers; the whole trace resource
union is still uploaded before any execution. See
[RESOURCE-SUBSETS.md](RESOURCE-SUBSETS.md) for the 2/4/8-pass cases, bounds and
validation status. Native v7 passed
[three-way CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/34010989175).

## Binary AIR encodings

[suite-v8.json](suite-v8.json) reuses the v7 2/4/8-pass cases and adds an
`air_encoding` field: `raw` submits bare LLVM bitcode and `wrapped` submits the
Apple wrapper with an exact offset/size payload. Text IR remains the default
for v1-v7. The Vulkan rails assemble the reviewed `.ll` fixture with `llvm-as`
and submit the selected encoding; the Swift and Rust native rails keep using
the reviewed MSL fixture. Native v8 passed
[five-path CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/34223294821).

[suite-v9.json](suite-v9.json) adds a command-buffer boundary dimension:
`provider-capture --api objects` and the Swift oracle split each reviewed
`subset_chain_*` dispatch sequence across two to four command buffers that commit
and complete in order, and the capture merges the per-view final landed bytes.
See [SUITE-V9.md](SUITE-V9.md).

Because every command buffer is its own submission, provider reports for these
cases carry `group_counts`: one `copy_in`/`copy_out` pair per committed command
buffer, each counting one device-buffer operation per allocation that group's
own dispatches touch or write. The flat `copy_in`/`copy_out` of the case remain
the sum over the groups. `compare.py` refuses a provider capture whose group
number, per-group counters or summed totals disagree with the fixture, instead
of skipping split cases; single-submission cases keep the case-level contract,
and the Swift oracle stays outside both (research/docs/15 §5b).

## Ranged aliasing

[suite-v10.json](suite-v10.json) binds two **disjoint views of one allocation**
through the reviewed `copy_word` program, so a real provider must admit the
ranged-alias shape and land both views into a single allocation extent. The
reversed case swaps which offset is the source so an offset mix-up cannot pass.
The comparator compares each allocation once, refuses overlapping
initialization ranges, and refuses shared allocations outside this suite. See
[SUITE-V10.md](SUITE-V10.md).

## Provider object entry point

`provider-capture --api objects` runs the same v1-v12 fixtures through the new
shared object API. Its reports identify `vulkan-objects` or
`native-metal-provider-objects`; both report actual host-buffer landing.
The direct trace captures remain separate, and the Swift oracle is unchanged.

## Sampled textures

[suite-v11.json](suite-v11.json) adds the first sampled-texture case: one
invocation reading texel (0, 0) of a 4x4 `R32Uint` image, which is why it cannot
see a wrong row stride. v11 provider captures also carry the copy-in/copy-out
counters for the texture upload (`research/docs/18` step 3).

[suite-v12.json](suite-v12.json) reads **every** cell of that image under two
dispatch shapes (one 4x4 group and sixteen 1x1 groups) and expects `[100..115]`
for a textured image holding 0..15. This is the case that fails when a host
linear-image upload assumes tightly packed rows instead of the driver's
`VkSubresourceLayout.rowPitch`. See [SUITE-V12.md](SUITE-V12.md) for the raw
before/after captures, the count contract and the limits.
Compare all five paths with the additional `--vulkan-objects` and
`--metal-objects` arguments. See [PROVIDER-OBJECTS.md](../docs/PROVIDER-OBJECTS.md)
for object lifetimes, result validation and the current verification boundary.

## Suite coverage checks without macOS

`NativeOracle.swift` is compiled and executed only by the macOS job, and both its
suite list and its per-suite case ids are hand-written. The same case ids appear
again in `examples/metal-smoke/src/bin/provider-capture.rs`, and
`.github/workflows/ci.yml` names the suite file each rail runs. None of that is
executable on Linux, so `test_oracle_coverage.py` compares the three tables with
`conformance/suite*.json` as text:

- the suite identities in the oracle's `loadSuite` switch and in the provider's
  `validate_suite` match are exactly the committed suite files (`suite.json` is
  v1 and `suite-vN.json` is vN, and the name must agree with the identity the
  JSON declares);
- every suite pins exactly the case ids its JSON lists, in both the Swift and the
  Rust table, and the failure message names the missing and extra ids;
- the reviewed source pins in the oracle's `reviewedProgram` table are exactly
  the `air`/`metal` identities the suite files declare, and every pinned SHA-256
  still matches the committed shader or AIR bytes on disk;
- each explicit CI rail (`run_native.py`, the Vulkan capture, the native-metal
  provider capture and the three-way parity comparison) names every suite, and
  the four object-API `for version in ...; do` loops pin every version;
- the oracle's "v1 through vN" diagnostic names the last committed suite.
- a render case's `capture_rails` marker is checked against the backends that
  own a render execution path, so a marker cannot name an object-API rail,
  which has no render command encoder and therefore cannot report an
  attachment.

This is metadata consistency, not evidence: it cannot compile Swift and says
nothing about a case body, a shader hash, GPU execution or cross-backend
agreement. A render-bearing suite needs no exemption any more: every committed
suite is named on all four CI rails and in all four object-API version loops,
and the object-API captures report the declaring pass of a render-bearing
suite because the marker does not name them. If a table is reshaped so it can
no longer be parsed, the check fails instead of silently comparing empty sets.

## Offscreen render capture

[suite-v13.json](suite-v13.json) is the first suite that declares `render_cases`:
one compute case whose pass declares the attachment view, and one render case
that draws the reviewed full-screen triangle into a 2x2 `rgba8_unorm`
allocation. The attachment's texels are reported through the same
writebacks/allocations shape every compute case uses, so the comparator's
per-texel byte comparison is the whole assertion. See
[RENDER-CAPTURE.md](RENDER-CAPTURE.md) for the case schema field by field, the
report shape, the count contract and the one-command device check.

Three rails execute and report the case: the Vulkan trace rail
(`provider-capture --suite conformance/suite-v13.json`), the native provider's
trace rail (`--backend native-metal-provider`, which registers the reviewed MSL
pipeline on the native context) and the Swift oracle's suite path
(`run_native.py --suite conformance/suite-v13.json`). The two object-API rails
report the declaring case only: they carry no render command encoder in this
increment, so the suite's `capture_rails` marker names the three rails above
and `compare.py` requires the attachment exactly from them. §4 of
RENDER-CAPTURE.md lists what that wiring covers and what only an Apple GPU can
still confirm; the macOS job's `--render-selftest` step is the one Apple-side
render run observed so far.

## MRT render capture

[suite-v18.json](suite-v18.json) is the first suite that declares an
`attachments` list: one draw writes two colour locations, so the render case
carries one `expected_hex` per attachment and the comparator owes one writeback
and one allocation image per location, in location order. The declaring case
runs the reviewed `mrt_declare` xor kernel over both attachment views, so the
render submission also proves the two views were declared and read.
[RENDER-CAPTURE.md](RENDER-CAPTURE.md) §10 describes the reviewed dual MSL/SPIR-V
pair, the per-rail wiring and the `--mrt-selftest` one-device check; the suite's
schema, plan and refusals are pinned by
`conformance/test_suite_v18.py`.
