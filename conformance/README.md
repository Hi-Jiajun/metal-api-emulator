# Native Metal reference capture

This directory prepares the first native Metal/Vulkan comparison for two small
buffer-compute cases. The native runner is a standalone Swift program using
Metal; it is not a Rust `ComputeProvider` implementation. Its source is prepared
for Apple silicon macOS 11+, but has not yet been compiled or run on a Mac in
this workspace. A successful Metal capture is still required.

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
Changing a source requires updating and reviewing both runners' source pins.
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

The prepared GitHub workflow compiles Swift and runs `--validate-suite` on a
macOS runner; it does not attempt GPU execution or manufacture a native
capture. That job has not yet been run for this local increment. No Swift
compiler, macOS SDK or Apple GPU was available for local native validation.
