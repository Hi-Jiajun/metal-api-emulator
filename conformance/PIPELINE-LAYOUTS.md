# Pipeline-specific buffer layouts

The v6 suite closes a limitation of v5: v5 changed shader code while preserving
binding numbers and access roles. `suite-v6.json` now changes both between
successive dispatches, without changing the initialized resource pool.

| Program | Slot | Access | View length |
|---|---:|---|---:|
| transform_3d | 0 | Read/write array | 120 |
| transform_3d | 2 | Read-only bias | 4 |
| transform_3d | 5 | Write-only array | 120 |
| remap_3d | 1 | Read-only bias | 4 |
| remap_3d | 3 | Read-only input array | 120 |
| remap_3d | 7 | Write-only output array | 120 |

The second shader computes `(input * 7 + bias) XOR 0x3c3ca5a5`, modulo 2^32.
It consumes the first shader's output and writes back to the earlier input
allocation, while leaving its own input unchanged. The 2/3/8-pass cases use
the same reviewed 5x3x2 grid and alternate local shapes as earlier suites.

## Per-program declarations

V6 programs declare `buffer_slots` in canonical Metal binding order. Every
entry gives a binding index, access and required length. V1-v5 retain their
original files and implicit first-program layouts; their runners refuse
explicit layout additions. The first v6 program's slots must agree with the
initial buffer metadata. Each later pass maps its logical views into the
selected program's slots, not the initial program's table.

- Rust capture derives pass bindings/access from the selected compiled
  pipeline contract and first checks the declared fixture layout against it.
- Swift independently binds by its reviewed per-program slot table, validates
  matching view lengths before GPU work and derives all written resources from
  every selected layout.
- The comparator uses the selected layout for extent checks and the union of
  writable views. This prevents a scalar/array swap or reused access metadata
  from silently passing result validation.
- Both Rust providers already resolve each pass's pipeline/layout. Vulkan
  retains its per-pass shader/layout/descriptor owners and barriers; no new
  engine-level fallback or guest behavior is introduced by these fixtures.

The capture runners pin all source bytes, paths and declared layouts. Native
source admission now includes five exact MSL fixtures. This is not reflection
of arbitrary MSL or a general Metal shader compiler on Windows.

## Run

```sh
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --suite conformance/suite-v6.json --output conformance/captures/vulkan-v6.json
# macOS:
python3 conformance/run_native.py --oracle conformance/native-oracle \
  --suite conformance/suite-v6.json --output-dir conformance/captures/native-v6 \
  --require-metal
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --backend native-metal-provider --suite conformance/suite-v6.json \
  --output conformance/captures/rust-metal-v6.json
python3 conformance/compare.py --suite conformance/suite-v6.json \
  --native conformance/captures/native-v6/native-metal.json \
  --vulkan conformance/captures/vulkan-v6.json \
  --metal-provider conformance/captures/rust-metal-v6.json
```

## Local validation

- 101 Rust tests passed: core 57, native 6, Vulkan 29, capture 9.
- 81 Python tests passed; new cases independently compute results, reject
  incorrect layout/length/access metadata, reject old v5 outputs and check
  read-only/guard observations. Synthetic reports are unit tests only.
- Vulkan v6 passed on Linux/Lavapipe and Windows/RTX 5060. Complete result
  arrays are identical. Real reflection checks confirm slots 1/3/7, read/read/
  write access, scalar reach 4 and XYZ array byte strides 4/20/60.
- V5 Vulkan regression still matches the archived Swift v5 reference.
- Formatting, Clippy, rustdoc, Windows GNU build and macOS ARM64 Rust
  typecheck/Clippy passed. Swift and native v6 GPU execution remain pending a
  new cloud run; prior v1-v5 successes cannot prove this new layout coverage.

In this v6 suite, both programs bind three buffers of the same initialized pool. Unused or
new pool resources, varying resource counts, aliasing, async behavior, general
shader support and production reims/guest/render/display paths are not tested
here. Swift captures full GPU buffers; Rust providers expose host writeback
landing, as in the earlier suites.

## Subsequent verification

Commit `66c1106` passed [three-way CI](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/33983234385):
Swift Metal, Rust NativeMetalProvider and Vulkan captures agree for v1-v6.
The native device was Apple Paravirtual on the hosted macOS runner; this is an
independent Metal execution path, not bare-metal coverage.
The subsequent [v7 extension](RESOURCE-SUBSETS.md) changes per-pass resource
counts and supports later first use.
