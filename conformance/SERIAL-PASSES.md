# Serial compute passes: one upload and one completion

The compute providers now admit 1..8 ordered dispatches using one pipeline and
one unchanged buffer table. This is a bounded extension of the existing
`ComputeTrace.passes` interface. It is not arbitrary command-buffer support.

For more than one pass:

- `encoder_dispatch_type` must be Serial and every pass names the same pipeline.
- Every buffer declaration, binding, allocation/view ID, access, offset, length
  and source must match the first pass exactly. Repeated OwnedBytes specify
  the initial upload once, not data to upload again before each pass.
- Only grid/local dispatch dimensions may differ, and each pass must satisfy
  the pipeline contract, buffer footprints and device limits.
- The result contains one complete writeback per writable view, after all
  dispatches complete. It does not expose intermediate pass results.

Both providers revalidate the complete trace before submission. Vulkan plans
all passes first, creates a shared set of GPU buffers and pipeline variants,
records one command buffer and submits it once. Compute-write to compute-read/
write barriers order successive passes; a final compute-to-host barrier precedes
fence completion and CPU readback. The legacy snapshot executor enters the same
engine with a one-item dispatch list and retains its public behavior.

The native provider and Swift oracle create tracked shared buffers once and
record one serial compute encoder per pass in the same command buffer. Directly
bound tracked Metal resources establish ordering across encoders. They commit
once and read only after completion. The existing unknown-completion retention
policy remains in effect; adding passes does not create a cancellation path.

The `max_passes = 8` limit is policy for this increment. The older snapshot
executor's metadata helper still reports its single-pass interface. Changing
bindings, changing initial data partway through a trace, multiple pipelines,
untracked/no-copy resources and concurrent dispatch remain unsupported.

## Suite v3

`conformance/suite-v3.json` has three cases based on the unchanged reviewed
`transform_3d` source:

| Case | Dispatch count | Final in-place value |
|---|---:|---|
| transform_twice | 2 | initial + 2 * bias (mod 2^32) |
| transform_three_times | 3 | initial + 3 * bias (mod 2^32) |
| transform_eight_times | 8 | initial + 8 * bias (mod 2^32) |

The grid stays 5x3x2 and local sizes cycle through 4x2x2, 8x4x4 and 1x1x1.
Every pass reads the prior contents of slot 0, adds the read-only slot-2 bias,
and updates slot 0 plus the XOR result at slot 5. If the implementation uploads
initial data repeatedly or returns the previous pass's result, the final bytes
cannot match. Buffer offsets, guards and sparse binding identities are retained.

The optional `dispatches` array is present only in v3. Its first entry must
match the case's grid/local fields. Capture runners require the reviewed
2/3/8-item sequences; v1/v2 refuse extra sequences and remain byte-for-byte
unchanged. The report schema is still 1; the suite name and raw-byte digest
separate versions. CI preserves and compares each suite's reports separately.

```sh
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --suite conformance/suite-v3.json --output conformance/captures/vulkan-v3.json
# On macOS after building the Swift oracle:
python3 conformance/run_native.py --oracle conformance/native-oracle \
  --suite conformance/suite-v3.json --output-dir conformance/captures/native-v3 \
  --require-metal
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --backend native-metal-provider --suite conformance/suite-v3.json \
  --output conformance/captures/rust-metal-v3.json
python3 conformance/compare.py --suite conformance/suite-v3.json \
  --native conformance/captures/native-v3/native-metal.json \
  --vulkan conformance/captures/vulkan-v3.json \
  --metal-provider conformance/captures/rust-metal-v3.json
```

## Local verification

- 82 Rust tests passed: core 50, native contract/platform 5, Vulkan 21, capture 6.
- 55 Python tests passed; new tests independently compute the serial result and
  reject single-pass/reset outputs or duplicated intermediate writebacks. Those
  reports are synthetic unit fixtures only.
- Vulkan v3 capture passed on Linux/Lavapipe and Windows/RTX 5060; all final
  result arrays match. The v2 capture still matches the earlier Swift reference;
  provider smoke and its pre-submit refusal checks passed.
- Formatting, warnings-denied Clippy/rustdoc and Windows GNU build passed.
  macOS ARM64 native types and Clippy passed without running GPU work.
- The new Swift serial code and actual Rust Metal sequence execution still
  require a new cloud run. Earlier v1/v2 native results cannot prove v3.

This remains offline compute work. No production reims wiring, guest memory,
rendering or display behavior is claimed. Swift captures full GPU buffer bytes;
Rust providers expose host writeback landing, as documented in the main
[conformance instructions](README.md).
