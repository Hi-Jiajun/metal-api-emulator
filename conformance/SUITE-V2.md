# Compute buffer suite v2

`suite-v2.json` expands the input matrix while keeping the report schema at
version 1. The original `suite.json` bytes and its two-case native reference
remain unchanged. A v1 capture can never be used as a v2 result because both
suite name and exact-byte SHA-256 must match.

| Case | Grid | Nominal local | Main coverage |
|---|---|---|---|
| copy_seed_a | 1x1x1 | 1x1x1 | Copy with nonzero input/output offsets |
| copy_seed_b | 1x1x1 | 1x1x1 | Different input and offsets on the same registered pipeline |
| indexed_tail | 10x3x1 | 8x2x1 | Four full/tail regions with a barrier |
| indexed_full | 10x3x1 | 5x3x1 | Exactly divisible dispatch |
| indexed_small_grid | 10x3x1 | 16x4x1 | All threads fit in one partial group |
| indexed_unit | 10x3x1 | 1x1x1 | Unit-sized groups |
| transform_tail | 5x3x2 | 4x2x2 | 3D indexing, sparse bindings and in-place updates |
| transform_small_grid | 5x3x2 | 8x4x4 | Smaller grid on all axes, changed input and offsets |

The owned `transform_3d` source pair uses slots 0 (read/write), 2 (read-only bias)
and 5 (write-only output). Each invocation adds the bias to one word with uint32
wraparound, stores it back, and writes an XOR-transformed word to the output.
The allocation IDs deliberately sort in the opposite order from the two writable
Metal bindings, checking canonical output identity rather than binding order.

The Vulkan runner checks real translator reflection for XYZ byte strides
`4,20,60` and a 120-byte footprint, a scalar 4-byte bias, and the expected access
modes. It verifies that a 119-byte read/write view is refused during admission.
This does not deliberately submit out-of-bounds GPU work.

Both runners compile once per entry during a capture and reuse the registered
pipeline across the cases for that entry, while allocating fresh buffers and
commands per case. Vulkan retains its translated artifact/provider pipeline ID;
its existing execution machinery still creates Vulkan pipeline objects per
submission. This is not a Vulkan PSO-cache performance test, buffer-reuse test,
async ordering test or multi-pass command-buffer implementation.

## Run

Use the same capture and comparison commands as the [main instructions](README.md),
substituting `conformance/suite-v2.json` and fresh output paths:

```sh
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --suite conformance/suite-v2.json --output conformance/captures/vulkan-v2.json
python3 conformance/run_native.py --oracle conformance/native-oracle \
  --suite conformance/suite-v2.json --output-dir conformance/captures/native-v2 \
  --require-metal
python3 conformance/compare.py --suite conformance/suite-v2.json \
  --native conformance/captures/native-v2/native-metal.json \
  --vulkan conformance/captures/vulkan-v2.json
```

The updated CI executes both suites, preserves both reports and compares each
only to its matching native report. Native artifacts use `v1/` and `v2/`
subdirectories. If either suite has no eligible device, the comparison job is
skipped without claiming parity; capture failures still fail the native job.

## Current local verification

- 62 Rust tests and 44 Python tests passed, including independent golden
  calculations and malformed/missing second-writeback refusals. Python test
  captures are synthetic and do not constitute Metal evidence.
- All eight v2 Vulkan captures passed on Linux/Lavapipe and Windows/RTX 5060;
  both sets of host-visible results are identical.
- The v1 Vulkan capture still passes after the runner changes.
- Formatting, Clippy, rustdoc and Windows GNU build passed. Workflow YAML and
  embedded shell/Python syntax were checked locally.
- The new Swift branch has received static review but has not yet compiled or
  run remotely at this checkpoint. Native Metal v2 parity is pending.

The existing evidence distinction remains: the Swift runner reports complete
GPU buffer readback, while Vulkan reports writable views landed into initialized
host allocations. The latter cannot prove GPU-side canary or read-only safety.
