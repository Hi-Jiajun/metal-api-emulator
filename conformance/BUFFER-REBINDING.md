# Serial buffer-role rebinding

The providers now support rearranging the same initialized view pool across
bindings between serial compute passes. A command still uses one pipeline,
1..8 passes, one initial upload per view and one final readback per written view.

## Resource identity and access

The first pass declares every logical view and its initial contents. Later
passes must contain exactly that view set, once each. A view's allocation,
range, source kind and initial bytes stay unchanged; its Metal binding and
access role may change to match the pipeline's reflected binding contract.
Rebinding does not insert another CPU upload.

`ComputeTrace::serial_resources()` returns the first-pass ordered pool with
access aggregated across all passes. A resource first used as read-only but
written later is included in final writebacks. Every ever-written view is
returned once, sorted by allocation/view identity. This does not permit two
bindings to alias one view within a pass, changing a view's range, introducing
new resources mid-command, concurrent passes or mixed pipelines.

Vulkan encodes a distinct descriptor set for each pass; updating one shared
set would incorrectly apply the last binding map to earlier dispatches.
Buffers are keyed by stable pool identity. The existing compute barriers order
all earlier writes before subsequent reads/writes, and the command submits
once. Both Metal paths bind the original resource selected by `view_id`,
reusing the same tracked buffers across encoders. Typed failures, retained
pending resources and owner/epoch checks remain unchanged.

## Suite v4

`conformance/suite-v4.json` adds four cases; earlier suite files are unchanged.
The report schema remains 1. Each dispatch's `bindings` array lists logical
view IDs in the case's binding-slot order, and the capture runners accept only
the reviewed alternating permutations.

- `transform_pingpong_two`, `transform_pingpong_three`, and
  `transform_pingpong_eight` use the existing sparse 3D read/write shader for
  2, 3 or 8 passes. Slots 0 and 5 exchange their backing views after each pass;
  slot 2 retains the read-only bias. Thus the previous XOR output becomes the
  next pass's in-place input, checking ordering and real descriptor changes.
- `copy_pingpong` copies A to B, then binds B as input and A as output. A starts
  read-only but must also be returned because the second pass writes it.

The CPU expected-value calculations process each dispatch in order by view
identity. For the transform cases, using one fixed binding table, overwriting
all descriptor sets with the final map, or resetting initial bytes between
passes yields a different result.

```sh
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --suite conformance/suite-v4.json --output conformance/captures/vulkan-v4.json
# macOS:
python3 conformance/run_native.py --oracle conformance/native-oracle \
  --suite conformance/suite-v4.json --output-dir conformance/captures/native-v4 \
  --require-metal
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --backend native-metal-provider --suite conformance/suite-v4.json \
  --output conformance/captures/rust-metal-v4.json
python3 conformance/compare.py --suite conformance/suite-v4.json \
  --native conformance/captures/native-v4/native-metal.json \
  --vulkan conformance/captures/vulkan-v4.json \
  --metal-provider conformance/captures/rust-metal-v4.json
```

## Local verification

- 91 Rust tests passed: core 54, native contract/platform 5, Vulkan 25, capture 7.
- 61 Python tests passed, including independent per-dispatch CPU simulation,
  unknown/duplicate bindings, short-resource misuse and missing later-written
  outputs. Synthetic reports are unit fixtures, not Metal evidence.
- The four Vulkan v4 cases passed on Linux/Lavapipe and Windows/RTX 5060;
  all host-visible output arrays match exactly. The sequential v3 capture
  still matches the archived Swift reference and the original provider smoke
  passed its four cases and refusal checks.
- Formatting, Clippy, rustdoc and Windows GNU build passed. Native Metal ARM64
  typecheck/Clippy passed; the changed Swift/native GPU paths still require a
  new macOS CI run. No new native v4 result is claimed at this local checkpoint.

As before, Swift reports full GPU buffer contents while the providers report
view writebacks landed into initialized host allocations. No general buffer
aliasing, async scheduling, production reims routing, guest memory or display
behavior is established by this increment.
