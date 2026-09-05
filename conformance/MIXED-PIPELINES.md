# Mixed compute pipelines in one submission

`ComputeTrace` now uses schema 2 and an explicit `pipelines` table of
`CompiledComputePipeline` metadata. The old single `function` and
`pipeline_contract` envelope fields are removed. This is an intentional Rust
API/schema change in the experimental provider contract; capture JSON reports
and existing suite manifests still use schema 1 and their bytes are unchanged.

Every table entry must have a nonzero unique ID, the trace's device epoch,
valid function/contract metadata and at least one referencing pass. Each pass
is validated against the contract it selects. Before GPU work, the receiving
provider resolves all referenced pipeline IDs and compares every metadata
record with its registry. An invalid later pass or forged second pipeline is
rejected before any command in the sequence is submitted.

The same bounded resource-pool rules remain: at most eight serial passes,
every pass permutes all initialized views exactly once, fixed allocation/range/
initial bytes, one upload and one final writeback for every ever-written view.
New resources, aliases within a pass, concurrent work and async completion are
not introduced here.

## Execution

- Vulkan performs all pass-specific planning, footprint and limit checks before
  resource creation. Each pass has its own shader module, layouts, descriptor
  set, push-constant range and required local-size variants. They share one
  buffer upload, command buffer, queue submit, fence and final readback.
- RAII pipeline-object owners are explicitly retained if submitted work has
  unknown completion, including on unwind, alongside the existing buffer and
  device retention policy.
- Rust Metal retains the complete vector of selected PSOs. Each serial encoder
  selects its pass's PSO and directly binds the shared tracked resource pool.
- Swift independently compiles the reviewed programs, prechecks each selected
  PSO's limits, and switches pipelines between encoders in one command buffer.

## Suite v5

`suite-v5.json` includes a case-level `programs` table and each dispatch selects
its zero-based `program` index. Legacy v1-v4 reject these additional program
fields. The capture implementations allow exactly two reviewed programs in v5,
selected alternately over 2, 3 or 8 passes:

1. `transform_3d`: add scalar bias in place, emit an XOR output.
2. `mix_3d`: XOR input with the bias, multiply by three and add one in place;
   emit a different addition-based output.

The outputs alternate roles between passes as in v4. Uint32 operations wrap
modulo 2^32. Independent CPU expectations and counterexamples establish that
using only the first shader would produce incorrect bytes. No expected bytes
are substituted for observed capture results.

The two GPU-tested shaders currently share binding/access layout and grid.
This increment proves changing actual shader programs with data dependencies;
different-layout behavior is unit-tested in the core and Vulkan preflight,
not yet a measured native/Vulkan GPU conformance claim. Native source admission
expands from three to four exact reviewed MSL files, not arbitrary Metal source.

```sh
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --suite conformance/suite-v5.json --output conformance/captures/vulkan-v5.json
# macOS:
python3 conformance/run_native.py --oracle conformance/native-oracle \
  --suite conformance/suite-v5.json --output-dir conformance/captures/native-v5 \
  --require-metal
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --backend native-metal-provider --suite conformance/suite-v5.json \
  --output conformance/captures/rust-metal-v5.json
python3 conformance/compare.py --suite conformance/suite-v5.json \
  --native conformance/captures/native-v5/native-metal.json \
  --vulkan conformance/captures/vulkan-v5.json \
  --metal-provider conformance/captures/rust-metal-v5.json
```

## Local verification

- 99 Rust tests passed: core 57, native contract/platform 5, Vulkan 29, capture 8.
- 66 Python tests passed, including independent two-shader CPU expectations,
  rejecting missing/unknown/unused program selections and single-shader outputs.
- Vulkan v5 passed on Linux/Lavapipe and Windows/RTX 5060 with identical final
  result arrays. Both reject a forged or unknown second pipeline before queue
  submission. The prior v4 GPU regression and original provider smoke passed.
- Formatting, Clippy, rustdoc, Windows GNU build, optional reims check and
  macOS ARM64 typecheck/Clippy passed. The dependency's existing block 0.1.6
  future-incompatibility advisory remains.
- The new Swift/native Metal sequence still needs macOS CI compilation and
  GPU execution. Earlier v1-v4 captures do not establish v5 parity.

No production reims/guest/display behavior, arbitrary shader support or full
Metal conformance is inferred. Swift observes full GPU buffers; the two Rust
providers report view writes landed into host allocations.
