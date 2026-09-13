# Render capture: the native oracle's offscreen render path

This file describes the render side of `NativeOracle.swift` and
`crates/metal-api-native/src/render.rs`: what the reviewed fixture is, what a
render case looks like, what a render capture reports, and what still has to
change before a committed suite can run one. It is a **pending** path: no
committed suite declares render cases, both providers still refuse render
traces, and nothing here has run on an Apple GPU.

The design it implements is `research/docs/23` §1.2 (the milestone), §3 (the
contract), §5.1 (what the oracle needs) and §6 Steps 6–7 (where it lands).

## 1. The reviewed fixture

`conformance/shaders/render_offscreen_2x2.metal` is one MSL module with two
stage entries:

| Entry | Stage | What it does |
|---|---|---|
| `render_fullscreen_triangle` | vertex | position from `vertex_id` alone: `(-1,-1)`, `(3,-1)`, `(-1,3)` |
| `render_solid_rgba8` | fragment | writes `(64/255, 128/255, 192/255, 1)`, i.e. `40 80 c0 ff` in an 8-bit UNORM attachment |

Two rules in that file are deliberate and are what make the milestone
falsifiable:

* the triangle is the oversize one, so it covers every pixel centre of a 2x2
  viewport (the naive 2x pair leaves one centre uncovered);
* the colour constants are `byte / 255` and not round decimals. A half-integer
  tie such as one half times 255 resolves differently per driver: the render
  probe read it back as `0x80` on Lavapipe but `0x7f` on the NVIDIA driver and
  dzn (`research/docs/23` §3.5). The Rust rail's unit tests assert both rules,
  including that the fixture carries no `0.5` literal.

The module's identity is pinned **in code** in three places, because a matching
file name or an updated hash must not be enough to admit different source for
execution:

* `crates/metal-api-native/src/render.rs`: `REVIEWED_SOURCE`, `VERTEX_ENTRY`,
  `FRAGMENT_ENTRY`;
* `NativeOracle.swift`: `reviewedRenderModule()`, whose `RenderSourcePin` path
  and SHA-256 are checked against the file before any capture;
* `NativeOracle.swift`'s `--render-selftest` expectation `40 80 c0 ff` x4.

## 2. A render case

A suite may carry a top-level `render_cases` array next to `cases`. A render
case is not a compute case: its resource shape is one colour attachment, not a
buffer pool, and the compute case model (`buffers`, `expected_writebacks`,
`dispatches`) is what `conformance/compare.py` builds its plan from. Keeping the
render cases in their own array leaves that model untouched until the fixture
step below extends it.

```json
{
  "render_cases": [
    {
      "id": "offscreen_clear_triangle_2x2",
      "vertex_entry": "render_fullscreen_triangle",
      "fragment_entry": "render_solid_rgba8",
      "metal": {
        "path": "shaders/render_offscreen_2x2.metal",
        "sha256": "7430cd19a3497582618226066e95fb6f4ead9071f83b00c53398ccab8ba9d7de"
      },
      "vertices": 3,
      "viewport": [0, 0, 2, 2],
      "attachment": {
        "allocation": 900,
        "view": 910,
        "format": "rgba8_unorm",
        "width": 2,
        "height": 2,
        "load": "clear",
        "store": "store",
        "clear_hex": "fefefefe"
      },
      "expected_hex": "4080c0ff4080c0ff4080c0ff4080c0ff"
    }
  ]
}
```

The oracle validates this shape as a whitelist, not as a per-case table: the
first render increment has exactly one render shape (`research/docs/23` §1.2,
§3), so the shape *is* the review, and a fixture cannot widen it by renaming a
case. The admitted values are one `rgba8_unorm` `2x2` attachment, three drawn
vertices, the reviewed entry pair and module, `store` (a discarded attachment
could not be compared), and either

* `load: "clear"` with a four-byte `clear_hex`, or
* `load: "load"` with `initial_hex` covering the whole attachment.

Two falsifiability rules are enforced at load time, before a device exists:

* every texel of `expected_hex` has to be identical, so a partially covered
  attachment cannot be asserted as correct;
* the expectation has to differ from the clear colour and from the initial
  texels, so "the pass never ran" cannot satisfy it.

## 3. What a render capture reports

The observable is the attachment's tightly packed texel bytes, reported in the
same shape as every other case: one `writebacks` entry
(`allocation`, `view`, `offset: 0`, `bytes_hex`) and one `allocations` entry
(`allocation`, `bytes_hex`) with the observed bytes. A render case therefore
needs no second observation channel (`research/docs/23` §1.1); the comparator's
existing byte comparison is the whole assertion.

The Swift oracle is not a provider, so the `copy_in`/`copy_out` count contract
does not apply to it (`conformance/compare.py`, `validate_capture`). A provider
rail that reports a render case does have to decide those counters — the Vulkan
rail's shape is one `vkCmdCopyImageToBuffer`, i.e. `copy_out = 1`
(`research/docs/23` §3.5, §5.3).

## 4. Why no committed suite reaches it yet

`conformance/compare.py` has no attachment section in its plan model,
`examples/metal-smoke/src/bin/provider-capture.rs` has no render execution
shape, and both providers still declare `supports_render_passes = false`, so
admission refuses a render-bearing trace with `render_passes_unsupported`
before any execution code runs. Committing a suite that declares render cases
now would therefore make the oracle, the Vulkan rails and the native-provider
rails disagree about the same fixture.

Switching a suite on is a coordinated change. The places that have to move
together:

| # | Place | What changes |
|---|---|---|
| 1 | `crates/metal-api-*/src/provider.rs` | flip `supports_render_passes` (native: only after the check in §5 passes) |
| 2 | provider trace path | dispatch a render pass to the offscreen rail instead of the compute path |
| 3 | `examples/metal-smoke/src/bin/provider-capture.rs` | build the render trace, `validate_suite` case ids, `copy_in`/`copy_out` |
| 4 | `conformance/compare.py` | the attachment section in `_suite_plan`, and the render case's expected allocation |
| 5 | `conformance/suite-vN.json` | `cases` (the declaring pass) plus `render_cases` |
| 6 | `NativeOracle.swift` | the `loadSuite` `expectedIDs` arm for the new suite |
| 7 | `conformance/test_oracle_coverage.py` | extend the suite/case tables to cover `render_cases` |
| 8 | `.github/workflows/ci.yml` | name the suite on every rail and in the four version loops |
| 9 | `tools/lavapipe-smoke.sh` discovery | nothing to edit, but the suite must land only after 2–4 |

Steps 5–7 are the ones a reader can check from Linux; 1–4 need a device.

## 5. The one-device check

`native-oracle --render-selftest` runs the same reviewed fixture and the same
`runRenderCase` a suite would, without needing a suite. Run it from the
repository root (the reviewed module is resolved relative to the working
directory, and its hash is checked):

```sh
swiftc -swift-version 5 -warnings-as-errors -framework Foundation -framework Metal \
  -framework CoreGraphics -framework CryptoKit conformance/NativeOracle.swift -o /tmp/native-oracle
/tmp/native-oracle --render-selftest
```

It prints the case result as JSON and exits non-zero unless the readback matches
the reviewed expectation exactly. The evidence is the `bytes_hex`, which has to
be `4080c0ff` four times — the fragment's texel — and never the `fefefefe`
sentinel the pass started from.

This is the check the native provider's `supports_render_passes` flip condition
names (`crates/metal-api-native/src/native.rs`).

## 6. What is verified where

Verified on a Linux host, by `cargo test -p metal-api-native` and the
`aarch64-apple-darwin` cross-check in `tools/gates-local.sh`:

* the rail's plan: the reviewed (source, entry pair) allowlist, the format
  mapping, the extent cap, the viewport and draw shape, the load/store mapping,
  the `LoadOp::Load` agreement, the readback extent and row pitch;
* the clear-value decoding per format, including the B/G/R/A memory order of
  `bgra8_unorm` and the single-channel float format;
* that the rail's refusal slugs and classes are the ones core admission uses;
* that the reviewed fixture still carries the expected byte/255 constants, no
  half-integer tie, and both entry names;
* that the Swift oracle still accepts exactly the committed suites and pins
  exactly the committed sources (`conformance/test_oracle_coverage.py`).

**Only an Apple GPU can confirm**, and none of it has been checked yet:

* that `MTLRenderPipelineState` creation succeeds for the reviewed pair and
  `rgba8Unorm` attachment 0;
* that `loadAction = .clear` (and `.load` with `replace_region`-uploaded
  texels) behaves as the contract's `LoadOp` means;
* the readback bytes: that all four texels are `40 80 c0 ff`;
* that `getBytes` on a `usage = .renderTarget`, `storageMode = .shared` 2x2
  texture returns the tightly packed rows this report assumes;
* the oracle's `--render-selftest` output and, after the flip, a render case in
  a committed suite;
* the provider's render path end to end, including the writeback that would
  land the attachment bytes on the trace's view.
