# Render capture: the native oracle's offscreen render path

This file describes the render side of `NativeOracle.swift` and
`crates/metal-api-native/src/render.rs`: what the reviewed fixture is, what a
render case looks like, what a render capture reports, and which rails report
one today. `conformance/suite-v13.json` is the first committed suite that
declares render cases, `conformance/compare.py` has the matching attachment
section, and the Vulkan trace rail executes the case end to end. Two things are
still **pending**: the two object-API rails have no render command encoder, and
no run on an Apple GPU has been observed yet. The macOS job does run the
one-device check itself — `--render-selftest` in §5 — but its result is still an
outstanding observation, not a recorded one.

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

The module's identity is pinned **in code** in four places, because a matching
file name or an updated hash must not be enough to admit different source for
execution:

* `crates/metal-api-native/src/render.rs`: `REVIEWED_SOURCE`, `VERTEX_ENTRY`,
  `FRAGMENT_ENTRY`;
* `NativeOracle.swift`: `reviewedRenderModule()`, whose `RenderSourcePin` path
  and SHA-256 are checked against the file before any capture;
* `NativeOracle.swift`'s `--render-selftest` expectation `40 80 c0 ff` x4.
* `examples/metal-smoke/src/bin/provider-capture.rs`: the Vulkan rail's half —
  the two reviewed SPIR-V stage modules (`RENDER_VERTEX_SPV`,
  `RENDER_FRAGMENT_SPV`) and their entry names (`vertex_main`,
  `fragment_main`). The entries differ from the MSL ones because the reviewed
  SPIR-V sources declare them that way and core refuses a render contract whose
  two entries share a name; the suite still pins the MSL module, i.e. the
  identity both canonical rails compile.

## 2. A render case

A suite may carry a top-level `render_cases` array next to `cases`. A render
case is not a compute case: its resource shape is one colour attachment, not a
buffer pool, and the compute case model (`buffers`, `expected_writebacks`,
`dispatches`) is what `conformance/compare.py` builds its plan from. Keeping the
render cases in their own array leaves that model untouched.

A render attachment does not declare storage: it references an existing
resource by identity and restates the shape the pass draws into, exactly as
`metal_api_core::provider::RenderAttachment` does. The pass that declares that
view is a compute case of the same suite, named by `declaring_case`; the
structural rules (`validate_serial_buffer_reuse`) resolve the attachment
against that case's own view declarations, so the attachment has to name one of
its views, with the same allocation and a byte range agreeing with the restated
extent, and the declaring pass has to be read-only for it — a compute pass that
*wrote* the view the render pass stores would make the two writers' order
inexpressible (`research/docs/23` §3.6).

```json
{
  "render_cases": [
    {
      "id": "offscreen_triangle_clear_2x2",
      "declaring_case": "render_declaring_copy_word",
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
      "expected_hex": "4080c0ff4080c0ff4080c0ff4080c0ff",
      "capture_rails": ["vulkan"]
    }
  ]
}
```

Field by field, against the core values the rail builds:

| Suite field | Core value | Rule |
|---|---|---|
| `attachment.allocation` / `.view` | `RenderAttachment::allocation_id` / `view_id` | nonzero; must be declared by `declaring_case` |
| `attachment.format` | `RenderAttachment::format` | `rgba8_unorm` only |
| `attachment.width` / `.height` | `RenderAttachment::width` / `height` | `2` x `2` only |
| `attachment.load` + `clear_hex` | `LoadOp::Clear(ClearColor)` | four bytes in memory order, different from the expected texel; `load: "load"` with `initial_hex` covering the whole attachment is admitted by the schema but refused by the Vulkan rail, which has no upload path yet |
| `attachment.store` | `StoreOp::Store` | `store` only; a discarded attachment could not be compared |
| `vertices` / `viewport` | `RenderPassDescriptor::vertices` / `viewport` | three vertices and a viewport covering the attachment |
| `expected_hex` | `RenderAttachment::expected_bytes` | every texel identical, and different from the value the pass started from |
| `declaring_case` | — | a `cases` entry: one submission, one dispatch, declaring the attachment view read-only |
| `capture_rails` | — | the capture backends required to report the case; see §4 |

The oracle validates this shape as a whitelist, not as a per-case table: the
first render increment has exactly one render shape (`research/docs/23` §1.2,
§3), so the shape *is* the review, and a fixture cannot widen it by renaming a
case. `conformance/compare.py` repeats the same rules when it builds the render
plan, so a suite the oracle would refuse cannot pass the comparator either. The
admitted values are one `rgba8_unorm` `2x2` attachment, three drawn vertices,
the reviewed entry pair and module, `store`, and either

* `load: "clear"` with a four-byte `clear_hex`, or
* `load: "load"` with `initial_hex` covering the whole attachment.

Two falsifiability rules are enforced at load time, before a device exists:

* every texel of `expected_hex` has to be identical, so a partially covered
  attachment cannot be asserted as correct;
* the expectation has to differ from the clear colour and from the initial
  texels, so "the pass never ran" cannot satisfy it.

## 3. What a render capture reports

The observable is the attachment's own allocation, reported in the same shape as
every other case: one `writebacks` entry (`allocation`, `view`, the declaring
view's `offset`, `bytes_hex`) and one `allocations` entry (`allocation`,
`bytes_hex`) holding the observed image. A render case therefore needs no second
observation channel (`research/docs/23` §1.1); the comparator's per-texel byte
comparison is the whole assertion. The Swift oracle's `runRenderCase` already
reports that shape, and the Vulkan trace rail reports it from the writeback the
render rail pushes for the view the trace declares
(`VulkanComputeProvider::execute_render_passes`).

The suite's `cases` entry is the *declaring* pass, so one render submission
carries two observations' worth of work: the declaring case reports its own
buffer landing, and the render case reports the attachment and nothing else.
`conformance/compare.py` asserts exactly that split — the attachment's writeback
identity set has to be the attachment's own, and the declaring case's has to be
its buffers — which is how an attachment and a buffer writeback are kept from
standing in for each other.

The count contract is derived, not stored. The declaring pass copies its touched
allocations in and its written ones out, and the attachment allocation is one of
the written ones: its single `vkCmdCopyImageToBuffer` readback *is* that
allocation's copy-out, replacing the device-buffer readback the compute pool
would otherwise do (`research/docs/23` §3.5, §5.3). So a render case expects one
`copy_in` per touched allocation and one `copy_out` per written allocation, the
same rule a compute case follows; the Swift oracle reports bytes without
counters, so the contract does not apply to `native-metal`
(`conformance/compare.py`, `validate_capture`).

## 4. Which rails report a render case

The first render increment has exactly one executable rail: the Vulkan trace
rail. The two object-API rails have no render command encoder
(`crates/metal-api-core/src/provider_api.rs` exposes
`compute_command_encoder` only) and the native provider still declares
`supports_render_passes = false`, so a render-bearing trace is refused there
with `render_passes_unsupported` or cannot be expressed at all.

A render case therefore declares the capture backends that owe the attachment
in its `capture_rails` marker:

* a rail named by the marker has to report the case — `compare.py` refuses a
  capture from it that omits the case;
* a rail not named by it has to omit it — a capture that reports the case
  anyway is refused, because that observation did not come from a rail that can
  produce one.

`suite-v13.json` marks `["vulkan"]`, so the local and CI Vulkan rails report the
attachment while `--api objects`, `--api objects --async` and every macOS rail
report the declaring case only. `conformance/test_oracle_coverage.py` checks the
marker against `compare.py`'s backend vocabulary and keeps the marked suite out
of the four CI rails and version loops until the macOS half exists.

The macOS half is still the pending step, but its first part is now wired: the
one-device check runs in CI (§5). What is deliberately still **not** wired is
the suite itself:

* `NativeOracle.swift` accepts `compute-buffer-v13` and its render capture path
  reports the attachment in exactly this shape. The self-test (§5) has since run
  on the CI runner's Apple Paravirtual device and passed
  (`render_selftest: PASS (4080c0ff4080c0ff4080c0ff4080c0ff)`, run 34774478149), so
  the single-case shape is evidenced; the **suite** path
  (`run_native.py --suite conformance/suite-v13.json`) is what would put that
  byte comparison into the four-rail capture matrix;
* the Rust native provider needs its own `supports_render_passes` flip, whose
  condition is the same §5 check (`crates/metal-api-native/src/native.rs`);
* `.github/workflows/ci.yml` would then need `13` in the four version loops, the
  explicit suite lines, the native status list and the parity lines.

Until that lands, the suite is run by the Vulkan rails locally
(`tools/lavapipe-smoke.sh` discovers it from `conformance/suite*.json`) and by
the trace and object-API rails in CI, which do name its suite file; the macOS
and Rust-native rails deliberately do not, because neither reports an
attachment yet. The coverage check keeps that split explicit rather than
silent, and `conformance/test_suite_v13.py` holds the schema, the plan and the
refusals.

## 5. The one-device check

`native-oracle --render-selftest` runs the same reviewed fixture and the same
`runRenderCase` a suite would, without needing a suite. Run it from
`conformance/` — the reviewed module (`shaders/render_offscreen_2x2.metal`) is
resolved relative to the working directory, exactly like a suite resolves its
pins against the suite file's directory, and its hash is checked:

```sh
swiftc -swift-version 5 -warnings-as-errors -framework Foundation -framework Metal \
  -framework CoreGraphics -framework CryptoKit conformance/NativeOracle.swift -o /tmp/native-oracle
(cd conformance && /tmp/native-oracle --render-selftest)
```

It prints the case result as JSON and exits non-zero unless the readback matches
the reviewed expectation exactly. The evidence is the `bytes_hex`, which has to
be `4080c0ff` four times — the fragment's texel — and never the `fefefefe`
sentinel the pass started from.

CI runs this check in `native-oracle-build`, on that job's own
`/tmp/native-oracle`, after the compute capture steps so a render regression
cannot discard their evidence. The step decides from the oracle's `--probe` —
the same eligibility `renderSelfTest` itself enforces — so a host whose device
the oracle would refuse prints
`render selftest: SKIP (no Metal device)` and passes the job, while a probe that
cannot be read (`probe unreadable`, `unexpected probe kind`) or exits non-zero
fails it. On an eligible host the report is written to
`/tmp/native-evidence/render-selftest.json`, the oracle's diagnostics and the
step's log to `/tmp/native-evidence/render-selftest.log`, and the job fails
unless both the exit status and the report's single writeback and single
allocation read `4080c0ff` four times; a successful log ends with
`render_selftest: PASS`. The job's existing `native-evidence` artifact upload
archives those files on every run, including a SKIP.

The flip therefore needs one CI run whose log carries both `4080c0ff` four
times and `render_selftest: PASS`. A green job whose log says `SKIP` is not that
evidence: it reports that the runner had no eligible device, not that the
reviewed path ran.

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
  exactly the committed sources (`conformance/test_oracle_coverage.py`);
* that `suite-v13.json`'s attachment section is the reviewed shape, that the
  Vulkan capture pins the reviewed SPIR-V stage pair and entries, and that a
  tampered attachment byte, a dropped texel, a buffer writeback standing in for
  the attachment (in either direction) and a wrong copy count are all refused
  (`conformance/test_suite_v13.py`, run by `python3 -m unittest discover -s
  conformance`);
* that the render case actually executes on Lavapipe: `provider-capture --suite
  conformance/suite-v13.json` reports
  `4080c0ff4080c0ff4080c0ff4080c0ff`, and `compare.py --check` accepts it.

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

`native-oracle-build` now runs the first of those bullets on every eligible
runner (§5). Until a run reports `render_selftest: PASS`, however, none of the
six has an Apple observation behind it.
