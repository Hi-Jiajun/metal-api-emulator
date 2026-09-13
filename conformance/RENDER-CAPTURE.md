# Render capture: the native oracle's offscreen render path

This file describes the render side of `NativeOracle.swift` and
`crates/metal-api-native/src/render.rs`: what the reviewed fixture is, what a
render case looks like, what a render capture reports, and which rails report
one today. `conformance/suite-v13.json` is the first committed suite that
declares render cases, `conformance/compare.py` has the matching attachment
section, and the case is named by every rail that owns a render execution path:
the Vulkan trace rail, the native provider's trace rail and the Swift oracle's
suite path. One thing is still **pending**: the two object-API rails have no
render command encoder, so they run a render-bearing suite and report its
declaring pass only. The one-device check is no longer pending — `--render-selftest`
in §5 ran on an Apple Paravirtual device in CI run `34774478149` and read the
attachment back as `4080c0ff` four times — so the native provider declares
render support and its trace rail plans, encodes and reports the same reviewed
fixture. What no Apple GPU has run yet is the Rust provider's own encoder path
and the oracle's suite path; CI now asks both for it (§6).

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
* `examples/metal-smoke/src/bin/provider-capture.rs`: each trace rail's half.
  The Vulkan rail pins the two reviewed SPIR-V stage modules
  (`RENDER_VERTEX_SPV`, `RENDER_FRAGMENT_SPV`) and their entry names
  (`vertex_main`, `fragment_main`); the native rail pins the reviewed MSL entry
  pair (`RENDER_MSL_VERTEX_ENTRY`, `RENDER_MSL_FRAGMENT_ENTRY`), which the
  rail's registration re-checks. The entries differ from the MSL ones because
  the reviewed SPIR-V sources declare them that way and core refuses a render
  contract whose two entries share a name; the suite still pins the MSL module,
  i.e. the identity both canonical rails compile.

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

The first render increment has three executable rails: the Vulkan trace rail,
the native trace rail (`crates/metal-api-native/src/render.rs`, wired into
`NativeMetalProvider::submit` in Step 7) and the Swift oracle's suite path
(`NativeOracle.swift::capture`, which runs `suite.renderCases` after the compute
cases exactly as it runs them for its own `--render-selftest`). The native
provider declares `supports_render_passes = true` with the rail's own limits —
one colour attachment, 2x2, the three admitted formats — because the flip
condition below is met; before that flip it refused a render-bearing trace with
`render_passes_unsupported`. On the Rust side both trace rails reach the case
through the same `provider-capture` code path, and each registers the reviewed
pipeline on its own concrete context: the Vulkan rail's SPIR-V pair
(`VulkanComputeProvider::register_render_pipeline`) and the native rail's
reviewed MSL module (`NativeMetalProvider::register_render_pipeline`). The two
object-API rails still have no render command encoder
(`crates/metal-api-core/src/provider_api.rs` exposes `compute_command_encoder`
only), so a render case cannot be expressed there at all; they run the suite and
report its declaring pass, and the marker does not name them.

A render case therefore declares the capture backends that owe the attachment
in its `capture_rails` marker:

* a rail named by the marker has to report the case — `compare.py` refuses a
  capture from it that omits the case;
* a rail not named by it has to omit it — a capture that reports the case
  anyway is refused, because that observation did not come from a rail that can
  produce one.

`suite-v13.json` marks `["vulkan", "native-metal", "native-metal-provider"]`, so
the attachment is reported by the Vulkan trace rail, the native provider's trace
rail and the Swift oracle, while `--api objects` and `--api objects --async`
report the declaring case only. `conformance/test_oracle_coverage.py` checks the marker
against `compare.py`'s backend vocabulary, refuses a marker that names an
object-API rail (they cannot report the attachment yet), and requires every
committed suite to be named on every CI rail and in every object-API version
loop.

The one-device check (§5) and the suite path are both wired now:

* `NativeOracle.swift` accepts `compute-buffer-v13`, and its render capture path
  reports the attachment in exactly this shape. The self-test (§5) ran on CI's
  Apple Paravirtual device and passed
  (`render_selftest: PASS (4080c0ff4080c0ff4080c0ff4080c0ff)`, run 34774478149);
  the suite path is now named on the macOS rail
  (`run_native.py --suite conformance/suite-v13.json`), so the same byte
  comparison is asked for as part of the capture matrix rather than beside it;
* the Rust native provider's `supports_render_passes` flip has landed on the
  same §5 evidence (`crates/metal-api-native/src/native.rs`), and its trace rail
  plans, encodes and reports every render pass the suite declares. The capture
  tool registers the reviewed MSL pipeline on the provider and runs the render
  case through the same trace path as the Vulkan rail, so the rail is admitted
  *and* asked for the observation;
* `.github/workflows/ci.yml` carries `13` in the four version loops, the
  explicit suite lines, the native status list and the parity lines.

What is still pending is the Apple-side execution itself, not the wiring: the
same CI job's `native-oracle-build` now runs the oracle's v13 suite capture and
the Rust provider's v13 capture, and either has to report
`4080c0ff4080c0ff4080c0ff4080c0ff` or fail the job. Local evidence for the
Vulkan rail stays with `tools/lavapipe-smoke.sh`, which discovers every suite
from `conformance/suite*.json`. `conformance/test_suite_v13.py` holds the
schema, the plan and the refusals; `conformance/test_oracle_coverage.py` holds
the marker, the case-id tables and the CI wiring.

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

The flip therefore needed one CI run whose log carries both `4080c0ff` four
times and `render_selftest: PASS`. A green job whose log says `SKIP` is not that
evidence: it reports that the runner had no eligible device, not that the
reviewed path ran.

That run is `34774478149` (`native-oracle-build`, commit `fb4f8da`): on an Apple
Paravirtual device it reported `bytes_hex =
4080c0ff4080c0ff4080c0ff4080c0ff` and `render_selftest: PASS`. It is the
evidence the native provider's render bits point at
(`crates/metal-api-native/src/render.rs::capability_bits`,
`crates/metal-api-native/src/native.rs`), and it is what checks the Swift
oracle's half of Step 6: the reviewed MSL pin matched, the oversize triangle
covered every texel of the 2x2 attachment, and the readback was the fragment's
texel rather than the clear sentinel.

## 6. What is verified where

Verified on a Linux host, by `cargo test -p metal-api-native` and the
`aarch64-apple-darwin` cross-check in `tools/gates-local.sh`:

* the rail's plan: the reviewed (source, entry pair) allowlist, the format
  mapping, the extent cap, the viewport and draw shape, the load/store mapping,
  the `LoadOp::Load` agreement, the readback extent and row pitch;
* the clear-value decoding per format, including the B/G/R/A memory order of
  `bgra8_unorm` and the single-channel float format;
* that the rail's refusal slugs and classes are the ones core admission uses,
  including the trace path's own refusals (`attachment_load_op_unsupported` for
  a `Load` the trace cannot carry, `render_attachment_landing_unsupported` for
  an attachment no declared view covers, `render_pass_order_unsupported` for a
  compute pass that would read after a render store);
* the trace path's device-free plan (`render::plan_trace`) and writeback merge
  (`render::merge_writebacks`): the planned extent and readback length, the
  attachment's landing view, the canonical one-writeback-per-view order, and
  that the render bytes replace the pre-render bytes of the attachment's own
  view;
* that the declared render bits are the rail's limits and that core admission
  admits exactly the trace the rail plans, while the pre-flip snapshot refuses
  it with `render_passes_unsupported`;
* that the reviewed fixture still carries the expected byte/255 constants, no
  half-integer tie, and both entry names;
* that both trace rails register the reviewed pipeline they own and refuse the
  other rail's identity: the native registration re-runs the reviewed MSL entry
  pair and returns the trace-table entry the render pass names, so
  `provider-capture` can put the same case through either rail without letting
  a caller-supplied entry widen the allowlist;
* that the Swift oracle still accepts exactly the committed suites and pins
  exactly the committed sources (`conformance/test_oracle_coverage.py`);
* that the marker, the two case-id tables and the CI wiring agree:
  `conformance/test_oracle_coverage.py` requires every committed suite on all
  four named CI rails and in all four object-API version loops, and refuses a
  marker that names an object-API rail, which has no render command encoder;
* that `suite-v13.json`'s attachment section is the reviewed shape, that the
  capture pins the reviewed SPIR-V stage pair and entries, that every rail the
  marker names reports the attachment while the object-API rails omit it, and
  that a tampered attachment byte, a dropped texel, a buffer writeback standing
  in for the attachment (in either direction) and a wrong copy count are all
  refused (`conformance/test_suite_v13.py`, run by `python3 -m unittest
  discover -s conformance`);
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
* a render case from the native rail in a committed suite: the oracle's
  `--render-selftest` output exists (§5), and CI asks for the suite capture
  (`run_native.py --suite conformance/suite-v13.json`) plus the Rust provider's
  v13 capture. Both ran on CI run `34781060564`: the oracle reported the v13
  attachment, and the Rust provider printed
  `render case completed: offscreen_triangle_clear_2x2 attachment=4080c0ff...`;
* the Rust provider's render path end to end — `render::plan_trace` plus the
  encoder body, and the writeback that lands the attachment bytes on the
  trace's view. Its refusal and planning halves are host-verified; the encoder
  half has only the oracle's observation behind it, not its own.

`native-oracle-build` runs the `--render-selftest` half of those bullets on
every eligible runner (§5). On the Paravirtual device of run `34774478149` that
check built the reviewed two-stage pipeline state, cleared the 2x2 `rgba8Unorm`
attachment and read it back through `getBytes` as `4080c0ff` four times — the
bullets that name the pipeline state, `loadAction = .clear`, the texel bytes and
the tightly packed readback, on the oracle's own path. The `.load` branch is what
`--present-selftest` exercises (§7). Since the capability flip, the same runner
also captures committed suites through the Rust provider's own encoder: run
`34781060564` captured v13 and run `34782615760` captured v14 with
`render case completed: present_triangle_clear_2x2 attachment=4080c0ff...`.

## 7. The present milestone (suite-v14)

`suite-v14.json` is v13's 2x2 `rgba8_unorm` render case plus a `present` section
(`research/docs/24` §3.1, §6 Step 3): the render pass's own attachment is a
provider-owned present target that is pre-seeded with the `efefefef` sentinel,
is acquired once, is rendered into, and is made readable by one explicit
`COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL` transition followed by the
same `vkCmdCopyImageToBuffer` readback every other case uses. A successful case
reports the target through the existing `writebacks`/`allocations` shape and adds
`"present": {"acquire": 1, "present": 1}`.

What each observable proves, precisely: the case's `load: "clear"` overwrites
the pre-seeded sentinel inside the same submission, so the target's bytes prove
that the clear and the draw ran (the clear colour `fefefefe` differs from the
expected texel) and the reported `acquire`/`present` counts prove that the
present action was taken; the sentinel is not itself a surviving observable in
this shape — it pins the target's initial state and the compare rule refuses a
sentinel equal to the expectation. The `--present-selftest` shape uses
`load: "load"`, where the sentinel *is* the pass's starting content and is
therefore directly observable if the draw does not run.

The marker rule is the same one §4 describes, with one extra consequence: the
Swift oracle does not report provider counters (`research/docs/24` §5.1), so a
present-bearing case cannot be marked for `native-metal`. Suite-v14 marks
`vulkan` and `native-metal-provider`; the object-API rails skip it exactly as
they skip v13's render case. The Apple half of the evidence is the oracle's
`--present-selftest` (§5): it presets the sentinel, runs the reviewed render
equivalent and fails if the readback is the sentinel instead of `4080c0ff` x4.

What the comparator enforces for a marked rail (`conformance/compare.py`,
`conformance/test_suite_v14.py`):

* the suite's `present` object is a whitelist (`mode = "fifo"`, `image_count = 1`,
  `acquire = present = 1`), and `initial_hex`, when present, is four bytes that
  differ from the expected texel — the same falsifiability rule the clear value
  and the loaded initial texels already follow;
* a rail the marker names has to report the counts the suite declares, and a rail
  the marker does not name has to leave them out; a suite with no `present`
  section may not see the key at all;
* the counts are additive observations, not substitutes: the attachment bytes,
  the `copy_in`/`copy_out` pair and the v9 `group_counts` rules are unchanged;
* the target's bytes still travel the attachment's writeback, so a buffer
  writeback standing in for it, a dropped texel or a tampered byte is refused as
  it is for v13.

Evidence. Lavapipe (`tools/lavapipe-smoke.sh`) runs all three Vulkan shapes for
v14 and every `compare.py --check` passes; CI run `34782615760` is green with
five-rail parity for v14, the Rust native provider executing the presenting case
on an Apple Paravirtual device, and `--present-selftest` passing; the RTX 5060
run is archived in `evidence/windows-rtx5060-v14-1f8adc1-2026-09-14/` (direct
capture: `present = {"acquire": 1, "present": 1}`, bytes
`4080c0ff4080c0ff4080c0ff4080c0ff`; object and async-object captures pass because
the marker does not name them). The object-API present action is still open
(`research/docs/24` §6 Step 5), so the object rails continue to skip the case.
