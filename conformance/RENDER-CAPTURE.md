# Render capture: the native oracle's offscreen render path

This file describes the render side of `NativeOracle.swift` and
`crates/metal-api-native/src/render.rs`: what the reviewed fixture is, what a
render case looks like, what a render capture reports, and which rails report
one today. `conformance/suite-v13.json` is the first committed suite that
declares render cases, `conformance/compare.py` has the matching attachment
section, and the case is named by every rail that owns a render execution path:
the Vulkan trace and object rails, the native provider's trace and object rails,
and the Swift oracle's suite path. The one-device check ran on an Apple
Paravirtual device in CI run `34774478149` (`--render-selftest`, §5), and both
Apple-side suite captures have since run too: the oracle and the Rust provider
each read `4080c0ff` four times back for v13 (run `34781060564`) and v14
(run `34782615760`), and the Rust provider's own encoder path is what those
captures exercise (§6).

The design it implements is `research/docs/23` §1.2 (the milestone), §3 (the
contract), §5.1 (what the oracle needs) and §6 Steps 6–7 (where it lands).

## 1. The reviewed fixture

`conformance/shaders/render_offscreen_2x2.metal` is one MSL module with two
stage entries:

| Entry | Stage | What it does |
|---|---|---|
| `render_fullscreen_triangle` | vertex | position from `vertex_id` alone: `(-1,-1)`, `(3,-1)`, `(-1,3)` |
| `render_solid_rgba8` | fragment | writes `(64/255, 128/255, 192/255, 1)`, i.e. `40 80 c0 ff` in an 8-bit UNORM attachment |

`conformance/shaders/quad_indexed_2x2.metal` is the second reviewed module, the
one the vertex-input increment adds (`research/docs/23` §3.3), with the same
fragment entry and a vertex stage that reads a caller-held stream:

| Entry | Stage | What it does |
|---|---|---|
| `render_quad_vertex` | vertex | `float2 position [[attribute(0)]]` from `[[stage_in]]`, i.e. from the pass's bound stream, returned as `float4(position, 0, 1)` |
| `render_solid_rgba8` | fragment | the same fixed colour texel as above |

The indexed fixture's stream is four NDC corners — `(-1,-1)`, `(1,-1)`,
`(-1,1)`, `(1,1)`, `float32x2` little-endian, 32 bytes — and its index buffer is
six `uint16` values `(0,1,2)` and `(2,1,3)`, 12 bytes. The two triangles cover
the whole square, so every pixel centre of the 2x2 viewport is covered just as
the oversize triangle covers it: the second fixture changes where the positions
come from, not what a capture can conclude from the texels.

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

* `crates/metal-api-native/src/render.rs`: `REVIEWED_MODULES`, i.e. the
  `vertex_id` module (`REVIEWED_SOURCE`, `VERTEX_ENTRY`, `FRAGMENT_ENTRY`) and
  the indexed one (`REVIEWED_VERTEX_SOURCE`, `QUAD_VERTEX_ENTRY`,
  `FRAGMENT_ENTRY`). A pipeline's `VertexLayout` selects which of the two may
  compile, so a stream-bearing layout cannot reach the `vertex_id` source and a
  `VertexLayout::None` pipeline cannot reach the indexed one;
* `NativeOracle.swift`: `reviewedRenderModule()` and `reviewedIndexedModule()`,
  whose `RenderSourcePin` paths and SHA-256s are checked against the files
  before any capture, plus the vertex layout `reviewedIndexedModule()` pins;
* `NativeOracle.swift`'s `--render-selftest` and `--vertex-selftest`
  expectations `40 80 c0 ff` x4.
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
| `attachment.load` + `clear_hex` | `LoadOp::Clear(ClearColor)` | four bytes in memory order, different from the expected texel; `load: "load"` with `initial_hex` covering the whole attachment is admitted and uploaded into the attachment before the pass opens (§9) |
| `attachment.store` | `StoreOp::Store` | `store` only; a discarded attachment could not be compared |
| `vertices` / `viewport` | `RenderPassDescriptor::vertices` / `viewport` | three vertices and a viewport covering the attachment, or six *indices* for the indexed shape below |
| `vertex_layout` | `RenderPipelineContract::vertex_layout` | optional; when present it has to equal the reviewed indexed layout: one stream, stride 8, one `float32x2` attribute at location 0, offset 0 |
| `vertex_buffers[]` | `RenderPassDescriptor::vertex_buffers` | optional; one read-only `BufferView` per reviewed stream, in binding order, carrying its own bytes: `allocation`, `view`, `offset`, `length`, `initial_hex` (exactly `length` bytes) |
| `indices` | `RenderPassDescriptor::indices` | optional; the index view (`view.view` = the same five fields) plus `uint16`/`uint32` |
| `expected_hex` | `RenderAttachment::expected_bytes` | every texel identical, and different from the value the pass started from |
| `declaring_case` | — | a `cases` entry: one submission, one dispatch, declaring the attachment view read-only |
| `capture_rails` | — | the capture backends required to report the case; see §4 |

The oracle's whitelist admits two shapes: the `vertex_id` triangle, where all
three vertex-input fields are absent, and the indexed quad, where all three are
present. The contract itself also admits a non-indexed draw out of streams
(`RenderPassDescriptor::validate`), which `render.rs::plan_vertex_input` plans;
the oracle's suite path stays with the reviewed indexed fixture, the same way it
reviews one render shape per increment.

A render input declares its own source (`research/docs/23` §3.6): each
`vertex_buffers[]` entry *is* the pass's `BufferView`, bytes included, and the
index buffer carries its view beside the width. That is what lets an object-API
trace bind a caller's buffer into a trace with no compute pass at all, and it is
why the declaring case does not have to declare the streams: the streams are not
part of the serial pool — the compute rail's binding set — and each rail uploads
them itself. What the ordering rules still add is that no compute binding may
write bytes the draw reads, and no later compute pass may bind overlapping
bytes, exactly as for the attachment. The indexed shape looks like this:

```json
{
  "id": "offscreen_quad_indexed_clear_2x2",
  "declaring_case": "render_declaring_copy_word",
  "vertex_entry": "render_quad_vertex",
  "fragment_entry": "render_solid_rgba8",
  "metal": {
    "path": "shaders/quad_indexed_2x2.metal",
    "sha256": "aeb662f5d0515ddc4711d821626a72e389506191d11fa03adc9e21ad097379e8"
  },
  "vertices": 6,
  "viewport": [0, 0, 2, 2],
  "vertex_layout": {
    "buffers": [
      {"stride": 8, "attributes": [{"location": 0, "offset": 0, "format": "float32x2"}]}
    ]
  },
  "vertex_buffers": [
    {
      "allocation": 940,
      "view": 950,
      "offset": 0,
      "length": 32,
      "initial_hex": "000080bf000080bf0000803f000080bf000080bf0000803f0000803f0000803f"
    }
  ],
  "indices": {
    "allocation": 960,
    "view": 970,
    "offset": 0,
    "length": 12,
    "initial_hex": "000001000200010003000200",
    "format": "uint16"
  },
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
  "capture_rails": ["vulkan", "native-metal"]
}
```

The oracle validates this shape as a whitelist, not as a per-case table: the
first render increment has exactly one render shape (`research/docs/23` §1.2,
§3) and the vertex-input increment adds exactly one more, so the shape *is* the
review, and a fixture cannot widen it by renaming a case.
`conformance/compare.py` repeats the same rules when it builds the render plan,
so a suite the oracle would refuse cannot pass the comparator either. The
admitted values are one `rgba8_unorm` `2x2` attachment, `store`, the reviewed
module and its entry pair — the `vertex_id` pair with three drawn vertices, or
the indexed pair with the reviewed layout, six indices and their views — and
either

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

The first render increment has five executable rails: the Vulkan trace rail,
the Vulkan object rail (the render command encoder in
`crates/metal-api-core/src/provider_api.rs`, executed by
`VulkanComputeProvider` for both the object and object-async shapes), the native
trace rail (`crates/metal-api-native/src/render.rs`, wired into
`NativeMetalProvider::submit` in Step 7), the native object rail (the same
encoder surface executed by `NativeMetalProvider`, `8793b7a`) and the Swift
oracle's suite path
(`NativeOracle.swift::capture`, which runs the render cases its marker names
after the compute cases, exactly as it runs them for its own `--render-selftest`).
The native provider declares `supports_render_passes = true` with the rail's own
limits — one colour attachment, 2x2, the three admitted formats — because the
flip condition below is met; before that flip it refused a render-bearing trace
with `render_passes_unsupported`. Each Rust rail registers the reviewed pipeline
on its own concrete context: the Vulkan rail's SPIR-V pair
(`VulkanComputeProvider::register_render_pipeline`) and the native rail's
reviewed MSL module (`NativeMetalProvider::register_render_pipeline`).

A render case therefore declares the capture backends that owe the attachment
in its `capture_rails` marker:

* a rail named by the marker has to report the case — `compare.py` refuses a
  capture from it that omits the case;
* a rail not named by it has to omit it — a capture that reports the case
  anyway is refused, because that observation did not come from a rail that can
  produce one.

`suite-v13.json` marks `["vulkan", "vulkan-objects", "native-metal",
"native-metal-provider", "native-metal-provider-objects"]`, so the attachment is
reported by all five backends. `suite-v14.json` marks the four provider rails
(`["vulkan", "vulkan-objects", "native-metal-provider",
"native-metal-provider-objects"]`): the Swift
oracle does not report provider counters, so its present evidence is the
`--present-selftest` check (§7) rather than a marked capture. The v14 Vulkan
object rail executes the present action too, because the object encoder carries
the same optional present tail as the trace path.
`conformance/test_oracle_coverage.py` checks the marker against `compare.py`'s
backend vocabulary, refuses a marker that names a rail with no render command
encoder, and requires every committed suite to be named on every CI rail and in
every object-API version loop.

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
* `.github/workflows/ci.yml` names every committed suite in the four version
  loops, the explicit suite lines, the native status list and the parity lines.

The Apple side is no longer pending: `native-oracle-build` runs the oracle's and
the Rust provider's captures for every committed suite and either reports
`4080c0ff4080c0ff4080c0ff4080c0ff` or fails the job (run `34781060564` for v13,
run `34782615760` for v13 and v14). Local evidence for the Vulkan rails stays
with `tools/lavapipe-smoke.sh`, which discovers every suite from
`conformance/suite*.json` and runs the trace, object and object-async shapes.
`conformance/test_suite_v13.py` holds the schema, the plan and the refusals;
`conformance/test_oracle_coverage.py` holds the marker, the case-id tables and
the CI wiring.

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
  the previous bytes a loading pass resolves from its declaring view
  (`render::previous_bytes`, §9), the readback extent and row pitch;
* the clear-value decoding per format, including the B/G/R/A memory order of
  `bgra8_unorm` and the single-channel float format;
* the vertex-input plan (`render::plan_vertex_input`): the stream identity rule
  (`render_vertex_buffer_unsupported` / `render_index_buffer_unsupported` for a
  view the trace does not declare and for a lease-backed stream this rail holds
  no bytes for), the footprint proof (`render_vertex_footprint_unsupported` for
  a non-indexed draw whose stream is shorter than `vertices * stride`,
  `render_index_footprint_unsupported` for an index view shorter than
  `index_count * index_bytes`), and `render_index_value_out_of_range` for an
  index value that selects a vertex no stream covers (including the
  `vertex_id` shape indexed through a buffer, where the module's three
  positions are the bound);
* the descriptor translation itself: one `PlannedVertexStream` per bound
  layout entry with the binding index, stride, attribute locations/offsets and
  the `VertexFormat` → `RenderVertexFormat` mapping, one `PlannedIndexStream`
  with the `IndexFormat` → `RenderIndexType` mapping, the index count and the
  vertex span, and the two module/entry pairs the rail's allowlist is made of
  (a stream-bearing layout cannot compile the `vertex_id` module or the
  reverse);
* that the declared vertex-input bits are the rail's limits and that core
  admission admits exactly the indexed trace the rail plans, while a
  pre-flip snapshot refuses it with `vertex_buffer_limit`,
  `index_format_unsupported` or `vertex_format_unsupported` — and that an
  indirect draw whose replayed pass binds streams is refused with
  `icb_command_unsupported` before any Metal object exists;
* that the rail's refusal slugs and classes are the ones core admission uses,
  including the trace path's own refusals (`attachment_load_op_unsupported` for
  a loading pass whose declaring view owns no bytes, §9,
  `render_attachment_landing_unsupported` for an attachment no declared view
  covers, `render_pass_order_unsupported` for a compute pass that would read
  after a render store);
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
  marker names reports the attachment (and that a marker which drops one rail
  makes that rail's report of the case refused), and that a tampered attachment
  byte, a dropped texel, a buffer writeback
  standing in for the attachment (in either direction) and a wrong copy count
  are all
  refused (`conformance/test_suite_v13.py`, run by `python3 -m unittest
  discover -s conformance`);
* that the render case actually executes on Lavapipe: `provider-capture --suite
  conformance/suite-v13.json` reports
  `4080c0ff4080c0ff4080c0ff4080c0ff`, and `compare.py --check` accepts it.

**Only an Apple GPU can confirm**, and none of it has been checked yet:

* that `MTLRenderPipelineState` creation succeeds for the reviewed pair and
  `rgba8Unorm` attachment 0;
* that `loadAction = .clear` (and `.load` with the `replaceRegion`-uploaded
  previous bytes of §9) behaves as the contract's `LoadOp` means;
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

The vertex-input increment's Apple-side half is §8's `--vertex-selftest`: the
descriptor, the two `setVertexBuffer`-style bindings and the indexed draw are
exercised there and nowhere else yet. Its host-side half is what the bullets
above cover — the plan, the footprint proof, the index-value bound, the
descriptor translation and the admission agreement — so the two together are
the increment's evidence, and neither is a claim about the other.

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
`vulkan`, `vulkan-objects`, `native-metal-provider` and
`native-metal-provider-objects`; the Apple half of the evidence is the oracle's
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
`4080c0ff4080c0ff4080c0ff4080c0ff`; the object and async-object captures pass with
the same case and counts, run through the object API's render command encoder on
both backends).

## 8. The vertex-input milestone (`--vertex-selftest`)

The vertex-input increment (`research/docs/23` §3.3, §6 Step 3.3) adds the
caller-held half of the render contract: a pass may bind vertex streams and an
index buffer, and the pipeline says what their strides and attributes are. The
rail reads each stream out of the view the pass itself carries — a render input
declares its own bytes (`research/docs/23` §3.6), and the compute rail's pool
holds only the views a compute binding declares — proves every footprint and
every index value on the host (`render::plan_vertex_input`), and only then
translates the plan into an `MTLVertexDescriptor`
(`MTLVertexBufferLayoutDescriptor` per stream, `MTLVertexAttributeDescriptor`
per attribute), one `setVertexBuffer` binding per stream and one
`drawIndexedPrimitives` call whose index buffer is named in the draw, which is
where Metal takes it.

The one-device check is the suite-free equivalent, in the same shape as §5:

```sh
swiftc -swift-version 5 -warnings-as-errors -framework Foundation -framework Metal \
  -framework CoreGraphics -framework CryptoKit conformance/NativeOracle.swift -o /tmp/native-oracle
(cd conformance && /tmp/native-oracle --vertex-selftest)
```

It resolves `shaders/quad_indexed_2x2.metal` relative to the working directory
and checks its pinned SHA-256, builds the reviewed indexed module and the
fixture's `float32x2` stream (four NDC corners) and `uint16` index buffer
(`(0,1,2)`, `(2,1,3)`) through the same `validateRenderCase` a suite uses, draws
with `drawIndexedPrimitives` into a cleared 2x2 `rgba8Unorm` attachment, and
prints the case result as JSON. The self-test fixture spells its stream and
index views the way a suite does — each view carries its own `initial_hex`,
exactly `length` bytes — so the bytes the draw reads travel with the case rather
than being resolved out of a declaring case's `buffers[]`.

The judgement condition is `conformance/run_native.py::validate_vertex_selftest`,
which a CI step of the probe-gated shape §5 describes reuses and which
`conformance/test_run_native.py` exercises on a host without Metal. A report
passes only when all four of these hold, and the step then prints
`vertex_selftest: PASS (4080c0ff)`:

* the report's `id` is the reviewed fixture's (`vertex_quad_indexed_2x2`). This
  is not decoration: the streams are *inputs*, so the attachment's texels are
  the same `4080c0ff` x4 the plain `--render-selftest` reports, and the id is
  the only field that says those bytes were drawn through the caller-held stream
  and index buffer instead of `vertex_id`;
* `completion` is `CompletedVisible`;
* there is exactly one `writebacks` entry, naming the attachment's own view
  (`allocation` 900, `view` 910, `offset` 0) with `bytes_hex` `4080c0ff` x4 —
  the fragment's texel, never the `fefefefe` sentinel the pass started from;
* there is exactly one `allocations` entry, the same allocation, holding those
  same four texels.

`--vertex-selftest`'s Rust counterpart is the plan the same bytes feed: the
footprint proof (`stride * vertices` for a non-indexed draw, `count * width` for
an indexed one), the index-value bound (`index < bytes / stride`), the reviewed
layout and the reviewed module's exact bytes.

The provider's vertex-input bits (`crates/metal-api-native/src/render.rs`,
`vertex_input_capability_bits`; `crates/metal-api-native/src/native.rs`) name
that observation as their flip condition: the three bits say this rail builds
vertex-input state, a stream count of `MAX_VERTEX_BUFFERS` and the admitted
`VertexFormat`/`IndexFormat` sets, and a trace that binds a stream is now
admitted and executed instead of being refused during admission. Before the flip
the same trace was refused with `vertex_buffer_limit` /
`index_format_unsupported`, which is the refusal the unit tests still pin on a
constructed pre-flip snapshot, so the flip is falsifiable in both directions.

What the host cannot answer, and this self-test exists to answer, is the Apple
half: that the reviewed indexed module compiles against the descriptor, that the
bound stream and the index buffer are read from the byte ranges the trace
declared, that `drawIndexedPrimitives` covers the whole 2x2 attachment through
the six indices, and that the readback is the fragment's texel rather than the
clear sentinel. Until the CI job that runs it is green, that half is a
condition, not an observation: the host half above is what `cargo test -p
metal-api-native` and the `aarch64-apple-darwin` cross-check cover today.

## 9. The load milestone (`LoadOp::Load` offscreen)

The first render increments admitted `load: "load"` through the schema but had
no rail that could execute it offscreen: the render pass restates the
attachment's identity and shape and carries no contents, so a `Load` executed
before this increment would have had to become a clear. The load increment gives
that load op a source: the declaring case's view — the same view that is already
the attachment's landing for the writeback — names the attachment's previous
bytes, and the bytes that view owns (`BufferSource::OwnedBytes`) are uploaded
into the attachment before the pass opens (`research/docs/23` §3.3).

What each rail does with those bytes:

* the Vulkan rail stages them through a host-visible buffer and
  `vkCmdCopyBufferToImage` (`UNDEFINED -> TRANSFER_DST_OPTIMAL`), then hands the
  image to the render pass as `COLOR_ATTACHMENT_OPTIMAL` with `LOAD_OP_LOAD`;
  the image declares `TRANSFER_DST` usage exactly when a load uploads;
* the native rail presets the attachment with `MTLTexture.replaceRegion`
  (`render::upload_texels`) and opens the render pass with `MTLLoadAction.Load`;
  `render::plan_trace` resolves the bytes at plan time and `RenderPlan::initial`
  carries them to the encoder.

The judgement conditions, all host-side:

* `render::plan_trace` resolves previous bytes exactly for an offscreen `Load`,
  and from the *declaring view*: the trace's serial pool is the only channel
  that carries them, and the plan re-checks their length against the attachment
  (`width * height * bytes_per_texel`, the tightly packed extent). A wrong
  length is `render_attachment_initial_mismatch`;
* a loading pass whose declaring view owns no bytes — `StagedLease` or
  `BorrowedNoCopy` — is refused with `attachment_load_op_unsupported`
  (Capability, Resolve), naming the storage mode, rather than executed as a
  clear. An attachment that no view covers at all stays
  `render_attachment_landing_unsupported`: the landing rail and the previous
  bytes are the same declaration;
* a `Clear` pass resolves no previous bytes, and `plan` still refuses initial
  bytes a clear would overwrite;
* a present pass's `Load` resolves no bytes either: it keeps the present
  target's own initial state (`InitialState::Sentinel` or `Undefined`), which is
  the shape §7's `--present-selftest` exercises (`research/docs/24` §3.1).

Evidence. On Lavapipe, `crates/metal-api-vulkan/tests/render_e2e.rs` runs
`a_loading_pass_keeps_the_bytes_the_draw_does_not_cover`: a quadrant-sized
triangle covers exactly one of four texels, the other three read back the
declaring view's bytes, and the same fixture with `LOAD_OP_CLEAR` reads back the
clear sentinel. On the native side the plan half above is what
`cargo test -p metal-api-native` covers; the encoder half —
`replaceRegion` followed by `MTLLoadAction.Load` — is the same `.load` branch
the oracle's `--present-selftest` already exercises on Apple hardware, and an
offscreen loading capture of its own has not been run there yet. That is the
half the increment still owes: the Rust provider executes an offscreen loading
pass end to end once a trace declares `Load` with a view whose bytes it owns,
but no committed suite or self-test captures that shape on an Apple GPU today.

## 10. The MRT milestone (`--mrt-selftest`, dual colour outputs)

The MRT increment (wave3 R1) lets one render pass carry two colour attachments
whose texels come from two `[[color(n)]]` outputs of one reviewed fragment
stage. The core contract already carried the shape (`color_formats` is a list,
`MAX_COLOR_ATTACHMENTS` is 4); what this increment adds on the native rail is
the execution of two of those locations and the capability bit that admits
exactly that.

The reviewed fixture is `conformance/shaders/quad_indexed_2x2_dual.metal`
(SHA-256 `5afc95dd177ba64e3d2e115ab84805fad2b56a7450914a0ab6f8572f26ba7eba`):
the same `render_quad_vertex` vertex stage as `quad_indexed_2x2.metal`, and a
fragment stage `render_solid_rgba8_dual` that returns a two-member struct,
`[[color(0)]]` the vertex-input texel and `[[color(1)]]` the byte-reversed
texel. Both are byte/255 constants, so an 8-bit UNORM attachment stores
location 0 as `4080c0ff` and location 1 as `ff8040c0`, and neither equals the
`fefefefe` clear sentinel. The pin is byte-exact in both
`crates/metal-api-native/src/render.rs` (`REVIEWED_DUAL_SOURCE`) and
`conformance/NativeOracle.swift` (`reviewedDualModule()`).

What the host-side plan pins, before a device exists:

* module selection is now the `(VertexLayout, color_formats)` pair, not the
  layout alone. A single output keeps the two pre-existing modules byte for
  byte, so the earlier milestones' captures cannot drift; an indexed pipeline
  with `[Rgba8Unorm, Rgba8Unorm]` selects only the dual module; every other
  two-format list, a `vertex_id` pipeline with two locations and any wider
  list have no reviewed module and are refused with
  `native_render_source_not_reviewed`;
* `MAX_COLOR_ATTACHMENTS` is 2 and `capability_bits()` publishes it, so core
  admission admits a two-attachment pass and refuses a wider one with
  `color_attachment_limit`. The rail's own `plan` keeps the same boundary as
  defense in depth: three or more attachments are
  `render_mrt_attachment_count_unsupported` (Capability, Resolve), with the
  fields `attachments` and `maximum: 2`, instead of rendering two locations
  and dropping the rest;
* every attachment of one pass has to share an extent, because Metal renders
  all locations into one raster. A disagreement is refused before the
  contract's viewport rule (which would report the same shape only as a
  viewport mismatch) with `render_attachment_extent_mismatch` (Args, Resolve),
  naming the attachment index and both extents;
* `plan_trace` resolves one landing view and one previous-bytes entry per
  attachment, reusing `previous_bytes` unchanged, so a loading dual pass
  uploads each attachment's own declaring view and a lease-backed declaration
  is refused per location with `attachment_load_op_unsupported`. A present
  pass still hands exactly one attachment to its target, so a present action
  with two locations is refused with `render_attachment_count_unsupported`;
* the macOS encoder body sets `colorAttachments[0..N]` with one texture and
  one load/store pair per location, compiles one pipeline attachment per
  location, and after `wait` reads every attachment back through the existing
  `read_texels` path — the returned `copy_out` count equals the attachment
  count — and `TraceRenderPlan::writebacks` turns each readback into the
  writeback of that attachment's own landing view.

The one-device check is `--mrt-selftest`: fixture id `mrt_dual_output_2x2`,
the vertex self-test's stream and six `uint16` indices, two 2x2 `rgba8_unorm`
attachments (allocation 900/view 910 and allocation 901/view 911), both
cleared with `fefefefe`. The report has to be `CompletedVisible` with one
writeback and one allocation per location, location 0 `4080c0ff` four times
and location 1 `ff8040c0` four times. The comparison lives in
`conformance/run_native.py::validate_mrt_selftest` — the function the CI step
reuses — and `test_run_native.py` pins it on a host without Metal, including
the negative case that the plain `--render-selftest` report (same location-0
bytes, no second location) must not pass.

Not yet achieved, and therefore still a condition rather than an observation:

* no Apple CI run of `--mrt-selftest` has been committed yet, so the dual
  fragment module has not been compiled or drawn on an Apple GPU; the Rust
  provider's two-attachment encoder body has likewise never executed there
  (`cargo check --target aarch64-apple-darwin` is the compile evidence today,
  not execution evidence);
* the suite-side wiring has since landed: `suite-v18.json` carries the
  `attachments` list, `compare.py` plans and compares one writeback and one
  allocation image per location in location order, and
  `conformance/test_suite_v18.py` pins the schema, the plan, the 3/3 count
  contract and the per-rail markers. The Vulkan trace, object and async-object
  rails execute the case on Lavapipe; the Apple-side capture of that suite is
  still owed;
* the flip of `max_color_attachments` to 2 is host-side evidence alone until
  that Apple run is green; before the flip the same two-attachment trace was
  refused by admission, and the unit tests keep that pre-flip snapshot pinned.

## 11. The store-dontcare milestone (`StoreOp::DontCare`, v19)

The v19 increment admits `StoreOp::DontCare` on a per-attachment basis. Its
semantic is that a discarded attachment must *disappear from the observable
surface*: the pass still renders into it, but its bytes are neither read back
nor reported, so a discarded attachment can never pass as "landed correctly"
(`research/docs/23` §3.6). Core admission states the companion rule — a pass
whose every attachment discards is refused with
`AllRenderAttachmentsDiscarded`, classified like `EmptyAttachmentList` as Args
/ `trace_contract_invalid` — so one `Store` action always keeps the pass
observable. `LoadOp::DontCare` stays refused: it would make the compared bytes
depend on state no earlier pass defined.

What the native rail does:

* `RenderStoreAction` gains a `DontCare` variant and
  `render::store_action(StoreOp::DontCare)` returns it instead of the old
  `attachment_store_op_unsupported` refusal. That slug stays alive for the
  contract mirror and the core legacy mapping, but no store operation reaches
  it on this rail anymore;
* the plan keeps one `PlannedAttachment` per colour location with its own
  `store` action; the previous-bytes / load resolution is unchanged per
  attachment, so a discarded attachment with `LoadOp::Load` still uploads the
  bytes its declaring view owns;
* the macOS encoder body sets `colorAttachments[i].storeAction` to
  `.dontCare` for a discarded attachment and, after `wait`, reads back only
  the stored attachments. `TraceRenderPlan::writebacks` filters its landing
  views the same way, so a discarded attachment produces neither a readback
  nor a writeback;
* `plan` re-runs core's pass validation, so the all-discarded pass is refused
  before any Metal object exists. The unit tests pin the `DontCare` mapping,
  the dual `Store` + `DontCare` plan (one writeback, location 0 only), and the
  all-discarded refusal.

What the Swift oracle does, for a suite render case:

* an attachment's `store` decodes as `"store"` or `"dontcare"`; a `dontcare`
  attachment must not carry `expected_hex`, and a stored attachment must.
  `validateRenderCase` also refuses a case whose every attachment discards,
  mirroring core's pass-level rule;
* the expectation is optional per attachment, so the byte-level review only
  runs where an observation exists. The load-side rules (`clear_hex`,
  `initial_hex` and their length checks) stay per attachment unchanged;
* `runRenderCase` sets `MTLRenderPassDescriptor` store actions to `.dontCare`
  for discarded attachments and, after completion, reads back only the stored
  ones; the resulting `writebacks`/`allocations` contain only the stored
  attachment's observation. A discarded attachment's view still takes part in
  the declaring-case validation, just not in the observation set.

The one-device check is the frozen v19 fixture, id
`discard_second_attachment_2x2`: the dual module's indexed quad, two 2x2
`rgba8_unorm` attachments cleared with `00000000`, location 0
`allocation: 900, view: 910, store: "store"` (expectation `4080c0ff` four
times) and location 1 `allocation: 920, view: 930, store: "dontcare"` with no
expectation. The report has to be `CompletedVisible` with exactly one
writeback and one allocation, both allocation 900 — any observation naming
the discarded allocation 920 or view 930 is refused. The comparison lives in
`conformance/run_native.py::validate_store_dontcare_selftest` — the function
the CI step reuses — and `test_run_native.py` pins it on a host without
Metal, including the negative case that a report carrying the discarded
attachment must not pass, and that the MRT self-test report (same location-0
bytes, but location 1 also reported) must not pass either.

Not yet achieved, and therefore still a condition rather than an observation:

* no Apple CI run of the v19 fixture has been committed yet, so the
  `.dontCare` store action and the skipped readback have not executed on an
  Apple GPU; `cargo check --target aarch64-apple-darwin` is compile evidence
  today, not execution evidence;
* the suite-side wiring has since landed: `suite-v19.json` freezes the fixture,
  `compare.py` owes one writeback/allocation to the stored attachment and
  refuses an observation of the discarded one, `provider-capture.rs` carries
  the store op through the Vulkan trace and object rails, `NativeOracle.swift`
  accepts v19 in `loadSuite`, `test_suite_v19.py` pins the schema/plan/marker
  gates and refusals, and `ci.yml` names the suite on all four capture rails
  and all four object-API version loops. The Apple capture of that suite is
  still the observation that closes this section.

## 12. The undefined-load milestone (`LoadOp::DontCare`, v20)

The v20 increment admits `LoadOp::DontCare` as undefined pre-pass contents. A
`dontcare` attachment still stores its rendered bytes — it differs from the v19
`StoreOp::DontCare` shape, which discards the landing — but the pass starts
from nothing, so neither a clear colour nor the declaring view's bytes travel
with the attachment. What makes the observation falsifiable is the comparison
discipline: the expectation has to be the uniform fragment output, and it has
to differ from the bytes the declaring case pins for the same view — otherwise
"the pass ignored the undefined contents and the renderer drew anyway" is
indistinguishable from "the load actually happened".

What the native rail does:

* `render.rs` opens a `DontCare` attachment from `UNDEFINED` with
  `VK_ATTACHMENT_LOAD_OP_DONT_CARE`, uploads no previous bytes and drops the
  `TRANSFER_DST` requirement the `Load` arm enforces; a `DontCare` entry that
  still carries previous bytes is refused under the preserved
  `UnsupportedAttachmentLoadOp` slug;
* the macOS encoder body maps `LoadOp::DontCare` onto `MTLLoadAction::DontCare`,
  with no `MTLClearColor` and no pre-seeded texture; the Swift oracle decodes
  `load: "dontcare"` the same way and refuses a fixture that carries
  `clear_hex` or `initial_hex` on the attachment;
* the known v19/v20 asymmetry stays deliberate for this step: a present target
  still hard-codes `CLEAR` on the Vulkan rail while the native rail can spell
  `DontCare`. The byte observation cannot distinguish them for the reviewed
  full-coverage quad, so the present path is left out of the v20 fixture until
  a present + dontcare case defines the contract (`docs/24` §3.1).

The one-device fixture is the frozen `suite-v20.json`, render case
`dontcare_load_quad_2x2`: the reviewed indexed quad, one 2x2 `rgba8_unorm`
attachment at `allocation: 900, view: 910` with `load: "dontcare"`,
`store: "store"` and the uniform `4080c0ff` expectation. Its declaring case is
the reviewed `copy_word` whose read view carries `cdcdcdcd` four times, so the
expectation differs from the declared bytes by construction. The count
contract is two uploads (`copy_in == 2`) and two readbacks (`copy_out == 2`).
`conformance/test_suite_v20.py` pins the schema, plan, marker gates, the
`clear_hex`/`initial_hex` refusals, the declared-bytes equality refusal and the
non-uniform-expectation refusal.

Not yet achieved, and therefore still a condition rather than an observation:

* no Apple CI run of the v20 fixture has been committed yet, so
  `MTLLoadAction::DontCare` and the no-pre-seed texture path have not executed
  on an Apple GPU; `cargo check --target aarch64-apple-darwin` and the Swift
  source review are compile evidence today, not execution evidence;
* the Vulkan execution evidence for this step is Lavapipe only in the local
  gates; the RTX 5060 / dzn real-device capture of the v20 suite is still owed.

## 13. The instancing milestone (v31)

`research/docs/23` §3.3 schedules instancing as its own increment: a draw runs
more than one instance, a vertex stream may advance once per *instance* instead
of once per vertex, and the vertex stage may read the `instance_id` builtin.
`conformance/shaders/instanced_quad_2x2.metal` is the third reviewed module and
the one that makes both halves observable at once:

| Entry | Stage | What it does |
|---|---|---|
| `render_instanced_quad_vertex` | vertex | reads `float2 position [[attribute(0)]]` from binding 0 (per vertex) and `float4 tint [[attribute(1)]]` from binding 1 (per instance), shifts its copy of the quad by half the viewport with `[[instance_id]]`, and forwards the tint |
| `render_instanced_tint` | fragment | stores the forwarded tint |

The fixture `instanced_pair_4x4` declares the four reviewed corners as the
position stream, two `float32x4` tint records — `(1, 0, 0, 1)` and
`(0, 1, 0, 1)` — as the per-instance stream, the reviewed six `uint16` indices
and `instance_count: 2`, drawn into one 4x4 `rgba8_unorm` attachment cleared to
`11223344`. Instance 0 covers the left half and instance 1 the right, so the
expected bytes are `ff0000ff` twice then `00ff00ff` twice, on every row.

That shape is what makes the increment falsifiable rather than merely observed:

* a rail that ignores `instance_count` draws only the left half and leaves the
  right half at the clear colour;
* a rail that ignores `instance_id` draws the left half twice and leaves the
  right half at the clear colour;
* a rail that treats the tint stream as per-vertex reads the wrong records (or
  past the view), so the green half never appears.

The two schema fields this increment adds are:

| Suite field | Core value | Rule |
|---|---|---|
| `vertex_layout.buffers[].step` | `VertexBufferLayout::step` | `"per_vertex"` (the default a pre-v31 fixture leaves implicit) or `"per_instance"`; the reviewed instanced layout is the only one that uses the latter |
| `instance_count` | `RenderPassDescriptor::instance_count` | positive; absent means `1`, and the reviewed instanced case declares exactly `2` |

Both rails execute the step function the contract carries: the Vulkan rail
builds each binding's `VkVertexInputBindingDescription.inputRate` from it
(`VK_VERTEX_INPUT_RATE_VERTEX` / `..._INSTANCE`) and passes the pass's instance
count to `vkCmdDraw`/`vkCmdDrawIndexed`; the native rail sets
`MTLVertexBufferLayoutDescriptor.stepFunction` (`.perVertex` / `.perInstance`,
`stepRate = 1`) and draws with `drawPrimitives(...:instanceCount:)` /
`drawIndexedPrimitives(...:instanceCount:)`. The wire carries the step through
the stepped vertex-layout tag and the count through the pass's
`RENDER_FEATURE_INSTANCING` bit, so a frame that instances nothing keeps its
pre-v31 bytes exactly.

Both trace rails and (from v32) both object rails execute the fixture: the
object API's `RenderCommandEncoder::draw_primitives_instanced` and
`draw_indexed_primitives_instanced` are the `instanceCount:` half of Metal's own
draw calls — the count belongs to the draw, not to encoder state — and every
other rule (the bound streams, the positional attachments, the pipeline's own
step functions) is the shape the trace entries already state. The fixture's
marker therefore names all five rails; an object capture that omitted the case
would be a marker refusal.

The evidence boundary of this milestone:

* the comparator's per-half expectation and the Swift oracle's matching rule are
  what refuse a fixture that carries one tint twice or swaps the halves;
* the Apple evidence is the macOS job's `native-metal` capture of this suite,
  and the Windows evidence is the RTX 5060 capture of the same suite on both
  trace rails and both object rails; a green job without either is compile
  evidence only.

## 14. The wildcard-texel channel (v33)

The expectation rules in §2 and §3 assume the case can name every byte it claims:
a clearing pass covers the whole attachment, a loading pass keeps what it was
handed, and a `dontcare` load observes every texel it draws into. A draw that
covers only *part* of a `dontcare`-loaded attachment is the shape none of those
rules admits: the uncovered texels are undefined by construction, so a case that
claimed them would be asserting bytes no rail ever promised.

`wildcard_texels` is that claim's complement, stated in advance: a row-major
list of texel indices whose bytes the case does **not** observe. The rules keep
the fixture falsifiable:

* only the single-attachment shape may carry the list, and only with
  `load: "dontcare"` — the undefined pre-pass contents are exactly what makes an
  unclaimed byte legitimate;
* the list has to be non-empty and strictly smaller than the texel count, so the
  case always claims at least one texel and never claims none;
* every entry names a texel of that attachment, once.

The comparison then skips exactly the named texels' four bytes, in both the
attachment's writeback and its allocation image; every other byte still has to
be the reviewed fragment output, and a wrong byte in a claimed texel is refused
with the offset that differs. The expectation's own bytes at wildcard positions
are placeholders — the fixture keeps `4080c0ff` there for readability — because
what lands there is whatever the driver left.

The reviewed fixture `dontcare_scissor_half_4x4` is the pair this channel was
added for: a 4x4 `rgba8_unorm` attachment opened with `load: "dontcare"`, the
reviewed indexed quad clipped to the left half by `scissor: [0, 0, 2, 4]`, and
`wildcard_texels: [2, 3, 6, 7, 10, 11, 14, 15]` for the right half. Inside the
rectangle every texel is the fragment output; outside it the pass neither read
nor promises anything, and the capture may report whatever the rail left. It is
the observation the partial-coverage fixtures (base vertex offsets, culling,
depth) will need, and it is the reason the channel landed before them.

The evidence boundary is the same as §13's: the five-rail CI run plus the
RTX 5060 capture of the suite, with the Apple half in the `native-oracle-build`
job.

## 15. The base-vertex offset (v34)

An indexed draw reads the vertex its index names; both APIs let the caller add a
constant to every index, which is how several meshes share one vertex buffer.
`RenderPassDescriptor::base_vertex` is that offset (Metal's `baseVertex`,
Vulkan's `vkCmdDrawIndexed(…, vertexOffset, …)`), and it is a *draw* parameter:
`0` is the shape every earlier increment published, so a pass that does not
offset anything keeps its pre-v34 bytes on the wire and in every rail.

The field's rules are two:

* it needs an index buffer — the two APIs add it to index *values*, and a
  non-indexed draw has none to add it to — which is
  `ContractError::BaseVertexRequiresIndices`;
* the bound streams have to cover the offset vertices too: the fetched vertex is
  `base_vertex + index`, so both rails' footprint proofs add the offset before
  they compare against the stream (Vulkan's `vertex_capacity` check, the native
  rail's index-span proof). A stream that fits the indices on their own but not
  the offset is refused before a device object exists.

The reviewed fixture `base_vertex_quad_4x4` is what makes the offset observable
rather than merely accepted: the reviewed quad's layout, module and index buffer
over a five-vertex stream whose first vertex is a degenerate centre, drawn with
`base_vertex: 1`. With the offset the indices reach the four reviewed corners and
every texel of the cleared 4x4 attachment is the fragment output; without it the
degenerate centre replaces a corner and six texel centres keep the clear colour.
Both trace rails and (from v35) both object rails execute the fixture: the
object API's `draw_indexed_primitives_base_vertex` is the `baseVertex:` half of
Metal's indexed draw, and an encoder without an index buffer is refused before
a pass is recorded, exactly as the contract refuses the combination. The
fixture's marker therefore names all five rails.

The evidence boundary is §13's: the five-rail CI run (with the Apple half in
`native-oracle-build`) plus the RTX 5060 capture of the same suite.

## 16. The depth milestone (v36)

`research/docs/23` §3.3 schedules a depth attachment as its own increment. The
first one is deliberately narrow: a pass may open **one** rail-owned depth
surface — `depth32float`, the pass's own extent, cleared or loaded — and declare
the depth state its draw tests and writes with. No depth readback channel, no
stencil, no depth-only passes:

* the surface is rail-owned. It has no view or allocation identity in the
  trace's resource table, because nothing observes it yet; the readback channel
  is what would add those fields. Both rails create it from the declared shape
  (Vulkan: a `D32_SFLOAT` image and view in the render pass's attachment list,
  store `DONT_CARE`; native: a private `.depth32Float` texture in the pass's
  depth attachment, store `.dontCare`);
* the state is the pass's own, mirroring Metal, where the compare function and
  the write enable live in the `MTLDepthStencilState` the encoder sets. The
  Vulkan rail bakes the same pair into the per-pass pipeline it builds, so one
  description serves both;
* the extent has to match every colour attachment's, the format is the one
  admitted value, and a depth state without a depth attachment is
  `ContractError::DepthTestWithoutAttachment`.

The wire keeps every pre-v36 frame byte-identical: a pass with a depth
attachment takes the extended render kind plus `RENDER_FEATURE_DEPTH`, whose
block carries the format, extent, load operation (the clear depth travels as
its IEEE-754 bits, so a frame is byte-stable) and, when present, the compare
function and write flag.

The reviewed fixture `depth_pair_4x4` is falsifiable by construction. Its stream
is two oversize triangles — the same triangle, twice — the first at `z = 0.5`
tinted red and the second at `z = 0.9` tinted green, drawn in one indexed draw
into a pass whose depth surface is cleared to one. With `less` and depth writes
the near triangle wins everywhere the two overlap, which is the whole
attachment, so the expectation is the red texel sixteen times. A rail that
dropped the attachment, the clear or the test would show the green one: the
same fixture run with `compare: "always"` reads back sixteen `00ff00ff` texels
on Lavapipe, which is the control that makes the red run evidence rather than a
coincidence.

Both trace rails and (from v37) both object rails execute the fixture: the
object API's `draw_primitives_with_depth` and `draw_indexed_primitives_with_depth`
are the depth-bearing recording entries, and the pass they record carries the
same rail-owned surface and state the trace contract names. The fixture's
marker therefore names all five rails.

## 17. The non-indexed draw arm (v39)

The render narrow class's one draw has two arms (`research/docs/23` §3.3, v39).
The first is the indexed one every pre-v39 case declares: the reviewed
`float32x2` position stream at stride eight plus an index buffer. The second is
the same streams drawn *without* an index buffer, which the contract has always
carried — `RenderPassDescriptor::indices` is an `Option`, and `None` reads as
"the draw names its vertices `0..vertices`".

What the arm owes is not a new module but the footprint rule the two rails
already prove for it: a non-indexed draw reads one record per vertex, so every
per-vertex stream has to cover the whole `vertices * stride` span (the indexed
arm only has to cover the span its index values reach, so this is the stricter
of the two), and a draw of fewer than three vertices rasterizes no triangle at
all. A case carrying no `vertex_layout` at all is the milestone's `vertex_id`
triangle and stays pinned to `vertices == 3`; one with a layout may name any
`vertices >= 3` its streams cover.

`suite-v39.json` states the arm four ways, so the parity and the falsifiers sit
in one suite: the milestone's `vertex_id` triangle (which the widened coverage
now admits rather than refuses), the reviewed indexed quad carried unchanged,
the same six vertices expanded out of the index buffer and drawn without one
(whose frame has to be the indexed case's frame byte for byte), and a neighbour
that draws one triangle of the same layout over a single texel (whose frame has
to move, which is what makes the vertex bytes — not the draw's shape — the thing
the attachment reads). The neighbour declares the partial coverage it resolves
and `conformance/narrow-class.json` refuses it by that rule, but it still owes
all five rails.

The object rails record the arm through `draw_primitives_with_attachments`,
which binds the encoder's streams and carries no index buffer: the same entry
the indexed arm's `draw_indexed_primitives_with_attachments` mirrors. The
milestone's `draw_render_pass` stays the one entry that binds nothing, so a
stream-carrying case can never be recorded through it by accident.
