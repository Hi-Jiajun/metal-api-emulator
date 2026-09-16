"""Scissor and instancing checks for `suite-v28.json`.

These are comparator and schema checks, not GPU execution evidence. They pin two
increments of `research/docs/23` §3.3: the scissor (v29), where a pass clips its
draw to a rectangle and the comparison then knows the coverage exactly — inside
the rectangle every texel is the fragment output, outside it every texel keeps
the clear colour — and the instanced pair (v31), where a per-instance tint
stream covers each half of the attachment with its own instance's colour, so
both the instance count and the stream's per-instance step are observable. A
scissor that covers the whole attachment (or nothing), or an expectation that
carries one tint twice, is refused: it could not show that the rail executed the
feature.

The same file pins the depth increments: v36's pair of oversize triangles over a
cleared depth attachment, and v43's stored depth attachment, which the pass hands
back as a resource of its own — one more allocation, observed through the
writeback and allocation channel the colour side already uses, and one more
touched and written allocation behind the provider counts. The v45 increment
makes that landing the pass's whole observation: the colour attachment renders
and then discards, so the stored depth texels are everything a capture owes.
The v46 increment removes the colour attachment altogether: a pass whose
fragment stage produces no output still writes depth, so the same stored depth
texels stay the case's whole observation with no colour surface at all.
The v47 increment adds the first stencil shape: a rail-owned `stencil8`
attachment the pass clears, masks with and drops, plus the depth fixture's own
pair of triangles. The near triangle passes an `equal 0` test and increments the
stored value, so the far one fails the test and is discarded — the expectation
is the near tint sixteen times, and a rail that ignored the stencil state would
show the far one exactly as it would for the depth pair. The attachment is
rail-owned like the pre-v43 depth surface, so it adds no writeback, no
allocation image and no count; v48 carries the same surface and the same test
on every rail, so the marker names all five and each one owes the same colour
pair's landing.

The v49 increment upgrades that attachment the way v43 upgraded the depth one:
the pass keeps the surface, states where the texels land and what the readback
has to contain — one byte per texel — and the comparison observes them through
the same writeback and allocation channel the colour side uses. The case then
owes two landings, so the marker names all five rails and a provider capture
owes the pair's own two touched and two written allocations plus the stencil
landing's one of each.
"""

import copy
import hashlib
import json
from pathlib import Path
import struct
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V28_PATH = CONFORMANCE / "suite-v28.json"

DECLARING_ID = "render_declaring_quad_extent"
DEPTH_DECLARING_ID = "render_declaring_depth_store"
RENDER_ID = "scissor_left_half_4x4"
INSTANCED_ID = "instanced_pair_4x4"
WILDCARD_ID = "dontcare_scissor_half_4x4"
BASE_VERTEX_ID = "base_vertex_quad_4x4"
DEPTH_ID = "depth_pair_4x4"
# The v43 fixture: the same pair of triangles, but the depth attachment is a
# resource the pass stores instead of one it drops.
DEPTH_STORE_ID = "depth_store_pair_4x4"
# The v45 fixture: the same pair, the same depth resource the pass stores, and a
# colour attachment that discards — the stored depth texels are the only thing
# the pass observes.
DEPTH_ONLY_ID = "depth_only_store_4x4"
# The v46 fixture: the same stored depth landing again, but the pass binds no
# colour attachment at all — the fragment stage produces no output, and the
# stored depth texels stay the whole observation.
DEPTH_NO_COLOUR_ID = "depth_only_no_colour_4x4"
# The v47 fixture: the depth pair's geometry and module over a cleared
# `stencil8` attachment instead of a depth one. The near triangle passes an
# `equal 0` test and increments the stored value, so the far one fails and is
# discarded.
STENCIL_ID = "stencil_increment_pair_4x4"
# The v49 fixture: the same stencil surface this time *kept* by the pass. The
# store hands the incremented value back through the same writeback and
# allocation channel the colour and depth landings use, one byte per texel.
STENCIL_STORE_ID = "stencil_store_pair_4x4"
# The declaring case of the v49 fixture: the v43 declaring kernel with a
# one-byte-per-texel third *read* binding in place of the depth view.
STENCIL_DECLARING_ID = "render_declaring_stencil_store"
# The declaring case of the v57d device-gated pair: the same v43 declaring
# kernel whose third *read* binding is the pair's own depth landing view, so
# both edge cases resolve against one declared resource.
DEPTH_RESOLVE_DECLARING_ID = "render_declaring_depth_resolve"
# The declaring case of the v60 pair: the same declaring kernel with a fourth
# *read* binding, so both the depth and the stencil landing of the combined
# shape are declared by one compute pass (`research/docs/23` §3.3, v60).
STENCIL_RESOLVE_DECLARING_ID = "render_declaring_stencil_resolve"
# The reviewed depth resource of the v43 fixture: the allocation and the view
# the stored texels land in, and the view's whole extent.
DEPTH_STORE_ALLOCATION = 940
DEPTH_STORE_VIEW = 950
DEPTH_STORE_ATTACHMENT = (DEPTH_STORE_ALLOCATION, DEPTH_STORE_VIEW, 0, 64)
# The reviewed stencil resource of the v49 fixture: one byte per texel, so the
# view covers sixteen bytes of its sixty-four byte allocation — the guard bytes
# around the landing stay part of the comparison.
STENCIL_STORE_ALLOCATION = 940
STENCIL_STORE_VIEW = 951
STENCIL_STORE_ATTACHMENT = (STENCIL_STORE_ALLOCATION, STENCIL_STORE_VIEW, 0, 16)
# The reviewed depth resource of the v57d pair: the landing both edge cases
# share, a whole-allocation sixty-four byte view like the v43 depth store's.
DEPTH_RESOLVE_ALLOCATION = 960
DEPTH_RESOLVE_VIEW = 961
DEPTH_RESOLVE_ATTACHMENT = (DEPTH_RESOLVE_ALLOCATION, DEPTH_RESOLVE_VIEW, 0, 64)
# The reviewed stencil landing of the v60 combined pair: the v49 stencil
# allocation with its own view, one byte per texel.
STENCIL_RESOLVE_VIEW = 952
ALIGNMENT_ID = "top_half_quad_4x4"
CULL_ID = "cull_back_half_quad_4x4"
BLEND_ID = "blend_alpha_quad_4x4"
# The v51 fixture: one colour attachment opened from a clear, rendered with a
# four-sample raster and resolved into the attachment view itself. The quad's
# right edge sits halfway through the third texel column, so exactly two of that
# column's four samples are covered and the resolved texel is the arithmetic
# mean of the fragment output and the clear colour — a byte pattern a
# single-sample raster cannot produce.
MSAA_ID = "msaa_edge_4x4"
# The v53 fixture: the depth pair's own two triangles over the same four-sample
# raster the edge fixture states, with the rail-owned `depth32float` surface
# cleared to one and a `less` test with writes on. Both primitives cover every
# sample, so the near tint wins on every texel and a rail that ignored the depth
# test would land the far tint instead.
MSAA_DEPTH_ID = "msaa_depth_pair_4x4"
# The v55 fixture: the v47 stencil pair over the same four-sample raster — the
# first triangle passes `equal 0` and increment-wraps the samples it covers, the
# second is tested against the value the first wrote, so it is rejected on every
# sample it covers and the resolve target carries the first tint.
MSAA_STENCIL_ID = "msaa_stencil_pair_4x4"
# The v57 fixture: the v43 stored depth pair over the same four-sample raster
# the edge fixture states, resolved with the `sample0` filter into the depth
# landing's own single-sample image. The near triangle still wins on every
# texel, so the colour observation is the v43 pair's and the stored depth
# observation is the resolved 0.5 (`0000003f`).
MSAA_DEPTH_RESOLVE_ID = "msaa_depth_resolve_sample0_4x4"
# The v57d device-gated pair: the same module over the v51 edge geometry — the
# near triangle stops at x = 0.25 NDC, so the third texel column is half
# covered — with both triangles tinted red. The Min and Max reductions of that
# column then disagree (0.5 vs 0.9), which the pre-v57d full-coverage probe
# could not show (`research/docs/23` §3.3, v57d).
MSAA_DEPTH_RESOLVE_MIN_EDGE_ID = "msaa_depth_resolve_min_edge_4x4"
MSAA_DEPTH_RESOLVE_MAX_EDGE_ID = "msaa_depth_resolve_max_edge_4x4"
# The v60 pair: the same edge geometry over a combined depth-stencil surface,
# whose stencil resolve reduces the near triangle's stencil 1 and the far
# triangle's stencil 0 (`research/docs/23` §3.3, v60).
MSAA_STENCIL_RESOLVE_SAMPLE0_ID = "msaa_stencil_resolve_sample0_4x4"
MSAA_STENCIL_RESOLVE_DRS_ID = "msaa_stencil_resolve_drs_4x4"
# The v61 pair: the same quad module over the full-coverage geometry — all four
# corners sit on the attachment's edges, so every sample of every texel is
# covered and the expectation is the fragment output sixteen times. The two
# cases state the 2x and 8x rasters, whose partial-coverage sample positions
# are not documented, so only the full-coverage shape is pinned and each case
# carries a device gate (`requires_sample_count`) that omits it from captures
# whose device ceiling cannot run that raster (`research/docs/23` §3.3, v61).
MSAA_UNIFORM_2X_ID = "msaa_uniform_2x_4x4"
MSAA_UNIFORM_8X_ID = "msaa_uniform_8x_4x4"
# The v66 case: the rail-owned combined depth-stencil pair, whose three
# triangles share the v51 edge coverage and whose two faces are both discarded
# with the pass (`research/docs/23` §3.3, v66).
MSAA_DS_ID = "msaa_ds_pair_4x4"
# The v43 case sits between the v36 depth pair and the v38 alignment fixture, and
# the v45 depth-only case and v46 no-colour case follow it, so every case after
# the stored depth pair moved by three positions.
DEPTH_STORE_INDEX = 5
DEPTH_ONLY_INDEX = 6
DEPTH_NO_COLOUR_INDEX = 7
# The v47 stencil case follows the v46 no-colour case, so the alignment, culling
# and blending fixtures moved by one more position.
STENCIL_INDEX = 8
# The v49 stencil store case follows the v47 stencil pair, so the alignment,
# culling and blending fixtures moved by one more position.
STENCIL_STORE_INDEX = 9
ALIGNMENT_INDEX = 10
CULL_INDEX = 11
BLEND_INDEX = 12
# The v51 multisample fixture is the newest case, so every earlier position is
# unchanged and the new one is last.
MSAA_INDEX = 13
# The v53 depth-tested multisample fixture follows it.
MSAA_DEPTH_INDEX = 14
# The v55 stencil-masked multisample fixture follows that.
MSAA_STENCIL_INDEX = 15
# The v57 depth-resolve fixture is the newest case.
MSAA_DEPTH_RESOLVE_INDEX = 16
# The v57d device-gated pair follows it.
MSAA_DEPTH_RESOLVE_MIN_EDGE_INDEX = 17
MSAA_DEPTH_RESOLVE_MAX_EDGE_INDEX = 18
# The v60 stencil-resolve pair follows it.
MSAA_STENCIL_RESOLVE_SAMPLE0_INDEX = 19
MSAA_STENCIL_RESOLVE_DRS_INDEX = 20
# The v61 sample-count fixtures are the newest cases.
MSAA_UNIFORM_2X_INDEX = 21
MSAA_UNIFORM_8X_INDEX = 22
# The v66 rail-owned combined pair is the newest case.
MSAA_DS_INDEX = 23
REVIEWED_ORDER = (RENDER_ID, INSTANCED_ID, WILDCARD_ID, BASE_VERTEX_ID, DEPTH_ID,
                  DEPTH_STORE_ID, DEPTH_ONLY_ID, DEPTH_NO_COLOUR_ID,
                  STENCIL_ID, STENCIL_STORE_ID, ALIGNMENT_ID, CULL_ID, BLEND_ID,
                  MSAA_ID, MSAA_DEPTH_ID, MSAA_STENCIL_ID, MSAA_DEPTH_RESOLVE_ID,
                  MSAA_DEPTH_RESOLVE_MIN_EDGE_ID, MSAA_DEPTH_RESOLVE_MAX_EDGE_ID,
                  MSAA_STENCIL_RESOLVE_SAMPLE0_ID, MSAA_STENCIL_RESOLVE_DRS_ID,
                  MSAA_UNIFORM_2X_ID, MSAA_UNIFORM_8X_ID, MSAA_DS_ID)
ATTACHMENT = (900, 910, 0, 64)
PROBE = (920, 930, 4)
QUAD_VIEW = (1000, 1010, 0, 32)
INDEX_VIEW = (1020, 1030, 0, 12)
SCISSOR = [0, 0, 2, 4]
OUTPUT = "4080c0ff"
CLEAR = "11223344"
EXPECTED = "".join(OUTPUT if (index % 4) < 2 else CLEAR for index in range(16))
# The reviewed instance tints: red for the left half, green for the right.
INSTANCE_TINTS = ("ff0000ff", "00ff00ff")
INSTANCED_EXPECTED = "".join(
    INSTANCE_TINTS[0] if (index % 4) < 2 else INSTANCE_TINTS[1]
    for index in range(16))
# The v33 fixture: a `dontcare` load with a left-half scissor. The right half is
# unclaimed, so its bytes are whatever the driver left; the synthetic capture
# below reports a byte the expectation does not carry to prove the mask skips it.
WILDCARD_EXPECTED = OUTPUT * 16
WILDCARD_TEXELS = [2, 3, 6, 7, 10, 11, 14, 15]
WILDCARD_REPORTED = "".join(
    OUTPUT if index not in WILDCARD_TEXELS else "cdcdcdcd"
    for index in range(16))
# The v34 fixture: the reviewed quad's indices offset by one over a five-vertex
# stream, drawn into a cleared 4x4 attachment. A rail that ignored the offset
# would leave the six texels the degenerate shape misses at the clear colour.
BASE_VERTEX_EXPECTED = OUTPUT * 16
# The v36 fixture: two oversize triangles at z = 0.5 (red) and z = 0.9 (green)
# over a cleared depth32float attachment with a `less` test. The near triangle
# wins, so the expectation is the red texel sixteen times; a rail that dropped
# the depth state would show the green one.
DEPTH_EXPECTED = INSTANCE_TINTS[0] * 16
# The v43 fixture stores the near triangle's depth (0.5, float32 little endian)
# in the sixteen texels of its own depth resource. The clear depth is the byte
# string the expectation has to differ from: it is exactly what a pass that
# never stored the attachment leaves behind.
DEPTH_STORE_EXPECTED = "0000003f" * 16
DEPTH_STORE_CLEAR = "0000803f" * 16
# The v47 fixture: the same two triangles over a cleared `stencil8` surface with
# an `equal 0` test whose pass op increments and wraps. The near triangle passes
# and stores one; the far one then meets that stored value, fails the test and is
# discarded, so the near (red) tint covers the attachment — the same texel the
# v36 depth pair lands.
STENCIL_SECTION = {"format": "stencil8", "width": 4, "height": 4, "load": "clear",
                   "clear_value": 0}
STENCIL_TEST = {"compare": "equal", "reference": 0, "read_mask": 255, "write_mask": 255,
                "fail_op": "keep", "depth_fail_op": "keep", "pass_op": "increment_wrap"}
STENCIL_EXPECTED = INSTANCE_TINTS[0] * 16
# The v49 fixture stores the value the near triangle leaves in the stencil
# surface: one byte per texel, sixteen `01`, and the clear value is the byte
# string the expectation has to differ from — it is exactly what a pass that
# never stored the surface leaves behind.
STENCIL_STORE_EXPECTED = "01" * 16
# The allocation image the landing leaves behind on the declaring case's own
# sixty-four byte allocation: the sixteen stored bytes at the view's offset,
# then the declaring image's remaining bytes (the suite's zero guard byte).
# The declaring case's stencil allocation is the view itself — sixteen bytes for the
# 4x4 `stencil8` surface, the shape the depth sibling's 64-byte landing has
# (`research/docs/23` §3.3, v43/v49).
STENCIL_STORE_ALLOCATION_BYTES = STENCIL_STORE_EXPECTED
# The v38 fixture: the reviewed quad over the attachment's top half in Metal's
# NDC convention, with the `partial` coverage claim. The expectation mixes the
# fragment output (top two rows) with the clear colour (bottom two), and the
# Vulkan rail's reviewed vertex modules flip y so both rails cover the same
# framebuffer rows.
ALIGNMENT_EXPECTED = "".join(
    OUTPUT if (index // 4) < 2 else CLEAR for index in range(16))

# Every rail executes the scissor from v30 on: the object API's encoder carries
# `set_scissor`, so the fixture names all five. The instanced pair is the same
# story from v32 on: `draw_indexed_primitives_instanced_with_attachments` is
# the object API's `drawIndexedPrimitives(...:instanceCount:)`, so its marker
# names all five rails too.
TRACE_RAILS = ("native-metal", "vulkan", "native-metal-provider")
OBJECT_RAILS = ("vulkan-objects", "native-metal-provider-objects")
ALL_RAILS = TRACE_RAILS + OBJECT_RAILS
INSTANCED_RAILS = ALL_RAILS
# The base-vertex draw is the same story from v35 on:
# `draw_indexed_primitives_base_vertex_with_attachments` is the object API's
# `drawIndexedPrimitives(…:baseVertex:baseInstance:)`, so its marker names all
# five rails too.
BASE_VERTEX_RAILS = ALL_RAILS
# The depth fixture is the same story from v37 on:
# `draw_indexed_primitives_with_depth` is the object API's depth-bearing draw
# entry, so its marker names all five rails too.
DEPTH_RAILS = ALL_RAILS
# The stored depth attachment names the three trace rails: the object API has no
# render command encoder, so the object rails cannot hand the texels back yet
# (`research/docs/23` §3.3, v43).
# The v44 object entries record the same store action and landing identity the
# trace contract carries — `draw_indexed_primitives_with_depth` takes one
# `RenderDepthAttachment` whose `store`/`identity` are the contract's own — so
# the stored-depth fixture's marker names all five rails.
DEPTH_STORE_RAILS = ALL_RAILS
# The depth-only case names the same five rails: the object entries carry the
# same store action and landing identity, and the pass has nothing else to hand
# back — every colour attachment discards (`research/docs/23` §3.3, v45).
DEPTH_ONLY_RAILS = ALL_RAILS
# The no-colour case names the same five rails for the same reason: the object
# entries carry the same store action and landing identity, and the pass binds
# no colour attachment at all (`research/docs/23` §3.3, v46).
DEPTH_NO_COLOUR_RAILS = ALL_RAILS
# The v47 marker named the two rails that first executed the stencil fixture:
# the Swift oracle and the Vulkan rail. The v48 increment carries the same
# rail-owned `stencil8` surface and the same `equal 0` test on the Rust native
# rail and through both object rails — the object entries record the surface
# the way v44's depth entry recorded its own store action — so the marker
# widens to every rail the way the stored-depth one did
# (`research/docs/23` §3.3, v48). The case is still observed through the
# colour pair's own landing, so the wider marker owes the same shape on all
# five rails.
STENCIL_RAILS = ALL_RAILS
# The v49 marker names the same five rails: the store action and the landing
# identity travel through the trace contract and both object entries exactly as
# the depth store's do since v44, and the Rust native rail's readback hands the
# stored bytes back — so every rail owes both landings
# (`research/docs/23` §3.3, v49).
STENCIL_STORE_RAILS = ALL_RAILS
# The alignment fixture names every rail: both trace and object rails execute
# the reviewed quad, and the Vulkan rail's reviewed vertex modules flip y so the
# framebuffer rows agree with Metal's convention (`research/docs/23` §3.3, v38).
ALIGNMENT_RAILS = ALL_RAILS
# The v39 fixture: two copies of one oversize triangle whose vertex orders are
# opposite, drawn with `cull: back` and a counter-clockwise front. The green
# (counter-clockwise) copy survives; a rail that ignored the state would show
# the later red-to-green order instead — the copies are ordered so the ignored
# state lands the other tint.
CULL_EXPECTED = INSTANCE_TINTS[1] * 16
# The v40 fixture: one oversize triangle whose tint carries alpha 128/255,
# blended over a cleared-to-zero attachment with source-alpha against
# one-minus-source-alpha. The stored texel is (32, 64, 96, 64) = `20406040`;
# a rail that ignored the blend state would store the tint's own bytes
# (`4080c080`).
BLEND_EXPECTED = "20406040" * 16
# The blend state is the same story from v42 on:
# `draw_indexed_primitives_with_blend` is the object API's own blending entry,
# so its marker names all five rails.
BLEND_RAILS = ALL_RAILS
# The culling state is the same story from v41 on:
# `draw_indexed_primitives_with_cull` is the object API's own culling entry, so
# its marker names all five rails.
CULL_RAILS = ALL_RAILS
# The v51 multisample raster named the three trace rails first; the v52
# recording entry (`draw_indexed_primitives_with_multisample`) carries the same
# state on both object rails, so the marker widens to every rail
# (`research/docs/23` §3.3, v51/v52).
MSAA_RAILS = ALL_RAILS
# The depth-tested sibling named the three trace rails first; the v54 recording
# entry (`draw_indexed_primitives_with_multisample_depth`) carries the same two
# halves on both object rails, so the marker widens to every rail
# (`research/docs/23` §3.3, v53/v54).
MSAA_DEPTH_RAILS = ALL_RAILS
# The stencil sibling named the three trace rails first; the v56 recording entry
# (`draw_indexed_primitives_with_multisample_stencil`) carries the same two
# halves on both object rails, so the marker widens to every rail
# (`research/docs/23` §3.3, v55/v56).
MSAA_STENCIL_RAILS = ALL_RAILS
# The depth resolve is device-gated: this increment's Lavapipe device reports
# `SAMPLE_ZERO` only, and the native rail declares Sample0 alone, so the
# reviewed fixture states `sample0`. The v57c marker named the three trace
# rails; the v58 recording entry
# (`draw_indexed_primitives_with_multisample_depth_resolve`) carries the same
# resolve on both object rails, so the marker widens to every rail
# (`research/docs/23` §3.3, v57c/v58).
MSAA_DEPTH_RESOLVE_RAILS = ALL_RAILS
# The device-gated pair named the Vulkan trace rail alone through v57d: the
# RTX 5060 reports Min and Max and Lavapipe reports Sample0 alone, so the pair
# is the mask-gated observation that tells the two devices apart. The v57e
# depth-resolve self-test then measured the Apple Paravirtual device executing
# all three filters (`f4d70e4`, CI run `35112569688`), so v57f widens the
# marker to the three trace rails, and v58 widens it again to every rail —
# the object rails run the same gated pair, and the device mask still decides
# presence (`research/docs/23` §3.3, v57d/v57f/v58).
MSAA_DEPTH_RESOLVE_MIN_EDGE_RAILS = ALL_RAILS
MSAA_DEPTH_RESOLVE_MAX_EDGE_RAILS = ALL_RAILS
# The v60 Sample0 case names every rail: the trace rails execute the combined
# shape from this increment on, and the object rails follow the same recorded
# state (`research/docs/23` §3.3, v60).
MSAA_STENCIL_RESOLVE_SAMPLE0_RAILS = ALL_RAILS
# The DepthResolvedSample case names the two native trace rails alone: Vulkan
# has no stencil mode for Metal's depth-following filter, so its rail refuses
# the filter and the mask never carries its bit (`research/docs/23` §3.3, v60).
MSAA_STENCIL_RESOLVE_DRS_RAILS = ("native-metal", "native-metal-provider")
# The mask the rails publish once the device proves the filters: Sample0|Min|
# Max, bit `i` = filter code `i`. The v57e self-test is the native half of
# that proof, and the RTX 5060 reports the same value; the v57f marker named
# the three trace rails and v58 widens it to every rail, so the per-rail
# synthetic captures declare the mask on every rail that owns the edge pair
# (`research/docs/23` §3.3, v57f/v58).
DEPTH_RESOLVE_ALL_FILTERS_BITS = (1 << 0) | (1 << 1) | (1 << 2)
# The stencil mask the native rails publish on the v59 evidence: Sample0 |
# DepthResolvedSample, bit `i` = filter code `i`. The Vulkan rail has no mode
# for DepthResolvedSample, so its mask carries Sample0 alone
# (`research/docs/23` §3.3, v60).
STENCIL_RESOLVE_ALL_FILTERS_BITS = (1 << 0) | (1 << 1)
MSAA_STENCIL_EXPECTED = "ff0000ff" * 16
MSAA_DEPTH_EXPECTED = "ff0000ff" * 16
# The v57 fixture's two observations: the colour pair's near tint sixteen
# times, and the resolved depth surface's near depth (0.5, float32 little
# endian) sixteen times — the v43 stored depth pair's own bytes.
MSAA_DEPTH_RESOLVE_EXPECTED = MSAA_DEPTH_EXPECTED
MSAA_DEPTH_RESOLVE_DEPTH = DEPTH_STORE_EXPECTED
# The device-gated pair's colour observation stays the near tint sixteen times
# — both triangles are red, so the colour side cannot tell the two filters
# apart and must not claim to.
MSAA_DEPTH_RESOLVE_MIN_EDGE_EXPECTED = MSAA_DEPTH_RESOLVE_EXPECTED
MSAA_DEPTH_RESOLVE_MAX_EDGE_EXPECTED = MSAA_DEPTH_RESOLVE_EXPECTED
# The depth observation is what makes the pair falsifiable. The near triangle
# covers the whole of columns 0-1 and half of column 2's four samples (the v51
# edge at x = 0.25 NDC), and the far one covers everything, so per row the
# Min reduction lands 0.5 in columns 0-2 and 0.9 in column 3, while the Max
# reduction lands 0.5 in columns 0-1 and 0.9 in columns 2-3 — the two
# expectations differ exactly where the filters differ.
MSAA_DEPTH_RESOLVE_MIN_EDGE_DEPTH = ("0000003f" * 3 + "6666663f") * 4
MSAA_DEPTH_RESOLVE_MAX_EDGE_DEPTH = ("0000003f" * 2 + "6666663f" * 2) * 4
# The v60 pair's three observations. The combined depth clear is 0.7 between
# the two triangles' depths, so the near triangle's depth pass writes stencil 1
# while the far triangle's depth failures leave the rest at zero: the Sample0
# stencil landing reads 01 in columns 0-2 (its third-column sample 0 is near)
# and 00 in column 3, and its depth landing reads 0.5 in columns 0-2 and the
# 0.7 clear in column 3. The DepthResolvedSample landing follows the Max depth
# resolve: columns 2-3 take the far sample's stencil 00 and depth 0.7, which is
# the split that makes the pair falsifiable (`research/docs/23` §3.3, v60).
MSAA_STENCIL_RESOLVE_SAMPLE0_STENCIL = "01010100" * 4
MSAA_STENCIL_RESOLVE_SAMPLE0_DEPTH = ("0000003f" * 3 + "3333333f") * 4
MSAA_STENCIL_RESOLVE_DRS_STENCIL = "01010000" * 4
MSAA_STENCIL_RESOLVE_DRS_DEPTH = ("0000003f" * 2 + "3333333f" * 2) * 4
# The combined pair's colour observation: the near triangle covers columns
# 0-1 fully, its edge splits column 2 in half, and the far triangle's depth
# failures leave column 3 at the clear — so column 2 carries the 2-of-4 mix of
# the red tint and the clear (`research/docs/23` §3.3, v60).
MSAA_STENCIL_RESOLVE_COLOUR = ("ff0000ff" * 2 + "88111aa2" + "11223445") * 4
# The v61 fixtures' uniform expectation: the full-coverage quad lands the
# fragment output on every texel, so both the 2x and 8x cases pin the same
# sixteen bytes. The device gate is the count half of the same question —
# the marker names all five rails (the v52 recording entry carries the
# raster), and each capture omits a case whose count its device ceiling
# cannot run (`research/docs/23` §3.3, v61).
MSAA_UNIFORM_CLEAR = "11223344"
MSAA_UNIFORM_2X_EXPECTED = OUTPUT * 16
MSAA_UNIFORM_8X_EXPECTED = OUTPUT * 16
MSAA_UNIFORM_2X_RAILS = ALL_RAILS
MSAA_UNIFORM_8X_RAILS = ALL_RAILS
# The reviewed multisample expectation: the fragment output where the quad
# covers every sample, the clear colour where it covers none, and the
# `2`-of-`4` resolve of the two in the column the quad's right edge crosses.
MSAA_CLEAR = "22446689"
MSAA_MIXED = "316293c4"
MSAA_EXPECTED = "".join(
    OUTPUT if (index % 4) < 2 else (MSAA_MIXED if (index % 4) == 2 else MSAA_CLEAR)
    for index in range(16))

# The v66 fixture: the rail-owned combined depth-stencil pair
# (`research/docs/23` §3.3, v66). Three copies of the v51 edge triangle share
# one coverage — the half-plane triangle whose vertical edge splits the third
# texel column — at z = 0.5 (red), z = 0.9 (green) and z = 0.4 (blue), over
# one combined surface whose two faces are both discarded with the pass. The
# first triangle's depth pass writes depth, the second one's depth failure
# writes stencil through the reviewed depth-failure increment, and the third
# one is rejected by the stencil the second wrote: a rail that ignored either
# face would land the third triangle's tint on the covered columns, so the
# expectation is the first triangle's tint, its `2`-of-`4` resolve against the
# clear in the split column, and the clear where nothing drew. The three trace
# rails named it first, and the v68 recording entry
# (`draw_indexed_primitives_with_multisample_depth_stencil`) carries the same
# pair on both object rails, so the marker names all five
# (`research/docs/23` §3.3, v66/v68). The tints carry only zero and one
# channels, so every expected byte is an exact tint, clear or two-of-four mean:
# no channel sits on a rounding tie, which is what the RTX 5060 direct track
# measured when a half-intensity tint read back one step below its Lavapipe
# byte.
MSAA_DS_RAILS = ALL_RAILS
MSAA_DS_STENCIL_TEST = {"compare": "equal", "reference": 0, "read_mask": 255,
                        "write_mask": 255, "fail_op": "keep",
                        "depth_fail_op": "increment_wrap", "pass_op": "keep"}
MSAA_DS_CLEAR = "11223445"
MSAA_DS_EXPECTED = (INSTANCE_TINTS[0] * 2 + "88111aa2" + MSAA_DS_CLEAR) * 4


def render_result(provider_backend=True, copy_in=2, copy_out=2):
    result = {
        "id": RENDER_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def instanced_result(provider_backend=True, copy_in=2, copy_out=2):
    result = {
        "id": INSTANCED_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": INSTANCED_EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": INSTANCED_EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def other_rail(rail):
    """A known rail that is not `rail`, for "this capture owes nothing" markers."""
    for candidate in ALL_RAILS:
        if candidate != rail:
            return candidate
    raise AssertionError("no other rail exists")


def depth_store_marker(suite, rail):
    """Point the v43 case at `rail` when that rail owes it, and elsewhere when not.

    The committed marker names all five rails: v44 widened it when the object
    API's depth entry took the same store action and landing identity the trace
    contract carries. A capture on a rail the marker names is owed the stored
    depth observation, so the rewritten marker names that rail — the same
    rewriting the scissor and instanced loops apply to their own case. A capture
    on any other rail owes nothing, and its marker has to name another rail
    instead, which keeps the case out of both the required set and the report.
    Returns whether `rail` owes the case.
    """
    suite["render_cases"][DEPTH_STORE_INDEX]["capture_rails"] = (
        [rail] if rail in DEPTH_STORE_RAILS else [other_rail(rail)])
    return rail in DEPTH_STORE_RAILS


def depth_only_marker(suite, rail):
    """Point the v45 case at `rail` when that rail owes it, and elsewhere when not.

    The depth-only case is the stored depth landing with every colour attachment
    discarded, so its marker names the same five rails and the same rule
    applies: a capture on a rail the marker names is owed the stored depth
    observation, and one on any other rail has to leave the case out entirely.
    Returns whether `rail` owes the case.
    """
    suite["render_cases"][DEPTH_ONLY_INDEX]["capture_rails"] = (
        [rail] if rail in DEPTH_ONLY_RAILS else [other_rail(rail)])
    return rail in DEPTH_ONLY_RAILS


def depth_no_colour_marker(suite, rail):
    """Point the v46 case at `rail` when that rail owes it, and elsewhere when not.

    The no-colour case is the stored depth landing with no colour attachment at
    all, so its marker names the same five rails and the same rule applies: a
    capture on a rail the marker names is owed the stored depth observation,
    and one on any other rail has to leave the case out entirely. Returns
    whether `rail` owes the case.
    """
    suite["render_cases"][DEPTH_NO_COLOUR_INDEX]["capture_rails"] = (
        [rail] if rail in DEPTH_NO_COLOUR_RAILS else [other_rail(rail)])
    return rail in DEPTH_NO_COLOUR_RAILS


def stencil_marker(suite, rail):
    """Point the v47 case at `rail` when that rail owes it, and elsewhere when not.

    The stencil case is rail-owned and observed through the colour it leaves
    behind, so the marker rule is the same one the depth fixtures state: a
    capture on a rail the marker names is owed the colour observation, and one
    on any other rail has to leave the case out entirely. The committed marker
    names all five rails: v48 widened it when the Rust native rail's stencil
    path and both object rails took the same surface and test
    (`research/docs/23` §3.3, v48). Returns whether `rail` owes the case.
    """
    suite["render_cases"][STENCIL_INDEX]["capture_rails"] = (
        [rail] if rail in STENCIL_RAILS else [other_rail(rail)])
    return rail in STENCIL_RAILS


def stencil_store_marker(suite, rail):
    """Point the v49 case at `rail` when that rail owes it, and elsewhere when not.

    The v49 case observes two resources: the colour pair's own landing and the
    stored stencil surface's. The committed marker names all five rails — the
    store action and the landing identity travel through the trace contract and
    both object entries exactly as the depth store's do, and the Rust native
    rail's readback carries the bytes — so a capture on a rail the marker names
    is owed both writebacks, and one on any other rail has to leave the case out
    entirely. Returns whether `rail` owes the case.
    """
    suite["render_cases"][STENCIL_STORE_INDEX]["capture_rails"] = (
        [rail] if rail in STENCIL_STORE_RAILS else [other_rail(rail)])
    return rail in STENCIL_STORE_RAILS


def msaa_marker(suite, rail):
    """Point the v51 case at `rail` when that rail owes it, and elsewhere when not.

    The multisample case is the trace rail's first increment, so its committed
    marker names the three trace rails and not the two object rails: a capture
    on a rail the marker names is owed the resolved attachment's own landing,
    and one on any other rail has to leave the case out entirely
    (`research/docs/23` §3.3, v51). Returns whether `rail` owes the case.
    """
    suite["render_cases"][MSAA_INDEX]["capture_rails"] = (
        [rail] if rail in MSAA_RAILS else [other_rail(rail)])
    return rail in MSAA_RAILS


def msaa_depth_marker(suite, rail):
    """Point the v53 case at `rail` when that rail owes it, and elsewhere when not.

    The depth-tested multisample case is the trace rails' first increment, so its
    committed marker names the three trace rails: a capture on a rail the marker
    names is owed the resolved attachment's landing, and one on any other rail
    has to leave the case out entirely (`research/docs/23` §3.3, v53). Returns
    whether `rail` owes the case.
    """
    suite["render_cases"][MSAA_DEPTH_INDEX]["capture_rails"] = (
        [rail] if rail in MSAA_DEPTH_RAILS else [other_rail(rail)])
    return rail in MSAA_DEPTH_RAILS


def msaa_stencil_marker(suite, rail):
    """Point the v55 case at `rail` when that rail owes it, and elsewhere when not.

    The stencil-masked multisample case is the trace rails' first increment, so
    its committed marker names the three trace rails: a capture on a rail the
    marker names is owed the resolved attachment's landing, and one on any other
    rail has to leave the case out entirely (`research/docs/23` §3.3, v55).
    Returns whether `rail` owes the case.
    """
    suite["render_cases"][MSAA_STENCIL_INDEX]["capture_rails"] = (
        [rail] if rail in MSAA_STENCIL_RAILS else [other_rail(rail)])
    return rail in MSAA_STENCIL_RAILS


def msaa_depth_resolve_marker(suite, rail):
    """Point the v57 case at `rail` when that rail owes it, and elsewhere when not.

    The v57c marker named the three trace rails; the v58 recording entry
    (`draw_indexed_primitives_with_multisample_depth_resolve`) carries the same
    resolve on both object rails, so the marker names all five. A capture on a
    rail the marker names is owed both landings — the resolved colour
    attachment and the resolved depth surface — and `sample0` is the API's own
    baseline, so no device mask gates it (`research/docs/23` §3.3, v57c/v58).
    Returns whether `rail` owes the case.
    """
    suite["render_cases"][MSAA_DEPTH_RESOLVE_INDEX]["capture_rails"] = (
        [rail] if rail in MSAA_DEPTH_RESOLVE_RAILS else [other_rail(rail)])
    return rail in MSAA_DEPTH_RESOLVE_RAILS


def msaa_depth_resolve_min_edge_marker(suite, rail):
    """Point the v57d Min edge case at `rail` when that rail owes it, and elsewhere when not.

    The v57f marker names the three trace rails — the v57e self-test measured
    the Apple Paravirtual device executing Min, so the native rails own the
    case — and v58 widens it to every rail, because the object recording entry
    carries the same gated pair. The marker only decides which rail *owns* the
    case, and the device mask then decides presence (`research/docs/23` §3.3,
    v57d/v57f/v58). Returns whether `rail` owes the case.
    """
    suite["render_cases"][MSAA_DEPTH_RESOLVE_MIN_EDGE_INDEX]["capture_rails"] = (
        [rail] if rail in MSAA_DEPTH_RESOLVE_MIN_EDGE_RAILS else [other_rail(rail)])
    return rail in MSAA_DEPTH_RESOLVE_MIN_EDGE_RAILS


def msaa_depth_resolve_max_edge_marker(suite, rail):
    """Point the v57d Max edge case at `rail` when that rail owes it, and elsewhere when not.

    The same v57f-then-v58 marker shape as the Min sibling
    (`research/docs/23` §3.3, v57d/v57f/v58). Returns whether `rail` owes the
    case.
    """
    suite["render_cases"][MSAA_DEPTH_RESOLVE_MAX_EDGE_INDEX]["capture_rails"] = (
        [rail] if rail in MSAA_DEPTH_RESOLVE_MAX_EDGE_RAILS else [other_rail(rail)])
    return rail in MSAA_DEPTH_RESOLVE_MAX_EDGE_RAILS


def msaa_stencil_resolve_sample0_marker(suite, rail):
    """Point the v60 Sample0 case at `rail` when that rail owes it.

    The combined shape is executed by the three trace rails; the object rails
    record no stencil-resolve entry yet, so the marker stays at the trace three
    (`research/docs/23` §3.3, v60). Returns whether `rail` owes the case.
    """
    suite["render_cases"][MSAA_STENCIL_RESOLVE_SAMPLE0_INDEX]["capture_rails"] = (
        [rail] if rail in MSAA_STENCIL_RESOLVE_SAMPLE0_RAILS else [other_rail(rail)])
    return rail in MSAA_STENCIL_RESOLVE_SAMPLE0_RAILS


def msaa_stencil_resolve_drs_marker(suite, rail):
    """Point the v60 DepthResolvedSample case at `rail` when that rail owes it.

    The case names the two native trace rails — Vulkan has no stencil mode for
    the depth-following filter — and the device mask then decides presence
    (`research/docs/23` §3.3, v60). Returns whether `rail` owes the case.
    """
    suite["render_cases"][MSAA_STENCIL_RESOLVE_DRS_INDEX]["capture_rails"] = (
        [rail] if rail in MSAA_STENCIL_RESOLVE_DRS_RAILS else [other_rail(rail)])
    return rail in MSAA_STENCIL_RESOLVE_DRS_RAILS


def msaa_uniform_2x_marker(suite, rail):
    """Point the v61 2x case at `rail` when that rail owes it.

    The recording entry that carries the raster (`draw_indexed_primitives_with_
    multisample`) exists on both object rails, so the marker names all five
    rails and the device's sample-count mask then decides presence
    (`research/docs/23` §3.3, v61). Returns whether `rail` owes the case.
    """
    suite["render_cases"][MSAA_UNIFORM_2X_INDEX]["capture_rails"] = (
        [rail] if rail in MSAA_UNIFORM_2X_RAILS else [other_rail(rail)])
    return rail in MSAA_UNIFORM_2X_RAILS


def msaa_uniform_8x_marker(suite, rail):
    """Point the v61 8x case at `rail` when that rail owes it.

    The same marker shape as the 2x sibling: the marker names every rail and
    the device's sample-count mask decides presence (`research/docs/23` §3.3,
    v61). Returns whether `rail` owes the case.
    """
    suite["render_cases"][MSAA_UNIFORM_8X_INDEX]["capture_rails"] = (
        [rail] if rail in MSAA_UNIFORM_8X_RAILS else [other_rail(rail)])
    return rail in MSAA_UNIFORM_8X_RAILS


def msaa_ds_marker(suite, rail):
    """Point the v66 combined pair at `rail` when that rail owes it.

    The rail-owned combined depth-stencil pair is the trace rails' own
    increment, and from v68 both object rails carry it too: the v68 recording
    entry (`draw_indexed_primitives_with_multisample_depth_stencil`) states
    both faces of the one surface at once, so the committed marker names all
    five rails (`research/docs/23` §3.3, v66/v68). Returns whether `rail` owes
    the case.
    """
    suite["render_cases"][MSAA_DS_INDEX]["capture_rails"] = (
        [rail] if rail in MSAA_DS_RAILS else [other_rail(rail)])
    return rail in MSAA_DS_RAILS


def wildcard_result(provider_backend=True, copy_in=2, copy_out=2):
    result = {
        "id": WILDCARD_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": WILDCARD_REPORTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": WILDCARD_REPORTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def base_vertex_result(provider_backend=True, copy_in=2, copy_out=2):
    result = {
        "id": BASE_VERTEX_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": BASE_VERTEX_EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": BASE_VERTEX_EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def msaa_result(provider_backend=True, copy_in=2, copy_out=2):
    """The v51 landing: the resolve target's own bytes.

    The pass observes one resource — the attachment view the four-sample raster
    resolves into — so the writeback, the allocation image and the provider
    counts are the colour side's own shape (`research/docs/23` §3.3, v51).
    """
    result = {
        "id": MSAA_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": MSAA_EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": MSAA_EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def msaa_uniform_2x_result(provider_backend=True, copy_in=2, copy_out=2):
    """The v61 2x landing: the full-coverage raster's own resolve target.

    The shape is the v51 case's — one colour attachment resolved into its own
    view — so the writeback, the allocation image and the provider counts are
    identical; only the expectation and the device gate change
    (`research/docs/23` §3.3, v61).
    """
    result = {
        "id": MSAA_UNIFORM_2X_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": MSAA_UNIFORM_2X_EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0],
                         "bytes_hex": MSAA_UNIFORM_2X_EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def msaa_uniform_8x_result(provider_backend=True, copy_in=2, copy_out=2):
    """The v61 8x landing: the 2x sibling with the wider raster's own gate."""
    result = {
        "id": MSAA_UNIFORM_8X_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": MSAA_UNIFORM_8X_EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0],
                         "bytes_hex": MSAA_UNIFORM_8X_EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def msaa_ds_result(provider_backend=True, copy_in=2, copy_out=2):
    """The v66 landing: the combined pair's resolved colour attachment.

    Both faces of the combined surface are rail-owned — they disappear with the
    pass — so the pair contributes no writeback and no allocation image, and
    the observation is the colour attachment's resolved texels, exactly as the
    v53 and v55 siblings' is (`research/docs/23` §3.3, v66).
    """
    result = {
        "id": MSAA_DS_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": MSAA_DS_EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": MSAA_DS_EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def msaa_depth_result(provider_backend=True, copy_in=2, copy_out=2):
    """The v53 landing: the depth-tested raster's own resolve target.

    The rail-owned depth surface contributes no writeback and no allocation
    image — it disappears with the pass — so the observation is the colour
    attachment's resolved texels, exactly as the v52 fixture's is
    (`research/docs/23` §3.3, v53).
    """
    result = {
        "id": MSAA_DEPTH_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": MSAA_DEPTH_EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": MSAA_DEPTH_EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def msaa_stencil_result(provider_backend=True, copy_in=2, copy_out=2):
    """The v55 landing: the stencil-masked raster's own resolve target.

    The rail-owned stencil surface contributes no writeback and no allocation
    image — it disappears with the pass — so the observation is the colour
    attachment's resolved texels (`research/docs/23` §3.3, v55).
    """
    result = {
        "id": MSAA_STENCIL_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": MSAA_STENCIL_EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": MSAA_STENCIL_EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def msaa_depth_resolve_result(provider_backend=True, copy_in=3, copy_out=3):
    """The v57 landing: the colour resolve plus the stored depth resolve target.

    Both resources use the same two surfaces the v43 stored depth pair uses —
    one writeback each, in the order the suite fixes, and one allocation image
    each — so a provider capture owes one copy-in and one copy-out for the
    depth allocation on top of the declaring pass's two and one, the stored
    depth pair's own three and three (`research/docs/23` §3.3, v43/v57).
    """
    result = {
        "id": MSAA_DEPTH_RESOLVE_ID,
        "completion": "CompletedVisible",
        "writebacks": [
            {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
             "offset": ATTACHMENT[2], "bytes_hex": MSAA_DEPTH_RESOLVE_EXPECTED},
            {"allocation": DEPTH_STORE_ALLOCATION, "view": DEPTH_STORE_VIEW,
             "offset": 0, "bytes_hex": MSAA_DEPTH_RESOLVE_DEPTH},
        ],
        "allocations": [
            {"allocation": ATTACHMENT[0], "bytes_hex": MSAA_DEPTH_RESOLVE_EXPECTED},
            {"allocation": DEPTH_STORE_ALLOCATION, "bytes_hex": MSAA_DEPTH_RESOLVE_DEPTH},
        ],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def msaa_depth_resolve_min_edge_result(provider_backend=True, copy_in=3, copy_out=3):
    """The v57d Min edge landing: the uniform colour pair plus the resolved depth.

    Both triangles are red, so the colour observation is the near tint sixteen
    times, and the depth landing carries the Min reduction — 0.5 in columns
    0-2, 0.9 in column 3, per row. A provider capture owes the stored depth
    pair's three and three (`research/docs/23` §3.3, v57d).
    """
    result = {
        "id": MSAA_DEPTH_RESOLVE_MIN_EDGE_ID,
        "completion": "CompletedVisible",
        "writebacks": [
            {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
             "offset": ATTACHMENT[2], "bytes_hex": MSAA_DEPTH_RESOLVE_MIN_EDGE_EXPECTED},
            {"allocation": DEPTH_RESOLVE_ALLOCATION, "view": DEPTH_RESOLVE_VIEW,
             "offset": 0, "bytes_hex": MSAA_DEPTH_RESOLVE_MIN_EDGE_DEPTH},
        ],
        "allocations": [
            {"allocation": ATTACHMENT[0],
             "bytes_hex": MSAA_DEPTH_RESOLVE_MIN_EDGE_EXPECTED},
            {"allocation": DEPTH_RESOLVE_ALLOCATION,
             "bytes_hex": MSAA_DEPTH_RESOLVE_MIN_EDGE_DEPTH},
        ],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def msaa_depth_resolve_max_edge_result(provider_backend=True, copy_in=3, copy_out=3):
    """The v57d Max edge landing: the same colour pair plus the Max reduction.

    The Max reduction lands 0.5 in columns 0-1 and 0.9 in columns 2-3, per
    row — the four bytes the Min sibling's own landing leaves at 0.5, which is
    the difference that makes the pair falsifiable (`research/docs/23` §3.3,
    v57d).
    """
    result = {
        "id": MSAA_DEPTH_RESOLVE_MAX_EDGE_ID,
        "completion": "CompletedVisible",
        "writebacks": [
            {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
             "offset": ATTACHMENT[2], "bytes_hex": MSAA_DEPTH_RESOLVE_MAX_EDGE_EXPECTED},
            {"allocation": DEPTH_RESOLVE_ALLOCATION, "view": DEPTH_RESOLVE_VIEW,
             "offset": 0, "bytes_hex": MSAA_DEPTH_RESOLVE_MAX_EDGE_DEPTH},
        ],
        "allocations": [
            {"allocation": ATTACHMENT[0],
             "bytes_hex": MSAA_DEPTH_RESOLVE_MAX_EDGE_EXPECTED},
            {"allocation": DEPTH_RESOLVE_ALLOCATION,
             "bytes_hex": MSAA_DEPTH_RESOLVE_MAX_EDGE_DEPTH},
        ],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def msaa_stencil_resolve_sample0_result(provider_backend=True, copy_in=4, copy_out=4):
    """The v60 Sample0 landing: colour, resolved depth and resolved stencil.

    The combined shape's three landings — the colour pair's near tint, the
    depth resolve's 0.5/0.7 columns and the stencil resolve's 01/00 columns —
    and the declaring pass declares all three resources, so a provider capture
    owes four copies in and four out (`research/docs/23` §3.3, v60).
    """
    result = {
        "id": MSAA_STENCIL_RESOLVE_SAMPLE0_ID,
        "completion": "CompletedVisible",
        "writebacks": [
            {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
             "offset": ATTACHMENT[2], "bytes_hex": MSAA_STENCIL_RESOLVE_COLOUR},
            {"allocation": DEPTH_RESOLVE_ALLOCATION, "view": DEPTH_RESOLVE_VIEW,
             "offset": 0, "bytes_hex": MSAA_STENCIL_RESOLVE_SAMPLE0_DEPTH},
            {"allocation": STENCIL_STORE_ALLOCATION, "view": STENCIL_RESOLVE_VIEW,
             "offset": 0, "bytes_hex": MSAA_STENCIL_RESOLVE_SAMPLE0_STENCIL},
        ],
        "allocations": [
            {"allocation": ATTACHMENT[0], "bytes_hex": MSAA_STENCIL_RESOLVE_COLOUR},
            {"allocation": DEPTH_RESOLVE_ALLOCATION,
             "bytes_hex": MSAA_STENCIL_RESOLVE_SAMPLE0_DEPTH},
            {"allocation": STENCIL_STORE_ALLOCATION,
             "bytes_hex": MSAA_STENCIL_RESOLVE_SAMPLE0_STENCIL},
        ],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def msaa_stencil_resolve_drs_result(provider_backend=True, copy_in=4, copy_out=4):
    """The v60 DepthResolvedSample landing: the Max-selected stencil split.

    The same three landings with the Max depth resolve: columns 2-3 take the
    far sample's stencil 00 and depth 0.7, which is the split that makes the
    filter falsifiable against Sample0 (`research/docs/23` §3.3, v60).
    """
    result = {
        "id": MSAA_STENCIL_RESOLVE_DRS_ID,
        "completion": "CompletedVisible",
        "writebacks": [
            {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
             "offset": ATTACHMENT[2], "bytes_hex": MSAA_STENCIL_RESOLVE_COLOUR},
            {"allocation": DEPTH_RESOLVE_ALLOCATION, "view": DEPTH_RESOLVE_VIEW,
             "offset": 0, "bytes_hex": MSAA_STENCIL_RESOLVE_DRS_DEPTH},
            {"allocation": STENCIL_STORE_ALLOCATION, "view": STENCIL_RESOLVE_VIEW,
             "offset": 0, "bytes_hex": MSAA_STENCIL_RESOLVE_DRS_STENCIL},
        ],
        "allocations": [
            {"allocation": ATTACHMENT[0], "bytes_hex": MSAA_STENCIL_RESOLVE_COLOUR},
            {"allocation": DEPTH_RESOLVE_ALLOCATION,
             "bytes_hex": MSAA_STENCIL_RESOLVE_DRS_DEPTH},
            {"allocation": STENCIL_STORE_ALLOCATION,
             "bytes_hex": MSAA_STENCIL_RESOLVE_DRS_STENCIL},
        ],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def depth_result(provider_backend=True, copy_in=2, copy_out=2):
    result = {
        "id": DEPTH_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": DEPTH_EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": DEPTH_EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def depth_store_result(provider_backend=True, copy_in=3, copy_out=3):
    """The v43 landing: the colour observation plus the stored depth one.

    Both resources use the same two surfaces — one writeback each, in the order
    the suite fixes, and one allocation image each — and a provider capture owes
    one copy-in and one copy-out for the depth allocation on top of the
    declaring pass's two and one, so the counts are three and three.
    """
    result = {
        "id": DEPTH_STORE_ID,
        "completion": "CompletedVisible",
        "writebacks": [
            {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
             "offset": ATTACHMENT[2], "bytes_hex": DEPTH_EXPECTED},
            {"allocation": DEPTH_STORE_ALLOCATION, "view": DEPTH_STORE_VIEW,
             "offset": 0, "bytes_hex": DEPTH_STORE_EXPECTED},
        ],
        "allocations": [
            {"allocation": ATTACHMENT[0], "bytes_hex": DEPTH_EXPECTED},
            {"allocation": DEPTH_STORE_ALLOCATION, "bytes_hex": DEPTH_STORE_EXPECTED},
        ],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def depth_only_result(provider_backend=True, copy_in=3, copy_out=2):
    """The v45 landing: the stored depth observation and nothing else.

    The colour attachment discards, so no writeback and no allocation image are
    owed for it — a capture that reports one is not the run the suite asked for.
    The depth resource is the pass's whole observation and uses the channel the
    colour side already uses. The declaring pass still reads the colour view,
    the probe and the depth view, and it still writes the probe, so a provider
    capture owes the plan's three touched and two written allocations.
    """
    result = {
        "id": DEPTH_ONLY_ID,
        "completion": "CompletedVisible",
        "writebacks": [
            {"allocation": DEPTH_STORE_ALLOCATION, "view": DEPTH_STORE_VIEW,
             "offset": 0, "bytes_hex": DEPTH_STORE_EXPECTED},
        ],
        "allocations": [
            {"allocation": DEPTH_STORE_ALLOCATION, "bytes_hex": DEPTH_STORE_EXPECTED},
        ],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def depth_no_colour_result(provider_backend=True, copy_in=3, copy_out=2):
    """The v46 landing: the stored depth observation and nothing else.

    The pass binds no colour attachment at all, so the stored depth surface is
    the case's whole observation and uses the channel the colour side already
    uses. The declaring pass still reads the colour view, the probe and the
    depth view, and it still writes the probe, so a provider capture owes the
    plan's three touched and two written allocations.
    """
    result = {
        "id": DEPTH_NO_COLOUR_ID,
        "completion": "CompletedVisible",
        "writebacks": [
            {"allocation": DEPTH_STORE_ALLOCATION, "view": DEPTH_STORE_VIEW,
             "offset": 0, "bytes_hex": DEPTH_STORE_EXPECTED},
        ],
        "allocations": [
            {"allocation": DEPTH_STORE_ALLOCATION, "bytes_hex": DEPTH_STORE_EXPECTED},
        ],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def stencil_result(provider_backend=True, copy_in=2, copy_out=2):
    """The v47 landing: the colour observation and nothing else.

    The stencil attachment is rail-owned — the pass opens it, masks with it and
    lets it disappear — so it owes neither a writeback nor an allocation image,
    and the case's observation is the v36 pair's own colour landing. The
    declaring pass still reads the colour view and the probe and still writes the
    probe, so a provider capture owes the plan's two touched and two written
    allocations exactly as the plain pair case does.
    """
    result = {
        "id": STENCIL_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": STENCIL_EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": STENCIL_EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def stencil_store_result(provider_backend=True, copy_in=3, copy_out=3):
    """The v49 landing: the colour observation plus the stored stencil one.

    Both resources use the same two surfaces — one writeback each, in the order
    the suite fixes, and one allocation image each — and the stencil landing
    covers one byte per texel inside the declaring case's own sixty-four byte
    allocation. A provider capture owes one copy-in and one copy-out for that
    allocation on top of the declaring pass's own two and one, so the counts are
    three and three, the stored depth pair's own numbers
    (`research/docs/23` §3.3, v43/v49).
    """
    result = {
        "id": STENCIL_STORE_ID,
        "completion": "CompletedVisible",
        "writebacks": [
            {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
             "offset": ATTACHMENT[2], "bytes_hex": STENCIL_EXPECTED},
            {"allocation": STENCIL_STORE_ALLOCATION, "view": STENCIL_STORE_VIEW,
             "offset": 0, "bytes_hex": STENCIL_STORE_EXPECTED},
        ],
        "allocations": [
            {"allocation": ATTACHMENT[0], "bytes_hex": STENCIL_EXPECTED},
            {"allocation": STENCIL_STORE_ALLOCATION,
             "bytes_hex": STENCIL_STORE_ALLOCATION_BYTES},
        ],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def alignment_result(provider_backend=True, copy_in=2, copy_out=2):
    result = {
        "id": ALIGNMENT_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": ALIGNMENT_EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": ALIGNMENT_EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def cull_result(provider_backend=True, copy_in=2, copy_out=2):
    result = {
        "id": CULL_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": CULL_EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": CULL_EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def blend_result(provider_backend=True, copy_in=2, copy_out=2):
    result = {
        "id": BLEND_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": BLEND_EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": BLEND_EXPECTED}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def counted_declaring(suite, digest, rail, depth_resolve_modes=0,
                      stencil_resolve_modes=0, render_sample_counts=0):
    report = synthetic_report(suite, digest, rail,
                              depth_resolve_modes=depth_resolve_modes,
                              stencil_resolve_modes=stencil_resolve_modes,
                              render_sample_counts=render_sample_counts)
    if rail != "native-metal":
        for result in report["results"]:
            # The v43 declaring case reads one more view than its v27 sibling:
            # it declares the depth attachment's own view beside the colour
            # attachment's, so it touches three allocations and still writes
            # one (`research/docs/23` §3.3, v43). The v49 declaring case is the
            # same shape with the one-byte-per-texel stencil view in place of
            # the depth one (`research/docs/23` §3.3, v49), and the v57d
            # declaring case is the depth shape again with its own landing
            # view (`research/docs/23` §3.3, v57d), and the v60 declaring case
            # declares both landings (`research/docs/23` §3.3, v60).
            result["copy_in"] = (
                4 if result["id"] == STENCIL_RESOLVE_DECLARING_ID
                else 3 if result["id"] in (DEPTH_DECLARING_ID,
                                           STENCIL_DECLARING_ID,
                                           DEPTH_RESOLVE_DECLARING_ID)
                else 2)
            result["copy_out"] = 1
    return report


class ScissorObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = V28_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v28_pins_the_scissored_fixture(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v28")
        case = self.suite["render_cases"][0]
        self.assertEqual(case["id"], RENDER_ID)
        self.assertEqual(case["scissor"], SCISSOR)
        self.assertEqual(case["attachment"]["clear_hex"], CLEAR)
        self.assertEqual(case["expected_hex"], EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(ALL_RAILS))

    def test_v28_pins_the_instanced_fixture(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v28")
        case = self.suite["render_cases"][1]
        self.assertEqual(case["id"], INSTANCED_ID)
        self.assertEqual(case["instance_count"], 2)
        self.assertEqual([stream["step"] for stream in case["vertex_layout"]["buffers"]],
                         ["per_vertex", "per_instance"])
        self.assertEqual(case["attachment"]["clear_hex"], CLEAR)
        self.assertEqual(case["expected_hex"], INSTANCED_EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(INSTANCED_RAILS))

    def test_v28_plan_knows_the_coverage(self):
        plan = compare._suite_plan(self.suite)
        render_plan = compare._render_plan(plan, self.suite)
        expectation = render_plan[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(EXPECTED))])
        self.assertEqual(expectation.touched, {900, 920})
        self.assertEqual(expectation.written, {900, 920})
        instanced = render_plan[INSTANCED_ID]
        self.assertEqual(instanced.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(INSTANCED_EXPECTED))])
        self.assertEqual(instanced.touched, {900, 920})
        self.assertEqual(instanced.written, {900, 920})
        self.assertEqual(instanced.rails, frozenset(INSTANCED_RAILS))

    def test_v28_clips_on_every_rail(self):
        # The scissor case is widened to every rail; the instanced case keeps
        # its own marker, so each rail reports exactly the render cases it owes.
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            suite["render_cases"][0]["capture_rails"] = [rail]
            owes_depth_store = depth_store_marker(suite, rail)
            owes_depth_only = depth_only_marker(suite, rail)
            owes_depth_no_colour = depth_no_colour_marker(suite, rail)
            owes_stencil = stencil_marker(suite, rail)
            owes_stencil_store = stencil_store_marker(suite, rail)
            owes_depth_resolve = msaa_depth_resolve_marker(suite, rail)
            owes_stencil_resolve_sample0 = msaa_stencil_resolve_sample0_marker(suite, rail)
            owes_stencil_resolve_drs = msaa_stencil_resolve_drs_marker(suite, rail)
            owes_msaa_ds = msaa_ds_marker(suite, rail)
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(
                suite, digest, rail,
                depth_resolve_modes=(DEPTH_RESOLVE_ALL_FILTERS_BITS
                                     if rail in MSAA_DEPTH_RESOLVE_MIN_EDGE_RAILS else 0),
                stencil_resolve_modes=(STENCIL_RESOLVE_ALL_FILTERS_BITS
                                       if rail in MSAA_STENCIL_RESOLVE_DRS_RAILS else 0))
            report["results"].append(render_result(rail != "native-metal"))
            if rail in INSTANCED_RAILS:
                report["results"].append(instanced_result(rail != "native-metal"))
            report["results"].append(wildcard_result(rail != "native-metal"))
            if rail in BASE_VERTEX_RAILS:
                report["results"].append(base_vertex_result(rail != "native-metal"))
            if rail in DEPTH_RAILS:
                report["results"].append(depth_result(rail != "native-metal"))
            if owes_depth_store:
                report["results"].append(depth_store_result(rail != "native-metal"))
            if owes_depth_only:
                report["results"].append(depth_only_result(rail != "native-metal"))
            if owes_depth_no_colour:
                report["results"].append(depth_no_colour_result(rail != "native-metal"))
            if owes_stencil:
                report["results"].append(stencil_result(rail != "native-metal"))
            if owes_stencil_store:
                report["results"].append(stencil_store_result(rail != "native-metal"))
            if rail in ALIGNMENT_RAILS:
                report["results"].append(alignment_result(rail != "native-metal"))
            if rail in CULL_RAILS:
                report["results"].append(cull_result(rail != "native-metal"))
            if rail in BLEND_RAILS:
                report["results"].append(blend_result(rail != "native-metal"))
            if rail in MSAA_RAILS:
                report["results"].append(msaa_result(rail != "native-metal"))
            if rail in MSAA_DEPTH_RAILS:
                report["results"].append(msaa_depth_result(rail != "native-metal"))
            if rail in MSAA_STENCIL_RAILS:
                report["results"].append(msaa_stencil_result(rail != "native-metal"))
            if owes_depth_resolve:
                report["results"].append(msaa_depth_resolve_result(rail != "native-metal"))
            if rail in MSAA_DEPTH_RESOLVE_MIN_EDGE_RAILS:
                report["results"].append(
                    msaa_depth_resolve_min_edge_result(rail != "native-metal"))
            if rail in MSAA_DEPTH_RESOLVE_MAX_EDGE_RAILS:
                report["results"].append(
                    msaa_depth_resolve_max_edge_result(rail != "native-metal"))
            if owes_stencil_resolve_sample0:
                report["results"].append(
                    msaa_stencil_resolve_sample0_result(rail != "native-metal"))
            if owes_stencil_resolve_drs:
                report["results"].append(
                    msaa_stencil_resolve_drs_result(rail != "native-metal"))
            if owes_msaa_ds:
                report["results"].append(msaa_ds_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                if not owes_depth_store:
                    self.assertNotIn(DEPTH_STORE_ID,
                                     [result["id"] for result in report["results"]])
                if not owes_depth_only:
                    self.assertNotIn(DEPTH_ONLY_ID,
                                     [result["id"] for result in report["results"]])
                if not owes_depth_no_colour:
                    self.assertNotIn(DEPTH_NO_COLOUR_ID,
                                     [result["id"] for result in report["results"]])
                if not owes_stencil:
                    self.assertNotIn(STENCIL_ID,
                                     [result["id"] for result in report["results"]])
                compare.validate_capture(suite, digest, report, rail)

    def test_v28_instances_on_every_rail(self):
        for rail in INSTANCED_RAILS:
            suite = copy.deepcopy(self.suite)
            # The scissor case names one other rail: this capture owes the
            # instanced result and nothing else.
            suite["render_cases"][0]["capture_rails"] = [other_rail(rail)]
            suite["render_cases"][1]["capture_rails"] = [rail]
            owes_depth_store = depth_store_marker(suite, rail)
            owes_depth_only = depth_only_marker(suite, rail)
            owes_depth_no_colour = depth_no_colour_marker(suite, rail)
            owes_stencil = stencil_marker(suite, rail)
            owes_stencil_store = stencil_store_marker(suite, rail)
            owes_depth_resolve = msaa_depth_resolve_marker(suite, rail)
            owes_stencil_resolve_sample0 = msaa_stencil_resolve_sample0_marker(suite, rail)
            owes_stencil_resolve_drs = msaa_stencil_resolve_drs_marker(suite, rail)
            owes_msaa_ds = msaa_ds_marker(suite, rail)
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(
                suite, digest, rail,
                depth_resolve_modes=(DEPTH_RESOLVE_ALL_FILTERS_BITS
                                     if rail in MSAA_DEPTH_RESOLVE_MIN_EDGE_RAILS else 0),
                stencil_resolve_modes=(STENCIL_RESOLVE_ALL_FILTERS_BITS
                                       if rail in MSAA_STENCIL_RESOLVE_DRS_RAILS else 0))
            report["results"].append(instanced_result(rail != "native-metal"))
            report["results"].append(wildcard_result(rail != "native-metal"))
            if rail in BASE_VERTEX_RAILS:
                report["results"].append(base_vertex_result(rail != "native-metal"))
            if rail in DEPTH_RAILS:
                report["results"].append(depth_result(rail != "native-metal"))
            if owes_depth_store:
                report["results"].append(depth_store_result(rail != "native-metal"))
            if owes_depth_only:
                report["results"].append(depth_only_result(rail != "native-metal"))
            if owes_depth_no_colour:
                report["results"].append(depth_no_colour_result(rail != "native-metal"))
            if owes_stencil:
                report["results"].append(stencil_result(rail != "native-metal"))
            if owes_stencil_store:
                report["results"].append(stencil_store_result(rail != "native-metal"))
            if rail in ALIGNMENT_RAILS:
                report["results"].append(alignment_result(rail != "native-metal"))
            if rail in CULL_RAILS:
                report["results"].append(cull_result(rail != "native-metal"))
            if rail in BLEND_RAILS:
                report["results"].append(blend_result(rail != "native-metal"))
            if rail in MSAA_RAILS:
                report["results"].append(msaa_result(rail != "native-metal"))
            if rail in MSAA_DEPTH_RAILS:
                report["results"].append(msaa_depth_result(rail != "native-metal"))
            if rail in MSAA_STENCIL_RAILS:
                report["results"].append(msaa_stencil_result(rail != "native-metal"))
            if owes_depth_resolve:
                report["results"].append(msaa_depth_resolve_result(rail != "native-metal"))
            if rail in MSAA_DEPTH_RESOLVE_MIN_EDGE_RAILS:
                report["results"].append(
                    msaa_depth_resolve_min_edge_result(rail != "native-metal"))
            if rail in MSAA_DEPTH_RESOLVE_MAX_EDGE_RAILS:
                report["results"].append(
                    msaa_depth_resolve_max_edge_result(rail != "native-metal"))
            if owes_stencil_resolve_sample0:
                report["results"].append(
                    msaa_stencil_resolve_sample0_result(rail != "native-metal"))
            if owes_stencil_resolve_drs:
                report["results"].append(
                    msaa_stencil_resolve_drs_result(rail != "native-metal"))
            if owes_msaa_ds:
                report["results"].append(msaa_ds_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                compare.validate_capture(suite, digest, report, rail)

    def test_v28_refuses_a_rail_whose_marker_does_not_name_it(self):
        # The scissor marker names every rail, so a capture whose marker omits
        # its own rail is refused rather than compared.
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            suite["render_cases"][0]["capture_rails"] = [
                other for other in ALL_RAILS if other != rail]
            owes_depth_store = depth_store_marker(suite, rail)
            owes_depth_only = depth_only_marker(suite, rail)
            owes_depth_no_colour = depth_no_colour_marker(suite, rail)
            owes_stencil = stencil_marker(suite, rail)
            owes_stencil_store = stencil_store_marker(suite, rail)
            owes_depth_resolve = msaa_depth_resolve_marker(suite, rail)
            owes_stencil_resolve_sample0 = msaa_stencil_resolve_sample0_marker(suite, rail)
            owes_stencil_resolve_drs = msaa_stencil_resolve_drs_marker(suite, rail)
            owes_msaa_ds = msaa_ds_marker(suite, rail)
            msaa_depth_resolve_min_edge_marker(suite, rail)
            msaa_depth_resolve_max_edge_marker(suite, rail)
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(
                suite, digest, rail,
                depth_resolve_modes=DEPTH_RESOLVE_ALL_FILTERS_BITS,
                stencil_resolve_modes=(STENCIL_RESOLVE_ALL_FILTERS_BITS
                                       if rail in MSAA_STENCIL_RESOLVE_DRS_RAILS else 0))
            report["results"].append(render_result(rail != "native-metal"))
            if rail in INSTANCED_RAILS:
                report["results"].append(instanced_result(rail != "native-metal"))
            report["results"].append(wildcard_result(rail != "native-metal"))
            if rail in BASE_VERTEX_RAILS:
                report["results"].append(base_vertex_result(rail != "native-metal"))
            if rail in DEPTH_RAILS:
                report["results"].append(depth_result(rail != "native-metal"))
            if owes_depth_store:
                report["results"].append(depth_store_result(rail != "native-metal"))
            if owes_depth_only:
                report["results"].append(depth_only_result(rail != "native-metal"))
            if owes_depth_no_colour:
                report["results"].append(depth_no_colour_result(rail != "native-metal"))
            if owes_stencil:
                report["results"].append(stencil_result(rail != "native-metal"))
            if owes_stencil_store:
                report["results"].append(stencil_store_result(rail != "native-metal"))
            if rail in ALIGNMENT_RAILS:
                report["results"].append(alignment_result(rail != "native-metal"))
            if rail in CULL_RAILS:
                report["results"].append(cull_result(rail != "native-metal"))
            if rail in BLEND_RAILS:
                report["results"].append(blend_result(rail != "native-metal"))
            if rail in MSAA_RAILS:
                report["results"].append(msaa_result(rail != "native-metal"))
            if rail in MSAA_DEPTH_RAILS:
                report["results"].append(msaa_depth_result(rail != "native-metal"))
            if rail in MSAA_STENCIL_RAILS:
                report["results"].append(msaa_stencil_result(rail != "native-metal"))
            if owes_depth_resolve:
                report["results"].append(msaa_depth_resolve_result(rail != "native-metal"))
            if owes_stencil_resolve_sample0:
                report["results"].append(
                    msaa_stencil_resolve_sample0_result(rail != "native-metal"))
            if owes_stencil_resolve_drs:
                report["results"].append(
                    msaa_stencil_resolve_drs_result(rail != "native-metal"))
            if owes_msaa_ds:
                report["results"].append(msaa_ds_result(rail != "native-metal"))
            if rail in MSAA_DEPTH_RESOLVE_MIN_EDGE_RAILS:
                report["results"].append(
                    msaa_depth_resolve_min_edge_result(rail != "native-metal"))
            if rail in MSAA_DEPTH_RESOLVE_MAX_EDGE_RAILS:
                report["results"].append(
                    msaa_depth_resolve_max_edge_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                with self.assertRaisesRegex(compare.CaptureError,
                                            "is not a rail this render case runs on"):
                    compare.validate_capture(suite, digest, report, rail)

    def test_v28_pins_the_wildcard_fixture(self):
        case = self.suite["render_cases"][2]
        self.assertEqual(case["id"], WILDCARD_ID)
        self.assertEqual(case["attachment"]["load"], "dontcare")
        self.assertEqual(case["scissor"], SCISSOR)
        self.assertEqual(case["wildcard_texels"], WILDCARD_TEXELS)
        self.assertEqual(case["expected_hex"], WILDCARD_EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(ALL_RAILS))

    def test_v28_wildcards_skip_exactly_the_unclaimed_texels(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[WILDCARD_ID]
        mask = expectation.wildcards[(ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2])]
        for texel in WILDCARD_TEXELS:
            for byte in range(4):
                self.assertIn(texel * 4 + byte, mask)
        for texel in (0, 1, 4, 5, 8, 9, 12, 13):
            for byte in range(4):
                self.assertNotIn(texel * 4 + byte, mask)
        # The synthetic report carries `cdcdcdcd` at every wildcard texel, so a
        # comparator that compared them would fail here.
        digest = hashlib.sha256(
            json.dumps(self.suite, sort_keys=True).encode("utf-8")).hexdigest()
        report = counted_declaring(self.suite, digest, "vulkan",
                                   render_sample_counts=1 << 1 | 1 << 3)
        report["results"].append(render_result())
        report["results"].append(instanced_result())
        report["results"].append(wildcard_result())
        report["results"].append(base_vertex_result())
        report["results"].append(depth_result())
        report["results"].append(depth_store_result())
        report["results"].append(depth_only_result())
        report["results"].append(depth_no_colour_result())
        report["results"].append(stencil_result())
        report["results"].append(stencil_store_result())
        report["results"].append(alignment_result())
        report["results"].append(cull_result())
        report["results"].append(blend_result())
        report["results"].append(msaa_result())
        report["results"].append(msaa_depth_result())
        report["results"].append(msaa_stencil_result())
        report["results"].append(msaa_depth_resolve_result())
        report["results"].append(msaa_stencil_resolve_sample0_result())
        report["results"].append(msaa_uniform_2x_result())
        report["results"].append(msaa_uniform_8x_result())
        report["results"].append(msaa_ds_result())
        compare.validate_capture(self.suite, digest, report, "vulkan")

    def test_v28_pins_the_base_vertex_fixture(self):
        case = self.suite["render_cases"][3]
        self.assertEqual(case["id"], BASE_VERTEX_ID)
        self.assertEqual(case["base_vertex"], 1)
        self.assertEqual(case["vertex_buffers"][0]["length"], 40)
        self.assertEqual(case["attachment"]["clear_hex"], CLEAR)
        self.assertEqual(case["expected_hex"], BASE_VERTEX_EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(BASE_VERTEX_RAILS))

    def test_v28_plans_the_base_vertex_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[BASE_VERTEX_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(BASE_VERTEX_EXPECTED))])
        self.assertEqual(expectation.rails, frozenset(BASE_VERTEX_RAILS))
        self.assertEqual(expectation.wildcards, {})

    def test_v28_refuses_a_base_vertex_the_stream_cannot_cover(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][3]["base_vertex"] = 2
        with self.assertRaisesRegex(compare.CaptureError,
                                    "offsets its indices by 1"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_wrong_base_vertex_stream(self):
        broken = copy.deepcopy(self.suite)
        # Drop the degenerate centre: the stream is then the plain reviewed
        # quad and the offset would read past it.
        broken["render_cases"][3]["vertex_buffers"][0]["length"] = 32
        broken["render_cases"][3]["vertex_buffers"][0]["initial_hex"] = (
            "000080bf000080bf0000803f000080bf000080bf0000803f0000803f0000803f")
        with self.assertRaisesRegex(compare.CaptureError,
                                    "degenerate centre plus the quad"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_base_vertex_on_the_plain_quad(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["base_vertex"] = 1
        with self.assertRaisesRegex(compare.CaptureError,
                                    "degenerate centre plus the quad"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_pins_the_depth_fixture(self):
        case = self.suite["render_cases"][4]
        self.assertEqual(case["id"], DEPTH_ID)
        self.assertEqual(case["depth"]["format"], "depth32float")
        self.assertEqual(case["depth"]["clear_depth"], 1.0)
        self.assertEqual(case["depth_test"], {"compare": "less", "write": True})
        self.assertEqual(case["attachment"]["clear_hex"], CLEAR)
        self.assertEqual(case["expected_hex"], DEPTH_EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(DEPTH_RAILS))

    def test_v28_plans_the_depth_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[DEPTH_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(DEPTH_EXPECTED))])
        self.assertEqual(expectation.rails, frozenset(DEPTH_RAILS))

    def test_v28_refuses_a_depth_state_that_is_not_the_reviewed_one(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][4]["depth_test"]["compare"] = "always"
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the reviewed depth state is a less test"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_depth_attachment_that_is_not_the_reviewed_one(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][4]["depth"]["clear_depth"] = 0.5
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the reviewed depth clear is one"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_depth_extent_mismatch(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][4]["depth"]["width"] = 2
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the depth attachment has to match the colour extent"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_pins_the_render_case_order(self):
        # Every per-case assertion below addresses the fixture by position, so
        # the reviewed order is pinned here.
        self.assertEqual([case["id"] for case in self.suite["render_cases"]],
                         list(REVIEWED_ORDER))

    def test_v28_pins_the_depth_store_fixture(self):
        case = self.suite["render_cases"][DEPTH_STORE_INDEX]
        self.assertEqual(case["id"], DEPTH_STORE_ID)
        self.assertEqual(case["depth"]["store"], "store")
        self.assertEqual(case["depth"]["allocation"], DEPTH_STORE_ALLOCATION)
        self.assertEqual(case["depth"]["view"], DEPTH_STORE_VIEW)
        self.assertEqual(len(case["depth"]["expected_hex"]), 128)
        self.assertEqual(case["depth"]["expected_hex"], DEPTH_STORE_EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(DEPTH_STORE_RAILS))
        # The colour side is the v36 pair's: the near red triangle still wins.
        self.assertEqual(case["expected_hex"], DEPTH_EXPECTED)
        self.assertEqual(case["expected_hex"],
                         self.suite["render_cases"][4]["expected_hex"])

    def test_v28_plans_the_depth_store_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[DEPTH_STORE_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(DEPTH_EXPECTED)),
                          ((DEPTH_STORE_ALLOCATION, DEPTH_STORE_VIEW, 0),
                           bytes.fromhex(DEPTH_STORE_EXPECTED))])
        self.assertEqual(expectation.allocations,
                         {ATTACHMENT[0]: bytes.fromhex(DEPTH_EXPECTED),
                          DEPTH_STORE_ALLOCATION: bytes.fromhex(DEPTH_STORE_EXPECTED)})
        # Both landings are observed, so the case's identity list covers the
        # colour resource and the depth resource and nothing else.
        self.assertEqual(list(expectation.attachment),
                         [ATTACHMENT, DEPTH_STORE_ATTACHMENT])
        self.assertEqual(expectation.touched, {900, 920, 940})
        self.assertEqual(expectation.written, {900, 920, 940})
        self.assertEqual(expectation.rails, frozenset(DEPTH_STORE_RAILS))

    def test_v28_counts_the_depth_store_and_its_neighbours(self):
        # The stored depth resource is one more touched and one more written
        # allocation behind the provider counts (three and three); the v45 and
        # v46 cases touch the same three but write only the probe and the depth
        # surface — the v45 colour attachment discards, and the v46 pass binds
        # none at all. The v49 stencil store is the stored depth pair's own
        # shape one byte wide: the colour attachment still stores, so the case
        # owes the stencil landing's one touched and one written allocation on
        # top of the declaring pass's two and two (`research/docs/23` §3.3,
        # v43/v49). Every other render case keeps the declaring pass's two and
        # two.
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        for case in self.suite["render_cases"]:
            expectation = plan[case["id"]]
            counts = (len(expectation.touched), len(expectation.written))
            if case["id"] in (DEPTH_STORE_ID, STENCIL_STORE_ID, MSAA_DEPTH_RESOLVE_ID,
                              MSAA_DEPTH_RESOLVE_MIN_EDGE_ID,
                              MSAA_DEPTH_RESOLVE_MAX_EDGE_ID):
                expected = (3, 3)
            elif case["id"] in (MSAA_STENCIL_RESOLVE_SAMPLE0_ID,
                                MSAA_STENCIL_RESOLVE_DRS_ID):
                # The combined shape's two extra landings touch and write two
                # more allocations on top of the declaring pass's two and two
                # (`research/docs/23` §3.3, v60).
                expected = (4, 4)
            elif case["id"] in (DEPTH_ONLY_ID, DEPTH_NO_COLOUR_ID):
                expected = (3, 2)
            else:
                expected = (2, 2)
            with self.subTest(case=case["id"]):
                self.assertEqual(counts, expected)

    def test_v28_reports_the_depth_store_on_every_rail_its_marker_names(self):
        # The marker is the rule, whichever rails it names: a capture on a rail
        # the fixture's marker names is owed the stored depth observation, and a
        # capture on any other rail has to leave the case out. v44 widened the
        # marker to all five rails, because the object API's depth entry now
        # records the same store action and landing identity.
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            owes = depth_store_marker(suite, rail)
            for position, case in enumerate(suite["render_cases"]):
                if position != DEPTH_STORE_INDEX:
                    case["capture_rails"] = [other_rail(rail)]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(depth_store_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                if owes:
                    compare.validate_capture(suite, digest, report, rail)
                else:
                    with self.assertRaisesRegex(
                            compare.CaptureError,
                            "is not a rail this render case runs on"):
                        compare.validate_capture(suite, digest, report, rail)

    def test_v28_reports_the_depth_only_store_on_every_rail_its_marker_names(self):
        # The v45 case follows the same marker rule: a capture on a rail the
        # fixture's marker names is owed its stored depth observation, and a
        # capture on any other rail has to leave the case out.
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            owes = depth_only_marker(suite, rail)
            for position, case in enumerate(suite["render_cases"]):
                if position != DEPTH_ONLY_INDEX:
                    case["capture_rails"] = [other_rail(rail)]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(depth_only_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                if owes:
                    compare.validate_capture(suite, digest, report, rail)
                else:
                    with self.assertRaisesRegex(
                            compare.CaptureError,
                            "is not a rail this render case runs on"):
                        compare.validate_capture(suite, digest, report, rail)

    def test_v28_compares_the_stored_depth_texels(self):
        # The depth landing goes through the same byte comparison the colour
        # side uses, so a capture that reports the clear depth instead of the
        # stored texels is refused at the first differing byte.
        suite = copy.deepcopy(self.suite)
        for position, case in enumerate(suite["render_cases"]):
            if position != DEPTH_STORE_INDEX:
                case["capture_rails"] = ["native-metal"]
        suite["render_cases"][DEPTH_STORE_INDEX]["capture_rails"] = ["vulkan"]
        digest = hashlib.sha256(
            json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
        report = counted_declaring(suite, digest, "vulkan")
        broken = depth_store_result()
        broken["writebacks"][1]["bytes_hex"] = DEPTH_STORE_CLEAR
        broken["allocations"][1]["bytes_hex"] = DEPTH_STORE_CLEAR
        report["results"].append(broken)
        with self.assertRaisesRegex(compare.CaptureError, "first differing byte at offset 2"):
            compare.validate_capture(suite, digest, report, "vulkan")
        # The reported landing the suite declares is the one that passes.
        report["results"][-1] = depth_store_result()
        compare.validate_capture(suite, digest, report, "vulkan")

    def test_v28_refuses_a_depth_expectation_that_is_the_clear_depth(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][DEPTH_STORE_INDEX]["depth"]["expected_hex"] = DEPTH_STORE_CLEAR
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the expected depth texels equal the clear depth"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_depth_identity_without_a_store_action(self):
        broken = copy.deepcopy(self.suite)
        del broken["render_cases"][DEPTH_STORE_INDEX]["depth"]["store"]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a discarded depth attachment carries no identity or expectation"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_store_action_without_its_identity(self):
        broken = copy.deepcopy(self.suite)
        del broken["render_cases"][DEPTH_STORE_INDEX]["depth"]["expected_hex"]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a stored depth attachment needs its store action, its identity"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_depth_resource_that_is_the_colour_attachment(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][DEPTH_STORE_INDEX]["depth"]["allocation"] = ATTACHMENT[0]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "the depth resource has to differ from the colour attachment"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_pins_the_depth_only_fixture(self):
        case = self.suite["render_cases"][DEPTH_ONLY_INDEX]
        self.assertEqual(case["id"], DEPTH_ONLY_ID)
        # The colour attachment is still the reviewed cleared 4x4 surface and the
        # pass still renders into it; the v45 increment only drops its bytes, so
        # the case carries no colour expectation at all.
        self.assertEqual(case["attachment"],
                         {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                          "format": "rgba8_unorm", "width": 4, "height": 4,
                          "load": "clear", "clear_hex": CLEAR, "store": "dontcare"})
        self.assertNotIn("expected_hex", case)
        # The depth section is the v43 fixture's, down to the texels the store
        # hands back, so both cases observe the same landing.
        store = self.suite["render_cases"][DEPTH_STORE_INDEX]
        self.assertEqual(case["depth"], store["depth"])
        self.assertEqual(case["depth"]["allocation"], DEPTH_STORE_ALLOCATION)
        self.assertEqual(case["depth"]["view"], DEPTH_STORE_VIEW)
        self.assertEqual(case["depth"]["expected_hex"], DEPTH_STORE_EXPECTED)
        # Everything else — the module, the layout, the streams, the indices, the
        # viewport and the depth state — is the stored pair's own shape.
        for field in ("declaring_case", "vertex_entry", "fragment_entry", "metal",
                      "vertex_layout", "vertex_buffers", "indices", "vertices",
                      "viewport", "depth_test"):
            self.assertEqual(case[field], store[field])
        self.assertEqual(sorted(case["capture_rails"]), sorted(DEPTH_ONLY_RAILS))

    def test_v28_plans_the_depth_only_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[DEPTH_ONLY_ID]
        # The stored depth surface is the case's whole observation: one
        # writeback and one allocation image under the depth view, and nothing
        # for the colour attachment the pass discards.
        self.assertEqual(expectation.writes,
                         [((DEPTH_STORE_ALLOCATION, DEPTH_STORE_VIEW, 0),
                           bytes.fromhex(DEPTH_STORE_EXPECTED))])
        self.assertEqual(expectation.allocations,
                         {DEPTH_STORE_ALLOCATION: bytes.fromhex(DEPTH_STORE_EXPECTED)})
        # The discarded colour attachment still resolves against the declaring
        # case and still counts as touched, but it is not written.
        self.assertEqual(expectation.touched, {900, 920, 940})
        self.assertEqual(expectation.written, {920, 940})
        # One landing means the identity list is that one attachment rather than
        # the stored pair's two.
        self.assertEqual(expectation.attachment, DEPTH_STORE_ATTACHMENT)
        self.assertEqual(expectation.rails, frozenset(DEPTH_ONLY_RAILS))

    def test_v28_refuses_a_depth_only_case_that_drops_its_landing(self):
        # The one landing the case declares is its stored depth surface: a
        # fixture that drops the store action while the identity and expectation
        # stay behind is the half-declared shape the depth parser refuses.
        broken = copy.deepcopy(self.suite)
        del broken["render_cases"][DEPTH_ONLY_INDEX]["depth"]["store"]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a discarded depth attachment carries no identity or expectation"):
            compare._render_plan(compare._suite_plan(broken), broken)
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][DEPTH_ONLY_INDEX]["depth"]["store"] = "dontcare"
        with self.assertRaisesRegex(compare.CaptureError,
                                    "unsupported depth store op 'dontcare'"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_colour_expectation_on_the_depth_only_case(self):
        # The case-level expectation is the single-attachment shape's claim about
        # stored colour bytes, so the depth-only case must not carry one: the
        # bytes it would claim are the ones the pass discards
        # (`research/docs/23` §3.3, v45).
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][DEPTH_ONLY_INDEX]["expected_hex"] = OUTPUT * 16
        with self.assertRaisesRegex(compare.CaptureError,
                                    "a discarded attachment carries no expectation"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_colour_landing_the_depth_only_case_discards(self):
        # A capture that reports the colour attachment as well has not run the
        # case the suite declares: the depth texels are the whole observation and
        # the extra surface is refused.
        suite = copy.deepcopy(self.suite)
        for position, case in enumerate(suite["render_cases"]):
            if position != DEPTH_ONLY_INDEX:
                case["capture_rails"] = ["native-metal"]
        suite["render_cases"][DEPTH_ONLY_INDEX]["capture_rails"] = ["vulkan"]
        digest = hashlib.sha256(
            json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
        report = counted_declaring(suite, digest, "vulkan")
        broken = depth_only_result()
        broken["writebacks"].insert(0, {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                                        "offset": ATTACHMENT[2],
                                        "bytes_hex": DEPTH_EXPECTED})
        broken["allocations"].insert(0, {"allocation": ATTACHMENT[0],
                                         "bytes_hex": DEPTH_EXPECTED})
        report["results"].append(broken)
        with self.assertRaisesRegex(compare.CaptureError, "writable set mismatch"):
            compare.validate_capture(suite, digest, report, "vulkan")
        # The observation the case does declare is the one that passes.
        report["results"][-1] = depth_only_result()
        compare.validate_capture(suite, digest, report, "vulkan")

    def test_v28_refuses_a_present_action_on_the_depth_only_case(self):
        # A present action hands a *stored* colour attachment on, and the v45
        # case discards its own, so the shape cannot carry one: the depth texels
        # are the whole observation and there is nothing on the colour side to
        # present (`research/docs/24` §5.3). The case is an indexed vertex-input
        # case, so the reviewed vertex-input rule refuses it first; the depth
        # parser's own "a present action needs a stored colour attachment" check
        # is the rule for any shape that reaches it.
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][DEPTH_ONLY_INDEX]["present"] = {
            "mode": "fifo", "image_count": 1, "acquire": 1, "present": 1,
            "initial_hex": "efefefef"}
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a vertex-input case carries neither a present action nor an ICB"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_pins_the_depth_no_colour_fixture(self):
        case = self.suite["render_cases"][DEPTH_NO_COLOUR_INDEX]
        self.assertEqual(case["id"], DEPTH_NO_COLOUR_ID)
        # The v46 shape binds no colour attachment at all: neither the single
        # attachment object nor the MRT list appears, and there is no colour
        # expectation for either shape to carry.
        self.assertNotIn("attachment", case)
        self.assertNotIn("attachments", case)
        self.assertNotIn("expected_hex", case)
        # The reviewed module is the no-output pair — a vertex stage plus a
        # `fragment void` stage — so the pass produces depth and nothing else.
        self.assertEqual(case["metal"]["path"], "shaders/depth_only_4x4.metal")
        self.assertEqual(case["vertex_entry"], "render_depth_only_vertex")
        self.assertEqual(case["fragment_entry"], "render_depth_only_fragment")
        # The depth section is the v45 case's, down to the texels the store
        # hands back, so both cases observe the same landing.
        store = self.suite["render_cases"][DEPTH_ONLY_INDEX]
        self.assertEqual(case["depth"], store["depth"])
        self.assertEqual(case["depth"]["store"], "store")
        self.assertEqual(case["depth"]["allocation"], DEPTH_STORE_ALLOCATION)
        self.assertEqual(case["depth"]["view"], DEPTH_STORE_VIEW)
        self.assertEqual(case["depth"]["expected_hex"], DEPTH_STORE_EXPECTED)
        # Everything else — the layout, the streams, the indices, the viewport
        # and the depth state — is the stored pair's own shape.
        for field in ("declaring_case", "vertex_layout", "vertex_buffers", "indices",
                      "vertices", "viewport", "depth_test"):
            self.assertEqual(case[field], store[field])
        self.assertEqual(sorted(case["capture_rails"]), sorted(DEPTH_NO_COLOUR_RAILS))

    def test_v28_plans_the_depth_no_colour_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[DEPTH_NO_COLOUR_ID]
        # The stored depth surface is the case's whole observation: one
        # writeback and one allocation image under the depth view, and nothing
        # on the colour side, because there is no colour attachment.
        self.assertEqual(expectation.writes,
                         [((DEPTH_STORE_ALLOCATION, DEPTH_STORE_VIEW, 0),
                           bytes.fromhex(DEPTH_STORE_EXPECTED))])
        self.assertEqual(expectation.allocations,
                         {DEPTH_STORE_ALLOCATION: bytes.fromhex(DEPTH_STORE_EXPECTED)})
        # The declaring pass still reads the colour view, the probe and the
        # depth view and still writes the probe and the depth surface, so the
        # counts are the v45 case's own (three touched, two written).
        self.assertEqual(expectation.touched, {900, 920, 940})
        self.assertEqual(expectation.written, {920, 940})
        # One landing means the identity list is that one attachment.
        self.assertEqual(list(expectation.attachment), [DEPTH_STORE_ATTACHMENT])
        self.assertEqual(expectation.rails, frozenset(DEPTH_NO_COLOUR_RAILS))

    def test_v28_reports_the_depth_no_colour_store_on_every_rail_its_marker_names(self):
        # The v46 case follows the same marker rule: a capture on a rail the
        # fixture's marker names is owed its stored depth observation, and a
        # capture on any other rail has to leave the case out.
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            owes = depth_no_colour_marker(suite, rail)
            for position, case in enumerate(suite["render_cases"]):
                if position != DEPTH_NO_COLOUR_INDEX:
                    case["capture_rails"] = [other_rail(rail)]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(depth_no_colour_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                if owes:
                    compare.validate_capture(suite, digest, report, rail)
                else:
                    with self.assertRaisesRegex(
                            compare.CaptureError,
                            "is not a rail this render case runs on"):
                        compare.validate_capture(suite, digest, report, rail)

    def test_v28_refuses_a_depth_no_colour_case_that_drops_its_landing(self):
        # The stored depth surface is the case's one landing. Without it the
        # pass observes nothing at all, and the plan refuses the shape exactly
        # as the all-discarded colour arm does.
        broken = copy.deepcopy(self.suite)
        depth = broken["render_cases"][DEPTH_NO_COLOUR_INDEX]["depth"]
        for field in ("store", "allocation", "view", "expected_hex"):
            del depth[field]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "every colour attachment discards, leaving no observable landing point"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_half_declared_depth_landing_on_the_depth_no_colour_case(self):
        # Dropping only the action leaves the identity and the expectation
        # behind: the depth parser's own half-declaration rule refuses it.
        broken = copy.deepcopy(self.suite)
        del broken["render_cases"][DEPTH_NO_COLOUR_INDEX]["depth"]["store"]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a discarded depth attachment carries no identity or expectation"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_colour_expectation_on_the_depth_no_colour_case(self):
        # The case-level expectation is the single-attachment shape's claim
        # about stored colour bytes, and the v46 pass binds no colour
        # attachment to store them, so the shape cannot carry one
        # (`research/docs/23` §3.3, v46).
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][DEPTH_NO_COLOUR_INDEX]["expected_hex"] = OUTPUT * 16
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a case with no colour attachment carries no expectation"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_allows_the_discarding_colour_shape_on_the_depth_no_colour_case(self):
        # Re-adding the v45 shape — one cleared colour attachment that discards
        # — turns the case back into `depth_only_store_4x4`'s own form, which
        # the suite already reviews: the plan accepts it and the observation is
        # unchanged.
        suite = copy.deepcopy(self.suite)
        suite["render_cases"][DEPTH_NO_COLOUR_INDEX]["attachment"] = copy.deepcopy(
            self.suite["render_cases"][DEPTH_ONLY_INDEX]["attachment"])
        plan = compare._render_plan(compare._suite_plan(suite), suite)
        expectation = plan[DEPTH_NO_COLOUR_ID]
        self.assertEqual(expectation.writes,
                         [((DEPTH_STORE_ALLOCATION, DEPTH_STORE_VIEW, 0),
                           bytes.fromhex(DEPTH_STORE_EXPECTED))])
        self.assertEqual(expectation.touched, {900, 920, 940})
        self.assertEqual(expectation.written, {920, 940})

    def test_v28_refuses_a_depth_no_colour_case_whose_viewport_misses_the_depth_extent(self):
        # The zero-colour shape binds no colour attachment to state the pass's
        # render area, so the stored depth surface is the raster the case's
        # viewport has to cover — the rule the Swift oracle's own zero-colour
        # branch already states, now mirrored here (`research/docs/23` §3.3,
        # v46/v50).
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][DEPTH_NO_COLOUR_INDEX]["viewport"] = [0, 0, 3, 4]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "the viewport must cover the depth attachment"):
            compare._render_plan(compare._suite_plan(broken), broken)
        # The colour-carrying shape keeps its own spelling of the rule: the
        # viewport covers the *colour* attachment, and the stored depth surface
        # follows that extent rather than stating it.
        colour = copy.deepcopy(self.suite)
        colour["render_cases"][DEPTH_ONLY_INDEX]["viewport"] = [0, 0, 3, 4]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the viewport must cover the attachment"):
            compare._render_plan(compare._suite_plan(colour), colour)

    def test_v28_refuses_a_depth_no_colour_case_whose_scissor_leaves_the_depth_extent(self):
        # With no colour attachment to clip, the scissor is measured against the
        # stored depth raster (`research/docs/23` §3.3, v46/v50): a rectangle
        # that leaves that extent — on either axis, by its origin or its size —
        # or one with no area at all describes coverage the case's one landing
        # cannot show.
        for scissor in ([0, 0, 5, 4], [2, 0, 3, 4], [0, 0, 4, 0], [3, 3, 1, 2]):
            broken = copy.deepcopy(self.suite)
            broken["render_cases"][DEPTH_NO_COLOUR_INDEX]["scissor"] = scissor
            with self.subTest(scissor=scissor), self.assertRaisesRegex(
                    compare.CaptureError,
                    "a scissor has to be a non-empty rectangle inside the depth "
                    "attachment"):
                compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_accepts_a_depth_no_colour_case_whose_scissor_stays_inside_the_depth_extent(self):
        # A scissor inside the stored depth raster is the reviewed shape, and it
        # leaves the case's observation alone: the depth texels are still the
        # whole landing, because there is no colour surface for the rectangle to
        # classify (`research/docs/23` §3.3, v46/v50).
        suite = copy.deepcopy(self.suite)
        suite["render_cases"][DEPTH_NO_COLOUR_INDEX]["scissor"] = [1, 0, 2, 4]
        plan = compare._render_plan(compare._suite_plan(suite), suite)
        expectation = plan[DEPTH_NO_COLOUR_ID]
        self.assertEqual(expectation.writes,
                         [((DEPTH_STORE_ALLOCATION, DEPTH_STORE_VIEW, 0),
                           bytes.fromhex(DEPTH_STORE_EXPECTED))])
        self.assertEqual(expectation.allocations,
                         {DEPTH_STORE_ALLOCATION: bytes.fromhex(DEPTH_STORE_EXPECTED)})
        self.assertEqual(expectation.touched, {900, 920, 940})
        self.assertEqual(expectation.written, {920, 940})
        self.assertEqual(list(expectation.attachment), [DEPTH_STORE_ATTACHMENT])
        self.assertEqual(expectation.rails, frozenset(DEPTH_NO_COLOUR_RAILS))
        # The whole-raster rectangle is the shape the fixture's own viewport
        # states, so it is inside the depth extent and accepted too.
        whole = copy.deepcopy(self.suite)
        whole["render_cases"][DEPTH_NO_COLOUR_INDEX]["scissor"] = [0, 0, 4, 4]
        compare._render_plan(compare._suite_plan(whole), whole)

    def test_v28_pins_the_stencil_fixture(self):
        case = self.suite["render_cases"][STENCIL_INDEX]
        self.assertEqual(case["id"], STENCIL_ID)
        # The surface is a cleared `stencil8` attachment whose extent is the
        # colour attachment's, and the state is the reviewed `equal 0` test
        # whose pass op increments and wraps.
        self.assertEqual(case["stencil"], STENCIL_SECTION)
        self.assertEqual(case["stencil_test"], STENCIL_TEST)
        # The module, the layout, the streams, the indices, the viewport and the
        # colour landing are the v36 pair's own: the increment changes the state
        # that decides which triangle survives, not the geometry or the
        # observation surface.
        pair = self.suite["render_cases"][4]
        for field in ("declaring_case", "vertex_entry", "fragment_entry", "metal",
                      "vertex_layout", "vertex_buffers", "indices", "vertices",
                      "viewport", "attachment"):
            self.assertEqual(case[field], pair[field])
        # The near (red) tint is the expectation sixteen times over: a rail that
        # ignored the test, the reference or the pass op would let the far
        # (green) triangle cover the attachment instead.
        self.assertEqual(case["expected_hex"], STENCIL_EXPECTED)
        self.assertEqual(INSTANCE_TINTS[0], "ff0000ff")
        self.assertNotEqual(INSTANCE_TINTS[0], INSTANCE_TINTS[1])
        self.assertEqual(sorted(case["capture_rails"]), sorted(STENCIL_RAILS))

    def test_v28_plans_the_stencil_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[STENCIL_ID]
        # The rail-owned stencil surface is not an allocation of the pass, so
        # the case observes the colour attachment and nothing else and its plan
        # is the v36 pair's own: one writeback, one allocation image, and the
        # declaring pass's two touched and two written allocations.
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(STENCIL_EXPECTED))])
        self.assertEqual(expectation.allocations,
                         {ATTACHMENT[0]: bytes.fromhex(STENCIL_EXPECTED)})
        self.assertEqual(expectation.touched, {900, 920})
        self.assertEqual(expectation.written, {900, 920})
        self.assertEqual(expectation.attachment, ATTACHMENT)
        self.assertEqual(expectation.rails, frozenset(STENCIL_RAILS))

    def test_v28_reports_the_stencil_case_on_every_rail_its_marker_names(self):
        # The marker is the rule, whichever rails it names: a capture on a rail
        # the fixture's marker names is owed the colour observation the stencil
        # state produced, and a capture on any other rail has to leave the case
        # out. The committed marker names all five rails, because v48 carried
        # the same surface and test through the Rust native rail and both
        # object rails (`research/docs/23` §3.3, v48), so every rail's capture
        # owes the same colour landing.
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            owes = stencil_marker(suite, rail)
            for position, case in enumerate(suite["render_cases"]):
                if position != STENCIL_INDEX:
                    case["capture_rails"] = [other_rail(rail)]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(stencil_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                if owes:
                    compare.validate_capture(suite, digest, report, rail)
                else:
                    with self.assertRaisesRegex(
                            compare.CaptureError,
                            "is not a rail this render case runs on"):
                        compare.validate_capture(suite, digest, report, rail)

    def test_v28_compares_the_stencil_pair_texels(self):
        # The stencil state is observed through the colour the pass leaves
        # behind, so a capture that reports the far triangle's tint — what a
        # rail that ignored the test or the op would land — is refused at the
        # first differing byte, and the near tint the suite declares is the
        # report that passes.
        suite = copy.deepcopy(self.suite)
        for position, case in enumerate(suite["render_cases"]):
            if position != STENCIL_INDEX:
                case["capture_rails"] = ["native-metal"]
        suite["render_cases"][STENCIL_INDEX]["capture_rails"] = ["vulkan"]
        digest = hashlib.sha256(
            json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
        report = counted_declaring(suite, digest, "vulkan")
        broken = stencil_result()
        broken["writebacks"][0]["bytes_hex"] = INSTANCE_TINTS[1] * 16
        broken["allocations"][0]["bytes_hex"] = INSTANCE_TINTS[1] * 16
        report["results"].append(broken)
        with self.assertRaisesRegex(compare.CaptureError, "first differing byte at offset 0"):
            compare.validate_capture(suite, digest, report, "vulkan")
        report["results"][-1] = stencil_result()
        compare.validate_capture(suite, digest, report, "vulkan")

    def test_v28_refuses_a_capture_that_counts_the_stencil_surface(self):
        # The rail-owned surface owes no copy-in and no copy-out, so a capture
        # that counted it — the stored-depth shape's three and three — is
        # refused, and the plan's own two and two are the counts that pass.
        suite = copy.deepcopy(self.suite)
        for position, case in enumerate(suite["render_cases"]):
            if position != STENCIL_INDEX:
                case["capture_rails"] = ["native-metal"]
        suite["render_cases"][STENCIL_INDEX]["capture_rails"] = ["vulkan"]
        digest = hashlib.sha256(
            json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
        report = counted_declaring(suite, digest, "vulkan")
        report["results"].append(stencil_result(copy_in=3, copy_out=3))
        with self.assertRaisesRegex(
                compare.CaptureError,
                "copy_in 3 does not match 2 touched allocations"):
            compare.validate_capture(suite, digest, report, "vulkan")
        report["results"][-1] = stencil_result()
        compare.validate_capture(suite, digest, report, "vulkan")

    def test_v28_allows_a_stencil_attachment_without_a_test(self):
        # The attachment and the state are separate shapes: a pass may open a
        # stencil surface and declare no test, which is what the contract states
        # as "no test" (`research/docs/23` §3.3, v47). The comparison reviews the
        # shape rather than the colour a state machine would leave behind, so the
        # case stays valid and its observation is unchanged.
        suite = copy.deepcopy(self.suite)
        del suite["render_cases"][STENCIL_INDEX]["stencil_test"]
        plan = compare._render_plan(compare._suite_plan(suite), suite)
        expectation = plan[STENCIL_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(STENCIL_EXPECTED))])
        self.assertEqual(expectation.touched, {900, 920})
        self.assertEqual(expectation.written, {900, 920})

    def test_v28_refuses_a_stencil_test_that_is_not_the_reviewed_one(self):
        # `always` is the v60 combined shape's own compare, so a third value
        # exercises the refusal; a keep pass op stays outside both reviewed
        # shapes (`research/docs/23` §3.3, v47/v60).
        for field, value in (("compare", "less"), ("pass_op", "keep")):
            broken = copy.deepcopy(self.suite)
            broken["render_cases"][STENCIL_INDEX]["stencil_test"][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                        compare.CaptureError,
                        "the reviewed stencil state is one of the two reviewed shapes"):
                    compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_stencil_clear_that_is_not_the_reviewed_one(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][STENCIL_INDEX]["stencil"]["clear_value"] = 1
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the reviewed stencil clear is zero"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_stencil_test_without_its_attachment(self):
        # The surface is what makes the state readable: a case that states the
        # test without the attachment it reads is refused rather than read as a
        # pass whose fragments are all discarded.
        broken = copy.deepcopy(self.suite)
        del broken["render_cases"][STENCIL_INDEX]["stencil"]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a stencil test needs the stencil attachment it reads"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_an_unreviewed_combined_depth_stencil_pair(self):
        # The combined depth-stencil pair is one surface both faces share, and
        # the only rail-owned shape reviewed over it is the v66 write-then-test
        # pair with its own stencil state: grafting another stencil state onto
        # a depth-bearing case is refused by the state rule rather than read as
        # that pair (`research/docs/23` §3.3, v66).
        for position in (STENCIL_INDEX, 4):
            broken = copy.deepcopy(self.suite)
            case = broken["render_cases"][position]
            if "depth" not in case:
                case["depth"] = {"format": "depth32float", "width": 4, "height": 4,
                                 "load": "clear", "clear_depth": 1.0}
                case["depth_test"] = {"compare": "less", "write": True}
            else:
                case["stencil"] = copy.deepcopy(STENCIL_SECTION)
                case["stencil_test"] = copy.deepcopy(STENCIL_TEST)
            with self.subTest(case=case["id"]):
                with self.assertRaisesRegex(
                        compare.CaptureError,
                        "the rail-owned combined pair's stencil state"):
                    compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_pins_the_stencil_store_fixture(self):
        case = self.suite["render_cases"][STENCIL_STORE_INDEX]
        self.assertEqual(case["id"], STENCIL_STORE_ID)
        # The surface is the v47 stencil attachment plus the v49 store action:
        # the pass keeps it, names the allocation and the view the texels land
        # in, and states the bytes the readback has to carry — one per texel.
        self.assertEqual(case["stencil"]["store"], "store")
        self.assertEqual(case["stencil"]["allocation"], STENCIL_STORE_ALLOCATION)
        self.assertEqual(case["stencil"]["view"], STENCIL_STORE_VIEW)
        self.assertEqual(len(case["stencil"]["expected_hex"]), 32)
        self.assertEqual(case["stencil"]["expected_hex"], STENCIL_STORE_EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(STENCIL_STORE_RAILS))
        # The colour side is the v47 pair's own: the near red triangle still
        # wins, and the stored stencil value is the one it left behind.
        self.assertEqual(case["expected_hex"], STENCIL_EXPECTED)
        self.assertEqual(case["expected_hex"],
                         self.suite["render_cases"][STENCIL_INDEX]["expected_hex"])
        # Everything else is the v47 case's shape — the module, the layout, the
        # streams, the indices, the viewport, the attachment and the state — so
        # the only new fields are the surface's store trio and the declaring
        # case whose third read binding is the stencil view.
        increment = self.suite["render_cases"][STENCIL_INDEX]
        for field in ("vertex_entry", "fragment_entry", "metal", "vertex_layout",
                      "vertex_buffers", "indices", "vertices", "viewport",
                      "attachment", "stencil_test"):
            self.assertEqual(case[field], increment[field])
        for field in ("format", "width", "height", "load", "clear_value"):
            self.assertEqual(case["stencil"][field], increment["stencil"][field])
        # The declaring case is the v43 declaring kernel with the stencil view in
        # place of the depth one: a third *read* binding covering exactly the
        # sixteen bytes of the stencil extent inside the sixty-four byte
        # allocation.
        declaring = [entry for entry in self.suite["cases"]
                     if entry["id"] == STENCIL_DECLARING_ID]
        self.assertEqual(len(declaring), 1)
        bindings = declaring[0]["buffers"]
        self.assertEqual([binding["binding"] for binding in bindings], [0, 1, 2])
        self.assertEqual((bindings[2]["allocation"], bindings[2]["view"],
                          bindings[2]["offset"], bindings[2]["length"],
                          bindings[2]["allocation_size"], bindings[2]["access"]),
                         (STENCIL_STORE_ALLOCATION, STENCIL_STORE_VIEW, 0, 16, 16,
                          "read"))

    def test_v28_plans_the_stencil_store_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[STENCIL_STORE_ID]
        # Both landings use the colour side's own two surfaces: one writeback
        # each, in the order the suite fixes, and one allocation image each. The
        # stencil image is the declaring case's own sixty-four byte allocation
        # with the sixteen stored bytes overlaid, so the guard bytes around the
        # view stay part of the comparison.
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(STENCIL_EXPECTED)),
                          ((STENCIL_STORE_ALLOCATION, STENCIL_STORE_VIEW, 0),
                           bytes.fromhex(STENCIL_STORE_EXPECTED))])
        self.assertEqual(expectation.allocations,
                         {ATTACHMENT[0]: bytes.fromhex(STENCIL_EXPECTED),
                          STENCIL_STORE_ALLOCATION:
                              bytes.fromhex(STENCIL_STORE_ALLOCATION_BYTES)})
        self.assertEqual(list(expectation.attachment),
                         [ATTACHMENT, STENCIL_STORE_ATTACHMENT])
        # The colour attachment still stores, so the case owes the stencil
        # landing's one touched and one written allocation on top of the
        # declaring pass's two and two: the stored depth pair's own three and
        # three (`research/docs/23` §3.3, v43/v49).
        self.assertEqual(expectation.touched, {900, 920, 940})
        self.assertEqual(expectation.written, {900, 920, 940})
        self.assertEqual(expectation.rails, frozenset(STENCIL_STORE_RAILS))

    def test_v28_reports_the_stencil_store_on_every_rail_its_marker_names(self):
        # The marker is the rule, whichever rails it names: a capture on a rail
        # the fixture's marker names is owed both landings — the colour pair's
        # and the stored stencil surface's — and a capture on any other rail has
        # to leave the case out rather than report a run it does not own.
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            owes = stencil_store_marker(suite, rail)
            for position, case in enumerate(suite["render_cases"]):
                if position != STENCIL_STORE_INDEX:
                    case["capture_rails"] = [other_rail(rail)]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(stencil_store_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                if owes:
                    compare.validate_capture(suite, digest, report, rail)
                else:
                    with self.assertRaisesRegex(
                            compare.CaptureError,
                            "is not a rail this render case runs on"):
                        compare.validate_capture(suite, digest, report, rail)

    def test_v28_compares_the_stored_stencil_texels(self):
        # The stencil landing goes through the same byte comparison the colour
        # side uses, so a capture that reports the clear value instead of the
        # stored texels is refused at the first differing byte, and the reported
        # landing the suite declares is the one that passes.
        suite = copy.deepcopy(self.suite)
        for position, case in enumerate(suite["render_cases"]):
            if position != STENCIL_STORE_INDEX:
                case["capture_rails"] = ["native-metal"]
        suite["render_cases"][STENCIL_STORE_INDEX]["capture_rails"] = ["vulkan"]
        digest = hashlib.sha256(
            json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
        report = counted_declaring(suite, digest, "vulkan")
        broken = stencil_store_result()
        broken["writebacks"][1]["bytes_hex"] = "00" * 16
        broken["allocations"][1]["bytes_hex"] = "00" * 64
        report["results"].append(broken)
        with self.assertRaisesRegex(compare.CaptureError,
                                    "first differing byte at offset 0"):
            compare.validate_capture(suite, digest, report, "vulkan")
        report["results"][-1] = stencil_store_result()
        compare.validate_capture(suite, digest, report, "vulkan")

    def test_v28_refuses_a_capture_that_counts_only_the_pair(self):
        # The stored stencil surface is one more touched and one more written
        # allocation, so the pair's own two and two are refused and the plan's
        # three and three are the counts that pass.
        suite = copy.deepcopy(self.suite)
        for position, case in enumerate(suite["render_cases"]):
            if position != STENCIL_STORE_INDEX:
                case["capture_rails"] = ["native-metal"]
        suite["render_cases"][STENCIL_STORE_INDEX]["capture_rails"] = ["vulkan"]
        digest = hashlib.sha256(
            json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
        report = counted_declaring(suite, digest, "vulkan")
        report["results"].append(stencil_store_result(copy_in=2, copy_out=2))
        with self.assertRaisesRegex(
                compare.CaptureError,
                "copy_in 2 does not match 3 touched allocations"):
            compare.validate_capture(suite, digest, report, "vulkan")
        report["results"][-1] = stencil_store_result()
        compare.validate_capture(suite, digest, report, "vulkan")

    def test_v28_refuses_a_stencil_expectation_that_is_the_clear_value(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][STENCIL_STORE_INDEX]["stencil"]["expected_hex"] = (
            "00" * 16)
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the expected stencil texels equal the clear value"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_stencil_expectation_of_the_wrong_length(self):
        # The landing is one byte per texel, so a fifteen-byte expectation is
        # not the surface's own extent and the shape is refused rather than
        # compared.
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][STENCIL_STORE_INDEX]["stencil"]["expected_hex"] = (
            "01" * 15)
        with self.assertRaisesRegex(
                compare.CaptureError,
                "the expected stencil texels do not match the attachment"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_stencil_identity_without_a_store_action(self):
        broken = copy.deepcopy(self.suite)
        del broken["render_cases"][STENCIL_STORE_INDEX]["stencil"]["store"]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a discarded stencil attachment carries no identity or expectation"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_stencil_store_action_without_its_identity(self):
        broken = copy.deepcopy(self.suite)
        del broken["render_cases"][STENCIL_STORE_INDEX]["stencil"]["allocation"]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a stored stencil attachment needs its store action, its identity"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_stencil_resource_that_is_the_colour_attachment(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][STENCIL_STORE_INDEX]["stencil"]["allocation"] = ATTACHMENT[0]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "the stencil resource has to differ from the colour attachment"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_stencil_store_the_declaring_case_does_not_declare(self):
        # The landing resolves against the declaring case's own table, exactly
        # as the depth landing does: without a declaration of the stencil view
        # the texels have no writeback channel to leave through.
        broken = copy.deepcopy(self.suite)
        declaring = [case for case in broken["cases"]
                     if case["id"] == STENCIL_DECLARING_ID][0]
        declaring["buffers"] = [buffer for buffer in declaring["buffers"]
                                if buffer["binding"] != 2]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "the declaring case has to declare exactly the stencil attachment view"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_stencil_view_the_declaring_pass_writes(self):
        # The declaring kernel only reads the stencil view: a compute write
        # would race the store, so the shape is refused exactly as the depth
        # landing's is.
        broken = copy.deepcopy(self.suite)
        declaring = [case for case in broken["cases"]
                     if case["id"] == STENCIL_DECLARING_ID][0]
        binding = declaring["buffers"][2]
        binding["access"] = "read_write"
        declaring["expected_writebacks"].append(
            {"allocation": binding["allocation"], "view": binding["view"],
             "offset": binding["offset"], "bytes_hex": binding["initial_hex"]})
        with self.assertRaisesRegex(
                compare.CaptureError,
                "the declaring pass must only read the stencil attachment view"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_stencil_view_of_the_wrong_byte_range(self):
        # The declaration's byte range is the one-byte-per-texel extent the
        # attachment restates, so an eight-byte view cannot carry the sixteen
        # stored bytes.
        broken = copy.deepcopy(self.suite)
        declaring = [case for case in broken["cases"]
                     if case["id"] == STENCIL_DECLARING_ID][0]
        binding = declaring["buffers"][2]
        binding["length"] = 8
        binding["initial_hex"] = binding["initial_hex"][:16]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "the stencil view's byte range disagrees with the attachment"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_pins_the_alignment_fixture(self):
        case = self.suite["render_cases"][ALIGNMENT_INDEX]
        self.assertEqual(case["id"], ALIGNMENT_ID)
        self.assertEqual(case["coverage"], "partial")
        self.assertEqual(case["attachment"]["load"], "clear")
        self.assertEqual(case["expected_hex"], ALIGNMENT_EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(ALIGNMENT_RAILS))

    def test_v28_plans_the_alignment_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[ALIGNMENT_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(ALIGNMENT_EXPECTED))])

    def test_v28_pins_the_cull_fixture(self):
        case = self.suite["render_cases"][CULL_INDEX]
        self.assertEqual(case["id"], CULL_ID)
        self.assertEqual(case["cull"], {"mode": "back", "winding": "counter_clockwise"})
        self.assertEqual(case["expected_hex"], CULL_EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(CULL_RAILS))

    def test_v28_plans_the_cull_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[CULL_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(CULL_EXPECTED))])

    def test_v28_pins_the_blend_fixture(self):
        case = self.suite["render_cases"][BLEND_INDEX]
        self.assertEqual(case["id"], BLEND_ID)
        self.assertEqual(case["blend"][0]["source_rgb"], "source_alpha")
        self.assertEqual(case["blend"][0]["destination_rgb"], "one_minus_source_alpha")
        self.assertEqual(case["attachment"]["clear_hex"], "00000000")
        self.assertEqual(case["expected_hex"], BLEND_EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(BLEND_RAILS))

    def test_v28_plans_the_blend_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[BLEND_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(BLEND_EXPECTED))])

    def test_v28_refuses_a_blend_state_that_is_not_the_reviewed_one(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][BLEND_INDEX]["blend"][0]["destination_rgb"] = "one"
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the reviewed blend state is source alpha against"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_blend_case_that_also_culls(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][BLEND_INDEX]["cull"] = {
            "mode": "back", "winding": "counter_clockwise"}
        with self.assertRaisesRegex(compare.CaptureError,
                                    "carries neither a depth attachment nor a culling state"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_cull_state_that_is_not_the_reviewed_one(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][CULL_INDEX]["cull"]["winding"] = "clockwise"
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the reviewed cull state is back faces"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_cull_case_that_also_opens_a_depth_surface(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][CULL_INDEX]["depth"] = {"format": "depth32float", "width": 4,
                                                       "height": 4, "load": "clear",
                                                       "clear_depth": 1.0}
        broken["render_cases"][CULL_INDEX]["depth_test"] = {"compare": "less", "write": True}
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the reviewed cull shape carries no depth attachment"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_partial_claim_that_covers_everything(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][ALIGNMENT_INDEX]["expected_hex"] = OUTPUT * 16
        with self.assertRaisesRegex(compare.CaptureError,
                                    "a partial coverage claim needs both drawn and clear"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_texel_that_is_neither_output_nor_clear(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][ALIGNMENT_INDEX]["expected_hex"] = \
            "ff0000ff" + ALIGNMENT_EXPECTED[8:]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "has to be the fragment output or the clear colour"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_coverage_claim_that_is_not_partial(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][ALIGNMENT_INDEX]["coverage"] = "full"
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the only coverage claim"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_wildcard_beyond_the_attachment(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][2]["wildcard_texels"] = WILDCARD_TEXELS + [16]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "wildcard texel 16 is outside the attachment"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_wildcard_list_that_claims_nothing(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][2]["wildcard_texels"] = list(range(16))
        with self.assertRaisesRegex(compare.CaptureError,
                                    "has to leave at least one texel observed"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_wildcards_on_a_non_dontcare_load(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][2]["attachment"]["load"] = "clear"
        broken["render_cases"][2]["attachment"]["clear_hex"] = CLEAR
        with self.assertRaisesRegex(compare.CaptureError,
                                    "only a dontcare load may leave texels unclaimed"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_wildcards_on_a_discarded_attachment(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][2]["attachment"]["store"] = "dontcare"
        with self.assertRaisesRegex(compare.CaptureError,
                                    "a wildcard list needs a stored attachment"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_wrong_observed_texel(self):
        digest = hashlib.sha256(
            json.dumps(self.suite, sort_keys=True).encode("utf-8")).hexdigest()
        report = counted_declaring(self.suite, digest, "vulkan")
        report["results"].append(render_result())
        report["results"].append(instanced_result())
        report["results"].append(base_vertex_result())
        report["results"].append(depth_result())
        report["results"].append(depth_store_result())
        report["results"].append(depth_only_result())
        report["results"].append(depth_no_colour_result())
        report["results"].append(stencil_result())
        report["results"].append(stencil_store_result())
        report["results"].append(alignment_result())
        report["results"].append(cull_result())
        report["results"].append(blend_result())
        report["results"].append(msaa_result())
        report["results"].append(msaa_depth_result())
        report["results"].append(msaa_stencil_result())
        report["results"].append(msaa_depth_resolve_result())
        broken = wildcard_result()
        # The left half is claimed, so a wrong byte there is a refusal even
        # though the right half stays wild.
        broken["writebacks"][0]["bytes_hex"] = "ff0000ff" + WILDCARD_REPORTED[8:]
        broken["allocations"][0]["bytes_hex"] = "ff0000ff" + WILDCARD_REPORTED[8:]
        report["results"].append(broken)
        with self.assertRaisesRegex(compare.CaptureError,
                                    "first differing byte at offset 0"):
            compare.validate_capture(self.suite, digest, report, "vulkan")

    def test_v28_refuses_an_instanced_marker_that_omits_its_rail(self):
        for rail in INSTANCED_RAILS:
            suite = copy.deepcopy(self.suite)
            suite["render_cases"][0]["capture_rails"] = [other_rail(rail)]
            suite["render_cases"][1]["capture_rails"] = [
                other for other in INSTANCED_RAILS if other != rail]
            owes_depth_store = depth_store_marker(suite, rail)
            owes_depth_only = depth_only_marker(suite, rail)
            owes_depth_no_colour = depth_no_colour_marker(suite, rail)
            owes_stencil = stencil_marker(suite, rail)
            owes_stencil_store = stencil_store_marker(suite, rail)
            owes_depth_resolve = msaa_depth_resolve_marker(suite, rail)
            owes_stencil_resolve_sample0 = msaa_stencil_resolve_sample0_marker(suite, rail)
            owes_stencil_resolve_drs = msaa_stencil_resolve_drs_marker(suite, rail)
            owes_msaa_ds = msaa_ds_marker(suite, rail)
            msaa_depth_resolve_min_edge_marker(suite, rail)
            msaa_depth_resolve_max_edge_marker(suite, rail)
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(
                suite, digest, rail,
                depth_resolve_modes=DEPTH_RESOLVE_ALL_FILTERS_BITS,
                stencil_resolve_modes=(STENCIL_RESOLVE_ALL_FILTERS_BITS
                                       if rail in MSAA_STENCIL_RESOLVE_DRS_RAILS else 0))
            report["results"].append(instanced_result(rail != "native-metal"))
            report["results"].append(wildcard_result(rail != "native-metal"))
            if rail in BASE_VERTEX_RAILS:
                report["results"].append(base_vertex_result(rail != "native-metal"))
            if rail in DEPTH_RAILS:
                report["results"].append(depth_result(rail != "native-metal"))
            if owes_depth_store:
                report["results"].append(depth_store_result(rail != "native-metal"))
            if owes_depth_only:
                report["results"].append(depth_only_result(rail != "native-metal"))
            if owes_depth_no_colour:
                report["results"].append(depth_no_colour_result(rail != "native-metal"))
            if owes_stencil:
                report["results"].append(stencil_result(rail != "native-metal"))
            if owes_stencil_store:
                report["results"].append(stencil_store_result(rail != "native-metal"))
            if rail in ALIGNMENT_RAILS:
                report["results"].append(alignment_result(rail != "native-metal"))
            if rail in CULL_RAILS:
                report["results"].append(cull_result(rail != "native-metal"))
            if rail in BLEND_RAILS:
                report["results"].append(blend_result(rail != "native-metal"))
            if rail in MSAA_RAILS:
                report["results"].append(msaa_result(rail != "native-metal"))
            if rail in MSAA_DEPTH_RAILS:
                report["results"].append(msaa_depth_result(rail != "native-metal"))
            if rail in MSAA_STENCIL_RAILS:
                report["results"].append(msaa_stencil_result(rail != "native-metal"))
            if owes_depth_resolve:
                report["results"].append(msaa_depth_resolve_result(rail != "native-metal"))
            if owes_stencil_resolve_sample0:
                report["results"].append(
                    msaa_stencil_resolve_sample0_result(rail != "native-metal"))
            if owes_stencil_resolve_drs:
                report["results"].append(
                    msaa_stencil_resolve_drs_result(rail != "native-metal"))
            if owes_msaa_ds:
                report["results"].append(msaa_ds_result(rail != "native-metal"))
            if rail in MSAA_DEPTH_RESOLVE_MIN_EDGE_RAILS:
                report["results"].append(
                    msaa_depth_resolve_min_edge_result(rail != "native-metal"))
            if rail in MSAA_DEPTH_RESOLVE_MAX_EDGE_RAILS:
                report["results"].append(
                    msaa_depth_resolve_max_edge_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                with self.assertRaisesRegex(compare.CaptureError,
                                            "is not a rail this render case runs on"):
                    compare.validate_capture(suite, digest, report, rail)

    def test_v28_refuses_a_uniform_instanced_expectation(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][1]["expected_hex"] = INSTANCE_TINTS[0] * 16
        with self.assertRaisesRegex(compare.CaptureError,
                                    "instance tint of its half"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_swapped_instanced_expectation(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][1]["expected_hex"] = "".join(
            INSTANCE_TINTS[1] if (index % 4) < 2 else INSTANCE_TINTS[0]
            for index in range(16))
        with self.assertRaisesRegex(compare.CaptureError,
                                    "instance tint of its half"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_single_instance_instanced_case(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][1]["instance_count"] = 1
        with self.assertRaisesRegex(compare.CaptureError,
                                    "runs exactly two instances"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_scissor_outside_the_attachment(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["scissor"] = [0, 0, 5, 4]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "non-empty rectangle inside the attachment"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_scissor_that_covers_everything(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["scissor"] = [0, 0, 4, 4]
        broken["render_cases"][0]["expected_hex"] = OUTPUT * 16
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the scissor has to clip part of the attachment"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_an_unclipped_expectation(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["expected_hex"] = OUTPUT * 16
        with self.assertRaisesRegex(compare.CaptureError,
                                    "has to be the clear colour under the declared scissor"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_pins_the_msaa_fixture(self):
        case = self.suite["render_cases"][MSAA_INDEX]
        self.assertEqual(case["id"], MSAA_ID)
        self.assertEqual(case["multisample"], {"sample_count": 4})
        self.assertEqual(case["coverage"], "partial")
        self.assertEqual(case["attachment"]["load"], "clear")
        self.assertEqual(case["attachment"]["clear_hex"], MSAA_CLEAR)
        self.assertEqual(case["expected_hex"], MSAA_EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(MSAA_RAILS))

    def test_v28_plans_the_msaa_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[MSAA_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(MSAA_EXPECTED))])

    def test_v28_refuses_a_multisample_state_outside_the_reviewed_family(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_INDEX]["multisample"]["sample_count"] = 3
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the reviewed multisample rasters are two, four or eight"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_full_coverage_multisample_expectation_with_a_mix(self):
        # Deleting the edge fixture's coverage claim admits the v61
        # full-coverage colour-only shape, whose expectation then has to be
        # uniform — the mixed texel the edge fixture pins is refused by the
        # uniform rule instead of the old coverage gate (`research/docs/23`
        # §3.3, v61).
        broken = copy.deepcopy(self.suite)
        del broken["render_cases"][MSAA_INDEX]["coverage"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "every texel of the expectation has to be the "
                                    "fragment output"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_multisample_case_with_a_loading_attachment(self):
        broken = copy.deepcopy(self.suite)
        attachment = broken["render_cases"][MSAA_INDEX]["attachment"]
        attachment["load"] = "load"
        attachment["initial_hex"] = MSAA_CLEAR * 16
        del attachment["clear_hex"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the reviewed multisample pass opens its attachment"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_multisample_case_that_widens_to_an_attachment_list(self):
        broken = copy.deepcopy(self.suite)
        case = broken["render_cases"][MSAA_INDEX]
        case["attachments"] = [case.pop("attachment")]
        case["attachments"][0]["expected_hex"] = case.pop("expected_hex")
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the reviewed MRT shapes are two to four attachments"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_multisample_expectation_that_is_not_a_resolve(self):
        # A single-sample raster can only produce the fragment output or the
        # clear colour, so an expectation without the 2-of-4 mix is refused:
        # that mix is the one byte pattern this fixture exists to observe.
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_INDEX]["expected_hex"] = "".join(
            OUTPUT if (index % 4) < 2 else MSAA_CLEAR for index in range(16))
        with self.assertRaisesRegex(compare.CaptureError,
                                    "needs at least one partially covered texel"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_multisample_expectation_of_a_wrong_byte(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_INDEX]["expected_hex"] = (
            MSAA_EXPECTED.replace(MSAA_MIXED, "316293c5", 1))
        with self.assertRaisesRegex(compare.CaptureError,
                                    "is not the resolve of any coverage"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_reports_the_msaa_case_on_every_rail_its_marker_names(self):
        # The marker is the rule: every rail the marker names owes the resolve
        # target's landing, and a capture on any other rail has to leave the
        # case out rather than report a run it does not own
        # (`research/docs/23` §3.3, v51/v52).
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            owes = msaa_marker(suite, rail)
            for position, case in enumerate(suite["render_cases"]):
                if position != MSAA_INDEX:
                    case["capture_rails"] = [other_rail(rail)]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(msaa_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                if owes:
                    compare.validate_capture(suite, digest, report, rail)
                else:
                    with self.assertRaisesRegex(
                            compare.CaptureError,
                            "is not a rail this render case runs on"):
                        compare.validate_capture(suite, digest, report, rail)

    def test_v28_pins_the_msaa_uniform_fixtures(self):
        for index, case_id, sample_count, gate in (
                (MSAA_UNIFORM_2X_INDEX, MSAA_UNIFORM_2X_ID, 2, 2),
                (MSAA_UNIFORM_8X_INDEX, MSAA_UNIFORM_8X_ID, 8, 8)):
            with self.subTest(case=case_id):
                case = self.suite["render_cases"][index]
                self.assertEqual(case["id"], case_id)
                self.assertEqual(case["declaring_case"], "render_declaring_quad_extent")
                self.assertEqual(case["multisample"], {"sample_count": sample_count})
                self.assertEqual(case["requires_sample_count"], gate)
                self.assertNotIn("coverage", case)
                self.assertEqual(case["attachment"]["load"], "clear")
                self.assertEqual(case["attachment"]["clear_hex"], MSAA_UNIFORM_CLEAR)
                self.assertEqual(case["expected_hex"], OUTPUT * 16)
                self.assertEqual(sorted(case["capture_rails"]), sorted(ALL_RAILS))
                # The stream is the full-coverage quad: all four corners sit
                # on the attachment's edges, so every sample of every texel is
                # covered and no partial-coverage byte can be pinned
                # (`research/docs/23` §3.3, v61).
                stream = case["vertex_buffers"][0]["initial_hex"]
                vertices = [struct.unpack("<2f", bytes.fromhex(stream)[offset:offset + 8])
                            for offset in range(0, 32, 8)]
                self.assertEqual(vertices,
                                 [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)])

    def test_v28_plans_the_msaa_uniform_fixtures(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        for case_id, gate in ((MSAA_UNIFORM_2X_ID, 2), (MSAA_UNIFORM_8X_ID, 8)):
            with self.subTest(case=case_id):
                expectation = plan[case_id]
                self.assertEqual(expectation.writes,
                                 [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                                   bytes.fromhex(OUTPUT * 16))])
                self.assertEqual(expectation.sample_count_gate, gate)

    def test_v28_refuses_a_sample_count_gate_that_does_not_name_the_raster(self):
        # The gate is the case's own admission condition, so it has to name the
        # raster the case states rather than drift into a second spelling
        # (`research/docs/23` §3.3, v61).
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_UNIFORM_2X_INDEX]["requires_sample_count"] = 8
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the device gate has to name the sample count"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_sample_count_gate_outside_two_and_eight(self):
        # 4x is the v51 baseline every multisampling device admits, so only 2x
        # and 8x are gateable (`research/docs/23` §3.3, v61).
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_UNIFORM_2X_INDEX]["requires_sample_count"] = 4
        broken["render_cases"][MSAA_UNIFORM_2X_INDEX]["multisample"]["sample_count"] = 4
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the device gate names the two- or eight-sample raster"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_gates_the_msaa_uniform_cases_on_the_sample_mask(self):
        # Presence iff the mask carries the count's bit: absent without the
        # bit passes, present without it is refused, absent with it is
        # refused, and present with it passes (`research/docs/23` §3.3, v61).
        for index, case_id, gate, bit, result_builder in (
                (MSAA_UNIFORM_2X_INDEX, MSAA_UNIFORM_2X_ID, 2, 1 << 1,
                 msaa_uniform_2x_result),
                (MSAA_UNIFORM_8X_INDEX, MSAA_UNIFORM_8X_ID, 8, 1 << 3,
                 msaa_uniform_8x_result)):
            suite = copy.deepcopy(self.suite)
            for position, case in enumerate(suite["render_cases"]):
                if position != index:
                    case["capture_rails"] = ["native-metal"]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            with self.subTest(case=case_id, direction="absent without the bit"):
                report = counted_declaring(suite, digest, "vulkan")
                compare.validate_capture(suite, digest, report, "vulkan")
            with self.subTest(case=case_id, direction="present without the bit"):
                report = counted_declaring(suite, digest, "vulkan")
                report["results"].append(result_builder())
                with self.assertRaisesRegex(
                        compare.CaptureError,
                        f"lacks the {gate}-sample raster"):
                    compare.validate_capture(suite, digest, report, "vulkan")
            with self.subTest(case=case_id, direction="absent with the bit"):
                report = counted_declaring(suite, digest, "vulkan",
                                           render_sample_counts=bit)
                with self.assertRaisesRegex(compare.CaptureError, "missing cases"):
                    compare.validate_capture(suite, digest, report, "vulkan")
            with self.subTest(case=case_id, direction="present with the bit"):
                report = counted_declaring(suite, digest, "vulkan",
                                           render_sample_counts=bit)
                report["results"].append(result_builder())
                compare.validate_capture(suite, digest, report, "vulkan")

    def test_v28_the_sample_mask_is_per_count_not_a_ladder(self):
        # Lavapipe is the measured counterexample: its framebuffer admits 4x
        # and 8x but not 2x, so a capture whose mask carries only 8x owes the
        # 8x case and leaves the 2x case out (`research/docs/23` §3.3, v61).
        suite = copy.deepcopy(self.suite)
        for position, case in enumerate(suite["render_cases"]):
            if position not in (MSAA_UNIFORM_2X_INDEX, MSAA_UNIFORM_8X_INDEX):
                case["capture_rails"] = ["native-metal"]
        digest = hashlib.sha256(
            json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
        report = counted_declaring(suite, digest, "vulkan",
                                   render_sample_counts=1 << 3)
        report["results"].append(msaa_uniform_8x_result())
        compare.validate_capture(suite, digest, report, "vulkan")

    def test_v28_pins_the_msaa_depth_fixture(self):
        case = self.suite["render_cases"][MSAA_DEPTH_INDEX]
        self.assertEqual(case["id"], MSAA_DEPTH_ID)
        self.assertEqual(case["multisample"], {"sample_count": 4})
        self.assertNotIn("coverage", case)
        self.assertEqual(case["depth"]["format"], "depth32float")
        self.assertNotIn("store", case["depth"])
        self.assertEqual(case["depth_test"], {"compare": "less", "write": True})
        self.assertEqual(case["expected_hex"], MSAA_DEPTH_EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(MSAA_DEPTH_RAILS))

    def test_v28_plans_the_msaa_depth_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[MSAA_DEPTH_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(MSAA_DEPTH_EXPECTED))])

    def test_v28_refuses_a_msaa_depth_surface_that_is_stored(self):
        # A multisampled depth surface's texels are only observable through a
        # depth resolve, so a stored surface without the resolve the case has
        # to state is refused rather than compared (`research/docs/23` §3.3,
        # v57). The rail-owned shape is the *absent* store action.
        broken = copy.deepcopy(self.suite)
        depth = broken["render_cases"][MSAA_DEPTH_INDEX]["depth"]
        depth["store"] = "store"
        depth["allocation"] = 940
        depth["view"] = 951
        depth["expected_hex"] = "00" * 64
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a stored multisampled depth surface needs its depth resolve"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_msaa_depth_case_that_claims_partial_coverage(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_DEPTH_INDEX]["coverage"] = "partial"
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a multisample pass with a depth surface claims no partial coverage"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_msaa_depth_expectation_that_is_not_one_output(self):
        # The depth pair's two primitives both cover every sample, so the
        # expectation is one fragment output repeated — a mixed texel would
        # claim a raster the fixture does not describe.
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_DEPTH_INDEX]["expected_hex"] = (
            "ff0000ff" * 15 + "00ff00ff")
        with self.assertRaisesRegex(compare.CaptureError,
                                    "has to be the fragment output"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_reports_the_msaa_depth_case_on_every_rail_its_marker_names(self):
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            owes = msaa_depth_marker(suite, rail)
            for position, case in enumerate(suite["render_cases"]):
                if position != MSAA_DEPTH_INDEX:
                    case["capture_rails"] = [other_rail(rail)]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(msaa_depth_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                if owes:
                    compare.validate_capture(suite, digest, report, rail)
                else:
                    with self.assertRaisesRegex(
                            compare.CaptureError,
                            "is not a rail this render case runs on"):
                        compare.validate_capture(suite, digest, report, rail)

    def test_v28_pins_the_msaa_stencil_fixture(self):
        case = self.suite["render_cases"][MSAA_STENCIL_INDEX]
        self.assertEqual(case["id"], MSAA_STENCIL_ID)
        self.assertEqual(case["multisample"], {"sample_count": 4})
        self.assertNotIn("coverage", case)
        self.assertEqual(case["stencil"]["format"], "stencil8")
        self.assertNotIn("store", case["stencil"])
        self.assertEqual(case["stencil_test"]["compare"], "equal")
        self.assertEqual(case["stencil_test"]["pass_op"], "increment_wrap")
        self.assertEqual(case["expected_hex"], MSAA_STENCIL_EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(MSAA_STENCIL_RAILS))

    def test_v28_plans_the_msaa_stencil_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[MSAA_STENCIL_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(MSAA_STENCIL_EXPECTED))])

    def test_v28_refuses_a_msaa_stencil_surface_that_is_stored(self):
        # A multisampled stencil surface's texels are only observable through
        # a resolve, so a case that keeps the surface without stating one is
        # refused rather than compared (`research/docs/23` §3.3, v55/v60).
        broken = copy.deepcopy(self.suite)
        stencil = broken["render_cases"][MSAA_STENCIL_INDEX]["stencil"]
        stencil["store"] = "store"
        stencil["allocation"] = 940
        stencil["view"] = 951
        stencil["expected_hex"] = "01" * 16
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a stored multisampled stencil surface needs its stencil resolve"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_msaa_stencil_case_that_claims_partial_coverage(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_STENCIL_INDEX]["coverage"] = "partial"
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a multisample pass with a stencil surface claims no partial coverage"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_msaa_stencil_expectation_that_is_not_one_output(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_STENCIL_INDEX]["expected_hex"] = (
            "ff0000ff" * 15 + "00ff00ff")
        with self.assertRaisesRegex(compare.CaptureError,
                                    "has to be the fragment output"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_reports_the_msaa_stencil_case_on_every_rail_its_marker_names(self):
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            owes = msaa_stencil_marker(suite, rail)
            for position, case in enumerate(suite["render_cases"]):
                if position != MSAA_STENCIL_INDEX:
                    case["capture_rails"] = [other_rail(rail)]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(msaa_stencil_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                if owes:
                    compare.validate_capture(suite, digest, report, rail)
                else:
                    with self.assertRaisesRegex(
                            compare.CaptureError,
                            "is not a rail this render case runs on"):
                        compare.validate_capture(suite, digest, report, rail)

    def test_v28_pins_the_combined_depth_stencil_pair_fixture(self):
        case = self.suite["render_cases"][MSAA_DS_INDEX]
        self.assertEqual(case["id"], MSAA_DS_ID)
        self.assertEqual(case["multisample"], {"sample_count": 4})
        self.assertEqual(case["coverage"], "partial")
        self.assertNotIn("store", case["depth"])
        self.assertNotIn("store", case["stencil"])
        self.assertNotIn("allocation", case["depth"])
        self.assertNotIn("allocation", case["stencil"])
        self.assertEqual(case["depth"]["clear_depth"], 1.0)
        self.assertEqual(case["stencil"]["clear_value"], 0)
        self.assertEqual(case["stencil_test"], MSAA_DS_STENCIL_TEST)
        self.assertEqual(case["expected_hex"], MSAA_DS_EXPECTED)
        self.assertEqual(case["vertices"], 9)
        self.assertEqual(len(case["vertex_buffers"][0]["initial_hex"]), 2 * 288)
        self.assertEqual(case["indices"]["initial_hex"],
                         "000001000200030004000500060007000800")
        self.assertEqual(case["attachment"]["clear_hex"], MSAA_DS_CLEAR)
        self.assertEqual(sorted(case["capture_rails"]), sorted(MSAA_DS_RAILS))

    def test_v28_plans_the_combined_depth_stencil_pair(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[MSAA_DS_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(MSAA_DS_EXPECTED))])
        self.assertEqual(expectation.rails, frozenset(MSAA_DS_RAILS))

    def test_v28_refuses_a_combined_pair_that_keeps_one_face(self):
        # The two faces share one surface, so the pair keeps both or neither:
        # a case that stores one while discarding the other is refused rather
        # than read as either reviewed shape (`research/docs/23` §3.3, v66).
        for stored in ("depth", "stencil"):
            broken = copy.deepcopy(self.suite)
            case = broken["render_cases"][MSAA_DS_INDEX]
            case[stored]["store"] = "store"
            case[stored]["allocation"] = 940
            case[stored]["view"] = 952
            case[stored]["expected_hex"] = (
                "01" * 16 if stored == "stencil" else "0000003f" * 16)
            with self.subTest(stored=stored):
                # The two faces' rules refuse the lopsided pair from either
                # side: the general rule names "both faces or neither" and the
                # depth face's own key-set rule names what it found.
                with self.assertRaisesRegex(
                        compare.CaptureError,
                        "keeps both faces or neither|keeps neither face"):
                    compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_combined_pair_that_claims_no_partial_coverage(self):
        # The pair's whole point is its partially covered column, so the case
        # has to claim the coverage its expectation then shows.
        broken = copy.deepcopy(self.suite)
        del broken["render_cases"][MSAA_DS_INDEX]["coverage"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "claims the partial coverage it resolves"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_combined_pair_whose_stencil_state_is_the_v47_one(self):
        # The pair's own reviewed state is the depth-failure increment; the
        # v47 equal-zero pass increment is a different shape and is refused by
        # name (`research/docs/23` §3.3, v66).
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_DS_INDEX]["stencil_test"] = copy.deepcopy(STENCIL_TEST)
        with self.assertRaisesRegex(
                compare.CaptureError,
                "the rail-owned combined pair's stencil state"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_combined_pair_with_a_uniform_expectation(self):
        # One tint on every texel is exactly what a rail that ignored one of
        # the faces would land; the resolve rule refuses it because no texel is
        # partially covered (`research/docs/23` §3.3, v66).
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_DS_INDEX]["expected_hex"] = INSTANCE_TINTS[0] * 16
        with self.assertRaisesRegex(compare.CaptureError, "partially covered texel"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_combined_pair_whose_expectation_drops_the_clear(self):
        # The other extreme: an expectation that claims every texel covered
        # would pass as "everything drew" and could not show the clear the
        # untouched column keeps.
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_DS_INDEX]["expected_hex"] = (
            (INSTANCE_TINTS[0] * 2 + "88111aa2" + INSTANCE_TINTS[0]) * 4)
        with self.assertRaisesRegex(
                compare.CaptureError,
                "needs both a fully covered and an uncovered texel"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_combined_pair_whose_expectation_is_not_a_resolve(self):
        # A texel the resolve arithmetic cannot produce — here the third
        # triangle's blue — is refused by the same rule that admits the mix.
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_DS_INDEX]["expected_hex"] = (
            "ff0000ffff0000ff0000ffff11223445" * 4)
        with self.assertRaisesRegex(compare.CaptureError,
                                    "is not the resolve of any coverage"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_reports_the_combined_pair_on_every_rail_its_marker_names(self):
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            owes = msaa_ds_marker(suite, rail)
            for position, case in enumerate(suite["render_cases"]):
                if position != MSAA_DS_INDEX:
                    case["capture_rails"] = [other_rail(rail)]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(msaa_ds_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                if owes:
                    compare.validate_capture(suite, digest, report, rail)
                else:
                    with self.assertRaisesRegex(
                            compare.CaptureError,
                            "is not a rail this render case runs on"):
                        compare.validate_capture(suite, digest, report, rail)

    def test_v28_pins_the_msaa_depth_resolve_fixture(self):
        case = self.suite["render_cases"][MSAA_DEPTH_RESOLVE_INDEX]
        self.assertEqual(case["id"], MSAA_DEPTH_RESOLVE_ID)
        self.assertEqual(case["multisample"], {"sample_count": 4})
        self.assertEqual(case["depth_resolve"], {"filter": "sample0"})
        self.assertNotIn("coverage", case)
        self.assertEqual(case["depth"]["format"], "depth32float")
        self.assertEqual(case["depth"]["store"], "store")
        self.assertEqual(case["depth"]["allocation"], DEPTH_STORE_ALLOCATION)
        self.assertEqual(case["depth"]["view"], DEPTH_STORE_VIEW)
        self.assertEqual(case["depth"]["expected_hex"], MSAA_DEPTH_RESOLVE_DEPTH)
        self.assertEqual(case["depth_test"], {"compare": "less", "write": True})
        self.assertEqual(case["expected_hex"], MSAA_DEPTH_RESOLVE_EXPECTED)
        self.assertEqual(case["capture_rails"], list(MSAA_DEPTH_RESOLVE_RAILS))

    def test_v28_plans_the_msaa_depth_resolve_fixture(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[MSAA_DEPTH_RESOLVE_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(MSAA_DEPTH_RESOLVE_EXPECTED)),
                          ((DEPTH_STORE_ALLOCATION, DEPTH_STORE_VIEW, 0),
                           bytes.fromhex(MSAA_DEPTH_RESOLVE_DEPTH))])
        self.assertEqual(list(expectation.attachment),
                         [ATTACHMENT, DEPTH_STORE_ATTACHMENT])
        self.assertEqual(expectation.touched, {900, 920, 940})
        self.assertEqual(expectation.written, {900, 920, 940})
        self.assertEqual(expectation.rails, frozenset(MSAA_DEPTH_RESOLVE_RAILS))

    def test_v28_refuses_a_stored_msaa_depth_surface_without_its_resolve(self):
        # The stored surface's texels are only observable as the resolve's
        # reduction, so a case that keeps the surface without stating the
        # filter is refused rather than compared (`research/docs/23` §3.3,
        # v57).
        broken = copy.deepcopy(self.suite)
        del broken["render_cases"][MSAA_DEPTH_RESOLVE_INDEX]["depth_resolve"]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a stored multisampled depth surface needs its depth resolve"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_depth_resolve_without_the_stored_surface(self):
        # The resolve is the stored surface's own tail: a filter beside a
        # surface the pass discards is refused instead of silently ignored
        # (`research/docs/23` §3.3, v57).
        broken = copy.deepcopy(self.suite)
        depth = broken["render_cases"][MSAA_DEPTH_RESOLVE_INDEX]["depth"]
        del depth["store"]
        del depth["allocation"]
        del depth["view"]
        del depth["expected_hex"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "a depth resolve needs a stored depth surface"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_an_unknown_depth_resolve_filter(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_DEPTH_RESOLVE_INDEX]["depth_resolve"] = {
            "filter": "average"}
        with self.assertRaisesRegex(compare.CaptureError,
                                    "unsupported depth resolve filter"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_reports_the_msaa_depth_resolve_case_on_every_rail_its_marker_names(self):
        # The marker is the rule: the v57c marker named the three trace rails,
        # and the v58 recording entry carries the same resolve on both object
        # rails, so a capture on any of the five owes the two landings. The
        # `sample0` filter is the API's own baseline, so no device mask gates
        # this case (`research/docs/23` §3.3, v57c/v58).
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            owes = msaa_depth_resolve_marker(suite, rail)
            for position, case in enumerate(suite["render_cases"]):
                if position != MSAA_DEPTH_RESOLVE_INDEX:
                    case["capture_rails"] = [other_rail(rail)]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(msaa_depth_resolve_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                if owes:
                    compare.validate_capture(suite, digest, report, rail)
                else:
                    with self.assertRaisesRegex(
                            compare.CaptureError,
                            "is not a rail this render case runs on"):
                        compare.validate_capture(suite, digest, report, rail)

    def test_v28_pins_the_msaa_depth_resolve_edge_fixtures(self):
        for index, case_id, resolve_filter, depth_hex, gate, rails in (
                (MSAA_DEPTH_RESOLVE_MIN_EDGE_INDEX, MSAA_DEPTH_RESOLVE_MIN_EDGE_ID,
                 "min", MSAA_DEPTH_RESOLVE_MIN_EDGE_DEPTH, "min",
                 MSAA_DEPTH_RESOLVE_MIN_EDGE_RAILS),
                (MSAA_DEPTH_RESOLVE_MAX_EDGE_INDEX, MSAA_DEPTH_RESOLVE_MAX_EDGE_ID,
                 "max", MSAA_DEPTH_RESOLVE_MAX_EDGE_DEPTH, "max",
                 MSAA_DEPTH_RESOLVE_MAX_EDGE_RAILS)):
            with self.subTest(case=case_id):
                case = self.suite["render_cases"][index]
                self.assertEqual(case["id"], case_id)
                self.assertEqual(case["declaring_case"], DEPTH_RESOLVE_DECLARING_ID)
                self.assertEqual(case["multisample"], {"sample_count": 4})
                self.assertEqual(case["depth_resolve"], {"filter": resolve_filter})
                self.assertEqual(case["requires_depth_resolve_filter"], gate)
                self.assertNotIn("coverage", case)
                self.assertEqual(case["depth"]["format"], "depth32float")
                self.assertEqual(case["depth"]["store"], "store")
                self.assertEqual(case["depth"]["allocation"], DEPTH_RESOLVE_ALLOCATION)
                self.assertEqual(case["depth"]["view"], DEPTH_RESOLVE_VIEW)
                self.assertEqual(case["depth"]["expected_hex"], depth_hex)
                self.assertEqual(case["depth_test"], {"compare": "less", "write": True})
                self.assertEqual(case["expected_hex"], MSAA_DEPTH_RESOLVE_MIN_EDGE_EXPECTED)
                self.assertEqual(case["capture_rails"], list(rails))
                # The stream is the device-gated pair: the near triangle's
                # right edge at x = 0.25 NDC and red tints on both triangles,
                # so the colour side stays uniform and only the depth resolve
                # can tell the two filters apart.
                stream = case["vertex_buffers"][0]["initial_hex"]
                self.assertEqual(len(bytes.fromhex(stream)), 192)
                vertices = [struct.unpack("<8f", bytes.fromhex(stream)[offset:offset + 32])
                            for offset in range(0, 192, 32)]
                self.assertAlmostEqual(vertices[0][0], 0.25)
                self.assertAlmostEqual(vertices[0][2], 0.5)
                self.assertAlmostEqual(vertices[3][2], 0.9)
                for vertex in vertices:
                    self.assertEqual(vertex[4:8], (1.0, 0.0, 0.0, 1.0))

    def test_v28_plans_the_msaa_depth_resolve_edge_fixtures(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        for case_id, depth_hex, gate, rails in (
                (MSAA_DEPTH_RESOLVE_MIN_EDGE_ID, MSAA_DEPTH_RESOLVE_MIN_EDGE_DEPTH, "min",
                 MSAA_DEPTH_RESOLVE_MIN_EDGE_RAILS),
                (MSAA_DEPTH_RESOLVE_MAX_EDGE_ID, MSAA_DEPTH_RESOLVE_MAX_EDGE_DEPTH, "max",
                 MSAA_DEPTH_RESOLVE_MAX_EDGE_RAILS)):
            with self.subTest(case=case_id):
                expectation = plan[case_id]
                self.assertEqual(expectation.writes,
                                 [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                                   bytes.fromhex(MSAA_DEPTH_RESOLVE_MIN_EDGE_EXPECTED)),
                                  ((DEPTH_RESOLVE_ALLOCATION, DEPTH_RESOLVE_VIEW, 0),
                                   bytes.fromhex(depth_hex))])
                self.assertEqual(list(expectation.attachment),
                                 [ATTACHMENT, DEPTH_RESOLVE_ATTACHMENT])
                self.assertEqual(expectation.touched, {900, 920, 960})
                self.assertEqual(expectation.written, {900, 920, 960})
                self.assertEqual(expectation.rails, frozenset(rails))
                self.assertEqual(expectation.filter, gate)

    def test_v28_refuses_a_device_gate_that_does_not_name_the_resolve_filter(self):
        # The gate is the case's own admission condition, so it has to name the
        # resolve the case states rather than drift into a second spelling
        # (`research/docs/23` §3.3, v57d).
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_DEPTH_RESOLVE_MIN_EDGE_INDEX][
            "requires_depth_resolve_filter"] = "max"
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the device gate has to name the resolve filter"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_refuses_a_device_gate_outside_min_and_max(self):
        # Sample0 is the API's own baseline — every device that resolves at all
        # carries it — so only Min and Max are gateable (`research/docs/23`
        # §3.3, v57d).
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_DEPTH_RESOLVE_MAX_EDGE_INDEX][
            "requires_depth_resolve_filter"] = "sample0"
        broken["render_cases"][MSAA_DEPTH_RESOLVE_MAX_EDGE_INDEX]["depth_resolve"] = {
            "filter": "sample0"}
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the device gate names the min or max"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v28_the_marker_names_every_rail_for_the_edge_resolve_cases(self):
        # The marker is the rail half and the mask the device half of one
        # question. The v57f marker named the three trace rails, so only the
        # two object-rail captures had to refuse the pair; the v58 recording
        # entry carries the gated pair on both object rails too, so the marker
        # names every rail and a capture on any rail is owed the pair when its
        # mask carries the filter's bit (`research/docs/23` §3.3,
        # v57d/v57f/v58).
        for index, marker, result_builder, bit in (
                (MSAA_DEPTH_RESOLVE_MIN_EDGE_INDEX, msaa_depth_resolve_min_edge_marker,
                 msaa_depth_resolve_min_edge_result, 2),
                (MSAA_DEPTH_RESOLVE_MAX_EDGE_INDEX, msaa_depth_resolve_max_edge_marker,
                 msaa_depth_resolve_max_edge_result, 4)):
            for rail in ALL_RAILS:
                suite = copy.deepcopy(self.suite)
                marker(suite, rail)
                for position, case in enumerate(suite["render_cases"]):
                    if position != index:
                        case["capture_rails"] = [other_rail(rail)]
                digest = hashlib.sha256(
                    json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
                report = counted_declaring(suite, digest, rail,
                                           depth_resolve_modes=bit)
                report["results"].append(result_builder(rail != "native-metal"))
                with self.subTest(case=suite["render_cases"][index]["id"], rail=rail):
                    compare.validate_capture(suite, digest, report, rail)

    def test_v28_gates_the_edge_resolve_cases_on_the_device_mask(self):
        # Presence iff the bit: absent without the bit passes, present without
        # it is refused, absent with it is refused, and present with it passes
        # (`research/docs/23` §3.3, v57d).
        for index, case_id, bit, filter_name, result_builder in (
                (MSAA_DEPTH_RESOLVE_MIN_EDGE_INDEX, MSAA_DEPTH_RESOLVE_MIN_EDGE_ID,
                 2, "min", msaa_depth_resolve_min_edge_result),
                (MSAA_DEPTH_RESOLVE_MAX_EDGE_INDEX, MSAA_DEPTH_RESOLVE_MAX_EDGE_ID,
                 4, "max", msaa_depth_resolve_max_edge_result)):
            suite = copy.deepcopy(self.suite)
            for position, case in enumerate(suite["render_cases"]):
                if position != index:
                    case["capture_rails"] = ["native-metal"]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            with self.subTest(case=case_id, direction="absent without the bit"):
                report = counted_declaring(suite, digest, "vulkan")
                compare.validate_capture(suite, digest, report, "vulkan")
            with self.subTest(case=case_id, direction="present without the bit"):
                report = counted_declaring(suite, digest, "vulkan")
                report["results"].append(result_builder())
                with self.assertRaisesRegex(
                        compare.CaptureError,
                        f"lacks the {filter_name} depth resolve filter"):
                    compare.validate_capture(suite, digest, report, "vulkan")
            with self.subTest(case=case_id, direction="absent with the bit"):
                report = counted_declaring(suite, digest, "vulkan",
                                           depth_resolve_modes=bit)
                with self.assertRaisesRegex(compare.CaptureError, "missing cases"):
                    compare.validate_capture(suite, digest, report, "vulkan")
            with self.subTest(case=case_id, direction="present with the bit"):
                report = counted_declaring(suite, digest, "vulkan",
                                           depth_resolve_modes=bit)
                report["results"].append(result_builder())
                compare.validate_capture(suite, digest, report, "vulkan")

    def test_v28_refuses_a_rail_the_msaa_marker_does_not_name(self):
        # The v51/v52 shape: a capture whose msaa marker names every rail
        # *except* the one it runs on is refused rather than compared, exactly
        # as the scissor and instanced markers state it.
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][MSAA_INDEX]["capture_rails"] = ["native-metal"]
        digest = hashlib.sha256(
            json.dumps(broken, sort_keys=True).encode("utf-8")).hexdigest()
        for position, case in enumerate(broken["render_cases"]):
            if position != MSAA_INDEX:
                case["capture_rails"] = ["native-metal"]
        report = counted_declaring(broken, digest, "vulkan")
        report["results"].append(msaa_result())
        report["results"].append(msaa_depth_result())
        report["results"].append(msaa_stencil_result())
        with self.assertRaisesRegex(compare.CaptureError,
                                    "is not a rail this render case runs on"):
            compare.validate_capture(broken, digest, report, "vulkan")


if __name__ == "__main__":
    unittest.main()
