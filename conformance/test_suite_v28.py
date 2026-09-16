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
touched and written allocation behind the provider counts.
"""

import copy
import hashlib
import json
from pathlib import Path
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
# The reviewed depth resource of the v43 fixture: the allocation and the view
# the stored texels land in, and the view's whole extent.
DEPTH_STORE_ALLOCATION = 940
DEPTH_STORE_VIEW = 950
DEPTH_STORE_ATTACHMENT = (DEPTH_STORE_ALLOCATION, DEPTH_STORE_VIEW, 0, 64)
ALIGNMENT_ID = "top_half_quad_4x4"
CULL_ID = "cull_back_half_quad_4x4"
BLEND_ID = "blend_alpha_quad_4x4"
# The v43 case sits between the v36 depth pair and the v38 alignment fixture, so
# every case after it moved by one position.
DEPTH_STORE_INDEX = 5
ALIGNMENT_INDEX = 6
CULL_INDEX = 7
BLEND_INDEX = 8
REVIEWED_ORDER = (RENDER_ID, INSTANCED_ID, WILDCARD_ID, BASE_VERTEX_ID, DEPTH_ID,
                  DEPTH_STORE_ID, ALIGNMENT_ID, CULL_ID, BLEND_ID)
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

    The committed marker names the three trace rails. A capture on one of them
    is owed the stored depth observation, so its marker names that rail — the
    same rewriting the scissor and instanced loops apply to their own case. A
    capture on any other rail owes nothing, and its marker has to name another
    rail instead, which keeps the case out of both the required set and the
    report. Returns whether `rail` owes the case.
    """
    suite["render_cases"][DEPTH_STORE_INDEX]["capture_rails"] = (
        [rail] if rail in DEPTH_STORE_RAILS else [other_rail(rail)])
    return rail in DEPTH_STORE_RAILS


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


def counted_declaring(suite, digest, rail):
    report = synthetic_report(suite, digest, rail)
    if rail != "native-metal":
        for result in report["results"]:
            # The v43 declaring case reads one more view than its v27 sibling:
            # it declares the depth attachment's own view beside the colour
            # attachment's, so it touches three allocations and still writes
            # one (`research/docs/23` §3.3, v43).
            result["copy_in"] = 3 if result["id"] == DEPTH_DECLARING_ID else 2
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
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
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
            if rail in ALIGNMENT_RAILS:
                report["results"].append(alignment_result(rail != "native-metal"))
            if rail in CULL_RAILS:
                report["results"].append(cull_result(rail != "native-metal"))
            if rail in BLEND_RAILS:
                report["results"].append(blend_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                if not owes_depth_store:
                    self.assertNotIn(DEPTH_STORE_ID,
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
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(instanced_result(rail != "native-metal"))
            report["results"].append(wildcard_result(rail != "native-metal"))
            if rail in BASE_VERTEX_RAILS:
                report["results"].append(base_vertex_result(rail != "native-metal"))
            if rail in DEPTH_RAILS:
                report["results"].append(depth_result(rail != "native-metal"))
            if owes_depth_store:
                report["results"].append(depth_store_result(rail != "native-metal"))
            if rail in ALIGNMENT_RAILS:
                report["results"].append(alignment_result(rail != "native-metal"))
            if rail in CULL_RAILS:
                report["results"].append(cull_result(rail != "native-metal"))
            if rail in BLEND_RAILS:
                report["results"].append(blend_result(rail != "native-metal"))
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
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
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
            if rail in ALIGNMENT_RAILS:
                report["results"].append(alignment_result(rail != "native-metal"))
            if rail in CULL_RAILS:
                report["results"].append(cull_result(rail != "native-metal"))
            if rail in BLEND_RAILS:
                report["results"].append(blend_result(rail != "native-metal"))
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
        report = counted_declaring(self.suite, digest, "vulkan")
        report["results"].append(render_result())
        report["results"].append(instanced_result())
        report["results"].append(wildcard_result())
        report["results"].append(base_vertex_result())
        report["results"].append(depth_result())
        report["results"].append(depth_store_result())
        report["results"].append(alignment_result())
        report["results"].append(cull_result())
        report["results"].append(blend_result())
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
        # allocation behind the provider counts (three and three); every other
        # render case keeps the declaring pass's two and two.
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        for case in self.suite["render_cases"]:
            expectation = plan[case["id"]]
            counts = (len(expectation.touched), len(expectation.written))
            with self.subTest(case=case["id"]):
                self.assertEqual(counts,
                                 (3, 3) if case["id"] == DEPTH_STORE_ID else (2, 2))

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
        report["results"].append(alignment_result())
        report["results"].append(cull_result())
        report["results"].append(blend_result())
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
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(instanced_result(rail != "native-metal"))
            report["results"].append(wildcard_result(rail != "native-metal"))
            if rail in BASE_VERTEX_RAILS:
                report["results"].append(base_vertex_result(rail != "native-metal"))
            if rail in DEPTH_RAILS:
                report["results"].append(depth_result(rail != "native-metal"))
            if owes_depth_store:
                report["results"].append(depth_store_result(rail != "native-metal"))
            if rail in ALIGNMENT_RAILS:
                report["results"].append(alignment_result(rail != "native-metal"))
            if rail in CULL_RAILS:
                report["results"].append(cull_result(rail != "native-metal"))
            if rail in BLEND_RAILS:
                report["results"].append(blend_result(rail != "native-metal"))
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


if __name__ == "__main__":
    unittest.main()
