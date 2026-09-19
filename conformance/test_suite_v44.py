"""The layout-free vertex count above the milestone's three vertices.

2026-09-19, census v45's `vertex_span` bucket: the population is one shape —
`1920x1080`, no declared vertex layout, and draws that name **six** vertices
(`290x`) or twenty-four (`4x`). The contract admitted exactly three of them
before this widening; from here the triangle list fixes the count's lower
bound, and the count itself is the draw's own.

These tests are comparator and schema checks, not GPU execution evidence. What
they pin is the suite's half of the widening: the two new cases name counts the
arm now admits, their expectation is the reviewed module's own covered frame,
and the rails they name are the ones whose `vertex_id` module carries the
positions — both native faces compile a three-entry position table, so a case
that named them would claim a capture they cannot report.
"""

import copy
import hashlib
import json
import unittest
from pathlib import Path

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
REPOSITORY = CONFORMANCE.parent
V44_PATH = CONFORMANCE / "suite-v44.json"
PROVIDER_PATH = REPOSITORY / "examples" / "metal-smoke" / "src" / "bin" / "provider-capture.rs"
SPV_DIRECTORY = REPOSITORY / "crates" / "metal-api-vulkan" / "src" / "render_spv"

DECLARING_ID = "render_declaring_copy_word"
SIX_ID = "nonindexed_vertex_id_quad_six_clear_2x2"
FIVE_ID = "nonindexed_vertex_id_five_clear_2x2"
ATTACHMENT = (900, 910, 0, 16)
TEXELS = "4080c0ff" * 4
# The rails whose `vertex_id` module carries the positions the widened count
# names (`research/docs/23` §3.3; the reviewed module's three-entry table is
# what keeps the two native faces out).
VULKAN_RAILS = ("vulkan", "vulkan-objects")


def render_result(case_id):
    """The attachment observation a reporting rail of this suite has to emit."""
    return {
        "id": case_id,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": TEXELS}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": TEXELS}],
        "copy_in": 2,
        "copy_out": 2,
    }


def synthetic_capture(suite, digest, backend="vulkan"):
    """A capture with the two hand-written attachment observations attached."""
    report = synthetic_report(suite, digest, backend)
    for result in report["results"]:
        if result["id"] == DECLARING_ID:
            result["copy_in"], result["copy_out"] = 2, 1
    for case_id in (SIX_ID, FIVE_ID):
        report["results"].append(render_result(case_id))
    return report


class LayoutFreeCountTests(unittest.TestCase):
    def setUp(self):
        self.raw = V44_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def cases(self):
        return {case["id"]: case for case in self.suite["render_cases"]}

    def test_v44_declares_the_widened_counts_on_the_rails_that_carry_them(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v44")
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        cases = self.cases()
        self.assertEqual(sorted(cases), [FIVE_ID, SIX_ID])
        for case_id, vertices in ((SIX_ID, 6), (FIVE_ID, 5)):
            case = cases[case_id]
            self.assertEqual(case["declaring_case"], DECLARING_ID)
            self.assertEqual(case["capture_rails"], list(VULKAN_RAILS))
            self.assertEqual(case["vertices"], vertices)
            self.assertEqual(case["viewport"], [0, 0, 2, 2])
            # The layout-free arm states no vertex layout and no stream: the
            # shape is the `vertex_id` one, with the count as its only other
            # statement.
            self.assertNotIn("vertex_layout", case)
            self.assertNotIn("vertex_buffers", case)
            self.assertNotIn("indices", case)
            self.assertEqual((case["vertex_entry"], case["fragment_entry"]),
                             ("render_fullscreen_triangle", "render_solid_rgba8"))
            attachment = case["attachment"]
            self.assertEqual((attachment["allocation"], attachment["view"]),
                             (ATTACHMENT[0], ATTACHMENT[1]))
            self.assertEqual((attachment["format"], attachment["width"], attachment["height"]),
                             ("rgba8_unorm", 2, 2))
            self.assertEqual((attachment["load"], attachment["store"]), ("clear", "store"))
            self.assertEqual(attachment["clear_hex"], "fefefefe")
            self.assertNotIn("initial_hex", attachment)
            expected = bytes.fromhex(case["expected_hex"])
            self.assertEqual(len(expected), 16)
            self.assertEqual(set(expected[index:index + 4] for index in range(0, 16, 4)),
                             {bytes.fromhex("4080c0ff")})
            self.assertNotEqual(expected[:4], bytes.fromhex(attachment["clear_hex"]))

    def test_v44_pins_the_reviewed_module_and_its_vulkan_sibling(self):
        for case in self.suite["render_cases"]:
            module = V44_PATH.parent / case["metal"]["path"]
            self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                             case["metal"]["sha256"])
        # The reviewed `vertex_id` module reads a three-entry position table by
        # index, which is what the two native faces compile and why the widened
        # cases cannot name them. The Vulkan rail's sibling is a *total*
        # function of the index — its `isTop` select sends every index above the
        # first two to the third corner — so the extra vertices resolve to
        # degenerate triangles there.
        module = (V44_PATH.parent / self.suite["render_cases"][0]["metal"]["path"]).read_text()
        self.assertIn("const float2 positions[3]", module)
        source = (SPV_DIRECTORY / "fullscreen_triangle.vert.spvasm").read_text()
        self.assertIn("OpSGreaterThan", source,
                      "the Vulkan sibling selects the third corner for the later indices")
        # The capture binary embeds that sibling, so the case's rails read the
        # module whose bytes the rail ran.
        self.assertIn('render_spv/fullscreen_triangle.vert.spv',
                      PROVIDER_PATH.read_text(encoding="utf-8"))

    def test_v44_plan_reaches_every_texel_of_both_cases(self):
        plan = compare._suite_plan(self.suite)
        render_plan = compare._render_plan(plan, self.suite)
        for case_id in (SIX_ID, FIVE_ID):
            expectation = render_plan[case_id]
            self.assertEqual(expectation.writes,
                             [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                               bytes.fromhex(TEXELS))])
            self.assertEqual(expectation.allocations, {ATTACHMENT[0]: bytes.fromhex(TEXELS)})
            self.assertEqual(expectation.attachment, ATTACHMENT)
            self.assertEqual(expectation.rails, set(VULKAN_RAILS))

    def test_v44_every_rail_that_reports_the_attachment_validates(self):
        for backend in VULKAN_RAILS:
            with self.subTest(backend=backend):
                compare.validate_capture(self.suite, self.digest,
                                         synthetic_capture(self.suite, self.digest, backend),
                                         backend)

    def test_v44_refuses_tampered_attachment_bytes(self):
        for case_id in (SIX_ID, FIVE_ID):
            with self.subTest(case=case_id):
                report = synthetic_capture(self.suite, self.digest)
                for result in report["results"]:
                    if result["id"] == case_id:
                        # One texel still the clear sentinel: a rail that landed
                        # the milestone's three-vertex frame where the count says
                        # otherwise cannot pass.
                        result["writebacks"][0]["bytes_hex"] = "fefefefe" + TEXELS[8:]
                        result["allocations"][0]["bytes_hex"] = "fefefefe" + TEXELS[8:]
                with self.assertRaisesRegex(compare.CaptureError, "first differing byte"):
                    compare.validate_capture(self.suite, self.digest, report)

    def test_v44_refuses_a_native_rail_on_a_widened_case(self):
        # The two native faces compile the reviewed module whose position table
        # carries exactly three entries, so a widened case that named them would
        # claim a capture they cannot report.
        for rail in ("native-metal", "native-metal-provider",
                     "native-metal-provider-objects"):
            with self.subTest(rail=rail):
                suite = copy.deepcopy(self.suite)
                for case in suite["render_cases"]:
                    case["capture_rails"] = list(VULKAN_RAILS) + [rail]
                with self.assertRaisesRegex(compare.CaptureError,
                                            "capture_rails has to stay inside that list"):
                    compare.validate_capture(suite, self.digest,
                                             synthetic_capture(suite, self.digest))

    def test_v44_refuses_a_count_below_the_triangle(self):
        suite = copy.deepcopy(self.suite)
        for case in suite["render_cases"]:
            case["vertices"] = 2
        with self.assertRaisesRegex(compare.CaptureError, "rasterizes no triangle"):
            compare.validate_capture(suite, self.digest,
                                     synthetic_capture(suite, self.digest))

    def test_v44_admits_the_three_vertex_shape_the_arm_always_carried(self):
        # The widening moves the *upper* end of the arm only: the milestone's
        # three-vertex triangle is admitted by the same rule, on the same rails,
        # with the same frame — the reviewed module's later indices resolve to
        # its third corner, so the six-vertex case lands the three-vertex frame.
        for case_id in (SIX_ID, FIVE_ID):
            with self.subTest(case=case_id):
                suite = copy.deepcopy(self.suite)
                for case in suite["render_cases"]:
                    if case["id"] == case_id:
                        case["vertices"] = 3
                compare.validate_capture(suite, self.digest,
                                         synthetic_capture(suite, self.digest))


if __name__ == "__main__":
    unittest.main()
