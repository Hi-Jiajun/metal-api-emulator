"""The heap observation segment (`research/docs/25` §5.1, Step 5).

`suite-v15.json` (Step 8) does not exist yet: this file pins the rules against a
synthetic suite built from `suite-v13.json`'s declaring compute case plus a heap
section, so the comparator's contract is exercised without a committed fixture
or a GPU. The bytes of the case stay the ordinary writeback comparison; the new
segment only proves the placements landed in one slab at the declared offsets.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V13_PATH = CONFORMANCE / "suite-v13.json"

# The v13 declaring case reads allocation 900 (16 bytes) and writes allocation
# 920 (12 bytes), which is exactly the two-resource placement the first heap
# increment is about (`research/docs/25` §6 Step 3).
HEAP = {
    "size": 512,
    "storage_mode": "owned_bytes",
    "allows_aliasing": False,
    "placements": [
        {"allocation": 900, "offset": 0, "byte_size": 16},
        {"allocation": 920, "offset": 256, "byte_size": 12},
    ],
}
OBSERVATION = {
    "heap": 61,
    "same_slab": True,
    "placements": [
        {"allocation": 900, "offset": 0, "byte_size": 16},
        {"allocation": 920, "offset": 256, "byte_size": 12},
    ],
}
RAILS = ("vulkan",)


def synthetic_suite(heap=HEAP, rails=RAILS):
    """`suite-v13.json`'s declaring case plus (or minus) the heap section."""
    suite = json.loads(V13_PATH.read_text(encoding="utf-8"))
    # The heap rules are about the compute half; the render case of v13 is not
    # part of this file's contract and would be a missing case in every
    # synthetic capture.
    suite.pop("render_cases", None)
    case = suite["cases"][0]
    if heap is not None:
        case["heap"] = copy.deepcopy(heap)
    if rails is not None:
        case["capture_rails"] = list(rails)
    digest = hashlib.sha256(json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
    return suite, digest


def capture_for(suite, digest, backend="vulkan", observation=OBSERVATION, report_case=True):
    report = synthetic_report(suite, digest, backend)
    if backend != "native-metal":
        for result in report["results"]:
            result["copy_in"], result["copy_out"] = 2, 1
    if report_case and observation is not None:
        report["results"][0]["heap"] = copy.deepcopy(observation)
    return report


class HeapDeclarationTests(unittest.TestCase):
    """The suite-side section: one slab, no aliasing, placements that fit."""

    def plan(self, heap):
        suite = json.loads(V13_PATH.read_text(encoding="utf-8"))
        suite["cases"][0]["heap"] = heap
        suite["cases"][0]["capture_rails"] = ["vulkan"]
        return compare._suite_plan(suite)

    def reject(self, heap, message):
        with self.assertRaisesRegex(compare.CaptureError, message):
            self.plan(copy.deepcopy(heap))

    def test_declaration_is_planned_in_allocation_order(self):
        plan = self.plan(copy.deepcopy(HEAP))
        case = plan["render_declaring_copy_word"]
        self.assertIsNotNone(case[4])
        self.assertEqual(case[4].size, 512)
        self.assertEqual(case[4].storage_mode, "owned_bytes")
        self.assertEqual(case[4].placements, ((900, 0, 16), (920, 256, 12)))
        self.assertEqual(case[5], ["vulkan"])

    def test_aliasing_is_refused_in_the_first_increment(self):
        heap = copy.deepcopy(HEAP)
        heap["allows_aliasing"] = True
        self.reject(heap, "refuses aliasing")

    def test_a_placement_outside_the_heap_is_refused(self):
        heap = copy.deepcopy(HEAP)
        heap["size"] = 200
        self.reject(heap, "placement exceeds the heap")

    def test_a_placement_that_does_not_cover_its_allocation_is_refused(self):
        heap = copy.deepcopy(HEAP)
        heap["placements"][0]["byte_size"] = 8
        self.reject(heap, "does not match allocation 900")

    def test_an_unknown_allocation_is_refused(self):
        heap = copy.deepcopy(HEAP)
        heap["placements"][1]["allocation"] = 930
        self.reject(heap, "unknown allocation 930")

    def test_overlapping_placements_are_refused(self):
        heap = copy.deepcopy(HEAP)
        heap["placements"][1]["offset"] = 8
        self.reject(heap, "placements overlap")

    def test_placements_must_be_in_allocation_order(self):
        heap = copy.deepcopy(HEAP)
        heap["placements"].reverse()
        self.reject(heap, "must be in allocation order")

    def test_a_marker_without_a_heap_section_is_refused(self):
        suite = json.loads(V13_PATH.read_text(encoding="utf-8"))
        suite["cases"][0]["capture_rails"] = ["vulkan"]
        with self.assertRaisesRegex(compare.CaptureError, "declared together"):
            compare._suite_plan(suite)

    def test_a_heap_section_without_a_marker_is_refused(self):
        suite = json.loads(V13_PATH.read_text(encoding="utf-8"))
        suite["cases"][0]["heap"] = copy.deepcopy(HEAP)
        with self.assertRaisesRegex(compare.CaptureError, "declared together"):
            compare._suite_plan(suite)

    def test_an_unknown_rail_is_refused(self):
        suite = json.loads(V13_PATH.read_text(encoding="utf-8"))
        suite["cases"][0]["heap"] = copy.deepcopy(HEAP)
        suite["cases"][0]["capture_rails"] = ["lavapipe"]
        with self.assertRaisesRegex(compare.CaptureError, "distinct known backends"):
            compare._suite_plan(suite)


class HeapObservationTests(unittest.TestCase):
    """The capture side: the segment is required where declared and refused elsewhere."""

    def setUp(self):
        self.suite, self.digest = synthetic_suite()

    def validate(self, report, backend="vulkan"):
        compare.validate_capture(self.suite, self.digest, report, backend)

    def reject(self, report, message, backend="vulkan"):
        with self.assertRaisesRegex(compare.CaptureError, message):
            self.validate(report, backend)

    def test_the_named_rail_reports_the_placements(self):
        self.validate(capture_for(self.suite, self.digest))

    def test_a_missing_observation_is_refused(self):
        report = capture_for(self.suite, self.digest, observation=None)
        self.reject(report, "has to report the heap placement")

    def test_an_undeclared_observation_is_refused(self):
        suite, digest = synthetic_suite(heap=None, rails=None)
        report = synthetic_report(suite, digest)
        report["results"][0]["heap"] = copy.deepcopy(OBSERVATION)
        with self.assertRaisesRegex(compare.CaptureError, "declares no heap section"):
            compare.validate_capture(suite, digest, report)

    def test_a_different_slab_is_refused(self):
        observation = copy.deepcopy(OBSERVATION)
        observation["same_slab"] = False
        self.reject(capture_for(self.suite, self.digest, observation=observation),
                    "share one slab")

    def test_a_different_offset_is_refused(self):
        observation = copy.deepcopy(OBSERVATION)
        observation["placements"][1]["offset"] = 128
        self.reject(capture_for(self.suite, self.digest, observation=observation),
                    "placements do not match")

    def test_a_zero_heap_identity_is_refused(self):
        observation = copy.deepcopy(OBSERVATION)
        observation["heap"] = 0
        self.reject(capture_for(self.suite, self.digest, observation=observation),
                    "expected integer")

    def test_a_widened_observation_shape_is_refused(self):
        observation = copy.deepcopy(OBSERVATION)
        observation["target"] = 900
        self.reject(capture_for(self.suite, self.digest, observation=observation),
                    "expected fields")

    def test_an_unnamed_rail_must_not_report_the_case(self):
        # A marker that names another rail is what "unnamed" means now that
        # every committed rail can report a heap case.
        suite, digest = synthetic_suite(rails=("native-metal-provider",))
        report = capture_for(suite, digest)
        with self.assertRaisesRegex(compare.CaptureError,
                                    "not a rail this heap case runs on"):
            compare.validate_capture(suite, digest, report, "vulkan")

    def test_a_rail_without_counters_can_report_the_heap_segment(self):
        # The Apple oracle reports bytes but no device-buffer counters; a heap
        # case marked for it still validates, because the segment is the
        # placement observation and the counters stay a provider surface.
        suite, digest = synthetic_suite(rails=("native-metal",))
        report = capture_for(suite, digest, backend="native-metal")
        compare.validate_capture(suite, digest, report, "native-metal")

    def test_the_heap_section_does_not_change_the_byte_plan(self):
        suite, digest = synthetic_suite(heap=None, rails=None)
        baseline = compare._suite_plan(suite)["render_declaring_copy_word"]
        planned = compare._suite_plan(self.suite)["render_declaring_copy_word"]
        self.assertEqual(baseline[:4], planned[:4])


if __name__ == "__main__":
    unittest.main()
