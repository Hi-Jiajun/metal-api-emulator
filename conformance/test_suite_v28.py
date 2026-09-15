"""Scissor checks for `suite-v28.json`.

These are comparator and schema checks, not GPU execution evidence. What they
pin is the first dynamic-state field the contract carries (`research/docs/23`
§3.3, v29): a pass may clip its draw to a rectangle, and the comparison then
knows the coverage exactly — inside the rectangle every texel is the fragment
output, outside it every texel keeps the clear colour. A scissor that covers the
whole attachment (or nothing) is refused, because it could not show that the
rail executed it.
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
RENDER_ID = "scissor_left_half_4x4"
ATTACHMENT = (900, 910, 0, 64)
PROBE = (920, 930, 4)
QUAD_VIEW = (1000, 1010, 0, 32)
INDEX_VIEW = (1020, 1030, 0, 12)
SCISSOR = [0, 0, 2, 4]
OUTPUT = "4080c0ff"
CLEAR = "11223344"
EXPECTED = "".join(OUTPUT if (index % 4) < 2 else CLEAR for index in range(16))
# The two object rails cannot express a scissor yet (their encoder has no field
# for it), so the fixture names the three trace rails only.
TRACE_RAILS = ("native-metal", "vulkan", "native-metal-provider")
OBJECT_RAILS = ("vulkan-objects", "native-metal-provider-objects")


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


def counted_declaring(suite, digest, rail):
    report = synthetic_report(suite, digest, rail)
    if rail != "native-metal":
        for result in report["results"]:
            result["copy_in"], result["copy_out"] = 2, 1
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
        self.assertEqual(sorted(case["capture_rails"]), sorted(TRACE_RAILS))

    def test_v28_plan_knows_the_coverage(self):
        plan = compare._suite_plan(self.suite)
        expectation = compare._render_plan(plan, self.suite)[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(EXPECTED))])
        self.assertEqual(expectation.touched, {900, 920})
        self.assertEqual(expectation.written, {900, 920})

    def test_v28_clips_on_every_trace_rail(self):
        for rail in TRACE_RAILS:
            suite = copy.deepcopy(self.suite)
            for case in suite["render_cases"]:
                case["capture_rails"] = [rail]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(render_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                compare.validate_capture(suite, digest, report, rail)

    def test_v28_refuses_an_object_rail_that_reports_the_case(self):
        # The marker does not name the object rails, so a capture from one of
        # them that reports the case is refused rather than compared.
        for rail in OBJECT_RAILS:
            report = counted_declaring(self.suite, self.digest, rail)
            report["results"].append(render_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                with self.assertRaisesRegex(compare.CaptureError,
                                            "is not a rail this render case runs on"):
                    compare.validate_capture(self.suite, self.digest, report, rail)

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
