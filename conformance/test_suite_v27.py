"""MRT-plus-load checks for `suite-v27.json`.

These are comparator and schema checks, not GPU execution evidence. The v17
increment taught the comparison that a loading pass keeps its uploaded bytes
where the draw misses; v18 taught it that one draw may write several targets.
This suite is the two together (`research/docs/23` §3.3, §11, §12): each of the
two attachments starts from *its own* uploaded bytes, the left column is drawn
in both, and the right column keeps each attachment's own bytes — so a rail that
shared one upload between the two targets, or that reloaded the same bytes for
both, reads the wrong right column somewhere.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V27_PATH = CONFORMANCE / "suite-v27.json"

DECLARING_ID = "render_declaring_two_attachments"
RENDER_ID = "mrt_partial_load_2x2"
ATTACHMENTS = ((900, 910, 0, 16), (920, 930, 0, 16))
PROBE = (940, 950, 4)
QUAD_VIEW = (1000, 1010, 0, 32)
INDEX_VIEW = (1020, 1030, 0, 12)
PREVIOUS = ("fefefefe" * 4, "01010101" * 4)
OUTPUTS = ("4080c0ff", "ff8040c0")
# The fixture's quad covers the left column (texels 0 and 2); the right column
# keeps each attachment's own uploaded bytes.
EXPECTED = tuple(output + previous[8:16] + output + previous[24:32]
                 for output, previous in zip(OUTPUTS, PREVIOUS))
ALL_RAILS = ("native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
             "native-metal-provider-objects")
V27_REPORTING_RAILS = ALL_RAILS


def render_result(provider_backend=True, copy_in=3, copy_out=3):
    result = {
        "id": RENDER_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": allocation, "view": view, "offset": offset,
                        "bytes_hex": texels}
                       for (allocation, view, offset, _), texels
                       in zip(ATTACHMENTS, EXPECTED)],
        "allocations": [{"allocation": allocation, "bytes_hex": texels}
                        for (allocation, _, _, _), texels in zip(ATTACHMENTS, EXPECTED)],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def counted_declaring(suite, digest, rail):
    report = synthetic_report(suite, digest, rail)
    if rail != "native-metal":
        for result in report["results"]:
            result["copy_in"], result["copy_out"] = 3, 1
    return report


class MrtLoadObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = V27_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v27_pins_the_mrt_loading_fixture(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v27")
        case = self.suite["render_cases"][0]
        self.assertEqual(case["id"], RENDER_ID)
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        for attachment, previous, expected in zip(case["attachments"], PREVIOUS,
                                                  EXPECTED):
            self.assertEqual(attachment["load"], "load")
            self.assertEqual(attachment["initial_hex"], previous)
            self.assertEqual(attachment["expected_hex"], expected)
            self.assertNotIn("clear_hex", attachment)
        module = V27_PATH.parent / case["metal"]["path"]
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                         case["metal"]["sha256"])
        self.assertEqual(sorted(case["capture_rails"]), sorted(V27_REPORTING_RAILS))

    def test_v27_plan_keeps_each_attachments_own_bytes(self):
        plan = compare._suite_plan(self.suite)
        expectation = compare._render_plan(plan, self.suite)[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((allocation, view, offset), bytes.fromhex(texels))
                          for (allocation, view, offset, _), texels
                          in zip(ATTACHMENTS, EXPECTED)])
        self.assertEqual(expectation.written, {900, 920, 940})
        self.assertEqual(expectation.touched, {900, 920, 940})

    def test_v27_marker_gates_each_rail(self):
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            for case in suite["render_cases"]:
                case["capture_rails"] = [rail]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(render_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                compare.validate_capture(suite, digest, report, rail)

    def test_v27_refuses_an_expectation_that_swaps_the_uploads(self):
        # Each attachment keeps its *own* bytes: an expectation built from the
        # other attachment's upload is refused at plan time.
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["attachments"][1]["expected_hex"] = EXPECTED[0]
        with self.assertRaises(compare.CaptureError):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v27_refuses_a_clearing_result(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        cleared = render_result(True)
        cleared["writebacks"][1]["bytes_hex"] = EXPECTED[1][:8] + "00000000" + EXPECTED[1][12:]
        report["results"].append(cleared)
        with self.assertRaises(compare.CaptureError):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")


if __name__ == "__main__":
    unittest.main()
