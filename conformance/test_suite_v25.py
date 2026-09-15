"""Mixed-layout checks for the observation channel of `suite-v25.json`.

These are comparator and schema checks, not GPU execution evidence. The v26
increment generalises the reviewed 8-bit UNORM modules: the same fragment store
lands in whichever channel order each attachment declares, so one draw may write
an `rgba8_unorm` location and a `bgra8_unorm` location side by side
(`research/docs/23` §3.3). The fixture's two texels are therefore `40 80 c0 ff`
and `40 80 ff c0` — the same colour, two layouts — and a capture that reported
one layout's bytes for both locations is refused.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V25_PATH = CONFORMANCE / "suite-v25.json"

DECLARING_ID = "render_declaring_two_attachments"
RENDER_ID = "mixed_layout_dual_2x2"
ATTACHMENTS = ((900, 910, 0, 16), (920, 930, 0, 16))
PROBE = (940, 950, 4)
QUAD_VIEW = (1000, 1010, 0, 32)
INDEX_VIEW = (1020, 1030, 0, 12)
TEXELS = ("4080c0ff" * 4, "4080ffc0" * 4)
ALL_RAILS = ("native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
             "native-metal-provider-objects")
V25_REPORTING_RAILS = ALL_RAILS


def render_result(provider_backend=True, copy_in=3, copy_out=3):
    result = {
        "id": RENDER_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": allocation, "view": view, "offset": offset,
                        "bytes_hex": texels}
                       for (allocation, view, offset, _), texels
                       in zip(ATTACHMENTS, TEXELS)],
        "allocations": [{"allocation": allocation, "bytes_hex": texels}
                        for (allocation, _, _, _), texels in zip(ATTACHMENTS, TEXELS)],
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


class MixedLayoutObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = V25_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v25_pins_the_mixed_layout_fixture(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v25")
        case = self.suite["render_cases"][0]
        self.assertEqual(case["id"], RENDER_ID)
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual([attachment["format"] for attachment in case["attachments"]],
                         ["rgba8_unorm", "bgra8_unorm"])
        self.assertEqual([attachment["expected_hex"] for attachment in case["attachments"]],
                         list(TEXELS))
        module = V25_PATH.parent / case["metal"]["path"]
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                         case["metal"]["sha256"])
        self.assertEqual(sorted(case["capture_rails"]), sorted(V25_REPORTING_RAILS))

    def test_v25_plan_observes_both_layouts(self):
        plan = compare._suite_plan(self.suite)
        expectation = compare._render_plan(plan, self.suite)[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((allocation, view, offset), bytes.fromhex(texels))
                          for (allocation, view, offset, _), texels
                          in zip(ATTACHMENTS, TEXELS)])
        self.assertEqual(expectation.written, {900, 920, 940})
        self.assertEqual(expectation.touched, {900, 920, 940})

    def test_v25_marker_gates_each_rail(self):
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

    def test_v25_refuses_one_layout_for_both_locations(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        wrong = render_result(True)
        wrong["writebacks"][1]["bytes_hex"] = TEXELS[0]
        wrong["allocations"][1]["bytes_hex"] = TEXELS[0]
        report["results"].append(wrong)
        with self.assertRaises(compare.CaptureError):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")


if __name__ == "__main__":
    unittest.main()
