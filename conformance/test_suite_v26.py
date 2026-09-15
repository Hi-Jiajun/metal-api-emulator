"""Four-by-four attachment checks for `suite-v26.json`.

These are comparator and schema checks, not GPU execution evidence. What they
pin is the extent ceiling the rails execute from v27 on (`research/docs/23`
§3.3): the reviewed fragment stages are extent-independent, so a 4x4 attachment
is the same draw with sixteen texels — but the declaring view, the expectation
and the copy-out counter all have to grow with it. The fixture's declaring view
is 64 bytes of `cdcdcdcd`; a rail that read back only a corner would report the
wrong length and be refused.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V26_PATH = CONFORMANCE / "suite-v26.json"

DECLARING_ID = "render_declaring_quad_extent"
RENDER_ID = "quad_extent_clear_4x4"
ATTACHMENT = (900, 910, 0, 64)
PROBE = (920, 930, 4)
QUAD_VIEW = (1000, 1010, 0, 32)
INDEX_VIEW = (1020, 1030, 0, 12)
TEXELS = "4080c0ff" * 16
ALL_RAILS = ("native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
             "native-metal-provider-objects")
V26_REPORTING_RAILS = ALL_RAILS


def render_result(provider_backend=True, copy_in=2, copy_out=2):
    result = {
        "id": RENDER_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": TEXELS}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": TEXELS}],
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


class QuadExtentObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = V26_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v26_pins_the_4x4_fixture(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v26")
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["id"], DECLARING_ID)
        buffers = {buffer["binding"]: buffer for buffer in declaring["buffers"]}
        self.assertEqual((buffers[0]["access"], buffers[0]["length"],
                          buffers[0]["allocation_size"]),
                         ("read", 64, 64))
        case = self.suite["render_cases"][0]
        attachment = case["attachment"]
        self.assertEqual((attachment["width"], attachment["height"]),
                         (4, 4))
        self.assertEqual(case["expected_hex"], TEXELS)
        self.assertEqual(case["viewport"], [0, 0, 4, 4])
        self.assertEqual(sorted(case["capture_rails"]), sorted(V26_REPORTING_RAILS))

    def test_v26_plan_observes_sixteen_texels(self):
        plan = compare._suite_plan(self.suite)
        expectation = compare._render_plan(plan, self.suite)[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(TEXELS))])
        self.assertEqual(len(expectation.allocations[ATTACHMENT[0]]), 64)
        self.assertEqual(expectation.touched, {900, 920})
        self.assertEqual(expectation.written, {900, 920})

    def test_v26_marker_gates_each_rail(self):
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

    def test_v26_refuses_an_eight_texel_axis(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["attachment"]["width"] = 8
        with self.assertRaisesRegex(compare.CaptureError,
                                    "one to four texels per axis"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v26_refuses_a_short_expectation(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["expected_hex"] = "4080c0ff" * 4
        with self.assertRaises(compare.CaptureError):
            compare._render_plan(compare._suite_plan(broken), broken)


if __name__ == "__main__":
    unittest.main()
