"""MRT checks for the observation channel of `suite-v18.json`.

These are comparator and schema checks, not GPU execution evidence. What they
pin is the observation surface the MRT increment (`research/docs/23` wave3)
adds on top of v17's loading pass: one draw writes two colour locations, so a
render case declares an `attachments` list whose entries each carry their own
`expected_hex`, the comparator owes one writeback and one allocation image per
location, and the two locations have to read back different texels or the
"two outputs landed" claim is not falsifiable.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V18_PATH = CONFORMANCE / "suite-v18.json"

DECLARING_ID = "render_declaring_two_attachments"
RENDER_ID = "mrt_dual_output_2x2"
FIRST = (900, 910, 0)
SECOND = (920, 930, 0)
PROBE = (940, 950, 4)
FIRST_TEXELS = "4080c0ff" * 4
SECOND_TEXELS = "ff8040c0" * 4
ALL_RAILS = ("native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
             "native-metal-provider-objects")
# The rails v18 marks: every rail. The three trace rails execute the
# dual-output render pass, and the two object rails record it through the
# M6 `draw_*_with_attachments` entry points.
V18_REPORTING_RAILS = ALL_RAILS


def render_result(provider_backend=True, copy_in=3, copy_out=3):
    """The attachment observation a reporting rail of v18 has to emit."""
    result = {
        "id": RENDER_ID,
        "completion": "CompletedVisible",
        "writebacks": [
            {"allocation": FIRST[0], "view": FIRST[1], "offset": FIRST[2],
             "bytes_hex": FIRST_TEXELS},
            {"allocation": SECOND[0], "view": SECOND[1], "offset": SECOND[2],
             "bytes_hex": SECOND_TEXELS},
        ],
        "allocations": [
            {"allocation": FIRST[0], "bytes_hex": FIRST_TEXELS},
            {"allocation": SECOND[0], "bytes_hex": SECOND_TEXELS},
        ],
    }
    if provider_backend:
        # Three uploads (the two attachment views and the declaring pass's
        # probe view) and three readbacks (the probe's landing plus one
        # attachment readback per location).
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def counted_declaring(suite, digest, rail):
    """The declaring case's result with the v18 compute counts (3 in, 1 out)."""
    report = synthetic_report(suite, digest, rail)
    if rail != "native-metal":
        for result in report["results"]:
            result["copy_in"], result["copy_out"] = 3, 1
    return report


class DualOutputObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = V18_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v18_pins_the_dual_attachments_and_their_locations(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v18")
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], [RENDER_ID])
        case = self.suite["render_cases"][0]
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual(sorted(case["capture_rails"]), sorted(V18_REPORTING_RAILS))
        self.assertEqual(case["vertices"], 6)
        self.assertEqual(case["viewport"], [0, 0, 2, 2])
        self.assertEqual((case["vertex_entry"], case["fragment_entry"]),
                         ("render_quad_vertex", "render_solid_rgba8_dual"))
        self.assertNotIn("attachment", case)
        self.assertNotIn("expected_hex", case)
        attachments = case["attachments"]
        self.assertEqual(len(attachments), 2)
        self.assertEqual([(attachment["allocation"], attachment["view"])
                          for attachment in attachments],
                         [(FIRST[0], FIRST[1]), (SECOND[0], SECOND[1])])
        for attachment in attachments:
            self.assertEqual((attachment["format"], attachment["width"],
                              attachment["height"]), ("rgba8_unorm", 2, 2))
            self.assertEqual((attachment["load"], attachment["store"],
                              attachment["clear_hex"]),
                             ("clear", "store", "00000000"))
            self.assertEqual(len(attachment["expected_hex"]), 32)
        self.assertEqual(attachments[0]["expected_hex"], FIRST_TEXELS)
        self.assertEqual(attachments[1]["expected_hex"], SECOND_TEXELS)
        # The two locations read back different texels, which is what keeps a
        # capture that wrote one output twice from passing.
        self.assertNotEqual(attachments[0]["expected_hex"], attachments[1]["expected_hex"])
        for kind in ("air", "metal"):
            source = V18_PATH.parent / self.suite["cases"][0][kind]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             self.suite["cases"][0][kind]["sha256"], kind)
        module = V18_PATH.parent / case["metal"]["path"]
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                         case["metal"]["sha256"])

    def test_v18_declaring_pass_declares_both_attachment_views_read_only(self):
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["entry"], "mrt_declare")
        self.assertEqual([declaring["grid"], declaring["local"]], [[1, 1, 1], [1, 1, 1]])
        buffers = {buffer["binding"]: buffer for buffer in declaring["buffers"]}
        self.assertEqual(buffers[0]["access"], "read")
        self.assertEqual((buffers[0]["allocation"], buffers[0]["view"],
                          buffers[0]["offset"], buffers[0]["length"],
                          buffers[0]["allocation_size"]), (900, 910, 0, 16, 16))
        self.assertEqual(buffers[1]["access"], "read")
        self.assertEqual((buffers[1]["allocation"], buffers[1]["view"],
                          buffers[1]["offset"], buffers[1]["length"],
                          buffers[1]["allocation_size"]), (920, 930, 0, 16, 16))
        self.assertEqual(buffers[2]["access"], "write")
        self.assertEqual((buffers[2]["allocation"], buffers[2]["view"],
                          buffers[2]["offset"], buffers[2]["length"],
                          buffers[2]["allocation_size"]), (940, 950, 4, 4, 16))
        self.assertEqual(declaring["expected_writebacks"],
                         [{"allocation": 940, "view": 950, "offset": 4,
                           "bytes_hex": "ffffffff"}])

    def test_v18_plan_carries_one_writeback_and_allocation_per_location(self):
        plan = compare._suite_plan(self.suite)
        render_plan = compare._render_plan(plan, self.suite)
        expectation = render_plan[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((FIRST[0], FIRST[1], FIRST[2]),
                           bytes.fromhex(FIRST_TEXELS)),
                          ((SECOND[0], SECOND[1], SECOND[2]),
                           bytes.fromhex(SECOND_TEXELS))])
        self.assertEqual(expectation.allocations,
                         {FIRST[0]: bytes.fromhex(FIRST_TEXELS),
                          SECOND[0]: bytes.fromhex(SECOND_TEXELS)})
        self.assertEqual(expectation.attachment,
                         [(900, 910, 0, 16), (920, 930, 0, 16)])
        # Three allocations are touched by the submission (both attachment
        # views and the probe view), and three are written (the probe's
        # landing plus one attachment landing per location).
        self.assertEqual(expectation.touched, {900, 920, 940})
        self.assertEqual(expectation.written, {900, 920, 940})
        self.assertEqual(expectation.rails, set(V18_REPORTING_RAILS))
        self.assertIsNone(expectation.present)
        self.assertIsNone(expectation.icb)

    def test_v18_marker_gates_each_rail(self):
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

    def test_v18_refuses_a_rail_whose_marker_does_not_name_it(self):
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            for case in suite["render_cases"]:
                case["capture_rails"] = [other for other in ALL_RAILS if other != rail]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = counted_declaring(suite, digest, rail)
            report["results"].append(render_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                with self.assertRaisesRegex(compare.CaptureError,
                                            "is not a rail this render case runs on"):
                    compare.validate_capture(suite, digest, report, rail)

    def test_v18_refuses_a_tampered_attachment_byte_string(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        result = render_result(True)
        result["writebacks"][1]["bytes_hex"] = "00" * 16
        result["allocations"][1]["bytes_hex"] = "00" * 16
        report["results"].append(result)
        with self.assertRaisesRegex(compare.CaptureError,
                                    "writeback allocation 920/view 930"):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v18_refuses_swapped_locations(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        result = render_result(True)
        result["writebacks"][0]["bytes_hex"] = SECOND_TEXELS
        result["writebacks"][1]["bytes_hex"] = FIRST_TEXELS
        result["allocations"][0]["bytes_hex"] = SECOND_TEXELS
        result["allocations"][1]["bytes_hex"] = FIRST_TEXELS
        report["results"].append(result)
        with self.assertRaisesRegex(compare.CaptureError,
                                    "writeback allocation 900/view 910"):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v18_count_contract_is_three_in_and_three_out(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        report["results"].append(render_result(True, copy_in=3, copy_out=3))
        compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v18_refuses_wrong_copy_in(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        report["results"].append(render_result(True, copy_in=2, copy_out=3))
        with self.assertRaisesRegex(compare.CaptureError,
                                    "does not match 3 touched allocations"):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v18_refuses_wrong_copy_out(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        report["results"].append(render_result(True, copy_in=3, copy_out=2))
        with self.assertRaisesRegex(compare.CaptureError,
                                    "does not match 3 written allocations"):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v18_refuses_both_attachment_shapes(self):
        broken = copy.deepcopy(self.suite)
        case = broken["render_cases"][0]
        case["attachment"] = copy.deepcopy(case["attachments"][0])
        case["expected_hex"] = case["attachments"][0]["expected_hex"]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "exactly one of attachment and attachments is required"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v18_refuses_a_list_plus_a_case_level_expectation(self):
        broken = copy.deepcopy(self.suite)
        case = broken["render_cases"][0]
        case["expected_hex"] = case["attachments"][0]["expected_hex"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "carries its own expected_hex"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v18_refuses_two_locations_that_read_back_the_same_texels(self):
        broken = copy.deepcopy(self.suite)
        case = broken["render_cases"][0]
        case["attachments"][1]["expected_hex"] = case["attachments"][0]["expected_hex"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the attachments read back the same texels"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v18_refuses_a_declaring_pass_that_writes_an_attachment_view(self):
        broken = copy.deepcopy(self.suite)
        broken["cases"][0]["buffers"][0]["access"] = "read_write"
        broken["cases"][0]["expected_writebacks"].insert(
            0, {"allocation": 900, "view": 910, "offset": 0,
                "bytes_hex": "fefefefe" * 4})
        with self.assertRaisesRegex(compare.CaptureError,
                                    "must only read the attachment view"):
            compare._render_plan(compare._suite_plan(broken), broken)


if __name__ == "__main__":
    unittest.main()
