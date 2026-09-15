"""Store/dontcare checks for the observation channel of `suite-v19.json`.

These are comparator and schema checks, not GPU execution evidence. What they
pin is the observation surface the v19 increment (`research/docs/23` §3.6) adds
on top of v18's MRT: one draw writes two colour locations, the second location
is discarded, so a render case marks that attachment `dontcare`, it carries no
`expected_hex`, and the capture owes one writeback and one allocation image —
the stored location only. Reporting the discarded attachment is the increment's
falsifiability point, refused here before any backend runs.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V19_PATH = CONFORMANCE / "suite-v19.json"

DECLARING_ID = "render_declaring_store_and_discard"
RENDER_ID = "discard_second_attachment_2x2"
FIRST = (900, 910, 0)          # stored attachment
SECOND = (920, 930, 0)         # discarded attachment
PROBE = (940, 950, 4)
FIRST_TEXELS = "4080c0ff" * 4
DISCARDED_TEXELS = "01010101" * 4
ALL_RAILS = ("native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
             "native-metal-provider-objects")
# The rails v19 marks: every rail. The three trace rails execute the render
# pass with one stored and one discarded attachment, and the two object rails
# record the store op through the M6 `draw_*_with_attachments` entry points.
V19_REPORTING_RAILS = ALL_RAILS


def render_result(provider_backend=True, copy_in=3, copy_out=2):
    """The attachment observation a reporting rail of v19 has to emit: the
    stored attachment only, never the discarded one."""
    result = {
        "id": RENDER_ID,
        "completion": "CompletedVisible",
        "writebacks": [
            {"allocation": FIRST[0], "view": FIRST[1], "offset": FIRST[2],
             "bytes_hex": FIRST_TEXELS},
        ],
        "allocations": [
            {"allocation": FIRST[0], "bytes_hex": FIRST_TEXELS},
        ],
    }
    if provider_backend:
        # Three uploads (the two attachment views and the declaring pass's
        # probe view) and two readbacks (the probe's landing plus the stored
        # attachment's readback; the discarded attachment is not read back).
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def counted_declaring(suite, digest, rail):
    """The declaring case's result with the v19 compute counts (3 in, 1 out)."""
    report = synthetic_report(suite, digest, rail)
    if rail != "native-metal":
        for result in report["results"]:
            result["copy_in"], result["copy_out"] = 3, 1
    return report


class StoreDontCareObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = V19_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v19_pins_the_stored_and_discarded_attachments(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v19")
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], [RENDER_ID])
        case = self.suite["render_cases"][0]
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual(sorted(case["capture_rails"]), sorted(V19_REPORTING_RAILS))
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
            self.assertEqual((attachment["load"], attachment["clear_hex"]),
                             ("clear", "00000000"))
        # Location 0 is stored and carries the expectation; location 1 is
        # discarded and carries none.
        self.assertEqual(attachments[0]["store"], "store")
        self.assertEqual(attachments[0]["expected_hex"], FIRST_TEXELS)
        self.assertEqual(attachments[1]["store"], "dontcare")
        self.assertNotIn("expected_hex", attachments[1])
        for kind in ("air", "metal"):
            source = V19_PATH.parent / self.suite["cases"][0][kind]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             self.suite["cases"][0][kind]["sha256"], kind)
        module = V19_PATH.parent / case["metal"]["path"]
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                         case["metal"]["sha256"])

    def test_v19_declaring_pass_declares_both_attachment_views_read_only(self):
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

    def test_v19_plan_carries_only_the_stored_attachment(self):
        plan = compare._suite_plan(self.suite)
        render_plan = compare._render_plan(plan, self.suite)
        expectation = render_plan[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((FIRST[0], FIRST[1], FIRST[2]),
                           bytes.fromhex(FIRST_TEXELS))])
        self.assertEqual(expectation.allocations,
                         {FIRST[0]: bytes.fromhex(FIRST_TEXELS)})
        self.assertEqual(expectation.attachment, [(900, 910, 0, 16)])
        # Three allocations are touched by the submission (both attachment
        # views and the probe view), and two are written (the probe's landing
        # plus the stored attachment's landing; the discarded attachment's
        # bytes disappear).
        self.assertEqual(expectation.touched, {900, 920, 940})
        self.assertEqual(expectation.written, {900, 940})
        self.assertEqual(expectation.rails, set(V19_REPORTING_RAILS))
        self.assertIsNone(expectation.present)
        self.assertIsNone(expectation.icb)

    def test_v19_marker_gates_each_rail(self):
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

    def test_v19_refuses_a_rail_whose_marker_does_not_name_it(self):
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

    def test_v19_refuses_a_writeback_of_the_discarded_attachment(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        result = render_result(True)
        result["writebacks"].append(
            {"allocation": SECOND[0], "view": SECOND[1], "offset": SECOND[2],
             "bytes_hex": DISCARDED_TEXELS})
        report["results"].append(result)
        with self.assertRaisesRegex(compare.CaptureError, "writable set mismatch"):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v19_refuses_an_allocation_observation_of_the_discarded_attachment(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        result = render_result(True)
        result["allocations"].append(
            {"allocation": SECOND[0], "bytes_hex": DISCARDED_TEXELS})
        report["results"].append(result)
        with self.assertRaisesRegex(compare.CaptureError,
                                    "unknown allocation 920"):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v19_refuses_a_tampered_stored_attachment_byte_string(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        result = render_result(True)
        result["writebacks"][0]["bytes_hex"] = "00" * 16
        result["allocations"][0]["bytes_hex"] = "00" * 16
        report["results"].append(result)
        with self.assertRaisesRegex(compare.CaptureError,
                                    "writeback allocation 900/view 910"):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v19_count_contract_is_three_in_and_two_out(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        report["results"].append(render_result(True, copy_in=3, copy_out=2))
        compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v19_refuses_wrong_copy_in(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        report["results"].append(render_result(True, copy_in=2, copy_out=2))
        with self.assertRaisesRegex(compare.CaptureError,
                                    "does not match 3 touched allocations"):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v19_refuses_wrong_copy_out(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        report["results"].append(render_result(True, copy_in=3, copy_out=3))
        with self.assertRaisesRegex(compare.CaptureError,
                                    "does not match 2 written allocations"):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v19_refuses_an_expectation_on_a_discarded_attachment(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["attachments"][1]["expected_hex"] = DISCARDED_TEXELS
        with self.assertRaisesRegex(compare.CaptureError,
                                    "a discarded attachment carries no expected_hex"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v19_refuses_a_stored_attachment_without_an_expectation(self):
        broken = copy.deepcopy(self.suite)
        del broken["render_cases"][0]["attachments"][0]["expected_hex"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "a stored attachment needs expected_hex"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v19_refuses_a_pass_that_discards_every_attachment(self):
        broken = copy.deepcopy(self.suite)
        attachments = broken["render_cases"][0]["attachments"]
        attachments[0]["store"] = "dontcare"
        del attachments[0]["expected_hex"]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "every colour attachment discards, leaving no observable landing point"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v19_refuses_a_declaring_pass_that_writes_an_attachment_view(self):
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
