"""Attachment-format checks for the observation channel of `suite-v21.json`.

These are comparator and schema checks, not GPU execution evidence. What they
pin is the observation surface the v21 increment (`research/docs/23` §3.3) adds
on top of the v13 render shape: the attachment may declare `bgra8_unorm` as
well as `rgba8_unorm`. The reviewed fragment stage stores the same colour either
way — which channel lands in which byte is the *attachment format's* decision —
so the fixture's expected texels are `c0 80 40 ff` per texel for a B,G,R,A
layout, and the comparator has to admit the second layout without loosening any
of the existing shape rules.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V21_PATH = CONFORMANCE / "suite-v21.json"

DECLARING_ID = "render_declaring_copy_word"
RENDER_ID = "bgra8_clear_2x2"
ATTACHMENT = (900, 910, 0, 16)
PROBE = (920, 930, 4)
QUAD_VIEW = (960, 970, 0, 32)
INDEX_VIEW = (980, 990, 0, 12)
DECLARED = "fefefefe"
# The same float4 store as the `rgba8_unorm` fixtures (64/255, 128/255,
# 192/255, 1) landing in a B8G8R8A8_UNORM attachment: the memory bytes carry
# blue first, so the texel is `c0 80 40 ff` (`research/docs/23` §3.3, v21).
TEXELS = "c08040ff" * 4
ALL_RAILS = ("native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
             "native-metal-provider-objects")
V21_REPORTING_RAILS = ALL_RAILS


def render_result(provider_backend=True, copy_in=2, copy_out=2):
    """The attachment observation a reporting rail of v21 has to emit."""
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
    """The declaring case's result with the v13 compute counts (2 in, 1 out)."""
    report = synthetic_report(suite, digest, rail)
    if rail != "native-metal":
        for result in report["results"]:
            result["copy_in"], result["copy_out"] = 2, 1
    return report


class AttachmentFormatObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = V21_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v21_pins_the_bgra8_clear_quad(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v21")
        self.assertEqual(self.suite["guard_byte"], 0)
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], [RENDER_ID])
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["entry"], "copy_word")
        for kind in ("air", "metal"):
            source = V21_PATH.parent / declaring[kind]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             declaring[kind]["sha256"], kind)
        buffers = {buffer["binding"]: buffer for buffer in declaring["buffers"]}
        self.assertEqual(buffers[0]["access"], "read")
        self.assertEqual((buffers[0]["allocation"], buffers[0]["view"],
                          buffers[0]["offset"], buffers[0]["length"],
                          buffers[0]["allocation_size"]), ATTACHMENT + (16,))
        self.assertEqual(buffers[0]["initial_hex"], DECLARED * 4)
        self.assertEqual(buffers[1]["access"], "write")
        self.assertEqual((buffers[1]["allocation"], buffers[1]["view"],
                          buffers[1]["offset"], buffers[1]["length"],
                          buffers[1]["allocation_size"]), (920, 930, 4, 4, 12))
        case = self.suite["render_cases"][0]
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        module = V21_PATH.parent / case["metal"]["path"]
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                         case["metal"]["sha256"])
        self.assertEqual(case["vertices"], 6)
        self.assertEqual(case["viewport"], [0, 0, 2, 2])
        vertex = case["vertex_buffers"][0]
        self.assertEqual((vertex["allocation"], vertex["view"], vertex["offset"],
                          vertex["length"]), QUAD_VIEW)
        indices = case["indices"]
        self.assertEqual((indices["allocation"], indices["view"], indices["offset"],
                          indices["length"], indices["format"]),
                         INDEX_VIEW + ("uint16",))
        attachment = case["attachment"]
        self.assertEqual((attachment["allocation"], attachment["view"],
                          attachment["format"], attachment["width"],
                          attachment["height"]),
                         (900, 910, "bgra8_unorm", 2, 2))
        self.assertEqual((attachment["load"], attachment["store"]),
                         ("clear", "store"))
        self.assertEqual(attachment["clear_hex"], "00000000")
        self.assertNotIn("initial_hex", attachment)
        self.assertEqual(case["expected_hex"], TEXELS)
        self.assertEqual(sorted(case["capture_rails"]), sorted(V21_REPORTING_RAILS))

    def test_v21_plan_observes_the_bgra8_layout(self):
        plan = compare._suite_plan(self.suite)
        render_plan = compare._render_plan(plan, self.suite)
        expectation = render_plan[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(TEXELS))])
        self.assertEqual(expectation.allocations,
                         {ATTACHMENT[0]: bytes.fromhex(TEXELS)})
        self.assertEqual(expectation.attachment, ATTACHMENT)
        self.assertEqual(expectation.touched, {ATTACHMENT[0], PROBE[0]})
        self.assertEqual(expectation.written, {ATTACHMENT[0], PROBE[0]})
        self.assertEqual(expectation.rails, set(V21_REPORTING_RAILS))

    def test_v21_marker_gates_each_rail(self):
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

    def test_v21_refuses_a_rail_whose_marker_does_not_name_it(self):
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

    def test_v21_refuses_an_unsupported_attachment_format(self):
        # `R32Float` is admitted by the contract and both rails, but the suite
        # observation surface does not pin its component shape yet, so the
        # comparator keeps refusing it rather than inventing a byte rule.
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["attachment"]["format"] = "r32float"
        with self.assertRaisesRegex(compare.CaptureError,
                                    "unsupported attachment format"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v21_refuses_the_rgba8_bytes_in_a_bgra8_attachment(self):
        # The layout is the point of the fixture: the same colour lands in a
        # different byte order, so the `rgba8_unorm` texel is a wrong answer
        # here and the comparison has to refuse it.
        report = counted_declaring(self.suite, self.digest, "vulkan")
        swapped = render_result(True)
        swapped["writebacks"][0]["bytes_hex"] = "4080c0ff" * 4
        swapped["allocations"][0]["bytes_hex"] = "4080c0ff" * 4
        report["results"].append(swapped)
        with self.assertRaises(compare.CaptureError):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v21_refuses_a_clear_colour_equal_to_the_expectation(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["attachment"]["clear_hex"] = "c08040ff"
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the clear colour equals the expected texel"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v21_count_contract_is_two_in_and_two_out(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        report["results"].append(render_result(True, copy_in=2, copy_out=2))
        compare.validate_capture(self.suite, self.digest, report, "vulkan")


if __name__ == "__main__":
    unittest.main()
