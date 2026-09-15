"""Single-channel float attachment checks for `suite-v22.json`.

These are comparator and schema checks, not GPU execution evidence. What they
pin is the third attachment format the observation surface admits
(`research/docs/23` §3.3, v22): `r32float` stores one component, and the
reviewed fragment stage hands it `64/255` — whose little-endian `f32` bytes are
`81 80 80 3e`. The fixture therefore proves two things at once: the comparator
can observe a one-component attachment, and a capture that reports the
four-component `4080c0ff` texel of the UNORM fixtures is refused.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V22_PATH = CONFORMANCE / "suite-v22.json"

DECLARING_ID = "render_declaring_copy_word"
RENDER_ID = "r32float_clear_2x2"
ATTACHMENT = (900, 910, 0, 16)
PROBE = (920, 930, 4)
QUAD_VIEW = (960, 970, 0, 32)
INDEX_VIEW = (980, 990, 0, 12)
DECLARED = "fefefefe"
# `float 64/255` (`0x3e808081`) little-endian, one component per texel.
TEXELS = "8180803e" * 4
ALL_RAILS = ("native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
             "native-metal-provider-objects")
# The rails v22 marks: every rail. The single-channel float stage has a reviewed
# SPIR-V module on the Vulkan rail and a reviewed MSL sibling
# (`shaders/quad_indexed_2x2_r32f.metal`) on the native rails, so both halves of
# the rail set execute the same one-component store.
V22_REPORTING_RAILS = ALL_RAILS


def render_result(provider_backend=True, copy_in=2, copy_out=2):
    """The attachment observation a reporting rail of v22 has to emit."""
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


class SingleChannelAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.raw = V22_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v22_pins_the_r32float_clear_quad(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v22")
        self.assertEqual(self.suite["guard_byte"], 0)
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], [RENDER_ID])
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["entry"], "copy_word")
        for kind in ("air", "metal"):
            source = V22_PATH.parent / declaring[kind]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             declaring[kind]["sha256"], kind)
        buffers = {buffer["binding"]: buffer for buffer in declaring["buffers"]}
        self.assertEqual(buffers[0]["access"], "read")
        self.assertEqual((buffers[0]["allocation"], buffers[0]["view"],
                          buffers[0]["offset"], buffers[0]["length"],
                          buffers[0]["allocation_size"]), ATTACHMENT + (16,))
        self.assertEqual(buffers[0]["initial_hex"], DECLARED * 4)
        case = self.suite["render_cases"][0]
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        module = V22_PATH.parent / case["metal"]["path"]
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
                         (900, 910, "r32float", 2, 2))
        self.assertEqual((attachment["load"], attachment["store"]),
                         ("clear", "store"))
        self.assertEqual(attachment["clear_hex"], "00000000")
        self.assertNotIn("initial_hex", attachment)
        self.assertEqual(case["expected_hex"], TEXELS)
        self.assertEqual(sorted(case["capture_rails"]), sorted(V22_REPORTING_RAILS))

    def test_v22_plan_observes_the_single_component_texel(self):
        plan = compare._suite_plan(self.suite)
        render_plan = compare._render_plan(plan, self.suite)
        expectation = render_plan[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(TEXELS))])
        self.assertEqual(expectation.allocations,
                         {ATTACHMENT[0]: bytes.fromhex(TEXELS)})
        self.assertEqual(expectation.touched, {ATTACHMENT[0], PROBE[0]})
        self.assertEqual(expectation.written, {ATTACHMENT[0], PROBE[0]})
        self.assertEqual(expectation.rails, set(V22_REPORTING_RAILS))

    def test_v22_marker_gates_each_rail(self):
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

    def test_v22_refuses_a_rail_whose_marker_does_not_name_it(self):
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

    def test_v22_refuses_the_unorm_texel_in_a_float_attachment(self):
        # The component shape is the point of the fixture: a capture that
        # reports the four-channel UNORM texel is reporting another format's
        # bytes, so the comparison has to refuse it.
        report = counted_declaring(self.suite, self.digest, "vulkan")
        wrong = render_result(True)
        wrong["writebacks"][0]["bytes_hex"] = "4080c0ff" * 4
        wrong["allocations"][0]["bytes_hex"] = "4080c0ff" * 4
        report["results"].append(wrong)
        with self.assertRaises(compare.CaptureError):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v22_refuses_an_unsupported_attachment_format(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["attachment"]["format"] = "r32_uint"
        with self.assertRaisesRegex(compare.CaptureError,
                                    "unsupported attachment format"):
            compare._render_plan(compare._suite_plan(broken), broken)


if __name__ == "__main__":
    unittest.main()
