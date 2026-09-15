"""Undefined-load checks for the observation channel of `suite-v20.json`.

These are comparator and schema checks, not GPU execution evidence. What they
pin is the observation surface the v20 increment (`research/docs/23` §13) adds
on top of v16's vertex-input quad: a render case may declare `load: "dontcare"`
so the pass starts from undefined pre-pass contents. The fixture's declaring
case still pins the attachment view at `cdcdcdcd`, but a `dontcare` load never
hands those bytes to the pass — so the comparator refuses a fixture that
carries `clear_hex` or `initial_hex` on the attachment, one whose expectation
equals the declared bytes, and one whose expectation is not a uniform fragment
output.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V20_PATH = CONFORMANCE / "suite-v20.json"

DECLARING_ID = "render_declaring_copy_word"
RENDER_ID = "dontcare_load_quad_2x2"
ATTACHMENT = (900, 910, 0, 16)
PROBE = (920, 930, 4)
QUAD_VIEW = (960, 970, 0, 32)
INDEX_VIEW = (980, 990, 0, 12)
DECLARED = "cdcdcdcd"
TEXELS = "4080c0ff" * 4
ALL_RAILS = ("native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
             "native-metal-provider-objects")
# The rails v20 marks: every rail. The three trace rails execute the render
# pass with undefined pre-pass contents, and the two object rails express it
# through `RenderAttachmentLoad::DontCare`.
V20_REPORTING_RAILS = ALL_RAILS


def render_result(provider_backend=True, copy_in=2, copy_out=2):
    """The attachment observation a reporting rail of v20 has to emit: the
    fragment output over the whole 2x2 attachment, with the v13-v17 counters."""
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
    """The declaring case's result with the v20 compute counts (2 in, 1 out)."""
    report = synthetic_report(suite, digest, rail)
    if rail != "native-metal":
        for result in report["results"]:
            result["copy_in"], result["copy_out"] = 2, 1
    return report


class UndefinedLoadObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = V20_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v20_pins_the_dontcare_load_quad(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v20")
        self.assertEqual(self.suite["guard_byte"], 0)
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], [RENDER_ID])
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["entry"], "copy_word")
        self.assertEqual([declaring["grid"], declaring["local"]], [[1, 1, 1], [1, 1, 1]])
        for kind in ("air", "metal"):
            source = V20_PATH.parent / declaring[kind]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             declaring[kind]["sha256"], kind)
        buffers = {buffer["binding"]: buffer for buffer in declaring["buffers"]}
        self.assertEqual(buffers[0]["access"], "read")
        self.assertEqual((buffers[0]["allocation"], buffers[0]["view"],
                          buffers[0]["offset"], buffers[0]["length"],
                          buffers[0]["allocation_size"]), (900, 910, 0, 16, 16))
        self.assertEqual(buffers[0]["initial_hex"], DECLARED * 4)
        self.assertEqual(buffers[1]["access"], "write")
        self.assertEqual((buffers[1]["allocation"], buffers[1]["view"],
                          buffers[1]["offset"], buffers[1]["length"],
                          buffers[1]["allocation_size"]), (920, 930, 4, 4, 12))
        self.assertEqual(declaring["expected_writebacks"],
                         [{"allocation": 920, "view": 930, "offset": 4,
                           "bytes_hex": DECLARED}])
        case = self.suite["render_cases"][0]
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual((case["vertex_entry"], case["fragment_entry"]),
                         ("render_quad_vertex", "render_solid_rgba8"))
        module = V20_PATH.parent / case["metal"]["path"]
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                         case["metal"]["sha256"])
        self.assertEqual(case["vertices"], 6)
        self.assertEqual(case["viewport"], [0, 0, 2, 2])
        layout = case["vertex_layout"]["buffers"]
        self.assertEqual(layout,
                         [{"stride": 8,
                           "attributes": [{"location": 0, "offset": 0,
                                           "format": "float32x2"}]}])
        vertex = case["vertex_buffers"][0]
        self.assertEqual((vertex["allocation"], vertex["view"], vertex["offset"],
                          vertex["length"]), QUAD_VIEW)
        self.assertEqual(len(bytes.fromhex(vertex["initial_hex"])), QUAD_VIEW[3])
        indices = case["indices"]
        self.assertEqual((indices["allocation"], indices["view"], indices["offset"],
                          indices["length"], indices["format"]),
                         INDEX_VIEW + ("uint16",))
        self.assertEqual(bytes.fromhex(indices["initial_hex"]),
                         bytes([0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 2, 0]))
        attachment = case["attachment"]
        self.assertEqual((attachment["allocation"], attachment["view"],
                          attachment["format"], attachment["width"],
                          attachment["height"]),
                         (900, 910, "rgba8_unorm", 2, 2))
        self.assertEqual((attachment["load"], attachment["store"]),
                         ("dontcare", "store"))
        self.assertNotIn("clear_hex", attachment)
        self.assertNotIn("initial_hex", attachment)
        self.assertEqual(case["expected_hex"], TEXELS)
        self.assertEqual(sorted(case["capture_rails"]), sorted(V20_REPORTING_RAILS))

    def test_v20_plan_carries_the_attachment_but_not_the_undefined_contents(self):
        plan = compare._suite_plan(self.suite)
        render_plan = compare._render_plan(plan, self.suite)
        expectation = render_plan[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(TEXELS))])
        self.assertEqual(expectation.allocations,
                         {ATTACHMENT[0]: bytes.fromhex(TEXELS)})
        self.assertEqual(expectation.attachment, ATTACHMENT)
        # Two allocations are touched by the submission (the attachment view
        # and the probe view), and two are written (the probe's landing plus
        # the attachment's landing): the vertex/index streams belong to the
        # render pass and are not part of the declaring case's observation set.
        self.assertEqual(expectation.touched, {ATTACHMENT[0], PROBE[0]})
        self.assertEqual(expectation.written, {ATTACHMENT[0], PROBE[0]})
        self.assertEqual(expectation.rails, set(V20_REPORTING_RAILS))
        self.assertIsNone(expectation.present)
        self.assertIsNone(expectation.icb)

    def test_v20_declaring_pass_reads_the_attachment_view_only(self):
        declaring = self.suite["cases"][0]
        buffers = {buffer["binding"]: buffer for buffer in declaring["buffers"]}
        self.assertEqual((buffers[0]["allocation"], buffers[0]["view"],
                          buffers[0]["offset"], buffers[0]["length"],
                          buffers[0]["allocation_size"]), ATTACHMENT + (16,))
        self.assertEqual(buffers[0]["access"], "read")
        self.assertEqual(declaring["expected_writebacks"],
                         [{"allocation": PROBE[0], "view": PROBE[1],
                           "offset": PROBE[2], "bytes_hex": DECLARED}])

    def test_v20_marker_gates_each_rail(self):
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

    def test_v20_refuses_a_rail_whose_marker_does_not_name_it(self):
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

    def test_v20_refuses_a_clear_colour_on_a_dontcare_load(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["attachment"]["clear_hex"] = "fefefefe"
        with self.assertRaisesRegex(compare.CaptureError,
                                    "a dontcare load carries no clear colour"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v20_refuses_initial_bytes_on_a_dontcare_load(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["attachment"]["initial_hex"] = DECLARED * 4
        with self.assertRaisesRegex(compare.CaptureError,
                                    "a dontcare load carries no initial bytes"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v20_refuses_an_expectation_equal_to_the_declared_bytes(self):
        broken = copy.deepcopy(self.suite)
        broken["cases"][0]["buffers"][0]["initial_hex"] = TEXELS
        broken["cases"][0]["expected_writebacks"][0]["bytes_hex"] = "4080c0ff"
        broken["render_cases"][0]["expected_hex"] = TEXELS
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the declared view's bytes equal the expectation"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v20_refuses_a_nonuniform_expectation(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["expected_hex"] = \
            "4080c0ff" + "fefefefe" + "4080c0ff" + "4080c0ff"
        with self.assertRaisesRegex(compare.CaptureError,
                                    "every texel of a dontcare load has to be "
                                    "the fragment output"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v20_count_contract_is_two_in_and_two_out(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        report["results"].append(render_result(True, copy_in=2, copy_out=2))
        compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v20_refuses_wrong_copy_in(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        report["results"].append(render_result(True))
        report["results"][-1]["copy_in"] = 3
        with self.assertRaisesRegex(compare.CaptureError,
                                    "does not match 2 touched allocations"):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v20_refuses_wrong_copy_out(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        report["results"].append(render_result(True))
        report["results"][-1]["copy_out"] = 1
        with self.assertRaisesRegex(compare.CaptureError,
                                    "does not match 2 written allocations"):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")


if __name__ == "__main__":
    unittest.main()
