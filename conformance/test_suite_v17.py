"""Load-section checks for the observation channel of `suite-v17.json`.

These are comparator and schema checks, not GPU execution evidence. What they
pin is the observation surface `research/docs/23` §3.3 adds on top of v16's
vertex input: a render case may ask the pass to *load* the attachment's previous
bytes instead of clearing them, and the expectation then has to keep the loaded
bytes wherever the draw missed while still showing the fragment output wherever
it covered. Both halves have to appear, or neither half of the pass is
falsifiable.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V17_PATH = CONFORMANCE / "suite-v17.json"

RENDER_ID = "load_partial_quad_2x2"
ATTACHMENT = (900, 910, 0, 16)
QUAD_VIEW = (940, 950, 0, 32)
INDEX_VIEW = (960, 970, 0, 12)
PREVIOUS = "fefefefe"
TEXEL = "4080c0ff"
EXPECTED = TEXEL + PREVIOUS * 3
ALL_RAILS = ("native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
             "native-metal-provider-objects")
# The rails v17 marks: the Vulkan trace rail executes the upload path today. The
# native rails follow once `MTLLoadAction.Load` lands, and the object rails once
# their binding surface can express "load" rather than "clear".
V17_REPORTING_RAILS = ("vulkan",)


def render_result(provider_backend=True):
    """The attachment observation a reporting rail of v17 has to emit."""
    result = {
        "id": RENDER_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": EXPECTED}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": EXPECTED}],
    }
    if provider_backend:
        # Two uploads (the declaring case's attachment and probe views) and two
        # readbacks (the probe's landing and the attachment's landing), exactly
        # as the v13-v16 render results report.
        result["copy_in"], result["copy_out"] = 2, 2
    return result


class LoadObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = V17_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v17_pins_the_loading_pass_and_its_partial_coverage(self):
        case = self.suite["render_cases"][0]
        self.assertEqual(case["id"], RENDER_ID)
        self.assertEqual(sorted(case["capture_rails"]), sorted(V17_REPORTING_RAILS))
        attachment = case["attachment"]
        self.assertEqual(attachment["load"], "load")
        self.assertEqual(attachment["store"], "store")
        self.assertNotIn("clear_hex", attachment)
        self.assertEqual(attachment["initial_hex"], PREVIOUS * 4)
        expected = case["expected_hex"]
        self.assertEqual(expected, EXPECTED)
        # The stream is the reviewed layout moved into the top-left quadrant, so
        # exactly one of the four texels is covered.
        self.assertEqual(case["vertex_buffers"][0]["length"], 32)
        self.assertEqual(case["vertex_buffers"][0]["initial_hex"][:16], "000080bf000080bf")
        self.assertEqual(case["indices"]["format"], "uint16")

    def test_v17_plan_keeps_both_halves_of_the_observation(self):
        plan = compare._suite_plan(self.suite)
        render_plan = compare._render_plan(plan, self.suite)
        expectation = render_plan[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(EXPECTED))])
        self.assertEqual(expectation.allocations, {ATTACHMENT[0]: bytes.fromhex(EXPECTED)})
        self.assertEqual(expectation.rails, set(V17_REPORTING_RAILS))

    def test_v17_refuses_an_expectation_that_loses_either_half(self):
        for mutate, message in (
            # Everything drawn: the load becomes unobservable.
            (lambda case: case.__setitem__("expected_hex", TEXEL * 4),
             "needs both drawn and kept texels"),
            # Everything loaded: the draw becomes unobservable.
            (lambda case: case.__setitem__("expected_hex", PREVIOUS * 4),
             "initial texels equal"),
            # A texel that is neither the fragment output nor the loaded byte.
            (lambda case: case.__setitem__(
                "expected_hex", TEXEL + PREVIOUS + "00ff00ff" + PREVIOUS),
             "drawn texels disagree"),
            # Two drawn texels that disagree about the fragment output.
            (lambda case: case.__setitem__(
                "expected_hex", TEXEL + "01020304" + PREVIOUS * 2),
             "drawn texels disagree"),
        ):
            broken = copy.deepcopy(self.suite)
            mutate(broken["render_cases"][0])
            with self.subTest(message=message):
                with self.assertRaises(compare.CaptureError) as caught:
                    compare._render_plan(compare._suite_plan(broken), broken)
                self.assertIn(message, str(caught.exception))

    def test_v17_refuses_a_declaration_that_disagrees_with_the_expectation(self):
        broken = copy.deepcopy(self.suite)
        # The loading pass uploads the declaring case's bytes, so a case whose
        # declaration says something else describes bytes the rail never held.
        broken["cases"][0]["buffers"][0]["initial_hex"] = "01020304" * 4
        with self.assertRaises(compare.CaptureError) as caught:
            compare._render_plan(compare._suite_plan(broken), broken)
        self.assertIn("declared view's bytes", str(caught.exception))

    def test_v17_marker_gates_each_rail(self):
        for rail in ALL_RAILS:
            suite = copy.deepcopy(self.suite)
            for case in suite["render_cases"]:
                case["capture_rails"] = [rail]
            digest = hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
            report = synthetic_report(suite, digest, rail)
            if rail != "native-metal":
                for result in report["results"]:
                    result["copy_in"], result["copy_out"] = 2, 1
            report["results"].append(render_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                compare.validate_capture(suite, digest, report, rail)

    def test_v17_refuses_a_capture_that_reports_a_clearing_result(self):
        # A clearing pass on the same geometry leaves the clear colour where the
        # draw missed; the marker names only the rails that load, so a capture
        # whose bytes are cleared has to be refused rather than compared.
        suite = self.suite
        report = synthetic_report(suite, self.digest, "vulkan")
        for result in report["results"]:
            result["copy_in"], result["copy_out"] = 2, 1
        cleared = render_result(True)
        cleared["writebacks"][0]["bytes_hex"] = TEXEL + PREVIOUS + PREVIOUS + "00000000"
        cleared["allocations"][0]["bytes_hex"] = cleared["writebacks"][0]["bytes_hex"]
        report["results"].append(cleared)
        with self.assertRaises(compare.CaptureError):
            compare.validate_capture(suite, self.digest, report, "vulkan")


if __name__ == "__main__":
    unittest.main()
