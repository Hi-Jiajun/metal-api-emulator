"""Vertex-input section checks for the observation channel of `suite-v16.json`.

These are comparator and schema checks, not GPU execution evidence. What they
pin is the observation surface `research/docs/23` §3.3 adds on top of v14's
attachment: a render case may declare a vertex layout, the streams that layout
reads and an index buffer, every rail its `capture_rails` marker names has to
report that case, and a rail the marker does not name must omit it.

The committed fixture is `suite-v16.json`; the execution evidence lives in
`crates/metal-api-vulkan/tests/render_e2e.rs` (Lavapipe) and in the RTX 5060
capture archived under `evidence/`.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


def render_result(provider_backend=True):
    """The attachment observation a reporting rail of v16 has to emit.

    Identical in shape to v13's: one writeback covering the attachment's own
    range and one allocation image whose texels are the fragment output. The
    streams are inputs, so they are not part of the observation.
    """
    result = {
        "id": RENDER_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": TEXELS}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": TEXELS}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = 2, 2
    return result


CONFORMANCE = Path(__file__).resolve().parent
V16_PATH = CONFORMANCE / "suite-v16.json"
PROVIDER_PATH = CONFORMANCE.parent / "examples/metal-smoke/src/bin/provider-capture.rs"
SPV_DIRECTORY = CONFORMANCE.parent / "crates/metal-api-vulkan/src/render_spv"

RENDER_ID = "quad_indexed_clear_2x2"
TEXELS = "4080c0ff" * 4
ATTACHMENT = (900, 910, 0, 16)
QUAD_VIEW = (940, 950, 0, 32)
INDEX_VIEW = (960, 970, 0, 12)
ALL_RAILS = ("native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
             "native-metal-provider-objects")
# Every rail now marks the case: the Vulkan trace and object rails execute the
# reviewed quad on Lavapipe and the RTX 5060, and the native rails observed the
# same four texels through `MTLVertexDescriptor` + `drawIndexedPrimitives` on an
# Apple Paravirtual device (CI run `34866107438`,
# `vertex_selftest: PASS (vertex_quad_indexed_2x2 4080c0ff...)`).
V16_REPORTING_RAILS = ALL_RAILS


def suite_with_rail(suite, rail):
    """A copy of the suite whose render-case marker names one rail."""
    trimmed = copy.deepcopy(suite)
    for case in trimmed["render_cases"]:
        case["capture_rails"] = [rail] if rail in ALL_RAILS else []
    digest = hashlib.sha256(json.dumps(trimmed, sort_keys=True).encode("utf-8")).hexdigest()
    return trimmed, digest


class VertexInputObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = V16_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v16_pins_the_reviewed_quad_and_its_streams(self):
        case = self.suite["render_cases"][0]
        self.assertEqual(case["id"], RENDER_ID)
        self.assertEqual(sorted(case["capture_rails"]), sorted(V16_REPORTING_RAILS))
        layout = case["vertex_layout"]["buffers"]
        self.assertEqual(len(layout), 1)
        self.assertEqual(layout[0]["stride"], 8)
        self.assertEqual(layout[0]["attributes"],
                         [{"location": 0, "offset": 0, "format": "float32x2"}])
        self.assertEqual(case["vertices"], 6)
        buffers = case["vertex_buffers"]
        self.assertEqual(len(buffers), 1)
        self.assertEqual((buffers[0]["allocation"], buffers[0]["view"], buffers[0]["offset"],
                          buffers[0]["length"]), QUAD_VIEW)
        self.assertEqual(len(bytes.fromhex(buffers[0]["initial_hex"])), QUAD_VIEW[3])
        indices = case["indices"]
        self.assertEqual((indices["allocation"], indices["view"], indices["offset"],
                          indices["length"], indices["format"]),
                         INDEX_VIEW + ("uint16",))
        self.assertEqual(bytes.fromhex(indices["initial_hex"]),
                         bytes([0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 2, 0]))
        # The reviewed MSL module is pinned by bytes, exactly as v13 pins its
        # own module: a renamed stage cannot slip past the review.
        module = V16_PATH.parent / case["metal"]["path"]
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                         case["metal"]["sha256"])
        self.assertIn(b"render_quad_vertex", module.read_bytes())
        self.assertIn(b"[[stage_in]]", module.read_bytes())

    def test_v16_plan_carries_the_streams_beside_the_attachment(self):
        plan = compare._suite_plan(self.suite)
        render_plan = compare._render_plan(plan, self.suite)
        expectation = render_plan[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(TEXELS))])
        self.assertEqual(expectation.allocations, {ATTACHMENT[0]: bytes.fromhex(TEXELS)})
        self.assertEqual(expectation.attachment, ATTACHMENT)
        # The declaring pass uploads the attachment and its probe view; the
        # streams belong to the render pass, so they are not part of the
        # declaring case's own observation set.
        self.assertEqual(expectation.touched, {ATTACHMENT[0], 920})
        self.assertEqual(expectation.written, {ATTACHMENT[0], 920})
        self.assertEqual(expectation.rails, set(V16_REPORTING_RAILS))

    def test_v16_prefers_the_reviewed_shape_over_a_stream_without_a_layout(self):
        # A stream without a layout describes nothing, and a layout without its
        # binding cannot be executed: both are refused before any rail runs.
        for mutation, message in (
            ({"vertex_layout": None}, "without a vertex layout"),
            ({"vertex_buffers": []}, "binds one vertex stream"),
            ({"indices": None}, "is indexed"),
        ):
            broken = copy.deepcopy(self.suite)
            case = broken["render_cases"][0]
            for key, value in mutation.items():
                if value is None:
                    case.pop(key, None)
                else:
                    case[key] = value
            with self.subTest(mutation=mutation):
                with self.assertRaises(compare.CaptureError) as caught:
                    compare._render_plan(compare._suite_plan(broken), broken)
                self.assertIn(message, str(caught.exception))

    def test_v16_refuses_an_index_outside_the_reviewed_quad(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["indices"]["initial_hex"] = "040001000200010003000200"
        with self.assertRaises(compare.CaptureError) as caught:
            compare._render_plan(compare._suite_plan(broken), broken)
        self.assertIn("outside the quad", str(caught.exception))

    def test_v16_refuses_a_short_vertex_stream(self):
        broken = copy.deepcopy(self.suite)
        case = broken["render_cases"][0]
        case["vertex_buffers"][0]["length"] = 16
        case["vertex_buffers"][0]["initial_hex"] = "00" * 16
        with self.assertRaises(compare.CaptureError) as caught:
            compare._render_plan(compare._suite_plan(broken), broken)
        self.assertIn("shorter than the reviewed quad", str(caught.exception))

    def test_v16_marker_gates_each_rail(self):
        # A rail the marker names has to report the case; a rail it does not
        # name must omit it. The synthetic capture carries the compute case for
        # every rail and the attachment observation only where the marker asks
        # for it.
        for rail in ALL_RAILS:
            suite, digest = suite_with_rail(self.suite, rail)
            report = synthetic_report(suite, digest, rail)
            if rail != "native-metal":
                for result in report["results"]:
                    result["copy_in"], result["copy_out"] = 2, 1
            report["results"].append(render_result(rail != "native-metal"))
            with self.subTest(rail=rail):
                compare.validate_capture(suite, digest, report, rail)

    def test_v16_refuses_a_capture_whose_rail_was_not_named(self):
        suite, digest = suite_with_rail(self.suite, "vulkan")
        report = synthetic_report(suite, digest, "vulkan-objects")
        report["results"].append(render_result(True))
        with self.assertRaises(compare.CaptureError) as caught:
            compare.validate_capture(suite, digest, report, "vulkan-objects")
        self.assertIn("is not a rail this render case runs on", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
