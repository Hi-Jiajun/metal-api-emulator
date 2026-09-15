"""Four-location checks for the observation channel of `suite-v23.json`.

These are comparator and schema checks, not GPU execution evidence. What they
pin is the ceiling `research/docs/23` §3.3 allows: a render pass may carry
`MAX_COLOR_ATTACHMENTS` colour attachments, the pipeline's `color_formats` list
agrees with them position by position, and the reviewed four-location module
writes four *pairwise distinct* texels — `40 80 c0 ff`, `ff 80 40 c0`,
`c0 40 ff 80`, `80 c0 40 ff`. A capture that landed one target twice, or
swapped two locations, reports the wrong bytes for its own allocation and is
refused by the identity check rather than by a byte comparison alone.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V23_PATH = CONFORMANCE / "suite-v23.json"

DECLARING_ID = "render_declaring_four_attachments"
RENDER_ID = "four_attachments_2x2"
ATTACHMENTS = ((900, 910, 0, 16), (920, 930, 0, 16), (940, 950, 0, 16),
               (960, 970, 0, 16))
SCRATCH = (980, 990, 4)
QUAD_VIEW = (1000, 1010, 0, 32)
INDEX_VIEW = (1020, 1030, 0, 12)
TEXELS = ("4080c0ff" * 4, "ff8040c0" * 4, "c040ff80" * 4, "80c040ff" * 4)
ALL_RAILS = ("native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
             "native-metal-provider-objects")
# Every rail executes the four-location shape now that the reviewed modules
# cover the ceiling (`docs/23` §3.3, v24).
V23_REPORTING_RAILS = ALL_RAILS


def render_result(provider_backend=True, copy_in=5, copy_out=5):
    """The four-attachment observation a reporting rail of v23 has to emit."""
    result = {
        "id": RENDER_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": allocation, "view": view, "offset": offset,
                        "bytes_hex": texels}
                       for (allocation, view, offset, _), texels
                       in zip(ATTACHMENTS, TEXELS)],
        "allocations": [{"allocation": allocation, "bytes_hex": texels}
                        for (allocation, _, _, _), texels
                        in zip(ATTACHMENTS, TEXELS)],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = copy_in, copy_out
    return result


def counted_declaring(suite, digest, rail):
    """The declaring case's result with the v24 compute counts (5 in, 1 out)."""
    report = synthetic_report(suite, digest, rail)
    if rail != "native-metal":
        for result in report["results"]:
            result["copy_in"], result["copy_out"] = 5, 1
    return report


class FourLocationObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = V23_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v23_pins_the_four_location_fixture(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v23")
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], [RENDER_ID])
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["entry"], "mrt_declare4")
        self.assertEqual([declaring["grid"], declaring["local"]], [[1, 1, 1], [1, 1, 1]])
        for kind in ("air", "metal"):
            source = V23_PATH.parent / declaring[kind]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             declaring[kind]["sha256"], kind)
        buffers = {buffer["binding"]: buffer for buffer in declaring["buffers"]}
        self.assertEqual(sorted(buffers), [0, 1, 2, 3, 4])
        for binding, (allocation, view, offset, length) in enumerate(ATTACHMENTS):
            buffer = buffers[binding]
            self.assertEqual((buffer["access"], buffer["allocation"], buffer["view"],
                              buffer["offset"], buffer["length"],
                              buffer["allocation_size"]),
                             ("read", allocation, view, offset, length, length))
        self.assertEqual((buffers[4]["access"], buffers[4]["allocation"],
                          buffers[4]["view"], buffers[4]["offset"],
                          buffers[4]["length"], buffers[4]["allocation_size"]),
                         ("write", SCRATCH[0], SCRATCH[1], SCRATCH[2], 4, 16))
        case = self.suite["render_cases"][0]
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual((case["vertex_entry"], case["fragment_entry"]),
                         ("render_quad_vertex", "render_solid_rgba8_quad"))
        module = V23_PATH.parent / case["metal"]["path"]
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                         case["metal"]["sha256"])
        declared = [attachment for attachment in case["attachments"]]
        self.assertEqual(len(declared), 4)
        for attachment, (allocation, view, _, _), texels in zip(declared, ATTACHMENTS,
                                                                TEXELS):
            self.assertEqual((attachment["allocation"], attachment["view"],
                              attachment["format"], attachment["load"],
                              attachment["store"], attachment["expected_hex"]),
                             (allocation, view, "rgba8_unorm", "clear", "store", texels))
        self.assertEqual(sorted(case["capture_rails"]), sorted(V23_REPORTING_RAILS))

    def test_v23_plan_observes_all_four_locations(self):
        plan = compare._suite_plan(self.suite)
        expectation = compare._render_plan(plan, self.suite)[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((allocation, view, offset), bytes.fromhex(texels))
                          for (allocation, view, offset, _), texels
                          in zip(ATTACHMENTS, TEXELS)])
        self.assertEqual(expectation.allocations,
                         {allocation: bytes.fromhex(texels)
                          for (allocation, _, _, _), texels in zip(ATTACHMENTS, TEXELS)})
        self.assertEqual(expectation.touched,
                         {allocation for allocation, _, _, _ in ATTACHMENTS} | {SCRATCH[0]})
        self.assertEqual(expectation.written, expectation.touched)
        self.assertEqual(expectation.rails, set(V23_REPORTING_RAILS))

    def test_v23_marker_gates_each_rail(self):
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

    def test_v23_refuses_a_capture_that_swaps_two_locations(self):
        # The byte strings are pairwise distinct, so a rail that landed one
        # target twice cannot satisfy the plan: the swapped capture reports the
        # wrong bytes for its own allocation.
        report = counted_declaring(self.suite, self.digest, "vulkan")
        swapped = render_result(True)
        swapped["writebacks"][0]["bytes_hex"], swapped["writebacks"][1]["bytes_hex"] = \
            swapped["writebacks"][1]["bytes_hex"], swapped["writebacks"][0]["bytes_hex"]
        swapped["allocations"][0]["bytes_hex"], swapped["allocations"][1]["bytes_hex"] = \
            swapped["allocations"][1]["bytes_hex"], swapped["allocations"][0]["bytes_hex"]
        report["results"].append(swapped)
        with self.assertRaises(compare.CaptureError):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_v23_refuses_identical_expectations(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["attachments"][1]["expected_hex"] = TEXELS[0]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the attachments read back the same texels"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_v23_refuses_three_and_five_attachments(self):
        for count, message in ((3, "three attachments have no reviewed module yet"),
                               (5, "two to four attachments")):
            broken = copy.deepcopy(self.suite)
            attachments = broken["render_cases"][0]["attachments"]
            while len(attachments) > count:
                attachments.pop()
            while len(attachments) < count:
                clone = copy.deepcopy(attachments[0])
                clone["allocation"] += 1000
                clone["view"] += 1000
                attachments.append(clone)
            with self.subTest(count=count):
                with self.assertRaisesRegex(compare.CaptureError, message):
                    compare._render_plan(compare._suite_plan(broken), broken)

    def test_v23_count_contract_is_five_in_and_five_out(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        report["results"].append(render_result(True, copy_in=5, copy_out=5))
        compare.validate_capture(self.suite, self.digest, report, "vulkan")


if __name__ == "__main__":
    unittest.main()
