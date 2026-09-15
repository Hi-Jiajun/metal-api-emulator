"""Three-location checks for the observation channel of `suite-v24.json`.

These are comparator and schema checks, not GPU execution evidence. Three
attachments are not the ceiling (`MAX_COLOR_ATTACHMENTS` is four) but they are
their own reviewed shape: the four-location module cannot stand in, because a
fragment that writes a location with no attachment beside it is undefined. The
fixture reuses the four-read declaring kernel with three attachment views plus a
4-byte scratch read — a declared view the render pass does not attach has to
keep its guard bytes — and its three texels are pairwise distinct.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V24_PATH = CONFORMANCE / "suite-v24.json"

DECLARING_ID = "render_declaring_three_attachments"
RENDER_ID = "three_attachments_2x2"
ATTACHMENTS = ((900, 910, 0, 16), (920, 930, 0, 16), (940, 950, 0, 16))
SCRATCH_READ = (960, 970, 4)
SCRATCH_WRITE = (980, 990, 4)
QUAD_VIEW = (1000, 1010, 0, 32)
INDEX_VIEW = (1020, 1030, 0, 12)
TEXELS = ("4080c0ff" * 4, "ff8040c0" * 4, "c040ff80" * 4)
ALL_RAILS = ("native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
             "native-metal-provider-objects")
V24_REPORTING_RAILS = ALL_RAILS


def render_result(provider_backend=True, copy_in=5, copy_out=4):
    """The three-attachment observation a reporting rail of v24 has to emit."""
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
    """The declaring case's result with the v25 compute counts (5 in, 1 out)."""
    report = synthetic_report(suite, digest, rail)
    if rail != "native-metal":
        for result in report["results"]:
            result["copy_in"], result["copy_out"] = 5, 1
    return report


class ThreeLocationObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = V24_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v24_pins_the_three_location_fixture(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v24")
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], [RENDER_ID])
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["entry"], "mrt_declare4")
        for kind in ("air", "metal"):
            source = V24_PATH.parent / declaring[kind]["path"]
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
        # The fourth read is scratch: the render pass does not attach it, so the
        # fixture has to keep its guard bytes (offset 4, length 4).
        self.assertEqual((buffers[3]["access"], buffers[3]["allocation"],
                          buffers[3]["view"], buffers[3]["offset"],
                          buffers[3]["length"], buffers[3]["allocation_size"]),
                         ("read", SCRATCH_READ[0], SCRATCH_READ[1], SCRATCH_READ[2],
                          4, 16))
        self.assertEqual((buffers[4]["access"], buffers[4]["allocation"],
                          buffers[4]["view"], buffers[4]["offset"],
                          buffers[4]["length"], buffers[4]["allocation_size"]),
                         ("write", SCRATCH_WRITE[0], SCRATCH_WRITE[1],
                          SCRATCH_WRITE[2], 4, 16))
        case = self.suite["render_cases"][0]
        self.assertEqual((case["vertex_entry"], case["fragment_entry"]),
                         ("render_quad_vertex", "render_solid_rgba8_triple"))
        module = V24_PATH.parent / case["metal"]["path"]
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                         case["metal"]["sha256"])
        for attachment, (allocation, view, _, _), texels in zip(case["attachments"],
                                                                ATTACHMENTS, TEXELS):
            self.assertEqual((attachment["allocation"], attachment["view"],
                              attachment["format"], attachment["store"],
                              attachment["expected_hex"]),
                             (allocation, view, "rgba8_unorm", "store", texels))
        self.assertEqual(sorted(case["capture_rails"]), sorted(V24_REPORTING_RAILS))

    def test_v24_plan_observes_three_locations_and_counts_the_scratch_read(self):
        plan = compare._suite_plan(self.suite)
        expectation = compare._render_plan(plan, self.suite)[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((allocation, view, offset), bytes.fromhex(texels))
                          for (allocation, view, offset, _), texels
                          in zip(ATTACHMENTS, TEXELS)])
        # Five allocations are touched (three attachments, the scratch read and
        # the scratch write) but only four are written back (the three
        # attachments plus the declaring pass's own landing).
        self.assertEqual(expectation.touched,
                         {allocation for allocation, _, _, _ in ATTACHMENTS}
                         | {SCRATCH_READ[0], SCRATCH_WRITE[0]})
        self.assertEqual(expectation.written,
                         {allocation for allocation, _, _, _ in ATTACHMENTS}
                         | {SCRATCH_WRITE[0]})
        self.assertEqual(expectation.rails, set(V24_REPORTING_RAILS))

    def test_v24_marker_gates_each_rail(self):
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

    def test_v24_count_contract_is_five_in_and_four_out(self):
        report = counted_declaring(self.suite, self.digest, "vulkan")
        report["results"].append(render_result(True, copy_in=5, copy_out=4))
        compare.validate_capture(self.suite, self.digest, report, "vulkan")
        wrong = counted_declaring(self.suite, self.digest, "vulkan")
        wrong["results"].append(render_result(True, copy_in=5, copy_out=5))
        with self.assertRaisesRegex(compare.CaptureError,
                                    "does not match 4 written allocations"):
            compare.validate_capture(self.suite, self.digest, wrong, "vulkan")

    def test_v24_refuses_identical_expectations(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["attachments"][2]["expected_hex"] = TEXELS[0]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the attachments read back the same texels"):
            compare._render_plan(compare._suite_plan(broken), broken)


if __name__ == "__main__":
    unittest.main()
