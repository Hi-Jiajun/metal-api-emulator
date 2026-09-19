"""The superset fragment interface: a module that stores more than is attached.

2026-09-20, the third door behind census v46's `stage_buffer_footprint` bucket.
The population is one LPF pipeline whose fragment stage stores three colour
locations while the draw attaches one. Vulkan defines what the rail does with
the extra stores — a fragment output whose location has no attachment behind it
is discarded — so the shape is the attached location's frame, and the extra
declaration is what the provider's registration gate had to learn to admit.

These tests are comparator and schema checks, not GPU execution evidence. What
they pin is the suite's half of the arm: the case's two AIR stages by digest,
its provenance — the pinned fragment module really declares two
`air.render_target` locations, one of which no attachment covers — the frame
the attached location's own texel produces, and the rails the arm can run on.
The two native faces select a reviewed module by the colour format list's exact
shape, so a module that stores a location the pass does not attach matches no
arm there, and a case that named them would claim a capture they cannot report.
"""

import copy
import hashlib
import json
import re
import unittest
from pathlib import Path

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
REPOSITORY = CONFORMANCE.parent
V45_PATH = CONFORMANCE / "suite-v45.json"
PROVIDER_PATH = REPOSITORY / "examples" / "metal-smoke" / "src" / "bin" / "provider-capture.rs"

DECLARING_ID = "render_declaring_copy_word"
CASE_ID = "declared_superset_fragment_two_outputs_2x2"
ATTACHMENT = (900, 910, 0, 16)
# The attached location's texel — `(64/255, 128/255, 192/255, 1)` — in every
# one of the 2x2 attachment's pixels: the fixture's triangle covers every pixel
# centre.
TEXELS = "4080c0ff" * 4
# The texel the fixture's *second*, unattached location stores. A rail that
# bound that store to the attached location would land this instead, which is
# the frame the suite's expectation refuses.
DROPPED_TEXELS = "ff0000ff" * 4
# The rails whose translator registers the case's own two AIR stages.
VULKAN_RAILS = ("vulkan", "vulkan-objects")
# The two native faces, whose reviewed-module table has no arm for the shape.
NATIVE_RAILS = ("native-metal", "native-metal-provider", "native-metal-provider-objects")


def render_result(case_id):
    """The attachment observation a reporting rail of this suite has to emit."""
    return {
        "id": case_id,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": TEXELS}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": TEXELS}],
        "copy_in": 2,
        "copy_out": 2,
    }


def synthetic_capture(suite, digest, backend="vulkan"):
    """A capture with the case's own attachment observation attached."""
    report = synthetic_report(suite, digest, backend)
    for result in report["results"]:
        if result["id"] == DECLARING_ID:
            result["copy_in"], result["copy_out"] = 2, 1
    report["results"].append(render_result(CASE_ID))
    return report


class SupersetFragmentOutputTests(unittest.TestCase):
    def setUp(self):
        self.raw = V45_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def cases(self):
        return {case["id"]: case for case in self.suite["render_cases"]}

    def test_v45_declares_the_arm_on_the_rails_that_translate_its_stages(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v45")
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual(sorted(self.cases()), [CASE_ID])
        case = self.cases()[CASE_ID]
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual(case["capture_rails"], list(VULKAN_RAILS))
        self.assertEqual(case["vertices"], 3)
        self.assertEqual(case["viewport"], [0, 0, 2, 2])
        # The arm is a *translated* shape: the case pins its own two AIR
        # modules and no MSL sibling, and it states no vertex layout, no
        # stream, no index buffer, no slot and no sampled declaration — the
        # draw is the `vertex_id` triangle's own.
        self.assertIn("translated_stages", case)
        self.assertNotIn("metal", case)
        for absent in ("vertex_layout", "vertex_buffers", "indices", "stage_buffers",
                       "fragment_textures", "attachments"):
            self.assertNotIn(absent, case)
        self.assertEqual((case["vertex_entry"], case["fragment_entry"]),
                         ("render_fullscreen_triangle", "render_two_output_rgba8"))
        attachment = case["attachment"]
        self.assertEqual((attachment["allocation"], attachment["view"]),
                         (ATTACHMENT[0], ATTACHMENT[1]))
        self.assertEqual((attachment["format"], attachment["width"], attachment["height"]),
                         ("rgba8_unorm", 2, 2))
        self.assertEqual((attachment["load"], attachment["store"]), ("clear", "store"))
        self.assertEqual(attachment["clear_hex"], "fefefefe")
        expected = bytes.fromhex(case["expected_hex"])
        self.assertEqual(len(expected), 16)
        self.assertEqual(set(expected[index:index + 4] for index in range(0, 16, 4)),
                         {bytes.fromhex("4080c0ff")})
        self.assertNotEqual(expected[:4], bytes.fromhex(attachment["clear_hex"]))
        self.assertNotEqual(case["expected_hex"], DROPPED_TEXELS,
                            "the frame is the attached location's texel, not the dropped one")

    def test_v45_pins_the_module_that_stores_one_location_more_than_it_attaches(self):
        case = self.cases()[CASE_ID]
        for kind, entry in (("vertex", case["vertex_entry"]),
                            ("fragment", case["fragment_entry"])):
            source = case["translated_stages"][kind]
            module = (V45_PATH.parent / source["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(module).hexdigest(), source["sha256"])
            self.assertIn(("@" + entry).encode(), module,
                          "the pinned module defines the entry the case names")
        fragment = (V45_PATH.parent
                    / case["translated_stages"]["fragment"]["path"]).read_text()
        # The pinned module is the arm's own shape: two `air.render_target`
        # metadata entries, at locations 0 and 1, and the case attaches one.
        self.assertEqual(re.findall(r'!"air\.render_target", i32 (\d+)', fragment),
                         ["0", "1"],
                         "the fixture's fragment stage declares exactly two colour locations")
        self.assertEqual(len(re.findall(r'!"air\.render_target"', fragment)), 2)
        # The module *stores* both: a fixture whose second location were only
        # declared would not state the shape the census's LPF stage has.
        self.assertRegex(fragment, r"define <\{ <4 x float>, <4 x float> \}>")
        # The capture binary knows the suite and pins its declaring pass.
        provider = PROVIDER_PATH.read_text(encoding="utf-8")
        self.assertIn('(1, "compute-buffer-v45")', provider)
        self.assertIn("SupersetFragmentOutput", provider)

    def test_v45_plan_reaches_every_texel(self):
        plan = compare._suite_plan(self.suite)
        render_plan = compare._render_plan(plan, self.suite)
        expectation = render_plan[CASE_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(TEXELS))])
        self.assertEqual(expectation.allocations, {ATTACHMENT[0]: bytes.fromhex(TEXELS)})
        self.assertEqual(expectation.attachment, ATTACHMENT)
        self.assertEqual(expectation.rails, set(VULKAN_RAILS))

    def test_v45_every_rail_that_reports_the_attachment_validates(self):
        for backend in VULKAN_RAILS:
            with self.subTest(backend=backend):
                compare.validate_capture(self.suite, self.digest,
                                         synthetic_capture(self.suite, self.digest, backend),
                                         backend)

    def test_v45_refuses_the_dropped_locations_frame(self):
        # A rail that bound the second store to the attached location — or that
        # summed the two — lands the *dropped* texel everywhere. The suite's
        # expectation is the attached location's texel, so that frame cannot
        # pass.
        for tampered in (DROPPED_TEXELS, "fefefefe" + TEXELS[8:]):
            with self.subTest(tampered=tampered):
                report = synthetic_capture(self.suite, self.digest)
                for result in report["results"]:
                    if result["id"] == CASE_ID:
                        result["writebacks"][0]["bytes_hex"] = tampered
                        result["allocations"][0]["bytes_hex"] = tampered
                with self.assertRaisesRegex(compare.CaptureError, "first differing byte"):
                    compare.validate_capture(self.suite, self.digest, report)

    def test_v45_refuses_a_native_rail(self):
        # The two native faces select a reviewed module by the colour format
        # list's exact shape, so a module that stores a location the pass does
        # not attach matches no arm and is refused by name: a case that named
        # them would claim a capture they cannot report.
        for rail in NATIVE_RAILS:
            with self.subTest(rail=rail):
                suite = copy.deepcopy(self.suite)
                for case in suite["render_cases"]:
                    case["capture_rails"] = list(VULKAN_RAILS) + [rail]
                with self.assertRaisesRegex(compare.CaptureError,
                                            "capture_rails has to stay inside that list"):
                    compare.validate_capture(suite, self.digest,
                                             synthetic_capture(suite, self.digest))

    def test_v45_refuses_two_attached_locations(self):
        # The arm's whole statement is that the *pass* attaches fewer locations
        # than the module stores: a case that declared two attachments would be
        # the reviewed MRT shape's arithmetic, not this one.
        suite = copy.deepcopy(self.suite)
        case = suite["render_cases"][0]
        case["attachments"] = [dict(case["attachment"]), dict(case["attachment"])]
        del case["attachment"]
        del case["expected_hex"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the milestone vertex_id shape renders one attachment"):
            compare.validate_capture(suite, self.digest, synthetic_capture(self.suite, self.digest))


if __name__ == "__main__":
    unittest.main()
