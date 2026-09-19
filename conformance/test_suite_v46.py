"""The three-dimensional sampled volume: a texture3d argument whose third axis is real.

2026-09-20, the `D3` sampled texture arm. The census's LPF pipeline samples
three `texture3d<float, sample>` arguments beside its stage buffers, so the
canonical rail's sampled window had to grow a third spatial axis — and the
suite's half of that arm is a `4 x 4 x 2` volume whose two slices carry
different texels, sampled at four texel centres two of which stand in the
second slice. A rail that uploaded one slice, read `z` as an array layer or
ignored the source lands another frame; a rail that binds the declaration at
another extent is refused by name.

These tests are comparator and schema checks, not GPU execution evidence. What
they pin is the suite's half of the arm: the case's two AIR stages by digest,
its provenance — the pinned fragment module really declares a
`texture3d<float, sample>` argument sampled with a three-component coordinate —
the volume's own bytes (one run of slices, `width x height x depth` texels), the
frame its samples land, and the rails the arm can run on. The two native faces
compile a reviewed module selected by the colour format list's exact shape and
refuse a three-dimensional declaration by name, so a case that named them would
claim a capture they cannot report.
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
V46_PATH = CONFORMANCE / "suite-v46.json"
PROVIDER_PATH = REPOSITORY / "examples" / "metal-smoke" / "src" / "bin" / "provider-capture.rs"

DECLARING_ID = "render_declaring_gathered_extent"
CASE_ID = "volume_slice_pair_4x4x2"
DEPTH = 2
EXTENT = (4, 4)
VOLUME_TEXELS = 4 * 4 * 2
ATTACHMENT = (900, 910, 0, 64)
TEXELS = "40ffbfdf" * 16
SLICELESS_TEXELS = "4000bf00" * 16
VULKAN_RAILS = ("vulkan", "vulkan-objects")
NATIVE_RAILS = ("native-metal", "native-metal-provider", "native-metal-provider-objects")


def render_result(case_id):
    """The attachment observation a reporting rail of this suite has to emit."""
    return {
        "id": case_id,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": TEXELS}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": TEXELS}],
        "copy_in": 3,
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


class SampledVolumeTests(unittest.TestCase):
    def setUp(self):
        self.raw = V46_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def cases(self):
        return {case["id"]: case for case in self.suite["render_cases"]}

    def test_v46_declares_the_volume_on_the_rails_that_translate_its_stages(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v46")
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual(sorted(self.cases()), [CASE_ID])
        case = self.cases()[CASE_ID]
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual(case["capture_rails"], list(VULKAN_RAILS))
        self.assertEqual(case["vertices"], 3)
        self.assertEqual(case["viewport"], [0, 0, EXTENT[0], EXTENT[1]])
        self.assertIn("translated_stages", case)
        self.assertNotIn("metal", case)
        for absent in ("vertex_layout", "vertex_buffers", "indices", "stage_buffers",
                       "attachments"):
            self.assertNotIn(absent, case)
        self.assertEqual((case["vertex_entry"], case["fragment_entry"]),
                         ("render_fullscreen_triangle", "render_sample_texture_3d_volume"))
        texture = case["fragment_textures"][0]
        self.assertEqual((texture["format"], texture["width"], texture["height"]),
                         ("rgba8_unorm", EXTENT[0], EXTENT[1]))
        self.assertEqual(texture["depth"], DEPTH)
        self.assertEqual(len(bytes.fromhex(texture["initial_hex"])), VOLUME_TEXELS * 4)
        attachment = case["attachment"]
        self.assertEqual((attachment["allocation"], attachment["view"]),
                         (ATTACHMENT[0], ATTACHMENT[1]))
        self.assertEqual((attachment["format"], attachment["width"], attachment["height"]),
                         ("rgba8_unorm", EXTENT[0], EXTENT[1]))
        self.assertEqual((attachment["load"], attachment["store"]), ("clear", "store"))
        self.assertEqual(attachment["clear_hex"], "fefefefe")
        expected = bytes.fromhex(case["expected_hex"])
        self.assertEqual(len(expected), ATTACHMENT[3])
        self.assertEqual(set(expected[index:index + 4] for index in range(0, len(expected), 4)),
                         {bytes.fromhex("40ffbfdf")})
        self.assertNotEqual(expected[:4], bytes.fromhex(attachment["clear_hex"]))
        self.assertNotEqual(case["expected_hex"], SLICELESS_TEXELS,
                            "the frame reads both slices, not the first one twice")

    def test_v46_pins_the_module_that_samples_a_volume(self):
        case = self.cases()[CASE_ID]
        for kind, entry in (("vertex", case["vertex_entry"]),
                            ("fragment", case["fragment_entry"])):
            source = case["translated_stages"][kind]
            module = (V46_PATH.parent / source["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(module).hexdigest(), source["sha256"])
            self.assertIn(("@" + entry).encode(), module,
                          "the pinned module defines the entry the case names")
        fragment = (V46_PATH.parent
                    / case["translated_stages"]["fragment"]["path"]).read_text()
        self.assertIn('!"texture3d<float, sample>"', fragment)
        self.assertRegex(fragment, r"@air\.sample_texture_3d\.v4f32")
        self.assertEqual(len(re.findall(r"@air\.sample_texture_3d\.v4f32\(", fragment)), 5,
                         "four sample sites beside the intrinsic's own declaration")
        self.assertEqual(fragment.count("float 7.500000e-01>"), 2)
        provider = PROVIDER_PATH.read_text(encoding="utf-8")
        self.assertIn('(1, "compute-buffer-v46")', provider)
        self.assertIn("new_volume_texture_with_bytes", provider)

    def test_v46_plan_reaches_every_texel(self):
        plan = compare._suite_plan(self.suite)
        render_plan = compare._render_plan(plan, self.suite)
        expectation = render_plan[CASE_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(TEXELS))])
        self.assertEqual(expectation.allocations, {ATTACHMENT[0]: bytes.fromhex(TEXELS)})
        self.assertEqual(expectation.attachment, ATTACHMENT)
        self.assertEqual(expectation.rails, set(VULKAN_RAILS))
        self.assertEqual(expectation.texture_uploads, 1)

    def test_v46_every_rail_that_reports_the_attachment_validates(self):
        for backend in VULKAN_RAILS:
            with self.subTest(backend=backend):
                compare.validate_capture(self.suite, self.digest,
                                         synthetic_capture(self.suite, self.digest, backend),
                                         backend)

    def test_v46_refuses_a_frame_that_never_read_the_second_slice(self):
        for tampered in (SLICELESS_TEXELS, "40bfbfdf" * 16):
            with self.subTest(tampered=tampered):
                report = synthetic_capture(self.suite, self.digest)
                for result in report["results"]:
                    if result["id"] == CASE_ID:
                        result["writebacks"][0]["bytes_hex"] = tampered
                        result["allocations"][0]["bytes_hex"] = tampered
                with self.assertRaisesRegex(compare.CaptureError, "first differing byte"):
                    compare.validate_capture(self.suite, self.digest, report)

    def test_v46_refuses_a_native_rail(self):
        for rail in NATIVE_RAILS:
            with self.subTest(rail=rail):
                suite = copy.deepcopy(self.suite)
                for case in suite["render_cases"]:
                    case["capture_rails"] = list(VULKAN_RAILS) + [rail]
                with self.assertRaisesRegex(compare.CaptureError,
                                            "capture_rails has to stay inside that list"):
                    compare.validate_capture(suite, self.digest,
                                             synthetic_capture(self.suite, self.digest))

    def test_v46_refuses_a_volume_that_shares_the_render_area(self):
        suite = copy.deepcopy(self.suite)
        del suite["render_cases"][0]["fragment_textures"][0]["depth"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "differ from the render area in at least one axis"):
            compare.validate_capture(suite, self.digest,
                                     synthetic_capture(self.suite, self.digest))


if __name__ == "__main__":
    unittest.main()
