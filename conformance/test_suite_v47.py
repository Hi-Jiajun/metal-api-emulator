"""The volume lane the device answers for: the census's own eight-bit order.

2026-09-20, census v48's volume lane gate. The `D3` arm's own increment shipped
with one lane — a `texture3d<float, sample>` over `R32_SFLOAT` — because that is
the lane the census's boot carried then. The LPF family's three volumes are
declared in the `B8G8R8A8_UNORM` lane instead, and whether a device can create
and fill a `TYPE_3D` image in a lane at all is a *device* answer rather than a
reading of the surface format list beside it (the `D3` arm's first increment
measured that: the RTX 5060 refuses a linear `R32_SFLOAT` volume Lavapipe
accepts). So the provider's frame grew a lane list, and this suite carries the
same `4 x 4 x 2` volume one lane over: `bgra8_unorm` texels whose *third* byte
is the component a view of that order hands the module's `.x` sample.

The case is falsifiable in the two ways the lane's own arithmetic can go wrong.
A rail that uploaded one slice leaves the two slice-1 samples reading the fill,
and a rail that named the *other* four-byte order (`rgba8_unorm` over the same
bytes) would hand the module the first byte of each texel instead of the third —
both land a frame this suite states and refuses to accept.

These tests are comparator and schema checks, not GPU execution evidence. What
they pin is the suite's half of the lane: the case's two AIR stages by digest,
its provenance — the pinned fragment module really declares a
`texture3d<float, sample>` argument sampled with a three-component coordinate —
the volume's own bytes (one run of slices, `width x height x depth` texels at
this lane's own order), the frame its samples land, and the rails the arm can
run on. The two native faces compile a reviewed module selected by the colour
format list's exact shape and refuse a three-dimensional declaration by name, so
a case that named them would claim a capture they cannot report.
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
V47_PATH = CONFORMANCE / "suite-v47.json"
PROVIDER_PATH = REPOSITORY / "examples" / "metal-smoke" / "src" / "bin" / "provider-capture.rs"

DECLARING_ID = "render_declaring_gathered_extent"
CASE_ID = "volume_slice_pair_bgra_4x4x2"
DEPTH = 2
EXTENT = (4, 4)
VOLUME_TEXELS = 4 * 4 * 2
ATTACHMENT = (900, 910, 0, 64)
TEXELS = "40bfffdf" * 16
# The two frames the case's own arithmetic may not land: one slice (the two
# slice-1 samples read the fill's red byte) and the other four-byte order (the
# module's `.x` comes from the texel's first byte instead of its third).
SLICELESS_TEXELS = "4000ff00" * 16
BYTE_ORDER_TEXELS = "1022182a" * 16
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


class SampledVolumeLaneTests(unittest.TestCase):
    def setUp(self):
        self.raw = V47_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def cases(self):
        return {case["id"]: case for case in self.suite["render_cases"]}

    def test_v47_declares_the_eight_bit_volume_on_the_rails_that_translate_its_stages(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v47")
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
                         ("bgra8_unorm", EXTENT[0], EXTENT[1]))
        self.assertEqual(texture["depth"], DEPTH)
        texels = bytes.fromhex(texture["initial_hex"])
        self.assertEqual(len(texels), VOLUME_TEXELS * 4)
        # The volume's texels are pairwise distinct, so a capture that landed
        # one texel twice cannot pass — the same falsifier the four-byte lane's
        # own case states one byte order over.
        chunks = [texels[offset:offset + 4] for offset in range(0, len(texels), 4)]
        self.assertEqual(len(set(chunks)), len(chunks))
        # The four texels the module samples carry the frame's four bytes in the
        # lane's own third component (`bgra8_unorm`'s red), and the frame is
        # *not* those texels' first component — which is what a rail that named
        # the other four-byte order would read.
        self.assertEqual([texels[index * 4 + 2] for index in (0, 18, 8, 26)],
                         [0x40, 0xbf, 0xff, 0xdf])
        self.assertEqual([texels[index * 4] for index in (0, 18, 8, 26)],
                         [0x10, 0x22, 0x18, 0x2a])
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
                         {bytes.fromhex("40bfffdf")})
        self.assertNotEqual(expected[:4], bytes.fromhex(attachment["clear_hex"]))
        self.assertNotEqual(case["expected_hex"], SLICELESS_TEXELS,
                            "the frame reads both slices, not the first one twice")
        self.assertNotEqual(case["expected_hex"], BYTE_ORDER_TEXELS,
                            "the frame reads the lane's own third byte, not its first")

    def test_v47_pins_the_module_that_samples_a_volume(self):
        case = self.cases()[CASE_ID]
        for kind, entry in (("vertex", case["vertex_entry"]),
                            ("fragment", case["fragment_entry"])):
            source = case["translated_stages"][kind]
            module = (V47_PATH.parent / source["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(module).hexdigest(), source["sha256"])
            self.assertIn(("@" + entry).encode(), module,
                          "the pinned module defines the entry the case names")
        fragment = (V47_PATH.parent
                    / case["translated_stages"]["fragment"]["path"]).read_text()
        self.assertIn('!"texture3d<float, sample>"', fragment)
        self.assertRegex(fragment, r"@air\.sample_texture_3d\.v4f32")
        self.assertEqual(len(re.findall(r"@air\.sample_texture_3d\.v4f32\(", fragment)), 5,
                         "four sample sites beside the intrinsic's own declaration")
        self.assertEqual(fragment.count("float 7.500000e-01>"), 2)
        provider = PROVIDER_PATH.read_text(encoding="utf-8")
        self.assertIn('(1, "compute-buffer-v47")', provider)
        self.assertIn("new_volume_texture_with_bytes", provider)

    def test_v47_plan_reaches_every_texel(self):
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

    def test_v47_every_rail_that_reports_the_attachment_validates(self):
        for backend in VULKAN_RAILS:
            with self.subTest(backend=backend):
                compare.validate_capture(self.suite, self.digest,
                                         synthetic_capture(self.suite, self.digest, backend),
                                         backend)

    def test_v47_refuses_a_frame_that_never_read_the_second_slice(self):
        for tampered in (SLICELESS_TEXELS, BYTE_ORDER_TEXELS):
            with self.subTest(tampered=tampered):
                report = synthetic_capture(self.suite, self.digest)
                for result in report["results"]:
                    if result["id"] == CASE_ID:
                        result["writebacks"][0]["bytes_hex"] = tampered
                        result["allocations"][0]["bytes_hex"] = tampered
                with self.assertRaisesRegex(compare.CaptureError, "first differing byte"):
                    compare.validate_capture(self.suite, self.digest, report)

    def test_v47_refuses_a_native_rail(self):
        for rail in NATIVE_RAILS:
            with self.subTest(rail=rail):
                suite = copy.deepcopy(self.suite)
                for case in suite["render_cases"]:
                    case["capture_rails"] = list(VULKAN_RAILS) + [rail]
                with self.assertRaisesRegex(compare.CaptureError,
                                            "capture_rails has to stay inside that list"):
                    compare.validate_capture(suite, self.digest,
                                             synthetic_capture(self.suite, self.digest))

    def test_v47_refuses_a_volume_that_shares_the_render_area(self):
        suite = copy.deepcopy(self.suite)
        del suite["render_cases"][0]["fragment_textures"][0]["depth"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "differ from the render area in at least one axis"):
            compare.validate_capture(suite, self.digest,
                                     synthetic_capture(self.suite, self.digest))


if __name__ == "__main__":
    unittest.main()
