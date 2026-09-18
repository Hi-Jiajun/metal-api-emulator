"""The declared-superset vertex interface's own suite (`research/docs/23` §3.3, E-TX11).

`suite-v37.json` carries one translated render case against one declaring pass:
the case's contract declares four `float32x2` attribute locations on one stream
while its translated vertex module reads only the first two, so the extra
declared attributes are bound with the stream and ignored. The frame is
falsifiable in both directions — the two attributes the module never reads carry
values far outside the clip space, so a rail that read them (or bound the wrong
attribute to an input) lands a different frame, and the attribute the module
does read decides where the triangle covers the 2x2 attachment.

These are comparator and schema checks, not GPU execution evidence: the Lavapipe
readings live in
`crates/metal-api-vulkan/tests/render_vertex_superset_e2e.rs` (including the
reading where the ignored bytes leave the frame alone and the read stream moves
it) and in the capture archives the evidence directory keeps.
"""

import copy
import hashlib
import json
import struct
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V37_PATH = CONFORMANCE / "suite-v37.json"

DECLARING_ID = "render_declaring_vertex_superset"
CASE_ID = "declared_superset_4_attributes_2_read"

ATTACHMENT = (900, 910)
STREAM = (960, 970)
INDICES = (980, 990)
CLEAR = "fefefefe"

# The fragment stage's own texel (`render_offscreen_2x2.frag.ll`) and the frame
# the fixture's triangle lands: the module's offset moves the triangle so that
# three of the four texels are covered.
TEXEL = "4080c0ff"
FRAME_HEX = TEXEL + CLEAR + TEXEL + TEXEL
# What a rail that ignored the vertex stream altogether would land: the
# milestone's full-screen triangle covers every texel.
FULL_COVERAGE_HEX = TEXEL * 4

# The two AIR fixtures the case pins, by their own digests.
VERTEX_SHA = "85e63eafd8ce43bef286ad102a753e39e7cc36798dd1ab07466714b2f3d19a3c"
FRAGMENT_SHA = "26788da076aa8958c5fcf6363f43699da41ce695ac7f64b50bcb0944a7c67acb"

VULKAN_RAILS = ("vulkan", "vulkan-objects")

STRIDE = 32
RECORDS = 3


class DeclaredSupersetTests(unittest.TestCase):
    def setUp(self):
        self.raw = V37_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def case(self, case_id, suite=None):
        suite = suite or self.suite
        for case in suite["render_cases"]:
            if case["id"] == case_id:
                return case
        self.fail(f"the suite carries no {case_id} case")

    def plan(self, case_id, suite=None):
        suite = suite or self.suite
        return compare._render_plan(compare._suite_plan(suite), suite)[case_id]

    def refused(self, mutate, message):
        suite = copy.deepcopy(self.suite)
        mutate(self.case(CASE_ID, suite))
        with self.assertRaisesRegex(compare.CaptureError, message):
            compare._render_plan(compare._suite_plan(suite), suite)

    def test_the_suite_carries_the_declaring_pass_and_the_superset_case(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v37")
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], [CASE_ID])
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["entry"], "copy_word")
        for kind in ("air", "metal"):
            source = CONFORMANCE / declaring[kind]["path"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                declaring[kind]["sha256"],
                kind,
            )
        buffers = {buffer["binding"]: buffer for buffer in declaring["buffers"]}
        self.assertEqual(
            (buffers[0]["access"], buffers[0]["allocation"], buffers[0]["view"]),
            ("read",) + ATTACHMENT,
        )
        self.assertEqual(buffers[0]["length"], 2 * 2 * 4, "the render area's own view")
        self.assertEqual(
            (buffers[1]["access"], buffers[1]["allocation"], buffers[1]["view"]),
            ("write", 920, 930),
        )

    def test_the_case_pins_the_translated_pair_and_names_the_vulkan_rails(self):
        case = self.case(CASE_ID)
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual(
            (case["vertex_entry"], case["fragment_entry"]),
            ("reims_two_stream_vertex", "render_solid_rgba8"),
        )
        self.assertNotIn("metal", case)
        translated = case["translated_stages"]
        for kind, sha, suffix in (
            ("vertex", VERTEX_SHA, "_two_stream.ll"),
            ("fragment", FRAGMENT_SHA, "_2x2.frag.ll"),
        ):
            source = CONFORMANCE / translated[kind]["path"]
            self.assertTrue(
                source.name.endswith(suffix),
                f"the {kind} stage pins its own AIR module ({source.name})",
            )
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                sha,
                kind,
            )
            self.assertEqual(translated[kind]["sha256"], sha)
        self.assertEqual(sorted(case["capture_rails"]), sorted(VULKAN_RAILS))

    def test_the_layout_declares_four_attributes_and_the_module_reads_two(self):
        # The increment's own shape: the contract's layout names four locations,
        # the translated module reads two of them, and the declared set is a
        # strict superset of the read one.
        case = self.case(CASE_ID)
        streams = case["vertex_layout"]["buffers"]
        self.assertEqual(len(streams), 1)
        stream = streams[0]
        self.assertEqual((stream["stride"], stream["step"]), (STRIDE, "per_vertex"))
        attributes = stream["attributes"]
        self.assertEqual(len(attributes), 4)
        for location, attribute in enumerate(attributes):
            self.assertEqual(
                (attribute["location"], attribute["offset"], attribute["format"]),
                (location, location * 8, "float32x2"),
                f"attribute {location}",
            )
        # The module's own interface is the fixture's (`reims_indexed_tri_two_stream`):
        # a `float2` position at location 0 and a `float2` offset at location 1.
        installed = CONFORMANCE / case["translated_stages"]["vertex"]["path"]
        self.assertIn(b"air.location_index\", i32 0", installed.read_bytes())
        self.assertIn(b"air.location_index\", i32 1", installed.read_bytes())

    def test_the_ignored_attributes_stay_outside_the_read_streams_clip_space(self):
        # What makes the case falsifiable: the bytes behind the two declared
        # attributes the module never reads are far outside the clip space the
        # read attributes use, so a rail that read them as an input lands a
        # different frame.
        case = self.case(CASE_ID)
        binding = case["vertex_buffers"][0]
        self.assertEqual(binding["length"], STRIDE * RECORDS)
        bytes_ = bytes.fromhex(binding["initial_hex"])
        self.assertEqual(len(bytes_), STRIDE * RECORDS)
        read = [struct.unpack("<2f", bytes_[record * STRIDE:record * STRIDE + 8])
                for record in range(RECORDS)]
        self.assertEqual(read[0], (-1.0, -1.0))
        self.assertEqual(read[1], (3.0, -1.0))
        self.assertEqual(read[2], (-1.0, 3.0))
        for record in range(RECORDS):
            head = bytes_[record * STRIDE + 8:record * STRIDE + 16]
            self.assertEqual(struct.unpack("<2f", head), (0.0, -1.75),
                             "the offset the module reads at location 1")
            for component in struct.unpack(
                    "<4f", bytes_[record * STRIDE + 16:record * STRIDE + 32]):
                self.assertGreaterEqual(abs(component), 2.0,
                                        "the ignored attributes leave the clip space")

    def test_the_draw_selects_the_three_vertices_in_order(self):
        case = self.case(CASE_ID)
        indices = case["indices"]
        self.assertEqual(indices["format"], "uint16")
        self.assertEqual(indices["length"], 2 * RECORDS)
        values = struct.unpack("<3H", bytes.fromhex(indices["initial_hex"]))
        self.assertEqual(values, (0, 1, 2))
        self.assertEqual(case["vertices"], RECORDS)
        self.assertNotIn("instance_count", case)
        self.assertNotIn("scissor", case)

    def test_the_expectation_is_the_fixtures_partial_coverage(self):
        case = self.case(CASE_ID)
        self.assertEqual(case["coverage"], "partial")
        self.assertEqual(case["attachment"]["clear_hex"], CLEAR)
        self.assertEqual(case["expected_hex"], FRAME_HEX)
        self.assertNotEqual(case["expected_hex"], FULL_COVERAGE_HEX)
        texels = [FRAME_HEX[at:at + 8] for at in range(0, 32, 8)]
        self.assertEqual(texels, [TEXEL, CLEAR, TEXEL, TEXEL])
        plan = self.plan(CASE_ID)
        self.assertEqual(
            plan.writes,
            [((ATTACHMENT[0], ATTACHMENT[1], 0), bytes.fromhex(FRAME_HEX))],
        )
        self.assertEqual(plan.allocations, {ATTACHMENT[0]: bytes.fromhex(FRAME_HEX)})
        self.assertEqual(plan.rails, set(VULKAN_RAILS))

    def test_the_marker_gates_the_capture(self):
        reporting = synthetic_report(self.suite, self.digest)
        reporting["results"].append(
            {
                "id": CASE_ID,
                "completion": "CompletedVisible",
                "writebacks": [
                    {
                        "allocation": ATTACHMENT[0],
                        "view": ATTACHMENT[1],
                        "offset": 0,
                        "bytes_hex": FRAME_HEX,
                    }
                ],
                "allocations": [
                    {"allocation": ATTACHMENT[0], "bytes_hex": FRAME_HEX}
                ],
            }
        )
        compare.validate_capture(self.suite, self.digest, reporting, "vulkan")

        # A rail the markers do not name may not report the case: the two AIR
        # modules are the Vulkan translator's, and this rail selects its
        # reviewed MSL module by the layout's exact shape.
        native = synthetic_report(self.suite, self.digest, backend="native-metal")
        native["results"].append(
            {
                "id": CASE_ID,
                "completion": "CompletedVisible",
                "writebacks": [
                    {
                        "allocation": ATTACHMENT[0],
                        "view": ATTACHMENT[1],
                        "offset": 0,
                        "bytes_hex": FRAME_HEX,
                    }
                ],
                "allocations": [
                    {"allocation": ATTACHMENT[0], "bytes_hex": FRAME_HEX}
                ],
            }
        )
        with self.assertRaisesRegex(
            compare.CaptureError, "not a rail this render case runs on"
        ):
            compare.validate_capture(self.suite, self.digest, native, "native-metal")

    def test_a_frame_that_ignored_the_stream_is_refused(self):
        # The falsifiability of the case: a rail that ignored the vertex stream
        # — or bound the wrong attribute to an input — lands the milestone's
        # full-coverage frame, and the capture that reports it is refused.
        reporting = synthetic_report(self.suite, self.digest)
        reporting["results"].append(
            {
                "id": CASE_ID,
                "completion": "CompletedVisible",
                "writebacks": [
                    {
                        "allocation": ATTACHMENT[0],
                        "view": ATTACHMENT[1],
                        "offset": 0,
                        "bytes_hex": FULL_COVERAGE_HEX,
                    }
                ],
                "allocations": [
                    {"allocation": ATTACHMENT[0], "bytes_hex": FULL_COVERAGE_HEX}
                ],
            }
        )
        with self.assertRaises(compare.CaptureError):
            compare.validate_capture(self.suite, self.digest, reporting, "vulkan")

    def test_a_layout_that_declares_only_the_read_attributes_is_refused(self):
        # The shape is the *superset*: a layout that declares exactly what the
        # module reads is the pre-increment shape, which this arm does not
        # measure.
        def narrow(case):
            attributes = case["vertex_layout"]["buffers"][0]["attributes"]
            del attributes[2:]

        self.refused(narrow, "declares four attributes")

    def test_ignored_bytes_inside_the_clip_space_are_refused(self):
        # A fixture whose ignored bytes could pass for inputs would not be
        # falsifiable, so the comparator holds them outside the clip space.
        def tame(case):
            binding = case["vertex_buffers"][0]
            bytes_ = bytearray.fromhex(binding["initial_hex"])
            for record in range(RECORDS):
                struct.pack_into("<4f", bytes_, record * STRIDE + 16, 0.0, 0.0, 0.0, 0.0)
            binding["initial_hex"] = bytes_.hex()

        self.refused(tame, "outside the clip space")

    def test_a_reviewed_case_cannot_claim_the_superset_shape(self):
        # The arm is the translator's: a case without `translated_stages` names
        # a reviewed MSL pair, whose layout and module this shape is not.
        def reviewed(case):
            del case["translated_stages"]
            case["metal"] = {
                "path": "shaders/quad_indexed_2x2.metal",
                "sha256": "0" * 64,
            }

        self.refused(reviewed, "expected fields stride, attributes")


if __name__ == "__main__":
    unittest.main()
