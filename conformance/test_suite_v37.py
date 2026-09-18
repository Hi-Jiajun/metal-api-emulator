"""The render sampler's narrow lanes (`research/docs/23` §3.3, §113).

These are comparator and schema checks, not GPU execution evidence. They pin
the v37 increment: a render case whose sampled texture is a one-byte
`r8_unorm` guest view — the census's 1,778-line `texture_bind` shape
(`evidence/gate3-census-v25b-2026-09-18/`) — while the pass renders into the
reviewed `rgba8_unorm` attachment.

The expectation is a *derivation*, not a copy: a texel-centre sample of an
`r8_unorm` texture reads `(r, 0, 0, 1)`, so the attachment carries the source
byte in red and the format's own fill in the other three lanes. That is what
makes the fixture falsifiable — a rail that uploaded the single byte under a
four-component name, or logged it into another lane, would land something
other than the fill the format's rule states.

The case is marked for the Vulkan rails alone: the native rail's reviewed
table names the four-component surface until its own Apple-side reading lands
(`crates/metal-api-native/src/render.rs`), so naming a native rail here would
claim a capture it refuses.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V37_PATH = CONFORMANCE / "suite-v37.json"

DECLARING_ID = "render_declaring_quad_extent"
CASE_ID = "sampled_texel_r8_4x4"
TEXTURE = (1060, 1070)
ATTACHMENT = (900, 910)
CLEAR = "11223344"

# The fixture's pattern: texel `(i, j)` carries the byte `0x10 * (1 + i + 4j)`,
# so the sixteen texels of the 4x4 surface are pairwise distinct and the last
# one wraps to zero — a value the clear colour does not contain either.
TEXELS = "".join("%02x" % (0x10 * (1 + i + 4 * j) & 0xff)
                 for j in range(4) for i in range(4))
# The frame: the source byte in red, then the sampling rule's fills.
EXPECTED = "".join("%02x0000ff" % (0x10 * (1 + i + 4 * j) & 0xff)
                   for j in range(4) for i in range(4))

VULKAN_RAILS = ("vulkan", "vulkan-objects")


class NarrowSampledFormatTests(unittest.TestCase):
    def setUp(self):
        self.raw = V37_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def case(self, suite=None):
        suite = suite or self.suite
        for case in suite["render_cases"]:
            if case["id"] == CASE_ID:
                return case
        self.fail(f"the suite carries no {CASE_ID} case")

    def plan(self, suite=None):
        suite = suite or self.suite
        return compare._render_plan(compare._suite_plan(suite), suite)[CASE_ID]

    def refused(self, mutate, message):
        suite = copy.deepcopy(self.suite)
        mutate(self.case(suite))
        with self.assertRaisesRegex(compare.CaptureError, message):
            compare._render_plan(compare._suite_plan(suite), suite)

    def test_the_case_states_the_narrow_upload_and_the_fill_expectation(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v37")
        self.assertEqual(self.suite["guard_byte"], 165)
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], [CASE_ID])
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["entry"], "copy_word")
        for kind in ("air", "metal"):
            source = CONFORMANCE / declaring[kind]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             declaring[kind]["sha256"], kind)
        case = self.case()
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual((case["vertex_entry"], case["fragment_entry"]),
                         ("render_sampled_quad_vertex", "render_sampled_texel"))
        module = CONFORMANCE / case["metal"]["path"]
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                         case["metal"]["sha256"])
        self.assertEqual((case["vertices"], case["viewport"]), (3, [0, 0, 4, 4]))
        self.assertEqual(case["fragment_textures"], [{
            "allocation": TEXTURE[0],
            "view": TEXTURE[1],
            "format": "r8_unorm",
            "width": 4,
            "height": 4,
            "initial_hex": TEXELS,
        }])
        attachment = case["attachment"]
        self.assertEqual((attachment["allocation"], attachment["view"],
                          attachment["format"], attachment["width"],
                          attachment["height"]), ATTACHMENT + ("rgba8_unorm", 4, 4))
        self.assertEqual((attachment["load"], attachment["store"]), ("clear", "store"))
        self.assertEqual(attachment["clear_hex"], CLEAR)
        self.assertEqual(case["expected_hex"], EXPECTED)
        self.assertEqual(sorted(case["capture_rails"]), sorted(VULKAN_RAILS))
        # One byte per texel: the upload is exactly the 4x4 surface's own bytes,
        # not four per texel with three quarters of them invented.
        self.assertEqual(len(TEXELS), 32)
        # The observation is the derivation, and the derivation is the format's
        # rule rather than the upload: the two differ in three of every four
        # bytes, which is what a rail that ignored the narrow fill would land.
        self.assertEqual(
            EXPECTED,
            compare._sampled_expectation("r8_unorm", "rgba8_unorm",
                                         bytes.fromhex(TEXELS), 16).hex())
        self.assertNotEqual(EXPECTED, TEXELS)
        # Sixteen pairwise-distinct frame texels, none of them the clear colour:
        # a rail that ignored the texture, filtered it or read the byte into
        # another lane cannot land the expectation.
        chunks = [EXPECTED[offset:offset + 8] for offset in range(0, len(EXPECTED), 8)]
        self.assertEqual(len(chunks), 16)
        self.assertEqual(len(set(chunks)), len(chunks))
        self.assertNotIn(CLEAR, chunks)

    def test_the_plan_holds_the_fill_in_the_attachment_order(self):
        expectation = self.plan()
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], 0), bytes.fromhex(EXPECTED))])
        self.assertEqual(expectation.allocations, {ATTACHMENT[0]: bytes.fromhex(EXPECTED)})
        self.assertEqual(expectation.attachment, (ATTACHMENT[0], ATTACHMENT[1], 0, 64))
        self.assertEqual(expectation.texture_uploads, 1)
        self.assertEqual(expectation.rails, set(VULKAN_RAILS))

    def test_the_two_byte_lane_derives_its_own_fill(self):
        # The same fixture with the two-byte sibling (`rg8_unorm`): red and
        # green come from the texel's bytes, blue and alpha from the rule.
        suite = copy.deepcopy(self.suite)
        case = self.case(suite)
        case["fragment_textures"][0]["format"] = "rg8_unorm"
        case["fragment_textures"][0]["initial_hex"] = "".join(
            "%02x%02x" % (0x20 + 0x10 * i, 0x30 + 0x10 * j)
            for j in range(4) for i in range(4))
        case["expected_hex"] = "".join(
            "%02x%02x00ff" % (0x20 + 0x10 * i, 0x30 + 0x10 * j)
            for j in range(4) for i in range(4))
        sibling = compare._render_plan(compare._suite_plan(suite), suite)[CASE_ID]
        self.assertEqual(
            sibling.writes,
            [((ATTACHMENT[0], ATTACHMENT[1], 0), bytes.fromhex(case["expected_hex"]))])

    def test_a_four_component_upload_is_refused(self):
        # The upload is one byte per texel, so a four-byte spelling is four
        # times too long and the length rule is the one that answers.
        suite = copy.deepcopy(self.suite)
        case = self.case(suite)
        case["fragment_textures"][0]["initial_hex"] = "".join(
            TEXELS[i:i + 2] + "0000ff" for i in range(0, len(TEXELS), 2))
        with self.assertRaisesRegex(compare.CaptureError, "texture initial length"):
            compare._render_plan(compare._suite_plan(suite), suite)

    def test_the_uploaded_bytes_are_not_the_expectation(self):
        # The narrow lane's reading is the format's rule, so a frame with the
        # fill lanes in another order is the wrong answer: the spelling below
        # carries the same red byte with the rule's alpha and blue lanes
        # swapped, which is exactly what a rail that read a narrow texel as
        # `(r, 0, 1, 0)` would land.
        shifted = "".join("%s000100" % TEXELS[index:index + 2]
                          for index in range(0, len(TEXELS), 2))
        self.assertEqual(len(shifted), len(EXPECTED))
        self.refused(lambda case: case.update(expected_hex=shifted),
                     "uploaded texels in the attachment's own byte order")

    def test_a_short_spelling_is_refused_by_the_extent_rule(self):
        # The frame is the render area's four bytes per texel, so the source's
        # own one-byte spelling is not a frame at all.
        self.refused(lambda case: case.update(expected_hex=TEXELS),
                     "frame's texels do not match the extent")

    def test_the_bytes_shifted_one_lane_are_refused(self):
        # A rail that read the single byte into green instead of red would land
        # `00xx00ff`; the derivation refuses it.
        shifted = "".join("00%s00ff" % EXPECTED[offset:offset + 2]
                          for offset in range(0, len(EXPECTED), 8))
        self.refused(lambda case: case.update(expected_hex=shifted),
                     "uploaded texels in the attachment's own byte order")

    def test_a_format_outside_the_lanes_is_refused(self):
        self.refused(lambda case: case["fragment_textures"][0].update(
            format="rgba16_float"), "one 8-bit unorm surface")

    def test_a_narrow_attachment_is_refused(self):
        # The narrow lanes are *sampling* sources: the colour attachment is
        # still one of the two four-component layouts, because that is the
        # surface both rails read back.
        self.refused(lambda case: case["attachment"].update(format="r8_unorm"),
                     "colour attachment is one 8-bit four-component unorm surface")

    def test_another_extent_is_refused(self):
        self.refused(lambda case: case["fragment_textures"][0].update(width=2, height=2),
                     "share the attachment's extent")

    def test_the_marker_gates_the_capture(self):
        reporting = synthetic_report(self.suite, self.digest)
        reporting["results"].append({
            "id": CASE_ID,
            "completion": "CompletedVisible",
            "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                            "offset": 0, "bytes_hex": EXPECTED}],
            "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": EXPECTED}],
        })
        compare.validate_capture(self.suite, self.digest, reporting, "vulkan")

        # A rail the case's marker does not name may not report it, and the
        # marker names the Vulkan rails alone: the native rail's reviewed table
        # still names the four-component surface.
        native = synthetic_report(self.suite, self.digest, backend="native-metal")
        native["results"].append({
            "id": CASE_ID,
            "completion": "CompletedVisible",
            "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                            "offset": 0, "bytes_hex": EXPECTED}],
            "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": EXPECTED}],
        })
        with self.assertRaisesRegex(compare.CaptureError, "not a rail this render case runs on"):
            compare.validate_capture(self.suite, self.digest, native, "native-metal")

    def test_a_capture_of_the_upload_alone_is_refused(self):
        # The reading this fixture exists to falsify: a capture that logged the
        # source's own one-byte texels as if they were the frame.
        reporting = synthetic_report(self.suite, self.digest)
        reporting["results"].append({
            "id": CASE_ID,
            "completion": "CompletedVisible",
            "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                            "offset": 0, "bytes_hex": TEXELS}],
            "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": TEXELS}],
        })
        with self.assertRaises(compare.CaptureError):
            compare.validate_capture(self.suite, self.digest, reporting, "vulkan")


if __name__ == "__main__":
    unittest.main()
