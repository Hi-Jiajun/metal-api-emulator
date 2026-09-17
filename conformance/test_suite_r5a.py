"""The reviewed attachment window's 2048x2048 boundary (R5a, `research/docs/23` §73).

These are comparator and schema checks, not GPU execution evidence. They pin the
increment that widened the declared window from R1b's 64x64 to the desktop sizes
the guest profile measured, and that solved what the widening owes: a 2048x2048
attachment is four million texels, so the fixture states its expectation as a
*rule* over the texel coordinates instead of 32 MiB of hex, and a capture
reports the plane's digest plus the declared readback windows instead of the
plane's bytes. The rule, the digest and the windows are implemented in all three
review surfaces — this comparator, `provider-capture.rs` and `NativeOracle.swift`
— and the negative probes below are what keep that form a check rather than a
summary.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
import test_suite_v28 as v28

CONFORMANCE = Path(__file__).resolve().parent
REPOSITORY = CONFORMANCE.parent
V28_PATH = CONFORMANCE / "suite-v28.json"

RULE_ID = v28.SAMPLED_RULE_2048_ID
RULE_DECLARING = "render_declaring_attachment_2048x2048"
RULE = compare.XY_U16LE_V1
CEILING = 2048
RULE_ATTACHMENT = v28.SAMPLED_RULE_2048_ATTACHMENT
RULE_DIGEST = v28.SAMPLED_RULE_2048_DIGEST
RULE_WINDOWS = v28.SAMPLED_RULE_2048_WINDOWS

# The four sources that state the reviewed window, each with the line that has
# to carry the same number the comparator does. The Rust rails declare the
# per-axis pair, the capture tool and the Swift oracle one value, so the
# spellings differ; what the test is about is that no surface drifts.
CEILING_SOURCES = {
    "crates/metal-api-vulkan/src/provider.rs":
        r"REVIEWED_ATTACHMENT_CEILING: \[u64; 2\] = \[2048, 2048\]",
    "crates/metal-api-native/src/render.rs":
        r"REVIEWED_ATTACHMENT_CEILING: \[u64; 2\] = \[2048, 2048\]",
    "examples/metal-smoke/src/bin/provider-capture.rs":
        r"REVIEWED_ATTACHMENT_CEILING: u64 = 2048;",
    "conformance/NativeOracle.swift":
        r"reviewedAttachmentCeiling = 2048",
}


class WideAttachmentTests(unittest.TestCase):
    """The R5a fixture and the comparator rules that bound it."""

    def setUp(self):
        self.raw = V28_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def case(self, case_id, suite=None):
        suite = suite or self.suite
        for case in suite["render_cases"]:
            if case["id"] == case_id:
                return case
        self.fail(f"the suite carries no {case_id} case")

    def declaring(self, declaring_id, suite=None):
        suite = suite or self.suite
        for case in suite["cases"]:
            if case["id"] == declaring_id:
                return case
        self.fail(f"the suite carries no {declaring_id} declaring case")

    def plan(self, suite=None):
        suite = suite or self.suite
        return compare._render_plan(compare._suite_plan(suite), suite)

    def refused(self, mutate, message):
        suite = copy.deepcopy(self.suite)
        mutate(self.case(RULE_ID, suite))
        with self.assertRaisesRegex(compare.CaptureError, message):
            compare._render_plan(compare._suite_plan(suite), suite)

    def initial_bytes(self, buffer):
        pattern = bytes.fromhex(buffer["initial_repeat_hex"])
        return pattern * (buffer["length"] // len(pattern))

    def capture(self):
        """A synthetic capture for the wide case and its declaring pass.

        The suite is narrowed the way the v28 tests narrow it: every other
        render case is marked on another rail, so this capture owes exactly the
        wide landing and the declaring pass its own marker names.
        """
        suite = copy.deepcopy(self.suite)
        for case in suite["render_cases"]:
            case["capture_rails"] = (["vulkan"] if case["id"] == RULE_ID
                                     else ["native-metal"])
        digest = hashlib.sha256(
            json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
        # The compute cases — including the wide declaring pass, whose image is
        # 16 MiB and therefore reported by digest — come from the v28 helper
        # every other observation test uses.
        report = v28.counted_declaring(suite, digest, "vulkan")
        report["results"].append(v28.sampled_rule_2048_result())
        return suite, digest, report

    # -- the window and the rule -------------------------------------------

    def test_the_four_review_surfaces_share_the_ceiling(self):
        self.assertEqual(compare.REVIEWED_ATTACHMENT_CEILING, CEILING)
        for path, pattern in CEILING_SOURCES.items():
            with self.subTest(source=path):
                text = (REPOSITORY / path).read_text(encoding="utf-8")
                self.assertRegex(text, pattern)

    def test_the_rule_is_the_texel_coordinates_little_endian(self):
        # Hand-computed texels: the first two bytes are x, the last two y, each
        # little-endian. Nothing here reads a constant of the implementation
        # under test, so a moved channel or a swapped axis fails this file.
        for x, y, expect in ((0, 0, "00000000"), (1, 0, "01000000"), (0, 1, "00000100"),
                             (255, 3, "ff000300"), (256, 3, "00010300"),
                             (2047, 2047, "ff07ff07"), (1984, 1984, "c007c007")):
            with self.subTest(texel=(x, y)):
                self.assertEqual(compare._rule_texel(RULE, x, y).hex(), expect)
        # The rule is injective over the reviewed window: different texels never
        # carry the same four bytes, including one 256-row step apart.
        self.assertNotEqual(compare._rule_texel(RULE, 0, 1),
                            compare._rule_texel(RULE, 1, 0))
        self.assertNotEqual(compare._rule_texel(RULE, 7, 5),
                            compare._rule_texel(RULE, 7, 5 + 256))

    def test_the_rule_plane_matches_a_row_wise_spelling(self):
        # A small plane built by hand against the strided row builder the
        # comparator uses for the 2048x2048 plane.
        plane = compare._rule_plane(RULE, 3, 2)
        self.assertEqual(plane.hex(), "00000000" "01000000" "02000000"
                                      "00000100" "01000100" "02000100")
        self.assertEqual(compare._rule_digest(RULE, 3, 2),
                         hashlib.sha256(plane).hexdigest())

    def test_the_wide_case_is_the_sampled_rule_shape(self):
        case = self.case(RULE_ID)
        self.assertEqual(case["declaring_case"], RULE_DECLARING)
        self.assertEqual(case["vertex_entry"], "render_sampled_quad_vertex")
        self.assertEqual(case["fragment_entry"], "render_sampled_texel")
        self.assertEqual(case["viewport"], [0, 0, CEILING, CEILING])
        self.assertEqual(case["expected_rule"], RULE)
        self.assertNotIn("expected_hex", case)
        attachment = case["attachment"]
        self.assertEqual((attachment["width"], attachment["height"]), (CEILING, CEILING))
        self.assertEqual(attachment["load"], "clear")
        self.assertEqual(attachment["clear_hex"], "80808080")
        texture = case["fragment_textures"][0]
        self.assertEqual(texture["texel_rule"], RULE)
        self.assertNotIn("initial_hex", texture)
        self.assertEqual((texture["width"], texture["height"]), (CEILING, CEILING))
        self.assertEqual([[window["x"], window["y"], window["width"], window["height"]]
                          for window in case["readback_windows"]],
                         [list(window) for window in RULE_WINDOWS])
        self.assertEqual(case["capture_rails"], list(v28.ALL_RAILS))
        # The clear colour is outside the rule's reach: the two halves of
        # 80808080 name texel (0x8080, 0x8080), which the plane does not have.
        self.assertFalse(compare._rule_reaches_colour(
            RULE, bytes.fromhex(attachment["clear_hex"]), CEILING, CEILING))

    def test_the_wide_declaring_view_is_a_repeated_pattern(self):
        case = self.case(RULE_ID)
        declared = [buffer for buffer in self.declaring(RULE_DECLARING)["buffers"]
                    if buffer["allocation"] == case["attachment"]["allocation"]
                    and buffer["view"] == case["attachment"]["view"]]
        self.assertEqual(len(declared), 1)
        declared = declared[0]
        self.assertEqual(declared["access"], "read")
        self.assertEqual(declared["length"], CEILING * CEILING * 4)
        self.assertEqual(declared["allocation_size"], declared["length"])
        self.assertEqual(declared["offset"], 0)
        # The 16 MiB view is spun from one pattern instead of 32 MiB of hex.
        self.assertEqual(declared["initial_repeat_hex"], "cdcdcdcd")
        self.assertNotIn("initial_hex", declared)
        self.assertEqual(compare._buffer_initial_bytes(
            declared, "synthetic", declared["allocation"], declared["view"],
            declared["length"]),
            bytes.fromhex("cdcdcdcd") * (declared["length"] // 4))

    def test_the_plan_routes_the_rule_landing(self):
        plan = self.plan()
        expectation = plan[RULE_ID]
        self.assertIsNotNone(expectation.rule)
        self.assertEqual(expectation.rule.rule, RULE)
        self.assertEqual((expectation.rule.width, expectation.rule.height),
                         (CEILING, CEILING))
        self.assertEqual(expectation.rule.digest, RULE_DIGEST)
        self.assertEqual(expectation.attachment, RULE_ATTACHMENT)
        self.assertEqual(expectation.texture_uploads, 1)
        self.assertEqual(len(expectation.rule.windows), len(RULE_WINDOWS))
        for window, declared in zip(expectation.rule.windows, RULE_WINDOWS):
            self.assertEqual(window[:4], declared)
            self.assertEqual(window[4], compare._rule_window_bytes(RULE, declared))

    # -- the refusals -------------------------------------------------------

    def test_one_texel_outside_the_ceiling_is_refused(self):
        over = CEILING + 1

        def widen(case):
            case["attachment"].update(width=over)
            case["fragment_textures"][0].update(width=over)
            case["viewport"] = [0, 0, over, CEILING]

        self.refused(widen, f"one to {CEILING} texels per axis")

    def test_a_rule_below_the_megapixel_form_is_refused(self):
        # The rule form is what replaces an impractical hex spelling; a small
        # case states its texels, which is what keeps the pairwise-distinctness
        # scan available.
        def shrink(case):
            case["attachment"].update(width=64, height=64)
            case["viewport"] = [0, 0, 64, 64]
            case["fragment_textures"][0].update(width=64, height=64)
            case["readback_windows"] = [{"x": 0, "y": 0, "width": 8, "height": 8}]

        self.refused(shrink, "a texel rule is the megapixel form")

    def test_a_rule_expectation_needs_its_windows(self):
        self.refused(lambda case: case.update(readback_windows=[]),
                     "needs its readback windows")

    def test_a_window_outside_the_plane_is_refused(self):
        self.refused(lambda case: case["readback_windows"].__setitem__(
            1, {"x": CEILING - 8, "y": CEILING - 8, "width": 64, "height": 64}),
            "the readback window leaves the attachment plane")

    def test_a_window_wider_than_the_reviewed_shape_is_refused(self):
        self.refused(lambda case: case["readback_windows"].__setitem__(
            0, {"x": 0, "y": 0, "width": 128, "height": 8}),
            "expected integer in 1..64")

    def test_a_clear_colour_the_rule_reaches_is_refused(self):
        # 00000000 is the rule's texel (0, 0): a rail that ignored the draw
        # could land it.
        self.refused(lambda case: case["attachment"].update(clear_hex="00000000"),
                     "a rule texel equals the clear colour")

    def test_the_expectation_has_to_be_the_textures_own_rule(self):
        self.refused(lambda case: case.update(expected_rule="unknown_rule_v9"),
                     "the expectation has to be the texture's own rule")

    def test_a_rule_case_has_to_bind_the_texture(self):
        self.refused(lambda case: case.pop("fragment_textures"),
                     "a rule expectation is the reviewed sampling shape's")

    # -- the observation is still falsifiable -------------------------------

    def test_the_reported_form_is_the_digest_and_the_windows(self):
        suite, digest, report = self.capture()
        compare.validate_capture(suite, digest, report)
        result = [entry for entry in report["results"] if entry["id"] == RULE_ID][0]
        writeback = result["writebacks"][0]
        self.assertEqual(writeback["bytes_sha256"], RULE_DIGEST)
        self.assertEqual(writeback["bytes_length"], CEILING * CEILING * 4)
        self.assertNotIn("bytes_hex", writeback)
        for window, declared in zip(writeback["observed_windows"], RULE_WINDOWS):
            self.assertEqual((window["x"], window["y"], window["width"], window["height"]),
                             declared)
        self.assertEqual(result["allocations"][0]["bytes_sha256"], RULE_DIGEST)

    def test_a_plane_with_one_changed_texel_is_refused(self):
        # The falsifiability claim: one texel of the four-million-texel plane
        # changes the digest, and the capture that reports the changed plane
        # cannot pass. The probe changes a single byte of one texel far from any
        # window, so only the digest can catch it.
        suite, digest, report = self.capture()
        result = [entry for entry in report["results"] if entry["id"] == RULE_ID][0]
        plane = bytearray(compare._rule_plane(RULE, CEILING, CEILING))
        plane[(1024 * CEILING + 1024) * 4 + 1] ^= 0x01
        mutated = hashlib.sha256(bytes(plane)).hexdigest()
        self.assertNotEqual(mutated, RULE_DIGEST)
        result["writebacks"][0]["bytes_sha256"] = mutated
        with self.assertRaisesRegex(compare.CaptureError, "is not the xy_u16le_v1 plane's"):
            compare.validate_capture(suite, digest, report)

    def test_one_changed_texel_inside_a_window_is_named(self):
        # The window itself is checked texel for texel, so the failure names the
        # coordinates and both four-byte values instead of only the digest.
        suite, digest, report = self.capture()
        result = [entry for entry in report["results"] if entry["id"] == RULE_ID][0]
        window = result["writebacks"][0]["observed_windows"][1]
        raw = bytearray.fromhex(window["bytes_hex"])
        raw[(3 * 64 + 5) * 4] ^= 0x01
        window["bytes_hex"] = raw.hex()
        with self.assertRaisesRegex(
                compare.CaptureError,
                r"texel \(1989, 1987\) reads .*, the xy_u16le_v1 rule stores"):
            compare.validate_capture(suite, digest, report)

    def test_a_dropped_or_moved_window_is_refused(self):
        suite, digest, report = self.capture()
        result = [entry for entry in report["results"] if entry["id"] == RULE_ID][0]
        result["writebacks"][0]["observed_windows"].pop()
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the capture has to report the 2 readback windows"):
            compare.validate_capture(suite, digest, report)
        suite, digest, report = self.capture()
        result = [entry for entry in report["results"] if entry["id"] == RULE_ID][0]
        result["writebacks"][0]["observed_windows"][0]["x"] = 1
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the reported rectangle is not the one the suite declares"):
            compare.validate_capture(suite, digest, report)

    def test_the_wide_declaring_image_is_reported_by_digest(self):
        # 16 MiB of hex would ride in every capture of the suite, so the image
        # form is the digest; the comparator recomputes it from the suite's own
        # guard fill and view bytes.
        suite, digest, report = self.capture()
        compare.validate_capture(suite, digest, report)
        result = [entry for entry in report["results"] if entry["id"] == RULE_DECLARING][0]
        image = result["allocations"][0]
        self.assertEqual(image["allocation"], RULE_ATTACHMENT[0])
        self.assertEqual(image["bytes_length"], CEILING * CEILING * 4)
        self.assertNotIn("bytes_hex", image)
        # One byte of the 16 MiB fill changed is refused.
        mutated = bytearray(self.initial_bytes(self.declaring(RULE_DECLARING)["buffers"][0]))
        mutated[12345] ^= 0x01
        image["bytes_sha256"] = hashlib.sha256(bytes(mutated)).hexdigest()
        with self.assertRaisesRegex(compare.CaptureError, "one byte of the"):
            compare.validate_capture(suite, digest, report)

    def test_the_verbatim_cap_is_the_same_number_in_the_capture_surfaces(self):
        for path, pattern in (
                ("examples/metal-smoke/src/bin/provider-capture.rs",
                 r"MAX_VERBATIM_ALLOCATION_BYTES: usize = 1024 \* 1024;"),
                ("conformance/NativeOracle.swift",
                 r"maximumVerbatimAllocationBytes = 1_048_576")):
            with self.subTest(source=path):
                text = (REPOSITORY / path).read_text(encoding="utf-8")
                self.assertRegex(text, pattern)
        self.assertEqual(compare.MAX_VERBATIM_ALLOCATION_BYTES, 1_048_576)
        self.assertLess(compare.MAX_VERBATIM_ALLOCATION_BYTES,
                        compare.MAX_ALLOCATION_BYTES)


if __name__ == "__main__":
    unittest.main()
