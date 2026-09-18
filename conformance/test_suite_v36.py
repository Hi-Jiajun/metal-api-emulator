"""The gathered-extent arm's own suite (`research/docs/23` §3.3, §111, E-TX10).

`suite-v36.json` carries one translated render case against one declaring pass:
the case's fragment stage states its own absolute sample coordinates and the
pass binds one `rgba8_unorm` texture whose extent (6x4) is *not* the render
area's (4x4), so the rail binds the source at its own extent and the attachment
is a function of the source's bytes. Over the suite's grid — column `x` holds
`16 * x` in red — the module's two absolute samples land column five (`50`) and
column one (`10`), so every fragment stores `50 10 00 ff`; a rail that gathered
the source into the destination grid would answer the second sample with the
gathered column two (`20`) instead, which is the frame this suite refuses.

These are comparator and schema checks, not GPU execution evidence: the Lavapipe
readings live in
`crates/metal-api-vulkan/tests/render_texture_extent_e2e.rs` (including the
reading where one replaced source byte moves exactly the channel it decides) and
in the capture archives the evidence directory keeps.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V36_PATH = CONFORMANCE / "suite-v36.json"

DECLARING_ID = "render_declaring_gathered_extent"
CASE_ID = "gathered_extent_6x4_into_4x4"

ATTACHMENT = (900, 910)
TEXTURE = (940, 950)
CLEAR = "fefefefe"

SOURCE_WIDTH = 6
SOURCE_HEIGHT = 4

# The module's own reading (`render_texture_extent_e2e.rs`'s `TRANSLATED_FRAME`).
FRAME_HEX = "501000ff" * 16
# What a rail that gathered the source into the destination grid would answer
# the second sample with: the gathered column one is source column two (`20`).
GATHERED_RULE_HEX = "502000ff" * 16

# The two AIR fixtures the case pins, by their own digests.
VERTEX_SHA = "be63f8b9efc815f929706fd5ee42e71ec3d12aa9b10d7444e0af8caf67755ddc"
FRAGMENT_SHA = "96c4de8def6de8d3776fef860d39638d22e71eac2db53cdf007fe27513cfeb8d"

VULKAN_RAILS = ("vulkan", "vulkan-objects")


def source_hex():
    """The source grid: every texel names its own position, and column `x`
    carries `16 * x` in red, so the module's two absolute samples are the
    frame's own two channels."""
    return "".join(
        "%02x%02x80ff" % (16 * x, y)
        for y in range(SOURCE_HEIGHT)
        for x in range(SOURCE_WIDTH)
    )


class GatheredExtentTests(unittest.TestCase):
    def setUp(self):
        self.raw = V36_PATH.read_bytes()
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

    def test_the_suite_carries_the_declaring_pass_and_the_gathered_case(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v36")
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
        self.assertEqual(buffers[0]["length"], 4 * 4 * 4, "the render area's own view")
        self.assertEqual(buffers[0]["initial_hex"], CLEAR * 16)
        self.assertEqual(
            (buffers[1]["access"], buffers[1]["allocation"], buffers[1]["view"]),
            ("write", 920, 930),
        )

    def test_the_case_pins_the_translated_pair_and_names_the_vulkan_rails(self):
        case = self.case(CASE_ID)
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual(
            (case["vertex_entry"], case["fragment_entry"]),
            ("render_fullscreen_triangle", "render_sample_texture_2d"),
        )
        self.assertNotIn("metal", case)
        translated = case["translated_stages"]
        for kind, sha, suffix in (
            ("vertex", VERTEX_SHA, ".vert.ll"),
            ("fragment", FRAGMENT_SHA, ".frag.ll"),
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

    def test_the_source_is_another_extent_and_its_texels_are_distinct(self):
        # The increment's own shape: the sampled source's extent is not the
        # render area's, and no two uploaded texels are equal, so a rail that
        # ignored the binding or read the wrong one cannot pass by accident.
        case = self.case(CASE_ID)
        texture = case["fragment_textures"][0]
        self.assertEqual((texture["width"], texture["height"]), (SOURCE_WIDTH, SOURCE_HEIGHT))
        self.assertNotEqual(texture["width"], case["attachment"]["width"])
        self.assertEqual(texture["initial_hex"], source_hex())
        chunks = [texture["initial_hex"][at:at + 8] for at in range(0, len(source_hex()), 8)]
        self.assertEqual(len(chunks), SOURCE_WIDTH * SOURCE_HEIGHT)
        self.assertEqual(len(set(chunks)), len(chunks))
        self.assertNotIn(CLEAR, chunks)

    def test_the_expectation_is_the_module_s_own_reading(self):
        case = self.case(CASE_ID)
        self.assertEqual(case["expected_hex"], FRAME_HEX)
        self.assertEqual(case["attachment"]["clear_hex"], CLEAR)
        self.assertNotEqual(case["expected_hex"], CLEAR * 16)
        self.assertNotIn("coverage", case)
        plan = self.plan(CASE_ID)
        self.assertEqual(
            plan.writes,
            [((ATTACHMENT[0], ATTACHMENT[1], 0), bytes.fromhex(FRAME_HEX))],
        )
        self.assertEqual(plan.allocations, {ATTACHMENT[0]: bytes.fromhex(FRAME_HEX)})
        self.assertEqual(plan.rails, set(VULKAN_RAILS))
        self.assertEqual(plan.texture_uploads, 1)

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
        # modules are the Vulkan translator's, and this rail answers every
        # source of another extent with its own refusal by name.
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

    def test_a_frame_that_gathered_the_source_is_refused(self):
        # The falsifiability of the case: a rail that gathered the 6x4 source
        # into the 4x4 destination grid would answer the module's second sample
        # with the gathered column two (`20`), and the capture that reports it
        # is refused.
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
                        "bytes_hex": GATHERED_RULE_HEX,
                    }
                ],
                "allocations": [
                    {"allocation": ATTACHMENT[0], "bytes_hex": GATHERED_RULE_HEX}
                ],
            }
        )
        with self.assertRaises(compare.CaptureError):
            compare.validate_capture(self.suite, self.digest, reporting, "vulkan")

    def test_a_gathered_source_that_shares_the_render_area_is_refused(self):
        # The arm's own definition: a source whose extent equals the render
        # area's is the reviewed same-extent window, not this shape.
        def share_the_area(case):
            texture = case["fragment_textures"][0]
            texture["width"] = case["attachment"]["width"]
            texture["height"] = case["attachment"]["height"]
            texture["initial_hex"] = "".join(
                "%02x%02x80ff" % (16 * x, y)
                for y in range(case["attachment"]["height"])
                for x in range(case["attachment"]["width"])
            )

        self.refused(share_the_area, "has to differ from the render area")

    def test_the_reviewed_arm_still_requires_the_shared_extent(self):
        # The rule the increment keeps: a *reviewed* case's sampled texture
        # still has to share the attachment's extent, because its fragment stage
        # samples at the fragment's own centre.
        def reviewed_arm(case):
            del case["translated_stages"]
            case["metal"] = {
                "path": "shaders/render_sampled_4x4.metal",
                "sha256": "0" * 64,
            }

        self.refused(reviewed_arm, "has to share the attachment's extent")


if __name__ == "__main__":
    unittest.main()
