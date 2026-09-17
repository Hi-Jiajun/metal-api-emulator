"""Attachment window widening (R1b, `research/docs/23` §70).

These are comparator and schema checks, not GPU execution evidence. They pin
the increment that turned the render rail's fixed 4x4 attachment window into a
declared, device-gated one: a 16x16 full-cover triangle whose texels are the
fragment output, and the 64x64 reviewed-window boundary whose sampled texture
of its own extent is the expectation. Both are inside every conformant
device's declared window -- the rails declare the smaller of the reviewed
ceiling and their device's own framebuffer limit, and Vulkan's minimum
`maxFramebufferWidth` is 4096 -- so neither case carries a device gate, and the
expected bytes are what a rail that refused the extent could never report.
"""

import copy
import json
from pathlib import Path
import unittest

import compare
import test_suite_v28 as v28

CONFORMANCE = Path(__file__).resolve().parent
V28_PATH = CONFORMANCE / "suite-v28.json"

FULL_COVER_ID = v28.FULL_COVER_16X16_ID
SAMPLED_64_ID = v28.SAMPLED_64_ID
FULL_COVER_DECLARING = "render_declaring_attachment_16x16"
SAMPLED_64_DECLARING = "render_declaring_attachment_64x64"
CEILING = 64


class AttachmentWindowTests(unittest.TestCase):
    """The R1b fixtures and the comparator rule that bounds them."""

    def setUp(self):
        self.raw = V28_PATH.read_bytes()
        self.suite = json.loads(self.raw)

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

    def test_the_three_review_surfaces_share_the_ceiling(self):
        # The comparator's ceiling is the number the fixtures measure; the Rust
        # rails and the Swift oracle state the same number in their own sources
        # (`crates/metal-api-vulkan/src/provider.rs`,
        # `crates/metal-api-native/src/render.rs`,
        # `examples/metal-smoke/src/bin/provider-capture.rs`,
        # `conformance/NativeOracle.swift`), and the boundary case sits on it.
        self.assertEqual(compare.REVIEWED_ATTACHMENT_CEILING, CEILING)
        boundary = self.case(SAMPLED_64_ID)
        self.assertEqual((boundary["attachment"]["width"], boundary["attachment"]["height"]),
                         (CEILING, CEILING))

    def test_the_16x16_case_is_the_full_cover_triangle(self):
        # A rail that rendered only part of the attachment would leave the clear
        # colour behind, so the uniform fragment output is what the case claims.
        case = self.case(FULL_COVER_ID)
        self.assertEqual(case["declaring_case"], FULL_COVER_DECLARING)
        self.assertEqual(case["vertex_entry"], "render_fullscreen_triangle")
        self.assertEqual(case["fragment_entry"], "render_solid_rgba8")
        self.assertEqual(case["viewport"], [0, 0, 16, 16])
        self.assertEqual(case["vertices"], 3)
        self.assertEqual(case["expected_hex"], "4080c0ff" * (16 * 16))
        self.assertNotEqual(case["attachment"]["clear_hex"], case["expected_hex"][:4])
        self.assertEqual(case["capture_rails"], list(v28.ALL_RAILS))

    def test_the_boundary_case_is_the_sampled_identity(self):
        case = self.case(SAMPLED_64_ID)
        self.assertEqual(case["declaring_case"], SAMPLED_64_DECLARING)
        self.assertEqual(case["vertex_entry"], "render_sampled_quad_vertex")
        self.assertEqual(case["fragment_entry"], "render_sampled_texel")
        self.assertEqual(case["viewport"], [0, 0, CEILING, CEILING])
        texture = case["fragment_textures"][0]
        self.assertEqual((texture["width"], texture["height"]), (CEILING, CEILING))
        self.assertEqual(case["expected_hex"], texture["initial_hex"])
        chunks = [texture["initial_hex"][index:index + 8]
                  for index in range(0, len(texture["initial_hex"]), 8)]
        self.assertEqual(len(chunks), CEILING * CEILING)
        self.assertEqual(len(set(chunks)), len(chunks),
                         "the boundary texels have to be pairwise distinct")
        self.assertNotIn(case["attachment"]["clear_hex"], chunks)
        self.assertEqual(case["capture_rails"], list(v28.ALL_RAILS))

    def test_the_declaring_views_cover_the_two_extents(self):
        for case_id, declaring_id, extent in (
                (FULL_COVER_ID, FULL_COVER_DECLARING, 16 * 16 * 4),
                (SAMPLED_64_ID, SAMPLED_64_DECLARING, CEILING * CEILING * 4)):
            case = self.case(case_id)
            attachment = case["attachment"]
            declared = [buffer for buffer in self.declaring(declaring_id)["buffers"]
                        if buffer["allocation"] == attachment["allocation"]
                        and buffer["view"] == attachment["view"]]
            self.assertEqual(len(declared), 1, f"{case_id}: exactly one declaring view")
            declared = declared[0]
            self.assertEqual(declared["access"], "read")
            self.assertEqual(declared["length"], extent)
            self.assertEqual(declared["length"],
                             attachment["width"] * attachment["height"] * 4)
            self.assertNotEqual(declared["initial_hex"], case["expected_hex"],
                                f"{case_id}: the declared bytes must differ from the landing")

    def test_the_plans_route_the_two_landings(self):
        plan = self.plan()
        full = plan[FULL_COVER_ID]
        self.assertEqual(full.attachment, (900, 910, 0, 16 * 16 * 4))
        self.assertEqual(full.texture_uploads, 0)
        boundary = plan[SAMPLED_64_ID]
        self.assertEqual(boundary.attachment, (900, 910, 0, CEILING * CEILING * 4))
        self.assertEqual(boundary.texture_uploads, 1)

    def refused(self, mutate, message):
        suite = copy.deepcopy(self.suite)
        mutate(self.case(FULL_COVER_ID, suite))
        with self.assertRaisesRegex(compare.CaptureError, message):
            compare._render_plan(compare._suite_plan(suite), suite)

    def test_one_texel_beyond_the_ceiling_is_refused(self):
        # The committed boundary case is the window's own edge; one texel more
        # is the refusal the widening moved out from four texels per axis.
        over = CEILING + 1
        self.refused(lambda case: case["attachment"].update(width=over),
                     f"one to {CEILING} texels per axis")

    def test_the_committed_boundary_case_is_admitted(self):
        # The "one texel beyond" refusal above is a boundary only because the
        # committed case at the ceiling still plans.
        self.assertIn(SAMPLED_64_ID, self.plan())


if __name__ == "__main__":
    unittest.main()
