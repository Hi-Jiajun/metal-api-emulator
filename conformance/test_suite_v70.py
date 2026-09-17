"""Render-side texture sampling (`research/docs/23` §3.3, v70).

These are comparator and schema checks, not GPU execution evidence. They pin
the v70 increment: a render pass whose fragment stage samples one
`rgba8_unorm` texture of the render area's own extent, so every fragment stands
on a texel centre and the sample is an identity copy. The case's expectation
*is* the uploaded texture, which is what makes the claim falsifiable: a rail
that ignored the binding reads back the clear colour, one that filtered the
sample reads a neighbour, and one that flipped or transposed the uv reads
another row or column — none of which can equal sixteen pairwise-distinct
texels.
"""

import copy
import json
from pathlib import Path
import unittest

import compare

CONFORMANCE = Path(__file__).resolve().parent
V28_PATH = CONFORMANCE / "suite-v28.json"

CASE_ID = "sampled_texel_4x4"
TEXTURE = (1060, 1070)
ATTACHMENT = (900, 910)
CLEAR = "11223344"
TEXELS = "".join(f"{x:02x}{y:02x}{x + y:02x}ff" for y in range(4) for x in range(4))
RAILS = ["native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
         "native-metal-provider-objects"]


class RenderSamplerSuiteTests(unittest.TestCase):
    def setUp(self):
        self.suite = json.loads(V28_PATH.read_bytes())

    def case(self, suite=None):
        suite = suite or self.suite
        for case in suite["render_cases"]:
            if case["id"] == CASE_ID:
                return case
        self.fail(f"the suite carries no {CASE_ID} case")

    def plan(self, suite=None):
        suite = suite or self.suite
        return compare._render_plan(compare._suite_plan(suite), suite)[CASE_ID]

    def test_the_reviewed_case_states_the_uploaded_texels(self):
        case = self.case()
        self.assertEqual(case["fragment_textures"], [{
            "allocation": TEXTURE[0],
            "view": TEXTURE[1],
            "format": "rgba8_unorm",
            "width": 4,
            "height": 4,
            "initial_hex": TEXELS,
        }])
        self.assertEqual(case["attachment"]["clear_hex"], CLEAR)
        self.assertEqual(case["expected_hex"], TEXELS)
        self.assertEqual(case["metal"]["path"], "shaders/render_sampled_4x4.metal")
        self.assertEqual(case["vertex_entry"], "render_sampled_quad_vertex")
        self.assertEqual(case["fragment_entry"], "render_sampled_texel")
        self.assertEqual(case["capture_rails"], RAILS)
        chunks = [TEXELS[index:index + 8] for index in range(0, len(TEXELS), 8)]
        self.assertEqual(len(set(chunks)), len(chunks))
        self.assertNotIn(CLEAR, chunks)
        expectation = self.plan()
        self.assertEqual(expectation.texture_uploads, 1)
        self.assertEqual(expectation.attachment, (ATTACHMENT[0], ATTACHMENT[1], 0, 64))

    def refused(self, mutate, message):
        suite = copy.deepcopy(self.suite)
        mutate(self.case(suite))
        with self.assertRaisesRegex(compare.CaptureError, message):
            compare._render_plan(compare._suite_plan(suite), suite)

    def test_the_expectation_has_to_be_the_uploaded_texels(self):
        def mutate(case):
            case["expected_hex"] = "0a0b0cff" + TEXELS[8:]
        self.refused(mutate, "the expectation has to be the uploaded texels")

    def test_a_repeated_texel_is_refused(self):
        def mutate(case):
            # The first two texels repeat, so the set is no longer pairwise
            # distinct while the expectation still is the uploaded texture.
            case["fragment_textures"][0]["initial_hex"] = (
                TEXELS[:8] + TEXELS[:8] + TEXELS[16:])
            case["expected_hex"] = case["fragment_textures"][0]["initial_hex"]
        self.refused(mutate, "pairwise distinct")

    def test_a_texel_equal_to_the_clear_is_refused(self):
        def mutate(case):
            case["fragment_textures"][0]["initial_hex"] = CLEAR + TEXELS[8:]
            case["expected_hex"] = case["fragment_textures"][0]["initial_hex"]
        self.refused(mutate, "equals the clear colour")

    def test_another_extent_is_refused(self):
        def mutate(case):
            case["fragment_textures"][0]["width"] = 2
            case["fragment_textures"][0]["height"] = 2
        self.refused(mutate, "share the attachment's extent")

    def test_a_second_texture_is_refused(self):
        def mutate(case):
            case["fragment_textures"].append(dict(case["fragment_textures"][0]))
        self.refused(mutate, "binds exactly one texture")

    def test_a_texture_of_another_format_is_refused(self):
        def mutate(case):
            # `bgra8_unorm` is the widened window's second layout
            # (`research/docs/23` §107, `test_suite_v33.py`); the format outside
            # it is the eight-byte `rgba16_float` texel.
            case["fragment_textures"][0]["format"] = "rgba16_float"
        self.refused(mutate, "four-component unorm surface")

    def test_the_shape_carries_no_other_state(self):
        def mutate(case):
            case["scissor"] = [0, 0, 2, 4]
        self.refused(mutate, "carries no scissor")

        def multisampled(case):
            case["multisample"] = {"sample_count": 4}
        self.refused(multisampled, "carries no multisample")

        def vertex_input(case):
            case["vertex_layout"] = {"buffers": []}
        self.refused(vertex_input, "carries no vertex_layout")

    def test_the_stored_attachment_has_to_be_cleared_and_stored(self):
        def mutate(case):
            case["attachment"]["store"] = "dontcare"
        self.refused(mutate, "clears and stores")


if __name__ == "__main__":
    unittest.main()
