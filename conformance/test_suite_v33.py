"""The render sampler's second 8-bit byte order (`research/docs/23` §3.3, §107).

These are comparator and schema checks, not GPU execution evidence. They pin
the v33 increment: a render case whose sampled texture is a `bgra8_unorm` guest
view — the census's 6455-line `texture_bind` shape — while the pass renders
into the reviewed `rgba8_unorm` attachment. A texel-centre sample is an
identity copy *of the colours*, so the expectation is the uploaded texel hex
spelled in the attachment's own byte order: across the two admitted layouts
every texel is the red/blue swap of the other, which is what makes the fixture
falsifiable — a rail that uploaded the bytes under the wrong name would land
the unswapped texture instead.

The case is marked for the Vulkan rails alone: the native rail's reviewed table
names one layout until its own Apple-side reading lands
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
V33_PATH = CONFORMANCE / "suite-v33.json"

DECLARING_ID = "render_declaring_quad_extent"
CASE_ID = "sampled_texel_bgra_4x4"
TEXTURE = (1060, 1070)
ATTACHMENT = (900, 910)
CLEAR = "11223344"

# The fixture's pattern, in the two admitted layouts. Texel `(i, j)` is the
# colour `(0x10 + 0x10i, 0x50 + 0x10j, 0x90, 0xc0 + 0x10(i + j))`; the
# `bgra8_unorm` upload carries it blue first, and the `rgba8_unorm` attachment
# carries it red first.
TEXELS = "".join("%02x%02x%02x%02x" % (
    0x90, 0x50 + 0x10 * j, 0x10 + 0x10 * i, (0xc0 + 0x10 * (i + j)) & 0xff,
) for j in range(4) for i in range(4))
EXPECTED = "".join("%02x%02x%02x%02x" % (
    0x10 + 0x10 * i, 0x50 + 0x10 * j, 0x90, (0xc0 + 0x10 * (i + j)) & 0xff,
) for j in range(4) for i in range(4))

VULKAN_RAILS = ("vulkan", "vulkan-objects")


def swapped(texels):
    """The same colours spelled in the other admitted byte order."""
    return "".join(texels[offset + index]
                   for offset in range(0, len(texels), 8)
                   for index in (4, 5, 2, 3, 0, 1, 6, 7))


class RenderSamplerByteOrderTests(unittest.TestCase):
    def setUp(self):
        self.raw = V33_PATH.read_bytes()
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

    def test_the_case_states_the_bgra8_upload_and_the_colour_expectation(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v33")
        self.assertEqual(self.suite["guard_byte"], 0)
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], [CASE_ID])
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["entry"], "copy_word")
        for kind in ("air", "metal"):
            source = CONFORMANCE / declaring[kind]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             declaring[kind]["sha256"], kind)
        buffers = {buffer["binding"]: buffer for buffer in declaring["buffers"]}
        self.assertEqual((buffers[0]["access"], buffers[0]["allocation"],
                          buffers[0]["view"], buffers[0]["offset"],
                          buffers[0]["length"]), ("read",) + ATTACHMENT + (0, 64))
        self.assertEqual(buffers[0]["initial_hex"], "cd" * 64)
        self.assertEqual((buffers[1]["access"], buffers[1]["allocation"],
                          buffers[1]["view"], buffers[1]["offset"],
                          buffers[1]["length"]), ("write", 920, 930, 4, 4))
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
            "format": "bgra8_unorm",
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
        # The observation is the swap of the upload, texel by texel, and the
        # upload is sixteen pairwise-distinct texels none of which is the clear
        # colour: a rail that ignored the texture, filtered it or read the bytes
        # under the other name cannot land the expectation.
        self.assertEqual(EXPECTED, swapped(TEXELS))
        chunks = [TEXELS[offset:offset + 8] for offset in range(0, len(TEXELS), 8)]
        self.assertEqual(len(set(chunks)), len(chunks))
        self.assertNotIn(CLEAR, chunks)

    def test_the_plan_holds_the_colours_in_the_attachment_order(self):
        expectation = self.plan()
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], 0), bytes.fromhex(EXPECTED))])
        self.assertEqual(expectation.allocations, {ATTACHMENT[0]: bytes.fromhex(EXPECTED)})
        self.assertEqual(expectation.attachment, (ATTACHMENT[0], ATTACHMENT[1], 0, 64))
        self.assertEqual(expectation.texture_uploads, 1)
        self.assertEqual(expectation.rails, set(VULKAN_RAILS))

    def test_the_unswapped_upload_is_refused(self):
        # The claim is the *colours*, so the texture's own bytes are the wrong
        # answer wherever the two layouts differ.
        self.refused(lambda case: case.update(expected_hex=TEXELS),
                     "attachment's own byte order")

    def test_the_sibling_layout_lands_the_same_colours(self):
        suite = copy.deepcopy(self.suite)
        case = self.case(suite)
        case["fragment_textures"][0]["format"] = "rgba8_unorm"
        case["fragment_textures"][0]["initial_hex"] = swapped(TEXELS)
        sibling = compare._render_plan(compare._suite_plan(suite), suite)[CASE_ID]
        self.assertEqual(sibling.writes, self.plan().writes)
        self.assertEqual(sibling.allocations, self.plan().allocations)

    def test_a_texture_of_another_layout_is_refused(self):
        # The narrow lanes are admitted too (`research/docs/23` §113,
        # `test_suite_v37.py`), so the format outside the window is the
        # eight-byte `rgba16_float` texel rather than a narrow one.
        self.refused(lambda case: case["fragment_textures"][0].update(
            format="rgba16_float"), "one 8-bit unorm surface")

    def test_an_attachment_outside_the_two_layouts_is_refused(self):
        self.refused(lambda case: case["attachment"].update(format="r32float"),
                     "colour attachment is one 8-bit four-component unorm surface")

    def test_a_rule_expectation_needs_one_layout(self):
        # The rule form states one layout's plane (`research/docs/23` §73), so
        # a rule texture whose attachment names the other layout has no closed
        # form to compare against.
        def mutate(case):
            case["fragment_textures"][0].pop("initial_hex")
            case["fragment_textures"][0]["texel_rule"] = "xy_u16le_v1"
            case["fragment_textures"][0]["width"] = 2048
            case["fragment_textures"][0]["height"] = 2048
            case.update(expected_rule="xy_u16le_v1")
            case.pop("expected_hex")
            case["attachment"]["width"] = 2048
            case["attachment"]["height"] = 2048
        self.refused(mutate, "have to name it together")

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
        # still names one layout.
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

    def test_a_swapped_writeback_is_refused(self):
        # A capture that lands the uploaded bytes instead of the colours is the
        # reading this fixture exists to falsify.
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
