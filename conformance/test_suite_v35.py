"""The folded stage-buffer pair's own suite (`research/docs/23` §3.3, E-TX9).

`suite-v35.json` carries one translated render case against one declaring pass:
both of the case's stages read `[[buffer(0)]]` — the vertex stage's 24-byte
affine positions and the fragment stage's 16-byte tint — and the frame is the
reviewed `stage_buffer_borrowed_tint_2x2` reading (`40 80 c0 ff` in the covered
texel, the clear sentinel in the other three). The two stages' Metal index
spaces are independent, so a rail that folds them onto one descriptor slot
lands a frame this suite refuses.

These are comparator and schema checks, not GPU execution evidence: the
Lavapipe readings live in
`crates/metal-api-vulkan/tests/render_stage_buffer_namespace_e2e.rs` and in the
capture archives the evidence directory keeps.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V35_PATH = CONFORMANCE / "suite-v35.json"

DECLARING_ID = "render_declaring_stage_buffer_namespace"
CASE_ID = "stage_buffer_namespace_tint_2x2"

ATTACHMENT = (900, 910)
CLEAR = "fefefefe"

# The reviewed case's own bytes (`conformance/suite-v31.json`'s
# `stage_buffer_borrowed_tint_2x2`): the three Metal-NDC positions and the
# tint whose quantisation is the frame.
POSITIONS_HEX = "666666bf6666663f000000006666663f666666bf00000000"
TINT_HEX = "8180803e8180003fc1c0403f0000803f"
FRAME_HEX = "4080c0ff" + CLEAR * 3

# The two AIR fixtures the case pins, by their own digests.
VERTEX_SHA = "9371ae8aa9c44da9b2bd59a156d4b516d758c2df7f68b7712432502786d8084f"
FRAGMENT_SHA = "09ee634f72785f1f86597946fcfa028717559f280cd3cfe2ba4cb7a17506d3d4"

VULKAN_RAILS = ("vulkan", "vulkan-objects")


class FoldedStageBufferNamespaceTests(unittest.TestCase):
    def setUp(self):
        self.raw = V35_PATH.read_bytes()
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

    def test_the_suite_carries_the_declaring_pass_and_the_folded_case(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v35")
        self.assertEqual(
            [case["id"] for case in self.suite["cases"]], [DECLARING_ID]
        )
        self.assertEqual(
            [case["id"] for case in self.suite["render_cases"]], [CASE_ID]
        )
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
        self.assertEqual(buffers[0]["initial_hex"], CLEAR * 4)
        self.assertEqual(
            (buffers[1]["access"], buffers[1]["allocation"], buffers[1]["view"]),
            ("write", 920, 930),
        )

    def test_the_case_pins_both_translated_fixtures_by_digest(self):
        case = self.case(CASE_ID)
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual(
            (case["vertex_entry"], case["fragment_entry"]),
            ("render_vertex_positions", "render_stage_buffer_rgba8"),
        )
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

    def test_both_stages_read_the_same_metal_index(self):
        # The increment's own shape: one Metal index, two stages, two slots. A
        # case whose stages read different indices would be the shape the rail
        # already executed before this change.
        slots = self.case(CASE_ID)["stage_buffers"]
        self.assertEqual(
            [(slot["stage"], slot["index"]) for slot in slots],
            [("vertex", 0), ("fragment", 0)],
        )
        for slot in slots:
            self.assertEqual(slot["access"], "read")
        vertex, fragment = slots
        self.assertEqual(
            vertex["footprint"],
            {
                "affine": {
                    "accesses": [
                        {
                            "base_offset": 0,
                            "access_size": 4,
                            "terms": [{"axis": 0, "stride": 8}],
                        },
                        {
                            "base_offset": 4,
                            "access_size": 4,
                            "terms": [{"axis": 0, "stride": 8}],
                        },
                    ]
                }
            },
        )
        self.assertEqual(fragment["footprint"], {"static": {"max_bytes": 16}})
        self.assertEqual(vertex["initial_hex"], POSITIONS_HEX)
        self.assertEqual(fragment["initial_hex"], TINT_HEX)
        self.assertEqual(
            (vertex["length"], fragment["length"]), (len(POSITIONS_HEX) // 2, 16)
        )
        self.assertEqual(
            len({slot["view"] for slot in slots}),
            2,
            "the two stages' slots are two views",
        )
        self.assertEqual(
            len({slot["allocation"] for slot in slots}),
            2,
            "the two stages' slots are two allocations",
        )

    def test_the_expectation_is_the_reviewed_frame(self):
        case = self.case(CASE_ID)
        self.assertEqual(case["expected_hex"], FRAME_HEX)
        self.assertEqual(case["coverage"], "partial")
        self.assertEqual(case["attachment"]["clear_hex"], CLEAR)
        self.assertNotEqual(case["expected_hex"], CLEAR * 4)
        plan = self.plan(CASE_ID)
        self.assertEqual(
            plan.writes,
            [((ATTACHMENT[0], ATTACHMENT[1], 0), bytes.fromhex(FRAME_HEX))],
        )
        self.assertEqual(plan.allocations, {ATTACHMENT[0]: bytes.fromhex(FRAME_HEX)})
        self.assertEqual(plan.rails, set(VULKAN_RAILS))

    def test_the_vertex_reach_has_to_fit_the_declared_view(self):
        # The affine footprint reaches `0 + 2 * 8 + 4 = 20` bytes for the three
        # vertices the draw issues; a view shorter than that cannot carry the
        # declaration, and the comparator refuses the suite before a rail
        # would have to.
        self.refused(
            lambda case: case["stage_buffers"][0].update(
                length=16, initial_hex=POSITIONS_HEX[:32]
            ),
            "past the view",
        )

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
        # modules are the Vulkan translator's, and the object rails bind the
        # slots through the object API's own entry point.
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

    def test_a_frame_that_folded_the_two_slots_is_refused(self):
        # The falsifiability of the case: a rail that folded the two stages'
        # `(set 0, binding 0)` onto one descriptor would land either the
        # vertex bytes or the fragment bytes where the reviewed frame puts a
        # mixed one, and the capture that reports it is refused.
        reporting = synthetic_report(self.suite, self.digest)
        folded = CLEAR * 4
        reporting["results"].append(
            {
                "id": CASE_ID,
                "completion": "CompletedVisible",
                "writebacks": [
                    {
                        "allocation": ATTACHMENT[0],
                        "view": ATTACHMENT[1],
                        "offset": 0,
                        "bytes_hex": folded,
                    }
                ],
                "allocations": [
                    {"allocation": ATTACHMENT[0], "bytes_hex": folded}
                ],
            }
        )
        with self.assertRaises(compare.CaptureError):
            compare.validate_capture(self.suite, self.digest, reporting, "vulkan")


if __name__ == "__main__":
    unittest.main()
