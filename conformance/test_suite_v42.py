"""The per-stage stage-buffer ceiling's own suite (`research/docs/23` §117, E-SB2).

`suite-v42.json` carries one translated render case against one declaring pass:
the two stages declare **thirteen** `[[buffer(n)]]` slots between them — the
vertex stage's seven (a 24-byte positions view and six 16-byte offsets folded
into the clip position) and the fragment stage's six (the widened increment's
own sum) — which is past the older *pipeline-level* bound of eight and inside
the contract's per-stage ceiling of eight. The frame is the six fragment
payloads' sum: `e0` on every texel of the 2x2 attachment.

These are comparator and schema checks, not GPU execution evidence: the
Lavapipe readings live in
`crates/metal-api-vulkan/tests/render_e2e.rs`
(`a_per_stage_stage_buffer_shape_enters_the_rail_and_lands_its_bytes`) and in
the capture archives the evidence directory keeps.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare


CONFORMANCE = Path(__file__).resolve().parent
V42_PATH = CONFORMANCE / "suite-v42.json"

DECLARING_ID = "render_declaring_stage_buffer_per_stage"
CASE_ID = "stage_buffer_thirteen_tint_2x2"

ATTACHMENT = (900, 910)
CLEAR = "fefefefe"

# The three `float2` clip positions (`b0`) and the six zero offsets the vertex
# stage's own arguments carry.
POSITIONS_HEX = "000080bf000080bf" + "00004040000080bf" + "000080bf00004040"
ZERO_HEX = "00" * 16
# The fragment stage's six payloads: five slots of 32/255 and one of 64/255.
LOW_HEX = "8180003e" * 4
HIGH_HEX = "8180803e" * 4
FRAME_HEX = "e0" * 16

# The two AIR fixtures the case pins, by their own digests.
VERTEX_SHA = "add96f5d4570900a32a5f4370f26c3bc432b963a3b771e8e83ce2733d75684ca"
FRAGMENT_SHA = "c8ec931d95df8db099d10582a9635171feef1f608a54727e82db978a26c5d791"

VULKAN_RAILS = ("vulkan", "vulkan-objects")
VERTEX_SLOTS = 7
FRAGMENT_SLOTS = 6


class PerStageStageBufferCeilingTests(unittest.TestCase):
    def setUp(self):
        self.raw = V42_PATH.read_bytes()
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

    def test_the_suite_carries_the_declaring_pass_and_the_thirteen_slot_case(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v42")
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
        self.assertEqual(buffers[0]["allocation"], ATTACHMENT[0])
        self.assertEqual(buffers[0]["view"], ATTACHMENT[1])
        self.assertEqual(buffers[0]["access"], "read")
        self.assertEqual(
            self.case(CASE_ID)["capture_rails"],
            list(VULKAN_RAILS),
            "the thirteen-slot case names the two rails that translate its stages",
        )
        self.assertEqual(self.case(CASE_ID)["declaring_case"], DECLARING_ID)
        self.assertEqual(self.case(CASE_ID)["vertices"], 3)
        self.assertEqual(self.case(CASE_ID)["viewport"], [0, 0, 2, 2])

    def test_the_case_declares_thirteen_slots_seven_and_six(self):
        case = self.case(CASE_ID)
        entries = case["stage_buffers"]
        self.assertEqual(len(entries), VERTEX_SLOTS + FRAGMENT_SLOTS)
        self.assertGreater(len(entries), compare.MAX_RENDER_STAGE_BUFFERS)
        self.assertEqual(
            [entry["stage"] for entry in entries],
            ["vertex"] * VERTEX_SLOTS + ["fragment"] * FRAGMENT_SLOTS,
            "the list is canonical: the vertex stage's slots first, then the fragment's",
        )
        vertex = [entry for entry in entries if entry["stage"] == "vertex"]
        fragment = [entry for entry in entries if entry["stage"] == "fragment"]
        self.assertEqual([entry["index"] for entry in vertex], list(range(VERTEX_SLOTS)))
        self.assertEqual(
            [entry["index"] for entry in fragment], list(range(FRAGMENT_SLOTS))
        )
        self.assertEqual(vertex[0]["footprint"], {"static": {"max_bytes": 24}})
        self.assertEqual(vertex[0]["length"], 24)
        self.assertEqual(vertex[0]["initial_hex"], POSITIONS_HEX)
        for entry in vertex[1:]:
            self.assertEqual(entry["footprint"], {"static": {"max_bytes": 16}})
            self.assertEqual(entry["length"], 16)
            self.assertEqual(entry["initial_hex"], ZERO_HEX)
        for index, entry in enumerate(fragment):
            self.assertEqual(entry["footprint"], {"static": {"max_bytes": 16}})
            self.assertEqual(entry["length"], 16)
            self.assertEqual(
                entry["initial_hex"], HIGH_HEX if index == FRAGMENT_SLOTS - 1 else LOW_HEX
            )
        allocations = [entry["allocation"] for entry in entries]
        views = [entry["view"] for entry in entries]
        self.assertEqual(len(set(allocations)), len(entries))
        self.assertEqual(len(set(views)), len(entries))

    def test_the_air_modules_are_the_two_declared_fixtures(self):
        case = self.case(CASE_ID)
        self.assertEqual(case["vertex_entry"], "render_stage_buffer_seven_positions")
        self.assertEqual(case["fragment_entry"], "render_stage_buffer_six_rgba8")
        for stage, digest in (("vertex", VERTEX_SHA), ("fragment", FRAGMENT_SHA)):
            declared = case["translated_stages"][stage]
            self.assertEqual(declared["sha256"], digest, stage)
            source = CONFORMANCE / declared["path"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                digest,
                f"{stage}: the AIR fixture on disk is the one the case pins",
            )

    def test_the_expectation_is_the_thirteen_slot_frame(self):
        case = self.case(CASE_ID)
        self.assertEqual(case["expected_hex"], FRAME_HEX)
        self.assertEqual(case["attachment"]["clear_hex"], CLEAR)
        self.assertNotEqual(case["expected_hex"], CLEAR * 4)
        plan = self.plan(CASE_ID)
        self.assertEqual(
            plan.writes,
            [((ATTACHMENT[0], ATTACHMENT[1], 0), bytes.fromhex(FRAME_HEX))],
        )
        self.assertEqual(plan.allocations, {ATTACHMENT[0]: bytes.fromhex(FRAME_HEX)})
        self.assertEqual(plan.rails, set(VULKAN_RAILS))

    def test_the_list_bound_is_the_pair_sum(self):
        # Seventeen slots is one past the pipeline-level list bound: the
        # comparator refuses the suite before a rail would have to, which is
        # the same reading the wire's length prefix takes (`research/docs/23`
        # §117, E-SB2).

        def widen(case):
            for index in range(VERTEX_SLOTS, VERTEX_SLOTS + 4):
                case["stage_buffers"].append(
                    {
                        "stage": "vertex",
                        "index": index,
                        "access": "read",
                        "footprint": {"static": {"max_bytes": 16}},
                        "allocation": 1300 + index,
                        "view": 1400 + index,
                        "offset": 0,
                        "length": 16,
                        "initial_hex": ZERO_HEX,
                    }
                )

        self.refused(widen, "list bound is 16 slots")
        self.assertEqual(
            compare.MAX_RENDER_STAGE_BUFFER_DECLARATIONS,
            2 * compare.MAX_RENDER_STAGE_BUFFERS,
        )

    def test_the_per_stage_bound_is_still_the_stages_own(self):
        # Thirteen slots are inside the pair's sum but a *stage* can still name
        # too many: moving three of the vertex stage's slots to the fragment
        # stage makes nine of them fragment slots, and the comparator refuses
        # the suite with the per-stage reading rather than the list's.
        def restage(case):
            for entry in case["stage_buffers"][2:5]:
                entry["stage"] = "fragment"

        self.refused(restage, "ceiling is 8 slots per stage")


if __name__ == "__main__":
    unittest.main()
