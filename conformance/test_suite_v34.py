"""The widened stage-buffer ceiling (`research/docs/23` §3.3, §108, E-SB1).

`suite-v34.json` carries two translated render cases against one declaring
pass: a six-slot `[[buffer(n)]]` list — census v13's `stage_buffer_shape_gt4`
family, inside the widened eight — whose summed colour is the frame, and the
old ceiling's largest list, four slots, whose frame the widening must leave
unchanged. A ninth slot is the comparator's own refusal: the ceiling is eight,
and a suite that states more is refused before either rail can run it.

These are comparator and schema checks, not GPU execution evidence: the
Lavapipe readings live in `crates/metal-api-vulkan/tests/render_e2e.rs` and in
the capture archives the evidence directory keeps.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V34_PATH = CONFORMANCE / "suite-v34.json"

DECLARING_ID = "render_declaring_widened_extent"
SIX_ID = "stage_buffer_six_tint_2x2"
FOUR_ID = "stage_buffer_four_tint_2x2"

# The attachment the declaring pass's own view names, and the clear sentinel
# both cases start from.
ATTACHMENT = (900, 910)
CLEAR = "fefefefe"

# One `float4` of `32/255` and one of `64/255`, the fixtures' payloads.
PAYLOAD_32 = "8180003e" * 4
PAYLOAD_64 = "8180803e" * 4
# Five 32/255 slots and one 64/255 slot sum to 224/255 on every channel; three
# and one sum to 160/255. Both sit far from a quantisation tie.
SIX_FRAME = "e0e0e0e0" * 4
FOUR_FRAME = "a0a0a0a0" * 4
# The largest list the ceiling admitted before the widening.
OLD_CEILING = 4

VULKAN_RAILS = ("vulkan", "vulkan-objects")


class WidenedStageBufferCeilingTests(unittest.TestCase):
    def setUp(self):
        self.raw = V34_PATH.read_bytes()
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

    def refused(self, case_id, mutate, message):
        suite = copy.deepcopy(self.suite)
        mutate(self.case(case_id, suite))
        with self.assertRaisesRegex(compare.CaptureError, message):
            compare._render_plan(compare._suite_plan(suite), suite)

    def test_the_suite_carries_the_two_lists_and_the_declaring_pass(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v34")
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual(
            [case["id"] for case in self.suite["render_cases"]], [SIX_ID, FOUR_ID]
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

    def test_every_case_pins_its_translated_fixtures_by_digest(self):
        for case_id, fragment_entry, fragment_sha in (
            (SIX_ID, "render_stage_buffer_six_rgba8",
             "c8ec931d95df8db099d10582a9635171feef1f608a54727e82db978a26c5d791"),
            (FOUR_ID, "render_stage_buffer_four_rgba8",
             "ee6cd84bc434bb0f3d83b238c6bdf3ccb9032b713f35b3c81209b85da4736ad2"),
        ):
            case = self.case(case_id)
            self.assertEqual(case["declaring_case"], DECLARING_ID)
            self.assertEqual(
                (case["vertex_entry"], case["fragment_entry"]),
                ("render_fullscreen_triangle", fragment_entry),
            )
            translated = case["translated_stages"]
            fragment = CONFORMANCE / translated["fragment"]["path"]
            self.assertEqual(
                hashlib.sha256(fragment.read_bytes()).hexdigest(), fragment_sha
            )
            vertex = CONFORMANCE / translated["vertex"]["path"]
            self.assertEqual(
                hashlib.sha256(vertex.read_bytes()).hexdigest(),
                translated["vertex"]["sha256"],
            )
            self.assertEqual(sorted(case["capture_rails"]), sorted(VULKAN_RAILS))

    def test_the_six_slot_shape_reads_six_distinct_static_slots(self):
        case = self.case(SIX_ID)
        slots = case["stage_buffers"]
        self.assertEqual(len(slots), 6)
        for position, slot in enumerate(slots):
            self.assertEqual(slot["stage"], "fragment")
            self.assertEqual(slot["index"], position)
            self.assertEqual(slot["access"], "read")
            self.assertEqual(slot["footprint"], {"static": {"max_bytes": 16}})
            self.assertEqual(slot["length"], 16)
        self.assertEqual(
            [slot["initial_hex"] for slot in slots],
            [PAYLOAD_32] * 5 + [PAYLOAD_64],
        )
        self.assertEqual(
            len({slot["view"] for slot in slots}), 6,
            "six declarations fill six distinct views",
        )
        self.assertEqual(
            len({slot["allocation"] for slot in slots}), 6,
            "six declarations come from six distinct allocations",
        )
        self.assertEqual(case["expected_hex"], SIX_FRAME)
        self.assertEqual(
            case["attachment"]["clear_hex"], CLEAR,
            "the frame differs from the value the pass started from",
        )

    def test_the_four_slot_shape_is_the_old_ceiling_largest_list(self):
        case = self.case(FOUR_ID)
        slots = case["stage_buffers"]
        self.assertEqual(len(slots), OLD_CEILING)
        self.assertEqual(
            [slot["initial_hex"] for slot in slots],
            [PAYLOAD_32] * 3 + [PAYLOAD_64],
        )
        self.assertEqual(case["expected_hex"], FOUR_FRAME)

    def test_the_plans_hold_the_summed_expectations(self):
        six = self.plan(SIX_ID)
        self.assertEqual(
            six.writes,
            [((ATTACHMENT[0], ATTACHMENT[1], 0), bytes.fromhex(SIX_FRAME))],
        )
        self.assertEqual(six.allocations, {ATTACHMENT[0]: bytes.fromhex(SIX_FRAME)})
        self.assertEqual(six.rails, set(VULKAN_RAILS))
        four = self.plan(FOUR_ID)
        self.assertEqual(four.allocations, {ATTACHMENT[0]: bytes.fromhex(FOUR_FRAME)})

    def test_a_ninth_slot_is_refused_at_the_widened_ceiling(self):
        # The count rule the widening moves: `MAX_RENDER_STAGE_BUFFERS` is
        # eight (`metal_api_core::provider`), and the comparator restates it, so
        # a ninth declaration cannot reach a rail through a suite either.
        self.assertEqual(compare.MAX_RENDER_STAGE_BUFFERS, 8)

        def mutate(case):
            # Six declarations plus three more reach the ninth slot, one past
            # the widened ceiling.
            for offset, index in enumerate((6, 7, 8)):
                extra = copy.deepcopy(case["stage_buffers"][0])
                extra.update(index=index, allocation=1190 + offset, view=1190 + offset)
                case["stage_buffers"].append(extra)

        self.refused(SIX_ID, mutate, "reviewed stage-buffer ceiling is 8 slots")

    def test_a_slot_that_disagrees_with_its_declaration_is_refused(self):
        # The declaration states the read and the footprint; a view shorter
        # than that footprint cannot carry it, and the comparator refuses the
        # suite before a rail would have to.
        self.refused(
            SIX_ID,
            lambda case: case["stage_buffers"][5].update(
                length=8, initial_hex=PAYLOAD_64[:16]
            ),
            "past the view",
        )

    def test_the_marker_gates_the_capture(self):
        reporting = synthetic_report(self.suite, self.digest)
        for case_id, frame in ((SIX_ID, SIX_FRAME), (FOUR_ID, FOUR_FRAME)):
            reporting["results"].append({
                "id": case_id,
                "completion": "CompletedVisible",
                "writebacks": [
                    {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                     "offset": 0, "bytes_hex": frame},
                ],
                "allocations": [
                    {"allocation": ATTACHMENT[0], "bytes_hex": frame},
                ],
            })
        compare.validate_capture(self.suite, self.digest, reporting, "vulkan")

        # A rail the markers do not name may not report the cases: the two
        # cases are the Vulkan rails' arrangement, whose six slots fill one
        # set's descriptor floor rather than a reviewed module's fixed pairs.
        native = synthetic_report(self.suite, self.digest, backend="native-metal")
        native["results"].append({
            "id": SIX_ID,
            "completion": "CompletedVisible",
            "writebacks": [
                {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                 "offset": 0, "bytes_hex": SIX_FRAME},
            ],
            "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": SIX_FRAME}],
        })
        with self.assertRaisesRegex(compare.CaptureError,
                                    "not a rail this render case runs on"):
            compare.validate_capture(self.suite, self.digest, native, "native-metal")

    def test_a_dropped_slot_lands_the_wrong_frame(self):
        # The falsifiability of both cases: a rail that binds four of the six
        # declarations lands 128/255 (0x80), not 224/255 (0xe0), and the
        # capture that reports it is refused.
        reporting = synthetic_report(self.suite, self.digest)
        dropped = "80808080" * 4
        reporting["results"].append({
            "id": SIX_ID,
            "completion": "CompletedVisible",
            "writebacks": [
                {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                 "offset": 0, "bytes_hex": dropped},
            ],
            "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": dropped}],
        })
        with self.assertRaises(compare.CaptureError):
            compare.validate_capture(self.suite, self.digest, reporting, "vulkan")


if __name__ == "__main__":
    unittest.main()
