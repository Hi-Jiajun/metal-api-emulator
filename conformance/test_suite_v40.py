"""The colour attachment's landing view (`research/docs/23` §115 之后的增量，E-TX13).

These are comparator and schema checks, not GPU execution evidence. What they
pin is the arm that separates "where the pass begins" from "where its frame
lands": `landing_view_quad_2x2` opens its attachment from the *caller's* bytes
(`fefefefe…`), and stores the frame into an owner window a **second** view
declaration names — the declaring pass's `borrowed_no_copy` binding, whose bytes
start as `11223344…`.

The readings the suite's own shape makes falsifiable:

* the frame the capture reports is the fragment output on the drawn texels and
  the caller's bytes on the rest, so a rail that read the pre-pass contents from
  the landing window would land `11223344` where `fefefefe` belongs;
* the owner window holds that same frame afterwards, which is what
  `expected_landing_hex` and the `landing` observation compare;
* the landing view cannot be the attachment's own identity, cannot be a copy arm
  and cannot be a different extent — each is refused by the comparator here
  before any rail runs.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare


CONFORMANCE = Path(__file__).resolve().parent
V40_PATH = CONFORMANCE / "suite-v40.json"

DECLARING_ID = "render_declaring_landing_view"
CASE_ID = "landing_view_quad_2x2"

ATTACHMENT = (900, 910, 0, 16)
SINK = (920, 930, 4, 4)
WINDOW = (940, 950, 16384, 16)
WINDOW_ALLOCATION_SIZE = 32768

CALLER_HEX = "fefefefefefefefefefefefefefefefe"
WINDOW_HEX = "11223344112233441122334411223344"
FRAME_HEX = "4080c0fffefefefe4080c0fffefefefe"

DECLARING_RAILS = ("vulkan", "native-metal", "native-metal-provider")
RENDER_RAILS = ("vulkan",)


class LandingViewTests(unittest.TestCase):
    def setUp(self):
        self.raw = V40_PATH.read_bytes()
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
        mutate(self.case(suite), suite)
        with self.assertRaisesRegex(compare.CaptureError, message):
            compare._render_plan(compare._suite_plan(suite), suite)

    def test_v40_pins_the_suite_and_its_sources(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v40")
        self.assertEqual(self.suite["guard_byte"], 167)
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], [CASE_ID])
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["entry"], "copy_word_with_witness")
        for kind in ("air", "metal"):
            source = CONFORMANCE / declaring[kind]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             declaring[kind]["sha256"], kind)
        case = self.case()
        source = CONFORMANCE / case["metal"]["path"]
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         case["metal"]["sha256"])

    def test_the_declaring_pass_declares_the_window_as_its_borrowed_view(self):
        declaring = self.suite["cases"][0]
        self.assertEqual(sorted(declaring["capture_rails"]), sorted(DECLARING_RAILS))
        buffers = {buffer["binding"]: buffer for buffer in declaring["buffers"]}
        self.assertEqual([buffers[binding]["access"] for binding in (0, 1, 2)],
                         ["read", "write", "read"])
        self.assertEqual((buffers[0]["allocation"], buffers[0]["view"],
                          buffers[0]["offset"], buffers[0]["length"]), ATTACHMENT)
        self.assertEqual(buffers[0]["initial_hex"], CALLER_HEX)
        self.assertEqual((buffers[1]["allocation"], buffers[1]["view"],
                          buffers[1]["offset"], buffers[1]["length"]), SINK)
        self.assertEqual((buffers[2]["allocation"], buffers[2]["view"],
                          buffers[2]["offset"], buffers[2]["length"]), WINDOW)
        self.assertEqual(buffers[2]["allocation_size"], WINDOW_ALLOCATION_SIZE)
        self.assertEqual(buffers[2]["storage_mode"], "borrowed_no_copy")
        # The window's own bytes are deliberately *not* the caller's bytes: a
        # rail that resolved the load from the landing view would show them in
        # the frame, and a rail that landed nothing would leave them in the
        # window the capture reports.
        self.assertEqual(buffers[2]["initial_hex"], WINDOW_HEX)
        self.assertNotEqual(buffers[2]["initial_hex"], CALLER_HEX)
        # The reviewed witness kernel writes the witness word last, so the
        # declaring pass's own writeback is the window's first word.
        self.assertEqual(declaring["expected_writebacks"],
                         [{"allocation": SINK[0], "view": SINK[1], "offset": SINK[2],
                           "bytes_hex": WINDOW_HEX[:8]}])

    def test_the_render_case_separates_the_load_from_the_landing(self):
        case = self.case()
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual(case["capture_rails"], list(RENDER_RAILS))
        attachment = case["attachment"]
        self.assertEqual((attachment["allocation"], attachment["view"]), ATTACHMENT[:2])
        self.assertEqual(attachment["load"], "load")
        self.assertEqual(attachment["store"], "landing_view")
        self.assertEqual(attachment["initial_hex"], CALLER_HEX)
        self.assertEqual(attachment["landing_view"],
                         {"allocation": WINDOW[0], "view": WINDOW[1]})
        self.assertEqual(case["expected_hex"], FRAME_HEX)
        self.assertEqual(case["expected_landing_hex"], FRAME_HEX)
        # The drawn texels are the fragment's and the rest the caller's bytes —
        # the frame the writeback channel carries and the window receives.
        self.assertEqual(case["expected_hex"],
                         "4080c0ff" + CALLER_HEX[:8] + "4080c0ff" + CALLER_HEX[:8])

    def test_the_plan_carries_the_landing_window_beside_the_frame(self):
        expectation = self.plan()
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(FRAME_HEX))])
        self.assertEqual(expectation.landing,
                         (WINDOW[0], WINDOW[1], bytes.fromhex(FRAME_HEX)))
        # The owner window is touched by the declaring pass but never copied in:
        # its bytes are the owner's own pages (`research/docs/23` §90, R9i).
        self.assertEqual(expectation.touched, {ATTACHMENT[0], SINK[0]})
        self.assertEqual(expectation.written, {ATTACHMENT[0], SINK[0]})
        self.assertEqual(expectation.rails, set(RENDER_RAILS))

    def test_a_landing_expectation_without_the_arm_is_refused(self):
        def mutate(case, suite):
            case["attachment"]["store"] = "store"

        self.refused(mutate, "expected_landing_hex needs a landing_view store")

    def test_a_landing_view_that_is_the_attachment_itself_is_refused(self):
        def mutate(case, suite):
            case["attachment"]["landing_view"] = {"allocation": ATTACHMENT[0],
                                                  "view": ATTACHMENT[1]}

        self.refused(mutate, "the landing view is the attachment's own identity")

    def test_a_landing_view_that_is_a_copy_arm_is_refused(self):
        def mutate(case, suite):
            # The same window bytes through the *staged* arm: the provider would
            # hold the owner's copy, so the frame would land in memory no
            # owner's ledger protects. The lease marker stays consistent, so the
            # refusal is this arm's own rule rather than the marker's.
            suite["cases"][0]["buffers"][2]["storage_mode"] = "staged_lease"

        self.refused(mutate, "borrowed_no_copy window")

    def test_a_landing_view_of_another_extent_is_refused(self):
        def mutate(case, suite):
            window = suite["cases"][0]["buffers"][2]
            window["length"] = 12
            window["initial_hex"] = WINDOW_HEX[:24]

        self.refused(mutate, "the landing view has to be the attachment's own extent")

    def test_a_landing_expectation_that_is_not_the_frame_is_refused(self):
        def mutate(case, suite):
            case["expected_landing_hex"] = WINDOW_HEX

        self.refused(mutate, "the window holds the frame the pass read back")

    def test_the_marker_keeps_the_native_rails_out_of_the_arm(self):
        # The native rails have no landing route that writes an owner's window
        # (`research/docs/23` §115 之后的增量): the case names the Vulkan trace
        # rail alone, and every rail it does not name has to leave the case out
        # of its capture rather than report a landing it cannot perform.
        case = self.case()
        self.assertEqual(case["capture_rails"], ["vulkan"])
        for rail in ("native-metal", "native-metal-provider",
                     "native-metal-provider-objects", "vulkan-objects"):
            self.assertNotIn(rail, case["capture_rails"])


if __name__ == "__main__":
    unittest.main()
