"""The kept-frame landing entry (`research/docs/23` §115 之后的增量，E-TX14/R4b).

These are comparator and schema checks, not GPU execution evidence. What they
pin is the *deferred* sibling of v40's landing view: `kept_frame_landing_quad_2x2`
opens its attachment from the caller's bytes (`fefefefe…`), draws the left
column, and **keeps** the frame in the provider's own image (a `"resident"`
store publishes nothing at all). A later landing *entry* — the `kept_frame_landing`
section's two identities — delivers that frame into the owner window the
declaring pass declares as its `borrowed_no_copy` binding, whose bytes start as
`11223344…`.

The readings the suite's own shape makes falsifiable:

* the owner window holds the frame the keeping pass left in the provider's
  image — the fragment output on the drawn texels and the caller's bytes on the
  rest — so a rail that delivered the window's own pre-pass bytes, or never ran
  the entry at all, lands `11223344` where `4080c0ff`/`fefefefe` belongs;
* the case carries no `expected_hex` and the attachment publishes no writeback:
  the pass really is the resident arm rather than a stored one wearing its name;
* the frame's identity has to be the resident attachment's own, the window has to
  be a *second* declaration of the declaring pass (read-only, `borrowed_no_copy`,
  the attachment's own extent), and the kept asset leaves the device exactly once
  for the delivery (`copy_out` counts the entry's readback), so "the frame
  stayed" and "the frame was delivered" are different observations.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare


CONFORMANCE = Path(__file__).resolve().parent
V41_PATH = CONFORMANCE / "suite-v41.json"

DECLARING_ID = "render_declaring_landing_view"
CASE_ID = "kept_frame_landing_quad_2x2"

ATTACHMENT = (900, 910, 0, 16)
SINK = (920, 930, 4, 4)
WINDOW = (940, 950, 16384, 16)
WINDOW_ALLOCATION_SIZE = 32768

CALLER_HEX = "fefefefefefefefefefefefefefefefe"
WINDOW_HEX = "11223344112233441122334411223344"
FRAME_HEX = "4080c0fffefefefe4080c0fffefefefe"

DECLARING_RAILS = ("vulkan", "native-metal", "native-metal-provider")
RENDER_RAILS = ("vulkan",)


class KeptFrameLandingTests(unittest.TestCase):
    def setUp(self):
        self.raw = V41_PATH.read_bytes()
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

    def test_v41_pins_the_suite_and_its_sources(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v41")
        self.assertEqual(self.suite["guard_byte"], 168)
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
        # The window's own bytes are deliberately *not* the frame: a rail that
        # never ran the landing entry would leave them in the window the capture
        # reports, which is the reading the case's expectation has to differ
        # from.
        self.assertEqual(buffers[2]["initial_hex"], WINDOW_HEX)
        self.assertNotEqual(buffers[2]["initial_hex"], CALLER_HEX)
        # The reviewed witness kernel writes the witness word last, so the
        # declaring pass's own writeback is the window's first word — the copy
        # landing the render case's frame is measured against.
        self.assertEqual(declaring["expected_writebacks"],
                         [{"allocation": SINK[0], "view": SINK[1], "offset": SINK[2],
                           "bytes_hex": WINDOW_HEX[:8]}])

    def test_the_render_case_keeps_its_frame_instead_of_storing_it(self):
        case = self.case()
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual(case["capture_rails"], list(RENDER_RAILS))
        attachment = case["attachment"]
        self.assertEqual((attachment["allocation"], attachment["view"]), ATTACHMENT[:2])
        self.assertEqual(attachment["load"], "load")
        self.assertEqual(attachment["initial_hex"], CALLER_HEX)
        # The arm's own statement: the pass publishes nothing, so neither the
        # attachment nor the case carries a writeback expectation, and the owner
        # window the later entry fills is the whole observation.
        self.assertEqual(attachment["store"], "resident")
        self.assertNotIn("expected_hex", case)
        self.assertNotIn("landing_view", attachment)
        self.assertEqual(case["kept_frame_landing"],
                         {"frame": {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1]},
                          "landing": {"allocation": WINDOW[0], "view": WINDOW[1]}})
        self.assertEqual(case["expected_landing_hex"], FRAME_HEX)
        # The frame is the one the keeping pass left: the fragment output on the
        # drawn column and the caller's bytes on the other — not the uniform
        # colour a rail that ignored the load would land, and not the window's
        # own pre-pass bytes.
        self.assertEqual(case["expected_landing_hex"],
                         "4080c0ff" + CALLER_HEX[:8] + "4080c0ff" + CALLER_HEX[:8])
        self.assertNotEqual(case["expected_landing_hex"], WINDOW_HEX)

    def test_the_plan_carries_the_window_and_no_attachment_observation(self):
        expectation = self.plan()
        # The attachment publishes no writeback and no allocation image: the
        # frame's home is the provider's image until the entry moves it.
        self.assertEqual(expectation.writes, [])
        self.assertEqual(expectation.allocations, {})
        self.assertEqual(expectation.attachment, [])
        self.assertEqual(expectation.landing,
                         (WINDOW[0], WINDOW[1], bytes.fromhex(FRAME_HEX)))
        # The owner window is touched by the declaring pass but never copied in:
        # its bytes are the owner's own pages (`research/docs/23` §90, R9i).
        self.assertEqual(expectation.touched, {ATTACHMENT[0], SINK[0]})
        # The frame's own allocation *does* leave the device exactly once — the
        # landing entry reads the provider's kept image back to deliver it — so
        # the copy-out contract counts it beside the declaring write, while the
        # writeback list above stays empty. That pair is the reading which
        # separates "the pass published the frame" from "the entry delivered it".
        self.assertEqual(expectation.written, {ATTACHMENT[0], SINK[0]})
        self.assertEqual(expectation.rails, set(RENDER_RAILS))

    def test_a_landing_expectation_without_the_kept_frame_section_is_refused(self):
        def mutate(case, suite):
            case.pop("kept_frame_landing")

        self.refused(mutate, "a resident store names the kept_frame_landing entry")

    def test_a_landing_expectation_without_a_landing_arm_is_refused(self):
        def mutate(case, suite):
            # Everything a *stored* case needs — a case-level expectation and
            # the attachment that carries it — plus the window expectation, but
            # no arm that lands the frame anywhere: the case would be claiming
            # an observation no pass performs.
            case["attachment"]["store"] = "store"
            case["expected_hex"] = FRAME_HEX

        self.refused(mutate, "expected_landing_hex needs a landing_view store")

    def test_a_kept_frame_section_without_a_resident_store_is_refused(self):
        def mutate(case, suite):
            # The mutation keeps everything a *stored* case needs, so the only
            # fact left wrong is the section, which then names an entry no pass
            # would ever keep a frame for.
            case["attachment"]["store"] = "store"
            case["expected_hex"] = FRAME_HEX
            case.pop("expected_landing_hex")

        self.refused(mutate, "kept_frame_landing needs a resident store")

    def test_a_stored_expectation_beside_the_resident_store_is_refused(self):
        def mutate(case, suite):
            case["expected_hex"] = FRAME_HEX

        self.refused(mutate, "a resident store publishes no writeback, so the case "
                             "carries no expected_hex")

    def test_a_texel_rule_beside_the_resident_store_is_refused(self):
        def mutate(case, suite):
            case["expected_rule"] = "xy_u16le_v1"

        self.refused(mutate, "a resident store states its frame as expected_landing_hex")

    def test_a_kept_frame_that_is_not_the_attachment_is_refused(self):
        def mutate(case, suite):
            case["kept_frame_landing"]["frame"] = {"allocation": SINK[0], "view": SINK[1]}

        self.refused(mutate, "the kept frame has to be the resident attachment's own identity")

    def test_a_window_the_declaring_pass_does_not_declare_is_refused(self):
        def mutate(case, suite):
            case["kept_frame_landing"]["landing"] = {"allocation": 941, "view": 951}

        self.refused(mutate, "the declaring case has to declare exactly the landing view")

    def test_a_window_that_is_a_copy_arm_is_refused(self):
        def mutate(case, suite):
            # The same window bytes through the *staged* arm: the provider would
            # hold the owner's copy, so the frame would land in memory no
            # owner's ledger protects. The lease marker stays consistent, so the
            # refusal is this arm's own rule rather than the marker's.
            suite["cases"][0]["buffers"][2]["storage_mode"] = "staged_lease"

        self.refused(mutate, "borrowed_no_copy window")

    def test_a_window_of_another_extent_is_refused(self):
        def mutate(case, suite):
            window = suite["cases"][0]["buffers"][2]
            window["length"] = 12
            window["initial_hex"] = WINDOW_HEX[:24]

        self.refused(mutate, "the landing view has to be the attachment's own extent")

    def test_an_expectation_that_is_the_window_own_bytes_is_refused(self):
        def mutate(case, suite):
            # With no writeback to compare the frame against, this is the one
            # mutation that would let a rail which never ran the landing entry
            # pass: the window's pre-pass bytes are exactly what a provider that
            # delivered nothing would report.
            case["expected_landing_hex"] = WINDOW_HEX

        self.refused(mutate, "the window already holds those bytes before the entry runs")

    def test_the_marker_keeps_the_native_rails_out_of_the_arm(self):
        # The native rails keep no frame and have no route that writes an
        # owner's window (`research/docs/23` §115 之后的增量): the case names the
        # Vulkan trace rail alone, and every rail it does not name has to leave
        # the case out of its capture rather than report a landing it cannot
        # perform.
        case = self.case()
        self.assertEqual(case["capture_rails"], ["vulkan"])
        for rail in ("native-metal", "native-metal-provider",
                     "native-metal-provider-objects", "vulkan-objects"):
            self.assertNotIn(rail, case["capture_rails"])


if __name__ == "__main__":
    unittest.main()
