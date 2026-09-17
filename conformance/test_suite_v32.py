"""The object rail's half of the stage-buffer face (`research/docs/23` §3.3, v87).

`suite-v32.json` carries the two stage-buffer render cases v31 pinned to the
Vulkan trace rail — the translated pair whose fragment stage writes a
`[[buffer(1)]]` sink, and the reviewed pair whose tint is imported through a
no-copy lease — with the object rails named beside the trace rail, plus one new
fixture: the reviewed pair again, whose tint arrives through a *staged* lease
(`research/docs/23` §90, R9i).

These are comparator and schema checks, not GPU execution evidence: they pin
the declarations all three rails (direct, objects, objects-async) execute, the
landings the capture owes, the source arm each slot ran with, and the refusals
every disagreement has to hit — on Linux, without a GPU or a compiler.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V32_PATH = CONFORMANCE / "suite-v32.json"
V31_PATH = CONFORMANCE / "suite-v31.json"

DECLARING_SINK_ID = "render_declaring_stage_buffer_sink"
DECLARING_LEASE_ID = "render_declaring_stage_buffer_lease"
WRITE_ID = "stage_buffer_write_affine_2x2"
BORROWED_ID = "stage_buffer_borrowed_tint_2x2"
STAGED_ID = "stage_buffer_staged_tint_2x2"

# The attachment the three render cases draw into: the declaring passes' own
# whole-allocation view of a 2x2 `rgba8_unorm` attachment.
ATTACHMENT = (900, 910, 0, 16)
# The writable stage buffer's landing: the pool view the declaring pass
# declares, at its own offset inside the allocation (`research/docs/23` §3.3,
# v86).
SINK = (940, 950, 4, 16)
# The fragment stages' payload, in every case: the reviewed tint the write stage
# copies into its sink, and the tint the two lease arms carry.
TINT = "8180803e8180003fc1c0403f0000803f"
# The frame the affine triangle leaves: one covered texel carrying the payload
# through the format's quantisation, three keeping the clear colour.
FRAME = "4080c0ff" + "fefefefe" * 3
# The two rails every case of this suite names: the Vulkan trace rail and the
# object rails, whose sync and async runs report one backend name.
RAILS = ["vulkan", "vulkan-objects"]


def committed_suite():
    return json.loads(V32_PATH.read_text(encoding="utf-8"))


def previous_suite():
    return json.loads(V31_PATH.read_text(encoding="utf-8"))


def suite_digest(suite):
    return hashlib.sha256(json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()


def case_by_id(suite, case_id):
    return next(case for case in suite["cases"] if case["id"] == case_id)


def render_case_by_id(suite, case_id):
    return next(case for case in suite["render_cases"] if case["id"] == case_id)


def declaring_images(suite, case):
    """The declaring case's allocation images, exactly as the plan builds them."""
    images = {}
    for buffer in case["buffers"]:
        data = images.setdefault(buffer["allocation"],
                                 bytearray([suite["guard_byte"]]) * buffer["allocation_size"])
        initial = bytes.fromhex(buffer["initial_hex"])
        data[buffer["offset"]:buffer["offset"] + len(initial)] = initial
    for write in case["expected_writebacks"]:
        payload = bytes.fromhex(write["bytes_hex"])
        data = images[write["allocation"]]
        data[write["offset"]:write["offset"] + len(payload)] = payload
    return {allocation: bytes(data) for allocation, data in images.items()}


def render_result(suite, case, counts=True):
    """The capture result one stage-buffer render case owes.

    The observation is the attachment's own landing plus — for a case whose
    slots include a writable one — the stage buffer's writeback, in the order
    the rails report them. The counters are the declaring pass's own touched
    allocations as inputs and its written ones as outputs: a stage buffer's
    bytes travel through the render input channel and move no counter, while
    the landing it receives is one more written allocation
    (`research/docs/23` §3.3, v86).
    """
    declaring = case_by_id(suite, case["declaring_case"])
    images = declaring_images(suite, declaring)
    landing_allocations = {ATTACHMENT[0]}
    frame_image = bytearray(images[ATTACHMENT[0]])
    frame_image[ATTACHMENT[2]:ATTACHMENT[2] + 16] = bytes.fromhex(FRAME)
    writebacks = [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                   "offset": ATTACHMENT[2], "bytes_hex": FRAME}]
    allocations = {ATTACHMENT[0]: bytes(frame_image)}
    for entry in case["stage_buffers"]:
        if entry["access"] == "read":
            continue
        payload = bytes.fromhex(entry["expected_hex"])
        image = bytearray(images[entry["allocation"]])
        image[entry["offset"]:entry["offset"] + len(payload)] = payload
        allocations[entry["allocation"]] = bytes(image)
        landing_allocations.add(entry["allocation"])
        writebacks.append({"allocation": entry["allocation"], "view": entry["view"],
                           "offset": entry["offset"], "bytes_hex": entry["expected_hex"]})
    result = {
        "id": case["id"],
        "completion": "CompletedVisible",
        "writebacks": writebacks,
        "allocations": [{"allocation": allocation, "bytes_hex": image.hex()}
                        for allocation, image in sorted(allocations.items())],
    }
    if counts:
        result["copy_in"] = len(images)
        result["copy_out"] = len(landing_allocations | {920})
    # A case reports the source arm of each slot exactly when the suite
    # declares a lease arm (`research/docs/23` §90, R9i): the owned-only case
    # keeps the result shape every pre-R9i fixture has.
    if any(entry.get("storage_mode", "owned_bytes") != "owned_bytes"
           for entry in case["stage_buffers"]):
        result["storage_modes"] = [
            {"view": entry["view"], "mode": entry.get("storage_mode", "owned_bytes")}
            for entry in case["stage_buffers"]
        ]
    return result


def capture_for(suite, digest=None, backend="vulkan"):
    """A synthetic capture of the whole stage-buffer suite."""
    digest = digest or suite_digest(suite)
    report = synthetic_report(suite, digest, backend)
    report["results"].extend(render_result(suite, case) for case in suite["render_cases"])
    return report


def check(suite, report, backend=None):
    compare.validate_capture(suite, suite_digest(suite), report, backend)


class StageBufferSuiteTests(unittest.TestCase):
    """The committed suite's declarations, held to the contract's own rules."""

    def test_the_suite_carries_the_two_declaring_passes_and_three_render_cases(self):
        suite = committed_suite()
        self.assertEqual(suite["suite"], "compute-buffer-v32")
        self.assertEqual([case["id"] for case in suite["cases"]],
                         [DECLARING_SINK_ID, DECLARING_LEASE_ID])
        self.assertEqual([case["id"] for case in suite["render_cases"]],
                         [WRITE_ID, BORROWED_ID, STAGED_ID])
        for case in suite["render_cases"]:
            self.assertEqual(case["capture_rails"], RAILS)

    def test_the_two_carried_cases_differ_from_v31_only_in_their_marker(self):
        # The suite restates v31's two render cases rather than rewriting them:
        # the declarations are byte-identical field for field, and the marker is
        # the one field the object rail's entry point moves
        # (`research/docs/23` §3.3, v87).
        suite, previous = committed_suite(), previous_suite()
        for case_id in (WRITE_ID, BORROWED_ID):
            carried = copy.deepcopy(render_case_by_id(suite, case_id))
            original = copy.deepcopy(render_case_by_id(previous, case_id))
            self.assertEqual(carried.pop("capture_rails"), RAILS)
            if case_id == WRITE_ID:
                # The translated case stays on the Vulkan trace rail.
                self.assertEqual(original.pop("capture_rails"), ["vulkan"])
            else:
                # The reviewed case names the native window the Apple device
                # readings opened (`research/docs/23` §3.3, v83).
                self.assertEqual(original.pop("capture_rails"),
                                 ["vulkan", "native-metal", "native-metal-provider"])
            self.assertEqual(carried, original)

    def test_the_staged_case_is_the_borrowed_case_with_the_staged_arm(self):
        suite = committed_suite()
        borrowed = copy.deepcopy(render_case_by_id(suite, BORROWED_ID))
        staged = copy.deepcopy(render_case_by_id(suite, STAGED_ID))
        for case in (borrowed, staged):
            case.pop("id")
            for slot in case["stage_buffers"]:
                for field in ("allocation", "view"):
                    slot.pop(field)
        borrowed_tint = borrowed["stage_buffers"][1]
        staged_tint = staged["stage_buffers"][1]
        self.assertEqual(borrowed_tint.pop("storage_mode"), "borrowed_no_copy")
        self.assertEqual(staged_tint.pop("storage_mode"), "staged_lease")
        self.assertEqual(borrowed_tint, staged_tint)
        self.assertEqual(staged["stage_buffers"][0], borrowed["stage_buffers"][0])
        self.assertEqual(staged["attachment"], borrowed["attachment"])
        self.assertEqual(staged["expected_hex"], borrowed["expected_hex"])

    def test_the_staged_case_binds_one_lease_view_of_its_own_allocation(self):
        suite = committed_suite()
        case = render_case_by_id(suite, STAGED_ID)
        tint = next(entry for entry in case["stage_buffers"] if entry["stage"] == "fragment")
        self.assertEqual(tint["storage_mode"], "staged_lease")
        # One Apple silicon host-import page (16 KiB), as in v31.
        self.assertEqual(tint["allocation_size"], 16384)
        self.assertEqual(tint["initial_hex"], TINT)
        # The owner window is the view's own range inside the registration, and
        # a lease-armed case declares one view per allocation (`docs/23` §90).
        self.assertEqual(tint["offset"], 0)
        self.assertEqual(tint["length"], 16)
        views = [entry["view"] for entry in case["stage_buffers"]]
        allocations = [entry["allocation"] for entry in case["stage_buffers"]]
        self.assertEqual(len(set(views)), len(views))
        self.assertEqual(len(set(allocations)), len(allocations))
        positions = next(entry for entry in case["stage_buffers"] if entry["stage"] == "vertex")
        self.assertNotIn("allocation_size", positions)
        self.assertNotIn("storage_mode", positions)


class StageBufferCaptureTests(unittest.TestCase):
    """The capture surface the three cases owe on every rail they name."""

    def setUp(self):
        self.suite = committed_suite()

    def result(self, report, case_id):
        return next(result for result in report["results"] if result["id"] == case_id)

    def test_the_trace_rail_capture_passes(self):
        check(self.suite, capture_for(self.suite), "vulkan")

    def test_the_object_rails_capture_passes(self):
        # The v87 increment is exactly this: an objects capture has to report
        # the three render cases, and the comparator validates their landings
        # and source arms exactly as it validates the trace rail's.
        check(self.suite, capture_for(self.suite, backend="vulkan-objects"),
              "vulkan-objects")

    def test_an_object_capture_that_omits_a_render_case_is_refused(self):
        report = capture_for(self.suite, backend="vulkan-objects")
        report["results"] = [result for result in report["results"]
                             if result["id"] != STAGED_ID]
        with self.assertRaisesRegex(compare.CaptureError, "missing cases"):
            check(self.suite, report, "vulkan-objects")

    def test_the_staged_case_reports_its_arm(self):
        report = capture_for(self.suite, backend="vulkan-objects")
        result = self.result(report, STAGED_ID)
        self.assertEqual([entry["mode"] for entry in result["storage_modes"]],
                         ["owned_bytes", "staged_lease"])

    def test_a_staged_case_reporting_the_wrong_arm_is_refused(self):
        report = capture_for(self.suite, backend="vulkan-objects")
        result = self.result(report, STAGED_ID)
        result["storage_modes"][1]["mode"] = "borrowed_no_copy"
        with self.assertRaisesRegex(compare.CaptureError, "reported source arms"):
            check(self.suite, report, "vulkan-objects")

    def test_the_borrowed_case_reports_its_arm(self):
        report = capture_for(self.suite)
        result = self.result(report, BORROWED_ID)
        self.assertEqual([entry["mode"] for entry in result["storage_modes"]],
                         ["owned_bytes", "borrowed_no_copy"])

    def test_the_writable_slot_still_lands_on_the_object_rails(self):
        report = capture_for(self.suite, backend="vulkan-objects")
        result = self.result(report, WRITE_ID)
        self.assertNotIn("storage_modes", result)
        self.assertEqual(result["writebacks"][1],
                         {"allocation": SINK[0], "view": SINK[1], "offset": SINK[2],
                          "bytes_hex": TINT})
        # One more written allocation: the declaring pass's own 920, the
        # attachment, and the sink (`research/docs/23` §3.3, v86).
        self.assertEqual(result["copy_out"], 3)

    def test_a_silent_skip_is_not_a_pass(self):
        # A rail the marker names owes every case: dropping the render results
        # entirely — the shape the pre-v87 object rail produced as a named skip
        # — is refused instead of read as "nothing to compare".
        report = capture_for(self.suite, backend="vulkan-objects")
        report["results"] = [result for result in report["results"]
                             if result["id"] in (DECLARING_SINK_ID, DECLARING_LEASE_ID)]
        with self.assertRaisesRegex(compare.CaptureError, "missing cases"):
            check(self.suite, report, "vulkan-objects")

    def test_a_translated_case_cannot_name_a_native_rail(self):
        # The translated arm pins AIR only the Vulkan rails translate, so a
        # native rail named there would claim an observation that rail cannot
        # report (`research/docs/23` §3.3, v83/v87).
        case = render_case_by_id(self.suite, WRITE_ID)
        case["capture_rails"] = RAILS + ["native-metal-provider"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "capture_rails has to stay inside that list"):
            compare._render_plan(compare._suite_plan(self.suite), self.suite)


class StageBufferRefusalTests(unittest.TestCase):
    """Every declaration the three rails refuse has to be refused here first."""

    def setUp(self):
        self.suite = committed_suite()

    def validate(self):
        compare._render_plan(compare._suite_plan(self.suite), self.suite)

    def reject(self, message):
        with self.assertRaisesRegex(compare.CaptureError, message):
            self.validate()

    def render_case(self, case_id):
        return render_case_by_id(self.suite, case_id)

    def slot(self, case_id, stage="fragment", index=0):
        case = self.render_case(case_id)
        return next(entry for entry in case["stage_buffers"]
                    if entry["stage"] == stage and entry["index"] == index)

    def test_a_marker_naming_no_rail_is_refused(self):
        # The marker has to name at least one rail: an empty list is not a
        # capture any backend can answer (`research/docs/23` §3.3, v87).
        self.render_case(STAGED_ID)["capture_rails"] = []
        self.reject("capture_rails")

    def test_a_staged_arm_without_its_owner_window_is_refused(self):
        del self.slot(STAGED_ID)["allocation_size"]
        self.reject("states the allocation size its owner window covers")

    def test_an_owned_slot_stating_an_owner_window_is_refused(self):
        self.slot(STAGED_ID, "vertex")["allocation_size"] = 4096
        self.reject("an owned stage buffer maps no owner window")

    def test_an_unknown_storage_mode_is_refused(self):
        self.slot(STAGED_ID)["storage_mode"] = "guest_window"
        self.reject("unknown stage-buffer storage mode")

    def test_a_staged_view_outside_its_registration_is_refused(self):
        self.slot(STAGED_ID)["allocation_size"] = 8
        self.reject("the view lies outside its owner's registration")

    def test_a_read_only_slot_stating_an_expectation_is_refused(self):
        self.slot(STAGED_ID)["expected_hex"] = "00" * 16
        self.reject("a read-only stage buffer carries no expectation")

    def test_the_three_cases_declare_one_view_per_allocation(self):
        case = self.render_case(STAGED_ID)
        duplicate = copy.deepcopy(self.slot(STAGED_ID))
        duplicate.update({"stage": "fragment", "index": 1, "view": 1440})
        case["stage_buffers"].append(duplicate)
        self.reject("declares one view per allocation")


if __name__ == "__main__":
    unittest.main()
