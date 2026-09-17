"""The stage-buffer face of the fixture format (`research/docs/23` §3.3, v83-v86).

`suite-v31.json` is the first fixture whose render cases declare what their two
stages read and write directly: one translated pair whose vertex stage reads its
positions through an *affine* footprint and whose fragment stage writes a
`[[buffer(1)]]` sink, and one reviewed pair whose fragment tint is imported
through a *borrowed* lease (`research/docs/23` §90, R9i).

These are comparator and schema checks, not GPU execution evidence: they pin the
declarations the two rails execute, the two landings the capture owes (the
attachment and the writable slot's own writeback), the copy counters those
landings move, and the refusals every disagreement has to hit — on Linux,
without a GPU or a compiler.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V31_PATH = CONFORMANCE / "suite-v31.json"

DECLARING_SINK_ID = "render_declaring_stage_buffer_sink"
DECLARING_LEASE_ID = "render_declaring_stage_buffer_lease"
WRITE_ID = "stage_buffer_write_affine_2x2"
LEASE_ID = "stage_buffer_borrowed_tint_2x2"

# The attachment the two render cases draw into: the declaring passes' own
# whole-allocation view of a 2x2 `rgba8_unorm` attachment.
ATTACHMENT = (900, 910, 0, 16)
# The writable stage buffer's landing: the pool view the declaring pass
# declares, at its own offset inside the allocation (`research/docs/23` §3.3,
# v86).
SINK = (940, 950, 4, 16)
# The fragment stages' payload, in both cases: the reviewed tint the write stage
# copies into its sink and the borrowed lease carries.
TINT = "8180803e8180003fc1c0403f0000803f"
# The frame the affine triangle leaves: one covered texel carrying the payload
# through the format's quantisation, three keeping the clear colour.
FRAME = "4080c0ff" + "fefefefe" * 3


def committed_suite():
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
    """A synthetic capture of the whole stage-buffer suite.

    The two declaring passes keep the shared synthetic builder's results (they
    are ordinary compute cases); the two render cases carry the stage-buffer
    landings this file is about.
    """
    digest = digest or suite_digest(suite)
    report = synthetic_report(suite, digest, backend)
    report["results"].extend(render_result(suite, case) for case in suite["render_cases"])
    return report


def check(suite, report, backend=None):
    compare.validate_capture(suite, suite_digest(suite), report, backend)


class StageBufferDeclarationTests(unittest.TestCase):
    """The committed suite's declarations, held to the contract's own rules."""

    def test_the_suite_is_the_two_declaring_passes_and_their_render_cases(self):
        suite = committed_suite()
        self.assertEqual(suite["suite"], "compute-buffer-v31")
        self.assertEqual([case["id"] for case in suite["cases"]],
                         [DECLARING_SINK_ID, DECLARING_LEASE_ID])
        self.assertEqual([case["id"] for case in suite["render_cases"]], [WRITE_ID, LEASE_ID])
        for case in suite["render_cases"]:
            self.assertEqual(case["capture_rails"], ["vulkan"])

    def test_the_translated_case_pins_its_two_air_modules_and_no_msl_sibling(self):
        case = render_case_by_id(committed_suite(), WRITE_ID)
        self.assertNotIn("metal", case)
        translated = case["translated_stages"]
        self.assertEqual(case["vertex_entry"], "render_vertex_positions")
        self.assertEqual(case["fragment_entry"], "render_stage_buffer_write_rgba8")
        for kind in ("vertex", "fragment"):
            source = V31_PATH.parent / translated[kind]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             translated[kind]["sha256"], kind)

    def test_the_reviewed_case_pins_the_reviewed_stage_buffer_module(self):
        case = render_case_by_id(committed_suite(), LEASE_ID)
        self.assertNotIn("translated_stages", case)
        module = V31_PATH.parent / case["metal"]["path"]
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                         case["metal"]["sha256"])
        self.assertEqual((case["vertex_entry"], case["fragment_entry"]),
                         ("render_stage_buffer_vertex", "render_stage_buffer_tint"))

    def test_the_affine_declaration_is_the_draws_own_reach(self):
        case = render_case_by_id(committed_suite(), WRITE_ID)
        vertex = next(entry for entry in case["stage_buffers"] if entry["stage"] == "vertex")
        accesses = vertex["footprint"]["affine"]["accesses"]
        self.assertEqual([(access["base_offset"], access["access_size"],
                           [(term["axis"], term["stride"]) for term in access["terms"]])
                          for access in accesses],
                         [(0, 4, [(0, 8)]), (4, 4, [(0, 8)])])
        # Three vertices reach 4 + 4 + 2 * 8 = 24 bytes, which is exactly the
        # view the case binds: the proof is the draw's own count, not a guess.
        self.assertEqual(vertex["length"], 24)
        self.assertEqual(case["vertices"], 3)

    def test_the_writable_slot_lands_in_the_declaring_passes_own_pool_view(self):
        suite = committed_suite()
        case = render_case_by_id(suite, WRITE_ID)
        slot = next(entry for entry in case["stage_buffers"] if entry["access"] == "write")
        declaring = case_by_id(suite, case["declaring_case"])
        pool = next(buffer for buffer in declaring["buffers"]
                    if buffer["allocation"] == slot["allocation"]
                    and buffer["view"] == slot["view"])
        self.assertEqual(pool["access"], "read")
        self.assertEqual((pool["offset"], pool["length"]), (slot["offset"], slot["length"]))
        self.assertEqual(pool["initial_hex"], slot["initial_hex"])
        self.assertEqual(slot["expected_hex"], TINT)

    def test_the_borrowed_lease_carries_the_owners_registration(self):
        case = render_case_by_id(committed_suite(), LEASE_ID)
        tint = next(entry for entry in case["stage_buffers"] if entry["stage"] == "fragment")
        self.assertEqual(tint["storage_mode"], "borrowed_no_copy")
        self.assertEqual(tint["allocation_size"], 4096)
        self.assertEqual(tint["initial_hex"], TINT)
        # The owned slot states no owner window, so the two arms cannot be
        # confused by a field either of them leaves out.
        positions = next(entry for entry in case["stage_buffers"] if entry["stage"] == "vertex")
        self.assertNotIn("allocation_size", positions)
        self.assertNotIn("storage_mode", positions)


class StageBufferCaptureTests(unittest.TestCase):
    """The capture surface the two stage-buffer cases owe and refuse."""

    def setUp(self):
        self.suite = committed_suite()
        self.report = capture_for(self.suite)

    def validate(self, report=None, backend=None):
        check(self.suite, report or self.report, backend)

    def reject(self, message, report=None, backend=None):
        with self.assertRaisesRegex(compare.CaptureError, message):
            self.validate(report, backend)

    def result(self, report, case_id):
        return next(result for result in report["results"] if result["id"] == case_id)

    def test_the_synthetic_capture_passes(self):
        self.validate()

    def test_the_objects_rail_is_not_a_rail_these_cases_run_on(self):
        # This suite's marker was pinned while the object API bound no stage
        # buffers, so a capture of that rail owes the two declaring passes and
        # nothing else. The increment that gave the object rails their own
        # entry point carries the same two cases in `suite-v32.json` with the
        # object rails named (`research/docs/23` §3.3, v87); this suite keeps
        # its own statement, and a capture of it that reports a case the marker
        # does not name is still refused.
        report = capture_for(self.suite, backend="vulkan-objects")
        report["results"] = [result for result in report["results"]
                             if result["id"] in (DECLARING_SINK_ID, DECLARING_LEASE_ID)]
        self.validate(report, backend="vulkan-objects")
        report["results"].append(render_result(self.suite,
                                               render_case_by_id(self.suite, WRITE_ID)))
        self.reject("case stage_buffer_write_affine_2x2: vulkan-objects is not a rail this "
                    "render case runs on", report, backend="vulkan-objects")

    def test_a_missing_stage_buffer_landing_is_refused(self):
        report = copy.deepcopy(self.report)
        result = self.result(report, WRITE_ID)
        result["writebacks"] = result["writebacks"][:1]
        self.reject("writable set mismatch", report)

    def test_a_wrong_stage_buffer_landing_is_refused(self):
        report = copy.deepcopy(self.report)
        result = self.result(report, WRITE_ID)
        result["writebacks"][1]["bytes_hex"] = "ff" * 16
        self.reject("writeback allocation 940/view 950", report)

    def test_the_landing_adds_one_written_allocation(self):
        report = copy.deepcopy(self.report)
        self.result(report, WRITE_ID)["copy_out"] = 2
        self.reject("copy_out 2 does not match 3 written allocations", report)

    def test_the_stage_buffers_own_bytes_move_no_copy_counter(self):
        # The three declaring allocations are the submission's inputs; the
        # borrowed owner window is imported rather than copied in, exactly as a
        # compute case's borrowed view is (`research/docs/23` §90).
        report = copy.deepcopy(self.report)
        self.result(report, LEASE_ID)["copy_in"] = 3
        self.reject("copy_in 3 does not match 2 touched allocations", report)

    def test_a_borrowed_view_reports_its_arm(self):
        report = copy.deepcopy(self.report)
        result = self.result(report, LEASE_ID)
        result["storage_modes"][1]["mode"] = "owned_bytes"
        self.reject("reported source arms", report)

    def test_an_owned_only_case_must_not_report_an_arm(self):
        report = copy.deepcopy(self.report)
        self.result(report, WRITE_ID)["storage_modes"] = [
            {"view": entry["view"], "mode": "owned_bytes"}
            for entry in render_case_by_id(self.suite, WRITE_ID)["stage_buffers"]
        ]
        self.reject("declares no lease arm for this case", report)

    def test_a_lease_case_must_report_its_arms(self):
        report = copy.deepcopy(self.report)
        del self.result(report, LEASE_ID)["storage_modes"]
        self.reject("has to report the source arm", report)

    def test_the_reference_oracle_is_not_a_provider_so_it_reports_no_arms(self):
        # `native-metal` is the Swift reference oracle: it compiles MSL, not
        # AIR, and its rail publishes no stage-buffer capability, so the cases'
        # marker leaves it out and a capture that reported one anyway is
        # refused. (Its "not a provider, so it reports no arm" branch is the
        # defensive sibling of the compute path's rule and is unreachable while
        # the marker is pinned to the Vulkan trace rail.)
        report = capture_for(self.suite, backend="native-metal")
        del self.result(report, LEASE_ID)["storage_modes"]
        self.reject("native-metal is not a rail this render case runs on",
                    report, backend="native-metal")


class StageBufferRefusalTests(unittest.TestCase):
    """Every declaration the two rails refuse has to be refused here first."""

    def setUp(self):
        self.suite = committed_suite()

    def validate(self):
        # These are suite-shape refusals: the comparator has to refuse the
        # declaration itself, before any capture is compared, so the plan is
        # what this file exercises (`compare._render_plan`, as the other
        # suite-shape tests do).
        compare._render_plan(compare._suite_plan(self.suite), self.suite)

    def reject(self, message):
        with self.assertRaisesRegex(compare.CaptureError, message):
            self.validate()

    def slot(self, case_id, stage="fragment", index=0):
        case = render_case_by_id(self.suite, case_id)
        return next(entry for entry in case["stage_buffers"]
                    if entry["stage"] == stage and entry["index"] == index)

    def render_case(self, case_id):
        return render_case_by_id(self.suite, case_id)

    def test_a_footprint_past_its_own_view_is_refused(self):
        self.slot(WRITE_ID, "fragment", 0)["footprint"] = {"static": {"max_bytes": 32}}
        self.reject("declared footprint reaches 32 bytes")

    def test_an_affine_footprint_reaching_past_the_draw_is_refused(self):
        # The same two `float2` accesses with a doubled stride reach
        # 4 + 4 + 2 * 16 = 40 bytes, past the 24-byte view the case binds: the
        # arithmetic is the draw's own vertex count, so a declaration the draw
        # could not satisfy is refused here rather than by a rail.
        slot = self.slot(WRITE_ID, "vertex", 0)
        for access in slot["footprint"]["affine"]["accesses"]:
            access["terms"][0]["stride"] = 16
        self.reject("declared footprint reaches 40 bytes")

    def test_a_writable_slot_without_an_expectation_is_refused(self):
        del self.slot(WRITE_ID, "fragment", 1)["expected_hex"]
        self.reject("a writable stage buffer states the bytes its writeback lands")

    def test_a_read_only_slot_with_an_expectation_is_refused(self):
        self.slot(WRITE_ID, "fragment", 0)["expected_hex"] = "00" * 16
        self.reject("a read-only stage buffer carries no expectation")

    def test_a_writable_slot_the_declaring_case_does_not_declare_is_refused(self):
        slot = self.slot(WRITE_ID, "fragment", 1)
        slot["view"] = 951
        self.reject("the declaring case has to declare exactly that view")

    def test_a_writable_slot_disagreeing_with_the_declaring_case_is_refused(self):
        slot = self.slot(WRITE_ID, "fragment", 1)
        slot["initial_hex"] = "00" * 16
        self.reject("carries different bytes than the declaring case's declaration")

    def test_a_read_only_slot_reusing_a_declared_view_is_refused(self):
        self.slot(WRITE_ID, "fragment", 0)["view"] = 950
        self.reject("cannot be one the declaring case already declares")

    def test_a_lease_arm_without_its_owner_window_is_refused(self):
        del self.slot(LEASE_ID, "fragment", 0)["allocation_size"]
        self.reject("states the allocation size its owner window covers")

    def test_an_owned_slot_stating_an_owner_window_is_refused(self):
        self.slot(LEASE_ID, "vertex", 0)["allocation_size"] = 4096
        self.reject("an owned stage buffer maps no owner window")

    def test_an_unknown_storage_mode_is_refused(self):
        self.slot(LEASE_ID, "fragment", 0)["storage_mode"] = "shared_memory"
        self.reject("unknown stage-buffer storage mode")

    def test_a_second_view_of_one_leased_allocation_is_refused(self):
        case = self.render_case(LEASE_ID)
        case["stage_buffers"].append({
            "stage": "fragment", "index": 1, "access": "read",
            "footprint": {"static": {"max_bytes": 16}},
            "allocation": 1220, "view": 1240, "offset": 0, "length": 16,
            "allocation_size": 4096, "storage_mode": "borrowed_no_copy",
            "initial_hex": "00" * 16,
        })
        self.reject("declares one view per allocation")

    def test_a_non_canonical_slot_order_is_refused(self):
        case = self.render_case(WRITE_ID)
        case["stage_buffers"][0], case["stage_buffers"][2] = (
            case["stage_buffers"][2], case["stage_buffers"][0])
        self.reject("vertex bindings before fragment bindings")

    def test_a_translated_case_naming_another_rail_is_refused(self):
        self.render_case(WRITE_ID)["capture_rails"] = ["vulkan", "vulkan-objects"]
        self.reject("runs on the Vulkan trace rail alone")

    def test_a_case_carrying_both_source_spellings_is_refused(self):
        case = self.render_case(WRITE_ID)
        case["metal"] = {"path": "shaders/render_stage_buffer_2x2.metal",
                         "sha256": "63c4d5ba60c187437d749d957033778f479bc9ecb3f0a408eeabe6095dba0de6"}
        self.reject("exactly one of metal and translated_stages is required")

    def test_an_unknown_stage_or_access_is_refused(self):
        self.slot(WRITE_ID, "fragment", 0)["stage"] = "kernel"
        self.reject("unknown stage-buffer stage")


if __name__ == "__main__":
    unittest.main()
