"""Synthetic resource-subset checks; these tests are not GPU execution evidence."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import compare
from test_compare import synthetic_report


SUITE_PATH = Path(__file__).with_name("suite-v7.json")


class ResourceSubsetTests(unittest.TestCase):
    def setUp(self):
        self.raw = SUITE_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()
        self.report = synthetic_report(self.suite, self.digest)

    def reject_suite(self, suite, message):
        with self.assertRaisesRegex(compare.CaptureError, message):
            compare.validate_capture(suite, self.digest, self.report)

    def test_all_backends_accept_later_resources_and_variable_slot_counts(self):
        self.assertEqual([len(case["dispatches"]) for case in self.suite["cases"]], [2, 4, 8])
        for case in self.suite["cases"]:
            self.assertGreater(len(case["buffers"]), len(case["programs"][0]["buffer_slots"]))
            self.assertEqual(len(case["programs"][1]["buffer_slots"]), 2)
        for backend in compare.ALLOCATION_OBSERVATIONS:
            with self.subTest(backend=backend):
                compare.validate_capture(
                    self.suite, self.digest,
                    synthetic_report(self.suite, self.digest, backend), backend,
                )

    def test_manifest_shader_hashes_match_actual_sources(self):
        for case in self.suite["cases"]:
            for program in case["programs"]:
                for kind in ("air", "metal"):
                    source = program[kind]
                    actual = hashlib.sha256((SUITE_PATH.parent / source["path"]).read_bytes()).hexdigest()
                    self.assertEqual(actual, source["sha256"], source["path"])

    def test_independent_cpu_reference_tracks_producer_consumer_resources(self):
        for case in self.suite["cases"]:
            pool = {buffer["view"]: bytearray.fromhex(buffer["initial_hex"])
                    for buffer in case["buffers"]}
            written = set()
            for dispatch in case["dispatches"]:
                self.assertEqual(dispatch["grid"], [5, 3, 2])
                program = case["programs"][dispatch["program"]]
                bindings = {slot["binding"]: pool[view]
                            for slot, view in zip(program["buffer_slots"], dispatch["bindings"])}
                written.update(view for slot, view in zip(program["buffer_slots"], dispatch["bindings"])
                               if slot["access"] != "read")
                if program["entry"] == "transform_3d":
                    source, scalar, target = (bindings[index] for index in (0, 2, 5))
                    bias = int.from_bytes(scalar, "little")
                    for offset in range(0, len(source), 4):
                        value = (int.from_bytes(source[offset:offset + 4], "little") + bias) & 0xffffffff
                        source[offset:offset + 4] = value.to_bytes(4, "little")
                        target[offset:offset + 4] = (value ^ 0xa5a55a5a).to_bytes(4, "little")
                elif program["entry"] == "copy_3d":
                    source, target = (bindings[index] for index in (4, 9))
                    before = bytes(source)
                    target[:] = source
                    self.assertEqual(bytes(source), before)
                elif program["entry"] == "remap_3d":
                    scalar, source, target = (bindings[index] for index in (1, 3, 7))
                    before = bytes(source)
                    bias = int.from_bytes(scalar, "little")
                    for offset in range(0, len(source), 4):
                        value = int.from_bytes(source[offset:offset + 4], "little")
                        output = ((value * 7 + bias) & 0xffffffff) ^ 0x3c3ca5a5
                        target[offset:offset + 4] = output.to_bytes(4, "little")
                    self.assertEqual(bytes(source), before)
                else:
                    self.fail(program["entry"])
            self.assertEqual({write["view"] for write in case["expected_writebacks"]}, written)
            for expected in case["expected_writebacks"]:
                self.assertEqual(pool[expected["view"]].hex(), expected["bytes_hex"], case["id"])
            for buffer in case["buffers"]:
                if buffer["view"] not in written:
                    self.assertEqual(pool[buffer["view"]].hex(), buffer["initial_hex"])

    def test_pool_order_does_not_change_explicit_subset_maps(self):
        suite = copy.deepcopy(self.suite)
        for case in suite["cases"]:
            case["buffers"].reverse()
        compare.validate_capture(suite, self.digest, self.report)

    def test_later_writeback_is_required_from_every_provider(self):
        for case_index, case in enumerate(self.suite["cases"]):
            first_views = set(case["dispatches"][0]["bindings"])
            later_views = {buffer["view"] for buffer in case["buffers"]} - first_views
            self.assertTrue(later_views)
            for view in later_views:
                for backend in compare.ALLOCATION_OBSERVATIONS:
                    with self.subTest(case=case["id"], view=view, backend=backend):
                        wrong = synthetic_report(self.suite, self.digest, backend)
                        wrong["results"][case_index]["writebacks"] = [
                            write for write in wrong["results"][case_index]["writebacks"]
                            if write["view"] != view
                        ]
                        with self.assertRaisesRegex(compare.CaptureError, "writable set mismatch"):
                            compare.validate_capture(self.suite, self.digest, wrong)

    def test_expected_writeback_cannot_omit_later_writable_resource(self):
        for case_index, view in ((0, 430), (1, 430), (1, 440), (2, 440)):
            suite = copy.deepcopy(self.suite)
            case = suite["cases"][case_index]
            case["expected_writebacks"] = [write for write in case["expected_writebacks"]
                                          if write["view"] != view]
            self.reject_suite(suite, "do not cover writable views")

    def test_later_access_is_selected_by_pipeline_not_initial_pool_label(self):
        suite = copy.deepcopy(self.suite)
        for case in suite["cases"]:
            for buffer in case["buffers"]:
                if buffer["view"] in (430, 440):
                    buffer["access"] = "read"
        compare.validate_capture(suite, self.digest, self.report)

    def test_missing_extra_unknown_and_same_pass_alias_maps_are_rejected(self):
        for mapping in ([400], [400, 430, 410], [400, 999], [400, 400], [], None):
            with self.subTest(mapping=mapping):
                suite = copy.deepcopy(self.suite)
                suite["cases"][0]["dispatches"][1]["bindings"] = mapping
                self.reject_suite(suite, "binding map must")

    def test_full_pool_is_not_an_implicit_selected_shader_layout(self):
        for dispatch_index in (0, 1):
            suite = copy.deepcopy(self.suite)
            case = suite["cases"][0]
            case["dispatches"][dispatch_index]["bindings"] = [buffer["view"] for buffer in case["buffers"]]
            self.reject_suite(suite, "binding map must")
        suite = copy.deepcopy(self.suite)
        del suite["cases"][0]["programs"][0]["buffer_slots"]
        self.reject_suite(suite, "binding map must")

    def test_scalar_cannot_satisfy_later_array_slot(self):
        for mapping in ([420, 430], [400, 420]):
            suite = copy.deepcopy(self.suite)
            suite["cases"][0]["dispatches"][1]["bindings"] = mapping
            self.reject_suite(suite, "binding extent")

    def test_unused_pool_resource_is_rejected_even_with_no_writeback(self):
        suite = copy.deepcopy(self.suite)
        ghost = copy.deepcopy(suite["cases"][0]["buffers"][-1])
        ghost.update(binding=12, allocation=350, view=450, access="read")
        suite["cases"][0]["buffers"].append(ghost)
        self.reject_suite(suite, "unused buffer pool resources")

    def test_first_subset_slots_match_corresponding_initial_metadata(self):
        for field, value in (("binding", 1), ("access", "read"), ("length", 116)):
            with self.subTest(field=field):
                suite = copy.deepcopy(self.suite)
                suite["cases"][0]["programs"][0]["buffer_slots"][0][field] = value
                self.reject_suite(suite, "first program buffer_slots")

    def test_program_subset_slots_must_be_nonempty_unique_and_sorted(self):
        for mutation, message in (
                (lambda slots: slots.clear(), "cover every resource"),
                (lambda slots: slots[1].update(binding=4), "unique and sorted"),
                (lambda slots: slots.reverse(), "unique and sorted"),
                (lambda slots: slots.append(dict(binding=10, access="read", length=120)), "binding map must")):
            suite = copy.deepcopy(self.suite)
            mutation(suite["cases"][0]["programs"][1]["buffer_slots"])
            self.reject_suite(suite, message)

    def test_total_resource_cap_is_checked_before_allocating_buffers(self):
        suite = copy.deepcopy(self.suite)
        suite["cases"][0]["buffers"] = [suite["cases"][0]["buffers"][0]] * (compare.MAX_SERIAL_RESOURCES + 1)
        self.reject_suite(suite, "buffer pool exceeds 64 resources")

    def test_correct_later_writeback_cannot_hide_bad_landing_or_guards(self):
        for allocation_id, offset in ((330, 20), (340, 28), (330, 0), (340, 163)):
            wrong = copy.deepcopy(self.report)
            allocation = next(value for value in wrong["results"][1]["allocations"]
                              if value["allocation"] == allocation_id)
            data = bytearray.fromhex(allocation["bytes_hex"])
            data[offset] ^= 0xff
            allocation["bytes_hex"] = data.hex()
            with self.assertRaisesRegex(compare.CaptureError, f"allocation {allocation_id}.*offset {offset}"):
                compare.validate_capture(self.suite, self.digest, wrong)

    def test_old_suites_keep_valid_captures_on_every_backend(self):
        for name in ("suite.json", *(f"suite-v{version}.json" for version in range(2, 7))):
            raw = SUITE_PATH.with_name(name).read_bytes()
            suite, digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
            for backend in compare.ALLOCATION_OBSERVATIONS:
                with self.subTest(suite=name, backend=backend):
                    compare.validate_capture(suite, digest, synthetic_report(suite, digest, backend), backend)

    def test_three_backend_comparison_cli_accepts_resource_subsets(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments = ["--suite", str(SUITE_PATH)]
            for option, backend in (("--native", "native-metal"), ("--vulkan", "vulkan"),
                                    ("--metal-provider", "native-metal-provider")):
                path = Path(temporary) / f"{backend}.json"
                path.write_text(json.dumps(synthetic_report(self.suite, self.digest, backend)))
                arguments.extend((option, str(path)))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = compare.main(arguments)
            self.assertEqual(status, 0)
            self.assertIn("PASS parity: native-metal / vulkan / native-metal-provider", output.getvalue())
            self.assertIn("compute-buffer-v7; 3 cases", output.getvalue())


if __name__ == "__main__":
    unittest.main()
