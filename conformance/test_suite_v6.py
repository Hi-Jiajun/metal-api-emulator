"""Synthetic layout-switching checks; these tests do not generate GPU evidence."""

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


SUITE_PATH = Path(__file__).with_name("suite-v6.json")


class PipelineLayoutTests(unittest.TestCase):
    def setUp(self):
        self.raw = SUITE_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()
        self.report = synthetic_report(self.suite, self.digest)

    def reject_suite(self, suite, message):
        with self.assertRaisesRegex(compare.CaptureError, message):
            compare.validate_capture(suite, self.digest, self.report)

    def test_all_three_backends_accept_selected_layout_results(self):
        for backend in compare.ALLOCATION_OBSERVATIONS:
            with self.subTest(backend=backend):
                compare.validate_capture(
                    self.suite, self.digest,
                    synthetic_report(self.suite, self.digest, backend), backend,
                )

    def test_independent_reference_uses_selected_shader_and_slots(self):
        for case in self.suite["cases"]:
            pool = {buffer["view"]: bytearray.fromhex(buffer["initial_hex"])
                    for buffer in case["buffers"]}
            for dispatch in case["dispatches"]:
                program = case["programs"][dispatch["program"]]
                bindings = {slot["binding"]: pool[view]
                            for slot, view in zip(program["buffer_slots"], dispatch["bindings"])}
                if program["entry"] == "transform_3d":
                    source, scalar, target = (bindings[index] for index in (0, 2, 5))
                    bias = int.from_bytes(scalar, "little")
                    for offset in range(0, len(source), 4):
                        value = (int.from_bytes(source[offset:offset + 4], "little") + bias) & 0xffffffff
                        source[offset:offset + 4] = value.to_bytes(4, "little")
                        target[offset:offset + 4] = (value ^ 0xa5a55a5a).to_bytes(4, "little")
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
            for expected in case["expected_writebacks"]:
                self.assertEqual(pool[expected["view"]].hex(), expected["bytes_hex"], case["id"])
            for buffer in case["buffers"]:
                if buffer["access"] == "read":
                    self.assertEqual(pool[buffer["view"]].hex(), buffer["initial_hex"])

    def test_previous_v5_outputs_cannot_pass(self):
        previous = json.loads(SUITE_PATH.with_name("suite-v5.json").read_text())
        for index, case in enumerate(previous["cases"]):
            with self.subTest(case=index):
                wrong = copy.deepcopy(self.report)
                wrong["results"][index]["writebacks"] = case["expected_writebacks"]
                with self.assertRaisesRegex(compare.CaptureError, "differing byte"):
                    compare.validate_capture(self.suite, self.digest, wrong)

    def test_scalar_mapped_to_array_slot_is_rejected(self):
        for mapping in ([400, 420, 410], [410, 400, 420]):
            with self.subTest(mapping=mapping):
                suite = copy.deepcopy(self.suite)
                suite["cases"][0]["dispatches"][1]["bindings"] = mapping
                self.reject_suite(suite, "binding extent")

    def test_selected_slot_extent_must_match_view(self):
        suite = copy.deepcopy(self.suite)
        suite["cases"][0]["programs"][1]["buffer_slots"][0]["length"] = 120
        self.reject_suite(suite, "binding extent")

    def test_duplicate_missing_unknown_or_unsorted_slot_metadata_rejected(self):
        for mutation, message in (
                (lambda slots: slots[1].update(binding=1), "unique and sorted"),
                (lambda slots: slots.reverse(), "unique and sorted"),
                (lambda slots: slots.pop(), "cover every resource"),
                (lambda slots: slots.append(copy.deepcopy(slots[-1])), "cover every resource"),
                (lambda slots: slots[0].pop("length"), "expected fields"),
                (lambda slots: slots[0].pop("binding"), "expected fields"),
                (lambda slots: slots[0].pop("access"), "expected fields"),
                (lambda slots: slots[0].update(unknown=True), "expected fields")):
            with self.subTest(message=message, mutation=mutation):
                suite = copy.deepcopy(self.suite)
                mutation(suite["cases"][0]["programs"][1]["buffer_slots"])
                self.reject_suite(suite, message)

    def test_slot_integer_bounds_and_access_are_strict(self):
        for field, values in (
                ("binding", (-1, 1 << 32, True, "1")),
                ("length", (0, compare.MAX_ALLOCATION_BYTES + 1, True, "4")),
                ("access", (None, "readonly", 1))):
            for value in values:
                with self.subTest(field=field, value=value):
                    suite = copy.deepcopy(self.suite)
                    suite["cases"][0]["programs"][1]["buffer_slots"][0][field] = value
                    self.reject_suite(suite, "unknown buffer access" if field == "access" else f"{field}: expected integer")

    def test_optional_slots_do_not_allow_unknown_program_keys_or_null(self):
        suite = copy.deepcopy(self.suite)
        suite["cases"][0]["programs"][1]["unknown"] = True
        self.reject_suite(suite, "expected fields")
        for value in (None, {}, "slots"):
            suite = copy.deepcopy(self.suite)
            suite["cases"][0]["programs"][1]["buffer_slots"] = value
            self.reject_suite(suite, "buffer_slots: expected a list")

    def test_first_program_layout_must_match_initial_metadata(self):
        for field, value in (("binding", 1), ("access", "read"), ("length", 116)):
            with self.subTest(field=field):
                suite = copy.deepcopy(self.suite)
                suite["cases"][0]["programs"][0]["buffer_slots"][0][field] = value
                self.reject_suite(suite, "first program buffer_slots")

    def test_absent_slots_keep_legacy_layout_and_cannot_hide_v6_remap(self):
        suite = copy.deepcopy(self.suite)
        del suite["cases"][0]["programs"][0]["buffer_slots"]
        compare.validate_capture(suite, self.digest, self.report)
        del suite["cases"][0]["programs"][1]["buffer_slots"]
        self.reject_suite(suite, "binding extent")

    def test_mapping_must_permute_existing_views(self):
        for mapping in ([420, 400, 400], [420, 400], [420, 400, 999]):
            with self.subTest(mapping=mapping):
                suite = copy.deepcopy(self.suite)
                suite["cases"][0]["dispatches"][1]["bindings"] = mapping
                self.reject_suite(suite, "permute every resource")

    def test_earlier_written_input_still_requires_final_writeback(self):
        # View 400 is read-only in the last dispatch, but program 0 wrote it.
        suite = copy.deepcopy(self.suite)
        suite["cases"][0]["expected_writebacks"].pop(0)
        self.reject_suite(suite, "do not cover writable views")
        wrong = copy.deepcopy(self.report)
        wrong["results"][0]["writebacks"].pop(0)
        with self.assertRaisesRegex(compare.CaptureError, "writable set mismatch"):
            compare.validate_capture(self.suite, self.digest, wrong)

    def test_read_only_input_scalar_and_allocation_guards_are_preserved(self):
        for allocation_id, offset in ((300, 64), (320, 32), (300, 0), (310, 151), (320, 51)):
            with self.subTest(allocation=allocation_id, offset=offset):
                wrong = copy.deepcopy(self.report)
                allocation = next(value for value in wrong["results"][0]["allocations"]
                                  if value["allocation"] == allocation_id)
                data = bytearray.fromhex(allocation["bytes_hex"])
                data[offset] ^= 0xff
                allocation["bytes_hex"] = data.hex()
                with self.assertRaisesRegex(compare.CaptureError, f"allocation {allocation_id}.*offset {offset}"):
                    compare.validate_capture(self.suite, self.digest, wrong)

    def test_old_suites_accept_legacy_layouts_on_all_backends(self):
        for name in ("suite.json", *(f"suite-v{version}.json" for version in range(2, 6))):
            raw = SUITE_PATH.with_name(name).read_bytes()
            suite, digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
            for backend in compare.ALLOCATION_OBSERVATIONS:
                with self.subTest(suite=name, backend=backend):
                    compare.validate_capture(suite, digest, synthetic_report(suite, digest, backend), backend)

    def test_three_reports_compare_through_existing_cli(self):
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
            self.assertIn("compute-buffer-v6; 3 cases", output.getvalue())


if __name__ == "__main__":
    unittest.main()
