"""Suite-v2 golden and synthetic comparator tests; no hardware evidence."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import unittest

import compare
from test_compare import synthetic_report


SUITE_PATH = Path(__file__).with_name("suite-v2.json")
V1_SHA256 = "b46f70267a9516b81a7b8b6e7256af6f3b54a3eb92e8a9777ed6a0afa62f2090"


def words(hex_data):
    data = bytes.fromhex(hex_data)
    return [word for (word,) in struct.iter_unpack("<I", data)]


class SuiteV2Tests(unittest.TestCase):
    def setUp(self):
        raw = SUITE_PATH.read_bytes()
        self.suite = json.loads(raw)
        self.digest = hashlib.sha256(raw).hexdigest()
        self.cases = {case["id"]: case for case in self.suite["cases"]}
        self.report = synthetic_report(self.suite, self.digest)

    def result(self, case_id):
        return next(result for result in self.report["results"] if result["id"] == case_id)

    def validate(self):
        compare.validate_capture(self.suite, self.digest, self.report)

    def reject(self, message):
        with self.assertRaisesRegex(compare.CaptureError, message):
            self.validate()

    def test_v1_suite_bytes_remain_unchanged(self):
        self.assertEqual(hashlib.sha256(SUITE_PATH.with_name("suite.json").read_bytes()).hexdigest(),
                         V1_SHA256)

    def test_all_eight_cases_accept_existing_report_schema_for_both_backends(self):
        self.assertEqual(self.suite["schema_version"], 1)
        self.assertEqual(self.suite["suite"], "compute-buffer-v2")
        self.assertEqual(set(self.cases), {
            "copy_seed_a", "copy_seed_b", "indexed_tail", "indexed_full",
            "indexed_small_grid", "indexed_unit", "transform_tail", "transform_small_grid",
        })
        for backend in ("native-metal", "vulkan"):
            with self.subTest(backend=backend):
                report = synthetic_report(self.suite, self.digest, backend)
                self.assertEqual(report["schema_version"], 1)
                self.assertEqual(len(report["results"]), 8)
                compare.validate_capture(self.suite, self.digest, report, backend)

    def test_sparse_bindings_keep_allocation_identity_despite_different_orders(self):
        for case_id in ("transform_tail", "transform_small_grid"):
            with self.subTest(case_id=case_id):
                case = self.cases[case_id]
                self.assertEqual([(buffer["binding"], buffer["allocation"], buffer["access"])
                                  for buffer in case["buffers"]],
                                 [(0, 310, "read_write"), (2, 320, "read"), (5, 300, "write")])
                self.assertEqual([write["allocation"] for write in case["expected_writebacks"]],
                                 [300, 310])
                self.result(case_id)["allocations"].sort(key=lambda value: value["allocation"],
                                                         reverse=True)
        self.report["results"].reverse()
        self.validate()

    def test_sparse_binding_writeback_order_cannot_follow_buffer_order(self):
        self.result("transform_tail")["writebacks"].reverse()
        self.reject("transform_tail: writeback order differs")

    def test_either_transform_writeback_cannot_be_missing(self):
        original = copy.deepcopy(self.report)
        for case_id in ("transform_tail", "transform_small_grid"):
            for allocation in (300, 310):
                with self.subTest(case_id=case_id, allocation=allocation):
                    self.report = copy.deepcopy(original)
                    result = self.result(case_id)
                    result["writebacks"] = [write for write in result["writebacks"]
                                            if write["allocation"] != allocation]
                    self.reject(f"{case_id}: writable set mismatch")

    def test_read_write_output_is_required_by_suite_validation(self):
        case = self.cases["transform_tail"]
        case["expected_writebacks"] = [write for write in case["expected_writebacks"]
                                       if write["allocation"] != 310]
        self.reject("transform_tail: expected writebacks do not cover writable views")

    def test_either_transform_writeback_rejects_wrong_data(self):
        original = copy.deepcopy(self.report)
        for case_id in ("transform_tail", "transform_small_grid"):
            for allocation in (300, 310):
                with self.subTest(case_id=case_id, allocation=allocation):
                    self.report = copy.deepcopy(original)
                    write = next(write for write in self.result(case_id)["writebacks"]
                                 if write["allocation"] == allocation)
                    data = bytearray.fromhex(write["bytes_hex"])
                    data[-1] ^= 0x80
                    write["bytes_hex"] = data.hex()
                    self.reject(f"{case_id} writeback allocation {allocation}/view.*"
                                f"offset {write['offset'] + len(data) - 1}:")

    def test_read_write_allocation_cannot_remain_at_initial_data(self):
        case_id = "transform_tail"
        buffer = next(buffer for buffer in self.cases[case_id]["buffers"]
                      if buffer["access"] == "read_write")
        allocation = next(value for value in self.result(case_id)["allocations"]
                          if value["allocation"] == buffer["allocation"])
        data = bytearray.fromhex(allocation["bytes_hex"])
        data[buffer["offset"]:buffer["offset"] + buffer["length"]] = bytes.fromhex(buffer["initial_hex"])
        allocation["bytes_hex"] = data.hex()
        self.reject("transform_tail allocation 310: first differing byte at offset 16:")

    def test_copy_reused_pipeline_cannot_reuse_previous_input_bytes(self):
        first = self.cases["copy_seed_a"]
        second = self.cases["copy_seed_b"]
        self.assertEqual(first["entry"], second["entry"])
        self.assertEqual(first["air"], second["air"])
        self.assertEqual(first["metal"], second["metal"])
        self.assertNotEqual(first["buffers"][0]["initial_hex"], second["buffers"][0]["initial_hex"])
        self.assertNotEqual(first["buffers"][0]["offset"], second["buffers"][0]["offset"])
        old_bytes = first["expected_writebacks"][0]["bytes_hex"]
        result = self.result("copy_seed_b")
        write = result["writebacks"][0]
        write["bytes_hex"] = old_bytes
        allocation = next(value for value in result["allocations"]
                          if value["allocation"] == write["allocation"])
        data = bytearray.fromhex(allocation["bytes_hex"])
        data[write["offset"]:write["offset"] + len(bytes.fromhex(old_bytes))] = bytes.fromhex(old_bytes)
        allocation["bytes_hex"] = data.hex()
        self.reject("copy_seed_b writeback allocation 101/view 201: first differing byte at offset 4:")

    def test_transform_goldens_independently_cover_3d_indexing_wrapping_add_and_xor(self):
        mask = 0xFFFFFFFF
        for case_id in ("transform_tail", "transform_small_grid"):
            with self.subTest(case_id=case_id):
                case = self.cases[case_id]
                self.assertEqual(case["grid"], [5, 3, 2])
                buffers = {buffer["binding"]: buffer for buffer in case["buffers"]}
                initial = words(buffers[0]["initial_hex"])
                bias, = words(buffers[2]["initial_hex"])
                self.assertEqual(len(initial), 30)
                writes = {write["allocation"]: words(write["bytes_hex"])
                          for write in case["expected_writebacks"]}
                visited = []
                for z in range(2):
                    for y in range(3):
                        for x in range(5):
                            index = x + y * 5 + z * 15
                            visited.append(index)
                            updated = (initial[index] + bias) & mask
                            self.assertEqual(writes[buffers[0]["allocation"]][index], updated)
                            self.assertEqual(writes[buffers[5]["allocation"]][index],
                                             updated ^ 0xA5A55A5A)
                self.assertEqual(visited, list(range(30)))
                self.assertTrue(any(value + bias > mask for value in initial))
        tail_writes = self.cases["transform_tail"]["expected_writebacks"]
        updated_tail = next(write for write in tail_writes if write["allocation"] == 310)
        self.assertEqual(words(updated_tail["bytes_hex"])[0], 0x13)

    def test_indexed_goldens_use_actual_boundary_group_sizes(self):
        uniform = {"indexed_full": ([5, 3, 1], 305),
                   "indexed_small_grid": ([16, 4, 1], 310),
                   "indexed_unit": ([1, 1, 1], 101)}
        for case_id, (local, expected) in uniform.items():
            with self.subTest(case_id=case_id):
                case = self.cases[case_id]
                self.assertEqual(case["grid"], [10, 3, 1])
                self.assertEqual(case["local"], local)
                self.assertEqual(words(case["expected_writebacks"][0]["bytes_hex"]), [expected] * 30)
        tail = self.cases["indexed_tail"]
        self.assertEqual(tail["grid"], [10, 3, 1])
        self.assertEqual(tail["local"], [8, 2, 1])
        self.assertEqual(words(tail["expected_writebacks"][0]["bytes_hex"]),
                         ([208] * 8 + [202] * 2) * 2 + [108] * 8 + [102] * 2)


if __name__ == "__main__":
    unittest.main()
