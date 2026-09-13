"""Texture-cell row-pitch checks; these tests are not GPU execution evidence.

The v12 suite exposes the upload defect the v11 case cannot see: v11 reads only
texel (0, 0), while v12 reads every cell of a 4x4 R32Uint image whose rows are
not tightly packed in the driver's linear image.
"""

import hashlib
import json
import struct
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


V12_PATH = Path(__file__).with_name("suite-v12.json")
V11_PATH = Path(__file__).with_name("suite-v11.json")

# Read by hand from suite-v12.json: a 4x4 R32Uint image holding 0..15 and a
# `texel(x, y) + 100` store, so every cell must land [100..115] whatever row
# pitch the driver picks for its linear image.
EXPECTED_CELLS = [value + 100 for value in range(16)]
# What a row-stride-blind upload produced on Lavapipe before the fix: only
# V = 0 survives, so rows 1..3 read texel 0.
COLLAPSED_CELLS = [100, 101, 102, 103] + [100] * 12


def counted(report, copy_in, copy_out):
    for result in report["results"]:
        result["copy_in"] = copy_in
        result["copy_out"] = copy_out
    return report


class TextureCellTests(unittest.TestCase):
    def setUp(self):
        self.raw = V12_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v12_pins_two_dispatch_forms_of_one_texel_read(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v12")
        self.assertEqual([case["id"] for case in self.suite["cases"]],
                         ["texture_cell_local_4x4", "texture_cell_local_1x1"])
        self.assertEqual([case["local"] for case in self.suite["cases"]],
                         [[4, 4, 1], [1, 1, 1]])
        identities = set()
        for case in self.suite["cases"]:
            self.assertEqual(case["entry"], "read_texture_2d_cell", case["id"])
            # The same 4x4 grid both times: only the threadgroup shape differs,
            # so a shape-dependent wrong answer cannot hide behind a second
            # grid.
            self.assertEqual(case["grid"], [4, 4, 1], case["id"])
            for kind in ("air", "metal"):
                source = V12_PATH.parent / case[kind]["path"]
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                self.assertEqual(digest, case[kind]["sha256"], case["id"])
                identities.add((kind, case[kind]["path"], case[kind]["sha256"]))
        self.assertEqual(len(identities), 2, "one program pair for both cases")

    def test_v12_reads_every_cell_of_a_sparse_row_pitch_image(self):
        for case in self.suite["cases"]:
            with self.subTest(case=case["id"]):
                texture = case["textures"][0]
                self.assertEqual((texture["width"], texture["height"]), (4, 4))
                initial = bytes.fromhex(texture["initial_hex"])
                self.assertEqual(list(struct.unpack("<16I", initial)), list(range(16)))
                self.assertEqual(len(case["buffers"]), 1, "one write-only output")
                self.assertEqual(case["buffers"][0]["access"], "write")
                self.assertEqual(case["buffers"][0]["length"], 64)
                writebacks = case["expected_writebacks"]
                self.assertEqual(len(writebacks), 1)
                observed = bytes.fromhex(writebacks[0]["bytes_hex"])
                self.assertEqual(list(struct.unpack("<16I", observed)), EXPECTED_CELLS)
                # The expectation is only reachable by reading V != 0 texels,
                # and it is a permutation no collapsed coordinate can produce.
                self.assertNotEqual(list(struct.unpack("<16I", observed)), COLLAPSED_CELLS)

    def test_v11_still_reads_only_texel_zero_and_cannot_see_a_row_stride(self):
        v11 = json.loads(V11_PATH.read_text())
        case = v11["cases"][0]
        self.assertEqual([case["grid"], case["local"]], [[1, 1, 1], [1, 1, 1]])
        observed = bytes.fromhex(case["expected_writebacks"][0]["bytes_hex"])
        self.assertEqual(list(struct.unpack("<16I", observed)), [0] * 16)

    def test_v12_plan_accepts_the_manifest_in_every_backend_shape(self):
        plan = compare._suite_plan(self.suite)
        self.assertEqual(len(plan), len(self.suite["cases"]))
        for backend in compare.ALLOCATION_OBSERVATIONS:
            with self.subTest(backend=backend):
                report = synthetic_report(self.suite, self.digest, backend)
                if backend != "native-metal":
                    counted(report, 2, 1)
                compare.validate_capture(self.suite, self.digest, report, backend)

    def test_v12_refuses_the_collapsed_row_stride_result(self):
        buggy = counted(synthetic_report(self.suite, self.digest), 2, 1)
        collapsed = struct.pack("<16I", *COLLAPSED_CELLS)
        for result in buggy["results"]:
            result["writebacks"][0]["bytes_hex"] = collapsed.hex()
            for allocation in result["allocations"]:
                image = bytearray(bytes.fromhex(allocation["bytes_hex"]))
                image[4:68] = collapsed
                allocation["bytes_hex"] = bytes(image).hex()
        with self.assertRaisesRegex(compare.CaptureError, "first differing byte"):
            compare.validate_capture(self.suite, self.digest, buggy)

    def test_v12_count_contract_accounts_for_the_texture_upload(self):
        # One texture plus one output allocation copied in, one copied out.
        compare.validate_capture(
            self.suite, self.digest, counted(synthetic_report(self.suite, self.digest), 2, 1)
        )
        for counts in ((1, 1), (2, 0), (3, 1)):
            with self.subTest(counts=counts):
                report = counted(synthetic_report(self.suite, self.digest), *counts)
                with self.assertRaises(compare.CaptureError):
                    compare.validate_capture(self.suite, self.digest, report)
        # The Swift reference oracle reports bytes without counters.
        compare.validate_capture(
            self.suite, self.digest, synthetic_report(self.suite, self.digest, "native-metal")
        )


if __name__ == "__main__":
    unittest.main()
