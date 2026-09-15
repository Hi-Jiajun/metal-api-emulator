"""Ranged-alias checks; these tests are not GPU execution evidence."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


V10_PATH = Path(__file__).with_name("suite-v10.json")
V1_PATH = Path(__file__).with_name("suite.json")


class DisjointViewTests(unittest.TestCase):
    def setUp(self):
        self.raw = V10_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v10_binds_disjoint_views_of_one_allocation(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v10")
        self.assertEqual(
            [case["id"] for case in self.suite["cases"]],
            ["alias_disjoint_pair", "alias_disjoint_pair_reversed"],
        )
        for case in self.suite["cases"]:
            grouped = {}
            for buffer in case["buffers"]:
                grouped.setdefault(buffer["allocation"], []).append(buffer)
            self.assertEqual(len(grouped), 1, case["id"])
            ranges = [(buffer["offset"], buffer["offset"] + buffer["length"])
                      for buffer in case["buffers"]]
            for index, (start, end) in enumerate(ranges):
                for other_start, other_end in ranges[index + 1:]:
                    self.assertTrue(end <= other_start or other_end <= start, case["id"])
            for source, kind in (("air", "air"), ("metal", "metal")):
                digest = hashlib.sha256(
                    (V10_PATH.parent / case[source]["path"]).read_bytes()
                ).hexdigest()
                self.assertEqual(digest, case[source]["sha256"], case["id"])

    def test_v10_plan_accepts_the_manifest_and_rejects_overlapping_views(self):
        plan = compare._suite_plan(self.suite)
        self.assertEqual(len(plan), len(self.suite["cases"]))
        for backend in compare.ALLOCATION_OBSERVATIONS:
            with self.subTest(backend=backend):
                compare.validate_capture(
                    self.suite, self.digest,
                    synthetic_report(self.suite, self.digest, backend), backend,
                )
        overlapping = copy.deepcopy(self.suite)
        overlapping["cases"][0]["buffers"][1]["offset"] = 4
        overlapping["cases"][0]["expected_writebacks"][0]["offset"] = 4
        with self.assertRaisesRegex(compare.CaptureError, "overlapping initialization"):
            compare._suite_plan(overlapping)

    def test_shared_allocations_are_only_qualified_by_v10(self):
        # A v1 fixture whose second buffer names the first buffer's allocation,
        # with ranges kept disjoint so only the suite qualification can refuse it.
        legacy = json.loads(V1_PATH.read_text())
        case = legacy["cases"][0]
        first, second = case["buffers"]
        second["allocation"] = first["allocation"]
        second["offset"] = 24
        second["allocation_size"] = first["allocation_size"]
        case["expected_writebacks"][0]["allocation"] = first["allocation"]
        case["expected_writebacks"][0]["offset"] = 24
        with self.assertRaisesRegex(compare.CaptureError, "only qualified by the v10 suite"):
            compare._suite_plan(legacy)


if __name__ == "__main__":
    unittest.main()
