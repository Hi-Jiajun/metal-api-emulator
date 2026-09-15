"""Command-buffer boundary checks; these tests are not GPU execution evidence."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


V7_PATH = Path(__file__).with_name("suite-v7.json")
V9_PATH = Path(__file__).with_name("suite-v9.json")


class CommandBufferBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.raw = V9_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v9_reuses_v7_cases_and_splits_each_sequence(self):
        v7 = json.loads(V7_PATH.read_text())
        self.assertEqual(self.suite["suite"], "compute-buffer-v9")
        self.assertEqual(
            [case["id"] for case in self.suite["cases"]],
            [case["id"] for case in v7["cases"]],
        )
        for v9_case, v7_case in zip(self.suite["cases"], v7["cases"]):
            stripped = {key: value for key, value in v9_case.items()
                        if key != "command_buffers"}
            self.assertEqual(stripped, v7_case)
            groups = v9_case["command_buffers"]
            self.assertGreaterEqual(len(groups), 2)
            self.assertLessEqual(len(groups), 4)
            flat = [index for group in groups for index in group]
            self.assertEqual(flat, list(range(len(v7_case["dispatches"]))))
            self.assertTrue(all(group for group in groups))
        for case in self.suite["cases"]:
            for program in case["programs"]:
                for kind in ("air", "metal"):
                    source = program[kind]
                    actual = hashlib.sha256(
                        (V9_PATH.parent / source["path"]).read_bytes()
                    ).hexdigest()
                    self.assertEqual(actual, source["sha256"], source["path"])

    def test_v9_plan_accepts_the_manifest_and_rejects_malformed_groups(self):
        plan = compare._suite_plan(self.suite)
        self.assertEqual(len(plan), len(self.suite["cases"]))
        for backend in compare.ALLOCATION_OBSERVATIONS:
            with self.subTest(backend=backend):
                compare.validate_capture(
                    self.suite, self.digest,
                    synthetic_report(self.suite, self.digest, backend), backend,
                )
        for mutate, message in (
            (lambda suite: suite["cases"][0].pop("command_buffers"), "requires command buffer groups"),
            (lambda suite: suite["cases"][0].update(command_buffers=[[0, 1], [1]]), "partition"),
            (lambda suite: suite["cases"][0].update(command_buffers=[[0], [1], [2]]), "partition"),
            (lambda suite: suite["cases"][0].update(command_buffers=[[1], [0]]), "partition"),
            (lambda suite: suite["cases"][0].update(command_buffers=[[0], [], [1]]), "cannot be empty"),
            (lambda suite: suite["cases"][0].update(command_buffers=[[0, 1]]), "two to four"),
        ):
            invalid = copy.deepcopy(self.suite)
            mutate(invalid)
            with self.assertRaisesRegex(compare.CaptureError, message):
                compare.validate_capture(
                    invalid, self.digest,
                    synthetic_report(invalid, self.digest),
                )

    def test_legacy_suites_cannot_carry_command_buffer_groups(self):
        suite = json.loads((V7_PATH.parent / "suite-v7.json").read_text())
        suite["cases"][0]["command_buffers"] = [[0], [1]]
        with self.assertRaisesRegex(compare.CaptureError, "only qualified by the v9 suite"):
            compare._suite_plan(suite)
        legacy = json.loads((V7_PATH.parent / "suite.json").read_text())
        legacy["suite"] = "compute-buffer-v9"
        with self.assertRaisesRegex(compare.CaptureError, "requires command buffer groups"):
            compare._suite_plan(legacy)


if __name__ == "__main__":
    unittest.main()
