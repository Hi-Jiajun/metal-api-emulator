"""Synthetic binary-AIR encoding checks; these tests are not GPU execution evidence."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


V7_PATH = Path(__file__).with_name("suite-v7.json")
V8_PATH = Path(__file__).with_name("suite-v8.json")


class BinaryAirEncodingTests(unittest.TestCase):
    def setUp(self):
        self.raw = V8_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v8_reuses_v7_cases_and_covers_raw_and_wrapped(self):
        v7 = json.loads(V7_PATH.read_text())
        self.assertEqual(self.suite["suite"], "compute-buffer-v8")
        self.assertEqual(
            [case["id"] for case in self.suite["cases"]],
            [case["id"] for case in v7["cases"]],
        )
        encodings = [case["air_encoding"] for case in self.suite["cases"]]
        self.assertIn("raw", encodings)
        self.assertIn("wrapped", encodings)
        for v8_case, v7_case in zip(self.suite["cases"], v7["cases"]):
            stripped = {key: value for key, value in v8_case.items()
                        if key != "air_encoding"}
            self.assertEqual(stripped, v7_case)
        for case in self.suite["cases"]:
            for program in case["programs"]:
                for kind in ("air", "metal"):
                    source = program[kind]
                    actual = hashlib.sha256(
                        (V8_PATH.parent / source["path"]).read_bytes()
                    ).hexdigest()
                    self.assertEqual(actual, source["sha256"], source["path"])

    def test_v8_plan_accepts_the_manifest_and_rejects_unknown_encodings(self):
        plan = compare._suite_plan(self.suite)
        self.assertEqual(len(plan), len(self.suite["cases"]))
        for backend in compare.ALLOCATION_OBSERVATIONS:
            with self.subTest(backend=backend):
                compare.validate_capture(
                    self.suite, self.digest,
                    synthetic_report(self.suite, self.digest, backend), backend,
                )
        invalid = copy.deepcopy(self.suite)
        invalid["cases"][0]["air_encoding"] = "spirv"
        with self.assertRaisesRegex(compare.CaptureError, "unknown air_encoding"):
            compare.validate_capture(
                invalid, self.digest,
                synthetic_report(invalid, self.digest),
            )


if __name__ == "__main__":
    unittest.main()
