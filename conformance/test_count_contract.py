"""Allocation-level copy counters in the capture report (research/docs/15 §5)."""

import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


V10_PATH = Path(__file__).with_name("suite-v10.json")


def counted(report, copy_in, copy_out):
    for result in report["results"]:
        result["copy_in"] = copy_in
        result["copy_out"] = copy_out
    return report


class CountContractTests(unittest.TestCase):
    def setUp(self):
        self.raw = V10_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_reports_without_counters_still_validate(self):
        # Older captures, and the Swift reference oracle, carry no counters.
        compare.validate_capture(
            self.suite, self.digest, synthetic_report(self.suite, self.digest)
        )

    def test_matching_counters_validate(self):
        report = counted(synthetic_report(self.suite, self.digest), 1, 1)
        compare.validate_capture(self.suite, self.digest, report)

    def test_wrong_copy_in_is_refused(self):
        report = counted(synthetic_report(self.suite, self.digest), 2, 1)
        with self.assertRaisesRegex(compare.CaptureError,
                                    "does not match 1 touched allocations"):
            compare.validate_capture(self.suite, self.digest, report)

    def test_wrong_copy_out_is_refused(self):
        report = counted(synthetic_report(self.suite, self.digest), 1, 2)
        with self.assertRaisesRegex(compare.CaptureError, "written allocations"):
            compare.validate_capture(self.suite, self.digest, report)

    def test_half_a_counter_pair_is_refused(self):
        # Only copy_in is present, so the result object is neither the base nor
        # the counted field set.
        report = synthetic_report(self.suite, self.digest)
        report["results"][0]["copy_in"] = 1
        with self.assertRaisesRegex(compare.CaptureError,
                                    "optionally with copy_in and copy_out"):
            compare.validate_capture(self.suite, self.digest, report)

    def test_unknown_result_fields_are_refused(self):
        report = synthetic_report(self.suite, self.digest)
        report["results"][0]["extra"] = 1
        with self.assertRaisesRegex(compare.CaptureError,
                                    "optionally with copy_in and copy_out"):
            compare.validate_capture(self.suite, self.digest, report)


if __name__ == "__main__":
    unittest.main()
