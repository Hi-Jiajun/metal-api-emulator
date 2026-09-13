"""Allocation-level copy counters in the capture report (research/docs/15 §5)."""

import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


V10_PATH = Path(__file__).with_name("suite-v10.json")
V9_PATH = Path(__file__).with_name("suite-v9.json")


def counted(report, copy_in, copy_out):
    for result in report["results"]:
        result["copy_in"] = copy_in
        result["copy_out"] = copy_out
    return report


# Read by hand from suite-v9.json: each group copies in one operation per
# allocation its own dispatches touch and copies out one per allocation they
# write (research/docs/15 §5b). The split follows the fixture's
# command_buffers, so a capture that reported the whole case in one bucket
# would not match even with the same totals.
V9_GROUP_COUNTS = {
    "subset_chain_two": [(3, 2), (2, 1)],
    "subset_chain_four": [(4, 3), (4, 2)],
    "subset_chain_eight": [(4, 3), (4, 2), (4, 3), (4, 2)],
}


def grouped(report, groups_by_case):
    """Attach the summed totals and the per-command-buffer counters."""
    for result in report["results"]:
        groups = groups_by_case[result["id"]]
        result["copy_in"] = sum(copy_in for copy_in, _ in groups)
        result["copy_out"] = sum(copy_out for _, copy_out in groups)
        result["group_counts"] = [{"copy_in": copy_in, "copy_out": copy_out}
                                  for copy_in, copy_out in groups]
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


class CommandBufferCountContractTests(unittest.TestCase):
    """v9 splits one case across several submissions, so counters are per group."""

    def setUp(self):
        self.raw = V9_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def report(self, backend="vulkan"):
        return grouped(synthetic_report(self.suite, self.digest, backend),
                       V9_GROUP_COUNTS)

    def test_plan_matches_the_hand_computed_group_expectations(self):
        plan = compare._suite_plan(self.suite)
        derived = {case_id: values[3] for case_id, values in plan.items()}
        expected = {case_id: [tuple(group) for group in groups]
                    for case_id, groups in V9_GROUP_COUNTS.items()}
        self.assertEqual(derived, expected)

    def test_grouped_counters_validate_on_every_provider_backend(self):
        for backend in ("vulkan", "vulkan-objects", "native-metal-provider",
                        "native-metal-provider-objects"):
            with self.subTest(backend=backend):
                compare.validate_capture(self.suite, self.digest,
                                         self.report(backend), backend)

    def test_group_count_mismatch_is_refused(self):
        report = self.report()
        report["results"][0]["group_counts"].pop()
        with self.assertRaisesRegex(compare.CaptureError,
                                    "expected 2 command buffer groups, got 1"):
            compare.validate_capture(self.suite, self.digest, report)

    def test_one_wrong_group_counter_is_refused(self):
        report = self.report()
        report["results"][0]["group_counts"][1]["copy_out"] = 2
        with self.assertRaisesRegex(
                compare.CaptureError,
                "command buffer 1 copy_out 2 does not match 1 written allocations"):
            compare.validate_capture(self.suite, self.digest, report)

    def test_wrong_group_copy_in_is_refused(self):
        report = self.report()
        report["results"][1]["group_counts"][0]["copy_in"] = 5
        with self.assertRaisesRegex(
                compare.CaptureError,
                "command buffer 0 copy_in 5 does not match 4 touched allocations"):
            compare.validate_capture(self.suite, self.digest, report)

    def test_flat_totals_alone_are_refused_for_a_split_case(self):
        report = self.report()
        report["results"][0].pop("group_counts")
        with self.assertRaisesRegex(compare.CaptureError,
                                    "must report per-command-buffer group_counts"):
            compare.validate_capture(self.suite, self.digest, report)

    def test_totals_must_sum_the_groups(self):
        report = self.report()
        report["results"][0]["copy_out"] = 4
        with self.assertRaisesRegex(compare.CaptureError,
                                    "copy_out 4 does not match the 3 summed"):
            compare.validate_capture(self.suite, self.digest, report)

    def test_group_counts_are_refused_on_single_submission_cases(self):
        raw = V10_PATH.read_bytes()
        suite = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
        report = counted(synthetic_report(suite, digest), 1, 1)
        report["results"][0]["group_counts"] = [{"copy_in": 1, "copy_out": 1}]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "only recorded for cases split"):
            compare.validate_capture(suite, digest, report)

    def test_native_oracle_stays_outside_the_group_contract(self):
        # The Swift reference oracle is not a provider: it reports no counters,
        # and neither the totals nor the groups are required of it.
        report = self.report("native-metal")
        for result in report["results"]:
            result["copy_in"] = 99
            result["copy_out"] = 99
            result.pop("group_counts")
        compare.validate_capture(self.suite, self.digest, report, "native-metal")


if __name__ == "__main__":
    unittest.main()
