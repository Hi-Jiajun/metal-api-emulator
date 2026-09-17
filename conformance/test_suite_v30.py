"""The lease face of the fixture format (`research/docs/23` §90, R9i).

The committed `suite-v30.json` is the first fixture that says *where* a view's
bytes come from: the trace-owned bytes every pre-R9i case used, a staged lease
the provider copies, or a borrowed lease it maps without copying. This file
pins the three review surfaces of that declaration — the comparator's
validation of the suite, its check of the capture's own report, and the
copy-in count the borrowed arm changes — against the committed fixture and
against synthetic captures, so the rules are exercised without a GPU.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
V30_PATH = CONFORMANCE / "suite-v30.json"

STAGED_ID = "staged_lease_copy_word"
BORROWED_ID = "borrowed_lease_copy_word"
STAGED_READ_VIEW = 1010
BORROWED_READ_VIEW = 1012


def committed_suite():
    return json.loads(V30_PATH.read_text(encoding="utf-8"))


def suite_digest(suite):
    return hashlib.sha256(json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()


def observed_modes(case, modes):
    """The report segment the capture tool writes for one case."""
    return [{"view": buffer["view"], "mode": mode}
            for buffer, mode in zip(case["buffers"], modes)]


def capture_for(suite, digest, backend="vulkan", counts=None, modes=None,
                report_modes=True):
    """A synthetic capture of the lease fixture.

    `counts` overrides the per-case `(copy_in, copy_out)` pair, whose default
    is the run the rules expect: a staged view is copied in beside the owned
    one, a borrowed view is not.
    """
    report = synthetic_report(suite, digest, backend)
    default_counts = {
        STAGED_ID: (2, 1),
        BORROWED_ID: (1, 1),
    }
    for result in report["results"]:
        pair = (counts or {}).get(result["id"], default_counts[result["id"]])
        if backend != "native-metal":
            result["copy_in"], result["copy_out"] = pair
        if not report_modes:
            continue
        case = next(case for case in suite["cases"] if case["id"] == result["id"])
        if modes is not None and result["id"] in modes:
            result["storage_modes"] = observed_modes(case, modes[result["id"]])
        elif modes is None:
            default = ["owned_bytes"] * len(case["buffers"])
            mode = "staged_lease" if result["id"] == STAGED_ID else "borrowed_no_copy"
            default[0] = mode
            result["storage_modes"] = observed_modes(case, default)
    return report


def check(suite, report, backend="vulkan"):
    compare.validate_capture(suite, suite_digest(suite), report,
                             required_backend=backend)


class SuiteDeclarationTests(unittest.TestCase):
    """The committed fixture declares one arm per view, and only one."""

    def plan(self, case_id=BORROWED_ID):
        return compare._suite_plan(committed_suite())[case_id]

    def test_the_committed_suite_declares_both_lease_arms(self):
        plan = compare._suite_plan(committed_suite())
        self.assertEqual(plan[STAGED_ID][7],
                         {STAGED_READ_VIEW: "staged_lease", 1011: "owned_bytes"})
        self.assertEqual(plan[BORROWED_ID][7],
                         {BORROWED_READ_VIEW: "borrowed_no_copy", 1013: "owned_bytes"})
        self.assertEqual(plan[BORROWED_ID][8], frozenset({1002}))

    def test_a_leased_case_names_the_rails_that_execute_it(self):
        plan = compare._suite_plan(committed_suite())
        self.assertEqual(plan[STAGED_ID][6],
                         ["vulkan", "native-metal", "native-metal-provider"])

    def test_an_unknown_storage_mode_is_refused(self):
        suite = committed_suite()
        suite["cases"][0]["buffers"][0]["storage_mode"] = "shared_memory"
        with self.assertRaisesRegex(compare.CaptureError, "unknown buffer storage mode"):
            compare._suite_plan(suite)

    def test_one_allocation_carries_one_arm(self):
        suite = committed_suite()
        # A second view of the leased allocation, this time owned: the owner's
        # window cannot be two arms at once.
        suite["cases"][0]["buffers"].append({
            "binding": 2,
            "allocation": 1000,
            "view": 1012,
            "offset": 24,
            "length": 4,
            "allocation_size": 32,
            "access": "read",
            "initial_hex": "ffffffff",
        })
        with self.assertRaisesRegex(compare.CaptureError, "declares two source arms"):
            compare._suite_plan(suite)

    def test_a_leased_case_declares_one_view_per_allocation(self):
        suite = committed_suite()
        suite["cases"][0]["buffers"].append({
            "binding": 2,
            "allocation": 1000,
            "view": 1012,
            "offset": 24,
            "length": 4,
            "allocation_size": 32,
            "access": "read",
            "initial_hex": "ffffffff",
            "storage_mode": "staged_lease",
        })
        with self.assertRaisesRegex(compare.CaptureError, "one view per allocation"):
            compare._suite_plan(suite)

    def test_an_owned_case_keeps_the_old_shape(self):
        suite = committed_suite()
        for case in suite["cases"]:
            for buffer in case["buffers"]:
                buffer.pop("storage_mode", None)
            case.pop("capture_rails", None)
        plan = compare._suite_plan(suite)
        self.assertIsNone(plan[STAGED_ID][7])
        self.assertIsNone(plan[BORROWED_ID][7])
        self.assertIsNone(plan[STAGED_ID][6])


class CaptureObservationTests(unittest.TestCase):
    """The capture's own report of the arm each view ran with."""

    def test_the_expected_arms_pass(self):
        suite = committed_suite()
        check(suite, capture_for(suite, suite_digest(suite)))

    def test_a_fallback_to_owned_bytes_is_refused(self):
        suite = committed_suite()
        report = capture_for(suite, suite_digest(suite),
                             modes={STAGED_ID: ["owned_bytes", "owned_bytes"]})
        with self.assertRaisesRegex(compare.CaptureError, "are not the suite's"):
            check(suite, report)

    def test_a_missing_report_is_refused_for_a_provider(self):
        suite = committed_suite()
        report = capture_for(suite, suite_digest(suite), report_modes=False)
        with self.assertRaisesRegex(compare.CaptureError, "has to report the source arm"):
            check(suite, report)

    def test_the_reference_oracle_reports_no_arm(self):
        suite = committed_suite()
        # The oracle is not a provider: it places the same bytes in its own
        # buffer, reports no counters and no arm.
        report = capture_for(suite, suite_digest(suite), backend="native-metal",
                             report_modes=False)
        check(suite, report, backend="native-metal")
        reported = capture_for(suite, suite_digest(suite), backend="native-metal")
        with self.assertRaisesRegex(compare.CaptureError, "cannot report a source arm"):
            check(suite, reported, backend="native-metal")

    def test_a_case_without_a_lease_declaration_must_not_report_one(self):
        suite = committed_suite()
        for case in suite["cases"]:
            for buffer in case["buffers"]:
                buffer.pop("storage_mode", None)
            case.pop("capture_rails", None)
        report = synthetic_report(suite, suite_digest(suite))
        for result in report["results"]:
            result["copy_in"], result["copy_out"] = 2, 1
            case = next(case for case in suite["cases"] if case["id"] == result["id"])
            result["storage_modes"] = observed_modes(
                case, ["owned_bytes"] * len(case["buffers"]))
        with self.assertRaisesRegex(compare.CaptureError,
                                    "declares no lease arm for this case"):
            check(suite, report)

    def test_a_borrowed_view_is_not_copied_in(self):
        suite = committed_suite()
        # The same run reported with the owned arm's count: one copy-in too
        # many, which is what a rail that uploaded the owner's bytes would
        # report.
        report = capture_for(suite, suite_digest(suite), counts={BORROWED_ID: (2, 1)})
        with self.assertRaisesRegex(compare.CaptureError,
                                    "copy_in 2 does not match 1"):
            check(suite, report)

    def test_a_rail_outside_the_marker_must_omit_the_case(self):
        suite = committed_suite()
        report = capture_for(suite, suite_digest(suite), backend="vulkan-objects")
        with self.assertRaisesRegex(compare.CaptureError,
                                    "is not a rail this marked case runs on"):
            check(suite, report, backend="vulkan-objects")


if __name__ == "__main__":
    unittest.main()
