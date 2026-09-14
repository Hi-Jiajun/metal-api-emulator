"""Simulated command responses only; no synthetic fixture is Metal evidence."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import run_native
from test_compare import SUITE_PATH, synthetic_report


class NativeCaptureFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.suite_path = self.root / "suite.json"
        self.suite_path.write_bytes(SUITE_PATH.read_bytes())
        self.output = self.root / "evidence"
        self.suite = json.loads(self.suite_path.read_text())
        digest = hashlib.sha256(self.suite_path.read_bytes()).hexdigest()
        self.report = synthetic_report(self.suite, digest, "native-metal")
        self.probe = {"schema_version": 1, "kind": "metal-device-probe",
                      "platform": self.report["platform"], "device": self.report["device"],
                      "eligible": True, "reason": "eligible", "supports_apple4": True,
                      "has_unified_memory": True}
        self.commands = []
        self.fail_capture = False

    def fake_command(self, argv, *, stdout, stderr, check, timeout):
        self.assertTrue(check)
        self.assertGreater(timeout, 0)
        self.commands.append(argv)
        if "--probe" in argv:
            stdout.write(json.dumps(self.probe).encode())
        elif "--output" in argv:
            if self.fail_capture:
                stderr.write(b"SIMULATED Metal execution failure")
                raise subprocess.CalledProcessError(1, argv)
            Path(argv[argv.index("--output") + 1]).write_text(json.dumps(self.report))

    def run_flow(self, **kwargs):
        return run_native.run_capture(self.root / "fake-oracle", self.suite_path, self.output,
                                      revision="synthetic-test-revision", run_command=self.fake_command,
                                      **kwargs)

    def status(self):
        return json.loads((self.output / "status.json").read_text())

    def test_eligible_device_runs_and_validates_capture(self):
        result = self.run_flow()
        self.assertEqual(result["capture_status"], "captured")
        self.assertEqual(result, self.status())
        self.assertEqual(len(self.commands), 3)
        self.assertTrue((self.output / "probe.json").exists())

    def test_no_gpu_is_unavailable_and_never_runs_capture(self):
        self.probe.update(device=None, eligible=False, reason="no_default_device",
                          supports_apple4=False, has_unified_memory=False)
        result = self.run_flow()
        self.assertEqual(result["capture_status"], "unavailable")
        self.assertEqual(len(self.commands), 2)
        self.assertFalse((self.output / "native-metal.json").exists())

    def test_required_gpu_unavailable_fails_with_evidence(self):
        self.probe.update(eligible=False, reason="unsupported_features", supports_apple4=False)
        with self.assertRaisesRegex(run_native.NativeRunError, "unavailable"):
            self.run_flow(require_metal=True)
        self.assertEqual(self.status()["capture_status"], "unavailable")

    def test_unnamed_device_is_ineligible_not_a_broken_probe(self):
        self.probe.update(device=" ", eligible=False, reason="unsupported_features")
        result = self.run_flow()
        self.assertEqual(result["capture_status"], "unavailable")
        self.assertEqual(len(self.commands), 2)

    def test_eligible_capture_error_is_not_downgraded_to_no_gpu(self):
        self.fail_capture = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_flow()
        self.assertEqual(self.status()["capture_status"], "failed")
        self.assertIn(b"SIMULATED", (self.output / "capture.stderr").read_bytes())

    def test_inconsistent_probe_is_a_failure(self):
        original = copy.deepcopy(self.probe)
        for change in ({"eligible": False}, {"supports_apple4": 1}, {"reason": "unknown"},
                       {"device": None}, {"schema_version": True}, {"platform": ""}):
            with self.subTest(change=change):
                probe = dict(original, **change)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_probe(probe)

    def test_bad_capture_cannot_pass(self):
        self.report["results"][0]["writebacks"][0]["bytes_hex"] = "00000000"
        with self.assertRaisesRegex(ValueError, "differing byte"):
            self.run_flow()
        self.assertEqual(self.status()["capture_status"], "failed")

    def test_vulkan_capture_cannot_substitute_for_native(self):
        self.report["backend"] = "vulkan"
        with self.assertRaisesRegex(ValueError, "expected backend native-metal"):
            self.run_flow()
        self.assertEqual(self.status()["capture_status"], "failed")

    def test_probe_capture_device_disagreement_is_failure(self):
        self.report["device"] = "SYNTHETIC DIFFERENT DEVICE"
        with self.assertRaisesRegex(run_native.NativeRunError, "differs from probe"):
            self.run_flow()

    def test_output_directory_is_not_reused(self):
        self.output.mkdir()
        (self.output / "prior.json").write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_flow()
        self.assertEqual(self.commands, [])
        self.assertEqual((self.output / "prior.json").read_text(), "keep")


class PresentSelftestValidationTests(unittest.TestCase):
    """The CI step's present-selftest byte comparison, exercised without Metal.

    `run_native.validate_present_selftest` is the function the workflow reuses,
    so the sentinel rule is pinned here rather than only in the inline heredoc.
    """

    TARGET = "4080c0ff" * 4
    SENTINEL = "fefefefe" * 4

    def reviewed_report(self, writeback_bytes=None, allocation_bytes=None, completion="CompletedVisible"):
        return {
            "completion": completion,
            "writebacks": [{"bytes_hex": writeback_bytes}],
            "allocations": [{"bytes_hex": allocation_bytes}],
        }

    def test_accepts_the_reviewed_present_target(self):
        report = self.reviewed_report(self.TARGET, self.TARGET)
        self.assertEqual(run_native.validate_present_selftest(report), [self.TARGET, self.TARGET])

    def test_rejects_the_sentinel_in_either_channel(self):
        for writeback, allocation in (
            (self.SENTINEL, self.TARGET),
            (self.TARGET, self.SENTINEL),
            (self.SENTINEL, self.SENTINEL),
        ):
            with self.subTest(writeback=writeback, allocation=allocation):
                report = self.reviewed_report(writeback, allocation)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_present_selftest(report)

    def test_rejects_a_missing_or_extra_observation(self):
        for writebacks, allocations in (
            ([], []),
            ([{"bytes_hex": self.TARGET}], []),
            ([], [{"bytes_hex": self.TARGET}]),
            ([{"bytes_hex": self.TARGET}] * 2, [{"bytes_hex": self.TARGET}]),
            # The bytes concatenated still equal [expected, expected], so these
            # two only fail because the shape itself is pinned: two writebacks
            # with no allocation, and the reverse.
            ([{"bytes_hex": self.TARGET}] * 2, []),
            ([], [{"bytes_hex": self.TARGET}] * 2),
        ):
            with self.subTest(writebacks=writebacks, allocations=allocations):
                report = {"completion": "CompletedVisible",
                          "writebacks": writebacks, "allocations": allocations}
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_present_selftest(report)

    def test_rejects_a_non_visible_completion(self):
        report = self.reviewed_report(self.TARGET, self.TARGET, completion="Submitted")
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_present_selftest(report)

    def test_rejects_a_non_object_report(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_present_selftest([])


class HeapSelftestValidationTests(unittest.TestCase):
    """The CI step's heap-selftest byte comparison, exercised without Metal.

    `run_native.validate_heap_selftest` is the function the workflow reuses, so
    the sentinel and observation-shape rules are pinned here rather than only in
    the inline heredoc.
    """

    WORD = "fefefefe"
    READ_ALLOCATION = "fefefefefefefefefefefefefefefefe"
    WRITE_ALLOCATION = "fefefefeffffffffffffffffff"
    SENTINEL_WRITE = "ffffffff"

    def reviewed_report(self, completion="CompletedVisible", device="Apple GPU",
                        platform="macOS 15.0", writeback_bytes=None,
                        read_allocation=None, write_allocation=None,
                        writebacks=None, allocations=None):
        if writebacks is None:
            writebacks = [{"allocation": 920, "view": 930, "offset": 0,
                           "bytes_hex": self.WORD if writeback_bytes is None else writeback_bytes}]
        if allocations is None:
            allocations = [
                {"allocation": 900, "bytes_hex": self.READ_ALLOCATION if read_allocation is None else read_allocation},
                {"allocation": 920, "bytes_hex": self.WRITE_ALLOCATION if write_allocation is None else write_allocation},
            ]
        return {"id": "heap_placement_copy_word", "completion": completion,
                "writebacks": writebacks, "allocations": allocations,
                "device": device, "platform": platform}

    def test_accepts_the_reviewed_heap_observation(self):
        report = self.reviewed_report()
        self.assertEqual(run_native.validate_heap_selftest(report), self.WORD)

    def test_rejects_a_sentinel_writeback(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_heap_selftest(self.reviewed_report(writeback_bytes=self.SENTINEL_WRITE))

    def test_rejects_a_sentinel_in_either_allocation(self):
        sentinel_read = "ffffffff" * 4
        sentinel_write = "ffffffff" * 3
        for read_allocation, write_allocation in (
            (sentinel_read, self.WRITE_ALLOCATION),
            (self.READ_ALLOCATION, sentinel_write),
            (sentinel_read, sentinel_write),
        ):
            with self.subTest(read=read_allocation, write=write_allocation):
                report = self.reviewed_report(read_allocation=read_allocation,
                                              write_allocation=write_allocation)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_heap_selftest(report)

    def test_rejects_a_missing_or_extra_observation(self):
        writeback = [{"allocation": 920, "view": 930, "offset": 0,
                      "bytes_hex": self.WORD}]
        allocations = [
            {"allocation": 900, "bytes_hex": self.READ_ALLOCATION},
            {"allocation": 920, "bytes_hex": self.WRITE_ALLOCATION},
        ]
        for writebacks, extra_allocations in (
            ([], []),
            (writeback, []),
            ([], allocations),
            (writeback, allocations[:1]),
            (writeback, allocations + [{"allocation": 921, "bytes_hex": "00"}]),
            (writeback + writeback, allocations),
        ):
            with self.subTest(writebacks=writebacks, allocations=extra_allocations):
                report = self.reviewed_report(writebacks=writebacks,
                                              allocations=extra_allocations)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_heap_selftest(report)

    def test_rejects_a_wrong_writeback_offset_or_identity(self):
        for change in ({"offset": 4}, {"view": 931}, {"allocation": 921}):
            with self.subTest(change=change):
                writeback = [dict({"allocation": 920, "view": 930, "offset": 0,
                                   "bytes_hex": self.WORD}, **change)]
                report = self.reviewed_report(writebacks=writeback)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_heap_selftest(report)

    def test_rejects_a_non_visible_completion(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_heap_selftest(self.reviewed_report(completion="Submitted"))

    def test_rejects_a_missing_device_or_platform(self):
        for device, platform in (("", "macOS 15.0"), ("Apple GPU", ""),
                                 (None, "macOS 15.0"), ("Apple GPU", None)):
            with self.subTest(device=device, platform=platform):
                report = self.reviewed_report(device=device, platform=platform)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_heap_selftest(report)

    def test_rejects_a_non_object_report(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_heap_selftest([])


if __name__ == "__main__":
    unittest.main()
