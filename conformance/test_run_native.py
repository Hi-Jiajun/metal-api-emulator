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


if __name__ == "__main__":
    unittest.main()
