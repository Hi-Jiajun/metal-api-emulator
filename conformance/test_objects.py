"""Synthetic object API capture validation; these tests are not GPU evidence."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import compare
from test_compare import synthetic_report


SUITES = ("suite.json", *(f"suite-v{version}.json" for version in range(2, 8)))
OBJECT_BACKENDS = ("vulkan-objects", "native-metal-provider-objects")
CAPTURE_FLAGS = {
    "native-metal": "--native",
    "vulkan": "--vulkan",
    "native-metal-provider": "--metal-provider",
    "vulkan-objects": "--vulkan-objects",
    "native-metal-provider-objects": "--metal-objects",
}


class ObjectCaptureTests(unittest.TestCase):
    def setUp(self):
        raw = Path(__file__).with_name("suite-v7.json").read_bytes()
        self.suite = json.loads(raw)
        self.digest = hashlib.sha256(raw).hexdigest()

    def test_both_object_paths_accept_all_seven_suite_versions(self):
        for name in SUITES:
            raw = Path(__file__).with_name(name).read_bytes()
            suite, digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
            for backend in OBJECT_BACKENDS:
                with self.subTest(suite=name, backend=backend):
                    report = synthetic_report(suite, digest, backend)
                    compare.validate_capture(suite, digest, report, required_backend=backend)

    def test_object_backends_require_host_landing_observation(self):
        for backend in OBJECT_BACKENDS:
            with self.subTest(backend=backend):
                report = synthetic_report(self.suite, self.digest, backend)
                report["allocation_observation"] = "gpu-buffer-readback"
                with self.assertRaisesRegex(compare.CaptureError,
                                            f"{backend} requires allocation_observation host-writeback-landing"):
                    compare.validate_capture(self.suite, self.digest, report)

    def test_object_writebacks_must_match_final_bytes(self):
        for backend in OBJECT_BACKENDS:
            with self.subTest(backend=backend):
                report = synthetic_report(self.suite, self.digest, backend)
                write = report["results"][0]["writebacks"][-1]
                data = bytearray.fromhex(write["bytes_hex"])
                data[-1] ^= 1
                write["bytes_hex"] = data.hex()
                with self.assertRaisesRegex(compare.CaptureError, "writeback allocation.*first differing byte"):
                    compare.validate_capture(self.suite, self.digest, report)

    def test_later_first_use_resources_require_final_writebacks(self):
        for backend in OBJECT_BACKENDS:
            for case_index, case in enumerate(self.suite["cases"]):
                first_views = set(case["dispatches"][0]["bindings"])
                late_writes = [write for write in case["expected_writebacks"]
                               if write["view"] not in first_views]
                self.assertTrue(late_writes)
                for late_write in late_writes:
                    with self.subTest(backend=backend, case=case["id"], view=late_write["view"]):
                        report = synthetic_report(self.suite, self.digest, backend)
                        report["results"][case_index]["writebacks"].remove(late_write)
                        with self.assertRaisesRegex(compare.CaptureError, "writable set mismatch"):
                            compare.validate_capture(self.suite, self.digest, report)

    def test_late_resource_writeback_must_land_in_full_allocation(self):
        for backend in OBJECT_BACKENDS:
            with self.subTest(backend=backend):
                case = self.suite["cases"][0]
                first_views = set(case["dispatches"][0]["bindings"])
                buffer = next(buffer for buffer in case["buffers"] if buffer["view"] not in first_views)
                report = synthetic_report(self.suite, self.digest, backend)
                allocation = next(value for value in report["results"][0]["allocations"]
                                  if value["allocation"] == buffer["allocation"])
                data = bytearray.fromhex(allocation["bytes_hex"])
                data[buffer["offset"]:buffer["offset"] + buffer["length"]] = bytes.fromhex(buffer["initial_hex"])
                allocation["bytes_hex"] = data.hex()
                with self.assertRaisesRegex(compare.CaptureError, "allocation.*first differing byte"):
                    compare.validate_capture(self.suite, self.digest, report)

    def test_object_capture_preserves_prefix_and_suffix_guards(self):
        for backend in OBJECT_BACKENDS:
            for case_index, case in enumerate(self.suite["cases"]):
                for buffer in case["buffers"]:
                    for offset in (buffer["offset"] - 1, buffer["offset"] + buffer["length"]):
                        with self.subTest(backend=backend, case=case["id"],
                                          allocation=buffer["allocation"], offset=offset):
                            self.assertGreaterEqual(offset, 0)
                            self.assertLess(offset, buffer["allocation_size"])
                            report = synthetic_report(self.suite, self.digest, backend)
                            allocation = next(value for value in report["results"][case_index]["allocations"]
                                              if value["allocation"] == buffer["allocation"])
                            data = bytearray.fromhex(allocation["bytes_hex"])
                            data[offset] ^= 1
                            allocation["bytes_hex"] = data.hex()
                            with self.assertRaisesRegex(compare.CaptureError, f"allocation.*offset {offset}:"):
                                compare.validate_capture(self.suite, self.digest, report)

    def test_submitted_object_capture_cannot_claim_visible_completion(self):
        for backend in OBJECT_BACKENDS:
            with self.subTest(backend=backend):
                report = synthetic_report(self.suite, self.digest, backend)
                report["results"][-1]["completion"] = "Submitted"
                with self.assertRaisesRegex(compare.CaptureError, "completion must be CompletedVisible"):
                    compare.validate_capture(self.suite, self.digest, report)


class ObjectCaptureCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prepare_suite("suite-v7.json")

    def prepare_suite(self, name):
        raw = Path(__file__).with_name(name).read_bytes()
        self.suite, self.digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
        self.suite_path = self.root / "suite.json"
        self.suite_path.write_bytes(raw)
        self.paths = {backend: self.root / f"{backend}.json" for backend in CAPTURE_FLAGS}
        for backend in CAPTURE_FLAGS:
            self.write_report(backend, synthetic_report(self.suite, self.digest, backend))

    def write_report(self, backend, report):
        self.paths[backend].write_text(json.dumps(report), encoding="utf-8")

    def arguments(self, backends=tuple(CAPTURE_FLAGS)):
        return [item for backend in backends for item in (CAPTURE_FLAGS[backend], str(self.paths[backend]))]

    def run_cli(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = compare.main(["--suite", str(self.suite_path), *arguments])
        return status, output.getvalue(), errors.getvalue()

    def test_five_paths_compare_all_seven_suite_versions(self):
        for name in SUITES:
            with self.subTest(suite=name):
                self.prepare_suite(name)
                status, output, errors = self.run_cli(self.arguments())
                self.assertEqual(status, 0, errors)
                self.assertIn(f"PASS parity: {' / '.join(CAPTURE_FLAGS)};", output)
                self.assertIn("Swift native GPU buffer readback", output)
                self.assertIn("object API provider host writeback landing", output)
                self.assertIn(f"{len(self.suite['cases'])} cases", output)

    def test_object_paths_are_independently_optional(self):
        base = ("native-metal", "vulkan")
        for optional in (("vulkan-objects",), ("native-metal-provider-objects",), OBJECT_BACKENDS):
            with self.subTest(optional=optional):
                backends = (*base, *optional)
                status, output, errors = self.run_cli(self.arguments(backends))
                self.assertEqual(status, 0, errors)
                self.assertIn(f"PASS parity: {' / '.join(backends)};", output)
                self.assertIn("object API provider host writeback landing", output)

    def test_cli_rejects_every_mislabeled_backend_position(self):
        for expected in CAPTURE_FLAGS:
            for actual in CAPTURE_FLAGS:
                if expected == actual:
                    continue
                with self.subTest(expected=expected, actual=actual):
                    arguments = self.arguments()
                    arguments[arguments.index(CAPTURE_FLAGS[expected]) + 1] = str(self.paths[actual])
                    status, output, errors = self.run_cli(arguments)
                    self.assertEqual(status, 1)
                    self.assertEqual(output, "")
                    self.assertIn(f"expected backend {expected}, got {actual}", errors)

    def test_invalid_object_capture_prevents_any_success_output(self):
        for backend in OBJECT_BACKENDS:
            for mutation, expected in (
                    (lambda report: report.update(suite_sha256="0" * 64), "suite_sha256 mismatch"),
                    (lambda report: report["results"].pop(), "missing cases"),
                    (lambda report: report["results"][-1]["writebacks"].pop(), "writable set mismatch"),
                    (lambda report: report.update(allocation_observation="gpu-buffer-readback"),
                     f"{backend} requires allocation_observation host-writeback-landing")):
                with self.subTest(backend=backend, expected=expected):
                    self.prepare_suite("suite-v7.json")
                    report = synthetic_report(self.suite, self.digest, backend)
                    mutation(report)
                    self.write_report(backend, report)
                    status, output, errors = self.run_cli(self.arguments())
                    self.assertEqual(status, 1)
                    self.assertEqual(output, "")
                    self.assertIn(expected, errors)

    def test_check_accepts_each_object_capture_without_claiming_parity(self):
        for backend in OBJECT_BACKENDS:
            with self.subTest(backend=backend):
                status, output, errors = self.run_cli(["--check", str(self.paths[backend])])
                self.assertEqual(status, 0, errors)
                self.assertIn(f"PASS capture: {backend};", output)
                self.assertIn("allocations=host-writeback-landing", output)
                self.assertNotIn("parity", output)

    def test_check_excludes_all_comparison_arguments(self):
        for checked in OBJECT_BACKENDS:
            for extra in CAPTURE_FLAGS:
                with self.subTest(checked=checked, extra=extra):
                    with self.assertRaises(SystemExit) as error:
                        self.run_cli(["--check", str(self.paths[checked]), *self.arguments((extra,))])
                    self.assertEqual(error.exception.code, 2)

    def test_object_comparison_still_requires_both_original_reports(self):
        for objects in (("vulkan-objects",), ("native-metal-provider-objects",), OBJECT_BACKENDS):
            for base in ((), ("native-metal",), ("vulkan",)):
                with self.subTest(objects=objects, base=base):
                    with self.assertRaises(SystemExit) as error:
                        self.run_cli(self.arguments((*base, *objects)))
                    self.assertEqual(error.exception.code, 2)

    def test_original_two_and_three_path_output_is_unchanged(self):
        suffix = f"{self.suite['suite']}; {len(self.suite['cases'])} cases; host-visible bytes agreement; "
        for backends, expected in (
                (("native-metal", "vulkan"),
                 "PASS parity: native-metal / vulkan; " + suffix +
                 "native GPU buffer readback / Vulkan host writeback landing\n"),
                (("native-metal", "vulkan", "native-metal-provider"),
                 "PASS parity: native-metal / vulkan / native-metal-provider; " + suffix +
                 "Swift native GPU buffer readback / Vulkan and Rust Metal provider host writeback landing\n")):
            with self.subTest(backends=backends):
                status, output, errors = self.run_cli(self.arguments(backends))
                self.assertEqual(status, 0, errors)
                self.assertEqual(output, expected)


if __name__ == "__main__":
    unittest.main()
