"""Comparator tests using synthetic reports, never evidence of a Metal run."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import compare


SUITE_PATH = Path(__file__).with_name("suite.json")


def synthetic_report(suite, digest, backend="vulkan"):
    """Fabricate expected bytes solely to unit-test the comparator."""
    results = []
    for case in suite["cases"]:
        allocations = {}
        for buffer in case["buffers"]:
            allocation = buffer["allocation"]
            data = allocations.setdefault(allocation, bytearray([suite["guard_byte"]]) * buffer["allocation_size"])
            offset = buffer["offset"]
            data[offset:offset + buffer["length"]] = bytes.fromhex(buffer["initial_hex"])
        for write in case["expected_writebacks"]:
            data = bytes.fromhex(write["bytes_hex"])
            allocations[write["allocation"]][write["offset"]:write["offset"] + len(data)] = data
        results.append({
            "id": case["id"],
            "completion": "CompletedVisible",
            "writebacks": copy.deepcopy(case["expected_writebacks"]),
            "allocations": [{"allocation": key, "bytes_hex": data.hex()} for key, data in allocations.items()],
        })
    return {
        "schema_version": 1,
        "suite": suite["suite"],
        "suite_sha256": digest,
        "backend": backend,
        "allocation_observation": compare.ALLOCATION_OBSERVATIONS[backend],
        "device": "SYNTHETIC UNIT TEST; NOT HARDWARE EVIDENCE",
        "platform": "synthetic-test",
        "results": results,
    }


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.raw = SUITE_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()
        self.report = synthetic_report(self.suite, self.digest)

    def validate(self):
        compare.validate_capture(self.suite, self.digest, self.report)

    def reject(self, message):
        with self.assertRaisesRegex(compare.CaptureError, message):
            self.validate()

    def test_synthetic_valid_capture(self):
        self.validate()

    def test_case_and_allocation_order_is_irrelevant(self):
        self.report["results"].reverse()
        for result in self.report["results"]:
            result["allocations"].reverse()
        self.validate()

    def test_missing_unknown_and_duplicate_cases(self):
        original = copy.deepcopy(self.report)
        self.report["results"].pop()
        self.reject("missing cases.*indexed_boundary")
        self.report = copy.deepcopy(original)
        self.report["results"][0]["id"] = "unknown"
        self.reject("unknown case")
        self.report = original
        self.report["results"].append(copy.deepcopy(self.report["results"][0]))
        self.reject("duplicate case")

    def test_stale_digest(self):
        self.report["suite_sha256"] = "0" * 64
        self.reject("suite_sha256 mismatch")

    def test_wrong_backend(self):
        with self.assertRaisesRegex(compare.CaptureError, "expected backend native-metal"):
            compare.validate_capture(self.suite, self.digest, self.report, "native-metal")
        self.report["backend"] = "metal"
        self.reject("unknown backend")

    def test_allocation_observation_must_match_backend(self):
        for backend in compare.ALLOCATION_OBSERVATIONS:
            with self.subTest(backend=backend):
                self.report = synthetic_report(self.suite, self.digest, backend)
                self.report["allocation_observation"] = (
                    "host-writeback-landing" if backend == "native-metal" else "gpu-buffer-readback"
                )
                self.reject(f"{backend} requires allocation_observation")
        self.report = synthetic_report(self.suite, self.digest)
        del self.report["allocation_observation"]
        self.reject("expected fields.*allocation_observation")

    def test_completion_must_be_visible(self):
        for completion in ("Submitted", "NotCompleted", "Failed", None):
            with self.subTest(completion=completion):
                self.report["results"][0]["completion"] = completion
                self.reject("completion must be CompletedVisible")

    def test_unchanged_poisoned_writeback(self):
        self.report["results"][0]["writebacks"][0]["bytes_hex"] = "abababab"
        self.reject("copy_word writeback allocation 101/view 201.*offset 32.*expected 0x01, got 0xab")

    def test_output_mismatch_points_to_first_byte(self):
        self.report["results"][0]["writebacks"][0]["bytes_hex"] = "01230067"
        self.reject("offset 34.*expected 0x45, got 0x00")

    def test_truncated_writeback(self):
        self.report["results"][0]["writebacks"][0]["bytes_hex"] = "0123"
        self.reject("length mismatch: expected 4 bytes, got 2")

    def test_missing_extra_or_wrong_writeback(self):
        original = copy.deepcopy(self.report)
        self.report["results"][0]["writebacks"] = []
        self.reject("writable set mismatch")
        for key in ("allocation", "view", "offset"):
            with self.subTest(key=key):
                self.report = copy.deepcopy(original)
                self.report["results"][0]["writebacks"][0][key] += 1
                self.reject("writable set mismatch")
        self.report = original
        self.report["results"][0]["writebacks"].append({
            "allocation": 100, "view": 200, "offset": 16, "bytes_hex": "01234567",
        })
        self.reject("writable set mismatch")

    def test_duplicate_writeback(self):
        writes = self.report["results"][0]["writebacks"]
        writes.append(copy.deepcopy(writes[0]))
        self.reject("duplicate writeback")

    def test_writeback_order_must_match_manifest(self):
        case = self.suite["cases"][0]
        case["buffers"].append({
            "binding": 2, "allocation": 102, "view": 202, "offset": 16, "length": 4,
            "allocation_size": 36, "access": "write", "initial_hex": "aaaaaaaa",
        })
        case["expected_writebacks"].append({
            "allocation": 102, "view": 202, "offset": 16, "bytes_hex": "11223344",
        })
        self.report = synthetic_report(self.suite, self.digest)
        self.report["results"][0]["writebacks"].reverse()
        self.reject("writeback order differs")

    def test_read_only_and_canary_mutations(self):
        original = copy.deepcopy(self.report)
        for allocation_index, offset in ((0, 16), (0, 0), (1, 0), (1, 36), (1, 51)):
            with self.subTest(allocation_index=allocation_index, offset=offset):
                self.report = copy.deepcopy(original)
                allocation = self.report["results"][0]["allocations"][allocation_index]
                data = bytearray.fromhex(allocation["bytes_hex"])
                data[offset] ^= 0xFF
                allocation["bytes_hex"] = data.hex()
                self.reject(f"copy_word allocation {allocation['allocation']}.*offset {offset}:")

    def test_correct_writeback_cannot_hide_unchanged_full_allocation(self):
        allocation = self.report["results"][0]["allocations"][1]
        data = bytearray.fromhex(allocation["bytes_hex"])
        data[32:36] = bytes.fromhex("abababab")
        allocation["bytes_hex"] = data.hex()
        self.reject("copy_word allocation 101.*offset 32:")

    def test_missing_duplicate_unknown_and_truncated_allocations(self):
        original = copy.deepcopy(self.report)
        self.report["results"][0]["allocations"].pop(0)
        self.reject("missing allocations.*100")
        self.report = copy.deepcopy(original)
        allocations = self.report["results"][0]["allocations"]
        allocations.append(copy.deepcopy(allocations[0]))
        self.reject("duplicate allocation 100")
        self.report = copy.deepcopy(original)
        self.report["results"][0]["allocations"][0]["allocation"] = 999
        self.reject("unknown allocation 999")
        self.report = original
        self.report["results"][0]["allocations"][0]["bytes_hex"] = "01"
        self.reject("length mismatch: expected 36 bytes, got 1")

    def test_malformed_hex_and_schema(self):
        for malformed in ("0", "zz", "01 23", None):
            with self.subTest(malformed=malformed):
                self.report["results"][0]["writebacks"][0]["bytes_hex"] = malformed
                self.reject("even-length hexadecimal")
        self.report = synthetic_report(self.suite, self.digest)
        self.report["schema_version"] = True
        self.reject("unsupported schema_version")

    def test_empty_device_or_platform(self):
        for key in ("device", "platform"):
            with self.subTest(key=key):
                self.report = synthetic_report(self.suite, self.digest)
                self.report[key] = " "
                self.reject(f"capture.{key}: expected nonempty string")

    def test_overlapping_suite_initialization_is_rejected(self):
        buffer = copy.deepcopy(self.suite["cases"][0]["buffers"][0])
        buffer["view"] = 999
        buffer["binding"] = 2
        self.suite["cases"][0]["buffers"].append(buffer)
        self.reject("overlapping initialization would depend on write order")

    def test_suite_allocation_size_cap_precedes_allocation(self):
        for size in (compare.MAX_ALLOCATION_BYTES + 1, compare.U64_MAX):
            with self.subTest(size=size):
                self.suite["cases"][0]["buffers"][0]["allocation_size"] = size
                self.reject("allocation_size: expected integer in 1..1048576")

    def test_u64_integer_bounds_and_bool_rejection(self):
        original = copy.deepcopy(self.report)
        for field in ("allocation", "view", "offset"):
            for value in (-1, compare.U64_MAX + 1, True):
                with self.subTest(field=field, value=value):
                    self.report = copy.deepcopy(original)
                    self.report["results"][0]["writebacks"][0][field] = value
                    self.reject(f"{field}: expected integer in 0..{compare.U64_MAX}")
        self.report = original
        self.suite["cases"][0]["buffers"][0]["allocation"] = compare.U64_MAX + 1
        self.reject(f"allocation: expected integer in 0..{compare.U64_MAX}")

    def test_cli_single_check_and_synthetic_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite_path, native_path, vulkan_path = (root / name for name in ("suite.json", "native.json", "vulkan.json"))
            suite_path.write_bytes(self.raw)
            native = synthetic_report(self.suite, self.digest, "native-metal")
            native_path.write_text(json.dumps(native), encoding="utf-8")
            vulkan_path.write_text(json.dumps(self.report), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = compare.main(["--suite", str(suite_path), "--check", str(vulkan_path)])
            self.assertEqual(status, 0)
            self.assertTrue(output.getvalue().startswith("PASS capture:"))
            self.assertNotIn("parity", output.getvalue())
            self.assertIn("host-visible bytes agreement", output.getvalue())
            self.assertIn("host-writeback-landing", output.getvalue())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = compare.main(["--suite", str(suite_path), "--native", str(native_path),
                                       "--vulkan", str(vulkan_path)])
            self.assertEqual(status, 0)
            self.assertTrue(output.getvalue().startswith("PASS parity:"))
            self.assertIn("native GPU buffer readback / Vulkan host writeback landing", output.getvalue())
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                status = compare.main(["--suite", str(suite_path), "--native", str(vulkan_path),
                                       "--vulkan", str(vulkan_path)])
            self.assertEqual(status, 1)
            self.assertIn("expected backend native-metal", errors.getvalue())

    def test_duplicate_json_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"backend":"native-metal","backend":"vulkan"}', encoding="utf-8")
            with self.assertRaisesRegex(compare.CaptureError, "duplicate object key 'backend'"):
                compare._read_json(path)


class SharedProviderTests(unittest.TestCase):
    """Synthetic three-party captures test validation, never Metal execution."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepare_suite(SUITE_PATH)

    def prepare_suite(self, path):
        raw = path.read_bytes()
        self.suite = json.loads(raw)
        self.digest = hashlib.sha256(raw).hexdigest()
        self.suite_path = self.root / "suite.json"
        self.suite_path.write_bytes(raw)
        self.paths = {}
        for backend in compare.ALLOCATION_OBSERVATIONS:
            self.paths[backend] = self.root / f"{backend}.json"
            self.write_report(backend, synthetic_report(self.suite, self.digest, backend))

    def write_report(self, backend, report):
        self.paths[backend].write_text(json.dumps(report), encoding="utf-8")

    def compare_args(self):
        return ["--native", str(self.paths["native-metal"]),
                "--vulkan", str(self.paths["vulkan"]),
                "--metal-provider", str(self.paths["native-metal-provider"])]

    def run_cli(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = compare.main(["--suite", str(self.suite_path), *arguments])
        return status, output.getvalue(), errors.getvalue()

    def test_three_captures_validate_for_both_suite_versions(self):
        for name in ("suite.json", "suite-v2.json"):
            with self.subTest(suite=name):
                self.prepare_suite(SUITE_PATH.with_name(name))
                status, output, errors = self.run_cli(self.compare_args())
                self.assertEqual(status, 0, errors)
                self.assertIn("PASS parity: native-metal / vulkan / native-metal-provider;", output)
                self.assertIn("host-visible bytes agreement", output)
                self.assertIn("Swift native GPU buffer readback", output)
                self.assertIn("Rust Metal provider host writeback landing", output)
                self.assertNotIn("full allocation GPU", output)

    def test_single_metal_provider_capture_does_not_claim_parity(self):
        status, output, errors = self.run_cli(["--check", str(self.paths["native-metal-provider"])])
        self.assertEqual(status, 0, errors)
        self.assertIn("PASS capture: native-metal-provider;", output)
        self.assertIn("allocations=host-writeback-landing", output)
        self.assertNotIn("parity", output)

    def test_invalid_third_capture_prevents_any_success_message(self):
        for mutation, expected in (
                (lambda report: report.update(suite_sha256="0" * 64), "suite_sha256 mismatch"),
                (lambda report: report["results"].pop(), "missing cases"),
                (lambda report: report["results"][0]["writebacks"][0].update(bytes_hex="0123"),
                 "length mismatch"),
                (lambda report: report.update(allocation_observation="gpu-buffer-readback"),
                 "native-metal-provider requires allocation_observation host-writeback-landing"),
                (lambda report: report.update(backend="native-metal", allocation_observation="gpu-buffer-readback"),
                 "expected backend native-metal-provider")):
            with self.subTest(expected=expected):
                report = synthetic_report(self.suite, self.digest, "native-metal-provider")
                mutation(report)
                self.write_report("native-metal-provider", report)
                status, output, errors = self.run_cli(self.compare_args())
                self.assertEqual(status, 1)
                self.assertEqual(output, "")
                self.assertIn(expected, errors)

    def test_provider_cannot_replace_swift_reference_or_vulkan(self):
        for replaced in ("native-metal", "vulkan"):
            with self.subTest(replaced=replaced):
                paths = dict(self.paths)
                paths[replaced] = self.paths["native-metal-provider"]
                arguments = ["--native", str(paths["native-metal"]), "--vulkan", str(paths["vulkan"])]
                status, output, errors = self.run_cli(arguments)
                self.assertEqual(status, 1)
                self.assertEqual(output, "")
                self.assertIn(f"expected backend {replaced}, got native-metal-provider", errors)

    def test_third_capture_requires_both_original_reports(self):
        for remaining in ([], ["--vulkan", str(self.paths["vulkan"])],
                          ["--native", str(self.paths["native-metal"])],
                          ["--check", str(self.paths["native-metal-provider"])]):
            with self.subTest(remaining=remaining):
                arguments = ["--metal-provider", str(self.paths["native-metal-provider"]), *remaining]
                with self.assertRaises(SystemExit) as error:
                    self.run_cli(arguments)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
