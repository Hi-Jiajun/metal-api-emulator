"""Gate 2 G2-f class-parity tests, over synthetic captures only.

The captures here are fabricated from the suites' own declarations, exactly
like `test_compare.py`'s: they exercise the declaration pins, the covered
rules, the rail-to-rail byte comparison and the CLI without ever claiming
hardware evidence.
"""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import compare
import narrow_class


CONFORMANCE = Path(__file__).resolve().parent


def initial_bytes(buffer):
    """The bytes one declared view pre-seeds, in whichever form it states."""
    if "initial_repeat_hex" in buffer:
        pattern = bytes.fromhex(buffer["initial_repeat_hex"])
        return pattern * (buffer["length"] // len(pattern))
    return bytes.fromhex(buffer["initial_hex"])


def fabricate(suite, digest, backend):
    """A capture for every case in one suite, from the suite's own expectations."""
    results = []
    for case in suite.get("cases", []):
        allocations = {}
        for buffer in case["buffers"]:
            allocation = buffer["allocation"]
            data = allocations.setdefault(
                allocation, bytearray([suite["guard_byte"]]) * buffer["allocation_size"])
            offset = buffer["offset"]
            data[offset:offset + buffer["length"]] = initial_bytes(buffer)
        for write in case["expected_writebacks"]:
            data = bytes.fromhex(write["bytes_hex"])
            allocations[write["allocation"]][write["offset"]:write["offset"] + len(data)] = data
        results.append({
            "id": case["id"],
            "completion": "CompletedVisible",
            "writebacks": copy.deepcopy(case["expected_writebacks"]),
            "allocations": [{"allocation": allocation,
                             "bytes_hex": bytes(data).hex()}
                            for allocation, data in allocations.items()],
        })
    for case in suite.get("render_cases", []):
        attachment = case.get("attachment")
        if not isinstance(attachment, dict):
            raise unittest.SkipTest("fabrication covers one colour attachment only")
        results.append({
            "id": case["id"],
            "completion": "CompletedVisible",
            "writebacks": [{"allocation": attachment["allocation"],
                            "view": attachment["view"],
                            "offset": 0,
                            "bytes_hex": case["expected_hex"]}],
            "allocations": [{"allocation": attachment["allocation"],
                             "bytes_hex": case["expected_hex"]}],
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


def run_tool(arguments):
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = narrow_class.main(arguments)
    return code, stdout.getvalue(), stderr.getvalue()


class DeclarationTests(unittest.TestCase):
    def setUp(self):
        self.declaration = narrow_class.load_declaration()

    def class_named(self, name):
        return next(entry for entry in self.declaration["classes"] if entry["name"] == name)

    def test_every_class_claims_the_five_rails(self):
        for entry in self.declaration["classes"]:
            with self.subTest(entry["name"]):
                self.assertEqual(entry["required_rails"], list(compare.ALLOCATION_OBSERVATIONS))

    def test_declared_fixtures_are_covered_and_pinned(self):
        for entry in self.declaration["classes"]:
            with self.subTest(entry=entry["name"]):
                suite_path, raw, suite = narrow_class.check_class(entry)
                # The class's own suite is the file the fixtures were checked
                # against, and every declared fixture is a case of it.
                self.assertEqual(suite_path, (CONFORMANCE / entry["suite"]).resolve())
                for case_id in entry["fixtures"]:
                    kind, _ = narrow_class.find_case(suite, case_id)
                    self.assertIn(kind, ("compute", "render"))
                self.assertEqual(hashlib.sha256(raw).hexdigest(),
                                 hashlib.sha256(suite_path.read_bytes()).hexdigest())

    def test_a_declaration_cannot_shrink_the_rail_set(self):
        declaration = copy.deepcopy(self.declaration)
        declaration["classes"][0]["required_rails"] = list(compare.ALLOCATION_OBSERVATIONS)[:4]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "narrow-class.json"
            path.write_text(json.dumps(declaration), encoding="utf-8")
            with self.assertRaisesRegex(narrow_class.ClassParityError,
                                        "required_rails has to be"):
                narrow_class.load_declaration(path)

    def test_a_declaration_fixture_that_is_not_in_its_suite_is_refused(self):
        entry = copy.deepcopy(self.class_named("compute-buffer-narrow"))
        entry["fixtures"] = ["copy_word_typo"]
        with self.assertRaisesRegex(narrow_class.ClassParityError, "class-fixture-pin"):
            narrow_class.check_class(entry)

    def test_a_source_pin_that_the_tree_does_not_hash_is_refused(self):
        entry = self.class_named("compute-buffer-narrow")
        suite_path, _, suite = narrow_class.check_class(entry)
        tampered = copy.deepcopy(suite)
        tampered["cases"][0]["air"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(narrow_class.ClassParityError, "class-fixture-pin"):
            narrow_class.check_fixture(entry, tampered, suite_path, "copy_word")


class CoveredRuleTests(unittest.TestCase):
    def setUp(self):
        self.declaration = narrow_class.load_declaration()

    def class_named(self, name):
        return next(entry for entry in self.declaration["classes"] if entry["name"] == name)

    def test_out_of_class_neighbours_are_refused_by_their_rule(self):
        for entry in self.declaration["classes"]:
            for refusal in entry["refusals"]:
                with self.subTest(class_name=entry["name"], case=refusal["case"]):
                    path = CONFORMANCE / refusal["suite"]
                    suite = json.loads(path.read_bytes())
                    with self.assertRaises(narrow_class.ClassParityError) as caught:
                        narrow_class.check_fixture(entry, suite, path, refusal["case"])
                    self.assertEqual(caught.exception.rule, refusal["rule"])

    def test_a_second_dispatch_is_out_of_the_compute_class(self):
        entry = self.class_named("compute-buffer-narrow")
        suite_path, _, suite = narrow_class.check_class(entry)
        tampered = copy.deepcopy(suite)
        tampered["cases"][0]["dispatches"] = [{"grid": [1, 1, 1], "local": [1, 1, 1]}]
        with self.assertRaisesRegex(narrow_class.ClassParityError, "compute-covered-dispatch"):
            narrow_class.check_fixture(entry, tampered, suite_path, "copy_word")

    def test_a_rounding_up_grid_is_out_of_the_compute_class(self):
        entry = self.class_named("compute-buffer-narrow")
        suite_path, _, suite = narrow_class.check_class(entry)
        tampered = copy.deepcopy(suite)
        tampered["cases"][0]["local"] = [2, 1, 1]
        with self.assertRaisesRegex(narrow_class.ClassParityError, "compute-covered-grid"):
            narrow_class.check_fixture(entry, tampered, suite_path, "copy_word")

    def test_render_state_and_draw_are_covered_axes(self):
        entry = self.class_named("render-narrow")
        suite_path, _, suite = narrow_class.check_class(entry)
        fixture = entry["seam_fixture"]
        for mutation, rule in ((lambda case: case.update(multisample={"sample_count": 4}),
                                "render-covered-state"),
                               (lambda case: case.pop("indices"), "render-covered-draw"),
                               (lambda case: case["attachment"].update(load="load"),
                                "render-covered-attachment")):
            with self.subTest(rule):
                tampered = copy.deepcopy(suite)
                case = next(case for case in tampered["render_cases"]
                            if case["id"] == fixture)
                mutation(case)
                with self.assertRaises(narrow_class.ClassParityError) as caught:
                    narrow_class.check_fixture(entry, tampered, suite_path, fixture)
                self.assertEqual(caught.exception.rule, rule)


class RailParityTests(unittest.TestCase):
    def setUp(self):
        self.declaration = narrow_class.load_declaration()

    def class_named(self, name):
        return next(entry for entry in self.declaration["classes"] if entry["name"] == name)

    def write_captures(self, directory, entry, mutate=None):
        suite_path, raw, suite = narrow_class.check_class(entry)
        digest = hashlib.sha256(raw).hexdigest()
        paths = {}
        for rail in entry["required_rails"]:
            report = fabricate(suite, digest, rail)
            if mutate is not None:
                report = mutate(rail, report) or report
            path = Path(directory) / (rail + ".json")
            path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
            paths[rail] = path
        return paths

    def five_rail_arguments(self, entry, paths):
        arguments = ["--class", entry["name"]]
        for rail in entry["required_rails"]:
            if rail in paths:
                arguments += ["--rail", f"{rail}={paths[rail]}"]
        return arguments

    def test_both_classes_pass_the_five_rail_parity(self):
        for name in ("compute-buffer-narrow", "render-narrow"):
            entry = self.class_named(name)
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                paths = self.write_captures(directory, entry)
                code, stdout, stderr = run_tool(self.five_rail_arguments(entry, paths))
                self.assertEqual(code, 0, stderr)
                self.assertTrue(stdout.startswith("PASS class-parity: " + name), stdout)
                self.assertIn("rails=" + "/".join(entry["required_rails"]), stdout)

    def test_a_rail_that_differs_in_one_byte_is_refused(self):
        entry = self.class_named("compute-buffer-narrow")
        with tempfile.TemporaryDirectory() as directory:
            def mutate(rail, report):
                if rail != "vulkan":
                    return report
                writeback = report["results"][0]["writebacks"][0]
                writeback["bytes_hex"] = "00" + writeback["bytes_hex"][2:]
                return report
            paths = self.write_captures(directory, entry, mutate)
            code, stdout, stderr = run_tool(self.five_rail_arguments(entry, paths))
            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("rail vulkan", stderr)
            # The suite check refuses the byte before the rail-to-rail
            # comparison can name it: every rail is held to the suite's own
            # declaration, and the A/B statement then follows from the same
            # declared bytes.
            self.assertIn("writeback allocation 101/view 201", stderr)
            self.assertIn("first differing byte", stderr)

    def test_a_rail_that_differs_in_an_allocation_image_is_refused(self):
        entry = self.class_named("compute-buffer-narrow")
        with tempfile.TemporaryDirectory() as directory:
            def mutate(rail, report):
                if rail != "vulkan-objects":
                    return report
                image = report["results"][0]["allocations"][0]
                image["bytes_hex"] = "ff" + image["bytes_hex"][2:]
                return report
            paths = self.write_captures(directory, entry, mutate)
            code, _, stderr = run_tool(self.five_rail_arguments(entry, paths))
            self.assertEqual(code, 1)
            self.assertIn("case copy_word allocation 100", stderr)
            self.assertIn("first differing byte", stderr)

    def test_the_rail_comparison_names_the_first_differing_byte(self):
        # A suite whose expectation is a set (compare.py's wildcards) is the
        # case the direct rail-to-rail comparison exists for: both rails can
        # satisfy the suite and still disagree with each other. Exercise that
        # path directly, with the same tuple shape _observation builds.
        reference = {"writebacks": [(9, 1, 0, ("hex", "aabb"))], "allocations": []}
        other = {"writebacks": [(9, 1, 0, ("hex", "aacc"))], "allocations": []}
        with self.assertRaisesRegex(narrow_class.ClassParityError,
                                    "differs from rail native-metal at byte 1"):
            narrow_class._require_same("compute-buffer-narrow", "copy_word", "native-metal",
                                       "vulkan", reference, other)

    def test_a_missing_rail_is_incomplete_and_preview_never_claims_parity(self):
        entry = self.class_named("compute-buffer-narrow")
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_captures(directory, entry)
            short = {rail: paths[rail] for rail in ("vulkan", "vulkan-objects")}
            code, stdout, stderr = run_tool(self.five_rail_arguments(entry, short))
            self.assertEqual(code, 1)
            self.assertIn("missing rails", stderr)
            self.assertIn("native-metal", stderr)
            code, stdout, stderr = run_tool(
                self.five_rail_arguments(entry, short) + ["--preview"])
            self.assertEqual(code, 0, stderr)
            self.assertIn("no parity claim", stdout)
            self.assertNotIn("PASS", stdout)
            self.assertIn("PREVIEW", stdout)

    def test_a_stale_suite_digest_or_wrong_backend_is_refused(self):
        entry = self.class_named("compute-buffer-narrow")
        with tempfile.TemporaryDirectory() as directory:
            def stale(rail, report):
                if rail != "vulkan":
                    return report
                report["suite_sha256"] = "0" * 64
                return report
            paths = self.write_captures(directory, entry, stale)
            code, _, stderr = run_tool(self.five_rail_arguments(entry, paths))
            self.assertEqual(code, 1)
            self.assertIn("suite_sha256 mismatch", stderr)
            def wrong_backend(rail, report):
                if rail != "vulkan":
                    return report
                report["backend"] = "vulkan-objects"
                report["allocation_observation"] = compare.ALLOCATION_OBSERVATIONS[
                    "vulkan-objects"]
                return report
            paths = self.write_captures(directory, entry, wrong_backend)
            code, _, stderr = run_tool(self.five_rail_arguments(entry, paths))
            self.assertEqual(code, 1)
            self.assertIn("expected backend vulkan", stderr)

    def test_an_unknown_or_duplicated_rail_argument_is_refused(self):
        with self.assertRaisesRegex(narrow_class.ClassParityError, "class-parity-rails"):
            narrow_class.parse_rail("metal=/tmp/whatever.json")
        entry = self.class_named("compute-buffer-narrow")
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_captures(directory, entry)
            arguments = self.five_rail_arguments(entry, paths)
            arguments += ["--rail", f"vulkan={paths['vulkan']}"]
            code, _, stderr = run_tool(arguments)
            self.assertEqual(code, 1)
            self.assertIn("given twice", stderr)


if __name__ == "__main__":
    unittest.main()
