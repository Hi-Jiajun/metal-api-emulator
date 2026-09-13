"""Suite-coverage checks that run without macOS, Swift, a GPU or a compiler.

`NativeOracle.swift` is compiled and executed only by the macOS CI job, so a
Linux checkout cannot run it. Its accepted suite list and its per-suite case
list are hand-written anyway, and the same case lists appear again in the Rust
provider and in the CI workflow. A suite or case added to the JSON fixtures and
the shaders can therefore leave one runner behind while every executable check
still passes:

- `NativeOracle.swift` (`loadSuite`) pins the suite identities and their case ids,
- `examples/metal-smoke/src/bin/provider-capture.rs` (`validate_suite`) pins the
  same case ids for the Rust providers used by the Vulkan and native-metal rails,
- `.github/workflows/ci.yml` names the suite file every rail runs.

This module compares those three tables against `conformance/suite*.json` by
reading the sources as text. It is a metadata consistency check. It proves
nothing about Metal execution, MSL/AIR equivalence or conformance, and it cannot
see a case body that only the Swift compiler would reject. Text extraction is
strict on purpose: if a table is reshaped so it can no longer be parsed, the
check fails instead of silently comparing empty sets.
"""

import hashlib
import json
from pathlib import Path
import re
import unittest


CONFORMANCE = Path(__file__).resolve().parent
REPOSITORY = CONFORMANCE.parent
ORACLE_PATH = CONFORMANCE / "NativeOracle.swift"
PROVIDER_PATH = REPOSITORY / "examples" / "metal-smoke" / "src" / "bin" / "provider-capture.rs"
WORKFLOW_PATH = REPOSITORY / ".github" / "workflows" / "ci.yml"

SUITE_FILES = "suite*.json"
SUITE_IDENTITY = re.compile(r"compute-buffer-v([0-9]+)")

# The one capture backend that reports an attachment observation in the first
# render increment, and the ones that still cannot: the two object-API rails
# carry no render command encoder and the macOS rails have not run the render
# path on Apple hardware. A render case whose marker names only `vulkan` is
# therefore deliberately outside the macOS CI rails until
# `conformance/RENDER-CAPTURE.md` §4 lands.
VULKAN_TRACE_RAIL = "vulkan"
PENDING_RENDER_RAILS = ("native-metal", "native-metal-provider",
                        "native-metal-provider-objects", "vulkan-objects")

# The CI rail names that cannot carry an attachment observation yet. The
# Vulkan object-API rails are the exception: they run the same provider as the
# Vulkan trace rail, so they do report one.
RAILS_WITHOUT_ATTACHMENT_OBSERVATION = frozenset({
    "native oracle capture",
    "native-metal provider capture",
    "three-way parity comparison",
})


def _rail_cannot_report_attachments(rail_name):
    """Whether a CI rail is unable to report a render attachment today."""
    return rail_name in RAILS_WITHOUT_ATTACHMENT_OBSERVATION

# Every command rail in the CI workflow that writes its suites out explicitly.
# Order matters: a parity line also contains `compare.py --suite`, and the
# object-API lines contain a `--bin provider-capture --` prefix, so the first
# matching marker decides the rail. The object-API rails take their suite from a
# shell variable and are checked by the version-loop test instead.
CI_RAILS = (
    ("native oracle capture", "conformance/run_native.py"),
    ("Vulkan provider capture", "--bin provider-capture --"),
    ("native-metal provider capture", "--backend native-metal-provider"),
    ("three-way parity comparison", "--vulkan-objects"),
)

# `for version in ...; do` lists the object-API rails expand into suite files.
CI_VERSION_LOOP = re.compile(r"for version in ([0-9 ]+); do")
CI_EXPECTED_LOOPS = 4


def _suite_identity_from_name(name):
    """Return the suite identity a committed suite file name must declare."""
    if name == "suite.json":
        return "compute-buffer-v1"
    match = re.fullmatch(r"suite-v([0-9]+)\.json", name)
    if match is None:
        raise AssertionError("unrecognized suite file name " + name)
    return "compute-buffer-v" + match.group(1)


def _version(identity):
    match = SUITE_IDENTITY.fullmatch(identity)
    if match is None:
        raise AssertionError("unrecognized suite identity " + identity)
    return int(match.group(1))


def _ordered(identities):
    return sorted(identities, key=_version)


def _suite_difference(expected, found):
    """Describe a suite set mismatch by naming the missing and extra ids."""
    missing = _ordered(expected - found)
    extra = _ordered(found - expected)
    return ("missing " + (", ".join(missing) if missing else "none")
            + "; unexpected " + (", ".join(extra) if extra else "none"))


def _case_difference(expected, found):
    """Describe a case set mismatch by naming the missing and extra ids."""
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    return ("missing case ids " + (", ".join(missing) if missing else "none")
            + "; unexpected case ids " + (", ".join(extra) if extra else "none"))


def _pin_difference(expected, found):
    """Describe a source-pin mismatch by naming the missing and extra pins."""
    missing = sorted("%s %s" % pin for pin in expected - found)
    extra = sorted("%s %s" % pin for pin in found - expected)
    return ("missing " + (", ".join(missing) if missing else "none")
            + "; unexpected " + (", ".join(extra) if extra else "none"))


def suite_fixtures():
    """Map every committed suite identity to its declared case ids.

    The file name is checked as well: `suite.json` is v1 and `suite-vN.json` is
    vN, so a copy placed under a new name cannot pass as a new version.
    """
    fixtures = {}
    for path in sorted(CONFORMANCE.glob(SUITE_FILES)):
        document = json.loads(path.read_text(encoding="utf-8"))
        identity = document["suite"]
        expected = _suite_identity_from_name(path.name)
        if identity != expected:
            raise AssertionError("%s declares %s, but its name means %s"
                                 % (path.name, identity, expected))
        if identity in fixtures:
            raise AssertionError("two suite files declare " + identity)
        fixtures[identity] = [case["id"] for case in document["cases"]]
    if not fixtures:
        raise AssertionError("no suite JSON files found in " + str(CONFORMANCE))
    return fixtures


def locally_reported_suites():
    """Suites whose render cases are marked for rails this checkout can run.

    A suite that declares render cases and marks every one of them for the
    Vulkan trace rail only is reported by the local/lavapipe Vulkan rails; the
    macOS capture rails stay pending (`conformance/RENDER-CAPTURE.md` §4). The
    marker is read from the suite JSON rather than inferred, so a suite that
    never declares a render case keeps the full four-rail coverage, and a suite
    that widens the marker to a macOS rail starts failing the CI-coverage tests
    instead of silently leaving that rail behind.
    """
    pending = set()
    for path in sorted(CONFORMANCE.glob(SUITE_FILES)):
        document = json.loads(path.read_text(encoding="utf-8"))
        rails = set()
        for case in document.get("render_cases", []):
            rails.update(case["capture_rails"])
        if rails and rails <= {VULKAN_TRACE_RAIL}:
            pending.add(document["suite"])
        elif rails:
            raise AssertionError(
                "%s marks render cases for the still-pending capture rails %s; wire them "
                "into every CI rail first" % (document["suite"],
                                              ", ".join(sorted(rails))))
    if not pending:
        raise AssertionError("no committed suite declares a Vulkan-only render case")
    return pending


def suite_source_pins():
    """Every reviewed source identity the suite JSON files declare.

    A case names its own `air`/`metal` pair and a v7+ case may add a `programs`
    table with one pair per program. Both are collected: a program entry does
    not replace the case's own pair.
    """
    pins = set()
    for path in sorted(CONFORMANCE.glob(SUITE_FILES)):
        document = json.loads(path.read_text(encoding="utf-8"))
        for case in document["cases"]:
            for definition in [case, *(case.get("programs") or [])]:
                for kind in ("air", "metal"):
                    pins.add((definition[kind]["path"], definition[kind]["sha256"]))
    if not pins:
        raise AssertionError("no suite file declares a reviewed source pin")
    return pins


def _swift_table(text):
    """Return the loadSuite suite/case table from the oracle source text."""
    start = text.find("let expectedIDs: Set<String>")
    if start < 0:
        raise AssertionError("NativeOracle.swift no longer declares expectedIDs")
    start = text.find("switch suite.suite {", start)
    if start < 0:
        raise AssertionError("NativeOracle.swift no longer switches on suite.suite")
    end = text.find("default:", start)
    if end < 0:
        raise AssertionError("the loadSuite switch has no default arm")
    block = text[start:end]
    arms = re.findall(r'case\s+"(compute-buffer-v[0-9]+)":\s*expectedIDs\s*=\s*\[(.*?)\]',
                      block, re.S)
    if len(arms) != block.count('case "compute-buffer-'):
        raise AssertionError("the loadSuite switch has an arm that is not a plain "
                             "expectedIDs list")
    if not arms:
        raise AssertionError("the loadSuite switch declares no suite")
    table = {}
    for identity, body in arms:
        if identity in table:
            raise AssertionError("loadSuite declares " + identity + " twice")
        ids = re.findall(r'"([A-Za-z0-9_]+)"', body)
        if not ids:
            raise AssertionError("loadSuite declares no case ids for " + identity)
        table[identity] = ids
    return table


def oracle_case_ids():
    """Suite identities and case ids the Swift oracle accepts, read as text."""
    return _swift_table(ORACLE_PATH.read_text(encoding="utf-8"))


def oracle_source_pins(text):
    """Every `reviewedProgram` source identity the Swift oracle pins."""
    pins = set(re.findall(r'SourceDefinition\(path:\s*"([^"]+)",\s*\n?\s*'
                          r'sha256:\s*"([0-9a-f]{64})"\)', text))
    if len(pins) != text.count("SourceDefinition(path:"):
        raise AssertionError("a NativeOracle.swift source pin is not a plain path/sha256 pair")
    if not pins:
        raise AssertionError("NativeOracle.swift no longer pins reviewed sources")
    return pins


def _rust_table(text):
    """Return the provider-capture validate_suite suite/case table as text."""
    marker = "let case_ids: &[&str] = match (suite.schema_version, suite.suite.as_str()) {"
    start = text.find(marker)
    if start < 0:
        raise AssertionError("provider-capture.rs no longer builds case_ids from the suite")
    end = text.find("_ => return Err(", start)
    if end < 0:
        raise AssertionError("the provider-capture case_ids match has no fallback arm")
    block = text[start:end]
    arms = re.findall(r'\(1,\s*"(compute-buffer-v[0-9]+)"\)\s*=>\s*&\[(.*?)\]', block, re.S)
    if len(arms) != block.count('(1, "compute-buffer-'):
        raise AssertionError("the provider-capture case_ids match has an arm that is not a "
                             "plain list")
    if not arms:
        raise AssertionError("the provider-capture case_ids match declares no suite")
    table = {}
    for identity, body in arms:
        if identity in table:
            raise AssertionError("provider-capture.rs declares " + identity + " twice")
        ids = re.findall(r'"([A-Za-z0-9_]+)"', body)
        if not ids:
            raise AssertionError("provider-capture.rs declares no case ids for " + identity)
        table[identity] = ids
    return table


def rust_provider_case_ids():
    """Suite identities and case ids the Rust provider accepts, read as text."""
    return _rust_table(PROVIDER_PATH.read_text(encoding="utf-8"))


def _workflow_lines():
    return WORKFLOW_PATH.read_text(encoding="utf-8").splitlines()


def _referenced_suites(line):
    """Suite identities named by `conformance/suite[-vN].json` on one line."""
    return {"compute-buffer-v" + (match.group(1) or "1")
            for match in re.finditer(r"conformance/suite(?:-v([0-9]+))?\.json", line)}


class SuiteCoverageTests(unittest.TestCase):
    """Every hand-written suite and case table must match the JSON fixtures."""

    def test_oracle_accepts_exactly_the_committed_suites(self):
        fixtures = set(suite_fixtures())
        accepted = set(oracle_case_ids())
        self.assertEqual(accepted, fixtures, "NativeOracle.swift suite coverage: "
                         + _suite_difference(fixtures, accepted))

    def test_oracle_pins_every_case_id_of_each_suite(self):
        fixtures = suite_fixtures()
        table = oracle_case_ids()
        for identity in _ordered(fixtures):
            with self.subTest(suite=identity):
                self.assertEqual(set(table.get(identity, ())), set(fixtures[identity]),
                                 identity + " case coverage: "
                                 + _case_difference(set(fixtures[identity]),
                                                    set(table.get(identity, ()))))

    def test_rust_provider_accepts_exactly_the_committed_suites(self):
        fixtures = set(suite_fixtures())
        accepted = set(rust_provider_case_ids())
        self.assertEqual(accepted, fixtures, "provider-capture.rs suite coverage: "
                         + _suite_difference(fixtures, accepted))

    def test_rust_provider_pins_every_case_id_of_each_suite(self):
        fixtures = suite_fixtures()
        table = rust_provider_case_ids()
        for identity in _ordered(fixtures):
            with self.subTest(suite=identity):
                self.assertEqual(set(table.get(identity, ())), set(fixtures[identity]),
                                 identity + " case coverage: "
                                 + _case_difference(set(fixtures[identity]),
                                                    set(table.get(identity, ()))))

    def test_oracle_source_pins_match_the_suites_and_the_shader_bytes(self):
        # The oracle re-checks these pins at runtime, but only on macOS and only
        # after loading the suite. Comparing them here catches a shader that was
        # re-hashed without updating the oracle, and an oracle pin left behind by
        # a suite edit, from Linux.
        declared = suite_source_pins()
        pinned = oracle_source_pins(ORACLE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(pinned, declared, "oracle source pins against the suite JSON: "
                         + _pin_difference(declared, pinned))
        for relative, digest in sorted(pinned):
            # Pinned paths are relative to this directory, as loadSuite resolves
            # them against the directory holding the suite file.
            source = CONFORMANCE / relative
            with self.subTest(source=relative):
                self.assertTrue(source.is_file(), "pinned source is missing: " + relative)
                actual = hashlib.sha256(source.read_bytes()).hexdigest()
                self.assertEqual(actual, digest, relative + ": the pinned SHA-256 does not "
                                 "match the committed bytes")

    def test_ci_rails_each_run_every_committed_suite(self):
        # A suite whose render cases are marked for the Vulkan trace rail only
        # is reported by the Vulkan rails -- including the CI ones -- and stays
        # out of the rails that cannot carry an attachment observation until the
        # macOS capture step lands (`conformance/RENDER-CAPTURE.md` §4). Every
        # other suite has to be named on all four.
        locally_rendered = locally_reported_suites()
        found = {name: set() for name, _ in CI_RAILS}
        for line in _workflow_lines():
            for name, marker in CI_RAILS:
                if marker in line:
                    found[name].update(_referenced_suites(line))
                    break
        for name, marker in CI_RAILS:
            with self.subTest(rail=name):
                self.assertTrue(found[name], name + ": no suite file is named on any line "
                                "with marker " + repr(marker))
                # The two object-API rails and the macOS oracle cannot report a
                # render attachment yet, so the Vulkan-only suites are expected
                # to be absent from them.
                expected = (set(suite_fixtures()) - locally_rendered
                            if _rail_cannot_report_attachments(name)
                            else set(suite_fixtures()))
                self.assertEqual(found[name], expected, name + " suite coverage: "
                                 + _suite_difference(expected, found[name]))

    def test_ci_version_loops_pin_every_committed_suite(self):
        locally_rendered = locally_reported_suites()
        committed = {"compute-buffer-v" + str(version)
                     for version in sorted(_version(identity) for identity in suite_fixtures())}
        expected_without_attachments = committed - locally_rendered
        loops = []
        for line in _workflow_lines():
            for match in CI_VERSION_LOOP.finditer(line):
                loops.append({"compute-buffer-v" + version
                              for version in match.group(1).split()})
        self.assertEqual(len(loops), CI_EXPECTED_LOOPS,
                         "the workflow must list every object-API version loop; update "
                         "CI_EXPECTED_LOOPS with ci.yml if a rail was added or removed")
        for index, loop in enumerate(loops):
            with self.subTest(loop=index):
                # The loops appear in workflow order: the two Vulkan object-API
                # captures read the attachment, the two Rust Metal captures do
                # not report one yet (`RENDER-CAPTURE.md` §4).
                expected = (expected_without_attachments if index >= 2 else committed)
                self.assertEqual(loop, expected, "version loop %d coverage: " % index
                                 + _suite_difference(expected, loop))

    def test_render_cases_name_only_the_rails_that_can_report_them(self):
        # The first render increment has exactly one executable rail
        # (`research/docs/23` §1.2). A suite may therefore mark its render cases
        # for the Vulkan trace rail only; a marker that adds a pending rail would
        # make a capture that cannot report the attachment look compliant, so it
        # is refused here instead.
        marked = {}
        for path in sorted(CONFORMANCE.glob(SUITE_FILES)):
            document = json.loads(path.read_text(encoding="utf-8"))
            rails = set()
            for case in document.get("render_cases", []):
                rails.update(case["capture_rails"])
            if rails:
                marked[document["suite"]] = rails
        self.assertTrue(marked, "no committed suite declares render_cases")
        for identity, rails in sorted(marked.items()):
            with self.subTest(suite=identity):
                self.assertIn(VULKAN_TRACE_RAIL, rails,
                              identity + " must be reportable by the Vulkan trace rail")
                self.assertEqual(rails & set(PENDING_RENDER_RAILS), set(),
                                 identity + " names a capture rail that cannot report an "
                                 "attachment yet")

    def test_oracle_supported_version_range_names_the_last_suite(self):
        # The default arm is the diagnostic a real capture prints when it is
        # handed an unknown suite, so its bounded range must not go stale.
        text = ORACLE_PATH.read_text(encoding="utf-8")
        match = re.search(r"through compute-buffer-v([0-9]+) are supported", text)
        self.assertIsNotNone(match, "the loadSuite default arm no longer names a bounded range")
        last = max(_version(identity) for identity in suite_fixtures())
        self.assertEqual(int(match.group(1)), last,
                         "the loadSuite diagnostic range must name the last committed suite")


if __name__ == "__main__":
    unittest.main()
