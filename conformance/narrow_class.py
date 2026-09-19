#!/usr/bin/env python3
"""Gate 2 G2-f: the narrow classes' A/B parity, derived from the five rails.

Gate 2's exit criterion G2-f asks for one A/B parity per API class: the same
semantics observed identically on the native Metal oracle (the Swift collector
on an Apple Paravirtual device), on the Vulkan provider (Lavapipe or an RTX
5060) and on the Rust native Metal provider, through both the trace and the
object API.

This tool adds no rail, no capture and no second observation of the same bytes.
It reads the captures the five rails already produce for a suite
(`conformance/compare.py`; the CI jobs capture one per rail per suite) and:

1. holds each declared fixture to its class's covered rules, so a declaration
   cannot drift outside the shape it claims, and re-hashes the fixture's own
   sources against the files in this tree;
2. validates every rail's capture with the ordinary comparator, so each rail is
   still held to the suite's own declared bytes, counts and markers;
3. compares the declared fixtures' observations rail to rail -- every writeback
   and post-pass allocation image, byte for byte -- which is the A/B statement
   the class needs.

The legs G2-f does not settle here are named in the declarations instead of
being implied: the refusal surface above the class gate, and any widening of
the class, which owes its own covered fixture.

Output is one `PASS class-parity` line per class, naming the fixtures and the
rails. `--preview` runs the same validation and the same rail-to-rail
comparison over whatever rails are present and prints `PREVIEW` instead: a host
without Metal can exercise the tool without ever printing a parity claim.

Exit status: 0 for a PASS or a PREVIEW, 1 for any refusal.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import compare


CONFORMANCE = Path(__file__).resolve().parent
DECLARATION = CONFORMANCE / "narrow-class.json"
# The five rails, in the order the parities are stated everywhere else. A
# declaration that names fewer (or others) is refused: G2-f's A/B is the
# oracle plus both providers through both APIs, and a class may not quietly
# shrink that claim.
RAILS = tuple(compare.ALLOCATION_OBSERVATIONS)

# The covered rules of the compute class. A case's second dispatch is the
# `dispatches`/`programs` sections; a binding outside storage buffers is one of
# the image, heap, indirect or command-buffer sections.
COMPUTE_SECTIONS = ("dispatches", "programs")
COMPUTE_BINDINGS = ("textures", "heap", "icb", "command_buffers")
COMPUTE_BUFFER_ACCESS = ("read", "write", "read_write")
# The covered attachment formats and the state keys the first reviewed render
# shape leaves inert (research/docs/26 §10.1). A key here is not "forbidden
# forever" -- it is an axis the declaration does not cover yet, so a case
# carrying it cannot stand in for the covered fixture.
REVIEWED_ATTACHMENT_FORMATS = ("rgba8_unorm", "bgra8_unorm")
RENDER_INERT_KEYS = (
    "multisample",
    "depth",
    "stencil",
    "depth_test",
    "stencil_test",
    "blend",
    "cull",
    "scissor",
    "coverage",
    "depth_resolve",
    "stencil_resolve",
    "wildcard_texels",
    "wildcard_allowed_texels",
    "requires_sample_count",
    "requires_depth_resolve_filter",
    "requires_stencil_resolve_filter",
    "fragment_textures",
)


class ClassParityError(ValueError):
    """A refusal from the declaration, a covered rule, a capture or a rail."""

    def __init__(self, rule, detail):
        super().__init__(rule + ": " + detail)
        self.rule = rule


def _refuse(rule, detail):
    raise ClassParityError(rule, detail)


def load_declaration(path=DECLARATION):
    """Read `narrow-class.json`, refusing a malformed or shrinking declaration.

    The duplicate-key refusal is `compare._read_json`'s: a declaration that
    names the same key twice is ambiguous rather than last-wins.
    """
    try:
        _, declaration = compare._read_json(Path(path))
    except (OSError, UnicodeError, json.JSONDecodeError, compare.CaptureError) as error:
        _refuse("declaration", f"{path}: {error}")
    if not isinstance(declaration, dict) or set(declaration) != {"schema_version", "title",
                                                                 "note", "classes"}:
        _refuse("declaration", f"{path}: expected schema_version, title, note and classes")
    if type(declaration["schema_version"]) is not int or declaration["schema_version"] != 1:
        _refuse("declaration", f"{path}: unsupported schema_version")
    classes = declaration["classes"]
    if not isinstance(classes, list) or not classes:
        _refuse("declaration", f"{path}: classes has to be a non-empty list")
    names = []
    for entry in classes:
        expected = {"name", "api", "summary", "suite", "fixtures", "seam_fixture", "seam",
                    "required_rails", "bytes", "counts", "refusals"}
        if not isinstance(entry, dict) or set(entry) != expected:
            _refuse("declaration", f"{path}: a class is not exactly {sorted(expected)}")
        if entry["api"] not in ("compute", "render"):
            _refuse("declaration", f"{path}: {entry['name']}: unknown api {entry['api']!r}")
        if entry["required_rails"] != list(RAILS):
            _refuse("declaration", f"{path}: {entry['name']}: required_rails has to be "
                     + " / ".join(RAILS))
        if not isinstance(entry["fixtures"], list) or not entry["fixtures"]:
            _refuse("declaration", f"{path}: {entry['name']}: fixtures has to be a "
                     "non-empty list")
        if entry["seam_fixture"] not in entry["fixtures"]:
            _refuse("declaration", f"{path}: {entry['name']}: seam_fixture has to name "
                     "one of the fixtures")
        if not isinstance(entry["refusals"], list):
            _refuse("declaration", f"{path}: {entry['name']}: refusals has to be a list")
        for refusal in entry["refusals"]:
            if not isinstance(refusal, dict) or set(refusal) != {"suite", "case", "rule",
                                                                 "why"}:
                _refuse("declaration", f"{path}: {entry['name']}: a refusal is not exactly "
                         "suite, case, rule and why")
        names.append(entry["name"])
    if len(set(names)) != len(names):
        _refuse("declaration", f"{path}: duplicate class names")
    return declaration


def find_case(suite, case_id):
    """Return (kind, case) for one suite case, refusing a missing or ambiguous id."""
    matches = [("compute", case) for case in suite.get("cases", []) if case.get("id") == case_id]
    matches += [("render", case) for case in suite.get("render_cases", [])
                if case.get("id") == case_id]
    if not matches:
        _refuse("class-fixture-pin", f"{suite['suite']}: no case {case_id!r}")
    if len(matches) > 1:
        _refuse("class-fixture-pin", f"{suite['suite']}: case {case_id!r} appears twice")
    return matches[0]


def covered_compute_case(case, where):
    """The covered rules of the compute-buffer narrow class, on one case."""
    for key in COMPUTE_SECTIONS:
        if key in case:
            _refuse("compute-covered-dispatch",
                    f"{where}: the {key} section is a chain of dispatches; the class's "
                    "one direct dispatch is the case's own grid and local")
    for key in COMPUTE_BINDINGS:
        if key in case:
            _refuse("compute-covered-bindings",
                    f"{where}: the {key} section is outside the class's storage buffers")
    grid, local = case.get("grid"), case.get("local")
    if not isinstance(grid, list) or not isinstance(local, list):
        _refuse("compute-covered-grid", f"{where}: the case declares no exact-thread grid")
    if len(grid) != 3 or len(local) != 3:
        _refuse("compute-covered-grid", f"{where}: grid and local have to name three axes")
    if any(type(value) is not int or value < 1 for value in grid + local):
        _refuse("compute-covered-grid", f"{where}: grid and local have to be positive")
    for axis, (threads, group) in enumerate(zip(grid, local)):
        if group > threads:
            _refuse("compute-covered-grid",
                    f"{where}: axis {axis} launches {group} threads of a {threads}-thread "
                    "grid; an exact-thread dispatch never rounds up")
    if not case.get("buffers"):
        _refuse("compute-covered-bindings", f"{where}: the class binds storage buffers")
    for buffer in case["buffers"]:
        if buffer.get("access") not in COMPUTE_BUFFER_ACCESS:
            _refuse("compute-covered-bindings",
                    f"{where}: binding {buffer.get('binding')} has access "
                    f"{buffer.get('access')!r}, not a storage buffer")
    if not case.get("expected_writebacks"):
        _refuse("compute-covered-bindings",
                f"{where}: the compared bytes are the writebacks, and the case declares none")


def covered_render_case(case, suite, where):
    """The covered rules of the render narrow class, on one render case."""
    attachment = case.get("attachment")
    if "attachments" in case or not isinstance(attachment, dict):
        _refuse("render-covered-attachment",
                f"{where}: the class carries exactly one colour attachment")
    if attachment.get("format") not in REVIEWED_ATTACHMENT_FORMATS:
        _refuse("render-covered-attachment",
                f"{where}: format {attachment.get('format')!r} is not one of "
                + " / ".join(REVIEWED_ATTACHMENT_FORMATS))
    if attachment.get("load") != "clear":
        _refuse("render-covered-attachment",
                f"{where}: load {attachment.get('load')!r} is not the class's Clear")
    if attachment.get("store") != "store":
        _refuse("render-covered-attachment",
                f"{where}: store {attachment.get('store')!r} is not the class's Store")
    if case.get("instance_count", 1) != 1:
        _refuse("render-covered-draw", f"{where}: the class draws one instance")
    if case.get("base_vertex", 0) != 0:
        _refuse("render-covered-draw", f"{where}: the class draws from base vertex 0")
    vertices = case.get("vertices")
    if vertices is None:
        _refuse("render-covered-draw", f"{where}: the case declares no vertex count")
    # The class's one draw has two arms (`research/docs/23` §3.3, v39; the
    # covered rule's own widening). The indexed arm names its count through the
    # index buffer, so the draw's vertex count is whatever its indices reach and
    # the referenced fixture is the reviewed quad. The non-indexed arm names
    # its vertices `0..vertices` directly: with no layout at all it is the
    # milestone's three-vertex triangle, and with a layout every per-vertex
    # stream has to cover the whole `vertices * stride` span — the same counting
    # rule the contract states and the stricter of the two footprint rules the
    # rails prove, so a stream that is one record long cannot stand in for the
    # shape the class claims.
    #
    # The layout-free count is bounded *below* by the triangle's three and not
    # fixed at it since 2026-09-19 (census v45's `vertex_span` bucket): the
    # contract admits every count from three up. What a *declared* fixture may
    # name is still three, because this declaration is the five-rail parity
    # claim: both native faces compile the reviewed `vertex_id` module, whose
    # position table carries exactly three entries, so the widened count runs on
    # the Vulkan rails alone and owes fixtures there
    # (`conformance/suite-v44.json`, `crates/metal-api-vulkan/tests/render_vertex_count_e2e.rs`).
    if case.get("indices") is None:
        layout = case.get("vertex_layout")
        if layout is None:
            if vertices != 3:
                _refuse("render-covered-draw",
                        f"{where}: the resourceless non-indexed draw is the three-vertex "
                        "triangle")
        else:
            if type(vertices) is not int or vertices < 3:
                _refuse("render-covered-draw",
                        f"{where}: a non-indexed draw of {vertices!r} vertices rasterizes no "
                        "triangle")
            streams = layout.get("buffers") if isinstance(layout, dict) else None
            bindings = case.get("vertex_buffers")
            if not isinstance(streams, list) or not streams or not isinstance(bindings, list):
                _refuse("render-covered-draw",
                        f"{where}: the class's non-indexed draw names its streams")
            if len(bindings) != len(streams):
                _refuse("render-covered-draw",
                        f"{where}: one binding per declared stream")
            for position, stream in enumerate(streams):
                stride = stream.get("stride") if isinstance(stream, dict) else None
                step = stream.get("step", "per_vertex") if isinstance(stream, dict) else None
                if type(stride) is not int or stride < 1:
                    _refuse("render-covered-draw",
                            f"{where}: stream {position} declares no stride")
                if step != "per_vertex":
                    _refuse("render-covered-draw",
                            f"{where}: the class's non-indexed draw advances every stream per "
                            "vertex")
                binding = bindings[position]
                length = binding.get("length") if isinstance(binding, dict) else None
                if type(length) is not int or length < vertices * stride:
                    _refuse("render-covered-draw",
                            f"{where}: stream {position} declares {length!r} bytes, fewer than "
                            f"the {vertices * stride} the draw reads")
    for key in RENDER_INERT_KEYS:
        if key in case:
            _refuse("render-covered-state",
                    f"{where}: the {key} state is outside the covered shape")
    declaring = case.get("declaring_case")
    if not isinstance(declaring, str):
        _refuse("render-covered-declaring",
                f"{where}: a colour attachment owes a declaring compute case")
    kind, declaring_case = find_case(suite, declaring)
    if kind != "compute":
        _refuse("render-covered-declaring", f"{where}: declaring case {declaring} is not a "
                "compute case")
    covered_compute_case(declaring_case, f"{where} declaring case {declaring}")


def covered_rails(class_decl, case, where):
    """Every declared rail owes the case; a marked case has to name them all."""
    owed = case.get("capture_rails", list(RAILS))
    if sorted(owed) != sorted(class_decl["required_rails"]):
        _refuse("class-covered-rails",
                f"{where}: the case is owed by {owed} and the class's parity needs "
                + " / ".join(class_decl["required_rails"]))


def pinned_sources(case, suite_path, where):
    """Re-hash the fixture's own sources against the tree the suite lives in."""
    for key in ("air", "metal"):
        source = case.get(key)
        if source is None:
            continue
        path = (suite_path.parent / source["path"]).resolve()
        if not path.is_file():
            _refuse("class-fixture-pin", f"{where}: {key} source {source['path']} is missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source["sha256"]:
            _refuse("class-fixture-pin",
                    f"{where}: {key} source {source['path']} hashes {digest} while the "
                    f"suite declares {source['sha256']}")


def check_fixture(class_decl, suite, suite_path, case_id):
    """Hold one declared fixture to its class's covered rules and its pins."""
    kind, case = find_case(suite, case_id)
    where = f"{suite['suite']} fixture {case_id}"
    if class_decl["api"] == "compute":
        if kind != "compute":
            _refuse("compute-covered-bindings",
                    f"{where}: the compute class's fixtures are compute cases")
        covered_compute_case(case, where)
    elif kind == "compute":
        # A render class names its declaring pass as a fixture of its own: the
        # declaring pass is compute work, and the class owes it the same
        # storage-buffer discipline.
        covered_compute_case(case, where)
    else:
        covered_render_case(case, suite, where)
    covered_rails(class_decl, case, where)
    pinned_sources(case, suite_path, where)
    return case


def check_class(class_decl):
    """Load the class's suite, check every fixture: (suite_path, raw, suite)."""
    suite_path = (CONFORMANCE / class_decl["suite"]).resolve()
    if not suite_path.is_file():
        _refuse("class-fixture-pin", f"{class_decl['name']}: suite {class_decl['suite']} "
                "is missing")
    # The same reader every other tool uses: a suite that names a key twice is
    # refused rather than read as last-wins.
    raw, suite = compare._read_json(suite_path)
    for case_id in class_decl["fixtures"]:
        check_fixture(class_decl, suite, suite_path, case_id)
    return suite_path, raw, suite


def _observation(report, case_id, backend):
    """One fixture's compared bytes in one rail: writebacks and allocation images."""
    result = next((entry for entry in report["results"] if entry.get("id") == case_id), None)
    if result is None:
        _refuse("class-parity-capture",
                f"rail {backend}: capture reports no result for fixture {case_id}")
    writebacks = []
    for entry in result["writebacks"]:
        writebacks.append((entry["allocation"], entry["view"], entry["offset"],
                           _image(entry)))
    allocations = []
    for entry in result["allocations"]:
        allocations.append((entry["allocation"], _image(entry)))
    return {"writebacks": sorted(writebacks), "allocations": sorted(allocations)}


def _image(entry):
    """The bytes of one capture observation, spelled whichever way the rails do."""
    if "bytes_hex" in entry:
        return ("hex", entry["bytes_hex"])
    return ("digest", entry["bytes_sha256"], entry["bytes_length"])


def _bytes_difference(left, right):
    """The first differing byte of two hex images, or None when one is a digest."""
    if left[0] != "hex" or right[0] != "hex":
        return None
    try:
        left_bytes, right_bytes = bytes.fromhex(left[1]), bytes.fromhex(right[1])
    except ValueError:
        return None
    for index, (lhs, rhs) in enumerate(zip(left_bytes, right_bytes)):
        if lhs != rhs:
            return index, lhs, rhs
    return None


def _require_same(class_name, case_id, reference, backend, left, right):
    """Refuse a rail whose observation of one fixture differs from the reference."""
    if left == right:
        return
    for section in ("writebacks", "allocations"):
        expected, actual = left[section], right[section]
        if expected == actual:
            continue
        if len(expected) != len(actual):
            _refuse("class-parity-bytes",
                    f"{class_name} fixture {case_id}: rail {backend} reports "
                    f"{len(actual)} {section} while rail {reference} reports {len(expected)}")
        for index, (lhs, rhs) in enumerate(zip(expected, actual)):
            if lhs == rhs:
                continue
            if section == "writebacks":
                identity = (f"writeback allocation {rhs[0]}/view {rhs[1]}/offset {rhs[2]}")
            else:
                identity = f"allocation {rhs[0]}"
            difference = _bytes_difference(lhs[-1], rhs[-1])
            if difference is not None:
                index_byte, expected_byte, actual_byte = difference
                _refuse("class-parity-bytes",
                        f"{class_name} fixture {case_id}: rail {backend} {identity} differs "
                        f"from rail {reference} at byte {index_byte} "
                        f"(0x{actual_byte:02x} against 0x{expected_byte:02x})")
            _refuse("class-parity-bytes",
                    f"{class_name} fixture {case_id}: rail {backend} {identity} differs from "
                    f"rail {reference}")
    _refuse("class-parity-bytes",
            f"{class_name} fixture {case_id}: rail {backend} disagrees with rail {reference}")


def compare_rails(class_decl, suite, digest, captures):
    """Validate every present rail, then compare the fixtures' bytes rail to rail.

    `captures` maps a rail name to its already-parsed report. Every rail is held
    to the suite by the ordinary comparator first, so this comparison never
    replaces that check; it states the A/B claim itself.
    """
    observations = {}
    for rail in class_decl["required_rails"]:
        report = captures.get(rail)
        if report is None:
            continue
        try:
            compare.validate_capture(suite, digest, report, required_backend=rail)
        except compare.CaptureError as error:
            # The comparator's refusals name the case and the byte; the rail is
            # what this tool adds, since a class parity reads several captures
            # at once and "which rail" is the first question a failure raises.
            _refuse("class-parity-capture", f"rail {rail}: {error}")
        observations[rail] = {case_id: _observation(report, case_id, rail)
                              for case_id in class_decl["fixtures"]}
    present = [rail for rail in class_decl["required_rails"] if rail in captures]
    reference = present[0]
    for case_id in class_decl["fixtures"]:
        for rail in present[1:]:
            _require_same(class_decl["name"], case_id, reference, rail,
                          observations[reference][case_id], observations[rail][case_id])
    return observations


def parse_rail(value):
    """`backend=path`, refusing an unknown or unnamed rail."""
    backend, separator, path = value.partition("=")
    if not separator or backend not in RAILS or not path:
        _refuse("class-parity-rails",
                f"--rail expects one of {' / '.join(RAILS)} followed by '=' and a path, "
                f"got {value!r}")
    return backend, Path(path)


def main(argv=None):
    declaration = load_declaration()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--declaration", type=Path, default=DECLARATION,
                        help="the class declarations (default: conformance/narrow-class.json)")
    parser.add_argument("--class", dest="class_name", required=True,
                        choices=[entry["name"] for entry in declaration["classes"]])
    parser.add_argument("--rail", action="append", default=[], metavar="BACKEND=PATH",
                        help="one capture per rail; repeat for the class's rails")
    parser.add_argument("--preview", action="store_true",
                        help="run over the rails that are present and print PREVIEW, "
                             "never a parity claim")
    args = parser.parse_args(argv)
    try:
        class_decl = next(entry for entry in declaration["classes"]
                          if entry["name"] == args.class_name)
        _, raw, suite = check_class(class_decl)
        digest = hashlib.sha256(raw).hexdigest()
        captures = {}
        for value in args.rail:
            backend, path = parse_rail(value)
            if backend in captures:
                _refuse("class-parity-rails", f"rail {backend} was given twice")
            _, captures[backend] = compare._read_json(path)
        missing = [rail for rail in class_decl["required_rails"] if rail not in captures]
        if missing and not args.preview:
            _refuse("class-parity-rails",
                    f"{class_decl['name']}: missing rails " + " / ".join(missing)
                    + " (a parity claim needs " + " / ".join(class_decl["required_rails"]) + ")")
        compare_rails(class_decl, suite, digest, captures)
        present = [rail for rail in class_decl["required_rails"] if rail in captures]
        reference = present[0]
        suffix = ("; missing rails=" + ",".join(missing) + " (no parity claim)"
                  if missing else "")
        print(("PREVIEW" if missing else "PASS") + " class-parity: " + class_decl["name"]
              + "; " + suite["suite"] + "; fixtures=" + ",".join(class_decl["fixtures"])
              + "; rails=" + "/".join(present)
              + "; host-visible bytes identical rail to rail; reference=" + reference
              + " device=" + captures[reference]["device"] + suffix)
        return 0
    except (ClassParityError, compare.CaptureError, OSError, UnicodeError,
            json.JSONDecodeError, KeyError) as error:
        print("FAIL class-parity: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
