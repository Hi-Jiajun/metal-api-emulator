#!/usr/bin/env python3
"""Validate captures, or compare the Swift Metal oracle and compute providers.

Passing checks establishes agreement of the supplied captures with this suite.
It does not attest how a capture was produced or substitute for a native run.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


MAX_ALLOCATION_BYTES = 1_048_576
U64_MAX = (1 << 64) - 1
ALLOCATION_OBSERVATIONS = {
    "native-metal": "gpu-buffer-readback",
    "vulkan": "host-writeback-landing",
    "native-metal-provider": "host-writeback-landing",
}


class CaptureError(ValueError):
    """A suite or capture cannot establish the requested comparison."""


def _require(condition, message):
    if not condition:
        raise CaptureError(message)


def _object(value, keys, where):
    _require(isinstance(value, dict), f"{where}: expected an object")
    _require(set(value) == set(keys), f"{where}: expected fields {', '.join(keys)}")


def _integer(value, where, minimum=0, maximum=U64_MAX):
    _require(type(value) is int and minimum <= value <= maximum,
             f"{where}: expected integer in {minimum}..{maximum}")
    return value


def _string(value, where):
    _require(isinstance(value, str) and value.strip(), f"{where}: expected nonempty string")
    return value


def _hex(value, where):
    _require(isinstance(value, str) and re.fullmatch(r"(?:[0-9a-fA-F]{2})*", value) is not None,
             f"{where}: expected an even-length hexadecimal string")
    return bytes.fromhex(value)


def _list(value, where):
    _require(isinstance(value, list), f"{where}: expected a list")
    return value


def _same_bytes(actual, expected, where, offset=0):
    _require(len(actual) == len(expected),
             f"{where}: length mismatch: expected {len(expected)} bytes, got {len(actual)}")
    for index, (left, right) in enumerate(zip(actual, expected)):
        if left != right:
            raise CaptureError(
                f"{where}: first differing byte at offset {offset + index}: "
                f"expected 0x{right:02x}, got 0x{left:02x}"
            )


def _writeback(value, where):
    _object(value, ("allocation", "view", "offset", "bytes_hex"), where)
    identity = tuple(_integer(value[key], f"{where}.{key}") for key in ("allocation", "view", "offset"))
    data = _hex(value["bytes_hex"], f"{where}.bytes_hex")
    _require(data, f"{where}: empty writeback")
    return identity, data


def _suite_plan(suite):
    _require(isinstance(suite, dict), "suite: expected an object")
    _require(type(suite.get("schema_version")) is int and suite["schema_version"] == 1,
             "suite: unsupported schema_version")
    _string(suite.get("suite"), "suite.suite")
    guard = _integer(suite.get("guard_byte"), "suite.guard_byte")
    _require(guard <= 255, "suite.guard_byte: expected byte <= 255")
    cases = _list(suite.get("cases"), "suite.cases")
    _require(cases, "suite: no cases")
    plan = {}
    for case in cases:
        _require(isinstance(case, dict), "suite case: expected an object")
        case_id = _string(case.get("id"), "suite case.id")
        where = f"suite case {case_id}"
        _require(case_id not in plan, f"{where}: duplicate case")
        allocations, views, initial_ranges, bindings = {}, {}, {}, set()
        buffers = _list(case.get("buffers"), f"{where}.buffers")
        _require(buffers, f"{where}: no buffers")
        for buffer in buffers:
            _require(isinstance(buffer, dict), f"{where}: buffer must be an object")
            allocation = _integer(buffer.get("allocation"), f"{where}.allocation")
            view = _integer(buffer.get("view"), f"{where}.view")
            binding = _integer(buffer.get("binding"), f"{where}.binding")
            offset = _integer(buffer.get("offset"), f"{where}.offset")
            length = _integer(buffer.get("length"), f"{where}.length", 1)
            size = _integer(buffer.get("allocation_size"), f"{where}.allocation_size", 1, MAX_ALLOCATION_BYTES)
            access = buffer.get("access")
            _require(access in ("read", "write", "read_write"), f"{where}: unknown buffer access")
            _require(view not in views and binding not in bindings, f"{where}: duplicate view or binding")
            _require(offset + length <= size, f"{where}: buffer view outside allocation {allocation}")
            initial = _hex(buffer.get("initial_hex"), f"{where} allocation {allocation}.initial_hex")
            _require(len(initial) == length, f"{where}: initial length does not match view {view}")
            if allocation not in allocations:
                allocations[allocation] = bytearray([guard]) * size
                initial_ranges[allocation] = []
            _require(len(allocations[allocation]) == size, f"{where}: inconsistent allocation size")
            _require(all(offset + length <= start or end <= offset
                         for start, end in initial_ranges[allocation]),
                     f"{where}: overlapping initialization would depend on write order")
            initial_ranges[allocation].append((offset, offset + length))
            allocations[allocation][offset:offset + length] = initial
            views[view] = (allocation, offset, length, access)
            bindings.add(binding)

        writable_views = set()
        dispatches = case.get("dispatches")
        if dispatches is None:
            dispatches = [{}]
        _require(isinstance(dispatches, list) and 1 <= len(dispatches) <= 8,
                 f"{where}: invalid dispatch count")
        original_views = [buffer["view"] for buffer in buffers]
        programs = case.get("programs")
        if programs is not None:
            _list(programs, f"{where}.programs")
            _require(1 <= len(programs) <= 8, f"{where}: invalid program count")
            entries = []
            for program in programs:
                _object(program, ("entry", "air", "metal"), f"{where}.program")
                entries.append(_string(program["entry"], f"{where}.program.entry"))
                for kind in ("air", "metal"):
                    source = program[kind]
                    _object(source, ("path", "sha256"), f"{where}.program.{kind}")
                    _string(source["path"], f"{where}.source.path")
                    _require(isinstance(source["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", source["sha256"]), f"{where}: invalid source digest")
            _require(len(entries) == len(set(entries)), f"{where}: duplicate program entry")
        used_programs = set()
        for dispatch in dispatches:
            selection = dispatch.get("program") if isinstance(dispatch, dict) else None
            if programs is not None:
                _integer(selection, f"{where}.dispatch.program", 0, len(programs) - 1)
                used_programs.add(selection)
            else:
                _require(selection is None, f"{where}: program selection without table")
            _require(isinstance(dispatch, dict), f"{where}: dispatch must be an object")
            mapping = dispatch.get("bindings")
            if mapping is None:
                mapping = original_views
            _list(mapping, f"{where}.dispatch.bindings")
            for view in mapping:
                _integer(view, f"{where}.dispatch.view")
            _require(len(mapping) == len(buffers) and set(mapping) == set(original_views),
                     f"{where}: binding map must permute every resource exactly once")
            for slot, view in zip(buffers, mapping):
                _require(views[view][2] == slot["length"],
                         f"{where}: rebound view does not fit the binding extent")
                if slot["access"] != "read":
                    writable_views.add(view)

        if programs is not None:
            _require(used_programs == set(range(len(programs))), f"{where}: unused program entries")

        writes, written_views = [], set()
        for value in _list(case.get("expected_writebacks"), f"{where}.expected_writebacks"):
            identity, data = _writeback(value, f"{where} expected writeback")
            allocation, view, offset = identity
            _require(view in views, f"{where}: writeback references unknown view {view}")
            _require(view not in written_views, f"{where}: duplicate expected writeback view {view}")
            expected_allocation, expected_offset, length, access = views[view]
            _require(view in writable_views, f"{where}: writeback targets read-only view {view}")
            _require((allocation, offset, len(data)) == (expected_allocation, expected_offset, length),
                     f"{where}: writeback does not cover exact writable view {view}")
            allocations[allocation][offset:offset + length] = data
            written_views.add(view)
            writes.append((identity, data))
        _require(written_views == writable_views, f"{where}: expected writebacks do not cover writable views")
        plan[case_id] = (writes, allocations)
    return plan


def validate_capture(suite, digest, report, required_backend=None):
    """Raise CaptureError for invalid captures; success alone does not claim parity."""
    plan = _suite_plan(suite)
    _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
             "suite digest: expected lowercase SHA-256")
    _object(report, ("schema_version", "suite", "suite_sha256", "backend", "allocation_observation",
                     "device", "platform", "results"),
            "capture")
    _require(type(report["schema_version"]) is int and report["schema_version"] == 1,
             "capture: unsupported schema_version")
    _require(report["suite"] == suite["suite"], "capture: suite name mismatch")
    _require(report["suite_sha256"] == digest, "capture: suite_sha256 mismatch (stale or different suite)")
    _require(isinstance(report["backend"], str) and report["backend"] in ALLOCATION_OBSERVATIONS,
             "capture: unknown backend")
    _require(required_backend is None or report["backend"] == required_backend,
             f"capture: expected backend {required_backend}, got {report['backend']}")
    expected_observation = ALLOCATION_OBSERVATIONS[report["backend"]]
    _require(report["allocation_observation"] == expected_observation,
             f"capture: {report['backend']} requires allocation_observation {expected_observation}")
    _string(report["device"], "capture.device")
    _string(report["platform"], "capture.platform")
    results = _list(report["results"], "capture.results")
    seen = set()
    for result in results:
        _object(result, ("id", "completion", "writebacks", "allocations"), "capture result")
        case_id = _string(result["id"], "capture result.id")
        where = f"case {case_id}"
        _require(case_id in plan, f"{where}: unknown case")
        _require(case_id not in seen, f"{where}: duplicate case")
        seen.add(case_id)
        _require(result["completion"] == "CompletedVisible",
                 f"{where}: completion must be CompletedVisible, got {result['completion']!r}")
        expected_writes, expected_allocations = plan[case_id]
        actual_writes, identities = [], set()
        for value in _list(result["writebacks"], f"{where}.writebacks"):
            identity, data = _writeback(value, f"{where} writeback")
            _require(identity not in identities, f"{where}: duplicate writeback {identity}")
            identities.add(identity)
            actual_writes.append((identity, data))
        expected_identities = [identity for identity, _ in expected_writes]
        _require(identities == set(expected_identities),
                 f"{where}: writable set mismatch: expected {expected_identities}, got {[key for key, _ in actual_writes]}")
        _require([identity for identity, _ in actual_writes] == expected_identities,
                 f"{where}: writeback order differs from suite")
        for (identity, actual), (_, expected) in zip(actual_writes, expected_writes):
            allocation, view, offset = identity
            _same_bytes(actual, expected, f"{where} writeback allocation {allocation}/view {view}", offset)

        seen_allocations = set()
        for value in _list(result["allocations"], f"{where}.allocations"):
            _object(value, ("allocation", "bytes_hex"), f"{where} allocation")
            allocation = _integer(value["allocation"], f"{where}.allocation")
            _require(allocation not in seen_allocations, f"{where}: duplicate allocation {allocation}")
            _require(allocation in expected_allocations, f"{where}: unknown allocation {allocation}")
            seen_allocations.add(allocation)
            actual = _hex(value["bytes_hex"], f"{where} allocation {allocation}.bytes_hex")
            _same_bytes(actual, expected_allocations[allocation], f"{where} allocation {allocation}")
        missing = set(expected_allocations) - seen_allocations
        _require(not missing, f"{where}: missing allocations {sorted(missing)}")
    missing = set(plan) - seen
    _require(not missing, f"capture: missing cases {sorted(missing)}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"JSON: duplicate object key {key!r}")
        result[key] = value
    return result


def _read_json(path):
    raw = Path(path).read_bytes()
    return raw, json.loads(raw, object_pairs_hook=_unique_object)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--check", type=Path, help="validate one capture; does not establish parity")
    parser.add_argument("--native", type=Path, help="reference capture produced by the Swift Metal collector")
    parser.add_argument("--vulkan", type=Path, help="capture produced by Vulkan")
    parser.add_argument("--metal-provider", type=Path,
                        help="optional third capture produced by the Rust native Metal provider")
    args = parser.parse_args(argv)
    if args.check is not None:
        if args.native is not None or args.vulkan is not None or args.metal_provider is not None:
            parser.error("--check cannot be combined with --native, --vulkan, or --metal-provider")
    elif args.native is None or args.vulkan is None:
        parser.error("provide --check, or both --native and --vulkan")
    try:
        raw, suite = _read_json(args.suite)
        digest = hashlib.sha256(raw).hexdigest()
        if args.check is not None:
            _, report = _read_json(args.check)
            validate_capture(suite, digest, report)
            print(f"PASS capture: {report['backend']}; {suite['suite']}; {len(suite['cases'])} cases; "
                  f"host-visible bytes agreement with suite; allocations={report['allocation_observation']}")
        else:
            _, native = _read_json(args.native)
            _, vulkan = _read_json(args.vulkan)
            validate_capture(suite, digest, native, required_backend="native-metal")
            validate_capture(suite, digest, vulkan, required_backend="vulkan")
            if args.metal_provider is not None:
                _, metal_provider = _read_json(args.metal_provider)
                validate_capture(suite, digest, metal_provider, required_backend="native-metal-provider")
                print(f"PASS parity: native-metal / vulkan / native-metal-provider; "
                      f"{suite['suite']}; {len(suite['cases'])} cases; host-visible bytes agreement; "
                      "Swift native GPU buffer readback / Vulkan and Rust Metal provider host writeback landing")
            else:
                print(f"PASS parity: native-metal / vulkan; {suite['suite']}; {len(suite['cases'])} cases; "
                      "host-visible bytes agreement; native GPU buffer readback / Vulkan host writeback landing")
        return 0
    except (CaptureError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
