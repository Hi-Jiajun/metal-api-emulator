#!/usr/bin/env python3
"""Validate captures, or compare the Swift Metal oracle and compute providers.

Passing checks establishes agreement of the supplied captures with this suite.
It does not attest how a capture was produced or substitute for a native run.
"""

import argparse
from collections import namedtuple
import hashlib
import json
from pathlib import Path
import re
import struct
import sys


MAX_ALLOCATION_BYTES = 1_048_576
MAX_SERIAL_RESOURCES = 64
U32_MAX = (1 << 32) - 1
U64_MAX = (1 << 64) - 1
ALLOCATION_OBSERVATIONS = {
    "native-metal": "gpu-buffer-readback",
    "vulkan": "host-writeback-landing",
    "native-metal-provider": "host-writeback-landing",
    "vulkan-objects": "host-writeback-landing",
    "native-metal-provider-objects": "host-writeback-landing",
}

# The attachment observation a render case has to land
# (`research/docs/23` §1.1, §5.2). `writes` and `allocations` are the same
# shapes the compute plan builds; `touched`/`written` are the declaring pass's
# allocations, which the count contract is derived from; `rails` is the set of
# capture backends a suite declares this case executable on; `attachment` is
# the `(allocation, view, offset, length)` tuple the render result has to report
# and nothing else for a single-attachment case, or the tuple list in location
# order for an MRT case — and for the v46 pass that binds no colour attachment
# at all, the tuple list is the stored depth landing, the case's whole
# observation — which is what keeps an attachment from passing as a buffer
# writeback; and `present` is the case's optional present section, the
# acquire/present counts every rail its marker names has to report
# (`research/docs/24` §5.3), or `None` when the suite declares none.
RenderExpectation = namedtuple(
    "RenderExpectation",
    "writes allocations touched written rails attachment present icb wildcards",
    defaults=(None,))

# One render case's present section: the target mode and image count the first
# increment fixes, the counts a capture has to report, and the sentinel the
# provider pre-seeds the target with (`research/docs/24` §1.4, §5.3).
PresentExpectation = namedtuple(
    "PresentExpectation", "mode image_count acquire present sentinel")

# One compute case's heap section (`research/docs/25` §4.2, §5.1): the slab
# size, its storage mode, and the placements the capture has to prove landed in
# one slab. `placements` is a tuple of `(allocation, offset, byte_size)` in
# allocation order; `capture_rails` is a separate case-level marker, because a
# heap case runs only on the rails that declare heap support.
HeapExpectation = namedtuple("HeapExpectation", "size storage_mode placements")

# One case's indirect-command section (`research/docs/25` §4.3, §5.1): the
# command kind, the buffer's command cap and kind whitelist, and the replayed
# range. `command` parameters are validated with the section but do not enter
# the expectation: the capture reports what it replayed, not what it was asked
# to replay.
IcbExpectation = namedtuple("IcbExpectation", "kind max_commands kinds start count")


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


def _same_bytes(actual, expected, where, offset=0, wildcards=frozenset()):
    """Compare two byte strings, skipping the offsets the case leaves wild.

    `offset` is where the compared range starts inside its allocation, and
    `wildcards` holds absolute allocation offsets whose bytes the case does not
    claim (`research/docs/23` §3.3, v33). A byte at a wild offset is neither
    compared nor used to build an error message: the case said in advance that
    what landed there is undefined.
    """
    _require(len(actual) == len(expected),
             f"{where}: length mismatch: expected {len(expected)} bytes, got {len(actual)}")
    for index, (left, right) in enumerate(zip(actual, expected)):
        if offset + index in wildcards:
            continue
        if left != right:
            raise CaptureError(
                f"{where}: first differing byte at offset {offset + index}: "
                f"expected 0x{right:02x}, got 0x{left:02x}"
            )


def _compare_observation(result, expected_writes, expected_allocations, where,
                         wildcards=None):
    """Check one case's writebacks and allocation images against the plan.

    Both the compute and the render case report the same two surfaces, so the
    check is shared: the writeback identities are compared as a set *and* in
    order (the suite fixes the landing order), each writeback's bytes are
    compared byte for byte, and every declared allocation image has to be
    reported exactly. A capture that lands the right bytes under the wrong view
    identity, or that drops an allocation, is refused here.
    """
    actual_writes, identities = [], set()
    for value in _list(result["writebacks"], f"{where}.writebacks"):
        identity, data = _writeback(value, f"{where} writeback")
        _require(identity not in identities, f"{where}: duplicate writeback {identity}")
        identities.add(identity)
        actual_writes.append((identity, data))
    expected_identities = [identity for identity, _ in expected_writes]
    _require(identities == set(expected_identities),
             f"{where}: writable set mismatch: expected {expected_identities}, "
             f"got {[key for key, _ in actual_writes]}")
    _require([identity for identity, _ in actual_writes] == expected_identities,
             f"{where}: writeback order differs from suite")
    wildcards = wildcards or {}
    for (identity, actual), (_, expected) in zip(actual_writes, expected_writes):
        allocation, view, offset = identity
        _same_bytes(actual, expected, f"{where} writeback allocation {allocation}/view {view}",
                    offset, wildcards.get(identity, frozenset()))

    seen_allocations = set()
    for value in _list(result["allocations"], f"{where}.allocations"):
        _object(value, ("allocation", "bytes_hex"), f"{where} allocation")
        allocation = _integer(value["allocation"], f"{where}.allocation")
        _require(allocation not in seen_allocations, f"{where}: duplicate allocation {allocation}")
        _require(allocation in expected_allocations, f"{where}: unknown allocation {allocation}")
        seen_allocations.add(allocation)
        actual = _hex(value["bytes_hex"], f"{where} allocation {allocation}.bytes_hex")
        skipped = frozenset(
            index
            for identity, offsets in wildcards.items()
            if identity[0] == allocation
            for index in offsets
        )
        _same_bytes(actual, expected_allocations[allocation],
                    f"{where} allocation {allocation}", 0, skipped)
    missing = set(expected_allocations) - seen_allocations
    _require(not missing, f"{where}: missing allocations {sorted(missing)}")


def _writeback(value, where):
    _object(value, ("allocation", "view", "offset", "bytes_hex"), where)
    identity = tuple(_integer(value[key], f"{where}.{key}") for key in ("allocation", "view", "offset"))
    data = _hex(value["bytes_hex"], f"{where}.bytes_hex")
    _require(data, f"{where}: empty writeback")
    return identity, data


def _present_observation(value, expectation, where):
    """Check one case's present counts against the section its suite declares.

    The two counts are the whole observation: the first increment presents one
    target image once, so a capture that acquired or presented a different
    number of times is not the run the suite asked for (`research/docs/24` §5.3).
    """
    _object(value, ("acquire", "present"), f"{where}.present")
    acquire = _integer(value["acquire"], f"{where}.present.acquire", 0, U32_MAX)
    present = _integer(value["present"], f"{where}.present.present", 0, U32_MAX)
    _require(acquire == expectation.acquire,
             f"{where}: acquire {acquire} does not match the {expectation.acquire} "
             "the suite declares")
    _require(present == expectation.present,
             f"{where}: the present count {present} does not match the {expectation.present} "
             "the suite declares")


def _icb_declaration(value, expected_kinds, where):
    """Parse one case's indirect-command section (`research/docs/25` §4.3).

    The first indirect increment replays exactly one command of one kind per
    case, so the section is a whitelist: the command kind has to be the one the
    case's execution path can carry (`draw`/`draw_indexed` for a render case,
    `dispatch` for a compute case), the buffer's kind list has to contain it,
    the replayed range has to stay inside the buffer, and the command's own
    counts are validated here so a suite cannot spell a zero-vertex draw.
    """
    _object(value, ("kind", "max_commands", "kinds", "range", "command"),
            f"{where}.icb")
    kind = value["kind"]
    _require(kind in ("draw", "draw_indexed", "dispatch"),
             f"{where}.icb: unknown command kind")
    _require(kind in expected_kinds,
             f"{where}.icb: this case's execution path replays a "
             f"{' or '.join(expected_kinds)} command")
    max_commands = _integer(value["max_commands"], f"{where}.icb.max_commands", 1, U32_MAX)
    kinds = _list(value["kinds"], f"{where}.icb.kinds")
    _require(kinds and len(set(kinds)) == len(kinds)
             and all(named in ("draw", "draw_indexed", "dispatch") for named in kinds),
             f"{where}.icb: kinds has to name distinct known command kinds")
    _require(kind in kinds, f"{where}.icb: the buffer does not admit its own command kind")
    range_ = value["range"]
    _object(range_, ("start", "count"), f"{where}.icb.range")
    start = _integer(range_["start"], f"{where}.icb.range.start", 0, U32_MAX)
    count = _integer(range_["count"], f"{where}.icb.range.count", 1, U32_MAX)
    _require(start + count <= max_commands,
             f"{where}.icb: the replayed range exceeds the command buffer")
    command = value["command"]
    if kind == "draw":
        _object(command, ("vertex_count", "instance_count"), f"{where}.icb.command")
        _integer(command["vertex_count"], f"{where}.icb.command.vertex_count", 1, U32_MAX)
        _integer(command["instance_count"], f"{where}.icb.command.instance_count", 1, U32_MAX)
    elif kind == "draw_indexed":
        _object(command, ("index_count", "instance_count"), f"{where}.icb.command")
        _integer(command["index_count"], f"{where}.icb.command.index_count", 1, U32_MAX)
        _integer(command["instance_count"], f"{where}.icb.command.instance_count", 1, U32_MAX)
    else:
        _object(command, ("x", "y", "z"), f"{where}.icb.command")
        for axis in ("x", "y", "z"):
            _integer(command[axis], f"{where}.icb.command.{axis}", 1, U32_MAX)
    return IcbExpectation(kind=kind, max_commands=max_commands, kinds=tuple(kinds),
                          start=start, count=count)


def _icb_observation(value, expectation, where):
    """Check one case's `icb` segment against the section its suite declares.

    The segment is the provider's own record of the replay — the command kind,
    the buffer range it replayed and how many commands it encoded — not the
    suite's request echoed back. The bytes stay with the ordinary attachment or
    writeback comparison (`research/docs/25` §5.1).
    """
    _object(value, ("kind", "start", "count", "commands"), f"{where}.icb")
    _require(value["kind"] == expectation.kind,
             f"{where}.icb: the replayed kind does not match the suite")
    start = _integer(value["start"], f"{where}.icb.start", 0, U32_MAX)
    count = _integer(value["count"], f"{where}.icb.count", 1, U32_MAX)
    commands = _integer(value["commands"], f"{where}.icb.commands", 1, U32_MAX)
    _require((start, count) == (expectation.start, expectation.count),
             f"{where}.icb: the replayed range does not match the suite")
    _require(commands == count,
             f"{where}.icb: replayed {commands} commands for a {count}-command range")
    _require(commands <= expectation.max_commands,
             f"{where}.icb: the buffer holds fewer commands than were replayed")


def _heap_observation(value, expectation, where):
    """Check one case's heap segment against the section its suite declares.

    The segment proves the placements landed in one slab at the declared
    offsets; the bytes themselves stay with the ordinary writeback comparison
    (`research/docs/25` §5.1). A provider that echoes the request instead of
    reporting what it bound cannot be told apart by this check alone, which is
    why the smoke case asserts the real `vkBind*` result as well.
    """
    _object(value, ("heap", "same_slab", "placements"), f"{where}.heap")
    _require(value["same_slab"] is True,
             f"{where}.heap: the placements have to share one slab")
    _integer(value["heap"], f"{where}.heap.heap", 1)
    entries = _list(value["placements"], f"{where}.heap.placements")
    observed = []
    for index, entry in enumerate(entries):
        entry_where = f"{where}.heap.placements[{index}]"
        _object(entry, ("allocation", "offset", "byte_size"), entry_where)
        observed.append((
            _integer(entry["allocation"], f"{entry_where}.allocation"),
            _integer(entry["offset"], f"{entry_where}.offset"),
            _integer(entry["byte_size"], f"{entry_where}.byte_size", 1,
                     MAX_ALLOCATION_BYTES),
        ))
    _require(tuple(observed) == expectation.placements,
             f"{where}.heap: placements do not match the suite")


def _heap_declaration(value, allocations, where):
    """Parse one compute case's heap section (`research/docs/25` §4.2).

    The section is a whitelist and a falsifiability statement, not a knob: the
    first heap increment fixes one slab with `allows_aliasing = false`, so any
    two placements that overlap are refused here rather than left to a driver,
    and every placement has to cover exactly the allocation it names (the same
    `allocation_size` the buffer table declares). The capture's heap segment is
    compared against this expectation field for field.
    """
    _object(value, ("size", "storage_mode", "allows_aliasing", "placements"),
            f"{where}.heap")
    size = _integer(value["size"], f"{where}.heap.size", 1, MAX_ALLOCATION_BYTES)
    _require(value["storage_mode"] in ("owned_bytes", "staged_lease", "borrowed_no_copy"),
             f"{where}.heap: unknown storage mode")
    _require(value["allows_aliasing"] is False,
             f"{where}.heap: the first heap increment refuses aliasing")
    entries = _list(value["placements"], f"{where}.heap.placements")
    _require(entries, f"{where}.heap: no placements")
    placements = []
    covered = []
    for index, entry in enumerate(entries):
        entry_where = f"{where}.heap.placements[{index}]"
        _object(entry, ("allocation", "offset", "byte_size"), entry_where)
        allocation = _integer(entry["allocation"], f"{entry_where}.allocation")
        offset = _integer(entry["offset"], f"{entry_where}.offset")
        byte_size = _integer(entry["byte_size"], f"{entry_where}.byte_size", 1,
                             MAX_ALLOCATION_BYTES)
        _require(allocation in allocations,
                 f"{entry_where}: unknown allocation {allocation}")
        _require(byte_size == len(allocations[allocation]),
                 f"{entry_where}: byte_size does not match allocation {allocation}")
        _require(offset + byte_size <= size,
                 f"{entry_where}: placement exceeds the heap")
        end = offset + byte_size
        for start, other_end in covered:
            _require(end <= start or offset >= other_end,
                     f"{where}.heap: placements overlap while aliasing is refused")
        covered.append((offset, end))
        placements.append((allocation, offset, byte_size))
    _require([placement[0] for placement in placements]
             == sorted(placement[0] for placement in placements),
             f"{where}.heap: placements must be in allocation order")
    return HeapExpectation(size=size, storage_mode=value["storage_mode"],
                           placements=tuple(placements))


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
        air_encoding = case.get("air_encoding", "text")
        _require(air_encoding in ("text", "raw", "wrapped"),
                 f"{where}: unknown air_encoding")
        allocations, views, initial_ranges, bindings = {}, {}, {}, set()
        # v11 adds a sampled-texture section: textures are their own
        # allocations and the provider uploads each one once, so the count
        # contract has to account for them (`research/docs/18` step 3).
        textures = _list(case.get("textures", []), f"{where}.textures")
        texture_allocations = set()
        for texture in textures:
            _require(isinstance(texture, dict), f"{where}: texture must be an object")
            allocation = _integer(texture.get("allocation"), f"{where}.texture.allocation")
            view = _integer(texture.get("view"), f"{where}.texture.view")
            binding = _integer(texture.get("binding"), f"{where}.texture.binding")
            width = _integer(texture.get("width"), f"{where}.texture.width", 1)
            height = _integer(texture.get("height"), f"{where}.texture.height", 1)
            _require(texture.get("format") == "r32_uint",
                     f"{where}: unknown texture format")
            _require(texture.get("access") in ("sampled", "storage"),
                     f"{where}: unknown texture access")
            _require(allocation not in texture_allocations,
                     f"{where}: duplicate texture allocation {allocation}")
            _require(view not in views and binding not in bindings,
                     f"{where}: duplicate view or binding")
            initial = _hex(texture.get("initial_hex"),
                           f"{where} texture {allocation}.initial_hex")
            _require(len(initial) == width * height * 4,
                     f"{where}: texture initial length does not match its extent")
            texture_allocations.add(allocation)
        buffers = _list(case.get("buffers"), f"{where}.buffers")
        _require(buffers, f"{where}: no buffers")
        _require(len(buffers) <= MAX_SERIAL_RESOURCES,
                 f"{where}: buffer pool exceeds {MAX_SERIAL_RESOURCES} resources")
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
        if len(allocations) != len(buffers):
            # Several buffers naming one allocation is the ranged-alias shape.
            # Overlap was refused above, so only disjoint views survive.
            _require(suite["suite"] == "compute-buffer-v10",
                     f"{where}: several views of one allocation are only qualified "
                     "by the v10 suite")

        writable_views = set()
        dispatches = case.get("dispatches")
        if dispatches is None:
            dispatches = [{}]
        _require(isinstance(dispatches, list) and 1 <= len(dispatches) <= 8,
                 f"{where}: invalid dispatch count")
        original_views = [buffer["view"] for buffer in buffers]
        programs = case.get("programs")
        default_slots = [{key: buffer[key] for key in ("binding", "access", "length")}
                         for buffer in buffers]
        program_slots = []
        if programs is not None:
            _list(programs, f"{where}.programs")
            _require(1 <= len(programs) <= 8, f"{where}: invalid program count")
            entries = []
            for program in programs:
                keys = ("entry", "air", "metal")
                if isinstance(program, dict) and "buffer_slots" in program:
                    keys += ("buffer_slots",)
                _object(program, keys, f"{where}.program")
                entries.append(_string(program["entry"], f"{where}.program.entry"))
                for kind in ("air", "metal"):
                    source = program[kind]
                    _object(source, ("path", "sha256"), f"{where}.program.{kind}")
                    _string(source["path"], f"{where}.source.path")
                    _require(isinstance(source["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", source["sha256"]), f"{where}: invalid source digest")
                slots = program.get("buffer_slots", default_slots)
                if "buffer_slots" in program:
                    _list(slots, f"{where}.program.buffer_slots")
                    _require(1 <= len(slots) <= len(buffers),
                             f"{where}: buffer_slots must cover every resource selected by "
                             f"the program exactly once (1..{len(buffers)} slots)")
                    previous_binding = -1
                    for slot in slots:
                        slot_where = f"{where}.program.buffer_slot"
                        _object(slot, ("binding", "access", "length"), slot_where)
                        binding = _integer(slot["binding"], f"{slot_where}.binding", 0, U32_MAX)
                        _require(binding > previous_binding,
                                 f"{where}: buffer_slots bindings must be unique and sorted")
                        previous_binding = binding
                        _require(slot["access"] in ("read", "write", "read_write"),
                                 f"{slot_where}: unknown buffer access")
                        _integer(slot["length"], f"{slot_where}.length", 1, MAX_ALLOCATION_BYTES)
                program_slots.append(slots)
            _require(len(entries) == len(set(entries)), f"{where}: duplicate program entry")
            initial_slots = {slot["binding"]: slot for slot in default_slots}
            _require(all(initial_slots.get(slot["binding"]) == slot for slot in program_slots[0]),
                     f"{where}: first program buffer_slots must match initial buffer metadata")
        used_programs = set()
        used_views = set()
        # Per-dispatch allocation touches and writes. A v9 case commits one
        # command buffer per group, so these become the per-group count
        # expectations (`research/docs/15` §5b).
        dispatch_touches = []
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
            slots = program_slots[selection] if programs is not None else default_slots
            if programs is None:
                _require(len(mapping) == len(buffers) and set(mapping) == set(original_views),
                         f"{where}: binding map must permute every resource exactly once")
            else:
                _require(len(mapping) == len(slots) and len(set(mapping)) == len(mapping)
                         and set(mapping).issubset(views),
                         f"{where}: binding map must cover every resource selected by program "
                         "slots and permute every resource in that selection exactly once")
            used_views.update(mapping)
            touched, written = set(), set()
            for slot, view in zip(slots, mapping):
                _require(views[view][2] == slot["length"],
                         f"{where}: rebound view does not fit the binding extent")
                touched.add(views[view][0])
                if slot["access"] != "read":
                    writable_views.add(view)
                    written.add(views[view][0])
            dispatch_touches.append((touched, written))

        if programs is not None:
            _require(used_programs == set(range(len(programs))), f"{where}: unused program entries")
            _require(used_views == set(original_views),
                     f"{where}: unused buffer pool resources; every view must appear in a dispatch")

        command_buffers = case.get("command_buffers")
        group_expectations = None
        if command_buffers is None:
            _require(suite["suite"] != "compute-buffer-v9",
                     f"{where}: v9 fixture requires command buffer groups")
        else:
            _require(suite["suite"] == "compute-buffer-v9",
                     f"{where}: command buffer groups are only qualified by the v9 suite")
            _require(isinstance(command_buffers, list) and 2 <= len(command_buffers) <= 4,
                     f"{where}: v9 fixture needs two to four command buffers")
            expected = 0
            for group in command_buffers:
                _require(isinstance(group, list) and group,
                         f"{where}: command buffer group cannot be empty")
                for index in group:
                    _require(type(index) is int and index == expected,
                             f"{where}: command buffer groups must partition the dispatch order")
                    expected += 1
            _require(expected == len(dispatches),
                     f"{where}: command buffer groups must partition the dispatch order")
            # One submission per group: the provider copies in every
            # allocation that group's own dispatches touch and copies out
            # every allocation they write, so groups cannot be checked with
            # the case-level totals (`research/docs/15` §5b).
            group_expectations = []
            for group in command_buffers:
                touched, written = set(), set()
                for index in group:
                    touched.update(dispatch_touches[index][0])
                    written.update(dispatch_touches[index][1])
                group_expectations.append(
                    (len(touched) + len(texture_allocations), len(written))
                )

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
        # A heap-bearing compute case declares which rails owe it, exactly as a
        # render case does (`research/docs/25` §5.2): only a rail that declares
        # heap support can report the placement observation, so the marker is
        # required with the section and refused without it.
        heap = None
        icb = None
        rails = None
        if "heap" in case:
            heap = _heap_declaration(case["heap"], allocations, where)
        if "icb" in case:
            icb = _icb_declaration(case["icb"], ("dispatch",), where)
        if "capture_rails" in case:
            rails = _list(case["capture_rails"], f"{where}.capture_rails")
            _require(rails and len(set(rails)) == len(rails)
                     and all(isinstance(rail, str)
                             and rail in ALLOCATION_OBSERVATIONS for rail in rails),
                     f"{where}: capture_rails has to name distinct known backends")
        _require(((heap is None and icb is None) == (rails is None)),
                 f"{where}: capture_rails and a heap or icb section are declared together")
        plan[case_id] = (writes, allocations, len(texture_allocations),
                         group_expectations, heap, icb, rails)
    return plan


def _present_declaration(value, texel, where):
    """Parse one render case's optional present section (`research/docs/24` §5.3).

    The section is the suite's own falsifiability statement rather than a
    fixture knob: the first present increment presents one target image once,
    and the sentinel the provider pre-seeds the target with has to differ from
    the texel the render pass is expected to leave there, so a capture whose
    target still held its pre-seeded bytes could not be read as a pass. A
    missing `initial_hex` is Undefined, which pre-seeds nothing. What the
    sentinel proves depends on the pass: a `load: "clear"` case overwrites it
    in the same submission (its falsifier is the clear colour and the reported
    counts), while a `load: "load"` case starts from it directly — which is the
    shape `native-oracle --present-selftest` uses.
    """
    _require(isinstance(value, dict), f"{where}.present: expected an object")
    required = ("mode", "image_count", "acquire", "present")
    missing = [field for field in required if field not in value]
    _require(not missing, f"{where}.present: missing fields {', '.join(missing)}")
    _require(set(value).issubset(set(required) | {"initial_hex"}),
             f"{where}.present: unexpected fields "
             + ", ".join(sorted(set(value) - set(required))))
    mode = value["mode"]
    _require(mode == "fifo", f"{where}.present: the first present increment is fifo")
    image_count = _integer(value["image_count"], f"{where}.present.image_count", 1)
    _require(image_count == 1,
             f"{where}.present: image_count has to be 1, got {image_count}")
    acquire = _integer(value["acquire"], f"{where}.present.acquire", 0, U32_MAX)
    presented = _integer(value["present"], f"{where}.present.present", 0, U32_MAX)
    _require((acquire, presented) == (1, 1),
             f"{where}.present: the first increment acquires and presents exactly once")
    sentinel = None
    if "initial_hex" in value:
        sentinel = _hex(value["initial_hex"], f"{where}.present.initial_hex")
        _require(len(sentinel) == 4, f"{where}.present: a sentinel is one texel")
        _require(sentinel != texel, f"{where}.present: the sentinel equals the expected texel")
    return PresentExpectation(mode=mode, image_count=image_count, acquire=acquire,
                              present=presented, sentinel=sentinel)


def _vertex_input_declaration(case, where):
    """Pin the reviewed vertex-input shape of a render case (`docs/23` §3.3).

    Three geometries exist and no fourth: the milestone's `vertex_id` triangle
    (no `vertex_layout` at all), the reviewed indexed quad — one `float32x2`
    position stream at stride eight bound at index 0, plus six `uint16` or
    `uint32` indices whose values all name one of the four vertices the stream
    carries — and the reviewed instanced pair (`research/docs/23` §3.3, v31):
    the same position stream at binding 0, a `float32x4` tint stream at
    binding 1 that advances once per *instance*, and exactly two instances.
    The rules mirror `provider-capture`'s `render_geometry` and the Swift
    oracle's own validation, so a suite one rail would refuse cannot pass here
    either.
    """
    quad_vertices, quad_indices, quad_stride = 4, 6, 8
    layout = case.get("vertex_layout")
    vertex_buffers = case.get("vertex_buffers", [])
    indices = case.get("indices")
    if layout is None:
        _require(not vertex_buffers and indices is None,
                 f"{where}: vertex buffers without a vertex layout describe no stream")
        _require(case.get("base_vertex", 0) == 0,
                 f"{where}: a base vertex needs an index buffer")
        return None
    _object(layout, ("buffers",), f"{where}.vertex_layout")
    streams = _list(layout["buffers"], f"{where}.vertex_layout.buffers")
    if len(streams) == 2:
        return _instanced_declaration(case, streams, vertex_buffers, indices, where)
    # The two reviewed pair shapes are mutually exclusive and the culling one
    # is checked first, so a case that declares both is refused by the cull
    # rule rather than silently read as the depth fixture
    # (`research/docs/23` §3.3, v39).
    # The three reviewed pair shapes are mutually exclusive, and the blending
    # one is checked first so a case that declares more than one is refused by
    # the shape it claims rather than read as another fixture
    # (`research/docs/23` §3.3, v40).
    if case.get("blend") is not None:
        return _blend_declaration(case, streams, vertex_buffers, indices, where)
    if case.get("cull") is not None:
        return _cull_declaration(case, streams, vertex_buffers, indices, where)
    # The stencil pair is the depth fixture's geometry with a `stencil8` surface
    # in place of the depth one, and the two are mutually exclusive: this rule
    # is checked before the depth rule, so a case that declares both is refused
    # by the shape it claims rather than read as either one
    # (`research/docs/23` §3.3, v47).
    if case.get("stencil") is not None or case.get("stencil_test") is not None:
        return _stencil_declaration(case, streams, vertex_buffers, indices, where)
    if case.get("depth") is not None:
        return _depth_declaration(case, streams, vertex_buffers, indices, where)
    if case.get("base_vertex", 0) != 0:
        return _base_vertex_declaration(case, streams, vertex_buffers, indices, where)
    _require(len(streams) == 1, f"{where}: the reviewed shape is one vertex stream")
    _object(streams[0], ("stride", "attributes"), f"{where}.vertex_layout.buffers[0]")
    _require(streams[0]["stride"] == quad_stride,
             f"{where}: the reviewed stream has stride {quad_stride}")
    attributes = _list(streams[0]["attributes"], f"{where}.vertex_layout.buffers[0].attributes")
    _require(len(attributes) == 1, f"{where}: the reviewed stream has one attribute")
    _object(attributes[0], ("location", "offset", "format"),
            f"{where}.vertex_layout.buffers[0].attributes[0]")
    attribute = attributes[0]
    _require((attribute["location"], attribute["offset"], attribute["format"])
             == (0, 0, "float32x2"),
             f"{where}: the reviewed attribute is location 0, offset 0, float32x2")
    bindings = _list(vertex_buffers, f"{where}.vertex_buffers")
    _require(len(bindings) == 1, f"{where}: the reviewed shape binds one vertex stream")
    for position, binding in enumerate(bindings):
        _object(binding, ("allocation", "view", "offset", "length", "initial_hex"),
                f"{where}.vertex_buffers[{position}]")
        _require(binding["allocation"] > 0 and binding["view"] > 0,
                 f"{where}: zero vertex stream identity")
        _require(binding["length"] >= quad_stride * quad_vertices,
                 f"{where}: the vertex stream is shorter than the reviewed quad reads")
        _require(len(_hex(binding["initial_hex"],
                          f"{where}.vertex_buffers[{position}].initial_hex")) == binding["length"],
                 f"{where}: the vertex stream bytes do not match its length")
        _require("format" not in binding,
                 f"{where}: a vertex stream carries no index format")
    _require(indices is not None, f"{where}: the reviewed vertex-input shape is indexed")
    _object(indices, ("allocation", "view", "offset", "length", "initial_hex", "format"),
            f"{where}.indices")
    _require(indices["allocation"] > 0 and indices["view"] > 0,
             f"{where}: zero index buffer identity")
    width = {"uint16": 2, "uint32": 4}.get(indices["format"])
    _require(width is not None, f"{where}: unsupported index format")
    _require(indices["length"] >= quad_indices * width,
             f"{where}: the index buffer is shorter than the reviewed quad reads")
    index_bytes = _hex(indices["initial_hex"], f"{where}.indices.initial_hex")
    _require(len(index_bytes) == indices["length"],
             f"{where}: the index bytes do not match their length")
    for position in range(0, len(index_bytes), width):
        chunk = index_bytes[position:position + width]
        index = int.from_bytes(chunk, "little")
        _require(index < quad_vertices,
                 f"{where}: index {position // width} names vertex {index} outside the quad")
    return {"vertices": quad_vertices, "indices": quad_indices}


def _unorm_texel(record, where):
    """The four texel bytes a reviewed instance tint stores.

    Every component of a reviewed tint is exactly `0.0` or `1.0`, so the byte an
    8-bit UNORM attachment stores is exactly `0x00` or `0xff` and the comparison
    does not have to model a rounding rule. Any other component is refused: the
    expectation could not pin what a driver would store
    (`research/docs/23` §3.5).
    """
    _require(len(record) == 16, f"{where}: an instance tint is a float32x4")
    texel = bytearray(4)
    for index, component in enumerate(struct.unpack("<4f", record)):
        _require(component in (0.0, 1.0),
                 f"{where}: a reviewed instance tint is zero or one per component")
        texel[index] = 0xFF if component == 1.0 else 0x00
    return bytes(texel)


def _instanced_declaration(case, streams, vertex_buffers, indices, where):
    """Pin the reviewed instanced pair (`research/docs/23` §3.3, v31).

    The two streams are the whole review surface: binding 0 is the reviewed
    `float32x2` position stream that advances per vertex, binding 1 is a
    `float32x4` tint stream that advances per instance, and the draw runs
    exactly two instances so the first record's tint covers one half of the
    attachment and the second record's tint the other. The returned mapping
    carries the triangle's vertex and index counts plus the two tints, which
    the expectation rules below read.
    """
    quad_vertices, quad_indices, quad_stride = 4, 6, 8
    tint_stride = 16
    _require(case.get("instance_count") == 2,
             f"{where}: the reviewed instanced draw runs exactly two instances")
    _object(streams[0], ("stride", "step", "attributes"),
            f"{where}.vertex_layout.buffers[0]")
    _object(streams[1], ("stride", "step", "attributes"),
            f"{where}.vertex_layout.buffers[1]")
    _require((streams[0]["stride"], streams[0]["step"]) == (quad_stride, "per_vertex"),
             f"{where}: the reviewed instanced position stream is stride {quad_stride} "
             "and steps per vertex")
    _require((streams[1]["stride"], streams[1]["step"]) == (tint_stride, "per_instance"),
             f"{where}: the reviewed instanced tint stream is stride {tint_stride} "
             "and steps per instance")
    attributes = _list(streams[0]["attributes"],
                       f"{where}.vertex_layout.buffers[0].attributes")
    _require(len(attributes) == 1, f"{where}: the reviewed stream has one attribute")
    _object(attributes[0], ("location", "offset", "format"),
            f"{where}.vertex_layout.buffers[0].attributes[0]")
    _require((attributes[0]["location"], attributes[0]["offset"], attributes[0]["format"])
             == (0, 0, "float32x2"),
             f"{where}: the reviewed attribute is location 0, offset 0, float32x2")
    tints = _list(streams[1]["attributes"],
                  f"{where}.vertex_layout.buffers[1].attributes")
    _require(len(tints) == 1, f"{where}: the reviewed tint stream has one attribute")
    _object(tints[0], ("location", "offset", "format"),
            f"{where}.vertex_layout.buffers[1].attributes[0]")
    _require((tints[0]["location"], tints[0]["offset"], tints[0]["format"])
             == (1, 0, "float32x4"),
             f"{where}: the reviewed tint is location 1, offset 0, float32x4")
    bindings = _list(vertex_buffers, f"{where}.vertex_buffers")
    _require(len(bindings) == 2, f"{where}: the reviewed shape binds two vertex streams")
    for position, binding in enumerate(bindings):
        _object(binding, ("allocation", "view", "offset", "length", "initial_hex"),
                f"{where}.vertex_buffers[{position}]")
        _require(binding["allocation"] > 0 and binding["view"] > 0,
                 f"{where}: zero vertex stream identity")
        _require("format" not in binding,
                 f"{where}: a vertex stream carries no index format")
        _require(len(_hex(binding["initial_hex"],
                          f"{where}.vertex_buffers[{position}].initial_hex")) == binding["length"],
                 f"{where}: the vertex stream bytes do not match its length")
    _require(bindings[0]["length"] == quad_stride * quad_vertices,
             f"{where}: the position stream is the reviewed quad")
    _require(bindings[1]["length"] == tint_stride * 2,
             f"{where}: the tint stream carries exactly the two reviewed records")
    records = [_hex(bindings[1]["initial_hex"][index * tint_stride * 2:(index + 1) * tint_stride * 2],
                    f"{where}.vertex_buffers[1].initial_hex")
               for index in range(2)]
    _require(records[0] != records[1],
             f"{where}: the two instance tints have to differ, or the halves cannot "
             "show which record each instance read")
    tints = [_unorm_texel(record, where) for record in records]
    _require(indices is not None, f"{where}: the reviewed instanced shape is indexed")
    _object(indices, ("allocation", "view", "offset", "length", "initial_hex", "format"),
            f"{where}.indices")
    _require(indices["allocation"] > 0 and indices["view"] > 0,
             f"{where}: zero index buffer identity")
    width = {"uint16": 2, "uint32": 4}.get(indices["format"])
    _require(width is not None, f"{where}: unsupported index format")
    _require(indices["length"] == quad_indices * width,
             f"{where}: the index buffer is the reviewed six indices")
    index_bytes = _hex(indices["initial_hex"], f"{where}.indices.initial_hex")
    _require(len(index_bytes) == indices["length"],
             f"{where}: the index bytes do not match their length")
    for position in range(0, len(index_bytes), width):
        chunk = index_bytes[position:position + width]
        index = int.from_bytes(chunk, "little")
        _require(index < quad_vertices,
                 f"{where}: index {position // width} names vertex {index} outside the quad")
    return {"vertices": quad_vertices, "indices": quad_indices, "instanced": True,
            "tints": tints}


def _blend_declaration(case, streams, vertex_buffers, indices, where):
    """Pin the reviewed blend shape (`research/docs/23` §3.3, v40).

    The same reviewed pair module the depth and culling fixtures compile: one
    stream whose vertices carry a `float32x3` position at offset 0 and a
    `float32x4` tint at offset 16 (stride thirty-two), this time as one oversize
    triangle whose tint is `(64/255, 128/255, 192/255, 128/255)`. With the
    reviewed blend state — source alpha against one-minus-source-alpha, added —
    over a cleared-to-zero attachment the stored texel is `20406040`: a rail
    that ignored the state would store the tint itself, and one that swapped the
    factors would store the clear colour.
    """
    stride = 32
    _object(streams[0], ("stride", "attributes"), f"{where}.vertex_layout.buffers[0]")
    _require(streams[0]["stride"] == stride,
             f"{where}: the reviewed blend stream has stride {stride}")
    attributes = _list(streams[0]["attributes"],
                       f"{where}.vertex_layout.buffers[0].attributes")
    _require(len(attributes) == 2, f"{where}: the reviewed blend stream has two attributes")
    _object(attributes[0], ("location", "offset", "format"),
            f"{where}.vertex_layout.buffers[0].attributes[0]")
    _require((attributes[0]["location"], attributes[0]["offset"], attributes[0]["format"])
             == (0, 0, "float32x3"),
             f"{where}: the reviewed blend position is location 0, offset 0, float32x3")
    _object(attributes[1], ("location", "offset", "format"),
            f"{where}.vertex_layout.buffers[0].attributes[1]")
    _require((attributes[1]["location"], attributes[1]["offset"], attributes[1]["format"])
             == (1, 16, "float32x4"),
             f"{where}: the reviewed blend tint is location 1, offset 16, float32x4")
    _require(case.get("depth") is None and case.get("cull") is None,
             f"{where}: the reviewed blend shape carries neither a depth attachment nor a "
             "culling state")
    blend = case.get("blend")
    _require(isinstance(blend, list), f"{where}: the blend state is a list")
    _require(len(blend) == 1, f"{where}: the reviewed blend shape states one attachment")
    attachment = blend[0]
    _require(isinstance(attachment, dict), f"{where}.blend[0]: expected an object")
    _require(set(attachment) == {"source_rgb", "destination_rgb", "source_alpha",
                                 "destination_alpha", "operation"},
             f"{where}.blend[0]: expected the five blend fields")
    _require((attachment["source_rgb"], attachment["destination_rgb"],
              attachment["source_alpha"], attachment["destination_alpha"],
              attachment["operation"])
             == ("source_alpha", "one_minus_source_alpha", "source_alpha",
                 "one_minus_source_alpha", "add"),
             f"{where}: the reviewed blend state is source alpha against "
             "one-minus-source-alpha with an add")
    bindings = _list(vertex_buffers, f"{where}.vertex_buffers")
    _require(len(bindings) == 1, f"{where}: the reviewed blend shape binds one stream")
    binding = bindings[0]
    _object(binding, ("allocation", "view", "offset", "length", "initial_hex"),
            f"{where}.vertex_buffers[0]")
    _require(binding["allocation"] > 0 and binding["view"] > 0,
             f"{where}: zero vertex stream identity")
    _require(binding["length"] == stride * 3,
             f"{where}: the reviewed blend stream is three stride-{stride} vertices")
    _require(len(_hex(binding["initial_hex"], f"{where}.vertex_buffers[0].initial_hex"))
             == binding["length"],
             f"{where}: the vertex stream bytes do not match its length")
    _require(indices is not None, f"{where}: the reviewed blend shape is indexed")
    _object(indices, ("allocation", "view", "offset", "length", "initial_hex", "format"),
            f"{where}.indices")
    _require(indices["initial_hex"] == "000001000200",
             f"{where}: the reviewed blend indices are the reviewed triangle")
    return {"vertices": 3, "indices": 3}


def _cull_declaration(case, streams, vertex_buffers, indices, where):
    """Pin the reviewed cull pair (`research/docs/23` §3.3, v39).

    The same reviewed pair module the depth fixture compiles: one stream whose
    vertices carry a `float32x3` position at offset 0 and a `float32x4` tint at
    offset 16 (stride thirty-two), this time as two copies of one oversize
    triangle whose vertex orders are opposite. The pass culls back faces with a
    counter-clockwise front, so exactly the counter-clockwise copy survives and
    its tint covers the attachment; a rail that ignored the state would show the
    other copy (the later one wins when both draw).
    """
    quad_indices, stride = 6, 32
    _object(streams[0], ("stride", "attributes"), f"{where}.vertex_layout.buffers[0]")
    _require(streams[0]["stride"] == stride,
             f"{where}: the reviewed cull stream has stride {stride}")
    attributes = _list(streams[0]["attributes"],
                       f"{where}.vertex_layout.buffers[0].attributes")
    _require(len(attributes) == 2, f"{where}: the reviewed cull stream has two attributes")
    _object(attributes[0], ("location", "offset", "format"),
            f"{where}.vertex_layout.buffers[0].attributes[0]")
    _require((attributes[0]["location"], attributes[0]["offset"], attributes[0]["format"])
             == (0, 0, "float32x3"),
             f"{where}: the reviewed cull position is location 0, offset 0, float32x3")
    _object(attributes[1], ("location", "offset", "format"),
            f"{where}.vertex_layout.buffers[0].attributes[1]")
    _require((attributes[1]["location"], attributes[1]["offset"], attributes[1]["format"])
             == (1, 16, "float32x4"),
             f"{where}: the reviewed cull tint is location 1, offset 16, float32x4")
    cull = case.get("cull")
    _require(isinstance(cull, dict), f"{where}: a culling state is an object")
    _require(set(cull) == {"mode", "winding"},
             f"{where}.cull: expected fields mode, winding")
    _require((cull["mode"], cull["winding"]) == ("back", "counter_clockwise"),
             f"{where}: the reviewed cull state is back faces with a counter-clockwise front")
    _require(case.get("depth") is None,
             f"{where}: the reviewed cull shape carries no depth attachment")
    bindings = _list(vertex_buffers, f"{where}.vertex_buffers")
    _require(len(bindings) == 1, f"{where}: the reviewed cull shape binds one stream")
    binding = bindings[0]
    _object(binding, ("allocation", "view", "offset", "length", "initial_hex"),
            f"{where}.vertex_buffers[0]")
    _require(binding["allocation"] > 0 and binding["view"] > 0,
             f"{where}: zero vertex stream identity")
    _require(binding["length"] == stride * quad_indices,
             f"{where}: the reviewed cull stream is six stride-{stride} vertices")
    _require(len(_hex(binding["initial_hex"], f"{where}.vertex_buffers[0].initial_hex"))
             == binding["length"],
             f"{where}: the vertex stream bytes do not match its length")
    _require(indices is not None, f"{where}: the reviewed cull shape is indexed")
    _object(indices, ("allocation", "view", "offset", "length", "initial_hex", "format"),
            f"{where}.indices")
    _require(indices["initial_hex"] == "000001000200030004000500",
             f"{where}: the reviewed cull indices are the two reviewed triangles")
    return {"vertices": quad_indices, "indices": quad_indices}


def _depth_declaration(case, streams, vertex_buffers, indices, where):
    """Pin the reviewed depth pair (`research/docs/23` §3.3, v36 and v43).

    One stream whose vertices carry a `float32x3` position at offset 0 and a
    `float32x4` tint at offset 16 (stride thirty-two), two oversize triangles at
    `z = 0.5` and `z = 0.9`, a cleared `depth32float` attachment and a `less`
    test with writes on. With that state the near triangle wins everywhere the
    two overlap — which is the whole attachment — so the expectation is the
    near tint sixteen times; a rail that dropped the attachment, the clear or
    the test would show the far one.

    The v43 increment upgrades that attachment from a surface the pass owns and
    drops to a resource the caller keeps: with a `store` action the fixture also
    names the allocation and the view the texels land in and the texels it
    expects there, and the comparison observes them through the same writeback
    and allocation channel the colour side uses. A depth attachment without one
    is the v36 shape — cleared, tested against and discarded — so it carries
    neither identity nor expectation, and a fixture that half-declares the
    stored shape is refused rather than read as either one.
    """
    quad_indices, stride = 6, 32
    _object(streams[0], ("stride", "attributes"), f"{where}.vertex_layout.buffers[0]")
    _require(streams[0]["stride"] == stride,
             f"{where}: the reviewed depth stream has stride {stride}")
    attributes = _list(streams[0]["attributes"],
                       f"{where}.vertex_layout.buffers[0].attributes")
    _require(len(attributes) == 2, f"{where}: the reviewed depth stream has two attributes")
    _object(attributes[0], ("location", "offset", "format"),
            f"{where}.vertex_layout.buffers[0].attributes[0]")
    _require((attributes[0]["location"], attributes[0]["offset"], attributes[0]["format"])
             == (0, 0, "float32x3"),
             f"{where}: the reviewed depth position is location 0, offset 0, float32x3")
    _object(attributes[1], ("location", "offset", "format"),
            f"{where}.vertex_layout.buffers[0].attributes[1]")
    _require((attributes[1]["location"], attributes[1]["offset"], attributes[1]["format"])
             == (1, 16, "float32x4"),
             f"{where}: the reviewed depth tint is location 1, offset 16, float32x4")
    depth = case.get("depth")
    _require(isinstance(depth, dict), f"{where}: a depth attachment is an object")
    # The clear value rides on the `load: "clear"` arm, so the field set is the
    # one the fixture uses rather than a fixed key list. The store increment
    # adds the three fields of an observable depth landing (`research/docs/23`
    # §3.3, v43).
    store_fields = {"store", "allocation", "view", "expected_hex"}
    allowed = {"format", "width", "height", "load", "clear_depth"} | store_fields
    _require(set(depth) - allowed == set(),
             f"{where}.depth: unexpected fields "
             + ", ".join(sorted(set(depth) - allowed)))
    for field in ("format", "width", "height", "load"):
        _require(field in depth, f"{where}.depth: missing field {field}")
    _require(depth["format"] == "depth32float",
             f"{where}: the reviewed depth attachment is depth32float")
    _require(depth["load"] == "clear",
             f"{where}: the reviewed depth attachment is cleared")
    _require(depth.get("clear_depth") == 1.0,
             f"{where}: the reviewed depth clear is one")
    # Either the pass drops the attachment after testing against it — the only
    # shape before v43 — or it stores it, and then the identity and the expected
    # texels are what makes the landing observable. The two arms are mutually
    # exclusive in both directions: an identity without an action would be an
    # observation nobody claims, and an action without an identity an
    # observation nobody can read.
    store = depth.get("store")
    if store is None:
        _require(not (store_fields & set(depth)),
                 f"{where}: a discarded depth attachment carries no identity or expectation")
        depth_store = None
    else:
        _require(store == "store",
                 f"{where}.depth: unsupported depth store op {store!r}")
        missing = [field for field in ("allocation", "view", "expected_hex")
                   if field not in depth]
        _require(not missing,
                 f"{where}: a stored depth attachment needs its store action, its "
                 "identity and an expectation")
        allocation = _integer(depth["allocation"], f"{where}.depth.allocation")
        view = _integer(depth["view"], f"{where}.depth.view")
        _require(allocation > 0 and view > 0,
                 f"{where}: zero depth attachment identity")
        expected = _hex(depth["expected_hex"], f"{where}.depth.expected_hex")
        _require(depth["expected_hex"] == expected.hex(),
                 f"{where}.depth.expected_hex: the expectation has to be lowercase bytes")
        width = _integer(depth["width"], f"{where}.depth.width", 1)
        height = _integer(depth["height"], f"{where}.depth.height", 1)
        _require(len(expected) == width * height * 4,
                 f"{where}.depth: the expected depth texels do not match the attachment")
        # The falsifiability rule the colour side states for its clear colour:
        # clean depth texels are exactly what a pass that never stored the
        # attachment leaves behind, so an expectation equal to them could not
        # tell "stored" from "dropped".
        _require(expected != struct.pack("<f", depth["clear_depth"]) * (width * height),
                 f"{where}: the expected depth texels equal the clear depth")
        # The depth attachment is a second resource of the same pass, so it
        # cannot be the colour attachment under another name
        # (`research/docs/23` §3.3, v43).
        colours = [case["attachment"]] if "attachment" in case \
            else case.get("attachments", [])
        _require(all(not isinstance(colour, dict)
                     or (colour.get("allocation") != allocation
                         and colour.get("view") != view)
                     for colour in colours),
                 f"{where}.depth: the depth resource has to differ from the colour "
                 "attachment")
        depth_store = (allocation, view, expected)
    test = case.get("depth_test")
    _require(isinstance(test, dict), f"{where}: a depth test is an object")
    _object(test, ("compare", "write"), f"{where}.depth_test")
    _require(test["compare"] == "less" and test["write"] is True,
             f"{where}: the reviewed depth state is a less test with writes on")
    bindings = _list(vertex_buffers, f"{where}.vertex_buffers")
    _require(len(bindings) == 1, f"{where}: the reviewed depth shape binds one stream")
    binding = bindings[0]
    _object(binding, ("allocation", "view", "offset", "length", "initial_hex"),
            f"{where}.vertex_buffers[0]")
    _require(binding["allocation"] > 0 and binding["view"] > 0,
             f"{where}: zero vertex stream identity")
    _require(binding["length"] == stride * quad_indices,
             f"{where}: the reviewed depth stream is six stride-{stride} vertices")
    _require(len(_hex(binding["initial_hex"], f"{where}.vertex_buffers[0].initial_hex"))
             == binding["length"],
             f"{where}: the vertex stream bytes do not match its length")
    _require(indices is not None, f"{where}: the reviewed depth shape is indexed")
    _object(indices, ("allocation", "view", "offset", "length", "initial_hex", "format"),
            f"{where}.indices")
    _require(indices["initial_hex"] == "000001000200030004000500",
             f"{where}: the reviewed depth indices are the two reviewed triangles")
    # The stored depth attachment, when the case declares one, is the
    # `(allocation, view, expected_texels)` triple the assembly below lands
    # beside the colour attachment (`research/docs/23` §3.3, v43).
    return {"vertices": quad_indices, "indices": quad_indices,
            "depth_store": depth_store}


def _stencil_declaration(case, streams, vertex_buffers, indices, where):
    """Pin the reviewed stencil pair (`research/docs/23` §3.3, v47).

    The same reviewed pair module and the same two oversize triangles the depth
    fixture draws — the near one at `z = 0.5` (red) and the far one at
    `z = 0.9` (green) — this time over a cleared `stencil8` attachment with an
    `equal` test against zero whose pass op increments and wraps. The near
    triangle meets the cleared zero, passes and stores the incremented value;
    the far one then meets that stored value and fails, so it is discarded and
    the near tint covers the whole attachment: the expectation is the red texel
    sixteen times, and a rail that ignored the stencil state would show the
    green one, exactly as it would for the depth pair.

    The attachment is rail-owned, the shape the first depth increment published
    (`research/docs/23` §3.3, v36): the pass opens it, masks with it and lets it
    disappear, so the fixture states its format, extent and clear value and
    carries no identity and no readback. The *test* is optional — a pass may
    open the surface without testing against it — but a test without the
    attachment it reads describes nothing, so that half-declared shape is
    refused rather than read as a pass whose fragments are all discarded.

    The v49 increment upgrades that attachment the way v43 upgraded the depth
    one: with a `store` action the fixture also names the allocation and the
    view the texels land in and the bytes it expects there — one byte per texel,
    the whole extent of a `stencil8` surface — and the comparison observes them
    through the same writeback and allocation channel the colour side uses. A
    stencil attachment without one is the v47 shape — opened, masked with and
    dropped — so it carries neither identity nor expectation, and a fixture
    that half-declares the stored shape is refused rather than read as either
    one.
    """
    quad_indices, stride = 6, 32
    _object(streams[0], ("stride", "attributes"), f"{where}.vertex_layout.buffers[0]")
    _require(streams[0]["stride"] == stride,
             f"{where}: the reviewed stencil stream has stride {stride}")
    attributes = _list(streams[0]["attributes"],
                       f"{where}.vertex_layout.buffers[0].attributes")
    _require(len(attributes) == 2, f"{where}: the reviewed stencil stream has two attributes")
    _object(attributes[0], ("location", "offset", "format"),
            f"{where}.vertex_layout.buffers[0].attributes[0]")
    _require((attributes[0]["location"], attributes[0]["offset"], attributes[0]["format"])
             == (0, 0, "float32x3"),
             f"{where}: the reviewed stencil position is location 0, offset 0, float32x3")
    _object(attributes[1], ("location", "offset", "format"),
            f"{where}.vertex_layout.buffers[0].attributes[1]")
    _require((attributes[1]["location"], attributes[1]["offset"], attributes[1]["format"])
             == (1, 16, "float32x4"),
             f"{where}: the reviewed stencil tint is location 1, offset 16, float32x4")
    _require(case.get("depth") is None,
             f"{where}: the reviewed stencil shape carries no depth attachment")
    stencil = case.get("stencil")
    # A stencil test without the attachment it reads has nothing to test
    # against (`research/docs/23` §3.3, v47), so the half-declared shape is
    # refused rather than read as a pass whose state is inert.
    _require(isinstance(stencil, dict),
             f"{where}: a stencil test needs the stencil attachment it reads")
    # The clear value rides on the `load: "clear"` arm, so the field set is the
    # one the fixture uses rather than a fixed key list. The store increment
    # adds the three fields of an observable stencil landing
    # (`research/docs/23` §3.3, v49).
    store_fields = {"store", "allocation", "view", "expected_hex"}
    allowed = {"format", "width", "height", "load", "clear_value"} | store_fields
    _require(set(stencil) - allowed == set(),
             f"{where}.stencil: unexpected fields "
             + ", ".join(sorted(set(stencil) - allowed)))
    for field in ("format", "width", "height", "load", "clear_value"):
        _require(field in stencil, f"{where}.stencil: missing field {field}")
    _require(stencil["format"] == "stencil8",
             f"{where}: the reviewed stencil attachment is stencil8")
    _require(stencil["load"] == "clear",
             f"{where}: the reviewed stencil attachment is cleared")
    clear_value = _integer(stencil["clear_value"], f"{where}.stencil.clear_value", 0, 255)
    _require(clear_value == 0, f"{where}: the reviewed stencil clear is zero")
    # Either the surface disappears with the pass — the only shape before v49 —
    # or the pass keeps it, and then the identity and the expected texels are
    # what makes the landing observable. The two arms are mutually exclusive in
    # both directions, exactly as the depth attachment states them
    # (`research/docs/23` §3.3, v43/v49): an identity without an action would be
    # an observation nobody claims, and an action without an identity an
    # observation nobody can read.
    store = stencil.get("store")
    if store is None:
        _require(not (store_fields & set(stencil)),
                 f"{where}: a discarded stencil attachment carries no identity or expectation")
        stencil_store = None
    else:
        _require(store == "store",
                 f"{where}.stencil: unsupported stencil store op {store!r}")
        missing = [field for field in ("allocation", "view", "expected_hex")
                   if field not in stencil]
        _require(not missing,
                 f"{where}: a stored stencil attachment needs its store action, its "
                 "identity and an expectation")
        allocation = _integer(stencil["allocation"], f"{where}.stencil.allocation")
        view = _integer(stencil["view"], f"{where}.stencil.view")
        _require(allocation > 0 and view > 0,
                 f"{where}: zero stencil attachment identity")
        expected = _hex(stencil["expected_hex"], f"{where}.stencil.expected_hex")
        _require(stencil["expected_hex"] == expected.hex(),
                 f"{where}.stencil.expected_hex: the expectation has to be lowercase bytes")
        width = _integer(stencil["width"], f"{where}.stencil.width", 1)
        height = _integer(stencil["height"], f"{where}.stencil.height", 1)
        _require(len(expected) == width * height,
                 f"{where}.stencil: the expected stencil texels do not match the attachment")
        # The falsifiability rule the colour and depth sides state: clean texels
        # are exactly what a pass that never stored the surface leaves behind,
        # so an expectation equal to them could not tell "stored" from
        # "dropped".
        _require(expected != bytes([clear_value]) * (width * height),
                 f"{where}: the expected stencil texels equal the clear value")
        # The stencil attachment is a second resource of the same pass, so it
        # cannot be the colour attachment under another name
        # (`research/docs/23` §3.3, v49).
        colours = [case["attachment"]] if "attachment" in case \
            else case.get("attachments", [])
        _require(all(not isinstance(colour, dict)
                     or (colour.get("allocation") != allocation
                         and colour.get("view") != view)
                     for colour in colours),
                 f"{where}.stencil: the stencil resource has to differ from the "
                 "colour attachment")
        stencil_store = (allocation, view, expected)
    test = case.get("stencil_test")
    if test is not None:
        # The reviewed state is the one fixture's, whole: `equal` against the
        # value the attachment is cleared to, both masks fully on, and a pass op
        # that increments and wraps. A rail that ignored any of those would land
        # the far triangle's tint instead of the near one's, so the state is
        # pinned as one shape rather than as individually checked fields
        # (`research/docs/23` §3.3, v47).
        _object(test, ("compare", "reference", "read_mask", "write_mask", "fail_op",
                       "depth_fail_op", "pass_op"), f"{where}.stencil_test")
        reference = _integer(test["reference"], f"{where}.stencil_test.reference", 0, 255)
        read_mask = _integer(test["read_mask"], f"{where}.stencil_test.read_mask", 0, 255)
        write_mask = _integer(test["write_mask"], f"{where}.stencil_test.write_mask", 0, 255)
        _require(test["compare"] == "equal" and test["fail_op"] == "keep"
                 and test["depth_fail_op"] == "keep" and test["pass_op"] == "increment_wrap"
                 and (reference, read_mask, write_mask) == (0, 255, 255),
                 f"{where}: the reviewed stencil state is an equal-zero test with both "
                 "masks on that increments and wraps on pass")
    bindings = _list(vertex_buffers, f"{where}.vertex_buffers")
    _require(len(bindings) == 1, f"{where}: the reviewed stencil shape binds one stream")
    binding = bindings[0]
    _object(binding, ("allocation", "view", "offset", "length", "initial_hex"),
            f"{where}.vertex_buffers[0]")
    _require(binding["allocation"] > 0 and binding["view"] > 0,
             f"{where}: zero vertex stream identity")
    _require(binding["length"] == stride * quad_indices,
             f"{where}: the reviewed stencil stream is six stride-{stride} vertices")
    _require(len(_hex(binding["initial_hex"], f"{where}.vertex_buffers[0].initial_hex"))
             == binding["length"],
             f"{where}: the vertex stream bytes do not match its length")
    _require(indices is not None, f"{where}: the reviewed stencil shape is indexed")
    _object(indices, ("allocation", "view", "offset", "length", "initial_hex", "format"),
            f"{where}.indices")
    _require(indices["initial_hex"] == "000001000200030004000500",
             f"{where}: the reviewed stencil indices are the two reviewed triangles")
    # The stored stencil attachment, when the case declares one, is the
    # `(allocation, view, expected_texels)` triple the assembly below lands
    # beside the colour attachment; without one the surface is rail-owned and
    # observed by its effect on the colour side only, so the shape declares no
    # landing and the assembly owes it neither a writeback nor an allocation
    # image (`research/docs/23` §3.3, v47/v49).
    return {"vertices": quad_indices, "indices": quad_indices,
            "stencil_store": stencil_store}


def _base_vertex_declaration(case, streams, vertex_buffers, indices, where):
    """Pin the reviewed base-vertex shape (`research/docs/23` §3.3, v34).

    The layout and the index buffer are the reviewed quad's; the shape adds a
    five-vertex stream whose first vertex is a degenerate centre and a
    `base_vertex` of one, so the same indices draw the reviewed quad only when
    the offset reaches the draw. A rail that ignored it would read the centre
    and three corners and leave part of the attachment at the load's colour,
    which the full-coverage expectation below refuses.
    """
    quad_vertices, quad_indices, quad_stride = 4, 6, 8
    offset = 1
    _require(case.get("base_vertex") == offset,
             f"{where}: the reviewed base-vertex draw offsets its indices by {offset}")
    _require(len(streams) == 1, f"{where}: the reviewed base-vertex shape is one stream")
    _object(streams[0], ("stride", "attributes"), f"{where}.vertex_layout.buffers[0]")
    _require(streams[0]["stride"] == quad_stride,
             f"{where}: the reviewed base-vertex stream has stride {quad_stride}")
    attributes = _list(streams[0]["attributes"],
                       f"{where}.vertex_layout.buffers[0].attributes")
    _require(len(attributes) == 1, f"{where}: the reviewed stream has one attribute")
    _object(attributes[0], ("location", "offset", "format"),
            f"{where}.vertex_layout.buffers[0].attributes[0]")
    _require((attributes[0]["location"], attributes[0]["offset"], attributes[0]["format"])
             == (0, 0, "float32x2"),
             f"{where}: the reviewed attribute is location 0, offset 0, float32x2")
    bindings = _list(vertex_buffers, f"{where}.vertex_buffers")
    _require(len(bindings) == 1, f"{where}: the reviewed base-vertex shape binds one stream")
    binding = bindings[0]
    _object(binding, ("allocation", "view", "offset", "length", "initial_hex"),
            f"{where}.vertex_buffers[0]")
    _require(binding["allocation"] > 0 and binding["view"] > 0,
             f"{where}: zero vertex stream identity")
    _require(binding["length"] == quad_stride * (quad_vertices + offset),
             f"{where}: the reviewed base-vertex stream is the degenerate centre plus the quad")
    _require(len(_hex(binding["initial_hex"], f"{where}.vertex_buffers[0].initial_hex"))
             == binding["length"],
             f"{where}: the vertex stream bytes do not match its length")
    _require(indices is not None, f"{where}: the reviewed base-vertex shape is indexed")
    _object(indices, ("allocation", "view", "offset", "length", "initial_hex", "format"),
            f"{where}.indices")
    _require(indices["allocation"] > 0 and indices["view"] > 0,
             f"{where}: zero index buffer identity")
    width = {"uint16": 2, "uint32": 4}.get(indices["format"])
    _require(width is not None, f"{where}: unsupported index format")
    _require(indices["length"] == quad_indices * width,
             f"{where}: the index buffer is the reviewed six indices")
    index_bytes = _hex(indices["initial_hex"], f"{where}.indices.initial_hex")
    _require(len(index_bytes) == indices["length"],
             f"{where}: the index bytes do not match their length")
    for position in range(0, len(index_bytes), width):
        chunk = index_bytes[position:position + width]
        index = int.from_bytes(chunk, "little")
        _require(index < quad_vertices,
                 f"{where}: index {position // width} names vertex {index} outside the quad")
    return {"vertices": quad_vertices, "indices": quad_indices}


def _inside_scissor(index, width, scissor):
    """Whether the row-major texel `index` falls inside a `[x, y, w, h]` scissor.

    Both rails state the rectangle in framebuffer coordinates with the origin at
    the render area's top-left, which is the same corner the fixture's expected
    texels start from (`research/docs/23` §3.3, v29).
    """
    x, y, scissor_width, scissor_height = scissor
    column = index % width
    row = index // width
    return x <= column < x + scissor_width and y <= row < y + scissor_height


def _resolve_texel(fragment, clear, covered, samples):
    """The resolve of one texel's samples (`research/docs/23` §3.3, v51).

    `covered` of `samples` samples carry the fragment output and the rest the
    colour the pass started from, so each resolved channel is their arithmetic
    mean. A mean that is not exactly representable answers `None` rather than a
    rounded byte: the fixture has to choose colours whose mixes divide exactly,
    which is what keeps the expectation independent of a driver's rounding rule
    — the reviewed case's channels are all chosen that way.
    """
    resolved = bytearray()
    for channel in range(4):
        total = fragment[channel] * covered + clear[channel] * (samples - covered)
        if total % samples:
            return None
        resolved.append(total // samples)
    return bytes(resolved)


def _render_plan(plan, suite):
    """Plan the render cases of a suite (`research/docs/23` §1.2, §5.2).

    A render case is not a compute case: its observable is one colour
    attachment's tightly packed texels, and the pass that draws into it does not
    declare that storage itself. The declaring pass of every render case is
    therefore also a compute case of the same suite, and the attachment resolves
    against that case's own resource table exactly as
    `ComputeTrace::validate_serial_buffer_reuse` resolves it at admission
    (`research/docs/23` §3.6).

    The shape is a whitelist and not a per-case table: the first render
    increment has exactly one render shape, so the shape *is* the review and a
    fixture cannot widen it by renaming a case. The rules mirror the Swift
    oracle's `validateRenderCase`, including the two falsifiability rules — the
    expectation has to cover every texel with the same bytes, and it has to
    differ from the value the pass started from — so a suite that
    `NativeOracle.swift` would refuse cannot pass here either.
    """
    by_id = {case["id"]: case for case in suite.get("cases", [])}
    render_plan = {}
    for case in _list(suite.get("render_cases", []), "suite.render_cases"):
        _require(isinstance(case, dict), "render case: expected an object")
        case_id = _string(case.get("id"), "render case.id")
        where = f"render case {case_id}"
        # The shape is a whitelist and not a per-case table: `present` is the one
        # field the first present increment adds, and it arrives on the same
        # reviewed shape rather than widening it (`research/docs/24` §5.3).
        # The attachment section is one of two mutually exclusive shapes: the
        # single `attachment` object plus the case-level `expected_hex`
        # (v13-v17), or the MRT `attachments` list whose entries each carry
        # their own `expected_hex` (v18).
        required = ("id", "declaring_case", "vertex_entry", "fragment_entry", "metal",
                    "vertices", "viewport", "capture_rails")
        missing = [field for field in required if field not in case]
        _require(not missing, f"{where}: missing fields {', '.join(missing)}")
        unexpected = sorted(set(case) - set(required)
                            - {"attachment", "expected_hex", "attachments", "present", "icb",
                               "vertex_layout", "vertex_buffers", "indices", "scissor",
                               "instance_count", "wildcard_texels", "base_vertex",
                               "depth", "depth_test", "coverage", "cull", "blend",
                               "stencil", "stencil_test", "multisample"})
        _require(not unexpected, f"{where}: unexpected fields {', '.join(unexpected)}")
        single = "attachment" in case
        multiple = "attachments" in case
        # The v46 pass binds *no* colour attachment at all (`research/docs/23`
        # §3.3): neither `attachment` nor `attachments` appears, and the depth
        # section is the whole landing surface. The third shape is admitted
        # only when a `depth` section is declared — the depth parser below
        # still has to find a stored landing — so "no attachments" can never
        # pass as "nothing to compare".
        no_colour = not single and not multiple and "depth" in case
        _require(single != multiple or no_colour,
                 f"{where}: exactly one of attachment and attachments is required")
        if single:
            _require("expected_hex" in case or "depth" in case,
                     f"{where}: missing fields expected_hex")
        elif multiple:
            _require("expected_hex" not in case,
                     f"{where}: an attachment list carries its own expected_hex")
        else:
            # The colour side discards nothing because it does not exist, so a
            # case-level expectation would claim bytes no attachment could
            # produce (`research/docs/23` §3.3, v46).
            _require("expected_hex" not in case,
                     f"{where}: a case with no colour attachment carries no expectation")
        definitions = [] if no_colour else (
            [case["attachment"]] if single
            else _list(case["attachments"], f"{where}.attachments"))
        if multiple:
            _require(2 <= len(definitions) <= 4,
                     f"{where}: the reviewed MRT shapes are two to four attachments")
        _require(case_id not in plan and case_id not in render_plan,
                 f"{where}: duplicate case")
        declaring = _string(case["declaring_case"], f"{where}.declaring_case")
        _require(declaring in plan and declaring in by_id,
                 f"{where}: unknown declaring case {declaring}")
        # A render case replays its declaring case's passes before the render
        # pass, so the declaring case has to be one submission with one
        # dispatch.
        declaring_writes, _, _, group_expectations, _, _, _ = plan[declaring]
        _require(group_expectations is None,
                 f"{where}: the declaring case must be one submission")
        _require(not any(key in by_id[declaring] for key in ("programs", "dispatches",
                                                            "command_buffers")),
                 f"{where}: the declaring case must be one pass over its whole view pool")

        source = case["metal"]
        _object(source, ("path", "sha256"), f"{where}.metal")
        _string(source["path"], f"{where}.metal.path")
        _require(isinstance(source["sha256"], str)
                 and re.fullmatch(r"[0-9a-f]{64}", source["sha256"]),
                 f"{where}: invalid source digest")
        vertex_entry = _string(case["vertex_entry"], f"{where}.vertex_entry")
        fragment_entry = _string(case["fragment_entry"], f"{where}.fragment_entry")
        _require(vertex_entry != fragment_entry,
                 f"{where}: the vertex and fragment entries have to differ")

        vertex_input = _vertex_input_declaration(case, where)
        # The single attachment form spells its expectation at the case level —
        # unless the pass's landing is its stored depth attachment, which is the
        # v45 depth-only shape: the colour attachment still renders, its bytes
        # disappear, and the depth texels are the whole observation. The depth
        # declaration above is what tells the two apart, so this rule can only
        # be stated once that declaration is parsed.
        depth_landing = vertex_input is not None and vertex_input.get("depth_store") is not None
        if single and not depth_landing:
            _require("expected_hex" in case, f"{where}: missing fields expected_hex")
        if vertex_input is None:
            _require(case["vertices"] == 3, f"{where}: expected the full-screen triangle")
            _require(not multiple,
                     f"{where}: the milestone vertex_id shape renders one attachment")
        else:
            _require(case["vertices"] == vertex_input["indices"],
                     f"{where}: the reviewed indexed quad draws {vertex_input['indices']} indices")
            _require("present" not in case and "icb" not in case,
                     f"{where}: a vertex-input case carries neither a present action nor an ICB")
        # One attachment at a time: the single shape is the v13-v17 branch with
        # its expectation at the case level, and every MRT entry restates the
        # same per-field rules with its own expectation. The v19 increment adds
        # the store operation: a `dontcare` attachment still renders and still
        # resolves against its declaring view, but its bytes disappear from the
        # observable surface, so it carries no expectation and no observation
        # (`research/docs/23` §3.6).
        # The pass's scissor, when it declares one, makes the coverage of every
        # attachment exact: texels inside the rectangle carry the fragment
        # output and the rest keep whatever the load op handed the pass
        # (`research/docs/23` §3.3, v29). Both rails put the rectangle in
        # framebuffer coordinates, so the comparison can reason about it
        # directly instead of asking "both halves have to appear".
        scissor = case.get("scissor")
        if scissor is not None:
            scissor = _list(scissor, f"{where}.scissor")
            _require(len(scissor) == 4, f"{where}: a scissor is four numbers")
            scissor = [_integer(value, f"{where}.scissor[{index}]")
                       for index, value in enumerate(scissor)]
        # The zero-colour shape (`research/docs/23` §3.3, v46/v50) has no
        # colour attachment to state the pass's render area, so the raster the
        # viewport and the scissor are measured against is the depth surface the
        # case stores — or the stencil surface, once a stencil-only shape is
        # reviewed (`§3.3`, v47). The viewport has to cover that raster's extent
        # and the scissor, when the case declares one, has to be a non-empty
        # rectangle inside it: the two rules the colour side states once per
        # attachment, and the two the Swift oracle's own zero-colour branch
        # states, so a suite `NativeOracle.swift` would refuse cannot pass here
        # either.
        if no_colour:
            raster = next(name for name in ("depth", "stencil") if name in case)
            section = case[raster]
            _require(isinstance(section, dict), f"{where}.{raster}: expected an object")
            width = _integer(section.get("width"), f"{where}.{raster}.width", 1)
            height = _integer(section.get("height"), f"{where}.{raster}.height", 1)
            viewport = _list(case["viewport"], f"{where}.viewport")
            _require(viewport == [0, 0, width, height],
                     f"{where}: the viewport must cover the {raster} attachment")
            if scissor is not None:
                x, y, scissor_width, scissor_height = scissor
                _require(scissor_width > 0 and scissor_height > 0
                         and x + scissor_width <= width and y + scissor_height <= height,
                         f"{where}: a scissor has to be a non-empty rectangle inside "
                         f"the {raster} attachment")
        # The wildcard channel (`research/docs/23` §3.3, v33): a case may name
        # the texels it does not claim, and only a `dontcare` load has bytes
        # that may legitimately be unclaimed. The list has to leave at least one
        # texel observed and at least one wild, or the fixture would prove
        # "everything" or "nothing" rather than a partial coverage.
        # The coverage claim (`research/docs/23` §3.3, v38): a clearing pass
        # whose draw covers only part of the attachment may say so, and the
        # comparison then reads every texel as "the fragment output or the
        # clear colour, and both have to appear". The default stays the
        # milestone's stricter rule — every texel is the output — because a
        # fixture that says nothing claims everything.
        coverage = case.get("coverage")
        if coverage is not None:
            _require(coverage == "partial",
                     f"{where}: the only coverage claim is \"partial\"")
            _require(single,
                     f"{where}: the coverage claim is the single-attachment shape")
        wildcard_texels = case.get("wildcard_texels")
        if wildcard_texels is not None:
            _require(single,
                     f"{where}: the wildcard channel is the single-attachment shape")
            wildcard_texels = _list(wildcard_texels, f"{where}.wildcard_texels")
            _require(wildcard_texels,
                     f"{where}: a wildcard list has to name at least one texel")
            seen_texels = set()
            for position, value in enumerate(wildcard_texels):
                texel = _integer(value, f"{where}.wildcard_texels[{position}]", 0)
                _require(texel not in seen_texels,
                         f"{where}: duplicate wildcard texel {texel}")
                seen_texels.add(texel)
            wildcard_texels = sorted(seen_texels)
            _require(definitions[0].get("store", "store") == "store",
                     f"{where}: a wildcard list needs a stored attachment")
        # The pass-wide multisample raster (`research/docs/23` §3.3, v51): the
        # first increment reviews one shape — a single colour attachment opened
        # from a clear, four samples, no depth or stencil surface, no present
        # action, no ICB and no wildcard texels — and its expectation follows
        # the resolve rule instead of the coverage rule below. Every rail the
        # marker names executes it: the trace rails since v51, the object rails
        # since the v52 recording entry (`research/docs/23` §3.3, v51/v52).
        multisample = case.get("multisample")
        if multisample is not None:
            _require(single,
                     f"{where}: the multisample raster is the single-attachment shape")
            _object(multisample, ("sample_count",), f"{where}.multisample")
            _require(_integer(multisample["sample_count"],
                              f"{where}.multisample.sample_count", 1) == 4,
                     f"{where}: the reviewed multisample raster is four samples")
            _require(case["attachment"].get("load") == "clear",
                     f"{where}: the reviewed multisample pass opens its attachment from a "
                     "clear")
            # The depth surface beside the raster is admitted from v53 on and
            # the stencil surface from v55 (`research/docs/23` §3.3, v53/v55):
            # both are rail-owned — the pass tests and writes them, but keeping
            # their texels would need a resolve the two APIs spell differently —
            # and the expectation then follows the pair's own uniform rule
            # instead of the resolve rule. The two surfaces stay mutually
            # exclusive.
            _require(not ("depth" in case and "stencil" in case),
                     f"{where}: the multisample raster opens one depth-stencil surface")
            if "depth" in case:
                _require(case["depth"].get("store") is None,
                         f"{where}: a multisampled depth surface is rail-owned: the depth "
                         "resolve filters are a later increment")
                _require(coverage is None,
                         f"{where}: a multisample pass with a depth surface claims no partial "
                         "coverage")
            elif "stencil" in case:
                _require(case["stencil"].get("store") is None,
                         f"{where}: a multisampled stencil surface is rail-owned: the stencil "
                         "resolve is a later increment")
                _require(coverage is None,
                         f"{where}: a multisample pass with a stencil surface claims no partial "
                         "coverage")
            else:
                _require(coverage == "partial",
                         f"{where}: the multisample raster has to claim partial coverage")
            _require("present" not in case and "icb" not in case,
                     f"{where}: a multisample case carries neither a present action nor "
                     "an ICB")
            _require(wildcard_texels is None,
                     f"{where}: the multisample raster claims every texel it resolves")
            # The trace rails' footprint proof is the only gate that would
            # notice an offset index span, and no reviewed fixture covers the
            # shape; the object rail has no entry that carries both. Refusing it
            # here keeps the fixture gate as strict as the rails
            # (`research/docs/23` §3.3, v54 review H1).
            _require(case.get("base_vertex", 0) == 0,
                     f"{where}: the reviewed multisample shapes carry no base vertex")
        expected_bytes = []
        parsed = []
        for position, attachment in enumerate(definitions):
            attachment_where = (f"{where}.attachment" if single
                                else f"{where}.attachments[{position}]")
            _require(isinstance(attachment, dict), f"{attachment_where}: expected an object")
            allowed = {"allocation", "view", "format", "width", "height",
                       "load", "store", "clear_hex", "initial_hex"}
            if multiple:
                allowed.add("expected_hex")
            _require(set(attachment).issubset(allowed),
                     f"{attachment_where}: unexpected fields")
            allocation = _integer(attachment.get("allocation"),
                                  f"{attachment_where}.allocation")
            view = _integer(attachment.get("view"), f"{attachment_where}.view")
            _require(allocation > 0 and view > 0,
                     f"{attachment_where}: zero attachment identity")
            # Both 8-bit UNORM layouts are admitted (`research/docs/23` §3.3,
            # v21): the shader stores the same colour either way and the memory
            # bytes follow the attachment's channel order, so the fixture's
            # texels pin which layout the comparison is observing.
            _require(attachment.get("format") in ("rgba8_unorm", "bgra8_unorm", "r32float"),
                     f"{attachment_where}: unsupported attachment format")
            width = _integer(attachment.get("width"), f"{attachment_where}.width", 1)
            height = _integer(attachment.get("height"), f"{attachment_where}.height", 1)
            # The rail executes attachments up to `max_attachment_dimension`
            # (`research/docs/23` §3.3, v27): the reviewed fragment stages are
            # extent-independent, so any 1..=4 square or rectangle is a shape a
            # fixture may pin, as long as every attachment of one pass shares it.
            _require(1 <= width <= 4 and 1 <= height <= 4,
                     f"{attachment_where}: the attachment extent is one to four "
                     "texels per axis")
            viewport = _list(case["viewport"], f"{attachment_where}.viewport")
            _require(viewport == [0, 0, width, height],
                     f"{attachment_where}: the viewport must cover the attachment")
            # The depth attachment is a second raster with the pass's own
            # extent (`research/docs/23` §3.3, v36): the two have to agree, the
            # same rule the viewport states for the colour side.
            if case.get("depth") is not None:
                _require(case["depth"].get("width") == width
                         and case["depth"].get("height") == height,
                         f"{attachment_where}: the depth attachment has to match the "
                         "colour extent")
            # The stencil attachment is a raster with the pass's own extent, the
            # same rule the depth surface states (`research/docs/23` §3.3, v47).
            if case.get("stencil") is not None:
                _require(case["stencil"].get("width") == width
                         and case["stencil"].get("height") == height,
                         f"{attachment_where}: the stencil attachment has to match the "
                         "colour extent")
            if scissor is not None:
                x, y, scissor_width, scissor_height = scissor
                _require(scissor_width > 0 and scissor_height > 0
                         and x + scissor_width <= width and y + scissor_height <= height,
                         f"{where}: a scissor has to be a non-empty rectangle inside "
                         "the attachment")
            store = attachment.get("store", "store")
            _require(store in ("store", "dontcare"),
                     f"{attachment_where}: a discarded attachment cannot be compared")
            extent = width * height * 4
            # A discarded attachment carries no expectation and no observation:
            # the single-attachment form spells its expectation at the case
            # level, so a discard cannot be expressed there, and an expectation
            # arriving for a discarded entry is refused. Its load shape stays
            # pinned — the pass still performs the load — only the byte
            # comparison disappears.
            if store == "dontcare":
                _require(not single or depth_landing,
                         f"{attachment_where}: a discarded attachment carries no expectation")
                _require("expected_hex" not in attachment,
                         f"{attachment_where}: a discarded attachment carries no expected_hex")
                if single:
                    # The depth-only shape: the case-level expectation must stay
                    # absent too, so nothing claims bytes the pass discards.
                    _require("expected_hex" not in case,
                             f"{attachment_where}: a discarded attachment carries no expectation")
                load = attachment.get("load")
                if load == "clear":
                    clear = _hex(attachment.get("clear_hex"), f"{attachment_where}.clear_hex")
                    _require(len(clear) == 4, f"{attachment_where}: a clear colour is four bytes")
                    _require("initial_hex" not in attachment,
                             f"{attachment_where}: a cleared attachment carries no initial bytes")
                elif load == "load":
                    previous = _hex(attachment.get("initial_hex"),
                                    f"{attachment_where}.initial_hex")
                    _require(len(previous) == extent,
                             f"{attachment_where}: initial texels do not match the attachment")
                    _require("clear_hex" not in attachment,
                             f"{attachment_where}: a loaded attachment carries no clear colour")
                elif load == "dontcare":
                    _require("clear_hex" not in attachment,
                             f"{attachment_where}: a dontcare load carries no clear colour")
                    _require("initial_hex" not in attachment,
                             f"{attachment_where}: a dontcare load carries no initial bytes")
                else:
                    raise CaptureError(f"{attachment_where}: unknown attachment load op {load!r}")
                parsed.append((attachment, allocation, view, None))
                continue
            # The stored arm keeps the v13-v18 shape: its expectation is the
            # whole reason the attachment is comparable.
            if multiple:
                _require("expected_hex" in attachment,
                         f"{attachment_where}: a stored attachment needs expected_hex")
            expected = _hex(case["expected_hex"] if single else attachment.get("expected_hex"),
                            f"{attachment_where}.expected_hex")
            _require(len(expected) == extent,
                     f"{attachment_where}: expected texel bytes do not match the attachment")
            texel = expected[:4]
            texels = [expected[offset:offset + 4] for offset in range(0, len(expected), 4)]
            # Clearing and loading agree about what a drawn texel is — one
            # fragment output, repeated — and disagree about the rest: a
            # cleared attachment has no previous bytes to compare against
            # (`research/docs/23` §1.3), while a loaded one is expected to keep
            # them where the draw missed (§3.3). The v20 increment adds the
            # undefined pre-pass contents: a `dontcare` load carries neither
            # clear colour nor initial bytes, so its expectation follows the
            # clearing arm's uniform rule and still has to differ from the
            # bytes the declaring case pins for the same view (`research/docs/23`
            # §13). The classification below is the loading rule; the clearing
            # and dontcare arms keep the milestone's stricter one.
            load = attachment.get("load")
            if wildcard_texels is not None:
                texel_count = width * height
                _require(load == "dontcare",
                         f"{attachment_where}: only a dontcare load may leave texels "
                         "unclaimed")
                _require(len(wildcard_texels) < texel_count,
                         f"{attachment_where}: a wildcard list has to leave at least one "
                         "texel observed")
                for wildcard in wildcard_texels:
                    _require(wildcard < texel_count,
                             f"{attachment_where}: wildcard texel {wildcard} is outside the "
                             "attachment")
            if load == "clear":
                clear = _hex(attachment.get("clear_hex"), f"{attachment_where}.clear_hex")
                _require(len(clear) == 4, f"{attachment_where}: a clear colour is four bytes")
                _require("initial_hex" not in attachment,
                         f"{attachment_where}: a cleared attachment carries no initial bytes")
                _require(clear != texel,
                         f"{attachment_where}: the clear colour equals the expected texel")
                if (multisample is not None and case.get("depth") is None
                        and case.get("stencil") is None):
                    # The multisample resolve (`research/docs/23` §3.3, v51):
                    # every texel is the mean of the samples a primitive
                    # covered, so the expectation has to be a k-of-`sample_count`
                    # mix of the fragment output and the clear colour — and both
                    # extremes and at least one partial mix have to appear, or
                    # the fixture would claim a raster a single-sample pass could
                    # produce (`_resolve_texel` refuses a mix that is not exactly
                    # representable).
                    samples = multisample["sample_count"]
                    covered_seen = set()
                    for index, chunk in enumerate(texels):
                        for covered in range(samples + 1):
                            if chunk == _resolve_texel(texel, clear, covered, samples):
                                covered_seen.add(covered)
                                break
                        else:
                            raise CaptureError(
                                f"{attachment_where}: texel {index} is not the resolve of "
                                f"any coverage of the {samples}-sample raster")
                    _require(any(0 < covered < samples for covered in covered_seen),
                             f"{attachment_where}: a multisample expectation needs at "
                             "least one partially covered texel")
                    _require(0 in covered_seen and samples in covered_seen,
                             f"{attachment_where}: a multisample expectation needs both "
                             "a fully covered and an uncovered texel")
                elif coverage == "partial":
                    # The draw covers part of the attachment: every texel is
                    # the fragment output or the colour the pass started from,
                    # and both have to appear, or the fixture would claim
                    # either "everything" or "nothing"
                    # (`research/docs/23` §3.3, v38).
                    drawn = 0
                    kept = 0
                    for index, chunk in enumerate(texels):
                        if chunk == texel:
                            drawn += 1
                        elif chunk == clear:
                            kept += 1
                        else:
                            raise CaptureError(
                                f"{attachment_where}: texel {index} has to be the "
                                "fragment output or the clear colour")
                    _require(0 < drawn < len(texels) and kept > 0,
                             f"{attachment_where}: a partial coverage claim needs both "
                             "drawn and clear texels")
                elif scissor is None:
                    if vertex_input and vertex_input.get("instanced"):
                        # The instanced fixture covers each half of the
                        # attachment with its own instance tint
                        # (`research/docs/23` §3.3, v31): the left half has to
                        # carry the first record's tint and the right half the
                        # second's, each uniform. A uniform expectation could
                        # not show that the per-instance stream stepped, and a
                        # swapped pair would read as agreement on the wrong
                        # halves.
                        left, right = vertex_input["tints"]
                        _require(left != right and left != clear and right != clear,
                                 f"{attachment_where}: the two instance tints have to "
                                 "differ from each other and from the clear colour")
                        for index, chunk in enumerate(texels):
                            column = index % width
                            expected_chunk = left if column < width // 2 else right
                            _require(chunk == expected_chunk,
                                     f"{attachment_where}: texel {index} has to carry the "
                                     "instance tint of its half")
                    else:
                        _require(all(chunk == texel for chunk in texels),
                                 f"{attachment_where}: every texel of the expectation has to be "
                                 "the fragment output")
                else:
                    covered = 0
                    for index, chunk in enumerate(texels):
                        inside = _inside_scissor(index, width, scissor)
                        expected_chunk = texel if inside else clear
                        _require(chunk == expected_chunk,
                                 f"{attachment_where}: texel {index} has to be "
                                 + ("the fragment output" if inside else "the clear colour")
                                 + " under the declared scissor")
                        covered += 1 if inside else 0
                    _require(0 < covered < len(texels),
                             f"{attachment_where}: the scissor has to clip part of the "
                             "attachment, or the fixture cannot show it ran")
            elif load == "load":
                previous = _hex(attachment.get("initial_hex"),
                                f"{attachment_where}.initial_hex")
                _require(len(previous) == len(expected),
                         f"{attachment_where}: initial texels do not match the attachment")
                _require(previous != expected,
                         f"{attachment_where}: the initial texels equal the expectation")
                _require("clear_hex" not in attachment,
                         f"{attachment_where}: a loaded attachment carries no clear colour")
                # The loading pass uploads the declaring case's own bytes, so
                # the case's `initial_hex` has to be exactly what that case
                # declares.
                declaring_buffers = by_id[declaring]["buffers"]
                declared_bytes = [buffer for buffer in declaring_buffers
                                  if buffer["allocation"] == allocation
                                  and buffer["view"] == view]
                _require(len(declared_bytes) == 1,
                         f"{attachment_where}: the declaring case has to declare exactly "
                         "the attachment view")
                _require(declared_bytes[0].get("initial_hex")
                         == attachment.get("initial_hex"),
                         f"{attachment_where}: the declared view's bytes are not the "
                         "attachment's initial texels")
                # Partial coverage, both directions: every texel is either the
                # fragment output or the byte the load handed it, every drawn
                # texel carries the same output, and both halves appear.
                drawn = {chunk for position, chunk in enumerate(texels)
                         if chunk != previous[position * 4:position * 4 + 4]}
                kept = sum(1 for position, chunk in enumerate(texels)
                           if chunk == previous[position * 4:position * 4 + 4])
                _require(len(drawn) == 1,
                         f"{attachment_where}: drawn texels disagree about the fragment "
                         f"output: {sorted(drawn)}")
                _require(0 < kept < len(texels),
                         f"{attachment_where}: a loaded attachment needs both drawn and "
                         "kept texels")
            elif load == "dontcare":
                # A `dontcare` load claims no previous bytes, so every texel it
                # *observes* has to be the fragment output and the unclaimed
                # ones are named by the wildcard list (`research/docs/23` §3.3,
                # v33). Without a list the milestone's stricter rule — and its
                # exact message — stay: the fixture then has no way to say
                # which bytes it does not claim, so it claims all of them.
                if wildcard_texels is None:
                    _require(all(chunk == texel for chunk in texels),
                             f"{attachment_where}: every texel of a dontcare load has to be "
                             "the fragment output")
                else:
                    claimed = [position for position in range(len(texels))
                               if position not in wildcard_texels]
                    _require(claimed,
                             f"{attachment_where}: a wildcard list has to leave at least one "
                             "texel observed")
                    for position in claimed:
                        _require(texels[position] == texel,
                                 f"{attachment_where}: texel {position} of a dontcare load "
                                 "has to be the fragment output")
                _require("clear_hex" not in attachment,
                         f"{attachment_where}: a dontcare load carries no clear colour")
                _require("initial_hex" not in attachment,
                         f"{attachment_where}: a dontcare load carries no initial bytes")
                # The declaring pass still pins the view's bytes, but a
                # `dontcare` load does not hand them to the pass, so they must
                # not equal the expectation: that is what shows the undefined
                # contents never entered the observation.
                declared_bytes = [buffer for buffer in by_id[declaring]["buffers"]
                                  if buffer["allocation"] == allocation
                                  and buffer["view"] == view]
                _require(len(declared_bytes) == 1,
                         f"{attachment_where}: the declaring case has to declare exactly "
                         "the attachment view")
                _require(_hex(declared_bytes[0].get("initial_hex"),
                              f"{attachment_where} declared initial bytes") != expected,
                         f"{attachment_where}: the declared view's bytes equal the "
                         "expectation")
            else:
                raise CaptureError(f"{attachment_where}: unknown attachment load op {load!r}")
            parsed.append((attachment, allocation, view, expected))
            expected_bytes.append(expected)
        # Core admission refuses an all-discarded pass
        # (`AllRenderAttachmentsDiscarded`), so the suite has to keep at least
        # one attachment on the observable surface or "nothing landed" would
        # pass as "landed correctly". A stored depth attachment is a landing too
        # (`research/docs/23` §3.3, v43/v45): the depth-only shape discards every
        # colour attachment and keeps the depth surface, and the depth texels are
        # then the whole comparison.
        _require(expected_bytes or depth_landing,
                 f"{where}: every colour attachment discards, leaving no observable landing point")
        # The two reviewed MRT locations write two different byte strings, so a
        # dual case whose locations read back the same texels could not show
        # that both outputs landed (`4080c0ff` vs `ff8040c0`). A discarded
        # location has no expectation and takes no part in the comparison.
        if multiple and len(expected_bytes) >= 2:
            _require(len(set(expected_bytes)) == len(expected_bytes),
                     f"{where}: the attachments read back the same texels")
        texel = expected_bytes[0][:4] if expected_bytes else None

        # The present section is optional: a case without it is the v13 case and
        # must not grow a present observation in a capture (the exact-set rule
        # `validate_capture` applies to every result).
        present = None
        if "present" in case:
            # A present action hands a *stored* colour attachment on, so a case
            # whose only landing is its depth surface cannot carry one.
            _require(texel is not None,
                     f"{where}: a present action needs a stored colour attachment")
            present = _present_declaration(case["present"], texel, where)

        # The indirect section is optional too: a case that carries it replays
        # its full-screen triangle from one indirect draw command instead of a
        # direct draw, and a capture that reports bytes without the replay
        # record cannot prove which path produced them.
        icb = None
        if "icb" in case:
            icb = _icb_declaration(case["icb"], ("draw", "draw_indexed"), where)

        rails = _list(case["capture_rails"], f"{where}.capture_rails")
        _require(rails and len(set(rails)) == len(rails)
                 and all(isinstance(rail, str) and rail in ALLOCATION_OBSERVATIONS
                         for rail in rails),
                 f"{where}: capture_rails has to name distinct known backends")

        # Every attachment resolves against the declaring case's own table:
        # one of its declared views has to be the attachment, it has to be
        # read-only (a compute pass that *wrote* the view the render pass
        # stores would make the order inexpressible), and its byte range has to
        # agree with the extent the attachment restates.
        declaring_buffers = by_id[declaring]["buffers"]
        writes = []
        images = {}
        identities = []
        seen_allocations = set()
        written = {identity[0] for identity, _ in declaring_writes}
        for position, (attachment, allocation, view, expected) in enumerate(parsed):
            attachment_where = (f"{where}.attachment" if single
                                else f"{where}.attachments[{position}]")
            _require(allocation not in seen_allocations,
                     f"{attachment_where}: the attachments have to name distinct allocations")
            seen_allocations.add(allocation)
            declared = [buffer for buffer in declaring_buffers
                        if buffer["allocation"] == allocation and buffer["view"] == view]
            _require(len(declared) == 1,
                     f"{attachment_where}: the declaring case has to declare exactly "
                     "the attachment view")
            declared = declared[0]
            _require(declared["access"] == "read",
                     f"{attachment_where}: the declaring pass must only read the "
                     "attachment view")
            extent = attachment["width"] * attachment["height"] * 4
            _require(declared["length"] == extent,
                     f"{attachment_where}: attachment extent disagrees with the "
                     "declaring view")
            offset = declared["offset"]
            size = declared["allocation_size"]
            _require(offset + extent <= size,
                     f"{attachment_where}: the declaring view is outside its allocation")
            # A discarded attachment's declaring view still takes part in the
            # touched count and in the declaration resolution, but its landing
            # never enters the observation surface: no writeback and no
            # allocation image are owed for it (`research/docs/23` §3.6, v19).
            if expected is None:
                continue

            # The allocation image is the declaring case's own image with the
            # attachment's landing overlaid: the render result observes the
            # whole allocation, so a guard byte or an untouched neighbour that
            # the render pass did not store into stays part of the comparison.
            image = bytearray(plan[declaring][1][allocation])
            _require(len(image) == size, f"{attachment_where}: inconsistent allocation size")
            image[offset:offset + len(expected)] = expected
            images[allocation] = bytes(image)
            writes.append(((allocation, view, offset), expected))
            identities.append((allocation, view, offset, len(expected)))
            written.add(allocation)
        touched = set(plan[declaring][1])
        # The stored depth attachment (`research/docs/23` §3.3, v43) lands
        # through the channel the colour attachments already use: one writeback
        # under the depth view and one allocation image. Its view is declared by
        # the declaring pass exactly as a colour attachment's is — the reviewed
        # declaring kernel carries a third *read* binding for it — so the same
        # three questions are asked here: the declaring case declares that one
        # view, it only reads it (a compute write would race the store), and the
        # declaration's byte range is the depth extent the attachment restates.
        # The allocation image is the declaring case's own image with the landing
        # overlaid, which is what makes the guard bytes around the view part of
        # the comparison rather than something the fixture could forget.
        if vertex_input is not None and vertex_input.get("depth_store") is not None:
            depth_allocation, depth_view, depth_expected = vertex_input["depth_store"]
            declared = [buffer for buffer in by_id[declaring]["buffers"]
                        if buffer["allocation"] == depth_allocation
                        and buffer["view"] == depth_view]
            _require(len(declared) == 1,
                     f"{where}: the declaring case has to declare exactly the depth "
                     "attachment view")
            declared = declared[0]
            _require(declared["access"] == "read",
                     f"{where}: the declaring pass must only read the depth attachment view")
            _require(declared["length"] == len(depth_expected),
                     f"{where}: the depth view's byte range disagrees with the attachment")
            _require(_hex(declared.get("initial_hex"), f"{where} declared depth bytes")
                     != depth_expected,
                     f"{where}: the declared depth view's bytes equal the expectation")
            image = bytearray(plan[declaring][1][depth_allocation])
            _require(len(image) == declared["allocation_size"],
                     f"{where}: inconsistent depth allocation size")
            image[declared["offset"]:declared["offset"] + len(depth_expected)] = depth_expected
            images[depth_allocation] = bytes(image)
            writes.append(((depth_allocation, depth_view, declared["offset"]), depth_expected))
            identities.append((depth_allocation, depth_view, declared["offset"],
                               len(depth_expected)))
            written.add(depth_allocation)
            touched.add(depth_allocation)
        # The stored stencil attachment (`research/docs/23` §3.3, v49) lands
        # through the same channel one byte wide: one writeback under the
        # stencil view and one allocation image, and the same questions the
        # depth landing asks. Its view is declared by the declaring pass exactly
        # as a colour attachment's is — the reviewed declaring kernel carries a
        # third *read* binding for it — so the declaring case declares that one
        # view, it only reads it (a compute write would race the store), the
        # declaration's byte range is the stencil extent the attachment
        # restates, and the bytes it pins are not the stored texels, or the
        # fixture could not tell "stored" from "never written". The allocation
        # image is the declaring case's own image with the landing overlaid,
        # which is what makes the guard bytes around the one-byte-per-texel view
        # part of the comparison rather than something the fixture could
        # forget.
        if vertex_input is not None and vertex_input.get("stencil_store") is not None:
            stencil_allocation, stencil_view, stencil_expected = (
                vertex_input["stencil_store"])
            declared = [buffer for buffer in by_id[declaring]["buffers"]
                        if buffer["allocation"] == stencil_allocation
                        and buffer["view"] == stencil_view]
            _require(len(declared) == 1,
                     f"{where}: the declaring case has to declare exactly the stencil "
                     "attachment view")
            declared = declared[0]
            _require(declared["access"] == "read",
                     f"{where}: the declaring pass must only read the stencil attachment view")
            _require(declared["length"] == len(stencil_expected),
                     f"{where}: the stencil view's byte range disagrees with the attachment")
            _require(_hex(declared.get("initial_hex"), f"{where} declared stencil bytes")
                     != stencil_expected,
                     f"{where}: the declared stencil view's bytes equal the expectation")
            image = bytearray(plan[declaring][1][stencil_allocation])
            _require(len(image) == declared["allocation_size"],
                     f"{where}: inconsistent stencil allocation size")
            image[declared["offset"]:declared["offset"] + len(stencil_expected)] = (
                stencil_expected)
            images[stencil_allocation] = bytes(image)
            writes.append(((stencil_allocation, stencil_view, declared["offset"]),
                           stencil_expected))
            identities.append((stencil_allocation, stencil_view, declared["offset"],
                               len(stencil_expected)))
            written.add(stencil_allocation)
            touched.add(stencil_allocation)
        # The wildcard mask is stated once per observed attachment, in the
        # absolute byte offsets of its allocation, so the writeback comparison
        # and the allocation-image comparison read the same set.
        wildcards = {}
        if wildcard_texels is not None and writes:
            (allocation, view, offset), _ = writes[0]
            wildcards[(allocation, view, offset)] = frozenset(
                offset + texel * 4 + byte
                for texel in wildcard_texels
                for byte in range(4))
        render_plan[case_id] = RenderExpectation(
            writes=writes,
            allocations=images,
            touched=touched,
            written=written,
            rails=frozenset(rails),
            # The single-attachment shape declares one landing, so one identity
            # is the whole review surface. A case that also stores its depth
            # attachment observes two resources, and then the identity list is
            # what has to cover both (`research/docs/23` §3.3, v43).
            attachment=identities[0] if single and len(identities) == 1 else identities,
            present=present,
            icb=icb,
            wildcards=wildcards)
    return render_plan


def validate_capture(suite, digest, report, required_backend=None):
    """Raise CaptureError for invalid captures; success alone does not claim parity."""
    plan = _suite_plan(suite)
    render_plan = _render_plan(plan, suite)
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
    # The Swift reference oracle reports bytes but not device-buffer copy
    # counters: it is not a provider. The count contract therefore applies to
    # every other backend (research/docs/15 §5).
    provider_backend = report["backend"] != "native-metal"
    for result in results:
        _require(isinstance(result, dict) and "id" in result,
                 "capture result: expected an object with an id")
        base = {"id", "completion", "writebacks", "allocations"}
        counted = base | {"copy_in", "copy_out"}
        grouped = counted | {"group_counts"}
        # The present and heap observations are the keys a suite may declare on
        # top of an otherwise unchanged result shape: they replace no existing
        # field and they do not relax the counter-pair rule below.
        _require(set(result) - {"present", "heap", "icb"} in (base, counted, grouped),
                 "capture result: expected fields "
                 + ", ".join(sorted(base)) + ", optionally with copy_in and copy_out"
                 " plus the per-command-buffer group_counts")
        counts = (result.get("copy_in"), result.get("copy_out"))
        _require((counts[0] is None) == (counts[1] is None),
                 "capture result: copy_in and copy_out are recorded together")
        case_id = _string(result["id"], "capture result.id")
        where = f"case {case_id}"
        _require(case_id in plan or case_id in render_plan, f"{where}: unknown case")
        _require(case_id not in seen, f"{where}: duplicate case")
        seen.add(case_id)
        _require(result["completion"] == "CompletedVisible",
                 f"{where}: completion must be CompletedVisible, got {result['completion']!r}")
        if case_id in render_plan:
            # A render case reports one observation and one only: the
            # attachments' own allocations and their writebacks, one of each
            # per attachment. The identity check below is the
            # attachment-versus-buffer rule — an attachment cannot be satisfied
            # by a buffer writeback, and a buffer writeback cannot be reported
            # where the attachment belongs.
            expectation = render_plan[case_id]
            _compare_observation(result, expectation.writes, expectation.allocations, where,
                                 expectation.wildcards)
            attachments = expectation.attachment
            if isinstance(attachments, tuple):
                attachments = [attachments]
            _require({identity for identity, _ in expectation.writes}
                     == {attachment[:3] for attachment in attachments},
                     f"{where}: the attachment writebacks have to be exactly the "
                     "declared attachments")
            _require(set(expectation.allocations)
                     == {attachment[0] for attachment in attachments},
                     f"{where}: the attachment allocations have to be exactly the "
                     "declared attachments")
            for attachment, (_, expected) in zip(attachments, expectation.writes):
                _require(attachment[3] == len(expected),
                         f"{where}: an attachment plan has to cover its own texels")
            if counts[0] is not None and provider_backend:
                # One submission carries the declaring pass and the render pass.
                # The declaring pass copies its own allocations in and the ones
                # it writes out; the attachment allocation is one of the written
                # ones, and its single `vkCmdCopyImageToBuffer` readback *is*
                # that allocation's copy-out (it replaces the device-buffer
                # readback the pool would otherwise do), so the counts stay one
                # per touched and one per written allocation
                # (`research/docs/23` §3.5, §5.3).
                expected_in = len(expectation.touched)
                expected_out = len(expectation.written)
                _require(counts[0] == expected_in,
                         f"{where}: copy_in {counts[0]} does not match {expected_in} "
                         "touched allocations")
                _require(counts[1] == expected_out,
                         f"{where}: copy_out {counts[1]} does not match {expected_out} "
                         "written allocations")
            # The present observation follows the same marker rule the bytes do:
            # a rail the case's `capture_rails` names has to report the counts
            # the suite declares, and a rail the marker does not name has to
            # leave the observation out rather than report a run it does not
            # own (`research/docs/24` §5.3, §5.5).
            if expectation.present is None:
                _require("present" not in result,
                         f"{where}: the suite declares no present observation for this case")
            elif report["backend"] in expectation.rails:
                _require("present" in result,
                         f"{where}: {report['backend']} has to report the present observation "
                         "the suite declares")
                _present_observation(result["present"], expectation.present, where)
            else:
                _require("present" not in result,
                         f"{where}: {report['backend']} must not report the present observation "
                         "of a case its marker does not name")
            _require("heap" not in result,
                     f"{where}: a render case carries no heap observation")
            if expectation.icb is None:
                _require("icb" not in result,
                         f"{where}: the suite declares no indirect command for this case")
            else:
                _require("icb" in result,
                         f"{where}: {report['backend']} has to report the replayed indirect "
                         "command the suite declares")
                _icb_observation(result["icb"], expectation.icb, where)
        else:
            (expected_writes, expected_allocations, texture_count, group_expectations,
             heap, icb, case_rails) = plan[case_id]
            # A marked compute case (heap or indirect) is owed only by the
            # rails its marker names: a rail that is not named must not report
            # it at all, which this refuses before the per-observation checks
            # can read a missing segment as a malformed capture.
            if case_rails is not None and report["backend"] not in case_rails:
                raise CaptureError(
                    f"{where}: {report['backend']} is not a rail this marked case runs on")
            _compare_observation(result, expected_writes, expected_allocations, where)
            _require("present" not in result,
                     f"{where}: the suite declares no present observation for this case")
            if heap is None:
                _require("heap" not in result,
                         f"{where}: the suite declares no heap section for this case")
            else:
                _require("heap" in result,
                         f"{where}: {report['backend']} has to report the heap placement "
                         "the suite declares")
                _heap_observation(result["heap"], heap, where)
            if icb is None:
                _require("icb" not in result,
                         f"{where}: the suite declares no indirect command for this case")
            else:
                _require("icb" in result,
                         f"{where}: {report['backend']} has to report the replayed indirect "
                         "command the suite declares")
                _icb_observation(result["icb"], icb, where)

        if case_id in render_plan:
            if counts[0] is not None:
                _integer(counts[0], f"{where}.copy_in")
                _integer(counts[1], f"{where}.copy_out")
            continue
        if provider_backend and suite["suite"] in ("compute-buffer-v11", "compute-buffer-v12"):
            _require(counts[0] is not None,
                     f"{where}: the v11/v12 count contract requires copy_in and copy_out")
        if counts[0] is not None:
            _integer(counts[0], f"{where}.copy_in")
            _integer(counts[1], f"{where}.copy_out")
        groups = result.get("group_counts")
        if groups is not None:
            _list(groups, f"{where}.group_counts")
            _require(counts[0] is not None,
                     f"{where}: group_counts require the summed copy_in and copy_out")
            _require(group_expectations is not None,
                     f"{where}: group_counts are only recorded for cases split "
                     "across command buffers")
        elif provider_backend and counts[0] is not None and group_expectations is not None:
            # The v9 count contract is per submission, so a provider capture of
            # a split case that only reports the accumulated totals stays
            # unverifiable and is refused (`research/docs/15` §5b).
            raise CaptureError(
                f"{where}: a case split across command buffers must report "
                "per-command-buffer group_counts when it reports counters")
        if groups is not None and provider_backend:
            # One submission per command buffer: the expectation is that
            # group's own touched and written allocations, and the flat
            # counters are their sums (research/docs/15 §5b).
            _require(len(groups) == len(group_expectations),
                     f"{where}: expected {len(group_expectations)} command buffer "
                     f"groups, got {len(groups)}")
            total_in = total_out = 0
            for position, (group, (expected_in, expected_out)) in enumerate(
                    zip(groups, group_expectations)):
                _object(group, ("copy_in", "copy_out"),
                        f"{where} command buffer {position}")
                group_in = _integer(group["copy_in"],
                                    f"{where} command buffer {position}.copy_in")
                group_out = _integer(group["copy_out"],
                                     f"{where} command buffer {position}.copy_out")
                _require(group_in == expected_in,
                         f"{where}: command buffer {position} copy_in {group_in} does "
                         f"not match {expected_in} touched allocations")
                _require(group_out == expected_out,
                         f"{where}: command buffer {position} copy_out {group_out} does "
                         f"not match {expected_out} written allocations")
                total_in += group_in
                total_out += group_out
            _require(counts[0] == total_in,
                     f"{where}: copy_in {counts[0]} does not match the {total_in} "
                     f"summed over {len(groups)} command buffers")
            _require(counts[1] == total_out,
                     f"{where}: copy_out {counts[1]} does not match the {total_out} "
                     f"summed over {len(groups)} command buffers")
        # A case with a single submission submits its whole sequence once, so
        # the derived expectation is the case-level one (research/docs/15 §5).
        if provider_backend and group_expectations is None and counts[0] is not None:
            expected_in = len(expected_allocations) + texture_count
            expected_out = len({identity[0] for identity, _ in expected_writes})
            _require(counts[0] == expected_in,
                     f"{where}: copy_in {counts[0]} does not match {expected_in} "
                     "touched allocations")
            _require(counts[1] == expected_out,
                     f"{where}: copy_out {counts[1]} does not match {expected_out} "
                     "written allocations")
    # A compute case without a heap section is required from every rail. A
    # compute case with one, and a render case, is required only from the rails
    # its own `capture_rails` marker names (`research/docs/25` §5.2): a rail
    # that declares no heap support cannot report a placement observation, and
    # the object-API rails carry no render command encoder, so those captures
    # would have nothing to report. A rail that is not named must not report
    # the case either, which is the same exact-set rule the per-result check
    # applies.
    required = {case_id for case_id, expectation in plan.items()
                if expectation[6] is None or report["backend"] in expectation[6]}
    required |= {case_id for case_id, expectation in render_plan.items()
                 if report["backend"] in expectation.rails}
    missing = required - seen
    _require(not missing, f"capture: missing cases {sorted(missing)}")
    for case_id in sorted(set(render_plan) - required):
        _require(case_id not in seen,
                 f"case {case_id}: {report['backend']} is not a rail this render case runs on")
    for case_id, expectation in sorted(plan.items()):
        if expectation[6] is not None and case_id not in required:
            _require(case_id not in seen,
                     f"case {case_id}: {report['backend']} is not a rail this heap case runs on")


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
    parser.add_argument("--vulkan-objects", type=Path,
                        help="optional Vulkan capture through the provider object API")
    parser.add_argument("--metal-objects", type=Path,
                        help="optional native Metal provider capture through the object API")
    args = parser.parse_args(argv)
    if args.check is not None:
        if any(path is not None for path in (args.native, args.vulkan, args.metal_provider,
                                            args.vulkan_objects, args.metal_objects)):
            parser.error("--check cannot be combined with --native, --vulkan, --metal-provider, "
                         "--vulkan-objects, or --metal-objects")
    elif args.native is None or args.vulkan is None:
        parser.error("provide --check, or both --native and --vulkan")
    try:
        raw, suite = _read_json(args.suite)
        digest = hashlib.sha256(raw).hexdigest()
        shape = (f"{len(suite['cases'])} cases"
                 + (f"; {len(suite['render_cases'])} render cases"
                    if suite.get("render_cases") else ""))
        if args.check is not None:
            _, report = _read_json(args.check)
            validate_capture(suite, digest, report)
            print(f"PASS capture: {report['backend']}; {suite['suite']}; {shape}; "
                  f"host-visible bytes agreement with suite; allocations={report['allocation_observation']}")
        else:
            _, native = _read_json(args.native)
            _, vulkan = _read_json(args.vulkan)
            validate_capture(suite, digest, native, required_backend="native-metal")
            validate_capture(suite, digest, vulkan, required_backend="vulkan")
            backends = ["native-metal", "vulkan"]
            for path, backend in ((args.metal_provider, "native-metal-provider"),
                                  (args.vulkan_objects, "vulkan-objects"),
                                  (args.metal_objects, "native-metal-provider-objects")):
                if path is not None:
                    _, report = _read_json(path)
                    validate_capture(suite, digest, report, required_backend=backend)
                    backends.append(backend)
            if args.vulkan_objects is not None or args.metal_objects is not None:
                print(f"PASS parity: {' / '.join(backends)}; "
                      f"{suite['suite']}; {shape}; host-visible bytes agreement; "
                      "Swift native GPU buffer readback / trace and object API provider host writeback landing")
            elif args.metal_provider is not None:
                print(f"PASS parity: native-metal / vulkan / native-metal-provider; "
                      f"{suite['suite']}; {shape}; host-visible bytes agreement; "
                      "Swift native GPU buffer readback / Vulkan and Rust Metal provider host writeback landing")
            else:
                print(f"PASS parity: native-metal / vulkan; {suite['suite']}; {shape}; "
                      "host-visible bytes agreement; native GPU buffer readback / Vulkan host writeback landing")
        return 0
    except (CaptureError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
