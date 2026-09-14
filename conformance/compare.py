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
# and nothing else, which is what keeps an attachment from passing as a buffer
# writeback; and `present` is the case's optional present section, the
# acquire/present counts every rail its marker names has to report
# (`research/docs/24` §5.3), or `None` when the suite declares none.
RenderExpectation = namedtuple(
    "RenderExpectation", "writes allocations touched written rails attachment present icb",
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


def _same_bytes(actual, expected, where, offset=0):
    _require(len(actual) == len(expected),
             f"{where}: length mismatch: expected {len(expected)} bytes, got {len(actual)}")
    for index, (left, right) in enumerate(zip(actual, expected)):
        if left != right:
            raise CaptureError(
                f"{where}: first differing byte at offset {offset + index}: "
                f"expected 0x{right:02x}, got 0x{left:02x}"
            )


def _compare_observation(result, expected_writes, expected_allocations, where):
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
    for (identity, actual), (_, expected) in zip(actual_writes, expected_writes):
        allocation, view, offset = identity
        _same_bytes(actual, expected, f"{where} writeback allocation {allocation}/view {view}",
                    offset)

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

    Two geometries exist and no third: the milestone's `vertex_id` triangle
    (no `vertex_layout` at all), and the reviewed indexed quad — one
    `float32x2` position stream at stride eight bound at index 0, plus six
    `uint16` or `uint32` indices whose values all name one of the four
    vertices the stream carries. The rules mirror `provider-capture`'s
    `render_geometry` and the Swift oracle's own validation, so a suite one
    rail would refuse cannot pass here either.
    """
    quad_vertices, quad_indices, quad_stride = 4, 6, 8
    layout = case.get("vertex_layout")
    vertex_buffers = case.get("vertex_buffers", [])
    indices = case.get("indices")
    if layout is None:
        _require(not vertex_buffers and indices is None,
                 f"{where}: vertex buffers without a vertex layout describe no stream")
        return None
    _object(layout, ("buffers",), f"{where}.vertex_layout")
    streams = _list(layout["buffers"], f"{where}.vertex_layout.buffers")
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
        required = ("id", "declaring_case", "vertex_entry", "fragment_entry", "metal",
                    "vertices", "viewport", "attachment", "expected_hex", "capture_rails")
        missing = [field for field in required if field not in case]
        _require(not missing, f"{where}: missing fields {', '.join(missing)}")
        unexpected = sorted(set(case) - set(required)
                            - {"present", "icb", "vertex_layout", "vertex_buffers", "indices"})
        _require(not unexpected, f"{where}: unexpected fields {', '.join(unexpected)}")
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

        attachment = case["attachment"]
        _require(isinstance(attachment, dict), f"{where}.attachment: expected an object")
        _require(set(attachment).issubset({"allocation", "view", "format", "width", "height",
                                          "load", "store", "clear_hex", "initial_hex"}),
                 f"{where}.attachment: unexpected fields")
        allocation = _integer(attachment.get("allocation"), f"{where}.attachment.allocation")
        view = _integer(attachment.get("view"), f"{where}.attachment.view")
        _require(allocation > 0 and view > 0, f"{where}: zero attachment identity")
        _require(attachment.get("format") == "rgba8_unorm",
                 f"{where}: unsupported attachment format")
        width = _integer(attachment.get("width"), f"{where}.attachment.width", 1)
        height = _integer(attachment.get("height"), f"{where}.attachment.height", 1)
        _require((width, height) == (2, 2),
                 f"{where}: the first render increment renders into a 2x2 attachment")
        _require(attachment.get("store") == "store",
                 f"{where}: a discarded attachment cannot be compared")
        vertex_input = _vertex_input_declaration(case, where)
        if vertex_input is None:
            _require(case["vertices"] == 3, f"{where}: expected the full-screen triangle")
        else:
            _require(case["vertices"] == vertex_input["indices"],
                     f"{where}: the reviewed indexed quad draws {vertex_input['indices']} indices")
            _require("present" not in case and "icb" not in case,
                     f"{where}: a vertex-input case carries neither a present action nor an ICB")
        viewport = _list(case["viewport"], f"{where}.viewport")
        _require(viewport == [0, 0, width, height],
                 f"{where}: the viewport must cover the attachment")
        expected = _hex(case["expected_hex"], f"{where}.expected_hex")
        _require(len(expected) == width * height * 4,
                 f"{where}: expected texel bytes do not match the attachment")
        texel = expected[:4]
        # Full coverage is the milestone's whole point (`research/docs/23` §1.3):
        # an expectation that admits different texels could be satisfied by a
        # partially covered attachment.
        _require(all(expected[offset:offset + 4] == texel
                     for offset in range(0, len(expected), 4)),
                 f"{where}: every texel of the expectation has to be the fragment output")
        load = attachment.get("load")
        if load == "clear":
            clear = _hex(attachment.get("clear_hex"), f"{where}.attachment.clear_hex")
            _require(len(clear) == 4, f"{where}: a clear colour is four bytes")
            _require("initial_hex" not in attachment,
                     f"{where}: a cleared attachment carries no initial bytes")
            _require(clear != texel, f"{where}: the clear colour equals the expected texel")
        elif load == "load":
            previous = _hex(attachment.get("initial_hex"), f"{where}.attachment.initial_hex")
            _require(len(previous) == len(expected),
                     f"{where}: initial texels do not match the attachment")
            _require(previous != expected,
                     f"{where}: the initial texels equal the expectation")
            _require("clear_hex" not in attachment,
                     f"{where}: a loaded attachment carries no clear colour")
        else:
            raise CaptureError(f"{where}: unknown attachment load op {load!r}")

        # The present section is optional: a case without it is the v13 case and
        # must not grow a present observation in a capture (the exact-set rule
        # `validate_capture` applies to every result).
        present = None
        if "present" in case:
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

        # The attachment resolves against the declaring case's own table: one of
        # its declared views has to be the attachment, it has to be read-only
        # (a compute pass that *wrote* the view the render pass stores would make
        # the order inexpressible), and its byte range has to agree with the
        # extent the attachment restates.
        declaring_buffers = by_id[declaring]["buffers"]
        declared = [buffer for buffer in declaring_buffers
                    if buffer["allocation"] == allocation and buffer["view"] == view]
        _require(len(declared) == 1,
                 f"{where}: the declaring case has to declare exactly the attachment view")
        declared = declared[0]
        _require(declared["access"] == "read",
                 f"{where}: the declaring pass must only read the attachment view")
        _require(declared["length"] == len(expected),
                 f"{where}: attachment extent disagrees with the declaring view")
        offset = declared["offset"]
        size = declared["allocation_size"]
        _require(offset + len(expected) <= size,
                 f"{where}: the declaring view is outside its allocation")

        # The allocation image is the declaring case's own image with the
        # attachment's landing overlaid: the render result observes the whole
        # allocation, so a guard byte or an untouched neighbour that the render
        # pass did not store into stays part of the comparison.
        image = bytearray(plan[declaring][1][allocation])
        _require(len(image) == size, f"{where}: inconsistent allocation size")
        image[offset:offset + len(expected)] = expected
        touched = set(plan[declaring][1])
        written = {identity[0] for identity, _ in declaring_writes} | {allocation}
        render_plan[case_id] = RenderExpectation(
            writes=[((allocation, view, offset), expected)],
            allocations={allocation: bytes(image)},
            touched=touched,
            written=written,
            rails=frozenset(rails),
            attachment=(allocation, view, offset, len(expected)),
            present=present,
            icb=icb)
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
            # attachment's own allocation and its writeback. The identity check
            # below is the attachment-versus-buffer rule — an attachment cannot
            # be satisfied by a buffer writeback, and a buffer writeback cannot
            # be reported where the attachment belongs.
            expectation = render_plan[case_id]
            _compare_observation(result, expectation.writes, expectation.allocations, where)
            attachment_allocation, attachment_view, attachment_offset, attachment_length = \
                expectation.attachment
            _require({identity for identity, _ in expectation.writes}
                     == {(attachment_allocation, attachment_view, attachment_offset)},
                     f"{where}: the attachment plan has to be one writeback")
            _require(len(expectation.allocations) == 1
                     and next(iter(expectation.allocations)) == attachment_allocation,
                     f"{where}: the attachment plan has to be its own allocation")
            _require(attachment_length == len(expectation.writes[0][1]),
                     f"{where}: the attachment plan has to cover its own texels")
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
