#!/usr/bin/env python3
"""Validate captures, or compare the Swift Metal oracle and compute providers.

Passing checks establishes agreement of the supplied captures with this suite.
It does not attest how a capture was produced or substitute for a native run.
"""

import argparse
from collections import namedtuple
import functools
import hashlib
import json
from pathlib import Path
import re
import struct
import sys


MAX_SERIAL_RESOURCES = 64
U32_MAX = (1 << 32) - 1
U64_MAX = (1 << 64) - 1
# The reviewed attachment ceiling per axis (R1b, `research/docs/23` §70; R5a,
# §73): the window the render rails' fixtures measure. The rails declare the
# smaller of this ceiling and the device's own framebuffer limit, so a fixture
# at the ceiling is inside every conformant device's window (Vulkan's minimum
# `maxFramebufferWidth` is 4096; Metal's 2D texture ceiling is 16384) and the
# boundary case below runs unconditionally. A wider extent is a deliberate
# change that owes a boundary fixture at the new value, in all three review
# surfaces.
REVIEWED_ATTACHMENT_CEILING = 2048
# The largest byte extent one declared view may carry: the reviewed window's own
# attachment (`REVIEWED_ATTACHMENT_CEILING`² texels of four bytes). R5a (§73)
# derives the cap from the reviewed window so the wide case's declaring view is
# admitted while the guard keeps an unchecked suite from asking for more than
# the review measured.
MAX_ALLOCATION_BYTES = REVIEWED_ATTACHMENT_CEILING * REVIEWED_ATTACHMENT_CEILING * 4
# The largest allocation image a capture spells out as hex (R5a,
# `research/docs/23` §73). A wider image is reported as its digest: the wide
# attachment's declaring view is 16 MiB, so its image would be 32 MiB of hex in
# every capture of the suite, and this comparator recomputes the digest from the
# suite's own declarations. Narrower images keep their bytes, so the review of
# every pre-R5a case is unchanged.
MAX_VERBATIM_ALLOCATION_BYTES = 1_048_576
ALLOCATION_OBSERVATIONS = {
    "native-metal": "gpu-buffer-readback",
    "vulkan": "host-writeback-landing",
    "native-metal-provider": "host-writeback-landing",
    "vulkan-objects": "host-writeback-landing",
    "native-metal-provider-objects": "host-writeback-landing",
}
# The source arms one buffer view's bytes may come from (`research/docs/23`
# §90, R9i). The fixture format could spell only owned bytes before this
# increment, so the lease arms the provider rails execute had no case: the
# declaration now names the arm beside the bytes, and every provider capture
# owes the arm each view actually ran with.
BUFFER_STORAGE_MODES = ("owned_bytes", "staged_lease", "borrowed_no_copy")

# The stage-buffer face (`research/docs/23` §3.3, v83-v86): a render case may
# declare slots whose bytes its stages read — and, for a writable one, land —
# directly. The vocabulary is the contract's own (`StageBufferBinding` /
# `StageBufferView`), and the bounds restate `metal_api_core`'s constants: the
# count ceiling is the widened one (`research/docs/23` §108), the descriptor
# floor Vulkan states for one set's storage buffers.
STAGE_BUFFER_STAGES = ("vertex", "fragment")
STAGE_BUFFER_ACCESSES = ("read", "write", "read_write")
MAX_RENDER_STAGE_BUFFERS = 8
MAX_RENDER_STAGE_BUFFER_INDEX = 16
# The invocation axes an affine footprint may stride over
# (`metal_api_core::provider::RENDER_AFFINE_AXES`): `0` is the vertex index,
# `1` the instance index.
RENDER_AFFINE_AXES = 2
# The Vulkan trace rail: the one rail that translates a stage-buffer case's AIR
# stages (`research/docs/23` §3.3, v83-v86).
VULKAN_TRACE_RAIL = "vulkan"
# The Vulkan object rails (`research/docs/23` §3.3, v87): one marker covers the
# synchronous and the deferred run, because both captures report this backend,
# and the object API binds stage buffers from v87's own entry point on.
VULKAN_OBJECTS_RAIL = "vulkan-objects"
# The rails that bind a *reviewed* stage-buffer case's slots (`research/docs/23`
# §83, R9g): the Vulkan trace rail and the two native faces, which compile the
# reviewed MSL module the case pins — the Swift oracle through its own render
# encoder and the Rust native provider through the same module the rail
# embeds. A translated case's AIR pair is only executable on the rail whose
# translator mints its descriptor sets, so that arm names the Vulkan rails.
STAGE_BUFFER_RAILS = (VULKAN_TRACE_RAIL, "native-metal", "native-metal-provider")
# The rails a *translated* stage-buffer case may name: the Vulkan trace rail
# translates its AIR, and the object rails bind its slots through the object
# API's own stage-buffer entry point (`research/docs/23` §3.3, v87).
TRANSLATED_STAGE_BUFFER_RAILS = (VULKAN_TRACE_RAIL, VULKAN_OBJECTS_RAIL)

# One stage-buffer slot a render case declares (`research/docs/23` §3.3,
# v83-v86): the pipeline's declaration (`stage`, `index`, `access`, `footprint`)
# beside the pass's view (`view` identity, range and bytes). `expected` is the
# landing a writable slot's writeback has to carry, or `None` for a read-only
# one; `mode` is the source arm the view's bytes come from (`research/docs/23`
# §90, R9i), and `borrowed` says whether that arm copies the bytes in.
StageBuffer = namedtuple("StageBuffer", "stage index access footprint view expected mode borrowed")

# The reviewed per-texel rule of the wide attachment (R5a, `research/docs/23`
# §73): texel `(x, y)` carries `x` and `y` as two little-endian `u16`s, i.e.
# `[x & 0xff, (x >> 8) & 0xff, y & 0xff, (y >> 8) & 0xff]`. The rule is how a
# 2048x2048 fixture states its expectation — four million texels of hex is what
# it replaces — and it is injective over any window up to `RULE_ADDRESS_CEILING`
# texels per axis, so a flipped, transposed or row-shifted readback differs from
# it. The three review surfaces implement this one function: this comparator,
# `examples/metal-smoke/src/bin/provider-capture.rs` and
# `conformance/NativeOracle.swift`.
XY_U16LE_V1 = "xy_u16le_v1"
# The smallest extent a rule expectation is admissible at: the rule form exists
# for the megapixel-class attachments whose hex spelling the fixture cannot
# carry, so a small case keeps stating its texels.
RULE_MIN_DIMENSION = 1024
# The largest extent the reviewed rule addresses: both coordinates travel as
# little-endian `u16`s, so a wider axis could not be spelled injectively.
RULE_ADDRESS_CEILING = 65_536
# How many readback windows one rule-expected case may declare, and how wide
# each may be: the capture reports these bytes as hex, so the reviewed shape
# keeps them small (two 64x64 windows are 32 KiB of hex).
MAX_READBACK_WINDOWS = 8
MAX_READBACK_WINDOW_DIMENSION = 64

# The sampled textures the render sampler admits (`research/docs/23` §3.3,
# §107, §113), in the order the rails' capability lists name them: the two
# four-byte 8-bit UNORM byte orders and the two narrow lanes. A sampled
# texture may name any of the four; the colour attachment stays one of the two
# four-component layouts, and the case's hex expectation is the sampled
# *colours* spelled in the attachment's own order — which is what the slot
# table below computes. A narrow source's missing channels are the API
# sampling rule's own fill (zero, and one for alpha), so its expectation is a
# derivation rather than a copy. The rule form stays the same four-component
# pair's, because its closed-form expectation is stated over one layout.
SAMPLED_TEXTURE_FORMATS = ("rgba8_unorm", "bgra8_unorm", "r8_unorm", "rg8_unorm")
# The colour attachment's admitted layouts: the sampled shape renders into the
# eight-bit four-component surface both rails read back, which is what the
# loaded clear colour is stated in.
COLOUR_ATTACHMENT_FORMATS = ("rgba8_unorm", "bgra8_unorm")
# The byte slot each channel occupies in one 8-bit four-component layout: the
# two layouts differ in the red and blue halves alone.
SAMPLED_CHANNEL_SLOTS = {
    "rgba8_unorm": {"red": 0, "green": 1, "blue": 2, "alpha": 3},
    "bgra8_unorm": {"red": 2, "green": 1, "blue": 0, "alpha": 3},
}
# How many bytes one texel of each sampled format occupies, and which of its
# channels the format itself carries (`research/docs/23` §113). The channels a
# narrow format does not carry keep the sampling rule's fill.
SAMPLED_TEXEL_BYTES = {
    "rgba8_unorm": 4,
    "bgra8_unorm": 4,
    "r8_unorm": 1,
    "rg8_unorm": 2,
}
SAMPLED_FILL = {"r8_unorm": {"green": 0x00, "blue": 0x00, "alpha": 0xff},
                "rg8_unorm": {"blue": 0x00, "alpha": 0xff}}


def _sampled_expectation(texture_format, attachment_format, texels, texel_count=None):
    """The sampled colours of `texels`, spelled in the attachment's own order.

    `texels` are the texture's memory bytes as the fixture declares them; a
    texel-centre sample is an identity copy, so the attachment holds the same
    colours. When the two sides name the same layout the bytes are the same
    byte for byte; across the two layouts every texel is the red/blue swap of
    the other (`research/docs/23` §107). A narrow source states one or two
    bytes per texel and the sampling rule fills the channels it does not carry
    (`research/docs/23` §113), so the expectation is that fill spelled in the
    attachment's own order.

    `texel_count` is the extent's own texel count when the caller knows it: a
    narrow source's byte string is *shorter* than the frame it derives, so the
    count cannot be read off `texels` for those lanes.
    """
    target = SAMPLED_CHANNEL_SLOTS[attachment_format]
    # Where each of the four sampled channels comes from in the source's own
    # texel: a byte slot for a layout that carries it, the sampling rule's fill
    # for a narrow format that does not.
    stride = SAMPLED_TEXEL_BYTES[texture_format]
    if texture_format in SAMPLED_CHANNEL_SLOTS:
        slots = SAMPLED_CHANNEL_SLOTS[texture_format]
        source = {channel: slots[channel] for channel in slots}
    else:
        # A narrow format carries one or two channels and the sampling rule
        # fills the rest: a channel with no byte slot has that rule's value.
        fill = SAMPLED_FILL[texture_format]
        source = {"red": 0,
                  "green": 1 if "green" not in fill else None,
                  "blue": None,
                  "alpha": None}
    # A caller that states the extent's texel count means the *whole* extent;
    # the derivation never reads past the bytes it was handed, so a source
    # shorter than that count derives what it can and the caller's own length
    # rule is what refuses the short spelling.
    count = len(texels) // stride
    if texel_count is not None:
        count = min(count, texel_count)
    wanted = bytearray()
    for index in range(count):
        offset = index * stride
        order = [None, None, None, None]
        for channel in ("red", "green", "blue", "alpha"):
            slot = source[channel]
            if texture_format in SAMPLED_CHANNEL_SLOTS:
                order[target[channel]] = texels[offset + slot]
            elif slot is None:
                order[target[channel]] = fill[channel]
            else:
                order[target[channel]] = texels[offset + slot]
        wanted.extend(order)
    return bytes(wanted)


def _rule_texel(rule, x, y):
    """The four bytes the reviewed rule stores at texel `(x, y)`."""
    if rule != XY_U16LE_V1:
        raise CaptureError(f"unknown texel rule {rule!r}")
    if not (0 <= x < RULE_ADDRESS_CEILING and 0 <= y < RULE_ADDRESS_CEILING):
        raise CaptureError("the reviewed rule addresses at most "
                           f"{RULE_ADDRESS_CEILING} texels per axis")
    return struct.pack("<HH", x, y)


@functools.lru_cache(maxsize=4)
def _rule_plane(rule, width, height):
    """The whole plane the reviewed rule describes, row-major."""
    if rule != XY_U16LE_V1:
        raise CaptureError(f"unknown texel rule {rule!r}")
    if not (1 <= width <= RULE_ADDRESS_CEILING and 1 <= height <= RULE_ADDRESS_CEILING):
        raise CaptureError("the reviewed rule addresses one to "
                           f"{RULE_ADDRESS_CEILING} texels per axis")
    # The rule's channels are the two halves of one texel address each, so a row
    # is four strided copies: the x halves are fixed, the y halves are the row's
    # own value in every column.
    row = bytearray(width * 4)
    row[0::4] = bytes(x & 0xFF for x in range(width))
    row[1::4] = bytes((x >> 8) & 0xFF for x in range(width))
    plane = bytearray()
    for y in range(height):
        row[2::4] = bytes([y & 0xFF]) * width
        row[3::4] = bytes([(y >> 8) & 0xFF]) * width
        plane += row
    return bytes(plane)


@functools.lru_cache(maxsize=4)
def _rule_digest(rule, width, height):
    """The SHA-256 of the plane the rule describes, as the capture reports it."""
    return hashlib.sha256(_rule_plane(rule, width, height)).hexdigest()


def _rule_reaches_colour(rule, colour, width, height):
    """Whether a four-byte colour is one of the rule's texels over this window.

    The closed form of the rule's own injectivity: the first two bytes are a
    little-endian `x` and the last two a little-endian `y`, so the rule reaches
    the colour exactly when both addresses are inside the extent. A clear colour
    the rule reaches would let a rail that ignored the draw land bytes the
    expectation claims, which is why it is refused without scanning the plane.
    """
    if rule != XY_U16LE_V1:
        raise CaptureError(f"unknown texel rule {rule!r}")
    if len(colour) != 4:
        raise CaptureError("a clear colour is four bytes")
    x = int.from_bytes(colour[:2], "little")
    y = int.from_bytes(colour[2:], "little")
    return x < width and y < height


def _rule_window_bytes(rule, window):
    """One declared readback window's expected bytes, tightly packed rows."""
    x, y, width, height = window
    out = bytearray()
    for row in range(height):
        for column in range(width):
            out += _rule_texel(rule, x + column, y + row)
    return bytes(out)


def _readback_windows(case, rule, width, height, where):
    """Parse one rule-expected case's readback windows.

    Each window is a rectangle inside the attachment plane; the tuple carries
    its own expected bytes, so the capture's reported bytes are compared texel
    for texel instead of being trusted as "the digest was right".
    """
    windows = _list(case.get("readback_windows"), f"{where}.readback_windows")
    _require(windows, f"{where}: a rule-expected attachment needs its readback windows")
    _require(len(windows) <= MAX_READBACK_WINDOWS,
             f"{where}.readback_windows: one to {MAX_READBACK_WINDOWS} windows")
    parsed = []
    for index, window in enumerate(windows):
        window_where = f"{where}.readback_windows[{index}]"
        _require(isinstance(window, dict), f"{window_where}: expected an object")
        _require(set(window) == {"x", "y", "width", "height"},
                 f"{window_where}: expected x, y, width and height")
        x = _integer(window["x"], f"{window_where}.x")
        y = _integer(window["y"], f"{window_where}.y")
        window_width = _integer(window["width"], f"{window_where}.width", 1,
                                MAX_READBACK_WINDOW_DIMENSION)
        window_height = _integer(window["height"], f"{window_where}.height", 1,
                                 MAX_READBACK_WINDOW_DIMENSION)
        _require(x + window_width <= width and y + window_height <= height,
                 f"{window_where}: the readback window leaves the attachment plane")
        parsed.append((x, y, window_width, window_height,
                       _rule_window_bytes(rule, (x, y, window_width, window_height))))
    return tuple(parsed)

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
# (`research/docs/24` §5.3), or `None` when the suite declares none; and
# `filter` is the case's optional device gate (`research/docs/23` §3.3, v57d):
# the `"min"`/`"max"` depth resolve filter a capture has to carry in its
# `depth_resolve_modes` mask for the case to appear, or `None` for a case every
# rail its marker names reports unconditionally.
# `wildcards` maps each observed attachment's `(allocation, view, offset)`
# identity to a dict of its own byte offsets: `None` is a byte the case leaves
# unclaimed (`research/docs/23` §3.3, v33), and a tuple is the closed candidate
# set a constrained wildcard texel may carry (`research/docs/23` §3.3, v67).
RenderExpectation = namedtuple(
    "RenderExpectation",
    "writes allocations touched written rails attachment present icb wildcards filter "
    "stencil_filter sample_count_gate texture_uploads rule stage_buffer_modes landing",
    defaults=(None, None, None, None, None, None))

# One rule-expected attachment (R5a, `research/docs/23` §73): the rule's name,
# the extent it covers, the digest of the whole plane the rule describes, and
# the declared readback windows as `(x, y, width, height, bytes)`. A capture
# cannot report four million texels of hex, so this is the observation the
# comparator checks: the reported `bytes_sha256` has to be `digest`, and every
# window's reported bytes have to equal its own `bytes` entry, texel for texel.
RuleExpectation = namedtuple("RuleExpectation", "rule width height digest windows")

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

# The capability-mask bit each depth resolve filter occupies
# (`research/docs/23` §3.3, v57d): bit `i` is the filter whose wire code is
# `i`, the same numbering `metal-api-core` uses for
# `ProviderCapabilities::depth_resolve_modes`.
DEPTH_RESOLVE_FILTER_BITS = {"sample0": 1, "min": 2, "max": 4}

# The capability-mask bit each stencil resolve filter occupies
# (`research/docs/23` §3.3, v60): bit `i` is the filter whose wire code is
# `i`, the same numbering `metal-api-core` uses for
# `ProviderCapabilities::stencil_resolve_modes`.
STENCIL_RESOLVE_FILTER_BITS = {"sample0": 1, "depth_resolved_sample": 2}

# The capability-mask bit each device-gated sample count occupies
# (`research/docs/23` §3.3, v61): bit `i` is the `SampleCount` whose wire code
# is `i`, so 2x carries bit 1 and 8x carries bit 3. The capture's
# `render_sample_counts` mask carries exactly the reviewed counts the device
# admits, and a gated case is owed only when its count's bit is present.
SAMPLE_COUNT_BITS = {2: 1 << 1, 8: 1 << 3}


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


def _buffer_initial_bytes(buffer, where, allocation, view, length):
    """The bytes one declared buffer view is pre-seeded with.

    R5a (`research/docs/23` §73) adds the repeated-pattern form for the wide
    attachment's declaring view: the view is 16 MiB, so spelling its fill out
    would put 32 MiB of hex in the fixture. The fixture states one pattern and
    the view length it repeats to, and exactly one of the two forms is present,
    so a declaration cannot carry a pattern beside bytes that disagree with it.
    """
    has_hex = "initial_hex" in buffer
    has_repeat = "initial_repeat_hex" in buffer
    _require(has_hex != has_repeat,
             f"{where}: view {view} needs exactly one of initial_hex and "
             "initial_repeat_hex")
    if has_hex:
        initial = _hex(buffer["initial_hex"], f"{where} allocation {allocation}.initial_hex")
        _require(len(initial) == length, f"{where}: initial length does not match view {view}")
        return initial
    pattern = _hex(buffer["initial_repeat_hex"],
                   f"{where} allocation {allocation}.initial_repeat_hex")
    _require(pattern, f"{where}: an initial repeat pattern is at least one byte")
    _require(length % len(pattern) == 0,
             f"{where}: the initial repeat pattern of {len(pattern)} bytes does not divide "
             f"view {view}'s {length} bytes")
    return pattern * (length // len(pattern))


def _same_bytes(actual, expected, where, offset=0, wildcards=frozenset(), allowed=None):
    """Compare two byte strings against the case's claim on each byte.

    `offset` is where the compared range starts inside its allocation.
    `wildcards` holds absolute allocation offsets whose bytes the case does not
    claim (`research/docs/23` §3.3, v33): a byte at a wild offset is neither
    compared nor used to build an error message, because the case said in
    advance that what landed there is undefined. `allowed` maps absolute
    offsets to the closed set of byte values a *constrained* wildcard texel may
    carry (`research/docs/23` §3.3, v67): the measured byte has to be one of
    them, and a byte outside the set is refused with the same first-differing
    shape the exact comparison uses.
    """
    _require(len(actual) == len(expected),
             f"{where}: length mismatch: expected {len(expected)} bytes, got {len(actual)}")
    allowed = allowed or {}
    for index, (left, right) in enumerate(zip(actual, expected)):
        position = offset + index
        if position in wildcards:
            continue
        candidates = allowed.get(position)
        if candidates is not None:
            if left in candidates:
                continue
            raise CaptureError(
                f"{where}: first differing byte at offset {position}: "
                f"got 0x{left:02x}, which is none of "
                + ", ".join(f"0x{candidate:02x}" for candidate in candidates)
            )
        if left != right:
            raise CaptureError(
                f"{where}: first differing byte at offset {position}: "
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
        claims = wildcards.get(identity, {})
        skips = frozenset(position for position, candidates in claims.items()
                          if candidates is None)
        allows = {position: candidates for position, candidates in claims.items()
                  if candidates is not None}
        _same_bytes(actual, expected, f"{where} writeback allocation {allocation}/view {view}",
                    offset, skips, allows)

    seen_allocations = set()
    for value in _list(result["allocations"], f"{where}.allocations"):
        _require(isinstance(value, dict), f"{where} allocation: expected an object")
        allocation = _integer(value.get("allocation"), f"{where}.allocation")
        _require(allocation not in seen_allocations, f"{where}: duplicate allocation {allocation}")
        _require(allocation in expected_allocations, f"{where}: unknown allocation {allocation}")
        seen_allocations.add(allocation)
        _allocation_image(value, expected_allocations[allocation], allocation, where, wildcards)
    missing = set(expected_allocations) - seen_allocations
    _require(not missing, f"{where}: missing allocations {sorted(missing)}")


def _allocation_image(value, expected, allocation, where, wildcards):
    """Compare one reported allocation image against the plan's own image.

    A capture spells an image out as hex while it is at most
    `MAX_VERBATIM_ALLOCATION_BYTES` wide, and reports the digest of anything
    wider (R5a, `research/docs/23` §73): the wide attachment's declaring view is
    16 MiB, so 32 MiB of hex would ride in every capture of the suite. The
    digest is recomputed here from the plan's own image, so the form that avoids
    the bytes is still a byte-for-byte check.
    """
    has_hex = "bytes_hex" in value
    has_digest = "bytes_sha256" in value
    _require(has_hex != has_digest,
             f"{where} allocation {allocation}: exactly one of bytes_hex and bytes_sha256")
    if has_digest:
        _object(value, ("allocation", "bytes_sha256", "bytes_length"),
                f"{where} allocation {allocation}")
        length = _integer(value["bytes_length"],
                          f"{where} allocation {allocation}.bytes_length", 1)
        _require(length == len(expected),
                 f"{where} allocation {allocation}: the reported length is not the image's")
        _require(length > MAX_VERBATIM_ALLOCATION_BYTES,
                 f"{where} allocation {allocation}: only an image wider than "
                 f"{MAX_VERBATIM_ALLOCATION_BYTES} bytes is reported by digest")
        digest = value["bytes_sha256"]
        _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                 f"{where} allocation {allocation}: expected the image's lowercase SHA-256")
        expected_digest = hashlib.sha256(expected).hexdigest()
        _require(digest == expected_digest,
                 f"{where} allocation {allocation}: the image's digest {digest} is not the "
                 f"plan's {expected_digest}: one byte of the {length}-byte image differs")
        return
    _object(value, ("allocation", "bytes_hex"), f"{where} allocation {allocation}")
    actual = _hex(value["bytes_hex"], f"{where} allocation {allocation}.bytes_hex")
    _require(len(actual) <= MAX_VERBATIM_ALLOCATION_BYTES,
             f"{where} allocation {allocation}: an image wider than "
             f"{MAX_VERBATIM_ALLOCATION_BYTES} bytes is reported by digest")
    skipped, allowed = set(), {}
    for identity, claims in wildcards.items():
        if identity[0] != allocation:
            continue
        for position, candidates in claims.items():
            if candidates is None:
                skipped.add(position)
            else:
                allowed[position] = candidates
    _same_bytes(actual, expected,
                f"{where} allocation {allocation}", 0, frozenset(skipped), allowed)


def _writeback(value, where):
    _object(value, ("allocation", "view", "offset", "bytes_hex"), where)
    identity = tuple(_integer(value[key], f"{where}.{key}") for key in ("allocation", "view", "offset"))
    data = _hex(value["bytes_hex"], f"{where}.bytes_hex")
    _require(data, f"{where}: empty writeback")
    return identity, data


def _rule_plane_field(value, rule, where):
    """Check one reported plane against the rule it claims to observe.

    Both forms carry the plane's own byte length and SHA-256; the digest is
    recomputed from the rule here, so a capture cannot pass by reporting a
    digest of the bytes it was *asked* for: the two agree only if the bytes the
    rail landed are the rule's own plane (`research/docs/23` §73).
    """
    length = _integer(value["bytes_length"], f"{where}.bytes_length", 1)
    _require(length == rule.width * rule.height * 4,
             f"{where}: the plane's length does not match the rule's extent")
    digest = value["bytes_sha256"]
    _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
             f"{where}: expected the plane's lowercase SHA-256")
    _require(digest == rule.digest,
             f"{where}: the plane's digest {digest} is not the {rule.rule} plane's "
             f"{rule.digest}: one texel of the {rule.width}x{rule.height} attachment differs")


def _rule_window(observed, window, rule, window_where):
    """Check one reported readback window against the rule, texel for texel."""
    _object(observed, ("x", "y", "width", "height", "bytes_hex"), window_where)
    x = _integer(observed["x"], f"{window_where}.x")
    y = _integer(observed["y"], f"{window_where}.y")
    width = _integer(observed["width"], f"{window_where}.width", 1)
    height = _integer(observed["height"], f"{window_where}.height", 1)
    _require((x, y, width, height) == window[:4],
             f"{window_where}: the reported rectangle is not the one the suite declares")
    expected = window[4]
    actual = _hex(observed["bytes_hex"], f"{window_where}.bytes_hex")
    _require(len(actual) == len(expected),
             f"{window_where}: the reported bytes do not match the window's extent")
    for texel in range(width * height):
        chunk = slice(texel * 4, texel * 4 + 4)
        if actual[chunk] != expected[chunk]:
            column = x + texel % width
            row = y + texel // width
            raise CaptureError(
                f"{window_where}: texel ({column}, {row}) reads {actual[chunk].hex()}, the "
                f"{rule.rule} rule stores {expected[chunk].hex()}")


def _compare_rule_observation(result, expectation, where):
    """Check one rule-expected attachment's digest-and-windows observation.

    The capture cannot carry four million texels, so the observation is the
    plane's SHA-256 plus the declared readback windows' bytes
    (`research/docs/23` §73). The digest is recomputed from the rule, and the
    windows are compared texel for texel, so a single wrong texel anywhere in
    the plane changes the digest and a single wrong texel inside a window is
    named by its coordinates.
    """
    rule = expectation.rule
    attachments = expectation.attachment
    if isinstance(attachments, tuple):
        attachments = [attachments]
    identities = []
    for value in _list(result["writebacks"], f"{where}.writebacks"):
        _object(value, ("allocation", "view", "offset", "bytes_sha256", "bytes_length",
                        "observed_windows"), f"{where} writeback")
        identity = tuple(_integer(value[key], f"{where} writeback.{key}")
                         for key in ("allocation", "view", "offset"))
        _require(identity not in identities, f"{where}: duplicate writeback {identity}")
        identities.append(identity)
        _rule_plane_field(value, rule, f"{where} writeback {identity}")
        observed_windows = _list(value["observed_windows"],
                                 f"{where} writeback.observed_windows")
        _require(len(observed_windows) == len(rule.windows),
                 f"{where} writeback: the capture has to report the "
                 f"{len(rule.windows)} readback windows the suite declares")
        for index, (window, observed) in enumerate(zip(rule.windows, observed_windows)):
            _rule_window(observed, window, rule, f"{where} writeback window[{index}]")
    _require(set(identities) == {attachment[:3] for attachment in attachments},
             f"{where}: the attachment writebacks have to be exactly the declared "
             "attachments")
    _require(identities == [attachment[:3] for attachment in attachments],
             f"{where}: writeback order differs from suite")

    seen = set()
    for value in _list(result["allocations"], f"{where}.allocations"):
        _object(value, ("allocation", "bytes_sha256", "bytes_length"),
                f"{where} allocation")
        allocation = _integer(value["allocation"], f"{where}.allocation")
        _require(allocation not in seen, f"{where}: duplicate allocation {allocation}")
        _require(allocation in expectation.allocations,
                 f"{where}: unknown allocation {allocation}")
        seen.add(allocation)
        _rule_plane_field(value, rule, f"{where} allocation {allocation}")
    missing = set(expectation.allocations) - seen
    _require(not missing, f"{where}: missing allocations {sorted(missing)}")


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


def _storage_modes_observation(value, declared, where):
    """Check one case's reported source arms against its suite's declaration.

    The report is the run's own statement of where each view's bytes came from
    (`research/docs/23` §90, R9i): the owned arm reads the trace's own bytes,
    the staged arm imports the owner's window into provider storage, and the
    borrowed arm maps it without copying. A rail that fell back to owned bytes
    — or that reported an arm the suite did not declare — is refused here, so
    the bytes a lease case lands and the arm it claims stay one statement.
    """
    entries = _list(value, f"{where}.storage_modes")
    _require(entries, f"{where}: storage_modes is empty")
    observed = {}
    for index, entry in enumerate(entries):
        entry_where = f"{where}.storage_modes[{index}]"
        _object(entry, ("view", "mode"), entry_where)
        view = _integer(entry["view"], f"{entry_where}.view")
        mode = entry["mode"]
        _require(mode in BUFFER_STORAGE_MODES,
                 f"{entry_where}: unknown buffer storage mode {mode!r}")
        _require(view in declared,
                 f"{entry_where}: view {view} is not a leased view of this case")
        _require(view not in observed, f"{entry_where}: duplicate view {view}")
        observed[view] = mode
    _require(observed == dict(declared),
             f"{where}: reported source arms {observed} are not the suite's "
             f"{dict(declared)}")


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
        # The source arm each view declares (`research/docs/23` §90, R9i). One
        # allocation carries one arm: the lease window is the view's own range
        # inside the owner's registration, so two views of one allocation
        # cannot come from two different owners without a second window the
        # format does not state.
        view_modes, allocation_modes = {}, {}
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
            mode = buffer.get("storage_mode", "owned_bytes")
            _require(mode in BUFFER_STORAGE_MODES,
                     f"{where}: unknown buffer storage mode {mode!r}")
            if allocation in allocation_modes:
                _require(allocation_modes[allocation] == mode,
                         f"{where}: allocation {allocation} declares two source arms "
                         f"({allocation_modes[allocation]} and {mode})")
            else:
                allocation_modes[allocation] = mode
            view_modes[view] = mode
            _require(view not in views and binding not in bindings, f"{where}: duplicate view or binding")
            _require(offset + length <= size, f"{where}: buffer view outside allocation {allocation}")
            initial = _buffer_initial_bytes(buffer, where, allocation, view, length)
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
        lease_views = {view: mode for view, mode in view_modes.items()
                       if mode != "owned_bytes"}
        borrowed_allocations = frozenset(
            allocation for allocation, mode in allocation_modes.items()
            if mode == "borrowed_no_copy")
        if lease_views:
            # A lease-armed view names the window the owner registers for it,
            # so the case states one view per allocation — the shape the
            # provider's own lease range rule (`reservation` covers the view)
            # and the harness's one-window-per-view import both assume.
            _require(len(allocations) == len(buffers),
                     f"{where}: a case that declares a lease arm declares one view "
                     "per allocation")
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
        _require(((heap is None and icb is None and not lease_views) == (rails is None)),
                 f"{where}: capture_rails and a heap, icb or lease section are declared "
                 "together")
        plan[case_id] = (writes, allocations, len(texture_allocations),
                         group_expectations, heap, icb, rails,
                         view_modes if lease_views else None,
                         borrowed_allocations if lease_views else None)
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
    (no `vertex_layout` at all), the reviewed quad over one `float32x2`
    position stream at stride eight bound at index 0 — either with six `uint16`
    or `uint32` indices whose values all name one of the four vertices the
    stream carries, or, since v39, drawn *without* an index buffer so the draw
    names its vertices `0..vertices` and the stream has to cover the whole
    `vertices * stride` span (`research/docs/23` §3.3, v39) — and the reviewed
    instanced pair (`research/docs/23` §3.3, v31): the same position stream at
    binding 0, a `float32x4` tint stream at binding 1 that advances once per
    *instance*, and exactly two instances.
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
    # The declared-superset arm (`research/docs/23` §3.3, E-TX11): a
    # *translated* case whose contract declares more attribute locations than
    # its module reads. It is its own closed shape — one stream, four
    # `float32x2` attributes at `8 * location` over a 32-byte record — and it is
    # classified before the reviewed arms, because a translated case pins AIR
    # modules the reviewed shapes have no MSL module for.
    if case.get("translated_stages") is not None:
        return _superset_declaration(case, streams, vertex_buffers, indices, where)
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
    # The non-indexed arm (`research/docs/23` §3.3, v39): the same one-stream
    # shape drawn without an index buffer. A non-indexed draw names its
    # vertices `0..vertices`, so every per-vertex stream has to cover the whole
    # `vertices * stride` span — the footprint rule both rails prove, and the
    # stricter of the two rules the class carries, since the indexed arm only
    # has to cover the span its index values reach. A draw of fewer than three
    # vertices rasterizes no triangle, so it could only ever produce the frame
    # the pass started from.
    nonindexed = indices is None
    if nonindexed:
        draw = _integer(case.get("vertices"), f"{where}.vertices", 1)
        _require(draw >= 3,
                 f"{where}: the reviewed non-indexed draw is at least one triangle")
        required_span = quad_stride * draw
    else:
        required_span = quad_stride * quad_vertices
    bindings = _list(vertex_buffers, f"{where}.vertex_buffers")
    _require(len(bindings) == 1, f"{where}: the reviewed shape binds one vertex stream")
    for position, binding in enumerate(bindings):
        _object(binding, ("allocation", "view", "offset", "length", "initial_hex"),
                f"{where}.vertex_buffers[{position}]")
        _require(binding["allocation"] > 0 and binding["view"] > 0,
                 f"{where}: zero vertex stream identity")
        _require(binding["length"] >= required_span,
                 f"{where}: the vertex stream is shorter than the reviewed "
                 + ("non-indexed draw reads" if nonindexed else "quad reads"))
        _require(len(_hex(binding["initial_hex"],
                          f"{where}.vertex_buffers[{position}].initial_hex")) == binding["length"],
                 f"{where}: the vertex stream bytes do not match its length")
        _require("format" not in binding,
                 f"{where}: a vertex stream carries no index format")
    if nonindexed:
        return {"vertices": draw, "indices": None, "draw": draw}
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
    return {"vertices": quad_vertices, "indices": quad_indices, "draw": quad_indices}


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


def _superset_declaration(case, streams, vertex_buffers, indices, where):
    """Pin the declared-superset vertex interface (`research/docs/23` §3.3, E-TX11).

    The shape is the census's: a *translated* vertex module reads a subset of
    the locations its contract's layout declares, and the extra declared
    attributes are bound and ignored. The geometry is one stream of 32-byte
    records carrying four `float32x2` attributes — the two the module reads at
    the record's head (locations 0 and 1, offsets 0 and 8) and the two it never
    reads at locations 2 and 3 (offsets 16 and 24) — so the fixture can state
    both halves of the rule: the frame is the read attributes' function, and the
    ignored bytes do not enter it. Anything else is refused here rather than
    read as a shape the review did not cover: two streams, another stride,
    another attribute set or a selected draw are not this arm.
    """
    quad_vertices, quad_indices, quad_stride = 4, 6, 8
    superset_stride = 32
    _require(len(streams) == 1,
             f"{where}: the declared-superset shape is one vertex stream")
    stream = streams[0]
    _object(stream, ("stride", "step", "attributes"),
            f"{where}.vertex_layout.buffers[0]")
    _require((stream["stride"], stream["step"]) == (superset_stride, "per_vertex"),
             f"{where}: the declared-superset stream is stride {superset_stride} and steps "
             "per vertex")
    attributes = _list(stream["attributes"],
                       f"{where}.vertex_layout.buffers[0].attributes")
    _require(len(attributes) == 4,
             f"{where}: the declared-superset layout declares four attributes")
    for location, attribute in enumerate(attributes):
        attribute_where = f"{where}.vertex_layout.buffers[0].attributes[{location}]"
        _object(attribute, ("location", "offset", "format"), attribute_where)
        _require((attribute["location"], attribute["offset"], attribute["format"])
                 == (location, location * 8, "float32x2"),
                 f"{attribute_where}: the declared attribute is location {location}, offset "
                 f"{location * 8}, float32x2")
    bindings = _list(vertex_buffers, f"{where}.vertex_buffers")
    _require(len(bindings) == 1,
             f"{where}: the declared-superset shape binds one vertex stream")
    binding = bindings[0]
    _object(binding, ("allocation", "view", "offset", "length", "initial_hex"),
            f"{where}.vertex_buffers[0]")
    _require(binding["allocation"] > 0 and binding["view"] > 0,
             f"{where}: zero vertex stream identity")
    _require("format" not in binding,
             f"{where}: a vertex stream carries no index format")
    _require(binding["length"] == superset_stride * 3,
             f"{where}: the declared-superset stream carries exactly the three records the "
             "fixture's triangle draws")
    _require(len(_hex(binding["initial_hex"],
                      f"{where}.vertex_buffers[0].initial_hex")) == binding["length"],
             f"{where}: the vertex stream bytes do not match its length")
    # The draw is indexed, which is the reviewed vertex-input shape's own
    # spelling: the object rails record a vertex-buffer draw through their
    # indexed entry points, and the three indices name the three vertices in
    # order, so the geometry the module reads is the fixture's own triangle.
    _object(indices, ("allocation", "view", "offset", "length", "initial_hex", "format"),
            f"{where}.indices")
    _require(indices["allocation"] > 0 and indices["view"] > 0,
             f"{where}: zero index buffer identity")
    index_width = {"uint16": 2, "uint32": 4}.get(indices["format"])
    _require(index_width is not None,
             f"{where}.indices: the declared-superset shape selects through a uint16 or "
             "uint32 index buffer")
    _require(indices["length"] == index_width * 3,
             f"{where}.indices: the declared-superset shape draws the three vertices the "
             "stream carries")
    index_bytes = _hex(indices["initial_hex"], f"{where}.indices.initial_hex")
    _require(len(index_bytes) == indices["length"],
             f"{where}.indices: the index bytes do not match their length")
    for position in range(3):
        chunk = index_bytes[position * index_width:(position + 1) * index_width]
        _require(int.from_bytes(chunk, "little") == position,
                 f"{where}.indices: the index at position {position} has to name vertex "
                 f"{position}")
    _require(case.get("base_vertex", 0) == 0,
             f"{where}: a base vertex needs an index buffer")
    _require(case["vertices"] == 3,
             f"{where}: the declared-superset shape draws the translated stage's three "
             "vertices")
    _require(case.get("instance_count", 1) == 1,
             f"{where}: the declared-superset shape draws one instance")
    # The two ignored attributes are what makes the arm falsifiable: their bytes
    # have to be far outside the clip space the read attributes use, so a rail
    # that read them as an input would rasterize a different frame.
    records = [_hex(binding["initial_hex"][index * superset_stride * 2:
                                           (index + 1) * superset_stride * 2],
                    f"{where}.vertex_buffers[0].initial_hex")
               for index in range(3)]
    ignored = [record[16:32] for record in records]
    for index, record in enumerate(ignored):
        for component in struct.unpack("<4f", record):
            _require(abs(component) >= 2.0,
                     f"{where}.vertex_buffers[0]: record {index}'s ignored attributes are "
                     "outside the clip space the read attributes cover, or a rail that read "
                     "them would land the same frame")
    return {"indices": 3, "draw": 3, "superset": True}


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
    return {"vertices": quad_vertices, "indices": quad_indices, "draw": quad_indices,
            "instanced": True,
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
    return {"vertices": 3, "indices": 3, "draw": 3}


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
    return {"vertices": quad_indices, "indices": quad_indices, "draw": quad_indices}


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
    return {"vertices": quad_indices, "indices": quad_indices, "draw": quad_indices,
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

    The v66 increment opens the depth face beside the stencil one and states
    the third reviewed stencil state (`research/docs/23` §3.3, v66): an
    `equal 0` test that *keeps* the value on pass and increments-wraps it on
    depth failure. Its three oversize triangles carry the same coverage, so
    the near one writes depth where it passes, the far one's depth failure
    then writes stencil, and the third one is the test that reads it: the
    expectation is the covered region's tint, its partially covered column's
    resolve against the clear colour, and the untouched remainder — and a
    rail that ignored either face would land the third triangle's tint on all
    of them.
    """
    quad_indices, stride = 6, 32
    # The v66 pair draws three oversize triangles with one coverage instead of
    # the two the other stencil shapes draw (`research/docs/23` §3.3, v66).
    rail_owned_pair = (case.get("depth") is not None
                       and "stencil_resolve" not in case)
    stream_indices, stream_hex = (
        (9, "000001000200030004000500060007000800") if rail_owned_pair
        else (quad_indices, "000001000200030004000500"))
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
        # The v66 rail-owned combined pair is its own reviewed state
        # (`research/docs/23` §3.3, v66): the equal-zero test that keeps the
        # value on pass and increments-wraps it on depth failure. Every other
        # stencil shape keeps the two states below.
        if rail_owned_pair:
            _require(test["compare"] == "equal" and test["fail_op"] == "keep"
                     and test["depth_fail_op"] == "increment_wrap"
                     and test["pass_op"] == "keep"
                     and (reference, read_mask, write_mask) == (0, 255, 255),
                     f"{where}: the rail-owned combined pair's stencil state is the equal-zero "
                     "test that keeps on pass and increments-wraps on depth failure, with both "
                     "masks wide open")
        else:
            reviewed_states = (
            # v47's equal-zero shape: the near triangle writes 1 where it
            # passes and the far triangle keeps its mask-closed zero.
            test["compare"] == "equal" and test["fail_op"] == "keep"
            and test["depth_fail_op"] == "keep" and test["pass_op"] == "increment_wrap"
            and (reference, read_mask, write_mask) == (0, 255, 255),
            # v60's depth-differentiated shape: an always test that increments
            # on depth pass and keeps on depth fail, so the near triangle
            # writes 1 through its depth pass while the far triangle's depth
            # failures leave the rest at zero.
            test["compare"] == "always" and test["fail_op"] == "keep"
            and test["depth_fail_op"] == "keep" and test["pass_op"] == "increment_wrap"
            and (reference, read_mask, write_mask) == (0, 255, 255),
            )
            _require(any(reviewed_states),
                     f"{where}: the reviewed stencil state is one of the two reviewed shapes: an "
                     "equal-zero test or a depth-differentiated always test, both with both "
                     "masks on and an increment-wrap pass op")
    bindings = _list(vertex_buffers, f"{where}.vertex_buffers")
    _require(len(bindings) == 1, f"{where}: the reviewed stencil shape binds one stream")
    binding = bindings[0]
    _object(binding, ("allocation", "view", "offset", "length", "initial_hex"),
            f"{where}.vertex_buffers[0]")
    _require(binding["allocation"] > 0 and binding["view"] > 0,
             f"{where}: zero vertex stream identity")
    _require(binding["length"] == stride * stream_indices,
             f"{where}: the reviewed stencil stream is {stream_indices} stride-{stride} "
             "vertices")
    _require(len(_hex(binding["initial_hex"], f"{where}.vertex_buffers[0].initial_hex"))
             == binding["length"],
             f"{where}: the vertex stream bytes do not match its length")
    _require(indices is not None, f"{where}: the reviewed stencil shape is indexed")
    _object(indices, ("allocation", "view", "offset", "length", "initial_hex", "format"),
            f"{where}.indices")
    _require(indices["initial_hex"] == stream_hex,
             f"{where}: the reviewed stencil indices are the reviewed triangles")
    # The stored stencil attachment, when the case declares one, is the
    # `(allocation, view, expected_texels)` triple the assembly below lands
    # beside the colour attachment; without one the surface is rail-owned and
    # observed by its effect on the colour side only, so the shape declares no
    # landing and the assembly owes it neither a writeback nor an allocation
    # image (`research/docs/23` §3.3, v47/v49).
    # The combined depth-stencil shape (`research/docs/23` §3.3, v60/v66)
    # opens a depth surface beside the stencil one. The v60 resolve shape's
    # depth attachment clears to a value between the two triangles' depths, so
    # the near triangle's depth pass writes stencil 1 while the far triangle's
    # depth failures leave the rest at zero, and its stored depth is a second
    # landing, the same triple the depth pair states. The v66 rail-owned pair
    # keeps neither face: the depth surface clears to one, both faces are
    # discarded with the pass, and the only landing is the colour resolve's
    # own.
    depth_store = None
    if case.get("depth") is not None:
        depth = case["depth"]
        _require(isinstance(depth, dict), f"{where}: a depth attachment is an object")
        allowed = {"format", "width", "height", "load", "clear_depth",
                   "store", "allocation", "view", "expected_hex"}
        _require(set(depth) - allowed == set(),
                 f"{where}.depth: unexpected fields "
                 + ", ".join(sorted(set(depth) - allowed)))
        _require(depth["format"] == "depth32float" and depth["load"] == "clear",
                 f"{where}: the combined depth surface is a cleared depth32float")
        if rail_owned_pair:
            # The v66 pair: the surface is rail-owned, so it carries no store
            # action, no identity and no expectation, and its clear is the one
            # the first triangle's `less` test writes over
            # (`research/docs/23` §3.3, v66).
            _require(depth.get("clear_depth") == 1.0
                     and set(depth) == {"format", "width", "height", "load", "clear_depth"},
                     f"{where}: the rail-owned combined pair clears its depth to one and "
                     "keeps neither face")
            _require(case.get("depth_test") == {"compare": "less", "write": True},
                     f"{where}: the combined pair's depth state is a less test with writes on")
        else:
            _require("stencil_resolve" in case,
                     f"{where}: the combined depth-stencil shape needs its stencil resolve")
            _require(depth.get("clear_depth") == 0.7 and depth.get("store") == "store",
                     f"{where}: the combined depth surface clears to 0.7 and stores")
            allocation = _integer(depth["allocation"], f"{where}.depth.allocation")
            view = _integer(depth["view"], f"{where}.depth.view")
            _require(allocation > 0 and view > 0,
                     f"{where}: zero depth attachment identity")
            expected = _hex(depth["expected_hex"], f"{where}.depth.expected_hex")
            width = _integer(depth["width"], f"{where}.depth.width", 1)
            height = _integer(depth["height"], f"{where}.depth.height", 1)
            _require(len(expected) == width * height * 4,
                     f"{where}.depth: the expected depth texels do not match the attachment")
            _require(case.get("depth_test") == {"compare": "less", "write": True},
                     f"{where}: the combined depth state is a less test with writes on")
            depth_store = (allocation, view, expected)
    else:
        _require("stencil" in case or "stencil_test" in case,
                 f"{where}: the reviewed stencil shape carries a stencil surface")
    return {"vertices": stream_indices, "indices": stream_indices, "draw": stream_indices,
            "stencil_store": stencil_store, "depth_store": depth_store}


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
    return {"vertices": quad_vertices, "indices": quad_indices, "draw": quad_indices}


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


def _mix_candidates(fragment, clear, samples):
    """Every exact k-of-`samples` mix of two colours (`research/docs/23` §3.3, v67).

    A constrained wildcard texel of a multisample raster may carry the resolve
    of a texel whose covered samples the draw wrote and whose other samples
    stayed at the pass's reference colour, so its candidate set is the closed
    set of mixtures those two colours produce. A mixture whose channels are not
    exactly representable is not a value any resolve of the reviewed shape may
    produce and is left out (`_resolve_texel` answers `None`), which keeps the
    set independent of a driver's rounding rule. `samples=1` is the single
    sample shape: the two colours themselves.
    """
    candidates = set()
    for covered in range(samples + 1):
        resolved = _resolve_texel(fragment, clear, covered, samples)
        if resolved is not None:
            candidates.add(resolved)
    return candidates


def _wildcard_allowed_texels(case, where):
    """Parse the constrained wildcard channel (`research/docs/23` §3.3, v67).

    The unconstrained list (`wildcard_texels`, v33) names bytes the case does
    not claim at all: whatever the rail left is accepted. This channel names
    the same positions but states, in advance, the *closed* set of byte values
    each one may carry — a resolved texel is one of the exact mixes its
    reference colours produce, never an arbitrary byte. The two forms are
    mutually exclusive: a case that states an allowed set may not also declare
    the same texels unclaimed, or the claim would be whichever the comparator
    read first. Returns a dict of texel index to a tuple of allowed values in
    declaration order, or `None` for a case that states none.
    """
    declared = case.get("wildcard_allowed_texels")
    if declared is None:
        return None
    declared = _list(declared, f"{where}.wildcard_allowed_texels")
    _require(declared,
             f"{where}: an allowed set has to name at least one texel")
    parsed = {}
    for position, entry in enumerate(declared):
        entry_where = f"{where}.wildcard_allowed_texels[{position}]"
        _object(entry, ("index", "allowed"), entry_where)
        texel = _integer(entry["index"], f"{entry_where}.index", 0)
        _require(texel not in parsed, f"{where}: duplicate wildcard texel {texel}")
        allowed = _list(entry["allowed"], f"{entry_where}.allowed")
        _require(allowed, f"{entry_where}: an allowed set has to name a value")
        values = []
        for value_position, value in enumerate(allowed):
            value_hex = _string(value, f"{entry_where}.allowed[{value_position}]")
            value = _hex(value_hex, f"{entry_where}.allowed[{value_position}]")
            _require(value_hex == value.hex(),
                     f"{entry_where}.allowed[{value_position}]: an allowed value has to "
                     "be lowercase bytes")
            _require(len(value) == 4,
                     f"{entry_where}.allowed[{value_position}]: a 8-bit texel is four bytes")
            _require(value not in values,
                     f"{entry_where}: duplicate allowed value {value_hex}")
            values.append(value)
        parsed[texel] = tuple(values)
    return parsed


def _check_allowed_texels(allowed, texel_count, fragment, samples, clear, where):
    """Hold a constrained wildcard's declared sets to the shape's own mixes.

    `allowed` names the texels whose bytes the case does not pin, and each of
    them states the closed set its bytes may come from. The shape fixes that
    set: a resolve of `samples` samples where the draw wrote `k` of them and the
    rest stayed at the pass's reference colour can only produce the exact
    k-of-`samples` mixes of the fragment output and that colour, so a declared
    set has to be exactly those values — a value outside them would accept a
    byte no reviewed rail may produce, and a missing one would refuse a resolve
    that is correct (`research/docs/23` §3.3, v67). `samples` is 1 for the
    single-sample shape, whose mixes are the two colours themselves, and every
    named texel has to stay inside the attachment's `texel_count` texels.
    """
    for texel in allowed:
        _require(texel < texel_count,
                 f"{where}: wildcard texel {texel} is outside the attachment")
    reference = _hex(clear, f"{where} reference colour")
    _require(len(reference) == 4, f"{where}: a reference colour is four bytes")
    _require(reference != fragment,
             f"{where}: a constrained wildcard texel needs a reference colour other "
             "than the fragment output")
    candidates = _mix_candidates(fragment, reference, samples)
    _require(len(candidates) > 1,
             f"{where}: the fragment output and the reference colour have to differ")
    for texel, values in allowed.items():
        for value in values:
            _require(value in candidates,
                     f"{where}: texel {texel} states the allowed value 0x{value.hex()}, "
                     "which is not an exact mix of the fragment output and the reference colour")
        missing = candidates - set(values)
        _require(not missing,
                 f"{where}: texel {texel} leaves out the resolve "
                 + ", ".join(f"0x{value.hex()}" for value in sorted(missing)))


def _affine_required_bytes(declaration, counts, where):
    """The byte extent one affine footprint reaches over the draw's own counts.

    The arithmetic is `metal_api_core::provider::render_affine_required_bytes`'s
    (`research/docs/23` §3.3, v86): each access reaches `base_offset +
    access_size` at its lowest index and adds `(count - 1) * stride` per term,
    and the widest access decides. The comparator recomputes it so a suite whose
    declaration its own draw cannot cover is refused on Linux, before any rail
    sees it.
    """
    required = 0
    for access in declaration:
        end = access["base_offset"] + access["access_size"]
        for term in access["terms"]:
            if term["axis"] >= len(counts):
                raise CaptureError(f"{where}: affine axis {term['axis']} is not one of the "
                                   "draw's own invocation axes")
            end += max(counts[term["axis"]] - 1, 0) * term["stride"]
        required = max(required, end)
    return required


def _stage_buffer_footprint(value, counts, where):
    """Parse one stage-buffer slot's declared footprint (`research/docs/23` §3.3, v86).

    Two arms exist and no third: a static ceiling in bytes, or the affine access
    set a translated module's reflection states. The return value is the byte
    extent the pass's own view has to cover — the ceiling for the static arm,
    the recomputed affine reach for the other — so the caller can hold the two
    declarations to each other.
    """
    _require(isinstance(value, dict) and len(value) == 1,
             f"{where}: a footprint is one of static and affine")
    if "static" in value:
        static = value["static"]
        _object(static, ("max_bytes",), f"{where}.footprint.static")
        return _integer(static["max_bytes"], f"{where}.footprint.static.max_bytes", 1)
    accesses = value.get("affine")
    _require(isinstance(accesses, dict) and set(accesses) == {"accesses"},
             f"{where}: an affine footprint is an access list")
    entries = _list(accesses["accesses"], f"{where}.footprint.affine.accesses")
    _require(entries, f"{where}: an affine footprint names at least one access")
    parsed = []
    for position, entry in enumerate(entries):
        access_where = f"{where}.footprint.affine.accesses[{position}]"
        _object(entry, ("base_offset", "access_size", "terms"), access_where)
        base_offset = _integer(entry["base_offset"], f"{access_where}.base_offset")
        access_size = _integer(entry["access_size"], f"{access_where}.access_size", 1)
        terms = []
        for index, term in enumerate(_list(entry["terms"], f"{access_where}.terms")):
            term_where = f"{access_where}.terms[{index}]"
            _object(term, ("axis", "stride"), term_where)
            axis = _integer(term["axis"], f"{term_where}.axis")
            _require(axis < RENDER_AFFINE_AXES,
                     f"{term_where}: affine axis {axis} is above the draw's two invocation "
                     "axes (vertex, instance)")
            terms.append({"axis": axis,
                          "stride": _integer(term["stride"], f"{term_where}.stride", 1)})
        parsed.append({"base_offset": base_offset, "access_size": access_size, "terms": terms})
    return _affine_required_bytes(parsed, counts, where)


def _stage_buffer_section(case, declaring_case, declaring_images, where):
    """Plan one render case's stage-buffer slots (`research/docs/23` §3.3, v83-v86).

    Each entry carries both halves of one slot: the pipeline's declaration
    (`stage`, `index`, `access`, `footprint`) and the pass's view of it
    (`allocation`, `view`, `offset`, `length` and the bytes themselves, or the
    lease they come from). The comparator holds the same fields the two rails
    hold, so a suite neither rail could execute is refused here first.

    A *writable* slot is a landing: its bytes leave through the same byte-keyed
    writeback channel a stored attachment's texels use, so the entry states the
    bytes the writeback has to carry and the view has to be one the declaring
    pass also declares — the trace's own view pool is where a write lands. The
    caller appends these landings after the colour, depth and stencil ones, in
    the order the rails report them.

    Returns `(writes, images, views, written, modes)`:

    * `writes` are `((allocation, view, offset), bytes)` landings in slot order;
    * `images` are the landed allocations' own images, the declaring case's
      image with each landing overlaid;
    * `views` maps view id to `(allocation, offset, length, access)`;
    * `written` is the set of landed allocations;
    * `modes` is `None` for a case whose slots are all trace-owned, or the
      `{view: mode}` map a capture of a lease-armed case has to report
      (`research/docs/23` §90, R9i).
    """
    entries = _list(case.get("stage_buffers", []), f"{where}.stage_buffers")
    if not entries:
        return [], {}, {}, set(), None
    _require(len(entries) <= MAX_RENDER_STAGE_BUFFERS,
             f"{where}: the reviewed stage-buffer ceiling is {MAX_RENDER_STAGE_BUFFERS} slots")
    # The draw the declarations are proven against: the shape is the
    # `vertex_id` triangle, so the vertex axis is the draw's own vertex count
    # and the instance axis its instance count.
    _require("indices" not in case,
             f"{where}: a stage-buffer case binds no index buffer")
    counts = (_integer(case["vertices"], f"{where}.vertices"),
              _integer(case.get("instance_count", 1), f"{where}.instance_count", 1))
    required_keys = {"stage", "index", "access", "footprint", "allocation", "view", "offset",
                     "length", "initial_hex"}
    allowed_keys = required_keys | {"allocation_size", "storage_mode", "expected_hex"}
    slots = []
    views = {}
    modes = {}
    writes = []
    images = {}
    written = set()
    for position, entry in enumerate(entries):
        entry_where = f"{where}.stage_buffers[{position}]"
        _require(isinstance(entry, dict), f"{entry_where}: expected an object")
        missing = sorted(required_keys - set(entry))
        _require(not missing, f"{entry_where}: missing fields {', '.join(missing)}")
        unexpected = sorted(set(entry) - allowed_keys)
        _require(not unexpected, f"{entry_where}: unexpected fields {', '.join(unexpected)}")
        stage = _string(entry["stage"], f"{entry_where}.stage")
        _require(stage in STAGE_BUFFER_STAGES,
                 f"{entry_where}: unknown stage-buffer stage {stage!r}")
        index = _integer(entry["index"], f"{entry_where}.index")
        _require(index < MAX_RENDER_STAGE_BUFFER_INDEX,
                 f"{entry_where}: stage buffer index {index} is at or above the reviewed "
                 f"ceiling {MAX_RENDER_STAGE_BUFFER_INDEX}")
        access = entry["access"]
        _require(access in STAGE_BUFFER_ACCESSES,
                 f"{entry_where}: unknown stage-buffer access {access!r}")
        # The contract's list is canonical — vertex bindings before fragment
        # bindings, ascending inside each stage, no slot twice
        # (`validate_stage_buffer_bindings`).
        # The ordinal is the contract's own (`RenderPipelineStage::code`):
        # vertex before fragment, so the two stages' index spaces can share one
        # canonical list.
        slot = (STAGE_BUFFER_STAGES.index(stage), index)
        _require(not slots or slots[-1] < slot,
                 f"{entry_where}: stage buffers are declared once each, vertex bindings before "
                 "fragment bindings and ascending inside each stage")
        slots.append(slot)
        footprint = _stage_buffer_footprint(entry["footprint"], counts, entry_where)
        allocation = _integer(entry["allocation"], f"{entry_where}.allocation", 1)
        view = _integer(entry["view"], f"{entry_where}.view", 1)
        _require(view not in views, f"{entry_where}: duplicate stage-buffer view {view}")
        offset = _integer(entry["offset"], f"{entry_where}.offset")
        length = _integer(entry["length"], f"{entry_where}.length", 1, MAX_ALLOCATION_BYTES)
        initial = _hex(entry["initial_hex"], f"{entry_where}.initial_hex")
        _require(len(initial) == length,
                 f"{entry_where}: the view's bytes do not cover its declared length")
        _require(footprint <= length,
                 f"{entry_where}: the declared footprint reaches {footprint} bytes, past the "
                 f"view's own {length}")
        mode = entry.get("storage_mode", "owned_bytes")
        _require(mode in BUFFER_STORAGE_MODES,
                 f"{entry_where}: unknown stage-buffer storage mode {mode!r}")
        allocation_size = entry.get("allocation_size")
        if mode == "owned_bytes":
            _require(allocation_size is None,
                     f"{entry_where}: an owned stage buffer maps no owner window, so it states "
                     "no allocation size")
        else:
            _require(allocation_size is not None,
                     f"{entry_where}: a lease-armed stage buffer states the allocation size its "
                     "owner window covers")
        if allocation_size is not None:
            _integer(allocation_size, f"{entry_where}.allocation_size", 1, MAX_ALLOCATION_BYTES)
            _require(offset + length <= allocation_size,
                     f"{entry_where}: the view lies outside its owner's registration")
        expected = entry.get("expected_hex")
        landed = access != "read"
        if landed:
            _require(expected is not None,
                     f"{entry_where}: a writable stage buffer states the bytes its writeback "
                     "lands")
        else:
            _require(expected is None,
                     f"{entry_where}: a read-only stage buffer carries no expectation")
        views[view] = (allocation, offset, length, access)
        modes[view] = mode
        if not landed:
            # A read-only slot carries its own bytes, exactly as a vertex
            # stream does, so it cannot name a view the declaring case already
            # declares: one identity would then stand for two byte strings.
            # (A writable slot is the other arm: it *must* name the declaring
            # view, because that is the pool entry its writeback lands in.)
            _require(all(buffer["view"] != view for buffer in declaring_case["buffers"]),
                     f"{entry_where}: a read-only stage buffer declares its own bytes, so view "
                     f"{view} cannot be one the declaring case already declares")
            continue
        landed_bytes = _hex(expected, f"{entry_where}.expected_hex")
        _require(len(landed_bytes) == length,
                 f"{entry_where}: a landing covers exactly the bytes its view covers")
        declared = [buffer for buffer in declaring_case["buffers"]
                    if buffer["allocation"] == allocation and buffer["view"] == view]
        _require(len(declared) == 1,
                 f"{entry_where}: a writable stage buffer lands in the trace's own view pool, so "
                 "the declaring case has to declare exactly that view")
        declared = declared[0]
        _require(declared["access"] == "read",
                 f"{entry_where}: the declaring pass reads the writable stage buffer's view")
        _require((declared["offset"], declared["length"]) == (offset, length),
                 f"{entry_where}: the writable stage buffer's view and the declaring case's "
                 "declaration of it cover different ranges")
        _require(_buffer_initial_bytes(declared, where, allocation, view, length)
                 == initial,
                 f"{entry_where}: the writable stage buffer's view carries different bytes than "
                 "the declaring case's declaration of it")
        image = bytearray(declaring_images[allocation])
        _require(len(image) == declared["allocation_size"],
                 f"{entry_where}: inconsistent allocation size")
        image[offset:offset + length] = landed_bytes
        images[allocation] = bytes(image)
        writes.append(((allocation, view, offset), landed_bytes))
        written.add(allocation)
    # A lease-armed case states one view per allocation, exactly as a compute
    # case's own lease rule does: the owner window is the view's own range
    # inside the registration (`research/docs/23` §90).
    if any(mode != "owned_bytes" for mode in modes.values()):
        _require(len(set(views[view][0] for view in views)) == len(views),
                 f"{where}: a case that declares a lease arm declares one view per allocation")
        return writes, images, views, written, modes
    return writes, images, views, written, None


def _landing_window_declaration(declaring_buffers, where, path, allocation, view, extent):
    """The declaring pass's own declaration of a landing arm's owner window
    (`research/docs/23` §115 之后的增量，E-TX13/E-TX14).

    Both arms resolve their window here and nowhere else, so the two spellings
    cannot drift into two answers: exactly one buffer of the declaring case
    carries the identity, it is read-only (a compute write would race the
    landing), it sits on the `borrowed_no_copy` arm — the window is the owner's
    registered mapping rather than a copy the provider holds — and its byte
    range is the attachment's own tightly packed extent.

    Returns the declaring buffer, whose own `initial_hex` is what "the window
    held before the arm ran" means to a case that has to be falsifiable about
    it.
    """
    declared = [buffer for buffer in declaring_buffers
                if buffer["allocation"] == allocation and buffer["view"] == view]
    _require(len(declared) == 1,
             f"{where}.{path}: the declaring case has to declare exactly the landing view")
    declared = declared[0]
    _require(declared.get("storage_mode") == "borrowed_no_copy",
             f"{where}.{path}: the landing view has to be the declaring pass's "
             "borrowed_no_copy window")
    _require(declared["access"] == "read",
             f"{where}.{path}: the declaring pass reads the landing view")
    _require(declared["length"] == extent,
             f"{where}.{path}: the landing view has to be the attachment's own extent")
    return declared


def _kept_frame_landing_section(case, attachment, declaring_buffers, where, expectation):
    """The kept-frame landing entry's own review surface (`research/docs/23`
    §115 之后的增量，E-TX14/R4b): the identity a pass kept in the provider's own
    image, the owner window a later entry delivers it into, and the bytes that
    window has to hold afterwards.

    The entry is the *deferred* sibling of the landing-view store, so the rules
    are the same facts stated a second way — one owner window, declared by the
    declaring pass as a read-only `borrowed_no_copy` view of the attachment's own
    extent — beside the two facts only the deferred shape has:

    * the kept identity has to be the resident attachment's own, because a pass
      cannot keep a frame for a surface it did not write;
    * the case carries no `expected_hex`: the pass publishes no writeback, so a
      case-level frame expectation would be a claim no observation checks
      (`research/docs/23` §115.5's boundary). The window is the whole
      observation, and `expected_landing_hex` is what states it;
    * that expectation may not equal the bytes the window held before the entry
      ran — with no writeback to compare against, that is the reading which
      keeps "the entry never ran" from passing as "the frame landed".

    Returns `(allocation, view, bytes)`.
    """
    section = case.get("kept_frame_landing")
    _require(section is not None,
             f"{where}: a resident store names the kept_frame_landing entry it delivers "
             "through")
    _object(section, ("frame", "landing"), f"{where}.kept_frame_landing")
    frame = section["frame"]
    landing = section["landing"]
    _object(frame, ("allocation", "view"), f"{where}.kept_frame_landing.frame")
    _object(landing, ("allocation", "view"), f"{where}.kept_frame_landing.landing")
    frame_allocation = _integer(frame["allocation"],
                                f"{where}.kept_frame_landing.frame.allocation")
    frame_view = _integer(frame["view"], f"{where}.kept_frame_landing.frame.view")
    _require(frame_allocation > 0 and frame_view > 0,
             f"{where}.kept_frame_landing.frame: zero kept-frame identity")
    _require((frame_allocation, frame_view)
             == (attachment["allocation"], attachment["view"]),
             f"{where}.kept_frame_landing.frame: the kept frame has to be the resident "
             "attachment's own identity")
    allocation = _integer(landing["allocation"],
                          f"{where}.kept_frame_landing.landing.allocation")
    view = _integer(landing["view"], f"{where}.kept_frame_landing.landing.view")
    _require(allocation > 0 and view > 0,
             f"{where}.kept_frame_landing.landing: zero landing identity")
    extent = attachment["width"] * attachment["height"] * 4
    window = _landing_window_declaration(declaring_buffers, where,
                                         "kept_frame_landing.landing", allocation, view,
                                         extent)
    landed = _hex(expectation, f"{where}.expected_landing_hex")
    _require(len(landed) == extent,
             f"{where}.expected_landing_hex: the landing expectation does not match the "
             "attachment")
    held = _hex(window.get("initial_hex"), f"{where}.kept_frame_landing.landing initial "
                "bytes")
    _require(landed != held,
             f"{where}.expected_landing_hex: the window already holds those bytes before "
             "the entry runs, so a rail that never delivered the kept frame could pass")
    return allocation, view, landed


def _landing_view_section(case, declaring_buffers, parsed, where, single):
    """The landing arm's own review surface (`research/docs/23` §115 之后的增量，
    E-TX13/E-TX14): the second view declaration the frame lands in, and the bytes
    the owner's window has to hold afterwards.

    Two arms state the same reading. A `"landing_view"` store lands the pass's
    own frame in the window as the pass completes; a `"resident"` store leaves
    the frame in the provider's image for a later landing *entry* to deliver, and
    the case then carries the `kept_frame_landing` section instead of the
    attachment-level one. Either way exactly one attachment may declare the arm —
    the observation is one owner window under one case-level expectation — and
    the window is a second declaration of the *declaring* pass rather than the
    attachment's own identity, which `StoreOp::Borrowed` already states.

    Returns `(allocation, view, bytes)` for the case that declares an arm, and
    `None` for every case that does not.
    """
    stored = [attachment for attachment, _, _, _ in parsed
              if attachment.get("store", "store") == "landing_view"]
    kept = [attachment for attachment, _, _, _ in parsed
            if attachment.get("store", "store") == "resident"]
    section = case.get("kept_frame_landing")
    expectation = case.get("expected_landing_hex")
    if not stored and not kept:
        _require(expectation is None,
                 f"{where}: expected_landing_hex needs a landing_view store, or a resident "
                 "one whose kept_frame_landing entry delivers the frame")
        _require(section is None,
                 f"{where}: kept_frame_landing needs a resident store")
        for attachment in parsed:
            _require("landing_view" not in attachment[0],
                     f"{where}: only a landing_view store names a landing view")
        return None
    _require(not stored or not kept,
             f"{where}: one case lands one owner window")
    _require(len(stored) + len(kept) == 1,
             f"{where}: one pass lands one owner window")
    _require(single,
             f"{where}: the landing-view store is the single-attachment shape")
    if kept:
        _require("expected_hex" not in case,
                 f"{where}: a resident store publishes no writeback, so the case carries "
                 "no expected_hex")
        _require("expected_rule" not in case and "readback_windows" not in case,
                 f"{where}: a resident store states its frame as expected_landing_hex, "
                 "not as a texel rule")
        return _kept_frame_landing_section(case, kept[0], declaring_buffers, where,
                                           expectation)
    attachment = stored[0]
    definition = attachment.get("landing_view")
    _object(definition, ("allocation", "view"), f"{where}.attachment.landing_view")
    allocation = _integer(definition["allocation"],
                          f"{where}.attachment.landing_view.allocation")
    view = _integer(definition["view"], f"{where}.attachment.landing_view.view")
    _require(allocation > 0 and view > 0,
             f"{where}.attachment.landing_view: zero landing identity")
    _require((allocation, view) != (attachment["allocation"], attachment["view"]),
             f"{where}.attachment.landing_view: the landing view is the attachment's own "
             "identity, which the borrowed store already states")
    extent = attachment["width"] * attachment["height"] * 4
    _landing_window_declaration(declaring_buffers, where, "attachment.landing_view",
                                allocation, view, extent)
    landed = _hex(expectation, f"{where}.expected_landing_hex")
    frame = _hex(case.get("expected_hex"), f"{where}.expected_hex")
    _require(landed == frame,
             f"{where}.expected_landing_hex: the window holds the frame the pass read back")
    return allocation, view, landed


def _unpublished_load_shape(attachment, extent, where):
    """The load shape an attachment that publishes nothing still has to state
    (`research/docs/23` §3.6, v19).

    Two arms reach this: a `dontcare` store discards the pass's writes, and a
    `resident` store keeps them in the provider's own image
    (`research/docs/23` §115 之后的增量，E-TX14/R4b). Neither owes the comparison
    an attachment image, but the pass still performs its load, so the load's own
    shape stays pinned: a clear colour is four bytes and travels without initial
    bytes, a `load` states the previous texels the pass starts from, and a
    `dontcare` load states neither.
    """
    load = attachment.get("load")
    if load == "clear":
        clear = _hex(attachment.get("clear_hex"), f"{where}.clear_hex")
        _require(len(clear) == 4, f"{where}: a clear colour is four bytes")
        _require("initial_hex" not in attachment,
                 f"{where}: a cleared attachment carries no initial bytes")
    elif load == "load":
        previous = _hex(attachment.get("initial_hex"), f"{where}.initial_hex")
        _require(len(previous) == extent,
                 f"{where}: initial texels do not match the attachment")
        _require("clear_hex" not in attachment,
                 f"{where}: a loaded attachment carries no clear colour")
    elif load == "dontcare":
        _require("clear_hex" not in attachment,
                 f"{where}: a dontcare load carries no clear colour")
        _require("initial_hex" not in attachment,
                 f"{where}: a dontcare load carries no initial bytes")
    else:
        raise CaptureError(f"{where}: unknown attachment load op {load!r}")


def _kept_frame_attachment(case):
    """The attachment whose store keeps its frame in the provider's own image
    (`research/docs/23` §115 之后的增量，E-TX14/R4b), or `None`.

    Reads both attachment spellings — the single `attachment` object and the MRT
    list — because the two `expected_hex` rules that run before the attachment
    section is parsed have to see the arm: the resident attachment has no
    writeback to state an expectation for, so its frame is spelled as
    `expected_landing_hex` instead.
    """
    attachments = [case["attachment"]] if "attachment" in case else case.get("attachments") or []
    for attachment in attachments:
        if attachment.get("store", "store") == "resident":
            return attachment
    return None


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
        required = ("id", "declaring_case", "vertex_entry", "fragment_entry",
                    "vertices", "viewport", "capture_rails")
        missing = [field for field in required if field not in case]
        _require(not missing, f"{where}: missing fields {', '.join(missing)}")
        unexpected = sorted(set(case) - set(required)
                            - {"attachment", "expected_hex", "attachments", "present", "icb",
                               "vertex_layout", "vertex_buffers", "indices", "scissor",
                               "instance_count", "wildcard_texels", "wildcard_allowed_texels",
                               "base_vertex", "fragment_textures",
                               "expected_rule", "readback_windows",
                               "depth", "depth_test", "coverage", "cull", "blend",
                               "stencil", "stencil_test", "multisample", "depth_resolve",
                               "requires_depth_resolve_filter", "stencil_resolve",
                               "requires_stencil_resolve_filter", "requires_sample_count",
                               "metal", "translated_stages", "stage_buffers",
                               "expected_landing_hex", "kept_frame_landing"})
        _require(not unexpected, f"{where}: unexpected fields {', '.join(unexpected)}")
        # A reviewed case pins the MSL module its two stages were written as; a
        # *translated* case pins its two AIR modules instead
        # (`research/docs/23` §3.3, v84). Exactly one of the two spellings is
        # present, so a case cannot claim a canonical MSL sibling its stages do
        # not have.
        translated = case.get("translated_stages")
        _require(("metal" in case) != (translated is not None),
                 f"{where}: exactly one of metal and translated_stages is required")
        if translated is not None:
            _object(translated, ("vertex", "fragment"), f"{where}.translated_stages")
            for kind in ("vertex", "fragment"):
                source = translated[kind]
                _object(source, ("path", "sha256"), f"{where}.translated_stages.{kind}")
                _string(source["path"], f"{where}.translated_stages.{kind}.path")
                _require(isinstance(source["sha256"], str)
                         and re.fullmatch(r"[0-9a-f]{64}", source["sha256"]),
                         f"{where}.translated_stages.{kind}: invalid source digest")
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
            # The kept-frame arm (`research/docs/23` §115 之后的增量，E-TX14) is
            # the one stored shape whose *frame* has no case-level expectation:
            # the pass publishes no writeback, so the owner window the landing
            # entry fills is the whole observation and `expected_landing_hex`
            # states it. `_landing_view_section` holds that spelling to the
            # arm's own rules rather than reading a missing `expected_hex`.
            _require("expected_hex" in case or "expected_rule" in case or "depth" in case
                     or _kept_frame_attachment(case) is not None,
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
        declaring_writes, _, _, group_expectations, _, _, _, _, _ = plan[declaring]
        _require(group_expectations is None,
                 f"{where}: the declaring case must be one submission")
        _require(not any(key in by_id[declaring] for key in ("programs", "dispatches",
                                                            "command_buffers")),
                 f"{where}: the declaring case must be one pass over its whole view pool")

        if "metal" in case:
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

        # The render sampler (`research/docs/23` §3.3, v70): the case binds one
        # texture whose own texels *are* the expectation, so "the fragment stage
        # sampled the texture" is the claim under test rather than a colour a
        # module could store without one. The extent rule is the rail's own: a
        # texture of another size puts some fragment's sample on a texel boundary
        # or inside a neighbour, which is a filtered read the review never
        # covered.
        fragment_textures = case.get("fragment_textures")
        if fragment_textures is not None:
            textures = _list(fragment_textures, f"{where}.fragment_textures")
            _require(len(textures) == 1,
                     f"{where}: the reviewed sampling shape binds exactly one texture")
            _require(single,
                     f"{where}: the reviewed sampling shape stores one attachment")
            for absent in ("vertex_layout", "vertex_buffers", "indices", "scissor",
                           "instance_count", "base_vertex", "cull", "blend",
                           "multisample", "depth", "depth_test", "depth_resolve",
                           "stencil", "stencil_test", "stencil_resolve", "present",
                           "icb", "coverage", "wildcard_texels",
                           "wildcard_allowed_texels"):
                _require(absent not in case,
                         f"{where}: the reviewed sampling shape carries no {absent}")
            texture_where = f"{where}.fragment_textures[0]"
            texture = textures[0]
            _require(isinstance(texture, dict), f"{texture_where}: expected an object")
            _require(set(texture).issubset({"allocation", "view", "format", "width",
                                            "height", "initial_hex", "texel_rule"}),
                     f"{texture_where}: unexpected fields")
            _require(_integer(texture.get("allocation"), f"{texture_where}.allocation") > 0
                     and _integer(texture.get("view"), f"{texture_where}.view") > 0,
                     f"{texture_where}: zero texture identity")
            attachment = case["attachment"]
            texture_format = _string(texture.get("format"), f"{texture_where}.format")
            _require(texture_format in SAMPLED_TEXTURE_FORMATS,
                     f"{texture_where}: the reviewed sampling stage reads one 8-bit "
                     "unorm surface, in either four-component byte order "
                     "(rgba8_unorm/bgra8_unorm) or in the narrow r8_unorm/rg8_unorm "
                     "lanes")
            attachment_format = _string(attachment.get("format"),
                                        f"{where}.attachment.format")
            _require(attachment_format in COLOUR_ATTACHMENT_FORMATS,
                     f"{where}.attachment: the sampled shape's colour attachment is one "
                     "8-bit four-component unorm surface, in either byte order "
                     "(rgba8_unorm/bgra8_unorm)")
            if translated is None:
                _require(texture.get("width") == attachment.get("width")
                         and texture.get("height") == attachment.get("height"),
                         f"{texture_where}: the sampled texture has to share the "
                         "attachment's extent")
            else:
                # The gathered-extent arm (`research/docs/23` §3.3, §111,
                # E-TX10): a *translated* fragment module states its own
                # absolute sample coordinates, so the rail binds the source at
                # its own extent and the case's whole point is that the two
                # extents differ. A case whose source shared the render area's
                # extent would be the window the reviewed pair already measures.
                _require(texture.get("width") != attachment.get("width")
                         or texture.get("height") != attachment.get("height"),
                         f"{texture_where}: the gathered arm's source has to differ from the "
                         "render area in at least one axis")
            _require(attachment.get("load") == "clear"
                     and attachment.get("store", "store") == "store",
                     f"{texture_where}: the reviewed sampling shape clears and stores "
                     "its attachment")
            texture_width = _integer(texture.get("width"), f"{texture_where}.width", 1)
            texture_height = _integer(texture.get("height"), f"{texture_where}.height", 1)
            clear = _hex(attachment.get("clear_hex"), f"{where}.attachment.clear_hex")
            _require(len(clear) == 4,
                     f"{where}.attachment.clear_hex: a clear colour is four bytes")
            if "texel_rule" in texture:
                # The rule form (R5a, `research/docs/23` §73): the texture's
                # texels and the attachment's expectation are one function of
                # the texel coordinates, because the sampling stage's sample at
                # a texel centre is an identity copy. The rule has to stay
                # injective over the extent it addresses and the clear colour has
                # to stay outside its reach — the closed-form siblings of the
                # distinctness rules the hex form is held to below.
                rule = _string(texture["texel_rule"], f"{texture_where}.texel_rule")
                _require(texture_format == attachment_format,
                         f"{texture_where}: a rule expectation states one layout's plane, so "
                         "the sampled texture and the attachment have to name it together")
                _require(rule == XY_U16LE_V1,
                         f"{texture_where}: unknown texel rule {rule!r}")
                _require(case.get("expected_rule") == rule,
                         f"{where}: the expectation has to be the texture's own rule: the "
                         "sampling stage's sample at a texel centre is an identity copy")
                _require("initial_hex" not in texture,
                         f"{texture_where}: a rule texture carries no initial_hex")
                _require("expected_hex" not in case,
                         f"{where}: a rule expectation carries no expected_hex")
                _require(texture_width >= RULE_MIN_DIMENSION
                         and texture_height >= RULE_MIN_DIMENSION,
                         f"{texture_where}: a texel rule is the megapixel form; "
                         f"{texture_width}x{texture_height} states its texels instead")
                _require(texture_width <= RULE_ADDRESS_CEILING
                         and texture_height <= RULE_ADDRESS_CEILING,
                         f"{texture_where}: the reviewed rule addresses at most "
                         f"{RULE_ADDRESS_CEILING} texels per axis")
                _require(not _rule_reaches_colour(rule, clear, texture_width, texture_height),
                         f"{where}.attachment: a rule texel equals the clear colour, so a "
                         "rail that ignored the draw could pass")
            else:
                _require("expected_rule" not in case,
                         f"{where}: a rule expectation travels with the texture's own "
                         "texel rule")
                _require("readback_windows" not in case,
                         f"{where}: readback windows travel with the texture's own texel rule")
                texels = _hex(texture.get("initial_hex"), f"{texture_where}.initial_hex")
                expected = _hex(case.get("expected_hex"), f"{where}.expected_hex")
                # The texels the *frame* carries, not the source's bytes
                # (`research/docs/23` §113): a narrow source states one or two
                # bytes per texel, so the distinctness and clear-collision scans
                # run over the four-component texel each one is read out as. The
                # frame is the render area's, which the same-extent window makes
                # equal to the source's and the gathered arm states separately.
                if translated is None:
                    # The source's own byte extent, one texel wide for a narrow
                    # lane (`research/docs/23` §113): a spelling that disagrees
                    # with it states bytes the texture does not have, so the
                    # length rule answers before the expectation is derived.
                    _require(len(texels) == texture_width * texture_height
                             * SAMPLED_TEXEL_BYTES[texture_format],
                             f"{texture_where}: texture initial length does not match "
                             "its extent")
                    chunks = [expected[offset:offset + 4]
                              for offset in range(0, len(expected), 4)]
                    _require(len(chunks) == texture_width * texture_height,
                             f"{texture_where}: the frame's texels do not match the extent")
                    _require(len(set(chunks)) == len(chunks),
                             f"{texture_where}: the frame's texels have to be pairwise "
                             "distinct, or a repeated read could pass")
                    _require(clear not in chunks,
                             f"{texture_where}: a frame texel equals the clear colour, so a "
                             "rail that ignored the texture could pass")
                    _require(expected == _sampled_expectation(texture_format, attachment_format,
                                                              texels,
                                                              texture_width * texture_height),
                             f"{where}: the expectation has to be the uploaded texels in the "
                             "attachment's own byte order: the sampling stage's sample at a "
                             "texel centre is an identity copy")
                else:
                    # The gathered arm's expectation is the module's own reading
                    # of the source bytes, and only the module knows its sample
                    # coordinates, so the fixture states those bytes and the
                    # comparator holds them to the render area's own byte count
                    # — the same shape the translated stage-buffer case's frame
                    # has (`research/docs/23` §3.3, E-TX10).
                    destination_width = _integer(attachment.get("width"),
                                                 f"{where}.attachment.width", 1)
                    destination_height = _integer(attachment.get("height"),
                                                  f"{where}.attachment.height", 1)
                    _require(len(expected) == destination_width * destination_height * 4,
                             f"{where}.expected_hex: the gathered arm's expectation is the "
                             "render area's own bytes")
                    _require(_hex(case.get("expected_hex"),
                                  f"{where}.expected_hex") != clear * (destination_width
                                                                       * destination_height),
                             f"{where}.expected_hex: an all-clear frame is what a rail that "
                             "ignored the draw would land")

        vertex_input = _vertex_input_declaration(case, where)
        # The single attachment form spells its expectation at the case level —
        # unless the pass's landing is its stored depth attachment, which is the
        # v45 depth-only shape: the colour attachment still renders, its bytes
        # disappear, and the depth texels are the whole observation. The depth
        # declaration above is what tells the two apart, so this rule can only
        # be stated once that declaration is parsed.
        depth_landing = vertex_input is not None and vertex_input.get("depth_store") is not None
        if single and not depth_landing:
            # The kept-frame arm (`research/docs/23` §115 之后的增量，E-TX14/R4b)
            # is the stored shape whose frame the pass does not publish: its
            # owner window is the whole observation, and `expected_landing_hex`
            # is where the case states it. `_landing_view_section` holds that
            # spelling to the arm's own rules.
            _require("expected_hex" in case or "expected_rule" in case
                     or _kept_frame_attachment(case) is not None,
                     f"{where}: missing fields expected_hex")
        if vertex_input is None:
            _require(case["vertices"] == 3, f"{where}: expected the full-screen triangle")
            _require(not multiple,
                     f"{where}: the milestone vertex_id shape renders one attachment")
        else:
            # The count the case states is the draw's own: an indexed arm spends
            # its index count, the non-indexed arm its vertex count
            # (`research/docs/23` §3.3, v39).
            _require(case["vertices"] == vertex_input["draw"],
                     f"{where}: the reviewed draw spends {vertex_input['draw']} vertices")
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
        # The depth resolve (`research/docs/23` §3.3, v57) only means something
        # beside a multisample raster that keeps its depth surface: the resolve
        # is the reduction of the stored four-sample texels, so any other shape
        # is refused instead of silently ignored. The filter is the closed
        # three-value family, and the expectation then follows the depth
        # pair's own per-texel rule rather than the colour resolve arithmetic.
        depth_resolve = case.get("depth_resolve")
        if depth_resolve is not None:
            _require(case.get("multisample") is not None,
                     f"{where}: a depth resolve needs a multisample raster")
            _require(case.get("depth", {}).get("store") is not None,
                     f"{where}: a depth resolve needs a stored depth surface")
            _object(depth_resolve, ("filter",), f"{where}.depth_resolve")
            _require(depth_resolve["filter"] in ("sample0", "min", "max"),
                     f"{where}.depth_resolve: unsupported depth resolve filter")
        # The device gate (`research/docs/23` §3.3, v57d): a case that requires
        # a filter appears in a capture if and only if the capture's
        # `depth_resolve_modes` mask carries that filter's bit. The gate has to
        # name the resolve the case states — it is the case's own admission
        # condition, not a second spelling that could drift — and only the two
        # filters a device may lack are gateable: Sample0 is the API's own
        # baseline.
        requires_filter = case.get("requires_depth_resolve_filter")
        if requires_filter is not None:
            _require(requires_filter in ("min", "max"),
                     f"{where}: the device gate names the min or max depth resolve filter")
            _require(depth_resolve is not None
                     and depth_resolve["filter"] == requires_filter,
                     f"{where}: the device gate has to name the resolve filter the case states")
        # The stencil resolve (`research/docs/23` §3.3, v60) is the depth
        # resolve's sibling one byte wide: it only means something beside a
        # multisample raster that keeps its stencil surface, and the
        # `depth_resolved_sample` filter names the sample the depth resolve
        # selected, so it is refused without one instead of silently degrading
        # to sample zero.
        stencil_resolve = case.get("stencil_resolve")
        if stencil_resolve is not None:
            _require(case.get("multisample") is not None,
                     f"{where}: a stencil resolve needs a multisample raster")
            _require(case.get("stencil", {}).get("store") is not None,
                     f"{where}: a stencil resolve needs a stored stencil surface")
            _object(stencil_resolve, ("filter",), f"{where}.stencil_resolve")
            _require(stencil_resolve["filter"] in ("sample0", "depth_resolved_sample"),
                     f"{where}.stencil_resolve: unsupported stencil resolve filter")
            if stencil_resolve["filter"] == "depth_resolved_sample":
                _require(depth_resolve is not None,
                         f"{where}: the depth_resolved_sample stencil resolve names the "
                         "sample the depth resolve selects, so the case has to state a "
                         "depth resolve")
        # The stencil device gate (`research/docs/23` §3.3, v60): the sibling
        # of the depth gate, naming the one stencil filter a device may lack.
        requires_stencil_filter = case.get("requires_stencil_resolve_filter")
        if requires_stencil_filter is not None:
            _require(requires_stencil_filter == "depth_resolved_sample",
                     f"{where}: the device gate names the depth_resolved_sample stencil "
                     "resolve filter")
            _require(stencil_resolve is not None
                     and stencil_resolve["filter"] == requires_stencil_filter,
                     f"{where}: the device gate has to name the resolve filter the case states")
        # The sample-count device gate (`research/docs/23` §3.3, v61): a case
        # that requires a sample count appears in a capture if and only if the
        # capture's sample ceiling is at least that count. The gate has to name
        # the raster the case states, and only the two counts a device may lack
        # are gateable: 4x is the v51 baseline every multisampling device
        # admits.
        requires_sample_count = case.get("requires_sample_count")
        if requires_sample_count is not None:
            _require(_integer(requires_sample_count,
                              f"{where}.requires_sample_count", 1) in (2, 8),
                     f"{where}: the device gate names the two- or eight-sample raster")
            _require(case.get("multisample") is not None
                     and case["multisample"]["sample_count"] == requires_sample_count,
                     f"{where}: the device gate has to name the sample count the case states")
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
        else:
            wildcard_texels = None
        # The constrained wildcard channel (`research/docs/23` §3.3, v67): the
        # same shape as the unconstrained list, but every named texel states
        # the closed set of byte values it may carry instead of leaving the
        # bytes unclaimed. The shape belongs to a `dontcare` load's undefined
        # pre-pass contents just as the v33 list does — nothing else makes an
        # unobserved texel legitimate — and the two forms cannot be stated
        # together. The per-texel candidate rules (soundness against the
        # reference mix, completeness, and the multisample shape) are checked
        # once the attachment loop has parsed the case's colours.
        wildcard_allowed = _wildcard_allowed_texels(case, where) if single else None
        if wildcard_allowed is not None:
            _require(wildcard_texels is None,
                     f"{where}: the two wildcard channels are mutually exclusive")
            _require(definitions[0].get("store", "store") == "store",
                     f"{where}: an allowed set needs a stored attachment")
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
                              f"{where}.multisample.sample_count", 1) in (2, 4, 8),
                     f"{where}: the reviewed multisample rasters are two, four or eight "
                     "samples")
            # The attachment's load (`research/docs/23` §3.3, v51/v67): the
            # reviewed pass opens it from a clear, whose colour is then the
            # reference every expectation is a resolve of — or, from v67 on,
            # from `dontcare` while every unclaimed texel states the closed set
            # its resolve may land in — or, from v82 on, from `load`, whose
            # declared window is one repeated texel the rail seeds every sample
            # of the raster with before the measured pass opens it, so the
            # reference is that texel itself. The free `wildcard_texels` list
            # stays refused here: a nominal colour the expectation does not mix
            # with is what the constrained form exists for.
            _require(case["attachment"].get("load") in ("clear", "dontcare", "load"),
                     f"{where}: the reviewed multisample pass opens its attachment from a "
                     "clear, a loaded seed or a dontcare load with constrained wildcard "
                     "texels")
            # The depth surface beside the raster is admitted from v53 on and
            # the stencil surface from v55 (`research/docs/23` §3.3, v53/v55):
            # both are rail-owned — the pass tests and writes them, but keeping
            # their texels would need a resolve the two APIs spell differently —
            # and the expectation then follows the pair's own uniform rule
            # instead of the resolve rule. Opening both faces is the combined
            # depth-stencil surface — one attachment both faces share — and two
            # shapes of it are reviewed: the v60 resolve shape, which states
            # the stencil resolve its stored faces resolve through, and the v66
            # rail-owned write-then-test pair, whose two faces are discarded
            # with the pass and whose observation is the colour resolve
            # (`research/docs/23` §3.3, v60/v66).
            combined_pair = "depth" in case and "stencil" in case
            rail_owned_combined = combined_pair and "stencil_resolve" not in case
            _require(not rail_owned_combined
                     or (case["depth"].get("store") is None
                         and case["stencil"].get("store") is None),
                     f"{where}: the combined depth-stencil pair keeps both faces or neither")
            if "depth" in case:
                # A stored multisampled depth surface is admitted from v57 on,
                # through the resolve the case then has to state: its texels
                # are only observable as the resolve's reduction, so a stored
                # surface without one is refused, and the resolve's filter was
                # already held to the closed family above.
                if case["depth"].get("store") is not None:
                    _require("depth_resolve" in case,
                             f"{where}: a stored multisampled depth surface needs its "
                             "depth resolve")
                # The combined shape's stencil half states the same stored
                # surface rule its single-surface sibling does
                # (`research/docs/23` §3.3, v60).
                if "stencil" in case and case["stencil"].get("store") is not None:
                    _require("stencil_resolve" in case,
                             f"{where}: a stored multisampled stencil surface needs its "
                             "stencil resolve")
                if rail_owned_combined:
                    # The v66 pair's whole point is the partially covered
                    # column: the two faces' tests decide per sample, so the
                    # case claims the partial coverage its expectation then
                    # shows (`research/docs/23` §3.3, v66).
                    _require(coverage == "partial",
                             f"{where}: the rail-owned combined pair claims the partial "
                             "coverage it resolves")
                else:
                    _require(coverage is None,
                             f"{where}: a multisample pass with a depth surface claims no "
                             "partial coverage")
            elif "stencil" in case:
                # A stored multisampled stencil surface is admitted from v60
                # on, through the resolve the case then has to state: its
                # texels are only observable as the resolve's reduction, so a
                # stored surface without one is refused.
                if case["stencil"].get("store") is not None:
                    _require("stencil_resolve" in case,
                             f"{where}: a stored multisampled stencil surface needs its "
                             "stencil resolve")
                _require(coverage is None,
                         f"{where}: a multisample pass with a stencil surface claims no partial "
                         "coverage")
            else:
                # The colour-only raster admits both expectation shapes the
                # v61 increment reviews: a partial coverage claim is the v51
                # edge fixture's resolve rule, and an absent claim is the v61
                # full-coverage fixtures' uniform rule — every texel is the
                # fragment output. A present action beside the raster is
                # admitted from v62 on: it hands on the resolve landing, which
                # the fixture's attachment view already is, so the present
                # section's own rules apply unchanged.
                pass
            # The reviewed multisample shapes still carry no ICB: an indirect
            # replay beside a four-sample resolve is the increment that
            # reviews the two together (`research/docs/25` §5.2).
            _require("icb" not in case,
                     f"{where}: a multisample case carries no ICB")
            # A multisample expectation claims every texel it resolves, and a
            # texel it does *not* claim needs the closed candidate set the
            # constrained channel states (`research/docs/23` §3.3, v67): a
            # resolve of undefined pre-pass contents is not free to be any
            # byte, so the unconstrained list of a single-sample `dontcare`
            # load has no meaning beside a raster.
            _require(wildcard_texels is None,
                     f"{where}: the multisample raster claims every texel it resolves, or "
                     "states the closed allowed set of a constrained wildcard texel")
            # The trace rails' footprint proof is the only gate that would
            # notice an offset index span, and no reviewed fixture covers the
            # shape; the object rail has no entry that carries both. Refusing it
            # here keeps the fixture gate as strict as the rails
            # (`research/docs/23` §3.3, v54 review H1).
            _require(case.get("base_vertex", 0) == 0,
                     f"{where}: the reviewed multisample shapes carry no base vertex")
        expected_bytes = []
        parsed = []
        rule_expectation = None
        for position, attachment in enumerate(definitions):
            attachment_where = (f"{where}.attachment" if single
                                else f"{where}.attachments[{position}]")
            _require(isinstance(attachment, dict), f"{attachment_where}: expected an object")
            allowed = {"allocation", "view", "format", "width", "height",
                       "load", "store", "clear_hex", "initial_hex", "landing_view"}
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
            # (`research/docs/23` §3.3, v27; widened by R1b, §70): the reviewed
            # fragment stages are extent-independent, so any 1..=64 square or
            # rectangle is a shape a fixture may pin, as long as every
            # attachment of one pass shares it. The ceiling is the reviewed
            # window every conformant device's declared window covers — the
            # 16×16 case and the 64×64 boundary measure it.
            _require(1 <= width <= REVIEWED_ATTACHMENT_CEILING
                     and 1 <= height <= REVIEWED_ATTACHMENT_CEILING,
                     f"{attachment_where}: the attachment extent is one to "
                     f"{REVIEWED_ATTACHMENT_CEILING} texels per axis")
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
            # The landing-view store (`research/docs/23` §115 之后的增量，
            # E-TX13) is the stored arm plus its own second declaration, so it
            # keeps every rule the stored arm states; the section below pins the
            # declaration itself. The kept-frame arm (E-TX14/R4b) is the stored
            # arm minus its writeback: the pass keeps its frame in the
            # provider's own image and the owner window a later landing entry
            # fills is the whole observation, so the attachment carries no
            # expectation of its own either.
            _require(store in ("store", "dontcare", "landing_view", "resident"),
                     f"{attachment_where}: a discarded attachment cannot be compared")
            extent = width * height * 4
            if store == "resident":
                # The pass publishes nothing through the writeback channel, so
                # neither the attachment nor the single-attachment case may
                # state an expectation: `expected_landing_hex` is the arm's own
                # spelling of the frame, and `_landing_view_section` holds it
                # to the window it lands in.
                _require("expected_hex" not in attachment,
                         f"{attachment_where}: a resident store publishes no writeback, "
                         "so the attachment carries no expected_hex")
                _unpublished_load_shape(attachment, extent, attachment_where)
                parsed.append((attachment, allocation, view, None))
                continue
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
                _unpublished_load_shape(attachment, extent, attachment_where)
                parsed.append((attachment, allocation, view, None))
                continue
            # The stored arm keeps the v13-v18 shape: its expectation is the
            # whole reason the attachment is comparable.
            if multiple:
                _require("expected_hex" in attachment,
                         f"{attachment_where}: a stored attachment needs expected_hex")
            if case.get("expected_rule") is not None:
                # The rule form (R5a, `research/docs/23` §73): the expectation
                # is the reviewed function of the texel coordinates rather than
                # four million texels of hex. The sampled-shape block above
                # already held the rule, the extent and the clear colour to the
                # rule's own admissibility; what this arm adds is the landing
                # shape — the windows the capture reports, the single-attachment
                # form, and the clearing load a rule-expected attachment has.
                _require(single,
                         f"{where}: the rule expectation belongs to the single attachment "
                         "form")
                _require(case.get("fragment_textures") is not None,
                         f"{where}: a rule expectation is the reviewed sampling shape's")
                _require(attachment.get("load") == "clear",
                         f"{attachment_where}: a rule-expected attachment clears")
                _require("initial_hex" not in attachment,
                         f"{attachment_where}: a cleared attachment carries no initial bytes")
                rule = _string(case["expected_rule"], f"{where}.expected_rule")
                rule_expectation = RuleExpectation(
                    rule, width, height, _rule_digest(rule, width, height),
                    _readback_windows(case, rule, width, height, where))
                expected = _rule_plane(rule, width, height)
                _require(len(expected) == extent,
                         f"{attachment_where}: expected texel bytes do not match the attachment")
                parsed.append((attachment, allocation, view, expected))
                expected_bytes.append(expected)
                continue
            # A stored attachment's expectation is the whole reason it is
            # comparable: the MRT form states it on the entry and the
            # single-attachment form at the case level. A case that states
            # neither has no observation at all — the kept-frame arm is the one
            # such shape, and it continued above — so this is a refusal rather
            # than a lookup that could raise a bare `KeyError`.
            declared_expectation = (case.get("expected_hex") if single
                                    else attachment.get("expected_hex"))
            _require(declared_expectation is not None,
                     f"{attachment_where}: a stored attachment needs expected_hex")
            expected = _hex(declared_expectation, f"{attachment_where}.expected_hex")
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
                # A cleared raster has no undefined pre-pass contents, so the
                # fixture can pin every byte and the *free* list stays refused
                # here. The constrained channel (`research/docs/23` §3.3, v69)
                # is the one exception, and only beside a multisample raster
                # that claims the partial coverage its allowed set resolves:
                # the named texels are the ones whose covered-sample count the
                # rails' own sample *positions* do not pin, while the colours
                # the resolve is built from stay the case's own.
                if wildcard_allowed is not None:
                    _require(multisample is not None,
                             f"{attachment_where}: a cleared attachment has no unclaimed texel")
                    _require(coverage == "partial",
                             f"{attachment_where}: a cleared multisample raster states the "
                             "partial coverage its allowed set resolves")
                _require(clear != texel,
                         f"{attachment_where}: the clear colour equals the expected texel")
                # The combined depth-stencil shape's mixed column carries the
                # k-of-four colour resolve, so it follows the resolve rule; the
                # single-surface shapes keep the uniform pair rule, and so do
                # the v61 full-coverage colour-only fixtures — only a raster
                # that *claims* partial coverage is owed a mixed texel
                # (`research/docs/23` §3.3, v53/v55/v60/v61/v66).
                combined_pair = (case.get("depth") is not None
                                 and case.get("stencil") is not None)
                if multisample is not None and (
                        (case.get("depth") is None and case.get("stencil") is None
                         and coverage == "partial")
                        or case.get("stencil_resolve") is not None
                        or combined_pair):
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
                        # A texel the case leaves to its allowed set
                        # (`research/docs/23` §3.3, v69) states its claim as
                        # that closed set instead of one byte, so the exact
                        # resolve rule below holds for every texel the case
                        # does pin.
                        if wildcard_allowed is not None and index in wildcard_allowed:
                            continue
                        for covered in range(samples + 1):
                            if chunk == _resolve_texel(texel, clear, covered, samples):
                                covered_seen.add(covered)
                                break
                        else:
                            raise CaptureError(
                                f"{attachment_where}: texel {index} is not the resolve of "
                                f"any coverage of the {samples}-sample raster")
                    if wildcard_allowed is None:
                        _require(any(0 < covered < samples for covered in covered_seen),
                                 f"{attachment_where}: a multisample expectation needs at "
                                 "least one partially covered texel")
                    _require(0 in covered_seen and samples in covered_seen,
                             f"{attachment_where}: a multisample expectation needs both "
                             "a fully covered and an uncovered texel")
                    if wildcard_allowed is not None:
                        # The allowed set is held to the case's own arithmetic
                        # by the same rule the `dontcare` shape states
                        # (`research/docs/23` §3.3, v67): the reference colour
                        # is the attachment's own clear, and the declared sets
                        # have to be exactly the exact k-of-`sample_count`
                        # mixes of the two colours. The pinned texels then
                        # carry both extremes, and the allowed sets have to
                        # admit a partial mix — the one shape a single-sample
                        # raster could not produce.
                        _check_allowed_texels(wildcard_allowed, len(texels), texel, samples,
                                              attachment["clear_hex"], attachment_where)
                        _require(any(_resolve_texel(texel, clear, covered, samples) is not None
                                     for covered in range(1, samples)),
                                 f"{attachment_where}: a cleared multisample raster that leaves "
                                 "a texel to its allowed set needs an exactly representable "
                                 "partial mix of its colours")
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
                    if case.get("fragment_textures") is not None:
                        # The render sampler's expectation is the uploaded
                        # texture itself (`research/docs/23` §3.3, v70), so its
                        # claim is the identity the fragment-texture block
                        # above already pinned: the texels are pairwise distinct
                        # and none is the clear colour, which is what rules out
                        # a uniform store or a repeated read.
                        pass
                    elif vertex_input and vertex_input.get("instanced"):
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
                if multisample is not None:
                    # The loaded multisampled raster (`research/docs/23` §82,
                    # v82): a multisampled image cannot be uploaded into — the
                    # transfer commands are single-sample at both ends — so the
                    # rail states the declared window as one clear its own seed
                    # pass writes into every sample, and the measured pass then
                    # opens the image with `LOAD`. A clear value is one colour
                    # for the whole attachment, so the window has to be one
                    # repeated texel, and every pinned texel is the exact
                    # k-of-`sample_count` resolve of that seed and the fragment
                    # output. The reference colour of the mixes is therefore the
                    # seed itself, not a `clear_hex` a load does not carry.
                    _require(wildcard_texels is None,
                             f"{attachment_where}: the multisample raster claims every "
                             "texel it resolves, or states the closed allowed set of a "
                             "constrained wildcard texel")
                    _require(coverage == "partial",
                             f"{attachment_where}: a loaded multisample raster states the "
                             "partial coverage its seed resolves")
                    samples = multisample["sample_count"]
                    seed = previous[:4]
                    _require(all(previous[offset:offset + 4] == seed
                                 for offset in range(0, len(previous), 4)),
                             f"{attachment_where}: a seeded multisample raster loads one "
                             "repeated texel; a per-texel seed is not a shape the reviewed "
                             "rails execute")
                    _require(seed != texel,
                             f"{attachment_where}: the seed equals the fragment output")
                    if wildcard_allowed is not None:
                        # The constrained channel, hosted by the seed: the named
                        # texels state the closed set of exact k-of-N mixes of
                        # the fragment output and the seed, and the pinned texels
                        # still carry both extremes.
                        _check_allowed_texels(wildcard_allowed, len(texels), texel, samples,
                                              seed.hex(), attachment_where)
                        _require(any(_resolve_texel(texel, seed, covered, samples) is not None
                                     for covered in range(1, samples)),
                                 f"{attachment_where}: a seeded multisample raster that "
                                 "leaves a texel to its allowed set needs an exactly "
                                 "representable partial mix of its colours")
                    covered_seen = set()
                    for position, chunk in enumerate(texels):
                        if wildcard_allowed is not None and position in wildcard_allowed:
                            continue
                        for covered in range(samples + 1):
                            if chunk == _resolve_texel(texel, seed, covered, samples):
                                covered_seen.add(covered)
                                break
                        else:
                            raise CaptureError(
                                f"{attachment_where}: texel {position} is not the resolve of "
                                f"any coverage of the {samples}-sample raster")
                    if wildcard_allowed is None:
                        _require(any(0 < covered < samples for covered in covered_seen),
                                 f"{attachment_where}: a multisample expectation needs at "
                                 "least one partially covered texel")
                    _require(0 in covered_seen and samples in covered_seen,
                             f"{attachment_where}: a multisample expectation needs both "
                             "a fully covered and an uncovered texel")
                else:
                    # A single-sample loaded attachment hands the pass its own
                    # bytes, so no texel is unclaimed and neither wildcard
                    # channel has a meaning here: the expectation compares
                    # against what the load carried (`research/docs/23` §3.3,
                    # v69). The free list states the same rule where it is
                    # parsed. Partial coverage, both directions: every texel is
                    # either the fragment output or the byte the load handed it,
                    # every drawn texel carries the same output, and both halves
                    # appear.
                    _require(wildcard_allowed is None,
                             f"{attachment_where}: a loaded attachment has no unclaimed texel")
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
                if wildcard_allowed is not None:
                    # The constrained channel (`research/docs/23` §3.3, v67):
                    # the named texels' bytes are the resolve of a raster's
                    # samples, so beside a raster they are exactly the resolve
                    # mixes of the case's own colours, and on the single-sample
                    # shape they are the two colours themselves. Both are the
                    # closed set the case has to state.
                    _require("clear_hex" in attachment,
                             f"{attachment_where}: a constrained wildcard texel needs the "
                             "reference colour of its mixes")
                    _check_allowed_texels(
                        wildcard_allowed, len(texels), texel,
                        multisample["sample_count"] if multisample is not None else 1,
                        attachment["clear_hex"], attachment_where)
                    for position in range(len(texels)):
                        if position not in wildcard_allowed:
                            _require(texels[position] == texel,
                                     f"{attachment_where}: texel {position} of a dontcare load "
                                     "has to be the fragment output")
                elif wildcard_texels is not None:
                    _require(multisample is None,
                             f"{attachment_where}: a multisample case states the closed "
                             "allowed set of its unclaimed texels")
                    claimed = [position for position in range(len(texels))
                               if position not in wildcard_texels]
                    _require(claimed,
                             f"{attachment_where}: a wildcard list has to leave at least one "
                             "texel observed")
                    for position in claimed:
                        _require(texels[position] == texel,
                                 f"{attachment_where}: texel {position} of a dontcare load "
                                 "has to be the fragment output")
                else:
                    # A case that claims every texel states the raster's own
                    # uniform rule (`research/docs/23` §3.3, v61): every texel
                    # is the fragment output.
                    _require(all(chunk == texel for chunk in texels),
                             f"{attachment_where}: every texel of a dontcare load has to be "
                             "the fragment output")
                if wildcard_allowed is None:
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
        # The kept-frame arm's attachment carries no case-level expectation
        # either, for the reason above: its frame's only landing is the owner
        # window the landing entry fills.
        _require(expected_bytes or depth_landing
                 or any(attachment.get("store", "store") == "resident"
                        for attachment, _, _, _ in parsed),
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
        # A stage-buffer case names the rails that bind its slots
        # (`research/docs/23` §3.3, v83-v87), and the two arms name different
        # ones. A *translated* case pins two AIR modules only the Vulkan rails
        # translate, and the object rails bind its slots through the object
        # API's own entry point. A *reviewed* case pins the MSL module the two
        # native faces compile — the Swift oracle through its own render encoder
        # and the native provider through the same bytes the rail embeds — so it
        # may name those beside the Vulkan rails, now that the Apple device
        # readings flipped `supports_render_stage_buffers` (R9g/R9k, §83/§92).
        # Either way the marker stays a subset of the arm's list: a case that
        # names a rail outside it would claim a capture that rail cannot report.
        if case.get("stage_buffers"):
            if case.get("translated_stages") is not None:
                allowed = TRANSLATED_STAGE_BUFFER_RAILS
            else:
                allowed = STAGE_BUFFER_RAILS + (VULKAN_OBJECTS_RAIL,)
            _require(rails and all(rail in allowed for rail in rails),
                     f"{where}: a stage-buffer case runs on the rails that bind its slots ("
                     + ", ".join(allowed) + "), so its capture_rails has to stay inside that list")

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
            # A rule-expected attachment reports the digest of its declaring
            # allocation's whole image as well as the view's own plane, so the
            # view has to *be* the image (`research/docs/23` §73): the R5a
            # fixture declares offset 0 and a view that fills its allocation.
            if rule_expectation is not None:
                _require(offset == 0 and size == extent,
                         f"{attachment_where}: a rule-expected attachment's declaring view "
                         "has to be its whole allocation")
            # A discarded attachment's declaring view still takes part in the
            # touched count and in the declaration resolution, but its landing
            # never enters the observation surface: no writeback and no
            # allocation image are owed for it (`research/docs/23` §3.6, v19).
            if expected is None:
                # The kept-frame arm is the one arm that publishes no writeback
                # and still moves bytes: the landing entry reads the provider's
                # kept image back exactly once to deliver it into the owner's
                # window, so the frame's own allocation does leave the device
                # even though the pass that kept it reported nothing
                # (`research/docs/23` §115 之后的增量，E-TX14/R4b). The
                # copy-out count below is where that transfer is observable.
                if attachment.get("store", "store") == "resident":
                    written.add(allocation)
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
        # A borrowed owner window is one of the declaring pass's touched
        # allocations, but the provider never copies it in: its bytes are the
        # owner's own pages, imported rather than staged
        # (`research/docs/23` §90, R9i). The counter rule below is the one place
        # that difference is observable, exactly as it is on the compute path.
        touched -= set(plan[declaring][8] or ())
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
        # The stage-buffer landings (`research/docs/23` §3.3, v83-v86) follow
        # the colour, depth and stencil ones, in the order the rails report
        # them. A stage buffer's own bytes travel through the render input
        # channel — the host-visible upload a vertex stream or index buffer
        # takes — so a slot moves no device-buffer copy counter, whether its
        # bytes are trace-owned, staged or borrowed (`research/docs/23` §90):
        # the landed allocation is one more *written* allocation, and the
        # touched set stays the declaring pass's own.
        stage_writes, stage_images, stage_views, stage_written, stage_modes = (
            _stage_buffer_section(case, by_id[declaring], plan[declaring][1], where))
        writes.extend(stage_writes)
        images.update(stage_images)
        written = written | stage_written
        # The landing identities carry the same four fields an attachment's
        # does — identity, range and the byte count the writeback covers — so
        # the set and order rules below read one list for both kinds of
        # landing.
        identities.extend((allocation, view, offset, len(payload))
                          for (allocation, view, offset), payload in stage_writes)
        # The wildcard claims are stated once per observed attachment, in the
        # absolute byte offsets of its allocation, so the writeback comparison
        # and the allocation-image comparison read the same map. A free byte
        # (`wildcard_texels`) maps to `None` and is skipped; a constrained one
        # maps to its closed candidate set, and the measured byte has to be one
        # of those values (`research/docs/23` §3.3, v33/v67).
        wildcards = {}
        claims = None
        if writes:
            (allocation, view, offset), _ = writes[0]
            if wildcard_texels is not None:
                claims = {
                    offset + texel * 4 + byte: None
                    for texel in wildcard_texels
                    for byte in range(4)
                }
            elif wildcard_allowed is not None:
                claims = {
                    offset + texel * 4 + byte: tuple(value[byte] for value in values)
                    for texel, values in wildcard_allowed.items()
                    for byte in range(4)
                }
        if claims:
            wildcards[(allocation, view, offset)] = claims
        # The render sampler's own texture upload is a copy-in the count
        # contract owes (`research/docs/23` §3.3, v70): the rail uploads the
        # pass's sampled texels exactly as it uploads every allocation the
        # declaring pass touches, and the compute plan's textured cases count
        # their bindings the same way. The texture carries no allocation in the
        # policy table, so it is counted here rather than added to `touched`.
        texture_uploads = 0
        if fragment_textures is not None:
            texture_uploads = len(fragment_textures)
        # The landing-view store's own section (`research/docs/23` §115 之后的
        # 增量，E-TX13): the second declaration the frame lands in, and the bytes
        # the owner's window has to hold after the pass.
        landing = _landing_view_section(case, by_id[declaring]["buffers"], parsed, where, single)
        render_plan[case_id] = RenderExpectation(
            writes=writes,
            allocations=images,
            touched=touched,
            written=written,
            texture_uploads=texture_uploads,
            rails=frozenset(rails),
            # The single-attachment shape declares one landing, so one identity
            # is the whole review surface. A case that also stores its depth
            # attachment observes two resources, and then the identity list is
            # what has to cover both (`research/docs/23` §3.3, v43).
            attachment=identities[0] if single and len(identities) == 1 else identities,
            present=present,
            icb=icb,
            wildcards=wildcards,
            filter=requires_filter,
            stencil_filter=requires_stencil_filter,
            sample_count_gate=requires_sample_count,
            rule=rule_expectation,
            # The lease face of a stage-buffer case (`research/docs/23` §90,
            # R9i): the source arm each of its slots ran with, or `None` for a
            # case whose slots are all trace-owned.
            stage_buffer_modes=stage_modes,
            landing=landing)
    return render_plan


def validate_capture(suite, digest, report, required_backend=None):
    """Raise CaptureError for invalid captures; success alone does not claim parity."""
    plan = _suite_plan(suite)
    render_plan = _render_plan(plan, suite)
    _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
             "suite digest: expected lowercase SHA-256")
    # The depth resolve mask (`research/docs/23` §3.3, v57d) rides on top of
    # the pre-v57d capture shape: every capture the device-gated mechanism
    # compares carries it, while a capture for a suite with no gated case may
    # still omit it. The pre-v57d eight keys stay required and nothing else
    # may appear.
    _require(isinstance(report, dict), "capture: expected an object")
    required_keys = {"schema_version", "suite", "suite_sha256", "backend",
                     "allocation_observation", "device", "platform", "results"}
    _require(required_keys <= set(report),
             "capture: expected fields "
             + ", ".join(sorted(required_keys - set(report))))
    unexpected = set(report) - required_keys - {"depth_resolve_modes",
                                                "stencil_resolve_modes",
                                                "render_sample_counts"}
    _require(not unexpected,
             "capture: unexpected fields " + ", ".join(sorted(unexpected)))
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
    # The device's depth resolve capability mask (`research/docs/23` §3.3,
    # v57d). A capture that reports device-gated cases has to carry the mask —
    # otherwise a missing field would read as "the device lacks every filter"
    # and the presence rule could not be checked. A suite without gated cases
    # leaves the field optional, so the pre-v57d capture shape keeps passing.
    modes = report.get("depth_resolve_modes", 0)
    if "depth_resolve_modes" in report:
        _integer(modes, "capture.depth_resolve_modes")
    stencil_modes = report.get("stencil_resolve_modes", 0)
    if "stencil_resolve_modes" in report:
        _integer(stencil_modes, "capture.stencil_resolve_modes")
    sample_counts = report.get("render_sample_counts", 0)
    if "render_sample_counts" in report:
        _integer(sample_counts, "capture.render_sample_counts")
    if any(expectation.filter is not None for expectation in render_plan.values()):
        _require("depth_resolve_modes" in report,
                 "capture: missing depth_resolve_modes (the suite declares "
                 "device-gated cases)")
    if any(expectation.stencil_filter is not None
           for expectation in render_plan.values()):
        _require("stencil_resolve_modes" in report,
                 "capture: missing stencil_resolve_modes (the suite declares "
                 "device-gated stencil cases)")
    if any(expectation.sample_count_gate is not None
           for expectation in render_plan.values()):
        _require("render_sample_counts" in report,
                 "capture: missing render_sample_counts (the suite declares "
                 "device-gated sample-count cases)")
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
        # The present, heap, indirect and lease observations are the keys a
        # suite may declare on top of an otherwise unchanged result shape: they
        # replace no existing field and they do not relax the counter-pair rule
        # below.
        _require(set(result) - {"present", "heap", "icb", "storage_modes"}
                 - {"landing"}
                 in (base, counted, grouped),
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
            if expectation.rule is not None:
                # A rule-expected attachment reports a digest and its windows
                # instead of bytes (`research/docs/23` §73), so the byte-for-byte
                # comparison is the rule's own.
                _compare_rule_observation(result, expectation, where)
            else:
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
                # The render sampler's own texture upload rides the same
                # submission, so it is one more copy-in than the declaring
                # pass's touched allocations (`research/docs/23` §3.3, v70).
                expected_in = len(expectation.touched) + expectation.texture_uploads
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
            # The landing-view store's own reading (`research/docs/23` §115
            # 之后的增量，E-TX13): a case that declares the arm has to report the
            # owner window's bytes, and a case that does not may not report them.
            if expectation.landing is None:
                _require("landing" not in result,
                         f"{where}: the suite declares no landing observation for this case")
            else:
                _require(report["backend"] in expectation.rails,
                         f"{where}: a rail the case's marker does not name must not report the "
                         "landing observation")
                observed = result.get("landing")
                _object(observed, ("allocation", "view", "bytes_hex"), f"{where}.landing")
                allocation, view, bytes_ = expectation.landing
                _require((observed["allocation"], observed["view"]) == (allocation, view),
                         f"{where}.landing: the observation is not the declared landing view")
                landed = _hex(observed["bytes_hex"], f"{where}.landing.bytes_hex")
                _require(landed == bytes_,
                         f"{where}.landing: the owner window holds {landed} against the "
                         f"reviewed {bytes_.hex()}")
            if expectation.icb is None:
                _require("icb" not in result,
                         f"{where}: the suite declares no indirect command for this case")
            else:
                _require("icb" in result,
                         f"{where}: {report['backend']} has to report the replayed indirect "
                         "command the suite declares")
                _icb_observation(result["icb"], expectation.icb, where)
            # The stage-buffer lease face (`research/docs/23` §90, R9i): a case
            # whose slots name a source arm other than owned bytes owes the arm
            # each of them ran with, exactly as a compute case's lease section
            # does. The Swift reference oracle is not a provider — it places the
            # same bytes in its own buffer — so it reports no arms, and a case
            # that names one is not owed by the rails that cannot.
            if expectation.stage_buffer_modes is None:
                _require("storage_modes" not in result,
                         f"{where}: the suite declares no lease arm for this case")
            elif provider_backend:
                _require("storage_modes" in result,
                         f"{where}: a case whose stage buffers declare a lease arm has to "
                         "report the source arm each of them ran with")
                _storage_modes_observation(result["storage_modes"],
                                           expectation.stage_buffer_modes, where)
            else:
                _require("storage_modes" not in result,
                         f"{where}: {report['backend']} is not a provider, so it cannot report "
                         "a source arm it did not execute")
        else:
            (expected_writes, expected_allocations, texture_count, group_expectations,
             heap, icb, case_rails, declared_modes, borrowed_allocations) = plan[case_id]
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
            # The lease face (`research/docs/23` §90, R9i): a case whose suite
            # declares a source arm other than owned bytes owes the arm each
            # view ran with, and the borrowed arm's bytes are imported rather
            # than copied in — so the provider's own copy-in count drops by one
            # per borrowed allocation. The Swift reference oracle is not a
            # provider: it places the same bytes in its own buffer and reports
            # neither the arms nor the counters.
            if declared_modes is None:
                _require("storage_modes" not in result,
                         f"{where}: the suite declares no lease arm for this case")
            else:
                if provider_backend:
                    _require("storage_modes" in result,
                             f"{where}: a case whose suite declares a lease arm has to "
                             "report the source arm each view ran with")
                    _storage_modes_observation(result["storage_modes"], declared_modes, where)
                else:
                    _require("storage_modes" not in result,
                             f"{where}: {report['backend']} is not a provider, so it cannot "
                             "report a source arm it did not execute")
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
            # A borrowed view's bytes are the owner's mapping, imported rather
            # than copied in (`research/docs/23` §90, R9i), so its allocation
            # is the one touched allocation the provider's copy-in count does
            # not carry. The bytes still land byte for byte; only the arm they
            # arrived through changes the counter.
            expected_in = (len(expected_allocations) + texture_count
                           - len(borrowed_allocations or ()))
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
    # applies. A device-gated render case (`research/docs/23` §3.3,
    # v57d/v60/v61) adds the device half: even a rail its marker names owes the
    # case only when the capture's mask carries the filter's bit, or its sample
    # mask carries the count the case requires.
    required = {case_id for case_id, expectation in plan.items()
                if expectation[6] is None or report["backend"] in expectation[6]}
    required |= {case_id for case_id, expectation in render_plan.items()
                 if report["backend"] in expectation.rails
                 and (expectation.filter is None
                      or modes & DEPTH_RESOLVE_FILTER_BITS[expectation.filter])
                 and (expectation.stencil_filter is None
                      or stencil_modes
                      & STENCIL_RESOLVE_FILTER_BITS[expectation.stencil_filter])
                 and (expectation.sample_count_gate is None
                      or sample_counts
                      & SAMPLE_COUNT_BITS[expectation.sample_count_gate])}
    missing = required - seen
    _require(not missing, f"capture: missing cases {sorted(missing)}")
    for case_id in sorted(set(render_plan) - required):
        expectation = render_plan[case_id]
        if expectation.filter is not None and report["backend"] in expectation.rails:
            message = (f"case {case_id}: {report['backend']} lacks the "
                       f"{expectation.filter} depth resolve filter the case requires")
        elif (expectation.stencil_filter is not None
              and report["backend"] in expectation.rails):
            message = (f"case {case_id}: {report['backend']} lacks the "
                       f"{expectation.stencil_filter} stencil resolve filter the case "
                       "requires")
        elif (expectation.sample_count_gate is not None
              and report["backend"] in expectation.rails):
            message = (f"case {case_id}: {report['backend']} lacks the "
                       f"{expectation.sample_count_gate}-sample raster the case requires")
        else:
            message = (f"case {case_id}: {report['backend']} is not a rail this render "
                       f"case runs on")
        _require(case_id not in seen, message)
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
