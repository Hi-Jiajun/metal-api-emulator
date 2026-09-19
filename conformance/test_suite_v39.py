"""The non-indexed draw arm (`research/docs/23` §3.3, v39).

These are comparator and schema checks, not GPU execution evidence. What they
pin is the render narrow class's widened coverage: the class's one draw may be
drawn through its index buffer *or*, since R39 (`07b68ca`), without one — the
draw then names its vertices `0..vertices` and every per-vertex stream has to
cover the whole `vertices * stride` span the two rails prove.

`suite-v39.json` carries that arm four ways, so the claims are falsifiable
inside one suite rather than across revisions:

- `offscreen_triangle_clear_2x2` is the milestone's `vertex_id` triangle: a
  non-indexed draw that carries no stream at all, which the widened coverage
  now admits;
- `quad_indexed_clear_2x2` is the reviewed indexed quad (`research/docs/23`
  §3.3, v16) carried unchanged, so the index arm stays observable beside it;
- `nonindexed_quad_clear_2x2` is the *same* six vertices the index buffer
  names, written out as six records and drawn without an index buffer, so its
  frame MUST be the indexed case's frame byte for byte — the parity the class
  claims;
- `nonindexed_triangle_clear_2x2` draws one triangle of the same stream, which
  covers one of the four texels. It is the class's *neighbour*: it declares the
  partial coverage it resolves and `conformance/narrow-class.json` refuses it
  by that rule, but it still owes the five rails — a rail that ignored the
  vertex bytes could not land the frame it states.

The streams of the two non-indexed cases are *derivations* of the indexed one,
not copies: this module expands the index buffer itself and compares.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare


CONFORMANCE = Path(__file__).resolve().parent
V39_PATH = CONFORMANCE / "suite-v39.json"

DECLARING_ID = "render_declaring_copy_word"
TRIANGLE_ID = "offscreen_triangle_clear_2x2"
INDEXED_ID = "quad_indexed_clear_2x2"
NONINDEXED_ID = "nonindexed_quad_clear_2x2"
PARTIAL_ID = "nonindexed_triangle_clear_2x2"
RENDER_IDS = (TRIANGLE_ID, INDEXED_ID, NONINDEXED_ID, PARTIAL_ID)

TEXELS = "4080c0ff" * 4
# The neighbour's frame: the fragment output on the texel its triangle covers
# and the clear on the other three. The first texel of a partial expectation is
# the comparator's own spelling of "the fragment output", which is why the
# triangle sits over the frame's first texel rather than any other.
PARTIAL_TEXELS = "4080c0fffefefefefefefefefefefefe"

ATTACHMENT = (900, 910, 0, 16)
QUAD_VIEW = (940, 950, 0, 32)
INDEX_VIEW = (960, 970, 0, 12)
NONINDEXED_VIEW = (1040, 1050, 0, 48)
TRIANGLE_VIEW = (1060, 1070, 0, 24)
CLEAR_HEX = "fefefefe"

ALL_RAILS = ("native-metal", "vulkan", "native-metal-provider", "vulkan-objects",
             "native-metal-provider-objects")
V39_REPORTING_RAILS = ALL_RAILS


def expanded(stream_hex, index_hex, width):
    """The index buffer's own expansion: one record per index value."""
    records = bytes.fromhex(stream_hex)
    indices = bytes.fromhex(index_hex)
    out = bytearray()
    for position in range(0, len(indices), width):
        index = int.from_bytes(indices[position:position + width], "little")
        out += records[index * 8:(index + 1) * 8]
    return bytes(out)


class NonIndexedDrawTests(unittest.TestCase):
    def setUp(self):
        self.raw = V39_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def case(self, case_id, suite=None):
        suite = suite or self.suite
        for case in suite["render_cases"]:
            if case["id"] == case_id:
                return case
        self.fail(f"the suite carries no {case_id} case")

    def plan(self, case_id, suite=None):
        suite = suite or self.suite
        return compare._render_plan(compare._suite_plan(suite), suite)[case_id]

    def refused(self, case_id, mutate, message):
        suite = copy.deepcopy(self.suite)
        mutate(self.case(case_id, suite))
        with self.assertRaisesRegex(compare.CaptureError, message):
            compare._render_plan(compare._suite_plan(suite), suite)

    def test_v39_pins_the_suite_and_its_sources(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v39")
        self.assertEqual(self.suite["guard_byte"], 166)
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], list(RENDER_IDS))
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["entry"], "copy_word")
        for kind in ("air", "metal"):
            source = CONFORMANCE / declaring[kind]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             declaring[kind]["sha256"], kind)
        for case_id in RENDER_IDS:
            case = self.case(case_id)
            self.assertEqual(case["declaring_case"], DECLARING_ID)
            source = CONFORMANCE / case["metal"]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             case["metal"]["sha256"], case_id)

    def test_v39_pins_the_non_indexed_streams_as_the_index_buffer_expands(self):
        # The two index-carrying cases share the reviewed pair and the reviewed
        # layout; the neighbour draws one of the same corners' triangles.
        indexed = self.case(INDEXED_ID)
        nonindexed = self.case(NONINDEXED_ID)
        partial = self.case(PARTIAL_ID)
        self.assertEqual((indexed["vertex_entry"], indexed["fragment_entry"]),
                         ("render_quad_vertex", "render_solid_rgba8"))
        self.assertEqual((nonindexed["vertex_entry"], nonindexed["fragment_entry"]),
                         (indexed["vertex_entry"], indexed["fragment_entry"]))
        self.assertEqual((partial["vertex_entry"], partial["fragment_entry"]),
                         (indexed["vertex_entry"], indexed["fragment_entry"]))
        self.assertEqual(nonindexed["vertex_layout"], indexed["vertex_layout"])
        self.assertEqual(partial["vertex_layout"], indexed["vertex_layout"])
        self.assertNotIn("indices", nonindexed)
        self.assertNotIn("indices", partial)
        stream = indexed["vertex_buffers"][0]["initial_hex"]
        index_bytes = indexed["indices"]["initial_hex"]
        width = 2
        self.assertEqual(nonindexed["vertex_buffers"][0]["initial_hex"],
                         expanded(stream, index_bytes, width).hex())
        self.assertEqual(nonindexed["vertices"], len(bytes.fromhex(index_bytes)) // width)
        # The neighbour states three records of its own: the same reviewed
        # layout, a triangle that sits inside one texel's cell rather than the
        # quad's four corners — the byte change the frame has to follow.
        self.assertEqual(len(bytes.fromhex(partial["vertex_buffers"][0]["initial_hex"])),
                         3 * 8)
        self.assertNotEqual(partial["vertex_buffers"][0]["initial_hex"],
                            expanded(stream, index_bytes[:6], width).hex())
        self.assertEqual(partial["vertices"], 3)
        self.assertEqual(partial["coverage"], "partial")
        for case in (indexed, nonindexed, partial):
            self.assertEqual(sorted(case["capture_rails"]), sorted(ALL_RAILS))

    def test_v39_plan_carries_the_non_indexed_draw_beside_the_index_arm(self):
        # Every case in this suite lands in the same attachment, and the two
        # index-bearing shapes owe the same observable surface: one writeback
        # under the attachment's identity and one allocation image. The streams
        # are inputs, so they are not part of the observation.
        for case_id, texels in ((TRIANGLE_ID, TEXELS), (INDEXED_ID, TEXELS),
                                (NONINDEXED_ID, TEXELS), (PARTIAL_ID, PARTIAL_TEXELS)):
            with self.subTest(case=case_id):
                expectation = self.plan(case_id)
                self.assertEqual(expectation.writes,
                                 [(ATTACHMENT[:3], bytes.fromhex(texels))])
                self.assertEqual(expectation.allocations, {ATTACHMENT[0]: bytes.fromhex(texels)})
                self.assertEqual(expectation.touched, {ATTACHMENT[0], 920})
                self.assertEqual(expectation.written, {ATTACHMENT[0], 920})
                self.assertEqual(expectation.rails, set(V39_REPORTING_RAILS))

    def test_v39_the_non_indexed_draw_lands_the_indexed_frame(self):
        # The parity the class claims: the same six vertices, one arm through
        # the index buffer and one without it, land the same bytes.
        indexed = self.plan(INDEXED_ID)
        nonindexed = self.plan(NONINDEXED_ID)
        self.assertEqual(indexed.writes, nonindexed.writes)
        self.assertEqual(indexed.allocations, nonindexed.allocations)
        self.assertEqual(self.case(NONINDEXED_ID)["vertices"],
                         self.case(INDEXED_ID)["vertices"])

    def test_v39_the_partial_neighbour_moves_the_frame(self):
        # One triangle of the same stream covers one texel instead of four, so
        # the frame has to move: that is what makes the vertex bytes — not the
        # draw's shape — the thing the attachment reads.
        quad = self.plan(NONINDEXED_ID)
        partial = self.plan(PARTIAL_ID)
        self.assertNotEqual(quad.writes, partial.writes)
        texels = bytes.fromhex(PARTIAL_TEXELS)
        clear = bytes.fromhex(CLEAR_HEX)
        fragment = texels[:4]
        self.assertEqual(sum(texel == fragment for texel in
                             (texels[offset:offset + 4] for offset in range(0, 16, 4))), 1)
        self.assertEqual(sum(texel == clear for texel in
                             (texels[offset:offset + 4] for offset in range(0, 16, 4))), 3)

    def test_v39_refuses_a_non_indexed_draw_that_cannot_cover_itself(self):
        # The widening is exactly as wide as the two rails' own footprint
        # proof: fewer than three vertices rasterizes nothing, a stream shorter
        # than `vertices * stride` does not cover the draw, and a base vertex
        # has no index value to be added to.
        def drop_a_vertex(case):
            case["vertices"] = 2

        def shorten_the_stream(case):
            case["vertex_buffers"][0]["length"] = 40
            case["vertex_buffers"][0]["initial_hex"] = "00" * 40

        def add_a_base_vertex(case):
            case["base_vertex"] = 1

        for mutate, message in (
            (drop_a_vertex, "at least one triangle"),
            (shorten_the_stream, "shorter than the reviewed non-indexed draw reads"),
            # A base vertex is refused even earlier: the class carries that
            # shape as its own reviewed fixture (`research/docs/23` §3.3, v34),
            # so the refusal names that stream rather than the plain one.
            (add_a_base_vertex, "the reviewed base-vertex stream is the degenerate centre"),
        ):
            with self.subTest(message=message):
                self.refused(NONINDEXED_ID, mutate, message)

    def test_v39_refuses_a_resourceless_draw_that_is_not_the_triangle(self):
        # The `vertex_id` arm carries no stream to cover, so its count is
        # bounded by the reviewed triangle's three — and, since 2026-09-19
        # (census v45's `vertex_span` bucket), only *below*: a count from three
        # up is the same shape with more vertices, and the five-rail marker of
        # this suite cannot claim the widened arm, whose module the two native
        # faces do not carry.
        self.refused(TRIANGLE_ID, lambda case: case.update(vertices=2),
                     "rasterizes no triangle")
        self.refused(TRIANGLE_ID, lambda case: case.update(vertices=6),
                     "capture_rails has to stay inside that list")


if __name__ == "__main__":
    unittest.main()
