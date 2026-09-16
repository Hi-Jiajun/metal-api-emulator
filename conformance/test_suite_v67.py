"""Constrained wildcard texels (`research/docs/23` §3.3, v67).

These are comparator and schema checks, not GPU execution evidence. They pin
the channel that admits a texel whose bytes the case does not pin but does
*bound*: beside a multisample raster, the resolve of a texel whose covered
samples the draw wrote and whose other samples kept the pass's reference
colour can only be one of the exact mixes of the fragment output and that
colour, and the case states that closed set in advance. The unconstrained list
of v33 is the other half of the same idea — bytes with no claim at all — and
the two cannot be stated together.

The probe of the same shape (`evidence/msaa-partial-probe-2026-09-17`) shows
why the channel stops at the arithmetic: the drivers measured there land bytes
outside every mix of the fixture's own colours, so the `dontcare` shape stays
expressible and unit-tested rather than reviewed into the suite's own
expectation.
"""

import copy
import hashlib
import json
import unittest

import compare
from test_compare import synthetic_report


SUITE_NAME = "compute-buffer-v67-constrained-wildcard"
DECLARING_ID = "render_declaring_quad_extent"
CASE_ID = "constrained_wildcard_scissor_4x4"
ATTACHMENT = (900, 910, 0)
OUTPUT = "4080c0ff"
# The reference colour is the one the reviewed multisample fixture uses
# (`msaa_edge_4x4`): every k-of-N mix it makes with the fragment output below
# is exactly representable, so the same case states its closed set beside a
# single-sample and a multisample raster.
REFERENCE = "22446689"
# The observed half: the scissor covers columns 0 and 1, so those texels are
# the fragment output and nothing else — they are simply not named, the rule
# every case before v33 states.
OBSERVED_TEXELS = (0, 1, 4, 5, 8, 9, 12, 13)
# The claimed half: nothing rasterises there, so the byte is one of the two
# values the case's own colours produce.
CLAIMED_TEXELS = (2, 3, 6, 7, 10, 11, 14, 15)
CANDIDATES = (REFERENCE, OUTPUT)
# The 2-of-4 mix of the two colours, the only other exact mix the reviewed
# colours admit (`316293c4`).
MIXED = "316293c4"


def declaring_case():
    return {
        "id": DECLARING_ID,
        "entry": "copy_word",
        "grid": [1, 1, 1],
        "local": [1, 1, 1],
        "air": {"path": "air/copy_word.ll", "sha256": "00" * 32},
        "metal": {"path": "metal/copy_word.metal", "sha256": "11" * 32},
        "buffers": [
            {"binding": 0, "allocation": 900, "view": 910, "offset": 0, "length": 64,
             "allocation_size": 64, "access": "read", "initial_hex": "cd" * 64},
            {"binding": 1, "allocation": 920, "view": 930, "offset": 4, "length": 4,
             "allocation_size": 16, "access": "write", "initial_hex": "fefefefe"},
        ],
        "expected_writebacks": [
            {"allocation": 920, "view": 930, "offset": 4, "bytes_hex": "cdcdcdcd"},
        ],
    }


def render_case():
    return {
        "id": CASE_ID,
        "declaring_case": DECLARING_ID,
        "vertex_entry": "render_quad_vertex",
        "fragment_entry": "render_solid_rgba8",
        "metal": {"path": "shaders/quad_indexed_2x2.metal", "sha256": "22" * 32},
        "vertex_layout": {"buffers": [{"stride": 8, "attributes": [
            {"location": 0, "offset": 0, "format": "float32x2"}]}]},
        "vertex_buffers": [{
            "allocation": 1000, "view": 1010, "offset": 0, "length": 32,
            "initial_hex": "000080bf000080bf0000803f000080bf"
                            "000080bf0000803f0000803f0000803f",
        }],
        "indices": {
            "allocation": 1020, "view": 1030, "offset": 0, "length": 12,
            "initial_hex": "000001000200010003000200", "format": "uint16",
        },
        "vertices": 6,
        "viewport": [0, 0, 4, 4],
        "scissor": [0, 0, 2, 4],
        "attachment": {
            "allocation": 900, "view": 910, "format": "rgba8_unorm",
            "width": 4, "height": 4, "load": "dontcare",
            "clear_hex": REFERENCE, "store": "store",
        },
        "expected_hex": OUTPUT * 16,
        "wildcard_allowed_texels": [
            {"index": texel, "allowed": list(CANDIDATES)} for texel in CLAIMED_TEXELS
        ],
        "capture_rails": ["vulkan"],
    }


def suite_document():
    return {
        "schema_version": 1,
        "suite": SUITE_NAME,
        "guard_byte": 0xAB,
        "cases": [declaring_case()],
        "render_cases": [render_case()],
    }


def digest(suite):
    return hashlib.sha256(json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()


def report_for(suite, observed_texels, backend="vulkan"):
    """A synthetic capture whose measured bytes are `observed_texels`."""
    digest_value = digest(suite)
    report = synthetic_report(suite, digest_value, backend)
    for result in report["results"]:
        result["copy_in"] = 2
        result["copy_out"] = 1
    report["results"].append({
        "id": CASE_ID,
        "completion": "CompletedVisible",
        "writebacks": [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                        "offset": ATTACHMENT[2], "bytes_hex": observed_texels}],
        "allocations": [{"allocation": ATTACHMENT[0], "bytes_hex": observed_texels}],
    })
    return report


class ConstrainedWildcardTests(unittest.TestCase):
    def setUp(self):
        self.suite = suite_document()
        # The synthetic report is the observed half plus the first candidate of
        # every claimed texel: a capture that agrees with the case.
        measured = "".join(
            OUTPUT if texel in OBSERVED_TEXELS else CANDIDATES[0] for texel in range(16))
        self.report = report_for(self.suite, measured)

    def test_the_case_plans_a_constrained_claim(self):
        plan = compare._render_plan(compare._suite_plan(self.suite), self.suite)
        expectation = plan[CASE_ID]
        claims = expectation.wildcards[(ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2])]
        # The claimed texels carry their closed set, one set per byte; the
        # observed half carries no claim at all and stays an exact comparison.
        for texel in OBSERVED_TEXELS:
            for byte in range(4):
                self.assertNotIn(texel * 4 + byte, claims)
        for texel in CLAIMED_TEXELS:
            for byte in range(4):
                self.assertEqual(sorted(claims[texel * 4 + byte]),
                                 sorted(bytes.fromhex(value)[byte] for value in CANDIDATES))

    def test_every_declared_candidate_is_accepted(self):
        for candidate in CANDIDATES:
            measured = "".join(
                OUTPUT if texel in OBSERVED_TEXELS else candidate for texel in range(16))
            with self.subTest(candidate=candidate):
                compare.validate_capture(self.suite, digest(self.suite),
                                         report_for(self.suite, measured), "vulkan")

    def test_a_value_outside_the_set_is_refused(self):
        measured = OUTPUT * 16
        measured = measured[:2 * 8] + "ff0000ff" + measured[3 * 8:]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "which is none of 0x22, 0x40"):
            compare.validate_capture(self.suite, digest(self.suite),
                                     report_for(self.suite, measured), "vulkan")

    def test_a_claimed_texel_may_not_be_left_unclaimed(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["wildcard_texels"] = list(CLAIMED_TEXELS)
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the two wildcard channels are mutually exclusive"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_a_value_that_is_not_a_mix_is_refused(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["wildcard_allowed_texels"][0]["allowed"] = [
            REFERENCE, "00ff00ff"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "which is not an exact mix of the fragment output"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_a_missing_mix_is_refused(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["wildcard_allowed_texels"][0]["allowed"] = [REFERENCE]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "leaves out the resolve"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_a_reference_colour_equal_to_the_output_is_refused(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["attachment"]["clear_hex"] = OUTPUT
        with self.assertRaisesRegex(compare.CaptureError,
                                    "needs a reference colour other than the fragment output"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_an_allowed_value_that_is_not_lowercase_bytes_is_refused(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["wildcard_allowed_texels"][0]["allowed"] = [
            "00FF00FF", OUTPUT]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "has to be lowercase bytes"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_a_duplicate_texel_is_refused(self):
        broken = copy.deepcopy(self.suite)
        entry = copy.deepcopy(broken["render_cases"][0]["wildcard_allowed_texels"][0])
        broken["render_cases"][0]["wildcard_allowed_texels"].append(entry)
        with self.assertRaisesRegex(compare.CaptureError, "duplicate wildcard texel"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_a_texel_outside_the_attachment_is_refused(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["wildcard_allowed_texels"].append(
            {"index": 16, "allowed": list(CANDIDATES)})
        with self.assertRaisesRegex(compare.CaptureError,
                                    "wildcard texel 16 is outside the attachment"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_a_cleared_raster_may_not_claim_an_allowed_set(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["attachment"]["load"] = "clear"
        broken["render_cases"][0]["multisample"] = {"sample_count": 4}
        with self.assertRaisesRegex(compare.CaptureError,
                                    "a cleared attachment has no unclaimed texel"):
            compare._render_plan(compare._suite_plan(broken), broken)

    def test_a_multisample_case_states_mix_candidates(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][0]["multisample"] = {"sample_count": 4}
        # Beside a four-sample raster the same two colours admit the
        # 2-of-4 mix, so the case's declared set grows with the shape.
        for entry in broken["render_cases"][0]["wildcard_allowed_texels"]:
            entry["allowed"] = [REFERENCE, MIXED, OUTPUT]
        plan = compare._render_plan(compare._suite_plan(broken), broken)
        claims = plan[CASE_ID].wildcards[(ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2])]
        candidates = compare._mix_candidates(bytes.fromhex(OUTPUT),
                                             bytes.fromhex(REFERENCE), 4)
        for texel in CLAIMED_TEXELS:
            for byte in range(4):
                self.assertEqual(sorted(claims[texel * 4 + byte]),
                                 sorted(value[byte] for value in candidates))


if __name__ == "__main__":
    unittest.main()
