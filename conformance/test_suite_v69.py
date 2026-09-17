"""Cleared-raster edge coverage (`research/docs/23` §3.3, v69).

These are comparator and schema checks, not GPU execution evidence. They pin
the v69 increment: the v51 edge geometry over the 2x and 8x rasters, which the
v61 pair could only state as full coverage because the sample positions of
those rasters are not documented. A cleared attachment gives the resolve a
reference colour the fixture owns, so the column the right edge crosses is
stated as the *closed set* of exact k-of-`sample_count` mixes of the fragment
output and that clear — the constrained wildcard channel v67 introduced beside
`dontcare` — while the two fully covered columns stay pinned to the fragment
output and the uncovered one to the clear colour. Both measured devices
(Lavapipe's 8x and the RTX 5060's 2x and 8x, `evidence/conformance-v69-*`)
land the crossed column on the 4x fixture's own `2`-of-`4` mix, which is what
the reviewed expectations carry there; the set is what is admitted, so a
device whose raster covers a different exactly representable count is not
refused.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
import test_suite_v28 as v28


CONFORMANCE = Path(__file__).resolve().parent
V28_PATH = CONFORMANCE / "suite-v28.json"

CASE_2X = v28.MSAA_EDGE_2X_ID
CASE_8X = v28.MSAA_EDGE_8X_ID
ATTACHMENT = v28.ATTACHMENT
OUTPUT = v28.OUTPUT
CLEAR = v28.MSAA_CLEAR
MIXED = v28.MSAA_MIXED
CLAIMED_TEXELS = v28.MSAA_EDGE_TEXELS
CANDIDATES = v28.MSAA_EDGE_CANDIDATES
# The 2-of-8 mix of the fixture's colours rounded per channel: two samples
# carry the output and six the clear, and R, B and A do not divide exactly, so
# no reviewed resolve may land on this byte (`_resolve_texel` answers `None`).
ROUNDED_TWO_OF_EIGHT = "29537ca7"
# The sample mask both edge rasters need, the way the v61 gates are stated.
EDGE_MASK = (1 << 1) | (1 << 3)


def expected_bytes(texels):
    """The reviewed expectation with `texels` carrying their own byte."""
    base = [OUTPUT if (index % 4) < 2
            else (MIXED if (index % 4) == 2 else CLEAR)
            for index in range(16)]
    return "".join(texels.get(index, byte) for index, byte in enumerate(base))


def full_capture(suite, overrides=None):
    """A vulkan capture that owes every case of the reviewed suite.

    The capture's sample mask carries both edge rasters, so the gated cases are
    owed and the report has to carry them; `overrides` replaces a result's
    measured bytes by case id.
    """
    digest = hashlib.sha256(
        json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
    report = v28.counted_declaring(suite, digest, "vulkan",
                                   render_sample_counts=EDGE_MASK)
    builders = (v28.render_result, v28.instanced_result, v28.wildcard_result,
                v28.base_vertex_result, v28.depth_result, v28.depth_store_result,
                v28.depth_only_result, v28.depth_no_colour_result,
                v28.stencil_result, v28.stencil_store_result,
                v28.alignment_result, v28.cull_result, v28.blend_result,
                v28.msaa_result, v28.msaa_depth_result, v28.msaa_stencil_result,
                v28.msaa_depth_resolve_result,
                v28.msaa_stencil_resolve_sample0_result,
                v28.msaa_uniform_2x_result, v28.msaa_uniform_8x_result,
                v28.msaa_edge_2x_result, v28.msaa_edge_8x_result,
                v28.sampled_result,
                v28.full_cover_16x16_result, v28.sampled_64_result,
                v28.msaa_ds_result)
    for builder in builders:
        result = builder(True)
        if overrides and result["id"] in overrides:
            bytes_hex = overrides[result["id"]]
            result["writebacks"] = [{"allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                                     "offset": ATTACHMENT[2], "bytes_hex": bytes_hex}]
            result["allocations"] = [{"allocation": ATTACHMENT[0],
                                      "bytes_hex": bytes_hex}]
        report["results"].append(result)
    return report, digest


class ClearedEdgeClaimTests(unittest.TestCase):
    def setUp(self):
        self.suite = json.loads(V28_PATH.read_bytes())

    def plan(self, suite=None):
        suite = suite or self.suite
        return compare._render_plan(compare._suite_plan(suite), suite)

    def test_the_crossed_column_states_the_closed_mix_set(self):
        plan = self.plan()
        for case_id, samples in ((CASE_2X, 2), (CASE_8X, 8)):
            with self.subTest(case=case_id):
                expectation = plan[case_id]
                claims = expectation.wildcards[(ATTACHMENT[0], ATTACHMENT[1],
                                                ATTACHMENT[2])]
                candidates = compare._mix_candidates(bytes.fromhex(OUTPUT),
                                                     bytes.fromhex(CLEAR), samples)
                # The closed set is the case's own arithmetic: only k = 0, N/2
                # and N divide exactly for these two colours, so the set is
                # the two endpoints and the `2`-of-`4` mix the 4x fixture
                # pins.
                self.assertEqual(sorted(candidates), sorted(bytes.fromhex(value)
                                                            for value in CANDIDATES))
                for texel in CLAIMED_TEXELS:
                    for byte in range(4):
                        self.assertEqual(sorted(claims[texel * 4 + byte]),
                                         sorted(bytes.fromhex(value)[byte]
                                                for value in CANDIDATES))
                for texel in (0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15):
                    for byte in range(4):
                        self.assertNotIn(texel * 4 + byte, claims)

    def test_every_candidate_of_the_set_is_accepted(self):
        for case_id in (CASE_2X, CASE_8X):
            for candidate in CANDIDATES:
                with self.subTest(case=case_id, candidate=candidate):
                    measured = expected_bytes(
                        {texel: candidate for texel in CLAIMED_TEXELS})
                    report, digest = full_capture(self.suite, {case_id: measured})
                    compare.validate_capture(self.suite, digest, report, "vulkan")

    def test_a_value_outside_the_set_is_refused(self):
        for case_id in (CASE_2X, CASE_8X):
            with self.subTest(case=case_id):
                measured = expected_bytes({CLAIMED_TEXELS[0]: "ff0000ff"})
                report, digest = full_capture(self.suite, {case_id: measured})
                with self.assertRaisesRegex(
                        compare.CaptureError,
                        "which is none of 0x22, 0x31, 0x40"):
                    compare.validate_capture(self.suite, digest, report, "vulkan")

    def test_a_pinned_column_that_is_not_a_resolve_is_refused(self):
        # The columns the case does pin stay under the resolve rule: a byte
        # that is not the resolve of any coverage of the raster is refused
        # (`research/docs/23` §3.3, v69).
        broken = copy.deepcopy(self.suite)
        case = broken["render_cases"][v28.MSAA_EDGE_8X_INDEX]
        # The uncovered column is pinned to the clear colour; a byte that is
        # the resolve of no coverage is refused there. (The first texel is the
        # fragment output the rule below compares against, so it is the one
        # texel this mutation cannot reach.)
        case["expected_hex"] = expected_bytes({3: "ff0000ff"})
        with self.assertRaisesRegex(
                compare.CaptureError,
                "texel 3 is not the resolve of any coverage of the 8-sample raster"):
            self.plan(broken)

    def test_the_pinned_columns_show_both_extremes(self):
        # A case that leaves texels to their allowed set still has to pin both
        # a fully covered and an uncovered texel, or the claim would not say
        # which raster it resolves (`research/docs/23` §3.3, v69).
        broken = copy.deepcopy(self.suite)
        case = broken["render_cases"][v28.MSAA_EDGE_2X_INDEX]
        # Claim the first column too and leave every pinned texel at the clear
        # colour: the case then pins k = 0 and nothing else.
        case["wildcard_allowed_texels"].append({"index": 0, "allowed": CANDIDATES})
        case["expected_hex"] = OUTPUT + CLEAR * 15
        with self.assertRaisesRegex(
                compare.CaptureError,
                "needs both a fully covered and an uncovered texel"):
            self.plan(broken)

    def test_a_capture_that_breaks_a_pinned_column_is_refused(self):
        measured = expected_bytes({0: CLEAR})
        report, digest = full_capture(self.suite, {CASE_2X: measured})
        with self.assertRaisesRegex(compare.CaptureError,
                                    "first differing byte at offset 0"):
            compare.validate_capture(self.suite, digest, report, "vulkan")

    def test_a_set_that_leaves_out_a_mix_is_refused(self):
        for case_id, index in ((CASE_2X, v28.MSAA_EDGE_2X_INDEX),
                               (CASE_8X, v28.MSAA_EDGE_8X_INDEX)):
            with self.subTest(case=case_id):
                broken = copy.deepcopy(self.suite)
                case = broken["render_cases"][index]
                case["wildcard_allowed_texels"][0]["allowed"] = [CLEAR, OUTPUT]
                with self.assertRaisesRegex(compare.CaptureError,
                                            "leaves out the resolve"):
                    self.plan(broken)

    def test_a_set_that_names_a_value_that_is_not_a_mix_is_refused(self):
        for case_id, index, value in (
                (CASE_2X, v28.MSAA_EDGE_2X_INDEX, "00ff00ff"),
                # The 8x raster's 2-of-8 mix does not divide exactly, so its
                # rounded spelling is not a value a reviewed rail may produce
                # (`research/docs/23` §3.3, v67/v69).
                (CASE_8X, v28.MSAA_EDGE_8X_INDEX, ROUNDED_TWO_OF_EIGHT)):
            with self.subTest(case=case_id, value=value):
                broken = copy.deepcopy(self.suite)
                case = broken["render_cases"][index]
                case["wildcard_allowed_texels"][0]["allowed"] = [CLEAR, value, OUTPUT]
                with self.assertRaisesRegex(
                        compare.CaptureError,
                        f"states the allowed value 0x{value}, which is not an exact mix"):
                    self.plan(broken)

    def test_the_two_wildcard_channels_stay_mutually_exclusive(self):
        broken = copy.deepcopy(self.suite)
        broken["render_cases"][v28.MSAA_EDGE_2X_INDEX]["wildcard_texels"] = CLAIMED_TEXELS
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the two wildcard channels are mutually exclusive"):
            self.plan(broken)

    def test_a_cleared_single_sample_attachment_may_not_claim_a_set(self):
        # The channel is the multisample raster's own claim; a cleared
        # single-sample attachment keeps the v67 refusal
        # (`research/docs/23` §3.3, v69).
        broken = copy.deepcopy(self.suite)
        case = broken["render_cases"][v28.MSAA_EDGE_2X_INDEX]
        del case["multisample"]
        del case["requires_sample_count"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "a cleared attachment has no unclaimed texel"):
            self.plan(broken)

    def test_a_cleared_multisample_raster_needs_the_partial_coverage_claim(self):
        broken = copy.deepcopy(self.suite)
        case = broken["render_cases"][v28.MSAA_EDGE_2X_INDEX]
        del case["coverage"]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a cleared multisample raster states the partial coverage its allowed set "
                "resolves"):
            self.plan(broken)

    def test_a_loaded_attachment_may_not_claim_a_set(self):
        # A loaded attachment hands the pass its own bytes, so nothing is
        # unclaimed beside it (`research/docs/23` §3.3, v69).
        broken = copy.deepcopy(self.suite)
        case = broken["render_cases"][v28.MSAA_EDGE_2X_INDEX]
        del case["multisample"]
        del case["requires_sample_count"]
        case["attachment"]["load"] = "load"
        case["attachment"]["initial_hex"] = "cd" * 64
        with self.assertRaisesRegex(compare.CaptureError,
                                    "a loaded attachment has no unclaimed texel"):
            self.plan(broken)

    def test_the_dontcare_shape_still_admits_the_channel(self):
        # The v67 shape is untouched by v69: beside a `dontcare` load every
        # texel the case does not name stays the fragment output, and the
        # named ones keep their closed set (`research/docs/23` §3.3, v67).
        broken = copy.deepcopy(self.suite)
        case = broken["render_cases"][v28.MSAA_EDGE_2X_INDEX]
        del case["coverage"]
        case["attachment"]["load"] = "dontcare"
        case["expected_hex"] = OUTPUT * 16
        plan = self.plan(broken)
        claims = plan[CASE_2X].wildcards[(ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2])]
        for texel in CLAIMED_TEXELS:
            for byte in range(4):
                self.assertEqual(sorted(claims[texel * 4 + byte]),
                                 sorted(bytes.fromhex(value)[byte]
                                        for value in CANDIDATES))

    def test_the_device_gate_has_to_name_the_raster(self):
        broken = copy.deepcopy(self.suite)
        case = broken["render_cases"][v28.MSAA_EDGE_2X_INDEX]
        case["requires_sample_count"] = 8
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the device gate has to name the sample count"):
            self.plan(broken)


if __name__ == "__main__":
    unittest.main()
