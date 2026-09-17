"""Seeded multisampled loads (`research/docs/23` §82, v82).

These are comparator and schema checks, not GPU execution evidence. They pin
the v82 increment: a multisampled colour attachment a pass opens with
`LoadOp::Load`. A multisampled image cannot be uploaded into — every transfer
command that could define its texels is single-sample at one end or both — so
the reviewed route is a *seed pass*: the seam opens the same image from a clear
whose value is the declared window (one repeated texel, because a clear value is
one colour for the whole attachment), stores it, and lets the measured pass
load every sample it just defined. The fixtures therefore state the v69 pair's
own bytes: a seeded raster whose window is the colour the cleared pair opens
with must resolve to the same picture, and that equality is what the trio
pins. What the shape adds beyond v69 is the *declaration*: a rail that dropped
the seeded window would land the driver's own undefined contents where the
expectation carries the seed, which is exactly the byte the v67 probe measured
the two device families disagreeing about.
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

ATTACHMENT = v28.ATTACHMENT
OUTPUT = v28.OUTPUT
SEED = v28.MSAA_CLEAR
CANDIDATES = v28.MSAA_EDGE_CANDIDATES
CLAIMED_TEXELS = v28.MSAA_EDGE_TEXELS
DECLARING_ID = v28.DECLARING_MULTISAMPLE_SEED_ID
TRIO = ((v28.MSAA_LOAD_4X_ID, 4, None),
        (v28.MSAA_LOAD_2X_ID, 2, 2),
        (v28.MSAA_LOAD_8X_ID, 8, 8))
EDGE_MASK = (1 << 1) | (1 << 3)


def expected_bytes(texels):
    """The reviewed v69 expectation with `texels` carrying their own byte."""
    base = [OUTPUT if (index % 4) < 2
            else (v28.MSAA_MIXED if (index % 4) == 2 else SEED)
            for index in range(16)]
    return "".join(texels.get(index, byte) for index, byte in enumerate(base))


def full_capture(suite, overrides=None):
    """A vulkan capture that owes every case of the reviewed suite."""
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
                v28.msaa_load_4x_result, v28.msaa_load_2x_result,
                v28.msaa_load_8x_result,
                v28.sampled_result,
                v28.full_cover_16x16_result, v28.sampled_64_result,
                v28.sampled_rule_2048_result,
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


class SeededLoadTests(unittest.TestCase):
    def setUp(self):
        self.suite = json.loads(V28_PATH.read_bytes())

    def plan(self, suite=None):
        suite = suite or self.suite
        return compare._render_plan(compare._suite_plan(suite), suite)

    def cases(self):
        by_id = {case["id"]: case for case in self.suite["render_cases"]}
        return [by_id[case_id] for case_id, _, _ in TRIO]

    def test_the_trio_states_the_seed_route_and_the_v69_bytes(self):
        for (case_id, samples, gate), case in zip(TRIO, self.cases()):
            with self.subTest(case=case_id):
                self.assertEqual(case["id"], case_id)
                self.assertEqual(case["declaring_case"], DECLARING_ID)
                self.assertEqual(case["multisample"], {"sample_count": samples})
                self.assertEqual(case.get("requires_sample_count"), gate)
                self.assertEqual(case["coverage"], "partial")
                self.assertEqual(case["attachment"], {
                    "allocation": ATTACHMENT[0], "view": ATTACHMENT[1],
                    "format": "rgba8_unorm", "width": 4, "height": 4,
                    "load": "load", "store": "store",
                    "initial_hex": SEED * 16})
                self.assertNotIn("clear_hex", case["attachment"])
                self.assertEqual(sorted(case["capture_rails"]), sorted(v28.ALL_RAILS))

    def test_the_seeded_expectation_is_the_cleared_pair_s_own(self):
        # A seeded raster whose declared window is the colour the cleared pair
        # opens with resolves to the same bytes: the resolve cannot tell the two
        # apart, and that equality is the claim the trio states.
        by_id = {case["id"]: case for case in self.suite["render_cases"]}
        for case_id, _, _ in TRIO:
            with self.subTest(case=case_id):
                self.assertEqual(by_id[case_id]["expected_hex"],
                                 by_id[v28.MSAA_ID]["expected_hex"])
                self.assertEqual(by_id[case_id]["expected_hex"],
                                 expected_bytes({}))

    def test_the_declared_window_is_what_the_declaring_case_holds(self):
        declared = next(case for case in self.suite["cases"] if case["id"] == DECLARING_ID)
        buffer = next(entry for entry in declared["buffers"]
                      if entry["allocation"] == ATTACHMENT[0])
        self.assertEqual(buffer["initial_hex"], SEED * 16)
        self.assertEqual(buffer["length"], 64)
        # The declaring pass copies the view's first word into its own output,
        # so the seed travels with the expectation of the case the render trio
        # replays.
        self.assertEqual(declared["expected_writebacks"][0]["bytes_hex"], SEED)

    def test_the_4x_case_pins_the_crossed_column_and_the_pair_states_its_set(self):
        plan = self.plan()
        for case_id, samples, _ in TRIO:
            with self.subTest(case=case_id):
                expectation = plan[case_id]
                self.assertEqual(expectation.writes,
                                 [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                                   bytes.fromhex(expected_bytes({})))])
                candidates = compare._mix_candidates(bytes.fromhex(OUTPUT),
                                                     bytes.fromhex(SEED), samples)
                self.assertEqual(sorted(candidates), sorted(bytes.fromhex(value)
                                                            for value in CANDIDATES))
                key = (ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2])
                if case_id == v28.MSAA_LOAD_4X_ID:
                    self.assertNotIn(key, expectation.wildcards)
                else:
                    claims = expectation.wildcards[key]
                    for texel in CLAIMED_TEXELS:
                        for byte in range(4):
                            self.assertEqual(
                                sorted(claims[texel * 4 + byte]),
                                sorted(bytes.fromhex(value)[byte]
                                       for value in CANDIDATES))
                    for texel in range(16):
                        if texel in CLAIMED_TEXELS:
                            continue
                        for byte in range(4):
                            self.assertNotIn(texel * 4 + byte, claims)

    def test_a_rail_that_dropped_the_seed_is_refused(self):
        # The falsifiability the shape exists for: a rail that opened the
        # multisampled image from undefined contents lands bytes no reviewed
        # expectation carries. The RTX 5060's `dontcare` probe measured exactly
        # these bytes (`research/docs/23` §64.3): a zero-filled uncovered
        # column, and a crossed column mixed with zero rather than with the
        # fixture's own reference colour.
        for case_id, _, _ in TRIO:
            with self.subTest(case=case_id, column="uncovered"):
                dropped = expected_bytes({3: "00000000", 7: "00000000",
                                          11: "00000000", 15: "00000000"})
                report, digest = full_capture(self.suite, {case_id: dropped})
                with self.assertRaisesRegex(compare.CaptureError,
                                            "first differing byte"):
                    compare.validate_capture(self.suite, digest, report, "vulkan")
            if case_id == v28.MSAA_LOAD_4X_ID:
                continue
            with self.subTest(case=case_id, column="crossed"):
                mixed_with_zero = expected_bytes({2: "2040607f", 6: "2040607f",
                                                  10: "2040607f", 14: "2040607f"})
                report, digest = full_capture(self.suite, {case_id: mixed_with_zero})
                with self.assertRaisesRegex(compare.CaptureError,
                                            "which is none of"):
                    compare.validate_capture(self.suite, digest, report, "vulkan")

    def test_a_per_texel_seed_is_refused(self):
        broken = copy.deepcopy(self.suite)
        case = next(case for case in broken["render_cases"]
                    if case["id"] == v28.MSAA_LOAD_4X_ID)
        mixed = case["attachment"]["initial_hex"]
        case["attachment"]["initial_hex"] = mixed[:8] + "ff" + mixed[10:]
        declaring = next(case for case in broken["cases"] if case["id"] == DECLARING_ID)
        declared = next(buffer for buffer in declaring["buffers"]
                        if buffer["allocation"] == ATTACHMENT[0])
        declared["initial_hex"] = case["attachment"]["initial_hex"]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "loads one repeated texel"):
            self.plan(broken)

    def test_a_seed_equal_to_the_fragment_output_is_refused(self):
        broken = copy.deepcopy(self.suite)
        case = next(case for case in broken["render_cases"]
                    if case["id"] == v28.MSAA_LOAD_4X_ID)
        case["attachment"]["initial_hex"] = OUTPUT * 16
        declaring = next(case for case in broken["cases"] if case["id"] == DECLARING_ID)
        declared = next(buffer for buffer in declaring["buffers"]
                        if buffer["allocation"] == ATTACHMENT[0])
        declared["initial_hex"] = OUTPUT * 16
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the seed equals the fragment output"):
            self.plan(broken)

    def test_a_loaded_multisample_raster_needs_the_partial_coverage_it_resolves(self):
        broken = copy.deepcopy(self.suite)
        case = next(case for case in broken["render_cases"]
                    if case["id"] == v28.MSAA_LOAD_4X_ID)
        del case["coverage"]
        with self.assertRaisesRegex(
                compare.CaptureError,
                "a loaded multisample raster states the partial coverage its seed resolves"):
            self.plan(broken)

    def test_the_free_wildcard_list_stays_refused_beside_the_seed(self):
        broken = copy.deepcopy(self.suite)
        case = next(case for case in broken["render_cases"]
                    if case["id"] == v28.MSAA_LOAD_4X_ID)
        case["wildcard_texels"] = [2, 6]
        with self.assertRaisesRegex(compare.CaptureError,
                                    "the multisample raster claims every texel it resolves"):
            self.plan(broken)

    def test_the_allowed_set_is_held_to_the_seed_s_mixes(self):
        # A set with a missing candidate and a set with a value outside the mix
        # set are both refused: the channel is a closed claim, not a licence.
        for mutation, message in (("missing", "leaves out the resolve"),
                                  ("outside", "is not an exact mix")):
            broken = copy.deepcopy(self.suite)
            case = next(case for case in broken["render_cases"]
                        if case["id"] == v28.MSAA_LOAD_2X_ID)
            entry = case["wildcard_allowed_texels"][0]
            if mutation == "missing":
                entry["allowed"] = entry["allowed"][1:]
            else:
                entry["allowed"] = entry["allowed"] + ["ff0000ff"]
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(compare.CaptureError, message):
                    self.plan(broken)

    def test_the_pair_pins_both_extremes(self):
        # The covered columns carry the fragment output and the uncovered one
        # the seed; a fixture that only stated mixes could not show the load
        # happened at all.
        broken = copy.deepcopy(self.suite)
        case = next(case for case in broken["render_cases"]
                    if case["id"] == v28.MSAA_LOAD_2X_ID)
        case["expected_hex"] = expected_bytes({3: OUTPUT, 7: OUTPUT,
                                               11: OUTPUT, 15: OUTPUT})
        with self.assertRaisesRegex(compare.CaptureError,
                                    "needs both a fully covered and an uncovered texel"):
            self.plan(broken)

    def test_the_device_gate_has_to_name_the_raster(self):
        for case_id, samples, gate in TRIO:
            if gate is None:
                continue
            broken = copy.deepcopy(self.suite)
            case = next(case for case in broken["render_cases"] if case["id"] == case_id)
            case["requires_sample_count"] = 2 if samples == 8 else 8
            with self.subTest(case=case_id):
                with self.assertRaisesRegex(
                        compare.CaptureError,
                        "the device gate has to name the sample count"):
                    self.plan(broken)

    def test_every_candidate_of_the_pair_s_set_is_accepted(self):
        for case_id in (v28.MSAA_LOAD_2X_ID, v28.MSAA_LOAD_8X_ID):
            for candidate in CANDIDATES:
                with self.subTest(case=case_id, candidate=candidate):
                    measured = expected_bytes(
                        {texel: candidate for texel in CLAIMED_TEXELS})
                    report, digest = full_capture(self.suite, {case_id: measured})
                    compare.validate_capture(self.suite, digest, report, "vulkan")


if __name__ == "__main__":
    unittest.main()
