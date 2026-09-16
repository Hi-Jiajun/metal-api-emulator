"""Present-section checks for the observation channel of the first present increment.

These tests are comparator and schema checks, not GPU execution evidence. What
they pin is the observation surface `research/docs/24` §5.3 adds on top of v13's
attachment: a render case may declare a present target, every rail its
`capture_rails` marker names has to report that target's acquire/present counts,
the counts have to match the declaration exactly, and a rail the marker does not
name must not report the observation.

The suite under test is `suite-v13.json` plus the present section. Step 8 owns
the committed `suite-v14.json` fixture and it cannot be written before a provider
executes a present (`research/docs/24` §6 Step 8), so these tests build the
fixture they need instead of pinning one that does not exist yet.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare
from test_compare import synthetic_report
from test_suite_v13 import (ATTACHMENT, REPORTING_RAILS, RENDER_ID, TEXELS,
                            render_result, suite_without_marker_rail)


CONFORMANCE = Path(__file__).resolve().parent
V13_PATH = CONFORMANCE / "suite-v13.json"
V14_PATH = CONFORMANCE / "suite-v14.json"

# The present section the provider side parses (`research/docs/24` §5.3): one
# target image, one acquire and one present, and a sentinel that cannot be
# confused with the expected texel.
PRESENT = {"mode": "fifo", "image_count": 1, "acquire": 1, "present": 1,
           "initial_hex": "fefefefe"}
OBSERVATION = {"acquire": 1, "present": 1}

# The v62 case: the v14 present fixture's own shape over a four-sample raster.
# The full-coverage triangle resolves every sample to the fragment output, so
# the expectation is the v14 texel four times; the marker names the Vulkan
# trace rail alone, because the object rails record no present entry and the
# native/oracle present path is the separate self-test channel
# (`research/docs/24` §5.3, v62).
MSAA_PRESENT_ID = "present_msaa_triangle_2x2"

# The v1-v13 render plans as they are shipped (`research/docs/24` §6 Step 4:
# "v1-v13 的 plan 不受影响"). Twelve of the thirteen committed suites carry no
# render case at all, and v13's one render case is pinned field for field; the
# present section is the only thing the v14 increment adds to this surface.
PINNED_PLANS = {
    "compute-buffer-v13": {
        "offscreen_triangle_clear_2x2": {
            "writes": [[[900, 910, 0], TEXELS]],
            "allocations": [[900, TEXELS]],
            "touched": [900, 920],
            "written": [900, 920],
            "rails": sorted(REPORTING_RAILS),
            "attachment": list(ATTACHMENT),
            "present": None,
        },
    },
}

# The rails `suite-v14.json` marks (`research/docs/24` §5.1/§5.5): every rail
# that reports provider present counts. The Swift oracle reports no counters,
# so it is the one render-capable rail v14 leaves out; gating a present case on
# its capture would ask for an observation it cannot produce.
V14_REPORTING_RAILS = ("vulkan", "vulkan-objects", "native-metal-provider",
                       "native-metal-provider-objects")


def synthetic_suite(present=PRESENT):
    """`suite-v13.json` reshaped into v14: present section, provider-only marker.

    The synthetic suite keeps v13's case ids (the committed v14 fixture renames
    the render case, but the comparator rules under test do not depend on the
    name) while dropping `native-metal` from the marker, because the Swift
    oracle reports no provider counters (`research/docs/24` §5.1). Its marker
    therefore matches the committed `suite-v14.json`'s rail set.
    """
    suite = json.loads(V13_PATH.read_text(encoding="utf-8"))
    if present is not None:
        suite["render_cases"][0]["present"] = copy.deepcopy(present)
    for case in suite.get("render_cases", []):
        case["capture_rails"] = [rail for rail in case["capture_rails"]
                                 if rail != "native-metal"]
    digest = hashlib.sha256(json.dumps(suite, sort_keys=True).encode("utf-8")).hexdigest()
    return suite, digest


def plan_signature(render_plan):
    """The plan's whole content, in a form a test can pin and a reviewer can read."""
    signature = {}
    for case_id, expectation in sorted(render_plan.items()):
        present = expectation.present
        signature[case_id] = {
            "writes": [[list(identity), data.hex()] for identity, data in expectation.writes],
            "allocations": [[allocation, data.hex()]
                            for allocation, data in sorted(expectation.allocations.items())],
            "touched": sorted(expectation.touched),
            "written": sorted(expectation.written),
            "rails": sorted(expectation.rails),
            "attachment": list(expectation.attachment),
            "present": None if present is None else {
                "mode": present.mode,
                "image_count": present.image_count,
                "acquire": present.acquire,
                "present": present.present,
                "sentinel": None if present.sentinel is None else present.sentinel.hex(),
            },
        }
    return signature


def capture_for(suite, digest, backend="vulkan", observation=OBSERVATION, report_case=None):
    """A capture that reports the present observation exactly where a rail does.

    `report_case` overrides the membership of `capture_rails` so a test can make
    an unnamed rail report the render case, which the byte rule refuses on its
    own (`test_suite_v13.py`).
    """
    report = synthetic_report(suite, digest, backend)
    if backend != "native-metal":
        for result in report["results"]:
            result["copy_in"], result["copy_out"] = 2, 1
    if report_case is None:
        report_case = backend in V14_REPORTING_RAILS
    if report_case:
        result = render_result(backend != "native-metal")
        if observation is not None:
            result["present"] = copy.deepcopy(observation)
        report["results"].append(result)
    return report


class PresentDeclarationTests(unittest.TestCase):
    """The suite-side section: a whitelisted shape and a falsifiable sentinel."""

    def test_present_section_is_planned_with_its_sentinel(self):
        suite, _ = synthetic_suite()
        present = compare._render_plan(compare._suite_plan(suite), suite)[RENDER_ID].present
        self.assertEqual(present.mode, "fifo")
        self.assertEqual(present.image_count, 1)
        self.assertEqual((present.acquire, present.present), (1, 1))
        self.assertEqual(present.sentinel, bytes.fromhex("fefefefe"))
        # The byte observation of the same case is the v13 one, unchanged.
        plan = compare._render_plan(compare._suite_plan(suite), suite)
        self.assertEqual(plan_signature(plan)[RENDER_ID]["attachment"], list(ATTACHMENT))

    def test_an_omitted_sentinel_is_undefined_rather_than_a_value(self):
        suite, _ = synthetic_suite({key: value for key, value in PRESENT.items()
                                    if key != "initial_hex"})
        present = compare._render_plan(compare._suite_plan(suite), suite)[RENDER_ID].present
        self.assertIsNone(present.sentinel)
        self.assertEqual((present.image_count, present.acquire, present.present), (1, 1, 1))

    def test_a_case_without_the_section_plans_no_present(self):
        suite, _ = synthetic_suite(None)
        self.assertIsNone(
            compare._render_plan(compare._suite_plan(suite), suite)[RENDER_ID].present)

    def test_present_section_has_to_be_an_object(self):
        suite, _ = synthetic_suite(None)
        suite["render_cases"][0]["present"] = "fifo"
        with self.assertRaisesRegex(compare.CaptureError, "present: expected an object"):
            compare._render_plan(compare._suite_plan(suite), suite)

    def test_present_declaration_rejects_widened_shapes(self):
        for name, mutate, message in (
            ("mode", lambda present: present.update(mode="mailbox"), "fifo"),
            ("mode-type", lambda present: present.update(mode=0), "fifo"),
            ("image_count", lambda present: present.update(image_count=2), "image_count"),
            ("image_count-zero", lambda present: present.update(image_count=0), "image_count"),
            ("acquire", lambda present: present.update(acquire=2), "exactly once"),
            ("acquire-zero", lambda present: present.update(acquire=0), "exactly once"),
            ("present", lambda present: present.update(present=2), "exactly once"),
            ("sentinel-length", lambda present: present.update(initial_hex="fefefe"),
             "one texel"),
            ("sentinel-copy", lambda present: present.update(initial_hex="4080c0ff"),
             "equals the expected texel"),
            ("sentinel-odd", lambda present: present.update(initial_hex="fefefef"),
             "hexadecimal"),
            ("unknown-field", lambda present: present.update(frames=1), "unexpected fields"),
            ("missing-field", lambda present: present.pop("image_count"), "missing fields"),
            ("missing-mode", lambda present: present.pop("mode"), "missing fields"),
        ):
            with self.subTest(mutation=name):
                present = copy.deepcopy(PRESENT)
                mutate(present)
                suite, _ = synthetic_suite(present)
                with self.assertRaisesRegex(compare.CaptureError, message):
                    compare._render_plan(compare._suite_plan(suite), suite)

    def test_the_render_case_field_set_gains_only_present(self):
        suite, _ = synthetic_suite(None)
        suite["render_cases"][0]["present_target"] = copy.deepcopy(PRESENT)
        with self.assertRaisesRegex(compare.CaptureError, "unexpected fields"):
            compare._render_plan(compare._suite_plan(suite), suite)
        suite, _ = synthetic_suite()
        del suite["render_cases"][0]["capture_rails"]
        with self.assertRaisesRegex(compare.CaptureError, "missing fields"):
            compare._render_plan(compare._suite_plan(suite), suite)


class PresentObservationTests(unittest.TestCase):
    """The capture side: declared observations are required, undeclared ones refused."""

    def setUp(self):
        self.suite, self.digest = synthetic_suite()

    def validate(self, report, backend="vulkan"):
        compare.validate_capture(self.suite, self.digest, report, backend)

    def reject(self, report, message, backend="vulkan"):
        with self.assertRaisesRegex(compare.CaptureError, message):
            self.validate(report, backend)

    def validate_trimmed(self, suite, digest, report, backend):
        compare.validate_capture(suite, digest, report, backend)

    def reject_trimmed(self, suite, digest, report, message, backend):
        with self.assertRaisesRegex(compare.CaptureError, message):
            self.validate_trimmed(suite, digest, report, backend)

    def test_every_named_rail_reports_the_counts_the_suite_declares(self):
        for backend in V14_REPORTING_RAILS:
            with self.subTest(backend=backend):
                self.validate(capture_for(self.suite, self.digest, backend), backend)

    def test_a_named_rail_has_to_report_the_observation(self):
        for backend in V14_REPORTING_RAILS:
            with self.subTest(backend=backend):
                self.reject(capture_for(self.suite, self.digest, backend, observation=None),
                            "has to report the present observation", backend)

    def test_a_named_rail_has_to_report_the_case_at_all(self):
        for backend in V14_REPORTING_RAILS:
            with self.subTest(backend=backend):
                self.reject(capture_for(self.suite, self.digest, backend, report_case=False),
                            "missing cases", backend)

    def test_a_replayed_sentinel_is_refused(self):
        # The sentinel exists to make "the present never happened" falsifiable,
        # so a target that still holds it cannot pass even though the counts and
        # the identities are the ones the suite asked for.
        sentinel = PRESENT["initial_hex"] * 4
        report = capture_for(self.suite, self.digest)
        for result in report["results"]:
            if result["id"] == RENDER_ID:
                result["writebacks"][0]["bytes_hex"] = sentinel
                result["allocations"][0]["bytes_hex"] = sentinel
        self.reject(report, "first differing byte")

    def test_counts_that_disagree_with_the_suite_are_refused(self):
        for observation, message in (
            ({"acquire": 2, "present": 1}, "acquire 2 does not match the 1"),
            ({"acquire": 1, "present": 2}, "present count 2 does not match the 1"),
            ({"acquire": 0, "present": 1}, "acquire 0 does not match the 1"),
            ({"acquire": 1, "present": 0}, "present count 0 does not match the 1"),
        ):
            with self.subTest(observation=observation):
                self.reject(capture_for(self.suite, self.digest, observation=observation),
                            message)

    def test_the_observation_is_exactly_two_unsigned_counts(self):
        for observation, message in (
            ({"acquire": 1}, "expected fields"),
            ({"present": 1}, "expected fields"),
            ({"acquire": 1, "present": 1, "dropped": 0}, "expected fields"),
            ({"acquire": "1", "present": 1}, "expected integer"),
            ({"acquire": True, "present": 1}, "expected integer"),
            ({"acquire": 1, "present": 1.0}, "expected integer"),
            ({"acquire": -1, "present": 1}, "expected integer"),
            ({"acquire": 1 << 32, "present": 1}, "expected integer"),
        ):
            with self.subTest(observation=observation):
                self.reject(capture_for(self.suite, self.digest, observation=observation),
                            message)

    def test_a_rail_the_marker_does_not_name_must_not_report_the_observation(self):
        # Every committed rail is named now, so the unnamed-rail rule is
        # exercised against a copy of the suite whose marker drops one rail.
        suite, digest = suite_without_marker_rail(self.suite, "native-metal-provider-objects")
        for backend in ("native-metal-provider-objects",):
            with self.subTest(backend=backend, reported="the case only"):
                # The byte rule already refuses the case itself; the present
                # rule has to refuse it as well rather than let an unnamed rail
                # report the counts of a case it does not run.
                self.reject_trimmed(suite, digest,
                                    capture_for(suite, digest, backend, observation=None,
                                                report_case=True),
                                    "not a rail this render case runs on", backend)
            with self.subTest(backend=backend, reported="the observation"):
                self.reject_trimmed(suite, digest,
                                    capture_for(suite, digest, backend, report_case=True),
                                    "must not report the present observation", backend)
            with self.subTest(backend=backend, reported="nothing"):
                self.validate_trimmed(suite, digest,
                                      capture_for(suite, digest, backend, report_case=False),
                                      backend)

    def test_an_undeclared_observation_is_refused(self):
        # The committed v13 suite declares no present section, so no rail may
        # report one: an observation the suite does not ask for is not evidence.
        suite, digest = synthetic_suite(None)
        with self.assertRaisesRegex(compare.CaptureError, "declares no present observation"):
            compare.validate_capture(suite, digest, capture_for(suite, digest), "vulkan")
        report = capture_for(suite, digest, observation=None)
        report["results"][0]["present"] = copy.deepcopy(OBSERVATION)
        with self.assertRaisesRegex(compare.CaptureError, "declares no present observation"):
            compare.validate_capture(suite, digest, report, "vulkan")

    def test_the_observation_does_not_relax_the_counter_pair_rule(self):
        report = capture_for(self.suite, self.digest)
        for result in report["results"]:
            if result["id"] == RENDER_ID:
                del result["copy_out"]
        self.reject(report, "optionally with copy_in and copy_out")


class MsaaPresentObservationTests(unittest.TestCase):
    """The v62 case: the v14 present shape over a four-sample raster.

    The committed suite carries the case on the Vulkan trace rail alone, so a
    Vulkan capture owes the resolved texels *and* the acquire/present counts,
    while every other rail has to leave the case out entirely — the same
    marker rule the v14 case states, with the narrower marker spelling which
    rail owns the multisampled present path (`research/docs/24` §5.3, v62).
    """

    def setUp(self):
        self.raw = V14_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def present_result(self, case_id, provider_backend=True, observation=OBSERVATION):
        result = {
            "id": case_id,
            "completion": "CompletedVisible",
            "writebacks": [{"allocation": 900, "view": 910, "offset": 0,
                            "bytes_hex": TEXELS}],
            "allocations": [{"allocation": 900, "bytes_hex": TEXELS}],
        }
        if provider_backend:
            result["copy_in"], result["copy_out"] = 2, 2
        if observation is not None:
            result["present"] = copy.deepcopy(observation)
        return result

    def report(self, backend, msaa=True, observation=OBSERVATION):
        report = synthetic_report(self.suite, self.digest, backend)
        if backend != "native-metal":
            for result in report["results"]:
                result["copy_in"], result["copy_out"] = 2, 1
        if backend in V14_REPORTING_RAILS:
            report["results"].append(self.present_result("present_triangle_clear_2x2",
                                                         backend != "native-metal"))
        if backend == "vulkan":
            if msaa:
                report["results"].append(
                    self.present_result(MSAA_PRESENT_ID, backend != "native-metal",
                                        observation))
        return report

    def test_v14_pins_the_multisample_present_fixture(self):
        case = self.suite["render_cases"][1]
        self.assertEqual(case["id"], MSAA_PRESENT_ID)
        self.assertEqual(case["declaring_case"], "render_declaring_copy_word")
        self.assertEqual(case["vertex_entry"], "render_fullscreen_triangle")
        self.assertEqual(case["multisample"], {"sample_count": 4})
        self.assertEqual(case["attachment"]["allocation"], 900)
        self.assertEqual(case["attachment"]["view"], 910)
        self.assertEqual(case["attachment"]["load"], "clear")
        self.assertEqual(case["attachment"]["clear_hex"], "fefefefe")
        self.assertEqual(case["present"], {
            "mode": "fifo", "image_count": 1, "acquire": 1, "present": 1,
            "initial_hex": "efefefef"})
        self.assertEqual(case["expected_hex"], TEXELS)
        self.assertEqual(case["capture_rails"], ["vulkan"])
        # The full-coverage triangle resolves every sample to the fragment
        # output, so the expectation is the uniform texel and the sentinel
        # stays distinguishable from it (`research/docs/24` §3.1, v62).
        self.assertNotEqual(case["attachment"]["clear_hex"], case["expected_hex"][:8])
        self.assertNotEqual(case["present"]["initial_hex"], case["expected_hex"][:8])

    def test_the_vulkan_capture_reports_both_present_cases(self):
        report = self.report("vulkan")
        compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_the_msaa_present_case_needs_its_own_observation(self):
        report = self.report("vulkan", observation=None)
        with self.assertRaisesRegex(compare.CaptureError,
                                    "has to report the present observation"):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_the_msaa_present_case_needs_the_vulkan_landing(self):
        report = self.report("vulkan", msaa=False)
        with self.assertRaisesRegex(compare.CaptureError, "missing cases"):
            compare.validate_capture(self.suite, self.digest, report, "vulkan")

    def test_a_rail_the_marker_does_not_name_must_not_report_the_msaa_case(self):
        # The committed marker names `vulkan` alone, so a native-metal capture
        # that reports the case is refused by the byte rule before the present
        # rule can even run (`research/docs/24` §5.5, v62).
        report = self.report("native-metal")
        report["results"].append(self.present_result(MSAA_PRESENT_ID, False))
        with self.assertRaisesRegex(compare.CaptureError,
                                    "must not report the present observation"):
            compare.validate_capture(self.suite, self.digest, report, "native-metal")

    def test_a_rail_the_marker_does_not_name_omits_the_case(self):
        # Every rail the marker does not name owes the v14 case's own landing
        # where its marker names that one, and owes nothing of the v62 case
        # (`research/docs/24` §5.5, v62).
        for backend in ("native-metal", "vulkan-objects", "native-metal-provider",
                        "native-metal-provider-objects"):
            with self.subTest(backend=backend):
                compare.validate_capture(self.suite, self.digest,
                                         self.report(backend), backend)

    def test_the_msaa_present_counts_do_not_relax_the_counter_pair_rule(self):
        for observation, message in (
                ({"acquire": 2, "present": 1}, "acquire 2 does not match the 1"),
                ({"acquire": 1, "present": 2}, "present count 2 does not match the 1"),
                ({"acquire": 0, "present": 1}, "acquire 0 does not match the 1"),
                ({"acquire": 1, "present": 0}, "present count 0 does not match the 1"),
        ):
            with self.subTest(observation=observation):
                report = self.report("vulkan", observation=observation)
                with self.assertRaisesRegex(compare.CaptureError, message):
                    compare.validate_capture(self.suite, self.digest, report, "vulkan")


# The committed v14 suite's render plan (`research/docs/24` §6 Step 8): the
# same 2x2 attachment as v13, plus the present section the provider rails
# report. It is pinned here for the same reason the v13 plan is: a change to
# the shipped fixture has to be a deliberate edit to this table, not a side
# effect of touching the comparator.
PINNED_V14_PLAN = {
    "present_triangle_clear_2x2": {
        "writes": [[[900, 910, 0], TEXELS]],
        "allocations": [[900, TEXELS]],
        "touched": [900, 920],
        "written": [900, 920],
        "rails": sorted(["vulkan", "vulkan-objects", "native-metal-provider",
                         "native-metal-provider-objects"]),
        "attachment": list(ATTACHMENT),
        "present": {
            "mode": "fifo",
            "image_count": 1,
            "acquire": 1,
            "present": 1,
            "sentinel": "efefefef",
        },
    },
    MSAA_PRESENT_ID: {
        "writes": [[[900, 910, 0], TEXELS]],
        "allocations": [[900, TEXELS]],
        "touched": [900, 920],
        "written": [900, 920],
        "rails": ["vulkan"],
        "attachment": list(ATTACHMENT),
        "present": {
            "mode": "fifo",
            "image_count": 1,
            "acquire": 1,
            "present": 1,
            "sentinel": "efefefef",
        },
    },
}

# The committed v15 suite's render plan (`research/docs/25` §6 Step 8): v13's
# 2x2 attachment replayed from one indirect draw command, marked for the Vulkan
# trace rail only. The heap and dispatch cases are compute cases and do not
# appear in the render plan.
PINNED_V15_PLAN = {
    "icb_draw_clear_2x2": {
        "writes": [[[900, 910, 0], TEXELS]],
        "allocations": [[900, TEXELS]],
        "touched": [900, 920],
        "written": [900, 920],
        "rails": sorted(["vulkan", "vulkan-objects"]),
        "attachment": list(ATTACHMENT),
        "present": None,
    },
}


class ShippedSuitePlanTests(unittest.TestCase):
    """v1-v13 keep their plan; v14 is pinned with its present section."""

    def test_shipped_plans_are_pinned(self):
        paths = sorted(CONFORMANCE.glob("suite*.json"))
        # One suite with a render plan (v13), the committed v14 and v15
        # fixtures, and the twelve suites that carry no render case at all.
        # v16's and v17's plans are pinned by their own test modules, which own
        # the sections this file predates; v18's MRT plan is pinned by
        # `test_suite_v18.py`, v19's store/dontcare plan by `test_suite_v19.py`,
        # v20's dontcare-load plan by `test_suite_v20.py`, v21's
        # attachment-format plan by `test_suite_v21.py`, v22's single-channel
        # float plan by `test_suite_v22.py`, v23's four-location plan by
        # `test_suite_v23.py`, v24's three-location plan by `test_suite_v24.py`
        # v25's mixed-layout plan by `test_suite_v25.py` and v26's 4x4 extent
        # plan by `test_suite_v26.py` and v27's MRT-loading plan by
        # `test_suite_v27.py` and v28's scissored plan by `test_suite_v28.py`.
        self.assertEqual(len(paths), len(PINNED_PLANS) + 27)
        observed = {}
        for path in paths:
            suite = json.loads(path.read_text(encoding="utf-8"))
            signature = plan_signature(compare._render_plan(compare._suite_plan(suite), suite))
            expected = PINNED_PLANS.get(suite["suite"])
            if expected is None:
                if suite["suite"] in ("compute-buffer-v16", "compute-buffer-v17",
                                      "compute-buffer-v18", "compute-buffer-v19",
                                      "compute-buffer-v20", "compute-buffer-v21",
                                      "compute-buffer-v22", "compute-buffer-v23",
                                      "compute-buffer-v24", "compute-buffer-v25",
                                      "compute-buffer-v26", "compute-buffer-v27",
                                      "compute-buffer-v28"):
                    continue
                expected = {"compute-buffer-v14": PINNED_V14_PLAN,
                            "compute-buffer-v15": PINNED_V15_PLAN}.get(suite["suite"], {})
            self.assertEqual(signature, expected, path.name + ": the render plan drifted")
            observed[suite["suite"]] = signature
        for identity, signature in observed.items():
            for case_id, expectation in signature.items():
                if identity == "compute-buffer-v14":
                    self.assertIsNotNone(expectation["present"],
                                         identity + "/" + case_id + ": v14 declares present")
                else:
                    self.assertIsNone(expectation["present"],
                                      identity + "/" + case_id + ": v1-v13 declare no present")


if __name__ == "__main__":
    unittest.main()
