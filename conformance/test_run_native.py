"""Simulated command responses only; no synthetic fixture is Metal evidence."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import run_native
from test_compare import SUITE_PATH, synthetic_report


class NativeCaptureFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.suite_path = self.root / "suite.json"
        self.suite_path.write_bytes(SUITE_PATH.read_bytes())
        self.output = self.root / "evidence"
        self.suite = json.loads(self.suite_path.read_text())
        digest = hashlib.sha256(self.suite_path.read_bytes()).hexdigest()
        self.report = synthetic_report(self.suite, digest, "native-metal")
        self.probe = {"schema_version": 1, "kind": "metal-device-probe",
                      "platform": self.report["platform"], "device": self.report["device"],
                      "eligible": True, "reason": "eligible", "supports_apple4": True,
                      "has_unified_memory": True}
        self.commands = []
        self.fail_capture = False

    def fake_command(self, argv, *, stdout, stderr, check, timeout):
        self.assertTrue(check)
        self.assertGreater(timeout, 0)
        self.commands.append(argv)
        if "--probe" in argv:
            stdout.write(json.dumps(self.probe).encode())
        elif "--output" in argv:
            if self.fail_capture:
                stderr.write(b"SIMULATED Metal execution failure")
                raise subprocess.CalledProcessError(1, argv)
            Path(argv[argv.index("--output") + 1]).write_text(json.dumps(self.report))

    def run_flow(self, **kwargs):
        return run_native.run_capture(self.root / "fake-oracle", self.suite_path, self.output,
                                      revision="synthetic-test-revision", run_command=self.fake_command,
                                      **kwargs)

    def status(self):
        return json.loads((self.output / "status.json").read_text())

    def test_eligible_device_runs_and_validates_capture(self):
        result = self.run_flow()
        self.assertEqual(result["capture_status"], "captured")
        self.assertEqual(result, self.status())
        self.assertEqual(len(self.commands), 3)
        self.assertTrue((self.output / "probe.json").exists())

    def test_no_gpu_is_unavailable_and_never_runs_capture(self):
        self.probe.update(device=None, eligible=False, reason="no_default_device",
                          supports_apple4=False, has_unified_memory=False)
        result = self.run_flow()
        self.assertEqual(result["capture_status"], "unavailable")
        self.assertEqual(len(self.commands), 2)
        self.assertFalse((self.output / "native-metal.json").exists())

    def test_required_gpu_unavailable_fails_with_evidence(self):
        self.probe.update(eligible=False, reason="unsupported_features", supports_apple4=False)
        with self.assertRaisesRegex(run_native.NativeRunError, "unavailable"):
            self.run_flow(require_metal=True)
        self.assertEqual(self.status()["capture_status"], "unavailable")

    def test_unnamed_device_is_ineligible_not_a_broken_probe(self):
        self.probe.update(device=" ", eligible=False, reason="unsupported_features")
        result = self.run_flow()
        self.assertEqual(result["capture_status"], "unavailable")
        self.assertEqual(len(self.commands), 2)

    def test_eligible_capture_error_is_not_downgraded_to_no_gpu(self):
        self.fail_capture = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_flow()
        self.assertEqual(self.status()["capture_status"], "failed")
        self.assertIn(b"SIMULATED", (self.output / "capture.stderr").read_bytes())

    def test_inconsistent_probe_is_a_failure(self):
        original = copy.deepcopy(self.probe)
        for change in ({"eligible": False}, {"supports_apple4": 1}, {"reason": "unknown"},
                       {"device": None}, {"schema_version": True}, {"platform": ""}):
            with self.subTest(change=change):
                probe = dict(original, **change)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_probe(probe)

    def test_bad_capture_cannot_pass(self):
        self.report["results"][0]["writebacks"][0]["bytes_hex"] = "00000000"
        with self.assertRaisesRegex(ValueError, "differing byte"):
            self.run_flow()
        self.assertEqual(self.status()["capture_status"], "failed")

    def test_vulkan_capture_cannot_substitute_for_native(self):
        self.report["backend"] = "vulkan"
        with self.assertRaisesRegex(ValueError, "expected backend native-metal"):
            self.run_flow()
        self.assertEqual(self.status()["capture_status"], "failed")

    def test_probe_capture_device_disagreement_is_failure(self):
        self.report["device"] = "SYNTHETIC DIFFERENT DEVICE"
        with self.assertRaisesRegex(run_native.NativeRunError, "differs from probe"):
            self.run_flow()

    def test_output_directory_is_not_reused(self):
        self.output.mkdir()
        (self.output / "prior.json").write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_flow()
        self.assertEqual(self.commands, [])
        self.assertEqual((self.output / "prior.json").read_text(), "keep")


class PresentSelftestValidationTests(unittest.TestCase):
    """The CI step's present-selftest byte comparison, exercised without Metal.

    `run_native.validate_present_selftest` is the function the workflow reuses,
    so the sentinel rule is pinned here rather than only in the inline heredoc.
    """

    TARGET = "4080c0ff" * 4
    SENTINEL = "fefefefe" * 4

    def reviewed_report(self, writeback_bytes=None, allocation_bytes=None, completion="CompletedVisible"):
        return {
            "completion": completion,
            "writebacks": [{"bytes_hex": writeback_bytes}],
            "allocations": [{"bytes_hex": allocation_bytes}],
        }

    def test_accepts_the_reviewed_present_target(self):
        report = self.reviewed_report(self.TARGET, self.TARGET)
        self.assertEqual(run_native.validate_present_selftest(report), [self.TARGET, self.TARGET])

    def test_rejects_the_sentinel_in_either_channel(self):
        for writeback, allocation in (
            (self.SENTINEL, self.TARGET),
            (self.TARGET, self.SENTINEL),
            (self.SENTINEL, self.SENTINEL),
        ):
            with self.subTest(writeback=writeback, allocation=allocation):
                report = self.reviewed_report(writeback, allocation)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_present_selftest(report)

    def test_rejects_a_missing_or_extra_observation(self):
        for writebacks, allocations in (
            ([], []),
            ([{"bytes_hex": self.TARGET}], []),
            ([], [{"bytes_hex": self.TARGET}]),
            ([{"bytes_hex": self.TARGET}] * 2, [{"bytes_hex": self.TARGET}]),
            # The bytes concatenated still equal [expected, expected], so these
            # two only fail because the shape itself is pinned: two writebacks
            # with no allocation, and the reverse.
            ([{"bytes_hex": self.TARGET}] * 2, []),
            ([], [{"bytes_hex": self.TARGET}] * 2),
        ):
            with self.subTest(writebacks=writebacks, allocations=allocations):
                report = {"completion": "CompletedVisible",
                          "writebacks": writebacks, "allocations": allocations}
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_present_selftest(report)

    def test_rejects_a_non_visible_completion(self):
        report = self.reviewed_report(self.TARGET, self.TARGET, completion="Submitted")
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_present_selftest(report)

    def test_rejects_a_non_object_report(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_present_selftest([])


class HeapSelftestValidationTests(unittest.TestCase):
    """The CI step's heap-selftest byte comparison, exercised without Metal.

    `run_native.validate_heap_selftest` is the function the workflow reuses, so
    the sentinel and observation-shape rules are pinned here rather than only in
    the inline heredoc.
    """

    WORD = "fefefefe"
    READ_ALLOCATION = "fefefefefefefefefefefefefefefefe"
    WRITE_ALLOCATION = "fefefefeffffffffffffffff"
    SENTINEL_WRITE = "ffffffff"

    def reviewed_report(self, completion="CompletedVisible", device="Apple GPU",
                        platform="macOS 15.0", writeback_bytes=None,
                        read_allocation=None, write_allocation=None,
                        writebacks=None, allocations=None):
        if writebacks is None:
            writebacks = [{"allocation": 920, "view": 930, "offset": 0,
                           "bytes_hex": self.WORD if writeback_bytes is None else writeback_bytes}]
        if allocations is None:
            allocations = [
                {"allocation": 900, "bytes_hex": self.READ_ALLOCATION if read_allocation is None else read_allocation},
                {"allocation": 920, "bytes_hex": self.WRITE_ALLOCATION if write_allocation is None else write_allocation},
            ]
        return {"id": "heap_placement_copy_word", "completion": completion,
                "writebacks": writebacks, "allocations": allocations,
                "device": device, "platform": platform}

    def test_accepts_the_reviewed_heap_observation(self):
        report = self.reviewed_report()
        self.assertEqual(run_native.validate_heap_selftest(report), self.WORD)

    def test_rejects_a_sentinel_writeback(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_heap_selftest(self.reviewed_report(writeback_bytes=self.SENTINEL_WRITE))

    def test_rejects_a_sentinel_in_either_allocation(self):
        sentinel_read = "ffffffff" * 4
        sentinel_write = "ffffffff" * 3
        for read_allocation, write_allocation in (
            (sentinel_read, self.WRITE_ALLOCATION),
            (self.READ_ALLOCATION, sentinel_write),
            (sentinel_read, sentinel_write),
        ):
            with self.subTest(read=read_allocation, write=write_allocation):
                report = self.reviewed_report(read_allocation=read_allocation,
                                              write_allocation=write_allocation)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_heap_selftest(report)

    def test_rejects_a_missing_or_extra_observation(self):
        writeback = [{"allocation": 920, "view": 930, "offset": 0,
                      "bytes_hex": self.WORD}]
        allocations = [
            {"allocation": 900, "bytes_hex": self.READ_ALLOCATION},
            {"allocation": 920, "bytes_hex": self.WRITE_ALLOCATION},
        ]
        for writebacks, extra_allocations in (
            ([], []),
            (writeback, []),
            ([], allocations),
            (writeback, allocations[:1]),
            (writeback, allocations + [{"allocation": 921, "bytes_hex": "00"}]),
            (writeback + writeback, allocations),
        ):
            with self.subTest(writebacks=writebacks, allocations=extra_allocations):
                report = self.reviewed_report(writebacks=writebacks,
                                              allocations=extra_allocations)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_heap_selftest(report)

    def test_rejects_a_wrong_writeback_offset_or_identity(self):
        for change in ({"offset": 4}, {"view": 931}, {"allocation": 921}):
            with self.subTest(change=change):
                writeback = [dict({"allocation": 920, "view": 930, "offset": 0,
                                   "bytes_hex": self.WORD}, **change)]
                report = self.reviewed_report(writebacks=writeback)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_heap_selftest(report)

    def test_rejects_a_non_visible_completion(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_heap_selftest(self.reviewed_report(completion="Submitted"))

    def test_rejects_a_missing_device_or_platform(self):
        for device, platform in (("", "macOS 15.0"), ("Apple GPU", ""),
                                 (None, "macOS 15.0"), ("Apple GPU", None)):
            with self.subTest(device=device, platform=platform):
                report = self.reviewed_report(device=device, platform=platform)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_heap_selftest(report)

    def test_rejects_a_non_object_report(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_heap_selftest([])


class VertexSelftestValidationTests(unittest.TestCase):
    """The vertex-input self-test's byte comparison, exercised without Metal.

    `run_native.validate_vertex_selftest` is the function the CI step reuses, so
    the sentinel and observation-shape rules are pinned here rather than only in
    an inline heredoc. The fixture id is part of the claim: the indexed quad
    writes the same four texels the plain render self-test does, so the id is
    what separates the two observations.
    """

    TARGET = "4080c0ff" * 4
    SENTINEL = "fefefefe" * 4

    def reviewed_report(self, report_id="vertex_quad_indexed_2x2",
                        completion="CompletedVisible", writebacks=None,
                        allocations=None):
        if writebacks is None:
            writebacks = [{"allocation": 900, "view": 910, "offset": 0,
                           "bytes_hex": self.TARGET}]
        if allocations is None:
            allocations = [{"allocation": 900, "bytes_hex": self.TARGET}]
        return {"id": report_id, "completion": completion,
                "writebacks": writebacks, "allocations": allocations}

    def test_accepts_the_reviewed_indexed_quad_observation(self):
        report = self.reviewed_report()
        self.assertEqual(run_native.validate_vertex_selftest(report), self.TARGET)

    def test_rejects_the_clear_sentinel_in_either_channel(self):
        for writeback, allocation in (
            (self.SENTINEL, self.TARGET),
            (self.TARGET, self.SENTINEL),
            (self.SENTINEL, self.SENTINEL),
        ):
            with self.subTest(writeback=writeback, allocation=allocation):
                report = self.reviewed_report(
                    writebacks=[{"allocation": 900, "view": 910, "offset": 0,
                                 "bytes_hex": writeback}],
                    allocations=[{"allocation": 900, "bytes_hex": allocation}])
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_vertex_selftest(report)

    def test_rejects_the_plain_render_selftest_report(self):
        # The `vertex_id` fixture reaches the same attachment bytes, so only the
        # id distinguishes it; a report from that self-test must not pass here.
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_vertex_selftest(
                self.reviewed_report(report_id="render_offscreen_2x2"))

    def test_rejects_a_missing_or_extra_observation(self):
        writeback = [{"allocation": 900, "view": 910, "offset": 0,
                      "bytes_hex": self.TARGET}]
        allocations = [{"allocation": 900, "bytes_hex": self.TARGET}]
        for writebacks, observed in (
            ([], []),
            (writeback, []),
            ([], allocations),
            (writeback, allocations + [{"allocation": 950, "bytes_hex": self.TARGET}]),
            (writeback + writeback, allocations),
        ):
            with self.subTest(writebacks=writebacks, allocations=observed):
                report = self.reviewed_report(writebacks=writebacks, allocations=observed)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_vertex_selftest(report)

    def test_rejects_a_wrong_writeback_identity_or_offset(self):
        for change in ({"offset": 4}, {"view": 911}, {"allocation": 901}):
            with self.subTest(change=change):
                writeback = [dict({"allocation": 900, "view": 910, "offset": 0,
                                   "bytes_hex": self.TARGET}, **change)]
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_vertex_selftest(
                        self.reviewed_report(writebacks=writeback))

    def test_rejects_a_non_visible_completion(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_vertex_selftest(
                self.reviewed_report(completion="Submitted"))

    def test_rejects_a_non_object_report(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_vertex_selftest([])


class MrtSelftestValidationTests(unittest.TestCase):
    """The MRT self-test's byte comparison, exercised without Metal.

    `run_native.validate_mrt_selftest` is the function the CI step reuses, so
    the sentinel, location-order and observation-shape rules are pinned here
    rather than only in an inline heredoc. The fixture id and both locations'
    identities are part of the claim: the dual quad writes the same stream and
    index buffers as the vertex self-test, so only the id and the per-location
    bytes separate this observation from the single-output ones.
    """

    FIRST = "4080c0ff" * 4
    SECOND = "ff8040c0" * 4
    SENTINEL = "fefefefe" * 4

    def reviewed_report(self, report_id="mrt_dual_output_2x2",
                        completion="CompletedVisible", writebacks=None,
                        allocations=None):
        if writebacks is None:
            writebacks = [
                {"allocation": 900, "view": 910, "offset": 0,
                 "bytes_hex": self.FIRST},
                {"allocation": 901, "view": 911, "offset": 0,
                 "bytes_hex": self.SECOND},
            ]
        if allocations is None:
            allocations = [
                {"allocation": 900, "bytes_hex": self.FIRST},
                {"allocation": 901, "bytes_hex": self.SECOND},
            ]
        return {"id": report_id, "completion": completion,
                "writebacks": writebacks, "allocations": allocations}

    def test_accepts_the_reviewed_dual_output_observation(self):
        report = self.reviewed_report()
        self.assertEqual(run_native.validate_mrt_selftest(report),
                         [self.FIRST, self.SECOND])

    def test_rejects_the_sentinel_in_either_location(self):
        for first, second in (
            (self.SENTINEL, self.SECOND),
            (self.FIRST, self.SENTINEL),
            (self.SENTINEL, self.SENTINEL),
        ):
            with self.subTest(first=first, second=second):
                report = self.reviewed_report(
                    writebacks=[
                        {"allocation": 900, "view": 910, "offset": 0,
                         "bytes_hex": first},
                        {"allocation": 901, "view": 911, "offset": 0,
                         "bytes_hex": second},
                    ],
                    allocations=[
                        {"allocation": 900, "bytes_hex": first},
                        {"allocation": 901, "bytes_hex": second},
                    ])
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_mrt_selftest(report)

    def test_rejects_swapped_locations(self):
        report = self.reviewed_report(
            writebacks=[
                {"allocation": 900, "view": 910, "offset": 0,
                 "bytes_hex": self.SECOND},
                {"allocation": 901, "view": 911, "offset": 0,
                 "bytes_hex": self.FIRST},
            ],
            allocations=[
                {"allocation": 900, "bytes_hex": self.SECOND},
                {"allocation": 901, "bytes_hex": self.FIRST},
            ])
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_mrt_selftest(report)

    def test_rejects_the_plain_render_selftest_report(self):
        # The single-output self-test reaches the same location-0 bytes, so
        # only the id and the second location distinguish it; a report from
        # that self-test must not pass here.
        report = {
            "id": "render_offscreen_2x2",
            "completion": "CompletedVisible",
            "writebacks": [{"allocation": 900, "view": 910, "offset": 0,
                            "bytes_hex": self.FIRST}],
            "allocations": [{"allocation": 900, "bytes_hex": self.FIRST}],
        }
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_mrt_selftest(report)

    def test_rejects_a_missing_or_extra_observation(self):
        first_writeback = {"allocation": 900, "view": 910, "offset": 0,
                           "bytes_hex": self.FIRST}
        second_writeback = {"allocation": 901, "view": 911, "offset": 0,
                            "bytes_hex": self.SECOND}
        first_allocation = {"allocation": 900, "bytes_hex": self.FIRST}
        second_allocation = {"allocation": 901, "bytes_hex": self.SECOND}
        for writebacks, allocations in (
            ([], []),
            ([first_writeback], []),
            ([], [first_allocation]),
            ([first_writeback, second_writeback], [first_allocation]),
            ([first_writeback], [first_allocation, second_allocation]),
            ([first_writeback, second_writeback, second_writeback],
             [first_allocation, second_allocation]),
        ):
            with self.subTest(writebacks=writebacks, allocations=allocations):
                report = self.reviewed_report(writebacks=writebacks,
                                              allocations=allocations)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_mrt_selftest(report)

    def test_rejects_a_wrong_writeback_identity_or_offset(self):
        base = [
            {"allocation": 900, "view": 910, "offset": 0, "bytes_hex": self.FIRST},
            {"allocation": 901, "view": 911, "offset": 0, "bytes_hex": self.SECOND},
        ]
        for index, change in ((0, {"offset": 4}), (0, {"view": 911}),
                              (1, {"view": 910}), (1, {"allocation": 900})):
            with self.subTest(index=index, change=change):
                writebacks = [dict(entry) for entry in base]
                writebacks[index] = dict(writebacks[index], **change)
                report = self.reviewed_report(writebacks=writebacks)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_mrt_selftest(report)

    def test_rejects_a_non_visible_completion(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_mrt_selftest(
                self.reviewed_report(completion="Submitted"))

    def test_rejects_a_non_object_report(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_mrt_selftest([])


class StoreDontCareSelftestValidationTests(unittest.TestCase):
    """The store-dontcare self-test's byte comparison, exercised without Metal.

    `run_native.validate_store_dontcare_selftest` is the function the CI step
    reuses, so the observation-shape rule — the stored location reports, the
    discarded location disappears — is pinned here rather than only in an
    inline heredoc. The fixture id is part of the claim: the stored location's
    bytes are the same four texels the MRT self-test stores at location 0, so
    only the id and the absence of location 1's observation separate the two.
    """

    TARGET = "4080c0ff" * 4
    DISCARDED = "ff8040c0" * 4

    STORED_WRITEBACK = {"allocation": 900, "view": 910, "offset": 0,
                        "bytes_hex": TARGET}
    STORED_ALLOCATION = {"allocation": 900, "bytes_hex": TARGET}
    DISCARDED_WRITEBACK = {"allocation": 920, "view": 930, "offset": 0,
                           "bytes_hex": DISCARDED}
    DISCARDED_ALLOCATION = {"allocation": 920, "bytes_hex": DISCARDED}

    def reviewed_report(self, report_id="discard_second_attachment_2x2",
                        completion="CompletedVisible", writebacks=None,
                        allocations=None):
        if writebacks is None:
            writebacks = [dict(self.STORED_WRITEBACK)]
        if allocations is None:
            allocations = [dict(self.STORED_ALLOCATION)]
        return {"id": report_id, "completion": completion,
                "writebacks": writebacks, "allocations": allocations}

    def test_accepts_the_reviewed_discard_observation(self):
        report = self.reviewed_report()
        self.assertEqual(run_native.validate_store_dontcare_selftest(report),
                         self.TARGET)

    def test_rejects_a_reported_discarded_attachment(self):
        # The falsifiability point of the increment: a report that reads the
        # discarded location back must not pass. Allocation 920/view 930 has
        # to disappear from the observation, in either channel
        # (`research/docs/23` §3.6, v19).
        for writebacks, allocations in (
            ([dict(self.STORED_WRITEBACK), dict(self.DISCARDED_WRITEBACK)],
             [dict(self.STORED_ALLOCATION)]),
            ([dict(self.STORED_WRITEBACK)],
             [dict(self.STORED_ALLOCATION), dict(self.DISCARDED_ALLOCATION)]),
        ):
            with self.subTest(writebacks=writebacks, allocations=allocations):
                report = self.reviewed_report(writebacks=writebacks,
                                              allocations=allocations)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_store_dontcare_selftest(report)

    def test_rejects_the_mrt_selftest_report(self):
        # The MRT report stores the same location-0 bytes and also reports
        # location 1, so only the id distinguishes the two; a report from that
        # self-test must not pass here.
        report = {
            "id": "mrt_dual_output_2x2",
            "completion": "CompletedVisible",
            "writebacks": [dict(self.STORED_WRITEBACK),
                           dict(self.DISCARDED_WRITEBACK)],
            "allocations": [dict(self.STORED_ALLOCATION),
                            dict(self.DISCARDED_ALLOCATION)],
        }
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_store_dontcare_selftest(report)

    def test_rejects_a_missing_or_extra_observation(self):
        stored_writeback = dict(self.STORED_WRITEBACK)
        stored_allocation = dict(self.STORED_ALLOCATION)
        for writebacks, allocations in (
            ([], []),
            ([stored_writeback], []),
            ([], [stored_allocation]),
            ([stored_writeback, stored_writeback], [stored_allocation]),
            ([stored_writeback], [stored_allocation, stored_allocation]),
        ):
            with self.subTest(writebacks=writebacks, allocations=allocations):
                report = self.reviewed_report(writebacks=writebacks,
                                              allocations=allocations)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_store_dontcare_selftest(report)

    def test_rejects_a_wrong_writeback_identity_or_offset(self):
        for change in ({"offset": 4}, {"view": 911}, {"allocation": 901}):
            with self.subTest(change=change):
                writeback = [dict(self.STORED_WRITEBACK, **change)]
                report = self.reviewed_report(writebacks=writeback)
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_store_dontcare_selftest(report)

    def test_rejects_a_non_visible_completion(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_store_dontcare_selftest(
                self.reviewed_report(completion="Submitted"))

    def test_rejects_a_non_object_report(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_store_dontcare_selftest([])


class StageBufferSelftestValidationTests(unittest.TestCase):
    """The stage-buffer self-test's byte comparison, exercised without Metal.

    `run_native.validate_stage_buffer_selftest` is the function the CI step
    reuses, so the three frames the fixture's own `[[buffer(0)]]` arguments
    produce — the reviewed pair, a swapped tint and the full-screen positions —
    are pinned here rather than only in an inline heredoc
    (`research/docs/23` §83, R9g).
    """

    SENTINEL = "fefefefe"
    REVIEWED_FRAME = "4080c0ff" + SENTINEL * 3
    SWAPPED_FRAME = "00ff00ff" + SENTINEL * 3
    FULL_FRAME = "4080c0ff" * 4
    POSITIONS = "000080bf0000803f0000803e0000803f000080bf000080be"
    FULL_POSITIONS = "000080bf000080bf00004040000080bf000080bf00004040"
    TINT = "8180803e8180003fc1c0403f0000803f"
    SWAPPED_TINT = "000000000000803f000000000000803f"

    def reviewed_report(self, report_id="stage_buffer_positions_2x2",
                        completion="CompletedVisible", writebacks=None,
                        allocations=None, observations=None):
        if writebacks is None:
            writebacks = [{"allocation": 900, "view": 910, "offset": 0,
                           "bytes_hex": self.REVIEWED_FRAME}]
        if allocations is None:
            allocations = [{"allocation": 900, "bytes_hex": self.REVIEWED_FRAME}]
        if observations is None:
            observations = [
                {"positions_hex": self.POSITIONS, "tint_hex": self.TINT,
                 "attachment_hex": self.REVIEWED_FRAME},
                {"positions_hex": self.POSITIONS, "tint_hex": self.SWAPPED_TINT,
                 "attachment_hex": self.SWAPPED_FRAME},
                {"positions_hex": self.FULL_POSITIONS, "tint_hex": self.TINT,
                 "attachment_hex": self.FULL_FRAME},
            ]
        return {"id": report_id, "completion": completion, "writebacks": writebacks,
                "allocations": allocations, "observations": observations}

    def test_accepts_the_reviewed_stage_buffer_runs(self):
        report = self.reviewed_report()
        self.assertEqual(
            run_native.validate_stage_buffer_selftest(report),
            " ".join([self.REVIEWED_FRAME, self.SWAPPED_FRAME, self.FULL_FRAME]))

    def test_rejects_the_clear_sentinel_in_either_channel(self):
        sentinel_frame = self.SENTINEL * 4
        for writeback, allocation in (
            (sentinel_frame, self.REVIEWED_FRAME),
            (self.REVIEWED_FRAME, sentinel_frame),
        ):
            with self.subTest(writeback=writeback, allocation=allocation):
                report = self.reviewed_report(
                    writebacks=[{"allocation": 900, "view": 910, "offset": 0,
                                 "bytes_hex": writeback}],
                    allocations=[{"allocation": 900, "bytes_hex": allocation}])
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_stage_buffer_selftest(report)

    def test_rejects_a_run_that_landed_the_sentinel_or_a_foreign_frame(self):
        # A sentinel frame means no binding arrived; the reviewed frame in the
        # swapped run, or the swapped frame in the full-screen run, means the
        # run's own bytes were not what the draw read.
        for index, attachment in ((0, self.SENTINEL * 4), (1, self.REVIEWED_FRAME),
                                  (2, self.SWAPPED_FRAME)):
            with self.subTest(index=index, attachment=attachment):
                observations = self.reviewed_report()["observations"]
                observations[index]["attachment_hex"] = attachment
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_stage_buffer_selftest(
                        self.reviewed_report(observations=observations))

    def test_rejects_a_run_that_names_no_payload(self):
        for field in ("positions_hex", "tint_hex"):
            with self.subTest(field=field):
                observations = self.reviewed_report()["observations"]
                observations[1][field] = ""
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_stage_buffer_selftest(
                        self.reviewed_report(observations=observations))

    def test_rejects_a_missing_or_extra_run(self):
        observations = self.reviewed_report()["observations"]
        for runs in (observations[:2], observations + [observations[0]]):
            with self.subTest(runs=len(runs)):
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_stage_buffer_selftest(
                        self.reviewed_report(observations=runs))

    def test_rejects_the_plain_render_selftest_report(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_stage_buffer_selftest(
                self.reviewed_report(report_id="render_offscreen_2x2"))

    def test_rejects_a_non_visible_completion(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_stage_buffer_selftest(
                self.reviewed_report(completion="Submitted"))

    def test_rejects_a_non_object_report(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_stage_buffer_selftest([])


class StageBufferWriteSelftestValidationTests(unittest.TestCase):
    """The writable stage-buffer self-test's byte comparison, exercised without
    Metal (`research/docs/23` §92, R9k).

    `run_native.validate_stage_buffer_write_selftest` is the function the CI
    step reuses, so the five readings each run carries — the bound positions and
    source, the attachment the fragment returned, the sink the stage wrote and
    the accumulator it read and wrote — are pinned here rather than only in an
    inline heredoc.
    """

    SENTINEL = "fefefefe"
    REVIEWED_FRAME = "4080c0ff" + SENTINEL * 3
    FULL_FRAME = "00ff00ff" * 4
    POSITIONS = "000080bf0000803f0000803e0000803f000080bf000080be"
    FULL_POSITIONS = "000080bf000080bf00004040000080bf000080bf00004040"
    SOURCE = "8180803e8180003fc1c0403f0000803f"
    GREEN = "000000000000803f000000000000803f"
    ACCUMULATOR_INITIAL = "0000803e" * 4
    ACCUMULATOR_REVIEWED = "0000a03f" * 4
    ACCUMULATOR_GREEN = "0000a03f00000040" + "0000a03f00000040"

    def reviewed_report(self, report_id="stage_buffer_write_2x2",
                        completion="CompletedVisible", writebacks=None,
                        allocations=None, observations=None):
        if writebacks is None:
            writebacks = [
                {"allocation": 900, "view": 910, "offset": 0,
                 "bytes_hex": self.REVIEWED_FRAME},
                {"allocation": 901, "view": 911, "offset": 0,
                 "bytes_hex": self.SOURCE},
                {"allocation": 902, "view": 912, "offset": 0,
                 "bytes_hex": self.ACCUMULATOR_REVIEWED},
            ]
        if allocations is None:
            allocations = [
                {"allocation": 900, "bytes_hex": self.REVIEWED_FRAME},
                {"allocation": 901, "bytes_hex": self.SOURCE},
                {"allocation": 902, "bytes_hex": self.ACCUMULATOR_REVIEWED},
            ]
        if observations is None:
            observations = [
                {"positions_hex": self.POSITIONS, "source_hex": self.SOURCE,
                 "accumulator_initial_hex": self.ACCUMULATOR_INITIAL,
                 "attachment_hex": self.REVIEWED_FRAME, "sink_hex": self.SOURCE,
                 "accumulator_hex": self.ACCUMULATOR_REVIEWED},
                {"positions_hex": self.FULL_POSITIONS, "source_hex": self.GREEN,
                 "accumulator_initial_hex": self.ACCUMULATOR_INITIAL,
                 "attachment_hex": self.FULL_FRAME, "sink_hex": self.GREEN,
                 "accumulator_hex": self.ACCUMULATOR_GREEN},
            ]
        return {"id": report_id, "completion": completion, "writebacks": writebacks,
                "allocations": allocations, "observations": observations}

    def test_accepts_the_reviewed_writable_stage_buffer_runs(self):
        report = self.reviewed_report()
        self.assertEqual(
            run_native.validate_stage_buffer_write_selftest(report),
            "frames=[" + self.REVIEWED_FRAME + ", " + self.FULL_FRAME
            + "] sinks=[" + self.SOURCE + ", " + self.GREEN
            + "] accumulators=[" + self.ACCUMULATOR_REVIEWED + ", "
            + self.ACCUMULATOR_GREEN + "]")

    def test_rejects_a_frame_that_landed_no_source_or_a_foreign_one(self):
        # The sentinel frame means the fragment's source never arrived; the
        # other run's frame means the run's own bytes were not what the draw
        # returned.
        for index, attachment in ((0, self.SENTINEL * 4), (0, self.FULL_FRAME),
                                  (1, self.REVIEWED_FRAME)):
            with self.subTest(index=index, attachment=attachment):
                observations = self.reviewed_report()["observations"]
                observations[index]["attachment_hex"] = attachment
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_stage_buffer_write_selftest(
                        self.reviewed_report(observations=observations))

    def test_rejects_a_sink_that_was_not_written(self):
        # The sink starts as zeros, so a rail that executed the pass but landed
        # nothing reports them; a rail that bound the wrong slot reports the
        # other run's payload.
        for sink in ("00" * 16, self.GREEN):
            with self.subTest(sink=sink):
                observations = self.reviewed_report()["observations"]
                observations[0]["sink_hex"] = sink
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_stage_buffer_write_selftest(
                        self.reviewed_report(observations=observations))

    def test_rejects_an_accumulator_that_never_read_its_previous_bytes(self):
        # One alone (`0000803f` / `00000040`) is what a rail that bound zeros
        # publishes; the initial bytes are what the read half added to.
        for accumulator in ("0000803f" * 4,
                            "0000803f000000400000803f00000040"):
            with self.subTest(accumulator=accumulator):
                observations = self.reviewed_report()["observations"]
                observations[0]["accumulator_hex"] = accumulator
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_stage_buffer_write_selftest(
                        self.reviewed_report(observations=observations))

    def test_rejects_a_run_that_names_no_bound_payload(self):
        for field in ("positions_hex", "source_hex", "accumulator_initial_hex"):
            with self.subTest(field=field):
                observations = self.reviewed_report()["observations"]
                observations[1][field] = ""
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_stage_buffer_write_selftest(
                        self.reviewed_report(observations=observations))

    def test_rejects_a_missing_sink_or_a_moved_identity(self):
        # The writeback shape is part of the claim: one entry per landing, the
        # attachment first and the two writable bindings behind it, each at its
        # own view and offset.
        report = self.reviewed_report()
        for writebacks in (
            report["writebacks"][:2],
            report["writebacks"] + [{"allocation": 903, "view": 913, "offset": 0,
                                     "bytes_hex": self.SOURCE}],
            [report["writebacks"][0],
             dict(report["writebacks"][1], view=912),
             report["writebacks"][2]],
            [report["writebacks"][0],
             dict(report["writebacks"][1], offset=4),
             report["writebacks"][2]],
        ):
            with self.subTest(writebacks=writebacks):
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_stage_buffer_write_selftest(
                        self.reviewed_report(writebacks=writebacks))

    def test_rejects_a_missing_or_extra_run(self):
        observations = self.reviewed_report()["observations"]
        for runs in (observations[:1], observations + [observations[0]]):
            with self.subTest(runs=len(runs)):
                with self.assertRaises(run_native.NativeRunError):
                    run_native.validate_stage_buffer_write_selftest(
                        self.reviewed_report(observations=runs))

    def test_rejects_the_read_only_stage_buffer_selftest_report(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_stage_buffer_write_selftest(
                self.reviewed_report(report_id="stage_buffer_positions_2x2"))

    def test_rejects_a_non_visible_completion(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_stage_buffer_write_selftest(
                self.reviewed_report(completion="Submitted"))

    def test_rejects_a_non_object_report(self):
        with self.assertRaises(run_native.NativeRunError):
            run_native.validate_stage_buffer_write_selftest([])


if __name__ == "__main__":
    unittest.main()
