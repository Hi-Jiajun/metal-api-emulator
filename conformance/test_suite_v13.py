"""Attachment-section checks for the first render suite.

These tests are comparator and schema checks, not GPU execution evidence. What
they pin is the observation surface `research/docs/23` §5.2 adds: a render case
reports one colour attachment's texels through the same writebacks/allocations
shape every compute case uses, its declaring pass declares that view and stays
read-only, and the attachment's bytes cannot be satisfied by - or reported as -
a buffer writeback.
"""

import copy
import hashlib
import json
from pathlib import Path
import re
import unittest

import compare
from test_compare import synthetic_report


CONFORMANCE = Path(__file__).resolve().parent
REPOSITORY = CONFORMANCE.parent
V13_PATH = CONFORMANCE / "suite-v13.json"
PROVIDER_PATH = REPOSITORY / "examples" / "metal-smoke" / "src" / "bin" / "provider-capture.rs"
SPV_DIRECTORY = REPOSITORY / "crates" / "metal-api-vulkan" / "src" / "render_spv"

DECLARING_ID = "render_declaring_copy_word"
RENDER_ID = "offscreen_triangle_clear_2x2"
ATTACHMENT = (900, 910, 0, 16)
PROBE = (920, 930, 4, 4)
TEXELS = "4080c0ff" * 4

# The capture backends `suite-v13.json` marks this case executable on
# (`conformance/RENDER-CAPTURE.md` §4): every rail that owns a render execution
# path has to report the attachment. All five backends own one now: the Vulkan
# object rail gained a render command encoder first, the native object rail
# followed (`8793b7a`).
REPORTING_RAILS = ("vulkan", "vulkan-objects", "native-metal", "native-metal-provider",
                   "native-metal-provider-objects")

# Read by hand from suite-v13.json: the declaring pass copies the attachment
# view's first word (`fe fe fe fe`, the clear sentinel) into the probe view, and
# the render pass then stores the reviewed fragment output into all four texels.
DECLARING_WRITEBACK = {"allocation": 920, "view": 930, "offset": 4,
                       "bytes_hex": "fefefefe"}
RENDER_WRITEBACK = {"allocation": 900, "view": 910, "offset": 0, "bytes_hex": TEXELS}


def render_result(provider_backend=True):
    """The attachment observation a reporting rail of this suite has to emit.

    The Swift oracle reports bytes without device-buffer copy counters, so the
    two surfaces a provider rail also carries stay absent for `native-metal`
    exactly as they do on every compute case.
    """
    result = {
        "id": RENDER_ID,
        "completion": "CompletedVisible",
        "writebacks": [copy.deepcopy(RENDER_WRITEBACK)],
        "allocations": [{"allocation": 900, "bytes_hex": TEXELS}],
    }
    if provider_backend:
        result["copy_in"], result["copy_out"] = 2, 2
    return result


def synthetic_capture(suite, digest, backend="vulkan"):
    """A capture with the hand-written attachment observation attached."""
    report = synthetic_report(suite, digest, backend)
    if backend != "native-metal":
        for result in report["results"]:
            result["copy_in"], result["copy_out"] = 2, 1
    if backend in REPORTING_RAILS:
        report["results"].append(render_result(backend != "native-metal"))
    return report


def suite_without_marker_rail(suite, rail):
    """A copy of `suite` whose render-case marker omits `rail`.

    The committed suites name every rail, so the "a rail the marker does not
    name must not report the case" rule is exercised against a fixture whose
    marker drops one rail rather than against a committed suite that no longer
    exists in that shape.
    """
    trimmed = copy.deepcopy(suite)
    for case in trimmed.get("render_cases", []):
        case["capture_rails"] = [named for named in case["capture_rails"] if named != rail]
    digest = hashlib.sha256(json.dumps(trimmed, sort_keys=True).encode("utf-8")).hexdigest()
    return trimmed, digest


class RenderObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = V13_PATH.read_bytes()
        self.suite = json.loads(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_v13_declares_one_render_case_and_its_declaring_pass(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v13")
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], [RENDER_ID])
        case = self.suite["render_cases"][0]
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual(case["capture_rails"], list(REPORTING_RAILS))
        self.assertEqual(case["vertices"], 3)
        self.assertEqual(case["viewport"], [0, 0, 2, 2])
        self.assertEqual((case["vertex_entry"], case["fragment_entry"]),
                         ("render_fullscreen_triangle", "render_solid_rgba8"))
        attachment = case["attachment"]
        # Field for field with `metal_api_core::provider::RenderAttachment` and
        # `RenderPassDescriptor`: identity, format, 2x2 extent, load and store.
        self.assertEqual((attachment["allocation"], attachment["view"]),
                         (ATTACHMENT[0], ATTACHMENT[1]))
        self.assertEqual((attachment["format"], attachment["width"], attachment["height"]),
                         ("rgba8_unorm", 2, 2))
        self.assertEqual((attachment["load"], attachment["store"]), ("clear", "store"))
        self.assertEqual(attachment["clear_hex"], "fefefefe")
        self.assertNotIn("initial_hex", attachment)
        # The expectation covers every texel with the fragment output, and it
        # differs from the clear sentinel the pass started from.
        expected = bytes.fromhex(case["expected_hex"])
        self.assertEqual(len(expected), 16)
        self.assertEqual(set(expected[i:i + 4] for i in range(0, 16, 4)),
                         {bytes.fromhex("4080c0ff")})
        self.assertNotEqual(expected[:4], bytes.fromhex(attachment["clear_hex"]))

    def test_v13_declaring_pass_declares_the_attachment_view_read_only(self):
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["entry"], "copy_word")
        self.assertEqual([declaring["grid"], declaring["local"]], [[1, 1, 1], [1, 1, 1]])
        buffers = {buffer["binding"]: buffer for buffer in declaring["buffers"]}
        self.assertEqual(buffers[0]["access"], "read")
        # The attachment is its own allocation: the render rail reads the image
        # back directly, so the guard-byte discipline of a compute case does not
        # apply to this view, and the observed allocation *is* the attachment.
        self.assertEqual((buffers[0]["allocation"], buffers[0]["view"], buffers[0]["offset"],
                          buffers[0]["length"], buffers[0]["allocation_size"]),
                         ATTACHMENT + (16,))
        self.assertEqual(buffers[0]["allocation_size"], buffers[0]["length"])
        self.assertEqual(buffers[1]["access"], "write")
        self.assertEqual((buffers[1]["allocation"], buffers[1]["view"], buffers[1]["offset"],
                          buffers[1]["length"], buffers[1]["allocation_size"]),
                         PROBE + (12,))
        # The declaring pass's own observation stays a buffer writeback: it is
        # the probe view, never the attachment.
        self.assertEqual(declaring["expected_writebacks"],
                         [copy.deepcopy(DECLARING_WRITEBACK)])
        for kind in ("air", "metal"):
            source = V13_PATH.parent / declaring[kind]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             declaring[kind]["sha256"], kind)

    def test_v13_pins_the_reviewed_render_module_and_the_executed_stages(self):
        case = self.suite["render_cases"][0]
        module = V13_PATH.parent / case["metal"]["path"]
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),
                         case["metal"]["sha256"])
        text = PROVIDER_PATH.read_text(encoding="utf-8")
        pinned = re.findall(r'include_bytes!\(\s*"([^"]*render_spv/[^"]+)"\s*\)', text)
        # Two reviewed pairs: the milestone's `vertex_id` triangle, and the
        # vertex-input quad (`research/docs/23` §3.3). Both are pinned by bytes
        # rather than by name, which is what keeps a renamed module from
        # slipping past the review.
        self.assertEqual(len(pinned), 4, "the capture pins two reviewed stage pairs")
        for relative in pinned:
            module = (PROVIDER_PATH.parent / relative).resolve()
            self.assertTrue(module.is_file(), "pinned stage module is missing: " + relative)
            self.assertEqual(module.parent, SPV_DIRECTORY)
        entries = re.findall(
            r'const (?:RENDER|QUAD)_(?:VERTEX|FRAGMENT)_ENTRY: &str = "([^"]+)";', text)
        self.assertEqual(entries,
                         ["vertex_main", "fragment_main", "vertex_buffer_main", "fragment_main"])
        vertex = (SPV_DIRECTORY / "fullscreen_triangle.vert.spv").read_bytes()
        quad = (SPV_DIRECTORY / "quad_indexed.vert.spv").read_bytes()
        # The unorm8 fragment module serves both Rgba8Unorm and Bgra8Unorm,
        # which is why it is not named after one layout (see `render.rs`).
        fragment = (SPV_DIRECTORY / "solid_unorm8.frag.spv").read_bytes()
        # The reviewed SPIR-V sources declare these entries; the MSL entry names
        # are what the native rails compile, so the two identities are distinct.
        self.assertIn(b"vertex_main", vertex)
        self.assertIn(b"vertex_buffer_main", quad)
        self.assertNotIn(b"vertex_main", quad)
        self.assertIn(b"fragment_main", fragment)
        self.assertNotIn(b"render_fullscreen_triangle", vertex)
        self.assertNotIn(b"render_solid_unorm8", fragment)

    def test_v13_plan_is_attachment_only_and_one_copy_per_allocation(self):
        plan = compare._suite_plan(self.suite)
        render_plan = compare._render_plan(plan, self.suite)
        expectation = render_plan[RENDER_ID]
        self.assertEqual(expectation.writes,
                         [((ATTACHMENT[0], ATTACHMENT[1], ATTACHMENT[2]),
                           bytes.fromhex(TEXELS))])
        self.assertEqual(expectation.allocations, {ATTACHMENT[0]: bytes.fromhex(TEXELS)})
        self.assertEqual(expectation.attachment, ATTACHMENT)
        # One allocation is written by the render pass, but two are touched by
        # the submission: the declaring pass uploads both.
        self.assertEqual(expectation.touched, {ATTACHMENT[0], PROBE[0]})
        self.assertEqual(expectation.written, {ATTACHMENT[0], PROBE[0]})
        self.assertEqual(expectation.rails, set(REPORTING_RAILS))

    def test_v13_every_rail_that_reports_the_attachment_validates(self):
        for backend in REPORTING_RAILS:
            with self.subTest(backend=backend):
                compare.validate_capture(self.suite, self.digest,
                                         synthetic_capture(self.suite, self.digest, backend),
                                         backend)

    def test_v13_refuses_tampered_attachment_bytes(self):
        for mutation, message in (
            # Channel order reversed: the same four texels, wrong bytes.
            ("ffc08040" * 4, "first differing byte"),
            # One texel short: a partially covered attachment cannot pass.
            ("4080c0ff" * 3, "length mismatch"),
        ):
            with self.subTest(mutation=mutation):
                report = synthetic_capture(self.suite, self.digest)
                for result in report["results"]:
                    if result["id"] == RENDER_ID:
                        result["writebacks"][0]["bytes_hex"] = mutation
                        result["allocations"][0]["bytes_hex"] = mutation
                with self.assertRaisesRegex(compare.CaptureError, message):
                    compare.validate_capture(self.suite, self.digest, report)

    def test_v13_refuses_a_buffer_writeback_where_the_attachment_belongs(self):
        # The render result reports the declaring pass's buffer landing instead
        # of the attachment.
        report = synthetic_capture(self.suite, self.digest)
        for result in report["results"]:
            if result["id"] == RENDER_ID:
                result["writebacks"] = [copy.deepcopy(DECLARING_WRITEBACK)]
                result["allocations"] = [{"allocation": 920,
                                          "bytes_hex": "00000000fefefefe00000000"}]
        with self.assertRaisesRegex(compare.CaptureError, "writable set mismatch"):
            compare.validate_capture(self.suite, self.digest, report)
        # The other direction: the declaring case claims the attachment.
        report = synthetic_capture(self.suite, self.digest)
        for result in report["results"]:
            if result["id"] == DECLARING_ID:
                result["writebacks"].append(copy.deepcopy(RENDER_WRITEBACK))
        with self.assertRaisesRegex(compare.CaptureError, "writable set mismatch"):
            compare.validate_capture(self.suite, self.digest, report)
        # An attachment has to be its own allocation: a capture that reports the
        # probe's image under the attachment's identity is refused.
        report = synthetic_capture(self.suite, self.digest)
        for result in report["results"]:
            if result["id"] == RENDER_ID:
                result["allocations"] = [{"allocation": ATTACHMENT[0],
                                          "bytes_hex": "00000000fefefefe00000000"}]
        with self.assertRaisesRegex(compare.CaptureError, "length mismatch"):
            compare.validate_capture(self.suite, self.digest, report)

    def test_v13_refuses_a_rail_the_marker_does_not_name(self):
        suite, digest = suite_without_marker_rail(self.suite, "native-metal-provider-objects")
        # Build the capture by hand: `synthetic_capture` appends the attachment
        # for every rail in REPORTING_RAILS, and this rail is named there even
        # though this trimmed copy's marker drops it.
        report = synthetic_report(suite, digest, "native-metal-provider-objects")
        report["results"].append(render_result())
        with self.assertRaisesRegex(compare.CaptureError, "not a rail this render case runs on"):
            compare.validate_capture(suite, digest, report)

    def test_v13_count_contract_is_one_copy_per_touched_and_written_allocation(self):
        for counts, message in (
            ((2, 2), None),
            # The attachment's readback is the attachment allocation's own
            # copy-out, so an extra copy-out has nothing to pay for.
            ((2, 3), "copy_out 3 does not match 2 written allocations"),
            ((2, 1), "copy_out 1 does not match 2 written allocations"),
            ((1, 2), "copy_in 1 does not match 2 touched allocations"),
            ((3, 2), "copy_in 3 does not match 2 touched allocations"),
        ):
            with self.subTest(counts=counts):
                report = synthetic_capture(self.suite, self.digest)
                for result in report["results"]:
                    if result["id"] == RENDER_ID:
                        result["copy_in"], result["copy_out"] = counts
                if message is None:
                    compare.validate_capture(self.suite, self.digest, report)
                else:
                    with self.assertRaisesRegex(compare.CaptureError, message):
                        compare.validate_capture(self.suite, self.digest, report)
        # The Swift reference oracle reports bytes without counters.
        compare.validate_capture(self.suite, self.digest,
                                 synthetic_capture(self.suite, self.digest, "native-metal"),
                                 "native-metal")

    def test_v13_plan_rejects_a_widened_render_shape(self):
        def rejected(mutate, message):
            suite = copy.deepcopy(self.suite)
            mutate(suite["render_cases"][0])
            with self.assertRaisesRegex(compare.CaptureError, message):
                compare.validate_capture(suite, self.digest,
                                         synthetic_capture(suite, self.digest))

        rejected(lambda case: case["attachment"].update(width=4, height=4),
                 "2x2 attachment")
        # The comparator admits both 8-bit UNORM layouts from v21 on and the
        # single-channel float from v22 (`research/docs/23` §3.3/§15); the
        # integer format is still unadmitted, so it is the probe for "an
        # unsupported format is refused".
        rejected(lambda case: case["attachment"].update(format="r32_uint"),
                 "unsupported attachment format")
        rejected(lambda case: case["attachment"].update(store="discard"),
                 "cannot be compared")
        rejected(lambda case: case["attachment"].update(clear_hex="4080c0ff"),
                 "clear colour equals the expected texel")
        rejected(lambda case: case["attachment"].update(initial_hex="00000000" * 4),
                 "carries no initial bytes")
        rejected(lambda case: case.update(vertices=4), "full-screen triangle")
        rejected(lambda case: case.update(viewport=[0, 0, 1, 2]), "viewport")
        rejected(lambda case: case.update(capture_rails=[]), "capture_rails")
        rejected(lambda case: case.update(capture_rails=["vulkan", "vulkan"]), "capture_rails")
        rejected(lambda case: case.update(capture_rails=["dozen"]), "capture_rails")
        rejected(lambda case: case.update(declaring_case="missing"), "unknown declaring case")
        rejected(lambda case: case.update(expected_hex="4080c0ff" * 3),
                 "do not match the attachment")
        rejected(lambda case: case.update(expected_hex="4080c0ff" * 3 + "ffc08040"),
                 "every texel")

    def test_v13_declaring_pass_cannot_write_the_attachment_view(self):
        # A compute pass that *wrote* the view the render pass stores would make
        # the two writers' order inexpressible (`research/docs/23` §3.6), so the
        # declaring pass is pinned read-only. The declaring case is kept valid as
        # a compute case — its writable view set grows with the mutation — so the
        # render plan's own rule is what refuses it.
        suite = copy.deepcopy(self.suite)
        suite["cases"][0]["buffers"][0]["access"] = "read_write"
        suite["cases"][0]["expected_writebacks"].insert(
            0, {"allocation": ATTACHMENT[0], "view": ATTACHMENT[1], "offset": ATTACHMENT[2],
                "bytes_hex": "fefefefe" * 4})
        with self.assertRaisesRegex(compare.CaptureError,
                                    "must only read the attachment view"):
            compare.validate_capture(suite, self.digest, synthetic_capture(suite, self.digest))


if __name__ == "__main__":
    unittest.main()
