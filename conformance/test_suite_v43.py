"""The pass-entry snapshot arm's own suite (`research/docs/23` §118, E-TX15).

`suite-v43.json` carries one declaring pass and one translated render case: the
pass loads its 4x4 `rgba8_unorm` attachment from the caller's own bytes, clips
its draw to the attachment's left half, and its sampled declaration names that
very attachment through the new arm — the read is the bytes the attachment
holds when the pass opens, taken by a device-side copy before the render pass
starts. The frame is therefore two readings at once: the drawn columns carry
the fragment module's own reading of the entry bytes (`c04000ff`, the two fixed
texel centres' reds), while the columns the clip leaves alone keep the entry
bytes themselves.

These are comparator and schema checks, not GPU execution evidence: the
Lavapipe readings live in
`crates/metal-api-vulkan/tests/render_pass_entry_snapshot_e2e.rs` and in the
capture archives the evidence directory keeps.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import compare


CONFORMANCE = Path(__file__).resolve().parent
V43_PATH = CONFORMANCE / "suite-v43.json"

DECLARING_ID = "render_declaring_pass_entry_snapshot"
CASE_ID = "pass_entry_snapshot_4x4"

ATTACHMENT = (900, 910)
ATTACHMENT_BYTES = 64
# The attachment's entry content: column `i` holds `64 * i` in red and row `j`
# holds `0x11 * (j + 1)` in green, so every texel is its own word.
ENTRY_HEX = "".join(
    f"{64 * i:02x}{0x11 * (j + 1):02x}00ff" for j in range(4) for i in range(4)
)
# The drawn half: the translated module's two fixed samples read entry texel
# (3, 0)'s red (`c0`) and entry texel (1, 0)'s red (`40`).
DRAWN_HEX = "c04000ff"
# The frame: two drawn columns beside the two the clip leaves to the load.
FRAME_HEX = "".join(
    DRAWN_HEX * 2 + f"{0x80:02x}{0x11 * (j + 1):02x}00ff" + f"{0xc0:02x}{0x11 * (j + 1):02x}00ff"
    for j in range(4)
)

VERTEX_SHA = "be63f8b9efc815f929706fd5ee42e71ec3d12aa9b10d7444e0af8caf67755ddc"
FRAGMENT_SHA = "96c4de8def6de8d3776fef860d39638d22e71eac2db53cdf007fe27513cfeb8d"

VULKAN_RAILS = ("vulkan",)
SNAPSHOT_COPIES = 1


class PassEntrySnapshotSuiteTests(unittest.TestCase):
    def setUp(self):
        self.raw = V43_PATH.read_bytes()
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

    def refused(self, mutate, message):
        suite = copy.deepcopy(self.suite)
        mutate(self.case(CASE_ID, suite))
        with self.assertRaisesRegex(compare.CaptureError, message):
            compare._render_plan(compare._suite_plan(suite), suite)

    def test_the_suite_carries_the_declaring_pass_and_the_snapshot_case(self):
        self.assertEqual(self.suite["suite"], "compute-buffer-v43")
        self.assertEqual([case["id"] for case in self.suite["cases"]], [DECLARING_ID])
        self.assertEqual([case["id"] for case in self.suite["render_cases"]], [CASE_ID])
        declaring = self.suite["cases"][0]
        self.assertEqual(declaring["entry"], "copy_word")
        for kind in ("air", "metal"):
            source = CONFORMANCE / declaring[kind]["path"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                declaring[kind]["sha256"],
                kind,
            )
        buffers = {buffer["binding"]: buffer for buffer in declaring["buffers"]}
        self.assertEqual(buffers[0]["allocation"], ATTACHMENT[0])
        self.assertEqual(buffers[0]["view"], ATTACHMENT[1])
        self.assertEqual(buffers[0]["access"], "read")
        self.assertEqual(buffers[0]["length"], ATTACHMENT_BYTES)
        self.assertEqual(buffers[0]["initial_hex"], ENTRY_HEX)
        self.assertEqual(
            declaring["expected_writebacks"][0]["bytes_hex"], ENTRY_HEX[:8]
        )
        case = self.case(CASE_ID)
        self.assertEqual(case["declaring_case"], DECLARING_ID)
        self.assertEqual(case["vertices"], 3)
        self.assertEqual(case["viewport"], [0, 0, 4, 4])
        self.assertEqual(case["capture_rails"], list(VULKAN_RAILS))

    def test_the_declaration_names_the_pass_own_attachment(self):
        case = self.case(CASE_ID)
        texture = case["fragment_textures"][0]
        self.assertEqual(texture["source"], "pass_entry_snapshot")
        self.assertEqual((texture["allocation"], texture["view"]), ATTACHMENT)
        self.assertEqual(texture["format"], case["attachment"]["format"])
        self.assertEqual(texture["width"], case["attachment"]["width"])
        self.assertEqual(texture["height"], case["attachment"]["height"])
        self.assertNotIn("initial_hex", texture)
        self.assertNotIn("texel_rule", texture)
        self.assertEqual(case["attachment"]["load"], "load")
        self.assertEqual(case["attachment"]["initial_hex"], ENTRY_HEX)
        self.assertEqual(case["scissor"], [0, 0, 2, 4])

    def test_the_two_air_modules_are_the_declared_fixtures(self):
        case = self.case(CASE_ID)
        self.assertEqual(case["vertex_entry"], "render_fullscreen_triangle")
        self.assertEqual(case["fragment_entry"], "render_sample_texture_2d")
        for stage, digest in (("vertex", VERTEX_SHA), ("fragment", FRAGMENT_SHA)):
            declared = case["translated_stages"][stage]
            self.assertEqual(declared["sha256"], digest, stage)
            source = CONFORMANCE / declared["path"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                digest,
                f"{stage}: the AIR fixture on disk is the one the case pins",
            )

    def test_the_expectation_carries_both_halves_of_the_frame(self):
        case = self.case(CASE_ID)
        self.assertEqual(case["expected_hex"], FRAME_HEX)
        self.assertNotEqual(case["expected_hex"], ENTRY_HEX)
        self.assertNotEqual(case["expected_hex"], DRAWN_HEX * 16)
        entry = bytes.fromhex(ENTRY_HEX)
        frame = bytes.fromhex(FRAME_HEX)
        x, y, width, height = case["scissor"]
        for position in range(16):
            column = position % 4
            row = position // 4
            texel = frame[position * 4:position * 4 + 4]
            if x <= column < x + width and y <= row < y + height:
                self.assertEqual(texel.hex(), DRAWN_HEX)
            else:
                self.assertEqual(texel, entry[position * 4:position * 4 + 4])
        self.assertNotIn(
            bytes.fromhex(DRAWN_HEX),
            [entry[offset:offset + 4] for offset in range(0, len(entry), 4)],
            "the fragment's reading has to be unlike every entry texel",
        )

    def test_the_plan_carries_the_attachment_and_the_snapshot_reading(self):
        plan = self.plan(CASE_ID)
        self.assertEqual(
            plan.writes,
            [((ATTACHMENT[0], ATTACHMENT[1], 0), bytes.fromhex(FRAME_HEX))],
        )
        self.assertEqual(plan.allocations, {ATTACHMENT[0]: bytes.fromhex(FRAME_HEX)})
        self.assertEqual(plan.rails, set(VULKAN_RAILS))
        self.assertEqual(plan.texture_uploads, 0, "the declaration uploads no bytes")
        self.assertEqual(plan.snapshots, (SNAPSHOT_COPIES, ATTACHMENT_BYTES))

    def test_the_declaration_has_to_name_the_pass_own_attachment(self):
        def retarget(case):
            case["fragment_textures"][0]["view"] = 911

        self.refused(
            retarget,
            "a pass-entry snapshot declaration names the pass's own attachment",
        )

    def test_the_declaration_has_to_restate_the_attachment_shape(self):
        def reshape(case):
            case["fragment_textures"][0]["width"] = 2

        self.refused(
            reshape,
            "the sampled texture has to share the attachment's extent",
        )

    def test_the_declaration_may_not_carry_bytes(self):
        def carry(case):
            case["fragment_textures"][0]["initial_hex"] = ENTRY_HEX

        self.refused(
            carry,
            "a pass-entry snapshot declaration carries no bytes",
        )

    def test_the_load_arm_has_to_establish_the_entry_content(self):
        def clear(case):
            case["attachment"].pop("initial_hex")
            case["attachment"]["load"] = "clear"
            case["attachment"]["clear_hex"] = "fefefefe"

        self.refused(clear, "the pass-entry snapshot reads the bytes the attachment holds")

    def test_the_frame_may_not_be_the_entry_content_itself(self):
        def identity(case):
            case["expected_hex"] = ENTRY_HEX

        self.refused(
            identity,
            "the drawn texels disagree about the fragment's reading",
        )

    def test_the_frame_may_not_repeat_one_word_over_the_drawn_half_only(self):
        def collapse(case):
            # The kept half still carries the entry bytes, but the drawn half
            # claims a word the entry already holds: "the snapshot copy ran"
            # and "the load ran" then land the same frame.
            drawn = ENTRY_HEX[8:16]
            case["expected_hex"] = "".join(
                drawn * 2
                + ENTRY_HEX[(4 * j + 2) * 8:(4 * j + 3) * 8]
                + ENTRY_HEX[(4 * j + 3) * 8:(4 * j + 4) * 8]
                for j in range(4)
            )

        self.refused(
            collapse,
            "the fragment's reading equals an entry texel",
        )

    def test_the_clip_is_this_shape_own_falsifier(self):
        def unclipped(case):
            case.pop("scissor")

        self.refused(
            unclipped,
            "the pass-entry snapshot case clips its draw to a strict sub-rectangle",
        )

    def test_the_marker_stays_inside_the_rail_that_executes_the_arm(self):
        def widen(case):
            case["capture_rails"] = ["vulkan", "vulkan-objects"]

        self.refused(
            widen,
            "a pass-entry snapshot case runs on the rail whose texture walk resolves the arm",
        )

    def test_the_reviewed_sampling_shape_still_refuses_the_arm_without_the_source(self):
        # The same case with the source field dropped is the *reviewed* sampling
        # shape, whose source is the bytes the case carries — and whose load arm
        # is the clear. The comparator refuses it, so the new arm cannot be
        # reached by an older suite entry that never named it.
        def unname(case):
            case["fragment_textures"][0].pop("source")
            case["attachment"].pop("initial_hex")
            case["attachment"]["load"] = "clear"
            case["attachment"]["clear_hex"] = "fefefefe"

        self.refused(unname, "the reviewed sampling shape carries no .*scissor|"
                             "the reviewed sampling shape clears and stores|"
                             "the gathered arm|the reviewed sampling shape")


if __name__ == "__main__":
    unittest.main()
