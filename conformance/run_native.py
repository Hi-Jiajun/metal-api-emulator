#!/usr/bin/env python3
"""Probe native Metal, capture if eligible, and preserve explicit result status.

An unavailable GPU is an infrastructure outcome, never native parity success.
Once a device is eligible, compilation/execution/comparison failures are fatal.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from compare import CaptureError, _read_json, validate_capture


class NativeRunError(ValueError):
    pass


def validate_probe(probe):
    fields = {"schema_version", "kind", "platform", "device", "eligible", "reason",
              "supports_apple4", "has_unified_memory"}
    if not isinstance(probe, dict) or set(probe) != fields:
        raise NativeRunError("probe: unexpected fields")
    if type(probe["schema_version"]) is not int or probe["schema_version"] != 1:
        raise NativeRunError("probe: unsupported schema version")
    if probe["kind"] != "metal-device-probe":
        raise NativeRunError("probe: wrong report kind")
    if not isinstance(probe["platform"], str) or not probe["platform"].strip():
        raise NativeRunError("probe: missing platform")
    if any(type(probe[key]) is not bool for key in
           ("eligible", "supports_apple4", "has_unified_memory")):
        raise NativeRunError("probe: capability fields must be booleans")
    device = probe["device"]
    if device is not None and not isinstance(device, str):
        raise NativeRunError("probe: invalid device name")
    if device is None:
        if probe["supports_apple4"] or probe["has_unified_memory"]:
            raise NativeRunError("probe: no device but capabilities reported")
        reason = "no_default_device"
    elif device.strip() and probe["supports_apple4"] and probe["has_unified_memory"]:
        reason = "eligible"
    else:
        reason = "unsupported_features"
    if probe["eligible"] != (reason == "eligible") or probe["reason"] != reason:
        raise NativeRunError("probe: eligibility and capabilities disagree")


def validate_present_selftest(report):
    """One present self-test report carries exactly one writeback and one
    allocation, both the reviewed 2x2 present target (`4080c0ff` four times),
    and never the `fefefefe` sentinel (`research/docs/24` §6 Step 7).

    The CI step reuses this instead of inlining its byte comparison, so the
    comparison is exercised by `test_run_native.py` on a host without Metal.
    """
    if not isinstance(report, dict):
        raise NativeRunError("present selftest: report is not an object")
    if report.get("completion") != "CompletedVisible":
        raise NativeRunError("present selftest: completion is not CompletedVisible")
    writebacks = report.get("writebacks", [])
    allocations = report.get("allocations", [])
    expected = "4080c0ff" * 4
    # The shape is part of the claim: the reviewed fixture produces exactly one
    # writeback and one allocation, so a report that reached the same bytes by
    # another route (for example two writebacks and no allocation) is refused
    # rather than compared as if it were the same observation.
    if len(writebacks) != 1 or len(allocations) != 1:
        raise NativeRunError(
            "present selftest: expected one writeback and one allocation, got "
            + repr((len(writebacks), len(allocations)))
        )
    observed = [entry.get("bytes_hex") for entry in writebacks + allocations]
    if observed != [expected, expected]:
        raise NativeRunError(
            "present selftest: observed bytes " + repr(observed)
            + " do not match the reviewed target " + expected
        )
    return observed


def validate_heap_selftest(report):
    """One heap self-test report carries exactly one reviewed writeback and two
    reviewed allocations: the read buffer (16 `fe` bytes) and the write buffer
    whose first word `copy_word` overwrote with `fefefefe`, leaving the eight
    sentinel `ff` bytes after it. The write word must be `fefefefe`, never the
    `ffffffff` sentinel the buffer was preset with (`research/docs/25` §6
    Step 7a).

    The CI step reuses this instead of inlining its byte comparison, so the
    comparison is exercised by `test_run_native.py` on a host without Metal.
    """
    if not isinstance(report, dict):
        raise NativeRunError("heap selftest: report is not an object")
    device = report.get("device")
    platform = report.get("platform")
    if not isinstance(device, str) or not device.strip():
        raise NativeRunError("heap selftest: missing or empty device")
    if not isinstance(platform, str) or not platform.strip():
        raise NativeRunError("heap selftest: missing or empty platform")
    if report.get("completion") != "CompletedVisible":
        raise NativeRunError("heap selftest: completion is not CompletedVisible")
    writebacks = report.get("writebacks", [])
    allocations = report.get("allocations", [])
    expected_writeback = {
        "allocation": 920, "view": 930, "offset": 0, "bytes_hex": "fefefefe"
    }
    expected_allocations = [
        {"allocation": 900, "bytes_hex": "fefefefefefefefefefefefefefefefe"},
        {"allocation": 920, "bytes_hex": "fefefefeffffffffffffffff"},
    ]
    # The shape is part of the claim: the reviewed fixture produces exactly one
    # writeback and two allocations, so a report that reached the same bytes by
    # another route (for example one allocation and no writeback) is refused
    # rather than compared as if it were the same observation.
    if writebacks != [expected_writeback] or allocations != expected_allocations:
        raise NativeRunError(
            "heap selftest: observations do not match the reviewed shape, got "
            + repr((writebacks, allocations))
        )
    return writebacks[0]["bytes_hex"]


def validate_vertex_selftest(report):
    """One vertex-input self-test report is the reviewed indexed quad's
    observation: the fixture id, one writeback naming the attachment's own view
    (900/910) and one allocation of that view, both holding `4080c0ff` four
    times and never the `fefefefe` sentinel the pass started from
    (`research/docs/23` §6 Step 3.3, `conformance/RENDER-CAPTURE.md` §8).

    The fixture id is part of the check rather than decoration: the streams are
    inputs, so the attachment's bytes are the *same* four texels the plain
    `--render-selftest` reports, and the id is the only field that says those
    bytes were drawn through the caller-held stream and index buffer instead of
    `vertex_id`. The CI step reuses this instead of inlining its byte
    comparison, so the comparison is exercised by `test_run_native.py` on a host
    without Metal.
    """
    if not isinstance(report, dict):
        raise NativeRunError("vertex selftest: report is not an object")
    if report.get("id") != "vertex_quad_indexed_2x2":
        raise NativeRunError(
            "vertex selftest: report id " + repr(report.get("id"))
            + " is not the reviewed indexed fixture"
        )
    if report.get("completion") != "CompletedVisible":
        raise NativeRunError("vertex selftest: completion is not CompletedVisible")
    writebacks = report.get("writebacks", [])
    allocations = report.get("allocations", [])
    expected = "4080c0ff" * 4
    expected_writeback = {"allocation": 900, "view": 910, "offset": 0,
                          "bytes_hex": expected}
    expected_allocations = [{"allocation": 900, "bytes_hex": expected}]
    # The shape is part of the claim: the reviewed fixture reports exactly one
    # writeback and one allocation, both the attachment. The streams do not
    # appear in the observation because they are inputs.
    if writebacks != [expected_writeback] or allocations != expected_allocations:
        raise NativeRunError(
            "vertex selftest: observations do not match the reviewed indexed quad, got "
            + repr((writebacks, allocations))
        )
    return writebacks[0]["bytes_hex"]


def validate_mrt_selftest(report):
    """One MRT self-test report is the reviewed dual-output quad's observation:
    the fixture id, one writeback and one allocation per colour location,
    location 0's four `4080c0ff` texels and location 1's four `ff8040c0`
    texels, never the `fefefefe` clear sentinel the pass started from
    (`conformance/RENDER-CAPTURE.md` §10).

    The fixture id and the location order are part of the check rather than
    decoration: both locations are drawn through the same stream and index
    buffer, so only the id and the per-location bytes say the two outputs
    landed in the right attachments instead of being swapped or duplicated.
    The CI step reuses this instead of inlining its byte comparison, so the
    comparison is exercised by `test_run_native.py` on a host without Metal.
    """
    if not isinstance(report, dict):
        raise NativeRunError("mrt selftest: report is not an object")
    if report.get("id") != "mrt_dual_output_2x2":
        raise NativeRunError(
            "mrt selftest: report id " + repr(report.get("id"))
            + " is not the reviewed dual-output fixture"
        )
    if report.get("completion") != "CompletedVisible":
        raise NativeRunError("mrt selftest: completion is not CompletedVisible")
    writebacks = report.get("writebacks", [])
    allocations = report.get("allocations", [])
    first = "4080c0ff" * 4
    second = "ff8040c0" * 4
    expected_writebacks = [
        {"allocation": 900, "view": 910, "offset": 0, "bytes_hex": first},
        {"allocation": 901, "view": 911, "offset": 0, "bytes_hex": second},
    ]
    expected_allocations = [
        {"allocation": 900, "bytes_hex": first},
        {"allocation": 901, "bytes_hex": second},
    ]
    # The shape is part of the claim: the reviewed fixture reports exactly one
    # writeback and one allocation per location, in location order, so a report
    # that reached the same bytes by another route (for example swapped
    # locations or one writeback and no allocation) is refused rather than
    # compared as if it were the same observation.
    if writebacks != expected_writebacks or allocations != expected_allocations:
        raise NativeRunError(
            "mrt selftest: observations do not match the reviewed dual output, got "
            + repr((writebacks, allocations))
        )
    return [first, second]


def validate_stage_buffer_selftest(report):
    """One stage-buffer self-test report is the reviewed module's observation
    (`research/docs/23` §83, R9g): `shaders/render_stage_buffer_2x2.metal`
    compiled and run three times, each run binding its two `[[buffer(0)]]`
    arguments at the stages' own slots with `setVertexBuffer` and
    `setFragmentBuffer`.

    The three frames are the claim rather than decoration: the reviewed payload
    pair lands `4080c0ff` in the covered top-left texel and the `fefefefe`
    clear sentinel in the other three, a swapped tint moves that texel to
    `00ff00ff`, and the full-screen positions move the reviewed tint into all
    four texels. A rail that binds no buffer reads the sentinel everywhere, one
    that binds the wrong stage's buffer reads the other argument's bytes, and
    one that never reads the vertex stage cannot cover the frame — all three
    read back a frame this function refuses. The writeback/allocation pair
    repeats the first run's attachment in the shape every other self-test
    reports, so a report without it cannot pass.

    The comparison lives here instead of in the CI step's heredoc so
    `test_run_native.py` exercises it on a host without Metal.
    """
    if not isinstance(report, dict):
        raise NativeRunError("stage buffer selftest: report is not an object")
    if report.get("id") != "stage_buffer_positions_2x2":
        raise NativeRunError(
            "stage buffer selftest: report id " + repr(report.get("id"))
            + " is not the reviewed stage-buffer fixture"
        )
    if report.get("completion") != "CompletedVisible":
        raise NativeRunError("stage buffer selftest: completion is not CompletedVisible")
    sentinel = "fefefefe"
    reviewed_frame = "4080c0ff" + sentinel * 3
    swapped_frame = "00ff00ff" + sentinel * 3
    full_frame = "4080c0ff" * 4
    writebacks = report.get("writebacks", [])
    allocations = report.get("allocations", [])
    expected_writeback = {"allocation": 900, "view": 910, "offset": 0,
                          "bytes_hex": reviewed_frame}
    expected_allocations = [{"allocation": 900, "bytes_hex": reviewed_frame}]
    if writebacks != [expected_writeback] or allocations != expected_allocations:
        raise NativeRunError(
            "stage buffer selftest: observations do not match the reviewed run, got "
            + repr((writebacks, allocations))
        )
    observations = report.get("observations", [])
    if len(observations) != 3:
        raise NativeRunError(
            "stage buffer selftest: expected three runs (reviewed, swapped tint, "
            "full-screen positions), got " + repr(observations)
        )
    expected_attachments = [reviewed_frame, swapped_frame, full_frame]
    for index, (observation, expected) in enumerate(zip(observations, expected_attachments)):
        if not isinstance(observation, dict):
            raise NativeRunError(
                "stage buffer selftest: run " + str(index) + " is not an object"
            )
        if not observation.get("positions_hex") or not observation.get("tint_hex"):
            raise NativeRunError(
                "stage buffer selftest: run " + str(index)
                + " does not name both payloads it bound, got " + repr(observation)
            )
        if observation.get("attachment_hex") != expected:
            raise NativeRunError(
                "stage buffer selftest: run " + str(index) + " landed "
                + repr(observation.get("attachment_hex")) + " instead of " + repr(expected)
            )
    return " ".join(expected_attachments)


def validate_stage_buffer_write_selftest(report):
    """One writable-stage-buffer self-test report is the reviewed module's
    observation (`research/docs/23` §92, R9k):
    `shaders/render_stage_buffer_write_2x2.metal` compiled and run twice against
    a fresh 2x2 `rgba8_unorm` attachment cleared to the `fefefefe` sentinel.

    Each run binds the vertex stage's own `[[buffer(0)]]` positions, the
    fragment stage's readable `[[buffer(0)]]` source, its write-only
    `[[buffer(1)]]` sink and its read-write `[[buffer(2)]]` accumulator with
    `setVertexBuffer` and `setFragmentBuffer`, and reports five readings. The
    readings are the claim rather than decoration, one per access arm:

    * `attachment_hex` is the stage's *returned* source texel, so a rail that
      bound no source reads the clear sentinel everywhere;
    * `positions_hex` is the geometry the vertex stage read out of its own
      buffer, so the reviewed triangle covers one texel and the full-screen
      positions cover all four;
    * `sink_hex` is the write-only binding's bytes after the pass: the source
      payload, where a rail that dropped the write would report the zeros the
      sink started from;
    * `accumulator_hex` is the read-write binding's bytes after the pass: its
      previous value plus one, where a rail that bound no previous bytes (or no
      binding) would report one alone.

    The two runs move the source payload and the geometry independently, so no
    single constant frame can pass both. The second run's source is
    `(0, 1, 0, 1)`, which the 8-bit attachment stores as `00ff00ff`, while both
    runs start the accumulator from `0.25` in every component (the same
    previous value) and report `1.25` afterwards.
    """
    if not isinstance(report, dict):
        raise NativeRunError("stage buffer write selftest: report is not an object")
    if report.get("id") != "stage_buffer_write_2x2":
        raise NativeRunError(
            "stage buffer write selftest: report id " + repr(report.get("id"))
            + " is not the reviewed writable stage-buffer fixture"
        )
    if report.get("completion") != "CompletedVisible":
        raise NativeRunError(
            "stage buffer write selftest: completion is not CompletedVisible"
        )
    sentinel = "fefefefe"
    reviewed_frame = "4080c0ff" + sentinel * 3
    full_frame = "00ff00ff" * 4
    positions = "000080bf0000803f0000803e0000803f000080bf000080be"
    full_positions = "000080bf000080bf00004040000080bf000080bf00004040"
    tint = "8180803e8180003fc1c0403f0000803f"
    green = "000000000000803f000000000000803f"
    accumulator_initial = "0000803e" * 4
    accumulator_reviewed = "0000a03f" * 4
    accumulator_green = "0000a03f" * 4
    writebacks = report.get("writebacks", [])
    allocations = report.get("allocations", [])
    expected_writebacks = [
        {"allocation": 900, "view": 910, "offset": 0, "bytes_hex": reviewed_frame},
        {"allocation": 901, "view": 911, "offset": 0, "bytes_hex": tint},
        {"allocation": 902, "view": 912, "offset": 0, "bytes_hex": accumulator_reviewed},
    ]
    expected_allocations = [
        {"allocation": 900, "bytes_hex": reviewed_frame},
        {"allocation": 901, "bytes_hex": tint},
        {"allocation": 902, "bytes_hex": accumulator_reviewed},
    ]
    # The shape is part of the claim: the attachment and each writable binding
    # report exactly one writeback and one allocation, in that order, so a
    # report that reached the same bytes by another route (a missing sink, a
    # merged identity) is refused rather than compared as the same observation.
    if writebacks != expected_writebacks or allocations != expected_allocations:
        raise NativeRunError(
            "stage buffer write selftest: observations do not match the reviewed "
            "run, got " + repr((writebacks, allocations))
        )
    observations = report.get("observations", [])
    if len(observations) != 2:
        raise NativeRunError(
            "stage buffer write selftest: expected two runs (reviewed triangle, "
            "full-screen green), got " + repr(observations)
        )
    expected = [
        (positions, tint, reviewed_frame, tint, accumulator_reviewed),
        (full_positions, green, full_frame, green, accumulator_green),
    ]
    for index, (observation, run) in enumerate(zip(observations, expected)):
        if not isinstance(observation, dict):
            raise NativeRunError(
                "stage buffer write selftest: run " + str(index) + " is not an object"
            )
        fields = ("positions_hex", "source_hex", "attachment_hex", "sink_hex",
                  "accumulator_hex")
        for field, value in zip(fields, run):
            if observation.get(field) != value:
                raise NativeRunError(
                    "stage buffer write selftest: run " + str(index) + " " + field
                    + " is " + repr(observation.get(field)) + " instead of " + repr(value)
                )
        if observation.get("accumulator_initial_hex") != accumulator_initial:
            raise NativeRunError(
                "stage buffer write selftest: run " + str(index)
                + " does not name the previous bytes it bound, got "
                + repr(observation.get("accumulator_initial_hex"))
            )
    return ("frames=[" + ", ".join(run[2] for run in expected) + "] sinks=["
            + ", ".join(run[3] for run in expected) + "] accumulators=["
            + ", ".join(run[4] for run in expected) + "]")


def validate_store_dontcare_selftest(report):
    """One store-dontcare self-test report is the reviewed discard fixture's
    observation (`conformance/RENDER-CAPTURE.md` §11): the fixture id, one
    writeback and one allocation for the *stored* location (allocation 900 /
    view 910, four `4080c0ff` texels), and no observation at all for the
    discarded location (allocation 920 / view 930). A report that reads the
    discarded attachment back would present "nothing landed" as an
    observation, which the v19 rule refuses (`research/docs/23` §3.6).

    The fixture id is part of the check rather than decoration: the stored
    location's bytes are the same four texels the MRT self-test stores, and
    the id is what says the second location was drawn and then discarded.
    The CI step reuses this instead of inlining its byte comparison, so the
    comparison is exercised by `test_run_native.py` on a host without Metal.
    """
    if not isinstance(report, dict):
        raise NativeRunError("store-dontcare selftest: report is not an object")
    if report.get("id") != "discard_second_attachment_2x2":
        raise NativeRunError(
            "store-dontcare selftest: report id " + repr(report.get("id"))
            + " is not the reviewed discard fixture"
        )
    if report.get("completion") != "CompletedVisible":
        raise NativeRunError("store-dontcare selftest: completion is not CompletedVisible")
    writebacks = report.get("writebacks", [])
    allocations = report.get("allocations", [])
    expected = "4080c0ff" * 4
    expected_writebacks = [
        {"allocation": 900, "view": 910, "offset": 0, "bytes_hex": expected}
    ]
    expected_allocations = [{"allocation": 900, "bytes_hex": expected}]
    # The shape is part of the claim: the stored location reports exactly one
    # writeback and one allocation, and the discarded location reports
    # nothing. Any observation naming allocation 920 or view 930 — or any
    # other shape that reaches the same bytes by another route — is refused
    # rather than compared as if it were the same observation.
    if writebacks != expected_writebacks or allocations != expected_allocations:
        raise NativeRunError(
            "store-dontcare selftest: observations do not match the reviewed discard fixture, got "
            + repr((writebacks, allocations))
        )
    return expected


def run_capture(oracle, suite_path, output_dir, *, require_metal=False, revision=None,
                run_command=subprocess.run):
    """Create a new evidence directory. A partial or failed capture cannot pass."""
    oracle, suite_path, output_dir = map(lambda p: Path(p).resolve(),
                                        (oracle, suite_path, output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    status = {"schema_version": 1, "source_revision": revision,
              "capture_status": "failed", "reason": "not_started"}

    def command(name, arguments, timeout):
        with (output_dir / (name + ".stdout")).open("wb") as stdout, \
                (output_dir / (name + ".stderr")).open("wb") as stderr:
            run_command([str(oracle), *arguments], stdout=stdout, stderr=stderr,
                        check=True, timeout=timeout)

    try:
        raw, suite = _read_json(suite_path)
        digest = hashlib.sha256(raw).hexdigest()
        status.update(suite=suite["suite"], suite_sha256=digest)
        command("validate-suite", ["--suite", str(suite_path), "--validate-suite"], 30)
        command("probe", ["--probe"], 30)
        # Keep the exact probe stdout as an artifact before parsing it.
        probe_path = output_dir / "probe.json"
        probe_path.write_bytes((output_dir / "probe.stdout").read_bytes())
        _, probe = _read_json(probe_path)
        validate_probe(probe)
        status.update(device=probe["device"], platform=probe["platform"])
        if not probe["eligible"]:
            status.update(capture_status="unavailable", reason=probe["reason"])
            if require_metal:
                raise NativeRunError("required native Metal device unavailable: " + probe["reason"])
            return status

        path = output_dir / "native-metal.json"
        command("capture", ["--suite", str(suite_path), "--output", str(path)], 180)
        _, report = _read_json(path)
        validate_capture(suite, digest, report, required_backend="native-metal")
        if report["device"] != probe["device"] or report["platform"] != probe["platform"]:
            raise NativeRunError("capture device/platform differs from probe")
        # The suite must remain identical throughout validation and execution.
        if suite_path.read_bytes() != raw:
            raise NativeRunError("suite changed during capture")
        status.update(capture_status="captured", reason="native_capture_validated")
        return status
    except Exception as error:
        if status["capture_status"] != "unavailable":
            status.update(capture_status="failed", reason=str(error))
        raise
    finally:
        (output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-revision", help="source commit associated with this run")
    parser.add_argument("--require-metal", action="store_true",
                        help="return failure when a native Metal GPU is unavailable")
    args = parser.parse_args(argv)
    try:
        status = run_capture(args.oracle, args.suite, args.output_dir,
                             require_metal=args.require_metal, revision=args.source_revision)
        if status["capture_status"] == "captured":
            print("PASS native capture validated; cross-backend comparison still separate")
        else:
            print("UNAVAILABLE native Metal: " + status["reason"] + "; no GPU capture produced")
        return 0
    except (NativeRunError, CaptureError, KeyError, OSError, UnicodeError,
            json.JSONDecodeError, subprocess.SubprocessError) as error:
        print("FAIL native capture: " + str(error), file=sys.stderr)
        # The oracle writes its rejection reason to the captured stderr; surface
        # it so a suite or oracle mismatch is diagnosable from CI alone.
        for name in ("validate-suite", "capture"):
            diagnostic = args.output_dir / (name + ".stderr")
            if diagnostic.is_file():
                text = diagnostic.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    print("--- " + name + ".stderr ---", file=sys.stderr)
                    print(text, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
