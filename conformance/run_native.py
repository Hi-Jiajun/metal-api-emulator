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
        return 1


if __name__ == "__main__":
    sys.exit(main())
