#!/usr/bin/env python3
"""Fetch the pinned reims sources and apply the offline compute adapter patch."""

import argparse
from pathlib import Path
import subprocess

BASE = "69a57dd69a6958e946c03b73e02db331f330f435"
UPSTREAM = "https://github.com/steelbrain/reims-vgpu.git"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=UPSTREAM,
                        help="Git URL or local clone containing the pinned commit")
    args = parser.parse_args()
    integration = Path(__file__).resolve().parent
    destination = integration / "vendor" / "reims-vgpu"
    if destination.exists():
        parser.error(f"refusing to overwrite {destination}; move it aside before retrying")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def git(*arguments):
        subprocess.run(["git", "-C", str(destination), *arguments], check=True)

    subprocess.run(["git", "init", str(destination)], check=True)
    git("fetch", "--depth=1", "--no-tags", args.source, BASE)
    git("checkout", "--detach", "FETCH_HEAD")
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != BASE:
        raise RuntimeError(f"expected {BASE}, fetched {actual}")
    patch = str(integration / "compute-facade.patch")
    git("apply", "--check", patch)
    git("apply", patch)
    print(f"Prepared {destination} at {BASE} plus compute-facade.patch")


if __name__ == "__main__":
    main()
