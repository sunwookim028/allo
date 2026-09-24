#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check whether this machine can reproduce an ASIC synthesis run, and say what is missing.

This is NOT a push-button reproduction. A run needs a Synopsys DC licence, the
pinned mflowgen and sv2v, four EDA modules, and about 40-70 minutes of a machine
outside this repository's reach. What this script does is fail in seconds instead
of an hour in, and print the command sequence once every prerequisite is present.

It checks:

* ``dc_shell`` on PATH, and its version
* ``mflowgen``, and that it is the pinned commit's 0.8.0 -- PyPI's 0.7.0 lacks the
  ``Node`` construct these constructor files use, and the failure it produces is
  confusing rather than clean
* ``sv2v``, reporting the version, which is unpinned upstream
* the vendored node library and ADK definition under ``allo/backend/asic``
* the RTL directory and file list for the requested variant
* the fetched ``stdcells.db`` against the md5 recorded in the committed settings
  snapshots -- the ADK payload is not vendored (see
  ``allo/backend/asic/PROVENANCE.md``), so this is what ties a rerun to the
  library the published numbers were measured against

Run: python asic_synthesis/tools/preflight.py [--variant T8_MAXDIM64] [--build DIR]
Exit 0 if a run is possible, 1 otherwise.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASIC = os.path.dirname(HERE)
TINYTPU = os.path.dirname(ASIC)
REPO = os.path.abspath(os.path.join(TINYTPU, "..", ".."))
FLOW = os.environ.get("ALLO_ASIC_FLOW",
                      os.path.join(REPO, "allo", "backend", "asic"))
MFLOWGEN_PIN = "aee0e5d640638bd38b44007d7258828eef66c641"

ok, bad = [], []


def good(what, detail=""):
    ok.append(f"{what}{': ' + detail if detail else ''}")


def missing(what, fix):
    bad.append((what, fix))


def run(*cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return (p.stdout + p.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def check_tools():
    if shutil.which("dc_shell"):
        good("dc_shell", os.path.realpath(shutil.which("dc_shell")))
    else:
        missing("dc_shell not on PATH",
                "module load synopsys-dc-W-2024.09 synopsys-2024 (and a licence)")

    if shutil.which("mflowgen"):
        version = run("mflowgen", "--version")
        hit = re.search(r"\d+\.\d+\.\d+", version)
        found = hit.group(0) if hit else "unknown"
        if found.startswith("0.8"):
            good("mflowgen", found)
        else:
            missing(f"mflowgen is {found}, not 0.8.x",
                    "pip install --editable \"mflowgen @ git+https://github.com/"
                    f"mflowgen/mflowgen.git@{MFLOWGEN_PIN}\"  # PyPI 0.7.0 lacks Node")
    else:
        missing("mflowgen not on PATH",
                "pip install --editable \"mflowgen @ git+https://github.com/"
                f"mflowgen/mflowgen.git@{MFLOWGEN_PIN}\"")

    if shutil.which("sv2v"):
        # Unpinned upstream: record whatever resolved, so a later run can tell
        # whether it changed. Only needed when normalize_rtl is True.
        good("sv2v", run("sv2v", "--version") or "version unknown")
    else:
        missing("sv2v not on PATH (needed only when normalize_rtl is True)",
                "conda install ucb-bar::sv2v")


def check_flow():
    nodes = os.path.join(FLOW, "nodes")
    if os.path.isdir(nodes):
        good("node library", f"{len(os.listdir(nodes))} nodes at {nodes}")
    else:
        missing(f"no node library at {nodes}",
                "set ALLO_ASIC_FLOW, or check out allo/backend/asic")
    adk = os.path.join(FLOW, "adks", "freepdk-45nm", "configure.yml")
    if os.path.isfile(adk):
        good("ADK definition", adk)
    else:
        missing(f"no ADK definition at {adk}", "same as above")


def expected_library_md5():
    """Every stdcells.db md5 recorded in the committed settings snapshots."""
    out = {}
    for path in sorted(glob.glob(os.path.join(ASIC, "reports", "*",
                                              "settings.json"))):
        with open(path) as fh:
            digest = json.load(fh).get("stdcells_db_md5")
        if digest:
            out.setdefault(digest, []).append(
                os.path.basename(os.path.dirname(path)))
    return out


def check_library(build):
    expected = expected_library_md5()
    if not expected:
        missing("no stdcells_db_md5 in any settings snapshot",
                "run extract_results.py --capture-settings for a completed run")
        return
    if len(expected) > 1:
        # Worth stopping for: two published runs did not use the same library.
        listing = "; ".join(f"{d[:8]} ({', '.join(v)})" for d, v in expected.items())
        missing("the committed runs disagree on stdcells.db", listing)
        return
    digest, variants = next(iter(expected.items()))
    good("library recorded by the committed runs",
         f"md5 {digest[:8]} across {len(variants)} runs")

    if not build:
        print(f"  (pass --build DIR after the ADK step to verify the fetched "
              f"library against md5 {digest[:8]})")
        return
    found = glob.glob(os.path.join(build, "*", "inputs", "adk", "stdcells.db"))
    if not found:
        missing(f"no fetched stdcells.db under {build}",
                "run the ADK step first; the payload is not vendored, by licence")
        return
    h = hashlib.md5()
    with open(found[0], "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() == digest:
        good("fetched library matches the published runs", h.hexdigest()[:8])
    else:
        missing("fetched stdcells.db does NOT match the published runs",
                f"got {h.hexdigest()[:8]}, expected {digest[:8]} -- areas from "
                "this library are not comparable with the committed set")


def check_variant(variant):
    for base in ("rtl_handoff", "gemmini_rtl"):
        d = os.path.join(TINYTPU, base, variant)
        if os.path.isdir(d):
            lists = [f for f in sorted(os.listdir(d)) if f.startswith("sv2v_manifest")]
            count = len(glob.glob(os.path.join(d, "*.v"))) + \
                len(glob.glob(os.path.join(d, "*.sv")))
            good(f"RTL for {variant}", f"{count} files, lists: {', '.join(lists)}")
            return
    missing(f"no RTL directory for {variant}",
            f"expected {TINYTPU}/rtl_handoff/{variant} or gemmini_rtl/{variant}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="T8_MAXDIM64")
    ap.add_argument("--build", help="a build directory, to verify the fetched ADK")
    args = ap.parse_args()

    check_tools()
    check_flow()
    check_variant(args.variant)
    check_library(args.build)

    for line in ok:
        print(f"  ok      {line}")
    for what, fix in bad:
        print(f"  MISSING {what}\n            -> {fix}")

    if bad:
        print(f"\n{len(bad)} prerequisite(s) missing; a run would fail. "
              "Nothing was started.")
        return 1

    print(f"""
Prerequisites present. The sequence, for {args.variant}:

  mkdir -p <build-dir> && cd <build-dir>
  mflowgen run --design {ASIC}/construct-commercial.py
  make 4          # DC synthesis; 40-70 min depending on the variant

Then, back in the repository:

  python {os.path.relpath(HERE, REPO)}/extract_results.py \\
      --capture-settings {args.variant}=<build-dir>
  python {os.path.relpath(HERE, REPO)}/extract_results.py
  python {os.path.relpath(HERE, REPO)}/check_numbers.py

The licensed tool time is part of the cost and this script does not remove it.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
