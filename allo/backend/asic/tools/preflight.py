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
* the RTL directory and file list for the requested variant, under the export
  root given by ``--exports`` -- the flow is shared, neither the design nor its
  exports are, so both are named on the command line rather than derived. This
  is the same flag, with the same meaning, that ``check_pairing.py`` takes
* the fetched ``stdcells.db`` against the md5 recorded in the committed settings
  snapshots -- the ADK payload is not vendored (see
  ``allo/backend/asic/PROVENANCE.md``), so this is what ties a rerun to the
  library the published numbers were measured against

Run: python allo/backend/asic/tools/preflight.py --design DIR --exports DIR
         [--variant T8_MAXDIM64] [--build DIR]

For the TinyTPU design, from the repository root::

    python allo/backend/asic/tools/preflight.py \
        --design examples/tinytpu/asic_synthesis \
        --exports dev/records/tinytpu/rtl_handoff

The exports are generated RTL, so they are records and live with the records
(``dev/repo_layout.md``); Gemmini's are the sibling
``dev/records/tinytpu/gemmini_rtl``.

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


def _repo_root():
    """The checkout root, found by searching upward for the vendored flow.

    Not by counting directories up from ``__file__``: that has been wrong here
    three times in two days, including in a fix for itself, and the repository
    is mid-reorganisation. A path that encodes tree shape is a latent break.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(d, "allo", "backend", "asic", "nodes")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            # Not in a checkout (an installed copy, say). Only defaults and
            # printed paths depend on this, so degrade rather than refuse.
            return os.path.dirname(os.path.abspath(__file__))
        d = parent


# This script lives inside the flow, so the flow is its own parent -- no
# search needed for that one.
FLOW = os.environ.get("ALLO_ASIC_FLOW", os.path.dirname(HERE))
REPO = _repo_root()
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


def expected_library_md5(reports):
    """Every stdcells.db md5 recorded in the committed settings snapshots."""
    out = {}
    for path in sorted(glob.glob(os.path.join(reports, "*", "settings.json"))):
        with open(path) as fh:
            digest = json.load(fh).get("stdcells_db_md5")
        if digest:
            out.setdefault(digest, []).append(
                os.path.basename(os.path.dirname(path)))
    return out


def _show(path):
    """A path as a reader would type it: relative inside the checkout, else absolute."""
    rel = os.path.relpath(path, REPO)
    return path if rel.startswith(os.pardir) else rel


def check_library(reports, build):
    expected = expected_library_md5(reports)
    if not expected:
        # Paste-able: the reports directory is an argument now, so a hint that
        # omits it sends the reader to an argparse error instead of a fix.
        missing("no stdcells_db_md5 in any settings snapshot",
                f"python {_show(HERE)}/extract_results.py "
                f"--reports {_show(reports)} "
                "--capture-settings VARIANT=<build-dir>, for a completed run")
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


def check_variant(exports, variant):
    """The variant's RTL under the export root, which is given, not derived.

    This used to take the *design* directory and look one level above it for
    directories named ``rtl_handoff`` and ``gemmini_rtl``. Both assumptions --
    that the exports sit beside the design, and that they are named one of two
    fixed things -- were tree shape encoded in a tool, and both stopped being
    true when the exports moved to ``dev/records/tinytpu/``. It takes the root
    now, exactly as ``check_pairing.py`` does.
    """
    d = os.path.join(exports, variant)
    if not os.path.isdir(d):
        missing(f"no RTL directory for {variant}",
                f"expected {d} -- pass --exports for the right export root "
                f"(TinyTPU's are dev/records/tinytpu/rtl_handoff, Gemmini's "
                f"dev/records/tinytpu/gemmini_rtl)")
        return
    lists = [f for f in sorted(os.listdir(d)) if f.startswith("sv2v_manifest")]
    count = len(glob.glob(os.path.join(d, "*.v"))) + \
        len(glob.glob(os.path.join(d, "*.sv")))
    if not count:
        missing(f"no RTL files for {variant}", f"{d} holds no .v or .sv")
        return
    good(f"RTL for {variant}", f"{count} files, lists: {', '.join(lists)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", required=True,
                    help="a design's synthesis directory, holding "
                         "construct-commercial.py and reports/ "
                         "(e.g. examples/tinytpu/asic_synthesis)")
    ap.add_argument("--exports", required=True,
                    help="the design's committed RTL exports, the directory "
                         "holding one subdirectory per variant "
                         "(e.g. dev/records/tinytpu/rtl_handoff). The same "
                         "flag check_pairing.py takes")
    ap.add_argument("--variant", default="T8_MAXDIM64")
    ap.add_argument("--build", help="a build directory, to verify the fetched ADK")
    args = ap.parse_args()

    design = os.path.abspath(args.design)
    if not os.path.isdir(design):
        sys.exit(f"no design directory at {design}")
    exports = os.path.abspath(args.exports)
    if not os.path.isdir(exports):
        sys.exit(f"no exports directory at {exports}")
    reports = os.path.join(design, "reports")

    check_tools()
    check_flow()
    check_variant(exports, args.variant)
    check_library(reports, args.build)

    for line in ok:
        print(f"  ok      {line}")
    for what, fix in bad:
        print(f"  MISSING {what}\n            -> {fix}")

    if bad:
        print(f"\n{len(bad)} prerequisite(s) missing; a run would fail. "
              "Nothing was started.")
        return 1

    tools = _show(HERE)
    rel_reports = _show(reports)
    print(f"""
Prerequisites present. The sequence, for {args.variant}:

  mkdir -p <build-dir> && cd <build-dir>
  mflowgen run --design {design}/construct-commercial.py
  make 4          # DC synthesis; 40-70 min depending on the variant

Then, back in the repository:

  python {tools}/extract_results.py --reports {rel_reports} \\
      --capture-settings {args.variant}=<build-dir>
  python {tools}/extract_results.py --reports {rel_reports}
  python {tools}/check_numbers.py --reports {rel_reports}

The licensed tool time is part of the cost and this script does not remove it.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
