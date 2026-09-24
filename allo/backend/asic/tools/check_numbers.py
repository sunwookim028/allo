#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that every ASIC area figure quoted in the docs comes from a committed report.

Why this exists: area numbers were being retyped as prose in several pages, so a
corrected figure could be fixed on one page and left stale on another. That
happened -- 1,271,692 survived on the design page after being superseded in the
benchmarks page -- and nothing in the tree could see it.

What it does, in one pass and at zero cost:

1. Reads every ``<reports>/<variant>/area_summary.rpt`` and extracts its total
   cell area. That set is the ground truth. The reports directory is given on
   the command line: the results belong to a design, this checker does not, so
   it must not know which design it is checking.
2. Scans the docs for anything shaped like an area figure (a 6-or-more-digit
   number with thousands separators, or one followed by a um2 unit).
3. Reports any such figure that is neither in the ground-truth set nor in the
   allow-list below.

The allow-list is for figures that are legitimately not ours: third-party
published numbers, and derived quantities we state deliberately. Every entry
needs a reason, because an allow-list without reasons becomes a place to hide
failures.

Run: python allo/backend/asic/tools/check_numbers.py --reports DIR [--docs DIR]

For the TinyTPU design, from the repository root::

    python allo/backend/asic/tools/check_numbers.py \
        --reports examples/tinytpu/asic_synthesis/reports

Exit 0 if clean, 1 if any figure is unaccounted for.
"""

import argparse
import os
import re
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


REPO = _repo_root()

# Figures that are deliberately not from our own reports. Each needs a reason.
ALLOWED = {
    "1,029": "Gemmini DAC 2021 Fig. 6a, total, thousands of um2",
    "382,026": "Gemmini DIM=4 logic-only, committed under reports/ by the PD session",
    "990,938": "Gemmini DIM=4 full, same source",
    "23,885": "Gemmini DMA/adapter path, name-prefix sum, from its mapped database",
    "31,715": "Gemmini command queue and decode, same source",
    "840,160": "our four AXI adapters, name-prefix sum (flattening leaves no hierarchy line)",
    "681,537": "our gmem0 instance area, from report_area -hierarchy",
    "703,513": "our gmem0 at MAXDIM=64, same source",
    "703,151": "our gmem0 in the widened variant, same source",
    "744,092": "our gmem1 widened, same source",
    "745,280": "our gmem2 widened, same source",
    "1,388,710": "the widening delta, derived: 3,254,024 - 1,865,314",
    "1,161,801": "derived: T4_MAXDIM64 shipped less gmem0",
    "1,783,365": "derived: T8_MAXDIM64 less gmem0",
    "2,550,873": "derived: T4_MAXDIM64 widened less gmem0",
    "455,061": "derived: baseline less gmem0",
    "1,271,692": "explicitly cited as the SUPERSEDED figure, in its correction note",
    "2,621,440": "stock Gemmini memories as flip-flops; a memory-treatment figure, not a design",
}

# Only figures *claimed as areas*: a grouped number carrying a um2 unit, or one
# on a line that says "cell area". Anything broader flags cycle counts and LUT
# counts, and a checker that cries wolf is one nobody runs.
AREA_RE = re.compile(r"(\d{1,3}(?:,\d{3}){1,3})\s*(?:\\ )?(?:\u00b5m|um)\s*(?:\u00b2|2|\^2)")
AREA_LINE = re.compile(r"cell area", re.I)
GROUPED = re.compile(r"\b(\d{1,3}(?:,\d{3}){1,3})\b")


def ground_truth(reports):
    """Total cell area from every committed area_summary.rpt."""
    out = {}
    if not os.path.isdir(reports):
        sys.exit(f"no reports directory at {reports}")
    for variant in sorted(os.listdir(reports)):
        rpt = os.path.join(reports, variant, "area_summary.rpt")
        if not os.path.isfile(rpt):
            continue
        with open(rpt) as fh:
            for line in fh:
                if "Total cell area" in line:
                    raw = float(line.split(":")[-1].strip())
                    # The docs are not consistent about floor vs round on the
                    # fractional part, so accept either rather than impose one
                    # -- the failure this guards against is a *stale* figure,
                    # not a half-micron.
                    out[f"{int(raw):,}"] = variant
                    out[f"{round(raw):,}"] = variant
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", required=True,
                    help="a design's committed reports directory, holding "
                         "<variant>/area_summary.rpt")
    ap.add_argument("--docs", default=os.path.join(REPO, "docs", "source"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    truth = ground_truth(args.reports)
    if not truth:
        sys.exit(f"no area figures found under {args.reports}"
                 " -- has anything been committed?")

    unaccounted = {}
    for root, _, files in os.walk(args.docs):
        for name in files:
            if not name.endswith(".rst"):
                continue
            path = os.path.join(root, name)
            with open(path, errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    figs = set(AREA_RE.findall(line))
                    if AREA_LINE.search(line):
                        figs |= set(GROUPED.findall(line))
                    for fig in figs:
                        if len(fig.replace(",", "")) < 6:
                            continue
                        if fig in truth or fig in ALLOWED:
                            continue
                        unaccounted.setdefault(fig, []).append(
                            f"{os.path.relpath(path, REPO)}:{n}")

    if not args.quiet:
        print(f"ground truth: {len(truth)} committed totals")
        for fig, variant in sorted(truth.items()):
            print(f"  {fig:>12}  {variant}")

    if unaccounted:
        print("\nUNACCOUNTED area figures in the docs:")
        for fig, where in sorted(unaccounted.items()):
            print(f"  {fig:>12}  {', '.join(where[:3])}"
                  + (f" (+{len(where)-3} more)" if len(where) > 3 else ""))
        print("\nEach is either stale, or legitimately external and needs an"
              " ALLOWED entry with a reason.")
        return 1

    print("\nAREA NUMBERS OK: every figure quoted in the docs is accounted for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
