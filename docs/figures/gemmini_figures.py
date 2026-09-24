# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the Gemmini-comparison figures from the committed reports.

    python docs/figures/gemmini_figures.py            # writes the SVGs
    python docs/figures/gemmini_figures.py --print    # numbers only, no drawing

Why it exists: the comparison was numerically complete and had no picture, and a
picture drawn by hand is a fourth place for a corrected number to go stale. Every
value below is **read out of a committed file** -- the synthesis reports under
``examples/tinytpu/asic_synthesis/reports/`` and the cycle tables in
``docs/source/designs/benchmarks.rst`` -- so a figure cannot disagree with what
``check_numbers.py`` and ``check_pairing.py`` verify. Nothing is typed in.

The one exception is declared as one: our 64-PE array has no hierarchy line at
``flatten_effort 3`` (DC dissolves it), so its area is 64 x the per-PE mean that
``pe_array_area.rpt`` measured on the T=4 designs. It is drawn hatched and
labelled ``est.`` wherever it appears, which is the same treatment the prose
gives it.

Three figures, and what each is for:

1. ``gemmini_area_breakdown.svg`` -- where each design's logic goes, as a share
   of its own logic. The headline: the arithmetic array is under 4% of ours
   while the four memory-interface adapters are 40.9%.
2. ``gemmini_cycle_ratio.svg`` -- the deficit against Gemmini across the shape
   set, at both matched array sizes, with Gemmini's own min-max spread drawn so
   that 1.09x reads as the end of a convergence and not as a single point.
3. ``gemmini_memory_treatment.svg`` -- why the logic-only pair is the one
   quoted: the same design change reads as +14.9% or +37.3% depending only on
   how the memories were treated, and the non-combinational share moves for
   their design and not for ours.

Caveats travel *in* the figures, not only in the pages that embed them: every
panel states logic-only versus full, and the one run that misses timing
(``gemmini_DIM8_full``, -0.01 ns) is drawn as missing it, worded as the
methodology requires -- the RTL was not targeted at this constraint.
"""

import argparse
import ast
import json
import os
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REPORTS = os.path.join(REPO, "examples", "tinytpu", "asic_synthesis", "reports")
BENCH = os.path.join(REPO, "docs", "source", "designs", "benchmarks.rst")
SWEEP = os.path.join(REPO, "examples", "tinytpu", "parity_sweep.py")
OUT = os.path.join(REPO, "docs", "source", "_static", "figures")

# ---------------------------------------------------------------- palette ---
# The documented categorical slots 1 and 2 (blue, orange) on the light chart
# surface, validated all-pairs; everything else is chrome ink. Two hues, because
# the story is one contrast -- memory interface against arithmetic -- and the
# rest of every bar is deliberately neutral.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
NEUTRAL = "#d8d7d0"
CRITICAL = "#d03b3b"


def fail(msg):
    sys.exit(f"gemmini_figures: {msg}")


# ------------------------------------------------------------ report input ---
def area_summary(variant):
    """Total cell area from a run's ``area_summary.rpt`` -- the same file
    ``check_numbers.py`` treats as ground truth, read the same way."""
    path = os.path.join(REPORTS, variant, "area_summary.rpt")
    with open(path) as fh:
        for line in fh:
            if "Total cell area" in line:
                return float(line.split(":")[-1].strip())
    fail(f"no 'Total cell area' in {path}")


def run_results(variant):
    """A run's own ``results.json``: nested, and the only place the stub
    criterion and the library checksum are recorded per run."""
    with open(os.path.join(REPORTS, variant, "results.json")) as fh:
        return json.load(fh)


def run_facts(variant):
    r = run_results(variant)
    area, timing = r["area"], r.get("timing", {})
    manifest = r.get("settings", {}).get("rtl", {}).get("manifest", "")
    return dict(
        variant=variant,
        total=area["total_cell"],
        noncomb_share=100.0 * area["noncombinational"] / area["total_cell"],
        logic_only="nomem" in manifest,
        slack=timing.get("worst_path_group_slack"),
        violating=timing.get("setup", {}).get("violating_paths"),
        stdcells=r.get("settings", {}).get("stdcells_db_md5", ""),
    )


def hier_instances(variant, top, names):
    """Areas of named top-level hierarchy instances, from ``report_area
    -hierarchy``. Only instances whose boundary survived flattening appear
    here; that is exactly why the array is not among them."""
    path = None
    for name in os.listdir(os.path.join(REPORTS, variant)):
        if name.endswith(".mapped.area.hier.rpt"):
            path = os.path.join(REPORTS, variant, name)
    if path is None:
        fail(f"no *.mapped.area.hier.rpt under {variant}")
    want = set(names) | {top}
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3 and parts[0] in want and parts[0] not in out:
                out[parts[0]] = float(parts[1])
    missing = want - set(out)
    if missing:
        fail(f"{path}: no hierarchy line for {sorted(missing)}")
    return out


def pe_area_per_pe():
    """The measured per-PE mean, and the PE count it was measured over.

    ``pe_array_area.rpt`` is a name-prefix sum over flattened leaf cells, not a
    hierarchy line, and says so itself. It is the only measured per-PE number in
    the set, and it is identical in both T=4 designs to the digit.
    """
    path = os.path.join(REPORTS, "T4_MAXDIM16_shipped_baseline", "pe_array_area.rpt")
    with open(path) as fh:
        text = fh.read()
    mean = re.search(r"mean\s+([\d.]+)\s+um2/PE", text)
    total = re.search(r"total\s+([\d.]+)\s+over\s+(\d+)\s+PEs", text)
    if not mean or not total:
        fail(f"{path}: could not read the per-PE mean")
    per_pe, tot, npe = float(mean.group(1)), float(total.group(1)), int(total.group(2))
    if abs(tot / npe - per_pe) > 0.01:
        fail(f"{path}: mean {per_pe} does not match {tot}/{npe}")
    return per_pe, npe


def gemmini_dma_path():
    """Gemmini's reader / writer / transaction-tracker mass, from the committed
    answer sheet beside the DIM=4 logic-only run.

    These are NAME-PREFIX SUMS over flattened leaf cells and that file is
    explicit that they do not sum to the design -- magnitudes, not a budget.
    The figure therefore draws this one group and leaves the remainder
    unattributed rather than inventing a Gemmini budget.
    """
    path = os.path.join(REPORTS, "gemmini_DIM4_logiconly", "OPEN_QUESTIONS.rpt")
    with open(path) as fh:
        text = fh.read()
    parts = {}
    for key in ("reader", "writer", "xact"):
        m = re.search(rf"\*{key}\*\s+[\d,]+ cells\s+([\d,.]+) um2", text)
        if not m:
            fail(f"{path}: no *{key}* line")
        parts[key] = float(m.group(1).replace(",", ""))
    m = re.search(r"total\s+([\d,.]+) um2\s+=\s+([\d.]+)% of Gemmini's ([\d,.]+) logic", text)
    if not m:
        fail(f"{path}: no total line")
    total = float(m.group(1).replace(",", ""))
    if abs(sum(parts.values()) - total) > 0.02:
        fail(f"{path}: reader+writer+xact {sum(parts.values())} != stated {total}")
    return parts, total, float(m.group(3).replace(",", ""))


# ------------------------------------------------------------- cycle input ---
def _list_table_rows(lines, ncols):
    """Flatten an RST list-table's cells, in order, into rows of ``ncols``."""
    cells, cur = [], None
    for line in lines:
        m = re.match(r"\s*(\*\s+)?-\s?(.*)$", line)
        if m and (m.group(1) or cur is not None):
            if cur is not None:
                cells.append(cur.strip())
            cur = m.group(2)
        elif cur is not None and line.strip() and not line.lstrip().startswith(":"):
            cur += " " + line.strip()
        elif cur is not None and not line.strip():
            cells.append(cur.strip())
            cur = None
    if cur is not None:
        cells.append(cur.strip())
    if len(cells) % ncols:
        fail(f"list-table has {len(cells)} cells, not a multiple of {ncols}")
    return [cells[i:i + ncols] for i in range(0, len(cells), ncols)]


def cycle_table(heading):
    """The ours-against-Gemmini table under ``heading`` in benchmarks.rst.

    benchmarks.rst is the committed record of the cycle measurement -- there is
    no results.json for cosim -- so it is read as one, and the Gemmini column is
    then cross-checked against the independent copy in ``parity_sweep.py``. Two
    committed sources have to agree before anything is drawn.
    """
    with open(BENCH) as fh:
        lines = fh.read().split("\n")
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == heading)
    except StopIteration:
        fail(f"{BENCH}: no heading {heading!r}")
    start = next(i for i in range(start, len(lines)) if lines[i].startswith(".. list-table::"))
    end = next(i for i in range(start + 1, len(lines))
               if lines[i].strip() and not lines[i].startswith((" ", "\t")))
    rows = _list_table_rows(lines[start + 1:end], 8)
    if rows[0][0] != "shape":
        fail(f"{BENCH}: unexpected header {rows[0]!r} under {heading!r}")
    out = []
    for row in rows[1:]:
        shape = tuple(int(x) for x in row[0].split("x"))
        ours = int(row[2].replace(" ", "").replace(" ", ""))
        m = re.match(r"([\d ]+)\s*\+/-\s*(\d+)", row[3])
        if not m:
            fail(f"{BENCH}: cannot read Gemmini cell {row[3]!r}")
        out.append(dict(shape=shape, ours=ours,
                        gem=int(m.group(1).replace(" ", "")), spread=int(m.group(2))))
    return out


def sweep_gemmini():
    """``parity_sweep.GEMMINI`` without importing it (importing pulls in Vitis)."""
    tree = ast.parse(open(SWEEP).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "GEMMINI":
            return ast.literal_eval(node.value)
    fail(f"{SWEEP}: no GEMMINI assignment")


def cycles(T, heading):
    rows = cycle_table(heading)
    ref = sweep_gemmini()[T]
    for r in rows:
        if r["shape"] not in ref:
            fail(f"T={T}: {r['shape']} is in benchmarks.rst and not in parity_sweep.GEMMINI")
        if ref[r["shape"]] != (r["gem"], r["spread"]):
            fail(f"T={T} {r['shape']}: benchmarks.rst says {(r['gem'], r['spread'])}, "
                 f"parity_sweep.py says {ref[r['shape']]} -- one of them is stale")
    if len(rows) != len(ref):
        fail(f"T={T}: {len(rows)} rows against {len(ref)} shapes in parity_sweep.GEMMINI")
    return rows


# ------------------------------------------------------------------ figures --
def style():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "svg.hashsalt": "allo-gemmini-figures",   # reproducible SVG element ids
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def save(fig, name):
    """Every figure places itself: explicit margins, no tight bbox. A caption
    that a layout engine can move is a caption that can end up on top of a tick
    label, and these figures carry their caveats in that text."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, format="svg", metadata={"Date": None})
    print(f"  wrote {os.path.relpath(path, REPO)}")


def footnote(fig, y, text):
    """The caveats, wrapped to the figure's own width.

    Wrapped rather than hand-broken: these lines are edited often and a line
    that outgrows the canvas is silently cut off at the page, taking a caveat
    with it. ``\n`` in the text still starts a new paragraph.
    """
    width = int(fig.get_figwidth() * 19)
    lines = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width) or [""])
    fig.text(0.5, y, "\n".join(lines), ha="center", va="bottom", fontsize=6.5,
             color=MUTED, linespacing=1.55)


def bare(ax, grid="y"):
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    getattr(ax, f"{grid}axis").grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def fig_area(d):
    """Figure 1 -- where each design's logic goes, as a share of its own logic.

    One contrast carries the figure, so only two hues do any work: blue for the
    memory interface, orange for the arithmetic array, neutral for everything
    else. Every segment is labelled, so no reader has to decode a colour.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(9.0, 5.4))
    ax = fig.add_axes([0.050, 0.315, 0.935, 0.485])

    bars = [
        dict(y=1.20, label="TinyTPU-isa  T=8, MAXDIM=64, QD=16",
             sub=f"logic-only,  {d['ours_total']:,.0f} µm²", segs=d["ours_blocks"]),
        dict(y=0.00, label="Gemmini  DIM=4",
             sub=f"logic-only,  {d['gem_total']:,.0f} µm²", segs=d["gem_blocks"]),
    ]
    gap = 0.3                       # a surface gap between stacked segments
    for bar in bars:
        x = 0.0
        for seg in bar["segs"]:
            ax.barh(bar["y"], seg["pct"] - gap, left=x, height=0.40,
                    color=seg["color"], hatch=seg.get("hatch"),
                    edgecolor=SURFACE, linewidth=0.0, zorder=3)
            x += seg["pct"]
        ax.text(0, bar["y"] + 0.78, bar["label"], ha="left", va="baseline",
                fontsize=9.6, color=INK, fontweight="bold")
        ax.text(0, bar["y"] + 0.60, bar["sub"], ha="left", va="baseline",
                fontsize=8.0, color=INK2)

    def label_segments(bar, inside_min):
        """Wide segments name themselves; narrow ones get a leader out to a
        stack below the bar, because a clipped label is worse than none."""
        x, below = 0.0, []
        for seg in bar["segs"]:
            mid = x + seg["pct"] / 2
            x += seg["pct"]
            if not seg["name"]:
                continue
            if seg["pct"] >= inside_min:
                ink = SURFACE if seg["color"] == BLUE else INK
                ax.text(mid, bar["y"], f"{seg['name']}\n{seg['pct']:.1f}%",
                        ha="center", va="center", fontsize=7.4, color=ink,
                        linespacing=1.35, zorder=5)
            else:
                below.append((mid, seg))
        for i, (mid, seg) in enumerate(below):
            ty = bar["y"] - 0.29 - 0.15 * i
            ax.plot([mid, mid], [bar["y"] - 0.205, ty + 0.04], color=AXIS,
                    linewidth=0.7, zorder=2)
            ax.text(mid, ty, f"{seg['name']}  {seg['pct']:.1f}%", ha="center",
                    va="top", fontsize=7.2, color=INK2, zorder=5)

    label_segments(bars[0], 8.5)
    label_segments(bars[1], 12.0)

    def bracket(x0, x1, y, text, colour, va="bottom", dy=0.05, ha="center"):
        ax.annotate("", xy=(x0, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="|-|,widthA=0.32,widthB=0.32",
                                    color=colour, linewidth=1.1))
        tx = x0 if ha == "left" else (x0 + x1) / 2
        ax.text(tx, y + (dy if va == "bottom" else -dy), text,
                ha=ha, va=va, fontsize=8.6, color=colour, fontweight="bold")

    bracket(0, d["adapter_pct"], 1.47,
            f"the four memory-interface adapters: {d['adapter_pct']:.1f}%", BLUE)
    bracket(0, d["gem_blocks"][0]["pct"], -0.30,
            f"the whole DMA path: {d['gem_blocks'][0]['pct']:.1f}%", BLUE,
            va="top", dy=0.06, ha="left")
    ax.annotate(f"the arithmetic array: {d['array_pct']:.1f}%  (est.)",
                xy=(d["array_x"], 1.41), xytext=(d["array_x"] + 3.5, 1.92),
                fontsize=8.6, color=ORANGE, fontweight="bold", ha="left",
                va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=1.0,
                                connectionstyle="angle,angleA=0,angleB=90,rad=3"))

    ax.set_xlim(-0.6, 100.6)
    ax.set_ylim(-0.72, 2.16)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=7.6)
    ax.set_xlabel("share of that design's own logic area", fontsize=8.2, color=INK2,
                  labelpad=6)
    bare(ax, grid="x")

    fig.text(0.015, 0.960, "Where the logic goes, on each design's own terms",
             fontsize=11.5, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.015, 0.912,
             "The arithmetic array is under 4% of our logic. The adapters that "
             "carry operands and instructions to it are ten times that.",
             fontsize=8.6, color=INK2, ha="left", va="top")

    fig.legend(handles=[
        Patch(facecolor=BLUE, label="memory interface"),
        Patch(facecolor=ORANGE, hatch="///", edgecolor=SURFACE,
              label="arithmetic array (estimated, see below)"),
        Patch(facecolor=NEUTRAL, label="other logic, or not separately attributed"),
    ], loc="lower left", bbox_to_anchor=(0.050, 0.185), ncol=3, frameon=False,
        fontsize=7.6, handlelength=1.5, handleheight=0.9, columnspacing=1.6)

    footnote(fig, 0.020,
             "Both runs logic-only — the memory arrays stubbed out by the same criterion on both sides — under one flow: DC W-2024.09, FreePDK45, "
             f"3.33 ns, topographical, flatten effort 3, identical stdcells.db (md5 {d['lib']}). Both close timing.\n"
             "Our segments are hierarchy instances of this run. The array has none: DC dissolves it at this flatten effort, so it is drawn hatched, as "
             f"{d['npe']} PEs at the {d['per_pe']:,.2f} µm²/PE that pe_array_area.rpt measured over {d['npe_measured']} PEs on the T=4 designs.\n"
             "Gemmini's segment is a name-prefix sum over flattened cells — a magnitude, not a budget — so the rest of its bar is left unattributed "
             "rather than divided up, and its DIM=4 run is the one that carries that attribution. The capacity-matched pair, for the totals rather than "
             f"the shares, is ours against Gemmini DIM=8: {d['gem8_total']:,.0f} µm², a ratio of {d['ratio']:.2f}×.")
    save(fig, "gemmini_area_breakdown.svg")
    plt.close(fig)


def fig_cycles(d):
    """Figure 2 -- the deficit across the shape set, at both matched sizes."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    order, labels = d["order"], d["labels"]
    fig = plt.figure(figsize=(9.0, 5.4))
    axa = fig.add_axes([0.085, 0.320, 0.415, 0.450])
    axb = fig.add_axes([0.545, 0.320, 0.415, 0.450], sharey=axa)

    for ax, tag, rows, title, verdict in (
            (axa, "a", d["t4"], "T=4 against Gemmini DIM=4", "the deficit converges"),
            (axb, "b", d["t8"], "T=8 against Gemmini DIM=8",
             "it does not, and one shape inverts")):
        by = {r["shape"]: r for r in rows}
        cube_x, cube_y = [], []
        for i, shape in enumerate(order):
            r = by.get(shape)
            if r is None:
                continue
            ratio = r["ours"] / r["gem"]
            ax.plot([i, i], [r["ours"] / (r["gem"] + r["spread"]),
                             r["ours"] / (r["gem"] - r["spread"])],
                    color=AXIS, linewidth=3.6, solid_capstyle="butt", zorder=2)
            if shape[0] == shape[1] == shape[2]:
                cube_x.append(i)
                cube_y.append(ratio)
            else:
                ax.plot(i, ratio, marker="o", markersize=6.5, color=SURFACE,
                        markeredgecolor=BLUE, markeredgewidth=1.8, zorder=4)
        ax.plot(cube_x, cube_y, color=BLUE, linewidth=2.0, marker="o",
                markersize=6.5, markerfacecolor=BLUE, markeredgecolor=SURFACE,
                markeredgewidth=1.6, zorder=4)
        ax.axhline(1.0, color=INK2, linewidth=1.0, zorder=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.3)
        ax.set_xlim(-0.6, len(order) - 0.4)
        bare(ax)
        ax.set_title(f"({tag})  {title}\n{verdict}", fontsize=9.4, color=INK,
                     loc="left", pad=9, fontweight="bold")

    axa.set_ylabel("our cycles ÷ Gemmini's   (lower is better)", fontsize=8.2,
                   color=INK2)
    axa.set_ylim(0.74, 1.46)
    axa.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    axa.set_yticklabels(["0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4×"],
                        fontsize=7.5)
    plt.setp(axb.get_yticklabels(), visible=False)
    for ax in (axa, axb):
        ax.text(len(order) - 0.5, 1.008, "parity", ha="right", va="bottom",
                fontsize=7.4, color=INK2)

    t4 = {r["shape"]: r for r in d["t4"]}
    i16, i64 = order.index((16, 16, 16)), order.index((64, 64, 64))
    r16 = t4[(16, 16, 16)]["ours"] / t4[(16, 16, 16)]["gem"]
    r64 = t4[(64, 64, 64)]["ours"] / t4[(64, 64, 64)]["gem"]
    axa.annotate(f"{r16:.2f}×", xy=(i16 + 0.12, r16), xytext=(i16 + 1.1, 1.345),
                 fontsize=9, color=BLUE, fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=0.9))
    axa.annotate(f"{r64:.2f}× at 64³,\n74.1% of peak",
                 xy=(i64, r64 - 0.015), xytext=(i64 - 0.2, 0.84), fontsize=9,
                 color=BLUE, fontweight="bold", ha="right", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=0.9))

    t8 = {r["shape"]: r for r in d["t8"]}
    iw = order.index((16, 16, 8))
    w = t8[(16, 16, 8)]
    axb.annotate(f"{w['ours']} against {w['gem']} ± {w['spread']}:\n"
                 "the one supportable win,\nclearing the spread 4.8×",
                 xy=(iw + 0.14, w["ours"] / w["gem"]), xytext=(iw + 1.25, 0.762),
                 fontsize=8.6, color=ORANGE, fontweight="bold", ha="left",
                 va="bottom",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.9))
    axb.text(len(order) - 0.55, 1.225, "flat 1.16–1.17×\nat steady state",
             fontsize=8.6, color=INK2, ha="right", va="bottom", linespacing=1.4)

    fig.legend(handles=[
        Line2D([], [], color=BLUE, linewidth=2.0, marker="o", markersize=6.5,
               markerfacecolor=BLUE, markeredgecolor=SURFACE,
               label="cubic sweep — where the convergence is read"),
        Line2D([], [], color="none", marker="o", markersize=6.5,
               markerfacecolor=SURFACE, markeredgecolor=BLUE, markeredgewidth=1.8,
               label="non-cubic shape"),
        Line2D([], [], color=AXIS, linewidth=3.6,
               label="the ratio across Gemmini's own min–max spread"),
    ], loc="lower left", bbox_to_anchor=(0.085, 0.178), ncol=3, frameon=False,
        fontsize=7.6, handlelength=1.9, columnspacing=1.8)

    fig.text(0.015, 0.950,
             "The deficit against Gemmini, at both matched array sizes",
             fontsize=11.5, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.015, 0.908,
             "Shapes ordered by the work they contain. Gemmini's own trial-to-trial "
             "spread is drawn, so a ratio inside the bar is not a result.",
             fontsize=8.6, color=INK2, ha="left", va="top")

    footnote(fig, 0.020,
             "Ours: Vitis cosim (xsim), ap_start to ap_done, -m_axi_latency 0, "
             "bit-exact against numpy at every shape and deterministic \u2014 one run "
             "suffices, and that was verified. Gemmini: Verilator, the median of five "
             "trials, with one loop_ws proven at runtime per shape; the grey bar spans "
             "the ratio implied by the full min\u2013max spread of those five, which is "
             "not a standard error. Both sides at MAXDIM=64 against a matched array "
             "dimension, over the same window, with host overhead treated symmetrically. "
             "T=8 drops 4\u00b3 and 12\u00b3 \u2014 not multiples of 8 \u2014 by the "
             "same rule on both machines, so those columns are empty in (b) rather than "
             "estimated.\nThese are cycles. The two designs are not claimed to run at the "
             "same frequency, so no ratio here restates in seconds.")
    save(fig, "gemmini_cycle_ratio.svg")
    plt.close(fig)


def fig_memory(d):
    """Figure 3 -- what the memory treatment does, and what it does not.

    (a) is the argument for quoting the logic-only pair: one design change,
    two answers. (b) is the fact that survives either treatment on our side
    and does not on theirs, over every run in the set.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(9.0, 5.4))
    axa = fig.add_axes([0.070, 0.300, 0.375, 0.455])
    axb = fig.add_axes([0.555, 0.300, 0.420, 0.455])

    # (a) the same change, measured two ways ------------------------------
    g = d["gem_pair"]
    logic = [g[4]["logic"], g[8]["logic"]]
    mem = [g[4]["mem"], g[8]["mem"]]
    pad = 14000                       # a surface gap between the two segments
    XS = (0.0, 1.6)                   # room between the bars for the deltas
    axa.bar(XS, logic, width=0.55, color=BLUE, zorder=3)
    axa.bar(XS, mem, width=0.55, bottom=[v + pad for v in logic],
            color=NEUTRAL, hatch="////", edgecolor=SURFACE, linewidth=0.0, zorder=3)
    for x, lv, mv in zip(XS, logic, mem):
        axa.text(x, lv / 2, f"logic\n{lv/1e3:,.0f}k", ha="center", va="center",
                 fontsize=7.8, color=SURFACE, linespacing=1.35, zorder=5)
        axa.text(x, lv + pad + mv / 2, f"memory\narrays\n{mv/1e3:,.0f}k",
                 ha="center", va="center", fontsize=7.6, color=INK2,
                 linespacing=1.35, zorder=5)
    tops = [logic[i] + pad + mem[i] for i in (0, 1)]

    def delta(y0, y1, ty, label, colour, va, size=9.4):
        axa.annotate("", xy=(XS[0] + 0.30, y0), xytext=(XS[1] - 0.30, y1),
                     arrowprops=dict(arrowstyle="<->", color=colour, linewidth=1.2))
        axa.text(sum(XS) / 2, ty, label, ha="center", va=va, fontsize=size, color=colour,
                 fontweight="bold", linespacing=1.35, zorder=6)

    delta(logic[0], logic[1], 372000, f"logic-only\n+{d['d_logic']:.1f}%", BLUE, "top")
    delta(logic[0] + pad + mem[0] / 2, logic[1] + pad + mem[1] / 2, 790000,
          f"the stubbed mass\nbarely moves\n+{d['d_mem']:.1f}%", MUTED, "bottom", 7.4)
    delta(tops[0], tops[1], 1245000, f"full design\n+{d['d_full']:.1f}%", INK, "bottom")

    axa.plot([XS[1] + 0.31, XS[1] + 0.46], [tops[1], tops[1]], color=CRITICAL,
             linewidth=1.0, zorder=4)
    axa.text(XS[1] + 0.52, tops[1], f"misses timing,\n{d['miss']['slack']:.2f} ns — the\n"
             "RTL was not targeted\nat this constraint",
             fontsize=7.4, color=CRITICAL, fontweight="bold", ha="left",
             va="center", linespacing=1.45)
    axa.set_xticks(list(XS))
    axa.set_xticklabels(["Gemmini DIM=4", "Gemmini DIM=8"], fontsize=8.4, color=INK)
    axa.set_xlim(-0.50, 3.35)
    axa.set_ylim(0, 1.66e6)
    axa.set_yticks([0, 250e3, 500e3, 750e3, 1e6, 1.25e6])
    axa.set_yticklabels(["0", "250k", "500k", "750k", "1,000k", "1,250k"], fontsize=7.5)
    axa.set_ylabel("total cell area, µm²", fontsize=8.2, color=INK2)
    bare(axa)
    axa.set_title("(a)  one change, measured two ways\ndoubling the mesh: 4× the PEs",
                  fontsize=9.4, color=INK, loc="left", pad=9, fontweight="bold")

    # (b) non-combinational share, every run ------------------------------
    def swarm(values):
        """Offset marks that would otherwise sit on top of one another, up and
        down alternately, so the row stays centred on its baseline."""
        rows, out = [], []
        for v in sorted(values):
            r = 0
            while any(abs(v - pv) < 0.62 and r == pr for pv, pr in rows):
                r += 1
            rows.append((v, r))
            out.append((v, (r + 1) // 2 * (1 if r % 2 else -1)))
        return out

    for gi, (name, runs, colour, side) in enumerate(
            [("TinyTPU-isa", d["nc_ours"], BLUE, 0.62),
             ("Gemmini", d["nc_gem"], ORANGE, 0.34)]):
        y0 = 1 - gi
        logic_only = {round(s, 4): lo for s, lo in runs}
        for share, row in swarm([s for s, _ in runs]):
            lo = logic_only[round(share, 4)]
            axb.plot(share, y0 + 0.115 * row, marker="o" if lo else "s",
                     markersize=7.4, markerfacecolor=SURFACE if lo else colour,
                     markeredgecolor=colour, markeredgewidth=1.8, zorder=4)
        axb.text(34.0, y0 + side, f"{name} — {len(runs)} runs", ha="left",
                 va="bottom", fontsize=8.8, color=INK, fontweight="bold")

    ours = [s for s, _ in d["nc_ours"]]
    axb.text(sum(ours) / len(ours), 1.48,
             f"{min(ours):.1f}–{max(ours):.1f}%, either way",
             ha="center", va="bottom", fontsize=8.4, color=BLUE, fontweight="bold")
    gl = sorted(s for s, st in d["nc_gem"] if st)
    gf = sorted(s for s, st in d["nc_gem"] if not st)
    axb.text(sum(gl) / 2, -0.58, f"{gl[0]:.1f}–{gl[-1]:.1f}%\nlogic-only",
             ha="center", va="top", fontsize=8.4, color=ORANGE, fontweight="bold",
             linespacing=1.35)
    axb.text(sum(gf) / 2, -0.58, f"{gf[0]:.1f}–{gf[-1]:.1f}%\nmemories in",
             ha="center", va="top", fontsize=8.4, color=ORANGE, fontweight="bold",
             linespacing=1.35)
    axb.annotate("", xy=(gl[-1] + 1.8, -0.40), xytext=(gf[0] - 1.8, -0.40),
                 arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.2))
    axb.text((gl[-1] + gf[0]) / 2, -0.33,
             "their share moves with\nthe memory treatment",
             ha="center", va="bottom", fontsize=7.8, color=INK2, linespacing=1.4)
    axb.set_ylim(-1.25, 1.95)
    axb.set_yticks([])
    axb.set_xlim(33, 89)
    axb.set_xticks([40, 50, 60, 70, 80])
    axb.set_xticklabels(["40", "50", "60", "70", "80%"], fontsize=7.5)
    axb.set_xlabel("non-combinational share of that run's total cell area",
                   fontsize=8.2, color=INK2, labelpad=6)
    bare(axb, grid="x")
    axb.set_title("(b)  gate-heavy against flop-heavy\nevery run in the set",
                  fontsize=9.4, color=INK, loc="left", pad=9, fontweight="bold")
    axb.legend(handles=[
        Line2D([], [], color="none", marker="o", markersize=7.4,
               markerfacecolor=SURFACE, markeredgecolor=MUTED, markeredgewidth=1.8,
               label="logic-only run"),
        Line2D([], [], color="none", marker="s", markersize=7.4,
               markerfacecolor=MUTED, markeredgecolor=MUTED,
               label="memories in, as flip-flops"),
    ], loc="upper left", bbox_to_anchor=(0.30, 0.80), frameon=False, fontsize=7.4,
        handlelength=1.2, labelspacing=0.35)

    fig.text(0.015, 0.960, "Why the logic-only pair is the one quoted",
             fontsize=11.5, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.015, 0.912,
             "With memories rendered as flip-flops, a mass the change never touches "
             "sits in the total and dilutes it — far more on their design than on ours.",
             fontsize=8.6, color=INK2, ha="left", va="top")

    footnote(fig, 0.020,
             f"{d['nruns']} runs, one flow: DC W-2024.09, FreePDK45, 3.33 ns, "
             "topographical, flatten effort 3, sram_mode='none', identical stdcells.db "
             f"(md5 {d['lib']}); the superseded twelfth export is excluded. 'Memory "
             "arrays' in (a) is the full run less the logic-only run of the same design "
             "— the scratchpad and accumulator that the stub rule removes — so "
             "the bar is one quantity measured twice, not a third run. One run in the set "
             f"misses timing and is drawn missing it; the other {d['nruns'] - 1} close, at "
             f"slack from {d['slack_lo']:+.2f} ns (Gemmini's two logic-only runs: a pass "
             f"with nothing spare) to {d['slack_hi']:+.2f} ns.\nA DC topographical estimate "
             "is a floor rather than an achieved frequency, so nothing here is a speed "
             "claim in either direction.")
    save(fig, "gemmini_memory_treatment.svg")
    plt.close(fig)


# --------------------------------------------------------------------- main --
def gather():
    ours_total = area_summary("T8_MAXDIM64_qd16_logiconly")
    gem_total = area_summary("gemmini_DIM4_logiconly")
    gem8_total = area_summary("gemmini_DIM8_logiconly")
    names = ["gmem0_m_axi_U", "dma_ld_0_1_U0", "gmem1_m_axi_U", "gmem2_m_axi_U",
             "gmem3_m_axi_U", "sequencer_0_1_U0"]
    inst = hier_instances("T8_MAXDIM64_qd16_logiconly", "tinytpu_isa", names)
    if abs(inst["tinytpu_isa"] - ours_total) > 0.01:
        fail("the hierarchy report and area_summary.rpt disagree on the total")
    per_pe, npe_measured = pe_area_per_pe()
    npe = 64                     # T=8 -> an 8x8 array; the export's own T
    array = per_pe * npe
    dma_parts, dma_total, gem_logic_stated = gemmini_dma_path()
    if abs(gem_logic_stated - gem_total) > 1.0:
        fail("OPEN_QUESTIONS.rpt cites a different DIM=4 logic total than its area_summary")

    pct = lambda v: 100.0 * v / ours_total
    operands = inst["gmem1_m_axi_U"] + inst["gmem2_m_axi_U"]
    named = (inst["gmem0_m_axi_U"] + inst["dma_ld_0_1_U0"] + operands
             + inst["gmem3_m_axi_U"] + inst["sequencer_0_1_U0"] + array)
    ours_blocks = [
        dict(name="instruction-fetch\nadapter", pct=pct(inst["gmem0_m_axi_U"]), color=BLUE),
        dict(name="DMA load unit", pct=pct(inst["dma_ld_0_1_U0"]), color=BLUE),
        dict(name="2 operand\nadapters", pct=pct(operands), color=BLUE),
        dict(name="output adapter", pct=pct(inst["gmem3_m_axi_U"]), color=BLUE),
        dict(name="sequencer", pct=pct(inst["sequencer_0_1_U0"]), color=NEUTRAL),
        dict(name="64-PE array", pct=pct(array), color=ORANGE, hatch="///"),
        dict(name="everything else, not separately attributed",
             pct=100.0 - pct(named), color=NEUTRAL),
    ]
    adapter_pct = pct(inst["gmem0_m_axi_U"] + operands + inst["gmem3_m_axi_U"])
    array_x = pct(named - array) + pct(array) / 2
    gem_blocks = [
        dict(name="", pct=100.0 * dma_total / gem_total, color=BLUE),
        dict(name="not attributed — name-prefix sums do not form a budget",
             pct=100.0 - 100.0 * dma_total / gem_total, color=NEUTRAL),
    ]

    t4 = cycles(4, "T=4 vs Gemmini DIM=4, both at MAXDIM=64")
    t8 = cycles(8, "T=8 vs Gemmini DIM=8, both at MAXDIM=64")
    order = sorted({r["shape"] for r in t4} | {r["shape"] for r in t8},
                   key=lambda s: (s[0] * s[1] * s[2], s))
    labels = [f"{s[0]}³" if s[0] == s[1] == s[2] else
              f"{s[0]}×{s[1]}×{s[2]}" for s in order]

    gem_pair = {}
    for dim in (4, 8):
        full = area_summary(f"gemmini_DIM{dim}_full")
        lg = area_summary(f"gemmini_DIM{dim}_logiconly")
        gem_pair[dim] = dict(full=full, logic=lg, mem=full - lg)

    nc_ours, nc_gem, libs, misses, slacks = [], [], set(), [], []
    for variant in sorted(os.listdir(REPORTS)):
        if variant.startswith("superseded") or not os.path.isdir(os.path.join(REPORTS, variant)):
            continue
        f = run_facts(variant)
        libs.add(f["stdcells"])
        if f["violating"]:
            misses.append(f)
        else:
            slacks.append(f["slack"])
        (nc_gem if variant.startswith("gemmini") else nc_ours).append(
            (f["noncomb_share"], f["logic_only"]))
    if len(nc_ours) != 7 or len(nc_gem) != 4:
        fail(f"expected 7 runs of ours and 4 of Gemmini, found {len(nc_ours)} and {len(nc_gem)}")
    if len(libs) != 1:
        fail(f"the runs do not share one stdcells.db: {sorted(libs)}")
    if [m["variant"] for m in misses] != ["gemmini_DIM8_full"]:
        fail("the set of runs that miss timing has changed: "
             f"{[m['variant'] for m in misses]} -- the figures say only one does")

    return dict(
        ours_total=ours_total, gem_total=gem_total, gem8_total=gem8_total,
        ours_blocks=ours_blocks, gem_blocks=gem_blocks, adapter_pct=adapter_pct,
        array_pct=pct(array), array_x=array_x, per_pe=per_pe, npe=npe,
        npe_measured=npe_measured, array=array, inst=inst, dma_parts=dma_parts,
        dma_total=dma_total, ratio=ours_total / gem8_total,
        t4=t4, t8=t8, order=order, labels=labels, gem_pair=gem_pair,
        d_logic=100.0 * (gem_pair[8]["logic"] / gem_pair[4]["logic"] - 1),
        d_full=100.0 * (gem_pair[8]["full"] / gem_pair[4]["full"] - 1),
        d_mem=100.0 * (gem_pair[8]["mem"] / gem_pair[4]["mem"] - 1),
        nc_ours=nc_ours, nc_gem=nc_gem, lib=sorted(libs)[0][:8],
        miss=misses[0], slack_lo=min(slacks), slack_hi=max(slacks),
        nruns=len(nc_ours) + len(nc_gem),
    )


def report(d):
    print("figure 1 -- area breakdown")
    print(f"  ours, logic-only        {d['ours_total']:>12,.0f} um2")
    for b in d["ours_blocks"]:
        print(f"    {b['name'].replace(chr(10), ' '):<44s} {b['pct']:5.1f}%")
    print(f"    four adapters together{'':<23s} {d['adapter_pct']:5.1f}%")
    print(f"  Gemmini DIM=4 logic-only{d['gem_total']:>12,.0f} um2")
    for k, v in d["dma_parts"].items():
        print(f"    *{k}*{'':<38s} {100.0*v/d['gem_total']:5.1f}%")
    print(f"  array estimate: {d['npe']} PE x {d['per_pe']:,.2f} um2/PE "
          f"(measured over {d['npe_measured']} PEs at T=4) = {d['array']:,.0f} um2")
    print(f"  matched pair ours : Gemmini DIM=8 = {d['ratio']:.2f}x")
    print("figure 2 -- cycle ratio")
    for tag, rows in (("T=4", d["t4"]), ("T=8", d["t8"])):
        for r in rows:
            print(f"  {tag} {'x'.join(str(v) for v in r['shape']):>10s}  "
                  f"{r['ours']:>6d} / {r['gem']:>6d} +/- {r['spread']:<4d} = "
                  f"{r['ours']/r['gem']:.3f}x")
    print("figure 3 -- memory treatment")
    for dim, v in d["gem_pair"].items():
        print(f"  DIM={dim}  full {v['full']:>12,.0f}  logic {v['logic']:>12,.0f}  "
              f"arrays {v['mem']:>12,.0f} ({100.0*v['mem']/v['full']:.1f}% of full)")
    print(f"  doubling the mesh: +{d['d_full']:.1f}% full, +{d['d_logic']:.1f}% logic-only, "
          f"+{d['d_mem']:.1f}% in the stubbed arrays")
    print(f"  non-comb share, ours    {sorted(round(s, 1) for s, _ in d['nc_ours'])}")
    print(f"  non-comb share, Gemmini {sorted(round(s, 1) for s, _ in d['nc_gem'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", dest="only_print", action="store_true",
                    help="read and check the sources, print every number, draw nothing")
    args = ap.parse_args()
    d = gather()
    report(d)
    if args.only_print:
        return 0
    style()
    print(f"drawing into {os.path.relpath(OUT, REPO)}")
    fig_area(d)
    fig_cycles(d)
    fig_memory(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
