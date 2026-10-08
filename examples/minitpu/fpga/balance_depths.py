# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Size every dataflow link of a Vitis HLS build from its own schedule, so that
a composition of II-1 processes runs at one token per cycle.

    python3 balance_depths.py <prj> [--margin 1] [--min 2] [--out depths.tcl] [--json depths.json]

Why (minitpu_fpga_mxu_2026-10-08, section 2): a link's depth must hold the
tokens that arrive on it while its consumer waits on a *later* input. Allo
gives a stream array one depth for every element (``Stream[T, d][D, D+1]``);
on Vitis a PE's iteration latency is 7-9 cycles (it was 1 on Catapult), so the
skew between reconvergent paths -- the front's direct link to ``PE(r, 0)``
against ``PE(r-1, 0)``'s, ``px[D, 0]`` against ``px[D, D-1]`` into the back,
``ctl`` against the whole grid -- grows with the geometry, and no uniform
depth both fits and suffices.

The model (steady state, every process a stall-on-empty/full pipeline at II 1):
process ``p`` reads token ``t`` of link ``l`` at ``A_p + r_p(l) + t`` and
writes it at ``A_p + w_p(l) + t``, with ``r``/``w`` the pipeline states of
the ``ap_fifo`` read/write in ``<p>[_Pipeline_*].verbose.sched.rpt``.
A token written at cycle X is readable at X + 1 (FIFO_SRL), so
``A_dst = max_l (A_src + w_src(l) + 1 - r_dst(l))`` in topological order, and
a link holds ``(A_dst + r_dst) - (A_src + w_src)`` tokens just before its
consumer reads; it needs that plus one (``full_n`` is the current count) plus
``--margin``. Links to and from the top's ports are the wrapper's and are not
sized here. Writes ``set_directive_stream -depth`` lines for ``vhls_build.py
--tcl-file`` (a tcl directive overrides the emitted pragma; checked: the
RTMG line names ``fifo_w32_d8`` for a directive of 8 on a pragma of 2).
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("prj")
ap.add_argument("--margin", type=int, default=1)
ap.add_argument("--min", type=int, default=2)
ap.add_argument("--out", default=None)
ap.add_argument("--json", default=None)
a = ap.parse_args()

code = open(os.path.join(a.prj, "kernel.cpp")).read()
m = re.search(r"/// This is top function\.\nvoid (\w+)\((.*?)\)\s*\{(.*?)\n\}", code, re.S)
top, targs, body = m.group(1), m.group(2), m.group(3)
ports = set(re.findall(r"(\w+)\[\d+\]", targs))
streams = dict(re.findall(r"#pragma HLS stream variable=(\w+) depth=(\d+)", body))
calls = re.findall(r"^\s*(\w+)\((.*?)\);", body, re.M)
procs = [c for c, _ in calls]

db = os.path.join(a.prj, "out.prj/solution1/.autopilot/db")
op = re.compile(r'^ST_(\d+) : Operation \d+ .*?"%?[^"]*?@_ssdm_op_(Read|Write)\.ap_fifo[^"]*?%(\w+)[",]', re.M)
rd, wr = defaultdict(dict), defaultdict(dict)   # stream -> {proc: stage}
for proc in procs:
    files = glob.glob(os.path.join(db, f"{proc}.verbose.sched.rpt")) + \
        glob.glob(os.path.join(db, f"{proc}_Pipeline_*.verbose.sched.rpt"))
    assert files, f"no schedule report for {proc} in {db}"
    for f in files:
        for st, kind, var in op.findall(open(f).read()):
            if var in streams or var in ports:
                d = rd if kind == "Read" else wr
                d[var][proc] = min(int(st), d[var].get(proc, 1 << 30)) if kind == "Read" else max(int(st), d[var].get(proc, -1))

links, unused = [], []
for s in streams:
    if not wr[s] and not rd[s]:   # declared, never touched (the grid's edge links with EDGE_OUT 0)
        unused.append(s)
        continue
    assert len(wr[s]) == 1 and len(rd[s]) == 1, (s, wr[s], rd[s])
    (src, w), = wr[s].items()
    (dst, r), = rd[s].items()
    links.append((s, src, w, dst, r))

A = {p: 0 for p in procs}
ins = defaultdict(list)
for s, src, w, dst, r in links:
    ins[dst].append((s, src, w, r))
done = set()
order = []
while len(done) < len(procs):   # topological: a link-free process (the front) first
    progressed = False
    for p in procs:
        if p in done or any(src not in done for _, src, _, _ in ins[p]):
            continue
        A[p] = max([A[src] + w + 1 - r for _, src, w, r in ins[p]] or [0])
        done.add(p); order.append(p); progressed = True
    assert progressed, "the link graph has a cycle; this model is for feed-forward compositions"

out, rows = [], []
for s, src, w, dst, r in links:
    occ = (A[dst] + r) - (A[src] + w)
    depth = max(a.min, occ + 1 + a.margin)
    rows.append({"stream": s, "src": src, "w": w, "dst": dst, "r": r, "occupancy": occ,
                 "emitted_depth": int(streams[s]), "depth": depth})
    out.append(f"set_directive_stream -depth {depth} {top} {s}  ;# {src}@{w} -> {dst}@{r}, holds {occ}")
tot_bits = 0
for row in rows:
    width = int(re.search(r"hls::stream< (?:ap_uint<(\d+)>|uint(\d+)_t|(bool)) > " + row["stream"] + ";", body).group(1) or
                re.search(r"hls::stream< (?:ap_uint<(\d+)>|uint(\d+)_t|(bool)) > " + row["stream"] + ";", body).group(2) or 1)
    row["width"] = width
    tot_bits += width * row["depth"]
summ = {"unused_streams": unused, "top": top, "processes": len(procs), "links": len(rows), "A": A,
        "max_depth": max(r["depth"] for r in rows), "sum_depth": sum(r["depth"] for r in rows),
        "fifo_bits": tot_bits, "margin": a.margin}
print(f"BALANCE {top}: {len(procs)} processes, {len(rows)} links, depth max {summ['max_depth']} "
      f"sum {summ['sum_depth']}, {tot_bits} FIFO bits, {len(unused)} unused streams; back starts at A={A[procs[-1]]}")
for row in sorted(rows, key=lambda r: -r["depth"])[:8]:
    print(f"  {row['stream']}: {row['src']}@{row['w']} -> {row['dst']}@{row['r']} holds {row['occupancy']} -> depth {row['depth']}")
if a.out:
    open(a.out, "w").write(f"# GENERATED by balance_depths.py from {a.prj}\n" + "\n".join(out) + "\n")
if a.json:
    json.dump({"summary": summ, "links": rows}, open(a.json, "w"), indent=1)
