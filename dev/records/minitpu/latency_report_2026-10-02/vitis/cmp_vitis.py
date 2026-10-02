# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vitis HLS RTL of an Allo U1 unit kernel (``variant.py``: ap_fifo ports) vs
MiniTPU's RTL, in Verilator through the harness's ``stream`` shape, plus the
csynth report's numbers beside the measured latency.

    $ALLO_PYTHON cmp_vitis.py <unit> <vitis project dir> [--n 1024]

A 10-line Verilog wrapper maps ap_fifo onto the harness's valid/ready names
(``empty_n``->vld, ``read``->rdy, ``dout``->dat; ``write``->vld,
``full_n``->rdy, ``din``->dat), starts one call (``ap_start`` until ``ap_ready``) and inverts the reset; the ``#0`` initial delays are
stripped from simulation copies (Verilator 5 without ``--timing``).
"""
import argparse, glob, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, os.getcwd())
from examples.minitpu.harness import rtl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("unit"); ap.add_argument("prj")
ap.add_argument("--n", type=int, default=1024)
a = ap.parse_args()
u = __import__(f"examples.minitpu.units.{a.unit}", fromlist=["x"])
stim = u.stimulus()[: a.n].astype(np.uint64)
vdir = os.path.abspath(os.path.join(a.prj, "out.prj/solution1/syn/verilog"))
top = open(os.path.join(vdir, "add_0.v")).read()
ins = re.findall(r"input\s+\[(\d+):0\] (\w+)_dout;", top)
outs = re.findall(r"output\s+\[(\d+):0\] (\w+)_din;", top)
w = ["module vwrap(input clk, input rst,"]
w += [f"  input [{h}:0] {p}_dat, input {p}_vld, output {p}_rdy," for h, p in ins]
w += [f"  output [{h}:0] {p}_dat, output {p}_vld, input {p}_rdy," for h, p in outs]
w[-1] = w[-1].rstrip(",") + ");"
# ap_start: one call only. Held high, Vitis starts the next call while the
# last one drains, and the next call's blocked FIFO read stalls that drain.
w += ["  wire done, idle, ready; reg called;",
      "  always @(posedge clk) if (!rst) called <= 1'b0; else if (ready) called <= 1'b1;",
      "  add_0 u(.ap_clk(clk), .ap_rst(~rst), .ap_start(~called), .ap_done(done), .ap_idle(idle), .ap_ready(ready),"]
w += [f"    .{p}_dout({p}_dat), .{p}_empty_n({p}_vld), .{p}_read({p}_rdy)," for _, p in ins]
w += [f"    .{p}_din({p}_dat), .{p}_full_n({p}_rdy), .{p}_write({p}_vld)," for _, p in outs]
w[-1] = w[-1].rstrip(",") + ");"
w += ["endmodule"]
wrap = os.path.join(os.path.abspath(a.prj), "vwrap.v")
open(wrap, "w").write("\n".join(w) + "\n")
# Vitis's `initial begin #0 x = ...; end` needs --timing in Verilator 5; the
# harness builds without it, so simulate copies with the `#0 ` delays removed.
sdir = os.path.join(os.path.abspath(a.prj), "vsim")
os.makedirs(sdir, exist_ok=True)
srcs = [wrap]
for v in sorted(glob.glob(os.path.join(vdir, "*.v"))):
    dst = os.path.join(sdir, os.path.basename(v))
    open(dst, "w").write(open(v).read().replace("#0 ", ""))
    srcs.append(dst)
want, _ = rtl.run(u.RTL, stim)
vu = rtl.RtlUnit(top="vwrap", sources=srcs, inputs=[(p, int(h) + 1) for h, p in ins],
                 outputs=[(p, int(h) + 1) for h, p in outs], shape="stream", clk="clk", rst_n="rst")
got, cyc = rtl.run(vu, stim)
st = dict(rtl.last_stats or {})
lat = dict(zip(*[x.tolist() for x in np.unique(cyc, return_counts=True)]))
r = ET.parse(os.path.join(a.prj, "out.prj/solution1/syn/report/add_0_csynth.xml")).getroot()
lp = r.find("PerformanceEstimates/SummaryOfLoopLatency")[0]
est = r.findtext("PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")
tgt = r.findtext("UserAssignments/TargetClockPeriod")
print(f"VITIS-CMP {a.unit} {os.path.basename(a.prj.rstrip('/'))}: values {int((got[:, 0] == want[:, 0]).sum())}/{len(stim)}; "
      f"measured latency {lat}, {st.get('cycles', 0) / len(stim):.4f} cyc/vector; "
      f"report: PipelineDepth {lp.findtext('PipelineDepth')} II {lp.findtext('PipelineII')}, "
      f"est. clock {est} ns vs target {tgt} ns")
