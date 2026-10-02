# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive a Vitis HLS csynth project from Allo's emitted one (hand-patch, recorded):
synthesize the unit kernel itself (``set_top <k>``) with its arrays as ``ap_fifo``
ports (sequential access, so the port shape is a stream: the harness can drive
it), at a clock, optionally with ``#pragma HLS latency min=L max=L`` in the
pipelined loop body (Vitis's only latency directive; Allo cannot emit it).

    python variant.py <allo prj> <out dir> <kernel> <period ns> [--latency L] [--max-only]
"""
import argparse, os, re, shutil

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("kernel"); ap.add_argument("period")
ap.add_argument("--latency", type=int, default=None)
ap.add_argument("--max-only", action="store_true")
a = ap.parse_args()
os.makedirs(a.dst, exist_ok=True)
k = open(os.path.join(a.src, "kernel.cpp")).read()
shutil.copy(os.path.join(a.src, "kernel.h"), a.dst)
m = re.search(r"void %s\(\n((?:  .*\n)+?)\) \{[^\n]*\n" % a.kernel, k)
ports = re.findall(r"(\w+)\[\d+\]", m.group(1))
prag = "".join(f"  #pragma HLS interface mode=ap_fifo port={p}\n" for p in ports)
body_start = m.end()
k = k[:body_start] + prag + k[body_start:]
if a.latency is not None:
    # first pipeline pragma after the kernel's signature: the I/O loop
    i = k.index("#pragma HLS pipeline II=1", body_start)
    j = k.index("\n", i) + 1
    lat = f"max={a.latency}" if a.max_only else f"min={a.latency} max={a.latency}"
    k = k[:j] + f"  #pragma HLS latency {lat}\n" + k[j:]
open(os.path.join(a.dst, "kernel.cpp"), "w").write(k)
open(os.path.join(a.dst, "run.tcl"), "w").write(f"""open_project out.prj -reset
open_solution -reset solution1 -flow_target vivado
set_top {a.kernel}
add_files kernel.cpp
set_part {{xcu280-fsvh2892-2L-e}}
create_clock -period {a.period}
csynth_design
exit
""")
print("WROTE", a.dst, ports, a.latency)
