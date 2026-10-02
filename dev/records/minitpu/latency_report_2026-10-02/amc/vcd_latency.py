# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Measure per-element latency of the pipelined bf16 `bits` kernel from a VCD.
# Usage (source env.sh; under scl enable gcc-toolset-13 --):
#   AMC_TARGET_CLOCK_PERIOD_NS=<ns> python vcd_latency.py [N]
# Builds with s.unroll(offset)+s.pipeline(i), runs Verilator with AMC_VCD set,
# and reports, per element i, the posedge index at which mem0 (av) is read at
# address i (en=1) and at which mem2 (cv) is written at address i (we=1),
# plus start/done edges, next to the schedule dump's `II / latency`.
import sys, os, re, importlib.util
import numpy as np

U1 = "/work/shared/users/phd/sk3463/scratch/wt-lat/dev/records/minitpu/u1_bf16_add_amc"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
period = os.environ.get("AMC_TARGET_CLOCK_PERIOD_NS", "10.0")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


import allo
kp = os.path.join(os.environ["TMPDIR"], f"kbv_{N}_{os.getpid()}.py")
open(kp, "w").write(open(f"{U1}/kernel_bits_amc.py").read().replace("N = 16", f"N = {N}"))
K = load(f"kbv_{N}", kp)
s = allo.customize(K.bf16_add_bits_amc)
loops = s.get_loops()
s.unroll(loops["S_i_0"]["offset"])
s.pipeline(loops["S_i_0"]["i"])
f = s.build(target="amc")
ls = str(f.loopSchedule)
rep = re.findall(r"loopschedule\.pipeline II = (\d+) trip_count = (\d+) latency = (\d+)", ls)
vcd = os.path.join(os.environ["TMPDIR"], f"lat_{period}_{N}_{os.getpid()}.vcd")
os.environ["AMC_VCD"] = vcd
rng = np.random.default_rng(1)
a = rng.integers(0, 1 << 16, N, dtype=np.uint16)
b = rng.integers(0, 1 << 16, N, dtype=np.uint16)
c = np.zeros(N, np.uint16)
f(a, b, c)
cycles = f.rpt["cycles"]

# --- minimal VCD parser: map ids of the DUT-scope signals we need ---
want = {"clk", "rst", "start", "done", "mem0_bram0_addr", "mem0_bram0_en",
        "mem2_bram0_addr", "mem2_bram0_we"}
ids, scope = {}, []
vals, edges = {}, []
with open(vcd) as fh:
    for line in fh:
        t = line.split()
        if not t:
            continue
        if t[0] == "$scope":
            scope.append(t[2])
        elif t[0] == "$upscope":
            scope.pop()
        elif t[0] == "$var":
            name = t[4]
            # DUT instance: innermost scope is the kernel instance, not top
            if name in want and len(scope) >= 2 and name not in ids.values():
                ids.setdefault(t[3], name)
        elif t[0] == "$enddefinitions":
            break
    snap = {}
    for line in fh:
        line = line.strip()
        if not line:
            continue
        if line[0] == "#":
            continue
        if line[0] in "01xz" and line[1:] in ids:
            nm, v = ids[line[1:]], line[0]
        elif line[0] == "b":
            bits, i = line[1:].split()
            if i not in ids:
                continue
            nm, v = ids[i], bits
        else:
            continue
        if nm == "clk" and v == "1":
            # rising edge: record the values that were stable before the edge
            edges.append(dict(snap))
        snap[nm] = v
toint = lambda x: int(x, 2) if x and set(x) <= {"0", "1"} else None
rst_end = next(k for k, e in enumerate(edges) if e.get("rst") == "0")
E = edges[rst_end:]  # edge 0 = first posedge after reset deasserts
rd, wr, st, dn = {}, {}, None, None
for k, e in enumerate(E):
    if e.get("start") == "1" and st is None:
        st = k
    if e.get("done") == "1" and dn is None:
        dn = k
    if e.get("mem0_bram0_en") == "1":
        rd.setdefault(toint(e.get("mem0_bram0_addr")), k)
    if e.get("mem2_bram0_we") == "1":
        wr[toint(e.get("mem2_bram0_addr"))] = k
lat = sorted({wr[i] - rd[i] for i in range(N) if i in rd and i in wr})
print(f"RESULT period={period} N={N} report(II,trip,latency)={rep} tb_cycles={cycles}"
      f" start_edge={st} done_edge={dn} first_read={rd.get(0)} last_write={wr.get(N-1)}"
      f" read->write_edges_per_elem={lat} match_ok={bool((c != 0).any())}")
