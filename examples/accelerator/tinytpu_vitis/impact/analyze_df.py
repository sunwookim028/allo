"""Per-process timeline from Vitis cosim's dataflow monitor.

    python analyze_df.py <prj>/out.prj/solution1/sim/verilog

Needs a cosim run with `-enable_dataflow_profiling`, then the xsim snapshot
re-run in that directory (cosim deletes the CSVs; `profile.sh` does both).
Per process: start/done cycle, and of the cycles in between how many it was
RUNNING, STARVED (blocked reading an empty FIFO) or BLOCKED (writing a full
one). Processes are sorted by done time; the last to finish is the tail."""
import os, re, sys
d = sys.argv[1]
mon = open(os.path.join(d, "dataflow_monitor.sv")).read()
names = {int(n): p for n, p in re.findall(
    r"assign process_intf_(\d+)\.ap_start = AESL_inst_tinytpu_isa\.(\w+?)_U0\.ap_start", mon)}
rows = []
for n, p in sorted(names.items()):
    st = [l.strip() for l in open(os.path.join(d, f"status{n}.csv"))]
    sd = st[12].split(",") if len(st) > 12 else []
    stall = [int(x) for x in open(os.path.join(d, f"stalling{n}.csv")).read().split()]
    try:
        s0 = int(sd[0]); dn = int(sd[1])
    except Exception:
        s0, dn = 0, len(stall)
    win = stall[s0:dn]
    rows.append((dn, p, s0, win.count(0), win.count(1), win.count(2)))
print(f"{'process':34s} {'start':>5s} {'done':>5s} {'run':>5s} {'starve':>6s} {'block':>5s}")
for dn, p, s0, r0, r1, r2 in sorted(rows):
    print(f"{p:34s} {s0:5d} {dn:5d} {r0:5d} {r1:6d} {r2:5d}")
