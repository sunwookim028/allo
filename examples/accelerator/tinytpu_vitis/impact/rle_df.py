"""Run-length stall trace of chosen processes: R=running S=starved B=blocked.
    python rle_df.py <sim/verilog dir> proc [proc ...]"""
import os, re, sys
d = sys.argv[1]
mon = open(os.path.join(d, "dataflow_monitor.sv")).read()
names = {p: int(n) for n, p in re.findall(
    r"assign process_intf_(\d+)\.ap_start = AESL_inst_tinytpu_isa\.(\w+?)_U0\.ap_start", mon)}
for p in sys.argv[2:]:
    n = names[p]
    st = [int(x) for x in open(os.path.join(d, f"stalling{n}.csv")).read().split()]
    sd = [l.strip() for l in open(os.path.join(d, f"status{n}.csv"))][12].split(",")
    s0, dn = int(sd[0]), int(sd[1])
    out, cur, ln, t0 = [], None, 0, s0
    for t in range(s0, dn):
        c = "RSB?"[min(st[t], 3)] if st[t] in (0, 1, 2) else "?"
        if c != cur:
            if cur: out.append(f"{cur}{ln}@{t0}")
            cur, ln, t0 = c, 0, t
        ln += 1
    out.append(f"{cur}{ln}@{t0}")
    print(f"{p}: " + " ".join(out))
