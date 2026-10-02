"""One row per DC run: area (total/comb/non-comb), sequential cells, critical path, slack, wall.

    python summarize_dc.py <dc_dir>      (reads <dc_dir>/out/<run>/*.rpt and <dc_dir>/out_<run>.log)
"""
import glob, os, re, sys

d = sys.argv[1]
print("\t".join(["run", "total_um2", "comb_um2", "noncomb_um2", "seq_cells", "crit_path_ns", "slack_ns", "dc_wall"]))
for o in sorted(glob.glob(os.path.join(d, "out", "*"))):
    run = os.path.basename(o)
    try:
        area = open(glob.glob(os.path.join(o, "*.area.rpt"))[0]).read()
        qor = open(glob.glob(os.path.join(o, "*.qor.rpt"))[0]).read()
    except IndexError:
        print(f"{run}\tmissing reports")
        continue
    g = lambda pat, s: (re.search(pat, s).group(1) if re.search(pat, s) else "-")  # noqa: E731
    log = open(os.path.join(d, f"out_{run}.log")).read()
    print("\t".join([
        run,
        g(r"Total cell area:\s+([\d.]+)", area),
        g(r"Combinational area:\s+([\d.]+)", area),
        g(r"Noncombinational area:\s+([\d.]+)", area),
        g(r"Sequential Cell Count:\s+(\d+)", qor),
        g(r"Critical Path Length:\s+(-?[\d.]+)", qor),
        g(r"Critical Path Slack:\s+(-?[\d.]+)", qor),
        g(r"DC_EXIT \d+ WALL (\d+s)", log),
    ]))
