"""Build the D-12 regfile prototype for Catapult and print the cmp_rf.py ports.

    python d12_csyn.py <lowering> <prj> [--n 64] [--clock 3.33] [--width 16]
        [--no-run] [--pre-tcl LINE ...]
    python d12_csyn.py server <prj> --unit word_array --inst narrow|mid [--reset]
        [--ii 1]

From the worktree root, after ``source examples/minitpu/harness/env-zhang21.sh``.
The four port kernels (and a server, if any) are synthesized as one unit,
``rf_d12g`` (``configs["synth_group"]``); ``src``/``sink`` stay in csim only.
Each port kernel's steady-state loop is pipelined (``s.pipeline("<k>:_")``), as
the one-kernel forms' ``rf_0:_``. ``memory.json`` is written beside
``latency.json``. ``--pre-tcl`` is a hand-patch line before ``go assembly``
(recorded as such). Prints ``PORTS <in1..in6>:<out1..out3>`` in MiniTPU order.
"""
import argparse, os, re, shutil, subprocess, sys, time

sys.path.insert(0, os.getcwd())
import importlib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("lowering"); ap.add_argument("prj")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--clock", type=float, default=3.33)
ap.add_argument("--width", type=int, default=16)
ap.add_argument("--no-run", action="store_true")
ap.add_argument("--no-pipeline", action="store_true")
ap.add_argument("--pre-tcl", action="append", default=[])
ap.add_argument("--unit", default="regfile", choices=("regfile", "word_array"))
ap.add_argument("--inst", default="narrow")
ap.add_argument("--reset", action="store_true", help="word_array: reset storage (RAM-mappable)")
ap.add_argument("--ii", type=int, default=1)
ap.add_argument("--post-tcl", action="append", default=[],
                help="hand-patch line after `go architect`, before `go extract` (e.g. ignore_memory_precedences)")
a = ap.parse_args()

if a.unit == "regfile":
    d = importlib.import_module("examples.minitpu.units.vpu_regfile_d12")
    arch, mem, top_name = d.architecture(a.n, a.width), "vreg", "rf_d12"
else:
    d = importlib.import_module("examples.minitpu.units.vpu_word_array_d12")
    arch, mem, top_name = d.architecture(a.n, a.inst, reset=a.reset), "vmem", "wa_d12"
low = {mem: a.lowering}
kernels = arch.port_kernels("systemc", low)


def schedule(s):
    if not a.no_pipeline:
        for k in kernels:
            s.pipeline(f"{k}:_", initiation_interval=a.ii)


if os.path.isdir(a.prj):
    shutil.rmtree(a.prj)
group = f"{top_name}g"
cfg = {"clock_period": a.clock, "synth_group": {"name": group, "kernels": kernels}}
mod = arch.build("systemc", low, schedule, mode="csyn", project=a.prj, configs=cfg)
k = open(os.path.join(a.prj, "kernel.cpp")).read()
top = k[k.index(f"SC_MODULE({top_name}) {{"):]
top = top[: top.index("\n};")]
insts = dict((i, t) for t, i in re.findall(r"\n  (\w+) (u\d+);", top))
binds = re.findall(r"\n    (u\d+)\.(\w+)\((\w+)\);", top)
src = [s for i, p, s in binds if insts[i] == "src_0" and p not in ("clk", "rst", "done")]
sink = [s for i, p, s in binds if insts[i] == "sink_0" and p not in ("clk", "rst", "done")]
blk = k[k.index("SC_MODULE(src_0)"):]
outs = set(re.findall(r"sc_out< .+? > (\w+);", blk[: blk.index("\n};")]))
src = [s for (i, p, s) in binds if insts[i] == "src_0" and p in outs]
blk = k[k.index("SC_MODULE(sink_0)"):]
ins = set(re.findall(r"sc_in< .+? > (\w+);", blk[: blk.index("\n};")]))
sink = [s for (i, p, s) in binds if insts[i] == "sink_0" and p in ins]
print(f"KERNELS {kernels}")
print(f"PORTS {','.join(src)}:{','.join(sink)}", flush=True)
tcl = os.path.join(a.prj, "run.tcl")
t = open(tcl).read()
open(tcl + ".emitted", "w").write(t)
if a.pre_tcl:
    t = t.replace("go assembly\n", "".join(f"{x}  ;# [hand-patch]\n" for x in a.pre_tcl) + "go assembly\n")
if a.post_tcl:
    assert t.count("go extract\n") == 1
    t = t.replace("go extract\n", "go architect\n" + "".join(
        f"{x}  ;# [hand-patch]\n" for x in a.post_tcl) + "go extract\n")
if a.pre_tcl or a.post_tcl:
    open(tcl, "w").write(t)
if a.no_run:
    sys.exit(0)
t0 = time.time()
os.makedirs(os.path.join(a.prj, "build"), exist_ok=True)
r = subprocess.run("catapult -shell -f ../run.tcl > ../csyn.log 2>&1", shell=True,
                   cwd=os.path.join(a.prj, "build"))
with open(os.path.join(a.prj, "csyn.log"), "a") as f:
    f.write(f"CATAPULT_EXIT {r.returncode} WALL {time.time() - t0:.0f}s\n")
print(f"CATAPULT {a.lowering}: exit {r.returncode}, {time.time() - t0:.0f}s, log {a.prj}/csyn.log")
if r.returncode == 0:
    from allo.backend.catapult import write_latency_manifest
    man = write_latency_manifest(a.prj, group)
    for kk, u in sorted(man["units"].items()):
        print(f"[latency] {kk}: {u}")
