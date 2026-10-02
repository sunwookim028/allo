"""Emit one U1 unit variant for Catapult csyn (whole region ``top``), and run it.

    python emit_csyn.py <unit> <variant> <prj> [--n N] [--unroll LOOP ...] [--pipeline LOOP ...]
                        [--clock NS] [--cycles LOOP=L ...]

``--cycles LOOP=L`` is a hand-patch to the emitted ``run.tcl`` (a workaround,
not a feature: Allo has no way to say it): before ``go assembly`` it adds
``go architect`` and ``cycle set LOOP -equal L``, Catapult's cycle constraint
that fixes the number of c-steps of one iteration of LOOP (useref, "cycle
set"; it becomes the loop's CSTEPS_FROM directive). LOOP is Catapult's loop
name, e.g. ``l_S_i_0_i``. ``--io 'v12.Push():v10.Pop()=L'`` instead pins
the output write L c-steps after the input read (``cycle set ... -from``);
the op names are what ``cycle find_op *Push*`` returns after ``go architect``. Catapult runs from ``<prj>/build`` (needs MGC_HOME).
"""
import argparse, importlib, os, subprocess, sys, time

sys.path.insert(0, os.getcwd())
import allo.dataflow as df  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("unit"); ap.add_argument("variant"); ap.add_argument("prj")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--pipeline", action="append", default=[])
ap.add_argument("--unroll", action="append", default=[])
ap.add_argument("--clock", type=float, default=None)
ap.add_argument("--cycles", action="append", default=[])
ap.add_argument("--io", action="append", default=[],
                help="PUSH:POP=L -> cycle set {PUSH} -from {POP} -equal L")
a = ap.parse_args()

u = importlib.import_module(f"examples.minitpu.units.{a.unit}")
n = a.n or len(u.stimulus())
make, _ = u.VARIANTS[a.variant]
s = df.customize(make(n))
for loop in a.unroll:
    s.unroll(loop)
for loop in a.pipeline:
    s.pipeline(loop)
cfg = {}
if a.clock:
    cfg["clock_period"] = a.clock
s.build(target="systemc", mode="csyn", project=a.prj, configs=cfg)
tcl = os.path.join(a.prj, "run.tcl")
t = open(tcl).read()
if a.cycles or a.io:
    add = "go architect\n" + "".join(
        f"cycle set {c.split('=')[0]} -equal {c.split('=')[1]}  ;# [hand-patch: declared latency]\n"
        for c in a.cycles) + "".join(
        "cycle set {%s} -from {%s} -equal %s  ;# [hand-patch: declared latency]\n"
        % (c.split("=")[0].split(":")[0], c.split("=")[0].split(":")[1], c.split("=")[1])
        for c in a.io)
    # after the emitted library adds (none is allowed past 'libraries')
    assert t.count("go assembly\n") == 1
    t = t.replace("go assembly\n", add + "go assembly\n")
    open(tcl, "w").write(t)
os.makedirs(os.path.join(a.prj, "build"), exist_ok=True)
t0 = time.time()
r = subprocess.run("catapult -shell -f ../run.tcl > ../csyn.log 2>&1", shell=True,
                   cwd=os.path.join(a.prj, "build"))
print(f"CATAPULT {a.unit} {a.variant} n={n} pipeline={a.pipeline} unroll={a.unroll} clock={a.clock} "
      f"cycles={a.cycles} io={a.io}: exit {r.returncode}, {time.time() - t0:.0f}s, log {a.prj}/csyn.log")
