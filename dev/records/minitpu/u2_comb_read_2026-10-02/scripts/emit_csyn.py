"""Emit one unit variant for Catapult csyn and run Catapult on it (U2 comb-read copy).

The U2 pilot's ``emit_csyn.py`` plus ``--module FILE.py``: take ``<variant>`` from
``FILE.VARIANTS`` instead of the unit's (the record's ``rf_scalars.py``).

Original docstring follows.

Copied from ``u1_catapult_units_2026-10-02/scripts/emit_csyn.py``; adds
``--partition FN:ARR`` (Allo ``s.partition(..., partition_type=Complete)``),
``--width W`` (the regfile instance) and ``--pre-tcl`` (raw tcl before
``go assembly``, e.g. a ``MAP_TO_MODULE`` hand-patch).

Original docstring follows.

    python emit_csyn.py <unit> <variant> <prj> [--n N] [--unroll LOOP ...]
        [--pipeline LOOP ...] [--clock NS] [--io 'PUSH:POP=L' ...]
        [--synth-top KERNEL] [--flush] [--no-run]

From the worktree root, after ``source examples/minitpu/harness/env-zhang21.sh``.
Generalises the pilot's and the pipe record's emitters
(``u1_bf16_add_catapult_2026-10-02/scripts``, ``u1_pipe_2026-10-02/scripts``).

* ``<variant>`` is a key of the unit's ``VARIANTS``, or ``wire:<variant>`` for the
  Wire-port form ``gen_wire.py`` writes to ``wire/wire_<unit>_<variant>.py`` (then
  ``--synth-top`` names the kernel to synthesize, e.g. ``f_0``).
* ``--io 'v12.Push():v10.Pop()=L'`` is the **hand-patch** that pins a declared
  latency (Allo cannot say it, ``u1_pipe_2026-10-02.rst`` L1/L4): before
  ``go assembly`` it adds ``go architect`` and ``cycle set {PUSH} -from {POP}
  -equal L``. ``--io-all L`` pins every (output, input) pair of the synthesized
  kernel, op names read from the emitted SystemC (``<port>.Push()`` /
  ``<port>.Pop()`` for Connections, ``<port>.write()``/``<port>.read()``
  never needed so far).
* ``--flush`` is the pilot's P4 (``-PIPELINE_STALL_MODE flush``).

``run.tcl.emitted`` keeps the tcl as Allo wrote it; ``run.tcl`` is as run.
No kernel.cpp hand-patch is applied (the pilot's P1/P3 are fixed in the
emitter on ``u1-pilot``, ``c87462f7``/``bf6e4608``); ``kernel.cpp`` is as
emitted.
"""
import argparse, importlib, os, re, shutil, subprocess, sys, time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allo.dataflow as df  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("unit"); ap.add_argument("variant"); ap.add_argument("prj")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--pipeline", action="append", default=[])
ap.add_argument("--unroll", action="append", default=[])
ap.add_argument("--clock", type=float, default=None)
ap.add_argument("--io", action="append", default=[])
ap.add_argument("--io-all", type=int, default=None)
ap.add_argument("--synth-top", default=None)
ap.add_argument("--flush", action="store_true")
ap.add_argument("--tcl", action="append", default=[],
                help="raw tcl line added after go architect (hand-patch), e.g. a Wire-port cycle constraint")
ap.add_argument("--no-run", action="store_true")
ap.add_argument("--partition", action="append", default=[])
ap.add_argument("--width", type=int, default=0)
ap.add_argument("--module", default=None, help="take <variant> from this .py file instead of the unit")
ap.add_argument("--pre-tcl", action="append", default=[], help="raw tcl before go assembly (hand-patch)")
a = ap.parse_args()

u = importlib.import_module(f"examples.minitpu.units.{a.unit}")
if a.module:  # a variant file outside units/ (the record's rf_scalars.py)
    import importlib.util
    _spec = importlib.util.spec_from_file_location("rf_form", a.module)
    u_form = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(u_form)
n = a.n or len(u.stimulus())
if a.variant.startswith("wire:"):  # wire:<variant> -> wire/wire_<unit>_<variant>.py (gen_wire.py)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wire"))
    make = importlib.import_module(f"wire_{a.unit}_{a.variant[5:]}").wire
else:
    make, _ = (u_form if a.module else u).VARIANTS[a.variant]
s = df.customize(make(n, a.width) if a.width else make(n))
for arr in a.partition:
    from allo.customize import Partition
    fn, var = arr.split(":")
    s.partition(f"{fn}:{var}", partition_type=Partition.Complete, dim=1)
for loop in a.unroll:
    s.unroll(loop)
for loop in a.pipeline:
    s.pipeline(loop)
cfg = {}
if a.clock:
    cfg["clock_period"] = a.clock
if a.synth_top:
    cfg["synth_top"] = a.synth_top
if os.path.isdir(a.prj):
    shutil.rmtree(a.prj)
s.build(target="systemc", mode="csyn", project=a.prj, configs=cfg)
tcl = os.path.join(a.prj, "run.tcl")
t = open(tcl).read()
open(tcl + ".emitted", "w").write(t)
io = list(a.io)
if a.io_all is not None:
    k = open(os.path.join(a.prj, "kernel.cpp")).read()
    top = a.synth_top
    if not top:  # the kernel the region instantiates (one-kernel units)
        rb = k[k.index("SC_MODULE(top)"):]
        top = re.search(r"\n  (\w+) u0;", rb[: rb.index("\n};")]).group(1)
    blk = k[k.index(f"SC_MODULE({top})"):]
    blk = blk[: blk.index("\n};")]
    ins = re.findall(r"Connections::In< .+? > (\w+);", blk)
    outs = re.findall(r"Connections::Out< .+? > (\w+);", blk)
    io += [f"{o}.Push():{i}.Pop()={a.io_all}" for o in outs for i in ins]
add = "".join(f"{x}  ;# [hand-patch]\n" for x in a.pre_tcl)
if io or a.tcl:
    add += "go architect\n" + "".join(f"{x}  ;# [hand-patch: declared latency]\n" for x in a.tcl) + "".join(
        "cycle set {%s} -from {%s} -equal %s  ;# [hand-patch: declared latency]\n"
        % (c.split("=")[0].split(":")[0], c.split("=")[0].split(":")[1], c.split("=")[1])
        for c in io)
if a.flush:
    add = "directive set -PIPELINE_STALL_MODE flush  ;# [hand-patch P4]\n" + add
if add:
    assert t.count("go assembly\n") == 1, "run.tcl layout changed"
    t = t.replace("go assembly\n", add + "go assembly\n")
    open(tcl, "w").write(t)
print(f"EMITTED {a.unit} {a.variant} n={n} -> {a.prj} io={io} flush={a.flush}", flush=True)
if a.no_run:
    sys.exit(0)
os.makedirs(os.path.join(a.prj, "build"), exist_ok=True)
t0 = time.time()
r = subprocess.run("catapult -shell -f ../run.tcl > ../csyn.log 2>&1", shell=True,
                   cwd=os.path.join(a.prj, "build"))
with open(os.path.join(a.prj, "csyn.log"), "a") as f:
    f.write(f"CATAPULT_EXIT {r.returncode} WALL {time.time() - t0:.0f}s\n")
print(f"CATAPULT {a.unit} {a.variant} n={n} pipeline={a.pipeline} unroll={a.unroll} "
      f"clock={a.clock} io={io} tcl={a.tcl} flush={a.flush}: exit {r.returncode}, {time.time() - t0:.0f}s, "
      f"log {a.prj}/csyn.log", flush=True)
