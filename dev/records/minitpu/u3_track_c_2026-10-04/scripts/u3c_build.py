# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U3 track C: build one unit form through Allo's SystemC -> Catapult csyn flow.

    $ALLO_PYTHON u3c_build.py <unit> <variant> <prj> [--inst I] [--n N] [--clock NS]
        [--unroll LOOP ...] [--unroll-inner] [--pipeline LOOP] [--latency KERNEL=L ...]
        [--synth-top K] [--list-loops] [--no-run]

``<variant>`` is a key of the unit's ``VARIANTS`` (the landed track-A form) or
``form:<name>`` for a track-C form in ``../forms/<name>.py`` (``make(n, inst)``;
wide ports for the lane-array units, see the record). No hand-patch of
``kernel.cpp``; ``latency=`` is a build config (``configs["latency"]``) and
``run.tcl`` is as Allo wrote it. ``--unroll-inner`` unrolls every loop of the
kernel but the outermost band (the iteration loop), so a constant-trip inner
loop is not merged into the pipelined loop (latency_report: the one way a
manifest goes wrong). Prints ``BUILD ...`` and Allo's ``[latency]`` lines;
``<prj>/latency.json`` is the manifest.
"""
import argparse, importlib, inspect, os, shutil, sys, time

sys.path.insert(0, os.getcwd())
import allo.dataflow as df  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("unit"); ap.add_argument("variant"); ap.add_argument("prj")
ap.add_argument("--inst", default=None)
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--clock", type=float, default=None)
ap.add_argument("--unroll", action="append", default=[])
ap.add_argument("--unroll-inner", action="store_true")
ap.add_argument("--pipeline", action="append", default=[])
ap.add_argument("--pipeline-all", action="store_true", help="pipeline every kernel's iteration loop (composites)")
ap.add_argument("--pipeline-match", default=None, help="pipeline the iteration loop of the kernels whose name matches this regex")
ap.add_argument("--partition", action="append", default=[], help="s.partition(TARGET) complete, e.g. tx_0:tile (a Stateful into registers)")
ap.add_argument("--latency", action="append", default=[])
ap.add_argument("--synth-top", default=None)
ap.add_argument("--list-loops", action="store_true")
ap.add_argument("--tcl", action="append", default=[], help="Catapult directive line(s) inserted in run.tcl before `go analyze` (e.g. REGISTER_THRESHOLD); recorded as a hand-set directive")
ap.add_argument("--no-run", action="store_true")
a = ap.parse_args()

u = importlib.import_module(f"examples.minitpu.units.{a.unit}")
inst = a.inst or getattr(u, "DEFAULT", None)
if a.variant.startswith("form:"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "forms"))
    make = importlib.import_module(a.variant[5:]).make
else:
    make, _ = u.VARIANTS[a.variant]
n = a.n
if not n:
    if hasattr(u, "stimulus"):
        n = len(u.stimulus())
    else:
        from examples.minitpu.harness import check
        var = None if a.variant.startswith("form:") else a.variant
        n = len(next(iter(check._trace_all(u, inst, 0, var)[0].values())))
kw = {}
sig = inspect.signature(make).parameters
if "inst" in sig:
    kw["inst"] = inst
if "w" in sig:
    kw["w"] = getattr(u, "WIDTH", {}).get(inst, 0)
top = make(n, **kw)
s = df.customize(top)


def kernel_loops():
    """[(kernel func name, band name, [loop names outer..inner])] of every df.kernel
    (a band's loops are flattened by get_affine_loop_nests, outermost first)."""
    out = []
    for f in s.module.body.operations:
        try:
            f.attributes["df.kernel"]
        except (KeyError, AttributeError, IndexError):
            continue
        fn = str(f.name).strip('"')
        loops = s.get_loops(fn)
        for band_name in loops.loops:
            out.append((fn, band_name, list(loops[band_name].loops.keys())))
    return out


if a.list_loops:
    for fn, band, names in kernel_loops():
        print(f"LOOPS {fn} {band}: {names}")
unrolls = list(a.unroll)
def main_loops():
    """Per kernel, the iteration loop: the outermost loop of its deepest band."""
    kl = kernel_loops()
    out = {}
    for fn, band, names in kl:
        if fn not in out or len(names) > len(out[fn][1]):
            out[fn] = (band, names)
    return {fn: (band, names) for fn, (band, names) in out.items()}


if a.unroll_inner:
    # every loop but each kernel's iteration loop: a rolled constant-trip loop
    # would be merged into the pipelined loop
    mains = main_loops()
    for fn, band, names in kernel_loops():
        mb, mn = mains[fn]
        for nm in (names[1:] if band == mb else names):
            unrolls.append(f"{fn}:{nm}")
if a.pipeline_all or a.pipeline_match:
    import re as _re
    for fn, (band, names) in main_loops().items():
        if a.pipeline_all or _re.search(a.pipeline_match, fn):
            a.pipeline.append(f"{fn}:{names[0]}")
for lp in unrolls:
    s.unroll(lp)
for lp in a.pipeline:
    s.pipeline(lp)
for tg in a.partition:
    s.partition(tg)
cfg = {}
if a.clock:
    cfg["clock_period"] = a.clock
if a.synth_top:
    cfg["synth_top"] = a.synth_top
if a.latency:
    cfg["latency"] = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.latency}
if os.path.isdir(a.prj):
    shutil.rmtree(a.prj)
t0 = time.time()
tag = f"{a.unit} {a.variant} inst={inst} n={n} unroll={unrolls} pipeline={a.pipeline} partition={a.partition} tcl={a.tcl} {cfg}"
try:
    if a.no_run:
        s.build(target="systemc", mode="csyn", project=a.prj, configs=cfg)
        print(f"EMITTED {tag} -> {a.prj} ({time.time() - t0:.0f}s)", flush=True)
        sys.exit(0)
    mod = s.build(target="systemc", mode="csyn", project=a.prj, configs=cfg)
    if a.tcl:
        tp = os.path.join(a.prj, "run.tcl")
        t = open(tp).read()
        assert t.count("go analyze\n") == 1, "run.tcl layout changed"
        open(tp + ".emitted", "w").write(t)
        open(tp, "w").write(t.replace("go analyze\n", "".join(x + "  ;# [u3c --tcl]\n" for x in a.tcl) + "go analyze\n"))
    mod()
    print(f"BUILD {tag}: ok {time.time() - t0:.0f}s", flush=True)
except SystemExit:
    raise
except Exception as e:  # noqa: BLE001
    print(f"BUILD {tag}: {type(e).__name__}: {str(e)[:1500]} ({time.time() - t0:.0f}s)", flush=True)
    sys.exit(1)
