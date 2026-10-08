# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""D-21 inverse angle, FPGA half: build an Allo MXU form through Allo's Vitis
HLS (``target="vhls"``) emitter and run Vitis HLS csynth on it.

    $ALLO_PYTHON vhls_build.py <form.py> <prj> [--inst dim2] [--n N] [--clock 5]
        [--depth Q] [--partition KERNEL:ARRAY ...] [--iface ap_fifo|axis]
        [--ctrl ap_ctrl_none|ap_ctrl_hs] [--no-run] [--part xczu7ev-ffvc1156-2-e]

``<form.py>`` defines ``make(n, inst=...)`` returning an Allo region (the
``mxu_wide`` form of ``minitpu_rtl_mxu_2026-10-08``). The schedule is the
Catapult record's: every kernel's iteration loop pipelined (II 1), every
inner loop unrolled, the named arrays partitioned complete. ``--depth`` sets
the PE grid links' depth (``MRTL_PE_DEPTH``, read by the form).

Allo writes ``kernel.cpp`` (not edited). This script writes its OWN
``run.tcl`` (Allo's ``codegen_tcl`` has no interface directives): the region's
array arguments are read/written once per iteration, in order, so each is
given ``set_directive_interface -mode <iface>`` -- an ``ap_fifo`` port
(``_dout/_empty_n/_read``, ``_din/_full_n/_write``) or ``axis``
(``_TDATA/_TVALID/_TREADY``) -- and the top's block protocol ``--ctrl``.
These are tcl directives against the emitted names, recorded beside the
result; the C++ is Allo's verbatim. Prints ``BUILD ...`` lines; the csynth
report is ``<prj>/out.prj/solution1/syn/report/``.
"""
import argparse, importlib.util, inspect, os, re, shutil, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("form"); ap.add_argument("prj")
ap.add_argument("--inst", default="dim2")
ap.add_argument("--n", type=int, default=1 << 20)
ap.add_argument("--clock", type=float, default=5.0)
ap.add_argument("--depth", default=None)
ap.add_argument("--partition", action="append", default=[])
ap.add_argument("--tcl-partition", action="store_true",
                help="apply --partition as Vitis set_directive_array_partition -type complete (the arrays are locals of "
                     "the kernel functions in kernel.cpp, names kept) instead of Allo's s.partition, whose whole-module "
                     "use-def walk per call does not scale to DIM 16 (record, finding 6)")
ap.add_argument("--no-unroll-inner", action="store_true")
ap.add_argument("--no-pipeline", action="store_true")
ap.add_argument("--iface", default="ap_fifo", choices=["ap_fifo", "axis", "none"])
ap.add_argument("--ctrl", default="ap_ctrl_none", choices=["ap_ctrl_none", "ap_ctrl_hs", "ap_ctrl_chain", "none"])
ap.add_argument("--part", default="xczu7ev-ffvc1156-2-e")  # ZCU104, MiniTPU's board
ap.add_argument("--tcl", action="append", default=[], help="extra directive line(s) before csynth_design")
ap.add_argument("--tcl-file", default=None, help="directive lines from a file (balance_depths.py --out), before csynth_design")
ap.add_argument("--export", action="store_true", help="also export_design -rtl verilog -format ip_catalog")
ap.add_argument("--no-run", action="store_true")
a = ap.parse_args()

if a.depth:
    os.environ["MRTL_PE_DEPTH"] = str(a.depth)
sys.path.insert(0, os.getcwd())
import allo.dataflow as df  # noqa: E402

spec = importlib.util.spec_from_file_location("form", a.form)
form = importlib.util.module_from_spec(spec)
spec.loader.exec_module(form)
kw = {"inst": a.inst} if "inst" in inspect.signature(form.make).parameters else {}
t0 = time.time()
top = form.make(a.n, **kw)
print(f"PHASE make {time.time() - t0:.0f}s", flush=True)
s = df.customize(top)
print(f"PHASE customize {time.time() - t0:.0f}s", flush=True)


def kernel_loops():
    out = []
    for f in s.module.body.operations:
        try:
            f.attributes["df.kernel"]
        except (KeyError, AttributeError, IndexError):
            continue
        fn = str(f.name).strip('"')
        loops = s.get_loops(fn)
        for band in loops.loops:
            out.append((fn, band, list(loops[band].loops.keys())))
    return out


kl = kernel_loops()
mains = {}
for fn, band, names in kl:
    if fn not in mains or len(names) > len(mains[fn][1]):
        mains[fn] = (band, names)
unrolls, pipes = [], []
if not a.no_unroll_inner:
    for fn, band, names in kl:
        mb, _ = mains[fn]
        for nm in (names[1:] if band == mb else names):
            if sum(nm in ns for f2, b2, ns in kl if f2 == fn) > 1:
                print(f"UNROLL-SKIP {fn}:{nm} (ambiguous name across bands)")
                continue
            unrolls.append(f"{fn}:{nm}")
if not a.no_pipeline:
    pipes = [f"{fn}:{names[0]}" for fn, (band, names) in mains.items()]
for lp in unrolls:
    s.unroll(lp)
print(f"PHASE unroll x{len(unrolls)} {time.time() - t0:.0f}s", flush=True)
for lp in pipes:
    s.pipeline(lp)
print(f"PHASE pipeline x{len(pipes)} {time.time() - t0:.0f}s", flush=True)
if not a.tcl_partition:
    for tg in a.partition:
        s.partition(tg)
        print(f"PHASE partition {tg} {time.time() - t0:.0f}s", flush=True)
t_sched = time.time() - t0
if os.path.isdir(a.prj):
    shutil.rmtree(a.prj)
mod = s.build(target="vhls", mode="csyn", project=a.prj,
              configs={"device": "zcu104", "frequency": round(1000 / a.clock)})
t_emit = time.time() - t0
print(f"PHASE build(vhls) {t_emit:.0f}s", flush=True)
code = open(os.path.join(a.prj, "kernel.cpp")).read()
# the top: the function marked by Allo as top (the region), and its array arguments in order
top_name = s.top_func_name
m = re.search(r"^void " + re.escape(top_name) + r"\((.*?)\)\s*\{", code, re.M | re.S)
assert m, f"top {top_name} not found in kernel.cpp"
args = [x.strip() for x in m.group(1).split(",") if x.strip()]
arg_names = [re.search(r"(\w+)\s*(\[|$)", x).group(1) for x in args]
dirs = []
if a.ctrl != "none":
    dirs.append(f'set_directive_interface -mode {a.ctrl} "{top_name}"')
if a.iface != "none":
    for nm in arg_names:
        dirs.append(f'set_directive_interface -mode {a.iface} "{top_name}" {nm}')
if a.tcl_partition:
    for tg in a.partition:
        fn, arr = tg.split(":")
        dirs.append(f"set_directive_array_partition -type complete -dim 0 {fn} {arr}")
dirs += a.tcl
if a.tcl_file:
    dirs += [ln for ln in open(a.tcl_file).read().splitlines() if ln.strip() and not ln.startswith("#")]
tcl = f"""# GENERATED by examples/minitpu/fpga/vhls_build.py -- the flow's own run.tcl (Allo's kernel.cpp verbatim)
open_project out.prj -reset
open_solution -reset solution1 -flow_target vivado
set_top {top_name}
add_files kernel.cpp
set_part {{{a.part}}}
create_clock -period {a.clock}
{chr(10).join(dirs)}
csynth_design
{"export_design -rtl verilog -format ip_catalog" if a.export else ""}
exit
"""
open(os.path.join(a.prj, "run.tcl.allo"), "w").write(open(os.path.join(a.prj, "run.tcl")).read())
open(os.path.join(a.prj, "run.tcl"), "w").write(tcl)
tag = (f"{os.path.basename(a.form)} inst={a.inst} n={a.n} depth={a.depth} clock={a.clock} iface={a.iface} "
       f"ctrl={a.ctrl} unroll={len(unrolls)} pipeline={pipes} partition={a.partition}{" (tcl)" if a.tcl_partition else ""} tcl={a.tcl} tcl_file={a.tcl_file}")
print(f"EMITTED {tag} top={top_name} args={arg_names} ({t_emit:.0f}s, schedule {t_sched:.0f}s)", flush=True)
if a.no_run:
    sys.exit(0)
t1 = time.time()
with open(os.path.join(a.prj, "csynth.log"), "w") as f:
    rc = subprocess.call(["vitis_hls", "-f", "run.tcl"], cwd=a.prj, stdout=f, stderr=subprocess.STDOUT)
print(f"BUILD {tag}: vitis_hls rc={rc} {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s)", flush=True)
sys.exit(rc)
