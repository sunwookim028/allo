# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Can an exact unit depth be imposed AFTER AMC's scheduler? Build the bf16 bits
# kernel (unroll+pipeline) at 10 ns, where AMC schedules the datapath in one
# cycle (pipeline latency = 2), then hand-edit the loopschedule IR between
# lower_amc_to_loopschedule_vivado and lower_loopschedule_to_fsm: move the
# result store from stage 1 to a new stage 2 (one register on the datapath
# output) and set `latency = 3`. Simulate in Verilator and compare.
# Usage: python retime_by_hand.py [N] [mode]   mode: store2 (default) | latonly
#   latonly: only bump the `latency` attribute 2 -> 3, no op moved.
import sys, os, re, importlib.util
import numpy as np
import allo
import allo.backend.amc as amcmod
from amc_mlir.ir import Module

U1 = "/work/shared/users/phd/sk3463/scratch/wt-lat/dev/records/minitpu/u1_bf16_add_amc"
ROOT = "/work/shared/users/phd/sk3463/scratch/wt-lat"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
mode = sys.argv[2] if len(sys.argv) > 2 else "store2"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ref = load("ref", f"{ROOT}/examples/minitpu/harness/ref.py")
real = amcmod.amc.lower_amc_to_loopschedule_vivado


def edit(txt):
    m = re.search(r"loopschedule\.pipeline II = 1 trip_count = (\d+) latency = 2", txt)
    assert m, "expected a latency-2 pipeline"
    txt = txt.replace(m.group(0), m.group(0)[:-1] + "3")
    if mode == "latonly":
        return txt
    # stage 1: `%8 = loopschedule.at 1 -> i6 { ... amc.store %V, %P[%7#1 : i5] {..} : T
    #           loopschedule.yield %7#0 : i6 }`
    st = re.search(r"( *)amc\.store (%\w+), (%\w+)\[%7#1 : i5\] (\{[^}]*\}) : ([^\n]+)\n"
                   r"( *)loopschedule\.yield %7#0 : i6\n( *)\}\n", txt)
    assert st, "store pattern not found"
    ind, val, port, attrs, ty, yind, cind = st.groups()
    new = (f"{yind}loopschedule.yield %7#0, %7#1, {val} : i6, i5, i16\n{cind}}}\n"
           f"{cind}%9 = loopschedule.at 2 -> i6 {{\n"
           f"{ind}amc.store %8#2, {port}[%8#1 : i5] {attrs} : {ty}\n"
           f"{yind}loopschedule.yield %8#0 : i6\n{cind}}}\n")
    txt = txt[:st.start()] + new + txt[st.end():]
    txt = txt.replace("%8 = loopschedule.at 1 -> i6 {", "%8:3 = loopschedule.at 1 -> (i6, i5, i16) {")
    return txt


def patched(module, ctx, *a):
    ok = real(module, ctx, *a)
    if not ok:
        return ok
    new = Module.parse(edit(str(module)), ctx)
    for op in list(module.body.operations):
        op.erase()
    for op in list(new.body.operations):
        module.body.append(op)
    return ok


amcmod.amc.lower_amc_to_loopschedule_vivado = patched
kp = os.path.join(os.environ["TMPDIR"], f"kbr_{N}_{os.getpid()}.py")
open(kp, "w").write(open(f"{U1}/kernel_bits_amc.py").read().replace("N = 16", f"N = {N}"))
K = load(f"kbr_{N}", kp)
s = allo.customize(K.bf16_add_bits_amc)
loops = s.get_loops()
s.unroll(loops["S_i_0"]["offset"])
s.pipeline(loops["S_i_0"]["i"])
f = s.build(target="amc")
out = f"{os.path.dirname(os.path.abspath(__file__))}/out/retime_{mode}_N{N}"
f.dump_schedule(out + ".loopschedule.mlir")
f.dump_verilog(out + "_sv")
rng = np.random.default_rng(2)
a = rng.integers(0, 1 << 16, N, dtype=np.uint16)
b = rng.integers(0, 1 << 16, N, dtype=np.uint16)
c = np.zeros(N, np.uint16)
f(a, b, c)
gold = ref.vpu_bf16_add(a, b).astype(np.uint16)
print(f"RESULT mode={mode} N={N} cycles={f.rpt['cycles']} match={(c == gold).sum()}/{N}"
      f" report={re.findall(r'loopschedule.pipeline II = .*? iter_args', str(f.loopSchedule))}")
