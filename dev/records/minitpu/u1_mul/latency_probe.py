# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""L2: where SystemC csim puts each output in time, for the comb and pipe forms.

    $ALLO_PYTHON dev/records/minitpu/u1_mul/latency_probe.py [<scratch dir>]

Emits each variant for n=8 through ``target="systemc", mode="csim"``, then
stamps every testbench Push/Pop with ``sc_time_stamp()`` (a 1 ns clock) and
rebuilds. Prints, per element, the time its inputs were accepted and the time
its result left. The probe edits only the emitted ``kernel.cpp``.
"""

import os
import re
import subprocess
import sys

import numpy as np

import allo.dataflow as df
from allo.backend import systemc
from examples.minitpu.units import bf16_mul, bf16_mul_pipe

N = 8
STIM = np.array([[0x3F80, 0x3F80], [0x4000, 0x3F80]] * 4, dtype=np.uint16)
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/u1_mul_latency_probe"


def probe(name, make, sched=None):
    prj = os.path.join(ROOT, name)
    if sched:
        s = df.customize(make(N))
        sched(s)
        mod = s.build(target="systemc", mode="csim", project=prj)
    else:
        mod = df.build(make(N), target="systemc", mode="csim", project=prj)
    out = bf16_mul.run_bits(mod, STIM)
    assert list(out[:2]) == [0x3F80, 0x4000], out
    path = os.path.join(prj, "kernel.cpp")
    k = open(path).read()
    k = re.sub(r"(ch_(v\d+)\.Push\(\(([^)]*)\)_v\);)",
               r'\1 std::cerr << "IN " << f << " " << sc_time_stamp() << "\\n";', k)
    k = re.sub(r"_f << \(long long\)\((ch_v\d+)\.Pop\(\)\) << \"\\n\";",
               r'{ auto _x = \1.Pop(); std::cerr << "OUT " << f << " " '
               r'<< sc_time_stamp() << "\\n"; _f << (long long)(_x) << "\\n"; }', k)
    open(path, "w").write(k)
    inc = os.path.join(os.environ["MGC_HOME"], "shared/include")
    subprocess.check_call(systemc.compile_command(prj, inc, "csim"), shell=True,
                          stdout=subprocess.DEVNULL)
    r = subprocess.run(f"cd {prj}; ./sim", shell=True, capture_output=True, text=True)
    ins, outs = {}, {}
    for line in r.stderr.splitlines():
        p = line.split()
        if p and p[0] == "IN":
            ins[int(p[1])] = max(ins.get(int(p[1]), ""), "".join(p[2:]))
        elif p and p[0] == "OUT":
            outs[int(p[1])] = "".join(p[2:])
    print(name + ": element: inputs accepted -> output popped")
    for f in range(N):
        print(f"   {f}: {ins.get(f)} -> {outs.get(f)}")
    sys.stdout.flush()


if __name__ == "__main__":
    probe("comb_bits", bf16_mul.bits)
    probe("comb_bits_pipelined", bf16_mul.bits, lambda s: s.pipeline("mul_0:i"))
    probe("pipe_bits_pipe", bf16_mul_pipe.bits_pipe)
    probe("pipe_stages_depth1", bf16_mul_pipe.stages)
    probe("pipe_stages_depth2", lambda n: bf16_mul_pipe.stages(n, depth=2))
