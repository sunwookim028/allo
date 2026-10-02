# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# (c) Is a clockless / scalar-port AMC kernel possible? Builds the bf16 bits
# adder as a scalar function (kernel_bits_scalar.py: uint16 a, b -> uint16),
# with the leading-zero loop unrolled, optionally function-pipelined
# (s.pipeline() -> hls.pipeline on the func), and prints the top module's
# port list, the schedule header, and Verilator cycles/match on random pairs.
# Usage: AMC_TARGET_CLOCK_PERIOD_NS=<ns> python scalar_kernel.py [seq|fpipe] [npairs]
import sys, os, re, importlib.util, traceback
import numpy as np
import allo

ROOT = "/work/shared/users/phd/sk3463/scratch/wt-lat"
HERE = os.path.dirname(os.path.abspath(__file__))
mode = sys.argv[1] if len(sys.argv) > 1 else "fpipe"
npairs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
period = os.environ.get("AMC_TARGET_CLOCK_PERIOD_NS", "10.0")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ref = load("ref", f"{ROOT}/examples/minitpu/harness/ref.py")
K = load("kbs", f"{HERE}/kernel_bits_scalar.py")
s = allo.customize(K.bf16_add_bits_scalar)
loops = s.get_loops()
print("loops:", loops)
# leading-zero loop is unrolled in the source (kernel_bits_scalar.py)
if mode == "fpipe":
    s.pipeline()
f = s.build(target="amc")
out = f"{HERE}/out/scalar_{mode}_p{period}"
f.dump_schedule(out + ".loopschedule.mlir")
f.dump_verilog(out + "_sv")
sv = open(f"{out}_sv/bf16_add_bits_scalar.sv").read()
hdr = sv[sv.index("module bf16_add_bits_scalar("):]
print("PORTS", " ".join(hdr[:hdr.index(");")].split()))
ls = str(f.loopSchedule)
print("SCHED", re.findall(r"loopschedule\.func_\w+ @\w+\([^)]*\)[^{]*", ls)[:1],
      re.findall(r"loopschedule\.func_pipeline[^\n]*", ls)[:1])
rng = np.random.default_rng(3)
ok, cyc = 0, set()
for _ in range(npairs):
    a, b = rng.integers(0, 1 << 16, 2, dtype=np.uint16)
    r = f(np.uint16(a), np.uint16(b))
    want = int(ref.vpu_bf16_add(np.array([a]), np.array([b]))[0])
    ok += int(r) == want
    cyc.add(f.rpt["cycles"])
print(f"RESULT mode={mode} period={period} match={ok}/{npairs} cycles_per_call={sorted(cyc)}")
