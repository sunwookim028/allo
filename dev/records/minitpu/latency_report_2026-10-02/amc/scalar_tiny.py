# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# (c) control: smallest scalar-port kernel, `a + b` and `a & b` on uint16,
# sequential and function-pipelined. Prints top-module ports, schedule, cycles.
# Usage: python scalar_tiny.py
import os, re
import numpy as np
import allo
from allo.ir.types import uint16


def add16(a: uint16, b: uint16) -> uint16:
    c: uint16 = a + b
    return c


HERE = os.path.dirname(os.path.abspath(__file__))
for mode in ["seq", "fpipe"]:
    s = allo.customize(add16)
    if mode == "fpipe":
        s.pipeline()
    try:
        f = s.build(target="amc")
    except Exception as e:
        print(f"RESULT mode={mode} FAILED {type(e).__name__}: {e}")
        continue
    out = f"{HERE}/out/tiny_{mode}"
    f.dump_schedule(out + ".loopschedule.mlir")
    f.dump_verilog(out + "_sv")
    sv = open(f"{out}_sv/add16.sv").read()
    hdr = sv[sv.index("module add16("):]
    try:
        r = int(f(np.uint16(1234), np.uint16(4321)))
    except Exception as e:  # AMC's Python harness expects a memory for the result
        r = f"sim harness error: {type(e).__name__}: {e}"
    regd = re.findall(r"assign result0 = (\w+);", sv)
    print(f"RESULT mode={mode} ports=[{' '.join(hdr[:hdr.index(');')].split())}]"
          f" out={r} (want 5555) result0_driven_by={regd} cycles={f.rpt['cycles']}"
          f" sched={re.findall(r'loopschedule.func_\w+[^{]*', str(f.loopSchedule))[:1]}")
