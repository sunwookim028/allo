# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""(b) Did the CP-SAT (exact) path actually solve against prebuilt OR-Tools 9.15?
Prints compiler.solves telemetry for heuristic vs exact (+area_slack) on both kernels.
Usage (after sourcing env.sh, from this dir): python exp_cpsat.py"""
import sys; sys.dont_write_bytecode = True  # keep the read-only U1/harness dirs free of __pycache__
import os, sys
U1 = "/work/shared/users/phd/sk3463/scratch/wt-lat/dev/records/minitpu/u1_bf16_add_rtlgen"
os.environ["BF_N"] = "16"
sys.path.insert(0, U1)
from bits_kernel import bf16_add_bits
from bits_scalar_kernel import bf16_add_scalar

for k in (bf16_add_bits, bf16_add_scalar):
    for opts in ({}, {"scheduler": "exact", "budget": 10.0}, {"scheduler": "exact", "budget": 10.0, "area_slack": 1.0}):
        rtl = k.schedule().export("rtl", freq_mhz=300)
        if opts:
            rtl.set_scheduler_opt(**opts)
        sr = rtl.schedule()
        print(f"CPSAT {rtl.top} {opts}: latency={sr.func(rtl.top).latency} area={sr.area}")
        for s in sr.compiler.solves:
            print(f"   solve kind={s.kind} solver={s.solver!r} ii={s.interval} ms={s.ms:.0f} proven={s.proven} "
                  f"span_proven={s.span_proven} exhausted={s.budget_exhausted} fallback={s.fallback} "
                  f"model_area={s.model_area} bound={s.model_area_bound}")

# A kernel where CP-SAT really solves (Kai's tests/rtl/test_objectives.py::
# test_area_slack_pays_span_for_unit_folds): area_slack is the only knob that
# lets latency rise, and only as a fraction of the proven minimum.
from allo import kernel
from allo.lang import f32
import numpy as np


@kernel
def pairsum(A: f32[8], B: f32[8], C: f32[8], D: f32[8], out: f32[8]):
    for i in range(8):
        out[i] = (A[i] + B[i]) + (C[i] + D[i])


for opts in ({}, {"scheduler": "exact", "budget": 2.0}, {"scheduler": "exact", "budget": 2.0, "area_slack": 0.25},
             {"scheduler": "exact", "budget": 2.0, "area_slack": 1.0}):
    rtl = pairsum.schedule().export("rtl")
    if opts:
        rtl.set_scheduler_opt(**opts)
    sr = rtl.schedule()
    fn = sr.func("pairsum")
    A = np.arange(8, dtype=np.float32); out = np.zeros(8, np.float32)
    r = rtl.cosim(A, A, A, A, out)
    print(f"CPSAT pairsum {opts}: latency={fn.latency} II={[x.interval for x in fn.regions]} "
          f"measured={r.cycles} ok={np.array_equal(out, 4 * A)}")
    for s in sr.compiler.solves:
        print(f"   solve kind={s.kind} solver={s.solver!r} ii={s.interval} ms={s.ms:.0f} proven={s.proven} "
              f"span_proven={s.span_proven} fallback={s.fallback} model_area={s.model_area}")
