# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""(a) What RTLGen reports for latency/II vs what cosim measures.
Usage (after `source .../u1_bf16_add_rtlgen/env.sh`, from this dir):
    python exp_report.py <array|scalar> <N> <mhz> [<mhz> ...]
Kernels are imported read-only from the U1 record's directory."""
import sys; sys.dont_write_bytecode = True  # keep the read-only U1/harness dirs free of __pycache__
import json, os, sys, warnings
U1 = "/work/shared/users/phd/sk3463/scratch/wt-lat/dev/records/minitpu/u1_bf16_add_rtlgen"
HARN = "/work/shared/users/phd/sk3463/scratch/wt-lat/examples/minitpu/harness"
kind, N, mhzs = sys.argv[1], int(sys.argv[2]), [float(x) for x in sys.argv[3:]]
os.environ["BF_N"] = str(N)
sys.path[:0] = [U1, HARN]
import numpy as np, stimulus, ref
from bits_kernel import bf16_add_bits
from bits_scalar_kernel import bf16_add_scalar

st = stimulus.binary_bf16()
A = st[1936:1936 + N, 0].copy(); B = st[1936:1936 + N, 1].copy()
gold = ref.vpu_bf16_add(A, B)

for mhz in mhzs:
    k = bf16_add_bits if kind == "array" else bf16_add_scalar
    rtl = k.schedule().export("rtl", freq_mhz=mhz)
    s = rtl.schedule()
    fn = s.func(rtl.top)
    print(f"=== {kind} N={N} asked {mhz:g} MHz; schedule cycle_ns={s.cycle_ns:.3f}")
    print(f"  FuncSchedule: latency={fn.latency} bound={fn.latency_is_bound} determinacy={fn.determinacy}")
    for r in fn.regions:
        print(f"  region#{r.order} {r.kind.value} depth={r.depth} container={r.container} "
              f"II={r.interval} trip={r.trip_count} latency={r.latency} "
              f"iteration_latency={r.iteration_latency} drain={r.cost.drain} "
              f"n_ops={len(r.ops)} last_t={r.last_t() if r.ops else None}")
    iface = rtl.interfaces.of_symbol(rtl.top)
    print(f"  interfaces[{rtl.top}]: latency={iface.latency} bound={iface.latency_is_bound} "
          f"determinacy={iface.determinacy} control={iface.control}")
    q = rtl.estimation
    print(f"  estimation: latency={q.latency} latency_max={q.latency_max} latency_min={q.latency_min} "
          f"interval={q.interval} fmax={q.fmax:.0f} clock_mhz={getattr(q, 'clock_mhz', None)} "
          f"lut={q.area.lut} ff={q.area.ff}")
    # manifest written by scaffold_project
    prj = rtl.scaffold_project(f"prj_{kind}_{N}_{int(mhz)}")
    man = json.loads((prj / "manifest.json").read_text())
    def find(d, path=""):
        if isinstance(d, dict):
            for kk, v in d.items():
                if kk in ("latency", "latency_bound", "determinacy", "interval", "ii"):
                    print(f"  manifest{path}.{kk} = {v}")
                find(v, f"{path}.{kk}")
        elif isinstance(d, list):
            for i, v in enumerate(d):
                find(v, f"{path}[{i}]")
    find(man)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if kind == "array":
            C = np.zeros(N, np.uint16)
            r = rtl.cosim(A, B, C)
            ok = int((C == gold).sum())
        else:
            cyc = set(); ok = 0
            for i in range(N):
                r = rtl.cosim(int(A[i]), int(B[i])); cyc.add(r.cycles)
                ok += int(r.result) == int(gold[i])
            r_cycles = cyc
        for ww in w:
            print(f"  WARNING {ww.category.__name__}: {ww.message}")
    meas = r.cycles if kind == "array" else r_cycles
    print(f"  MEASURED cosim cycles={meas} match={ok}/{N} cosim_freq={rtl.freq_mhz:g}")
