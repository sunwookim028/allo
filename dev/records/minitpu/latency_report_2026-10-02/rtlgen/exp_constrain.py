# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""(b) Which knobs move the reported/measured latency of the bf16 `bits` kernel.
Usage (after sourcing env.sh, from this dir):
    python exp_constrain.py            # runs every config, each in a subprocess
    python exp_constrain.py <config>   # one config
Each config prints REPORT (schedule) and MEASURED (cosim) lines."""
import sys; sys.dont_write_bytecode = True  # keep the read-only U1/harness dirs free of __pycache__
import os, subprocess, sys, warnings
U1 = "/work/shared/users/phd/sk3463/scratch/wt-lat/dev/records/minitpu/u1_bf16_add_rtlgen"
HARN = "/work/shared/users/phd/sk3463/scratch/wt-lat/examples/minitpu/harness"

# name -> (kernel, mhz, schedule-directive fn or None, scheduler opts)
CONFIGS = {
    "array_base300": ("array", 300, None, {}),
    "array_ii2_300": ("array", 300, lambda s: s.pipeline("k", ii=2), {}),
    "array_ii4_300": ("array", 300, lambda s: s.pipeline("k", ii=4), {}),
    "array_nopipe300": ("array", 300, lambda s: s.pipeline("k", ii=-1), {}),
    "array_exact300": ("array", 300, None, {"scheduler": "exact", "budget": 10.0}),
    "array_exact_slack300": ("array", 300, None, {"scheduler": "exact", "budget": 10.0, "area_slack": 1.0}),
    "array_freq300": ("array", 300, None, {"O": "freq", "span_tolerance": 0.2}),
    "array_wall300": ("array", 300, None, {"O": "wall"}),
    "array_margin300": ("array", 300, None, {"clock_margin": 0.5}),
    "scalar_base300": ("scalar", 300, None, {}),
    "scalar_exact300": ("scalar", 300, None, {"scheduler": "exact", "budget": 10.0}),
    "scalar_exact_slack300": ("scalar", 300, None, {"scheduler": "exact", "budget": 10.0, "area_slack": 1.0}),
    "scalar_bogus_opt": ("scalar", 300, None, {"latency": 2}),
}


def run(name):
    kind, mhz, directive, opts = CONFIGS[name]
    os.environ["BF_N"] = "16"
    sys.path[:0] = [U1, HARN]
    import numpy as np, stimulus, ref
    from bits_kernel import bf16_add_bits
    from bits_scalar_kernel import bf16_add_scalar
    k = bf16_add_bits if kind == "array" else bf16_add_scalar
    s = k.schedule()
    if directive:
        directive(s)
    rtl = s.export("rtl", freq_mhz=mhz)
    if opts:
        rtl.set_scheduler_opt(**opts)
    sr = rtl.schedule()
    fn = sr.func(rtl.top)
    regs = [(r.kind.value, r.interval, r.trip_count, r.latency, r.iteration_latency, r.cost.drain)
            for r in fn.regions]
    print(f"REPORT {name}: latency={fn.latency} exact={fn.latency_is_exact} cycle_ns={sr.cycle_ns:.3f} "
          f"regions(kind,II,trip,lat,iter_lat,drain)={regs} unhonored={sr.unhonored_directives}")
    q = rtl.estimation
    print(f"REPORT {name}: est latency={q.latency} interval={q.interval} fmax={q.fmax:.0f} "
          f"lut={q.area.lut} ff={q.area.ff} freq_mhz={rtl.freq_mhz:.1f}")
    st = stimulus.binary_bf16()
    A = st[1936:1952, 0].copy(); B = st[1936:1952, 1].copy(); gold = ref.vpu_bf16_add(A, B)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if kind == "array":
            C = np.zeros(16, np.uint16); r = rtl.cosim(A, B, C); cyc = r.cycles; ok = int((C == gold).sum())
        else:
            cyc = set(); ok = 0
            for i in range(4):
                r = rtl.cosim(int(A[i]), int(B[i])); cyc.add(r.cycles); ok += int(r.result) == int(gold[i])
        for ww in w:
            print(f"WARNING {name}: {ww.category.__name__}: {ww.message}")
    print(f"MEASURED {name}: cycles={cyc} match={ok}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        for name in CONFIGS:
            p = subprocess.run([sys.executable, __file__, name], capture_output=True, text=True, timeout=900)
            lines = (p.stdout + p.stderr).splitlines()
            keep = [l for l in lines if l.startswith(("REPORT", "MEASURED", "WARNING", "WARN:", "ERROR", "INFO: [SCHED]"))
                    or "Error" in l or "error" in l or "Traceback" in l or "Abort" in l]
            print(f"##### {name}: exit={p.returncode}")
            print("\n".join(keep[-25:]))
            if p.returncode != 0:
                print("  last lines:", "\n  ".join(lines[-12:]))
