# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Reported-vs-measured latency for the bf16 `bits` kernel (U1 kernel_bits_amc.py).
# Usage (after `source env.sh`, under `scl enable gcc-toolset-13 --`):
#   AMC_TARGET_CLOCK_PERIOD_NS=<ns> python lat_sweep.py <sched> <outdir> [N ...]
# sched: none | unroll | unroll_iiK (pipeline with initiation_interval=K)
# For each N: builds target="amc", dumps loopschedule IR + SV, runs Verilator on
# random bf16 bit patterns, checks against the MiniTPU reference, prints cycles.
import sys, os, re, importlib.util, json, traceback
import numpy as np

U1 = "/work/shared/users/phd/sk3463/scratch/wt-lat/dev/records/minitpu/u1_bf16_add_amc"
ROOT = "/work/shared/users/phd/sk3463/scratch/wt-lat"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ref = load("ref", f"{ROOT}/examples/minitpu/harness/ref.py")
sched, outdir = sys.argv[1], sys.argv[2]
Ns = [int(x) for x in sys.argv[3:]] or [16, 64]
period = os.environ.get("AMC_TARGET_CLOCK_PERIOD_NS", "10.0")
os.makedirs(outdir, exist_ok=True)
import allo  # AMC's vendored fork

src = open(f"{U1}/kernel_bits_amc.py").read()
rng = np.random.default_rng(0)
res = []
for N in Ns:
    kp = os.path.join(os.environ["TMPDIR"], f"kba_{N}_{os.getpid()}.py")
    open(kp, "w").write(src.replace("N = 16", f"N = {N}"))
    K = load(f"kba_{N}", kp)
    tag = f"{sched}_p{period}_N{N}"
    rec = {"sched": sched, "period_ns": float(period), "N": N}
    try:
        s = allo.customize(K.bf16_add_bits_amc)
        if sched.startswith("unroll"):
            loops = s.get_loops()
            s.unroll(loops["S_i_0"]["offset"])
            ii = int(sched.split("_ii")[1]) if "_ii" in sched else 1
            s.pipeline(loops["S_i_0"]["i"], initiation_interval=ii)
        open(f"{outdir}/{tag}.allo.mlir", "w").write(str(s.module))
        f = s.build(target="amc")
    except Exception as e:
        traceback.print_exc()
        rec["error"] = f"{type(e).__name__}: {e}"
        res.append(rec)
        print("RESULT", json.dumps(rec), flush=True)
        continue
    f.dump_schedule(f"{outdir}/{tag}.loopschedule.mlir")
    f.dump_verilog(f"{outdir}/{tag}_sv")
    a = rng.integers(0, 1 << 16, N, dtype=np.uint16)
    b = rng.integers(0, 1 << 16, N, dtype=np.uint16)
    c = np.zeros(N, np.uint16)
    f(a, b, c)
    gold = ref.vpu_bf16_add(a, b).astype(np.uint16)
    rec["match"] = int((c == gold).sum())
    rec["cycles"] = f.rpt["cycles"]
    ls = open(f"{outdir}/{tag}.loopschedule.mlir").read()
    # loopschedule.pipeline II = a trip_count = b latency = c
    rec["report_pipeline"] = re.findall(
        r"loopschedule\.pipeline II = (\d+)(?: trip_count = (\d+))?(?: latency = (\d+))?", ls)
    # stage offsets (loopschedule.at K) anywhere in the schedule
    rec["report_at_offsets"] = sorted({int(x) for x in re.findall(r"loopschedule\.at (\d+)", ls)})
    rec["report_ops"] = sorted(set(re.findall(r"loopschedule\.(\w+)", ls)) - {"operator"})
    res.append(rec)
    print("RESULT", json.dumps(rec), flush=True)
json.dump(res, open(f"{outdir}/results_{sched}_p{period}.json", "w"), indent=1)
