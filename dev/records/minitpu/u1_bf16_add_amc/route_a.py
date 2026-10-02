# Route (a): the bits kernel through AMC's own vendored Allo frontend.
# Run in the AMC env (source env.sh) under `scl enable gcc-toolset-13`.
#   python route_a.py [N] [limit]   N: kernel size; limit: stimulus pairs (default all)
# Builds kernel_bits_amc.bf16_add_bits_amc with N elements, checks it first on
# AMC's own LLVM target (frontend semantics), then on target="amc" (Verilator
# RTL simulation), both against harness/ref.vpu_bf16_add.
import sys, os, time, importlib.util, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ref = load("ref", f"{ROOT}/examples/minitpu/harness/ref.py")
stimulus = load("stimulus", f"{ROOT}/examples/minitpu/harness/stimulus.py")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
targets = sys.argv[3].split(",") if len(sys.argv) > 3 else ["llvm", "amc"]
# sched: "none" (as written) or "unroll" (fully unroll the 17-step leading-zero
# loop and pipeline the element loop, as the RTLGen row needed for II=1)
sched = sys.argv[4] if len(sys.argv) > 4 else "none"

import allo  # AMC's vendored fork

K = load("kernel_bits_amc", f"{HERE}/kernel_bits_amc.py")
K.N = N
src = open(f"{HERE}/kernel_bits_amc.py").read().replace("N = 16", f"N = {N}")
kpath = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"kernel_bits_amc_{N}.py")
open(kpath, "w").write(src)
K = load(f"kernel_bits_amc_{N}", kpath)

stim = stimulus.binary_bf16()
if limit:
    stim = stim[:limit]
gold = ref.vpu_bf16_add(stim[:, 0], stim[:, 1]).astype(np.uint16)
m = len(stim)
pad = (-m) % N
a = np.concatenate([stim[:, 0], np.zeros(pad, np.uint16)])
b = np.concatenate([stim[:, 1], np.zeros(pad, np.uint16)])

for tgt in targets:
    print(f"\n######## target={tgt} N={N} pairs={m}", flush=True)
    t0 = time.time()
    try:
        s = allo.customize(K.bf16_add_bits_amc)
        if sched == "unroll":
            loops = s.get_loops()
            print(loops, flush=True)
            s.unroll(loops["S_i_0"]["offset"])
            s.pipeline(loops["S_i_0"]["i"])
        f = s.build(target=tgt)
    except Exception:
        traceback.print_exc()
        continue
    print(f"build {time.time() - t0:.1f}s", flush=True)
    if tgt == "amc":
        out = f"{os.environ.get('TMPDIR', '/tmp')}/u1_amc_sv_N{N}_{sched}"
        f.dump_schedule(out + ".loopschedule.mlir")
        print("dump_verilog ->", [p.name for p in f.dump_verilog(out)], out)
    got = np.zeros(m + pad, np.uint16)
    cycles = []
    t0 = time.time()
    for k in range(0, m + pad, N):
        c = np.zeros(N, np.uint16)
        f(np.ascontiguousarray(a[k:k + N]), np.ascontiguousarray(b[k:k + N]), c)
        got[k:k + N] = c
        if tgt == "amc":
            cycles.append(f.rpt["cycles"])
    got = got[:m]
    bad = np.nonzero(got != gold)[0]
    print(f"run {time.time() - t0:.1f}s; match {m - len(bad)}/{m}", flush=True)
    if cycles:
        print(f"cycles per call (N={N}): min {min(cycles)} max {max(cycles)}", flush=True)
    for j in bad[:10]:
        print(f"  a={stim[j,0]:#06x} b={stim[j,1]:#06x} got={got[j]:#06x} want={gold[j]:#06x}")
