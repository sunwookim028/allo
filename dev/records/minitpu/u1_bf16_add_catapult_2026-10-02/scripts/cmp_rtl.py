"""Catapult RTL of the Allo bf16_add unit vs MiniTPU's vpu_bf16_add.sv, in Verilator.

    python cmp_rtl.py <v1_dir> [--shape stream|bare] [--latency L] [--warmup W]
                      [--ready-period P] [--n N]
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
from examples.minitpu.harness import rtl, stimulus, check, ref
from examples.minitpu.units import bf16_add

ap = argparse.ArgumentParser()
ap.add_argument("v1_dir")
ap.add_argument("--shape", default="stream")
ap.add_argument("--latency", type=int, default=0)
ap.add_argument("--warmup", type=int, default=0)
ap.add_argument("--ready-period", type=int, default=0)
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--top", default="add_0")
a = ap.parse_args()

stim = stimulus.binary_bf16()
if a.n:
    stim = stim[: a.n]
n = len(stim)
t = time.time()
want, _ = rtl.run(bf16_add.RTL, stim.astype(np.uint64))
want = want[:, 0].astype(np.uint16)
print(f"MiniTPU RTL vpu_bf16_add: {n} vectors, {time.time()-t:.1f}s", flush=True)

kw = {}
if a.shape == "stream":
    kw["out_ready_period"] = a.ready_period
else:
    kw.update(reset=True, latency=a.latency, warmup=a.warmup)
cu = bf16_add.catapult_rtl(a.v1_dir, top=a.top, shape=a.shape, **kw)
t = time.time()
got, cyc = rtl.run(cu, stim.astype(np.uint64))
got = got[:, 0].astype(np.uint16)
print(f"Catapult RTL {a.v1_dir}: shape={a.shape} {time.time()-t:.1f}s stats={rtl.last_stats}", flush=True)
if a.shape == "stream":
    lat = np.unique(cyc, return_counts=True)
    print("  latency (output cycle - input accept cycle):", dict(zip(lat[0].tolist(), lat[1].tolist())))
    st = rtl.last_stats
    if st:
        print(f"  {n} outputs in {st['cycles']} cycles: {st['cycles']/n:.4f} cycles/vector, first output at cycle {st['first_out']}")
k = int((got != want).sum())
print(("MATCH" if k == 0 else "DIFF"), f"catapult-rtl vs minitpu-rtl: {n-k}/{n} equal")
ieee = ref.ieee_bf16_add(stim[:, 0], stim[:, 1])
print(f"  catapult-rtl vs IEEE RNE (ref.ieee_bf16_add): {int((got == ieee).sum())}/{n} equal")
if k:
    for name, idx in check.classify(stim, got, want, bf16_add.EXPLAIN).items():
        ex = ", ".join(f"{stim[i,0]:04x}+{stim[i,1]:04x}: catapult {got[i]:04x} minitpu {want[i]:04x}" for i in idx[:3])
        print(f"    {len(idx):7d}  {name}  e.g. {ex}")
