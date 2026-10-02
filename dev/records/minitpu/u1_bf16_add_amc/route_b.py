# Route (b): our frontend's MLIR text (emit_ours.py, N=16) handed to AMC's
# AMCModule, as the int32 experiments in amc_exploration_2026-10-02.rst did.
# Run in the AMC env (source env.sh) under `scl enable gcc-toolset-13`.
#   python route_b.py <mlir_dir> [limit] [name[+edit...]:top ...]
# N is read from the MLIR; the first limit - limit % N pairs are run.
import sys, os, re, time, importlib.util, traceback
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
from allo.backend.amc import AMCModule

d = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 256
cases = sys.argv[3:] or [
    "region:add_0", "region:top", "plain:bf16_add_bits",
    "region_amcedits:add_0", "plain_amcedits:bf16_add_bits_amc",
]
stim = stimulus.binary_bf16()
if limit:
    stim = stim[:limit]
m = len(stim)
gold = ref.vpu_bf16_add(stim[:, 0], stim[:, 1]).astype(np.uint16)

for case in cases:
    name, top = case.split(":")
    print(f"\n######## {name}.mlir top={top}", flush=True)
    base = name.split("+")[0]
    txt = open(f"{d}/{base}.mlir").read()
    N = int(re.search(r"memref<(\d+)xi16>", txt).group(1))
    n = m - m % N
    if "+rank1" in name:
        # Our fork keeps scalars as rank-0 memref<iN>; AMC's loopschedule
        # printer emits `[ : ]` for them, which its own parser rejects.
        # AMC's own frontend uses memref<1xiN>; rewrite to that.
        txt = re.sub(r"memref<(i\d+)>", r"memref<1x\1>", txt)
        txt = txt.replace("[]", "[0]")
    if "+droptop" in name:
        # The region wrapper func is named @top, and AMC's testbench module
        # is also `top`: "redefinition of symbol named 'top'". Drop it.
        a = txt.index("  func.func @top(")
        txt = txt[:a] + txt[txt.index("\n  }\n", a) + 5:]
    if "+nopipe" in name:
        txt = txt.replace(", pipeline_ii = 1 : ui32", "")
    if "+attrs" in name:
        # Our fork spells s.unroll as `unroll = 0 : i32`; AMC's spells it
        # `loopschedule.parallel = 0 : i32` (and pipeline_ii as i32, not ui32).
        txt = txt.replace("unroll = 0 : i32}", "loopschedule.parallel = 0 : i32, unroll = 0 : i32}")
        txt = txt.replace("pipeline_ii = 1 : ui32", "pipeline_ii = 1 : i32")
    try:
        f = AMCModule(txt, top_func_name=top, allocate_amc=True)
    except BaseException:
        traceback.print_exc(limit=2)
        continue
    print("AMCModule built", flush=True)
    got = np.zeros(n, np.uint16)
    try:
        for k in range(0, n, N):
            c = np.zeros(N, np.uint16)
            f(np.ascontiguousarray(stim[k:k + N, 0]), np.ascontiguousarray(stim[k:k + N, 1]), c)
            got[k:k + N] = c
    except BaseException:
        traceback.print_exc(limit=2)
        continue
    bad = np.nonzero(got != gold[:n])[0]
    print(f"match {n - len(bad)}/{n}; cycles (last call, N={N}) {f.rpt['cycles']}", flush=True)
    for j in bad[:5]:
        print(f"  a={stim[j,0]:#06x} b={stim[j,1]:#06x} got={got[j]:#06x} want={gold[j]:#06x}")
