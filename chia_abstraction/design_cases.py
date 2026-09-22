# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The design cases a candidate is measured on. FROZEN.

"Reusable across design cases" is the whole claim of an abstraction-level loop,
so it has to be measured on more than one design. Each case here is a complete,
self-contained design plus its own numpy golden, driven through Allo's Vivado
HLS path so that the C++ emitter (`mlir/lib/Translation/EmitVivadoHLS.cpp`) is
the thing under test:

    mode="csim"   the emitted C++ is compiled with g++ against Vitis's headers
                  and RUN, and its result compared to numpy. This is
                  bit-exactness THROUGH the emitter, and it is the only
                  correctness signal in the ladder that does not pass through
                  Allo's own LLVM simulator -- i.e. through code the candidate
                  edits.
    mode="csyn"   Vitis HLS 2023.2 csynth: latency, interval, area and the
                  estimated clock, from `csynth.xml`. ~30 s per case.

The cases, and why each one:

`systolic_1d` -- the SAME workload as TinyTPU-isa (int32 GEMM) on a completely
    different machine: a 1-D streaming systolic chain, fixed-function, no
    instruction word, no program memory, no scratchpad, streams carrying
    scalars between `mapping=[P]` kernel instances. This is the case that makes
    "reusable across design cases" mean something for a GEMM abstraction: if an
    abstraction helps the programmable machine and not the streaming one, that
    is a finding.

`blocks_stream` -- a producer/consumer region whose stream carries a whole 2-D
    block (`Stream[int32[M,N], d]`), i.e. the emitter's `hls::vector` path.
    Neither TinyTPU-isa (packed `UInt(T*8)` words) nor `systolic_1d` (scalar
    streams) touches it. Cheap. It is the case that catches an emitter change
    that only considered scalar streams.

`mlp_layered` -- a three-layer float32 MLP with `s.pipeline`, `s.unroll` and
    `s.partition` applied. Structurally the opposite of the other two: floating
    point, layered fixed function, a staged buffer rather than a systolic
    chain. If an abstraction pays off here too it is not a GEMM trick.

`tinytpu_isa` is a design case as well, but it is not defined here: it is run
through its own frozen gates (`bench_isa.py`, `stress_isa.py`, `cosim.py`) by
`evaluate_abs.py`, because those already exist, are byte-pinned to main, and
give RTL cosim CYCLES rather than an HLS estimate.

Every design is at MODULE level with module-level constants. That is not style:
`@df.region`'s kernel discovery is a syntactic source scan and its type
resolution uses the defining module's globals, so a type alias local to a
builder function resolves to nothing ("Unsupported type `Ty`"). The same hazard
is why `microarch_isa.py` keeps its `Stream`/`UInt` aliases at module scope.

Every case is run inside the vouched process (`abs_gate_runner.py design_case
<name>`), which has already frozen `numpy` and `builtins`, so the golden cannot
be computed with a monkeypatched numpy. The verdict is this module's RETURN
VALUE, never a printed line.

    printf '%s\\n' NONCE | python abs_gate_runner.py design_case NAME --work DIR
                                                     [--csyn] [--no-csim]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import allo
import allo.dataflow as df
from allo.customize import MockBuffer
from allo.ir.types import float32, int32, Stream

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: Same target clock as the TinyTPU-isa design case, so the two are comparable.
TARGET_NS = 3.33
#: Vitis HLS 2023.2. `hls.is_available("vitis_hls")` shells out to `which`, and
#: this host does not put Vitis on the login PATH (see CLAUDE.md).
VITIS_BIN = os.environ.get("CHIA_VITIS_BIN", "/opt/xilinx/Vitis_HLS/2023.2/bin")

#: Deterministic: every case's operands and weights come from this seed, so a
#: cycle or area delta between two candidates is a property of the compiler.
SEED = 20260922

# -- case 1: systolic_1d ------------------------------------------------------
S1_M, S1_K, S1_N = 4, 4, 4
S1_P = S1_K + 2


@df.region()
def systolic_1d(A: int32[S1_M, S1_K], B: int32[S1_K, S1_N],
                C: int32[S1_M, S1_N]):
    fifo_A: Stream[int32, 4][2, S1_P]
    fifo_B: Stream[int32, 4][2, S1_P]

    @df.kernel(mapping=[2, S1_P], args=[A, B, C])
    def gemm(lA: int32[S1_M, S1_K], lB: int32[S1_K, S1_N],
             lC: int32[S1_M, S1_N]):
        i, j = df.get_pid()
        with allo.meta_if(i == 0 and (j == 0 or j == S1_P - 1)):
            pass
        with allo.meta_elif(i == 0):
            for _ in range(S1_N):
                for k in range(S1_K):
                    fifo_B[i + 1, j].put(lB[k, j - 1])
        with allo.meta_elif(j == 0):
            for m in range(S1_M):
                for k in range(S1_K):
                    fifo_A[i, j + 1].put(lA[m, k])
        with allo.meta_elif(j == S1_P - 1):
            for m in range(S1_M):
                for _ in range(S1_K):
                    a: int32 = fifo_A[i, j].get()
        with allo.meta_else():
            for m in range(S1_M):
                c: int32 = 0
                for _ in range(S1_K):
                    a: int32 = fifo_A[i, j].get()
                    b: int32 = fifo_B[i, j].get()
                    c += a * b
                    fifo_A[i, j + 1].put(a)
                lC[m, j - 1] = c


def _systolic_1d_inputs():
    rng = np.random.default_rng(SEED)
    A = rng.integers(-64, 64, (S1_M, S1_K)).astype(np.int32)
    B = rng.integers(-64, 64, (S1_K, S1_N)).astype(np.int32)
    C = np.zeros((S1_M, S1_N), dtype=np.int32)
    return (A, B, C), (A.astype(np.int64) @ B.astype(np.int64)).astype(np.int32), 2


def _systolic_1d_schedule(s):
    # The compute PE's accumulation loop: the same shape of claim
    # `s.dependence` is about on TinyTPU-isa's accumulator.
    s.pipeline("gemm_1_1:m")
    return s


# -- case 2: blocks_stream ----------------------------------------------------
B2_M, B2_N, B2_NB = 4, 4, 3


@df.region()
def blocks_stream(A: int32[B2_M * B2_NB, B2_N], B: int32[B2_M * B2_NB, B2_N]):
    pipe: Stream[int32[B2_M, B2_N], 2]

    @df.kernel(mapping=[1], args=[A])
    def producer(lA: int32[B2_M * B2_NB, B2_N]):
        for k in range(B2_NB):
            blk: int32[B2_M, B2_N] = 0
            for m in range(B2_M):
                for n in range(B2_N):
                    blk[m, n] = lA[k * B2_M + m, n]
            pipe.put(blk)

    @df.kernel(mapping=[1], args=[B])
    def consumer(lB: int32[B2_M * B2_NB, B2_N]):
        for k in range(B2_NB):
            blk: int32[B2_M, B2_N] = pipe.get()
            for m in range(B2_M):
                for n in range(B2_N):
                    lB[k * B2_M + m, n] = blk[m, n] * 3 + 1


def _blocks_stream_inputs():
    rng = np.random.default_rng(SEED)
    A = rng.integers(-1000, 1000, (B2_M * B2_NB, B2_N)).astype(np.int32)
    B = np.zeros((B2_M * B2_NB, B2_N), dtype=np.int32)
    return (A, B), (A * 3 + 1).astype(np.int32), 1


def _blocks_stream_schedule(s):
    s.pipeline("consumer_0:m")
    return s


# -- case 3: mlp_layered ------------------------------------------------------
M3_BS, M3_D0, M3_D1, M3_D2, M3_D3 = 2, 32, 16, 8, 4
_rng3 = np.random.default_rng(SEED + 1)
M3_W0 = _rng3.standard_normal((M3_D1, M3_D0)).astype(np.float32)
M3_W1 = _rng3.standard_normal((M3_D2, M3_D1)).astype(np.float32)
M3_W2 = _rng3.standard_normal((M3_D3, M3_D2)).astype(np.float32)


@df.region()
def mlp_layered(X: float32[M3_BS, M3_D0], Y: float32[M3_BS, M3_D3]):
    h0: Stream[float32, 8]
    h1: Stream[float32, 8]

    @df.kernel(mapping=[1], args=[X])
    def layer0(lX: float32[M3_BS, M3_D0]):
        w0: float32[M3_D1, M3_D0] = M3_W0
        for b in range(M3_BS):
            for o in range(M3_D1):
                acc: float32 = 0.0
                for i in range(M3_D0):
                    acc += lX[b, i] * w0[o, i]
                relu: float32 = 0.0
                if acc > 0.0:
                    relu = acc
                h0.put(relu)

    @df.kernel(mapping=[1])
    def layer1():
        w1: float32[M3_D2, M3_D1] = M3_W1
        buf: float32[M3_BS, M3_D1]
        for b in range(M3_BS):
            for o in range(M3_D1):
                buf[b, o] = h0.get()
        for b2 in range(M3_BS):
            for o2 in range(M3_D2):
                acc2: float32 = 0.0
                for i2 in range(M3_D1):
                    acc2 += buf[b2, i2] * w1[o2, i2]
                relu2: float32 = 0.0
                if acc2 > 0.0:
                    relu2 = acc2
                h1.put(relu2)

    @df.kernel(mapping=[1], args=[Y])
    def layer2(lY: float32[M3_BS, M3_D3]):
        w2: float32[M3_D3, M3_D2] = M3_W2
        buf2: float32[M3_BS, M3_D2]
        for b in range(M3_BS):
            for o in range(M3_D2):
                buf2[b, o] = h1.get()
        for b3 in range(M3_BS):
            for o3 in range(M3_D3):
                acc3: float32 = 0.0
                for i3 in range(M3_D2):
                    acc3 += buf2[b3, i3] * w2[o3, i3]
                lY[b3, o3] = acc3


def _mlp_layered_inputs():
    rng = np.random.default_rng(SEED + 2)
    X = rng.standard_normal((M3_BS, M3_D0)).astype(np.float32)
    Y = np.zeros((M3_BS, M3_D3), dtype=np.float32)
    g0 = np.maximum(X @ M3_W0.T, 0.0)
    g1 = np.maximum(g0 @ M3_W1.T, 0.0)
    return (X, Y), (g1 @ M3_W2.T).astype(np.float32), 1


def _mlp_layered_schedule(s):
    s.pipeline("layer0_0:i")
    s.unroll("layer1_0:i2", factor=4)
    s.partition(MockBuffer("layer1_0", "buf"), partition_type=2, factor=4)
    return s


CASES = {
    "systolic_1d": (systolic_1d, _systolic_1d_inputs, _systolic_1d_schedule, 0),
    "blocks_stream": (blocks_stream, _blocks_stream_inputs,
                      _blocks_stream_schedule, 0),
    "mlp_layered": (mlp_layered, _mlp_layered_inputs, _mlp_layered_schedule,
                    1e-3),
}
#: Which cases csynth can actually build, measured 2026-09-22 on this host.
#: `systolic_1d` cannot: its top-level `lA`/`lB`/`lC` are read by several
#: kernel instances of one `mapping=[2, P]` kernel, and Vitis refuses that in a
#: dataflow region --
#:   ERROR [HLS 200-779] Non-shared array 'buf1' failed dataflow checking:
#:   it can only have a single reader and a single writer.
#: csim passes (Vitis's C model runs the processes sequentially and does not
#: apply the rule), so the case is a CORRECTNESS case, not a PPA one. This is
#: the same one-owner-per-memory wall as limitations register item "shared
#: memory" / issue #27, reached from the other side: there Allo refuses it, here
#: Vitis does. Recorded rather than worked around, because a candidate that
#: made this design synthesisable would be a real result -- and the harness
#: would then measure it, since `csyn` is asked for per case.
CSYN_OK = {"systolic_1d": False, "blocks_stream": True, "mlp_layered": True}
#: Run on every candidate inside the loop: csim only, ~5 s each.
LOOP_CASES = ("systolic_1d", "blocks_stream")
#: Run at acceptance.
ALL_CASES = tuple(CASES)
#: Cases that contribute a PPA row (csynth latency/interval/area).
PPA_CASES = tuple(c for c in CASES if CSYN_OK[c])


# -- running one case ---------------------------------------------------------
def parse_csynth(prj: Path) -> dict:
    xml = prj / "out.prj/solution1/syn/report/csynth.xml"
    if not xml.exists():
        return {"error": "no csynth.xml -- synthesis failed"}
    root = ET.parse(xml).getroot()
    lat = root.find(".//PerformanceEstimates/SummaryOfOverallLatency")
    res = root.find(".//AreaEstimates/Resources")

    def num(node, tag, cast=int):
        if node is None:
            return None
        try:
            return cast(node.findtext(tag))
        except (TypeError, ValueError):
            return None

    return {
        "target_ns": float(root.findtext(".//TargetClockPeriod")),
        "estimated_ns": float(root.findtext(".//EstimatedClockPeriod")),
        "latency_worst": num(lat, "Worst-caseLatency"),
        "latency_avg": num(lat, "Average-caseLatency"),
        "interval_max": num(lat, "Interval-max"),
        "area": {k.lower(): num(res, k) for k in
                 ("BRAM_18K", "DSP", "FF", "LUT", "URAM")},
    }


def run(argv) -> tuple[int, dict]:
    """(rc, report) for one design case. rc 0 iff the case emitted, was
    bit-exact under csim, and (when asked) met the clock under csynth."""
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(CASES))
    ap.add_argument("--work", required=True)
    ap.add_argument("--csyn", action="store_true",
                    help="run csynth. It is ATTEMPTED even for a case CSYN_OK "
                         "says Vitis refuses today: a candidate that made it "
                         "synthesise has made a second architecture "
                         "expressible, and the harness has to be able to see "
                         "that. The caller tolerates the failure.")
    ap.add_argument("--no-csim", action="store_true")
    a = ap.parse_args(argv)

    if VITIS_BIN not in os.environ.get("PATH", ""):
        os.environ["PATH"] = VITIS_BIN + os.pathsep + os.environ.get("PATH", "")
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    region, inputs, schedule, atol = CASES[a.name]
    args, golden, out_index = inputs()
    rep = {"case": a.name, "csim": None, "csyn": None, "seconds": {}}

    # 1. Emit. A candidate that breaks emission fails here, and the emitted
    #    text is kept: it is the artifact that shows what the change did.
    t = time.time()
    s = df.customize(region)
    schedule(s)
    code = s.build(target="vhls").hls_code
    (work / f"{a.name}.emitted.cpp").write_text(code)
    rep["seconds"]["emit"] = round(time.time() - t, 1)
    rep["emitted_bytes"] = len(code)

    # 2. csim: the emitted C++ compiled and run, against numpy.
    if not a.no_csim:
        t = time.time()
        s2 = df.customize(region)
        schedule(s2)
        mod = s2.build(target="vitis_hls", mode="csim",
                       project=str(work / f"{a.name}.csim.prj"))
        call_args = [np.array(x, copy=True) for x in args]
        mod(*call_args)
        got = call_args[out_index]
        err = float(np.max(np.abs(got.astype(np.float64)
                                  - golden.astype(np.float64))))
        exact = (np.array_equal(got, golden) if atol == 0
                 else bool(np.allclose(got, golden, atol=atol, rtol=1e-3)))
        rep["csim"] = {"exact": exact, "max_abs_err": err,
                       "reference": "numpy, in this process, numpy frozen by "
                                    "the gate runner"}
        rep["seconds"]["csim"] = round(time.time() - t, 1)
        if not exact:
            print(f"  CASE {a.name}: csim NOT exact (max abs err {err})",
                  flush=True)
            return 1, rep

    # 3. csynth: the PPA feedback for this case.
    if a.csyn:
        t = time.time()
        prj = work / f"{a.name}.csyn.prj"
        s3 = df.customize(region)
        schedule(s3)
        s3.build(target="vitis_hls", mode="csyn", project=str(prj))()
        rep["csyn"] = parse_csynth(prj)
        rep["seconds"]["csyn"] = round(time.time() - t, 1)
        if "error" in rep["csyn"]:
            print(f"  CASE {a.name}: {rep['csyn']['error']}", flush=True)
            return 1, rep
        if abs(rep["csyn"]["target_ns"] - TARGET_NS) > 1e-6:
            print(f"  CASE {a.name}: target clock {rep['csyn']['target_ns']} "
                  f"!= {TARGET_NS}", flush=True)
            return 1, rep
        if rep["csyn"]["estimated_ns"] > TARGET_NS:
            print(f"  CASE {a.name}: estimated clock "
                  f"{rep['csyn']['estimated_ns']} ns misses {TARGET_NS} ns; a "
                  f"latency at a clock the design cannot meet is not "
                  f"comparable", flush=True)
            return 1, rep

    print(f"  CASE {a.name} OK: csim="
          f"{rep['csim']['exact'] if rep['csim'] else 'skipped'} "
          f"latency={(rep['csyn'] or {}).get('latency_worst')} "
          f"interval={(rep['csyn'] or {}).get('interval_max')}", flush=True)
    return 0, rep


def main() -> int:
    rc, rep = run(None)
    print(json.dumps(rep, indent=1, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
