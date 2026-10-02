"""The composed (two-kernel ``Stream``) FIFO vs MiniTPU's ``vpu_fifo`` RTL, on
the Allo simulator and SystemC csim.

    python cmp_composed.py --inst w32d4 [--backend simulator] [--backend systemc]
        [--variant composed] [--n N] [--project DIR]

From the worktree root after ``source examples/minitpu/harness/env-zhang21.sh``.
The trace is ``units/vpu_fifo.py`` ``traces_composed(inst)`` (legal, reset
only at the start) plus the ``tb_mxu_single_port`` seeds of the instance,
joined end to end, through the RTL (Verilator) and through the variant.

What is compared (the consumer-visible contract, Part A: no peek):

* ``pop_data`` on the pop cycles only -- the k-th pop of the trace is the
  consumer's k-th ``get``, so the alignment is exact by construction;
* ``empty`` (the consumer's view) and ``full`` (the producer's) on every
  defined cycle, at the shift ``s`` in [-8, 8] where ``got[t] == rtl[t + s]``
  agrees most (the two kernels are not cycle-locked to the trace: the shift
  is the handshake's offset; the simulator is order-only there).
"""
import argparse, os, sys, time

import numpy as np

sys.path.insert(0, os.getcwd())
import allo.dataflow as df  # noqa: E402
from examples.minitpu.harness import rtl  # noqa: E402
from examples.minitpu.harness.traces import concat  # noqa: E402
from examples.minitpu.units import vpu_fifo as u  # noqa: E402


def trace_composed_all(inst, n_max=0):
    parts = [(lab, c) for lab, c, _legal in u.traces_composed(inst)]
    for lab, sinst, c, _seen, _legal in u.seeds():
        if sinst == inst:
            parts.append((lab, u._drained(inst, c)))  # a seed may end with words inside
    spans, t = [], 0
    for lab, c in parts:
        k = len(next(iter(c.values())))
        spans.append((lab, t, t + k))
        t += k
    cmd = concat(*[c for _, c in parts])
    if n_max:
        cmd = {p: v[:n_max] for p, v in cmd.items()}
        spans = [(lab, a, min(b, n_max)) for lab, a, b in spans if a < n_max]
    return cmd, spans


def compare(got, want_int, reason, cmd, spans, label):
    """One verdict line; ``got``/``want_int`` map port -> list of ints."""
    n = len(cmd["pop_i"])
    pop = np.asarray(cmd["pop_i"]) == 1
    out, first = [], []
    for p in u.RESP:
        d = reason[p] == ""
        g = np.asarray([int(x) for x in got[p]], dtype=object)
        wv = np.asarray(want_int[p], dtype=object)
        if p == "pop_data_o":
            d = d & pop
            bad = d & (g != wv)
            out.append(f"pop_data {int(d.sum()) - int(bad.sum())}/{int(d.sum())} on pop cycles")
            for i in np.flatnonzero(bad)[:3]:
                lab = next(lb for lb, a, b in spans if a <= i < b)
                first.append(f"{lab} cycle {i} pop_data: allo {int(g[i]):x} rtl {int(wv[i]):x}")
            continue
        hits = {}
        for s in range(-8, 9):
            lo, hi = max(0, -s), min(n, n - s)
            dd = d[lo + s: hi + s] if s >= 0 else d[lo + s: hi + s]
            dd = d[lo + s: hi + s]
            hits[s] = int((dd & (g[lo:hi] == wv[lo + s: hi + s])).sum())
        best = max(hits, key=lambda s: (hits[s], -abs(s)))
        tot = int(d.sum())
        out.append(f"{p[:-2]} {hits[best]}/{tot} at s={best:+d} ({hits[0]} at s=0)")
    print(f"COMPOSED {label}: " + "; ".join(out), flush=True)
    for f in first:
        print(f"    e.g. {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", default="w32d4")
    ap.add_argument("--backend", action="append", choices=("simulator", "systemc"))
    ap.add_argument("--variant", default="composed")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--project", default="/work/shared/users/phd/sk3463/scratch/u3fc/prj")
    a = ap.parse_args()
    inst, w = a.inst, u.WIDTH[a.inst]
    cmd, spans = trace_composed_all(inst, a.n)
    n = len(cmd["rst_ni"])
    unit = u.INSTANCES[inst]
    packed = {p: rtl.pack(cmd[p], ww) for p, ww in unit.inputs}
    t0 = time.time()
    want = rtl.run_trace(unit, packed, seed=1)
    _, reason, _ = u.REF(inst, packed)
    want_int = {p: rtl.unpack(want[p]) for p in want}
    ndef = sum(int((reason[p] == "").sum()) for p in want)
    print(f"RTL vpu_fifo:{inst}: {n} cycles in {len(spans)} traces ({', '.join(lb for lb, _, _ in spans)}), "
          f"{ndef} defined slots, {int(np.asarray(cmd['pop_i']).sum())} pops, {int(np.asarray(cmd['push_i']).sum())} pushes "
          f"({time.time() - t0:.1f}s)", flush=True)
    make, runner = u.VARIANTS[a.variant]
    for backend in a.backend or ["simulator"]:
        t0 = time.time()
        try:
            top = make(n, w)
            if backend == "simulator":
                mod = df.build(top, target="simulator")
            else:
                prj = os.path.join(a.project, f"vpu_fifo_{inst}_{a.variant}_{backend}")
                mod = df.build(top, target="systemc", mode="csim", project=prj)
            tb = time.time() - t0
            got = runner(mod, cmd, n, w)
        except Exception as e:  # a build or run failure is a finding
            print(f"UNIT-FAIL vpu_fifo:{inst} {a.variant} {backend}: {type(e).__name__}: {str(e)[:600]}")
            continue
        compare(got, want_int, reason, cmd, spans,
                f"vpu_fifo:{inst} {a.variant} {backend} (build {tb:.1f}s, run {time.time() - t0 - tb:.1f}s)")


if __name__ == "__main__":
    main()
