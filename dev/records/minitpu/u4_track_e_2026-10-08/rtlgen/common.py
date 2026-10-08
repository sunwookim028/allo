"""Shared driver for the U4 track E RTLGen scripts (RTLGen env; source env.sh).

``run(rtl, ins, outs_spec, n_total, N)``: drives the exported kernel's
cocotb+Verilator cosim over the oracle in chunks of ``N`` rows, returning the
concatenated outputs and the per-chunk cycle counts. ``verdict`` compares on
the defined mask exactly as ``check.py`` does (one slot = one output port in
one cycle; a slot differs if any lane differs).
"""
import os, sys, time
import numpy as np


def report_schedule(rtl, top):
    res = rtl.schedule()
    f = res.func(top)
    iis = [r.interval for r in res.cyclic()]
    q = rtl.estimation
    print(f"SCHED {top}: latency {f.latency} | cyclic II {iis} | QOR lat {q.latency} fmax {q.fmax:.1f} "
          f"area {q.area}", flush=True)
    return f.latency, iis


def run(rtl, ins, outs_spec, N, timeout_per_row=40, max_rows=None):
    """ins: list of numpy arrays with leading dim n (inputs, in kernel order);
    outs_spec: list of (shape_tail, dtype) for the outputs (kernel order, after
    the inputs). Returns (list of output arrays [n, ...], cycles list)."""
    n = len(ins[0]) if max_rows is None else min(max_rows, len(ins[0]))
    pad = (-n) % N
    ins = [np.concatenate([a[:n], np.zeros((pad,) + a.shape[1:], a.dtype)]) for a in ins]
    outs = [np.zeros((n + pad,) + tuple(tail), dt) for tail, dt in outs_spec]
    cyc = []
    t0 = time.time()
    for c in range((n + pad) // N):
        sl = slice(c * N, (c + 1) * N)
        o = [np.zeros((N,) + tuple(tail), dt) for tail, dt in outs_spec]
        r = rtl.cosim(*[np.ascontiguousarray(a[sl]) for a in ins], *o, timeout=timeout_per_row * N + 2000)
        cyc.append(r.cycles)
        for k in range(len(o)):
            outs[k][sl] = o[k]
    print(f"COSIM {(n + pad) // N} chunks of {N}: cycles/chunk {sorted(set(cyc))} ({time.time() - t0:.1f}s)",
          flush=True)
    return [a[:n] for a in outs], cyc


def verdict(tag, d, got, ports, n):
    """got: {port: uint32[n, lanes]}; d: the oracle npz."""
    tot = bad = 0
    ex = []
    spans = list(zip(d["span_labels"], d["span_bounds"])) if "span_labels" in d else []
    per = {}
    for p in ports:
        want = d[p][:n]
        g = got[p].reshape(want.shape).astype(np.uint32)
        m = d["def_" + p][:n]
        diff = (g != want).any(axis=1) & m
        tot += int(m.sum())
        bad += int(diff.sum())
        for i in np.flatnonzero(diff)[:3]:
            lab = next((str(l) for l, (a, b) in spans if a <= i < b), "?")
            per[lab] = per.get(lab, 0)
            if len(ex) < 6:
                ex.append(f"{lab} row {i} {p}: tool {[hex(x) for x in g[i]]} rtl {[hex(x) for x in want[i]]}")
        for i in np.flatnonzero(diff):
            lab = next((str(l) for l, (a, b) in spans if a <= i < b), "?")
            per[lab] = per.get(lab, 0) + 1
    t = "UNIT-MATCH" if bad == 0 else "UNIT-DIFF "
    print(f"{t} {tag}: {tot - bad}/{tot} defined slots (rows {n})", flush=True)
    if bad:
        print("    by trace: " + ", ".join(f"{k}={v}" for k, v in per.items() if v))
        for e in ex:
            print("    e.g. " + e)
    return bad
