"""Shared driver for the U4 track E AMC scripts (AMC env; source env.sh; run
under ``scl enable gcc-toolset-13 --``). A kernel is generated as text, loaded
as a module, built with AMC's vendored Allo (``allo.customize(fn).build``) on
``llvm`` (the frontend's semantics) and ``amc`` (Verilator RTL), and driven in
chunks of ``N`` rows (one ``f(...)`` call per chunk; ``f.rpt["cycles"]`` is
AMC's cycle count for the chunk)."""
import os, sys, time, traceback, importlib.util
import numpy as np

TMP = os.environ.get("TMPDIR", "/tmp")


def load(src, name):
    kp = f"{TMP}/{name}.py"
    open(kp, "w").write(src)
    open(f"{name}.py", "w").write(src)  # a copy beside the run, for the record
    spec = importlib.util.spec_from_file_location(name, kp)
    K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
    return K


def build(fn, tgt, sched, loop=None, tag="k"):
    import allo
    t0 = time.time()
    s = allo.customize(fn)
    if sched == "pipeline":
        loops = s.get_loops()
        s.pipeline(loops[loop[0]][loop[1]])
    f = s.build(target=tgt)
    print(f"BUILT {tag} {tgt} {sched} {time.time() - t0:.1f}s", flush=True)
    if tgt == "amc":
        try:
            f.dump_allocation(f"{tag}_{sched}.alloc.txt")
            f.dump_verilog(f"{tag}_{sched}_hdl")
        except Exception as e:
            print("NOTE dump failed:", type(e).__name__, str(e)[:200])
    return f


def run(f, tgt, ins, outs_spec, N, max_rows=None, tag=""):
    n = len(ins[0]) if max_rows is None else min(max_rows, len(ins[0]))
    pad = (-n) % N
    ins = [np.concatenate([a[:n], np.zeros((pad,) + a.shape[1:], a.dtype)]) for a in ins]
    outs = [np.zeros((n + pad,) + tuple(tail), dt) for tail, dt in outs_spec]
    cyc = []
    t0 = time.time()
    for c in range((n + pad) // N):
        sl = slice(c * N, (c + 1) * N)
        o = [np.zeros((N,) + tuple(tail), dt) for tail, dt in outs_spec]
        f(*[np.ascontiguousarray(a[sl]) for a in ins], *o)
        if tgt == "amc":
            cyc.append(int(f.rpt["cycles"]))
        for k in range(len(o)):
            outs[k][sl] = o[k]
    print(f"RAN {tag} {tgt}: {(n + pad) // N} chunks of {N}, cycles/chunk {sorted(set(cyc))} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return [a[:n] for a in outs], cyc


def verdict(tag, d, got, ports, n):
    tot = bad = 0
    ex = []
    spans = list(zip(d["span_labels"], d["span_bounds"])) if "span_labels" in d else []
    per = {}
    for p in ports:
        want = d[p][:n]
        g = got[p].reshape(want.shape).astype(np.uint32)
        m = d["def_" + p][:n]
        diff = (g != want).any(axis=1) & m
        tot += int(m.sum()); bad += int(diff.sum())
        for i in np.flatnonzero(diff):
            lab = next((str(l) for l, (a, b) in spans if a <= i < b), "?")
            per[lab] = per.get(lab, 0) + 1
            if len(ex) < 6:
                ex.append(f"{lab} row {i} {p}: tool {[hex(x) for x in g[i]]} rtl {[hex(x) for x in want[i]]}")
    t = "UNIT-MATCH" if bad == 0 else "UNIT-DIFF "
    print(f"{t} {tag}: {tot - bad}/{tot} defined slots (rows {n})", flush=True)
    if bad:
        print("    by trace: " + ", ".join(f"{k}={v}" for k, v in per.items()))
        for e in ex:
            print("    e.g. " + e)
    return bad


def guarded(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        print(f"FAIL {type(e).__name__}: {str(e)[:1500]}", flush=True)
        traceback.print_exc(limit=3)
        return None
