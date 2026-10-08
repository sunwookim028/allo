# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-23's no-stall condition on the CLOSED sequencer loop, in SystemC csim
(track B's ``d23_rate.py`` adapted to a five-unit composition).

    $ALLO_PYTHON dev/records/minitpu/u4_seqloop_2026-10-08/scripts/seqloop_rate.py [--qd 2] [--n N]

Builds ``template/sequencer.py`` for csim on Phase 0's joined sequencer traces
(``units/sequencer.py``, instance ``loop``), then stamps the emitted
``kernel.cpp`` (a hand patch to a generated file, the ``u1_pipe`` technique):
the clock cycle at the top of every cycle-loop iteration of every unit
(``issue``, ``fq``, ``lctl``, ``lcap``, ``sagu``, ``loader``), after every
command ``Push`` inside the issue loop (the D-23 V/X/M puts, found as the
streams the end-of-trace pad loop pushes 0 on), and after every receiver
``Pop``. Values are still held to the reference, so a timing patch cannot
hide a value bug.

Two questions, kept apart:

1. **Cycle-locked in iterations (D-24).** Iteration t of every unit is RTL
   cycle t; the issue unit issues in iteration t iff the RTL issues in cycle t.
   That is what the value check proves (``bundle_issued_o`` is a checked slot),
   and it is restated here: ``issues at the RTL's cycle``.
2. **What an iteration costs in csim time.** The interval between issue-loop
   iterations, split by iterations that put a command and those that do not
   (D-23: a command Stream that costs nothing leaves the two equal); and
   whether the interval is one constant c (``locked``). Stall cycles are
   counted against that constant (iterations slower than c), and c itself is
   reported against the RTL's 1.
"""

import argparse
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
STAMP = "(long long)(sc_time_stamp() / sc_time(1, SC_NS))"
UNITS = ("issue_0", "fq_0", "lctl_0", "lcap_0", "sagu_0", "loader_0")
RECEIVERS = ("vrx_0", "xrx_0", "mrx_0")


def module_span(src, name):
    a = src.index(f"SC_MODULE({name})")
    b = src.find("SC_MODULE(", a + 10)
    return a, (b if b > 0 else len(src))


def patch(path):
    s = open(path).read()
    head = '#include <fstream>\nstatic std::ofstream __stamp("stamp.txt");\n'
    counts = {}
    for u in UNITS:
        a, b = module_span(s, u)
        mod = s[a:b]
        # the cycle loop: the first top-level `for (int tN = 0; tN < N; tN++) {`
        mod, k = re.subn(r"(\n( +)(?:\w+: )?for \(int (t\d*) = 0; \3 < \d+; \3\+\+\) \{[^\n]*\n)",
                         lambda m: m.group(1) + f'{m.group(2)}  __stamp << "I {u} " << {m.group(3)} << " " << {STAMP} << "\\n";\n',
                         mod, count=1)
        counts[u] = k
        if u == "issue_0":
            pads = set(re.findall(r"(v\d+)\.Push\(0\);", mod))
            loop_var = re.search(r"for \(int (t\d*) = 0;", mod).group(1)
            mod, k = re.subn(r"\n( {8,})(v\d+)\.Push\((v\d+)\);([^\n]*)",
                             lambda m: (f'\n{m.group(1)}{m.group(2)}.Push({m.group(3)});{m.group(4)}\n{m.group(1)}'
                                        f'__stamp << "P {m.group(2)} " << {loop_var} << " " << {STAMP} << "\\n";')
                             if m.group(2) in pads else m.group(0), mod)
            counts["command pushes"] = k
        s = s[:a] + mod + s[b:]
    for r in RECEIVERS:
        a, b = module_span(s, r)
        mod = s[a:b]
        mod, k = re.subn(r"\n( +)(\w+ v\d+ = v\d+\.Pop\(\);[^\n]*)",
                         lambda m: f'\n{m.group(1)}{m.group(2)}\n{m.group(1)}__stamp << "R {r} " << {STAMP} << "\\n";',
                         mod, count=1)
        counts[r] = k
        s = s[:a] + mod + s[b:]
    open(path, "w").write(head + s)
    return counts


def hist(x):
    v, c = np.unique(np.asarray(x), return_counts=True)
    return dict(zip(v.tolist(), c.tolist()))


def analyse(prj, iss_rtl):
    I, P, R = {}, {}, {}
    for line in open(os.path.join(prj, "stamp.txt")):
        f = line.split()
        if f[0] == "I":
            I.setdefault(f[1], {})[int(f[2])] = int(f[3])
        elif f[0] == "P":
            P.setdefault(f[1], []).append((int(f[2]), int(f[3])))
        elif f[0] == "R":
            R.setdefault(f[1], []).append(int(f[2]))
    out = {}
    T = np.array([I["issue_0"][t] for t in range(len(I["issue_0"]))])
    n = len(T)
    d = np.diff(T)
    put_iters = sorted({t for v in P.values() for t, _ in v})
    mask = np.zeros(n - 1, dtype=bool)
    mask[[t for t in put_iters if t < n - 1]] = True
    c = int(np.bincount(d).argmax())
    out["n"] = n
    out["csim cycles per iteration (mode)"] = c
    out["interval, all"] = hist(d)
    out["interval, command put"] = hist(d[mask])
    out["interval, no command"] = hist(d[~mask])
    out["locked (one constant interval after start-up)"] = bool(np.all(d[1:] == c))
    out["stall cycles against the constant"] = int(np.clip(d - c, 0, None).sum())
    iss = [t for t, v in enumerate(iss_rtl) if v]
    # the first interval is the start-up (the units' first tokens); time is
    # held to T[1] + c * (t - 1) from iteration 1 on
    out["first interval (start-up)"] = int(d[0])
    out["stall cycles against the constant, after start-up"] = int(np.clip(d[1:] - c, 0, None).sum())
    off = [int(T[t] - T[1] - c * (t - 1)) for t in iss if t >= 1]
    out["issues at the RTL's cycle (iteration t = cycle t, time T1 + c*(t-1))"] = (
        f"{sum(1 for x in off if x == 0)}/{len(off)}, max late {max(off) if off else 0}")
    out["puts"] = {k: len(v) for k, v in P.items()}
    # lockstep between units: iteration t's start, unit vs issue
    for u in UNITS[1:]:
        if u in I:
            Tu = np.array([I[u][t] for t in range(min(n, len(I[u])))])
            out[f"{u} start - issue start"] = hist(Tu - T[:len(Tu)])
    lat = {}
    pairs = zip(sorted(P.items(), key=lambda kv: int(kv[0][1:])), RECEIVERS)
    for (pname, pv), rname in pairs:
        rv = R.get(rname, [])
        k = min(len(pv), len(rv))
        lat[f"{pname}->{rname}"] = hist([rv[i] - pv[i][1] for i in range(k)]) if k else {}
    out["put -> pop"] = lat
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qd", type=int, default=2)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--prj", default="/tmp/u4_seqloop_rate")
    a = ap.parse_args()
    import allo.dataflow as df
    from examples.minitpu.harness import check, rtl
    from examples.minitpu.template import sequencer as T
    from examples.minitpu.units import sequencer as S

    cmd, spans = check._trace_all(S, "loop", a.n, "loop")
    n = len(next(iter(cmd.values())))
    prj = os.path.join(a.prj, f"seqloop_qd{a.qd}_n{n}")
    mod = df.build(T.architecture(n, a.qd).region("simulator", {"iram": "registers", "lb": "registers"}),
                   target="systemc", mode="csim", project=prj)
    k = patch(os.path.join(prj, "kernel.cpp"))
    print(f"stamped: {k}")
    t0 = time.time()
    got = T.run(mod, cmd, n)
    packed = {p: rtl.pack(cmd[p], w) for p, w in S.LOOP_RTL.inputs}
    want, reason, _ = S.REF("loop", packed)
    bad = tot = 0
    for p in want:
        r = rtl.unpack(want[p])
        for t in range(n):
            if not reason[p][t]:
                tot += 1
                bad += int(got[p][t]) != int(r[t])
    iss = [int(x) for x in rtl.unpack(want["bundle_issued_o"])]
    res = analyse(prj, iss)
    print(f"SEQLOOP-RATE qd={a.qd} n={n}: values {tot - bad}/{tot} defined equal to the reference; "
          f"{sum(iss)} RTL issues; csim run {time.time() - t0:.1f}s")
    for key, v in res.items():
        print(f"  {key}: {v}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
