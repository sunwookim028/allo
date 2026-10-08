# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-23's condition, measured in SystemC csim: do the three command
Streams cost the issue unit a stall or an issue-rate loss?

    $ALLO_PYTHON dev/records/minitpu/u4_track_b_2026-10-08/scripts/d23_rate.py \\
        [--unit vpu_cmd] [--variant streams] [--n N] [--prj DIR]

Builds ``units/<unit>.VARIANTS[<variant>]`` for csim on Phase 0's joined
sequencer traces, then stamps the emitted ``kernel.cpp`` (a hand patch to a
generated file, the ``u1_pipe`` ``csim_cycles.py`` technique): the clock cycle
at the top of every iteration of the issue kernel's cycle loop, after every
command ``Push`` (the conditional pushes of the cycle loop: exactly the
V/X/M command puts), and after every receiver ``Pop``. Values are still held
to the RTL by the unit's runner, so a timing patch cannot hide a value bug.

What it reports:

* ``iteration interval``: cycles between consecutive iterations of the issue
  loop, split by iterations that send a command and those that do not. A
  command Stream that costs nothing leaves the two equal.
* ``issue cycles vs RTL``: iteration t of the loop is RTL cycle t (the trace is
  the RTL's cycle by cycle), so an issue at RTL cycle t happens at csim cycle
  ``T[t]``; the issue trace is cycle for cycle iff ``T[t] - T[0]`` is ``c * t``
  for one constant ``c`` (c = 1 is the RTL's own rate).
* ``put -> pop``: per slot, csim cycles from a command's push returning to
  its receiver's pop returning (the link latency), and its spread.
"""

import argparse
import importlib
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
STAMP = "(long long)(sc_time_stamp() / sc_time(1, SC_NS))"


def module_span(src, name):
    a = src.index(f"SC_MODULE({name})")
    b = src.find("SC_MODULE(", a + 10)
    return a, (b if b > 0 else len(src))


def patch(path, issue="issue_0", receivers=("vrx_0", "xrx_0", "mrx_0")):
    s = open(path).read()
    head = ('#include <fstream>\nstatic std::ofstream __stamp("stamp.txt");\n')
    a, b = module_span(s, issue)
    mod = s[a:b]
    # the cycle loop: the first `for (int t = 0; ...) {`
    mod, k1 = re.subn(r"(for \(int t = 0; t < \d+; t\+\+\) \{[^\n]*\n)",
                      lambda m: m.group(1) + f'      __stamp << "I " << t << " " << {STAMP} << "\\n";\n', mod, count=1)
    # conditional pushes inside the cycle loop (8+ spaces of indent), not the pads Push(0)
    mod, k2 = re.subn(r"\n( {8,})(v\d+)\.Push\((v\d+)\);([^\n]*)",
                      lambda m: f'\n{m.group(1)}{m.group(2)}.Push({m.group(3)});{m.group(4)}\n{m.group(1)}'
                                f'__stamp << "P {m.group(2)} " << t << " " << {STAMP} << "\\n";', mod)
    s = s[:a] + mod + s[b:]
    k3 = 0
    for r in receivers:
        a, b = module_span(s, r)
        mod = s[a:b]
        mod, k = re.subn(r"\n( +)(\w+ v\d+ = v\d+\.Pop\(\);[^\n]*)",
                         lambda m: f'\n{m.group(1)}{m.group(2)}\n{m.group(1)}'
                                   f'__stamp << "R {r} " << {STAMP} << "\\n";', mod, count=1)
        k3 += k
        s = s[:a] + mod + s[b:]
    s = head + s
    open(path, "w").write(s)
    return k1, k2, k3


def analyse(prj, iss_rtl, slots_rtl):
    I, P, R = {}, {}, {}
    for line in open(os.path.join(prj, "stamp.txt")):
        f = line.split()
        if f[0] == "I":
            I[int(f[1])] = int(f[2])
        elif f[0] == "P":
            P.setdefault(f[1], []).append((int(f[2]), int(f[3])))
        elif f[0] == "R":
            R.setdefault(f[1], []).append(int(f[2]))
    n = len(I)
    T = np.array([I[t] for t in range(n)])
    d = np.diff(T)
    put_iters = sorted({t for v in P.values() for t, _ in v})
    mask = np.zeros(n - 1, dtype=bool)
    mask[[t for t in put_iters if t < n - 1]] = True
    out = {"n": n, "first": int(T[0]), "last": int(T[-1]),
           "interval_all": dict(zip(*[x.tolist() for x in np.unique(d, return_counts=True)])),
           "interval_put": dict(zip(*[x.tolist() for x in np.unique(d[mask], return_counts=True)])),
           "interval_noput": dict(zip(*[x.tolist() for x in np.unique(d[~mask], return_counts=True)])),
           "puts": {k: len(v) for k, v in P.items()}}
    c = (T[-1] - T[0]) / (n - 1)
    out["cycles_per_iteration"] = c
    out["locked"] = bool(np.all(d == d[0]))
    # issue cycles: RTL issue at t -> csim T[t]
    iss = [t for t, v in enumerate(iss_rtl) if v]
    off = [int(T[t] - T[0] - round(c) * t) for t in iss]
    out["issue_offsets"] = ({"at_rtl_cycle": sum(1 for x in off if x == 0), "late": sum(1 for x in off if x),
                             "max_late": max(off), "issues": len(off)} if off else {})
    out["stall_cycles"] = int((d - 1).sum())
    # put -> pop per slot: the k-th real push against the k-th pop (pads come after the loop)
    lat = {}
    # push sites in source order are V, X, M (the kernel's text order); receivers likewise
    pairs = zip(sorted(P.items(), key=lambda kv: int(kv[0][1:])),
                [(r, R.get(r, [])) for r in ("vrx_0", "xrx_0", "mrx_0")])
    for (pname, pv), (rname, rv) in pairs:
        k = min(len(pv), len(rv))
        L = np.array([rv[i] - pv[i][1] for i in range(k)])
        lat[f"{pname}->{rname}"] = dict(zip(*[x.tolist() for x in np.unique(L, return_counts=True)])) if k else {}
    out["put_to_pop"] = lat
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="vpu_cmd")
    ap.add_argument("--variant", default="streams")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--prj", default="/tmp/u4b_d23")
    a = ap.parse_args()
    import allo.dataflow as df
    from examples.minitpu.harness import check, rtl

    u = importlib.import_module(f"examples.minitpu.units.{a.unit}")
    cmd, spans = check._trace_all(u, u.DEFAULT, a.n, a.variant)
    n = len(next(iter(cmd.values())))
    make, runner = u.VARIANTS[a.variant]
    prj = os.path.join(a.prj, f"{a.unit}_{a.variant}_n{n}")
    mod = df.build(make(n, 16, u.DEFAULT), target="systemc", mode="csim", project=prj)
    k = patch(os.path.join(prj, "kernel.cpp"))
    print(f"stamped: loop {k[0]}, command pushes {k[1]}, receiver pops {k[2]}")
    t0 = time.time()
    got = runner(mod, cmd, n, 16)
    packed = {p: rtl.pack(cmd[p], w) for p, w in u.RTL.inputs}
    want, reason, _ = u.REF(u.DEFAULT, packed)
    bad = 0
    for p in want:
        r = rtl.unpack(want[p])
        bad += sum(1 for t in range(n) if not reason[p][t] and int(got[p][t]) != int(r[t]))
    iss = [int(x) for x in got["bundle_issued_o"]]
    res = analyse(prj, iss, None)
    print(f"D23-RATE {a.unit} {a.variant} n={n}: values {'equal' if not bad else f'{bad} differ'} vs reference; "
          f"{sum(iss)} issues; csim run {time.time() - t0:.1f}s")
    for key, v in res.items():
        print(f"   {key}: {v}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
