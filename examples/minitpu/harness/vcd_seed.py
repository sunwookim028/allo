# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed U2 traces from MiniTPU's own testbenches (README D-7, second check).

``extract(tb, sources, dut, clk, unit)`` builds MiniTPU's ``tb/<tb>.sv``
unchanged under ``verilator --binary --timing --trace``, inside a two-line
wrapper that only adds ``$dumpfile``/``$dumpvars``, runs it (it must print
``PASS``), and reads the VCD back: at every rising edge of the DUT's clock the
DUT's input ports, as they were just before the edge, become one row of a
command trace for ``unit`` (an ``rtl.RtlUnit`` of shape ``trace``). So the
scenario is replayed exactly, without transcribing it by hand.

It also returns what the tb's DUT showed on each output, in ``rtl.py``'s row
convention (``"pre"`` ports before the edge, ``"post"`` ports after it; a
``"post"`` port's last row is unknown and is ``None``), so the replay can be
held to the tb's own run as well as to the reference. The trace ends with one
extra row holding the values at the end of the run (the tb's last check may
come after the last edge).
"""

import hashlib
import os
import re
import shutil
import subprocess

from examples.minitpu.harness import rtl


def _build_and_run(tb, sources):
    home = rtl.minitpu_home()
    cache = os.environ.get("MINITPU_HARNESS_CACHE", os.path.join(home, ".build", "allo_harness"))
    h = hashlib.sha256(tb.encode())
    for s in sources + [f"tb/{tb}.sv"]:
        with open(os.path.join(home, s), "rb") as f:
            h.update(f.read())
    d = os.path.join(cache, f"seed-{tb}-{h.hexdigest()[:12]}")
    vcd = os.path.join(d, "dump.vcd")
    if os.path.exists(vcd):
        return vcd
    os.makedirs(d, exist_ok=True)
    top = os.path.join(d, "u2_seed_top.sv")
    with open(top, "w", encoding="utf-8") as f:
        f.write(
            "`timescale 1ns/1ps\n"
            f"module u2_seed_top;\n  {tb} tb();\n"
            f'  initial begin $dumpfile("{vcd}"); $dumpvars; end\nendmodule\n'
        )
    verilator = shutil.which("verilator") or "verilator"
    cmd = (
        [verilator, "--binary", "--timing", "--trace", "-Wno-fatal", "-j", "8",
         "--top-module", "u2_seed_top", "--Mdir", d]
        + [os.path.join(home, s) for s in sources]
        + [os.path.join(home, "tb", f"{tb}.sv"), top]
    )
    env = dict(os.environ, CXX=rtl._cxx())
    r = subprocess.run(cmd, cwd=home, env=env, capture_output=True, text=True)
    exe = os.path.join(d, "Vu2_seed_top")
    if r.returncode != 0 or not os.path.exists(exe):
        raise RuntimeError(f"seed build of {tb} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    r = subprocess.run([exe], cwd=d, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or f"PASS {tb}" not in r.stdout:
        if os.path.exists(vcd):
            os.remove(vcd)
        raise RuntimeError(f"{tb} did not pass:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return vcd


def _norm(scope):
    return scope.replace("(", "[").replace(")", "]")


def read_vcd(path, scope_suffix, names):
    """``{name: [(time, int)]}`` changes of the vars ``names`` declared in the
    scope whose path ends with ``scope_suffix`` (a dotted path)."""
    want = _norm(scope_suffix).split(".")
    scope, codes = [], {}
    changes = {n: [] for n in names}
    with open(path, encoding="utf-8") as f:
        it = iter(f)
        for line in it:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "$scope":
                scope.append(_norm(tok[2]))
            elif tok[0] == "$upscope":
                scope.pop()
            elif tok[0] == "$var":
                name = _norm(tok[4])
                if scope[-len(want):] == want and name in changes:
                    codes.setdefault(tok[3], []).append(name)
            elif tok[0] == "$enddefinitions":
                break
        missing = set(names) - {n for v in codes.values() for n in v}
        if missing:
            raise KeyError(f"{missing} not found under scope ...{scope_suffix} in {path}")
        t = 0
        for line in it:
            if not line.strip():
                continue
            c = line[0]
            if c == "#":
                t = int(line[1:])
                continue
            if c in "bB":
                val, code = line[1:].split()
            elif c in "01xzXZ":
                val, code = c, line[1:].strip()
            else:
                continue
            if code in codes:
                v = int(re.sub("[xXzZ]", "0", val), 2)
                for n in codes[code]:
                    changes[n].append((t, v))
    return changes


def extract(tb, sources, dut, clk, unit):
    """Run ``tb`` and return ``(cmd, seen)`` for ``unit`` (module docstring)."""
    vcd = _build_and_run(tb, sources)
    ins = [p for p, _ in unit.inputs]
    outs = [rtl.out_spec(o) for o in unit.outputs]
    ch = read_vcd(vcd, f"tb.{dut}", [clk] + ins + [o[0] for o in outs])
    edges = [t for (t, v), (_, prev) in zip(ch[clk][1:], ch[clk][:-1]) if v == 1 and prev == 0]
    if ch[clk] and ch[clk][0][1] == 1 and ch[clk][0][0] > 0:
        edges.insert(0, ch[clk][0][0])

    def before(sig, times):
        """The value of ``sig`` just before each time in ``times``."""
        out, cur, i, c = [], 0, 0, ch[sig]
        for t in times:
            while i < len(c) and c[i][0] < t:
                cur = c[i][1]
                i += 1
            out.append(cur)
        return out

    # One more row with the values at the end of the run: a tb that checks an
    # asynchronous read just after an edge and then calls $finish (as
    # tb_vpu_alu_regfile's last expect_vreg does) is seen there and only there.
    times = edges + [float("inf")]
    cmd = {p: before(p, times) for p in ins}
    seen = {}
    for p, _, smp in outs:
        pre = before(p, times)
        seen[p] = pre if smp == "pre" else pre[1:] + [None]
    return cmd, seen
