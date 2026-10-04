# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Catapult RTL of a LANDED unit variant (as emitted: Connections streams for
the rank-1 boundary arrays, RAM pins for the others) vs the Phase 0 oracle.

    $ALLO_PYTHON u3c_cmp_landed.py <unit> <variant> <prj> [--inst I] [--n N]

The unit's own runner (``VARIANTS[variant][1]``, the one ``check.py`` uses on
the simulator and csim) is called with a stand-in ``mod`` that drives the
Catapult RTL in Verilator: every region argument is one port group of the
region top, in argument order -- a Connections In/Out stream (one element per
row) or a memory (``_radr/_re/_q/_rrdy``, ``_wadr/_d/_we/_wrdy``) served by the
driver exactly as the emitted ``AlloMemPins`` does (always ready; ``q`` holds
``mem[radr]`` from the edge after ``re``; a write lands at the edge). So the
verdict is ``check_trace``'s per-row comparison on the RTL, and every output
element carries the cycle it was pushed or written, against the cycle its
input row was accepted: the measured skew of a self-timed composition (H7).
"""
import argparse, hashlib, importlib, inspect, os, re, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.getcwd())
from examples.minitpu.harness import check, latency, rtl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("unit"); ap.add_argument("variant"); ap.add_argument("prj")
ap.add_argument("--inst", default=None)
ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()
u = importlib.import_module(f"examples.minitpu.units.{a.unit}")
inst = a.inst or getattr(u, "DEFAULT", None)
name = os.path.basename(a.prj.rstrip("/")).replace(".prj", "")
k = open(os.path.join(a.prj, "kernel.cpp")).read()
# the region top is the SC_MODULE emitted right before the testbench (`top` for a
# @df.region, the Architecture's name for a composed one)
mods = re.findall(r"^SC_MODULE\((\w+)\)", k, re.M)
TOP = mods[mods.index("tb") - 1]
blk = k[k.index(f"SC_MODULE({TOP})"):]
blk = blk[: blk.index("\n};")]
src = os.path.abspath(os.path.join(a.prj, "build", "Catapult", f"{TOP}.v1", "concat_sim_rtl.v"))
print(f"top {TOP}")
nw = rtl.nwords


def _w(t):
    m = re.match(r"(?:ac_int|ap_uint|ap_int)<(\d+)", t)
    return int(m.group(1)) if m else {"bool": 1}[t]


# region args in order: ('in'|'out'|'mr'|'mw', base name, data width, [addr width])
groups = {}
for t, p in re.findall(r"Connections::In< (.+?) > (\w+);", blk):
    groups[int(p[1:])] = ("in", p, _w(t), 0)
for t, p in re.findall(r"Connections::Out< (.+?) > (\w+);", blk):
    groups[int(p[1:])] = ("out", p, _w(t), 0)
for t, p in re.findall(r"sc_out< (.+?) > (\w+)_radr;", blk):
    q = re.search(rf"sc_in< (.+?) > {p}_q;", blk).group(1)
    groups[int(p[1:])] = ("mr", p, _w(q), _w(t))
for t, p in re.findall(r"sc_out< (.+?) > (\w+)_wadr;", blk):
    d = re.search(rf"sc_out< (.+?) > {p}_d;", blk).group(1)
    groups[int(p[1:])] = ("mw", p, _w(d), _w(t))
args = [groups[i] for i in sorted(groups)]
print(f"ports ({len(args)} region args): " + " ".join(f"{p}:{kd}{w}" for kd, p, w, _ in args))


def driver(n, sizes):
    """sizes[i]: elements of region arg i (n for streams)."""
    nl = "\n"
    decl, init, pre, post, fire, wr, cap = [], [], [], [], [], [], []
    done_terms = []
    ni = 0
    for i, (kd, p, w, aw) in enumerate(args):
        sz, k_ = sizes[i], nw(w)
        if kd == "in":
            decl.append(f"  std::vector<uint64_t> in{i}(n * {k_}); uint64_t k{i} = n;")
            init.append(f"  if (std::fread(in{i}.data(), 8, n * {k_}, fi) != n * {k_}) return 2;")
            pre.append(f"    dut.{p}_vld = k{i} < n; if (k{i} < n) setp(dut.{p}_dat, &in{i}[k{i} * {k_}]); else setp(dut.{p}_dat, zeros);")
            fire.append(f"    fire{i} = dut.{p}_vld && dut.{p}_rdy;")
            post.append(f"    if (fire{i}) {{ if (k{i} < n && acc[k{i}] < cyc) acc[k{i}] = cyc; ++k{i}; last = cyc; }}")
            decl.append(f"  bool fire{i} = false;")
        elif kd == "out":
            decl.append(f"  std::vector<uint64_t> out{i}(n * {k_ + 1}, 0); uint64_t j{i} = 0;")
            pre.append(f"    dut.{p}_rdy = 1;")
            fire.append(f"""    if (dut.{p}_vld && dut.{p}_rdy) {{ if (j{i} < n) {{ getp(dut.{p}_dat, &out{i}[j{i} * {k_ + 1}]); out{i}[j{i} * {k_ + 1} + {k_}] = cyc; }} ++j{i}; last = cyc; }}""")
            wr.append(f"  std::fwrite(out{i}.data(), 8, out{i}.size(), fo);")
            done_terms.append(f"j{i} >= n")
        elif kd == "mr":
            decl.append(f"  std::vector<uint64_t> mem{i}({sz} * {k_}); bool re{i} = false; uint64_t ra{i} = 0;")
            init.append(f"  if (std::fread(mem{i}.data(), 8, {sz} * {k_}, fi) != {sz} * {k_}) return 2;")
            pre.append(f"    dut.{p}_rrdy = 1;")
            fire.append(f"    re{i} = dut.{p}_re; ra{i} = (uint64_t)dut.{p}_radr;")
            post.append(f"    if (re{i} && ra{i} < {sz}) setp(dut.{p}_q, &mem{i}[ra{i} * {k_}]);")
        else:
            decl.append(f"  std::vector<uint64_t> mem{i}({sz} * {k_ + 1}, 0); bool we{i} = false; uint64_t wa{i} = 0; uint64_t wd{i}[{k_}];")
            pre.append(f"    dut.{p}_wrdy = 1;")
            fire.append(f"    we{i} = dut.{p}_we; wa{i} = (uint64_t)dut.{p}_wadr; getp(dut.{p}_d, wd{i});")
            post.append(f"    if (we{i} && wa{i} < {sz}) {{ for (int q = 0; q < {k_}; ++q) mem{i}[wa{i} * {k_ + 1} + q] = wd{i}[q]; mem{i}[wa{i} * {k_ + 1} + {k_}] = cyc; last = cyc; }}")
            wr.append(f"  std::fwrite(mem{i}.data(), 8, mem{i}.size(), fo);")
    done = " && ".join(["dut.done"] + done_terms)
    return f"""// generated by u3c_cmp_landed.py: Connections + AlloMemPins-protocol driver
#include "V{TOP}.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <vector>
{rtl._TRACE_HELPERS}
int main(int argc, char** argv) {{
  Verilated::commandArgs(argc, argv);
  const uint64_t n = {n};
  static const uint64_t zeros[64] = {{0}};
  FILE* fi = std::fopen(argv[1], "rb");
  std::vector<uint64_t> acc(n, 0);
{nl.join(decl)}
{nl.join(init)}
  std::fclose(fi);
  V{TOP} dut;
  uint64_t cyc = 0, last = 0;
  dut.rst = 0;
{nl.join(pre)}
  for (int r = 0; r < 4; ++r) {{ dut.clk = 0; dut.eval(); dut.clk = 1; dut.eval(); }}
  dut.rst = 1;
{nl.join(f"  k{i} = 0;" for i, (kd, p, w, aw) in enumerate(args) if kd == "in")}
  while (!({done})) {{
    dut.clk = 0;
{nl.join(pre)}
    dut.eval();
{nl.join(fire)}
    dut.clk = 1;
    dut.eval();
{nl.join(post)}
    dut.eval();
    ++cyc;
    if (cyc - last > 20000) {{ std::fprintf(stderr, "stalled at cycle %llu (done=%d)\\n", (unsigned long long)cyc, (int)dut.done); break; }}
  }}
  std::printf("STREAM cycles=%llu n=%llu\\n", (unsigned long long)cyc, (unsigned long long)n);
  FILE* fo = std::fopen(argv[2], "wb");
  std::fwrite(acc.data(), 8, n, fo);
{nl.join(wr)}
  std::fclose(fo);
  return 0;
}}
"""


def build(code):
    cache = os.environ.get("MINITPU_HARNESS_CACHE", "/work/shared/users/phd/sk3463/scratch/u3c/hcache")
    h = hashlib.sha256(code.encode() + open(src, "rb").read()).hexdigest()[:16]
    d = os.path.join(cache, f"catl-{TOP}-{h}")
    exe = os.path.join(d, f"V{TOP}")
    if os.path.exists(exe):
        return exe
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "driver.cpp"), "w").write(code)
    cmd = ["verilator", "--cc", "--exe", "--build", "-Wno-fatal", "-O3", "--top-module", TOP, "--Mdir", d,
           "-j", "8", src, os.path.join(d, "driver.cpp"), "-CFLAGS", "-std=c++17 -O2"]
    r = subprocess.run(cmd, env=dict(os.environ, CXX=rtl._cxx()), capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(exe):
        raise RuntimeError(f"verilator build failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return exe


stamps = {}  # region arg index -> int64 stamps per element (outputs)
acc_cycles = None
total_cycles = 0


def fake_mod(*arrays):
    """Drive the RTL with the runner's numpy arrays (region arg order)."""
    global acc_cycles, total_cycles
    assert len(arrays) == len(args), (len(arrays), len(args))
    n = None
    for (kd, p, w, _), arr in zip(args, arrays):
        if kd in ("in", "out"):
            n = len(arr) if n is None else n
            assert len(arr) == n, (p, len(arr), n)
    arrays = [np.asarray(a_) for a_ in arrays]
    if n is None:  # every port is a memory (no stream): the rows are the arrays' length
        n = len(arrays[0])
    sizes = [np.asarray(arr).size for arr in arrays]  # a 2-D lane array is one flat memory
    code = driver(n, sizes)
    exe = build(code)
    d = os.path.dirname(exe)
    fi, fo = os.path.join(d, f"in.{os.getpid()}"), os.path.join(d, f"out.{os.getpid()}")
    with open(fi, "wb") as f:
        for (kd, p, w, _), arr in zip(args, arrays):
            if kd in ("in", "mr"):
                rtl.pack(np.asarray(arr).reshape(-1).astype(np.int64), w).astype(np.uint64).tofile(f)
    r = subprocess.run([exe, fi, fo], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"driver exited {r.returncode}: {r.stderr[-1500:]}")
    if "stalled" in r.stderr:
        print("    " + r.stderr.strip().splitlines()[-1])
    raw = np.fromfile(fo, dtype=np.uint64)
    for p_ in (fi, fo):
        os.remove(p_)
    total_cycles = int(re.search(r"cycles=(\d+)", r.stdout).group(1))
    acc_cycles = raw[:n].astype(np.int64)
    off = n
    for i, ((kd, p, w, _), arr) in enumerate(zip(args, arrays)):
        if kd in ("out", "mw"):
            sz, k_ = arr.size, nw(w)
            blk_ = raw[off: off + sz * (k_ + 1)].reshape(sz, k_ + 1)
            off += sz * (k_ + 1)
            vals = rtl.unpack(blk_[:, :k_])
            m = (1 << w) - 1
            arr.reshape(-1)[:] = np.array([v & m for v in vals], dtype=arr.dtype)
            stamps[i] = blk_[:, k_].astype(np.int64)


if a.variant.startswith("form:"):  # a track-C form with its own make()/run()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "forms"))
    _f = importlib.import_module(a.variant[5:])
    make, runner = _f.make, _f.run
    trace_variant = None
else:
    make, runner = u.VARIANTS[a.variant]
    trace_variant = a.variant
t0 = time.time()
if u.RTL.shape == "valid":
    stim = u.stimulus()
    if a.n:
        stim = stim[: a.n]
    n = len(stim)
    want, _ = rtl.run(u.RTL, stim.astype(np.uint64))
    want = want[:, 0]
    got = runner(fake_mod, stim)
    kd_ = int((got != want).sum())
    oi = [i for i, g in enumerate(args) if g[0] in ("out", "mw")][0]
    lat = latency_hist = dict(zip(*[x.tolist() for x in np.unique(stamps[oi] - acc_cycles, return_counts=True)]))
    print(f"{'UNIT-MATCH' if kd_ == 0 else 'UNIT-DIFF '} {a.unit} {a.variant} {name} catapult-rtl {n - kd_}/{n} vs MiniTPU RTL; "
          f"latency {lat} (declared {u.RTL.latency}); {total_cycles / n:.4f} cyc/vector ({time.time() - t0:.1f}s)")
else:
    w = u.WIDTH[inst]
    unit = u.INSTANCES[inst]
    if hasattr(u, "_BUILT"):  # a geometry-bearing runner reads the instance its make() recorded (track A note A8)
        u._BUILT["inst"] = inst
    cmd, spans = check._trace_all(u, inst, a.n, trace_variant)
    n = len(next(iter(cmd.values())))
    packed = {p: rtl.pack(cmd[p], wd) for p, wd in unit.inputs}
    rtl_out = rtl.run_trace(unit, packed, seed=1)
    want, reason, _ = u.REF(inst, packed)
    rtl_int = {p: rtl.unpack(rtl_out[p]) for p in want}
    shift = getattr(u, "RESP_SHIFT", {}).get(a.variant)
    shift = shift(inst) if callable(shift) else (shift or {})
    got = runner(fake_mod, cmd, n, w)
    kb = tot = masked = masked_eq = 0
    per_label, first = {}, []
    for p in want:
        g = [int(x) for x in got[p]]
        r_ = rtl_int[p]
        ks = shift.get(p, 0)
        for i in range(max(0, ks), n + min(0, ks)):
            why = reason[p][i]
            if why:
                masked += 1
                masked_eq += int(g[i - ks] == r_[i])
                continue
            tot += 1
            if g[i - ks] != r_[i]:
                kb += 1
                lab = next(lb for lb, s0, s1 in spans if s0 <= i < s1)
                per_label[lab] = per_label.get(lab, 0) + 1
                if len(first) < 4:
                    first.append(f"{lab} cycle {i} {p}: catapult {g[i - ks]:x} rtl {r_[i]:x}")
    # per output region arg: stamps relative to the row's accept cycle
    skew = {}
    for i, (kd, p, w_, _) in enumerate(args):
        if i in stamps:
            st = stamps[i]
            per_row = len(st) // n
            rows = np.repeat(acc_cycles, per_row)
            d = st - rows
            v, c = np.unique(d, return_counts=True)
            h = dict(zip(v.tolist(), c.tolist()))
            skew[p] = h if len(h) <= 6 else {"min": int(d.min()), "max": int(d.max()), "median": int(np.median(d))}
    print(f"{'UNIT-MATCH' if kb == 0 else 'UNIT-DIFF '} {a.unit}:{inst} {a.variant} {name} catapult-rtl {tot - kb}/{tot} defined "
          f"({masked} masked; catapult==rtl on {masked_eq}) cycle=per-row; {total_cycles / n:.4f} cyc/row; "
          f"output skew vs input accept (cycles) {skew} ({time.time() - t0:.1f}s)")
    if kb:
        print("    differing defined slots by trace: " + ", ".join(f"{lb}={c}" for lb, c in per_label.items()))
        for f in first:
            print(f"    e.g. {f}")
    if a.unit == "mxu":  # the contract as the consumer sees it: pop data in pop order, first valid, drops
        wv, wd = rtl_int["output_valid_o"], rtl_int["output_data_o"]
        gv, gd = [int(x) for x in got["output_valid_o"]], [int(x) for x in got["output_data_o"]]
        pop = [int(x) for x in cmd["output_pop_i"]]
        push = [int(x) for x in cmd["input_push_i"]]
        # the VLD stream's push-edge stamps: the 1-bit output whose values are output_valid_o
        vst = None
        for i, (kd, p_, w_, _) in enumerate(args):
            if kd == "out" and w_ == 1 and i in stamps and [int(x) for x in got["output_valid_o"]] == gv and vst is None:
                pass
        vo_arrays = [i for i, (kd, p_, w_, _) in enumerate(args) if kd == "out" and w_ == 1 and i in stamps]
        if len(vo_arrays) >= 1:
            vst = stamps[vo_arrays[-1]]  # VLD is the last 1-bit output in both the lockstep and the Stream-FIFO forms
        for lab, s0, s1 in spans:
            pw = [wd[t] for t in range(s0, s1) if pop[t] and wv[t]]
            pg = [gd[t] for t in range(s0, s1) if pop[t] and gv[t]]
            ok = sum(int(x == y) for x, y in zip(pw, pg))
            rw = next((t - s0 for t in range(s0, s1) if wv[t]), None)
            rg = next((t - s0 for t in range(s0, s1) if gv[t]), None)
            vt = [t for t in range(s0, s1) if reason["output_valid_o"][t] == ""]
            vok = sum(int(gv[t] == wv[t]) for t in vt)
            pv = ""
            if vst is not None and rg is not None:
                lp = [t for t in range(s0, s0 + rg) if push[t]]
                if lp:
                    pv = f" | push->valid: rows {rg - (lp[-1] - s0)}, cycles {int(vst[s0 + rg] - acc_cycles[lp[-1]])}"
            print(f"    MXU-CONTRACT {lab:18s} pops rtl {len(pw)} catapult {len(pg)} in-order-equal {ok}/{len(pw)} | "
                  f"output_valid per cycle {vok}/{len(vt)} | first valid rtl {rw} catapult {rg}{pv}")
