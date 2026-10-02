"""Catapult RTL of the composed (two-kernel ``Stream``) FIFO region, driven
through its Connections ports in Verilator, vs MiniTPU's ``vpu_fifo`` RTL.

    python hs_cmp.py <prj> --inst w32d4 [--trace composed|<directed label>] [--n N]
        [--stall 2000] [--variant composed]

From the worktree root after ``source examples/minitpu/harness/env-zhang21.sh``.
The region top has one ``Connections::In`` per trace input (``rst``, ``push``,
``push_data``, ``pop``) and one ``Connections::Out`` per output (``pop_data``,
``empty``, ``full``); the port-to-trace mapping is read from ``kernel.cpp``
(the top's bindings to ``producer_0``/``consumer_0``, whose port order is the
kernel's parameter order). The driver offers every input vector k until it is
accepted, holds every output ready, and records per clock which ports fired
and what the outputs carried; it stops when every output has delivered ``n``
words or when nothing fires for ``--stall`` cycles (a deadlock: the blocking
``put`` past full).

Reported: per output port, the iteration-to-cycle offset (transfer j is
iteration j: constant when neither kernel ever stalls), the agreement with
MiniTPU's RTL as ``cmp_composed.py`` defines it (``pop_data`` on pop cycles
exactly; ``empty``/``full`` at the best shift), and the measured push-accept
to pop-deliver latency for the pops that follow their push by one trace cycle.
"""
import argparse, collections, hashlib, os, re, shutil, subprocess, sys, time

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from examples.minitpu.harness import rtl  # noqa: E402
from examples.minitpu.harness.traces import concat  # noqa: E402
from examples.minitpu.units import vpu_fifo as u  # noqa: E402
import cmp_composed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("prj")
ap.add_argument("--inst", default="w32d4")
ap.add_argument("--trace", default="composed", help="composed (the verdict trace) or a directed() label")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--stall", type=int, default=2000)
ap.add_argument("--variant", default="composed")
ap.add_argument("--scratch", default="/work/shared/users/phd/sk3463/scratch/u3fc/hs")
ap.add_argument("--dump", action="store_true", help="print every cycle (short traces)")
a = ap.parse_args()

prj = os.path.abspath(a.prj)
bdir = os.path.join(prj, "build")
cat = [d for d in sorted(os.listdir(bdir)) if d.startswith("Catapult")]
v1 = os.path.join(bdir, cat[-1], "top.v1")
src = os.path.join(v1, "concat_sim_rtl.v")
if not os.path.exists(src):
    src = os.path.join(v1, "concat_rtl.v")
k = open(os.path.join(prj, "kernel.cpp")).read()


def ports_of(mod):
    blk = k[k.index(f"SC_MODULE({mod})"):]
    blk = blk[: blk.index("\n};")]
    ins = re.findall(r"Connections::In< (.+?) > (\w+);", blk)
    outs = re.findall(r"Connections::Out< (.+?) > (\w+);", blk)
    return ins, outs


def width(t):
    m = re.match(r"(?:ac_int|ap_uint)<(\d+)", t)
    return int(m.group(1)) if m else {"bool": 1}[t]


tblk = k[k.index("SC_MODULE(top)"):]
tblk = tblk[: tblk.index("\n};")]
top_in = dict((p, width(t)) for t, p in re.findall(r"Connections::In< (.+?) > (\w+);", tblk))
top_out = dict((p, width(t)) for t, p in re.findall(r"Connections::Out< (.+?) > (\w+);", tblk))
inst_of = dict(re.findall(r"\n  (\w+_0) (u\d+);", tblk))  # module -> instance
bind = {}
for inst, port, sig in re.findall(r"(u\d+)\.(\w+)\((\w+)\);", tblk):
    bind[(inst, port)] = sig
prod_in, prod_out = ports_of("producer_0")
cons_in, cons_out = ports_of("consumer_0")
up, uc = inst_of["producer_0"], inst_of["consumer_0"]
# kernel parameter order: producer(rst, push, pd, qf), consumer(pop, qd, qe)
imap = {"rst_ni": bind[(up, prod_in[0][1])], "push_i": bind[(up, prod_in[1][1])],
        "push_data_i": bind[(up, prod_in[2][1])], "pop_i": bind[(uc, cons_in[0][1])]}
omap = {"full_o": bind[(up, prod_out[0][1])], "pop_data_o": bind[(uc, cons_out[0][1])],
        "empty_o": bind[(uc, cons_out[1][1])]}
assert set(imap.values()) == set(top_in) and set(omap.values()) == set(top_out), (imap, omap, top_in, top_out)
INS = [(p, imap[p], top_in[imap[p]]) for p in u.CMD]
OUTS = [(p, omap[p], top_out[omap[p]]) for p in u.RESP]

# ---- the trace ----
inst = a.inst
w, d = u.GEOM[inst]
if a.trace == "composed":
    cmd, spans = cmp_composed.trace_composed_all(inst, a.n)
else:
    lab, c, legal = next(x for x in u.directed(inst) if x[0] == a.trace)
    cmd, spans = c, [(lab, 0, len(c["rst_ni"]))]
    if a.n:
        cmd = {p: v[: a.n] for p, v in cmd.items()}
        spans = [(lab, 0, len(cmd["rst_ni"]))]
n = len(cmd["rst_ni"])
# the RTL's kernels run exactly the iteration count baked in at emission: pad the
# trace with idle cycles up to it (a shorter RTL count is refused)
n_rtl = int(re.search(r"for \(int t = 0; t < (\d+); t\+\+\)", k).group(1))
assert n_rtl >= n, f"RTL emitted for {n_rtl} iterations, trace has {n}: re-emit with --n {n}"
if n_rtl > n:
    pad = {"rst_ni": 1, "push_i": 0, "push_data_i": 0, "pop_i": 0}
    cmd = {p: list(v) + [pad[p]] * (n_rtl - n) for p, v in cmd.items()}
    spans = spans + [("idle-pad", n, n_rtl)]
    n = n_rtl
unit = u.INSTANCES[inst]
packed = {p: rtl.pack(cmd[p], ww) for p, ww in unit.inputs}
want = rtl.run_trace(unit, packed, seed=1)
_, reason, _ = u.REF(inst, packed)
want_int = {p: rtl.unpack(want[p]) for p in want}

# ---- the driver ----
win = sum(rtl.nwords(wd) for _, _, wd in INS)
wout = sum(rtl.nwords(wd) for _, _, wd in OUTS)
ni, no = len(INS), len(OUTS)
drive, off = [], 0
for i, (_, p, wd) in enumerate(INS):
    drive.append(f"    dut.{p}_vld = k[{i}] < n;")
    drive.append(f"    if (k[{i}] < n) setp(dut.{p}_dat, &in[k[{i}] * {win} + {off}]); else setp(dut.{p}_dat, zeros);")
    off += rtl.nwords(wd)
fire = "\n".join(f"    fire[{i}] = dut.{p}_vld && dut.{p}_rdy;" for i, (_, p, _) in enumerate(INS))
ordy = "\n".join(f"    dut.{p}_rdy = 1;" for _, p, _ in OUTS)
oidle = "\n".join(f"  dut.{p}_rdy = 0;" for _, p, _ in OUTS)
osample, off = [], 0
for i, (_, p, wd) in enumerate(OUTS):
    osample.append(f"    ofire[{i}] = dut.{p}_vld; getp(dut.{p}_dat, &odat[{off}]);")
    off += rtl.nwords(wd)
rowlen = 2 + wout  # in-fire mask, out-fire mask, output words
driver = f"""// generated by hs_cmp.py: Connections-port driver, per-cycle record
#include "Vtop.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
{rtl._TRACE_HELPERS}
int main(int argc, char** argv) {{
  VerilatedContext* ctx = Verilated::threadContextp();
  ctx->commandArgs(argc, argv);
  const uint64_t stall_limit = std::atoll(argv[3]);
  FILE* fi = std::fopen(argv[1], "rb");
  std::fseek(fi, 0, SEEK_END);
  const uint64_t n = std::ftell(fi) / 8 / {win};
  std::fseek(fi, 0, SEEK_SET);
  std::vector<uint64_t> in(n * {win});
  static const uint64_t zeros[64] = {{0}};
  if (std::fread(in.data(), 8, n * {win}, fi) != n * {win}) return 2;
  std::fclose(fi);
  std::vector<uint64_t> rec;
  Vtop dut;
  uint64_t k[{ni}] = {{0}}, j[{no}] = {{0}};
  bool fire[{ni}], ofire[{no}];
  uint64_t odat[{wout}];
  // reset: rst low 4 cycles, every valid low, outputs not ready, then 2 idle cycles high
  dut.clk = 0; dut.rst = 0;
  for (int i = 0; i < {ni}; ++i) k[i] = n;
{chr(10).join(drive)}
{oidle}
  for (int r = 0; r < 4; ++r) {{ dut.clk = 0; dut.eval(); dut.clk = 1; dut.eval(); }}
  dut.rst = 1;
  for (int r = 0; r < 2; ++r) {{ dut.clk = 0; dut.eval(); dut.clk = 1; dut.eval(); }}
  for (int i = 0; i < {ni}; ++i) k[i] = 0;
  uint64_t cyc = 0, last = 0;
  bool all_done = false;
  while (!all_done) {{
    dut.clk = 0;
{chr(10).join(drive)}
{ordy}
    dut.eval();
{fire}
{chr(10).join(osample)}
    dut.clk = 1;
    dut.eval();
    uint64_t im = 0, om = 0;
    for (int i = 0; i < {ni}; ++i) if (fire[i]) {{ im |= 1ull << i; ++k[i]; last = cyc; }}
    for (int i = 0; i < {no}; ++i) if (ofire[i]) {{ om |= 1ull << i; ++j[i]; last = cyc; }}
    rec.push_back(im); rec.push_back(om);
    for (int i = 0; i < {wout}; ++i) rec.push_back(odat[i]);
    ++cyc;
    all_done = true;
    for (int i = 0; i < {no}; ++i) if (j[i] < n) all_done = false;
    if (cyc - last > stall_limit) {{
      std::printf("STALL cycles=%llu accepted", (unsigned long long)cyc);
      for (int i = 0; i < {ni}; ++i) std::printf(" %llu", (unsigned long long)k[i]);
      std::printf(" delivered");
      for (int i = 0; i < {no}; ++i) std::printf(" %llu", (unsigned long long)j[i]);
      std::printf("\\n");
      break;
    }}
  }}
  std::printf("HS cycles=%llu\\n", (unsigned long long)cyc);
  FILE* fo = std::fopen(argv[2], "wb");
  std::fwrite(rec.data(), 8, rec.size(), fo);
  std::fclose(fo);
  return 0;
}}
"""
h = hashlib.sha256((driver + open(src, "rb").read().decode(errors="replace")).encode()).hexdigest()[:12]
bd = os.path.join(a.scratch, f"{os.path.basename(prj).replace('.prj', '')}-{h}")
exe = os.path.join(bd, "Vtop")
if not os.path.exists(exe):
    os.makedirs(bd, exist_ok=True)
    open(os.path.join(bd, "driver.cpp"), "w").write(driver)
    cmdline = [shutil.which("verilator") or "verilator", "--cc", "--exe", "--build", "-Wno-fatal", "-O3",
               "--top-module", "top", "--Mdir", bd, "-j", "8", src, os.path.join(bd, "driver.cpp"),
               "-CFLAGS", "-std=c++17 -O2"]
    r = subprocess.run(cmdline, cwd=bd, env=dict(os.environ, CXX=rtl._cxx()), capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(exe):
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit(1)

cols = [rtl.pack(list(cmd[p]), wd) for p, _, wd in INS]
stim = np.ascontiguousarray(np.concatenate(cols, axis=1), dtype=np.uint64)
fi, fo = os.path.join(bd, f"in.{os.getpid()}"), os.path.join(bd, f"out.{os.getpid()}")
stim.tofile(fi)
t0 = time.time()
r = subprocess.run([exe, fi, fo, str(a.stall)], capture_output=True, text=True)
rec = np.fromfile(fo, dtype=np.uint64).reshape(-1, rowlen)
for p in (fi, fo):
    os.remove(p)
status = r.stdout.strip().splitlines()
ncyc = len(rec)
name = os.path.basename(prj).replace(".prj", "")

# ---- transfers ----
in_cyc = {p: np.flatnonzero((rec[:, 0] >> i) & 1) for i, (p, _, _) in enumerate(INS)}
out_cyc, out_val = {}, {}
off = 2
for i, (p, _, wd) in enumerate(OUTS):
    nw = rtl.nwords(wd)
    c = np.flatnonzero((rec[:, 1] >> i) & 1)
    out_cyc[p] = c
    vals = rtl.unpack(rec[c, off: off + nw])
    if wd % 64:
        vals = [v & ((1 << wd) - 1) for v in vals]
    out_val[p] = vals
    off += nw

print(f"HS-RTL vpu_fifo:{inst} {name} trace={a.trace}: {n} iterations, {ncyc} cycles, "
      f"{'; '.join(status)} ({time.time() - t0:.1f}s)", flush=True)
if a.dump:
    print("  cycle | in fires (rst push pd pop) | out fires (pd e f) | values")
    for c in range(ncyc):
        im, om = int(rec[c, 0]), int(rec[c, 1])
        print(f"  {c:5d} | {' '.join(str((im >> i) & 1) for i in range(ni))} | {' '.join(str((om >> i) & 1) for i in range(no))} | "
              + " ".join(f"{p[:-2]}={int(rec[c, 2 + sum(rtl.nwords(wd2) for _, _, wd2 in OUTS[:i])]):x}" for i, (p, _, wd) in enumerate(OUTS)))


def offsets(cyc):
    o = collections.Counter((int(c) - j) for j, c in enumerate(cyc))
    return ", ".join(f"{k}:{v}" for k, v in sorted(o.items())[:6]) + (" ..." if len(o) > 6 else "")


for p, _, _ in INS:
    print(f"    in  {p:12s} accepted {len(in_cyc[p])}/{n}; cycle - iteration: {offsets(in_cyc[p])}")
for p, _, _ in OUTS:
    print(f"    out {p:12s} delivered {len(out_cyc[p])}/{n}; cycle - iteration: {offsets(out_cyc[p])}")

# ---- values vs MiniTPU ----
got = {}
for p, _, _ in OUTS:
    v = list(out_val[p]) + [0] * (n - len(out_val[p]))
    got[p] = v[:n]
if all(len(out_cyc[p]) == n for p, _, _ in OUTS):
    cmp_composed.compare(got, want_int, reason, cmd, spans, f"vpu_fifo:{inst} {name} catapult-rtl")
else:
    print("    (incomplete delivery: no value verdict; see STALL)")

# ---- push-accept -> pop-deliver latency, pops one trace cycle after their push ----
push = np.asarray(cmd["push_i"]) == 1
pop = np.asarray(cmd["pop_i"]) == 1
rstn = np.asarray(cmd["rst_ni"]) == 1
pending, pairs = [], []
for t in range(n):
    if not rstn[t]:
        pending = []
        continue
    if pop[t] and pending:
        it = pending.pop(0)
        pairs.append((it, t))
    if push[t]:
        pending.append(t)
pd_acc = in_cyc["push_data_i"]
qd_out = out_cyc["pop_data_o"]
lat = collections.Counter()
for it, t in pairs:
    if t == it + 1 and it < len(pd_acc) and t < len(qd_out):
        lat[int(qd_out[t]) - int(pd_acc[it])] += 1
print(f"    push-accept -> pop-deliver cycles, pops one trace cycle after their push: "
      + ", ".join(f"{k}:{v}" for k, v in sorted(lat.items())) + f" ({len(pairs)} push/pop pairs)")
