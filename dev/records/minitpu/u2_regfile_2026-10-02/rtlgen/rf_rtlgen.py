"""vpu_regfile (U2) in RTLGen's frontend (kkkaishao/allo allo-rtlgen), plan R1/R2.

    python rf_rtlgen.py <trace.npz> <variant> <N> [chunks]

``trace``: a kernel-local ``mem: u16[32]``, one cosim of N cycles (the
first N cycles of the joined command trace). ``stateful``: ``mem`` is
``Stateful[u16[32]]`` and the trace is fed in N-cycle chunks over
successive cosims of one RTL. Per iteration: three reads, then the write.
Port arrays: ``u8`` addresses and ``we``, ``u16`` data (RTLGen has no u5).
"""
import sys, time
import numpy as np
from allo import kernel
from allo.lang import i32, u8, u16, Stateful

path, variant, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
CH = int(sys.argv[4]) if len(sys.argv) > 4 else 1
d = dict(np.load(path))
for k in ("ra", "rb", "rc", "wa", "wd", "we"):  # pad with idle cycles (we = 0) to whole chunks
    d[k] = np.concatenate([d[k], np.zeros(max(0, N * CH - len(d[k])), d[k].dtype)])

if variant.startswith("trace"):
    @kernel
    def rf(RA: u8[N], RB: u8[N], RC: u8[N], WA: u8[N], WD: u16[N], WE: u8[N],
           QA: u16[N], QB: u16[N], QC: u16[N]):
        mem: u16[32]
        for t in range(N, name="t"):
            a: i32 = RA[t]
            b: i32 = RB[t]
            c: i32 = RC[t]
            QA[t] = mem[a]
            QB[t] = mem[b]
            QC[t] = mem[c]
            x: i32 = WA[t]
            dd: u16 = WD[t]
            if WE[t] != 0:
                mem[x] = dd
else:
    @kernel
    def rf(RA: u8[N], RB: u8[N], RC: u8[N], WA: u8[N], WD: u16[N], WE: u8[N],
           QA: u16[N], QB: u16[N], QC: u16[N]):
        mem: Stateful[u16[32]] = 0
        for t in range(N, name="t"):
            a: i32 = RA[t]
            b: i32 = RB[t]
            c: i32 = RC[t]
            QA[t] = mem[a]
            QB[t] = mem[b]
            QC[t] = mem[c]
            x: i32 = WA[t]
            dd: u16 = WD[t]
            if WE[t] != 0:
                mem[x] = dd

t0 = time.time()
sch = rf.schedule()
if variant.endswith("_part"):
    sch.partition("mem", dim=1, kind=sch.Complete)
m = sch.export("rtl")
print(f"EXPORTED {variant} N={N} in {time.time() - t0:.1f}s", flush=True)
open(f"rf_{variant}_{N}.sv", "w").write(m.sv if hasattr(m, "sv") else str(getattr(m, "verilog", "")))
try:
    from allo.backend.rtl import qor
    print("QOR", qor(m) if callable(qor) else qor, flush=True)
except Exception as e:  # noqa: BLE001
    print("QOR n/a", type(e).__name__, str(e)[:200], flush=True)
outs = {p: [] for p in ("qa", "qb", "qc")}
cyc = []
for c in range(CH):
    sl = slice(c * N, (c + 1) * N)
    ins = [d[k][sl].astype(np.uint8) for k in ("ra", "rb", "rc", "wa")]
    ins += [d["wd"][sl].astype(np.uint16), d["we"][sl].astype(np.uint8)]
    q = [np.zeros(N, np.uint16) for _ in range(3)]
    cyc.append(m.cosim(*ins, *q, timeout=4 * N + 1000))
    for p, o in zip(("qa", "qb", "qc"), q):
        outs[p].append(o)
np.savez(f"got_{variant}_{N}x{CH}.npz", **{p: np.concatenate(v) for p, v in outs.items()})
print(f"COSIM {variant} N={N} chunks={CH} cycles(first 2)={cyc[:2]} {time.time() - t0:.1f}s", flush=True)
