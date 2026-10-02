"""vpu_word_array (U2, narrow: 8 x 64 b, latencies 3/2) in RTLGen's frontend
(kkkaishao/allo allo-rtlgen), the ``trace`` variant (pipe as data).

    python wa_rtlgen.py <trace.npz> <variant> <N>

``trace``: kernel-local ``mem: u64[8]``, read pipes ``pc: u64[3]``,
``pd: u64[2]``; per iteration the pipes shift, each enabled port reads (old
word) or writes, ``q[t] = pipe[L - 1]``. ``trace_part``: the same with
``mem`` completely partitioned. ``trace_rw``: the simulation model (every
enabled port reads *and* writes). Port arrays: ``u8`` en/we/addr, ``u64``
data (RTLGen has no u1/u3).
"""
import sys, time
import numpy as np
from allo import kernel
from allo.lang import i32, u8, u64

path, variant, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = dict(np.load(path))

if variant.startswith("trace_rw"):
    @kernel
    def wa(CE: u8[N], CW: u8[N], CA: u8[N], CD: u64[N], DE: u8[N], DW: u8[N], DA: u8[N], DD: u64[N],
           QC: u64[N], QD: u64[N]):
        mem: u64[8]
        pc: u64[3]
        pd: u64[2]
        for t in range(N, name="t"):
            pc[2] = pc[1]
            pc[1] = pc[0]
            pd[1] = pd[0]
            ac: i32 = CA[t]
            ad: i32 = DA[t]
            dc: u64 = CD[t]
            dd: u64 = DD[t]
            if CE[t] != 0:
                pc[0] = mem[ac]
            if DE[t] != 0:
                pd[0] = mem[ad]
            if CE[t] != 0 and CW[t] != 0:
                mem[ac] = dc
            if DE[t] != 0 and DW[t] != 0:
                mem[ad] = dd
            QC[t] = pc[2]
            QD[t] = pd[1]
else:
    @kernel
    def wa(CE: u8[N], CW: u8[N], CA: u8[N], CD: u64[N], DE: u8[N], DW: u8[N], DA: u8[N], DD: u64[N],
           QC: u64[N], QD: u64[N]):
        mem: u64[8]
        pc: u64[3]
        pd: u64[2]
        for t in range(N, name="t"):
            pc[2] = pc[1]
            pc[1] = pc[0]
            pd[1] = pd[0]
            ac: i32 = CA[t]
            ad: i32 = DA[t]
            dc: u64 = CD[t]
            dd: u64 = DD[t]
            if CE[t] != 0 and CW[t] == 0:
                pc[0] = mem[ac]
            if DE[t] != 0 and DW[t] == 0:
                pd[0] = mem[ad]
            if CE[t] != 0 and CW[t] != 0:
                mem[ac] = dc
            if DE[t] != 0 and DW[t] != 0:
                mem[ad] = dd
            QC[t] = pc[2]
            QD[t] = pd[1]

t0 = time.time()
sch = wa.schedule()
if variant.endswith("_part"):
    sch.partition("mem", dim=1, kind=sch.Complete)
m = sch.export("rtl")
print(f"EXPORTED {variant} N={N} in {time.time() - t0:.1f}s", flush=True)
open(f"wa_{variant}_{N}.sv", "w").write(m.sv if hasattr(m, "sv") else str(getattr(m, "verilog", "")))
ins = [d[k][:N].astype(np.uint8) for k in ("ce", "cw", "ca")] + [d["cd"][:N].astype(np.uint64)]
ins += [d[k][:N].astype(np.uint8) for k in ("de", "dw", "da")] + [d["dd"][:N].astype(np.uint64)]
q = [np.zeros(N, np.uint64) for _ in range(2)]
cyc = m.cosim(*ins, *q, timeout=4 * N + 1000)
np.savez(f"got_{variant}_{N}.npz", qc=q[0], qd=q[1])
print(f"COSIM {variant} N={N} cycles={cyc} {time.time() - t0:.1f}s", flush=True)
