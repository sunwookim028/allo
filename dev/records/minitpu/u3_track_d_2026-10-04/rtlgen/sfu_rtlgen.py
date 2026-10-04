"""U3 ``sfu`` (plan S1 ``bits``) in RTLGen's frontend (kkkaishao/allo
allo-rtlgen @ 13b55a63). ``sfu.sv`` transcribed on i32 locals with masks, as
``examples/minitpu/units/sfu.py::bits`` does; the helpers are inlined (one
``fp32_abs_q7`` per element serves the three addresses). The four tables are
numpy arrays captured by the kernel (``rom: i32[2048] = GELU_ROM``), which is
the frontend's only constant-array form (``_visit_numpy_array_initializer``).

    python sfu_rtlgen.py <oracles/sfu.npz> <N> [limit] [freq_mhz]

Runs RTLGen's cocotb+Verilator cosim in chunks of N over the Phase 0 stimulus
and compares with the oracle (``want`` = ref.sfu = MiniTPU RTL, Phase 0).
"""
import sys, time
import numpy as np
from allo import kernel
from allo.lang import i32, u8, u16

path, N = sys.argv[1], int(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "all" else None
freq = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
d = np.load(path)
GELU_ROM = d["gelu"].astype(np.int32)
EXP_ROM = d["exp"].astype(np.int32)
RECIP_PWL = d["recip"].astype(np.int32)
RSQRT_PWL = d["rsqrt"].astype(np.int32)


@kernel
def sfu(OP: u8[N], X: u16[N], R: u16[N]):
    rom_gelu: i32[2048] = GELU_ROM
    rom_exp: i32[2048] = EXP_ROM
    recip_pwl: i32[32] = RECIP_PWL
    rsqrt_pwl: i32[32] = RSQRT_PWL
    for i in range(N, name="i"):
        SAT: i32 = 0x1FFF
        NANV: i32 = 0x7FC0
        ONE: i32 = 1
        ZERO: i32 = 0
        o: i32 = OP[i]
        x: i32 = X[i]
        sign: i32 = (x >> 15) & 1
        e: i32 = (x >> 7) & 0xFF
        frac: i32 = x & 0x7F
        # fp32_abs_q7 on {x, 16'b0}
        mant: i32 = (0x80 | frac) << 16
        rs: i32 = 16 - (e - 127)
        mag: i32 = 0
        if e == 0 or rs >= 24:
            mag = 0
        elif rs <= 0:
            mag = SAT
        else:
            v: i32 = mant >> rs
            mag = SAT if v > SAT else v
        # stage 1: addresses
        ga: i32 = 0
        if sign != 0:
            ga = (1024 - 1 - mag) & 0x7FF
        else:
            ga = (1024 + mag) & 0x7FF
        ea: i32 = (2048 - 1 - mag) & 0x7FF
        ra: i32 = (x >> 2) & 0x1F
        sa: i32 = ((((x >> 7) & 1) ^ 1) << 4) | ((x >> 3) & 0xF)
        recip_pos: i32 = (x & 3) << 10
        rsqrt_pos: i32 = (x & 7) << 9
        # stage 2: table reads, bias, slope products, rsqrt exponent
        gw: i32 = rom_gelu[ga]
        ew: i32 = rom_exp[ea]
        rw: i32 = recip_pwl[ra]
        sw: i32 = rsqrt_pwl[sa]
        r_bias: i32 = (((rw >> 8) & 0x1FFF) + 8321) & 0x3FFF
        s_bias: i32 = (((sw >> 8) & 0x1FFF) + 8322) & 0x3FFF
        r_prod: i32 = (rw & 0xFF) * recip_pos
        s_prod: i32 = (sw & 0xFF) * rsqrt_pos
        s_exp: i32 = 0
        if (e & 1) != 0:
            s_exp = (((379 - e) & 0x1FF) >> 1) & 0xFF
        else:
            s_exp = (((380 - e) & 0x1FF) >> 1) & 0xFF
        # stage 3: interpolation
        r_int: i32 = ((r_bias - ((r_prod >> 11) & 0x1FF)) & 0x3FFF) & 0x1FFF
        s_int: i32 = ((s_bias - ((s_prod >> 11) & 0x1FF)) & 0x3FFF) & 0x1FFF
        is_nan: i32 = ONE if (e == 0xFF and frac != 0) else ZERO
        is_inf: i32 = ONE if (e == 0xFF and frac == 0) else ZERO
        is_zero: i32 = ONE if (x & 0x7FFF) == 0 else ZERO
        res: i32 = NANV
        if o == 0:
            if is_nan != 0:
                res = NANV
            elif is_zero != 0:
                res = 0
            elif mag >= 1024:
                res = ZERO if sign == 1 else x
            else:
                res = gw
        elif o == 1:
            if is_nan != 0:
                res = NANV
            elif is_zero == 1 or sign == 0:
                res = 0x3F80
            elif mag >= 2048:
                res = 0
            else:
                res = ew
        elif o == 2:
            if is_nan != 0:
                res = NANV
            elif is_inf != 0:
                res = sign << 15
            elif is_zero != 0:
                res = (sign << 15) | 0x7F80
            else:
                res = (sign << 15) | (((253 - e) & 0xFF) << 7) | ((r_int >> 6) & 0x7F)
        else:
            if is_nan == 1 or sign == 1 or is_zero == 1:
                res = NANV
            elif is_inf != 0:
                res = 0
            else:
                res = (s_exp << 7) | ((s_int >> 6) & 0x7F)
        R[i] = res


t0 = time.time()
rtl = sfu.schedule().export("rtl", freq_mhz=freq)
print(f"EXPORTED N={N} freq={freq} in {time.time() - t0:.1f}s", flush=True)
res = rtl.schedule(); f = res.func("sfu")
print("SCHED latency:", f.latency, "| II:", [r.interval for r in res.cyclic()])
sv = rtl.verilog
open(f"sfu_N{N}_f{int(freq)}.sv", "w").write(sv)
q = rtl.estimation
print("QOR: lat", q.latency, "fmax %.1f" % q.fmax, "area", q.area)
print("CRIT:", q.critical_paths[0] if q.critical_paths else None)
try:
    ma = rtl.microarch
    txt = str(ma)
    open(f"sfu_N{N}_f{int(freq)}.microarch.txt", "w").write(txt)
    print("MICROARCH (storage lines):")
    for ln in txt.splitlines():
        if any(k in ln.lower() for k in ("rom", "storage", "mem", "bram", "lutram", "rom_gelu", "rom_exp", "pwl")):
            print("   ", ln[:200])
except Exception as ex:
    print("microarch unavailable:", ex)
op, x, want = d["op"], d["x"], d["want"]
if limit:
    op, x, want = op[:limit], x[:limit], want[:limit]
m = len(op); pad = (-m) % N
op = np.concatenate([op, np.zeros(pad, np.uint8)]); x = np.concatenate([x, np.zeros(pad, np.uint16)])
got = np.zeros(len(op), np.uint16); cyc = []; t0 = time.time()
for c in range(len(op) // N):
    o = np.zeros(N, np.uint16)
    cyc.append(rtl.cosim(op[c * N:(c + 1) * N].copy(), x[c * N:(c + 1) * N].copy(), o, timeout=4 * N + 1000).cycles)
    got[c * N:(c + 1) * N] = o
got = got[:m]
mm = np.nonzero(got != want)[0]
print(f"COSIM N={N} chunks={len(cyc)} cycles/chunk={sorted(set(cyc))} time={time.time() - t0:.1f}s "
      f"match={m - len(mm)}/{m}", flush=True)
for i in mm[:10]:
    print("   op", int(d["op"][i]), "x", hex(int(d["x"][i])), "got", hex(int(got[i])), "want", hex(int(want[i])))
np.save(f"sfu_got_N{N}.npy", got)
