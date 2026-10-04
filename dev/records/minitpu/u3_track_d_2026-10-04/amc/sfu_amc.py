"""U3 ``sfu`` (plan S1 ``bits``) through AMC's vendored Allo frontend
(cornell-zhang/amc-dialect @ fe60c121). ``sfu.sv`` on int32 locals with
masks, helpers inlined (AMC A4: a scalar-returning call aborts the backend),
two-operand booleans only (A5), no bit indexing (A2). The four tables are
module-level numpy arrays bound in the kernel (``rom: int32[2048] = ROM``),
the frontend's constant-array form (``build_constant_tensor`` ->
``memref.global``), which AMC's ``rom-allocation`` pass should turn into a ROM.

    python sfu_amc.py <oracles/sfu.npz> <N> <targets> <sched> [limit]

targets: comma list of llvm, amc; sched: none | pipeline (the element loop).
"""
import os, sys, time, traceback, importlib.util
import numpy as np

path, N, targets, sched = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4]
sel = sys.argv[5] if len(sys.argv) > 5 else "all"
limit = None if sel == "all" else (slice(*[int(v) if v else None for v in sel.split(":")]) if ":" in sel else slice(0, int(sel)))
d = np.load(path)
TMP = os.environ.get("TMPDIR", "/tmp")
np.save(f"{TMP}/u3d_sfu_tables.npy", np.concatenate([d["gelu"], d["exp"], d["recip"], d["rsqrt"]]).astype(np.int32))
src = f'''
import numpy as np
from allo.ir.types import int32, uint8, uint16
N = {N}
_T = np.load("{TMP}/u3d_sfu_tables.npy")
GELU_ROM = np.ascontiguousarray(_T[0:2048]).astype(np.int32)
EXP_ROM = np.ascontiguousarray(_T[2048:4096]).astype(np.int32)
RECIP_PWL = np.ascontiguousarray(_T[4096:4128]).astype(np.int32)
RSQRT_PWL = np.ascontiguousarray(_T[4128:4160]).astype(np.int32)

def sfu(opv: uint8[N], xv: uint16[N], rv: uint16[N]):
    rom_gelu: int32[2048] = GELU_ROM
    rom_exp: int32[2048] = EXP_ROM
    recip_pwl: int32[32] = RECIP_PWL
    rsqrt_pwl: int32[32] = RSQRT_PWL
    for i in range(N):
        o: int32 = opv[i]
        x: int32 = xv[i]
        sign: int32 = (x >> 15) & 1
        e: int32 = (x >> 7) & 0xFF
        frac: int32 = x & 0x7F
        mant: int32 = (0x80 | frac) << 16
        rs: int32 = 16 - (e - 127)
        mag: int32 = 0
        if e == 0 or rs >= 24:
            mag = 0
        elif rs <= 0:
            mag = 0x1FFF
        else:
            v: int32 = mant >> rs
            mag = 0x1FFF if v > 0x1FFF else v
        ga: int32 = 0
        if sign != 0:
            ga = (1024 - 1 - mag) & 0x7FF
        else:
            ga = (1024 + mag) & 0x7FF
        ea: int32 = (2048 - 1 - mag) & 0x7FF
        ra: int32 = (x >> 2) & 0x1F
        sa: int32 = ((((x >> 7) & 1) ^ 1) << 4) | ((x >> 3) & 0xF)
        recip_pos: int32 = (x & 3) << 10
        rsqrt_pos: int32 = (x & 7) << 9
        gw: int32 = rom_gelu[ga]
        ew: int32 = rom_exp[ea]
        rw: int32 = recip_pwl[ra]
        sw: int32 = rsqrt_pwl[sa]
        r_bias: int32 = (((rw >> 8) & 0x1FFF) + 8321) & 0x3FFF
        s_bias: int32 = (((sw >> 8) & 0x1FFF) + 8322) & 0x3FFF
        r_prod: int32 = (rw & 0xFF) * recip_pos
        s_prod: int32 = (sw & 0xFF) * rsqrt_pos
        s_exp: int32 = 0
        if (e & 1) != 0:
            s_exp = (((379 - e) & 0x1FF) >> 1) & 0xFF
        else:
            s_exp = (((380 - e) & 0x1FF) >> 1) & 0xFF
        r_int: int32 = ((r_bias - ((r_prod >> 11) & 0x1FF)) & 0x3FFF) & 0x1FFF
        s_int: int32 = ((s_bias - ((s_prod >> 11) & 0x1FF)) & 0x3FFF) & 0x1FFF
        is_nan: int32 = 1 if (e == 0xFF and frac != 0) else 0
        is_inf: int32 = 1 if (e == 0xFF and frac == 0) else 0
        is_zero: int32 = 1 if (x & 0x7FFF) == 0 else 0
        res: int32 = 0x7FC0
        if o == 0:
            if is_nan != 0:
                res = 0x7FC0
            elif is_zero != 0:
                res = 0
            elif mag >= 1024:
                res = 0 if sign == 1 else x
            else:
                res = gw
        elif o == 1:
            if is_nan != 0:
                res = 0x7FC0
            elif is_zero == 1 or sign == 0:
                res = 0x3F80
            elif mag >= 2048:
                res = 0
            else:
                res = ew
        elif o == 2:
            if is_nan != 0:
                res = 0x7FC0
            elif is_inf != 0:
                res = sign << 15
            elif is_zero != 0:
                res = (sign << 15) | 0x7F80
            else:
                res = (sign << 15) | (((253 - e) & 0xFF) << 7) | ((r_int >> 6) & 0x7F)
        else:
            if is_nan == 1 or (sign == 1 or is_zero == 1):
                res = 0x7FC0
            elif is_inf != 0:
                res = 0
            else:
                res = (s_exp << 7) | ((s_int >> 6) & 0x7F)
        rv[i] = res
'''
kp = f"{TMP}/u3d_sfu_amc_kernel_{N}.py"
open(kp, "w").write(src)
spec = importlib.util.spec_from_file_location(f"u3d_sfu_amc_kernel_{N}", kp)
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
import allo

op, x, want = d["op"], d["x"], d["want"]
if limit:
    op, x, want = op[limit], x[limit], want[limit]
idx = np.arange(len(d["op"]))[limit] if limit else np.arange(len(d["op"]))
m = len(op); pad = (-m) % N
op = np.concatenate([op, np.zeros(pad, np.uint8)]); x = np.concatenate([x, np.zeros(pad, np.uint16)])
for tgt in targets:
    t0 = time.time()
    try:
        s = allo.customize(K.sfu)
        if sched == "pipeline":
            loops = s.get_loops()
            s.pipeline(loops["S_i_0"]["i"])
        f = s.build(target=tgt)
        print(f"BUILT {tgt} {sched} N={N} {time.time() - t0:.1f}s", flush=True)
        if tgt == "amc":
            if hasattr(f, "dump_allocation"):
                f.dump_allocation(f"sfu_amc_{sched}_{N}.alloc.txt")
            if hasattr(f, "dump_schedule"):
                f.dump_schedule(f"sfu_amc_{sched}_{N}.loopschedule.mlir")
            if hasattr(f, "dump_verilog"):
                print("dump_verilog ->", [p.name for p in f.dump_verilog(f"sfu_amc_{sched}_{N}_hdl")], flush=True)
        got = np.zeros(len(op), np.uint16); cyc = []; t1 = time.time()
        for c in range(len(op) // N):
            r = np.zeros(N, np.uint16)
            f(np.ascontiguousarray(op[c * N:(c + 1) * N]), np.ascontiguousarray(x[c * N:(c + 1) * N]), r)
            got[c * N:(c + 1) * N] = r
            if tgt == "amc":
                cyc.append(int(f.rpt["cycles"]))
        got = got[:m]
        mm = np.nonzero(got != want)[0]
        print(f"RAN {tgt} {sched} N={N} chunks={len(op) // N} cycles/chunk={sorted(set(cyc))} "
              f"match={m - len(mm)}/{m} ({time.time() - t1:.1f}s)", flush=True)
        for i in mm[:8]:
            print("   op", int(op[i]), "x", hex(int(x[i])), "got", hex(int(got[i])), "want", hex(int(want[i])))
        np.save(f"sfu_got_{tgt}_{sched}_{N}.npy", got)
    except Exception as e:
        print(f"FAIL {tgt} {sched} N={N}: {type(e).__name__}: {str(e)[:800]}", flush=True)
        traceback.print_exc(limit=4)
