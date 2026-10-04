"""U3 ``xlu_reduction_tree`` (plan T1 ``bits``) in RTLGen's frontend
(kkkaishao/allo allo-rtlgen @ 13b55a63), generated as text: the N - 1 node
bodies (``bf16_add`` bits from u1_bf16_add_rtlgen/bits_kernel.py, or the
``bf16_gt`` max select, on the wavefront's op) and the ``1 + 2*LEVELS`` /
``1 + 2*LANE_LEVELS`` register pipes as explicit statements (a ``for`` is time
in RTLGen, U1 finding F4). Lane-array ports (P-8): ``D: u16[N, NL]``,
``LRO: u16[N, SUB]``. Valid pipes reset by ``rst_ni`` (synchronous, as the
.sv), payload pipes never (P-9).

    python tree_rtlgen.py <oracles/tree_<inst>.npz> <inst: n16|n64> <N cycles> [mode]

mode: ``call`` (one nested @kernel ``node_op(o, a, b) -> i32`` called per node;
the natural form) or ``inline`` (the body textually repeated per node).
"""
import sys, time, os, re
import numpy as np

path, inst, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else "call"
freq = float(sys.argv[5]) if len(sys.argv) > 5 else 300.0
LANES, SUB = {"n16": (4, 4), "n64": (16, 4)}[inst]
NL = LANES * SUB
LEVELS = NL.bit_length() - 1
TAP = LANES.bit_length() - 1
ROOT_D, TAP_D = 1 + 2 * LEVELS, 1 + 2 * TAP
TAP_BASE = 2 * NL - 2 * SUB

ADD_BODY = '''
        sa: i32 = (a >> 15) & 1
        sb: i32 = (b >> 15) & 1
        ea: i32 = (a >> 7) & 255
        eb: i32 = (b >> 7) & 255
        fa: i32 = a & 127
        fb: i32 = b & 127
        ha: i32 = H16 if ea != 0 else ZERO
        hb: i32 = H16 if eb != 0 else ZERO
        ma: i32 = (fa << 9) | ha
        mb: i32 = (fb << 9) | hb
        a_large: i32 = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1: i32 = ONE if ea == 0 else ea
        eb1: i32 = ONE if eb == 0 else eb
        sl: i32 = sa if a_large == 1 else sb
        el: i32 = ea1 if a_large == 1 else eb1
        es: i32 = eb1 if a_large == 1 else ea1
        ml: i32 = ma if a_large == 1 else mb
        ms: i32 = mb if a_large == 1 else ma
        ed: i32 = el - es
        sh: i32 = TEN if ed >= 10 else ed
        sm: i32 = ms >> sh
        mag: i32 = (ml + sm) if sa == sb else (ml - sm)
        er: i32 = el
        res: i32 = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka: i32 = a ^ S15
        kb: i32 = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt: i32 = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
            res = 0x7FC0
        elif ea == 255:
            res = (sa << 15) | 0x7F80
        elif eb == 255:
            res = (sb << 15) | 0x7F80
        elif (a & 0x7FFF) == 0:
            res = b
        elif (b & 0x7FFF) == 0:
            res = a
        elif ed >= 10:
            res = a if a_large == 1 else b
        elif mag == 0:
            res = 0
        else:
            if mag >= 131072:
                mag = (mag | ((mag & 1) << 1)) >> 1
                er = er + 1
            elif mag < 65536:
                x: i32 = mag
                lz: i32 = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx: i32 = 16 if er > 17 else er - 1
                ns: i32 = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g: i32 = (mag >> 8) & 1
            rb: i32 = (mag >> 7) & 1
            s: i32 = ONE if (mag & 127) != 0 else ZERO
            ru: i32 = g & (rb | s | ((mag >> 9) & 1))
            rnd: i32 = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
'''
CONSTS = "        H16: i32 = 65536\n        ONE: i32 = 1\n        ZERO: i32 = 0\n        TEN: i32 = 10\n        K16: i32 = 65535\n        S15: i32 = 32768\n"

lines = ["from allo import kernel", "from allo.lang import i32, u8, u16", f"N = {N}", ""]
if mode == "call":
    lines += ["@kernel", "def node_op(o: i32, a: i32, b: i32) -> i32:"]
    lines += ["    if True:", CONSTS.rstrip("\n")]
    lines += [ln for ln in ADD_BODY.splitlines() if ln.strip()]
    lines += ["        return res", ""]
lines += ["@kernel",
          f"def tree(RST: u8[N], VLD: u8[N], OP: u8[N], D: u16[N, {NL}], VO: u8[N], RO: u16[N], LVO: u8[N], LRO: u16[N, {SUB}]):"]
lines += [f"    vq{k}: i32 = 0" for k in range(ROOT_D)]
lines += [f"    rq{k}: i32 = 0" for k in range(ROOT_D)]
lines += [f"    lvq{k}: i32 = 0" for k in range(TAP_D)]
lines += [f"    lrq{k}_{s}: i32 = 0" for k in range(TAP_D) for s in range(SUB)]
lines += ['    for t in range(N, name="t"):', "        r: i32 = RST[t]", "        v: i32 = VLD[t]", "        o: i32 = OP[t]"]
if mode != "call":
    lines += [CONSTS.rstrip("\n")]
    decl = sorted(set(re.findall(r"^\s+([a-z_0-9]+): i32 =", ADD_BODY, re.M)))
    lines += [f"        {nm}: i32 = 0" for nm in decl]
    lines += ["        a: i32 = 0", "        b: i32 = 0"]
lines += [f"        n{l}: i32 = D[t, {l}]" for l in range(NL)]
for m in range(NL - 1):
    a, b, out = 2 * m, 2 * m + 1, NL + m
    if mode == "call":
        lines += [f"        n{out}: i32 = node_op(o, n{a}, n{b})"]
    else:
        lines += [f"        a = n{a}", f"        b = n{b}"]
        # every local is declared once above the copies; the copies assign
        body = "\n".join((ln.split(":")[0] + " =" + ln.split("=", 1)[1]) if (": i32 =" in ln) else ln for ln in ADD_BODY.splitlines())
        lines += [ln for ln in body.splitlines() if ln.strip()]
        lines += [f"        n{out}: i32 = res"]
# register edges: stage k takes k - 1, stage 0 the tree
for k in range(ROOT_D - 1, 0, -1):
    lines += [f"        vq{k} = vq{k - 1}", f"        rq{k} = rq{k - 1}"]
lines += ["        vq0 = v", f"        rq0 = n{2 * NL - 2}"]
for k in range(TAP_D - 1, 0, -1):
    lines += [f"        lvq{k} = lvq{k - 1}"] + [f"        lrq{k}_{s} = lrq{k - 1}_{s}" for s in range(SUB)]
lines += ["        lvq0 = v"] + [f"        lrq0_{s} = n{TAP_BASE + s}" for s in range(SUB)]
lines += ["        if r == 0:"] + [f"            vq{k} = 0" for k in range(ROOT_D)] + [f"            lvq{k} = 0" for k in range(TAP_D)]
lines += [f"        LRO[t, {s}] = lrq{TAP_D - 1}_{s}" for s in range(SUB)]
lines += [f"        VO[t] = vq{ROOT_D - 1}", f"        RO[t] = rq{ROOT_D - 1}", f"        LVO[t] = lvq{TAP_D - 1}"]
src = "\n".join(lines) + "\n"
kp = f"tree_{inst}_{mode}_kernel.py"
open(kp, "w").write(src)
print(f"GENERATED {kp}: {len(lines)} lines", flush=True)
import importlib.util
spec = importlib.util.spec_from_file_location(f"tree_{inst}_{mode}_kernel", os.path.abspath(kp))
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)

d = np.load(path)
n = len(d["rst"]); assert N <= n
t0 = time.time()
sch = K.tree.schedule()
rtl = sch.export("rtl", freq_mhz=freq)
print(f"EXPORTED {inst} {mode} N={N} in {time.time() - t0:.1f}s", flush=True)
res = rtl.schedule(); f = res.func("tree")
print("SCHED latency:", f.latency, "| II:", [r.interval for r in res.cyclic()])
open(f"tree_{inst}_{mode}_N{N}.sv", "w").write(rtl.verilog)
q = rtl.estimation
print("QOR: lat", q.latency, "fmax %.1f" % q.fmax, "area", q.area)
print("CRIT:", str(q.critical_paths[0])[:400] if q.critical_paths else None)
ins = [d["rst"][:N].astype(np.uint8), d["vld"][:N].astype(np.uint8), d["op"][:N].astype(np.uint8),
       np.ascontiguousarray(d["data"][:N]).astype(np.uint16)]
outs = [np.zeros(N, np.uint8), np.zeros(N, np.uint16), np.zeros(N, np.uint8), np.zeros((N, SUB), np.uint16)]
t0 = time.time()
cyc = rtl.cosim(*ins, *outs, timeout=int(os.environ.get("COSIM_TIMEOUT", 20 * N + 5000))).cycles
print(f"COSIM {inst} {mode} N={N} cycles={cyc} ({time.time() - t0:.1f}s)", flush=True)
np.savez(f"tree_got_{inst}_{mode}_N{N}.npz", valid_o=outs[0], result_o=outs[1], lane_valid_o=outs[2], lane_result_o=outs[3])
tot = bad = 0; ex = []
for p, g in zip(("valid_o", "result_o", "lane_valid_o", "lane_result_o"), outs):
    df = d["def_" + p][:N]; w = d["want_" + p][:N]
    diff = df & (g.reshape(N, -1) != w.reshape(N, -1)).any(axis=1)
    tot += int(df.sum()); bad += int(diff.sum())
    ex += [f"{p} cycle {i}: got {g.reshape(N, -1)[i]} rtl {w.reshape(N, -1)[i]}" for i in np.flatnonzero(diff)[:2]]
print(f"{'MATCH' if bad == 0 else 'DIFF '} {inst} {mode} N={N}: {tot - bad}/{tot} defined", flush=True)
for e in ex:
    print("    e.g.", e)
