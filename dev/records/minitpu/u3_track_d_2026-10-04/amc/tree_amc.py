"""U3 ``xlu_reduction_tree`` T1 ``bits`` through AMC's vendored Allo frontend,
generated as text from the RTLGen node body (tree_rtlgen.py ADD_BODY: bf16
add bits + bf16_gt max select on int32 masks), every node inlined, the
register pipes as 1-element arrays read once at the top and written once at
the bottom of the iteration (the PE's AMC form). Lane ports: ``d: uint16[N, NL]``,
``lro: uint16[N, SUB]``.

    python tree_amc.py <oracles/tree_<inst>.npz> <inst> <N> <targets> <sched: none|pipeline>
"""
import os, sys, re, time, importlib.util, traceback
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rtlgen"))
path, inst, N, targets, sched = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(","), sys.argv[5]
TMP = os.environ.get("TMPDIR", "/tmp")
LANES, SUB = {"n4": (2, 2), "n8": (2, 4), "n16": (4, 4), "n64": (16, 4)}[inst]
NL = LANES * SUB; LEVELS = NL.bit_length() - 1; TAP = LANES.bit_length() - 1
ROOT_D, TAP_D = 1 + 2 * LEVELS, 1 + 2 * TAP; TAP_BASE = 2 * NL - 2 * SUB
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rtlgen", "tree_rtlgen.py")).read()
ADD_BODY = src[src.index("ADD_BODY = '''") + len("ADD_BODY = '''"):]; ADD_BODY = ADD_BODY[:ADD_BODY.index("'''")]
body = ADD_BODY.replace(": i32 =", ": int32 =").replace("H16", "65536").replace("ZERO", "0").replace("ONE", "1").replace("TEN", "10").replace("K16", "65535").replace("S15", "32768")
# the 3-way or of the add's NaN test: AMC folds only two operands (A5)
body = body.replace("elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):",
                    "elif ((ea == 255 and fa != 0) or (eb == 255 and fb != 0)) or ((ea == 255 and eb == 255) and sa != sb):")
decl = sorted(set(re.findall(r"^\s+([a-z_0-9]+): int32 =", body, re.M)))
if os.environ.get("TREE_TRIVIAL"):  # bisection: the skeleton with a one-line node
    body = "        res: int32 = (a + b) & 0xFFFF if o == 0 else (a if a > b else b)\n"
    decl = ["res"]
assign = "\n".join((ln.split(":")[0] + " =" + ln.split("=", 1)[1]) if (": int32 =" in ln) else ln for ln in body.splitlines())
L = ["from allo.ir.types import int32, uint8, uint16", f"N = {N}", ""]
L += [f"def tree(rst: uint8[N], vld: uint8[N], op: uint8[N], d: uint16[N, {NL}], vo: uint8[N], ro: uint16[N], lvo: uint8[N], lro: uint16[N, {SUB}]):"]
pipes = [f"vq{k}" for k in range(ROOT_D)] + [f"rq{k}" for k in range(ROOT_D)] + [f"lvq{k}" for k in range(TAP_D)] + [f"lrq{k}_{s}" for k in range(TAP_D) for s in range(SUB)]
L += [f"    {p}_r: int32[1] = 0" for p in pipes]
L += ["    for t in range(N):"]
B = [f"{p}: int32 = {p}_r[0]" for p in pipes]
B += ["r: int32 = rst[t]", "v: int32 = vld[t]", "o: int32 = op[t]"]
B += [f"{nm}: int32 = 0" for nm in decl] + ["a: int32 = 0", "b: int32 = 0"]
B += [f"n{l}: int32 = d[t, {l}]" for l in range(NL)]
for m in range(NL - 1):
    B += [f"a = n{2 * m}", f"b = n{2 * m + 1}"]
    B += [ln[8:] for ln in assign.splitlines() if ln.strip()]  # body carries an 8-space base; 8 added below
    B += [f"n{NL + m}: int32 = res"]
for k in range(ROOT_D - 1, 0, -1):
    B += [f"vq{k} = vq{k - 1}", f"rq{k} = rq{k - 1}"]
B += ["vq0 = v", f"rq0 = n{2 * NL - 2}"]
for k in range(TAP_D - 1, 0, -1):
    B += [f"lvq{k} = lvq{k - 1}"] + [f"lrq{k}_{s} = lrq{k - 1}_{s}" for s in range(SUB)]
B += ["lvq0 = v"] + [f"lrq0_{s} = n{TAP_BASE + s}" for s in range(SUB)]
B += ["if r == 0:"] + [f"    vq{k} = 0" for k in range(ROOT_D)] + [f"    lvq{k} = 0" for k in range(TAP_D)]
B += [f"lro[t, {s}] = lrq{TAP_D - 1}_{s}" for s in range(SUB)]
B += [f"vo[t] = vq{ROOT_D - 1}", f"ro[t] = rq{ROOT_D - 1}", f"lvo[t] = lvq{TAP_D - 1}"]
B += [f"{p}_r[0] = {p}" for p in pipes]
L += ["        " + ln for ln in B]
kp = f"{TMP}/tree_amc_kernel_{inst}_{N}.py"; open(kp, "w").write("\n".join(L) + "\n")
open(f"tree_amc_kernel_{inst}_{N}.py", "w").write("\n".join(L) + "\n")
print(f"GENERATED {kp}: {len(L)} lines", flush=True)
spec = importlib.util.spec_from_file_location(f"tree_amc_kernel_{inst}_{N}", kp); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
import allo
dd = np.load(path)
BUILD_ONLY = dd["data"].shape[1] != NL  # a geometry without an oracle: build and run only
rng = np.random.default_rng(0)
ins = [dd["rst"][:N].astype(np.uint8), dd["vld"][:N].astype(np.uint8), dd["op"][:N].astype(np.uint8),
       np.ascontiguousarray(dd["data"][:N, :NL]).astype(np.uint16) if not BUILD_ONLY else rng.integers(0, 65536, (N, NL)).astype(np.uint16)]
for tgt in targets:
    t0 = time.time()
    try:
        s = allo.customize(K.tree)
        if sched == "pipeline":
            loops = s.get_loops(); s.pipeline(loops["S_t_0"]["t"])
        f = s.build(target=tgt)
        print(f"BUILT {tgt} {sched} {inst} N={N} {time.time() - t0:.1f}s", flush=True)
        if tgt == "amc":
            if hasattr(f, "dump_allocation"): f.dump_allocation(f"tree_amc_{sched}_{inst}_{N}.alloc.txt")
            if hasattr(f, "dump_verilog"): f.dump_verilog(f"tree_amc_{sched}_{inst}_{N}_hdl")
        outs = [np.zeros(N, np.uint8), np.zeros(N, np.uint16), np.zeros(N, np.uint8), np.zeros((N, SUB), np.uint16)]
        t1 = time.time(); f(*ins, *outs)
        cyc = int(f.rpt["cycles"]) if tgt == "amc" else None
        tot = bad = 0; ex = []
        for p, g in zip(("valid_o", "result_o", "lane_valid_o", "lane_result_o"), outs if not BUILD_ONLY else []):
            df = dd["def_" + p][:N]; w = dd["want_" + p][:N]
            diff = df & (g.reshape(N, -1) != w.reshape(N, -1)).any(axis=1)
            tot += int(df.sum()); bad += int(diff.sum())
            ex += [f"{p} cycle {i}: got {g.reshape(N, -1)[i]} rtl {w.reshape(N, -1)[i]}" for i in np.flatnonzero(diff)[:2]]
        print(f"{'MATCH' if bad == 0 else 'DIFF '} tree {tgt} {sched} {inst} N={N}: {tot - bad}/{tot} defined, cycles={cyc} ({time.time() - t1:.1f}s)", flush=True)
        for e in ex: print("    e.g.", e)
    except Exception as e:
        print(f"FAIL {tgt} {sched} {inst} N={N}: {type(e).__name__}: {str(e)[:600]}", flush=True); traceback.print_exc(limit=3)
