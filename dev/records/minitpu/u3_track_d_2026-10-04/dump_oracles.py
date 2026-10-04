"""Allo-env side of U3 track D: dump the Phase 0 stimulus and oracles for the
open-HLS tools (which run in their own envs and read only .npz).

    $ALLO_PYTHON dump_oracles.py <outdir> [sfu] [n16] [n64]

sfu.npz:  op uint8[m], x uint16[m], want uint16[m] (ref.sfu == RTL, Phase 0),
          gelu/exp int32[2048], recip/rsqrt int32[32] (the tables, from the clone)
tree_<inst>.npz: rst/vld/op uint8[n], data uint16[n, N]; want_* (the RTL's
          response, uint16/uint8), def_* (defined mask), ref_* (numpy ref).
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.environ.get("WT", "/work/shared/users/phd/sk3463/scratch/wt-u3d"))
from examples.minitpu.harness import check, ref, rtl
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
what = sys.argv[2:] or ["sfu", "n16", "n64"]
if "sfu" in what:
    from examples.minitpu.units import sfu as U
    st = U.stimulus()
    op = st[:, 0].astype(np.uint8); x = st[:, 1].astype(np.uint16)
    t0 = time.time(); want = ref.sfu(st[:, 0].astype(np.int64), st[:, 1].astype(np.int64)).astype(np.uint16)
    T = ref._sfu_tables()
    np.savez(f"{out}/sfu.npz", op=op, x=x, want=want, gelu=T["gelu"].astype(np.int32),
             exp=T["exp"].astype(np.int32), recip=T["recip"].astype(np.int32), rsqrt=T["rsqrt"].astype(np.int32))
    print(f"DUMPED sfu {len(op)} vectors ({time.time()-t0:.1f}s)", flush=True)
for inst in ("n16", "n64"):
    if inst not in what:
        continue
    from examples.minitpu.units import xlu_reduction_tree as U
    unit = U.INSTANCES[inst]; nl = U._n(inst); lanes, sub = U.GEOM[inst]
    cmd, spans = check._trace_all(U, inst)
    n = len(cmd["valid_i"])
    packed = {p: rtl.pack(cmd[p], w) for p, w in unit.inputs}
    t0 = time.time(); got = rtl.run_trace(unit, packed, seed=1)
    want, reason, _ = U.REF(inst, packed)
    d = {"rst": np.asarray(cmd["rst_ni"], np.uint8), "vld": np.asarray(cmd["valid_i"], np.uint8),
         "op": np.asarray(cmd["op_i"], np.uint8),
         "data": ref._split16(packed["data_i"], nl).astype(np.uint16),
         "spans": np.array([f"{a}:{b}:{c}" for a, b, c in spans])}
    for p in ("valid_o", "result_o", "lane_valid_o", "lane_result_o"):
        g = got[p]; r = want[p]
        if p == "lane_result_o":
            g = ref._split16(g, sub); r = ref._split16(r, sub)
        else:
            g = g[:, 0]; r = r[:, 0]
        d["want_" + p] = g.astype(np.uint16); d["ref_" + p] = r.astype(np.uint16)
        d["def_" + p] = reason[p] == ""
        bad = int((d["def_" + p] & (d["want_" + p] != d["ref_" + p]).reshape(n, -1).any(axis=1)).sum())
        print(f"  {inst} {p}: defined {int(d['def_' + p].sum())}, RTL vs ref bad {bad}")
    np.savez(f"{out}/tree_{inst}.npz", **d)
    print(f"DUMPED tree {inst}: {n} cycles, {len(spans)} traces ({time.time()-t0:.1f}s)", flush=True)
if "pe" in what:
    from examples.minitpu.units import mxu_pe as U
    unit = U.RTL
    cmd, spans = check._trace_all(U, "pe")
    n = len(cmd["rst_ni"])
    packed = {p: rtl.pack(cmd[p], w) for p, w in unit.inputs}
    t0 = time.time(); got = rtl.run_trace(unit, packed, seed=1)
    want, reason, _ = U.REF("pe", packed)
    d = {p: np.asarray([int(v) for v in cmd[p]], dtype=np.uint64) for p, _ in unit.inputs}
    d["spans"] = np.array([f"{a}:{b}:{c}" for a, b, c in spans])
    for p, w, _ in unit.outputs:
        g = got[p][:, 0]; r = want[p][:, 0]
        d["want_" + p] = g.astype(np.uint64); d["ref_" + p] = r.astype(np.uint64)
        d["def_" + p] = reason[p] == ""
        bad = int((d["def_" + p] & (d["want_" + p] != d["ref_" + p])).sum())
        cen = {}
        for why in reason[p]:
            if why: cen[why] = cen.get(why, 0) + 1
        print(f"  pe {p}: defined {int(d['def_' + p].sum())}, RTL vs ref bad {bad}, masked {cen}")
    np.savez(f"{out}/pe.npz", **d)
    print(f"DUMPED pe: {n} cycles, {len(spans)} traces ({time.time()-t0:.1f}s)", flush=True)
