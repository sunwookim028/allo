# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track C, probe (plan D2 as the plan first worded it; hypothesis H10):
the DMA as a SELF-TIMED composition, judged on its contract, not per cycle.

    feed -> issue -> [s_ord: Stream depth OUTSTANDING = the credits] -> land
              |  \\-> s_req/s_wd -> bridge (device memory DM) -> s_rsp -/  |
              |                                                          v
              +<- s_vrd <- vowner (owner of D-12 port vmem.d) <- [s_lnd: depth LANDING_DEPTH]

* The credit counter IS the order Stream's capacity: ``issue`` blocks on a
  full ``s_ord`` exactly where ``dma.sv`` waits for ``has_credit``; the
  landing FIFO IS ``s_lnd``'s capacity, and its back-pressure is
  ``dm_rsp_ready``. Neither is a counter in a body.
* ``bridge`` is the memory side of the credit pipe (the D-21 shim's role):
  in order, a load answers ``len + 1`` words with ``last`` on the final one,
  a store takes ``len + 1`` W beats and answers one beat with ``last``.
* ``vowner`` owns ``vmem.d`` (``rw``, latency 2): one port access per
  iteration; it drains a load's beats (gathering four into a word, committed
  on beat 3) and serves a store's beats, one descriptor after another.

What it is checked on (``main``): the request stream equals ``dma.sv``'s on
the same descriptors (run on the RTL with Phase 0's program generator), the
device memory after the program equals a sequential reference, and every
descriptor completes. Not modelled: SLVERR, the watchdog, reset, two
channels in flight at once (descriptors are served in order).
"""

from __future__ import annotations

import sys

import numpy as np

from allo.compose import Architecture, Channel, Memory, Port, unit

from examples.minitpu.units.dma_addr_gen import addr  # noqa: F401  (issue calls it)

M32 = (1 << 32) - 1


@unit(memories=("DESC",), writes=("s_d", "s_db", "s_ds", "s_v"), parameters=("ND",))
def st_feed(desc: UInt(32)[ND, 5]):
    for i in range(ND):
        d: UInt(64) = 0
        st: UInt(1) = desc[i, 0]
        vr: UInt(14) = desc[i, 1]
        rw: UInt(14) = desc[i, 2]
        d[0] = st
        d[1:15] = vr
        d[15:29] = rw
        s_d.put(d)
        s_v.put(d)
        s_db.put(desc[i, 3])
        s_ds.put(desc[i, 4])


@unit(reads=("s_d", "s_db", "s_ds", "s_vrd"), writes=("s_ord", "s_req", "s_wd"),
      parameters=("ND", "L", "ROW_M", "PW_M1", "ADDR_M"), calls=("addr",))
def st_issue():
    lanes: UInt(32)[L] = 0
    for i in range(ND):
        d: UInt(64) = s_d.get()
        base: UInt(32) = s_db.get()
        stride: UInt(32) = s_ds.get()
        store: int32 = d[0]
        vmem: int32 = d[1:15]
        lastrow: int32 = d[15:29]
        cur: int32 = 0
        busy: int32 = 1
        while busy == 1:
            cur14: UInt(14) = cur
            agu: UInt(32) = addr(base, cur14, stride)
            logical: int32 = agu & ADDR_M
            left_m1: int32 = (lastrow - cur) & ROW_M
            to_bound_m1: int32 = PW_M1 - (logical & PW_M1)
            left_sat: int32 = left_m1 & 255
            if (left_m1 >> 8) != 0:
                left_sat = 255
            blen: int32 = 0
            if stride == 1:
                blen = left_sat
                if to_bound_m1 < left_sat:
                    blen = to_bound_m1
            last: int32 = 0
            if blen == left_m1:
                last = 1
            o: UInt(64) = 0  # order token: store[0] last[1] vbase[2:18] len[18:26]
            o1: UInt(1) = store
            o2: UInt(1) = last
            o16: UInt(16) = (vmem + cur) & 0xFFFF
            o8: UInt(8) = blen
            o[0] = o1
            o[1] = o2
            o[2:18] = o16
            o[18:26] = o8
            s_ord.put(o)  # blocks while OUTSTANDING bursts are in flight: the credit
            r: UInt(64) = 0  # request: we[0] addr[1:30] len[30:38] stop[38]
            r29: UInt(29) = logical
            r[0] = o1
            r[1:30] = r29
            r[30:38] = o8
            s_req.put(r)
            if store == 1:
                for _b in range(blen + 1):
                    with allo.meta_for(L) as k:
                        lanes[k] = s_vrd[k].get()
                    with allo.meta_for(L) as k:
                        s_wd[k].put(lanes[k])
            if last == 1:
                busy = 0
            else:
                cur = (cur + blen + 1) & ROW_M
    stop: UInt(64) = 0
    stop[38] = 1
    s_req.put(stop)
    so: UInt(64) = 0
    so[26] = 1
    s_ord.put(so)


@unit(memories=("DM", "REQ"), reads=("s_req", "s_wd"), writes=("s_rsp", "s_rspd"),
      parameters=("L", "DMW", "NRMAX"))
def st_bridge(dm: UInt(32)[DMW, L], req: UInt(64)[NRMAX]):
    k_req: int32 = 0
    run: int32 = 1
    lanes: UInt(32)[L] = 0
    while run == 1:
        r: UInt(64) = s_req.get()
        req[k_req] = r
        k_req = k_req + 1
        if r[38] == 1:
            run = 0
        else:
            we: int32 = r[0]
            a: int32 = r[1:30]
            ln: int32 = r[30:38]
            if we == 0:
                for b in range(ln + 1):
                    w: int32 = (a + b) & (DMW - 1)
                    t: UInt(32) = 0
                    if b == ln:
                        t[0] = 1
                    s_rsp.put(t)
                    with allo.meta_for(L) as k:
                        s_rspd[k].put(dm[w, k])
            else:
                for b2 in range(ln + 1):
                    w2: int32 = (a + b2) & (DMW - 1)
                    with allo.meta_for(L) as k:
                        lanes[k] = s_wd[k].get()
                    with allo.meta_for(L) as k:
                        dm[w2, k] = lanes[k]
                t2: UInt(32) = 1
                s_rsp.put(t2)
                with allo.meta_for(L) as k:
                    s_rspd[k].put(0)


@unit(memories=("STAT",), reads=("s_ord", "s_rsp", "s_rspd"), writes=("s_lnd", "s_lndd"),
      parameters=("L",))
def st_land(stat: UInt(32)[4]):
    run: int32 = 1
    bursts: int32 = 0
    st_done: int32 = 0
    lanes: UInt(32)[L] = 0
    while run == 1:
        o: UInt(64) = s_ord.get()  # no peek: taken at the burst's first beat (C9)
        if o[26] == 1:
            run = 0
        else:
            bursts = bursts + 1
            store: int32 = o[0]
            vb: int32 = o[2:18]
            ln: int32 = o[18:26]
            if store == 0:
                for b in range(ln + 1):
                    t: UInt(32) = s_rsp.get()
                    with allo.meta_for(L) as k:
                        lanes[k] = s_rspd[k].get()
                    dst: UInt(32) = (vb + b) & 0xFFFF
                    s_lnd.put(dst)  # blocks on a full landing FIFO: dm_rsp_ready
                    with allo.meta_for(L) as k:
                        s_lndd[k].put(lanes[k])
            else:
                t2: UInt(32) = s_rsp.get()
                with allo.meta_for(L) as k:
                    lanes[k] = s_rspd[k].get()
                if o[1] == 1:
                    st_done = st_done + 1
    stat[0] = bursts
    stat[1] = st_done


@unit(memories=("vmem.d", "VSTAT"), reads=("s_v", "s_lnd", "s_lndd"), writes=("s_vrd",),
      parameters=("NP", "L", "SUB", "WB"))
def st_vowner(mem, vstat: UInt(32)[4]):
    # one port cycle per iteration; NP = every beat of the program + one
    # FLUSH iteration after each store descriptor: the server returns the read
    # of iteration t only in exchange for the access of t + 1 (latency 2 in
    # token time), so the last store beat is retrieved by a dummy access
    # before the next descriptor's blocking get (finding C10: without it a
    # store followed by a load deadlocks -- issue waits for that beat, the
    # owner waits for the load's landing beat, which issue has not requested)
    gather: UInt(WB) = 0
    have: int32 = 0       # a descriptor is being served
    store: int32 = 0
    vmem: int32 = 0
    left: int32 = 0       # beats still to serve in it
    r: int32 = 0          # beat index within it
    pend: int32 = 0       # a store read issued last iteration
    pidx: int32 = 0
    ld_done: int32 = 0
    flush: int32 = 0
    lanes: UInt(32)[L] = 0
    for _t in range(NP):
        if have == 0 and flush == 0:
            d: UInt(64) = s_v.get()
            store = d[0]
            vmem = d[1:15]
            left = d[15:29] + 1
            r = 0
            have = 1
        we: int32 = 0
        ptr: int32 = (vmem + r) & 0x3FFF
        if have == 1:
            if store == 0:
                dst: UInt(32) = s_lnd.get()
                ptr = dst & 0x3FFF
                with allo.meta_for(L) as k:
                    lanes[k] = s_lndd[k].get()
                we = 1
        word: int32 = ptr >> 2
        idx: int32 = ptr & 3
        wwd: UInt(WB) = gather
        if we == 1:
            with allo.meta_for(SUB) as s:
                if idx == s:
                    with allo.meta_for(L) as k:
                        wwd[32 * (s * L + k):32 * (s * L + k + 1)] = lanes[k]
        q: UInt(WB) = mem[word]  # the read of t - 1 (latency 2: token t = post edge t)
        cm: uint1 = 0
        if we == 1 and idx == SUB - 1:
            cm = 1
        if cm:
            mem[word] = wwd
        if pend == 1:  # the store beat read last iteration
            with allo.meta_for(SUB) as s:
                if pidx == s:
                    with allo.meta_for(L) as k:
                        s_vrd[k].put(q[32 * (s * L + k):32 * (s * L + k + 1)])
        pend = 0
        flush = 0
        if have == 1:
            if we == 1:
                gather = wwd
            else:
                pend = 1
                pidx = idx
            r = r + 1
            left = left - 1
            if left == 0:
                have = 0
                if store == 0:
                    ld_done = ld_done + 1
                else:
                    flush = 1
    vstat[0] = ld_done


@unit(memories=("vmem.c",), parameters=("NP", "WB"))
def st_vcompute_idle(mem):
    for _t in range(NP):
        z: int32 = 0
        x: UInt(WB) = mem[z]


def architecture(descs, inst="core", dmw=4096):
    from examples.minitpu.units.dma_params import GEOMETRIES  # noqa: PLC0415

    g = GEOMETRIES[inst].legality()
    nb = sum(d[2] + 1 for d in descs) + sum(d[0] for d in descs)  # + a flush per store (C10)
    nreq = sum(len(bursts(*d)) for d in descs) + 1
    p = {"ND": len(descs), "L": g.LANES, "ROW_M": (1 << g.ROW_BITS) - 1, "PW_M1": g.PAGE_WORDS - 1,
         "ADDR_M": (1 << g.DRAM_BEAT_ADDR_W) - 1, "DMW": dmw, "NRMAX": nreq, "NP": nb,
         "SUB": g.NUM_SUBLANES, "WB": g.NUM_SUBLANES * g.BEAT_BITS}
    vmem = Memory("vmem", "UInt(WB)", rows="256",
                  ports=(Port("c", "rw", latency=3, visible=1),
                         Port("d", "rw", latency=g.VMEM_DMA_READ_LATENCY, visible=1)),
                  collision="obligation", reset=False)
    lanes = lambda n, d: Channel(n, "UInt(32)", d, ("L",))  # noqa: E731
    return Architecture(
        name=f"dma_selftimed_{inst}",
        parameters=p,
        memories=(Memory("DESC", "UInt(32)[ND, 5]"), Memory("DM", "UInt(32)[DMW, L]"),
                  Memory("REQ", "UInt(64)[NRMAX]"), Memory("STAT", "UInt(32)[4]"),
                  Memory("VSTAT", "UInt(32)[4]"), vmem),
        channels=(Channel("s_d", "UInt(64)", "1"), Channel("s_db", "UInt(32)", "1"),
                  Channel("s_ds", "UInt(32)", "1"), Channel("s_v", "UInt(64)", "4"),
                  Channel("s_ord", "UInt(64)", str(g.OUTSTANDING)),
                  Channel("s_req", "UInt(64)", "2"), lanes("s_wd", "2"),
                  Channel("s_rsp", "UInt(32)", "2"), lanes("s_rspd", "2"),
                  Channel("s_lnd", "UInt(32)", str(g.LANDING_DEPTH)), lanes("s_lndd", str(g.LANDING_DEPTH)),
                  lanes("s_vrd", "2")),
        units=(st_feed, st_issue, st_bridge, st_land, st_vowner, st_vcompute_idle),
        obligations={"vmem": "compute port idle in this probe (never writes)"},
    )


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

def bursts(store, vmem, rows, base, stride):
    """``dma.sv``'s burst list for one descriptor: ``[(we, addr, len)]``."""
    out, cur = [], 0
    while True:
        logical = ((base + (cur & 0x3FFF) * stride) & M32) & ((1 << 29) - 1)
        left = (rows - cur) & 0x3FFF
        sat = 255 if left >> 8 else left
        blen = min(sat, 127 - (logical & 127)) if stride == 1 else 0
        out.append((store, logical, blen))
        if blen == left:
            return out
        cur = (cur + blen + 1) & 0x3FFF


def reference(descs, dm0, dmw):
    """Sequential semantics: VMEM beats as a dict (whole words only, as the
    gather commits them), the device memory as an array."""
    dm = dm0.copy()
    vm = {}
    for store, vmem, rows, base, stride in descs:
        for r in range(rows + 1):
            a = ((base + r * stride) & M32) & ((1 << 29) - 1) & (dmw - 1)
            v = (vmem + r) & 0x3FFF
            if store:
                dm[a] = vm[v]
            else:
                vm[v] = dm[a].copy()
    return dm


def program(seed, n=12):
    import random  # noqa: PLC0415
    rng = random.Random(seed)
    descs, loaded = [], []
    for i in range(n):
        if loaded and (i % 3 == 2 or rng.random() < 0.3):
            vmem, words = rng.choice(loaded)
            descs.append((1, vmem, 4 * words - 1, rng.randrange(0, 3000), rng.choice([1, 1, 2, 3])))
        else:
            words = rng.choice([1, 2, 4, 8, 16])
            vmem = 4 * rng.randrange(0, 256 - words)
            descs.append((0, vmem, 4 * words - 1, rng.choice([rng.randrange(0, 3500), 0x80 - 5]),
                          rng.choice([1, 1, 1, 2, 7])))
            loaded.append((vmem, words))
    return descs


def rtl_requests(descs, inst="core"):
    """``dma.sv``'s request stream on the same descriptors (one channel,
    waited between descriptors as ``wait.channel`` does), by the RTL."""
    from examples.minitpu.harness import rtl  # noqa: PLC0415
    from examples.minitpu.units import dma as D  # noqa: PLC0415
    rng = D.rng_for("dma-selftimed", inst)
    pr = D.Program(inst, rng, D.Bridge(rng, rtt=(2, 9), p_gap=0.1), p_ready=0.8)
    for store, vmem, rows, base, stride in descs:
        pr.desc(0, store, vmem, rows, base, stride)
        pr.drain()
    cmd = pr.cmd()
    u = D.INSTANCES[inst]
    out = rtl.run_trace(u, {p: rtl.pack(cmd[p], w) for p, w in u.inputs})
    col = {p: rtl.unpack(out[p]) for p in ("dm_req_valid", "dm_req_we", "dm_req_addr", "dm_req_len")}
    reqs, prev_store_beat = [], 0
    for t, v in enumerate(col["dm_req_valid"]):
        if v and cmd["dm_req_ready"][t]:
            if col["dm_req_we"][t]:
                if prev_store_beat == 0:
                    reqs.append((1, col["dm_req_addr"][t], col["dm_req_len"][t]))
                prev_store_beat = (prev_store_beat + 1) % (col["dm_req_len"][t] + 1)
            else:
                reqs.append((0, col["dm_req_addr"][t], col["dm_req_len"][t]))
    return reqs, len(cmd["rst_n"])


def run(target, descs, inst="core", dmw=4096, seed=0, project=None):
    import allo.dataflow as df  # noqa: F401,PLC0415
    arch = architecture(descs, inst, dmw)
    region = arch.region("simulator")
    if target == "simulator":
        mod = df.build(region, target="simulator")
    else:
        mod = df.build(region, target="systemc", mode="csim", project=project)
    rng = np.random.default_rng(seed)
    dm0 = rng.integers(0, 1 << 32, size=(dmw, 8), dtype=np.uint64).astype(np.uint32)
    desc = np.array(descs, dtype=np.uint64).astype(np.uint32)
    dm = dm0.copy()
    req = np.zeros(arch.parameters["NRMAX"], dtype=np.uint64)
    stat = np.zeros(4, dtype=np.uint32)
    vstat = np.zeros(4, dtype=np.uint32)
    mod(desc, dm, req, stat, vstat)
    return dm0, dm, req, stat, vstat


def main(argv=None):
    import argparse  # noqa: PLC0415
    import time  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", action="append", choices=("simulator", "systemc"))
    ap.add_argument("--inst", action="append")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--project", default="/tmp/minitpu_selftimed_prj")
    a = ap.parse_args(argv)
    bad = 0
    for inst in a.inst or ["core"]:
        for seed in range(a.seeds):
            descs = program(seed)
            want_req = [b for d in descs for b in bursts(*d)]
            rtl_req, rtl_cycles = rtl_requests(descs, inst)
            print(f"RTL dma:{inst} seed {seed}: {len(descs)} descriptors, {len(rtl_req)} requests in "
                  f"{rtl_cycles} cycles; burst rule == RTL: {rtl_req == want_req}", flush=True)
            bad += rtl_req != want_req
            for backend in a.backend or ["simulator"]:
                t = time.time()
                try:
                    dm0, dm, req, stat, vstat = run(backend, descs, inst, seed=seed,
                                                    project=f"{a.project}/{inst}_{seed}_{backend}")
                except Exception as e:  # noqa: BLE001
                    print(f"CONTRACT-FAIL dma_selftimed:{inst} seed {seed} {backend}: "
                          f"{type(e).__name__}: {str(e)[:400]}")
                    bad += 1
                    continue
                got_req = [(int(r) & 1, (int(r) >> 1) & ((1 << 29) - 1), (int(r) >> 30) & 255)
                           for r in req[:-1]]
                dm_ok = np.array_equal(dm, reference(descs, dm0, 4096))
                n_st = sum(d[0] for d in descs)
                ok = (got_req == rtl_req and int(req[-1]) >> 38 & 1 == 1 and dm_ok
                      and int(stat[1]) == n_st and int(vstat[0]) == len(descs) - n_st
                      and int(stat[0]) == len(rtl_req))
                tag = "CONTRACT-MATCH" if ok else "CONTRACT-DIFF "
                print(f"{tag} dma_selftimed:{inst} seed {seed} {backend}: requests {len(got_req)} "
                      f"(== RTL's: {got_req == rtl_req}), DM after == reference: {dm_ok}, completions "
                      f"store {int(stat[1])}/{n_st} load {int(vstat[0])}/{len(descs) - n_st} "
                      f"({time.time() - t:.1f}s)", flush=True)
                bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
