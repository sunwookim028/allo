"""Pipelined vs sequential fetch/dispatch on chia's RTL backend, measured by cosim.

This is the experiment the HLS path could not run. Two things the Vitis route
forbade are legal here:

  * a shared random-access scratchpad touched by several concurrent processes
    (Vitis dataflow: "single reader and a single writer"), and
  * region-scope shared state at all (broken VHLS codegen: per-kernel `static`
    copies).

So the Gemmini-style shape survives: loader and executor share `spad`, and the
overlap comes from double-buffer credits on `free` -- the loader stages tile
n+1 while the executor consumes tile n. `cosim` returns a real cycle count from
RTL simulation, not a synthesis estimate.

Both variants run the identical instruction stream over identical memory, so
the only difference measured is the sequencer.
"""
import numpy as np
from allo import kernel
from allo.lang import f32, i32, Stream

T = 4
TILE = T * T
M = K = N = 8
Mt, Nt, Kt = M // T, N // T, K // T
NBUF = 2

N_LD = Mt * Nt * Kt
N_EX = N_LD
N_ST = Mt * Nt
NPROG = N_LD + N_EX + N_ST
OP_LOAD, OP_EXEC, OP_STORE = 0, 1, 2

A_BASE = 0
B_BASE = M * K
C_BASE = B_BASE + K * N
DRAM = C_BASE + M * N

SPAD = 2 * NBUF * TILE
XSLOT = 0
WSLOT = NBUF * TILE
ACC = Mt * Nt * TILE

QD = 16


# ----------------------------------------------------------------- pipelined --
@kernel
async def sequencer(imem: i32[NPROG * 4], ld_q: Stream[i32, QD],
                    ex_q: Stream[i32, QD], st_q: Stream[i32, QD]):
    """Fetch, decode, dispatch. Routes in program order: draining one queue at a
    time would fill it while its consumer waits on another, and the machine
    deadlocks (the executor is what returns buffer credits)."""
    for pc in range(NPROG):
        op: i32 = imem[pc * 4]
        a0: i32 = imem[pc * 4 + 1]
        a1: i32 = imem[pc * 4 + 2]
        a2: i32 = imem[pc * 4 + 3]
        if op == OP_LOAD:
            ld_q.put(a0)
            ld_q.put(a1)
            ld_q.put(a2)
        elif op == OP_EXEC:
            ex_q.put(a0)
            ex_q.put(a1)
            ex_q.put(a2)
        else:
            st_q.put(a0)
            st_q.put(a1)


@kernel
async def loader(dram: f32[DRAM], spad: f32[SPAD], ld_q: Stream[i32, QD],
                 ld_done: Stream[i32, QD], free: Stream[i32, QD]):
    for c in range(N_LD):
        a_addr: i32 = ld_q.get()
        b_addr: i32 = ld_q.get()
        slot: i32 = ld_q.get()
        credit: i32 = free.get()          # blocks only when NBUF tiles ahead
        for e in range(TILE):
            spad[slot * TILE + e] = dram[a_addr + e]
        for e in range(TILE):
            spad[WSLOT + slot * TILE + e] = dram[b_addr + e]
        ld_done.put(slot)


@kernel
async def executor(spad: f32[SPAD], accbuf: f32[ACC], ex_q: Stream[i32, QD],
                   ld_done: Stream[i32, QD], free: Stream[i32, QD],
                   ex_done: Stream[i32, QD]):
    for c in range(N_EX):
        first: i32 = ex_q.get()
        acc_off: i32 = ex_q.get()
        last: i32 = ex_q.get()
        slot_r: i32 = ld_done.get()
        for i in range(T):
            for j in range(T):
                s: f32 = 0.0
                for k in range(T):
                    s += (spad[slot_r * TILE + i * T + k]
                          * spad[WSLOT + slot_r * TILE + j * T + k])
                if first == 1:
                    accbuf[acc_off + i * T + j] = s
                else:
                    accbuf[acc_off + i * T + j] += s
        free.put(1)                        # release the buffer for the next tile
        if last == 1:
            ex_done.put(acc_off)


@kernel
async def storer(accbuf: f32[ACC], out: f32[DRAM], st_q: Stream[i32, QD],
                 ex_done: Stream[i32, QD]):
    for c in range(N_ST):
        acc_off: i32 = st_q.get()
        c_addr: i32 = st_q.get()
        tok: i32 = ex_done.get()
        for e in range(TILE):
            out[c_addr + e] = accbuf[acc_off + e]


@kernel
async def tpu_pipe(imem: i32[NPROG * 4], dram: f32[DRAM], out: f32[DRAM]):
    spad: f32[SPAD]
    accbuf: f32[ACC]
    ld_q: Stream[i32, QD]
    ex_q: Stream[i32, QD]
    st_q: Stream[i32, QD]
    ld_done: Stream[i32, QD]
    ex_done: Stream[i32, QD]
    free: Stream[i32, QD] = [1, 1]   # NBUF initial credits: seeds the
                                     # loader->executor->loader cycle
    await sequencer(imem, ld_q, ex_q, st_q)
    await loader(dram, spad, ld_q, ld_done, free)
    await executor(spad, accbuf, ex_q, ld_done, free, ex_done)
    await storer(accbuf, out, st_q, ex_done)


# ---------------------------------------------------------------- sequential --
@kernel
def tpu_serial(imem: i32[NPROG * 4], dram: f32[DRAM], out: f32[DRAM]):
    """The current TinyTPU shape: one fetch-decode-dispatch loop, no overlap."""
    spad: f32[SPAD]
    accbuf: f32[ACC]
    slot_r: i32 = 0
    for pc in range(NPROG):
        op: i32 = imem[pc * 4]
        a0: i32 = imem[pc * 4 + 1]
        a1: i32 = imem[pc * 4 + 2]
        a2: i32 = imem[pc * 4 + 3]
        if op == OP_LOAD:
            slot_r = a2
            for e in range(TILE):
                spad[a2 * TILE + e] = dram[a0 + e]
            for e in range(TILE):
                spad[WSLOT + a2 * TILE + e] = dram[a1 + e]
        elif op == OP_EXEC:
            for i in range(T):
                for j in range(T):
                    s: f32 = 0.0
                    for k in range(T):
                        s += (spad[slot_r * TILE + i * T + k]
                              * spad[WSLOT + slot_r * TILE + j * T + k])
                    if a0 == 1:
                        accbuf[a1 + i * T + j] = s
                    else:
                        accbuf[a1 + i * T + j] += s
        else:
            for e in range(TILE):
                out[a1 + e] = accbuf[a0 + e]


# --------------------------------------------------------------------- driver --
def tile_pack(R, C_, Mat):
    out = np.empty((R // T) * (C_ // T) * TILE, np.float32)
    p = 0
    for i in range(R // T):
        for k in range(C_ // T):
            out[p:p + TILE] = Mat[i * T:(i + 1) * T, k * T:(k + 1) * T].ravel()
            p += TILE
    return out


def program():
    prog = []
    t = 0
    for i in range(Mt):
        for j in range(Nt):
            acc_off = (i * Nt + j) * TILE
            for k in range(Kt):
                slot = t % NBUF
                t += 1
                prog += [OP_LOAD, A_BASE + (i * Kt + k) * TILE,
                         B_BASE + (j * Kt + k) * TILE, slot]
                prog += [OP_EXEC, 1 if k == 0 else 0, acc_off,
                         1 if k == Kt - 1 else 0]
            prog += [OP_STORE, acc_off, C_BASE + (i * Nt + j) * TILE, 0]
    return np.array(prog, np.int32)


def run(k, name):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    dram = np.zeros(DRAM, np.float32)
    dram[A_BASE:A_BASE + M * K] = tile_pack(M, K, A)
    dram[B_BASE:B_BASE + K * N] = tile_pack(N, K, np.ascontiguousarray(B.T))
    out = np.zeros(DRAM, np.float32)

    rtl = k.schedule().export("rtl")
    res = rtl.cosim(program(), dram, out)
    got = out[C_BASE:C_BASE + M * N]
    gold = tile_pack(M, N, (A @ B).astype(np.float32))
    ok = np.allclose(got, gold, rtol=1e-4, atol=1e-4)
    print(f"{name:11s} cycles={res.cycles:6d}  correct={ok}  "
          f"maxerr={np.abs(got - gold).max():.2e}")
    return res.cycles, ok


def run_stall(k, name, prob):
    """Re-run under randomized input starvation and output back-pressure. The
    result must be unchanged: this is what actually validates a credit protocol,
    and the HLS path had no equivalent."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    dram = np.zeros(DRAM, np.float32)
    dram[A_BASE:A_BASE + M * K] = tile_pack(M, K, A)
    dram[B_BASE:B_BASE + K * N] = tile_pack(N, K, np.ascontiguousarray(B.T))
    out = np.zeros(DRAM, np.float32)
    rtl = k.schedule().export("rtl")
    res = rtl.cosim(program(), dram, out, stall_prob=prob)
    got = out[C_BASE:C_BASE + M * N]
    gold = tile_pack(M, N, (A @ B).astype(np.float32))
    ok = np.allclose(got, gold, rtol=1e-4, atol=1e-4)
    print(f"{name:11s} stall={prob}  cycles={res.cycles:6d}  correct={ok}")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "serial"):
        run(tpu_serial, "sequential")
    if which in ("both", "pipe"):
        run(tpu_pipe, "pipelined")
    if which == "stall":
        run_stall(tpu_pipe, "pipelined", 0.3)


