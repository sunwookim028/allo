# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The functional stress gate for TinyTPU-isa: what `bench_isa.py` cannot see.

`bench_isa.py` reproduces the published setup -- seed 0, operands in [-4, 4]
(Gemmini's `allo_cmp.c` distribution), `C` zeroed, only the `M x N` result
compared, GEMM plus one vadd program. Those inputs hide whole classes of bug:
at T=4 a PE's partial sum never leaves 9 bits, so **narrowing the int32
partial sum to int16 prints ALL EXACT** (shown on `chia-isa`; `mutate.py`
here re-proves it). This gate runs on the same simulator build and adds:

  * **operands**: full-range int8 with -128 and 127 forced in, a corner
    distribution drawn from {-128, -127, -1, 0, 1, 126, 127} (a T-deep
    partial sum of -128*-128 is 65536, which int16 wraps to 0), a
    mid range whose sums straddle int8, and a DIRECTED boundary case whose
    results are exactly 127, 128, -128, -129, 0, -1, ... so the clip and the
    ReLU are tested at their edges rather than hoped at;
  * **shapes**: every multiple of T up to MAXDIM in every dimension (64),
    not the five scored ones, so nothing special-cased to them survives;
  * **`C` prefilled with random bytes**, and the WHOLE of `C` compared: the
    result region must be overwritten and everything outside it untouched, so
    a design may not rely on `C` arriving zeroed nor scribble past `M x N`;
  * **programs other than GEMM**: `isa_dsl.vector_program`, which varies
    every field GEMM leaves constant (DRAM row, result region, distinct
    `vadd`/`vrelu` destinations), and random valid programs, both against
    `isa_ref.run` -- the ISA as numpy;
  * **many invocations of one build**, so a later run sees the state an
    earlier one left in `vmem`/`vr`/`ar`, which nothing clears;
  * **`kpn_model` on every distinct program first**, so a program whose
    header counts and dispatch disagree is reported with the blocked units
    instead of hanging the simulator, and **the validator's controls**:
    crafted programs that break the write-before-read contract (and the
    other `check_program` rules) must be rejected, every generated one
    accepted.

    python stress_isa.py          # everything, ~10 s
    python stress_isa.py quick    # scored shapes + corner/boundary only

Prints `STRESS OK: n/n` and exits 0, or lists each failing run and exits 1.
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
import allo.dataflow as df  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    tinytpu_isa, assemble, check_program, ProgramError, enc, enc_agu,
    gemm_program_flat, gemm_program_handwritten, vadd_program,
    OP_DMA_LD, OP_DMA_ST, OP_VLD, OP_MM, OP_VADD, OP_VRELU, OP_MVOUT,
    OP_VMATLOAD, OP_VMATPUSH, OP_VMATPOP, OP_VST, OP_VMEMST, OUTQ,
    OP_LOOP, OP_ENDLOOP, AGU_F0, AGU_F1,
    MAXDIM, T, WPR, BEATS, IMEM_SIZE, NHDR, IWORDS, VMEM_ROWS, NVREG, VR_RAW_DIST,
    OP_WAIT,
    SCORED_SHAPES, VEC_M,
)
from examples.accelerator.tinytpu_vitis.isa_dsl import (  # noqa: E402
    Program, Ref, gemm_program, vector_program, ar_distance_program,
)
from examples.accelerator.tinytpu_vitis import isa_ref, kpn_model  # noqa: E402

SCORED = list(SCORED_SHAPES)     # T, 2T, 3T, 4Tx4Tx2T, 4T -- v1's five at T=4
ALL_SHAPES = [s for s in itertools.product(range(T, MAXDIM + 1, T), repeat=3)]
# vector_program sizes: VEC_M and half of it (4 and 8 at MAXDIM=16)
VEC_MS = sorted({max(2, VEC_M // 2 // 2 * 2), VEC_M})
MAX_STATIC = (IMEM_SIZE - NHDR) // IWORDS
CORNERS = np.array([-128, -127, -1, 0, 1, 126, 127], np.int8)
CORNER_P = np.array([4, 2, 1, 1, 1, 2, 4], float) / 15


def operands(dist, seed):
    rng = np.random.default_rng(seed)
    shape = (MAXDIM, MAXDIM)
    if dist == "full":
        A = rng.integers(-128, 128, shape).astype(np.int8)
        B = rng.integers(-128, 128, shape).astype(np.int8)
        A[0, 0] = B[0, 0] = -128          # guaranteed, not left to chance
        A[0, 1] = B[1, 0] = 127
    elif dist == "corner":
        A = rng.choice(CORNERS, shape, p=CORNER_P)
        B = rng.choice(CORNERS, shape, p=CORNER_P)
    elif dist == "mid":
        A = rng.integers(-16, 17, shape).astype(np.int8)
        B = rng.integers(-16, 17, shape).astype(np.int8)
    elif dist == "small":
        A = rng.integers(-4, 5, shape).astype(np.int8)
        B = rng.integers(-4, 5, shape).astype(np.int8)
    else:
        raise ValueError(dist)
    return A, B


# Results the boundary case lands on, +-2 each (the row offset below).
TARGETS = [127, 128, -128, -129, 0, 1, -1, 126, 129, -127, -130, 2, 200,
           -200, 254, -254]


def boundary_operands(M, K, N, seed):
    """A and B chosen so C[i, j] = TARGETS[j] + s_i, s_i in -2..2, exactly.
    Everything the GEMM does not read stays random full-range, so a unit
    that reads outside its region still shows up."""
    assert K >= 3
    A, B = operands("full", seed)
    A[:M, :K] = 0
    A[:M, 0] = 1
    A[:M, 1] = 1
    A[:M, 2] = [((i % 5) - 2) for i in range(M)]
    B[:K, :N] = 0
    for j in range(N):
        t = TARGETS[j % len(TARGETS)]
        B[0, j], B[1, j], B[2, j] = t // 2, t - t // 2, 1
    return A, B


def gemm_gold(A, B, M, K, N, relu, C0):
    g = A[:M, :K].astype(np.int64) @ B[:K, :N].astype(np.int64)
    if relu:
        g = np.maximum(g, 0)
    C = C0.reshape(MAXDIM, MAXDIM).copy()
    C[:M, :N] = np.clip(g, -128, 127)
    return C.reshape(-1)


def execute(mod, prog, A, B, C0):
    words = assemble(prog)
    imem = np.zeros(IMEM_SIZE, np.uint64)
    imem[: len(words)] = np.array(words, np.uint64)
    C = C0.copy()
    mod(imem, A.reshape(-1), B.reshape(-1), C)
    return C


def compare(tag, got, want, M=None, N=None):
    """None if equal, else a line saying how many cells are wrong inside the
    result region and how many were clobbered outside it."""
    got = got.reshape(MAXDIM, MAXDIM)
    want = want.reshape(MAXDIM, MAXDIM)
    diff = got != want
    if not diff.any():
        return None
    if M is None:
        return f"{tag}: {int(diff.sum())}/{MAXDIM * MAXDIM} cells of C wrong"
    inside = int(diff[:M, :N].sum())
    return (f"{tag}: wrong={inside}/{M * N} "
            f"clobbered_outside={int(diff.sum()) - inside}")


# ------------------------------------------------------ random programs ---
def random_program(seed):
    """`_random_program(seed)`, retried deterministically on a derived seed
    in the rare case the program outgrows the imem."""
    for attempt in range(8):
        prog = _random_program(seed + 1000003 * attempt)
        if prog is not None:
            return prog
    raise AssertionError(f"no program fits the imem from seed {seed}")


def _random_program(seed):
    """A random program that is valid BY CONSTRUCTION: every read names rows
    an earlier instruction wrote, and every vreg read keeps the distance
    contract (`VR_RAW_DIST` vpu iterations after the write it depends on,
    counting every instruction the vpu runs). Memory windows are either low
    or at the top of each memory, so both overlapping reuse and the last rows
    get hit."""
    rng = np.random.default_rng(seed)
    k = Program(f"fuzz {seed}")
    ri = lambda lo, hi: int(rng.integers(lo, hi + 1))   # noqa: E731
    hi_win = bool(rng.integers(0, 2))
    win = max(64, 2 * MAXDIM)            # every window fits a MAXDIM-row run
    lo = {m: (s - win if hi_win else 0)
          for m, s in (("vmem", VMEM_ROWS), ("vr", NVREG))}
    wr = {"vmem": np.zeros(VMEM_ROWS, bool), "vr": np.zeros(NVREG, bool)}
    vr_at = np.full(NVREG, -VR_RAW_DIST)  # vpu iteration of each row's last write
    vpu = {"it": 0}                      # vpu iterations issued so far

    def place(mem, n):
        return lo[mem] + ri(0, win - n)

    def run_of(mem, n):
        ok = [s for s in range(lo[mem], lo[mem] + win - n + 1)
              if wr[mem][s:s + n].all()]
        return int(rng.choice(ok)) if ok else None

    def reads_ok(reads, writes=()):
        """reads/writes: (vr row, vpu iteration) pairs, iterations counted
        from the instruction's first. True if every read keeps the contract,
        against earlier instructions and against this one's own writes."""
        for row, t in reads:
            last = max([wt for wrow, wt in writes if wrow == row and wt < t],
                       default=None)
            at = vpu["it"] + t
            if last is not None and t - last < VR_RAW_DIST:
                return False
            if last is None and at - vr_at[row] < VR_RAW_DIST:
                return False
        return True

    def commit(writes, n_it):
        for row, t in writes:
            wr["vr"][row] = True
            vr_at[row] = vpu["it"] + t
        vpu["it"] += n_it

    chans = {}                           # DMA channel -> (writes?, VMEM rows)

    def fence(rows=(), writes=False, force=False):
        # wait for every outstanding descriptor this access would race --
        # MiniTPU's contract, which check_program enforces
        m = 0
        for ch, (dw, dr) in chans.items():
            if force or (set(rows) & dr and (dw or writes)):
                m |= 1 << ch
        if m:
            k.wait(m)
            for ch in [c for c in list(chans) if m >> c & 1]:
                del chans[ch]

    def channel():
        free = [c for c in range(2) if c not in chans]
        if free:
            return int(rng.choice(free))
        c = ri(0, 1)                     # both busy: fence one
        k.wait(1 << c)
        del chans[c]
        return c

    def descriptor(n, limit):
        # a stride that walks rows of a matrix, walks beats, or broadcasts
        s = int(rng.choice([WPR, WPR, 1, 0]))
        return s, ri(0, limit - 1 - (n - 1) * s)

    def dma_ld(n=None):
        # DMA writes only VMEM; the vregs are reached by `vld`
        n = n or ri(1, MAXDIM)
        d = place("vmem", n)
        rows = range(d, d + n)
        fence(rows, writes=True)
        ch = channel()
        src = ri(0, 1)
        stride, off = descriptor(n, BEATS)
        k.vmemld(ch, src * BEATS + off, stride, d, rows=n)
        chans[ch] = (True, set(rows))
        wr["vmem"][d:d + n] = True
        if rng.integers(0, 2):
            fence(force=True)

    def vld():
        n = ri(1, 24)
        s = run_of("vmem", n)
        if s is None:
            return False
        fence(range(s, s + n))
        d = place("vr", n)
        k.vld(d, s, rows=n)
        commit([(d + i, i) for i in range(n)], n)
        return True

    qout = {"n": 0}                      # rows pushed and not yet popped

    def pop(n=None):
        if qout["n"] == 0:
            return False
        n = n or ri(1, min(qout["n"], MAXDIM))
        d = place("vr", n)
        k.vmatpop(d, rows=n)
        commit([(d + i, i) for i in range(n)], n)
        qout["n"] -= n
        return True

    def mm():
        # a load and one push, sometimes a second load and push behind it
        # (pending weights, two tiles in the output FIFO); pops now, later,
        # or split
        for _ in range(ri(1, 2)):
            n = ri(1, MAXDIM)
            w, a = run_of("vr", T), run_of("vr", n)
            if (w is None or a is None or qout["n"] + n > OUTQ
                    or not reads_ok([(w + i, i) for i in range(T)])):
                break
            k.vmatload(w)
            commit([], T)
            if not reads_ok([(a + i, i) for i in range(n)]):
                k.vmatpush(w, rows=T)        # the weights themselves, which
                commit([], T)                # are T iterations old already
                qout["n"] += T
                continue
            k.vmatpush(a, rows=n)
            commit([], n)
            qout["n"] += n
        else:
            if rng.integers(0, 2):
                pop()
            return True
        return qout["n"] > 0

    def vec(op):
        n = ri(1, 12)
        s1, s2 = run_of("vr", n), run_of("vr", n)
        if s1 is None:
            return False
        d = place("vr", n)
        if op == "vadd":
            reads = [(r, t) for i in range(n)
                     for r, t in ((s1 + i, 2 * i), (s2 + i, 2 * i + 1))]
            writes, n_it = [(d + i, 2 * i + 1) for i in range(n)], 2 * n
        else:
            reads = [(s1 + i, i) for i in range(n)]
            writes, n_it = [(d + i, i) for i in range(n)], n
        if not reads_ok(reads, writes):
            return False
        if op == "vadd":
            k.vadd(d, s1, s2, rows=n)
        else:
            k.vrelu(d, s1, rows=n)
        commit(writes, n_it)
        return True

    def vst():
        n = ri(1, MAXDIM)
        s = run_of("vr", n)
        if s is None or not reads_ok([(s + i, i) for i in range(n)]):
            return False
        d = place("vmem", n)
        fence(range(d, d + n), writes=True)
        k.vst(s, d, rows=n)
        wr["vmem"][d:d + n] = True
        commit([], n)
        return True

    def vmemst():
        n = ri(1, MAXDIM)
        s = run_of("vmem", n)
        if s is None:
            return False
        fence(range(s, s + n))
        ch = channel()
        stride, off = descriptor(n, BEATS)
        k.vmemst(ch, off, stride, s, rows=n)
        chans[ch] = (False, set(range(s, s + n)))
        return True

    def vmemst_loop():
        # the AGU on the DRAM base and the VMEM row of one descriptor at
        # once, a wait in the body (a channel takes one descriptor at a time)
        if WPR < 2:
            return False                 # one column block: nothing to walk
        trip, n = ri(2, min(3, WPR)), ri(1, min(4, MAXDIM // 3))
        s = run_of("vmem", trip * n)
        if s is None:
            return False
        fence(force=True)
        ch = ri(0, 1)
        with k.loop(trip) as i:          # rows i*n.., column block i
            k.vmemst(ch, Ref(0).at(i, n * WPR + 1), WPR,
                     Ref(s).at(i, n), rows=n)
            k.wait(1 << ch)
        return True

    def mvout():
        # retire: vregs -> VMEM -> DRAM, as MiniTPU does
        return vst() and vmemst()

    dma_ld(ri(T, MAXDIM))                # weights
    dma_ld()                             # activations, once vld'd
    for _ in range(ri(0, 2)):
        dma_ld()
    while not mm():
        dma_ld()
        vld()
    while pop():                         # something to read before vec/mvout
        pass
    ops = [dma_ld, vld, mm, pop, lambda: vec("vadd"), lambda: vec("vrelu"),
           vst, vmemst, vmemst_loop, mvout]
    budget = ri(8, MAX_STATIC - 16)   # mm, fences, the drain pops and the last mvout overshoot it
    while len(k.words) < budget:
        ops[int(rng.integers(0, len(ops)))]()
    while qout["n"]:                     # every pushed row is popped
        pop(min(qout["n"], MAXDIM))
    while not mvout():                   # the program ends by retiring rows
        vld() or dma_ld()
    prog = k.emit()
    if len(prog) > MAX_STATIC:
        return None
    check_program(prog)
    return prog


def _bad_programs():
    """Programs `check_program` must REJECT, each breaking one rule. Raw
    `enc` words where the DSL would refuse first, so the check under test is
    the assembler's and not the generator's."""
    # every base run is T rows, so a vmatload of vr 0.. is fully written and
    # each program is rejected for the rule it names, not for another
    ld = (enc(OP_DMA_LD, f0=0, f2=WPR, f3=0, nr=T), 0)   # A rows 0.., ch 0
    wt = (enc(OP_WAIT, f0=1, nr=1), 0)                    # ... fenced
    vw = (enc(OP_VLD, f0=0, f1=0, nr=T), 0)
    wl = lambda v=0, n=T: (enc(OP_VMATLOAD, f0=v, nr=n), 0)  # noqa: E731
    wp = lambda v=0, n=4: (enc(OP_VMATPUSH, f0=v, nr=n), 0)  # noqa: E731
    wo = lambda a=0, n=4: (enc(OP_VMATPOP, f0=a, nr=n), 0)   # noqa: E731
    mm0 = [wl(), wp(), wo()]
    VO = 300                                 # a VMEM row nothing else uses
    out = lambda a=0: [(enc(OP_VST, f0=a, f1=VO, nr=4), 0),         # noqa: E731
                       (enc(OP_VMEMST, f3=VO, nr=4), 0)]
    U = 2 * T + 8        # one vreg file: rows here are written by nothing below
    return {
        "vst before any write to the vregs": [ld, wt, *out()],
        "vadd second source unwritten": [ld, wt, vw, *mm0, (enc(OP_VADD, f0=U + 8, f1=0, f2=U, nr=4), 0), *out(U + 8)],
        "vrelu source unwritten": [ld, wt, vw, *mm0, (enc(OP_VRELU, f0=0, f1=U, nr=4), 0), *out()],
        "vmatload weights never vld'd into vr": [ld, wt, vw, wl(40), wp(), wo(), *out()],
        "vmatpush activations only in vmem, never in vr": [ld, wt, vw, wl(), wp(U), wo(), *out()],
        "descriptor on channel 2 (there are 2)": [(enc(OP_DMA_LD, f0=2, nr=4), 0), ld, wt, vw, *mm0, *out()],
        # MiniTPU's DMA contract
        "wait on an idle channel (MiniTPU hangs)": [ld, wt, vw, *mm0, *out(), (enc(OP_WAIT, f0=2, nr=1), 0)],
        "wait with an empty mask": [ld, wt, vw, *mm0, *out(), (enc(OP_WAIT, f0=0, nr=1), 0)],
        "second descriptor on a busy channel (MiniTPU hangs)": [ld, (enc(OP_DMA_LD, f0=0, f2=WPR, f3=64, nr=T), 0), wt, vw, *mm0, *out()],
        "vld of rows an unfenced vmemld writes": [ld, vw, wt, *mm0, *out()],
        "vst into rows an unfenced vmemst reads": [ld, wt, vw, *mm0, *out(), (enc(OP_VST, f0=0, f1=VO, nr=4), 0)],
        "two descriptors racing on VMEM rows": [ld, (enc(OP_DMA_LD, f0=1, f2=WPR, f3=0, nr=T), 0), (enc(OP_WAIT, f0=3, nr=1), 0), vw, *mm0, *out()],
        "vmemld beat past A | B": [(enc(OP_DMA_LD, f0=0, f1=2 * BEATS - 2, f2=1, f3=0, nr=T), 0), wt, vw, *mm0, *out()],
        # The vreg distance contract, one iteration inside it. Relative
        # to VR_RAW_DIST on purpose: whether the contract is wide enough for
        # the RTL is a question only cosim can answer (TPU_TB=stress runs
        # ar_distance_program(VR_RAW_DIST); mutate.py's `ar_claim_false`).
        **({f"ar reads at distance VR_RAW_DIST - 1 = {VR_RAW_DIST - 1}":
            ar_distance_program(VR_RAW_DIST - 1)} if VR_RAW_DIST > 1 else {}),
        **({"vst of a 1-row pop's row, 1 vpu iteration later": [
            ld, vw, wl(), wp(0, 1), wo(0, 1), (enc(OP_VST, f1=VO, nr=1), 0),
            (enc(OP_VMEMST, f3=VO, nr=1), 0)]}
           if VR_RAW_DIST > 1 else {}),
        "vmatpush consumes a vld copy of unwritten vmem": [ld, wt, (enc(OP_VLD, f0=0, f1=100, nr=4), 0), *mm0, *out()],
        # Loop semantics: the body is fine on iteration 0 and reads ar 4..7,
        # which nothing wrote, on iteration 1.
        "loop iteration 1 reads unwritten vregs": [
            ld, vw, wl(), wp(), wo(U), (enc(OP_LOOP, nr=2), 0),
            (enc(OP_VST, f0=U, f1=VO, nr=4), enc_agu((AGU_F0, 0, 4), (AGU_F1, 0, 4))),
            (enc(OP_ENDLOOP), 0)],
        "zero-row vld (hangs the machine)": [ld, wt, (enc(OP_VLD, nr=0), 0), vw, *mm0, *out()],
        "loop trip count 0": [ld, wt, vw, *mm0, (enc(OP_LOOP, nr=0), 0), *out(), (enc(OP_ENDLOOP), 0)],
        "AGU term names a closed loop": [ld, wt, vw, *mm0, (enc(OP_VST, f1=VO, nr=4), enc_agu((AGU_F1, 0, 4)))],
        "vr row past NVREG": [ld, wt, vw, wl(), wp(), wo(NVREG - 2), *out(NVREG - 2)],
        "vmemst past the last C beat": [ld, wt, vw, *mm0, *out()[:1], (enc(OP_VMEMST, f1=BEATS - 2, f2=1, f3=VO, nr=4), 0)],
        "vmemst of VMEM rows nothing wrote": [ld, wt, vw, *mm0, (enc(OP_VMEMST, f3=VO + 100, nr=4), 0)],
        "retired mvout opcode": [ld, wt, vw, *mm0, (enc(OP_MVOUT, nr=4), 0)],
        "retired dma_st opcode": [ld, wt, vw, *mm0, *out(), (enc(OP_DMA_ST, nr=4), 0)],
        "retired mm opcode": [ld, wt, vw, (enc(OP_MM, nr=4), 0), *out()],
        "unbalanced loop": [ld, wt, vw, *mm0, (enc(OP_LOOP, nr=2), 0), *out()],
        # The array's queue.
        "vmatload moving T-1 rows": [ld, wt, vw, wl(0, T - 1), wp(), wo(), *out()],
        "vmatpush with no vmatload before it": [ld, wt, vw, wp(), wo(), *out()],
        "two vmatloads with no push between": [ld, wt, vw, wl(), wl(), wp(), wo(), *out()],
        # rebalanced afterwards, so only the pop rule can reject it
        "vmatpop of more rows than are pushed": [ld, wt, vw, wl(), wp(0, 4), wo(0, 8), wp(0, 4), *out()],
        "pushed rows never popped": [ld, wt, vw, *mm0, *out(), wl(), wp()],
        "more than OUTQ rows un-popped": [ld, wt, vw, wl()] + [wp()] * (OUTQ // 4 + 1)
                                         + [wo(0, 4)] * (OUTQ // 4 + 1) + [*out()],
        "the last vmatload is never pushed": [ld, wt, vw, *mm0, *out(), wl()],
    }


def validator_controls():
    """Every crafted bad program is rejected; every program the harness ships
    or generates is accepted (the random ones are checked as they are made).
    Returns a list of failure lines."""
    fails = []
    for name, prog in _bad_programs().items():
        try:
            check_program(prog)
            fails.append(f"validator ACCEPTED a bad program: {name}")
        except ProgramError:
            pass
    good = [(g.__name__, s, r, g(*s, r)) for s in ALL_SHAPES for r in (False, True)
            for g in (gemm_program, gemm_program_flat, gemm_program_handwritten)]
    good += [("vadd_program", SCORED[-1], False, vadd_program(*SCORED[-1]))]
    good += [("vector_program", (M,), False, vector_program(M)) for M in VEC_MS]
    good += [("ar_distance_program", (d,), False, ar_distance_program(d))
             for d in (VR_RAW_DIST, VR_RAW_DIST + 1, 2 * VR_RAW_DIST)]
    for name, s, r, prog in good:
        try:
            check_program(prog)
        except ProgramError as e:
            fails.append(f"validator REJECTED {name}{s} relu={r}: {e}")
    print(f"  validator: {len(_bad_programs())} crafted bad programs rejected, "
          f"{len(good)} generated programs accepted"
          if not fails else "  validator: FAILED")
    return fails


# ----------------------------------------------------------------- runs ---
def cases(quick=False):
    """(tag, program, A, B, C0, gold, M, N) for every run, lazily."""
    crng = np.random.default_rng(1234)
    prefill = lambda: crng.integers(-128, 128, MAXDIM * MAXDIM).astype(np.int8)  # noqa: E731
    seed = 100
    shapes = SCORED if quick else ALL_SHAPES
    for (M, K, N) in shapes:
        for relu in (False, True):
            dists = ["corner"] if quick else ["full", "corner"]
            if (M, K, N) in SCORED:
                dists = ["full", "corner", "mid", "boundary"]
            for dist in dists:
                seed += 1
                if dist == "boundary":
                    A, B = boundary_operands(M, K, N, seed)
                else:
                    A, B = operands(dist, seed)
                C0 = prefill()
                prog = gemm_program(M, K, N, relu)
                gold = gemm_gold(A, B, M, K, N, relu, C0)
                # The reference model is checked against numpy on every GEMM,
                # so a disagreement below is the design's, not the model's.
                ref = isa_ref.run(prog, A, B, C0)
                assert (ref == gold).all(), f"isa_ref disagrees with numpy at {M}x{K}x{N}"
                tag = f"gemm{'.relu' if relu else ''} {M}x{K}x{N} {dist} seed={seed}"
                yield tag, prog, A, B, C0, gold, M, N
                flat = gemm_program_flat(M, K, N, relu)
                if (M, K, N) in SCORED and dist == "full" and \
                        NHDR + IWORDS * len(flat) <= IMEM_SIZE:
                    yield tag + " flat", flat, A, B, C0, gold, M, N
    for M in VEC_MS:
        for dist in ("full", "mid", "small"):
            seed += 1
            A, B = operands(dist, seed)
            C0 = prefill()
            prog = vector_program(M)
            yield (f"vector {M} {dist} seed={seed}", prog, A, B, C0,
                   isa_ref.run(prog, A, B, C0), None, None)
    for dist in ("full", "corner"):
        seed += 1
        A, B = operands(dist, seed)
        C0 = prefill()
        prog = ar_distance_program(VR_RAW_DIST)
        yield (f"ar distance {VR_RAW_DIST} {dist} seed={seed}", prog, A, B, C0,
               isa_ref.run(prog, A, B, C0), None, None)
    for s in range(8 if quick else 200):
        A, B = operands(("mid", "full", "small")[s % 3], 7000 + s)
        C0 = prefill()
        prog = random_program(7000 + s)
        yield (f"fuzz seed={7000 + s} ({len(prog)} instrs)", prog, A, B, C0,
               isa_ref.run(prog, A, B, C0), None, None)


def main(argv):
    quick = "quick" in argv
    failures = validator_controls()
    for f in failures:
        print("  STRESS FAIL", f)
    n = len(failures)
    mod = df.build(tinytpu_isa, target="simulator")
    modelled = {}
    for tag, prog, A, B, C0, gold, M, N in cases(quick):
        n += 1
        # The channel model first: a program the protocol cannot complete
        # would hang the simulator with no message, and this names why.
        key = tuple(prog)
        if key not in modelled:
            modelled[key] = kpn_model.run(prog)
        live, rep = modelled[key]
        if not live:
            bad = f"{tag}: kpn_model: " + "; ".join(rep[:4])
            failures.append(bad)
            print("  STRESS FAIL", bad, flush=True)
            continue
        bad = compare(tag, execute(mod, prog, A, B, C0), gold, M, N)
        if bad:
            failures.append(bad)
            print("  STRESS FAIL", bad, flush=True)
    print(f"  STRESS {'OK' if not failures else 'FAILED'}: "
          f"{n - len(failures)}/{n} runs exact "
          f"({'quick' if quick else 'full'}: full-range/corner/boundary "
          f"operands, prefilled C compared in full, GEMM at "
          f"{len(SCORED) if quick else len(ALL_SHAPES)} shapes, vector and "
          f"random programs)")
    return 0 if not failures else 1


if __name__ == "__main__":
    _rc = main(sys.argv[1:])
    # os._exit: see bench_isa.py -- teardown can race the OpenMP threads
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
