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
  * **shapes**: every multiple of T up to MAXDIM in every dimension, not the
    scored ones, so nothing special-cased to them survives -- as a seeded
    stratified sample once the exhaustive set passes `TPU_STRESS_SHAPES`
    (4096 shapes at MAXDIM=64), keeping the scored shapes and every extreme;
  * **`C` prefilled with random bytes**, and the WHOLE of `C` compared: the
    result region must be overwritten and everything outside it untouched, so
    a design may not rely on `C` arriving zeroed nor scribble past `M x N`;
  * **programs other than GEMM**: `isa_dsl.vector_program`, which varies
    every field GEMM leaves constant (DRAM row, accumulator region, distinct
    `vadd`/`vrelu` destinations), and random valid programs, both against
    `isa_ref.run` -- the ISA as numpy;
  * **many invocations of one build**, so a later run sees the state an
    earlier one left in `spad`/`vr`/`ar`, which nothing clears;
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
    OP_LOOP, OP_ENDLOOP, AGU_F0, AGU_F1,
    MAXDIM, T, WPR, IMEM_SIZE, NHDR, IWORDS, SPAD_ROWS, NVR, NAR, AR_RAW_DIST,
)
from examples.accelerator.tinytpu_vitis.isa_dsl import (  # noqa: E402
    Program, Ref, gemm_program, vector_program, ar_distance_program,
)
from examples.accelerator.tinytpu_vitis import isa_ref, kpn_model  # noqa: E402
from examples.accelerator.tinytpu_vitis.bench_isa import (  # noqa: E402
    LATENCY, STEADY, runnable,
)

SCORED = LATENCY + [s for s in runnable(STEADY) if s not in LATENCY]
# Every multiple of T in every dimension -- the point being that nothing
# special-cased to the scored shapes survives.
#
# **That set is cubic in MAXDIM/T**: 64 shapes at MAXDIM=16 but 4096 at
# MAXDIM=64, times two relu settings and two operand distributions, and the
# simulator pass is the expensive one. Above a budget the set becomes a
# DETERMINISTIC stratified sample instead -- seeded, so a failure reproduces --
# that always keeps the scored shapes and every extreme (each dimension at its
# smallest and largest). Set TPU_STRESS_SHAPES=0 for the exhaustive set.
_EXHAUSTIVE = [s for s in itertools.product(range(T, MAXDIM + 1, T), repeat=3)]
SHAPE_BUDGET = int(os.environ.get("TPU_STRESS_SHAPES", 96))
if SHAPE_BUDGET and len(_EXHAUSTIVE) > SHAPE_BUDGET:
    _ends = (T, MAXDIM)
    _keep = {s for s in _EXHAUSTIVE if all(d in _ends for d in s)}
    _keep |= {s for s in SCORED if s in _EXHAUSTIVE}
    _rest = sorted(set(_EXHAUSTIVE) - _keep)
    _rng = np.random.default_rng(20260922)
    _pick = _rng.choice(len(_rest), max(0, SHAPE_BUDGET - len(_keep)),
                        replace=False)
    ALL_SHAPES = sorted(_keep | {_rest[i] for i in _pick})
else:
    ALL_SHAPES = _EXHAUSTIVE
MAX_STATIC = (IMEM_SIZE - NHDR) // IWORDS
# `vector_program`'s M. Written as T and 2T, which is what the literal (4, 8)
# meant at T=4: the program's third `mm` reads T weight rows out of the region
# its M-row `dma_ld` filled, so M < T loads nothing into the tail. Dropped
# entirely on a build the program's fixed addresses do not fit (MAXDIM//T < 4),
# where `vector_program` refuses to be built at all.
VECTOR_M = [M for M in (T, 2 * T) if 2 * M <= MAXDIM] if MAXDIM // T >= 4 \
    and 5 + 2 * T <= MAXDIM else []
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
    """A random program that is valid BY CONSTRUCTION: every read names rows
    an earlier instruction wrote, and every `ar` read keeps the accumulator
    distance contract (`AR_RAW_DIST` accu iterations after the write). Memory
    windows are either low or at the top of each memory, so both overlapping
    reuse and the last rows get hit."""
    rng = np.random.default_rng(seed)
    k = Program(f"fuzz {seed}")
    ri = lambda lo, hi: int(rng.integers(lo, hi + 1))   # noqa: E731
    hi_win = bool(rng.integers(0, 2))
    lo = {m: (s - 64 if hi_win else 0)
          for m, s in (("spad", SPAD_ROWS), ("vr", NVR), ("ar", NAR))}
    wr = {"spad": np.zeros(SPAD_ROWS, bool), "vr": np.zeros(NVR, bool),
          "ar": np.zeros(NAR, bool)}
    ar_at = np.full(NAR, -AR_RAW_DIST)   # accu iteration of each row's last write
    acc = {"it": 0}                      # accu iterations issued so far

    def place(mem, n):
        return lo[mem] + ri(0, 64 - n)

    def run_of(mem, n):
        ok = [s for s in range(lo[mem], lo[mem] + 64 - n + 1)
              if wr[mem][s:s + n].all()]
        return int(rng.choice(ok)) if ok else None

    def accu_ok(reads):
        """reads: (ar row, accu iteration) pairs, iterations counted from the
        instruction's first. True if each keeps the distance contract."""
        return all(acc["it"] + t - ar_at[row] >= AR_RAW_DIST for row, t in reads)

    def accu_commit(writes, n_it):
        for row, t in writes:
            wr["ar"][row] = True
            ar_at[row] = acc["it"] + t
        acc["it"] += n_it

    def dma_ld(to=None, n=None):
        n = n or ri(1, MAXDIM)
        to = to or ("vr" if rng.integers(0, 2) else "spad")
        s = place(to, n)
        k.dma_ld(src=ri(0, 1), dram_row=ri(0, MAXDIM - n),
                 col_block=ri(0, WPR - 1), rows=n, **{to: s})
        wr[to][s:s + n] = True

    def vld():
        n = ri(1, 24)
        s = run_of("spad", n)
        if s is None:
            return False
        d = place("vr", n)
        k.vld(d, s, rows=n)
        wr["vr"][d:d + n] = True
        return True

    def mm():
        n = ri(1, MAXDIM)
        w, a = run_of("spad", T), run_of("vr", n)
        if w is None or a is None:
            return False
        d = run_of("ar", n) if rng.integers(0, 2) else None   # accumulate onto
        d = place("ar", n) if d is None else d                # a live region?
        acc_ = (bool(wr["ar"][d:d + n].all()) and bool(rng.integers(0, 2))
                and accu_ok([(d + i, i) for i in range(n)]))
        k.mm(a, d, w, rows=n, acc=acc_)
        accu_commit([(d + i, i) for i in range(n)], n)
        return True

    def vec(op):
        n = ri(1, 12)
        s1, s2 = run_of("ar", n), run_of("ar", n)
        if s1 is None:
            return False
        d = place("ar", n)
        if op == "vadd":
            reads = [(r, t) for i in range(n)
                     for r, t in ((s1 + i, 2 * i), (s2 + i, 2 * i + 1))]
            writes, n_it = [(d + i, 2 * i + 1) for i in range(n)], 2 * n
        else:
            reads = [(s1 + i, i) for i in range(n)]
            writes, n_it = [(d + i, i) for i in range(n)], n
        # a read of a row this same instruction wrote earlier counts too
        ok = True
        for r, t in reads:
            last = max([wt for wrow, wt in writes if wrow == r and wt < t],
                       default=None)
            if last is not None and t - last < AR_RAW_DIST:
                ok = False
        if not ok or not accu_ok(reads):
            return False
        if op == "vadd":
            k.vadd(d, s1, s2, rows=n)
        else:
            k.vrelu(d, s1, rows=n)
        accu_commit(writes, n_it)
        return True

    def mvout():
        n = ri(1, MAXDIM)
        s = run_of("ar", n)
        if s is None or not accu_ok([(s + i, i) for i in range(n)]):
            return False
        k.mvout(s, dram_row=ri(0, MAXDIM - n), col_block=ri(0, WPR - 1), rows=n)
        acc["it"] += n
        return True

    def mvout_loop():
        # the AGU on three fields of one instruction at once
        trip, n = ri(2, 3), ri(1, 4)
        s = run_of("ar", trip * n)
        if s is None or not accu_ok([(s + i, i) for i in range(trip * n)]):
            return False
        with k.loop(trip) as i:
            k.mvout(Ref(s).at(i, n), dram_row=Ref(0).at(i, n),
                    col_block=Ref(0).at(i, 1), rows=n)
        acc["it"] += trip * n
        return True

    dma_ld("spad", ri(T, MAXDIM))        # weights
    dma_ld("vr")                         # activations
    for _ in range(ri(0, 2)):
        dma_ld()
    while not mm():
        dma_ld()
    ops = [dma_ld, vld, mm, lambda: vec("vadd"), lambda: vec("vrelu"),
           mvout, mvout_loop]
    budget = ri(8, MAX_STATIC - 4)
    while len(k.words) < budget:
        ops[int(rng.integers(0, len(ops)))]()
    mvout()
    prog = k.emit()
    assert len(prog) <= MAX_STATIC
    check_program(prog)
    return prog


# ------------------------------------------- the validator, both directions ---
def _bad_programs():
    """Programs `check_program` must REJECT, each breaking one rule. Raw
    `enc` words where the DSL would refuse first, so the check under test is
    the assembler's and not the generator's."""
    ld = (enc(OP_DMA_LD, f0=0, f3=0, nr=4), 0)
    vw = (enc(OP_VLD, f0=0, f1=0, nr=4), 0)
    mm0 = (enc(OP_MM, f0=0, f1=0, f2=0, f3=0, nr=4), 0)
    out = lambda a=0: (enc(OP_MVOUT, f0=a, nr=4), 0)  # noqa: E731
    return {
        "mvout before any write to ar": [ld, out()],
        "mm accumulate onto unwritten ar": [ld, vw, (enc(OP_MM, f2=1, nr=4), 0), out()],
        "vadd second source unwritten": [ld, vw, mm0, (enc(OP_VADD, f0=8, f1=0, f2=4, nr=4), 0), out(8)],
        "vrelu source unwritten": [ld, vw, mm0, (enc(OP_VRELU, f0=0, f1=20, nr=4), 0), out()],
        "mm weights never loaded into spad": [ld, vw, (enc(OP_MM, f3=40, nr=4), 0), out()],
        "mm activations only in spad, never in vr": [ld, (enc(OP_MM, f0=8, nr=4), 0), out()],
        "dma_ld f0 outside source|destination": [(enc(OP_DMA_LD, f0=4, nr=4), 0), ld, vw, mm0, out()],
        # The accumulator distance contract, one iteration inside it. Relative
        # to AR_RAW_DIST on purpose: whether the contract is wide enough for
        # the RTL is a question only cosim can answer (TPU_TB=stress runs
        # ar_distance_program(AR_RAW_DIST); mutate.py's `ar_claim_false`).
        **({f"ar reads at distance AR_RAW_DIST - 1 = {AR_RAW_DIST - 1}":
            ar_distance_program(AR_RAW_DIST - 1)} if AR_RAW_DIST > 1 else {}),
        **({"mvout of a 1-row mm's row, 1 accu iteration later": [
            ld, vw, (enc(OP_MM, nr=1), 0), (enc(OP_MVOUT, nr=1), 0)]}
           if AR_RAW_DIST > 1 else {}),
        "mm consumes a vld copy of unwritten spad": [ld, (enc(OP_VLD, f0=0, f1=100, nr=4), 0), mm0, out()],
        # Loop semantics: the body is fine on iteration 0 and reads ar 4..7,
        # which nothing wrote, on iteration 1.
        "loop iteration 1 reads unwritten ar": [
            ld, vw, mm0, (enc(OP_LOOP, nr=2), 0),
            (enc(OP_MVOUT, f0=0, f1=0, nr=4), enc_agu((AGU_F0, 0, 4), (AGU_F1, 0, 4))),
            (enc(OP_ENDLOOP), 0)],
        "zero-row vld (hangs the machine)": [ld, (enc(OP_VLD, nr=0), 0), vw, mm0, out()],
        "loop trip count 0": [ld, vw, mm0, (enc(OP_LOOP, nr=0), 0), out(), (enc(OP_ENDLOOP), 0)],
        "AGU term names a closed loop": [ld, vw, mm0, (enc(OP_MVOUT, nr=4), enc_agu((AGU_F1, 0, 4)))],
        "ar row past NAR": [ld, vw, (enc(OP_MM, f1=NAR - 2, nr=4), 0), out(NAR - 2)],
        "mvout past the last C row": [ld, vw, mm0, (enc(OP_MVOUT, f1=MAXDIM - 2, nr=4), 0)],
        "retired dma_st opcode": [ld, vw, mm0, out(), (enc(OP_DMA_ST, nr=4), 0)],
        "unbalanced loop": [ld, vw, mm0, (enc(OP_LOOP, nr=2), 0), out()],
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
    good += [("vector_program", (M,), False, vector_program(M))
             for M in VECTOR_M]
    good += [("ar_distance_program", (d,), False, ar_distance_program(d))
             for d in (AR_RAW_DIST, AR_RAW_DIST + 1, 2 * AR_RAW_DIST)]
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
    for M in VECTOR_M:
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
        prog = ar_distance_program(AR_RAW_DIST)
        yield (f"ar distance {AR_RAW_DIST} {dist} seed={seed}", prog, A, B, C0,
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
    sys.exit(main(sys.argv[1:]))
