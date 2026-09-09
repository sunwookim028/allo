"""Assemble and measure tiled GEMM / ReLU-MLP on CornellTPU v2.

The schedule the assembler emits is the one the machine is built for: hold a
weight tile stationary and stream a whole M-row panel of activations through it,
accumulate over K inside the array, and drain once per output column block. Each
operand is therefore read from DRAM exactly once.

Dependency tokens are attached here, not inferred by hardware: the first load of
a k-group takes a buffer credit, its last load posts a token, the group's first
compute waits on that token and its last returns the credit. That is what lets
the loader run NBUF k-groups ahead of the executor.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
_SERIAL = "--serial" in sys.argv
if _SERIAL:
    sys.argv.remove("--serial")
    from examples.accelerator.tinytpu import (  # noqa: E402
        microarch_v2_serial as m2,
    )
else:
    from examples.accelerator.tinytpu import microarch_v2 as m2  # noqa: E402
from allo.backend.rtl.sim import shell as _shell  # noqa: E402


# Verilator refuses the generated executor with
#   %Warning-UNOPTFLAT: Circular combinational logic: 'executor.r18_issue'
# and it exits on the warning. The cycle is
#   free.put's valid  ->  the FIFO's ready  ->  the issue enable  ->  valid
# i.e. a conditional stream `put` whose valid depends combinationally on the
# downstream ready. UNOPTFLAT is not a correctness failure -- Verilator settles
# such logic by iterating, and the results below are checked against numpy every
# run -- but it is a real property of the emitted RTL and would matter to a
# synthesis flow, so it is demoted here rather than fixed here, and recorded in
# FINDINGS.md. The cosim harness passes no Verilator flags of its own, hence the
# wrapper.
_orig_write_sources = _shell._write_sources


def _write_sources(*a, **k):
    srcs, build_args = _orig_write_sources(*a, **k)
    return srcs, [*build_args, "-Wno-UNOPTFLAT"]


_shell._write_sources = _write_sources

T = m2.DIM
IW = m2.IWIDTH
NBUF = m2.NBUF
MAXM = 16


def spad_layout(Kt, Nt, M):
    """Scratchpad *row* assignments. The scratchpad is DIM elements wide, so
    everything on-chip is counted in rows; only DRAM addresses are word
    addresses. ``G``/``H`` are the rows the array reads below a base and past
    the end -- its fill and drain -- which the ISA requires be reserved."""
    G, H = m2.GUARD, m2.SKEW
    apanel = [G + k * (M + H) for k in range(Kt)]       # all K panels resident
    wbase = G + Kt * (M + H)
    wslot = [wbase + b * (T + H) for b in range(NBUF)]  # double-buffered
    out = wbase + NBUF * (T + H)
    return apanel, wslot, out


def gemm_program(M, K, N, relu=False):
    """Emit the instruction stream and the DRAM layout it expects.

    Loop order is (output column block j, contraction k). The activations are
    DMA'd once and stay resident; each weight tile streams in once and is
    double-buffered against the array. So every operand crosses the DRAM bus
    exactly once, and only one accumulator block is ever live -- which is what
    lets the accumulator sit at a compile-time offset (see mmu)."""
    assert M % T == 0 and K % T == 0 and N % T == 0
    Kt, Nt = K // T, N // T
    assert M <= m2.MAXROWS, f"one mm streams at most {m2.MAXROWS} rows"
    apanel, wslot, out = spad_layout(Kt, Nt, M)

    A_BASE = 0
    B_BASE = A_BASE + M * K
    C_BASE = B_BASE + K * N
    prog, n_ld, n_ex, n_st = [], 0, 0, 0

    def emit(op, a0, a1, a2, dep=0):
        prog.extend([op, a0, a1, a2, dep])

    # Activations: one DMA per K panel, resident for the whole GEMM. These
    # post no token. The loader is strictly in-order, so every A panel has
    # landed before the first weight tile it is paired with, and giving them a
    # token would put one extra entry on `ld_tok` -- every later `loadw` would
    # then consume its *predecessor's* token and read a weight slot one DMA too
    # early. (That is a silent race: it happens to hold at 4x4x4 and 8x8x8 and
    # produces wrong results from 12x12x12 up.)
    for k in range(Kt):
        emit(m2.OP_DMA_LOAD, A_BASE + k * M * T, apanel[k], M)
        n_ld += 1
    for j in range(Nt):
        for k in range(Kt):
            b = (j * Kt + k) % NBUF
            # the first weight load also waits for the activation panels
            emit(m2.OP_DMA_LOAD, B_BASE + (k * Nt + j) * T * T, wslot[b], T,
                 m2.DEP_WAIT_FREE | m2.DEP_SIG_LOAD)
            n_ld += 1
            # one instruction per tile: latch the weight and stream the panel
            emit(m2.OP_MM0 if k == 0 else m2.OP_MM, apanel[k], wslot[b], M,
                 m2.DEP_WAIT_LOAD | m2.DEP_SIG_FREE)
            n_ex += 1
        # Each output column block drains to its own staging area. Sharing one
        # would be a write-after-read across processes -- the executor's next
        # `accst` can overwrite rows the storer has not yet pushed to DRAM, and
        # nothing in the token protocol orders those two. It shows up only from
        # Nt >= 3 (correct at 4x4x4, 8x8x8 and 16x16x8; wrong at 12x12x12 and
        # 16x16x16), and only in RTL -- the CPU backend runs the processes
        # sequentially and cannot expose it. The alternative is a credit stream
        # from the storer back to the executor; separate areas are free here
        # because Nt <= 4.
        emit(m2.OP_ACCRELU if relu else m2.OP_ACCST, 0, out + j * M, M,
             m2.DEP_SIG_EXEC)
        n_ex += 1
        emit(m2.OP_DMA_STORE, out + j * M, C_BASE + j * M * T, M,
             m2.DEP_WAIT_EXEC)
        n_st += 1
    return (np.array(prog, np.int32), n_ld, n_ex, n_st,
            (A_BASE, B_BASE, C_BASE))


def pack(M, K, N, A, B):
    """A as K-major panels (panel k = A[:, kT:(k+1)T], row-major), B as
    per-(k, j) transposed tiles, because the array consumes W as W[n][k]."""
    Kt, Nt = K // T, N // T
    a = np.concatenate([A[:, k * T:(k + 1) * T].ravel() for k in range(Kt)])
    bt = []
    for k in range(Kt):
        for j in range(Nt):
            bt.append(B[k * T:(k + 1) * T, j * T:(j + 1) * T].T.ravel())
    return a.astype(np.float32), np.concatenate(bt).astype(np.float32)


def unpack_c(M, N, flat):
    Nt = N // T
    C = np.empty((M, N), np.float32)
    for j in range(Nt):
        C[:, j * T:(j + 1) * T] = flat[j * M * T:(j + 1) * M * T].reshape(M, T)
    return C


def build_image(M, K, N, relu=False, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    prog, n_ld, n_ex, n_st, (A_B, B_B, C_B) = gemm_program(M, K, N, relu)
    ap, bp = pack(M, K, N, A, B)
    dmem = np.zeros(m2.DRAM_SIZE, np.float32)
    dmem[A_B:A_B + M * K] = ap
    dmem[B_B:B_B + K * N] = bp
    imem = np.zeros(m2.IMEM_SIZE, np.int32)
    imem[:len(prog)] = prog
    gold = (A @ B).astype(np.float32)
    if relu:
        gold = np.maximum(gold, 0.0)
    return dmem, imem, len(prog) // IW, n_ld, n_ex, n_st, C_B, gold


def check(dmem, C_B, M, N, gold):
    got = unpack_c(M, N, dmem[C_B:C_B + M * N])
    err = np.abs(got - gold).max()
    return np.allclose(got, gold, rtol=1e-4, atol=1e-4), err


_RTL = None


def rtl():
    """Compile once; every shape runs on the same hardware."""
    global _RTL
    if _RTL is None:
        _RTL = m2.export_backend("rtl")
    return _RTL


def run(mode, M, K, N, relu=False, stall=None):
    dmem, imem, ni, n_ld, n_ex, n_st, C_B, gold = build_image(M, K, N, relu)
    tag = f"{'relu-' if relu else ''}gemm {M}x{K}x{N}"
    if mode == "cpu":
        mod = m2.export_backend("cpu")
        mod(dmem, imem, ni, n_ld, n_ex, n_st)
        ok, err = check(dmem, C_B, M, N, gold)
        print(f"{tag:20s} instrs={ni:3d} (ld={n_ld} ex={n_ex} st={n_st})  "
              f"correct={ok}  maxerr={err:.2e}")
        return None
    kw = {} if stall is None else {"stall_prob": stall}
    res = rtl().cosim(dmem, imem, ni, n_ld, n_ex, n_st, **kw)
    ok, err = check(dmem, C_B, M, N, gold)
    s = "" if stall is None else f" stall={stall}"
    print(f"{tag:20s} instrs={ni:3d}  cycles={res.cycles:6d}  correct={ok}  "
          f"maxerr={err:.2e}{s}")
    return res.cycles


SHAPES = [(4, 4, 4), (8, 8, 8), (12, 12, 12), (16, 16, 8), (16, 16, 16)]


def stage_isolate(M, K, N):
    """Run each pipeline stage alone, to find the perfect-overlap bound.

    Take the real instruction stream, keep only the instructions one unit owns,
    and clear every dependency bit. Nothing then blocks, so the cycle count is
    that stage's own throughput on this workload. `max` over the three is what
    the machine would cost with a perfect scheduler; `sum` is what it would cost
    with none. Comparing the real number against both says how much is left on
    the table by *scheduling* as opposed to by throughput.

    Results are timing-only -- the isolated streams read uninitialized memory,
    so they are not checked against numpy and must not be read as correctness.
    """
    prog, _, _, _, _ = gemm_program(M, K, N)
    recs = prog.reshape(-1, IW)
    dmem0, imem0, ni, n_ld, n_ex, n_st, C_B, _ = build_image(M, K, N)
    out = {}
    sel = {
        "loader": ({m2.OP_DMA_LOAD}, lambda n: (n, 0, 0)),
        "storer": ({m2.OP_DMA_STORE}, lambda n: (0, 0, n)),
        "executor": (None, lambda n: (0, n, 0)),
    }
    for name, (ops, counts) in sel.items():
        if ops is None:
            keep = recs[~np.isin(recs[:, 0],
                                 [m2.OP_DMA_LOAD, m2.OP_DMA_STORE])].copy()
        else:
            keep = recs[np.isin(recs[:, 0], list(ops))].copy()
        keep[:, 4] = 0                      # clear every dependency bit
        imem = np.zeros(m2.IMEM_SIZE, np.int32)
        imem[:keep.size] = keep.ravel()
        dmem = dmem0.copy()
        a, b, c = counts(len(keep))
        res = rtl().cosim(dmem, imem, len(keep), a, b, c)
        out[name] = res.cycles
        print(f"  {name:9s} {len(keep):3d} instrs  {res.cycles:6d} cycles")
    seq = ni * m2.IWIDTH
    out["sequencer"] = seq
    print(f"  {'sequencer':9s} {ni:3d} instrs  {seq:6d} cycles (II={m2.IWIDTH}, computed)")
    return out

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    if len(sys.argv) > 2 and sys.argv[2] == "all":
        for sh in SHAPES:
            run(mode, *sh)
    elif len(sys.argv) > 2 and sys.argv[2] == "tall":
        # The per-instruction fixed cost of `mm` is 35 cycles against 2/row, so
        # it is amortized by streaming a longer panel. MAXROWS caps the panel,
        # and MAXROWS is set by the accumulator sitting at a compile-time
        # offset -- which is what bought the array II=1. This measures what
        # raising it is worth.
        for sh in [(16, 16, 16), (32, 16, 16), (64, 16, 16)]:
            c = run(mode, *sh)
            n = sh[0] * sh[1] * sh[2]
            print(f"    -> {c / n:.3f} cyc/MAC   ({0.0625 / (c / n) * 100:.0f}% "
                  f"of the 16-MAC/cycle roofline)")
    elif len(sys.argv) > 2 and sys.argv[2] == "stages":
        for sh in [(8, 8, 8), (16, 16, 16)]:
            real = run(mode, *sh)
            print(f"gemm {sh[0]}x{sh[1]}x{sh[2]}  measured {real}")
            st = stage_isolate(*sh)
            busy = {k: v for k, v in st.items()}
            mx, sm = max(busy.values()), sum(busy.values())
            print(f"  -> perfect-overlap bound (max stage) = {mx}")
            print(f"  -> zero-overlap bound   (sum stages) = {sm}")
            print(f"  -> measured {real}: {(sm - real) / max(sm - mx, 1) * 100:.0f}% "
                  f"of the available overlap captured\n")
    elif len(sys.argv) > 2 and sys.argv[2] == "mlp":
        for sh in [(4, 8, 8), (8, 8, 8), (8, 16, 16)]:
            run(mode, *sh, relu=True)
    elif len(sys.argv) > 2 and sys.argv[2] == "stall":
        for p in (0.0, 0.3):
            run(mode, 16, 16, 16, stall=p)
    elif len(sys.argv) > 4:
        run(mode, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
            relu=(len(sys.argv) > 5 and sys.argv[5] == "relu"))
    else:
        run(mode, 8, 8, 8)
