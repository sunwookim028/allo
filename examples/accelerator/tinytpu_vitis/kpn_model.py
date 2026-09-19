# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A KPN model of tinytpu_isa's channel protocol, with bounded FIFOs and
DEADLOCK REPORTING -- which the Allo simulator does not provide.

Each unit is a generator yielding `('get', ch)` (and receiving the token) or
`('put', ch, token)`, written to mirror its `df.kernel` in `microarch_isa.py`
statement for statement on the channel side: the same header reads, the same
flat row loops fetching an instruction when the row counter runs out, the same
`vru` prologue of one header word plus T weight words per `mm`, the same PE
chains. A cooperative scheduler runs them against FIFOs of depth `QD`.

It is driven by the ASSEMBLED program -- the header words `assemble()` writes
and the dynamic stream the sequencer dispatches -- so it checks the one place
the assembler and the hardware are coupled: a unit promised more work than it
is sent waits forever, one promised less leaves tokens behind. The simulator
shows either as a silent hang or a silent wrong answer; this names the blocked
process, the channel, and its occupancy.

    python kpn_model.py        # every shipped program: minimum depth, then QD

Data values are not modelled; `stress_isa.py` and `bench_isa.py` do that.
"""

import collections
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from examples.accelerator.tinytpu_vitis import microarch_isa as U  # noqa: E402

T = U.T


def build(prog):
    words = U.assemble(prog)
    hdr = words[:U.NHDR]
    dyn = U.expand(prog)

    def seq():
        for ch, w in (("c_dld", hdr[1]), ("c_dld", hdr[7]), ("c_spm", hdr[2]),
                      ("c_vru", hdr[3]), ("c_vru", hdr[4]), ("c_acc", hdr[5]),
                      ("c_dst", hdr[6])):
            yield ("put", ch, w)
        route = {U.OP_DMA_LD: ("c_dld", "c_spm"), U.OP_VLD: ("c_spm", "c_vru"),
                 U.OP_MM: ("c_vru", "c_acc"), U.OP_VADD: ("c_acc",),
                 U.OP_VRELU: ("c_acc",), U.OP_MVOUT: ("c_acc", "c_dst")}
        for op, nr, *_ in dyn:
            for ch in route.get(op, ()):
                yield ("put", ch, (op, nr))

    def flat(ctl, n_row, body):
        """The row-flattened loop every unit but `accu` runs: fetch an
        instruction when the row counter runs out, then one row of `body`.
        `vru` charges an `mm` T + 1 extra words (header + weights)."""
        r, cnt, op, nr = -1, 0, None, 0
        for _ in range(n_row):
            r += 1
            if r >= cnt:
                op, nr = yield ("get", ctl)
                cnt = nr + (T + 1 if (ctl == "c_vru" and op == U.OP_MM) else 0)
                r = 0
            yield from body(op, nr, r)

    def dma_ld():
        n_row = (yield ("get", "c_dld")) & 0xFFFF
        yield ("get", "c_dld")                       # the A/B spans

        def body(op, nr, r):
            yield ("put", "dma2sp", 0)
        yield from flat("c_dld", n_row, body)

    def spm():
        n_row = (yield ("get", "c_spm")) & 0xFFFF

        def body(op, nr, r):
            if op == U.OP_DMA_LD:
                yield ("get", "dma2sp")
            else:
                yield ("put", "sp2vr", 0)
        yield from flat("c_spm", n_row, body)

    def vru():
        n_word = (yield ("get", "c_vru")) & 0xFFFF
        mw = yield ("get", "c_vru")
        yield ("put", "wcol0", mw & 0xFFFF)          # the array's mm count

        def body(op, nr, r):
            if op == U.OP_VLD:
                yield ("get", "sp2vr")
            elif r == 0:
                yield ("put", "wcol0", nr)           # header: this mm's rows
            elif r <= T:
                yield ("put", "wcol0", "w")          # T weight words
            else:
                yield ("put", "acol0", 0)            # one activation word
        yield from flat("c_vru", n_word, body)

    def pe(i, j):
        def chain_in():
            v = yield ("get", f"wcol{i}" if j == 0 else f"wrow{i}_{j - 1}")
            if j == 0 and i != T - 1:
                yield ("put", f"wcol{i + 1}", v)
            if j != T - 1:
                yield ("put", f"wrow{i}_{j}", v)
            return v

        def g():
            nmm = yield from chain_in()
            for _ in range(nmm):
                nrows = yield from chain_in()                    # header
                if j == 0:                                       # weights
                    ww = yield ("get", f"wcol{i}")
                    for _ in range(T - 1 - i):
                        v = yield ("get", f"wcol{i}")
                        yield ("put", f"wcol{i + 1}", v)
                else:
                    ww = yield ("get", f"wrow{i}_{j - 1}")
                if j != T - 1:
                    yield ("put", f"wrow{i}_{j}", ww)
                for _ in range(nrows):
                    if j == 0:
                        yield ("get", f"acol{i}")
                        if i != T - 1:
                            yield ("put", f"acol{i + 1}", 0)
                    else:
                        yield ("get", f"a_fwd{i}_{j - 1}")
                    if i > 0:
                        yield ("get", f"p_fwd{i - 1}_{j}")
                    if i != T - 1:
                        yield ("put", f"p_fwd{i}_{j}", 0)
                    else:
                        if j > 0:
                            yield ("get", f"cw{j - 1}")
                        yield ("put", f"cw{j}", 0)
                    if j != T - 1:
                        yield ("put", f"a_fwd{i}_{j}", 0)
        return g()

    def accu():
        n_own = (yield ("get", "c_acc")) & 0xFFFF
        for _ in range(n_own):
            op, nr = yield ("get", "c_acc")
            for _ in range(nr):
                if op == U.OP_MM:
                    yield ("get", f"cw{T - 1}")
                if op == U.OP_MVOUT:
                    yield ("put", "ac2sp", 0)

    def dma_st():
        n_row = (yield ("get", "c_dst")) & 0xFFFF

        def body(op, nr, r):
            yield ("get", "ac2sp")
        yield from flat("c_dst", n_row, body)

    procs = {"sequencer": seq(), "dma_ld": dma_ld(), "spm": spm(), "vru": vru(),
             "accu": accu(), "dma_st": dma_st()}
    for i in range(T):
        for j in range(T):
            procs[f"pe{i}_{j}"] = pe(i, j)
    return procs


def run(prog, QD=U.QD):
    """-> (True, None) if the program runs to completion with every channel
    drained, else (False, report lines)."""
    procs = build(prog)
    q = collections.defaultdict(collections.deque)
    pending = {n: None for n in procs}
    recv = {n: None for n in procs}
    done = set()
    progress = True
    while progress:
        progress = False
        for n, gen in procs.items():
            if n in done:
                continue
            while True:
                if pending[n] is None:
                    try:
                        pending[n] = gen.send(recv[n])
                        recv[n] = None
                    except StopIteration:
                        done.add(n)
                        progress = True
                        break
                act = pending[n]
                if act[0] == "get":
                    if not q[act[1]]:
                        break
                    recv[n] = q[act[1]].popleft()
                else:
                    if len(q[act[1]]) >= QD:
                        break
                    q[act[1]].append(act[2])
                pending[n] = None
                progress = True
    left = {ch: len(v) for ch, v in q.items() if v}
    if len(done) == len(procs) and not left:
        return True, None
    rep = []
    if len(done) != len(procs):
        rep.append(f"DEADLOCK at QD={QD}: {len(procs) - len(done)} processes blocked")
        for n in procs:
            if n not in done:
                a = pending[n]
                rep.append(f"  {n:10s} blocked on {a[0]:3s} {a[1]:10s} "
                           f"(occupancy {len(q[a[1]])}/{QD})")
    if left:
        rep.append(f"tokens left in channels at exit: {left} -- a unit was "
                   f"sent more than its header count promised")
    return False, rep


if __name__ == "__main__":
    from examples.accelerator.tinytpu_vitis.isa_dsl import gemm_program, vector_program
    from examples.accelerator.tinytpu_vitis.bench_isa import SHAPES
    progs = [(f"gemm{'.relu' if r else ''} {M}x{K}x{N}", gemm_program(M, K, N, r))
             for (M, K, N) in SHAPES for r in (False, True)]
    progs += [("vadd 16x16x16", U.vadd_program(16, 16, 16)),
              ("vector 8", vector_program(8))]
    ok = True
    for name, prog in progs:
        lo = next((d for d in (1, 2, 3, 4, 8) if run(prog, d)[0]), None)
        good, rep = run(prog, U.QD)
        ok &= good
        print(f"  {name:22s} minimum depth {lo}   at QD={U.QD}: "
              f"{'ok' if good else 'FAIL'}")
        for line in rep or []:
            print("    " + line)
    print("  KPN OK" if ok else "  KPN FAILED")
    sys.exit(0 if ok else 1)
