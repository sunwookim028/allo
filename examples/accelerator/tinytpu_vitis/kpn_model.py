# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A KPN model of tinytpu_isa's channel protocol, with bounded FIFOs and
DEADLOCK REPORTING -- which the Allo simulator does not provide.

Each unit is a generator yielding `('get', ch)` (and receiving the token) or
`('put', ch, token)`, written to mirror its `df.kernel` in `microarch_isa.py`
statement for statement on the channel side: the same header reads, the same
flat row loops fetching an instruction when the row counter runs out, the same
`vmu` header word plus T weight words per `mm`, the same weight loaders and
flat PEs. A cooperative scheduler runs them against FIFOs of depth `QD` (and
the weight queues `wq` at their fixed depth 4).

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
        for ch, w in (("c_dld", hdr[1]), ("c_dld", hdr[7]), ("c_vmu", hdr[2]),
                      ("c_vmu", hdr[4]), ("c_vru", hdr[3]), ("c_acc", hdr[5]),
                      ("c_dst", hdr[6])):
            yield ("put", ch, w)
        for op, nr, f0, f1, f2, f3 in dyn:
            if op == U.OP_DMA_LD:
                yield ("put", "c_dld", (op, nr, f0))
                yield ("put", "c_vmu", (op, nr, f0))
            elif op == U.OP_VLD:
                yield ("put", "c_vmu", (op, nr, f0))
                yield ("put", "c_vru", (op, nr, f0))
            elif op == U.OP_MM:
                # vmu's copy: its own work count T + 1, the array's rows in f1
                yield ("put", "c_vmu", (op, T + 1, nr))
                yield ("put", "c_vru", (op, nr, f0))
                yield ("put", "c_acc", (op, nr, f0))
            elif op == U.OP_VADD:
                yield ("put", "c_acc", (op, 2 * nr, f0))
            elif op == U.OP_VRELU:
                yield ("put", "c_acc", (op, nr, f0))
            elif op == U.OP_MVOUT:
                yield ("put", "c_acc", (op, nr, f0))
                yield ("put", "c_dst", (op, nr, f0))

    def flat(ctl, n_row, body):
        """The row-flattened loop every unit runs: fetch an instruction when
        the row counter runs out (its work count is the word's `nr`), then one
        row of `body`."""
        r, cnt, word = -1, 0, None
        for _ in range(n_row):
            r += 1
            if r >= cnt:
                word = yield ("get", ctl)
                cnt = word[1]
                r = 0
            yield from body(word, r)

    def dma_ld():
        n_row = (yield ("get", "c_dld")) & 0xFFFF
        yield ("get", "c_dld")                       # the A/B spans

        def body(word, r):
            yield ("put", "dma2vm", 0)
        yield from flat("c_dld", n_row, body)

    def vmu():
        n_row = (yield ("get", "c_vmu")) & 0xFFFF
        mw = yield ("get", "c_vmu")
        yield ("put", "wcol0", (mw & 0xFFFF, mw >> 16))   # the array's counts

        def body(word, r):
            op = word[0]
            if op == U.OP_DMA_LD:
                yield ("get", "dma2vm")
            elif op == U.OP_VLD:
                yield ("put", "vm2vr", 0)
            elif r == 0:
                yield ("put", "wcol0", word[2])      # header: this mm's rows
            else:
                yield ("put", "wcol0", "w")          # T weight words
        yield from flat("c_vmu", n_row, body)

    def vru():
        n_word = (yield ("get", "c_vru")) & 0xFFFF

        def body(word, r):
            op = word[0]
            if op == U.OP_MM:
                yield ("put", "acol0", 0)            # one activation word
            else:
                yield ("get", "vm2vr")
        yield from flat("c_vru", n_word, body)

    def chain_in(i, j):
        v = yield ("get", f"wcol{i}" if j == 0 else f"wrow{i}_{j - 1}")
        if j == 0 and i != T - 1:
            yield ("put", f"wcol{i + 1}", v)
        if j != T - 1:
            yield ("put", f"wrow{i}_{j}", v)
        return v

    def wld(i, j):
        nmm, nrows = yield from chain_in(i, j)
        yield ("put", f"wq{i}_{j}", nrows)                   # the PE's trip
        for _ in range(nmm):
            rows = yield from chain_in(i, j)                  # header
            if j == 0:                                        # weights
                yield ("get", f"wcol{i}")
                for _ in range(T - 1 - i):
                    v = yield ("get", f"wcol{i}")
                    yield ("put", f"wcol{i + 1}", v)
            else:
                yield ("get", f"wrow{i}_{j - 1}")
            if j != T - 1:
                yield ("put", f"wrow{i}_{j}", "w")
            yield ("put", f"wq{i}_{j}", rows)

    def pe(i, j):
        nt = yield ("get", f"wq{i}_{j}")
        r, cnt = -1, 0
        for _ in range(nt):
            r += 1
            if r >= cnt:
                cnt = yield ("get", f"wq{i}_{j}")
                r = 0
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

    def accu():
        n_row = (yield ("get", "c_acc")) & 0xFFFF

        def body(word, r):
            if word[0] == U.OP_MM:
                yield ("get", f"cw{T - 1}")
            if word[0] == U.OP_MVOUT:
                yield ("put", "ac2sp", 0)
        yield from flat("c_acc", n_row, body)

    def dma_st():
        n_row = (yield ("get", "c_dst")) & 0xFFFF

        def body(word, r):
            yield ("get", "ac2sp")
        yield from flat("c_dst", n_row, body)

    procs = {"sequencer": seq(), "dma_ld": dma_ld(), "vmu": vmu(), "vru": vru(),
             "accu": accu(), "dma_st": dma_st()}
    for i in range(T):
        for j in range(T):
            procs[f"wld{i}_{j}"] = wld(i, j)
            procs[f"pe{i}_{j}"] = pe(i, j)
    return procs


def depth(ch, QD):
    """`wq` is declared depth 4 whatever QD is (the shadow weight register)."""
    return min(QD, 4) if ch.startswith("wq") else QD


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
                    if len(q[act[1]]) >= depth(act[1], QD):
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
                           f"(occupancy {len(q[a[1]])}/{depth(a[1], QD)})")
    if left:
        rep.append(f"tokens left in channels at exit: {left} -- a unit was "
                   f"sent more than its header count promised")
    return False, rep


if __name__ == "__main__":
    from examples.accelerator.tinytpu_vitis.isa_dsl import (
        gemm_program, vector_program, ar_distance_program)
    from examples.accelerator.tinytpu_vitis.bench_isa import SHAPES
    progs = [(f"gemm{'.relu' if r else ''} {M}x{K}x{N}", gemm_program(M, K, N, r))
             for (M, K, N) in SHAPES for r in (False, True)]
    progs += [("vadd 16x16x16", U.vadd_program(16, 16, 16)),
              ("vector 8", vector_program(8)),
              (f"ar distance {U.AR_RAW_DIST}", ar_distance_program(U.AR_RAW_DIST))]
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
