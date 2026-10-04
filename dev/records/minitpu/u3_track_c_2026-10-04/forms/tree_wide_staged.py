# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track C form of ``xlu_reduction_tree``: ``tree_wide`` with the RTL's
registers where the RTL has them -- two per level (``vpu_bf16_add_pipe``'s
two stages, the max path paced to it), written as data (pipe-as-data,
checkpoint 6), instead of one combinational tree followed by a delay line.

Why: the landed T1 body computes the whole ``log2 N``-level tree in one
iteration and delays the result ``1 + 2 LEVELS - 1`` rows. Catapult cannot
move a register that is written as data into the combinational cone, so the
tree's depth is scheduled on top of the delay line: the free latency is
``root + k - 1`` with ``k`` the c-steps the tree needs, and ``latency=1`` (the
value that makes the form cycle-equal) is infeasible at any clock (SCHD-30).
Here each iteration's cone is one ``add_bits`` (or the ``bf16_gt`` select),
so ``latency=1`` is the one c-step of one adder.

Row semantics are identical to T1's (same number of rows of delay per output:
``2 LEVELS`` for the root, ``2 TAP_LEVEL`` for the tap; the first edge is the
Connections Pop->Push edge, as in ``sfu.staged``): the valid pipes are reset
by ``rst_ni``, the payload never is (P-9). Level ``k`` node ``m`` (heap
numbering, ``node[N + m]``) reads its children's ``stB`` (the second stage
register) and writes ``stA``; ``stB <= stA``. Leaves are read combinationally
from the popped word (the RTL's leaf register is the Pop edge).
"""
import allo
import allo.dataflow as df
from allo.ir.types import UInt, uint8, uint16

from examples.minitpu.units import xlu_reduction_tree as X
from examples.minitpu.units.alu import bf16_gt
from examples.minitpu.units.bf16_add import add_bits


def make(n, inst="n64"):
    nl = X._n(inst)
    lanes, sub = X.GEOM[inst]
    root_d, tap_d = X.declared(inst)
    levels = nl.bit_length() - 1
    tap_level = lanes.bit_length() - 1
    tap_base = 2 * nl - 2 * sub
    WD = UInt(16 * nl)
    WL = UInt(16 * sub)
    NN = nl - 1  # internal nodes, heap index m -> node N + m
    VD = 2 * levels  # rows of delay for the root (+1 Pop->Push edge = root_d)
    TD = 2 * tap_level

    @df.region()
    def top(RST: uint8[n], VLD: uint8[n], OP: uint8[n], D: WD[n],
            VO: uint8[n], RO: uint16[n], LVO: uint8[n], LRO: WL[n]):
        @df.kernel(mapping=[1], args=[RST, VLD, OP, D, VO, RO, LVO, LRO])
        def tree(rst: uint8[n], vld: uint8[n], op: uint8[n], d: WD[n],
                 vo: uint8[n], ro: uint16[n], lvo: uint8[n], lro: WL[n]):
            # per internal node: stage A (adder stage 1 / max stage 1) and stage B
            stA: uint16[NN]
            stB: uint16[NN]
            opA: uint8[NN]  # the op travels with the wavefront (op_q per level)
            opB: uint8[NN]
            vq: uint8[VD]  # valid pipe to the root: reset
            lvq: uint8[TD]
            for k0 in range(VD):
                vq[k0] = 0
            for k1 in range(TD):
                lvq[k1] = 0
            for t in range(n):
                r: uint8 = rst[t]
                v: uint8 = vld[t]
                o: uint8 = op[t]
                w: WD = d[t]
                # outputs first: the root's stage B and the tap's stage B of the
                # previous iteration (the RTL's register outputs)
                lw: WL = 0
                with allo.meta_for(sub) as s:
                    lw[16 * s : 16 * s + 16] = stB[tap_base - nl + s]
                lro[t] = lw
                ro[t] = stB[NN - 1]
                vo[t] = vq[VD - 1]
                lvo[t] = lvq[TD - 1]
                # stage B <= stage A, every node
                for m0 in range(NN):
                    stB[m0] = stA[m0]
                    opB[m0] = opA[m0]
                # stage A <= f(children's stage B), levels above the leaves (nodes
                # whose children are internal): children of node N+m are 2m, 2m+1;
                # internal when 2m >= N
                for m1 in range(nl // 2, NN):
                    a: uint16 = stB[2 * m1 - nl]
                    b: uint16 = stB[2 * m1 + 1 - nl]
                    oc: uint8 = opB[2 * m1 - nl]
                    if oc == 0:
                        stA[m1] = add_bits(a, b)
                    else:
                        if bf16_gt(a, b):
                            stA[m1] = a
                        else:
                            stA[m1] = b
                    opA[m1] = oc
                # the leaf level: children are the popped word's lanes
                with allo.meta_for(nl // 2) as m:
                    a0: uint16 = w[32 * m : 32 * m + 16]
                    b0: uint16 = w[32 * m + 16 : 32 * m + 32]
                    if o == 0:
                        stA[m] = add_bits(a0, b0)
                    else:
                        if bf16_gt(a0, b0):
                            stA[m] = a0
                        else:
                            stA[m] = b0
                    opA[m] = o
                # valid pipes: shift, load, synchronous clear
                for k2 in range(1, VD):
                    vq[VD - k2] = vq[VD - k2 - 1]
                vq[0] = v
                for j in range(1, TD):
                    lvq[TD - j] = lvq[TD - j - 1]
                lvq[0] = v
                if r == 0:
                    for k3 in range(VD):
                        vq[k3] = 0
                    for k4 in range(TD):
                        lvq[k4] = 0

    return top
