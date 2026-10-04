# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track C form of ``xlu_reduction_tree`` T1 ``bits``: the landed body with the
lane arrays packed into one word per cycle (``data_i`` as ``UInt(16 N)[n]``,
``lane_result_o`` as ``UInt(16 SUB)[n]``), the RTL's own port shape.

The landed form's ``uint16[n, N]`` ports are 2-D, and the SystemC emitter
streams only a rank-1 array (``isSeqStreamable``): a 2-D boundary array
becomes RAM pins (``_radr/_re/_q``), which is a memory the integrator has to
provide, not a unit port -- so the landed form has no Connections RTL to hold
to the oracle per cycle (finding C1). ``make(n, inst)``: the body, loops and
pipes are ``xlu_reduction_tree.bits``'s; only the two ports and their
(un)packing (``allo.meta_for``, compile-time constant slices) differ.
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
    tap_base = 2 * nl - 2 * sub
    WD = UInt(16 * nl)
    WL = UInt(16 * sub)

    @df.region()
    def top(RST: uint8[n], VLD: uint8[n], OP: uint8[n], D: WD[n],
            VO: uint8[n], RO: uint16[n], LVO: uint8[n], LRO: WL[n]):
        @df.kernel(mapping=[1], args=[RST, VLD, OP, D, VO, RO, LVO, LRO])
        def tree(rst: uint8[n], vld: uint8[n], op: uint8[n], d: WD[n],
                 vo: uint8[n], ro: uint16[n], lvo: uint8[n], lro: WL[n]):
            node: uint16[2 * nl - 1]
            vq: uint8[root_d]
            rq: uint16[root_d]
            lvq: uint8[tap_d]
            lrq: uint16[tap_d, sub]
            for k0 in range(root_d):
                vq[k0] = 0
            for k1 in range(tap_d):
                lvq[k1] = 0
            for t in range(n):
                r: uint8 = rst[t]
                v: uint8 = vld[t]
                o: uint8 = op[t]
                w: WD = d[t]
                with allo.meta_for(nl) as l:
                    node[l] = w[16 * l : 16 * l + 16]
                for m in range(nl - 1):
                    a: uint16 = node[2 * m]
                    b: uint16 = node[2 * m + 1]
                    if o == 0:
                        node[nl + m] = add_bits(a, b)
                    else:
                        if bf16_gt(a, b):
                            node[nl + m] = a
                        else:
                            node[nl + m] = b
                for k2 in range(1, root_d):
                    vq[root_d - k2] = vq[root_d - k2 - 1]
                    rq[root_d - k2] = rq[root_d - k2 - 1]
                vq[0] = v
                rq[0] = node[2 * nl - 2]
                for j in range(1, tap_d):
                    lvq[tap_d - j] = lvq[tap_d - j - 1]
                    for s0 in range(sub):
                        lrq[tap_d - j, s0] = lrq[tap_d - j - 1, s0]
                lvq[0] = v
                for s1 in range(sub):
                    lrq[0, s1] = node[tap_base + s1]
                if r == 0:
                    for k3 in range(root_d):
                        vq[k3] = 0
                    for k4 in range(tap_d):
                        lvq[k4] = 0
                lw: WL = 0
                with allo.meta_for(sub) as s:
                    lw[16 * s : 16 * s + 16] = lrq[tap_d - 1, s]
                lro[t] = lw
                vo[t] = vq[root_d - 1]
                ro[t] = rq[root_d - 1]
                lvo[t] = lvq[tap_d - 1]

    return top
