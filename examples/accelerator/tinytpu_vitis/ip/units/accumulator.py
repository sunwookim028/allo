# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The accumulator vector registers, their sole owner, and the vector ALU.
Its II=1 rests on a dependence claim the assembler makes true; see
``docs/source/designs/tinytpu_isa.rst``."""

from __future__ import annotations

from ..compose import unit


def accumulator_directives(s, ctx):
    # `#pragma HLS dependence variable=ar inter false` on the row loop. Vitis
    # alone closes it at II=3; the claim is true only for programs that meet
    # the assembler's accumulator distance contract.
    s.dependence(f"{ctx.instance('accu')}:x", "ar", dep_type="inter",
                 dependent=False)


@unit(
    reads=("c_acc", "cw"),
    writes=("ac2sp",),
    parameters=("NAR", "AW", "VW", "T"),
    isa=("OP_MM", "OP_VADD", "OP_VRELU", "OP_MVOUT"),
    directives=accumulator_directives,
)
def accu():
    ar: UInt(AW)[NAR]
    nw: UInt(64) = c_acc.get()
    n_row: int32 = nw[0:16]
    op: int32 = 0
    f0: int32 = 0
    f1: int32 = 0
    f2: int32 = 0
    cnt: int32 = 0
    r: int32 = -1               # advanced at the TOP: see the II note
    xr: UInt(AW) = 0            # vadd's first operand, held for one row
    for x in range(n_row):
        r += 1
        if r >= cnt:
            w0: UInt(64) = c_acc.get()
            op = w0[0:6]
            f0 = w0[6:18]
            f1 = w0[18:30]
            f2 = w0[30:42]
            cnt = w0[54:62]
            r = 0
        rr: int32 = r
        ph: int32 = 0
        if op == OP_VADD:
            rr = r >> 1
            ph = r - (rr << 1)
        ra: int32 = f1 + rr
        wa: int32 = f0 + rr
        if op == OP_MM:
            wa = f1 + rr
        if op == OP_MVOUT:
            ra = f0 + rr
        if op == OP_VADD:
            if ph == 1:
                ra = f2 + rr
        rv: UInt(AW) = ar[ra]
        z: UInt(AW) = 0
        dw: int32 = 1
        if op == OP_MM:
            v: UInt(AW) = cw[T - 1].get()
            base: UInt(AW) = 0
            if f2 == 1:
                base = rv
            with allo.meta_for(T) as e:
                be: int32 = base[32 * e : 32 * (e + 1)]
                ve: int32 = v[32 * e : 32 * (e + 1)]
                se: int32 = be + ve
                z[32 * e : 32 * (e + 1)] = se
        elif op == OP_VADD:
            if ph == 0:
                xr = rv
                dw = 0
            else:
                with allo.meta_for(T) as e2:
                    xe: int32 = xr[32 * e2 : 32 * (e2 + 1)]
                    ye: int32 = rv[32 * e2 : 32 * (e2 + 1)]
                    xy: int32 = xe + ye
                    z[32 * e2 : 32 * (e2 + 1)] = xy
        elif op == OP_VRELU:
            with allo.meta_for(T) as e3:
                ue: int32 = rv[32 * e3 : 32 * (e3 + 1)]
                re: int32 = ue
                if re < 0:
                    re = 0
                z[32 * e3 : 32 * (e3 + 1)] = re
        else:
            dw = 0
            ow: UInt(VW) = 0
            with allo.meta_for(T) as e4:
                te: int32 = rv[32 * e4 : 32 * (e4 + 1)]
                if te > 127:
                    te = 127
                if te < -128:
                    te = -128
                tc: int8 = te
                ow[8 * e4 : 8 * (e4 + 1)] = tc
            ac2sp.put(ow)
        if dw == 1:
            ar[wa] = z
