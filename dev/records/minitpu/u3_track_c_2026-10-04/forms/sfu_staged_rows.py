# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track C form of ``sfu`` S2 ``staged``: the landed body with its port
accesses unconditional -- one Pop of each input and one Push of the result per
iteration over exactly ``n`` rows, so output row ``t`` is the result of input
row ``t - 4`` (rows 0-3 are the pipeline's initial state).

Why: the landed S2 warms up and drains (``if t < n: read``, ``if t >= 4:
write`` over ``n + 4`` iterations), and the SystemC emitter streams a boundary
array only when its one access sits directly in one loop (``isSeqStreamable``):
a conditional access makes the array RAM pins, so ``latency=`` is refused
("no Connections In/Out pair") and there is no stream RTL to hold to the
``valid``-shape oracle (finding C2). The comparator shifts by 4 rows; the
contract's 5 = 4 rows + the Pop->Push edge (``latency=1``).
"""
import allo.dataflow as df
from allo.ir.types import int32, uint8, uint16

from examples.minitpu.units import sfu as S

ROWS = 4  # rows of delay carried by the registers written as data


def make(n, inst=None):
    GELU_ROM, EXP_ROM, RECIP_PWL, RSQRT_PWL = S.GELU_ROM, S.EXP_ROM, S.RECIP_PWL, S.RSQRT_PWL
    fp32_abs_q7 = S.fp32_abs_q7

    @df.region()
    def top(OP: uint8[n], X: uint16[n], R: uint16[n]):
        @df.kernel(mapping=[1], args=[OP, X, R])
        def sfu_k(opv: uint8[n], xv: uint16[n], rv: uint16[n]):
            rom_gelu: int32[2048] = GELU_ROM
            rom_exp: int32[2048] = EXP_ROM
            recip_pwl: int32[32] = RECIP_PWL
            rsqrt_pwl: int32[32] = RSQRT_PWL
            in_op_q: int32 = 0
            in_x_q: int32 = 0
            s1_op_q: int32 = 0
            s1_x_q: int32 = 0
            s1_mag_q: int32 = 0
            s1_gaddr_q: int32 = 0
            s1_eaddr_q: int32 = 0
            s1_raddr_q: int32 = 0
            s1_saddr_q: int32 = 0
            s1_rpos_q: int32 = 0
            s1_spos_q: int32 = 0
            s2_op_q: int32 = 0
            s2_x_q: int32 = 0
            s2_mag_q: int32 = 0
            s2_gword_q: int32 = 0
            s2_eword_q: int32 = 0
            s2_rbias_q: int32 = 0
            s2_sbias_q: int32 = 0
            s2_rprod_q: int32 = 0
            s2_sprod_q: int32 = 0
            s2_sexp_q: int32 = 0
            s3_op_q: int32 = 0
            s3_x_q: int32 = 0
            s3_mag_q: int32 = 0
            s3_gword_q: int32 = 0
            s3_eword_q: int32 = 0
            s3_rint_q: int32 = 0
            s3_sint_q: int32 = 0
            s3_sexp_q: int32 = 0
            for t in range(n):
                op_in: int32 = opv[t]
                x_in: int32 = xv[t]
                x3: int32 = s3_x_q
                sign: int32 = (x3 >> 15) & 1
                e: int32 = (x3 >> 7) & 0xFF
                frac: int32 = x3 & 0x7F
                is_nan: int32 = 1 if (e == 0xFF and frac != 0) else 0
                is_inf: int32 = 1 if (e == 0xFF and frac == 0) else 0
                is_zero: int32 = 1 if (x3 & 0x7FFF) == 0 else 0
                res: int32 = 0x7FC0
                if s3_op_q == 0:
                    if is_nan != 0:
                        res = 0x7FC0
                    elif is_zero != 0:
                        res = 0
                    elif s3_mag_q >= 1024:
                        res = 0 if sign == 1 else x3
                    else:
                        res = s3_gword_q
                elif s3_op_q == 1:
                    if is_nan != 0:
                        res = 0x7FC0
                    elif is_zero == 1 or sign == 0:
                        res = 0x3F80
                    elif s3_mag_q >= 2048:
                        res = 0
                    else:
                        res = s3_eword_q
                elif s3_op_q == 2:
                    if is_nan != 0:
                        res = 0x7FC0
                    elif is_inf != 0:
                        res = sign << 15
                    elif is_zero != 0:
                        res = (sign << 15) | 0x7F80
                    else:
                        res = (sign << 15) | (((253 - e) & 0xFF) << 7) | ((s3_rint_q >> 6) & 0x7F)
                else:
                    if is_nan == 1 or sign == 1 or is_zero == 1:
                        res = 0x7FC0
                    elif is_inf != 0:
                        res = 0
                    else:
                        res = (s3_sexp_q << 7) | ((s3_sint_q >> 6) & 0x7F)
                rv[t] = res
                s3_op_q = s2_op_q
                s3_x_q = s2_x_q
                s3_mag_q = s2_mag_q
                s3_gword_q = s2_gword_q
                s3_eword_q = s2_eword_q
                s3_rint_q = ((s2_rbias_q - ((s2_rprod_q >> 11) & 0x1FF)) & 0x3FFF) & 0x1FFF
                s3_sint_q = ((s2_sbias_q - ((s2_sprod_q >> 11) & 0x1FF)) & 0x3FFF) & 0x1FFF
                s3_sexp_q = s2_sexp_q
                rw: int32 = recip_pwl[s1_raddr_q]
                sw: int32 = rsqrt_pwl[s1_saddr_q]
                e1: int32 = (s1_x_q >> 7) & 0xFF
                s2_op_q = s1_op_q
                s2_x_q = s1_x_q
                s2_mag_q = s1_mag_q
                s2_gword_q = rom_gelu[s1_gaddr_q]
                s2_eword_q = rom_exp[s1_eaddr_q]
                s2_rbias_q = (((rw >> 8) & 0x1FFF) + 8321) & 0x3FFF
                s2_sbias_q = (((sw >> 8) & 0x1FFF) + 8322) & 0x3FFF
                s2_rprod_q = (rw & 0xFF) * s1_rpos_q
                s2_sprod_q = (sw & 0xFF) * s1_spos_q
                if (e1 & 1) != 0:
                    s2_sexp_q = (((379 - e1) & 0x1FF) >> 1) & 0xFF
                else:
                    s2_sexp_q = (((380 - e1) & 0x1FF) >> 1) & 0xFF
                x0: int32 = in_x_q
                m0: int32 = fp32_abs_q7(x0)
                s1_op_q = in_op_q
                s1_x_q = x0
                s1_mag_q = m0
                if ((x0 >> 15) & 1) != 0:
                    s1_gaddr_q = (1024 - 1 - m0) & 0x7FF
                else:
                    s1_gaddr_q = (1024 + m0) & 0x7FF
                s1_eaddr_q = (2048 - 1 - m0) & 0x7FF
                s1_raddr_q = (x0 >> 2) & 0x1F
                s1_saddr_q = ((((x0 >> 7) & 1) ^ 1) << 4) | ((x0 >> 3) & 0xF)
                s1_rpos_q = (x0 & 3) << 10
                s1_spos_q = (x0 & 7) << 9
                in_op_q = op_in
                in_x_q = x_in

    return top
