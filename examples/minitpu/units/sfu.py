# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U3: ``sfu``, the special-function unit: vgelu, vexp, vrecip, vrsqrt.

One element per cycle, II=1, five registers deep (``input_*_q``, stages 1-3,
``result_o``); ``valid_i``/``valid_o`` and the op tag are reset, the payload
is not (``sfu.sv`` "No reset on payload"). vgelu/vexp are 2048-entry ROMs
(``$readmemh`` of ``gelu_bf16.mem``/``exp_bf16.mem``), vrecip/vrsqrt are
32-bin piecewise-linear tables written as ``case`` functions. ``sfu_group``
is ``LANES`` copies of this module sharing ``valid_i``/``op_i`` (4 groups of
16 lanes in ``vpu.sv``): replication, not characterized separately.

Declared latency 5: ``vpu_pkg.sv:64`` ``VPU_SFU_LATENCY = 5 // sfu_group``;
``docs/isa_latency.json`` ``WB_W_SFU = 7`` = 5 + ``VPU_WB_STAGES`` (2).

Reference ``ref.sfu``: the logic modelled bit for bit; the ROM contents and
the two PWL tables are read from the clone as data (``ref._sfu_tables``), so
``tb_sfu_math_sweep`` -- not this harness -- is what checks the tables.
"""

import os

import numpy as np

from examples.minitpu.harness import ref, rtl

_HOME = rtl.minitpu_home()

RTL = rtl.RtlUnit(
    top="sfu",
    sources=["src/core/vpu/vpu_pkg.sv", "src/core/sfu/sfu.sv"],
    inputs=[("op_i", 2), ("operand_i", 16)],
    outputs=[("result_o", 16)],
    shape="valid",
    latency=5,
    # the driver does not run in the clone: $readmemh needs absolute paths
    params={"GELU_MEM_FILE": '"%s"' % os.path.join(_HOME, "src/core/sfu/gelu_bf16.mem"),
            "EXP_MEM_FILE": '"%s"' % os.path.join(_HOME, "src/core/sfu/exp_bf16.mem")},
)
LATENCY_SOURCE = "vpu_pkg.sv:64 VPU_SFU_LATENCY = 5 (isa_latency.json WB_W_SFU 7 = 5 + VPU_WB_STAGES 2)"

REF = ref.sfu
IEEE = ref.ieee_sfu
# rsqrt(4) = 0.5 -> rsqrt(1) = 1: the output moves
PROBE = ((3, 0x4080), (3, 0x3F80))


def stimulus():
    """Every op over every bf16 operand (op-major), then 200,000 vectors with
    a random op on every cycle, so the op tag must travel with its operand."""
    x = np.arange(65536, dtype=np.uint64)
    blocks = [np.stack([np.full(65536, op, dtype=np.uint64), x], axis=1) for op in range(4)]
    rng = np.random.default_rng(0x5F0)
    mix = np.stack([rng.integers(0, 4, 200_000, dtype=np.uint64),
                    rng.integers(0, 65536, 200_000, dtype=np.uint64)], axis=1)
    return np.concatenate(blocks + [mix])


def _f(x):
    return float(ref._bf16_to_f32(np.uint16(x)))


def _ulps(g, w):
    """Distance in bf16 ulps between two finite same-sign patterns."""
    return abs(int(g) - int(w))


def _ulp(w):
    e = (int(w) >> 7) & 0xFF
    return 2.0 ** (max(e, 1) - 127 - 7)


def _rel(g, w):
    a, b = _f(g), _f(w)
    return abs(a - b) / max(abs(a), 1e-38)


_SUB = lambda x: (int(x) >> 7) & 0xFF == 0 and int(x) & 0x7F != 0

# Every difference between the math (float64, one RNE to bf16) and the RTL,
# by cause, first match wins. s = (op, x); g = IEEE; w = RTL.
DEVIATIONS = [
    ("NaN in -> +0x7fc0 (IEEE keeps the operand's NaN sign/payload)",
     lambda s, g, w: ref.is_nan(s[1]) and int(w) == 0x7FC0),
    ("vexp(x > 0) = 1.0 (unit covers x <= 0 only; clamps silently)",
     lambda s, g, w: s[0] == 1 and int(s[1]) >> 15 == 0 and not ref.is_zero(s[1])
     and int(w) == 0x3F80),
    ("vrsqrt(+-0) = NaN (IEEE: +-Inf)",
     lambda s, g, w: s[0] == 3 and ref.is_zero(s[1]) and int(w) == 0x7FC0),
    ("vrsqrt(x < 0) = +0x7fc0 (IEEE: NaN, sign differs)",
     lambda s, g, w: s[0] == 3 and ref.is_nan(g) and int(w) == 0x7FC0),
    ("vexp(x <= -16) = 0 (table domain [-16, 0); IEEE a tiny normal/subnormal)",
     lambda s, g, w: s[0] == 1 and _f(s[1]) <= -16 and int(w) == 0),
    ("vgelu(x < -8) = 0 (table domain; IEEE a tiny negative)",
     lambda s, g, w: s[0] == 0 and _f(s[1]) <= -8 and int(w) == 0),
    ("subnormal operand: exponent field 0 read as a normal exponent (recip/rsqrt)",
     lambda s, g, w: s[0] in (2, 3) and _SUB(s[1])),
    ("vrecip, |x| in [2^127, 2^128): exponent 253 - 254 wraps to 0xff -> Inf or a "
     "NON-canonical NaN pattern (e.g. 0x7fff)",
     lambda s, g, w: s[0] == 2 and ((int(s[1]) >> 7) & 0xFF) == 254 and (int(w) >> 7) & 0xFF == 0xFF),
    ("vrecip, |x| in [2^126, 2^127): exponent field 0 with the fraction bits -> a subnormal "
     "pattern worth half (hidden bit lost); IEEE the subnormal 1/x",
     lambda s, g, w: s[0] == 2 and ((int(s[1]) >> 7) & 0xFF) == 253),
    ("vgelu/vexp subnormal operand -> table centre bin",
     lambda s, g, w: s[0] in (0, 1) and _SUB(s[1])),
    ("vrecip/vrsqrt PWL truncation, <= 2 ulp",
     lambda s, g, w: s[0] in (2, 3) and int(g) >> 15 == int(w) >> 15 and _ulps(g, w) <= 2),
    ("vexp table step (input quantized to 1/128, bin centre): rel <= 1%",
     lambda s, g, w: s[0] == 1 and _rel(g, w) <= 0.01),
    ("vgelu table step: abs <= 4e-3 (half a 1/128 bin) + 1 ulp of the bf16 entry, or rel <= 1%",
     lambda s, g, w: s[0] == 0 and (abs(_f(g) - _f(w)) <= 4e-3 + _ulp(w) or _rel(g, w) <= 0.01)),
]


# ---------------------------------------------------------------------------
# Allo expressions (U3 track A; ``dev/records/minitpu/u3_track_a_2026-10-04.rst``)
#
# The DATA is loaded from the clone at build as module-level numpy constants
# (the two 2,048-entry ROMs and the two 32-entry PWL tables, the same reader
# ``ref`` uses): a ROM from a file has no Allo declaration of its own (plan
# H2), so a constant array is the form. The LOGIC is ``sfu.sv`` transcribed;
# everything is carried in ``int32`` and masked to the RTL's widths, which
# sidesteps the narrow-``UInt`` compare/extension class of U1/U2 (B1, B4).
#
# ``bits`` (plan S1)   one element per loop iteration, no registers: the
#                      function of ``(op, operand)``. Untimed.
# ``staged`` (S2)      the five register banks of ``sfu.sv`` (``input_*_q``,
#                      stage 1-3, ``result_o``) written as data, one iteration
#                      per cycle: output ``i`` leaves at iteration ``i + 4``
#                      (latency 5 counts the input register).
# ---------------------------------------------------------------------------

import allo.dataflow as df  # noqa: E402
from allo.ir.types import int32, uint8, uint16  # noqa: E402

_T = ref._sfu_tables()
GELU_ROM = _T["gelu"].astype(np.int32)
EXP_ROM = _T["exp"].astype(np.int32)
RECIP_PWL = _T["recip"].astype(np.int32)  # 21-bit words: 13 value + 8 slope
RSQRT_PWL = _T["rsqrt"].astype(np.int32)


def fp32_abs_q7(x: int32) -> int32:
    """``sfu.sv`` ``fp32_abs_q7`` on the fp32 view ``{x, 16'b0}``."""
    e: int32 = (x >> 7) & 0xFF
    mant: int32 = (0x80 | (x & 0x7F)) << 16
    rs: int32 = 16 - (e - 127)
    q: int32 = 0
    if e == 0 or rs >= 24:
        q = 0
    elif rs <= 0:
        q = 0x1FFF
    else:
        v: int32 = mant >> rs
        q = 0x1FFF if v > 0x1FFF else v
    return q




def gelu_addr(x: int32) -> int32:
    """``sfu.sv`` GELU ROM address: the fold of ``fp32_abs_q7`` on the sign."""
    mag: int32 = fp32_abs_q7(x)
    a: int32 = 0
    if ((x >> 15) & 1) != 0:
        a = (1024 - 1 - mag) & 0x7FF
    else:
        a = (1024 + mag) & 0x7FF
    return a


def exp_addr(x: int32) -> int32:
    mag: int32 = fp32_abs_q7(x)
    return (2048 - 1 - mag) & 0x7FF


def sfu_bits(op: int32, x: int32, gelu_word: int32, exp_word: int32, rw: int32, sw: int32) -> int32:
    """``sfu.sv`` on one ``(op, operand)``, with the stage boundaries erased.

    The table *words* are arguments, looked up by the kernel: a constant array
    bound inside a helper (``rom: int32[2048] = ROM``) builds on the simulator
    but the SystemC emitter leaves it out of the helper's scope (finding A2),
    and a constant array passed down as an argument is emitted ``const`` on
    the caller's side and non-const on the callee's (A4), so the lookups stay
    in the kernel (workaround; the record).
    """
    sign: int32 = (x >> 15) & 1
    e: int32 = (x >> 7) & 0xFF
    frac: int32 = x & 0x7F
    mag: int32 = fp32_abs_q7(x)
    is_nan: int32 = 1 if (e == 0xFF and frac != 0) else 0
    is_inf: int32 = 1 if (e == 0xFF and frac == 0) else 0
    is_zero: int32 = 1 if (x & 0x7FFF) == 0 else 0
    nonpos: int32 = 1 if (sign == 1 or is_zero == 1) else 0
    recip_pos: int32 = (x & 3) << 10  # fp32[17:6]
    rsqrt_pos: int32 = (x & 7) << 9  # fp32[18:7]
    # stage 2: bias (offset folded in), slope products
    r_bias: int32 = (((rw >> 8) & 0x1FFF) + 8321) & 0x3FFF
    s_bias: int32 = (((sw >> 8) & 0x1FFF) + 8322) & 0x3FFF
    r_prod: int32 = (rw & 0xFF) * recip_pos
    s_prod: int32 = (sw & 0xFF) * rsqrt_pos
    s_exp: int32 = 0
    if (e & 1) != 0:
        s_exp = ((379 - e) & 0x1FF) >> 1
    else:
        s_exp = ((380 - e) & 0x1FF) >> 1
    s_exp = s_exp & 0xFF
    # stage 3: interpolation (14-bit decode, bit 13 a borrow guard)
    r_int: int32 = ((r_bias - ((r_prod >> 11) & 0x1FF)) & 0x3FFF) & 0x1FFF
    s_int: int32 = ((s_bias - ((s_prod >> 11) & 0x1FF)) & 0x3FFF) & 0x1FFF
    # output stage
    res: int32 = 0x7FC0
    if op == 0:  # GELU
        if is_nan != 0:
            res = 0x7FC0
        elif is_zero != 0:
            res = 0
        elif mag >= 1024:
            res = 0 if sign == 1 else x
        else:
            res = gelu_word
    elif op == 1:  # EXP
        if is_nan != 0:
            res = 0x7FC0
        elif is_zero == 1 or sign == 0:
            res = 0x3F80
        elif mag >= 2048:
            res = 0
        else:
            res = exp_word
    elif op == 2:  # RECIP
        if is_nan != 0:
            res = 0x7FC0
        elif is_inf != 0:
            res = sign << 15
        elif is_zero != 0:
            res = (sign << 15) | 0x7F80
        else:
            res = (sign << 15) | (((253 - e) & 0xFF) << 7) | ((r_int >> 6) & 0x7F)
    else:  # RSQRT
        if is_nan == 1 or nonpos == 1:
            res = 0x7FC0
        elif is_inf != 0:
            res = 0
        else:
            res = (s_exp << 7) | ((s_int >> 6) & 0x7F)
    return res


def bits(n):
    @df.region()
    def top(OP: uint8[n], X: uint16[n], R: uint16[n]):
        @df.kernel(mapping=[1], args=[OP, X, R])
        def sfu_k(opv: uint8[n], xv: uint16[n], rv: uint16[n]):
            # A module-level array cannot be indexed directly (``Unsupported
            # global variable``): it is bound to a local constant first (A1).
            # Its name must not start with ``gelu`` (A3: ``allo/passes.py``
            # erases any such symbol as the library's gelu).
            rom_gelu: int32[2048] = GELU_ROM
            rom_exp: int32[2048] = EXP_ROM
            recip_pwl: int32[32] = RECIP_PWL
            rsqrt_pwl: int32[32] = RSQRT_PWL
            for i in range(n):
                o: int32 = opv[i]
                x: int32 = xv[i]
                # stage 1's addresses and stage 2's table reads, in the kernel (A2, A4)
                ga: int32 = gelu_addr(x)
                ea: int32 = exp_addr(x)
                ra: int32 = (x >> 2) & 0x1F  # fp32[22:18]
                sa: int32 = (((x >> 7) & 1) ^ 1) << 4 | ((x >> 3) & 0xF)  # {~fp32[23], fp32[22:19]}
                gw: int32 = rom_gelu[ga]
                ew: int32 = rom_exp[ea]
                rw: int32 = recip_pwl[ra]
                sw: int32 = rsqrt_pwl[sa]
                r: int32 = sfu_bits(o, x, gw, ew, rw, sw)
                rv[i] = r

    return top


def staged(n):
    """S2: ``sfu.sv``'s five register banks as data, one iteration per cycle.

    Every name ending ``_q`` is a register of ``sfu.sv`` (the subset the
    output depends on). Iteration ``t`` computes each bank's next value from
    the current ones, last stage first, so a value moves one bank per
    iteration; ``result_q`` after iteration ``t`` is the operand of ``t - 4``
    (the RTL's latency 5 counts the input register; a ``valid``-shape row is
    sampled one edge later than this array). Payload only: ``valid`` is the
    harness's, and the op tag travels as payload here (it is reset in
    ``sfu.sv``, which only matters before the first valid).
    """
    @df.region()
    def top(OP: uint8[n], X: uint16[n], R: uint16[n]):
        @df.kernel(mapping=[1], args=[OP, X, R])
        def sfu_k(opv: uint8[n], xv: uint16[n], rv: uint16[n]):
            rom_gelu: int32[2048] = GELU_ROM  # A1/A3: bound locally, not named gelu*
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
            for t in range(n + 4):
                # ---- result_o <= f(stage 3) ----
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
                if t >= 4:
                    rv[t - 4] = res
                # ---- stage 3 <= stage 2: the interpolation subtract ----
                s3_op_q = s2_op_q
                s3_x_q = s2_x_q
                s3_mag_q = s2_mag_q
                s3_gword_q = s2_gword_q
                s3_eword_q = s2_eword_q
                s3_rint_q = ((s2_rbias_q - ((s2_rprod_q >> 11) & 0x1FF)) & 0x3FFF) & 0x1FFF
                s3_sint_q = ((s2_sbias_q - ((s2_sprod_q >> 11) & 0x1FF)) & 0x3FFF) & 0x1FFF
                s3_sexp_q = s2_sexp_q
                # ---- stage 2 <= stage 1: ROM reads, PWL words, products ----
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
                # ---- stage 1 <= input register: magnitude, addresses ----
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
                # ---- input register <= ports ----
                if t < n:
                    in_op_q = opv[t]
                    in_x_q = xv[t]
                else:
                    in_op_q = 0
                    in_x_q = 0

    return top


def run(mod, stim):
    """``stim``: uint64[n, 2] = (op, operand). Returns uint16[n]."""
    op = np.ascontiguousarray(stim[:, 0]).astype(np.uint8)
    x = np.ascontiguousarray(stim[:, 1]).astype(np.uint16)
    r = np.zeros(len(stim), dtype=np.uint16)
    mod(op, x, r)
    return r


VARIANTS = {"bits": (bits, run), "staged": (staged, run)}
