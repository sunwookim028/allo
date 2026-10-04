# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U3: ``mxu_pe``, one weight-stationary PE of MiniTPU's systolic array.

``mxu_pe.sv``: a combinational exact bf16 x bf16 -> acc24 multiplier
(``mxu_bf16_mul_acc24``, U1) on ``lhs_i`` and the *active* weight, a product
register, a partial-sum register, and the 3-stage acc24 adder
(``mxu_acc24_add_pipe``, U1): ``psum_o = acc24_add(product, psum_i)`` four
edges after its operands (``MXU_PE_LATENCY = 1 + MXU_ACC_ADD_LATENCY``,
``vpu_pkg.sv:106``; the PE asserts its own depth in simulation, ``:84-96``).
Two pending weight banks (``MXU_WEIGHT_BANKS = 2``, ``vpu_pkg.sv:108``) each
load from ``weight_i[b]`` when ``weight_valid_i[b]``; ``weight_commit_i``
copies the bank ``weight_commit_bank_i`` names into the active weight.
``lhs``, ``lhs_valid`` and ``weight_commit`` are forwarded east one register
later; ``weight_o`` is the pending registers themselves (forwarded south).
Only the control is reset; ``lhs_o``, ``product_q``, ``psum_q`` and the
adder's payload load every cycle (``mxu_pe.sv:76-81``).

Driven as a ``trace`` unit, every output ``"post"``. Reference:
``harness/ref_mxu.py`` ``mxu_pe_trace`` (a cycle model; unreset payload is
tainted until known). No geometry: the PE has no parameter of its own.
"""

import numpy as np

import allo.dataflow as df
from allo.compose import Architecture, Memory
from allo.ir.types import UInt, uint1, uint8, uint16, uint32

from examples.minitpu.harness import ref_mxu, rtl
from examples.minitpu.harness.traces import Trace, rng_for
from examples.minitpu.units.mxu_pe_unit import pe_channels, pe_drive, pe_sink, pe_unit

SOURCES = ["src/core/vpu/vpu_pkg.sv", "src/core/mxu/mxu_bf16_mul_acc24.sv",
           "src/core/mxu/mxu_acc24_add_pipe.sv", "src/core/mxu/mxu_pe.sv"]

RTL = rtl.RtlUnit(
    top="mxu_pe",
    sources=SOURCES,
    inputs=[("rst_ni", 1), ("weight_commit_i", 1), ("weight_commit_bank_i", 1),
            ("lhs_i", 16), ("lhs_valid_i", 1), ("weight_i", 32), ("weight_valid_i", 2),
            ("psum_i", 24), ("psum_valid_i", 1)],
    outputs=[("weight_commit_o", 1, "post"), ("lhs_o", 16, "post"), ("lhs_valid_o", 1, "post"),
             ("weight_o", 32, "post"), ("psum_o", 24, "post"), ("psum_valid_o", 1, "post")],
    shape="trace",
    assertions=True,
)
INSTANCES = {"pe": RTL}
DEFAULT = "pe"
LATENCY_SOURCE = "vpu_pkg.sv:106 MXU_PE_LATENCY = 1 + MXU_ACC_ADD_LATENCY (= 4)"


def REF(inst, cmd):
    return ref_mxu.mxu_pe_trace(cmd)


# --- value generators shared by the MXU units --------------------------------
SPECIAL_BF16 = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x0001, 0x807F, 0x7F7F, 0xFF7F,
                0x3F80, 0xBF80, 0x0080, 0x8080]


def bf16(rng, p_special=0.1, p_any=0.1):
    """A bf16 pattern: mostly moderate normals (exponent 120..134), some
    specials (zeros, Inf, NaN, subnormals, extremes), some arbitrary."""
    r = rng.random()
    if r < p_special:
        return rng.choice(SPECIAL_BF16)
    if r < p_special + p_any:
        return rng.getrandbits(16)
    return (rng.getrandbits(1) << 15) | (rng.randrange(120, 135) << 7) | rng.getrandbits(7)


def acc24(rng):
    """An acc24 pattern: mostly moderate normals, some specials, some arbitrary."""
    r = rng.random()
    if r < 0.08:
        return rng.choice([0, 1 << 23, 0x7F8000, 0xFF8000, 0x7FC000, 0x000001, 0x7F7FFF])
    if r < 0.18:
        return rng.getrandbits(24)
    return (rng.getrandbits(1) << 23) | (rng.randrange(115, 140) << 15) | rng.getrandbits(15)


def _defaults():
    return {p: 0 for p, _ in RTL.inputs} | {"rst_ni": 1}


def random_trace(n, seed, p_rst=0.003):
    rng = rng_for("mxu_pe", seed)
    t = Trace(_defaults())
    t.idle(2, rst_ni=0)
    for _ in range(n):
        t.cycle(rst_ni=int(rng.random() > p_rst),
                weight_commit_i=int(rng.random() < 0.15), weight_commit_bank_i=rng.getrandbits(1),
                lhs_i=bf16(rng), lhs_valid_i=int(rng.random() < 0.7),
                weight_i=bf16(rng) | (bf16(rng) << 16), weight_valid_i=rng.getrandbits(2),
                psum_i=acc24(rng), psum_valid_i=int(rng.random() < 0.7))
    return t.cmd()


def directed():
    out = []
    # a column of 3 PEs written out by hand: load both banks, commit each in
    # turn, stream activations and partial sums, commit mid-stream
    t = Trace(_defaults())
    t.idle(2, rst_ni=0)
    t.cycle(weight_i=0x3F80 | (0x4000 << 16), weight_valid_i=0b11)  # bank0 = 1.0, bank1 = 2.0
    t.cycle(weight_commit_i=1, weight_commit_bank_i=0)
    for k in range(8):
        t.cycle(lhs_i=0x3F80 + k, lhs_valid_i=1, psum_i=0x3F8000 + (k << 4), psum_valid_i=1,
                weight_commit_i=int(k == 4), weight_commit_bank_i=1)
    t.idle(8)
    out.append(("banks-commit", t.cmd()))
    # pending loads while the active weight is used; commit and load in one cycle
    t = Trace(_defaults())
    t.idle(2, rst_ni=0)
    t.cycle(weight_i=0x4040, weight_valid_i=0b01)
    t.cycle(weight_commit_i=1, weight_commit_bank_i=0, weight_i=0x40A0 << 16, weight_valid_i=0b11)
    for k in range(6):
        t.cycle(lhs_i=0xBF80 - k, lhs_valid_i=1, psum_valid_i=1, psum_i=0,
                weight_commit_i=int(k % 2 == 1), weight_commit_bank_i=k % 2)
    t.idle(8)
    out.append(("commit+load-same-cycle", t.cmd()))
    # specials through product and adder: NaN, Inf x 0, -0 + +0 bypass
    t = Trace(_defaults())
    t.idle(2, rst_ni=0)
    t.cycle(weight_i=0x7F80 | (0x0000 << 16), weight_valid_i=0b11)
    t.cycle(weight_commit_i=1, weight_commit_bank_i=0)
    for x, ps in [(0x0000, 0x800000), (0x8000, 0x000000), (0x3F80, 0xFF8000), (0x7FC0, 0),
                  (0x0001, 0x400000), (0xFF80, 0x7F8000)]:
        t.cycle(lhs_i=x, lhs_valid_i=1, psum_i=ps, psum_valid_i=1)
    t.cycle(weight_commit_i=1, weight_commit_bank_i=1)
    for x, ps in [(0x7F80, 0), (0x8000, 0x800000), (0x0000, 0x800000)]:
        t.cycle(lhs_i=x, lhs_valid_i=1, psum_i=ps, psum_valid_i=1)
    t.idle(8)
    out.append(("specials", t.cmd()))
    # mid-stream reset: payload keeps flowing, control and the adder clear
    t = Trace(_defaults())
    t.idle(2, rst_ni=0)
    t.cycle(weight_i=0x3FC0, weight_valid_i=1)
    t.cycle(weight_commit_i=1)
    for k in range(12):
        t.cycle(lhs_i=0x3F80 + 3 * k, lhs_valid_i=1, psum_i=0x400000 + k, psum_valid_i=1,
                rst_ni=int(k not in (5, 6)))
    t.idle(8)
    out.append(("reset-mid-stream", t.cmd()))
    return out


def traces(inst):
    tr = [(lab, c, True) for lab, c in directed()]
    tr += [(f"random-{s}", random_trace(20000, s), True) for s in range(3)]
    return tr


def _probe_base():
    t = Trace(_defaults())
    t.idle(2, rst_ni=0)
    t.cycle(weight_i=0x4000 | (0x4040 << 16), weight_valid_i=0b11)
    t.cycle(weight_commit_i=1, weight_commit_bank_i=0)
    return t


def probes(inst):
    """``[(label, declared, measured)]``, step probes on the RTL."""
    res = []
    hold = dict(lhs_i=0x3F80, lhs_valid_i=1, psum_i=0x3F8000, psum_valid_i=1)
    # lhs -> lhs_o
    t = _probe_base().idle(8, **hold)
    ev = len(t)
    t.idle(8, **{**hold, "lhs_i": 0x4000})
    res.append(("lhs_i -> lhs_o", 1, rtl.probe_trace(RTL, t.cmd(), "lhs_o", ev)))
    # psum_i -> psum_o: MXU_PE_LATENCY
    t = _probe_base().idle(8, **hold)
    ev = len(t)
    t.idle(10, **{**hold, "psum_i": 0x408000})
    res.append(("psum_i -> psum_o (MXU_PE_LATENCY)", ref_mxu.PE_LATENCY,
                rtl.probe_trace(RTL, t.cmd(), "psum_o", ev)))
    # lhs_i -> psum_o: the product register, then the adder
    t = _probe_base().idle(8, **hold)
    ev = len(t)
    t.idle(10, **{**hold, "lhs_i": 0x4080})
    res.append(("lhs_i -> psum_o", ref_mxu.PE_LATENCY, rtl.probe_trace(RTL, t.cmd(), "psum_o", ev)))
    # commit -> psum_o: active switches at the edge, the next lhs uses it
    t = _probe_base().idle(8, **hold)
    ev = len(t)
    t.cycle(**hold, weight_commit_i=1, weight_commit_bank_i=1)
    t.idle(10, **hold)
    res.append(("weight_commit_i -> psum_o (1 + MXU_PE_LATENCY)", 1 + ref_mxu.PE_LATENCY,
                rtl.probe_trace(RTL, t.cmd(), "psum_o", ev)))
    # weight_i -> weight_o
    t = _probe_base().idle(8, **hold)
    ev = len(t)
    t.cycle(**hold, weight_i=0x4100, weight_valid_i=0b01)
    t.idle(4, **hold)
    res.append(("weight_i -> weight_o", 1, rtl.probe_trace(RTL, t.cmd(), "weight_o", ev)))
    # valid: lhs_valid & psum_valid -> psum_valid_o
    t = _probe_base().idle(8, **{**hold, "lhs_valid_i": 0})
    ev = len(t)
    t.idle(8, **hold)
    res.append(("valid -> psum_valid_o", ref_mxu.PE_LATENCY,
                rtl.probe_trace(RTL, t.cmd(), "psum_valid_o", ev)))
    # commit -> weight_commit_o (east)
    t = _probe_base().idle(8, **hold)
    ev = len(t)
    t.cycle(**hold, weight_commit_i=1)
    t.idle(4, **hold)
    res.append(("weight_commit_i -> weight_commit_o", 1,
                rtl.probe_trace(RTL, t.cmd(), "weight_commit_o", ev)))
    return res

# ---------------------------------------------------------------------------
# Allo variants (U3 track B, ``dev/records/minitpu/u3_track_b_2026-10-04.rst``).
# Every variant is driven by the SAME per-cycle command trace as the RTL (one
# array per input port, ``rst_ni`` included, one element per cycle) and
# returns one array per output port, every output "post": the register after
# the edge that ends cycle ``t``. ``make(n, w)`` takes a width because
# ``check.py`` does; the PE has none (``WIDTH`` below is a placeholder).
#
# ``bits`` (plan P1)
#     One kernel, ``mxu_pe.sv`` with every register written as data: the two
#     pending banks, the active weight, ``lhs_q``/``product_q``/``psum_q``,
#     the forwards, and the acc24 adder's three stage register banks exactly
#     as ``mxu_acc24_add_pipe.sv`` holds them (special class, sign, 20-bit
#     magnitude, 9-bit exponent, packed in a ``uint32`` as U1's ``staged``
#     does). The multiplier is U1's ``mul_acc24.bits`` inline (no engine
#     plug-in here: that is track E's Q1). Reset is modelled as the RTL's:
#     control, the special-class stages and the result clear; payload flows.
# ---------------------------------------------------------------------------

WIDTH = {"pe": 0}
IN_PORTS = [p for p, _ in RTL.inputs]
OUT_PORTS = [p for p, _, _ in RTL.outputs]


def _np(w):
    return np.uint8 if w <= 8 else np.uint16 if w <= 16 else np.uint32


def _in_arrays(cmd, n):
    return [np.asarray([int(x) for x in cmd[p][:n]], dtype=_np(w)) for p, w in RTL.inputs]


def _out_arrays(n):
    return [np.zeros(n, dtype=_np(w)) for _, w, _ in RTL.outputs]


def run_pe(mod, cmd, n, w=0):
    ins, outs = _in_arrays(cmd, n), _out_arrays(n)
    mod(*ins, *outs)
    return dict(zip(OUT_PORTS, outs))


def bits(n, w=0):
    @df.region()
    def top(RST: uint1[n], CMT: uint1[n], CBK: uint1[n], LHS: uint16[n], LHV: uint1[n],
            WGT: uint32[n], WGV: uint8[n], PSI: uint32[n], PSV: uint1[n],
            CMO: uint1[n], LHO: uint16[n], LVO: uint1[n], WGO: uint32[n], PSO: uint32[n],
            PVO: uint1[n]):
        @df.kernel(mapping=[1], args=[RST, CMT, CBK, LHS, LHV, WGT, WGV, PSI, PSV,
                                      CMO, LHO, LVO, WGO, PSO, PVO])
        def pe(rst: uint1[n], cmt: uint1[n], cbk: uint1[n], lhs: uint16[n], lhv: uint1[n],
               wgt: uint32[n], wgv: uint8[n], psi: uint32[n], psv: uint1[n],
               cmo: uint1[n], lho: uint16[n], lvo: uint1[n], wgo: uint32[n], pso: uint32[n],
               pvo: uint1[n]):
            def leading_zeros19(value: UInt(19)) -> UInt(5):
                lz: UInt(5) = 19
                found: uint1 = 0
                for offset in range(19):
                    if not found and value[18 - offset]:
                        lz = offset
                        found = 1
                return lz

            # --- the PE's registers (mxu_pe.sv) ---
            active: UInt(16) = 0
            pend0: UInt(16) = 0
            pend1: UInt(16) = 0
            lhs_q: UInt(16) = 0
            lhs_valid_q: uint1 = 0
            commit_q: uint1 = 0
            product_q: UInt(24) = 0
            psum_q: UInt(24) = 0
            product_valid_q: uint1 = 0
            # --- the adder's stage registers (mxu_acc24_add_pipe.sv) ---
            s1w: uint32 = 0  # mag[0:20] exp[20:29] sign[29] special[30:32]
            s1v: uint1 = 0
            s2w: uint32 = 0
            s2v: uint1 = 0
            result: UInt(24) = 0
            vout: uint1 = 0
            for t in range(n):
                r: uint1 = rst[t]  # S6: every port read unconditional
                cm: uint1 = cmt[t]
                cb: uint1 = cbk[t]
                x: UInt(16) = lhs[t]
                xv: uint1 = lhv[t]
                wg: uint32 = wgt[t]
                wv: uint8 = wgv[t]
                ps: uint32 = psi[t]
                pv: uint1 = psv[t]

                # ---- stage 3: round and pack (from the s2 registers) ----
                s2_mag: UInt(20) = s2w[0:20]
                s2_exp: UInt(10) = s2w[20:29]
                s2_sign: uint1 = s2w[29]
                s2_special: UInt(2) = s2w[30:32]
                guard_bit: uint1 = s2_mag[2]
                round_bit: uint1 = s2_mag[1]
                sticky_bit: uint1 = s2_mag[0]
                round_up: uint1 = guard_bit & (round_bit | sticky_bit | s2_mag[3])
                frac16: UInt(16) = 0
                frac16[0:15] = s2_mag[3:18]
                rounded: UInt(16) = frac16 + round_up
                inc_exp: UInt(10) = s2_exp + 1
                packed: UInt(24) = 0
                if s2_special == 3:
                    packed = 0x7FC000
                elif s2_special == 2:
                    packed[23] = s2_sign
                    packed[15:23] = 0xFF
                elif s2_special == 1:
                    packed[23] = s2_sign
                    if s2_mag[18]:
                        packed[15:23] = s2_exp[0:8]
                    packed[0:15] = s2_mag[3:18]
                elif s2_mag == 0:
                    packed = 0
                elif rounded[15]:
                    packed[23] = s2_sign
                    if inc_exp >= 255:
                        packed[15:23] = 0xFF
                    else:
                        packed[15:23] = inc_exp[0:8]
                elif s2_exp >= 255:
                    packed[23] = s2_sign
                    packed[15:23] = 0xFF
                elif s2_exp <= 1 and not s2_mag[18]:
                    packed[23] = s2_sign
                    packed[0:15] = rounded[0:15]
                else:
                    packed[23] = s2_sign
                    packed[15:23] = s2_exp[0:8]
                    packed[0:15] = rounded[0:15]

                # ---- stage 2: normalize (from the s1 registers) ----
                s1_mag: UInt(20) = s1w[0:20]
                s1_exp: UInt(9) = s1w[20:29]
                s1_sign: uint1 = s1w[29]
                s1_special: UInt(2) = s1w[30:32]
                norm_overflow: uint1 = s1_mag[19]
                norm_needed: uint1 = 0
                if s1_mag != 0 and not s1_mag[18] and not norm_overflow:
                    norm_needed = 1
                lzc: UInt(5) = leading_zeros19(s1_mag[0:19])
                max_ns: UInt(5) = 18 if s1_exp > 19 else s1_exp - 1
                lz6: UInt(6) = lzc  # spare top bit (B1)
                max6: UInt(6) = max_ns
                normalize_shift: UInt(5) = 0
                if norm_needed:
                    normalize_shift = lzc if lz6 < max6 else max_ns
                mag19: UInt(19) = s1_mag[0:19]
                mag_normalized: UInt(19) = mag19 << normalize_shift
                mag_s2: UInt(20) = s1_mag
                exp_s2: UInt(10) = s1_exp
                if s1_special == 1:
                    exp_s2 = s1_exp
                elif norm_overflow:
                    mag_s2[1] = mag_s2[1] | mag_s2[0]
                    mag_s2 >>= 1
                    exp_s2 = s1_exp + 1
                elif norm_needed:
                    mag_s2 = 0
                    mag_s2[0:19] = mag_normalized
                    exp_s2 = s1_exp - normalize_shift
                w2n: uint32 = 0
                w2n[0:20] = mag_s2
                w2n[20:29] = exp_s2[0:9]
                w2n[29] = s1_sign
                w2n[30:32] = s1_special

                # ---- stage 1: classify, align (jam), add (product_q + psum_q) ----
                a_i: UInt(24) = product_q
                b_i: UInt(24) = psum_q
                sign_a: uint1 = a_i[23]
                sign_b: uint1 = b_i[23]
                exp_a: UInt(8) = a_i[15:23]
                exp_b: UInt(8) = b_i[15:23]
                frac_a: UInt(15) = a_i[0:15]
                frac_b: UInt(15) = b_i[0:15]
                sig_a: UInt(16) = 0
                sig_a[15] = exp_a != 0
                sig_a[0:15] = frac_a
                sig_b: UInt(16) = 0
                sig_b[15] = exp_b != 0
                sig_b[0:15] = frac_b
                same_sign: uint1 = sign_a == sign_b
                key_a: UInt(24) = 0  # spare top bit (B1)
                key_a[0:23] = a_i[0:23]
                key_b: UInt(24) = 0
                key_b[0:23] = b_i[0:23]
                a_is_large: uint1 = key_a >= key_b
                sign_large: uint1 = sign_b
                exp_large: UInt(9) = 0
                exp_small: UInt(9) = 0
                sig_large: UInt(16) = 0
                sig_small: UInt(16) = 0
                if a_is_large:
                    sign_large = sign_a
                    exp_large = 1 if exp_a == 0 else exp_a
                    exp_small = 1 if exp_b == 0 else exp_b
                    sig_large = sig_a
                    sig_small = sig_b
                else:
                    exp_large = 1 if exp_b == 0 else exp_b
                    exp_small = 1 if exp_a == 0 else exp_a
                    sig_large = sig_b
                    sig_small = sig_a
                exp_diff: UInt(9) = exp_large - exp_small
                align_shift: UInt(5) = 19 if exp_diff >= 19 else exp_diff[0:5]
                wide: UInt(40) = 0
                wide[3:19] = sig_small
                one: UInt(40) = 1
                mask: UInt(40) = (one << align_shift) - one
                jam: uint1 = (wide & mask) != 0
                shifted: UInt(40) = wide >> align_shift
                small_aligned: UInt(19) = shifted[0:19]
                small_aligned[0] = small_aligned[0] | jam
                mant_large: UInt(19) = 0
                mant_large[3:19] = sig_large
                magnitude_s1: UInt(20) = (
                    (mant_large + small_aligned)
                    if same_sign
                    else (mant_large - small_aligned)
                )
                special_s1: UInt(2) = 0  # NORMAL 0, BYPASS 1, INF 2, NAN 3
                if (
                    (exp_a == 0xFF and frac_a != 0)
                    or (exp_b == 0xFF and frac_b != 0)
                    or (exp_a == 0xFF and exp_b == 0xFF and sign_a != sign_b)
                ):
                    special_s1 = 3
                elif exp_a == 0xFF or exp_b == 0xFF:
                    special_s1 = 2
                elif a_i[0:23] == 0:
                    special_s1 = 1
                    sign_large = sign_b
                elif b_i[0:23] == 0:
                    special_s1 = 1
                    sign_large = sign_a
                w1n: uint32 = 0
                w1n[0:20] = magnitude_s1
                w1n[20:29] = exp_large
                w1n[29] = sign_large
                w1n[30:32] = special_s1

                # ---- the multiplier (mxu_bf16_mul_acc24.sv): lhs_i x active ----
                ma: UInt(16) = x
                mb: UInt(16) = active
                msign: uint1 = ma[15] ^ mb[15]
                mexp_a: UInt(8) = ma[7:15]
                mexp_b: UInt(8) = mb[7:15]
                mfrac_a: UInt(7) = ma[0:7]
                mfrac_b: UInt(7) = mb[0:7]
                mant_a: UInt(8) = 0
                mant_a[7] = 1
                mant_a[0:7] = mfrac_a
                mant_b: UInt(8) = 0
                mant_b[7] = 1
                mant_b[0:7] = mfrac_b
                mprod: UInt(16) = mant_a * mant_b
                exp_sum: UInt(9) = mexp_a + mexp_b
                exp_low: UInt(9) = exp_sum - 127
                exp_high: UInt(9) = exp_sum - 126
                finite_low: UInt(24) = 0
                finite_low[23] = msign
                finite_low[15:23] = exp_low[0:8]
                finite_low[1:15] = mprod[0:14]
                if exp_sum <= 127:
                    finite_low = 0
                    finite_low[23] = msign
                elif exp_sum >= 382:
                    finite_low = 0
                    finite_low[23] = msign
                    finite_low[15:23] = 0xFF
                finite_high: UInt(24) = 0
                finite_high[23] = msign
                finite_high[15:23] = exp_high[0:8]
                finite_high[0:15] = mprod[0:15]
                if exp_sum <= 126:
                    finite_high = 0
                    finite_high[23] = msign
                elif exp_sum >= 381:
                    finite_high = 0
                    finite_high[23] = msign
                    finite_high[15:23] = 0xFF
                finite_result: UInt(24) = finite_high if mprod[15] else finite_low
                product: UInt(24) = 0
                if (
                    (mexp_a == 0xFF and mfrac_a != 0)
                    or (mexp_b == 0xFF and mfrac_b != 0)
                    or (mexp_a == 0xFF and mb[0:15] == 0)
                    or (mexp_b == 0xFF and ma[0:15] == 0)
                ):
                    product = 0x7FC000
                elif mexp_a == 0xFF or mexp_b == 0xFF:
                    product[23] = msign
                    product[15:23] = 0xFF
                elif mexp_a == 0 or mexp_b == 0:
                    product[23] = msign
                else:
                    product = finite_result

                # ---- the edge: adder stages (reset forces class and valid) ----
                if r == 0:
                    result = 0
                    vout = 0
                    s2w = w2n
                    s2w[30:32] = 0
                    s2v = 0
                    s1w = w1n
                    s1w[30:32] = 0
                    s1v = 0
                else:
                    result = packed
                    vout = s2v
                    s2w = w2n
                    s2v = s1v
                    s1w = w1n
                    s1v = product_valid_q
                # ---- the edge: the PE's registers ----
                product_q = product  # payload: every cycle, from the OLD active
                psum_q = ps[0:24]
                lhs_q = x
                if r == 0:
                    active = 0
                    pend0 = 0
                    pend1 = 0
                    product_valid_q = 0
                    lhs_valid_q = 0
                    commit_q = 0
                else:
                    lhs_valid_q = xv
                    if cm == 1:
                        active = pend1 if cb == 1 else pend0
                    if wv[0] == 1:
                        pend0 = wg[0:16]
                    if wv[1] == 1:
                        pend1 = wg[16:32]
                    commit_q = cm
                    product_valid_q = xv & pv
                # ---- outputs: the registers after the edge ----
                cmo[t] = commit_q
                lho[t] = lhs_q
                lvo[t] = lhs_valid_q
                wo: uint32 = 0
                wo[0:16] = pend0
                wo[16:32] = pend1
                wgo[t] = wo
                pso[t] = result
                pvo[t] = vout

    return top


def unit_pe(n, w=0):
    """Plan P2 standalone: driver -> ``pe_unit`` (1 x 1, ``EDGE_OUT``) -> sink."""
    widths = (("RST", 1), ("CMT", 1), ("CBK", 1), ("LHS", 16), ("LHV", 1), ("WGT", 32), ("WGV", 8),
              ("PSI", 32), ("PSV", 1), ("CMO", 1), ("LHO", 16), ("LVO", 1), ("WGO", 32), ("PSO", 32),
              ("PVO", 1))
    mems = tuple(Memory(nm, f"UInt({wd})[N]") for nm, wd in widths)
    arch = Architecture(name="mxu_pe_unit", parameters={"N": n, "D": 1, "ROW0_PSUM_ZERO": 0, "EDGE_OUT": 1},
                        memories=mems, channels=pe_channels(), units=(pe_drive, pe_unit, pe_sink))
    return arch.region()


# token t of ``unit`` is the PE's state DURING cycle t: the RTL's post output of t - 1
RESP_SHIFT = {"unit": {p: -1 for p in OUT_PORTS}}

VARIANTS = {"bits": (bits, run_pe), "unit": (unit_pe, run_pe)}
