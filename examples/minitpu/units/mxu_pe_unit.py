# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U3: MiniTPU's ``mxu_pe`` as a ``compose.unit`` (plan P2), with the driver
and sink units its standalone check wraps it in. Annotations are lazy here
(``from __future__ import annotations``) because a unit body's parameter
types name the architecture's parameters (``N``), bound only when the
``Architecture`` emits the region; the trace variants in ``mxu_pe.py`` keep
eager annotations for ``@df.region()``.
"""

from __future__ import annotations

from allo.compose import Channel, unit

# ---------------------------------------------------------------------------
# ``unit`` (plan P2): the PE as a ``compose.unit`` with stream ports.
#
# Tokens, one per cycle on every port (the token index IS the cycle):
#   west  ``lhsx[i, j]``  UInt(32): lhs[0:16] lhs_valid[16] commit[17] bank[18] rst_n[19]
#   north ``wx[i, j]``    UInt(64): pending0[0:16] pending1[16:32] weight_valid[32:34]
#   north ``px[i, j]``    UInt(32): psum[0:24] psum_valid[24]
# and the same layouts east (``lhsx[i, j+1]``) and south (``wx[i+1, j]``,
# ``px[i+1, j]``). The PE emits its registered state BEFORE the update, so the
# token a neighbour gets at ``t`` is the register it would see during cycle
# ``t`` in the RTL; the commit's bank bit travels east WITH the commit (in
# ``mxu.sv`` it is a delay line read only at the commit, so this is the same
# function at the ``mxu`` boundary), and ``rst_n`` and ``weight_valid`` pass
# through combinationally (global reset; ``mxu_systolic_array.sv`` broadcasts
# the load beat down a column). ``ROW0_PSUM_ZERO`` wires row 0's partial sum
# to ``'0`` with ``psum_valid = lhs_valid`` as the array does; ``EDGE_OUT``
# makes the last column / bottom row emit their east / south tokens too (the
# standalone check reads them; in the array nothing does).
#
# The datapath is ``bits`` line for line. A nested helper ``def`` is not
# possible here: ``compose.Unit.check`` counts its call as a free name (track
# B finding), so the leading-zero count is an inline loop.
# ---------------------------------------------------------------------------


@unit(instances=("D", "D"), reads=("lhsx", "wx", "px"), writes=("lhsx", "wx", "px"),
      parameters=("N", "D", "ROW0_PSUM_ZERO", "EDGE_OUT"))
def pe_unit():
    i, j = df.get_pid()
    # --- the PE's registers (mxu_pe.sv) ---
    active: UInt(16) = 0
    pend0: UInt(16) = 0
    pend1: UInt(16) = 0
    lhs_q: UInt(16) = 0
    lhs_valid_q: UInt(1) = 0
    commit_q: UInt(1) = 0
    bank_q: UInt(1) = 0
    product_q: UInt(24) = 0
    psum_q: UInt(24) = 0
    product_valid_q: UInt(1) = 0
    # --- the adder's stage registers (mxu_acc24_add_pipe.sv) ---
    s1w: UInt(32) = 0  # mag[0:20] exp[20:29] sign[29] special[30:32]
    s1v: UInt(1) = 0
    s2w: UInt(32) = 0
    s2v: UInt(1) = 0
    result: UInt(24) = 0
    vout: UInt(1) = 0
    for t in range(N):
        wt: UInt(32) = lhsx[i, j].get()
        nw: UInt(64) = wx[i, j].get()
        x: UInt(16) = wt[0:16]
        xv: UInt(1) = wt[16]
        cm: UInt(1) = wt[17]
        cb: UInt(1) = wt[18]
        r: UInt(1) = wt[19]
        wv0: UInt(1) = nw[32]
        wv1: UInt(1) = nw[33]
        ps: UInt(24) = 0
        pv: UInt(1) = xv
        with allo.meta_if(i > 0 or ROW0_PSUM_ZERO == 0):
            npt: UInt(32) = px[i, j].get()
            ps = npt[0:24]
            pv = npt[24]
        # ---- emit the registers as the neighbours see them this cycle ----
        with allo.meta_if(j != D - 1 or EDGE_OUT == 1):
            et: UInt(32) = 0
            et[0:16] = lhs_q
            et[16] = lhs_valid_q
            et[17] = commit_q
            et[18] = bank_q
            et[19] = r
            lhsx[i, j + 1].put(et)
        with allo.meta_if(i != D - 1 or EDGE_OUT == 1):
            sw: UInt(64) = 0
            sw[0:16] = pend0
            sw[16:32] = pend1
            sw[32] = wv0
            sw[33] = wv1
            wx[i + 1, j].put(sw)
        sp: UInt(32) = 0
        sp[0:24] = result
        sp[24] = vout
        px[i + 1, j].put(sp)

        # ---- stage 3: round and pack (from the s2 registers) ----
        s2_mag: UInt(20) = s2w[0:20]
        s2_exp: UInt(10) = s2w[20:29]
        s2_sign: UInt(1) = s2w[29]
        s2_special: UInt(2) = s2w[30:32]
        guard_bit: UInt(1) = s2_mag[2]
        round_bit: UInt(1) = s2_mag[1]
        sticky_bit: UInt(1) = s2_mag[0]
        round_up: UInt(1) = guard_bit & (round_bit | sticky_bit | s2_mag[3])
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
        s1_sign: UInt(1) = s1w[29]
        s1_special: UInt(2) = s1w[30:32]
        norm_overflow: UInt(1) = s1_mag[19]
        norm_needed: UInt(1) = 0
        if s1_mag != 0 and not s1_mag[18] and not norm_overflow:
            norm_needed = 1
        mag19: UInt(19) = s1_mag[0:19]
        lzc: UInt(5) = 19
        found: UInt(1) = 0
        for offset in range(19):
            if not found and mag19[18 - offset]:
                lzc = offset
                found = 1
        max_ns: UInt(5) = 18 if s1_exp > 19 else s1_exp - 1
        lz6: UInt(6) = lzc  # spare top bit (B1)
        max6: UInt(6) = max_ns
        normalize_shift: UInt(5) = 0
        if norm_needed:
            normalize_shift = lzc if lz6 < max6 else max_ns
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
        w2n: UInt(32) = 0
        w2n[0:20] = mag_s2
        w2n[20:29] = exp_s2[0:9]
        w2n[29] = s1_sign
        w2n[30:32] = s1_special

        # ---- stage 1: classify, align (jam), add (product_q + psum_q) ----
        a_i: UInt(24) = product_q
        b_i: UInt(24) = psum_q
        sign_a: UInt(1) = a_i[23]
        sign_b: UInt(1) = b_i[23]
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
        same_sign: UInt(1) = sign_a == sign_b
        key_a: UInt(24) = 0  # spare top bit (B1)
        key_a[0:23] = a_i[0:23]
        key_b: UInt(24) = 0
        key_b[0:23] = b_i[0:23]
        a_is_large: UInt(1) = key_a >= key_b
        sign_large: UInt(1) = sign_b
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
        jam: UInt(1) = (wide & mask) != 0
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
        w1n: UInt(32) = 0
        w1n[0:20] = magnitude_s1
        w1n[20:29] = exp_large
        w1n[29] = sign_large
        w1n[30:32] = special_s1

        # ---- the multiplier (mxu_bf16_mul_acc24.sv): lhs_i x active ----
        ma: UInt(16) = x
        mb: UInt(16) = active
        msign: UInt(1) = ma[15] ^ mb[15]
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
        psum_q = ps
        lhs_q = x
        bank_q = cb
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
            if wv0 == 1:
                pend0 = nw[0:16]
            if wv1 == 1:
                pend1 = nw[16:32]
            commit_q = cm
            product_valid_q = xv & pv


# The standalone wrapper: driver -> one PE -> sink, each a unit over the
# region's per-port arrays (``compose.Memory``), so the PE body is composed the
# way the array composes it.
@unit(memories=("RST", "CMT", "CBK", "LHS", "LHV", "WGT", "WGV", "PSI", "PSV"),
      writes=("lhsx", "wx", "px"), parameters=("N",))
def pe_drive(rst: UInt(1)[N], cmt: UInt(1)[N], cbk: UInt(1)[N], lhs: UInt(16)[N], lhv: UInt(1)[N],
             wgt: UInt(32)[N], wgv: UInt(8)[N], psi: UInt(32)[N], psv: UInt(1)[N]):
    for t in range(N):
        wt: UInt(32) = 0
        wt[0:16] = lhs[t]
        wt[16] = lhv[t]
        wt[17] = cmt[t]
        wt[18] = cbk[t]
        wt[19] = rst[t]
        lhsx[0, 0].put(wt)
        wg: UInt(32) = wgt[t]
        wv: UInt(8) = wgv[t]
        nw: UInt(64) = 0
        nw[0:16] = wg[0:16]
        nw[16:32] = wg[16:32]
        nw[32] = wv[0]
        nw[33] = wv[1]
        wx[0, 0].put(nw)
        ps: UInt(32) = psi[t]
        sp: UInt(32) = 0
        sp[0:24] = ps[0:24]
        sp[24] = psv[t]
        px[0, 0].put(sp)


@unit(memories=("CMO", "LHO", "LVO", "WGO", "PSO", "PVO"),
      reads=("lhsx", "wx", "px"), parameters=("N",))
def pe_sink(cmo: UInt(1)[N], lho: UInt(16)[N], lvo: UInt(1)[N], wgo: UInt(32)[N], pso: UInt(32)[N],
            pvo: UInt(1)[N]):
    for t in range(N):
        et: UInt(32) = lhsx[0, 1].get()
        sw: UInt(64) = wx[1, 0].get()
        sp: UInt(32) = px[1, 0].get()
        lho[t] = et[0:16]
        lvo[t] = et[16]
        cmo[t] = et[17]
        wo: UInt(32) = 0
        wo[0:16] = sw[0:16]
        wo[16:32] = sw[16:32]
        wgo[t] = wo
        po: UInt(32) = 0
        po[0:24] = sp[0:24]
        pso[t] = po
        pvo[t] = sp[24]


def pe_channels():
    """The grid's three link arrays, one element per edge of a D x D array
    plus the edge rows/columns the drivers and sinks use."""
    return (
        Channel("lhsx", "UInt(32)", "2", ("D", "D + 1"), "west -> east: lhs, valid, commit, bank, rst_n"),
        Channel("wx", "UInt(64)", "2", ("D + 1", "D"), "north -> south: both pending banks, weight_valid"),
        Channel("px", "UInt(32)", "2", ("D + 1", "D"), "north -> south: psum, psum_valid"),
    )




# ---------------------------------------------------------------------------
# The array's edge units (plan A1, ``mxu_systolic_array.py`` ``streams``):
# the driver turns one cycle of ``mxu_systolic_array.sv``'s ports into the
# west token of every row and the north weight token of every column; the
# sink reads the bottom row's psum tokens. Lane arrays are flat
# (``UInt(16)[N * D]``, lane ``r`` of cycle ``t`` at ``t * D + r``: P-8, and
# limitations item 14 for the shape), bit vectors are one word per cycle.
# The per-PE ``weight_commit_bank_i[r][c]`` has no port here: the bank bit
# rides east with the commit (``pe_unit``), so the driver sends row ``r``'s
# column-0 bit and a trace must be forward-consistent
# (``mxu_systolic_array.random_trace(..., fwd_bank=True)``).
# ---------------------------------------------------------------------------


@unit(memories=("RST", "CMT", "CBK", "LHS", "LHV", "RHS", "RHV"),
      writes=("lhsx", "wx"), parameters=("N", "D"))
def array_drive(rst: UInt(1)[N], cmt: UInt(16)[N], cbk: UInt(16)[N], lhs: UInt(16)[N * D],
                lhv: UInt(16)[N], rhs: UInt(16)[N * D], rhv: UInt(32)[N]):
    for t in range(N):
        r_n: UInt(1) = rst[t]
        cm: UInt(16) = cmt[t]
        cb: UInt(16) = cbk[t]
        lv: UInt(16) = lhv[t]
        rv: UInt(32) = rhv[t]
        with allo.meta_for(D) as row:
            wt: UInt(32) = 0
            wt[0:16] = lhs[t * D + row]
            wt[16] = lv[row]
            wt[17] = cm[row]
            wt[18] = cb[row]
            wt[19] = r_n
            lhsx[row, 0].put(wt)
        with allo.meta_for(D) as col:
            w16: UInt(16) = rhs[t * D + col]
            nw: UInt(64) = 0
            nw[0:16] = w16  # weight_in = {BANKS{rhs_i[col]}}
            nw[16:32] = w16
            nw[32] = rv[2 * col]
            nw[33] = rv[2 * col + 1]
            wx[0, col].put(nw)


@unit(memories=("RES", "RSV"), reads=("px",), parameters=("N", "D"))
def array_sink(res: UInt(32)[N * D], rsv: UInt(16)[N]):
    for t in range(N):
        v: UInt(16) = 0
        with allo.meta_for(D) as col:
            sp: UInt(32) = px[D, col].get()
            po: UInt(32) = 0
            po[0:24] = sp[0:24]
            res[t * D + col] = po
            v[col] = sp[24]
        rsv[t] = v
