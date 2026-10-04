# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bit-level numpy references for MiniTPU's arithmetic units.

Each reference states the RTL's semantics, not IEEE's: where the two differ,
the difference is named here and was measured against the RTL with
``harness/rtl.py``. The Allo side is compared with the RTL directly; these
references exist to *explain* a mismatch, not to replace the RTL as oracle.
"""

import numpy as np


def _bf16_to_f32(x):
    return (np.asarray(x, dtype=np.uint32) << 16).view(np.float32)


def f32_to_bf16_rne(f):
    """float32 bits -> bf16 bits, round to nearest even, NaN kept with sign."""
    f = np.asarray(f, dtype=np.float32).view(np.uint32)
    out = ((f + 0x7FFF + ((f >> 16) & 1)) >> 16).astype(np.uint16)
    nan = (((f >> 23) & 0xFF) == 0xFF) & ((f & 0x7FFFFF) != 0)
    out[nan] = (((f[nan] >> 31) << 15) | 0x7FC0).astype(np.uint16)
    return out


def ieee_bf16_add(a, b):
    """IEEE semantics: exact sum, one RNE rounding to bf16.

    fp32 has 24 significand bits >= 2*8+2, so rounding the exact sum to fp32
    first and then to bf16 is innocuous (no double-rounding error).
    """
    with np.errstate(all="ignore"):
        return f32_to_bf16_rne(_bf16_to_f32(a) + _bf16_to_f32(b))


def vpu_bf16_add(a, b):
    """``vpu_bf16_add.sv`` at b3ba0a4d: IEEE RNE add with subnormals, except

    * every NaN result is the positive canonical NaN ``0x7FC0`` (IEEE keeps a
      sign; MiniTPU's ``tb/tb_bf16_add.cpp`` skips specials, so its own test
      does not see this);
    * ``(+0) + (-0)`` is ``-0`` (IEEE RNE gives ``+0``). Measured on the RTL
      2026-10-02 over 251,936 vectors (corners crossed, ties, random): these
      two are the only differences.
    """
    a = np.asarray(a, dtype=np.uint16)
    b = np.asarray(b, dtype=np.uint16)
    out = ieee_bf16_add(a, b)
    nan = ((out & 0x7F80) == 0x7F80) & ((out & 0x7F) != 0)
    out[nan] = 0x7FC0
    zz = ((a & 0x7FFF) == 0) & ((b & 0x7FFF) == 0) & (a != b)
    out[zz] = b[zz]  # (+0)+(-0) -> -0 measured; (-0)+(+0) -> +0 measured
    return out


# ---------------------------------------------------------------------------
# Exact rounding, shared by the units below. A value is ``sign``, an integer
# magnitude ``mag`` and a power-of-two ``scale`` (value = mag * 2**scale), so
# products and sums are exact before the one rounding the hardware does.
# Formats have an 8-bit exponent, bias 127, and ``frac`` fraction bits: bf16
# is frac=7, MiniTPU's acc24 is frac=15.

ACC24_NAN = 0x7FC000  # sign 0, exponent 0xff, fraction MSB: every unit's NaN


def _decode(x, frac):
    """Fields of a 1+8+``frac`` float: sign, significand, effective exponent."""
    x = np.asarray(x, dtype=np.int64)
    sign = (x >> (8 + frac)) & 1
    exp = (x >> frac) & 0xFF
    f = x & ((1 << frac) - 1)
    sig = np.where(exp != 0, f | (1 << frac), f)
    eeff = np.maximum(exp, 1)  # subnormals share exponent 1's scale
    return sign, exp, f, sig, eeff


def _bitlen(m):
    n = np.zeros(m.shape, dtype=np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        big = (m >> s) > 0
        n += s * big
        m = np.where(big, m >> s, m)
    return n + (m > 0)


def _pack(sign, mag, scale, frac):
    """Round ``mag * 2**scale`` once to nearest-even in the 1+8+``frac`` format.

    Gradual underflow, overflow to signed infinity, a zero keeps ``sign``.
    ``mag`` must be below 2**62.
    """
    sign = np.asarray(sign, dtype=np.int64)
    mag = np.asarray(mag, dtype=np.int64)
    scale = np.asarray(scale, dtype=np.int64)
    n = _bitlen(mag)
    e = np.maximum(scale + n - 1, -126)  # exponent of the result's hidden bit
    shift = (e - frac) - scale  # bits below the result's LSB
    sh = np.clip(shift, 0, 62)
    q = mag >> sh
    rem = mag & ((np.int64(1) << sh) - 1)
    half = np.where(sh > 0, np.int64(1) << np.maximum(sh - 1, 0), 0)
    up = (sh > 0) & ((rem > half) | ((rem == half) & ((q & 1) == 1)))
    up &= shift <= 62  # beyond, the value is under half an LSB
    q = np.where(shift > 0, q + up, mag << np.clip(-shift, 0, 62))
    carry = q >> (frac + 1) != 0
    q = np.where(carry, q >> 1, q)
    e = e + carry
    normal = q >> frac != 0
    biased = np.where(normal, e + 127, 0)
    out = (sign << (8 + frac)) | (biased << frac) | (q & ((1 << frac) - 1))
    inf = (sign << (8 + frac)) | (0xFF << frac)
    return np.where(biased >= 255, inf, out)


def _mul_exact(a, b):
    """bf16 x bf16, exact: sign, 16-bit magnitude, scale."""
    sa, _, _, ma, ea = _decode(a, 7)
    sb, _, _, mb, eb = _decode(b, 7)
    return sa ^ sb, ma * mb, ea + eb - 2 * (127 + 7)


def _bf16_class(x):
    x = np.asarray(x, dtype=np.int64)
    nan = ((x & 0x7F80) == 0x7F80) & ((x & 0x7F) != 0)
    return nan, (x & 0x7FFF) == 0x7F80, (x & 0x7FFF) == 0


def _mul_specials(a, b):
    """NaN and Inf of a bf16 product: ``(nan, inf)`` masks.

    IEEE and the RTL agree on which operand pairs these are, because the RTL's
    "Inf x 0" test reads the literal pattern: ``Inf x subnormal`` is ``Inf``
    there too, even though the RTL flushes a subnormal operand otherwise.
    """
    na, ia, za = _bf16_class(a)
    nb, ib, zb = _bf16_class(b)
    nan = na | nb | (ia & zb) | (ib & za)
    return nan, ~nan & (ia | ib)


def ieee_bf16_mul(a, b):
    """IEEE: exact product, one RNE rounding to bf16, gradual underflow.

    A NaN result is ``0x7FC0`` with the product's sign (IEEE leaves the sign
    of a NaN unspecified; only "is NaN" is compared).
    """
    s, m, sc = _mul_exact(a, b)
    nan, inf = _mul_specials(a, b)
    out = np.where(inf, (s << 15) | 0x7F80, _pack(s, m, sc, 7))
    return np.where(nan, (s << 15) | 0x7FC0, out).astype(np.uint16)


def _ftz_mul(a, b, frac, nan_value):
    """The two multipliers' shared structure, as the RTL orders its tests."""
    s, m, sc = _mul_exact(a, b)
    nan, inf = _mul_specials(a, b)
    ea = (np.asarray(a, dtype=np.int64) >> 7) & 0xFF
    eb = (np.asarray(b, dtype=np.int64) >> 7) & 0xFF
    flush = (ea == 0) | (eb == 0)  # a zero or subnormal operand
    flush |= sc + _bitlen(m) - 1 < -126  # |exact product| < 2**-126
    top = 8 + frac
    out = np.where(flush, s << top, _pack(s, m, sc, frac))
    out = np.where(inf, (s << top) | (0xFF << frac), out)
    return np.where(nan, nan_value, out)


def vpu_bf16_mul(a, b):
    """``vpu_bf16_mul`` and ``vpu_bf16_mul_pipe`` at b3ba0a4d.

    The exact product rounded once to nearest-even, overflowing to signed
    infinity, except:

    * a subnormal operand is zero: the result is the signed zero, not the
      (possibly normal) product (flush-to-zero on input);
    * a product below ``2**-126`` before rounding is the signed zero, even one
      that would round up to the smallest normal (flush-to-zero on output);
    * ``Inf x subnormal`` is ``Inf``, not NaN: the Inf x 0 test reads the bit
      pattern before the input flush (ARITHMETIC.md section 6);
    * every NaN result is ``+0x7FC0``.
    """
    return _ftz_mul(a, b, 7, 0x7FC0).astype(np.uint16)


def ieee_bf16_mul_acc24(a, b):
    """IEEE: bf16 x bf16 rounded once into acc24 (1+8+15, gradual underflow).

    Exact unless the product is below acc24's smallest normal.
    """
    s, m, sc = _mul_exact(a, b)
    nan, inf = _mul_specials(a, b)
    out = np.where(inf, (s << 23) | 0x7F8000, _pack(s, m, sc, 15))
    return np.where(nan, (s << 23) | ACC24_NAN, out).astype(np.uint32)


def mxu_bf16_mul_acc24(a, b):
    """``mxu_bf16_mul_acc24`` at b3ba0a4d: the exact product in acc24, except

    * a subnormal operand is zero (flush on input), as in ``vpu_bf16_mul``;
    * a product below ``2**-126`` is the signed zero although acc24 has
      subnormals that could hold it (flush on output);
    * a product of ``2**128`` or more is signed infinity (acc24's own range);
    * ``Inf x subnormal`` is ``Inf``; every NaN is ``+0x7FC000``.
    """
    return _ftz_mul(a, b, 15, ACC24_NAN).astype(np.uint32)


_GAP = 40  # far operands become a sticky unit; RNE cannot tell the difference


def _add_exact(a, b, frac):
    """``a + b`` (finite) as sign, magnitude, scale; |a|>=|b| ordering inside."""
    sa, _, _, ma, ea = _decode(a, frac)
    sb, _, _, mb, eb = _decode(b, frac)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    low = (1 << (8 + frac)) - 1
    a_large = (a & low) >= (b & low)
    sl, ml, el = np.where(a_large, sa, sb), np.where(a_large, ma, mb), np.where(a_large, ea, eb)
    ss, ms, es = np.where(a_large, sb, sa), np.where(a_large, mb, ma), np.where(a_large, eb, ea)
    gap = el - es
    small = np.where(gap <= _GAP, ms << np.clip(_GAP - gap, 0, _GAP), (ms != 0).astype(np.int64))
    mag = np.where(sl == ss, (ml << _GAP) + small, (ml << _GAP) - small)
    return sl, mag, el - 127 - frac - _GAP


def ieee_acc24_add(a, b):
    """IEEE binary-style add in acc24: RNE once, gradual underflow, overflow
    to Inf, ``(+0)+(-0) = +0``; NaN keeps no particular sign."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    s, m, sc = _add_exact(a, b, 15)
    s = np.where(m == 0, ((a & b) >> 23) & 1, s)  # exact zero: +0 unless both -0
    out = _pack(s, m, sc, 15)
    isnan = ((a >> 15) & 0xFF == 0xFF) & ((a & 0x7FFF) != 0), ((b >> 15) & 0xFF == 0xFF) & ((b & 0x7FFF) != 0)
    ia, ib = (a & 0x7FFFFF) == 0x7F8000, (b & 0x7FFFFF) == 0x7F8000
    nan = isnan[0] | isnan[1] | (ia & ib & ((a ^ b) >> 23 == 1))
    out = np.where(ia, a, np.where(ib, b, out))
    out = np.where(nan, ACC24_NAN | (((a | b) >> 23 & 1) << 23), out)
    return out.astype(np.uint32)


def mxu_acc24_add(a, b):
    """``mxu_acc24_add_pipe`` at b3ba0a4d: IEEE RNE add in acc24 with
    subnormals and overflow to Inf (``ieee_acc24_add``), except

    * every NaN result is ``+0x7FC000``;
    * an operand whose 23 low bits are zero is a bypass: the result is the
      other operand bit for bit, so ``(+0) + (-0) = -0`` and
      ``(-0) + (+0) = +0`` (IEEE: ``+0`` for both), as in ``vpu_bf16_add``.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    out = ieee_acc24_add(a, b).astype(np.int64)
    nan = ((out >> 15) & 0xFF == 0xFF) & ((out & 0x7FFF) != 0)
    out = np.where(nan, ACC24_NAN, out)
    za, zb = (a & 0x7FFFFF) == 0, (b & 0x7FFFFF) == 0
    special = nan | ((a >> 15) & 0xFF == 0xFF) | ((b >> 15) & 0xFF == 0xFF)
    out = np.where(~special & za, b, np.where(~special & zb, a, out))
    return out.astype(np.uint32)


# vpu_pkg::vpu_alu_op_e, in declaration order (logic [3:0], from 0).
ALU_OPS = ("ADD", "SUB", "MUL", "MOV", "MAX", "MIN", "AND", "OR", "XOR")
ALU_OP = {name: i for i, name in enumerate(ALU_OPS)}


def bf16_gt(a, b):
    """``vpu_pkg::bf16_gt``: unsigned compare of a sign-flipped key.

    A total order on bit patterns: ``-NaN < -Inf < ... < -0 < +0 < ... <
    +Inf < +NaN``, and NaNs ordered among themselves by payload.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    key = lambda x: np.where(x >> 15 == 1, ~x & 0xFFFF, x ^ 0x8000)
    return key(a) > key(b)


def vpu_alu(op, a, b):
    """``vpu_alu`` at b3ba0a4d, one lane, by ``vpu_alu_op_e`` value.

    * ADD is ``vpu_bf16_add``; SUB is ``vpu_bf16_add(a, b ^ 0x8000)``, so
      ``(+0) - (+0) = -0`` (IEEE: ``+0``) and ``(-0) - (-0) = +0``;
    * MUL is ``vpu_bf16_mul`` (flush-to-zero both ways);
    * MAX / MIN select by ``bf16_gt``: ``-0 < +0``, a positive NaN wins MAX and
      a negative NaN wins MIN, payload and sign kept; a NaN on the losing side
      is dropped (neither IEEE 754-2019 ``maximum``, which returns NaN, nor
      ``maximumNumber``, which drops every NaN);
    * MOV, AND, OR, XOR and the unused codes 9..15 all return ``a`` bit for
      bit: the result mux's ``default`` arm. AND/OR/XOR are declared in the
      enum and listed in ``docs/UNITS.md`` but not implemented; the decoder
      never issues them.
    """
    op = np.asarray(op, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    gt = bf16_gt(a, b)
    out = np.select(
        [op == 0, op == 1, op == 2, op == 4, op == 5],
        [vpu_bf16_add(a, b), vpu_bf16_add(a, b ^ 0x8000), vpu_bf16_mul(a, b),
         np.where(gt, a, b), np.where(gt, b, a)],
        a,
    )
    return out.astype(np.uint16)


def ieee_vpu_alu(op, a, b):
    """What the op names promise: IEEE add/sub/mul, IEEE 754-2019
    ``maximum``/``minimum`` (NaN wins, ``-0 < +0``), bitwise AND/OR/XOR."""
    op = np.asarray(op, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    nan = lambda x: ((x & 0x7F80) == 0x7F80) & ((x & 0x7F) != 0)
    gt = bf16_gt(a, b)
    anynan = nan(a) | nan(b)
    mx = np.where(anynan, 0x7FC0, np.where(gt, a, b))
    mn = np.where(anynan, 0x7FC0, np.where(gt, b, a))
    out = np.select(
        [op == 0, op == 1, op == 2, op == 4, op == 5, op == 6, op == 7, op == 8],
        [ieee_bf16_add(a, b), ieee_bf16_add(a, b ^ 0x8000), ieee_bf16_mul(a, b),
         mx, mn, a & b, a | b, a ^ b],
        a,
    )
    return out.astype(np.uint16)


# Scalar predicates for classification rules (units' DEVIATIONS / EXPLAIN).
def is_nan(x, frac=7):
    x = int(x)
    return (x >> frac) & 0xFF == 0xFF and x & ((1 << frac) - 1) != 0


def exp_field(x, frac=7):
    return (int(x) >> frac) & 0xFF


def is_zero(x, frac=7):
    return int(x) & ((1 << (8 + frac)) - 1) == 0


# ---------------------------------------------------------------------------
# U2: storage units, as trace references.
#
# A storage unit is a function from a command trace to a response trace
# (``rtl.py``, the ``trace`` shape). Each reference below takes the command
# trace as ``{port: uint64[n, nwords]}`` and returns ``(resp, reason)``:
# ``resp[port]`` is ``uint64[n, nwords]`` in ``rtl.py``'s row convention and
# ``reason[port]`` is a ``str[n]`` that is ``""`` on a *defined* slot and
# otherwise names why the slot is undefined. A masked slot still carries the
# reference's best guess of what MiniTPU's simulation model returns there, so
# ``characterize`` can say how often the guess holds -- but only defined slots
# are held to the RTL. A third result, ``events``, counts the illegal or
# quirky commands the trace issued, by kind.

UNINIT = "uninit"  # a read of state never written (no reset clears it)


def _col(cmd, p):
    return np.asarray(cmd[p], dtype=np.uint64).reshape(len(cmd[p]), -1)


def _bit(cmd, p):
    return _col(cmd, p)[:, 0].astype(bool)


def _addr(cmd, p):
    return _col(cmd, p)[:, 0].astype(np.int64)


def vpu_regfile_trace(cmd, width, depth=32):
    """``vpu_regfile.sv``: three asynchronous reads, one synchronous write.

    Row ``t`` of ``rdata_{a,b,c}_o`` (sampled before edge ``t + 1``) is the
    entry as it was before cycle ``t``'s write: a read and a write of one
    VREG in one cycle return the **old** value, and a write is seen from the
    next cycle (write visibility 1). ``rst_ni`` does nothing. A read of a VREG
    never written is ``uninit``: the memory has no reset.
    """
    n = len(cmd["we_i"])
    nw = (width + 63) // 64
    mem = np.zeros((depth, nw), dtype=np.uint64)
    written = np.zeros(depth, dtype=bool)
    we, wa, wd = _bit(cmd, "we_i"), _addr(cmd, "waddr_i"), _col(cmd, "wdata_i")
    ra = {x: _addr(cmd, f"raddr_{x}_i") for x in "abc"}
    resp = {f"rdata_{x}_o": np.zeros((n, nw), dtype=np.uint64) for x in "abc"}
    reason = {f"rdata_{x}_o": np.full(n, "", dtype=object) for x in "abc"}
    for t in range(n):
        for x in "abc":
            a = ra[x][t]
            resp[f"rdata_{x}_o"][t] = mem[a]
            if not written[a]:
                reason[f"rdata_{x}_o"][t] = UNINIT
        if we[t]:
            mem[wa[t]] = wd[t]
            written[wa[t]] = True
    return resp, reason, {}


def vpu_word_array_trace(cmd, nw, read_latency=3, dma_read_latency=2, ww_winner="dma"):
    """``vpu_word_array.sv``'s simulation model (the ``ifndef SYNTHESIS`` branch).

    Two symmetric ports, ``compute`` and ``dma``. A port with ``en`` reads the
    word as it was before the cycle (NBA), so a read-during-write on one port
    returns the **old** word; the read enters a pipe of the port's latency
    (3 / 2) and leaves it on ``*_rdata_o`` in row ``t + L - 1`` (sampled after
    edge ``t + 1``, so latency ``L`` edges). Writes are seen from the next
    cycle (visibility 1), from either port. Undefined slots:

    * ``no read``: the read issued ``L`` cycles earlier had ``en`` low. The
      model's pipe then holds its last read (the guess), and XPM's
      ``no_change`` port holds its output too, but nothing promises it;
    * ``write cycle``: that read had ``we`` high. The model returns the old
      word (the guess); XPM ``WRITE_MODE no_change`` holds the previous output
      instead, so the board differs from the model here;
    * ``collision``: that read touched the word the other port wrote in the
      same cycle (``vpu_vmem_simd.sv``'s assertion; undefined on the board);
    * ``ww collision``: a read of a word both ports wrote in one cycle, until
      it is written again. The model's two ``always_ff`` blocks both write it;
      ``ww_winner`` is the guess of which lands;
    * ``uninit``: a read of a word never written.
    """
    n = len(cmd["compute_en_i"])
    ports = {"compute": read_latency, "dma": dma_read_latency}
    en = {p: _bit(cmd, f"{p}_en_i") for p in ports}
    we = {p: _bit(cmd, f"{p}_we_i") & en[p] for p in ports}
    ad = {p: _addr(cmd, f"{p}_addr_i") for p in ports}
    wd = {p: _col(cmd, f"{p}_wdata_i") for p in ports}
    mem, state = {}, {}  # word -> data, word -> "" | reason
    zero = np.zeros(nw, dtype=np.uint64)
    pipe = {p: [(zero, "no read")] * L for p, L in ports.items()}
    resp = {f"{p}_rdata_o": np.zeros((n, nw), dtype=np.uint64) for p in ports}
    reason = {f"{p}_rdata_o": np.full(n, "", dtype=object) for p in ports}
    events = {}
    for t in range(n):
        coll = (en["compute"][t] and en["dma"][t] and ad["compute"][t] == ad["dma"][t]
                and (we["compute"][t] or we["dma"][t]))
        if coll:
            k = "ww collision" if we["compute"][t] and we["dma"][t] else "rw collision"
            events[k] = events.get(k, 0) + 1
        for p, L in ports.items():
            o = "dma" if p == "compute" else "compute"
            if en[p][t]:
                a = ad[p][t]
                v = mem.get(a, zero)
                why = state.get(a, UNINIT)
                if we[p][t]:
                    why = "write cycle"
                elif coll and we[o][t]:
                    why = "collision"
                head = (v, why)
            else:
                head = (pipe[p][0][0], "no read")
            pipe[p] = [head] + pipe[p][:-1]
            resp[f"{p}_rdata_o"][t], reason[f"{p}_rdata_o"][t] = pipe[p][L - 1]
        if coll and we["compute"][t] and we["dma"][t]:
            a = ad["dma"][t]
            mem[a] = wd[ww_winner][t]
            state[a] = "ww collision"
        else:
            for p in ports:
                if we[p][t]:
                    mem[ad[p][t]] = wd[p][t]
                    state[ad[p][t]] = ""
    return resp, reason, events


def vpu_fifo_trace(cmd, width, depth):
    """``vpu_fifo.sv``: a ring of ``depth`` entries with a reset count.

    Row ``t`` (sampled before edge ``t + 1``) shows the state cycle ``t``
    starts in: ``empty_o``/``full_o`` from the count, ``pop_data_o =
    mem[rd_ptr]`` (first-word-fall-through, asynchronous). The update is the
    RTL's, illegal commands included, so the flags are defined on every trace
    after the first reset:

    * a push is taken if not full, or full with a pop (pass-through);
    * a pop is taken if not empty, or empty with a push: the pointers both
      move, the count stays 0 and **the pushed word is lost** -- the reader
      gets stale ``mem[rd_ptr]`` and no assertion fires (``events``:
      ``push+pop on empty``);
    * a push on full without a pop is dropped (``overflow``, asserted in
      simulation only); a pop on empty without a push does nothing
      (``underflow``, asserted).

    Undefined slots: everything before the first reset (``pre-reset``), and
    ``pop_data_o`` while empty (``empty``; the guess is the stale entry,
    ``uninit`` if that entry was never written). ``rst_ni`` low clears the
    pointers and the count at the edge, not ``mem``.
    """
    n = len(cmd["push_i"])
    nw = (width + 63) // 64
    rst = ~_bit(cmd, "rst_ni")
    push, pop, pd = _bit(cmd, "push_i"), _bit(cmd, "pop_i"), _col(cmd, "push_data_i")
    mem = np.zeros((depth, nw), dtype=np.uint64)
    written = np.zeros(depth, dtype=bool)
    known, count, rd, wr = False, 0, 0, 0
    resp = {p: np.zeros((n, nw if p == "pop_data_o" else 1), dtype=np.uint64)
            for p in ("pop_data_o", "empty_o", "full_o")}
    reason = {p: np.full(n, "", dtype=object) for p in resp}
    events = {}

    def ev(k):
        events[k] = events.get(k, 0) + 1

    for t in range(n):
        empty, full = count == 0, count == depth
        resp["empty_o"][t, 0], resp["full_o"][t, 0] = empty, full
        resp["pop_data_o"][t] = mem[rd]
        if not known:
            for p in resp:
                reason[p][t] = "pre-reset"
        elif empty:
            reason["pop_data_o"][t] = "empty" if written[rd] else UNINIT
        if rst[t]:
            known, count, rd, wr = True, 0, 0, 0
            continue
        if not known:
            continue
        do_push = push[t] and (not full or pop[t])
        do_pop = pop[t] and (not empty or push[t])
        if push[t] and full and not pop[t]:
            ev("overflow (push dropped)")
        if pop[t] and empty and not push[t]:
            ev("underflow (pop ignored)")
        if push[t] and pop[t] and empty:
            ev("push+pop on empty (word lost)")
        if push[t] and pop[t] and full:
            ev("push+pop on full (pass-through)")
        if do_push:
            mem[wr] = pd[t]
            written[wr] = True
            wr = (wr + 1) % depth
        if do_pop:
            rd = (rd + 1) % depth
        count += int(do_push) - int(do_pop)
    return resp, reason, events


# ---------------------------------------------------------------------------
# U3: cross-lane and special-function units.
#
# ``sfu``: a bit-level model of ``sfu.sv``'s logic. Its DATA -- the two
# 2048-entry ROMs (``gelu_bf16.mem``, ``exp_bf16.mem``) and the two 32-entry
# piecewise-linear case tables of ``recip_entry``/``rsqrt_entry`` -- is read
# from the pinned clone, not transcribed: the model is independent in its
# logic (magnitude, address fold, interpolation, packing, special cases) and
# shares the tables with the RTL. A wrong table therefore passes here, as it
# passes ``tb_sfu_equiv``; ``tb_sfu_math_sweep`` (the functions themselves)
# is the check that covers the tables.

SFU_OPS = ("GELU", "EXP", "RECIP", "RSQRT")  # vpu_pkg::vpu_sfu_op_e, from 0
SFU_OP = {name: i for i, name in enumerate(SFU_OPS)}
_SFU_TABLES = {}


def _sfu_tables():
    if _SFU_TABLES:
        return _SFU_TABLES
    import os
    import re

    from examples.minitpu.harness import rtl

    home = rtl.minitpu_home()
    for name in ("gelu", "exp"):
        with open(os.path.join(home, f"src/core/sfu/{name}_bf16.mem"), encoding="utf-8") as f:
            words = [int(w, 16) for w in f.read().split() if not w.startswith("//")]
        assert len(words) == 2048, (name, len(words))
        _SFU_TABLES[name] = np.array(words, dtype=np.int64)
    with open(os.path.join(home, "src/core/sfu/sfu.sv"), encoding="utf-8") as f:
        src = f.read()
    for name in ("recip", "rsqrt"):
        body = src[src.index(f"function automatic logic [PWL_WORD_W-1:0] {name}_entry"):]
        body = body[: body.index("endfunction")]
        ent = dict((int(i), int(v, 16)) for i, v in
                   re.findall(rf"5'd(\d+):\s*{name}_entry\s*=\s*21'h([0-9A-Fa-f]+);", body))
        assert sorted(ent) == list(range(32)), name
        _SFU_TABLES[name] = np.array([ent[i] for i in range(32)], dtype=np.int64)
    return _SFU_TABLES


def _fp32_abs_q7(x):
    """``sfu.sv`` ``fp32_abs_q7`` on a bf16 operand (its fp32 view's low 16
    bits are zero): |x| in Q7, truncated toward zero, saturated at 0x1fff."""
    e = (x >> 7) & 0xFF
    mant = (0x80 | (x & 0x7F)) << 16  # {1, value[22:0]}
    rs = 16 - (e - 127)
    sh = np.clip(rs, 0, 63)
    v = mant >> sh
    out = np.where(v > 0x1FFF, 0x1FFF, v)
    out = np.where(rs <= 0, 0x1FFF, out)
    return np.where((e == 0) | (rs >= 24), 0, out)


def sfu(op, x):
    """``sfu.sv`` at b3ba0a4d (module comment and ARITHMETIC.md §6/§9):

    * vgelu/vexp: a 2048-entry ROM indexed by the Q7 magnitude (truncated),
      folded around the table's centre; vgelu outside [-8, 8) is 0 or x;
      vexp of x > 0 (and of +-0) is 1.0, below -16 it is 0;
    * vrecip/vrsqrt: 32-bin piecewise-linear tables (13 value + 8 slope
      bits), the interpolation's 13-bit result truncated to 7 mantissa bits;
    * every NaN in is ``0x7fc0``; vrecip(+-Inf) = +-0, vrecip(+-0) = +-Inf;
      vrsqrt(x <= 0) = NaN, vrsqrt(+Inf) = 0.
    """
    T = _sfu_tables()
    op = np.asarray(op, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64) & 0xFFFF
    sign, e, frac = x >> 15, (x >> 7) & 0xFF, x & 0x7F
    mag = _fp32_abs_q7(x)
    nan = (e == 0xFF) & (frac != 0)
    inf = (e == 0xFF) & (frac == 0)
    zero = (x & 0x7FFF) == 0
    nonpos = (sign == 1) | zero
    gelu_addr = np.where(sign == 1, 1024 - 1 - mag, 1024 + mag) & 0x7FF
    exp_addr = (2048 - 1 - mag) & 0x7FF
    # fp32 view {x, 16'b0}: fp32[22:18] = x[6:2], fp32[17:6] = {x[1:0], 10'b0}
    recip_addr, recip_pos = (x >> 2) & 0x1F, (x & 3) << 10
    rsqrt_addr = (((~x >> 7) & 1) << 4) | ((x >> 3) & 0xF)  # {~fp32[23], fp32[22:19]}
    rsqrt_pos = (x & 7) << 9  # fp32[18:7]
    rw, sw = T["recip"][recip_addr], T["rsqrt"][rsqrt_addr]
    r_bias = (((rw >> 8) & 0x1FFF) + 8321) & 0x3FFF
    s_bias = (((sw >> 8) & 0x1FFF) + 8322) & 0x3FFF
    r_prod, s_prod = (rw & 0xFF) * recip_pos, (sw & 0xFF) * rsqrt_pos
    r_int = ((r_bias - ((r_prod >> 11) & 0x1FF)) & 0x3FFF) & 0x1FFF
    s_int = ((s_bias - ((s_prod >> 11) & 0x1FF)) & 0x3FFF) & 0x1FFF
    s_exp = np.where(e & 1, ((379 - e) & 0x1FF) >> 1, ((380 - e) & 0x1FF) >> 1) & 0xFF
    gelu = np.where(nan, 0x7FC0, np.where(zero, 0, np.where(
        mag >= 1024, np.where(sign == 1, 0, x), T["gelu"][gelu_addr])))
    expv = np.where(nan, 0x7FC0, np.where(zero | (sign == 0), 0x3F80, np.where(
        mag >= 2048, 0, T["exp"][exp_addr])))
    recip = np.where(nan, 0x7FC0, np.where(inf, sign << 15, np.where(
        zero, (sign << 15) | 0x7F80,
        (sign << 15) | (((253 - e) & 0xFF) << 7) | ((r_int >> 6) & 0x7F))))
    rsqrt = np.where(nan | nonpos, 0x7FC0, np.where(
        inf, 0, (s_exp << 7) | ((s_int >> 6) & 0x7F)))
    out = np.choose(op & 3, [gelu, expv, recip, rsqrt])
    return out.astype(np.uint16)


def ieee_sfu(op, x):
    """The functions themselves in float64, rounded once to bf16 (RNE, NaN
    kept with its sign): GELU with erf, exp, 1/x, 1/sqrt(x)."""
    from math import erf

    op = np.asarray(op, dtype=np.int64)
    xv = _bf16_to_f32(np.asarray(x, dtype=np.uint16)).astype(np.float64)
    verf = np.vectorize(erf, otypes=[np.float64])
    with np.errstate(all="ignore"):
        fin = np.isfinite(xv)
        g = np.where(fin, 0.5 * xv * (1.0 + verf(np.where(fin, xv, 0.0) / np.sqrt(2.0))),
                     np.where(xv > 0, xv, np.where(np.isnan(xv), xv, 0.0)))
        r = np.choose(op & 3, [g, np.exp(xv), 1.0 / xv, 1.0 / np.sqrt(xv)])
    return f32_to_bf16_rne(r.astype(np.float32))


# ``xlu_reduction_tree``: a pairwise tree, element e of level l combining
# elements 2e (as ``a``) and 2e+1 (as ``b``) of level l-1.

def bf16_max(a, b):
    """``xlu_reduction_tree``'s max select: ``bf16_gt(a, b) ? a : b``."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return np.where(bf16_gt(a, b), a, b).astype(np.uint16)


def reduce_tree_levels(op, leaves):
    """``[value[0], ..., value[LEVELS]]`` for leaves ``uint16[m, N]`` under a
    per-row op (0 = SUM, 1 = MAX): every level of the tree, so the root is
    ``[-1][:, 0]`` and the per-sublane tap is level ``log2(NUM_LANES)``."""
    v = np.asarray(leaves, dtype=np.uint16)
    op = np.asarray(op, dtype=np.int64).reshape(-1, 1)
    levels = [v]
    while v.shape[1] > 1:
        a, b = v[:, 0::2], v[:, 1::2]
        v = np.where(op == 0, vpu_bf16_add(a, b), bf16_max(a, b)).astype(np.uint16)
        levels.append(v)
    return levels


def _split16(col, n):
    """``uint64[m, nw]`` packed port -> ``int64[m, n]`` of 16-bit fields."""
    col = np.asarray(col, dtype=np.uint64)
    out = np.zeros((len(col), n), dtype=np.int64)
    for i in range(n):
        out[:, i] = ((col[:, (16 * i) // 64] >> np.uint64((16 * i) % 64)) & np.uint64(0xFFFF)).astype(np.int64)
    return out


def _join16(fields, nw):
    """``int64[m, n]`` of 16-bit fields -> packed ``uint64[m, nw]``."""
    fields = np.asarray(fields, dtype=np.uint64)
    out = np.zeros((len(fields), nw), dtype=np.uint64)
    for i in range(fields.shape[1]):
        out[:, (16 * i) // 64] |= (fields[:, i] & np.uint64(0xFFFF)) << np.uint64((16 * i) % 64)
    return out


def xlu_reduction_tree_trace(cmd, n_leaves, lanes):
    """``xlu_reduction_tree.sv``: ``1 + 2*LEVELS`` edges to the root, ``1 +
    2*LANE_LEVELS`` to the per-sublane tap, II=1, the op tag travelling with
    its wavefront. Rows are ``"post"`` (after edge ``t + 1``).

    The payload is never gated (leaves, both adder stages and the max
    registers load every cycle), so ``result_o`` row ``t`` is the tree of the
    data and op driven ``L - 1`` rows earlier on *every* cycle. Defined: the
    valids on every cycle after the first reset, the results on valid cycles.
    Other cycles are masked ``no valid`` (the guess is that window function);
    rows whose window reaches back to or before a reset cycle are ``fill``
    (the reset clears the adders' ``result_o``, the leaves keep stale data).
    """
    n = len(cmd["valid_i"])
    levels_n = int(np.log2(n_leaves))
    tap = int(np.log2(lanes))
    rst = ~_bit(cmd, "rst_ni")
    valid = _bit(cmd, "valid_i")
    op = _col(cmd, "op_i")[:, 0].astype(np.int64)
    data = _split16(_col(cmd, "data_i"), n_leaves)
    lv = reduce_tree_levels(op, data)
    sub = n_leaves // lanes
    nw_lane = (sub * 16 + 63) // 64
    resp = {
        "valid_o": np.zeros((n, 1), dtype=np.uint64),
        "result_o": np.zeros((n, 1), dtype=np.uint64),
        "lane_valid_o": np.zeros((n, 1), dtype=np.uint64),
        "lane_result_o": np.zeros((n, nw_lane), dtype=np.uint64),
    }
    reason = {p: np.full(n, "", dtype=object) for p in resp}
    first_rst = np.flatnonzero(rst)
    first_rst = first_rst[0] if len(first_rst) else n
    last_rst = np.full(n, -1, dtype=np.int64)  # last reset cycle at or before t
    r = -1
    for t in range(n):
        if rst[t]:
            r = t
        last_rst[t] = r
    for vp, rp, depth, src in (("valid_o", "result_o", 1 + 2 * levels_n, None),
                               ("lane_valid_o", "lane_result_o", 1 + 2 * tap, tap)):
        for t in range(n):
            s = t - (depth - 1)  # the row whose command shows in row t
            if t < first_rst:
                reason[vp][t] = reason[rp][t] = "pre-reset"
                continue
            # valid_q is reset: rows whose source is at or before the last reset read 0
            if s < 0 or s <= last_rst[t]:
                resp[vp][t, 0] = 0
                reason[rp][t] = "fill"
                continue
            resp[vp][t, 0] = int(valid[s])
            if src is None:
                resp[rp][t, 0] = int(lv[-1][s, 0])
            else:
                resp[rp][t] = _join16(lv[src][s : s + 1], nw_lane)[0]
            if not valid[s]:
                reason[rp][t] = "no valid"
    return resp, reason, {}


def xlu_transpose_trace(cmd, lanes=16, sublanes=4):
    """``xlu_transpose.sv``: a ``lanes x lanes`` tile written whole rows at a
    time (beat ``q`` writes rows ``4q..4q+3`` from sublanes 0..3) and read
    through a registered crossing mux: ``read_data_o[s][l] = tile[l][4*idx+s]``
    one edge after the read (``"post"`` row ``t``), from the tile as it was
    before cycle ``t``'s write (both happen at edge ``t + 1``). The tile is
    never reset: a read touching an unwritten element is ``uninit``.
    ``read_data_q`` loads every cycle, so ``read_data_o`` is defined on
    ``read_valid_o`` cycles only (``no valid`` otherwise, guess kept).
    """
    n = len(cmd["read_valid_i"])
    nw = (lanes * sublanes * 16 + 63) // 64
    rst = ~_bit(cmd, "rst_ni")
    wv, wi = _bit(cmd, "write_valid_i"), _addr(cmd, "write_index_i")
    rv, ri = _bit(cmd, "read_valid_i"), _addr(cmd, "read_index_i")
    wd = _split16(_col(cmd, "write_data_i"), lanes * sublanes)  # field s*lanes + l
    tile = np.zeros((lanes, lanes), dtype=np.int64)
    known = np.zeros((lanes, lanes), dtype=bool)
    resp = {"read_valid_o": np.zeros((n, 1), dtype=np.uint64),
            "read_data_o": np.zeros((n, nw), dtype=np.uint64)}
    reason = {p: np.full(n, "", dtype=object) for p in resp}
    seen_rst = False
    for t in range(n):
        idx = ri[t]
        cols = [4 * idx + s for s in range(sublanes)]
        out = np.zeros(lanes * sublanes, dtype=np.int64)
        ok = True
        for s, c in enumerate(cols):
            out[s * lanes : (s + 1) * lanes] = tile[:, c]
            ok &= bool(known[:, c].all())
        resp["read_data_o"][t] = _join16(out.reshape(1, -1), nw)[0]
        if rst[t]:
            seen_rst = True
            resp["read_valid_o"][t, 0] = 0
        else:
            resp["read_valid_o"][t, 0] = int(rv[t])
        if not seen_rst:
            reason["read_valid_o"][t] = "pre-reset"
        if not rv[t] or rst[t]:
            reason["read_data_o"][t] = "no valid"
        elif not ok:
            reason["read_data_o"][t] = UNINIT
        if wv[t]:
            for s in range(sublanes):
                tile[4 * wi[t] + s, :] = wd[t, s * lanes : (s + 1) * lanes]
                known[4 * wi[t] + s, :] = True
    return resp, reason, {}
