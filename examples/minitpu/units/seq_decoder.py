# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: ``sequencer_decoder``, MiniTPU's bundle decoder (combinational).

A 128-bit VLIW bundle (``encoded_bundle_t``) to the decoded per-slot view
``bundle_fields_t`` (233 bits, the packed struct's own MSB-first layout,
flattened by ``units/rtl/u4_seq_decoder.sv``). It is a pure function: every
input word, defined or not, has one output, and nothing is masked.

Reference: ``ref_ctrl_decode.decode``, written from ``sequencer_pkg.sv``'s
layout. Cross-check (``python -m examples.minitpu.units.seq_decoder
--asm-check``): MiniTPU's own assembler ``board_package/asm.py`` decodes the
same words with its own unpackers, and every op its builder emits round-trips.

Stimulus: random words biased over every opcode (undefined V/M/C/S codes,
MEM kind 3, IMM claimed by several slots), every bundle of the two committed
board images (``board_package/images/*.bin``), and programs built with
``asm.AsmBuilder`` and scheduled by ``asm.schedule``. Seed: ``tb_bundle_encoding``
(DUT = the decoder; no clock, so every time step is a row).
"""

import glob
import os
import random
import sys

import numpy as np

import allo.dataflow as df
from allo.ir.types import UInt

from examples.minitpu.harness import minitpu_asm, rtl
from examples.minitpu.harness import ref_ctrl_decode as R
from examples.minitpu.harness.traces import rng_for

HERE = os.path.dirname(os.path.abspath(__file__))
PKGS = ["src/pkg/minitpu_config_pkg.sv", "src/core/vpu/vpu_pkg.sv", "src/core/sequencer/sequencer_pkg.sv"]
SOURCES = PKGS + ["src/core/sequencer/sequencer_decoder.sv", os.path.join(HERE, "rtl", "u4_seq_decoder.sv")]

RTL = rtl.RtlUnit(top="u4_seq_decoder", sources=SOURCES, inputs=[("bundle_i", 128)],
                  outputs=[("fields_o", R.FIELDS_W, "pre")], shape="trace", clk="clk_i",
                  rst_n="rst_n_unused")
INSTANCES = {"base": RTL}
DEFAULT = "base"
LATENCY_SOURCE = "combinational (sequencer_decoder.sv: always_comb; sequencer.sv decodes the live fetch head)"


def REF(inst, cmd):
    return R.decoder_trace(cmd)


# ---- stimulus ---------------------------------------------------------------
def random_word(rng):
    """A bundle with every slot drawn over its full code space."""
    w = 0
    w |= rng.choice([0] * 4 + list(range(32))) << 123  # V op, incl. 16..31 undefined
    w |= rng.getrandbits(15) << 108
    w |= rng.choice([0, 0, 1, 2, 3, 4, 5, 6, 7]) << 105  # M subop, 4..7 undefined
    w |= rng.getrandbits(5) << 100
    w |= rng.choice([0, 1, 1, 2, 2, 3]) << 98  # MEM kind, 3 undefined
    w |= rng.getrandbits(32) << 66
    w |= rng.getrandbits(13) << 53  # S slot, any op 0..7
    w |= rng.choice([0, 0, 1, 2, 3, 4, 5, 6, 7]) << 50  # C op, 6..7 undefined
    w |= rng.getrandbits(4) << 46
    w |= rng.getrandbits(24) << 22
    w |= rng.getrandbits(7) << 15
    w |= rng.choice([0, 0, rng.getrandbits(15)])  # reserved bits, unread
    return w


def image_words():
    out = []
    for p in sorted(glob.glob(os.path.join(rtl.minitpu_home(), "board_package", "images", "*.bin"))):
        data = open(p, "rb").read()
        out += [int.from_bytes(data[i:i + 16], "little") for i in range(0, len(data), 16)]
    return out


def builder_words(rng, n_progs=40):
    """Every op the builder emits, as single bundles and combined, plus
    scheduled random programs (the delay field set by asm.schedule)."""
    asm = minitpu_asm.load()
    b = asm.AsmBuilder()
    r = lambda k=5: rng.getrandbits(k)  # noqa: E731
    ops = {
        "v": [lambda: b.vadd(r(), r(), r()), lambda: b.vsub(r(), r(), r()), lambda: b.vmul(r(), r(), r()),
              lambda: b.vmax(r(), r(), r()), lambda: b.vmin(r(), r(), r()), lambda: b.vmov(r(), r()),
              lambda: b.vgelu(r(), r()), lambda: b.vexp(r(), r()), lambda: b.vrecip(r(), r()),
              lambda: b.vrsqrt(r(), r()), lambda: b.vredsum(r(), r()), lambda: b.vredmax(r(), r()),
              lambda: b.vlanesum(r(), r()), lambda: b.vlanemax(r(), r()), lambda: b.vtxin(r(), r(2)),
              lambda: b.vtxout(r(), r(2))],
        "m": [lambda: b.vmatload(rng.randrange(0, 29)), lambda: b.vmatpush(r()), lambda: b.vmatpop(r())],
        "x": [lambda: b.vld(r(), 4 * r(12)), lambda: b.vst(r(), 4 * r(12)),
              lambda: b.vld(r(), 4 * r(12), agu=True, shift=rng.randrange(2, 18), level=r(3)),
              lambda: b.vst(r(), 4 * r(12), agu=True, shift=rng.randrange(2, 18), level=r(3))],
        "d": [lambda: b.dma_load(r(1), 4 * r(12), 4 * rng.randrange(1, 4097), r(2), r(2)),
              lambda: b.dma_store(r(1), 4 * r(12), 4 * rng.randrange(1, 4097), r(2), r(2),
                                  disp=rng.randrange(-(1 << 23), 1 << 23))],
        "s": [lambda: b.smovi(r(2), r(24)), lambda: b.smov_arg(r(2), r(2)),
              lambda: b.saddi(r(2), r(2), r(24))],
        "f": [lambda: b.lbegin(r(16), lo=r(4), step=r(4)), lambda: b.lbegin_r(r(2), from_arg=bool(r(1)), lo=r(4), step=r(4)),
              lambda: b.loop_end(), lambda: b.wait_channel(rng.randrange(1, 4)), lambda: b.halt()],
    }
    out = [f() for fs in ops.values() for f in fs for _ in range(8)]
    # combined bundles: the builder refuses illegal pairs, so try and keep the accepted ones
    for _ in range(3000):
        slots = {k: rng.choice(ops[k])() for k in rng.sample(sorted(ops), rng.randrange(1, 5))}
        try:
            b.bundles.clear()
            b.bundle(**slots)
            out.append(b.bundles[-1])
        except (ValueError, TypeError):
            pass
    # scheduled straight-line programs: delays from asm.schedule
    for _ in range(n_progs):
        words = []
        for _ in range(rng.randrange(4, 20)):
            k = rng.choice(["v", "v", "x", "s", "m"])
            if k == "x":
                words.append(b.vld(r(), 4 * r(12)))
            elif k == "m":
                words.append(b.vmatpop(r()))
            else:
                words.append(rng.choice(ops[k])())
        words.append(b.halt())
        try:
            out += asm.schedule(words)
        except ValueError:
            pass
    return out


def directed():
    """IMM claimed by every combination of S / C / MEM; each C and kind code."""
    out = []
    base_imm = 0xABCDEF << 22
    for s_on in (0, 1):
        for c_op in range(8):
            for kind in range(4):
                for disp in (0, 1):
                    w = base_imm
                    if s_on:
                        w |= (1 << 65) | (2 << 62)  # SADDI
                    w |= c_op << 50 | 0xD << 46
                    w |= kind << 98 | (disp << 4 | 0x5A5A5A00) << 66
                    out.append(w)
    out += [1 << 65 | 1 << 62 | base_imm]  # SMOV_ARG does not own IMM
    out += [v << 123 | 0x7FFF << 108 for v in range(32)]
    return out


def traces(inst):
    rng = rng_for("seq_decoder", inst)
    return [("directed", {"bundle_i": directed()}, True),
            ("random", {"bundle_i": [random_word(rng) for _ in range(20000)]}, True),
            ("images", {"bundle_i": image_words()}, True),
            ("asm-builder", {"bundle_i": builder_words(rng)}, True)]


def seeds():
    """tb_bundle_encoding: the decoder alone, driven by # delays."""
    rows = R.seed_rows("tb_bundle_encoding", PKGS + ["src/core/sequencer/sequencer_decoder.sv"], "dut",
                       ["bundle_i", "fields_o"])
    return [("tb_bundle_encoding", "base", {"bundle_i": rows["bundle_i"]},
             {"fields_o": rows["fields_o"]}, True)]


def probes(inst):
    rng = rng_for("seq_decoder-probe", inst)
    words = [random_word(rng) for _ in range(8)]
    a, b2 = R.decode(words[0]), R.decode(words[1])
    assert a != b2
    cmd = {"bundle_i": [words[0]] * 8 + [words[1]] * 8}
    return [("bundle_i -> fields_o (comb)", 0, rtl.probe_trace(RTL, {"bundle_i": rtl.pack(cmd["bundle_i"], 128)},
                                                                "fields_o", 8))]


# ---- cross-check against asm.py ---------------------------------------------
def asm_check(words):
    """Hold asm.py's own unpackers to the reference decode on every word.
    Returns {check: (agree, total, [examples of disagreement])}."""
    asm = minitpu_asm.load()
    res = {}

    def note(k, ok, w):
        a = res.setdefault(k, [0, 0, []])
        a[1] += 1
        a[0] += int(ok)
        if not ok and len(a[2]) < 3:
            a[2].append(hex(w))

    for w in words:
        f = R.decode(w)
        cop = R.bits(w, 52, 50)
        try:
            cop_asm = asm.control_op(w)
            note("control_op", cop_asm == cop, w)
        except ValueError:
            note("control_op refuses undefined C op (RTL: nop)", cop in (6, 7), w)
            continue
        lb = asm.loop_begin_of(w)
        if f["l.valid"]:
            ok = lb is not None and lb.lo == f["l.lo"] and lb.step == f["l.step"] and (
                (lb.hi == f["l.hi"]) if not f["l.hi_from_reg"] else
                (lb.skip == f["l.skip"] and (lb.bound_arg if f["l.hi_from_arg"] else lb.bound_sreg) == f["l.hi_idx"]))
        else:
            ok = lb is None
        note("loop_begin_of", ok, w)
        d = asm.descriptor_of(w)
        if f["d.valid"]:
            disp = f["d.disp"] - (1 << 24) if f["d.disp"] >> 23 else f["d.disp"]
            ok = d is not None and (d["store"], d["channel"], d["vmem"], d["rows"], d["has_disp"], d["disp"],
                                    d["base_sreg"], d["stride_sreg"]) == (
                bool(f["d.is_store"]), f["d.channel_sel"], 4 * f["d.vmem_address"], 4 * (f["d.rows"] + 1),
                bool(f["d.has_disp"]), disp, f["d.base_sreg"], f["d.stride_sreg"])
        else:
            ok = d is None
        note("descriptor_of", ok, w)
        if f["x.valid"]:
            payload = R.bits(w, 97, 66)
            st, vreg, lit, agu, lvl, sh = asm.unpack_ldst(payload)
            note("unpack_ldst", (st, vreg, lit, agu, lvl, sh) == (
                f["x.op"], f["x.vreg_idx"], f["x.literal"], f["x.agu_valid"], f["x.agu_level"], f["x.agu_shift"]), w)
        s = R.bits(w, 65, 53)
        if s:
            v, op, rd, rs, use_iv, lvl = asm.unpack_s_slot(s)
            note("unpack_s_slot", (v, op, rd, rs, use_iv, lvl) == (
                f["s.valid"], f["s.op"], f["s.rd"], f["s.rs"], f["s.use_iv"], f["s.level"]), w)
        own = R.owners(w)
        note("_imm_owners", sorted(asm._imm_owners(w)) == sorted(
            k for k, v in (("S", own["s"]), ("C", own["c"]), ("MEM", own["mem"])) if v), w)
        # the scheduler's view: which VREGs a bundle writes, and when
        dd = asm._decode(w)
        writes = set()
        if f["v.alu_valid"]:
            writes.add((f["v.alu_vd"], asm.W_ALU))
        elif f["v.sfu_valid"]:
            writes.add((f["v.sfu_vd"], asm.W_SFU))
        elif f["v.reduce_valid"]:
            writes.add((f["v.reduce_vd"], asm.W_LANE_REDUCE if f["v.reduce_lane"] else asm.W_REDUCE))
        elif f["v.txout_valid"]:
            writes.add((f["v.txout_vd"], asm.W_TRANSPOSE))
        if f["x.valid"] and not f["x.op"]:
            writes.add((f["x.vreg_idx"], asm.W_VLD))
        if f["m.subop"] == 3:
            writes.add((f["m.reg_idx"], asm.W_MPOP_FIRST))
        note("_decode writes (vreg, W)", writes == {(v, sp[0]) for v, sp in dd["writes"]}, w)
        # reads the RTL makes (sequencer.sv candidate_vreg_reads + port A of txin)
        reads = set()
        if f["v.alu_valid"] or f["v.sfu_valid"] or f["v.reduce_valid"] or f["v.txin_valid"]:
            reads.add(f["v.raddr_a"])
        if f["v.alu_valid"] and f["v.alu_op"] != 3:
            reads.add(f["v.raddr_b"])
        if f["x.valid"] and f["x.op"]:
            reads.add(f["x.vreg_idx"])
        if f["m.subop"] == 1:
            reads |= {(f["m.reg_idx"] + k) & 31 for k in range(4)}
        elif f["m.subop"] == 2:
            reads.add(f["m.reg_idx"])
        note("_decode reads == RTL reads", reads == dd["reads"], w)
        note("_decode reads >= RTL reads (conservative)", reads <= dd["reads"], w)
        note("_decode delay", asm._delay_of(w) == f["delay"], w)
    return res


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--asm-check", action="store_true")
    a = ap.parse_args(argv)
    if a.asm_check:
        rng = rng_for("seq_decoder", "base")
        sets = {"directed": directed(), "random": [random_word(rng) for _ in range(20000)],
                "images": image_words(), "asm-builder": builder_words(rng)}
        for name, words in sets.items():
            print(f"== asm.py vs reference decode: {name} ({len(words)} bundles)")
            for k, (ok, tot, ex) in sorted(asm_check(words).items()):
                print(f"   {'AGREE   ' if ok == tot else 'DISAGREE'} {k:45s} {ok}/{tot}" + (f"  e.g. {ex}" if ok != tot else ""))
    return 0


# ---------------------------------------------------------------------------
# Allo (U4 track A, plan C1). The decoder is one plain function per slot of
# ``bundle_fields_t`` -- the per-slot records the plan asks for instead of one
# 233-bit port (U3 P-8) -- each taking the 128-bit bundle and returning that
# slot's packed record (``sequencer_pkg``'s layout, MSB first, as
# ``R.<SLOT>_SLOT``). The issue unit (track B) calls the ones it needs. The
# region feeds the bundle as four 32-bit lanes and returns the eight records
# (two lanes where a record is wider than 32 bits); the runner concatenates
# them only to compare with the RTL's flattened struct.
# Bundle: V[127:108] = op rd ra rb; M[107:100]; MEM kind[99:98] payload[97:66];
# S[65:53]; C[52:46] = op operand; IMM[45:22]; DELAY[21:15].
# ---------------------------------------------------------------------------

WIDTH = {"base": R.FIELDS_W}
U32 = UInt(32)


def decode_v(w: UInt(128)) -> UInt(46):
    op: UInt(5) = w[123:128]
    rd: UInt(5) = w[118:123]
    rb: UInt(5) = w[108:113]
    f: UInt(46) = 0
    f[41:46] = w[113:118]  # raddr_a
    f[36:41] = rb  # raddr_b
    if op >= 1 and op <= 4:  # vadd vsub vmul vmov -> ALU op 0..3
        f[35] = 1
        f[21:24] = op - 1
        f[16:21] = rd
    elif op == 11 or op == 12:  # vmax vmin -> ALU op 4, 5
        f[35] = 1
        f[21:24] = op - 7
        f[16:21] = rd
    elif op >= 5 and op <= 8:  # vgelu vexp vrecip vrsqrt -> SFU op 0..3
        f[15] = 1
        f[13:15] = op - 5
        f[8:13] = rd
    elif op == 9 or op == 10:  # vredsum vredmax
        f[7] = 1
        f[6] = op - 9
        f[0:5] = rd
    elif op == 15:  # vlanered: rb[0] picks sum or max
        f[7] = 1
        f[6] = rb[0]
        f[5] = 1
        f[0:5] = rd
    elif op == 13:  # vtxin
        f[34] = 1
        f[32:34] = rb[0:2]
    elif op == 14:  # vtxout
        f[31] = 1
        f[29:31] = rb[0:2]
        f[24:29] = rd
    return f


def decode_m(w: UInt(128)) -> UInt(8):
    m: UInt(8) = w[100:108]
    return m


def decode_x(w: UInt(128)) -> UInt(27):
    f: UInt(27) = 0
    if w[98:100] == 1:  # MEM_LDST
        p: UInt(32) = w[66:98]
        f[26] = 1
        f[25] = p[31]  # dir
        f[20:25] = p[26:31]  # vreg_idx
        f[8:20] = p[14:26]  # literal (word, VMEM_ADDR_W)
        f[7] = p[13]  # agu_valid
        f[6] = p[0]  # agu_level_hi
        f[4:6] = p[11:13]  # agu_level_lo
        f[3] = p[7]  # agu_shift_hi
        f[0:3] = p[8:11]  # agu_shift_lo
    return f


def decode_d(w: UInt(128)) -> UInt(57):
    f: UInt(57) = 0
    if w[98:100] == 2:  # MEM_DESC
        p: UInt(32) = w[66:98]
        f[56] = 1
        f[55] = p[31]  # is_store
        f[53:55] = p[29:31]  # channel_sel
        f[41:53] = p[17:29]  # vmem_address
        f[29:41] = p[5:17]  # rows_m1
        f[28] = p[4]  # has_disp
        f[2:4] = p[2:4]  # base_sreg
        f[0:2] = p[0:2]  # stride_sreg
        if p[4]:  # mem_owns_imm
            f[4:28] = w[22:46]
    return f


def decode_c(w: UInt(128)) -> UInt(7):
    cop: UInt(3) = w[50:53]
    f: UInt(7) = 0
    if cop == 2:  # C_OP_LEND
        f[4:7] = 1
    elif cop == 3:  # C_OP_FLUSH -> wait.channel
        f[4:7] = 2
        f[0:4] = w[46:50]
    elif cop == 4:  # C_OP_HALT
        f[4:7] = 3
    return f


def decode_l(w: UInt(128)) -> UInt(45):
    cop: UInt(3) = w[50:53]
    f: UInt(45) = 0
    if cop == 1 or cop == 5:  # loop.begin / loop.begin.r: valid, and c_owns_imm
        f[44] = 1
        if cop == 5:
            f[43] = 1
            f[42] = w[48]  # loop_bound.from_arg
            f[40:42] = w[46:48]  # loop_bound.idx
            f[0:16] = w[22:38]  # skip
        else:
            f[16:32] = w[22:38]  # hi
        f[32:36] = w[38:42]  # step
        f[36:40] = w[42:46]  # lo
    return f


def decode_s(w: UInt(128)) -> UInt(36):
    f: UInt(36) = 0
    f[35] = w[65]  # valid
    f[32:35] = w[62:65]  # op
    f[30:32] = w[60:62]  # rd
    f[28:30] = w[58:60]  # rs
    f[27] = w[57]  # use_iv
    f[26] = w[53]  # level_hi
    f[24:26] = w[55:57]  # level_lo
    if w[65] == 1 and w[62:65] != 1:  # s_owns_imm: valid and not SMOV_ARG
        f[0:24] = w[22:46]
    return f


def c1(n, w_unused):
    @df.region()
    def top(B: U32[n, 4], V: U32[n, 2], D: U32[n, 2], L: U32[n, 2], S: U32[n, 2],
            M: U32[n], X: U32[n], C: U32[n], DL: U32[n]):
        @df.kernel(mapping=[1], args=[B, V, D, L, S, M, X, C, DL])
        def decoder(b: U32[n, 4], v: U32[n, 2], d: U32[n, 2], lp: U32[n, 2], s: U32[n, 2],
                    m: U32[n], x: U32[n], c: U32[n], dl: U32[n]):
            for t in range(n):
                w: UInt(128) = 0
                w[0:32] = b[t, 0]
                w[32:64] = b[t, 1]
                w[64:96] = b[t, 2]
                w[96:128] = b[t, 3]
                fv: UInt(46) = decode_v(w)
                fd: UInt(57) = decode_d(w)
                fl: UInt(45) = decode_l(w)
                fs: UInt(36) = decode_s(w)
                v[t, 0] = fv[0:32]
                v[t, 1] = fv[32:46]
                d[t, 0] = fd[0:32]
                d[t, 1] = fd[32:57]
                lp[t, 0] = fl[0:32]
                lp[t, 1] = fl[32:45]
                s[t, 0] = fs[0:32]
                s[t, 1] = fs[32:36]
                m[t] = decode_m(w)
                x[t] = decode_x(w)
                c[t] = decode_c(w)
                dl[t] = w[15:22]

    return top


#: (name, slot width, lanes) in ``bundle_fields_t`` order, MSB first
SLOT_OUTS = [("v", 46, 2), ("m", 8, 1), ("x", 27, 1), ("d", 57, 2), ("c", 7, 1), ("l", 45, 2),
             ("s", 36, 2), ("delay", 7, 1)]


def run_c1(mod, cmd, n, w):
    from examples.minitpu.units.ctrl_lanes import join, split

    outs = {k: np.zeros((n, nl) if nl > 1 else n, dtype=np.uint32) for k, _, nl in SLOT_OUTS}
    mod(split(cmd["bundle_i"][:n], 4), outs["v"], outs["d"], outs["l"], outs["s"], outs["m"], outs["x"],
        outs["c"], outs["delay"])
    vals = {k: (join(outs[k]) if nl > 1 else [int(x) for x in outs[k]]) for k, _, nl in SLOT_OUTS}
    res = []
    for t in range(n):
        f = 0
        for k, wd, _ in SLOT_OUTS:
            f = (f << wd) | vals[k][t]
        res.append(f)
    return {"fields_o": res}


VARIANTS = {"c1": (c1, run_c1)}

if __name__ == "__main__":
    sys.exit(main())
