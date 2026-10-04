# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One VPU lane from real units, with the SFU as an optional module (README D-19).

The base lane is ``issue -> alu -> writeback``; the ALU is U1's ``bits``
composition (``units/alu.py`` ``composed``: ``add_bits``, ``mul_bits`` and
``bf16_gt`` on every pair, then the result mux). ``SFU`` is a
``compose.Option``: track A's S1 ``bits`` SFU (``units/sfu.py``: two ROMs, two
PWL tables, ``sfu_bits``), its channel ``sfu_out``, the slots ``vgelu``,
``vexp``, ``vrecip``, ``vrsqrt``, and the rebind ``writeback: alu_out ->
sfu_out``.

That rebind mirrors ``vpu.sv``: ``sfu_group`` (``:88``) is fed from VREG port
A beside the ALU, and the writeback takes the SFU's result for an SFU op
through the op tag pipe (``:157-184``). Here the ALU forwards port A's
operand and the op tag on ``alu_out``; the SFU computes on that operand for
an SFU op and passes the ALU's result through otherwise. Functionally the
same per element; the timing (the SFU's declared latency 5, ``vpu_pkg.sv:64``,
against the ALU's 3) is not modelled by these untimed streams.

Op word (5 bits): bit 4 selects the SFU, then ``op[0:2]`` is ``sfu.sv``'s op
(gelu 0, exp 1, recip 2, rsqrt 3); otherwise ``op[0:4]`` is
``vpu_alu_op_e``. Oracles: ``ref.vpu_alu`` and ``ref.sfu``, the Phase 0 /
U1 references held bit-exact against the RTL.
"""

from __future__ import annotations

import numpy as np

from allo.compose import Architecture, Channel, Memory, Option, unit
from examples.minitpu.harness import ref
from examples.minitpu.units.alu import bf16_gt
from examples.minitpu.units.bf16_add import add_bits
from examples.minitpu.units.bf16_mul_pipe import mul_bits
from examples.minitpu.units.sfu import (EXP_ROM, GELU_ROM, RECIP_PWL, RSQRT_PWL,
                                        exp_addr, gelu_addr, sfu_bits)

ALU_SLOTS = {"vadd": 0, "vsub": 1, "vmul": 2, "vmov": 3, "vmax": 4, "vmin": 5}
SFU_SLOTS = {"vgelu": 16, "vexp": 17, "vrecip": 18, "vrsqrt": 19}


@unit(memories=("OPS", "A", "B"), writes=("to_alu",), parameters=("N_OPS",))
def issue(ops: UInt(32)[N_OPS], a_mem: UInt(32)[N_OPS], b_mem: UInt(32)[N_OPS]):
    for i in range(N_OPS):
        word: UInt(64) = 0
        word[0:16] = a_mem[i]
        word[16:32] = b_mem[i]
        word[32:37] = ops[i]
        to_alu.put(word)


@unit(reads=("to_alu",), writes=("alu_out",),
      parameters=("N_OPS", "OP_ADD", "OP_SUB", "OP_MUL", "OP_MAX", "OP_MIN"),
      calls=("add_bits", "mul_bits", "bf16_gt"))
def alu():
    for i in range(N_OPS):
        word: UInt(64) = to_alu.get()
        a: UInt(16) = word[0:16]
        b: UInt(16) = word[16:32]
        op: UInt(4) = word[32:36]
        # units/alu.py `composed`, line for line (B2: the compare via uint1)
        is_sub: uint1 = op == OP_SUB
        add_rhs: UInt(16) = b
        add_rhs[15] = b[15] ^ is_sub
        sum_r: UInt(16) = add_bits(a, add_rhs)
        mul_r: UInt(16) = mul_bits(a, b)
        gt: uint1 = bf16_gt(a, b)
        sel: UInt(16) = 0
        if op == OP_MAX:
            sel = a if gt else b
        else:
            sel = b if gt else a
        result: UInt(16) = a
        if op == OP_ADD or op == OP_SUB:
            result = sum_r
        elif op == OP_MUL:
            result = mul_r
        elif op == OP_MAX or op == OP_MIN:
            result = sel
        out: UInt(64) = 0
        out[0:16] = result
        out[16:32] = a          # VREG port A, forwarded for the SFU (vpu.sv:88)
        out[32:37] = word[32:37]  # the op tag (vpu.sv:157-184)
        alu_out.put(out)


@unit(reads=("alu_out",), writes=("sfu_out",),
      parameters=("N_OPS", "GELU_ROM", "EXP_ROM", "RECIP_PWL", "RSQRT_PWL"),
      calls=("gelu_addr", "exp_addr", "sfu_bits"))
def sfu():
    # units/sfu.py `bits`: the tables bound to local constants (A1; a name
    # starting with `gelu` is erased as the library's gelu, A3), the lookups
    # in the kernel (A2, A4)
    rom_g: int32[2048] = GELU_ROM
    rom_e: int32[2048] = EXP_ROM
    pwl_r: int32[32] = RECIP_PWL
    pwl_s: int32[32] = RSQRT_PWL
    for i in range(N_OPS):
        word: UInt(64) = alu_out.get()
        out: UInt(64) = word
        if word[36]:
            o: int32 = word[32:34]
            x16: UInt(16) = word[16:32]
            x: int32 = x16
            ga: int32 = gelu_addr(x)
            ea: int32 = exp_addr(x)
            ra: int32 = (x >> 2) & 0x1F
            sa: int32 = (((x >> 7) & 1) ^ 1) << 4 | ((x >> 3) & 0xF)
            gw: int32 = rom_g[ga]
            ew: int32 = rom_e[ea]
            rw: int32 = pwl_r[ra]
            sw: int32 = pwl_s[sa]
            r: int32 = sfu_bits(o, x, gw, ew, rw, sw)
            out[0:16] = r
        sfu_out.put(out)


@unit(memories=("OUT",), reads=("alu_out",), parameters=("N_OPS",))
def writeback(out_mem: UInt(32)[N_OPS]):
    for i in range(N_OPS):
        word: UInt(64) = alu_out.get()
        out_mem[i] = word[0:16]


def base(n_ops: int) -> Architecture:
    """The lane without the SFU: the ALU's slots only."""
    return Architecture(
        name="vpu_lane",
        parameters={"N_OPS": n_ops, "QD": 4} | {
            f"OP_{k}": ref.ALU_OP[k] for k in ("ADD", "SUB", "MUL", "MAX", "MIN")},
        memories=(Memory("OPS", "UInt(32)[N_OPS]"), Memory("A", "UInt(32)[N_OPS]"),
                  Memory("B", "UInt(32)[N_OPS]"), Memory("OUT", "UInt(32)[N_OPS]")),
        channels=(Channel("to_alu", "UInt(64)", "QD", carries="a, b, op"),
                  Channel("alu_out", "UInt(64)", "QD",
                          carries="result, port A operand, op tag")),
        units=(issue, alu, writeback),
        slots=tuple(ALU_SLOTS))


SFU = Option(
    name="sfu",
    units=(sfu,),
    channels=(Channel("sfu_out", "UInt(64)", "QD", carries="result, operand, op tag"),),
    rebind={"writeback": {"alu_out": "sfu_out"}},
    isa=tuple(SFU_SLOTS),
    parameters={"GELU_ROM": GELU_ROM, "EXP_ROM": EXP_ROM,
                "RECIP_PWL": RECIP_PWL, "RSQRT_PWL": RSQRT_PWL})

#: Malformed: the SFU added without rebinding the writeback -- what an
#: ``if with_sfu:`` over the unit list gives. ``alu_out`` gets two readers.
SFU_NO_REBIND = Option(name="sfu_norebind", units=SFU.units, channels=SFU.channels,
                       isa=SFU.isa, parameters=SFU.parameters)
#: Malformed: the rebind and the channel without the unit -- the module
#: "removed" but its wiring left. Nothing writes ``sfu_out``.
SFU_NO_UNIT = Option(name="sfu_nounit", channels=SFU.channels, rebind=SFU.rebind,
                     isa=SFU.isa)


def program(slots, n, seed=0):
    """``n`` random instructions over ``slots`` and random bf16 operands
    (every bit pattern, specials included)."""
    rng = np.random.default_rng(seed)
    codes = {**ALU_SLOTS, **SFU_SLOTS}
    names = list(slots)
    ops = np.array([codes[names[k]] for k in rng.integers(0, len(names), n)], dtype=np.uint32)
    a = rng.integers(0, 1 << 16, n).astype(np.uint32)
    b = rng.integers(0, 1 << 16, n).astype(np.uint32)
    return ops, a, b


def reference(ops, a, b):
    ops = np.asarray(ops, dtype=np.int64)
    alu_r = ref.vpu_alu(ops & 15, a, b).astype(np.int64)
    sfu_r = ref.sfu(ops & 3, a).astype(np.int64)
    return np.where(ops & 16, sfu_r, alu_r).astype(np.uint32)


def run(mod, ops, a, b):
    out = np.zeros(len(ops), dtype=np.uint32)
    mod(ops, a, b, out)
    return out
