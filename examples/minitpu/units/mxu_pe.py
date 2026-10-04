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

from examples.minitpu.harness import ref_mxu, rtl
from examples.minitpu.harness.traces import Trace, rng_for

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


VARIANTS = {}
