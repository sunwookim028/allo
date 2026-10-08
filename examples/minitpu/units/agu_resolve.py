# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: ``sequencer_agu_resolve``, the X slot's VMEM word address (combinational).

``word = literal + (iv[level] << shift)`` truncated to ``VMEM_ADDR_W = 12``
bits, with ``iv`` any of the eight loop levels' 32-bit induction variables
(``iv_by_level`` flattened by ``units/rtl/u4_agu_resolve.sv``). The RTL
narrows each iv to its low 12 bits before the 8:1 mux; the reference
(``ref_ctrl_decode.agu_resolve``) keeps all 32 bits as
``tb_agu_resolve_width``'s does, so a match is also the narrowing identity.

Stimulus: every (level, shift, agu_valid) with ivs whose high bits are set
(the bits the narrowing drops), random literals, plus random rows. Seed:
``tb_agu_resolve_width`` (no clock: every time step a row).
"""

import os
import sys

from examples.minitpu.harness import rtl
from examples.minitpu.harness import ref_ctrl_decode as R
from examples.minitpu.harness.traces import rng_for

HERE = os.path.dirname(os.path.abspath(__file__))
PKGS = ["src/pkg/minitpu_config_pkg.sv", "src/core/vpu/vpu_pkg.sv", "src/core/sequencer/sequencer_pkg.sv"]
SOURCES = PKGS + ["src/core/sequencer/sequencer_agu_resolve.sv", os.path.join(HERE, "rtl", "u4_agu_resolve.sv")]
INPUTS = [("iv_flat_i", 32 * R.STACK_DEPTH), ("x_literal_i", 12), ("x_agu_valid_i", 1),
          ("x_agu_level_i", 3), ("x_agu_shift_i", 4)]
RTL = rtl.RtlUnit(top="u4_agu_resolve", sources=SOURCES, inputs=INPUTS,
                  outputs=[("x_resolved_addr_o", 12, "pre")], shape="trace", clk="clk_i", rst_n="rst_n_unused")
INSTANCES = {"base": RTL}
DEFAULT = "base"
LATENCY_SOURCE = "combinational (sequencer_agu_resolve.sv; sequencer.sv feeds the live decode, vpu_adapter forwards it)"

IVS = [0, 1, 0xFFF, 0x1000, 0x1001, 0xFFFFF000, 0xFFFFFFFF, 0x80000000, 0x555, 0xDEADBEEF, 0x800, 0x12345678]


def REF(inst, cmd):
    return R.agu_trace(cmd)


def _flat(ivs):
    return sum((v & 0xFFFFFFFF) << (32 * i) for i, v in enumerate(ivs))


def exhaustive(rng):
    rows = {p: [] for p, _ in INPUTS}
    for which in range(len(IVS)):
        ivs = [IVS[(which + i) % len(IVS)] for i in range(8)]
        for lvl in range(8):
            for s in range(16):
                for valid in (0, 1):
                    for lit in (0, rng.getrandbits(12), 0xFFF):
                        for p, v in zip(rows, (_flat(ivs), lit, valid, lvl, s)):
                            rows[p].append(v)
    return rows


def random_rows(rng, n):
    rows = {p: [] for p, _ in INPUTS}
    for _ in range(n):
        for p, w in INPUTS:
            rows[p].append(rng.getrandbits(w))
    return rows


def traces(inst):
    rng = rng_for("agu_resolve", inst)
    return [("exhaustive-level-shift", exhaustive(rng), True), ("random", random_rows(rng, 20000), True)]


def seeds():
    ports = ["x_literal_i", "x_agu_valid_i", "x_agu_level_i", "x_agu_shift_i"]
    names = [f"iv_by_level[{i}]" for i in range(8)] + ports + ["x_resolved_addr_o"]
    rows = R.seed_rows("tb_agu_resolve_width", PKGS + ["src/core/sequencer/sequencer_agu_resolve.sv"], "dut",
                       names)
    n = len(rows["x_literal_i"])
    cmd = {p: rows[p] for p in ports}
    cmd["iv_flat_i"] = [_flat([rows[f"iv_by_level[{i}]"][t] for i in range(8)]) for t in range(n)]
    return [("tb_agu_resolve_width", "base", cmd, {"x_resolved_addr_o": rows["x_resolved_addr_o"]}, True)]


def probes(inst):
    cmd = {"iv_flat_i": [_flat([5] * 8)] * 16, "x_literal_i": [3] * 16, "x_agu_valid_i": [1] * 16,
           "x_agu_level_i": [2] * 16, "x_agu_shift_i": [1] * 8 + [2] * 8}
    packed = {p: rtl.pack(cmd[p], w) for p, w in INPUTS}
    return [("shift -> x_resolved_addr_o (comb)", 0, rtl.probe_trace(RTL, packed, "x_resolved_addr_o", 8))]


VARIANTS = {}

if __name__ == "__main__":
    sys.exit(0)
