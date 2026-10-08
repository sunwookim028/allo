# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: ``sequencer_vpu_adapter``, the producer of ``vpu_ctrl_t`` (combinational).

Decoded V, M and X slots plus ``issue_i`` (the sequencer's ``run_accept``) and
the resolved VMEM word (``sequencer_agu_resolve``) to the VPU's 85-bit control
struct. Every *valid* field is ANDed with ``issue_i``; every payload field
(register addresses, destinations, ops, the VMEM address and
``vmem_store_read_hint``) passes **ungated**, so the struct carries the live
fetch head's operands on stall cycles too (``ref_ctrl_decode.VPU_CTRL_SOURCE``).
``alu_op`` widens from 3 to 4 bits by a cast that relies on both enums
agreeing.

Stimulus: random slot vectors with ``issue_i`` both ways, and the decoded
slots of every bundle of the two board images. Seed:
``tb_bundle_vpu_adapter`` (whole sequencer; scope ``tb.dut.u_vpu_adapter``).
"""

import os
import sys

import numpy as np

import allo.dataflow as df
from allo.ir.types import UInt, uint1

from examples.minitpu.harness import rtl
from examples.minitpu.harness import ref_ctrl_decode as R
from examples.minitpu.harness.traces import rng_for
from examples.minitpu.units.seq_decoder import image_words

HERE = os.path.dirname(os.path.abspath(__file__))
PKGS = ["src/pkg/minitpu_config_pkg.sv", "src/core/vpu/vpu_pkg.sv", "src/core/sequencer/sequencer_pkg.sv"]
SOURCES = PKGS + ["src/core/sequencer/sequencer_vpu_adapter.sv", os.path.join(HERE, "rtl", "u4_vpu_adapter.sv")]
W = {n: sum(w for _, w in lay) for n, lay in (("v", R.V_SLOT), ("m", R.M_SLOT), ("x", R.X_SLOT))}
INPUTS = [("v_i", W["v"]), ("m_i", W["m"]), ("x_i", W["x"]), ("issue_i", 1), ("x_resolved_row_i", 12)]
RTL = rtl.RtlUnit(top="u4_vpu_adapter", sources=SOURCES, inputs=INPUTS,
                  outputs=[("vpu_ctrl_o", R.VPU_CTRL_W, "pre")], shape="trace", clk="clk_i", rst_n="rst_n_unused")
INSTANCES = {"base": RTL}
DEFAULT = "base"
LATENCY_SOURCE = "combinational (sequencer_vpu_adapter.sv always_comb); vpu_ctrl_t is valid in the issue cycle"


def REF(inst, cmd):
    return R.vpu_adapter_trace(cmd)


def _slots(f, slot, lay):
    return R.pack(lay, {n: f[f"{slot}.{n}"] for n, _ in lay})


def from_images(rng):
    rows = {p: [] for p, _ in INPUTS}
    for w in image_words():
        f = R.decode(w)
        for issue in (1, 0):
            for p, v in zip(rows, (_slots(f, "v", R.V_SLOT), _slots(f, "m", R.M_SLOT), _slots(f, "x", R.X_SLOT),
                                   issue, rng.getrandbits(12))):
                rows[p].append(v)
    return rows


def random_rows(rng, n):
    return {p: [rng.getrandbits(w) for _ in range(n)] for p, w in INPUTS}


def traces(inst):
    rng = rng_for("vpu_adapter", inst)
    return [("images", from_images(rng), True), ("random", random_rows(rng, 20000), True)]


def seeds():
    srcs = PKGS + [f"src/core/sequencer/{f}" for f in (
        "sequencer_decoder.sv", "sequencer_iram.sv", "sequencer_fetch_queue.sv", "sequencer_loop_buffer.sv",
        "sequencer_loop_ctrl.sv", "sequencer_agu_resolve.sv", "sequencer_scalar_agu.sv", "dma_desc_adapter.sv",
        "sequencer_vpu_adapter.sv", "sequencer.sv")]
    rows = R.seed_rows("tb_bundle_vpu_adapter", srcs, "dut.u_vpu_adapter",
                       [p for p, _ in INPUTS] + ["vpu_ctrl_o"], clk_scope="dut", clk="clk")
    return [("tb_bundle_vpu_adapter", "base", {p: rows[p] for p, _ in INPUTS}, {"vpu_ctrl_o": rows["vpu_ctrl_o"]},
             True)]


def probes(inst):
    cmd = {"v_i": [0] * 16, "m_i": [0] * 16, "x_i": [0] * 16, "issue_i": [1] * 8 + [0] * 8,
           "x_resolved_row_i": [0] * 16}
    cmd["v_i"] = [R.pack(R.V_SLOT, {"alu_valid": 1})] * 16
    return [("issue_i -> vpu_ctrl_o.alu_valid (comb)", 0,
             rtl.probe_trace(RTL, {p: rtl.pack(cmd[p], w) for p, w in INPUTS}, "vpu_ctrl_o", 8))]


def table():
    """vpu_ctrl_t, field by field: width, source, gated by issue."""
    lines = []
    for n, w in R.VPU_CTRL:
        src, gated = R.VPU_CTRL_SOURCE[n]
        lines.append(f"{n:22s} {w:3d}  {'issue &' if gated else 'ungated'}  {src}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Allo (U4 track A, plan C1; README D-23).
#
# ``vpu_ctrl_t`` is not one port here. The adapter is three plain functions,
# one per slot command D-23 declares -- V (vector: ALU/SFU/reduce/transpose,
# ``vpu_ctrl_t[84:38]``), X (memory: the VMEM request and the store hint,
# ``[37:18]``), M (matrix: vmatload/push/pop, ``[17:0]``) -- each a record
# of at most 47 bits, valid-qualified by ``issue``. The runner concatenates
# them only to hold them to the RTL's struct. Slice bounds are literals
# (``ctrl_lanes.lsb(R.VPU_CTRL)`` minus the command's base gives them; D-17:
# a named bound is refused).
#
# ``slots``        D-23 as the RTL behaves: valids gated, payload ungated.
# ``slots_gated``  D-23's allowed deviation: payload zero unless its op's
#                  valid is set. Differs from the RTL exactly on payload the
#                  VPU never samples (Phase 0: ``vpu_ctrl_gating.log``); its
#                  verdict is that census, not a match.
# ---------------------------------------------------------------------------

WIDTH = {"base": 85}
U32 = UInt(32)


def adapt_v(v: UInt(46), issue: uint1) -> UInt(47):
    """The V command: the decoded V slot, valids ANDed with ``issue``,
    ``alu_op`` widened 3 -> 4 bits (``vpu_alu_op_e'(...)``)."""
    c: UInt(47) = 0
    c[42:47] = v[41:46]  # raddr_a
    c[37:42] = v[36:41]  # raddr_b
    c[36] = v[35] & issue  # alu_valid
    c[35] = v[34] & issue  # txin_valid
    c[33:35] = v[32:34]  # txin_index
    c[32] = v[31] & issue  # txout_valid
    c[30:32] = v[29:31]  # txout_index
    c[25:30] = v[24:29]  # txout_vd
    c[21:24] = v[21:24]  # alu_op (bit 24 stays 0: the 4-bit cast)
    c[16:21] = v[16:21]  # alu_vd
    c[15] = v[15] & issue  # sfu_valid
    c[13:15] = v[13:15]  # sfu_op
    c[8:13] = v[8:13]  # sfu_vd
    c[7] = v[7] & issue  # reduce_valid
    c[6] = v[6]  # reduce_op
    c[5] = v[5]  # reduce_lane
    c[0:5] = v[0:5]  # reduce_vd
    return c


def adapt_x(x: UInt(27), issue: uint1, row: UInt(12)) -> UInt(20):
    """The X (memory) command: ``vmem`` request + ``vmem_store_read_hint``."""
    c: UInt(20) = 0
    c[19] = x[26] & issue  # vmem.valid
    c[18] = x[25]  # vmem.op
    c[13:18] = x[20:25]  # vmem.vreg_idx
    c[1:13] = row  # vmem.vmem_address (resolved)
    c[0] = x[26] & x[25]  # vmem_store_read_hint = x.valid && x.op (ungated)
    return c


def adapt_m(m: UInt(8), issue: uint1) -> UInt(18):
    """The M (matrix) command: one valid per subop, ``reg_idx`` on all three."""
    sub: UInt(3) = m[5:8]
    reg: UInt(5) = m[0:5]
    c: UInt(18) = 0
    c[17] = issue & (sub == 1)  # vmatload_valid (M_VMATLOAD)
    c[12:17] = reg
    c[11] = issue & (sub == 2)  # vmatpush_valid
    c[6:11] = reg
    c[5] = issue & (sub == 3)  # vmatpop_valid
    c[0:5] = reg
    return c


def gate_v(c: UInt(47)) -> UInt(47):
    """D-23's deviation: each payload field zero unless its op is valid.
    ``raddr_a`` is read by ALU, SFU, reduce and txin; ``raddr_b`` by the ALU."""
    g: UInt(47) = 0
    alu: uint1 = c[36]
    txin: uint1 = c[35]
    txout: uint1 = c[32]
    sfu: uint1 = c[15]
    red: uint1 = c[7]
    if alu | sfu | red | txin:
        g[42:47] = c[42:47]
    if alu:
        g[37:42] = c[37:42]
        g[21:25] = c[21:25]
        g[16:21] = c[16:21]
    g[36] = alu
    g[35] = txin
    if txin:
        g[33:35] = c[33:35]
    g[32] = txout
    if txout:
        g[30:32] = c[30:32]
        g[25:30] = c[25:30]
    g[15] = sfu
    if sfu:
        g[13:15] = c[13:15]
        g[8:13] = c[8:13]
    g[7] = red
    if red:
        g[0:7] = c[0:7]
    return g


def gate_x(c: UInt(20)) -> UInt(20):
    g: UInt(20) = 0
    if c[19]:
        g = c
    return g


def gate_m(c: UInt(18)) -> UInt(18):
    g: UInt(18) = 0
    if c[17]:
        g[12:18] = c[12:18]
    if c[11]:
        g[6:12] = c[6:12]
    if c[5]:
        g[0:6] = c[0:6]
    return g


def slots(n, w):
    @df.region()
    def top(VI: U32[n, 2], MI: UInt(8)[n], XI: U32[n], ISS: uint1[n], ROW: UInt(12)[n],
            VC: U32[n, 2], XC: U32[n], MC: U32[n]):
        @df.kernel(mapping=[1], args=[VI, MI, XI, ISS, ROW, VC, XC, MC])
        def adapter(vi: U32[n, 2], mi: UInt(8)[n], xi: U32[n], iss: uint1[n], row: UInt(12)[n],
                    vc: U32[n, 2], xc: U32[n], mc: U32[n]):
            for t in range(n):
                v: UInt(46) = 0
                v[0:32] = vi[t, 0]
                v[32:46] = vi[t, 1]
                x: UInt(27) = xi[t]
                cv: UInt(47) = adapt_v(v, iss[t])
                cx: UInt(20) = adapt_x(x, iss[t], row[t])
                cm: UInt(18) = adapt_m(mi[t], iss[t])
                vc[t, 0] = cv[0:32]  # A5: the 2-D output stored first
                vc[t, 1] = cv[32:47]
                xc[t] = cx
                mc[t] = cm

    return top


def slots_gated(n, w):
    """``slots`` with the three gates. A separate function, not a flag: a
    Python bool closed over by a kernel is compiled as an ``scf.if`` on an
    ``i32`` (U3 A9)."""

    @df.region()
    def top(VI: U32[n, 2], MI: UInt(8)[n], XI: U32[n], ISS: uint1[n], ROW: UInt(12)[n],
            VC: U32[n, 2], XC: U32[n], MC: U32[n]):
        @df.kernel(mapping=[1], args=[VI, MI, XI, ISS, ROW, VC, XC, MC])
        def adapter(vi: U32[n, 2], mi: UInt(8)[n], xi: U32[n], iss: uint1[n], row: UInt(12)[n],
                    vc: U32[n, 2], xc: U32[n], mc: U32[n]):
            for t in range(n):
                v: UInt(46) = 0
                v[0:32] = vi[t, 0]
                v[32:46] = vi[t, 1]
                x: UInt(27) = xi[t]
                cv: UInt(47) = gate_v(adapt_v(v, iss[t]))
                cx: UInt(20) = gate_x(adapt_x(x, iss[t], row[t]))
                cm: UInt(18) = gate_m(adapt_m(mi[t], iss[t]))
                vc[t, 0] = cv[0:32]  # A5: the 2-D output stored first
                vc[t, 1] = cv[32:47]
                xc[t] = cx
                mc[t] = cm

    return top


def run_slots(mod, cmd, n, w):
    from examples.minitpu.units.ctrl_lanes import col, join, split

    vc = np.zeros((n, 2), dtype=np.uint32)
    xc, mc = np.zeros(n, dtype=np.uint32), np.zeros(n, dtype=np.uint32)
    mod(split(cmd["v_i"][:n], 2), col(cmd, "m_i", n, np.uint8), col(cmd, "x_i", n),
        col(cmd, "issue_i", n, np.uint8), col(cmd, "x_resolved_row_i", n, np.uint16), vc, xc, mc)
    v = join(vc)
    return {"vpu_ctrl_o": [(v[t] << 38) | (int(xc[t]) << 18) | int(mc[t]) for t in range(n)]}


VARIANTS = {"slots": (slots, run_slots), "slots_gated": (slots_gated, run_slots)}

#: payload field -> the valid bits under which the VPU samples it (Phase 0:
#: ``vpu_ctrl_gating.log``; every other cycle the field is ignored)
SAMPLED_UNDER = {
    "raddr_a": ("alu_valid", "sfu_valid", "reduce_valid", "txin_valid"), "raddr_b": ("alu_valid",),
    "txin_index": ("txin_valid",), "txout_index": ("txout_valid",), "txout_vd": ("txout_valid",),
    "alu_op": ("alu_valid",), "alu_vd": ("alu_valid",), "sfu_op": ("sfu_valid",), "sfu_vd": ("sfu_valid",),
    "reduce_op": ("reduce_valid",), "reduce_lane": ("reduce_valid",), "reduce_vd": ("reduce_valid",),
    "vmem.op": ("vmem.valid",), "vmem.vreg_idx": ("vmem.valid",), "vmem.vmem_address": ("vmem.valid",),
    "vmem_store_read_hint": ("vmem.valid",), "vmatload_base": ("vmatload_valid",),
    "vmatpush_vs": ("vmatpush_valid",), "vmatpop_vd": ("vmatpop_valid",)}


def gated_contract(backend="simulator", project="/tmp/u4_vpu_adapter_gated"):
    """``slots_gated`` held to the RTL on what the VPU samples: every valid
    bit on every row, and each payload field in the rows its op is valid.
    Prints one ``CONTRACT-`` line."""
    import allo.dataflow as df
    from examples.minitpu.harness.traces import concat

    parts = [c for _, c, _ in traces("base")] + [c for _, _, c, _, _ in seeds()]
    cmd = concat(*parts)
    n = len(cmd["issue_i"])
    want, _, _ = REF("base", {p: rtl.pack(cmd[p], w) for p, w in INPUTS})
    want = rtl.unpack(want["vpu_ctrl_o"])
    top = slots_gated(n, 85)
    mod = (df.build(top, target="simulator") if backend == "simulator"
           else df.build(top, target="systemc", mode="csim", project=project))
    got = run_slots(mod, cmd, n, 85)["vpu_ctrl_o"]
    tot = bad = zeroed = 0
    for t in range(n):
        r, g = R.unpack(R.VPU_CTRL, want[t]), R.unpack(R.VPU_CTRL, got[t])
        for f, _ in R.VPU_CTRL:
            if f in SAMPLED_UNDER:
                if not any(r[v] for v in SAMPLED_UNDER[f]):
                    zeroed += int(g[f] != r[f])
                    continue
            tot += 1
            bad += int(g[f] != r[f])
    tag = "CONTRACT-MATCH" if bad == 0 else "CONTRACT-DIFF "
    print(f"{tag} vpu_adapter slots_gated {backend}: {tot - bad}/{tot} sampled fields equal "
          f"({zeroed} unsampled payload fields zeroed where the RTL drives the fetch head's)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--gated-contract":
        gated_contract(*sys.argv[2:])
    else:
        print(table())
    sys.exit(0)
