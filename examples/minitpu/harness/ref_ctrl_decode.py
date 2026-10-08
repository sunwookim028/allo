# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 references: the sequencer's decode and address units (MiniTPU ``b3ba0a4d``).

Written from ``sequencer_pkg.sv``'s layouts and the ``.sv`` bodies, not from
``asm.py`` (``asm.py`` is the *cross-check*, ``units/seq_decoder.py``):

* ``decode(word)``: ``sequencer_decoder`` -- a 128-bit bundle to the
  ``bundle_fields_t`` view, as a dict of named fields; ``pack(FIELDS, d)``
  flattens it the way the packed struct is laid out (MSB first).
* ``agu_resolve``: ``literal + (iv[level][11:0] << shift)`` mod ``2**12``.
* ``ScalarAgu``: ``sequencer_scalar_agu``'s cycle model (S_LAT-deep pipe, no
  bypass, SMAC's multiply before and add after the pipe register).
* ``DescAdapter``: ``dma_desc_adapter``'s cycle model (snapshot at start,
  words to beats).
* ``vpu_adapter``: ``sequencer_vpu_adapter`` -- the producer of ``vpu_ctrl_t``.

Every function of a trace returns ``(want, reason, events)`` in the
``characterize`` convention. Every unit here resets all of its state (or has
none), so nothing is masked except the cycles of an explicit reset.

``seed_rows`` extracts a MiniTPU tb's stimulus at any scope (also for
combinational DUTs with no clock), so a tb whose DUT is a whole sequencer
seeds the sub-unit it contains.
"""

import numpy as np

from examples.minitpu.harness import rtl

# ---- geometry (vpu_pkg / sequencer_pkg / minitpu_config_pkg at 16x4) -------
VREG_ADDR_W = 5
VMEM_ADDR_W = 12
LEVEL_SEL_W = 3
STACK_DEPTH = 8
X_SHIFT_W = 4
IMM_W = 24
DELAY_W = 7
L_HI_W = 16
SUBLANE_SEL_W = 2
DMA_CHANNEL_SEL_W = 1  # DMA_CHANNEL_COUNT = 2
S_LAT = 2

# ---- packed-struct layouts, MSB first ----------------------------------------
V_SLOT = [("raddr_a", 5), ("raddr_b", 5), ("alu_valid", 1), ("txin_valid", 1), ("txin_index", 2),
          ("txout_valid", 1), ("txout_index", 2), ("txout_vd", 5), ("alu_op", 3), ("alu_vd", 5),
          ("sfu_valid", 1), ("sfu_op", 2), ("sfu_vd", 5), ("reduce_valid", 1), ("reduce_op", 1),
          ("reduce_lane", 1), ("reduce_vd", 5)]
M_SLOT = [("subop", 3), ("reg_idx", 5)]
X_SLOT = [("valid", 1), ("op", 1), ("vreg_idx", 5), ("literal", 12), ("agu_valid", 1),
          ("agu_level", 3), ("agu_shift", 4)]
D_SLOT = [("valid", 1), ("is_store", 1), ("channel_sel", 2), ("vmem_address", 12), ("rows", 12),
          ("has_disp", 1), ("disp", 24), ("base_sreg", 2), ("stride_sreg", 2)]
C_SLOT = [("subop", 3), ("operand", 4)]
L_SLOT = [("valid", 1), ("hi_from_reg", 1), ("hi_from_arg", 1), ("hi_idx", 2), ("lo", 4),
          ("step", 4), ("hi", 16), ("skip", 16)]
S_SLOT = [("valid", 1), ("op", 3), ("rd", 2), ("rs", 2), ("use_iv", 1), ("level", 3), ("imm", 24)]
SLOTS = [("v", V_SLOT), ("m", M_SLOT), ("x", X_SLOT), ("d", D_SLOT), ("c", C_SLOT),
         ("l", L_SLOT), ("s", S_SLOT)]
FIELDS = [(f"{s}.{n}", w) for s, lay in SLOTS for n, w in lay] + [("delay", DELAY_W)]
FIELDS_W = sum(w for _, w in FIELDS)  # 233

VPU_REQ = [("valid", 1), ("op", 1), ("vreg_idx", 5), ("vmem_address", 12)]
VPU_CTRL = [("raddr_a", 5), ("raddr_b", 5), ("alu_valid", 1), ("txin_valid", 1), ("txin_index", 2),
            ("txout_valid", 1), ("txout_index", 2), ("txout_vd", 5), ("alu_op", 4), ("alu_vd", 5),
            ("sfu_valid", 1), ("sfu_op", 2), ("sfu_vd", 5), ("reduce_valid", 1), ("reduce_op", 1),
            ("reduce_lane", 1), ("reduce_vd", 5)] + [(f"vmem.{n}", w) for n, w in VPU_REQ] + [
            ("vmem_store_read_hint", 1), ("vmatload_valid", 1), ("vmatload_base", 5),
            ("vmatpush_valid", 1), ("vmatpush_vs", 5), ("vmatpop_valid", 1), ("vmatpop_vd", 5)]
VPU_CTRL_W = sum(w for _, w in VPU_CTRL)  # 85


def pack(layout, d):
    """Fields (dict, missing = 0) -> int, ``layout`` MSB first."""
    v = 0
    for n, w in layout:
        x = int(d.get(n, 0))
        assert 0 <= x < (1 << w), (n, x, w)
        v = (v << w) | x
    return v


def unpack(layout, v):
    out = {}
    for n, w in reversed(layout):
        out[n] = v & ((1 << w) - 1)
        v >>= w
    return out


def bits(v, hi, lo):
    return (v >> lo) & ((1 << (hi - lo + 1)) - 1)


# ---- encoded bundle (encoded_bundle_t) -----------------------------------------
# V[127:108] = op rd ra rb; M[107:100]; MEM[99:66] = kind[99:98] payload[97:66];
# S[65:53] = valid op[3] rd[2] rs[2] use_iv level_lo[2] spare[1] level_hi[1];
# C[52:46] = op[3] operand[4]; IMM[45:22]; DELAY[21:15]; reserved[14:0].
V_OPS = {1: ("alu", 0), 2: ("alu", 1), 3: ("alu", 2), 4: ("alu", 3), 11: ("alu", 4), 12: ("alu", 5),
         5: ("sfu", 0), 6: ("sfu", 1), 7: ("sfu", 2), 8: ("sfu", 3), 9: ("red", 0), 10: ("red", 1),
         13: ("txin", None), 14: ("txout", None), 15: ("lanered", None)}
MEM_LDST, MEM_DESC = 1, 2
C_OP_LBEGIN, C_OP_LEND, C_OP_FLUSH, C_OP_HALT, C_OP_LBEGIN_R = 1, 2, 3, 4, 5
C_LOOP_END, C_WAIT_CHANNEL, C_HALT = 1, 2, 3
SMOV_ARG = 1


def owners(word):
    """Which slots claim the shared IMM (sequencer_pkg s/c/mem_owns_imm)."""
    s_valid, s_op = bits(word, 65, 65), bits(word, 64, 62)
    c_op = bits(word, 52, 50)
    kind, payload = bits(word, 99, 98), bits(word, 97, 66)
    return {"s": bool(s_valid and s_op != SMOV_ARG),
            "c": c_op in (C_OP_LBEGIN, C_OP_LBEGIN_R),
            "mem": kind == MEM_DESC and bool(bits(payload, 4, 4))}


def decode(word):
    """``sequencer_decoder``: bundle -> ``{"v.raddr_a": ..., ..., "delay": ...}``."""
    f = {n: 0 for n, _ in FIELDS}
    op, rd, ra, rb = bits(word, 127, 123), bits(word, 122, 118), bits(word, 117, 113), bits(word, 112, 108)
    f["v.raddr_a"], f["v.raddr_b"] = ra, rb
    kind_op = V_OPS.get(op)
    if kind_op:
        k, sub = kind_op
        if k == "alu":
            f["v.alu_valid"], f["v.alu_op"], f["v.alu_vd"] = 1, sub, rd
        elif k == "sfu":
            f["v.sfu_valid"], f["v.sfu_op"], f["v.sfu_vd"] = 1, sub, rd
        elif k == "red":
            f["v.reduce_valid"], f["v.reduce_op"], f["v.reduce_vd"] = 1, sub, rd
        elif k == "lanered":
            f["v.reduce_valid"], f["v.reduce_lane"], f["v.reduce_op"], f["v.reduce_vd"] = 1, 1, rb & 1, rd
        elif k == "txin":
            f["v.txin_valid"], f["v.txin_index"] = 1, rb & 3
        elif k == "txout":
            f["v.txout_valid"], f["v.txout_index"], f["v.txout_vd"] = 1, rb & 3, rd
    f["m.subop"], f["m.reg_idx"] = bits(word, 107, 105), bits(word, 104, 100)
    kind, p = bits(word, 99, 98), bits(word, 97, 66)
    if kind == MEM_LDST:
        f["x.valid"], f["x.op"], f["x.vreg_idx"] = 1, bits(p, 31, 31), bits(p, 30, 26)
        f["x.literal"], f["x.agu_valid"] = bits(p, 25, 14), bits(p, 13, 13)
        f["x.agu_level"] = (bits(p, 0, 0) << 2) | bits(p, 12, 11)
        f["x.agu_shift"] = (bits(p, 7, 7) << 3) | bits(p, 10, 8)
    elif kind == MEM_DESC:
        f["d.valid"], f["d.is_store"], f["d.channel_sel"] = 1, bits(p, 31, 31), bits(p, 30, 29)
        f["d.vmem_address"], f["d.rows"], f["d.has_disp"] = bits(p, 28, 17), bits(p, 16, 5), bits(p, 4, 4)
        f["d.base_sreg"], f["d.stride_sreg"] = bits(p, 3, 2), bits(p, 1, 0)
    f["s.valid"], f["s.op"], f["s.rd"], f["s.rs"] = bits(word, 65, 65), bits(word, 64, 62), bits(word, 61, 60), bits(word, 59, 58)
    f["s.use_iv"], f["s.level"] = bits(word, 57, 57), (bits(word, 53, 53) << 2) | bits(word, 56, 55)
    cop, operand = bits(word, 52, 50), bits(word, 49, 46)
    if cop == C_OP_LBEGIN:
        f["l.valid"] = 1
    elif cop == C_OP_LBEGIN_R:
        f["l.valid"], f["l.hi_from_reg"], f["l.hi_from_arg"], f["l.hi_idx"] = 1, 1, (operand >> 2) & 1, operand & 3
    elif cop == C_OP_LEND:
        f["c.subop"] = C_LOOP_END
    elif cop == C_OP_FLUSH:
        f["c.subop"], f["c.operand"] = C_WAIT_CHANNEL, operand
    elif cop == C_OP_HALT:
        f["c.subop"] = C_HALT
    imm = bits(word, 45, 22)
    own = owners(word)
    if own["s"]:
        f["s.imm"] = imm
    if own["mem"]:
        f["d.disp"] = imm
    if own["c"]:
        f["l.skip" if cop == C_OP_LBEGIN_R else "l.hi"] = imm & 0xFFFF
        f["l.step"], f["l.lo"] = (imm >> 16) & 0xF, (imm >> 20) & 0xF
    f["delay"] = bits(word, 21, 15)
    return f


def decoder_trace(cmd):
    words = rtl.unpack(cmd["bundle_i"])
    want = rtl.pack([pack(FIELDS, decode(w)) for w in words], FIELDS_W)
    n = len(words)
    ev = {}
    for w in words:
        k = sum(owners(w).values())
        if k > 1:
            ev["imm-claim-conflict"] = ev.get("imm-claim-conflict", 0) + 1
        if V_OPS.get(bits(w, 127, 123)) is None and bits(w, 127, 123):
            ev["v-op-undefined"] = ev.get("v-op-undefined", 0) + 1
        if bits(w, 99, 98) == 3:
            ev["mem-kind-3"] = ev.get("mem-kind-3", 0) + 1
        if bits(w, 52, 50) in (6, 7):
            ev["c-op-undefined"] = ev.get("c-op-undefined", 0) + 1
    return {"fields_o": want}, {"fields_o": np.array([""] * n, dtype=object)}, ev


# ---- address resolve -----------------------------------------------------------
def agu_resolve(iv, literal, agu_valid, level, shift):
    """sequencer_agu_resolve, written as tb_agu_resolve_width's full 32-bit
    reference: keep every bit, shift in 64, add in 32, truncate at the end."""
    wide = ((iv[level] & 0xFFFFFFFF) << shift) if agu_valid else 0
    return (literal + (wide & 0xFFFFFFFF)) & ((1 << VMEM_ADDR_W) - 1)


def ivs_of(flat):
    return [(flat >> (32 * i)) & 0xFFFFFFFF for i in range(STACK_DEPTH)]


def agu_trace(cmd):
    cols = {p: rtl.unpack(cmd[p]) for p in cmd}
    n = len(cols["x_literal_i"])
    out = [agu_resolve(ivs_of(cols["iv_flat_i"][t]), cols["x_literal_i"][t], cols["x_agu_valid_i"][t],
                       cols["x_agu_level_i"][t], cols["x_agu_shift_i"][t]) for t in range(n)]
    hi = sum(1 for t in range(n) if cols["x_agu_valid_i"][t]
             and ivs_of(cols["iv_flat_i"][t])[cols["x_agu_level_i"][t]] >> VMEM_ADDR_W)
    return ({"x_resolved_addr_o": rtl.pack(out, VMEM_ADDR_W)},
            {"x_resolved_addr_o": np.array([""] * n, dtype=object)}, {"iv-high-bits-dropped": hi})


# ---- scalar AGU ------------------------------------------------------------------
SMOVI, SADDI, SMAC, SSHL = 0, 2, 3, 4
M32 = 0xFFFFFFFF


def _sext24(v):
    return (v - (1 << 24)) & M32 if v & (1 << 23) else v


class ScalarAgu:
    """sequencer_scalar_agu, one call of ``step`` per cycle (``rtl.py`` trace
    convention: ``pre`` outputs from the state before the edge, then the edge)."""

    def __init__(self, s_lat=S_LAT):
        self.s_lat = s_lat
        self.reset()

    def reset(self):
        self.sreg = [0] * 4
        self.written = 0
        nst = max(self.s_lat - 1, 0)
        self.stages = [(0, 0, 0, 0, 0)] * nst  # valid, rd, data, mac_product, is_mac

    def _results(self, r, iv, kargs):
        op, rs, imm = r["s_op_i"], r["s_rs_i"], r["s_imm_i"]
        operand = iv[r["s_level_i"]] if r["s_use_iv_i"] else self.sreg[rs]
        product = (operand * imm) & M32
        arg = (kargs >> (32 * rs)) & M32
        full = {SMOVI: imm, SMOV_ARG: arg, SADDI: (self.sreg[rs] + _sext24(imm)) & M32,
                SMAC: (self.sreg[rs] + product) & M32, SSHL: (self.sreg[rs] << (imm & 31)) & M32}.get(op, 0)
        pre = {SMAC: self.sreg[rs]}.get(op, full)
        return full, pre, product

    def outputs(self, r):
        kargs = r["kernel_arg_csr_i"]
        lb = ((kargs >> (32 * r["rd_sel_loop_bound_i"])) & M32 if r["rd_loop_bound_from_arg_i"]
              else self.sreg[r["rd_sel_loop_bound_i"]])
        return {"rd_data_d_base_o": self.sreg[r["rd_sel_d_base_i"]],
                "rd_data_d_stride_o": self.sreg[r["rd_sel_d_stride_i"]],
                "rd_data_loop_bound_o": 0xFFFF if lb >> 16 else lb,
                "sreg_written_o": self.written,
                "sreg_o": sum(v << (32 * i) for i, v in enumerate(self.sreg))}

    def edge(self, r):
        iv = ivs_of(r["iv_flat_i"])
        full, pre, product = self._results(r, iv, r["kernel_arg_csr_i"])
        if self.s_lat <= 1:
            self.written = 0
            if r["s_valid_i"]:
                self.sreg[r["s_rd_i"]] = full
                self.written = 1 << r["s_rd_i"]
            return
        v, rd, data, prod, is_mac = self.stages[-1]
        self.written = 0
        if v:
            self.sreg[rd] = (data + prod) & M32 if is_mac else data
            self.written = 1 << rd
        self.stages = [(r["s_valid_i"], r["s_rd_i"], pre, product, int(r["s_op_i"] == SMAC))] + self.stages[:-1]


def _rows(cmd):
    cols = {p: rtl.unpack(c) for p, c in cmd.items()}
    n = len(next(iter(cols.values())))
    return n, [{p: cols[p][t] for p in cols} for t in range(n)]


def _run(model, cmd, outs, widths, rst="rst_n"):
    """Drive a cycle model through a trace with an active-low async reset port."""
    n, rows = _rows(cmd)
    got = {p: [] for p in outs}
    reason = {p: np.array([""] * n, dtype=object) for p in outs}
    for t, r in enumerate(rows):
        if not r[rst]:
            model.reset()
        o = model.outputs(r)
        for p in outs:
            got[p].append(o[p])
        if r[rst]:
            model.edge(r)
        else:
            model.reset()
    return {p: rtl.pack(got[p], widths[p]) for p in outs}, reason


def scalar_agu_trace(cmd, s_lat=S_LAT):
    outs = ["rd_data_d_base_o", "rd_data_d_stride_o", "rd_data_loop_bound_o", "sreg_written_o", "sreg_o"]
    widths = dict(zip(outs, [32, 32, 16, 4, 128]))
    want, reason = _run(ScalarAgu(s_lat), cmd, outs, widths)
    n, rows = _rows(cmd)
    ev = {}
    for r in rows:
        if r["rst_n"] and r["s_valid_i"] and r["s_op_i"] > SSHL:
            ev["s-op-undefined"] = ev.get("s-op-undefined", 0) + 1
    return want, reason, ev


# ---- descriptor adapter ----------------------------------------------------------
class DescAdapter:
    """dma_desc_adapter: IDLE -start-> ISSUE (snapshot) -accept-> IDLE."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.issue, self.d, self.base, self.stride = 0, unpack(D_SLOT, 0), 0, 0

    def outputs(self, r):
        d = self.d
        return {"desc_valid_o": self.issue, "desc_is_store_o": d["is_store"],
                "desc_channel_o": d["channel_sel"] & ((1 << DMA_CHANNEL_SEL_W) - 1),
                "desc_vmem_row_o": d["vmem_address"] << SUBLANE_SEL_W,
                "desc_rows_o": (d["rows"] << SUBLANE_SEL_W) | 3, "desc_cols_o": 0,
                "desc_base_o": self.base, "desc_stride_o": self.stride,
                "done_o": int(self.issue and r["desc_accept_i"])}

    def edge(self, r):
        if not self.issue:
            if r["start_i"]:
                d = unpack(D_SLOT, r["d_i"])
                self.d = d
                self.base = ((r["sreg_rd_base_i"] + _sext24(d["disp"])) & M32 if d["has_disp"]
                             else r["sreg_rd_base_i"])
                self.stride = r["sreg_rd_stride_i"]
                self.issue = 1
        elif r["desc_accept_i"]:
            self.issue = 0


DESC_OUTS = [("desc_valid_o", 1), ("desc_is_store_o", 1), ("desc_channel_o", DMA_CHANNEL_SEL_W),
             ("desc_vmem_row_o", VMEM_ADDR_W + SUBLANE_SEL_W), ("desc_rows_o", 12 + SUBLANE_SEL_W),
             ("desc_cols_o", 7), ("desc_base_o", 32), ("desc_stride_o", 32), ("done_o", 1)]


def desc_adapter_trace(cmd):
    widths = dict(DESC_OUTS)
    want, reason = _run(DescAdapter(), cmd, list(widths), widths)
    n, rows = _rows(cmd)
    ev, issue = {}, 0
    for r in rows:  # replay the state to count protocol events
        if not r["rst_n"]:
            issue = 0
            continue
        if issue and r["start_i"]:
            ev["start-while-issuing (ignored)"] = ev.get("start-while-issuing (ignored)", 0) + 1
        if not issue and r["start_i"] and unpack(D_SLOT, r["d_i"])["channel_sel"] >> DMA_CHANNEL_SEL_W:
            ev["channel-bit-dropped"] = ev.get("channel-bit-dropped", 0) + 1
        issue = (0 if r["desc_accept_i"] else 1) if issue else int(bool(r["start_i"]))
    return want, reason, ev


# ---- VPU adapter: the producer of vpu_ctrl_t ----------------------------------------
M_VMATLOAD, M_VMATPUSH, M_VMATPOP = 1, 2, 3
# field -> (source, gated by issue_i)
VPU_CTRL_SOURCE = {
    "raddr_a": ("v.raddr_a", False), "raddr_b": ("v.raddr_b", False),
    "alu_valid": ("v.alu_valid", True), "txin_valid": ("v.txin_valid", True),
    "txin_index": ("v.txin_index", False), "txout_valid": ("v.txout_valid", True),
    "txout_index": ("v.txout_index", False), "txout_vd": ("v.txout_vd", False),
    "alu_op": ("v.alu_op (3b -> 4b cast)", False), "alu_vd": ("v.alu_vd", False),
    "sfu_valid": ("v.sfu_valid", True), "sfu_op": ("v.sfu_op", False), "sfu_vd": ("v.sfu_vd", False),
    "reduce_valid": ("v.reduce_valid", True), "reduce_op": ("v.reduce_op", False),
    "reduce_lane": ("v.reduce_lane", False), "reduce_vd": ("v.reduce_vd", False),
    "vmem.valid": ("x.valid", True), "vmem.op": ("x.op", False), "vmem.vreg_idx": ("x.vreg_idx", False),
    "vmem.vmem_address": ("x_resolved_row_i (agu_resolve)", False),
    "vmem_store_read_hint": ("x.valid & x.op", False),
    "vmatload_valid": ("m.subop == VMATLOAD", True), "vmatload_base": ("m.reg_idx", False),
    "vmatpush_valid": ("m.subop == VMATPUSH", True), "vmatpush_vs": ("m.reg_idx", False),
    "vmatpop_valid": ("m.subop == VMATPOP", True), "vmatpop_vd": ("m.reg_idx", False),
}


def vpu_adapter(v, m, x, issue, row):
    v, m, x = unpack(V_SLOT, v), unpack(M_SLOT, m), unpack(X_SLOT, x)
    c = {k: v[k] for k in ("raddr_a", "raddr_b", "txin_index", "txout_index", "txout_vd", "alu_op",
                           "alu_vd", "sfu_op", "sfu_vd", "reduce_op", "reduce_lane", "reduce_vd")}
    for k in ("alu_valid", "txin_valid", "txout_valid", "sfu_valid", "reduce_valid"):
        c[k] = issue & v[k]
    c["vmem.valid"], c["vmem.op"], c["vmem.vreg_idx"] = issue & x["valid"], x["op"], x["vreg_idx"]
    c["vmem.vmem_address"] = row
    c["vmem_store_read_hint"] = x["valid"] & x["op"]
    for k, sub in (("vmatload", M_VMATLOAD), ("vmatpush", M_VMATPUSH), ("vmatpop", M_VMATPOP)):
        c[f"{k}_valid"] = issue & int(m["subop"] == sub)
    c["vmatload_base"] = c["vmatpush_vs"] = c["vmatpop_vd"] = m["reg_idx"]
    return pack(VPU_CTRL, c)


def vpu_adapter_trace(cmd):
    cols = {p: rtl.unpack(c) for p, c in cmd.items()}
    n = len(cols["issue_i"])
    out = [vpu_adapter(cols["v_i"][t], cols["m_i"][t], cols["x_i"][t], cols["issue_i"][t],
                       cols["x_resolved_row_i"][t]) for t in range(n)]
    ev = {"ungated-payload-while-not-issued": sum(1 for t in range(n) if not cols["issue_i"][t]
                                                  and (cols["v_i"][t] or cols["x_i"][t] or cols["m_i"][t]))}
    return ({"vpu_ctrl_o": rtl.pack(out, VPU_CTRL_W)}, {"vpu_ctrl_o": np.array([""] * n, dtype=object)}, ev)


# ---- seeds from MiniTPU's tbs, at any scope ----------------------------------------
def seed_rows(tb, sources, scope, names, clk_scope=None, clk=None):
    """Run ``tb`` (unchanged, via ``vcd_seed``) and return ``{name: [values]}``
    for signals ``names`` under ``tb.<scope>``: sampled just before each rising
    edge of ``tb.<clk_scope>.<clk>`` plus once at the end, or, with no clock
    (a combinational tb driven by ``#`` delays), at the end of every time step
    in which anything listed changed."""
    from examples.minitpu.harness import vcd_seed

    vcd = vcd_seed._build_and_run(tb, sources)
    ch = vcd_seed.read_vcd(vcd, f"tb.{scope}", names)
    if clk:
        c = vcd_seed.read_vcd(vcd, f"tb.{clk_scope or scope}", [clk])[clk]
        times = [t for (t, v), (_, p) in zip(c[1:], c[:-1]) if v == 1 and p == 0] + [float("inf")]
    else:
        times = sorted({t for n in names for t, _ in ch[n]})
        times = [t + 0.5 for t in times]

    def before(sig):
        out, cur, i, c = [], 0, 0, ch[sig]
        for t in times:
            while i < len(c) and c[i][0] < t:
                cur = c[i][1]
                i += 1
            out.append(cur)
        return out

    return {n: before(n) for n in names}
