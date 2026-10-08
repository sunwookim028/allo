# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: the VREG write-port calendar, measured on ``vpu.sv`` behind ``vpu_ctrl_t``.

``vpu.sv`` has six writeback sources and one VREG write port: the sources are
OR-ed into the port with no arbiter (``vpu.sv:289-305``, ``:352-365``), two
registers follow (``wb_stage_*``, then the per-sublane ``wb_local_*``), and
the only guard is a simulation assertion (``$onehot0``, ``:370``). So the
schedule must keep the sources apart, and each op's *claim cycle* ``W`` is the
contract the assembler (``asm.py`` ``_Timeline.write_port``) and the
sequencer's sim-only calendar (``sequencer.sv:277-405``) book.

The unit is ``vpu.sv`` unchanged inside a harness wrapper
(``units/rtl/u4_vpu_wb.sv``) that takes ``vpu_ctrl_t`` as one flat 85-bit
vector and exposes the writeback by hierarchical reference. The reference
(``harness/ref_ctrl_wb.py``) is the calendar only: which source drives the
mux in which cycle, and the write enable/address two registers later, from the
command trace alone. Data is not modelled (U5); the VREG read port A is
observed only by the RAW probes, which measure ``W + 1`` by function.

Every trace also randomises every payload field of ``vpu_ctrl_t`` on every
cycle except the valid bits (and the fields of the op that is valid), so a
REF-MATCH says the writeback depends on a field only in the cycle its valid is
high -- the property that would let ``vpu_ctrl_t`` be declared as a
valid-qualified command interface.
"""

import numpy as np

from examples.minitpu.harness import ref_ctrl_wb as R
from examples.minitpu.harness import rtl
from examples.minitpu.harness.traces import rng_for

M = "/work/shared/users/phd/sk3463/minitpu/"  # noqa: F841 (documentation of the source root)
VPU_F = ["src/pkg/minitpu_config_pkg.sv", "src/core/vpu/vpu_pkg.sv",
         "src/core/vpu/vpu_word_array.sv", "src/core/vpu/vpu_dma_group.sv",
         "src/core/vpu/vpu_vmem_simd.sv", "src/core/vpu/vpu_regfile.sv",
         "src/core/vpu/vpu_vreg_stripe.sv", "src/core/vpu/vpu_bf16_mul.sv",
         "src/core/vpu/vpu_bf16_add.sv", "src/core/vpu/vpu_bf16_add_pipe.sv",
         "src/core/vpu/vpu_alu.sv", "src/core/vpu/vpu_fifo.sv",
         "src/core/xlu/xlu_reduction_tree.sv", "src/core/xlu/xlu.sv",
         "src/core/xlu/xlu_transpose.sv", "src/core/sfu/sfu.sv", "src/core/sfu/sfu_group.sv",
         "src/core/mxu/mxu_stream_engine.sv", "src/core/mxu/mxu_pop_engine.sv",
         "src/core/mxu/mxu_matrix_ctrl.sv", "src/core/mxu/mxu_bf16_mul_acc24.sv",
         "src/core/mxu/mxu_acc24_add_pipe.sv", "src/core/mxu/mxu_pe.sv",
         "src/core/mxu/mxu_systolic_array.sv", "src/core/mxu/mxu_serializer.sv",
         "src/core/mxu/mxu.sv", "src/core/vpu/vpu.sv"]
import os  # noqa: E402

WRAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtl", "u4_vpu_wb.sv")
LW = 16 * 16  # one lane stripe
RTL = rtl.RtlUnit(
    top="u4_vpu_wb",
    sources=VPU_F + [WRAP],
    inputs=[("rst_ni", 1), ("ctrl_i", R.CTRL_W), ("dma_en_i", 1), ("dma_we_i", 1),
            ("dma_addr_i", 14), ("dma_wdata_i", LW)],
    outputs=[("ctrl_w_o", 8), ("wb_src_o", 6), ("wb_stage_valid_o", 1), ("wb_local_valid_o", 4),
             ("wb_local_addr_o", 5), ("rdata_a_o", 4 * LW), ("matrix_busy_o", 1),
             ("unsupported_o", 1)],
    shape="trace",
    clk="clk_i",
    assertions=True,
)
INSTANCES = {"shipped": RTL}
DEFAULT = "shipped"
LATENCY_SOURCE = ("vpu_pkg.sv:57-67 VPU_*_LATENCY + VPU_WB_STAGES (2) = sequencer_pkg.sv:557-571 "
                  "WB_W_*; docs/isa_latency.json rtl_params; asm.py:707-713")
CLASSES = ("load", "alu", "sfu", "reduce", "lane_reduce", "txout")
CHECKED = ("ctrl_w_o", "wb_src_o", "wb_stage_valid_o", "wb_local_valid_o", "wb_local_addr_o")
RESULT_LATENCY = 85  # isa_latency.json matrix.result_latency.vmatpush


def _payload(rng):
    """Every non-valid field random: what the RTL must ignore."""
    f = {n: rng.getrandbits(w) for n, w in R.VPU_CTRL_FIELDS if n not in R.VALIDS}
    f["alu_op"] = rng.randrange(9)
    return f


GROUP = {"alu": ("alu_valid", "alu_vd", "alu_op", "raddr_a", "raddr_b"),
         "sfu": ("sfu_valid", "sfu_vd", "sfu_op", "raddr_a"),
         "reduce": ("reduce_valid", "reduce_lane", "reduce_vd", "reduce_op", "raddr_a"),
         "lane_reduce": ("reduce_valid", "reduce_lane", "reduce_vd", "reduce_op", "raddr_a"),
         "txout": ("txout_valid", "txout_vd", "txout_index"),
         "load": ("vmem_valid", "vmem_op", "vmem_vreg_idx", "vmem_address")}


def _op(rng, cls, vd, src=None):
    """The fields that start one ``cls`` op writing ``vd`` (random payload)."""
    f = _payload(rng)
    src = rng.randrange(32) if src is None else src
    if cls == "alu":
        f.update(alu_valid=1, alu_vd=vd, raddr_a=src, alu_op=rng.choice([0, 1, 2, 4, 5]))
    elif cls == "sfu":
        f.update(sfu_valid=1, sfu_vd=vd, raddr_a=src)
    elif cls in ("reduce", "lane_reduce"):
        f.update(reduce_valid=1, reduce_lane=int(cls == "lane_reduce"), reduce_vd=vd, raddr_a=src)
    elif cls == "txout":
        f.update(txout_valid=1, txout_vd=vd)
    elif cls == "load":
        f.update(vmem_valid=1, vmem_op=0, vmem_vreg_idx=vd)
    return f


class Prog:
    def __init__(self, rng, reset=3):
        self.rng = rng
        self.rows = []
        for _ in range(reset):
            self.rows.append({"rst_ni": 0, "ctrl": _payload(rng)})
        self.pops = {}

    def idle(self, k=1, raddr_a=None):
        for _ in range(k):
            f = _payload(self.rng)
            if raddr_a is not None:
                f["raddr_a"] = raddr_a
            self.rows.append({"rst_ni": 1, "ctrl": f})

    def preload(self):
        """Distinct VREG contents: DMA-write VMEM words 0..31 (four 32 B beats
        each, ``vpu_dma_group``), then ``vld v_k <- word k``. Needed because
        Verilator's ``--x-initial unique`` gives every entry of the unpacked
        VREG array the same random value (finding H1)."""
        for w in range(32):
            for b in range(4):
                self.idle()
                self.rows[-1].update(dma_en_i=1, dma_we_i=1, dma_addr_i=4 * w + b,
                                     dma_wdata_i=self.rng.getrandbits(LW))
        self.idle(4)
        for w in range(32):
            t = len(self.rows)
            self.put(t, _op(self.rng, "load", w), "load")
            self.at(t)["vmem_address"] = w
        self.idle(12)

    def at(self, t):
        while len(self.rows) <= t:
            self.idle()
        return self.rows[t]["ctrl"]

    def put(self, t, fields, cls=None):
        """Place an op; on a cycle that already holds one, only the new op's
        own fields are written (co-issue)."""
        self.at(t)
        cur = self.rows[t]["ctrl"]
        if any(cur.get(n) for n in R.VALIDS):
            assert cls is not None
            fields = {k: fields[k] for k in GROUP[cls] if k in fields}
        self.rows[t]["ctrl"] = {**cur, **fields}

    def cmd(self, tail=24):
        self.idle(tail)
        ctrl = []
        for r in self.rows:
            ctrl.append(R.pack_ctrl(**{k: v for k, v in r["ctrl"].items() if k in R._OFF}))
        col = lambda k: [r.get(k, 0) for r in self.rows]  # noqa: E731
        return {"rst_ni": col("rst_ni"), "ctrl_i": ctrl, "dma_en_i": col("dma_en_i"),
                "dma_we_i": col("dma_we_i"), "dma_addr_i": col("dma_addr_i"),
                "dma_wdata_i": col("dma_wdata_i")}


def matrix_prelude(p, t):
    """vmatload v0..v3 at t, vmatpush v4 at t+16; returns the push cycle."""
    f = _payload(p.rng)
    f.update(vmatload_valid=1, vmatload_base=0)
    p.put(t, f)
    f = _payload(p.rng)
    f.update(vmatpush_valid=1, vmatpush_vs=4)
    p.put(t + 16, f)
    return t + 16


def pop(p, t, vd, push):
    f = _payload(p.rng)
    f.update(vmatpop_valid=1, vmatpop_vd=vd)
    p.put(t, f)


def traces(inst):
    out = []
    rng = rng_for("u4-vpu-wb", "singles")
    p = Prog(rng)
    t = 6
    for _ in range(4):
        for cls in CLASSES:
            p.put(t, _op(rng, cls, rng.randrange(32)), cls)
            t += 20
    out.append(("singles", p, True))
    # every ordered pair of classes at every offset 0..16: a collision exactly when W_a = d + W_b
    rng = rng_for("u4-vpu-wb", "pairs")
    p = Prog(rng)
    t = 6
    legal = True
    for a in CLASSES:
        for b in CLASSES:
            for d in range(0, 17):
                if d == 0 and (a == b or (a != "load" and b != "load")):
                    continue  # one V op per cycle in vpu_ctrl_t's producer
                p.put(t, _op(rng, a, rng.randrange(32)), a)
                p.put(t + d, _op(rng, b, rng.randrange(32)), b)
                if R.W[a] == d + R.W[b]:
                    legal = False
                t += 24
    out.append(("pairs", p, legal))
    # random streams: a legal one (claims kept apart) and an unfiltered one
    for k, filt in ((0, True), (1, True), (2, False)):
        rng = rng_for("u4-vpu-wb", "random", k)
        p = Prog(rng)
        booked = set()
        for t in range(6, 3000):
            if rng.random() < 0.45:
                cls = rng.choice(CLASSES)
                if filt and t + R.W[cls] in booked:
                    continue
                booked.add(t + R.W[cls])
                p.put(t, _op(rng, cls, rng.randrange(32)), cls)
                if rng.random() < 0.3 and cls != "load":
                    c2 = "load"
                    if not filt or t + R.W[c2] not in booked:
                        booked.add(t + R.W[c2])
                        p.put(t, _op(rng, c2, rng.randrange(32)), c2)
        out.append((f"random-{'legal' if filt else 'any'}-{k}", p, filt))
    # matrix pops: late (legal, W = 3) and early (waits for its result)
    rng = rng_for("u4-vpu-wb", "matrix")
    p = Prog(rng)
    t = 6
    for early in (0, 0, 10, 40, 84):
        push = matrix_prelude(p, t)
        pop(p, push + RESULT_LATENCY - early if early else push + RESULT_LATENCY + 3,
            rng.randrange(8, 32), push)
        t = push + RESULT_LATENCY + 30
    out.append(("matrix-pops", p, True))
    return [(label, p.cmd(), legal) for label, p, legal in out]


def REF(inst, cmd):
    rst = [int(v) for v in np.asarray(cmd["rst_ni"]).reshape(-1)]
    ctrl = rtl.unpack(cmd["ctrl_i"]) if isinstance(cmd["ctrl_i"], np.ndarray) else cmd["ctrl_i"]
    want, claims, events = R.writeback_trace({"rst_ni": rst, "ctrl_i": ctrl})
    n = len(rst)
    out, reason = {}, {}
    for p, w, *_ in RTL.outputs:
        if p in CHECKED:
            out[p] = rtl.pack([int(x) for x in want[p]], w)
            r = np.array([""] * n, dtype=object)
            r[:3] = "reset"
        else:
            out[p] = np.zeros((n, rtl.nwords(w)), dtype=np.uint64)
            r = np.array(["not modelled"] * n, dtype=object)
        reason[p] = r
    return out, reason, events



def _probe_prog(cls, vd, rng):
    """Idle with raddr_a = vd (port A observes vd), one ``cls`` op writing vd at
    ``event``; the op reads vd itself where it reads port A, so the read port
    never leaves vd."""
    p = Prog(rng)
    p.preload()
    p.idle(20, raddr_a=vd)
    event = len(p.rows)
    f = _op(rng, cls, vd, src=vd)
    f["raddr_a"] = vd
    p.put(event, f)
    p.idle(30, raddr_a=vd)
    return p, event


def probes(inst):
    """``[(label, declared, measured)]``: per class, the mux cycle L, the write
    enable W, and RAW by function (the first cycle port A shows the new value,
    = W + 1, the earliest legal consumer)."""
    res = []
    for cls in CLASSES:
        for k, vd in enumerate((5, 17)):
            rng = rng_for("u4-vpu-wb-probe", cls, k)
            p, ev = _probe_prog(cls, vd, rng)
            cmd = p.cmd(tail=0)
            if k == 0:
                res.append((f"{cls}: issue -> wb_source_valid (L)", R.L[cls],
                            rtl.probe_trace(RTL, cmd, "wb_src_o", ev)))
                res.append((f"{cls}: issue -> VREG write enable (W)", R.W[cls],
                            rtl.probe_trace(RTL, cmd, "wb_local_valid_o", ev)))
            res.append((f"{cls}: issue -> port A reads the new value (W+1), vd={vd}", R.W[cls] + 1,
                        rtl.probe_trace(RTL, cmd, "rdata_a_o", ev)))
    # vmatpop after its result: L = 1, W = 3, W+1 = 4
    for k, vd in enumerate((9, 30)):
        rng = rng_for("u4-vpu-wb-probe", "mpop", k)
        p = Prog(rng)
        p.preload()
        p.idle(6, raddr_a=vd)
        push = matrix_prelude(p, len(p.rows))
        for t in range(len(p.rows), push + RESULT_LATENCY + 6):
            p.at(t)["raddr_a"] = vd
        ev = push + RESULT_LATENCY + 5
        pop(p, ev, vd, push)
        p.at(ev)["raddr_a"] = vd
        p.idle(30, raddr_a=vd)
        cmd = p.cmd(tail=0)
        if k == 0:
            res.append(("mpop (result waiting): issue -> wb_source_valid (L)", R.L["mpop"],
                        rtl.probe_trace(RTL, cmd, "wb_src_o", ev)))
            res.append(("mpop (result waiting): issue -> VREG write enable (W)", R.W["mpop"],
                        rtl.probe_trace(RTL, cmd, "wb_local_valid_o", ev)))
        res.append((f"mpop: issue -> port A reads the new value (W+1), vd={vd}", R.W["mpop"] + 1,
                    rtl.probe_trace(RTL, cmd, "rdata_a_o", ev)))
    return res
