# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track C, plan D2: ``dma.sv`` closed on MiniTPU's VMEM DMA port.

The DUT is ``dma.sv`` + ``vpu_vmem_simd`` (``vpu_dma_group`` and
``vpu_word_array``, compute port idle) wired as ``minitpu_core.sv`` wires
them (``units/rtl/u4_dma_vmem.sv``, no logic): ``vmem_gnt`` tied 1, a load
drain's write beat outranks a store's read, the 16-bit beat pointer cut to
``VMEM_BEAT_ADDR_W`` = 14 bits, ``dma_flush_i`` tied 0. Its inputs are the
DMA's minus ``vmem_rd_data``/``vmem_gnt`` (now internal); its outputs are the
DMA's, so a store's ``dm_req_wdata`` now carries what earlier loads wrote.

The VMEM side's contract (Phase 0 F3, measured here per cycle): a 32 B beat
per access; a read fetches the 128 B word on beat 0 and serves beats 1-3 from
a register, so a word's beats are read in order from 0, and the data reaches
``vmem_rd_data`` ``VMEM_DMA_READ_LATENCY`` = 2 cycles after ``vmem_rd_en``
(P-3); a write gathers beats and commits the whole word on beat 3 (a partial
word is never written; the gather register is not reset). In README D-12
terms the DMA side is one ``rw`` port of read latency 2 on the word array
(``vmem.d``), beside the compute port (``vmem.c``, ``rw`` latency 3), with the
same-word collision an obligation (MiniTPU issue #21).

Reference: ``ref_ctrl_dma.DmaModel`` (Phase 0's, unchanged) driven in closed
loop with ``VmemDmaSide`` below (``vpu_dma_group`` + the word array's DMA port
+ ``vpu_vmem_simd``'s rvalid pipe, transcribed). Undefined: what the RTL
never wrote -- VMEM words, the gather and scatter registers and the read
pipe are unreset, so a store beat whose source was never written makes
``dm_req_wdata`` undefined until the next capture (reason ``vmem uninit``);
the landing payload while empty and row 0 as in ``units/dma.py``.

Traces: closed-loop programs from Phase 0's generator (``dma.Program`` and
its in-order ``Bridge``, ``vmem_gnt`` 1): load/store round trips (a store
reading back what loads wrote, strides 1 and > 1), a partial word (the gather
never committed), two channels with
backpressure, reset mid-transfer (VMEM survives it), random programs over a
small hot set of VMEM rows so that stores read loaded words.
"""

import hashlib  # noqa: F401
import os

import numpy as np

from examples.minitpu.harness import rtl
from examples.minitpu.harness.ref_ctrl_dma import INPUTS as DMA_INPUTS, OUTPUTS, DmaModel, I_ST_RDW
from examples.minitpu.harness.traces import rng_for
from examples.minitpu.units import dma as D
from examples.minitpu.units.dma_params import GEOMETRIES

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["src/pkg/minitpu_config_pkg.sv", "src/core/vpu/vpu_pkg.sv", "src/core/dma/dma_addr_gen.sv",
           "src/core/dma/dma.sv", "src/core/vpu/vpu_dma_group.sv", "src/core/vpu/vpu_word_array.sv",
           "src/core/vpu/vpu_vmem_simd.sv", os.path.join(HERE, "rtl", "u4_dma_vmem.sv")]
INPUTS = [(p, w) for p, w in DMA_INPUTS if p not in ("vmem_rd_data", "vmem_gnt")]
PARAMS = {k: D.PARAMS[k] for k in ("core", "o1", "o4")}


def _unit(p):
    return rtl.RtlUnit(
        top="u4_dma_vmem", sources=SOURCES, inputs=list(INPUTS),
        outputs=[(q, w, "pre") for q, w in OUTPUTS], shape="trace", clk="clk", rst_n="rst_n",
        params={"OUTSTANDING": p["outstanding"], "TIMEOUT_CYCLES": p["timeout"],
                "LANDING_DEPTH": p["landing"]},
        assertions=True)


INSTANCES = {k: _unit(p) for k, p in PARAMS.items()}
DEFAULT = "core"
RTL = INSTANCES[DEFAULT]
WIDTH = {k: 256 for k in PARAMS}
LATENCY_SOURCE = "dma.sv as units/dma.py; VMEM DMA read 2 (vpu_pkg.sv VMEM_DMA_READ_LATENCY), pinned (P-3)"


class VmemDmaSide:
    """``vpu_dma_group`` + ``vpu_word_array``'s DMA port + ``vpu_vmem_simd``'s
    ``dma_rvalid_q`` (compute idle), per cycle. A beat or word value of
    ``None`` is undefined (never written since power-up)."""

    def __init__(self, latency=2, sub=4):
        self.L, self.SUB = latency, sub
        self.mem = {}
        self.pipe = [None] * latency      # word_array dma_read_pipe (unreset)
        self.rvalid = [0] * latency       # vpu_vmem_simd dma_rvalid_q (sync reset)
        self.gather = [None] * sub        # gather_q (unreset)
        self.gword, self.filled = 0, 0    # sync reset
        self.scatter = [None] * sub       # scatter_q (unreset)
        self.sidx, self.sfrom = 0, 0      # sync reset

    def _word(self, w):
        return self.mem.get(w, (None,) * self.SUB)

    def rdata(self):
        """``beat_rdata_o`` this cycle (combinational from state)."""
        if self.sfrom:
            return self.scatter[self.sidx]
        w = self.pipe[-1]
        return None if w is None else w[self.sidx]

    def edge(self, rst_n, o):
        wr, rd = o["vmem_wr_en"], o["vmem_rd_en"]
        en, we = bool(wr or rd), bool(wr)
        ptr = (o["vmem_wr_ptr"] if wr else o["vmem_rd_ptr"]) & 0x3FFF
        word, idx = ptr >> 2, ptr & (self.SUB - 1)
        commit = en and we and idx == self.SUB - 1
        fetch = en and not we and idx == 0
        word_en, word_we = commit or fetch, commit
        wwd = list(self.gather)
        if en and we:
            wwd[idx] = o["vmem_wr_data"]
        word_rdata, rv_last = self.pipe[-1], self.rvalid[-1]
        pipe = [self.pipe[0]] + self.pipe[:-1]  # stage 0 holds unless enabled
        if word_en:
            pipe[0] = self._word(word)
            if word_we:
                self.mem[word] = tuple(wwd)
        self.pipe = pipe
        if not rst_n:
            self.filled, self.gword = 0, 0
            self.sidx, self.sfrom = 0, 0
            self.rvalid = [0] * self.L
            return
        if en and we:
            self.gather[idx] = o["vmem_wr_data"]
            self.gword = word
            self.filled = 0 if commit else self.filled | (1 << idx)
        if en and not we:
            self.sidx, self.sfrom = idx, int(idx != 0)
        if rv_last:
            self.scatter = list(word_rdata) if word_rdata is not None else [None] * self.SUB
        self.rvalid = [int(word_en and not word_we)] + self.rvalid[:-1]


def closed_trace(cmd, n, params):
    """``dma_trace`` closed on ``VmemDmaSide``: ``({port: [int]}, {port: [reason]}, events)``."""
    m = DmaModel(**params)
    v = VmemDmaSide()
    names = [p for p, _ in INPUTS]
    out = {p: [] for p, _ in OUTPUTS}
    why = {p: [] for p, _ in OUTPUTS}
    ev = {}
    wd_def = True
    for t in range(n):
        x = {p: cmd[p][t] for p in names}
        x["vmem_gnt"] = 1
        rd = v.rdata()
        x["vmem_rd_data"] = 0 if rd is None else rd
        if not x["rst_n"]:
            m.reset()
            wd_def = True  # st_data_q resets to 0
        o, it = m.comb(x)
        empty = m.lnd_count == 0
        for p, _ in OUTPUTS:
            out[p].append(o[p])
            r = ""
            if t == 0:
                r = "power-up"
            elif p in ("vmem_wr_data", "vmem_wr_ptr") and (empty or m.lnd_mem[m.lnd_head] is None):
                r = "landing empty"
            elif p == "dm_req_wdata" and not wd_def:
                r = "vmem uninit"
            why[p].append(r)
        if x["rst_n"]:
            if m.eng == I_ST_RDW and m.rd_lat == m_lat(m) - 1:
                wd_def = rd is not None
                ev["store beat read"] = ev.get("store beat read", 0) + 1
                ev["store beat read undefined"] = ev.get("store beat read undefined", 0) + int(rd is None)
            m.edge(x, o, it, ev)
        v.edge(x["rst_n"], o)
    return out, why, ev


def m_lat(_m):
    from examples.minitpu.harness.ref_ctrl_dma import VMEM_DMA_READ_LATENCY  # noqa: PLC0415
    return VMEM_DMA_READ_LATENCY


def REF(inst, cmd):
    plain = {p: rtl.unpack(cmd[p]) for p, _ in INPUTS}
    n = len(plain["rst_n"])
    out, why, ev = closed_trace(plain, n, PARAMS[inst])
    want = {p: rtl.pack(out[p], w) for p, w in OUTPUTS}
    reason = {p: np.array(why[p], dtype=object) for p, _ in OUTPUTS}
    return want, reason, ev


# ---------------------------------------------------------------------------
# Traces: Phase 0's program generator, vmem_gnt 1, round trips
# ---------------------------------------------------------------------------

def _program(inst, tag, **kw):
    rng = rng_for("dma-vmem", inst, tag)
    br = D.Bridge(rng, **{k: v for k, v in kw.items() if k in ("rtt", "p_gap", "p_err", "drop", "late")})
    return D.Program(inst, rng, br, p_ready=kw.get("p_ready", 1.0), p_gnt=1.0), rng


def _cmd(p):
    c = p.cmd()
    return {q: c[q] for q, _ in INPUTS}


def directed(inst):
    out = []
    # load 8 words, store them back (stride 1), then read them back strided
    p, _ = _program(inst, "rt", rtt=(3, 7), p_gap=0.1)
    p.desc(0, 0, 0x100, 31, 0x2000, 1)
    p.drain()
    p.desc(1, 1, 0x100, 31, 0x9000, 1)
    p.drain()
    p.desc(0, 1, 0x100, 15, 0xA000, 3)
    p.drain()
    out.append(("round trip, stride 1 and 3", _cmd(p), True))
    # two channels, backpressure: loads on 0 while 1 stores rows loaded before
    p, _ = _program(inst, "rt2", rtt=(1, 12), p_gap=0.2, p_ready=0.6)
    p.desc(0, 0, 0x200, 15, 0x300, 1)
    p.desc(1, 0, 0x240, 15, 0x400, 2)
    p.drain()
    p.desc(0, 1, 0x200, 15, 0x500, 1)
    p.desc(1, 0, 0x280, 15, 0x600, 1)
    p.desc(0, 1, 0x240, 7, 0x700, 1)
    p.drain()
    out.append(("two channels, ready 0.6", _cmd(p), True))
    # a partial word: 6 beats loaded (the second word's two beats gathered, never
    # committed), stored back (its unwritten beats are undefined), then a fresh
    # aligned load that overwrites the stale gather before committing
    p, _ = _program(inst, "partial", rtt=(2, 5))
    p.desc(0, 0, 0x40, 5, 0x800, 1)
    p.drain()
    p.desc(1, 1, 0x40, 7, 0x880, 1)
    p.drain()
    p.desc(0, 0, 0x80, 7, 0x900, 1)
    p.drain()
    p.desc(1, 1, 0x80, 7, 0x980, 1)
    p.drain()
    out.append(("partial word, then an aligned load", _cmd(p), True))
    # a load across a 4 KiB page (two bursts), stored back across one too
    p, _ = _program(inst, "page", rtt=(2, 6), p_gap=0.05)
    p.desc(0, 0, 0x300, 4 * 40 - 1, 0x1000 * 3 + 100, 1)
    p.drain()
    p.desc(1, 1, 0x300, 4 * 40 - 1, 0x1000 * 5 + 120, 1)
    p.drain()
    out.append(("page-split load and store, 40 words", _cmd(p), True))
    # reset mid-transfer: VMEM is not reset, so the store after it reads the first load
    p, _ = _program(inst, "rst", rtt=(3, 8))
    p.desc(0, 0, 0x60, 15, 0x700, 1)
    p.drain()
    p.desc(1, 0, 0xA0, 31, 0x800, 1)
    p.idle(6)
    p.idle(2, rst_n=0)
    p.desc(1, 1, 0x60, 15, 0x900, 1)
    p.drain()
    out.append(("reset mid-transfer, VMEM survives", _cmd(p), True))
    return out


def random_closed(inst, seed, n=30, p_ready=0.8):
    rng = rng_for("dma-vmem-rand", inst, seed)
    br = D.Bridge(rng, rtt=(1, 16), p_gap=0.15)
    p = D.Program(inst, rng, br, p_ready=p_ready, p_gnt=1.0)
    hot = [0x000, 0x020, 0x100, 0x104, 0x2F0, 0x380]  # all below word 256 (CSIM_ROWS)
    for i in range(n):
        store = i >= 4 and rng.random() < 0.5
        words = rng.choice([1, 1, 2, 4, 8])
        rows = 4 * words - 1
        stride = rng.choice([1, 1, 1, 2, 5])
        base = rng.choice([rng.getrandbits(16), 0x1F80 + rng.randrange(0, 256)])
        p.desc(rng.randrange(2), int(store), rng.choice(hot), rows, base, stride)
        p.idle(rng.choice([0, 0, 1, 4]))
    p.drain()
    return _cmd(p)


def traces(inst):
    tr = directed(inst)
    k = {"core": 3, "o1": 2, "o4": 2}[inst]
    tr += [(f"random-closed-{s}", random_closed(inst, s), True) for s in range(k)]
    for lab, c, _ in tr:  # the reduced-rows workaround (C7) is exact only below CSIM_ROWS
        assert max(((r + k) & 0x3FFF) >> 2 for r, k, v in
                   zip(c["desc_vmem_row"], c["desc_rows"], c["desc_valid"]) if v) < CSIM_ROWS, lab
    return tr


# ---------------------------------------------------------------------------
# Allo variants: units/dma_unit.py vmem_architecture
# ---------------------------------------------------------------------------

def d12(n, w=256, inst="core", payload="unreset"):
    """D2: ``dma_engine`` + ``vmem_group`` (owner of the D-12 port ``vmem.d``,
    ``rw`` latency 2) + an idle owner of ``vmem.c``, between a source and a sink."""
    from examples.minitpu.units.dma_unit import vmem_architecture  # noqa: PLC0415

    return vmem_architecture(n, inst, payload).region("simulator")


def d12_reset(n, w=256, inst="core"):
    return d12(n, w, inst, payload="reset")


CSIM_ROWS = 256  # every trace here stays below VMEM word 256 (asserted in traces())


def d12_reset_r256(n, w=256, inst="core"):
    """``d12_reset`` with VMEM cut to 256 rows (finding C7's workaround for csim)."""
    from examples.minitpu.units.dma_unit import vmem_architecture  # noqa: PLC0415

    return vmem_architecture(n, inst, "reset", rows=CSIM_ROWS).region("simulator")


def run_closed(mod, cmd, n, w=256):
    c = dict(cmd)
    c["vmem_rd_data"] = [0] * n
    c["vmem_gnt"] = [1] * n
    return D.run_bits(mod, c, n, w)


VARIANTS = {"d12": (d12, run_closed), "d12_reset": (d12_reset, run_closed),
            "d12_reset_r256": (d12_reset_r256, run_closed)}

assert set(GEOMETRIES) >= set(PARAMS)
