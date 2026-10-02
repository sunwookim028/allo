# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U2: ``vpu_word_array``, VMEM: one array of whole words, two ports.

Ports ``compute`` and ``dma`` are symmetric: ``en``/``we``/``addr``/``wdata``
in, ``rdata`` out, whole words only. Reads are registered through a pipe of
``READ_LATENCY`` (compute, ``vpu_pkg::VMEM_READ_LATENCY = 3``) or
``DMA_READ_LATENCY`` (``VMEM_DMA_READ_LATENCY = 2``) stages. No reset. Driven
as a ``trace`` unit with both ``rdata`` ports sampled after the edge; this is
the simulation model (the ``ifndef SYNTHESIS`` branch), not the XPM
``xpm_memory_tdpram`` the board builds. Reference: ``harness/ref.py``
``vpu_word_array_trace``.

Instances (``vpu_pkg`` defines and ``-G`` parameters):

``narrow``      ``MINITPU_NUM_LANES=1``, ``MINITPU_VMEM_ENTRIES_PER_LANE=16``:
                8 words of 64 b, latencies 3/2 -- dense collisions, fast
``narrow_rl2``  the same with ``READ_LATENCY=2``
``full``        the default geometry: 4096 words of 1024 b, latencies 3/2
``full_rl2``    ``READ_LATENCY=2``: what ``tb_vpu_word_array`` instantiates

Legal traces keep the two ports off one word in a cycle when either writes
(``vpu_vmem_simd.sv:121-131`` asserts it in simulation; nothing else checks
it). Illegal traces break that rule on purpose.

No Allo variant yet (U2 plan, checkpoint 1).
"""

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness.traces import Trace, hot_addr, rng_for, word

SOURCES = ["src/core/vpu/vpu_pkg.sv", "src/core/vpu/vpu_word_array.sv"]
NARROW = ["MINITPU_NUM_LANES=1", "MINITPU_VMEM_ENTRIES_PER_LANE=16"]
GEOM = {  # instance: (word bits, words, address bits, compute latency, dma latency)
    "narrow": (64, 8, 3, 3, 2),
    "narrow_rl2": (64, 8, 3, 2, 2),
    "full": (1024, 4096, 12, 3, 2),
    "full_rl2": (1024, 4096, 12, 2, 2),
}
PORTS = ("compute", "dma")


def _unit(inst):
    ww, _, aw, rl, drl = GEOM[inst]
    ins = [("rst_ni", 1)]
    for p in PORTS:
        ins += [(f"{p}_en_i", 1), (f"{p}_we_i", 1), (f"{p}_addr_i", aw), (f"{p}_wdata_i", ww)]
    return rtl.RtlUnit(
        top="vpu_word_array",
        sources=SOURCES,
        inputs=ins,
        outputs=[(f"{p}_rdata_o", ww, "post") for p in PORTS],
        shape="trace",
        defines=NARROW if inst.startswith("narrow") else [],
        params={} if rl == 3 else {"READ_LATENCY": rl},
        assertions=True,
    )


INSTANCES = {k: _unit(k) for k in GEOM}
DEFAULT = "narrow"
RTL = INSTANCES[DEFAULT]
VARIANTS = {}
LATENCY_SOURCE = "vpu_pkg.sv:58-59 (VMEM_READ_LATENCY = 3, VMEM_DMA_READ_LATENCY = 2)"


def REF(inst, cmd):
    ww, _, _, rl, drl = GEOM[inst]
    return ref.vpu_word_array_trace(cmd, (ww + 63) // 64, rl, drl)


def _defaults():
    d = {"rst_ni": 1}
    for p in PORTS:
        d.update({f"{p}_en_i": 0, f"{p}_we_i": 0, f"{p}_addr_i": 0, f"{p}_wdata_i": 0})
    return d


def _acc(p, addr, data=None):
    """One port's access: a read, or a write of ``data``."""
    a = {f"{p}_en_i": 1, f"{p}_addr_i": addr}
    if data is not None:
        a.update({f"{p}_we_i": 1, f"{p}_wdata_i": data})
    return a


def random_trace(inst, n, seed, legal):
    ww, words, _, _, _ = GEOM[inst]
    rng = rng_for("word_array", inst, seed, legal)
    hot = rng.sample(range(words), 3)
    t = Trace(_defaults())
    for _ in range(n):
        row = {}
        for p in PORTS:
            if rng.random() < 0.75:
                row.update(_acc(p, hot_addr(rng, words, hot),
                                word(rng, ww) if rng.random() < 0.4 else None))
        c, d = row.get("compute_addr_i"), row.get("dma_addr_i")
        if legal and "compute_en_i" in row and "dma_en_i" in row and c == d and (
                "compute_we_i" in row or "dma_we_i" in row):
            row["dma_addr_i"] = (d + 1) % words
        row["rst_ni"] = int(rng.random() > 0.01)
        t.cycle(**row)
    return t.cmd()


def directed(inst):
    """``[(label, cmd, legal)]``: every port pair, write then read at every
    offset -1..+L; same-port read-during-write; both ports reading one word;
    both ports writing one word, and one writing while the other reads; the
    bottom, an interior and the top address; idle cycles between reads."""
    ww, words, _, rl, drl = GEOM[inst]
    lat = {"compute": rl, "dma": drl}
    rng = rng_for("word_array-directed", inst)
    targets = [0, words // 2 + 1, words - 1]
    out = []
    for wp in PORTS:
        for rp in PORTS:
            legal = Trace(_defaults())
            coll = Trace(_defaults())
            for a in targets:
                for t in (legal, coll):
                    t.cycle(**_acc(wp, a, word(rng, ww))).idle(lat[rp] + 1)
                for off in range(-1, lat[rp] + 1):
                    t = coll if (off == 0 and wp != rp) else legal
                    seq = [dict() for _ in range(off + 3)] if off >= 0 else [dict() for _ in range(3)]
                    wi = 1 if off >= 0 else 2
                    seq[wi].update(_acc(wp, a, word(rng, ww)))
                    ri = wi + off
                    if wp == rp and off == 0:
                        pass  # the write's own cycle: covered by "rdw-same-port"
                    else:
                        seq[ri].update(_acc(rp, a))
                    for row in seq:
                        t.cycle(**row)
                    t.idle(lat[rp] + 1)
            out.append((f"wr-offsets-{wp}->{rp}", legal.cmd(), True))
            if wp != rp:
                out.append((f"wr-same-cycle-{wp}->{rp}", coll.cmd(), False))
    t = Trace(_defaults())  # same-port read-during-write: the write cycle's read slot
    for p in PORTS:
        for a in targets:
            t.cycle(**_acc(p, a, word(rng, ww)))
            t.cycle(**_acc(p, a, word(rng, ww)))  # read-during-write of a written word
            t.cycle(**_acc(p, a)).idle(lat[p] + 1)
    out.append(("rdw-same-port", t.cmd(), True))
    t = Trace(_defaults())  # both ports read one word in one cycle: legal
    for a in targets:
        t.cycle(**_acc("compute", a, word(rng, ww))).idle(1)
        t.cycle(**_acc("compute", a), **_acc("dma", a)).idle(max(lat.values()) + 1)
    out.append(("both-read-one-word", t.cmd(), True))
    t = Trace(_defaults())  # both ports write one word: undefined until rewritten
    for a in targets:
        t.cycle(**_acc("compute", a, word(rng, ww)), **_acc("dma", a, word(rng, ww)))
        t.cycle(**_acc("compute", a)).cycle(**_acc("dma", a)).idle(4)
        t.cycle(**_acc("dma", a, word(rng, ww))).cycle(**_acc("compute", a)).idle(4)
    out.append(("ww-collision", t.cmd(), False))
    t = Trace(_defaults())  # gaps: the pipe holds its last read (masked "no read")
    for a in targets:
        t.cycle(**_acc("dma", a, word(rng, ww)))
    for a in targets:
        t.cycle(**_acc("compute", a), **_acc("dma", targets[0])).idle(5)
    out.append(("idle-gaps", t.cmd(), True))
    return out


def traces(inst):
    n = 20000 if inst.startswith("narrow") else 3000
    tr = directed(inst)
    tr += [(f"random-legal-{s}", random_trace(inst, n, s, True), True) for s in range(2)]
    tr += [(f"random-any-{s}", random_trace(inst, n, s, False), False) for s in range(2)]
    return tr


def probes(inst):
    """Read latency per port (a step from one word to another), and write
    visibility (the first read that sees a write, minus the read latency),
    same port and across ports."""
    ww, words, _, rl, drl = GEOM[inst]
    lat = {"compute": rl, "dma": drl}
    u = INSTANCES[inst]
    one, two = 1, (1 << ww) - 2
    res = {}
    out = []
    for p in PORTS:
        o = "dma" if p == "compute" else "compute"
        t = Trace(_defaults())
        t.cycle(**_acc(o, 1, one)).cycle(**_acc(o, 2, two)).idle(2)
        t.idle(8, **_acc(p, 1))
        ev = len(t)
        t.idle(8, **_acc(p, 2))
        res[p] = rtl.probe_trace(u, t.cmd(), f"{p}_rdata_o", ev)
        out.append((f"read {p}", lat[p], res[p]))
    for wp in PORTS:
        for rp in PORTS:
            t = Trace(_defaults())
            t.cycle(**_acc(wp, 3, one)).idle(2)
            t.idle(8, **_acc(rp, 3))
            ev = len(t)
            row = _acc(wp, 3, two)
            if wp != rp:
                row.update({f"{rp}_en_i": 0})  # the reader skips the write's cycle
            t.cycle(**row)
            t.idle(8, **_acc(rp, 3))
            out.append((f"write {wp} -> read {rp} (visibility)", 1,
                        rtl.probe_trace(u, t.cmd(), f"{rp}_rdata_o", ev) - res[rp]))
    return out


TB_SOURCES = ["src/core/vpu/vpu_pkg.sv", "src/core/vpu/vpu_word_array.sv",
              "src/core/vpu/vpu_dma_group.sv"]


def seeds():
    """``tb_vpu_word_array`` (DMA beats through ``vpu_dma_group``, compute
    words, ``READ_LATENCY = 2``), replayed at 2 against the tb's own DUT and
    at the production 3 against the reference only."""
    from examples.minitpu.harness import vcd_seed

    cmd, seen = vcd_seed.extract("tb_vpu_word_array", TB_SOURCES, dut="dut",
                                 clk="clk_i", unit=INSTANCES["full_rl2"])
    return [("tb_vpu_word_array", "full_rl2", cmd, seen, True),
            ("tb_vpu_word_array@RL3", "full", cmd, None, True)]
