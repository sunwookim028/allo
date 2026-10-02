# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U2: ``vpu_regfile``, one sublane's VREG stripe: 32 entries, 3R1W.

Three asynchronous reads (``assign rdata = mem[raddr]``) and one synchronous
write; never reset (``rst_ni`` is unused), so a VREG must be written before it
is read. Driven as a ``trace`` unit (``harness/rtl.py``): every read port is
sampled before the edge, so a read and a write of one VREG in one cycle show
the old value. Reference: ``harness/ref.py`` ``vpu_regfile_trace``.

Instances: ``w16`` is the module's default ``WIDTH = DATA_WIDTH``, what
``tb_vpu_alu_regfile`` drives; ``w256`` is the width ``vpu_vreg_stripe.sv:26``
instantiates (``VMEM_STRIPE_W``, 16 lanes x 16 b), four per VPU.

Declared latencies (``vpu_regfile.sv`` comment "Three async reads, one sync
write"; ``REGISTER_FILE.md``): read 0 edges on every port; write visible to a
read 1 edge later.

No Allo variant yet: the expression waits for the owner (U2 plan,
checkpoint 1).
"""

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness.traces import Trace, concat, hot_addr, rng_for, word

SOURCES = ["src/core/vpu/vpu_pkg.sv", "src/core/vpu/vpu_regfile.sv"]
DEPTH = 32


def _unit(width):
    return rtl.RtlUnit(
        top="vpu_regfile",
        sources=SOURCES,
        inputs=[("rst_ni", 1), ("raddr_a_i", 5), ("raddr_b_i", 5), ("raddr_c_i", 5),
                ("waddr_i", 5), ("wdata_i", width), ("we_i", 1)],
        outputs=[("rdata_a_o", width), ("rdata_b_o", width), ("rdata_c_o", width)],
        shape="trace",
        params={} if width == 16 else {"WIDTH": width},
        assertions=True,
    )


INSTANCES = {"w16": _unit(16), "w256": _unit(256)}
WIDTH = {"w16": 16, "w256": 256}
DEFAULT = "w16"
RTL = INSTANCES[DEFAULT]
VARIANTS = {}
LATENCY_SOURCE = "vpu_regfile.sv:29 comment (\"Three async reads, one sync write\")"


def REF(inst, cmd):
    return ref.vpu_regfile_trace(cmd, WIDTH[inst], DEPTH)


def _defaults():
    return {"rst_ni": 1, "raddr_a_i": 0, "raddr_b_i": 0, "raddr_c_i": 0,
            "waddr_i": 0, "wdata_i": 0, "we_i": 0}


def random_trace(inst, n, seed):
    rng = rng_for("regfile", inst, seed)
    w = WIDTH[inst]
    hot = rng.sample(range(DEPTH), 4)
    t = Trace(_defaults())
    for _ in range(n):
        t.cycle(rst_ni=int(rng.random() > 0.01),
                raddr_a_i=hot_addr(rng, DEPTH, hot), raddr_b_i=hot_addr(rng, DEPTH, hot),
                raddr_c_i=hot_addr(rng, DEPTH, hot), waddr_i=hot_addr(rng, DEPTH, hot),
                wdata_i=word(rng, w), we_i=int(rng.random() < 0.5))
    return t.cmd()


def directed(inst):
    """Write-then-read at every offset -1..+2 on every read port; all three
    ports on one VREG; every VREG written then read; reset mid-trace."""
    w = WIDTH[inst]
    rng = rng_for("regfile-directed", inst)
    out = []
    t = Trace(_defaults())
    for v in range(DEPTH):  # fill, then read back all three ports on one entry
        t.cycle(waddr_i=v, wdata_i=word(rng, w), we_i=1)
    for v in range(DEPTH):
        t.cycle(raddr_a_i=v, raddr_b_i=v, raddr_c_i=v)
    out.append(("fill-readback", t.cmd()))
    for port in "abc":
        t = Trace(_defaults())
        for v in range(DEPTH):
            t.cycle(waddr_i=v, wdata_i=word(rng, w), we_i=1)
        for off in (-1, 0, 1, 2):  # read of VREG 7 at cycle (write + off)
            seq = [dict() for _ in range(5)]
            seq[1].update(waddr_i=7, wdata_i=word(rng, w), we_i=1)
            seq[1 + off][f"raddr_{port}_i"] = 7
            for row in seq:
                t.cycle(**row)
        # back-to-back writes of one VREG, read every cycle
        for k in range(4):
            t.cycle(**{f"raddr_{port}_i": 9}, waddr_i=9, wdata_i=word(rng, w), we_i=1)
        t.cycle(**{f"raddr_{port}_i": 9})
        out.append((f"wr-offsets-port-{port}", t.cmd()))
    t = Trace(_defaults())  # reset does not clear: write, reset, read
    for v in range(4):
        t.cycle(waddr_i=v, wdata_i=word(rng, w), we_i=1)
    t.idle(4, rst_ni=0)
    for v in range(4):
        t.cycle(raddr_a_i=v, raddr_b_i=v, raddr_c_i=3 - v)
    t.cycle(rst_ni=0, waddr_i=1, wdata_i=word(rng, w), we_i=1)  # a write under reset lands
    t.cycle(raddr_a_i=1)
    out.append(("reset-keeps", t.cmd()))
    return out


def traces(inst):
    """``[(label, cmd, legal)]``. Every regfile trace is legal: it has no
    illegal command; reads of unwritten VREGs are masked, not refused."""
    tr = [(lab, c, True) for lab, c in directed(inst)]
    n = 20000 if inst == "w16" else 5000
    tr += [(f"random-{s}", random_trace(inst, n, s), True) for s in range(3)]
    return tr


def probes(inst):
    """``[(label, declared, measured)]`` by step probes on the RTL."""
    u = INSTANCES[inst]
    w = WIDTH[inst]
    res = []
    for port in "abc":
        t = Trace(_defaults())
        t.cycle(waddr_i=3, wdata_i=1, we_i=1).cycle(waddr_i=4, wdata_i=(1 << w) - 2, we_i=1)
        t.idle(8, **{f"raddr_{port}_i": 3})
        ev = len(t)
        t.idle(8, **{f"raddr_{port}_i": 4})
        res.append((f"read {port}", 0, rtl.probe_trace(u, t.cmd(), f"rdata_{port}_o", ev)))
        t = Trace(_defaults())
        t.cycle(waddr_i=5, wdata_i=1, we_i=1)
        t.idle(8, **{f"raddr_{port}_i": 5})
        ev = len(t)
        t.cycle(**{f"raddr_{port}_i": 5}, waddr_i=5, wdata_i=2, we_i=1)
        t.idle(8, **{f"raddr_{port}_i": 5})
        res.append((f"write -> read {port}", 1, rtl.probe_trace(u, t.cmd(), f"rdata_{port}_o", ev)))
    return res


def seeds():
    """MiniTPU tb scenarios replayed as traces (``harness/vcd_seed.py``)."""
    from examples.minitpu.harness import vcd_seed

    cmd, seen = vcd_seed.extract(
        "tb_vpu_alu_regfile",
        ["src/core/vpu/vpu_pkg.sv", "src/core/vpu/vpu_regfile.sv",
         "src/core/vpu/vpu_bf16_add_pipe.sv", "src/core/vpu/vpu_bf16_mul.sv",
         "src/core/vpu/vpu_alu.sv"],
        dut="i_rf", clk="clk_i", unit=INSTANCES["w16"],
    )
    return [("tb_vpu_alu_regfile", "w16", cmd, seen, True)]
