# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track-D forms of ``dma.streams_reset`` (D1 over Streams) and
``dma_vmem.d12_reset_r256`` (D2) for Catapult (the lane slices written out
for ``L = 8`` by a one-off script; ``dma_unit``'s boundary units otherwise
verbatim).

``dma_unit``'s engine, ``vmem_group`` and ``vmem_compute_idle`` are used
unchanged; only the boundary units change (U3 track C C1/C2): the column and
lane arrays ``CI: UInt(32)[N, NC]``, ``DI: UInt(32)[N, 2 L]``, ``CO``, ``DO``
become one ``UInt(32 k)`` port per row, read or written once per iteration,
lanes as literal slices (``allo.meta_for`` over a port is 2 L accesses of
one array: RAM pins). Runners: ``dma.run_bits`` / ``dma_vmem.run_closed``.
"""
from __future__ import annotations

from allo.compose import Architecture, Channel, Instance, Memory, Port, unit
from allo.ir.types import UInt  # noqa: F401

from examples.minitpu.units import dma as D
from examples.minitpu.units import dma_unit as DU
from examples.minitpu.units import dma_vmem as DV


@unit(memories=("CI", "DI"), writes=("x_desc", "x_base", "x_stride", "x_ctl", "x_rsp", "x_rspd", "x_vrd"),
      parameters=("N", "CIW", "DIW"))
def dma_src(ci: UInt(CIW)[N], di: UInt(DIW)[N]):
    for t in range(N):
        ciw: UInt(CIW) = ci[t]
        diw: UInt(DIW) = di[t]
        d: UInt(64) = 0
        v1: UInt(1) = ciw[32:64]
        s1: UInt(1) = ciw[64:96]
        c1: UInt(1) = ciw[96:128]
        vr: UInt(14) = ciw[128:160]
        rw: UInt(14) = ciw[160:192]
        cl: UInt(7) = ciw[192:224]
        d[0] = v1
        d[1] = s1
        d[2] = c1
        d[3:17] = vr
        d[17:31] = rw
        d[31:38] = cl
        x_desc.put(d)
        bs: UInt(32) = ciw[224:256]
        sd: UInt(32) = ciw[256:288]
        x_base.put(bs)
        x_stride.put(sd)
        c: UInt(32) = 0
        r1: UInt(1) = ciw[0:32]
        clr: UInt(2) = ciw[288:320]
        rdy: UInt(1) = ciw[320:352]
        gnt: UInt(1) = ciw[448:480]
        c[0] = r1
        c[1:3] = clr
        c[3] = rdy
        c[4] = gnt
        x_ctl.put(c)
        r: UInt(32) = 0
        rv: UInt(1) = ciw[352:384]
        rl: UInt(1) = ciw[384:416]
        rr: UInt(2) = ciw[416:448]
        r[0] = rv
        r[1] = rl
        r[2:4] = rr
        x_rsp.put(r)
        x_rspd[0].put(diw[0:32])
        x_rspd[1].put(diw[32:64])
        x_rspd[2].put(diw[64:96])
        x_rspd[3].put(diw[96:128])
        x_rspd[4].put(diw[128:160])
        x_rspd[5].put(diw[160:192])
        x_rspd[6].put(diw[192:224])
        x_rspd[7].put(diw[224:256])
        x_vrd[0].put(diw[256:288])
        x_vrd[1].put(diw[288:320])
        x_vrd[2].put(diw[320:352])
        x_vrd[3].put(diw[352:384])
        x_vrd[4].put(diw[384:416])
        x_vrd[5].put(diw[416:448])
        x_vrd[6].put(diw[448:480])
        x_vrd[7].put(diw[480:512])


@unit(memories=("CI", "DI"), writes=("x_desc", "x_base", "x_stride", "x_ctl", "x_rsp", "x_rspd"),
      parameters=("N", "CIW", "DIW"))
def dma_src_closed(ci: UInt(CIW)[N], di: UInt(DIW)[N]):
    for t in range(N):
        ciw: UInt(CIW) = ci[t]
        diw: UInt(DIW) = di[t]
        d: UInt(64) = 0
        v1: UInt(1) = ciw[32:64]
        s1: UInt(1) = ciw[64:96]
        c1: UInt(1) = ciw[96:128]
        vr: UInt(14) = ciw[128:160]
        rw: UInt(14) = ciw[160:192]
        cl: UInt(7) = ciw[192:224]
        d[0] = v1
        d[1] = s1
        d[2] = c1
        d[3:17] = vr
        d[17:31] = rw
        d[31:38] = cl
        x_desc.put(d)
        bs: UInt(32) = ciw[224:256]
        sd: UInt(32) = ciw[256:288]
        x_base.put(bs)
        x_stride.put(sd)
        c: UInt(32) = 0
        r1: UInt(1) = ciw[0:32]
        clr: UInt(2) = ciw[288:320]
        rdy: UInt(1) = ciw[320:352]
        gnt: UInt(1) = ciw[448:480]
        c[0] = r1
        c[1:3] = clr
        c[3] = rdy
        c[4] = 1  # vmem_gnt: minitpu_core.sv ties it to 1
        x_ctl.put(c)
        r: UInt(32) = 0
        rv: UInt(1) = ciw[352:384]
        rl: UInt(1) = ciw[384:416]
        rr: UInt(2) = ciw[416:448]
        r[0] = rv
        r[1] = rl
        r[2:4] = rr
        x_rsp.put(r)
        x_rspd[0].put(diw[0:32])
        x_rspd[1].put(diw[32:64])
        x_rspd[2].put(diw[64:96])
        x_rspd[3].put(diw[96:128])
        x_rspd[4].put(diw[128:160])
        x_rspd[5].put(diw[160:192])
        x_rspd[6].put(diw[192:224])
        x_rspd[7].put(diw[224:256])


@unit(memories=("CO", "DO"), reads=("y_req", "y_wd", "y_vreq", "y_vwd", "y_stat", "y_beats", "y_ovl"),
      parameters=("N", "COW", "DIW"))
def dma_sink(co: UInt(COW)[N], do: UInt(DIW)[N]):
    for t in range(N):
        q: UInt(64) = y_req.get()
        vq: UInt(64) = y_vreq.get()
        sq: UInt(64) = y_stat.get()
        cow: UInt(COW) = 0
        cow[0:32] = sq[0]
        cow[32:64] = sq[1:3]
        cow[64:96] = sq[3]
        cow[96:128] = q[0]
        cow[128:160] = q[1]
        cow[160:192] = q[2:31]
        cow[192:224] = q[31:39]
        cow[224:256] = 0xFFFFFFFF
        cow[256:288] = q[39]
        cow[288:320] = vq[1]
        cow[320:352] = vq[3:19]
        cow[352:384] = vq[2]
        cow[384:416] = vq[19:35]
        cow[416:448] = sq[4]
        cow[448:480] = sq[5]
        cow[480:512] = sq[6:8]
        cow[512:544] = sq[8:10]
        cow[544:576] = sq[10]
        cow[576:608] = y_beats.get()
        cow[608:640] = y_ovl.get()
        dow: UInt(DIW) = 0
        dow[0:32] = y_wd[0].get()
        dow[32:64] = y_wd[1].get()
        dow[64:96] = y_wd[2].get()
        dow[96:128] = y_wd[3].get()
        dow[128:160] = y_wd[4].get()
        dow[160:192] = y_wd[5].get()
        dow[192:224] = y_wd[6].get()
        dow[224:256] = y_wd[7].get()
        dow[256:288] = y_vwd[0].get()
        dow[288:320] = y_vwd[1].get()
        dow[320:352] = y_vwd[2].get()
        dow[352:384] = y_vwd[3].get()
        dow[384:416] = y_vwd[4].get()
        dow[416:448] = y_vwd[5].get()
        dow[448:480] = y_vwd[6].get()
        dow[480:512] = y_vwd[7].get()
        co[t] = cow
        do[t] = dow


def _ports(params):
    params.update({"CIW": 32 * len(D.CIN_NAMES), "COW": 32 * len(D.COUT_NAMES), "DIW": 64 * params["L"]})
    assert params["L"] == 8, "the lane slices are written out for L = 8"
    return (Memory("CI", "UInt(CIW)[N]"), Memory("DI", "UInt(DIW)[N]"),
            Memory("CO", "UInt(COW)[N]"), Memory("DO", "UInt(DIW)[N]"))


def streams_architecture(n, inst="core", payload="reset"):
    a = DU.streams_architecture(n, inst, payload)
    params = dict(a.parameters)
    return Architecture(name=a.name + "_packed", parameters=params, memories=_ports(params),
                        channels=a.channels, units=(dma_src, DU.dma_engine, dma_sink))


def vmem_architecture(n, inst="core", payload="reset", rows=DV.CSIM_ROWS):
    a = DU.vmem_architecture(n, inst, payload, rows=rows)
    params = dict(a.parameters)
    vmem = [m for m in a.memories if m.name == "vmem"][0]
    return Architecture(name=a.name + "_packed", parameters=params, memories=_ports(params) + (vmem,),
                        channels=a.channels,
                        units=(dma_src_closed, DU.dma_engine, DU.vmem_group, DU.vmem_compute_idle,
                               Instance(dma_sink, "dma_sink_d2", {"y_vreq": "g_vmo", "y_vwd": "g_vmd"})),
                        obligations=a.obligations)


def make(n, w=256, inst="core"):
    """``U4D_FORM`` = ``streams_reset`` (default) or ``d12_reset_r256``; the
    region as tracks C built it for the simulator and csim (Stream links)."""
    import os
    if os.environ.get("U4D_FORM", "streams_reset") == "d12_reset_r256":
        return vmem_architecture(n, inst).region("simulator")
    return streams_architecture(n, inst).region("simulator")


def run(mod, cmd, n, w=256):
    # dma_vmem's traces are closed on the VMEM port (no vmem_rd_data column)
    if "vmem_rd_data" not in cmd:
        return DV.run_closed(mod, cmd, n, w)
    return D.run_bits(mod, cmd, n, w)
