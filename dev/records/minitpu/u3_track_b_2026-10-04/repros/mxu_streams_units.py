"""Probe units: the MXU output FIFOs as per-lane Streams (u3_fifo_composed's
recommendation): the gather ``try_put``\\ s a finished group (refusal = the
RTL's drop), a separate pop engine polls every lane's ``empty()`` and
``get``\\ s on a pop. No reset handling (a Stream has no pointer clear)."""
from __future__ import annotations
from allo.compose import Channel, unit


def probe_channels(base):
    return base + (Channel("laneq", "UInt(64)", "ENTRIES", ("D",), "gather -> pop engine, one FIFO per lane"),
                   Channel("ctl", "UInt(32)", "2", (), "front -> gather: rst_n"))


@unit(memories=("DROP",), reads=("px", "ctl"), writes=("laneq",), parameters=("N", "D", "SUB"))
def gather_try(drop: UInt(16)[N]):
    gidx: UInt(2)[D] = 0
    gq: UInt(64)[D] = 0
    for t in range(N):
        c_t: UInt(32) = ctl.get()
        dropped: UInt(16) = 0
        with allo.meta_for(D) as lane:
            sp: UInt(32) = px[D, lane].get()
            res: UInt(24) = sp[0:24]
            rv: UInt(1) = sp[24]
            rounded: UInt(24) = res + 0x7F + res[8]
            bf: UInt(16) = rounded[8:24]
            gnext: UInt(64) = gq[lane]
            gi: UInt(2) = gidx[lane]
            with allo.meta_for(SUB) as sub:
                if gi == sub:
                    gnext[16 * sub:16 * (sub + 1)] = bf
            if rv == 1:
                gq[lane] = gnext
                gidx[lane] = gi + 1
                if gi == SUB - 1:
                    ok: UInt(1) = laneq[lane].try_put(gnext)
                    if ok == 0:
                        dropped[lane] = 1
        drop[t] = dropped


@unit(memories=("POP", "VLD", "ODATA"), reads=("laneq",), parameters=("N", "D", "SUB"))
def pop_engine(pop: UInt(1)[N], vld: UInt(1)[N], odata: UInt(16)[N * D * SUB]):
    for t in range(N):
        q: UInt(1) = pop[t]
        valid: UInt(1) = 1
        with allo.meta_for(D) as lane:
            e: UInt(1) = laneq[lane].empty()
            if e == 1:
                valid = 0
        vld[t] = valid
        if q == 1:
            if valid == 1:
                with allo.meta_for(D) as lane:
                    g: UInt(64) = laneq[lane].get()
                    with allo.meta_for(SUB) as sub:
                        odata[t * (D * SUB) + sub * D + lane] = g[16 * sub:16 * (sub + 1)]
