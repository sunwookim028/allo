# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-20 for MiniTPU's DMA: the declared parameters, what follows from
them, and the relations ``dma.sv``, ``dma_desc_adapter.sv`` and the VMEM DMA
port rest on.

In the RTL these live in four places: ``dma.sv``'s parameter comments
("must be >= OUTSTANDING"), its derived ``localparam``\\s (``CRED_W``,
``PAGE_WORDS``, ``LND_ENTRY_W``), a lint rule (``ROW_BITS`` must equal
``sequencer_pkg::DESC_BEAT_ROWS_W``) and a constant in another package
(``vpu_pkg::VMEM_DMA_READ_LATENCY``). Here the derived numbers are
properties (never fields, so none can be typed in beside the number it must
equal) and every relation is one ``legality`` condition naming the parameter
and the consequence, run when a variant is made.

``check_against_phase0`` holds the derived timing numbers to Phase 0's
measured ones (``u4_phase0_2026-10-08.rst``, the latency table) by running
the cycle model, which Phase 0 held bit-exact to the RTL.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clog2(x):
    return max(0, (int(x) - 1).bit_length())


def _log2(n, what):
    assert n >= 1 and n & (n - 1) == 0, f"{what}={n}: must be a power of two"
    return n.bit_length() - 1


@dataclass(frozen=True)
class DmaGeometry:  # pylint: disable=too-many-instance-attributes
    """``dma.sv``'s parameters (``minitpu_core.sv``'s instance by default),
    the descriptor adapter's field widths, and the beat as the Allo ports
    carry it (``LANES`` x ``LANE_BITS``, U3 P-8: a region port is a lane
    array; the lane stays below S8's 2**63)."""

    CHANNELS: int = 2
    OUTSTANDING: int = 2
    LANDING_DEPTH: int = 8
    TIMEOUT_CYCLES: int = 1024
    LANES: int = 8
    LANE_BITS: int = 32
    VMEM_BEAT_BITS: int = 256       # the vmem_* port: one 32 B sublane row (vpu_dma_group)
    NUM_SUBLANES: int = 4           # beats per 128 B VMEM word
    VMEM_ADDR_W: int = 12           # VMEM words (vpu_pkg)
    DESC_WORD_ROWS_W: int = 12      # D-slot ``rows`` (words - 1)
    ROW_BITS: int = 14              # dma.sv localparam (beats - 1)
    VMEM_DMA_READ_LATENCY: int = 2  # vpu_pkg.sv:52; the VMEM DMA port's declared latency (P-3)
    PAGE_BYTES: int = 4096
    DRAM_BEAT_ADDR_W: int = 29
    LEN_BITS: int = 8               # dm_req_len

    # ---- derived ---------------------------------------------------------
    @property
    def BEAT_BITS(self) -> int:
        return self.LANES * self.LANE_BITS

    @property
    def STRB_W(self) -> int:
        return self.BEAT_BITS // 8

    @property
    def PAGE_WORDS(self) -> int:
        """DM words per 4 KiB page: a fused burst is cut at each one."""
        return self.PAGE_BYTES // self.STRB_W

    @property
    def SUBLANE_SEL_W(self) -> int:
        return _log2(self.NUM_SUBLANES, "NUM_SUBLANES")

    @property
    def DESC_BEAT_ROWS_W(self) -> int:
        """``sequencer_pkg::DESC_BEAT_ROWS_W``: words - 1 with the sublane
        bits appended as ones (``{rows, 2'b11}``)."""
        return self.DESC_WORD_ROWS_W + self.SUBLANE_SEL_W

    @property
    def VMEM_BEAT_ADDR_W(self) -> int:
        return self.VMEM_ADDR_W + self.SUBLANE_SEL_W

    @property
    def CHANNEL_SEL_BITS(self) -> int:
        return _clog2(self.CHANNELS) if self.CHANNELS > 1 else 1

    @property
    def CRED_W(self) -> int:
        return _clog2(self.OUTSTANDING + 1)

    @property
    def OUT_PTR_W(self) -> int:
        return _clog2(self.OUTSTANDING) if self.OUTSTANDING > 1 else 1

    @property
    def LND_PTR_W(self) -> int:
        return _clog2(self.LANDING_DEPTH) if self.LANDING_DEPTH > 1 else 1

    @property
    def LND_CNT_W(self) -> int:
        return _clog2(self.LANDING_DEPTH + 1)

    @property
    def LND_ENTRY_W(self) -> int:
        """One landing entry: channel, is_store, VMEM pointer (16), the beat,
        resp (2), last (1). Its payload is the unreset part (P-7)."""
        return self.CHANNEL_SEL_BITS + 1 + 16 + self.BEAT_BITS + 2 + 1

    @property
    def TO_W(self) -> int:
        return 1 if self.TIMEOUT_CYCLES < 2 else _clog2(self.TIMEOUT_CYCLES)

    # ---- derived timing (bookings; D-10/D-20) -----------------------------
    @property
    def ACCEPT_TO_REQUEST(self) -> int:
        """``desc_accept`` -> first ``dm_req_valid``: PENDING at edge 1, the
        engine picks it at edge 2, the request is combinational from there."""
        return 2

    @property
    def LAST_BEAT_TO_DONE(self) -> int:
        """Last response beat -> ``dma_channel_done``: lands at edge 1,
        drains (and completes) at edge 2."""
        return 2

    @property
    def STORE_BEAT_INTERVAL(self) -> int:
        """Cycles per store beat with the bridge always ready: ``I_ST_RD``
        (1) + ``I_ST_RDW`` (the read latency) + ``I_ST_REQ`` (1). The store
        path is latency-bound, which is why the read latency is pinned (P-3)."""
        return self.VMEM_DMA_READ_LATENCY + 2

    def namespace(self) -> dict:
        return {k: getattr(self, k) for k in (
            "CHANNELS", "OUTSTANDING", "LANDING_DEPTH", "TIMEOUT_CYCLES", "LANES", "LANE_BITS",
            "BEAT_BITS", "PAGE_WORDS", "ROW_BITS", "DESC_BEAT_ROWS_W", "CRED_W", "OUT_PTR_W",
            "LND_PTR_W", "TO_W", "VMEM_DMA_READ_LATENCY", "STORE_BEAT_INTERVAL")}

    def legality(self):
        """Every relation the DMA's correctness rests on, as one condition set."""
        dma_legality(self)
        return self


def dma_legality(g: DmaGeometry):
    assert g.CHANNELS >= 1, f"CHANNELS={g.CHANNELS}: a DMA without a channel"
    assert g.OUTSTANDING >= 1, (
        f"OUTSTANDING={g.OUTSTANDING}: the credit counter starts at OUTSTANDING; "
        f"0 credits never issue a request (the engine waits in I_LD_REQ forever)")
    assert g.LANDING_DEPTH >= 1, f"LANDING_DEPTH={g.LANDING_DEPTH}: no landing slot"
    assert g.LANDING_DEPTH >= g.OUTSTANDING, (
        f"LANDING_DEPTH={g.LANDING_DEPTH} < OUTSTANDING={g.OUTSTANDING}: dma.sv:18 "
        f"requires LANDING_DEPTH >= OUTSTANDING so an in-flight burst can always land "
        f"(the RTL states it in a comment only; Phase 0's o4 instance sits on the edge)")
    assert g.ROW_BITS == g.DESC_BEAT_ROWS_W, (
        f"ROW_BITS={g.ROW_BITS} != DESC_BEAT_ROWS_W={g.DESC_BEAT_ROWS_W} "
        f"(= DESC_WORD_ROWS_W {g.DESC_WORD_ROWS_W} + SUBLANE_SEL_W {g.SUBLANE_SEL_W}): "
        f"a narrower row counter truncates the beat count silently (dma_addr_gen.sv:13, "
        f"dma.sv:26; a lint rule in the RTL)")
    assert g.BEAT_BITS == g.VMEM_BEAT_BITS, (
        f"BEAT_BITS={g.BEAT_BITS} (LANES {g.LANES} x LANE_BITS {g.LANE_BITS}) != "
        f"VMEM_BEAT_BITS={g.VMEM_BEAT_BITS}: the vmem_* port stays one VMEM beat wide; "
        f"any other bridge width needs a gearbox (dma.sv:20)")
    assert g.BEAT_BITS % 8 == 0, f"BEAT_BITS={g.BEAT_BITS}: not whole bytes (wstrb)"
    _log2(g.PAGE_WORDS, "PAGE_WORDS")
    assert g.PAGE_WORDS <= 1 << g.LEN_BITS, (
        f"PAGE_WORDS={g.PAGE_WORDS} > 2**LEN_BITS: a page-bounded burst length "
        f"(PAGE_WORDS - 1) would not fit dm_req_len[{g.LEN_BITS - 1}:0]; "
        f"burst_len <= words_to_bound_m1 is what keeps every burst a legal AXI length")
    assert g.VMEM_DMA_READ_LATENCY >= 1, (
        f"VMEM_DMA_READ_LATENCY={g.VMEM_DMA_READ_LATENCY}: I_ST_RDW counts to L-1; "
        f"an asynchronous port (0) wraps the counter and captures the wrong cycle")
    assert g.LANE_BITS <= 32, (
        f"LANE_BITS={g.LANE_BITS}: a region-port lane >= 2**63 is lost in SystemC csim (S8); "
        f"the Allo forms carry the beat as 32-bit lanes (a tool constraint, not the RTL's)")
    assert g.TIMEOUT_CYCLES >= 0


# The four RTL instances Phase 0 characterised (units/dma.py PARAMS).
GEOMETRIES = {
    "core": DmaGeometry(),
    "o1": DmaGeometry(OUTSTANDING=1),
    "o4": DmaGeometry(OUTSTANDING=4, LANDING_DEPTH=4),
    "bw": DmaGeometry(TIMEOUT_CYCLES=0),
}

# Wrong declarations the legality must refuse (check_legality_refusals).
WRONG = {
    "landing below outstanding": dict(OUTSTANDING=4, LANDING_DEPTH=2),
    "no credit": dict(OUTSTANDING=0),
    "row counter narrower than the descriptor": dict(ROW_BITS=12),
    "beat narrower than VMEM's": dict(LANES=4),
    "asynchronous VMEM DMA port": dict(VMEM_DMA_READ_LATENCY=0),
    "64-bit lanes (S8)": dict(LANES=4, LANE_BITS=64),
}


def check_legality_refusals():
    """Each wrong declaration is refused, naming its parameter."""
    out = []
    for what, kw in WRONG.items():
        try:
            DmaGeometry(**kw).legality()
            out.append((what, "ACCEPTED"))
        except AssertionError as e:
            out.append((what, f"refused: {str(e).split(':')[0]}"))
    return out


def check_against_phase0():
    """The derived timing against Phase 0's measurements, by the cycle model
    (``ref_ctrl_dma``, bit-exact to the RTL on every Phase 0 trace)."""
    from examples.minitpu.units import dma  # noqa: PLC0415 (slow import chain)

    res = []
    for inst, g in GEOMETRIES.items():
        g.legality()
        p = dma.PARAMS[inst]
        assert (p["outstanding"], p["landing"], p["timeout"], p["channels"]) == (
            g.OUTSTANDING, g.LANDING_DEPTH, g.TIMEOUT_CYCLES, g.CHANNELS), inst
        rng = dma.rng_for("dma-params", inst)
        pr = dma.Program(inst, rng, dma.Bridge(rng, rtt=(4, 4), p_gap=0.0))
        pr.idle(3)
        t_acc = pr.desc(0, 1, 0x40, 4 * 8 - 1, 0x100, 1)  # a 32-beat fused store
        fires, first_req = [], None
        while not pr.m.done & 1:
            t = len(pr.rows)
            o = pr.cycle()
            if o["dm_req_valid"] and first_req is None:
                first_req = t
            if o["dm_req_valid"] and pr.rows[-1]["dm_req_ready"] and o["dm_req_we"]:
                fires.append(t)
        gaps = sorted({b - a for a, b in zip(fires, fires[1:])})
        res.append((inst, "store beat interval", g.STORE_BEAT_INTERVAL, gaps))
        res.append((inst, "accept -> first request (store: + read)", g.ACCEPT_TO_REQUEST,
                    first_req - t_acc - (1 + g.VMEM_DMA_READ_LATENCY)))
    return res


if __name__ == "__main__":
    for inst, g in GEOMETRIES.items():
        g.legality()
        print(inst, g.namespace())
    for what, verdict in check_legality_refusals():
        print(f"LEGALITY {what}: {verdict}")
    for inst, what, decl, meas in check_against_phase0():
        ok = (meas == [decl]) if isinstance(meas, list) else meas == decl
        print(f"{'DERIVED-OK ' if ok else 'DERIVED-DIFF'} {inst} {what}: declared {decl} measured {meas}")
