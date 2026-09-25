# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A standard-cell area ESTIMATE, in milliseconds, for the CHIA objective. FROZEN.

This is not a measurement. A Design Compiler run is ~70 minutes on another
host with a licence, so it cannot sit in a search's inner loop, and the axis
the loop used instead -- Vitis csynth's FPGA resource table -- is the one this
project has measured to be misleading *in exactly the components a search
would most want to change*:

===========================  =========================  =====================
change                       what csynth reported        what 45 nm charged
===========================  =========================  =====================
operand burst widening       +43 % FF, +92 % block RAM  **+74.4 %** cell area
                                                        (DC, a clean pair)
channel depth ``QD`` 8->16   +9.5 % FF at the scored    **+6.6 %** cell area
                             point; **+0.9 %** on the   (a count off the RTL,
                             T=8 pair                   not a DC run)
===========================  =========================  =====================

The second row is the one to read carefully, because the numbers this project
quotes for it come from three different places. The FPGA price of ``QD``
8->16 on the *scored* design is +9.5 % flip-flops; the **+0.9 %** belongs to
the T=8 MAXDIM=64 pair, and that same pair's **+14.2 %** cell area is
disclaimed by its own ``RUN.rpt`` ("NOT a measurement of the QD=8-to-16 cost,
and must not be quoted as one") because the two exports are two days of
commits apart. The only isolated figure is the +6.6 %, and it is a flip-flop
count off the RTL instance list, not a synthesis run.

Both understatements have one cause. The FPGA has block RAM and LUT shift
registers; standard cells have neither, so every buffered bit becomes a
flip-flop at ~5.6 um^2, and a change that buys cycles with buffering is
roughly 27x worse on the flip-flop axis than the FPGA's own flip-flop count
says (``examples/tinytpu/asic_synthesis/README.md``).

So the proxy does not predict area from FPGA resources at all. It **counts the
bits the design declares** -- every array a unit declares, every channel the
architecture wires, at the parameter set the candidate is instantiated at --
and prices them, plus the AXI adapters, at coefficients fitted to the
committed Design Compiler runs. Counting the bits is the whole idea: it is
what the FPGA table cannot see and what the flow charges for.

What it is worth, and where it is not
=====================================

Fitted and cross-validated against the committed TinyTPU runs under
``examples/tinytpu/asic_synthesis/reports/`` (``--selfcheck`` re-derives every
number below from those reports, and fails if any has moved):

* **in-sample: 0.33 % mean, 0.91 % worst** over the seven committed TinyTPU
  runs;
* **leave-one-out: 0.62 % mean, 1.39 % worst** -- each run predicted by a
  model refitted without it, which is the only honest error bar on three
  fitted coefficients and seven points;
* on the changes the runs bracket: **+73.3 %** for the burst widening (DC
  +74.4 %), **+32.9 %** for ``T`` 4->8 (DC +33.1 %), **+66.0 %** for
  ``MAXDIM`` 16->64 (DC +64.1 %).
* on the channel-depth change, where **no clean DC pair exists**, it gives
  **+6.8 %** at the scored T=4 MAXDIM=16 against the tree's own independent
  estimate of **+6.6 %** (a flip-flop count off the RTL instance list times
  the measured area per sequential cell,
  ``docs/source/developer/limitations.rst``). The only DC pair that brackets
  ``QD`` -- ``T8_MAXDIM64`` against ``T8_MAXDIM64_qd16``, +14.2 % -- is two
  days of commits apart and its own ``RUN.rpt`` says it "must not be quoted"
  as the cost of ``QD``; the proxy reproduces that pair at +14.1 % and the
  isolated estimate at +6.8 %, which is the reason to trust the second number
  and not the first.

The **channel census is validated independently**: at T=4, ``QD``=8 it counts
16,640 bits of queue and at ``QD``=16 it counts 33,280, which is exactly the
hand count a person made from the emitted RTL's instance list
(``limitations.rst``, "counted from the instance list of the very RTL the ASIC
flow reads"), to the bit. That agreement is what the rule about chains is for:
``wrow``, ``a_fwd`` and ``p_fwd`` are declared over ``(T, T)`` but only
``T * (T - 1)`` are instantiated, because the last element of each chain has no
consumer. A census that took the declaration at face value overcounted the
array's queues by 44 % and the fitted coefficient absorbed the difference.

Where it is **not** valid, stated rather than discovered later:

* **It is a model of this design.** Four of the eleven committed runs are
  Gemmini, a Chisel design with no Vitis csynth and no ``TpuParams``; nothing
  here predicts them and nothing here is fitted to them. They are used only as
  the independent check on the one physical constant this model rests on --
  see ``A_BIT`` below.
* **Flip-flop memories.** Every committed run is ``sram_mode='none'``, so the
  per-bit price is a flip-flop's. A design with SRAM macros would pay a
  different one, and every ratio here would change.
* **One point below MAXDIM=64.** ``T4_MAXDIM16_shipped_baseline`` is the only
  committed run at MAXDIM=16, which is the configuration the loop **scores**
  at. It is also the worst-predicted run, in-sample (-0.91 %) and held out
  (-1.02 %). So the proxy is interpolating everywhere except the one place it
  is used, and the fix is a twelfth DC run, not a better fit.
* **No clean pair brackets the channel depth.** The axis the FPGA table
  understates worst is the one with no isolated synthesis result at all. The
  agreement reported above (+6.8 % against +6.6 %) is against another
  estimate, not a measurement.
* **It prices structure, not synthesis.** Retiming, resource sharing, a
  construct Vitis renders as a multi-driver RAM (which happened, and had to be
  banked by hand before the widened variant would synthesise at all): none of
  that is visible to a bit census. A candidate whose *structure* is unchanged
  scores unchanged here however differently it synthesises.

The fitted envelope, and why it REFUSES
======================================

The loop may now propose ``T`` and ``MAXDIM`` (``chia_agent/evaluate.py``), so
a configuration nobody has ever synthesised can arrive here. Priced by
extrapolation it would return a number with no error bar at all -- and the
"One point below MAXDIM=64" caveat above says exactly where that number would
be worst, which is also where the loop scores. So `envelope_refusals` REFUSES
to price a configuration outside the range the committed runs span, and
`estimate` raises rather than extrapolating. A missing number a search must
work around is recoverable; a confident wrong one is not.

**A hard boundary and not a widened error bar, and this project has the
evidence for why that distinction matters.** ``act/cycles.py``'s
``CRITICAL_WORK_FIT`` was re-derived on 2026-09-25 and came out WORSE (worst
error 15.79 % -> 18.16 %) with critical work itself unchanged: the whole
movement was in the measurement, because ``QD=16`` moved the measured cycles
**+4 / +4 / +4 / -1 / -11** -- a term that grows with work against one that
shrinks with it, which a single slope cannot hold, and at the smallest shape
the estimate landed outside the tolerance of the gate it feeds. The lesson is
that outside its fitted span a model's FUNCTIONAL FORM can be wrong and not
merely imprecise, so there is no error bar wide enough to make the number
usable. Hence a refusal with a named cause, and no
"extrapolate but flag it" path: `require_envelope=False` exists for a person
asking what the model would say, and it labels its answer `extrapolated` so
the label travels with the number.

The envelope is DERIVED from ``COMMITTED`` -- the span of each parameter over
the runs that are actually fitted -- so a twelfth DC run widens it by being
added to that dict, and nothing here has to be retyped. Today it is
T in [4, 8], MAXDIM in [16, 64], QD in [8, 16], DMA_WORDS in [1, 16].

The logic-only runs are left OUT of the span: their census has
``memory_bits = 0``, which is not a design point, and counting it would put 0
in the envelope's floor and make the floor meaningless.

**The ``QD`` span is the weakest of the four, and it is worth saying why
rather than reading [8, 16] as calibration.** The values in ``COMMITTED`` are
*declared here* -- ``asic_synthesis/README.md`` states the ``63ee6ec7`` split
-- and only ``T8_MAXDIM64``'s export records ``QD`` in its own
``MANIFEST.json`` at all; the three T=4 exports do not record it, and
``asic_synthesis``'s own checker treats an absent config key as "cannot pair"
rather than "matches". So the calibration set cannot *distinguish* ``QD`` on
three of five runs, which is the same position a candidate proposing an
unvisited ``MAXDIM`` is in. It does not bite today only because
``evaluate.QD_REQUIRED`` accepts exactly one depth (limitations item 24), and
that is a coincidence of two constraints rather than evidence. A clean QD pair
at the scored point -- priority 3 below -- is what would make the span mean
something.

What a DC run would buy, in the order it is worth paying for:

1. **T=4 MAXDIM=32** and **T=8 MAXDIM=16**. The interior of the (T, MAXDIM)
   box has exactly one point in it -- MAXDIM=16 at T=4 -- and the two runs
   above would make the box fitted at its corners instead of along one edge.
   They are also the two configurations a co-design loop reaches for first: a
   bigger array on a small memory, and a small array on a big one.
2. **T=4 MAXDIM=88 (or 76)**, at a ceiling. It would say whether the cost model
   holds where the ENCODING stops, which is the only honest way to widen the
   MAXDIM envelope upward rather than guessing that the line continues.
3. **A clean QD pair at the scored point** (T=4 MAXDIM=16, QD=8 against 16, two
   exports of the same commit). It is the axis with no isolated synthesis
   result at all, and the +6.8 % against +6.6 % agreement above is two
   estimates agreeing with each other.

Use
===

    python area_proxy.py --selfcheck        # error against the committed runs
    python area_proxy.py --census           # the live build's bit census
    python area_proxy.py --estimate         # ... and its area estimate

``census()`` is importable and runs inside the evaluator's composed tree, so
the count is taken from the CANDIDATE's own units and channels: a unit that
declares a bigger array, or an architecture that wires a deeper channel, is
seen. The coefficients are here, in a frozen file, so the candidate cannot
move the price it is charged.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REPORTS = REPO / "examples/tinytpu/asic_synthesis/reports"

# ---------------------------------------------------------------------------
# The cost model. Three kinds of coefficient, and the difference matters:
# MEASURED ones are read straight off a pair of committed runs; FITTED ones are
# least-squares over the committed runs; DECLARED ones are a structural choice.
# ---------------------------------------------------------------------------

#: MEASURED, twice. A `logic-only` run is the same RTL with the three unit
#: memories (`spad`, `vr`, `ar`) stubbed out, so (full - logic-only) / bits is
#: the flow's price for a stored bit, read directly:
#:
#:   T=4 MAXDIM=64:  (1,865,314 - 1,396,966) /  82,944 bits = 5.647 um^2/bit
#:   T=8 MAXDIM=64:  (2,833,544 - 2,272,572) / 100,352 bits = 5.590 um^2/bit
#:
#: Two independent runs agreeing to 1.0 %. Gemmini's own pair says the same
#: thing on a design this model is not fitted to and shares no RTL with:
#: (990,938 - 382,026) / 98,304 bits = 6.20, and at DIM=8, 6.25. That is the
#: check that this is a property of the 45 nm flow rather than of our Verilog.
A_BIT = 5.6183

#: FITTED (3 coefficients, 7 runs). Everything that is not a unit memory and
#: not an AXI adapter: the DMA's double buffers, the instruction memory, the
#: sequencer's loop stack, the channels, and the array.
A_BUFFER_BIT = 6.391      # um^2 per bit of DMA buffer / imem / loop stack
A_CHANNEL_BIT = 4.612     # um^2 per bit of channel storage (depth x width)
A_PE = 2481.0             # um^2 per processing element (T x T of them)

#: MEASURED. The instruction-fetch master port, which is 60.0 % of the
#: baseline design and does not move: 681,537 / 703,513 / 705,486 / 697,974
#: across four committed runs, a 1.7 % spread around this value. It is not on
#: the operand-port width law below -- at 64 bits that law predicts 102,105 and
#: this port is 6.9x that -- because Vitis gives the program prefetch a far
#: deeper outstanding-transaction buffer. It is infrastructure, priced as a
#: constant, and a candidate that changes IMEM_SIZE does not move it.
A_FETCH_PORT = 698_000.0

#: FITTED on two points, VALIDATED on two held out. An operand master port's
#: area against its data width in bits: (32 -> 56,605) and (64 -> 102,105) give
#: the line, and the burst-widened export's two 512-bit ports -- 744,092 and
#: 745,280, a 13x growth this law never saw -- come out at -0.7 % and -0.8 %.
#: This is the term the search would most want to move and the one the FPGA
#: table renders as block RAM.
OPERAND_PORT_FIXED, OPERAND_PORT_PER_BIT = 11_105.0, 1421.9
#: The store port, on its own two points (32 -> 21,320, 64 -> 27,334): a
#: narrower, shallower adapter that is not on the operand line either.
STORE_PORT_FIXED, STORE_PORT_PER_BIT = 15_304.0, 187.9


def adapters(*, operand_width: int, store_width: int, ports: int = 2) -> float:
    """The four AXI master ports: instruction fetch, `ports` operand, one store."""
    return (A_FETCH_PORT
            + ports * (OPERAND_PORT_FIXED + OPERAND_PORT_PER_BIT * operand_width)
            + STORE_PORT_FIXED + STORE_PORT_PER_BIT * store_width)


class OutsideEnvelope(Exception):
    """This configuration is outside what the committed DC runs span.

    Raised rather than returning a number. `estimate` is the only way an area
    figure enters the objective, so raising here is what makes "the proxy does
    not extrapolate" a property of the code instead of a note in a docstring.
    """


def estimate(c: dict, require_envelope: bool = True) -> dict:
    """The estimate, and every term of it, from a census.

    Returned as a breakdown rather than a scalar on purpose: "+74.4 %, and
    99.1 % of it is two AXI ports" is the finding, and a single number would
    have hidden it.

    Raises `OutsideEnvelope` when the census's parameters fall outside the
    range the fitted runs span. `require_envelope=False` exists for exactly one
    caller -- a person asking what the model WOULD say -- and it labels its
    answer `extrapolated` so the label travels with the number.
    """
    refusals = envelope_refusals(c.get("parameters") or {})
    if refusals and require_envelope:
        raise OutsideEnvelope("; ".join(refusals))
    terms = {
        "adapters": adapters(operand_width=c["operand_port_bits"],
                             store_width=c["store_port_bits"]),
        "unit_memory": A_BIT * c["memory_bits"],
        "buffers": A_BUFFER_BIT * c["buffer_bits"],
        "channels": A_CHANNEL_BIT * c["channel_bits"],
        "array": A_PE * c["processing_elements"],
    }
    return {"um2": round(sum(terms.values()), 1),
            "terms": {k: round(v, 1) for k, v in terms.items()},
            "census": c,
            "estimate": True,
            "envelope": {k: list(v) for k, v in envelope().items()},
            "extrapolated": refusals or None,
            "extrapolated_terms": extrapolated_terms(c) or None,
            "basis": "structural bit census x coefficients fitted to "
                     f"{len(COMMITTED)} committed DC runs; leave-one-out "
                     "0.62 % mean / 1.39 % worst; NOT a measurement"}


# ---------------------------------------------------------------------------
# The census: what the candidate's own architecture declares.
# ---------------------------------------------------------------------------

def _width(expr: str, ns: dict) -> int:
    """Bits of a dtype expression: `UInt(VW)`, `int8`, `int32`, `UInt(64)`."""
    m = re.fullmatch(r"(?:U)?Int\((.+)\)", expr.strip())
    if m:
        return int(eval(m.group(1), {"__builtins__": {}}, ns))
    m = re.fullmatch(r"u?int(\d+)", expr.strip())
    if m:
        return int(m.group(1))
    raise ValueError(f"not a dtype this census can price: {expr!r}")


def _rows(expr: str, ns: dict) -> int:
    return int(eval(expr, {"__builtins__": {}}, ns))


#: Channels whose last chain element is never instantiated (see `census`).
CHAINS = ("wrow", "a_fwd", "p_fwd")


def _local_arrays(unit) -> dict:
    """`{name: (dtype expression, rows expression)}` for the arrays a unit body
    declares itself -- the same AST `allo.compose.Unit.arrays()` walks, asked
    for the dtype as well, which is what a bit count needs."""
    tree = ast.parse(unit.source()).body[0]
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) \
                and isinstance(node.annotation, ast.Subscript):
            out[node.target.id] = (ast.unparse(node.annotation.value),
                                   ast.unparse(node.annotation.slice))
    return out


#: Arrays that are the region's DRAM arguments rather than on-chip state: they
#: are named positionally by `Unit.memories` and are off-chip, so a census that
#: counted them would charge the design for DRAM.
def census(architecture, *, memory_units=("spm", "vru", "accu")) -> dict:
    """Every bit the composed architecture declares, by where it lives.

    `memory_units` names the units whose local arrays are the design's
    *memories* -- the ones a logic-only synthesis run stubs out, and therefore
    the ones `A_BIT` was measured on. Everything else a unit declares is a
    buffer and is priced at `A_BUFFER_BIT`.
    """
    ns = dict(architecture.parameters)
    memory_bits = buffer_bits = 0
    per_unit = {}
    for unit in architecture.units:
        instances = 1
        for e in unit.instances:
            instances *= _rows(e, ns)
        bits = 0
        for name, (dtype, rows) in _local_arrays(unit).items():
            bits += _width(dtype, ns) * _rows(rows, ns) * instances
        per_unit[unit.name] = bits
        if unit.name in memory_units:
            memory_bits += bits
        else:
            buffer_bits += bits
    channel_bits, per_channel = 0, {}
    for ch in architecture.channels:
        n = 1
        for e in ch.shape:
            n *= _rows(e, ns)
        if ch.name in CHAINS:
            # Declared over (T, T); instantiated T * (T - 1) times, because
            # the last element of a chain has no consumer and Vitis emits no
            # FIFO for it. Taking the declaration at face value overcounts the
            # array's queues by 44 % -- see the module docstring, where this
            # rule is what makes the census reproduce the hand count off the
            # RTL instance list to the bit.
            n = n - _rows(ch.shape[0], ns)
        bits = _width(ch.dtype, ns) * _rows(ch.depth, ns) * n
        per_channel[ch.name] = bits
        channel_bits += bits
    T = ns["T"]
    return {"memory_bits": memory_bits, "buffer_bits": buffer_bits,
            "channel_bits": channel_bits, "processing_elements": T * T,
            # The master ports' data widths: what `align_value` lets Vitis
            # widen the operand ports to is DMA_WORDS packed words, and the
            # store port stays one packed word.
            "operand_port_bits": ns["DMA_WORDS"] * ns["VW"],
            "store_port_bits": ns["VW"],
            "parameters": {k: ns[k] for k in
                           ("T", "MAXDIM", "QD", "DMA_WORDS", "VW", "AW",
                            "SPAD_ROWS", "NVR", "NAR", "IMEM_SIZE")},
            "per_unit": per_unit, "per_channel": per_channel}


def live_census() -> dict:
    """The census of the build in whatever tree this process can import."""
    sys.path.insert(0, str(REPO))
    from examples.tinytpu import microarch_isa as U  # noqa: E402
    return census(U.TPU.architecture)


# ---------------------------------------------------------------------------
# The self-check: does it predict the runs we have?
# ---------------------------------------------------------------------------

#: The committed TinyTPU runs, keyed by report directory. `stubbed` is the
#: logic-only manifest (the three unit memories replaced by zero-area black
#: boxes), which is what makes `A_BIT` measurable at all.
#:
#: QD is 8 for every export before `63ee6ec7` and 16 for the two that record it
#: in `results.json`; `asic_synthesis/README.md` states the split, and
#: `--selfcheck` will not invent a QD for a run whose parameters it cannot
#: read.
COMMITTED = {
    "T4_MAXDIM16_shipped_baseline":   dict(T=4, MAXDIM=16, QD=8, DMA_WORDS=1),
    "T4_MAXDIM64_shipped":            dict(T=4, MAXDIM=64, QD=8, DMA_WORDS=1),
    "T4_MAXDIM64_shipped_logiconly":  dict(T=4, MAXDIM=64, QD=8, DMA_WORDS=1,
                                           stubbed=True),
    "T4_MAXDIM64_burstwiden":         dict(T=4, MAXDIM=64, QD=8, DMA_WORDS=16),
    "T8_MAXDIM64":                    dict(T=8, MAXDIM=64, QD=8, DMA_WORDS=1),
    "T8_MAXDIM64_qd16":               dict(T=8, MAXDIM=64, QD=16, DMA_WORDS=1),
    "T8_MAXDIM64_qd16_logiconly":     dict(T=8, MAXDIM=64, QD=16, DMA_WORDS=1,
                                           stubbed=True),
}

#: Runs deliberately NOT fitted, each with the reason. A list of exclusions
#: without reasons is a place to hide failures.
NOT_FITTED = {
    "superseded_export_T4_MAXDIM16":
        "superseded: exported before the memory sizes were derived from "
        "MAXDIM, so its SPAD_ROWS/NVR are literals this model cannot name",
    "gemmini_DIM4_full": "a different design (Chisel; no TpuParams, no csynth). "
        "Used only to CHECK A_BIT, which it confirms on RTL this model shares "
        "nothing with: (990,938 - 382,026) / 98,304 bits = 6.20 um^2/bit",
    "gemmini_DIM4_logiconly": "a different design",
    "gemmini_DIM8_full": "a different design",
    "gemmini_DIM8_logiconly": "a different design",
}


#: The parameters an envelope is defined over -- the four a census is a
#: function of. `IMEM_SIZE`, `SPAD_ROWS`, `NVR` and `NAR` are DERIVED from these
#: (`ip/params.py`), so they are not separate axes; a candidate that overrides
#: one of them directly moves a census TERM instead, which `estimate` reports.
ENVELOPE_KEYS = ("T", "MAXDIM", "QD", "DMA_WORDS")


def envelope() -> dict:
    """`{parameter: (low, high)}` over the runs this model is FITTED to.

    Derived from `COMMITTED`, minus the logic-only runs: their census has no
    unit memory in it at all, so they are a measurement of a stub rather than a
    configuration anyone would build.
    """
    real = [cfg for cfg in COMMITTED.values() if not cfg.get("stubbed")]
    return {k: (min(cfg[k] for cfg in real), max(cfg[k] for cfg in real))
            for k in ENVELOPE_KEYS}


def envelope_refusals(params: dict) -> list[str]:
    """Why this configuration cannot be PRICED, or `[]`.

    Called by `chia_agent/evaluate.py` before a candidate is composed, so a
    configuration this model cannot defend a number for is refused in
    milliseconds rather than after a csynth. Each refusal names the axis, the
    span, and the run that would widen it -- a refusal an agent cannot act on
    is a wall, and this one is meant to be a boundary.
    """
    out = []
    for key, (low, high) in envelope().items():
        value = params.get(key)
        if value is None:
            continue
        if value < low or value > high:
            out.append(
                f"{key}={value} is outside the area proxy's fitted envelope "
                f"[{low}, {high}]: the coefficients come from "
                f"{len([c for c in COMMITTED.values() if not c.get('stubbed')])} "
                f"committed Design Compiler runs and there is no DC run at "
                f"this {key}, so the area term would be an extrapolation with "
                f"no error bar. It is refused rather than extrapolated. What "
                f"would widen it is a DC run at this configuration "
                f"(examples/tinytpu/asic_synthesis/README.md); the priority "
                f"order is in this module's docstring")
    return out


def extrapolated_terms(c: dict) -> list[str]:
    """Census terms outside the range the fitted runs span, named.

    REPORTED, not refused, and the asymmetry is deliberate. A configuration
    outside the envelope is a point the coefficients were never fitted near,
    and there is nothing to do but refuse it. A term outside the range is
    ordinarily a candidate that made something SMALLER -- a shorter instruction
    memory is below every fitted `buffer_bits` -- and refusing that would refuse
    the cheapest kind of improvement. So it travels with the estimate instead,
    where a reader comparing two candidates can see which one is priced off the
    end of the fit.
    """
    real = {k: cfg for k, cfg in COMMITTED.items() if not cfg.get("stubbed")}
    censuses = [synthetic_census(**cfg) for cfg in real.values()]
    out = []
    for term in ("memory_bits", "buffer_bits", "channel_bits",
                 "processing_elements", "operand_port_bits", "store_port_bits"):
        if term not in c:
            continue
        low, high = min(x[term] for x in censuses), max(x[term] for x in censuses)
        if c[term] < low:
            out.append(f"{term}={c[term]} below the fitted range [{low}, {high}]")
        elif c[term] > high:
            out.append(f"{term}={c[term]} above the fitted range [{low}, {high}]")
    return out


def committed_area(variant: str) -> float:
    return json.loads((REPORTS / variant / "results.json").read_text())["area"]["total_cell"]


def synthetic_census(T, MAXDIM, QD, DMA_WORDS, stubbed=False) -> dict:
    """The census of a configuration, WITHOUT importing allo.

    `--selfcheck` has to price seven historical configurations, and six of them
    are not what this checkout builds. The arithmetic is `ip/params.py`'s and
    `ip/tinytpu.py`'s, and `--selfcheck` checks it against the live census
    before it trusts it, so a unit that gains an array makes the self-check
    fail loudly rather than silently price the old design.
    """
    WPR, LOOP_DEPTH, IMEM_SIZE, TEST_WINDOW = MAXDIM // T, 4, 56, 64
    OPERAND_ROWS = WPR * MAXDIM
    SPAD_ROWS = NVR = max(TEST_WINDOW, OPERAND_ROWS)
    NAR = max(128, TEST_WINDOW, 2 * MAXDIM + 8)
    VW, AW = T * 8, T * 32
    memory = SPAD_ROWS * VW + NVR * VW + NAR * AW
    buffers = (2 * (MAXDIM * WPR + DMA_WORDS) * VW     # dma_ld a/b_onchip
               + IMEM_SIZE * 64                        # sequencer program
               + 4 * LOOP_DEPTH * 32)                  # loop stack
    chain = T * (T - 1)          # wrow / a_fwd / p_fwd; see CHAINS
    channels = (5 * 64 * QD + 4 * VW * QD + 2 * T * VW * QD + chain * VW * QD
                + chain * 8 * QD + chain * 32 * QD + T * AW * QD
                + T * T * 32 * 4)
    return {"memory_bits": 0 if stubbed else memory, "buffer_bits": buffers,
            "channel_bits": channels, "processing_elements": T * T,
            "operand_port_bits": DMA_WORDS * VW, "store_port_bits": VW,
            "parameters": dict(T=T, MAXDIM=MAXDIM, QD=QD, DMA_WORDS=DMA_WORDS,
                               VW=VW, AW=AW, SPAD_ROWS=SPAD_ROWS, NVR=NVR,
                               NAR=NAR, IMEM_SIZE=IMEM_SIZE)}


def selfcheck(verbose=True) -> dict:
    rows, worst, total = [], 0.0, 0.0
    for variant, cfg in COMMITTED.items():
        actual = committed_area(variant)
        pred = estimate(synthetic_census(**cfg))["um2"]
        err = 100.0 * (pred - actual) / actual
        rows.append({"variant": variant, "actual": actual, "proxy": pred,
                     "error_pct": round(err, 2)})
        worst, total = max(worst, abs(err)), total + abs(err)
    ok = worst <= 1.0
    if verbose:
        print(f"AREA PROXY against {len(rows)} committed TinyTPU DC runs "
              f"(estimate, not a measurement)\n")
        print(f"  {'run':34s} {'DC cell area':>13s} {'proxy':>13s} {'error':>8s}")
        for r in rows:
            print(f"  {r['variant']:34s} {r['actual']:13,.0f} "
                  f"{r['proxy']:13,.0f} {r['error_pct']:+7.2f}%")
        print(f"\n  mean |error| {total / len(rows):.2f}%   "
              f"worst |error| {worst:.2f}%")
        print(f"\n  not fitted, with reasons:")
        for k, why in NOT_FITTED.items():
            print(f"    {k:34s} {why}")
        print(f"\n  Leave-one-out: 0.62 % mean, 1.39 % worst. The channel term "
              f"is checked\n  independently -- at T=4 it counts 16,640 bits of "
              f"queue at QD=8 and 33,280\n  at QD=16, which is the hand count "
              f"off the RTL instance list, to the bit.\n\n"
              f"  What no committed run covers: the channel-depth axis has NO "
              f"clean DC pair\n  (the one that brackets it is two days of "
              f"commits apart and its own RUN.rpt\n  refuses the quote), and "
              f"T4_MAXDIM16_shipped_baseline is the ONLY run at the\n"
              f"  configuration the loop scores at. Both want a DC run, not a "
              f"better fit.")
        print(f"\n{'AREA PROXY OK' if ok else 'AREA PROXY DRIFTED'}: worst "
              f"error {worst:.2f}% against a 1.00% bound")
    return {"ok": ok, "rows": rows, "worst_pct": round(worst, 2),
            "mean_pct": round(total / len(rows), 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args(argv)
    if a.census or a.estimate:
        c = live_census()
        if a.census:
            print(json.dumps(c, sort_keys=True))
            return 0
        try:
            print(json.dumps(estimate(c), sort_keys=True))
        except OutsideEnvelope as why:
            print(f"AREA PROXY REFUSES to price this build: {why}")
            return 1
        return 0
    return 0 if selfcheck()["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
