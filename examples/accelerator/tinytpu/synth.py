# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthesis-derived latency table for TinyTPU.

``microarch.py`` declares each unit's cycle model as a frozen pre-synthesis
guess::

    tpu.latency(mxu, ii=1, depth=20)

Those constants are what ``CompiledProgram.cycles()`` charges a program, so the
whole co-design objective rests on them. This module runs Vitis HLS
C-synthesis on the *same* composed schedule and rebuilds that table from the
measured report, which closes the loop: an ISA/microarchitecture edit changes
the hardware, synthesis re-measures it, and the compiler's cycle count moves
accordingly. Nothing here changes the ISA or the compiler interface -- only the
numbers the units declare about themselves.

Run ``python -m examples.accelerator.tinytpu.synth`` to synthesize and print the
derived table as JSON.

Calibration
-----------
The frontend charges an instruction ``depth + ii * trips``. Vitis reports, per
unit, the latency of one whole call plus the ``ii``/``depth``/trip count of each
loop inside it (loops that were outlined into ``<unit>_Pipeline_*`` modules
included). The two are reconciled so the model reproduces the measured call:

* ``ii``    -- the issue slots one trip occupies, i.e. the sum of the per-loop
  ``ii`` over the loops that iterate once per trip. A unit whose body is two
  sequential passes over the same trip range issues at ``ii=2``.
* ``depth`` -- everything the call costs that does not scale with ``trips``,
  taken as ``measured_call_latency - ii * trips`` so that
  ``depth + ii * trips`` is exactly the measured latency.

A data-dependent mover (``dma_load``/``dma_store``, whose trip count is a
runtime operand) has no measured call latency: Vitis reports ``undef``. Its
``depth`` is instead its loops' pipeline depth plus the call overhead measured
from the units in the same report that *are* statically bounded, rather than a
constant carried over from another design.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from allo.backend.vitis.report import ModuleReport, SynthReport, parse_report

from .isa import tpu
from .microarch import export_backend, tinytpu

#: The part the co-design checkpoint synthesizes against.
DEFAULT_PART = "xcu55c-fsvh2892-2L-e"

#: Display order for the units ``microarch.py`` ships with. The table itself is
#: built from whatever the ISA has *bound* (see :func:`bound_units`), so a
#: candidate that adds a unit gets that unit measured too -- and therefore
#: cannot lower its score by declaring an optimistic ``ISA.latency`` for it.
UNIT_ORDER = ("dma_load", "dma_store", "vload", "vstore", "vpu", "mxu")

#: Used only when a report carries no statically-bounded unit to measure the
#: per-call overhead from -- every real TinyTPU report has several.
CALL_OVERHEAD_FALLBACK = 2


@dataclass(frozen=True)
class UnitLatency:
    """One unit's measured cycle model, in the form ``ISA.latency`` declares."""

    ii: int
    depth: int
    trips: int | None  # the trip count synthesis saw; None when data-dependent
    call_cycles: int | None  # measured latency of one whole call

    def as_dict(self) -> dict:
        return {
            "ii": self.ii,
            "depth": self.depth,
            "trips": self.trips,
            "call_cycles": self.call_cycles,
        }


def bound_units(isa=tpu) -> dict[str, object]:
    """Every hardware unit the ISA has bound an instruction to, by unit name.

    Read from the bindings rather than a fixed list so the table follows the
    machine the candidate actually describes.
    """
    units: dict[str, object] = {}
    for instruction in isa._ops.values():  # pylint: disable=protected-access
        unit = getattr(instruction.spec, "unit", None)
        if unit is not None:
            units[unit.func_name] = unit
    order = {name: i for i, name in enumerate(UNIT_ORDER)}
    return dict(
        sorted(units.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0]))
    )


def _unit_modules(report: SynthReport, top: str, unit: str) -> list[ModuleReport]:
    """The unit's own module plus the ``_Pipeline_*`` modules Vitis split out.

    A pipelined loop is often outlined into its own module, which is where its
    ``ii``/``depth`` are then reported -- so the unit's loops are the union.
    """
    prefix = f"{top}_{unit}"
    return [
        module
        for name, module in report.modules.items()
        if name == prefix or name.startswith(f"{prefix}_Pipeline_")
    ]


def _unit_loops(report: SynthReport, top: str, unit: str) -> list:
    return [
        loop for module in _unit_modules(report, top, unit) for loop in module.loops
    ]


def _call_overhead(report: SynthReport, top: str) -> int:
    """Cycles a call costs on top of the loop it spends its time in.

    Measured only on units that carry their loops *directly* -- a unit whose
    loops Vitis outlined into ``_Pipeline_`` modules pays its own call overhead
    into each of those, so differencing it would count that overhead twice and
    overstate the residual. The median over the remaining units keeps one
    oddly-scheduled unit from skewing the result.
    """
    samples = []
    for unit in bound_units():
        own = report.modules.get(f"{top}_{unit}")
        if own is None or own.latency.worst_cycles is None or not own.loops:
            continue
        loop_cycles = [
            loop.latency_cycles for loop in own.loops if loop.latency_cycles is not None
        ]
        if loop_cycles:
            samples.append(own.latency.worst_cycles - sum(loop_cycles))
    if not samples:
        return CALL_OVERHEAD_FALLBACK
    return max(0, int(round(statistics.median(samples))))


def derive_latency_table(
    report: SynthReport, top: str | None = None
) -> dict[str, UnitLatency]:
    """Rebuild the ``(ii, depth)`` table for every unit from a csynth report."""
    top = top or report.top or tinytpu.func_name
    overhead = _call_overhead(report, top)
    table: dict[str, UnitLatency] = {}
    for unit in bound_units():
        own = report.modules.get(f"{top}_{unit}")
        if own is None:
            continue
        loops = _unit_loops(report, top, unit)
        if not loops:
            # A unit synthesized without a loop costs its call and nothing per
            # trip; charging ii=1 keeps the model monotone in trip count.
            call = own.latency.worst_cycles
            table[unit] = UnitLatency(1, max(0, (call or 1) - 1), 1, call)
            continue

        trip_counts = [lp.trip_count for lp in loops if lp.trip_count is not None]
        trips = max(trip_counts) if trip_counts else None
        # Loops that run once per trip contribute their ii to the issue rate;
        # a shorter inner loop contributes proportionally.
        issue = 0.0
        for loop in loops:
            ii = loop.ii if loop.ii is not None else 1
            if trips and loop.trip_count:
                issue += ii * (loop.trip_count / trips)
            else:
                issue += ii
        ii = max(1, int(round(issue)))

        call = own.latency.worst_cycles
        if call is not None and trips is not None:
            depth = call - ii * trips
        else:
            depth = sum(lp.depth or 0 for lp in loops) + overhead
        table[unit] = UnitLatency(ii, max(0, int(depth)), trips, call)
    return table


def apply_latency_table(table: dict[str, UnitLatency]) -> None:
    """Re-declare every unit's ``ISA.latency`` from the measured table.

    The ISA, the decoder, and the bindings are untouched: only the cycle model
    the compiler charges changes, which is what makes ``prog.cycles()`` a
    synthesis-grounded number rather than a hand-frozen one.
    """
    units = bound_units()
    for name, latency in table.items():
        unit = units.get(name)
        if unit is not None:
            tpu.latency(unit, ii=latency.ii, depth=latency.depth)


def synthesize(
    part: str = DEFAULT_PART,
    freq_mhz: float = 300.0,
    project: str | Path = "tinytpu_synth_prj",
) -> SynthReport:
    """Export the composed schedule to Vitis HLS and run C-synthesis on it."""
    mod = export_backend("vitis", part=part, freq_mhz=freq_mhz)
    mod.scaffold_project(str(project))
    mod.synth()
    return parse_report(mod.synth_report)


def report_summary(report: SynthReport) -> dict:
    """The PPA facts a run should record: area and clock, reported not scored."""
    timing = report.timing
    achieved = timing.estimated_clock_ns
    return {
        "part": report.part,
        "target_clock_ns": timing.target_clock_ns,
        "estimated_clock_ns": achieved,
        "fmax_mhz": round(1000.0 / achieved, 2) if achieved else None,
        "area": {
            "lut": report.resources.lut,
            "ff": report.resources.ff,
            "dsp": report.resources.dsp,
            "bram18k": report.resources.bram,
            "uram": report.resources.uram,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", default=DEFAULT_PART)
    parser.add_argument("--freq", type=float, default=300.0)
    parser.add_argument("--project", default="tinytpu_synth_prj")
    args = parser.parse_args()

    report = synthesize(args.part, args.freq, args.project)
    table = derive_latency_table(report)
    print(
        json.dumps(
            {
                "synthesis": report_summary(report),
                "latency_table": {n: l.as_dict() for n, l in table.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
