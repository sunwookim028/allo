# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the compile was asked for, split by which stage reads the knob."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PrepassOptions:
    """The IR rewrites run before the scheduler, deciding what problem it is
    handed rather than how it solves one. Not published in any report.

    Args:
        float_reassoc: rebalance float reduction chains into logarithmic trees.
            Not bit-exact.
        accumulators: rotate float reductions across this many accumulators,
            dropping their II to ``ceil(latency / accumulators)`` (0 = off).
        unroll_under_pipeline: fully unroll the loops nested inside a pipelined
            loop, so the nest pipelines at one II. ``False`` keeps them rolled and
            leaves the directive unhonored.
        perfectize: sink an imperfect nest's prologue/epilogue into the inner
            loop under a guard, fusing it into one pipeline. The scheduler
            handles imperfect nests without it.
        scalarize_threshold: keep arrays of at most this many elements in
            registers rather than a memory (0 = off).
    """

    float_reassoc: bool = True
    accumulators: int = 0
    unroll_under_pipeline: bool = True
    perfectize: bool = False
    scalarize_threshold: int = 16


#: `O` values the driver sweeps a period for, deciding the clock rather than
#: taking it as given.
PERIOD_POLICIES = frozenset({"freq", "wall", "area"})

#: The region objective a period policy solves each rung under, where it differs
#: from the knob's own name: `freq` and `wall` rank clocks by time and leave
#: every region minimizing span. `area` ranks on what its regions minimize, so
#: it keeps its own order.
REGION_ORDERS = {"freq": "cycles", "wall": "cycles"}


@dataclass(frozen=True)
class SchedulerOptions:  # pylint: disable=too-many-instance-attributes
    """What the scheduler itself was asked for.

    Every field is the effective value the solve ran under, the knob list
    ``RTL.set_scheduler_opt`` turns by field name.

    Args:
        scheduler: the solver settling the resource half of each problem.
            ``"heuristic"`` is the SDC simplex plus greedy placement; ``"exact"``
            is CP-SAT over the same problem.
        O: the optimization direction. ``"cycles"`` minimizes span, tie-breaking
            on area; ``"area"`` minimizes area, over the period as well as the
            schedule: candidate clocks are probed and the winner solved under a
            span leash no slower than that clock's own heuristic schedule (an
            explicit ``pipeline(ii=n)`` also caps II at ``n``). The requested
            clock is the reference that trade is measured against, so the
            winner is never faster than it nor larger than it costs; a
            ``wall_ns`` deadline replaces that reference. Both need
            ``scheduler="exact"``, the heuristic solving spans only. ``"freq"``
            makes the clock an output: periods below the requested one are
            probed, the tightest within ``span_tolerance`` solved, then the
            clock tightened to the realized critical path. ``"wall"`` minimizes
            span times period, probing both sides of the requested clock, which
            may come back slower than asked. Under all three the handle's
            ``freq_mhz`` follows the result.
        cycle_ns: the operating clock period in ns, from the handle's
            ``freq_mhz``; chains are cut to it less ``clock_margin``.
        clock_margin: fraction of the period withheld as timing headroom; chains
            are cut to ``(1 - clock_margin) * cycle_ns`` while the design is
            clocked at ``cycle_ns``.
        area_slack: span the area objective may pay beyond its leash, as a
            fraction of the heuristic schedule's span. Zero ships no slower than
            the heuristic.
        wall_ns: the deadline ``O="area"`` holds its period sweep to, in ns: a
            candidate clock is eligible only while span times achieved period
            stays within it, and the smallest eligible area wins. Zero leaves
            the sweep ranking on area times wall, which needs no deadline. Read
            only under ``O="area"``.
        span_tolerance: the cycle-count regression ``O="freq"`` may trade for a
            faster clock; a candidate is kept only while span stays within
            ``(1 + span_tolerance)`` of the span at the requested clock (or, with
            no composed span, while every region holds its II, iteration depth
            and drain to it). Zero pays no cycles for frequency.
        budget: what one exact solve may spend, in the solver's deterministic
            time units (roughly a second of one core each).
        workers: search workers per exact solve. The portfolio is interleaved, so
            the same budget buys more search and a budget-limited region can
            settle on a different schedule than at one worker.
        seed: the exact solver's random seed; shifts which equal-cost optimum a
            solve lands on.
        deterministic: whether workers advance in a fixed interleaved order, so
            two identical compiles emit identical RTL. Off, above one worker, they
            race, each held to ``budget / workers`` wall-clock seconds, so exact
            solves are not reproducible.
        resource_weights: multipliers on the per-resource scarcity prices, by
            name (``{"dsp": 0.25}`` prices DSPs at a quarter). Composes with the
            weight a device declares; unnamed resources keep 1.0.
        escalate: whether the heuristic scheduler hands a region to the exact
            solver when its own schedule is provably off (a placement gap the
            oracle could not retire, or a drain above the region's floor).
            Spends exact-solve time only where the compile-time certificate
            fails; most regions certify optimal and pay nothing. Read by the
            heuristic scheduler alone.
    """

    scheduler: str = "heuristic"
    O: str = "cycles"
    cycle_ns: float = 5.0
    clock_margin: float = 0.0
    area_slack: float = 0.0
    wall_ns: float = 0.0
    span_tolerance: float = 0.1
    budget: float = 30.0
    workers: int = 8
    seed: int = 0
    deterministic: bool = True
    resource_weights: dict[str, float] = field(default_factory=dict)
    escalate: bool = True
