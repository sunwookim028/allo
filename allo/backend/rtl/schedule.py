# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDC scheduling driver. The result it returns lives in `reports`."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace

from ..base import run_pipeline
from ..._mlir.ir import Module
from ..._mlir.dialects.allo import run_sdc_scheduling

from .options import REGION_ORDERS, PrepassOptions, SchedulerOptions
from .reports.schedule import ScheduleResult, SweepPoint

RTL_PREPARE_PIPELINE = """
builtin.module(
grid-mapping,
fold-constant-calls,
canonicalize,
cse,
materialize-topology,
canonicalize,
cse,
convert-allo-to-func,
elide-dead-init,
func.func(convert-linalg-to-affine-loops,float-to-int),legalize-arith,canonicalize,cse,
outline-loose-processes)
"""

# --- driver ----------------------------------------------------------------


def run_schedule(
    top,
    module,
    options: SchedulerOptions,
    prepass: PrepassOptions,
    allocate: bool,
) -> ScheduleResult:
    """Schedule ``top`` and return the :class:`ScheduleResult`. ``module`` is
    rewritten in place, left holding the ``allo.dcp.*`` ops the schedule reifies
    into. Operator/device timing is read from the ``dcp.device`` /
    ``dcp.operator`` ops injected into ``module`` before this call.

    ``prepass`` shapes the IR the scheduler is handed, ``options`` is what the
    scheduler itself is asked for, and ``allocate`` lets an exact solve decide
    how many copies of each operator a region builds.
    """
    run_pipeline(module, RTL_PREPARE_PIPELINE)
    # The model period every period-dependent stage below reads: the operating
    # clock less the margin withheld. The cosim clock stays the operating one.
    if not 0.0 <= options.clock_margin < 1.0:
        raise ValueError(f"clock_margin must lie in [0, 1); got {options.clock_margin}")
    model_ns = options.cycle_ns * (1.0 - options.clock_margin)
    thr = (
        "tree-height-reduction{enable-fp="
        f"{'true' if prepass.float_reassoc else 'false'} "
        f"period-ns={model_ns}}}"
    )
    rotate = (
        f"rotate-reductions{{accumulators={int(prepass.accumulators)} "
        f"period-ns={model_ns}}}"
    )
    loops = (
        "loop-canonicalization{"
        f"unroll-under-pipeline={'true' if prepass.unroll_under_pipeline else 'false'} "
        f"perfectize={'true' if prepass.perfectize else 'false'}}}"
    )
    part = f"reconcile-array-directives{{top={top}}}"
    scalarize = f"scalarize-memory{{max-elements={prepass.scalarize_threshold}}}"
    # `raise-memory-reductions` runs twice: once before `{loops}` so a reduction
    # nested in a pipelined loop is on an iter_arg before unrolling, and again
    # after `fold-if-statements`, which turns a guarded `if c: M += x` into a
    # plain reduction (its guard folded into the loop bound or a select) the
    # first run could not see. `float-to-int` runs a second time here, after
    # if-conversion and reduction raising, to demote the `select` cones and
    # reduction iter_args the earlier run in RTL_PREPARE could not yet see.
    pipeline = (
        f"builtin.module(canonicalize,cse,func.func(raise-to-affine,cse,"
        f"raise-counted-while,raise-memory-reductions,{loops},"
        f"canonicalize,fold-if-statements,cse,raise-memory-reductions,"
        f"float-to-int,{scalarize},"
        f"{thr},{rotate},narrow-demanded-bits),drop-trivial-func,"
        f"{part},func.func(hoist-invariant-reads,assign-banks),canonicalize,cse,"
        f"func.func(expand-region-bounds),"
        f"legalize-arith{{expand-const-arith=true period-ns={model_ns} "
        f"area-fuse-gate={'true' if options.O == 'area' else 'false'}}},"
        f"canonicalize,cse)"
    )
    run_pipeline(module, pipeline)
    diagnostics: list[str] = []
    handler = module.context.attach_diagnostic_handler(
        lambda d: diagnostics.append(d.message) or True
    )
    try:
        result = run_sdc_scheduling(
            module,
            top,
            model_ns,
            options.scheduler,
            # A period policy that ranks clocks by time leaves its regions on
            # the cycles order; "area" keeps its own (see `REGION_ORDERS`).
            REGION_ORDERS.get(options.O, options.O),
            options.budget,
            allocate,
            options.workers,
            options.seed,
            options.deterministic,
            options.area_slack,
            options.escalate,
        )
    finally:
        handler.detach()
    if result is None:
        raise RuntimeError(
            "An error occurred during scheduling process:\n" + "\n".join(diagnostics)
        )
    return ScheduleResult.from_json(result, options)


# `sweep_freq`'s ladder: this many geometric rungs between the discovered device
# floor and the requested period, beyond the two endpoint probes.
_SWEEP_RUNGS = 8

# `_descending`'s ladder: this many geometric rungs between the aggressive
# anchor and the slowest period any device row is built for.
_DESCENDING_RUNGS = 10


def _anchor_ns(options: SchedulerOptions, floor_ns: float) -> float:
    """The aggressive end of a ladder: twice the device's register floor, or an
    8x faster clock where the device declares none. A derate lifts an unholdable
    ask, so what a probe there achieves is the tightest clock on offer."""
    if floor_ns <= 0:
        return options.cycle_ns / 8.0
    return 2.0 * floor_ns / (1.0 - options.clock_margin)


def _descending(lo: float, hi: float, probed: float) -> Iterator[float]:
    """The periods a sweep walks from ``hi`` down to ``lo``, skipping the clock
    it has already probed."""
    for k in range(_DESCENDING_RUNGS):
        period = hi * (lo / hi) ** (k / (_DESCENDING_RUNGS - 1))
        if abs(period - probed) >= 1e-9:
            yield period


def _region_vector(result) -> dict:
    """Every solved per-region quantity a span composes from, keyed stably
    across probes of one kernel. None of them depend on trip counts."""
    out = {}
    for f in result.funcs:
        for r in f.regions:
            for name, v in (
                ("ii", r.interval),
                ("len", r.iteration_latency),
                ("drain", r.cost.drain),
            ):
                if v is not None:
                    out[(f.name, r.order, name)] = v
    return out


def _probe(
    top,
    make_module: Callable[[], Module],
    options: SchedulerOptions,
    prepass: PrepassOptions,
    allocate: bool,
    period: float,
) -> tuple[SweepPoint, dict]:
    """One heuristic probe at ``period``, on a fresh copy of the pristine IR:
    the sweep point and the region vector its span composes from."""
    opts = replace(options, scheduler="heuristic", cycle_ns=period)
    result = run_schedule(top, make_module(), opts, prepass, allocate)
    fn = result.func(top)
    point = SweepPoint(
        cycle_ns=period,
        achieved_ns=result.cycle_ns / (1.0 - options.clock_margin),
        latency=fn.latency,
        latency_is_bound=fn.latency_is_bound,
        area=result.area,
    )
    return point, _region_vector(result)


def _dedup(points: list[SweepPoint]) -> list[SweepPoint]:
    """Candidates that derate onto the same achieved period are one design;
    keep the laxest ask of each."""
    seen: set[float] = set()
    curve: list[SweepPoint] = []
    for p in sorted(points, key=lambda p: p.cycle_ns, reverse=True):
        if (key := round(p.achieved_ns, 3)) not in seen:
            seen.add(key)
            curve.append(p)
    return curve


def _solve_at(
    top,
    make_module: Callable[[], Module],
    options: SchedulerOptions,
    prepass: PrepassOptions,
    allocate: bool,
    period: float,
    curve: list[SweepPoint],
) -> tuple[Module, ScheduleResult]:
    """Solve once at the winning period under the caller's own scheduler
    settings, publishing the probed curve."""
    module = make_module()
    result = run_schedule(
        top, module, replace(options, cycle_ns=period), prepass, allocate
    )
    return module, replace(result, sweep=tuple(curve))


def sweep_freq(
    top,
    make_module: Callable[[], Module],
    options: SchedulerOptions,
    prepass: PrepassOptions,
    allocate: bool,
    floor_ns: float,
) -> tuple[Module, ScheduleResult]:
    """Minimize the operating period under ``O="freq"``: probe candidates below
    ``options.cycle_ns`` with the heuristic scheduler, keep those whose span
    stays within ``span_tolerance`` of the span at the requested clock, and
    solve once at the tightest survivor under the caller's own scheduler
    settings. A kernel with no composed span is leashed per region instead:
    every solved quantity the span composition is monotone in holds the same
    tolerance, which bounds the span at any trip counts. Every probe recompiles
    from pristine IR (``make_module``), since the legalized op set depends on
    the period. ``floor_ns`` is the device's register floor, which bounds how
    deep the probes reach. Returns the scheduled module and its result, with
    the probed curve published as ``ScheduleResult.sweep``."""
    if options.span_tolerance < 0.0:
        raise ValueError(
            f"span_tolerance must be non-negative; got {options.span_tolerance}"
        )
    vectors: dict[float, dict] = {}

    def probe(period: float) -> SweepPoint:
        point, vectors[period] = _probe(
            top, make_module, options, prepass, allocate, period
        )
        return point

    asked = probe(options.cycle_ns)
    anchor = _anchor_ns(options, floor_ns)
    points = [asked]
    lo = options.cycle_ns
    if anchor < options.cycle_ns:
        floor = probe(anchor)
        points.append(floor)
        lo = floor.achieved_ns
    if lo < options.cycle_ns:
        ratio = options.cycle_ns / lo
        points += [
            probe(lo * ratio ** (k / (_SWEEP_RUNGS + 1)))
            for k in range(1, _SWEEP_RUNGS + 1)
        ]
    curve = _dedup(points)
    # A bounded span compares as its worst case; `asked` always qualifies, so
    # there is a winner. With no span to compare, the per-region vector is
    # leashed instead.
    tol = 1.0 + options.span_tolerance
    if asked.latency is not None:
        leash = asked.latency * tol
        eligible = [p for p in curve if p.latency is not None and p.latency <= leash]
    else:
        ref = vectors[asked.cycle_ns]
        eligible = [
            p
            for p in curve
            if vectors[p.cycle_ns].keys() == ref.keys()
            and all(vectors[p.cycle_ns][k] <= v * tol for k, v in ref.items())
        ]
    winner = min(eligible, key=lambda p: (p.achieved_ns, p.latency or 0))
    return _solve_at(
        top, make_module, options, prepass, allocate, winner.cycle_ns, curve
    )


def sweep_wall(
    top,
    make_module: Callable[[], Module],
    options: SchedulerOptions,
    prepass: PrepassOptions,
    allocate: bool,
    floor_ns: float,
    cap_ns: float,
) -> tuple[Module, ScheduleResult]:
    """Minimize wall time under ``O="wall"``: probe candidate periods on both
    sides of the requested clock with the heuristic scheduler, take the one
    whose span times achieved period is least, and solve once there under the
    caller's own scheduler settings. The requested clock is a reference, not a
    bound, so a winning clock slower than asked is possible. ``cap_ns`` tops
    the ladder at the slowest period any device row is built for; ``floor_ns``
    is the register floor the aggressive anchor stands on. A kernel with no
    composed span has no wall time to compare and is refused. Returns the
    scheduled module and its result, with the probed curve published as
    ``ScheduleResult.sweep``."""

    def probe(period: float) -> SweepPoint:
        return _probe(top, make_module, options, prepass, allocate, period)[0]

    asked = probe(options.cycle_ns)
    if asked.latency is None:
        raise RuntimeError(
            f"O='wall' compares span times period, and '{top}' publishes no "
            "span at the requested clock; add allo.assume trip bounds or "
            "choose a different objective"
        )
    points = [asked]
    best = asked.latency * asked.achieved_ns
    lo = _anchor_ns(options, floor_ns)
    hi = max(cap_ns, options.cycle_ns, lo)
    # Walked with the incumbent: the laxest candidate's span bounds every
    # candidate's from below (feasible sets only grow with the period), so a
    # period that laxest span cannot win at is skipped unprobed.
    floor_span = None
    for period in _descending(lo, hi, options.cycle_ns):
        if floor_span is not None and floor_span * period >= best:
            continue
        p = probe(period)
        assert p.latency is not None, (
            "a span is a property of the kernel's trip structure, which no "
            "probed period changes"
        )
        floor_span = floor_span if floor_span is not None else p.latency
        best = min(best, p.latency * p.achieved_ns)
        points.append(p)
    curve = _dedup(points)
    # Fewer cycles breaks a wall tie: the shorter schedule spends less on
    # pipeline registers.
    winner = min(curve, key=lambda p: (p.latency * p.achieved_ns, p.latency))
    return _solve_at(
        top, make_module, options, prepass, allocate, winner.cycle_ns, curve
    )


def sweep_area(
    top,
    make_module: Callable[[], Module],
    options: SchedulerOptions,
    prepass: PrepassOptions,
    allocate: bool,
    floor_ns: float,
    cap_ns: float,
) -> tuple[Module, ScheduleResult]:
    """Minimize area under ``O="area"`` with the period free: probe candidate
    clocks with the heuristic scheduler, rank them by cost, and solve once at
    the winner under the caller's own scheduler settings, where the area
    objective proper runs. A slower clock chains deeper, so it breaks fewer
    chains and spends fewer delay registers; the rank stops it paying unbounded
    time for that.

    The rank depends on what area is weighed against:

    - ``wall_ns`` set: a candidate is eligible while span times achieved period
      holds the deadline, and the least-area eligible one wins. No eligible
      candidate, or no composed span, is refused rather than silently overrun.
    - No deadline, span present: only clocks no faster than the requested one
      and costing no more area are probed, and the winner minimizes area times
      wall, so this never returns more area than the requested clock would.
    - No deadline, no span: nothing may be traded, so a candidate qualifies only
      while every solved per-region quantity costs no more time than at the
      requested clock; this degrades to scheduling at the clock asked for.

    Without a deadline the winning clock is confirmed by solving beside the
    requested one and shipping the cheaper, since the probe ranks the heuristic
    schedule while the objective's own solve ships. The direction is then
    monotone: it never builds more than it would have without the sweep.

    ``cap_ns`` tops the ladder at the slowest period any device row is built
    for; ``floor_ns`` is the register floor the aggressive anchor stands on,
    reached only under a deadline. Returns the scheduled module and its result,
    with the probed curve published as ``ScheduleResult.sweep``."""
    if options.wall_ns < 0.0:
        raise ValueError(f"wall_ns must be non-negative; got {options.wall_ns}")
    vectors: dict[float, dict] = {}

    def probe(period: float) -> SweepPoint:
        point, vectors[period] = _probe(
            top, make_module, options, prepass, allocate, period
        )
        return point

    asked = probe(options.cycle_ns)
    if options.wall_ns and asked.latency is None:
        raise RuntimeError(
            f"O='area' holds wall_ns against span times period, and '{top}' "
            "publishes no span at the requested clock; add allo.assume trip "
            "bounds, or leave wall_ns unset to trade against the requested "
            "clock instead"
        )
    points = [asked]
    # A deadline lets the winner sit at any clock. Without one the requested
    # clock floors the walk, a faster rung being a frequency purchase this
    # direction does not make.
    lo = _anchor_ns(options, floor_ns) if options.wall_ns else options.cycle_ns
    hi = max(cap_ns, options.cycle_ns, lo)
    # Under a deadline the laxest candidate's span bounds every candidate's from
    # below (feasible sets only grow with the period), so a period no schedule
    # can meet the deadline at is skipped unprobed.
    floor_span = None
    for period in _descending(lo, hi, options.cycle_ns):
        if options.wall_ns and floor_span and floor_span * period > options.wall_ns:
            continue
        p = probe(period)
        if floor_span is None:
            floor_span = p.latency
        points.append(p)
    curve = _dedup(points)
    assert all(
        p.area is not None for p in curve
    ), "every schedule carries a modeled area, whatever solved it"

    def wall(p: SweepPoint) -> float:
        return p.latency * p.achieved_ns

    if options.wall_ns:
        eligible = [p for p in curve if p.latency and wall(p) <= options.wall_ns]
        if not eligible:
            best = min((p for p in curve if p.latency), key=wall, default=None)
            raise RuntimeError(
                f"O='area' found no clock at which '{top}' finishes within "
                f"{options.wall_ns:g} ns: the best of {len(curve)} probed "
                f"periods runs {best.latency} cycles at {best.achieved_ns:g} ns "
                f"= {wall(best):g} ns; raise wall_ns, or leave it unset to trade "
                "against the requested clock instead"
            )
        return _solve_at(
            top,
            make_module,
            options,
            prepass,
            allocate,
            min(eligible, key=lambda p: (p.area, wall(p))).cycle_ns,
            curve,
        )
    # Nothing faster than asked was probed; a candidate must also cost no more
    # than asked, so the rank never proposes a bigger design than the one the
    # requested clock would have built.
    slower = [
        p
        for p in curve
        if p.achieved_ns >= asked.achieved_ns - 1e-9 and p.area <= asked.area
    ]
    if asked.latency is not None:
        # Area times wall, not area times span: a span alone does not compare
        # across periods. Wall breaks the tie, so an equal-area slower clock is
        # never taken.
        winner = min(slower, key=lambda p: (p.area * wall(p), wall(p)))
    else:
        # No span, so no wall to price an area saving against: take only the
        # clocks that cost no region any time either.
        ref = vectors[asked.cycle_ns]
        free = [
            p
            for p in slower
            if vectors[p.cycle_ns].keys() == ref.keys()
            and all(
                vectors[p.cycle_ns][k] * p.achieved_ns <= v * asked.achieved_ns
                for k, v in ref.items()
            )
        ]
        winner = min(free, key=lambda p: (p.area, p.achieved_ns))
    solved = _solve_at(
        top, make_module, options, prepass, allocate, winner.cycle_ns, curve
    )
    if abs(winner.cycle_ns - options.cycle_ns) < 1e-9:
        return solved

    def cost(outcome: tuple[Module, ScheduleResult]) -> tuple[int, float]:
        """What the solver says the schedule costs, wall breaking a tie."""
        result = outcome[1]
        span = result.func(top).latency
        return (result.area, span * result.cycle_ns if span else 0.0)

    asked_solve = _solve_at(
        top, make_module, options, prepass, allocate, options.cycle_ns, curve
    )
    return min(solved, asked_solve, key=cost)
