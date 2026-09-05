# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The schedule result: what the scheduler decided, per kernel and per loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from ..options import SchedulerOptions
from .compiler import CompilerReport


class RegionKind(str, Enum):
    """The scheduling regime of a region."""

    CYCLIC = "cyclic"  # a pipelined loop (dcp.pipeline)
    ACYCLIC = "acyclic"  # a straight-line span (dcp.sequential)
    GUARD = "guard"  # a control select (dcp.select); carries no compute itself


@dataclass(frozen=True)
class ScheduledOp:
    """One scheduled operation inside a region."""

    kind: str  # operator mnemonic (addi/mulf/load/store/...)
    t: int  # start cycle within the region
    impl: str | None = None  # realization (device operator symbol / native)
    z: float | None = None  # SDC z-slack, when carried

    @classmethod
    def from_json(cls, d: dict) -> ScheduledOp:
        return cls(kind=d["kind"], t=d["t"], impl=d.get("impl"), z=d.get("z"))


@dataclass(frozen=True)
class RegionScheduleCost:
    """What composition needs of a region and no reader compares.

    ``drain`` is the terminal cycle: cycles after the last issue pulse the
    deepest output commits, so ``done`` rises one later. A span composes off
    this, not ``iteration_latency``, which may carry slack above the last
    commit."""

    drain: int | None = None

    @classmethod
    def from_json(cls, d: dict) -> RegionScheduleCost:
        return cls(drain=d.get("drain"))


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class RegionSchedule:
    """One scheduling region (a dcp.pipeline / dcp.sequential / dcp.select)."""

    kind: RegionKind
    order: int  # program order among the func's regions
    depth: int  # nesting depth among dcp regions (0 = outermost)
    container: bool  # nests another region (a loop / guard wrapper)
    ops: list[ScheduledOp] = field(default_factory=list)
    #: initiation interval. Cyclic only; None for a dynamic-trip wrapper.
    interval: int | None = None
    trip_count: int | None = None  # constant iteration count, when known
    latency: int | None = None  # the whole region's span (cycles)
    #: the depth of one iteration: the cycle by which every op has completed.
    iteration_latency: int | None = None
    #: composition quantities, apart from the latencies a reader compares.
    cost: RegionScheduleCost = field(default_factory=RegionScheduleCost)
    latency_is_bound: bool = False  # latency is an upper bound, not exact
    conditional: bool = False  # while-pipeline (dcp.condition) or a guard
    # The controller family that paces this region: `counted_static`,
    # `conditional`, `indeterminate` or `concurrent`.
    determinacy: str | None = None

    @classmethod
    def from_json(cls, d: dict) -> RegionSchedule:
        return cls(
            kind=RegionKind(d["kind"]),
            order=d["order"],
            depth=d["depth"],
            container=d["container"],
            ops=[ScheduledOp.from_json(o) for o in d["ops"]],
            interval=d.get("interval"),
            trip_count=d.get("trip_count"),
            latency=d.get("latency"),
            iteration_latency=d.get("iteration_latency"),
            cost=RegionScheduleCost.from_json(d["cost"]),
            latency_is_bound=d["latency_bound"],
            conditional=d["conditional"],
            determinacy=d.get("determinacy"),
        )

    @property
    def is_wrapper(self) -> bool:
        """A container region carrying no compute of its own (a residual outer
        loop around leaf regions): a derived nesting node, not a scheduling
        decision."""
        return self.container and not self.ops

    @property
    def is_leaf(self) -> bool:
        return not self.container

    def op(self, kind: str) -> ScheduledOp:
        """The first op of the given kind (raises ``StopIteration`` if none)."""
        return next(o for o in self.ops if o.kind == kind)

    def has(self, kind: str) -> bool:
        return any(o.kind == kind for o in self.ops)

    def last_t(self) -> int:
        """The latest start cycle among this region's ops."""
        return max(o.t for o in self.ops)


@dataclass(frozen=True)
class FuncSchedule:
    """The schedule of one kernel (an ``allo.dcp.module``)."""

    name: str
    regions: list[RegionSchedule] = field(default_factory=list)
    latency: int | None = None  # whole-func latency (cycles), when static
    latency_is_bound: bool = False
    # Composition class: `counted_static` (`latency` is an exact start->done
    # span), `indeterminate` (consumers gate on `done`), or `concurrent`
    # (children paced by back-pressure, so `latency` is a floor).
    determinacy: str | None = None

    @classmethod
    def from_json(cls, d: dict) -> FuncSchedule:
        return cls(
            name=d["name"],
            regions=[RegionSchedule.from_json(r) for r in d["regions"]],
            latency=d.get("latency"),
            latency_is_bound=d["latency_bound"],
            determinacy=d.get("determinacy"),
        )

    @property
    def latency_is_exact(self) -> bool:
        """Whether ``latency`` is an exact span the hardware must realize, and
        so a number a measured cycle count may be held to. A bounded, elastic or
        concurrent kernel publishes a figure that is deliberately not tight."""
        return (
            self.latency is not None
            and self.determinacy == "counted_static"
            and not self.latency_is_bound
        )

    def cyclic(self, *, wrappers: bool = False) -> list[RegionSchedule]:
        """This func's cyclic regions; pure sequential wrappers excluded unless
        ``wrappers=True``."""
        return [
            r
            for r in self.regions
            if r.kind is RegionKind.CYCLIC and (wrappers or not r.is_wrapper)
        ]


@dataclass(frozen=True)
class UnhonoredDirective:
    """A schedule directive the scheduler did not apply, and why. Only refusals
    are listed; a directive that lands leaves its mark on the region it shaped."""

    directive: str  # as the user spelled it (`pipeline`)
    where: str  # source anchor of the loop it was attached to
    reason: str  # one stable mnemonic rather than prose

    @classmethod
    def from_json(cls, d: dict) -> UnhonoredDirective:
        return cls(d["directive"], d["where"], d["reason"])


@dataclass(frozen=True)
class SweepPoint:
    """One probed point on the ``(period, span)`` curve a period sweep walks."""

    cycle_ns: float  # the operating period the probe was asked for
    achieved_ns: float  # the operating period the probe came back holding
    latency: int | None  # the top kernel's composed span at this period
    latency_is_bound: bool
    #: what the probed schedule costs in the device's currency; see
    #: :attr:`ScheduleResult.area`.
    area: int | None = None


@dataclass(frozen=True)
class ScheduleResult:
    """The whole-module schedule result: the schedule of every kernel."""

    funcs: list[FuncSchedule] = field(default_factory=list)
    #: directives the scheduler could not apply, in the order it met them.
    unhonored_directives: list[UnhonoredDirective] = field(default_factory=list)
    #: the compiler's account of itself, not a property of the design (see
    #: :class:`CompilerReport`).
    compiler: CompilerReport = field(default_factory=CompilerReport)
    #: the clock period the schedule holds (ns): the target, or the least period
    #: every device operator fits when the target was unreachable. Emission and
    #: QoR price against this.
    cycle_ns: float | None = None
    #: the ``(period, span)`` curve the period sweep probed before settling on
    #: this schedule, tightest last. Empty outside the period policies.
    sweep: tuple[SweepPoint, ...] = ()
    #: what the whole module's schedule costs in the device's own currency: the
    #: quantity the exact solver minimizes under the span, summed over every
    #: region and evaluated on the settled schedule. A model figure, not a
    #: synthesis estimate: it compares two schedules of one kernel, not two
    #: kernels.
    area: int | None = None

    @classmethod
    def from_json(
        cls, text: str | dict, options: SchedulerOptions | None = None
    ) -> ScheduleResult:
        """Parse the JSON schedule result the scheduler returns, either as the
        raw string or as an already-decoded object. ``options`` is what the
        scheduler was asked for, which only its caller knows."""
        d = json.loads(text) if isinstance(text, str) else text
        return cls(
            funcs=[FuncSchedule.from_json(f) for f in d["funcs"]],
            unhonored_directives=[
                UnhonoredDirective.from_json(u)
                for u in d.get("unhonored_directives", [])
            ],
            compiler=CompilerReport.from_json(d, options),
            cycle_ns=d.get("cycle_ns"),
            area=d.get("area"),
        )

    def func(self, suffix: str) -> FuncSchedule:
        """The sub-function whose name ends with ``suffix`` (kernels compose by
        calling sub-kernels, so results carry ``top.sub`` funcs)."""
        return next(f for f in self.funcs if f.name.endswith(suffix))

    def regions(
        self, kind: RegionKind | None = None, *, wrappers: bool = False
    ) -> list[RegionSchedule]:
        """Regions across all funcs, optionally filtered by kind. Pure
        sequential wrappers are excluded by default (they carry a derived II, not
        a scheduling decision); pass ``wrappers=True`` for the full nesting
        tree."""
        return [
            r
            for f in self.funcs
            for r in f.regions
            if (kind is None or r.kind is kind) and (wrappers or not r.is_wrapper)
        ]

    def cyclic(self, *, wrappers: bool = False) -> list[RegionSchedule]:
        return self.regions(RegionKind.CYCLIC, wrappers=wrappers)
