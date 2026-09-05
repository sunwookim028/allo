# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The whole compile, as one object."""

from __future__ import annotations

from dataclasses import dataclass

from ..interface import Interfaces
from .compiler import CompilerReport
from .microarch import MicroarchReport, FuncUarch, RegionUarch
from .schedule import ScheduleResult, FuncSchedule, RegionSchedule


@dataclass(frozen=True)
class CompileReport:
    """What the scheduler decided, what the emitter built, and the boundary it
    built it behind."""

    schedule: ScheduleResult
    microarch: MicroarchReport
    interfaces: Interfaces

    @property
    def compiler(self) -> CompilerReport:
        """The compiler's account of itself, which the schedule stage owns."""
        return self.schedule.compiler

    def func(self, suffix: str) -> tuple[FuncSchedule, FuncUarch]:
        """One kernel's schedule and its allocation, by name suffix."""
        return self.schedule.func(suffix), self.microarch.func(suffix)

    def region(self, func: str, order: int) -> tuple[RegionSchedule, RegionUarch]:
        """The one join: a region's schedule and its allocation, keyed by
        (func, program order)."""
        fs, fu = self.func(func)
        return next(r for r in fs.regions if r.order == order), fu.region(order)
