# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent units, each sequential and in program order: when does it finish."""

from dataclasses import dataclass


class StepError(Exception):
    """A step list that is not a program order."""


@dataclass(frozen=True)
class Region:
    space: str
    start: int
    length: int

    def overlaps(self, other):
        return (self.space == other.space
                and self.start < other.start + other.length
                and other.start < self.start + self.length)


@dataclass(frozen=True)
class Step:
    index: int
    loads: tuple
    reads: tuple = ()
    writes: tuple = ()

    @property
    def duration(self):
        return max((w for _, w in self.loads), default=0)

    @property
    def units(self):
        return tuple(u for u, _ in self.loads)

    def conflicts(self, other):
        for mine in self.writes:
            for theirs in other.reads + other.writes:
                if mine.overlaps(theirs):
                    return True
        for mine in self.reads:
            for theirs in other.writes:
                if mine.overlaps(theirs):
                    return True
        return False


@dataclass(frozen=True)
class Schedule:
    start: dict
    finish: dict
    unit_load: dict
    predecessors: dict

    @property
    def makespan(self):
        return max(self.finish.values(), default=0)

    @property
    def bottleneck(self):
        if not self.unit_load:
            return None, 0
        unit = max(sorted(self.unit_load), key=lambda u: self.unit_load[u])
        return unit, self.unit_load[unit]

    @property
    def critical_path(self):
        through = {}
        for index in sorted(self.finish):
            span = self.finish[index] - self.start[index]
            through[index] = span + max(
                (through[p] for p in self.predecessors[index]), default=0)
        return max(through.values(), default=0)


def in_program_order(steps):
    ordered = tuple(sorted(steps, key=lambda s: s.index))
    if len({s.index for s in ordered}) != len(ordered):
        raise StepError("two steps share an index")
    return ordered


def predecessors(steps):
    ordered = in_program_order(steps)
    return {step.index: frozenset(
        earlier.index for earlier in ordered[:position]
        if step.conflicts(earlier))
        for position, step in enumerate(ordered)}


def run(steps):
    ordered = in_program_order(steps)
    deps = predecessors(ordered)
    start, finish, busy, load = {}, {}, {}, {}
    for step in ordered:
        ready = max((finish[p] for p in deps[step.index]), default=0)
        begin = max([ready] + [busy.get(u, 0) for u in step.units])
        start[step.index] = begin
        finish[step.index] = begin + step.duration
        for unit, work in step.loads:
            busy[unit] = begin + work
            load[unit] = load.get(unit, 0) + work
    return Schedule(start, finish, dict(sorted(load.items())), deps)


@dataclass(frozen=True, order=True)
class Priced:
    cost: tuple
    schedule: object = None

    @staticmethod
    def of(schedule, emits):
        return Priced((schedule.makespan, emits), schedule)
