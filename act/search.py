# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every nest the mapspace offers, lowered, priced, and the refusals counted."""

from dataclasses import dataclass, field

from act import mapspace, schedule
from act.nest import Refused, emitted_order, order


@dataclass(frozen=True)
class Problem:
    workload: object
    extents: dict

    @property
    def title(self):
        shape = "x".join(f"{self.extents[r]}" for r in self.workload.ranks)
        return f"{self.workload.name} {shape}"


@dataclass(frozen=True)
class Candidate:
    nest: tuple
    program: tuple
    priced: schedule.Priced

    @property
    def cost(self):
        return self.priced.cost

    @property
    def label(self):
        return emitted_order(self.nest)


@dataclass
class Census:
    counts: dict = field(default_factory=dict)
    any_counts: dict = field(default_factory=dict)
    examples: dict = field(default_factory=dict)

    def record(self, refusal, nest):
        self.counts[refusal.cause] = self.counts.get(refusal.cause, 0) + 1
        for cause in refusal.also:
            self.any_counts[cause] = self.any_counts.get(cause, 0) + 1
        self.examples.setdefault(refusal.cause, (order(nest), refusal.detail))

    @property
    def total(self):
        return sum(self.counts.values())

    def rows(self):
        return sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class Result:
    problem: Problem
    candidates: list
    census: Census
    considered: int

    @property
    def best(self):
        return self.candidates[0] if self.candidates else None

    def ranked(self, top=None):
        return self.candidates[:top]


def price(target, program):
    return schedule.Priced.of(schedule.run(target.steps(program)),
                              target.emits(program))


def search(problem, target, slots=2):
    census, found, considered = Census(), [], 0
    for nest in mapspace.nests(problem.extents,
                               target.intrinsics(problem.workload,
                                                 problem.extents),
                               slots):
        considered += 1
        try:
            program = target.lower(problem.workload, problem.extents, nest)
        except Refused as refusal:
            census.record(refusal, nest)
            continue
        found.append(Candidate(nest, program, price(target, program)))
    found.sort(key=lambda c: (c.cost, c.label))
    return Result(problem, found, census, considered)
