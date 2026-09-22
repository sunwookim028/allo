# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The propositions the ASAP placement is supposed to satisfy.

ACT's `epoch.schedule` is an order-*preserving* greedy ASAP pass, so its
guarantee is pointwise minimality over the feasible set, not invariance under
reordering the stream. `act.schedule.run` is held to the same claims, against
two solvers that do not share its algorithm: a least-fixpoint iteration and,
for the smallest cases, brute force over every start vector.
"""

import itertools
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from act.schedule import Region, Step, StepError, predecessors, run  # noqa: E402

UNITS = ("alpha", "beta", "gamma")


def step(index, units, work, reads=(), writes=()):
    return Step(index=index,
                loads=tuple((u, work) for u in units),
                reads=tuple(Region("m", r, 1) for r in reads),
                writes=tuple(Region("m", w, 1) for w in writes))


def chain(length):
    return [step(i, (UNITS[i % len(UNITS)],), 1 + i % 4,
                 reads=(i - 1,) if i else (), writes=(i,))
            for i in range(length)]


def random_steps(rng, count, cells=4):
    return [step(i, tuple(rng.sample(UNITS, rng.randint(1, 2))),
                 rng.randint(1, 5),
                 reads=tuple(rng.sample(range(cells), rng.randint(0, 2))),
                 writes=(rng.randrange(cells),))
            for i in range(count)]


def constraints(steps):
    ordered = sorted(steps, key=lambda s: s.index)
    deps = predecessors(ordered)
    out = []
    for step_ in ordered:
        for earlier in deps[step_.index]:
            out.append((step_.index, earlier,
                        next(s for s in ordered
                             if s.index == earlier).duration))
    for unit in sorted({u for s in ordered for u in s.units}):
        on = [s for s in ordered if unit in s.units]
        for earlier, later in zip(on, on[1:]):
            out.append((later.index, earlier.index, dict(earlier.loads)[unit]))
    return out


def feasible(steps, start):
    return (all(t >= 0 for t in start.values())
            and all(start[late] >= start[early] + delay
                    for late, early, delay in constraints(steps)))


def least_fixpoint(steps):
    start = {s.index: 0 for s in steps}
    edges = constraints(steps)
    for _ in range(len(start) + 1):
        moved = False
        for late, early, delay in reversed(edges):
            want = start[early] + delay
            if start[late] < want:
                start[late] = want
                moved = True
        if not moved:
            return start
    raise AssertionError("the constraint system did not converge")


def brute_force_minimum(steps, horizon):
    indices = sorted(s.index for s in steps)
    best = None
    for point in itertools.product(range(horizon), repeat=len(indices)):
        start = dict(zip(indices, point))
        if not feasible(steps, start):
            continue
        best = start if best is None else {
            i: min(best[i], start[i]) for i in indices}
    return best


@pytest.mark.parametrize("length", [1, 2, 5, 9])
def test_degenerate_theorem(length):
    steps = [step(i, ("alpha",), 1) for i in range(length)]
    plan = run(steps)
    assert [plan.start[i] for i in range(length)] == list(range(length))
    assert plan.makespan == length


@pytest.mark.parametrize("length", [1, 4, 8, 16])
def test_respects_dependences_and_never_overbooks(length):
    steps = chain(length)
    assert feasible(steps, run(steps).start)


def test_random_programs_are_feasible():
    rng = random.Random(7)
    for _ in range(300):
        steps = random_steps(rng, rng.randint(1, 12))
        assert feasible(steps, run(steps).start)


def test_pointwise_minimal_against_a_least_fixpoint_solver():
    rng = random.Random(11)
    for _ in range(500):
        steps = random_steps(rng, rng.randint(1, 10))
        assert run(steps).start == least_fixpoint(steps)


def test_pointwise_minimal_against_brute_force():
    rng = random.Random(23)
    for _ in range(40):
        steps = random_steps(rng, rng.randint(1, 3), cells=2)
        plan = run(steps)
        horizon = max(plan.start.values()) + 2
        assert plan.start == brute_force_minimum(steps, horizon)


def test_makespan_is_bounded_below_by_both_bounds():
    rng = random.Random(13)
    for _ in range(300):
        steps = random_steps(rng, rng.randint(1, 12))
        plan = run(steps)
        assert plan.makespan >= plan.critical_path
        assert plan.makespan >= max(plan.unit_load.values())


def test_presentation_order_does_not_change_the_schedule():
    steps = chain(7)
    reference = run(steps)
    for shuffled in itertools.permutations(steps):
        plan = run(list(shuffled))
        assert plan.start == reference.start
        assert plan.unit_load == reference.unit_load


def test_repeated_index_is_refused():
    with pytest.raises(StepError):
        run([step(0, ("alpha",), 1), step(0, ("beta",), 1)])
