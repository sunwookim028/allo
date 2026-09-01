# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Searching the mapspace ourselves (refactor target 4).

Step 7 closed the loop *around* an external mapper: export the operation and the
machine, let Timeloop choose, read the answer back. This is the other end of the
same seam — the same site mapped with no mapper at all. What changes is not the
object (a ``Mapping`` either way, refused by the same obligations and lowered by
the same walk) but where its **score** comes from.

Timeloop ranks a nest by an analytical access count over an architecture YAML.
Here a nest is ranked by **compiling it**: ``mapping.assemble`` turns it into a
running program, the allocator answers whether it fits — and spills it when it
nearly does — the run decomposition answers how many instructions its transfers
actually are, and σ answers how long the result takes. Three things follow that
an access count cannot express, and all three are visible in the tests:

- **Capacity is the allocator's answer, per candidate.** A nest whose working set
  does not fit is removed by the thing that would have had to place it, not by a
  capacity model standing next to it — and one that *nearly* fits is priced with
  its spill traffic included, which is a number no analytical model has.
- **The instruction count is the real one.** A tile transfer is one instruction
  per maximal contiguous run under the layouts at both ends (``_run_offsets``), so
  two nests moving identical *words* can cost different numbers of instructions.
- **The objective is time, not traffic.** So the search will happily take a nest
  that moves *more* data if it finishes sooner — the spatial split an access count
  is indifferent to, or actively penalizes.

**The enumeration.** The innermost level is pinned to the instruction's intrinsic
tile — step 7 already exports that as a search constraint — and what is left of
each rank is distributed over the remaining slots: one temporal and (when the
caller declares instances) one spatial slot per storage level. Then every
permutation of the temporal loops at each level. That is the "factorizations ×
permutations × spatial/temporal splits" of the plan, exactly.

**What the enumeration deliberately does not walk**, since a search that quietly
bounds itself reads as a search that covered everything:

- **Bypass.** Timeloop's mapspace includes the bypass mask; this one fixes it at
  "every level keeps everything". With two levels the axis is degenerate — level 0
  is program I/O and cannot bypass, and a level-1 bypass puts the compute's operand
  in the wrong buffer, which ``Mapping.check`` refuses — so it buys nothing until a
  machine has three.
- **Imperfect factorizations.** A rank's factors must multiply to its extent
  *exactly*, because that is what constraint 1 says and what the walk emits. A tile
  that does not divide D would need a ragged last iteration, which nothing in this
  vocabulary can say. The refusal names it rather than rounding.
- **Spatial loop order**, and the order of the intrinsic's own loops: both only
  rename instances or are stripped by ``_split_body``, so exactly one of each class
  is enumerated.

**``instances`` is the one number the machine does not declare.** It is the same
gap ``to_timeloop`` has to list rather than export (Timeloop's ``meshX``/
``meshY``), and it turns up here as an *unbounded axis* instead of a missing
field: told it may fan out, the search fans out as far as it is allowed, because
nothing in the ISA says how far that is. So the caller states it and the default
is 1 — the only value that is not an invention, and the one the rest of the
frontend already assumes when it gives an instruction one unit per mnemonic.

**Two things the plan expected this step to need, and it did not.**
*Group-scope schedule params*: within one nest every compute step performs the
identical intrinsic tile, so the per-instruction chooser already returns one
configuration for the whole run — and where a field genuinely has to be shared
across instructions, the machine says so with ``ISA.configures``, whose setter is
an *emitted instruction* and therefore already in what pricing measures. Ragged
tiles would need a group scope, and exact factorization excludes them.
*Symbolic checking*: every candidate is a concrete program, so
``epoch.region_elements`` stays a finite set that decides its own questions
exactly. What would need Presburger is proving a *family* of nests correct — "for
every M divisible by 4" — which is a different question from choosing among
concrete ones, and is not what enumeration asks.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

from .errors import AllocationError, AssemblyError
from .mapping import _DERIVE, _derive, Loop, Mapping, assemble


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _splits(n: int, parts: int) -> list[tuple]:
    """Every ordered factorization of ``n`` into ``parts`` factors."""
    if parts == 1:
        return [(n,)]
    return [(d,) + rest for d in _divisors(n) for rest in _splits(n // d, parts - 1)]


def _prod(xs) -> int:
    out = 1
    for x in xs:
        out *= x
    return out


def mapspace(binding, ranks: dict, *, instances: int = 1):
    """Every nest this machine could perform for this problem instance.

    Yields ``Mapping``s, and only the *shape* of the machine is consulted —
    ``binding.levels`` and ``binding.body`` — so enumeration needs no dataspaces
    and no source op. Pricing does; that is the next function.

    The intrinsic is the innermost loops of every nest, so what is enumerated is
    the residual ``extent // intrinsic`` of each rank, distributed over one
    temporal slot per level (plus one spatial slot per level once ``instances``
    allows more than one), times a permutation of each level's temporal loops.
    Nests that come out identical — a factor of 1 is notation, not a loop — are
    yielded once."""
    levels = binding.levels
    body = {rank: n for rank, n in binding.body.items() if n != 1}
    residual = {}
    for rank, extent in ranks.items():
        inner = body.get(rank, 1)
        if extent % inner:
            raise AssemblyError(
                f"rank '{rank}': '{binding.compute}' performs {inner} of it "
                f"intrinsically and D has {extent}, which it does not divide — a "
                f"rank's factors have to multiply to its extent exactly, so there is "
                f"no nest here to enumerate rather than a ragged one to round to"
            )
        residual[rank] = extent // inner
    for rank in body:
        if rank not in ranks:
            raise AssemblyError(
                f"rank '{rank}': '{binding.compute}' performs it intrinsically, but "
                f"D does not have it"
            )

    order = list(ranks)
    # One slot per (level, temporal|spatial) a factor can go in. With one instance
    # there are no spatial slots at all, which is what makes 1 the default that
    # invents nothing rather than a fanout of one.
    kinds = (False, True) if instances > 1 else (False,)
    slots = [(d, spatial) for d in range(len(levels)) for spatial in kinds]
    temporal_slot = {d: slots.index((d, False)) for d in range(len(levels))}
    spatial_slot = [i for i, (_d, spatial) in enumerate(slots) if spatial]
    # The intrinsic, as the innermost loops. Their order among themselves never
    # reaches the walk — ``_split_body`` strips them off as the instruction's own.
    tail = tuple(Loop(rank, n, levels[-1]) for rank, n in body.items())

    seen = set()
    for combo in itertools.product(*(_splits(residual[r], len(slots)) for r in order)):
        factors = dict(zip(order, combo))
        if _prod(factors[r][i] for r in order for i in spatial_slot) > instances:
            continue
        by_level = [
            [r for r in order if factors[r][temporal_slot[d]] != 1]
            for d in range(len(levels))
        ]
        for perm in itertools.product(*(itertools.permutations(t) for t in by_level)):
            loops = []
            for d, level in enumerate(levels):
                loops += [
                    Loop(rank, factors[rank][temporal_slot[d]], level)
                    for rank in perm[d]
                ]
                for i in (i for i in spatial_slot if slots[i][0] == d):
                    # Canonical order: permuting spatial loops only renumbers the
                    # instances, which no cost this frontend reports can see.
                    loops += [
                        Loop(rank, factors[rank][i], level, spatial=True)
                        for rank in order
                        if factors[rank][i] != 1
                    ]
            nest = tuple(loops) + tail
            if nest not in seen:
                seen.add(nest)
                yield Mapping(dict(ranks), nest)


@dataclass(frozen=True)
class Priced:
    """One candidate, compiled — which is the whole of how it is scored.

    ``makespan`` is σ's, so it is in cycles wherever the ISA has a cycle model and
    in unit steps otherwise; either way it is the point the schedule actually
    achieves inside the ``bottleneck_cycles()`` / ``cycles()`` bracket, rather than
    a count of anything. ``emits`` breaks ties, so two nests the time model cannot
    separate are separated by the shorter program."""

    mapping: Mapping
    program: object
    sigma: object
    makespan: float
    emits: int

    @property
    def cost(self) -> tuple:
        """The key candidates are ranked by: time first, program length to break a tie."""
        return (self.makespan, self.emits)


@dataclass(frozen=True)
class Choice:
    """A search's answer, and what it had to reject to get there.

    ``refused`` counts the nests removed *before* pricing, keyed by the obligation
    that removed them — the machine's own refusals doing duty as search
    constraints — plus ``capacity`` for the ones the allocator could not place.
    A nest is tallied under its first violation, so the counts sum to nests
    removed rather than to obligations broken."""

    best: Priced
    considered: int
    refused: dict = field(default_factory=dict)

    @property
    def priced(self) -> int:
        """How many nests were actually compiled — the rest never got that far."""
        return self.considered - sum(self.refused.values())


def price(mapping: Mapping, binding) -> Priced:
    """What this nest costs on this machine — by assembling it and reading σ.

    The pricing function *is* the standalone assembly entry, which is the payoff
    of step 3 having made it a step generator over the planner's own pass 1: what
    was written to verify an external mapping turns out to be the thing that
    scores a native one. It raises what assembly raises — an ``AssemblyError`` for
    a nest the machine cannot perform, an ``AllocationError`` for one it cannot
    place — and those refusals are the search's pruning.

    A site is priced *in isolation*, the same isolation Timeloop maps a layer in:
    the level-0 blocks are this operation's own tensors in the I/O buffer, not
    whatever else the surrounding program has resident. Inter-layer effects are
    the compiler's job downstream (Stage 2b's unification and the planner's
    relayout movers), not the mapper's."""
    program, sigma = assemble(mapping, binding)
    return Priced(mapping, program, sigma, sigma.makespan, len(program.emits))


def search(
    binding, op=None, *, ranks: dict | None = None, instances: int = 1
) -> Choice:
    """Enumerate this site's mapspace and return the cheapest nest that compiles.

    With ``op`` the dataspaces and the iteration domain come from the source
    operation (``_derive``), exactly as they do at a mapped site; without one the
    binding has to declare its dataspaces and the caller has to supply ``ranks``,
    exactly as ``to_timeloop`` requires. So the native path takes its problem from
    the same place the exported problem file did.

    Raises when the whole space is refused, with the tally — a machine that can
    perform *no* mapping of an operation is a fact about the pair, and which
    obligation did the refusing is the useful half of it."""
    if op is not None:
        bound, ranks = _derive(op)
        binding = replace(binding, dataspaces=tuple(ds for ds, _v in bound))
    elif not binding.dataspaces or ranks is None:
        raise AssemblyError(
            "search: without a source op the binding has to declare its dataspaces "
            "and the caller has to supply `ranks` — a problem instance cannot be "
            "recovered from the projections alone"
        )
    best, considered, refused = None, 0, {}

    def refuse(reason: str) -> None:
        refused[reason] = refused.get(reason, 0) + 1

    for candidate in mapspace(binding, ranks, instances=instances):
        considered += 1
        violations = candidate.check(binding)
        if violations:
            refuse(violations[0].constraint)
            continue
        try:
            priced = price(candidate, binding)
        except AllocationError:
            refuse("capacity")
            continue
        except AssemblyError:
            refuse("assembly")
            continue
        if best is None or priced.cost < best.cost:
            best = priced
    if best is None:
        tally = ", ".join(f"{n} by {reason}" for reason, n in sorted(refused.items()))
        raise AssemblyError(
            f"no mapping of this operation onto '{binding.isa.name}' could be "
            f"performed: {considered} nest(s), all refused ({tally})"
        )
    return Choice(best, considered, refused)


def native_mapper(binding, **kw):
    """``search`` as a ``mapping_for`` driver, so a native mapping and an imported
    one enter the compiler through the same hole.

    That they do is the point rather than a convenience: a mapped site does not
    know or care where its nest was chosen, so the import path stays available as
    the control the searched one is measured against."""

    def mapping_for(op):
        if op.operation.name not in _DERIVE:
            return None
        return search(binding, op, **kw).best.mapping, binding

    return mapping_for
