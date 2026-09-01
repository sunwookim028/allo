# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importing an externally chosen mapping (refactor target 5).

``drafts/schedule-isa-summary.md`` §6 states the relation this module lands.
A Timeloop mapping is **σ without λ and without code**: per storage level a set
of tiling factors, a loop permutation, a spatial/temporal split and a bypass
mask, scored by an analytical access count. This frontend is its exact
complement — λ in full (every address is solved, allocated and emitted), σ until
now only a linear order, and it *emits running code*. So the two halves compose,
and the composition occupies a hole nobody fills: **Timeloop does not emit code,
ACT emits code but has no σ, MINISA's own layout search is a cost-blind
feasibility filter — nobody verifies a mapping and then runs it.**

Three objects, in the order the import goes through them:

1. ``Mapping`` — the mapping as our object, deliberately not Timeloop's: the
   problem instance (rank -> extent) *is* the iteration domain **D**, and the
   per-level directives flatten to one outermost-first loop nest. ``check`` is
   then **constraint 1 proper** (v2 §3.4): the factors of each rank multiply to
   its extent, so the nest enumerates D exactly once — the coverage obligation
   ``check.py`` could only shadow as *definedness*, because a compiled wire
   carries no D and an imported mapping does.
2. ``Binding`` — the half a mapping cannot carry, because Timeloop has no
   vocabulary for it: which buffer each storage level *is*, which instruction
   performs the innermost tile and with what intrinsic extents, and which operand
   of it is which tensor. That is all of it, and its size is a fair measure of the
   distance between the two tools. What a binding deliberately does *not* carry is
   anything the **operation** already fixes: a tensor's shape and its projection
   from the ranks are the same on every machine, so at a mapped site they are
   derived from the source op (``_derive``) in Timeloop's own problem-shape names.
3. ``assemble`` — mapping + binding -> a ``CompiledProgram`` (so it runs) and a
   ``Schedule`` (so it is verifiable). The nest is walked into the planner's own
   pass 1 (``search._Planner``), the one a matched site and an ``@expand`` body
   also go through: a tile whose origin changed is drained and refilled, one
   transfer per *maximal contiguous run* under the two layouts (a machine with
   only contiguous DMA has no other option, and the run count is exactly the
   access count Timeloop's evaluator predicts); the innermost body becomes one
   step per point. Liveness, allocation, spilling and emission are then the
   compiler's, unmodified — which is the sense in which *a mapping is an imported
   expansion*.

**Where the division of labour falls.** The mapping fixes tiling, loop order and
spatial assignment. Everything else stays ours and is *not* imported: tile
placement (λ — the mapping says what is resident, never where, and the allocator
answers where *and whether it fits*), the mover decomposition and the choice of
mover, the machine's own instruction-word fields (filled by
``InstructionSpec.configurations``, the same chooser the planner uses), and the
time base (σ's ``pe`` and ordering come from the mapping, its *times* from our
``ISA.latency`` model). ``assemble`` therefore returns a σ that no single tool
could have produced, and ``CompiledProgram.check(sigma=...)`` verifies it.

**What is deliberately not modelled**, since a checker that quietly assumes them
is worth less than one that names them:

- **No double buffering.** One tile slot per (level, dataspace, instance), so a
  refill is ordered against the compute that reads it and σ shows the stall.
  Timeloop prices double buffering without addressing it; here it would be a
  second slot and an alternation, and it is not written.
- **No reducing network.** A mapping with a *reduction* rank in a spatial loop
  claims partial sums combine across instances, and no machine description here
  declares a network that does it (v2 constraint 7). Such a mapping is
  **refused**, not assembled-and-checked: each instance would accumulate its own
  partial into its own slot and both would drain to the same address, and every
  constraint holds on that program — value identity dies at emission, so no
  address-level checker can see the lost sum. Refusal is the honest form of an
  obligation nothing can discharge.
- **No arch YAML.** Storage capacities, fanouts and energies live in Timeloop's
  architecture file; here the buffer sizes are the ISA's own and capacity is
  checked as ``check.py``'s *bounds*. Fanout has no machine declaration to check
  against, exactly like constraints 6 and 7.
- **No YAML parser.** ``from_timeloop`` takes the *parsed* structure — what
  ``yaml.safe_load`` returns — so the module has no dependency of its own and
  the translation is the mapping's semantics, not its tokenization.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np

from .check import Violation
from .core import (
    Tile,
    _access_chain,
    access_map,
    access_names,
    access_patterns,
    compute_params,
    dense_strides,
    param_roles,
    residence,
    show_map,
)
from .errors import AssemblyError
from .search import (
    _canon,
    _Compute,
    _i64_attr,
    _is_zero_const,
    _Planner,
    _source_ins,
    _static_shape,
    _strip_leading_units,
)


def _strip(shape) -> tuple:
    """A shape modulo leading unit dimensions — the rank alias the whole frontend
    looks through, so a torch-batched ``1xMxN`` compares equal to a plain ``MxN``."""
    return tuple(_strip_leading_units(list(shape)))


# ==========================================================================#
# The mapping: a loop nest over D, with each loop attributed to a level
# ==========================================================================#


@dataclass(frozen=True)
class Loop:
    """One loop of the nest: ``factor`` iterations of ``rank``, attributed to the
    storage level whose tiles it iterates over. ``spatial`` loops are unrolled
    across instances (Timeloop's fanout) rather than in time."""

    rank: str
    factor: int
    level: str
    spatial: bool = False


@dataclass(frozen=True)
class Mapping:
    """A mapping as an ordered nest plus the problem instance it maps.

    ``ranks`` is the iteration domain **D** — the one thing an emitted program
    does not carry and a mapping does, which is why constraint 1 becomes
    checkable here. ``loops`` is the whole nest **outermost first**, with each
    loop's level attribution kept: a level's tile of a dataspace is what the
    loops at that level *and inside it* touch, so the attribution is what turns
    an order into a data-movement schedule. ``bypass`` names, per level, the
    dataspaces that level does not hold (Timeloop's bypass mask); a bypassed
    level is transparent and the transfer skips it."""

    ranks: dict
    loops: tuple
    bypass: dict = field(default_factory=dict)

    def keeps(self, level: str, dataspace: str) -> bool:
        return dataspace not in self.bypass.get(level, ())

    def check(self, binding=None, source=None) -> list[Violation]:
        """The mapping's own obligations; ``[]`` means it is assemblable.

        **Constraint 1, in full.** A rank's factors across the whole nest are a
        mixed-radix decomposition of its index, so the nest enumerates D exactly
        once iff every rank's factors multiply to its extent — coverage and
        no-double-count in one equation.

        ``source`` — ``{dataspace: shape}`` from the op being lowered — closes the
        remaining link. Without it the nest is only checked against a **declared**
        D, which the binding's author could have declared to suit; with it, D is
        pinned to the tensors the source program actually passes, and the chain
        "factors multiply to the extent" → "level 0's tile is the whole dataspace"
        → "the dataspace is the source tensor" says the nest enumerates the source
        op's own iteration domain exactly once.

        With ``binding``, the machine-side agreements join, and each of them is a
        way a mapping can be un-implementable however good its access count: a
        level the binding does not name; a spatial loop over a reduction rank
        (constraint 7, see the module docstring); innermost loops that do not
        compose the instruction's intrinsic tile; an operand whose innermost
        keeper is not the buffer the instruction reads it from, or in which it
        does not sit the way the instruction reads it."""
        out = []
        got: dict = {}
        for loop in self.loops:
            got[loop.rank] = got.get(loop.rank, 1) * loop.factor
        for rank, extent in self.ranks.items():
            if got.get(rank, 1) != extent:
                out.append(
                    Violation(
                        "coverage",
                        (),
                        f"rank '{rank}': the nest's factors multiply to "
                        f"{got.get(rank, 1)}, but D has extent {extent}",
                    )
                )
        for rank in got:
            if rank not in self.ranks:
                out.append(
                    Violation(
                        "coverage",
                        (),
                        f"loop over rank '{rank}', which D does not have",
                    )
                )
        if binding is None:
            return out
        if not binding.dataspaces:
            out.append(
                Violation(
                    "binding",
                    (),
                    "this binding derives its dataspaces from a source op, so it can "
                    "only be checked where there is one (a mapped site)",
                )
            )
            return out
        for ds in binding.dataspaces:
            # Compared modulo leading unit dims, the same rank alias the rest of the
            # frontend looks through (a torch-batched 1xMxN *is* an MxN tensor).
            if source is not None and _strip(source[ds.name]) != _strip(ds.shape):
                out.append(
                    Violation(
                        "coverage",
                        (),
                        f"'{ds.name}': the mapping is over a {tuple(ds.shape)} "
                        f"tensor, but the source op passes {tuple(source[ds.name])}",
                    )
                )
        for loop in self.loops:
            if loop.level not in binding.levels:
                out.append(
                    Violation(
                        "binding",
                        (),
                        f"loop at level '{loop.level}', which the binding does not name",
                    )
                )
        for ds in binding.dataspaces:
            if not self.keeps(binding.levels[0], ds.name):
                out.append(
                    Violation(
                        "binding",
                        (),
                        f"level '{binding.levels[0]}' holds program I/O and cannot "
                        f"bypass '{ds.name}'",
                    )
                )
        # Constraint 7. A rank absent from every output projection is a reduction;
        # fanned across instances it leaves one partial sum per instance, and
        # combining them takes a reducing network. Refused unless the machine
        # *declares* the instruction that does the combining
        # (``ISA.network(reduces=...)``) — assembling it without one would emit a
        # program whose instances overwrite each other, and every constraint would
        # hold on it, because value identity dies at emission.
        reduced = set(_reduction_ranks(binding, self.ranks))
        if binding.isa.reduces is None:
            for loop in self.loops:
                if loop.spatial and loop.rank in reduced:
                    out.append(
                        Violation(
                            "reduction",
                            (),
                            f"spatial loop over reduction rank '{loop.rank}' at level "
                            f"'{loop.level}': the partial sums it fans out have to be "
                            f"combined, and this machine declares no instruction that "
                            f"combines them — see ISA.network(reduces=...)",
                        )
                    )
        split = _split_body(self, binding)
        if split is None:
            out.append(
                Violation(
                    "intrinsic",
                    (),
                    f"the nest's innermost loops do not compose the intrinsic tile "
                    f"{binding.body} that '{binding.compute}' performs",
                )
            )
            return out
        # Where the compute reaches its operands. An instruction's access is
        # rooted at a *particular* buffer and describes a *particular* residence,
        # so an operand's innermost keeper has to be the level bound to that
        # buffer, and the intrinsic tile has to *sit there the way the access
        # reads it* — otherwise the emitted offset addresses a block laid out
        # differently from the one described, and the program is wrong rather
        # than refused. Compared as maps, not as contiguity: a sub-block can be
        # one contiguous run and still hold its elements in another order.
        nest, _body = split
        depth = {level: d for d, level in enumerate(binding.levels)}
        # The outermost level holds the whole tensor — the mapping's coverage,
        # restated as a placement fact, and the link that carries the rank equation
        # above onto the shapes the site actually has. Reported only when the ranks
        # themselves check out, since otherwise it is the same defect twice.
        if not any(v.constraint == "coverage" for v in out):
            for ds in binding.dataspaces:
                whole = _tile(ds, _span(nest, binding.body, 0, depth))
                if whole != tuple(ds.shape):
                    out.append(
                        Violation(
                            "coverage",
                            (),
                            f"'{ds.name}': the nest's factors give level "
                            f"'{binding.levels[0]}' a {whole} tile, not the whole "
                            f"{tuple(ds.shape)} tensor",
                        )
                    )
        spaces = {ds.name: ds for ds in binding.dataspaces}
        spec = binding.isa._ops[binding.compute].spec
        rooted = [root.buffer.name for _pos, _stripped, root in _access_chain(spec)]
        names, roles = access_names(spec), param_roles(spec)[0]
        params = {
            i: binding.params[names[i]] for i, r in roles.items() if r != "offset"
        }
        reads = [access_map(p, params) for p in access_patterns(spec)]
        for pos, name in enumerate(binding.operands):
            ds = spaces[name]
            keeper = max(
                d
                for d in range(len(binding.levels))
                if self.keeps(binding.levels[d], name)
            )
            level = binding.levels[keeper]
            if binding.buffers[level] != rooted[pos]:
                out.append(
                    Violation(
                        "binding",
                        (),
                        f"'{name}' is innermost held at level '{level}' "
                        f"('{binding.buffers[level]}'), but '{binding.compute}' reads "
                        f"operand {pos} from '{rooted[pos]}'",
                    )
                )
                continue
            resident = _tile(ds, _span(nest, binding.body, keeper, depth))
            intrinsic = _tile(ds, binding.body)
            sits = residence(
                [(n, st) for n, (_s, st) in zip(intrinsic, _packing(resident))]
            )
            wants = residence(reads[pos])
            if sits != wants:
                out.append(
                    Violation(
                        "residence",
                        (),
                        f"'{name}': level '{level}' holds a {resident} tile, in which "
                        f"the intrinsic {intrinsic} sits as {show_map(sits)}, but "
                        f"'{binding.compute}' reads operand {pos} as "
                        f"{show_map(wants)} — the loops between them have to move "
                        f"outside that level, or the tile has to be packed the way "
                        f"the access reads it",
                    )
                )
        return out


def _factors(entry) -> dict:
    """Timeloop's ``factors``: ``{'C': 2}`` or the compact ``"C=2 K=4"``."""
    if isinstance(entry, dict):
        return {k: int(v) for k, v in entry.items()}
    out = {}
    for token in str(entry).replace(",", " ").split():
        rank, _, value = token.partition("=")
        out[rank] = int(value)
    return out


def _permutation(entry) -> list:
    """Timeloop's ``permutation``, **innermost first** — a list, a delimited
    string, or the bare letter sequence its own mappers print (``"RSPQCKN"``)."""
    if entry is None:
        return []
    if not isinstance(entry, str):
        return list(entry)
    return entry.replace(",", " ").split() if {" ", ","} & set(entry) else list(entry)


def from_timeloop(mapping, problem, *, levels) -> Mapping:
    """A Timeloop mapping + problem instance as a ``Mapping``.

    Both arguments are the *parsed* YAML (a ``dict``, or the bare ``mapping``
    list / ``instance`` dict). ``levels`` orders the storage levels **outermost
    first**: the mapping names its levels but does not order them — that lives in
    the architecture file, and the binding has to name them anyway.

    Two Timeloop conventions are honoured and worth stating, because getting
    either backwards yields a different, silently self-consistent program:
    ``permutation`` lists loops **innermost first**, and a level's *spatial*
    loops sit **inside** its temporal ones (they are the fanout to the level
    below). Factors of 1 are dropped — they are notation, not loops."""
    directives = mapping["mapping"] if isinstance(mapping, dict) else list(mapping)
    if isinstance(problem, dict) and "problem" in problem:
        problem = problem["problem"]["instance"]
    ranks = {k: int(v) for k, v in problem.items() if isinstance(v, int)}

    per_level: dict = {}
    bypass: dict = {}
    for directive in directives:
        target, kind = directive["target"], directive.get("type", "temporal")
        if kind == "bypass":
            bypass[target] = frozenset(directive.get("bypass", ()))
            continue
        factors = _factors(directive.get("factors", {}))
        order = _permutation(directive.get("permutation")) or list(factors)
        per_level.setdefault(target, {})[kind] = (factors, order)

    loops = []
    for level in levels:
        for kind in ("temporal", "spatial"):
            factors, order = per_level.get(level, {}).get(kind, ({}, []))
            for rank in reversed(order):  # innermost-first -> outermost-first
                if factors.get(rank, 1) != 1:
                    loops.append(
                        Loop(rank, factors[rank], level, spatial=kind == "spatial")
                    )
    return Mapping(ranks, tuple(loops), bypass)


# ==========================================================================#
# The binding: what a mapping cannot say, because Timeloop has no machine
# ==========================================================================#


@dataclass(frozen=True)
class Dataspace:
    """One tensor of the einsum: its shape, and how its coordinates project from
    the ranks — per data dimension, ``{rank: coefficient}``.

    An affine projection is all Timeloop's own problem shapes are, and the
    coefficients are what make a halo work: a tile's extent along a dimension is
    ``1 + Σ coeff·(rank span - 1)``, so a convolution's ``p + r`` widens the
    input tile exactly as it should. ``role`` says whether the host supplies the
    tensor or reads it back."""

    name: str
    shape: tuple
    projection: tuple
    role: str = "input"  # "input" | "output"


def _matmul_spaces(op) -> tuple[list, dict]:
    """``tosa.matmul``: ``[Batch, M, K] x [Batch, K, N] -> [Batch, M, N]``, in
    Timeloop's own GEMM rank names so an imported mapping needs no translation."""
    a, b = _source_ins(op)
    z = op.results[0]
    zs, as_ = _shape_of(z), _shape_of(a)
    ranks = {"Batch": zs[0], "M": zs[1], "N": zs[2], "K": as_[2]}
    return [
        (
            Dataspace("A", _shape_of(a), ({"Batch": 1}, {"M": 1}, {"K": 1}), "input"),
            a,
        ),
        (
            Dataspace("B", _shape_of(b), ({"Batch": 1}, {"K": 1}, {"N": 1}), "input"),
            b,
        ),
        (
            Dataspace("Z", _shape_of(z), ({"Batch": 1}, {"M": 1}, {"N": 1}), "output"),
            z,
        ),
    ], ranks


def _conv2d_spaces(op) -> tuple[list, dict]:
    """``tosa.conv2d``: NHWC input, OHWI weights, NHWC output, in Timeloop's own CNN
    rank names.

    The halo is the projection, and the coefficients are where stride and dilation
    live: an input row is ``stride·P + dilation·R``, so the tile extent
    ``1 + Σ c·(span - 1)`` widens to exactly the window the pass reads. Two things
    are refused rather than modelled, because both would silently compute something
    else: **padding**, which puts input coordinates outside the tensor and this
    address model has no notion of a zero that is not stored; and a **non-zero
    bias**, which no operand of the instruction would add — a zero bias is the
    accumulator's initial value, which the run already preloads."""
    ifm, flt, bias = _source_ins(op)
    if not _is_zero_const(bias):
        raise AssemblyError(
            "tosa.conv2d: a mapped site's run accumulates into a zeroed output, so a "
            "non-zero bias has nothing to add it — fold it into the source program "
            "or declare the dataspaces by hand"
        )
    if any(_i64_attr(op, "pad")):
        raise AssemblyError(
            f"tosa.conv2d: pad {_i64_attr(op, 'pad')} — a padded window reads "
            f"coordinates outside the tensor, and a tile placement has no address "
            f"for a zero that is not stored"
        )
    sh, sw = _i64_attr(op, "stride")
    dh, dw = _i64_attr(op, "dilation")
    o, w = _shape_of(op.results[0]), _shape_of(flt)
    ranks = {
        "N": o[0],
        "P": o[1],
        "Q": o[2],
        "M": o[3],
        "R": w[1],
        "S": w[2],
        "C": w[3],
    }
    return [
        (
            Dataspace(
                "Inputs",
                _shape_of(ifm),
                ({"N": 1}, {"P": sh, "R": dh}, {"Q": sw, "S": dw}, {"C": 1}),
                "input",
            ),
            ifm,
        ),
        (
            Dataspace(
                "Weights",
                _shape_of(flt),
                ({"M": 1}, {"R": 1}, {"S": 1}, {"C": 1}),
                "input",
            ),
            flt,
        ),
        (
            Dataspace(
                "Outputs",
                _shape_of(op.results[0]),
                ({"N": 1}, {"P": 1}, {"Q": 1}, {"M": 1}),
                "output",
            ),
            op.results[0],
        ),
    ], ranks


_DERIVE = {"tosa.matmul": _matmul_spaces, "tosa.conv2d": _conv2d_spaces}


def _shape_of(value) -> tuple:
    return tuple(_static_shape(value))


def _derive(op) -> tuple[list, dict]:
    """The dataspaces of a source op — each with the value it is — and its ranks.

    A tensor's shape and how its coordinates project from the ranks are properties
    of the **operation**, not of the machine — a convolution's halo is ``p + r``
    wherever it runs — so at a mapped site they come from here rather than from the
    binding. The names are Timeloop's own, which is what lets a mapping written
    against an exported problem file import without translation.

    Returning the *value* alongside each dataspace is what lets an operand the
    instruction does not take (a convolution's bias) simply not appear, instead of
    forcing a positional correspondence that would then be wrong. The ranks are the
    iteration domain **D** read off the same shapes — inverting the projections in
    general is not possible (a halo folds two ranks into one dimension), so each
    operation states its own."""
    make = _DERIVE.get(op.operation.name)
    if make is None:
        raise AssemblyError(
            f"'{op.operation.name}': no dataspace derivation for this operation — "
            f"its projections have to come from somewhere, so either add one here or "
            f"declare Binding.dataspaces by hand"
        )
    return make(op)


@dataclass(frozen=True)
class Binding:
    """The machine half of an import: everything a mapping leaves unsaid.

    ``levels`` orders the storage levels outermost first and ``levels[0]`` is the
    I/O level — the buffer program arguments are marshalled through. ``compute``
    is the instruction that performs the innermost tile and ``body`` the extents
    it performs intrinsically (a rank absent from ``body`` has extent 1);
    ``operands`` names the dataspace at each of its access positions, so an
    accumulating instruction simply lists its output twice — once read, once
    written. ``params`` / ``alpha`` supply ``compute``'s non-address access params
    and its computational attributes, which are constant across the program because
    the intrinsic tile is.

    There is deliberately no mover table: which instruction crosses a level pair is
    not a machine fact the binding has to state but a *route*, found in the same
    edge graph the planner routes with — at the buffer pair and run length the
    mapping implies, priced by the mover's own cost, and configured by its own
    ``@schedule``.

    ``dataspaces`` is likewise not a machine fact: a tensor's shape and how its
    coordinates project from the ranks are properties of the *operation* — a
    convolution's halo is ``p + r`` on any machine — so at a mapped site they are
    **derived from the source op** and this field is left empty. Declaring them is
    for standalone assembly, where there is no source op to derive from, and for an
    operation ``_derive`` does not know. What is left when they are gone is the
    honest measure of the distance between Timeloop and a compiler: levels,
    buffers, the instruction, its intrinsic tile, and which operand is which."""

    isa: object
    levels: tuple
    buffers: dict
    compute: str
    body: dict
    operands: tuple
    dataspaces: tuple = ()  # empty: derived from the source op at a mapped site
    params: dict = field(default_factory=dict)
    alpha: dict = field(default_factory=dict)


# ==========================================================================#
# Assembly
# ==========================================================================#


def _split_body(mapping: Mapping, binding: Binding):
    """``(the loops we emit, the loops the instruction performs)``, or ``None``
    when the nest's innermost loops do not compose the intrinsic tile exactly."""
    want = {rank: n for rank, n in binding.body.items() if n != 1}
    got: dict = {}
    cut = len(mapping.loops)
    while got != want and cut > 0:
        loop = mapping.loops[cut - 1]
        got[loop.rank] = got.get(loop.rank, 1) * loop.factor
        if got[loop.rank] > want.get(loop.rank, 1):
            return None
        cut -= 1
    return (mapping.loops[:cut], mapping.loops[cut:]) if got == want else None


def _weights(nest, body: dict) -> list:
    """Each emitted loop's radix weight — the product of its rank's factors
    inside it, the intrinsic tile included. The rank's index at a point is the
    mixed-radix number ``Σ index·weight``."""
    acc = dict(body)
    out = [0] * len(nest)
    for i in reversed(range(len(nest))):
        out[i] = acc.get(nest[i].rank, 1)
        acc[nest[i].rank] = out[i] * nest[i].factor
    return out


def _span(nest, body: dict, d: int, depth: dict) -> dict:
    """Per rank, the extent of one depth-``d`` tile: what the loops at that level
    and every level inside it sweep, times the intrinsic tile."""
    out = dict(body)
    for loop in nest:
        if depth[loop.level] >= d:
            out[loop.rank] = out.get(loop.rank, 1) * loop.factor
    return out


def _tile(ds: Dataspace, span: dict) -> tuple:
    return tuple(
        1 + sum(c * (span.get(rank, 1) - 1) for rank, c in dim.items())
        for dim in ds.projection
    )


def _project(ds: Dataspace, index: dict) -> tuple:
    return tuple(
        sum(c * index.get(rank, 0) for rank, c in dim.items()) for dim in ds.projection
    )


def _packing(shape) -> list[tuple]:
    """How a level packs a tile of ``shape``, as a ``(size, stride)`` map.

    Dense row-major, and it is a *decision* rather than a fact: Timeloop says what
    is resident, never how it is laid out, so the import has to choose. Everything
    downstream reads the packing from here instead of assuming it, which is what
    lets level 0 become the operand's own residence — whatever Stage 2b unified —
    when a mapping enters at a matched site."""
    return list(zip(shape, dense_strides(shape)))


def _run_offsets(outer, inner) -> tuple[list, int]:
    """``([(offset in the block it comes from, offset in the block it goes to)],
    run length)`` for one tile transfer, given the ``(size, stride)`` map the moved
    block has at each end.

    A dimension joins the contiguous run only if it is packed immediately inside
    the run **at both ends**: a transfer copies consecutive addresses to
    consecutive addresses, so a dimension the two sides stride differently becomes
    separate instructions. Every dimension in the run therefore has the same stride
    on both sides, which is what makes the copy element-for-element rather than a
    reordering. This is why a mapping's transfer is many instructions and not one —
    a machine whose DMA copies contiguous words has no other option — and the count
    is exactly the access count Timeloop's model predicts analytically."""
    assert [s for s, _ in outer] == [s for s, _ in inner]
    run, merged = 1, set()
    while True:
        nxt = next(
            (
                i
                for i, ((n, a), (_n, b)) in enumerate(zip(outer, inner))
                if i not in merged and n != 1 and a == run and b == run
            ),
            None,
        )
        if nxt is None:
            break
        merged.add(nxt)
        run *= outer[nxt][0]
    rest = [i for i, (n, _st) in enumerate(outer) if i not in merged and n != 1]
    offsets = [
        (
            sum(c * outer[i][1] for c, i in zip(coord, rest)),
            sum(c * inner[i][1] for c, i in zip(coord, rest)),
        )
        for coord in itertools.product(*(range(outer[i][0]) for i in rest))
    ]
    return offsets, run


def _configure(spec, shape_params: dict) -> dict:
    """The instruction-word fields the *machine* still owns, at this site — the
    cheapest configuration its ``@schedule`` predicate admits. An imported
    mapping fixes tiling and order; it says nothing about a burst factor or a
    packing, so those go through the same chooser the planner uses."""
    configs = spec.configurations(shape_params)
    if not configs:
        raise AssemblyError(
            f"{spec.name}: no configuration its @schedule admits at {shape_params}"
        )
    return min(configs, key=lambda c: c[1])[0]


def _reads(binding: Binding) -> dict:
    """Per dataspace, whether the compute reads it — an output it only writes is
    drained but never filled, and needs no initial value."""
    n_src = len(binding.isa._ops[binding.compute].spec.sources)
    read = {ds.name: False for ds in binding.dataspaces}
    for pos, name in enumerate(binding.operands):
        read[name] = read[name] or pos < n_src
    return read


def _pack_as(shape, res, who) -> list[tuple]:
    """``shape`` laid out as the residence ``res``, as a per-dimension map.

    A residence drops the dimensions that span one element — they hold their datum
    wherever their stride puts it — so this is the inverse: put the strides back on
    the dimensions that carry them, and leave the rest at 0, where they contribute
    the coordinate 0 they are entitled to."""
    span = [i for i, n in enumerate(shape) if n != 1]
    if [shape[i] for i in span] != [n for n, _st in res]:
        raise AssemblyError(
            f"{who}: it is laid out as {show_map(res)}, which is not a residence of "
            f"a {tuple(shape)} tensor"
        )
    out = [(n, 0) for n in shape]
    for i, (_n, st) in zip(span, res):
        out[i] = (shape[i], st)
    return out


def _compute_step(planner, binding: Binding, locs: list, offsets: list, pe) -> None:
    """The innermost body as one step: its operands at ``locs``, each shifted by
    the point's offset *within* that tile."""
    spec = binding.isa._ops[binding.compute].spec
    names = access_names(spec)
    roles, offset_of = param_roles(spec)
    params = {
        i: binding.params[names[i]] for i, role in roles.items() if role != "offset"
    }
    chosen = _configure(spec, {i: v for i, v in params.items() if roles[i] == "shape"})
    planner.steps.append(
        _Compute(
            spec.name,
            locs[: len(spec.sources)],
            locs[len(spec.sources)],
            offset_of,
            params,
            set(),  # the mapping decides residence; nothing coalesces behind its back
            [binding.alpha[n] for n in compute_params(spec)],
            [chosen[k] for k in spec.schedule_domains],
            [{0: o} for o in offsets],
            pe,
        )
    )


def _lower(mapping: Mapping, binding: Binding, planner, root: dict) -> None:
    """Walk the nest, appending the steps it prescribes to ``planner``.

    ``root`` gives the level-0 location of each dataspace — a whole-tensor block the
    caller has already placed. Standalone assembly makes them itself, in the I/O
    buffer under the host ABI; a mapped *site* passes the surrounding program's own
    locations, and their residence is then whatever Stage 2b settled. That is the
    only difference between the two entries, which is why ``_packing`` was made the
    single place that says how a level packs a tile.

    The walk is drain-then-fill: at each point of the emitted nest, any tile whose
    origin changed is written back (innermost level first, so a partial sum reaches
    its home before the level above it moves) and then refilled (outermost first, so
    a tile is read from an outer tile that is already current). A dataspace the
    compute never reads is not filled, only drained."""
    isa = binding.isa
    split = _split_body(mapping, binding)
    assert split is not None
    nest, _body_loops = split
    depth = {level: d for d, level in enumerate(binding.levels)}
    n_levels = len(binding.levels)
    spaces = {ds.name: ds for ds in binding.dataspaces}
    weights = _weights(nest, binding.body)

    for level in binding.levels:
        buf = isa.buffers[binding.buffers[level]]
        assert buf.address_rank == 1 and buf.slot_size == 1, (
            f"'{buf.name}': tiles are placed in a flat, word-addressable space; this "
            f"buffer already fixes part of the packing"
        )
    assert isa.buffers[binding.buffers[binding.levels[0]]] is planner.io, (
        f"level '{binding.levels[0]}' marshals program I/O, so the buffer bound to it "
        f"has to be the ISA's global one, not '{binding.buffers[binding.levels[0]]}'"
    )

    # Tiles: what each level holds of each dataspace. Depth 0 is the whole tensor
    # by constraint 1 — the mapping's coverage, restated as a placement fact.
    tiles = [
        {
            ds.name: _tile(ds, _span(nest, binding.body, d, depth))
            for ds in binding.dataspaces
        }
        for d in range(n_levels)
    ]
    for ds in binding.dataspaces:
        assert tiles[0][ds.name] == tuple(ds.shape), (
            f"'{ds.name}': the outermost level's tile is {tiles[0][ds.name]}, not the "
            f"whole tensor {tuple(ds.shape)}"
        )
    chain = {
        ds.name: [
            d for d in range(n_levels) if mapping.keeps(binding.levels[d], ds.name)
        ]
        for ds in binding.dataspaces
    }

    # λ is not ours to bump-allocate any more. Each (level, dataspace, instance) is a
    # *value the lowering invented* and one location holding it — written once per
    # round of the walk and read by whatever the mapping says reads it — so where it
    # sits, whether it fits, and what it costs in a buffer shared with the rest of
    # the program are the allocator's answers, on the same terms as everything else.
    held = {(0, ds.name, 0): root[ds.name] for ds in binding.dataspaces}
    packing = [
        {ds.name: _packing(tiles[d][ds.name]) for ds in binding.dataspaces}
        for d in range(n_levels)
    ]
    # Level 0 is packed the way the block already *is*, not the way this module
    # would have packed it: at a mapped site it is a value of the surrounding
    # program, and repacking it to suit the mapping is data movement nobody asked
    # for. Every run offset and every operand offset reads the packing from here.
    packing[0] = {
        ds.name: _pack_as(ds.shape, root[ds.name].map, f"'{ds.name}' at level 0")
        for ds in binding.dataspaces
    }

    def tile_at(d: int, name: str, pe: int):
        key = (d, name, pe)
        if key not in held:
            buf = isa.buffers[binding.buffers[binding.levels[d]]]
            tile = Tile(tuple(tiles[d][name]))
            held[key] = planner.at(tile, buf, residence(packing[d][name]))
        return held[key]

    reduced = set(_reduction_ranks(binding, mapping.ranks))

    def fans(d_out: int, d_in: int) -> bool:
        """Whether a reduction rank is fanned across instances between these two
        levels — so the inner level holds one *partial sum* per instance and the
        drain has to combine rather than overwrite."""
        return any(
            loop.spatial and loop.rank in reduced and d_out <= depth[loop.level] < d_in
            for loop in nest
        )

    def copy(ds, d_out, pe_out, origin_out, d_in, pe_in, origin_in, *, store):
        """One tile transfer between consecutive keepers, run by run."""
        sub = tiles[d_in][ds.name]
        # The moved block is the *inner* tile, seen at each end: the outer level
        # strides it by its own packing and holds only a window of it, the inner
        # level by that tile's own.
        pack = packing[d_out][ds.name]
        offsets, length = _run_offsets(
            [(n, st) for n, (_s, st) in zip(sub, pack)], packing[d_in][ds.name]
        )
        outer, inner = tile_at(d_out, ds.name, pe_out), tile_at(d_in, ds.name, pe_in)
        anchor = sum(
            (a - b) * st for a, b, (_s, st) in zip(origin_in, origin_out, pack)
        )
        who = (
            f"'{ds.name}' between levels '{binding.levels[d_out]}' and "
            f"'{binding.levels[d_in]}'"
        )
        combine = store and fans(d_out, d_in)
        for out_at, in_at in offsets:
            if combine:
                planner.reduce_run(
                    inner, in_at, outer, anchor + out_at, length, who, pe_in
                )
            elif store:
                planner.copy_run(
                    inner, in_at, outer, anchor + out_at, length, who, pe_in
                )
            else:
                planner.copy_run(
                    outer, anchor + out_at, inner, in_at, length, who, pe_in
                )

    # A dataspace the instruction reads at some access position must arrive; one
    # it only writes is drained but never filled.
    read = _reads(binding)

    current: dict = {}  # (depth, dataspace, instance) -> the origin resident there
    home: dict = {}  # ... -> (depth, instance, origin) it was filled from
    for point in itertools.product(*(range(loop.factor) for loop in nest)):
        outer: list = [{} for _ in range(n_levels + 1)]
        for i, loop in enumerate(nest):
            for d in range(depth[loop.level] + 1, n_levels + 1):
                outer[d][loop.rank] = outer[d].get(loop.rank, 0) + point[i] * weights[i]
        origin = [
            {ds.name: _project(ds, outer[d]) for ds in binding.dataspaces}
            for d in range(n_levels + 1)
        ]
        pe = [0] * (n_levels + 1)
        for d in range(n_levels + 1):
            for i, loop in enumerate(nest):
                if loop.spatial and depth[loop.level] < d:
                    pe[d] = pe[d] * loop.factor + point[i]

        for ds in binding.dataspaces:  # drain, innermost first
            if ds.role != "output":
                continue
            for d in reversed(chain[ds.name][1:]):
                key = (d, ds.name, pe[d])
                if key in current and current[key] != origin[d][ds.name]:
                    d_out, pe_out, origin_out = home.pop(key)
                    copy(
                        ds,
                        d_out,
                        pe_out,
                        origin_out,
                        d,
                        pe[d],
                        current.pop(key),
                        store=True,
                    )
        for ds in binding.dataspaces:  # fill, outermost first
            keepers = chain[ds.name]
            for idx in range(1, len(keepers)):
                d_out, d = keepers[idx - 1], keepers[idx]
                key = (d, ds.name, pe[d])
                if current.get(key) == origin[d][ds.name]:
                    continue
                if read[ds.name]:
                    copy(
                        ds,
                        d_out,
                        pe[d_out],
                        origin[d_out][ds.name],
                        d,
                        pe[d],
                        origin[d][ds.name],
                        store=False,
                    )
                current[key] = origin[d][ds.name]
                home[key] = (d_out, pe[d_out], origin[d_out][ds.name])

        locs, offsets = [], []
        for name in binding.operands:
            d = chain[name][-1]
            locs.append(tile_at(d, name, pe[d]))
            offsets.append(
                sum(
                    (a - b) * st
                    for a, b, (_s, st) in zip(
                        origin[n_levels][name],
                        origin[d][name],
                        packing[d][name],
                    )
                )
            )
        _compute_step(planner, binding, locs, offsets, pe[n_levels])

    for key in sorted(current, key=lambda k: -k[0]):  # flush, innermost first
        d, name, pe_in = key
        if spaces[name].role != "output":
            continue
        d_out, pe_out, origin_out = home[key]
        copy(
            spaces[name], d_out, pe_out, origin_out, d, pe_in, current[key], store=True
        )


def assemble(mapping: Mapping, binding: Binding):
    """Assemble ``mapping`` into a runnable program and the σ it prescribes.

    Returns ``(CompiledProgram, Schedule)``. The program is ordinary — it prints,
    runs on the functional simulator and reports cycles like a compiled one — and
    the σ carries what only the mapping knew: which spatial instance runs each
    epoch, and in what order. ``program.check(sigma=schedule)`` is then the whole
    constraint system evaluated on an imported mapping, which is the point.

    The nest is walked into ``_Planner``'s pass 1 — the same pass a match and an
    ``@expand`` body go through — so liveness, best-fit allocation, Belady spilling
    and emission are shared rather than reimplemented here. That is what makes the
    capacity question answerable: a mapping whose working set does not fit is an
    allocation error naming the buffer, not a bump-allocated address past its end."""
    violations = mapping.check(binding)
    if violations:
        raise AssemblyError(
            "this mapping cannot be assembled:\n  "
            + "\n  ".join(v.message for v in violations)
        )
    planner = _Planner(binding.isa)
    io = binding.isa.buffers[binding.buffers[binding.levels[0]]]
    # Standalone: the program *is* this mapping, so its dataspaces are its I/O and
    # this module places them — densely, in the host ABI, which is the one packing
    # the compiler does not get to choose.
    root = {
        ds.name: planner.at(Tile(tuple(ds.shape)), io, residence(_packing(ds.shape)))
        for ds in binding.dataspaces
    }
    read = _reads(binding)
    for ds in binding.dataspaces:
        if ds.role == "output" and read[ds.name]:
            # The accumulator's initial value. Making it a preloaded constant is what
            # lets *definedness* see that an accumulating instruction's first read is
            # defined.
            planner.constants.append(
                (root[ds.name], np.zeros(ds.shape, dtype=np.float32))
            )
    _lower(mapping, binding, planner, root)
    program = planner.finish(
        [root[ds.name] for ds in binding.dataspaces if ds.role == "input"],
        [
            (root[ds.name], tuple(ds.shape), ds.name)
            for ds in binding.dataspaces
            if ds.role == "output"
        ],
    )
    # σ's spatial axis rides on the program now (``CompiledProgram.instances``), so
    # the imported assignment is not this entry's private knowledge: a mapped site
    # compiled through ``compile_program`` keeps it too.
    return program, program.schedule()


# ==========================================================================#
# The export direction: our machine and one operation, as Timeloop's inputs
# ==========================================================================#


def _reduction_ranks(binding: Binding, ranks) -> list:
    """The ranks no output projection mentions — the reduction ones. Fanned across
    instances they leave a partial sum per instance (constraint 7)."""
    written = {
        rank
        for ds in binding.dataspaces
        if ds.role == "output"
        for dim in ds.projection
        for rank in dim
    }
    return sorted(set(ranks) - written)


def _problem(name: str, spaces, ranks: dict) -> dict:
    """Timeloop's problem file: the operation, as a shape plus an instance.

    A projection term is ``[rank]``, or ``[rank, coefficient]`` where the rank is
    scaled — which is how a convolution's stride and dilation reach the mapper, and
    the reason the halo it computes is the halo we compute. Coefficients are named
    after their value rather than after what they mean (``Coef2``), because a
    projection carries the number and not the story behind it."""
    scales = sorted(
        {c for ds in spaces for dim in ds.projection for c in dim.values() if c != 1}
    )
    coeff = {c: f"Coef{c}" for c in scales}
    shape: dict = {"name": name, "dimensions": list(ranks)}
    if coeff:
        shape["coefficients"] = [
            {"name": coeff[c], "default": c} for c in sorted(coeff)
        ]
    shape["data-spaces"] = [
        {
            "name": ds.name,
            "projection": [
                [[r] if c == 1 else [r, coeff[c]] for r, c in dim.items()]
                for dim in ds.projection
            ],
            **({"read-write": True} if ds.role == "output" else {}),
        }
        for ds in spaces
    ]
    return {
        "problem": {
            "shape": shape,
            "instance": dict(ranks) | {v: k for k, v in coeff.items()},
        }
    }


def _architecture(binding: Binding, gaps: list) -> dict:
    """Timeloop's architecture file, as far as an ISA declares one.

    A storage level per ``binding.levels``, outermost first, with the compute at the
    leaf — the same nesting the levels already are. What an ISA states is the
    *capacity* of each buffer and the width of a word; what it does not state is
    everything else Timeloop prices with, and those are recorded in ``gaps`` rather
    than filled with a plausible default, because a default here is a number the
    mapper will optimize against and nobody wrote down."""
    isa = binding.isa
    bits = _word_bits(isa.buffers[binding.buffers[binding.levels[-1]]])
    node: dict = {
        "name": "PE",
        "local": [
            {
                "name": binding.compute,
                "class": "compute",
                "attributes": {"word-bits": bits},
            }
        ],
    }
    for level in reversed(binding.levels):
        buf = isa.buffers[binding.buffers[level]]
        word = _word_bits(buf)
        node = {
            "name": f"{level}_level",
            "local": [
                {
                    "name": level,
                    "class": "DRAM" if buf.is_global else "SRAM",
                    "attributes": {
                        "depth": buf.capacity,
                        "block-size": buf.slot_size,
                        "word-bits": word,
                        "width": word * buf.slot_size,
                    },
                }
            ],
            "subtree": [node],
        }
        gaps.append(
            f"level '{level}': no fanout (meshX/meshY) — the ISA declares no "
            f"instance count for '{buf.name}', the same gap constraint 6 has"
        )
        gaps.append(
            f"level '{level}': no bandwidth or energy — Timeloop prices a mapping "
            f"with both and an ISA states neither (energy is Accelergy's table)"
        )
    return {"architecture": {"version": "0.3", "subtree": [node]}}


def _word_bits(buf) -> int:
    return buf.kind.dtype.primitive_width


def to_timeloop(binding: Binding, op=None, *, ranks=None) -> dict:
    """This machine and one operation, as the three files Timeloop reads.

    Returns ``{"problem", "architecture", "constraints", "gaps"}`` — parsed
    structures, not YAML text, mirroring ``from_timeloop``, which takes the parsed
    form so this module has no serializer and no dependency of its own.

    With ``op`` the problem comes from the source operation (``_derive``); without
    one, from ``binding.dataspaces`` and an explicit ``ranks``. This is the seam
    that closes the loop: a ``mapping_for`` driver exports here, runs the mapper,
    and hands the result straight back to ``from_timeloop`` — the names line up
    because both ends use the operation's own.

    **Two of this module's refusals become search constraints**, which is the point
    of exporting them rather than only checking them. The intrinsic tile is pinned
    as the innermost level's factors, so a mapping whose innermost loops the
    hardware cannot perform is never proposed; and every reduction rank is pinned to
    one spatially at every level, so a mapping needing a reducing network the
    machine does not declare is never proposed either. A mapper obeying both cannot
    hit those two refusals at all.

    **``gaps`` is the honest part.** It lists what the export could not derive, one
    line each, instead of writing a default the mapper would then optimize against:
    fanout and bandwidth and energy per level, and the one obligation that has no
    constraint form here — that the intrinsic must sit in its innermost keeper the
    way the instruction's access reads it, which depends on the tile shapes the
    mapper picks rather than on anything declarable up front."""
    if op is not None:
        bound, ranks = _derive(op)
        binding = replace(binding, dataspaces=tuple(ds for ds, _v in bound))
        name = op.operation.name.split(".")[-1]
    else:
        if not binding.dataspaces or ranks is None:
            raise AssemblyError(
                "to_timeloop: without a source op the binding has to declare its "
                "dataspaces and the caller has to supply `ranks` — a problem "
                "instance cannot be recovered from the projections alone"
            )
        name = binding.compute
    gaps: list = []
    architecture = _architecture(binding, gaps)
    targets = [
        {
            "target": binding.levels[-1],
            "type": "temporal",
            "factors": " ".join(f"{r}={binding.body.get(r, 1)}" for r in ranks),
        }
    ]
    reduced = _reduction_ranks(binding, ranks)
    if reduced:
        targets += [
            {
                "target": level,
                "type": "spatial",
                "factors": " ".join(f"{r}=1" for r in reduced),
            }
            for level in binding.levels
        ]
    gaps.append(
        "the residence obligation has no constraint form: whether the intrinsic "
        "sits in its innermost keeper the way the instruction reads it depends on "
        "the tile shapes the mapper picks, so a mapping violating it is refused on "
        "import rather than excluded from the search"
    )
    return {
        "problem": _problem(name, binding.dataspaces, ranks),
        "architecture": architecture,
        "constraints": {"architecture_constraints": {"targets": targets}},
        "gaps": gaps,
    }


def lower_site(m, planner) -> None:
    """Lower one matched source op to the tile run an imported mapping prescribes.

    This is the entry the repeal buys. A mapped site is **not** one instruction's
    worth of work: the source op is whatever the program wrote, the mapping says how
    to cut it up, and the instruction performs one innermost tile of the result. So
    Stage 2's exact fit does not apply here — it is the very thing tiling replaces —
    and the shapes flow the other way instead: the mapping's factors give the
    instruction its extents, and the *source op's tensors* are the dataspaces.

    **Which is where constraint 1 finally closes.** ``Mapping.check`` verifies that
    the nest's factors multiply to the declared extent of every rank; the walk then
    requires the outermost level's tile to be the whole dataspace; and here the
    dataspace has to be the tensor the source program actually passes. Chain those
    and the nest provably enumerates the source op's own iteration domain exactly
    once — the obligation a compiled wire could never carry, because it does not
    know what it was supposed to compute.

    The correspondence is positional and stated rather than guessed: the input
    dataspaces are the op's data operands in order, the output dataspace is its
    result. Step 6 makes it automatic by deriving the dataspaces from the op."""
    mapping, binding = m.mapping
    isa = binding.isa
    who = f"{m.instruction.name} at a mapped site"
    if binding.dataspaces:
        # Declared by hand: the correspondence is positional, and stated — the input
        # dataspaces are the op's data operands in order, the output is its result.
        ins = [ds for ds in binding.dataspaces if ds.role == "input"]
        outs = [ds for ds in binding.dataspaces if ds.role == "output"]
        if len(ins) != len(m.operand_values) or len(outs) != 1:
            raise AssemblyError(
                f"{who}: the binding declares {len(ins)} input and {len(outs)} output "
                f"dataspace(s), but the source op has {len(m.operand_values)} "
                f"operand(s) and one result"
            )
        bound = list(zip(ins + outs, list(m.operand_values) + [m.result_value]))
    else:
        bound, _ranks = _derive(m.result_value.owner)
        binding = replace(binding, dataspaces=tuple(ds for ds, _v in bound))
    source = {ds.name: _static_shape(_canon(v)) for ds, v in bound}
    violations = mapping.check(binding, source=source)
    if violations:
        raise AssemblyError(
            f"{who}: this mapping cannot be assembled:\n  "
            + "\n  ".join(v.message for v in violations)
        )

    level0 = isa.buffers[binding.buffers[binding.levels[0]]]
    read = _reads(binding)
    root = {}
    for ds, value in bound:
        value = _canon(value)
        want = _root_residence(planner, value, level0, ds.shape)
        root[ds.name] = (
            planner.make_loc(value, level0, want)
            if ds.role == "output"
            else planner.bring_to(value, level0, want, f"{who}: '{ds.name}'")
        )
        if ds.role == "output" and read[ds.name]:
            planner.constants.append(
                (root[ds.name], np.zeros(ds.shape, dtype=np.float32))
            )
    _lower(mapping, binding, planner, root)


def _root_residence(planner, value, buf, shape) -> tuple:
    """The residence a mapped site's level-0 block should have in ``buf``.

    Whatever the value already has there, if it has one: a mapping says what is
    resident and never how it is packed, so imposing a packing of our own would be a
    relayout nobody asked for. Otherwise the dense one, which is what the surrounding
    program will hand it anyway."""
    for name, res in planner.loc.get(value, {}):
        if name == buf.name:
            return res
    return residence(_packing(shape))
