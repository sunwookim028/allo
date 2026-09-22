# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one instruction word of a fixed-function target can carry, as a checkable
property of an Allo schedule. See ``docs/source/developer/extending_allo.rst``."""

from dataclasses import dataclass

from ._mlir.dialects import (
    affine as affine_d,
    func as func_d,
    memref as memref_d,
    scf as scf_d,
)
from ._mlir.ir import AffineMapAttr, Block, StringAttr
from ._mlir.exceptions import AlloError

LOOP_OPS = (affine_d.AffineForOp,)
GUARD_OPS = (affine_d.AffineIfOp, scf_d.IfOp)
AFFINE_ACCESS_OPS = (affine_d.AffineLoadOp, affine_d.AffineStoreOp)
UNRESTRICTED_ACCESS_OPS = (memref_d.LoadOp, memref_d.StoreOp)


@dataclass(frozen=True)
class Encoding:
    """The encoding budgets of one target. Every field left unset means the
    target places no limit, so the default constrains nothing."""

    name: str = "unconstrained"
    address_terms: int = None
    loop_depth: int = None
    has_predicated_fields: bool = True
    requires_static_trip_counts: bool = False
    requires_affine_addressing: bool = False


@dataclass(frozen=True)
class Violation:
    rule: str
    where: str
    found: str
    repair: str

    def __str__(self):
        return f"{self.rule:24s} {self.where}\n{' ' * 26}{self.found}\n{' ' * 26}repair: {self.repair}"


class EncodingError(AlloError):
    def __init__(self, encoding, found, after=None):
        self.encoding = encoding
        self.violations = found
        self.after = after
        body = "\n".join(str(v) for v in found)
        blame = f" after s.{after}()" if after else ""
        super().__init__(
            f"{len(found)} site(s) are not encodable on {encoding.name!r}"
            f"{blame}:\n{body}"
        )


@dataclass(frozen=True)
class Site:
    """One operation, together with the affine loops that enclose it."""

    func: str
    loops: tuple
    op: object

    @property
    def induction_variables(self):
        return tuple(loop.induction_variable for loop in self.loops)

    @property
    def where(self):
        axes = [_axis_name(loop) for loop in self.loops]
        return f"{self.func}:{'/'.join(axes)}" if axes else self.func

    @property
    def what(self):
        return self.op.operation.name


def _axis_name(loop):
    if "loop_name" in loop.attributes:
        return StringAttr(loop.attributes["loop_name"]).value
    return "?"


def _sites(module):
    def walk(func, ops, loops):
        for op in ops:
            yield Site(func, loops, op)
            if isinstance(op, LOOP_OPS):
                yield from walk(func, op.body.operations, loops + (op,))
            elif isinstance(op, GUARD_OPS):
                yield from walk(func, op.then_block.operations, loops)
                if op.else_block is not None:
                    yield from walk(func, op.else_block.operations, loops)

    for op in module.body.operations:
        if isinstance(op, func_d.FuncOp):
            name = StringAttr(op.attributes["sym_name"]).value
            yield from walk(name, op.entry_block.operations, ())


def _is_one_of(value, values):
    return any(value == other for other in values)


def varying_induction_variables(value, induction_variables):
    """The enclosing induction variables that ``value`` is derived from."""
    found, seen, stack = [], [], [value]
    while stack:
        current = stack.pop()
        owner = current.owner
        if isinstance(owner, Block):
            if _is_one_of(current, induction_variables) and not _is_one_of(
                current, found
            ):
                found.append(current)
            continue
        if _is_one_of(owner, seen):
            continue
        seen.append(owner)
        stack.extend(owner.operands)
    return tuple(found)


def address_terms(site):
    """The distinct induction variables one address generator must stride over
    to issue this access."""
    terms = []
    for index in site.op.indices:
        for iv in varying_induction_variables(index, site.induction_variables):
            if not _is_one_of(iv, terms):
                terms.append(iv)
    return tuple(terms)


def is_constant_trip_count(loop):
    bounds = (
        AffineMapAttr(loop.attributes["lowerBoundMap"]).value,
        AffineMapAttr(loop.attributes["upperBoundMap"]).value,
    )
    return all(
        m.n_dims == 0 and m.n_symbols == 0 and len(m.results) == 1 for m in bounds
    )


def guard_condition_operands(op):
    if isinstance(op, scf_d.IfOp):
        return [op.condition]
    return list(op.operands)


def predicated_field_violations(sites, encoding):
    if encoding.has_predicated_fields:
        return
    for site in (s for s in sites if isinstance(s.op, GUARD_OPS) and s.loops):
        varying = [
            iv
            for operand in guard_condition_operands(site.op)
            for iv in varying_induction_variables(operand, site.induction_variables)
        ]
        if varying:
            yield Violation(
                rule="predicated-field",
                where=site.where,
                found=f"{site.what} selects between bodies on a condition that varies "
                f"with {len(varying)} enclosing induction variable(s), and this "
                f"target's instruction fields are fixed at assembly time",
                repair="index-set split the loop so that each body needs one constant "
                "field value (Allo has no peel primitive; see the brief)",
            )


def address_term_violations(sites, encoding):
    if encoding.address_terms is None:
        return
    for site in (s for s in sites if isinstance(s.op, AFFINE_ACCESS_OPS)):
        terms = address_terms(site)
        if len(terms) > encoding.address_terms:
            yield Violation(
                rule="address-terms",
                where=site.where,
                found=f"{site.what} needs {len(terms)} address terms, and one "
                f"instruction word carries {encoding.address_terms}",
                repair="re-stage the operand so fewer induction variables reach one "
                "access, or widen the address-generator budget",
            )


def affine_addressing_violations(sites, encoding):
    if not encoding.requires_affine_addressing:
        return
    for site in (s for s in sites if isinstance(s.op, UNRESTRICTED_ACCESS_OPS)):
        if not site.loops:
            continue
        yield Violation(
            rule="affine-addressing",
            where=site.where,
            found=f"{site.what} computes its index outside the affine map, so this "
            f"target's address generator cannot issue it",
            repair="write the subscript as an affine expression of the induction "
            "variables, or move the computed index out of the band",
        )


def loop_depth_violations(sites, encoding):
    if encoding.loop_depth is None:
        return
    for site in (s for s in sites if isinstance(s.op, LOOP_OPS)):
        depth = len(site.loops) + 1
        if depth > encoding.loop_depth:
            yield Violation(
                rule="loop-depth",
                where=f"{site.where}/{_axis_name(site.op)}",
                found=f"nested {depth} deep, and the sequencer's loop stack holds "
                f"{encoding.loop_depth}",
                repair="fuse or fully unroll an outer level",
            )


def static_trip_count_violations(sites, encoding):
    if not encoding.requires_static_trip_counts:
        return
    for site in (s for s in sites if isinstance(s.op, LOOP_OPS)):
        if not is_constant_trip_count(site.op):
            yield Violation(
                rule="static-trip-count",
                where=f"{site.where}/{_axis_name(site.op)}",
                found=f"bounds {AffineMapAttr(site.op.attributes['lowerBoundMap']).value}"
                f" to {AffineMapAttr(site.op.attributes['upperBoundMap']).value} are not "
                f"constant, and the sequencer's trip count is an instruction field",
                repair="split by a factor that divides the extent exactly, so no "
                "min/max bound is introduced",
            )


RULES = (
    predicated_field_violations,
    address_term_violations,
    affine_addressing_violations,
    loop_depth_violations,
    static_trip_count_violations,
)


def violations(module, encoding):
    sites = list(_sites(module))
    return [v for rule in RULES for v in rule(sites, encoding)]


def check(module, encoding):
    found = violations(module, encoding)
    if found:
        raise EncodingError(encoding, found)


UNCONSTRAINED = Encoding()

TINYTPU_ISA = Encoding(
    name="tinytpu-isa",
    address_terms=3,
    loop_depth=4,
    has_predicated_fields=False,
    requires_static_trip_counts=True,
    requires_affine_addressing=True,
)
