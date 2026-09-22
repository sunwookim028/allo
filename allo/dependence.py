# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The legality rule behind ``s.dependence``, and the obligation it leaves.

The rule is one-sided on purpose. A dependence that **is** provable at a
distance the claim denies makes the claim false, and is refused. A dependence
that is **not** provable is what the primitive exists to assert away, and is
accepted. So this module can only ever catch a demonstrably false claim: it
never confirms a true one, and it must never refuse a claim for lack of proof
-- that would make the primitive useless, which is worse than no rule at all.

What survives the rule is an :class:`Obligation`, not a result. See
``docs/source/developer/extending_allo.rst``."""

import warnings
from dataclasses import dataclass

from ._mlir.ir import (
    AffineAddExpr,
    AffineConstantExpr,
    AffineDimExpr,
    AffineMapAttr,
    AffineMulExpr,
    AffineSymbolExpr,
    ArrayAttr,
    Block,
    BoolAttr,
    DictAttr,
    IntegerAttr,
    StringAttr,
)
from ._mlir.exceptions import AlloError

FUNC_OP = "func.func"
LOOP_OPS = ("affine.for", "scf.for")
GUARD_OPS = ("affine.if", "scf.if")
ALLOC_OPS = ("memref.alloc", "memref.alloca")
AFFINE_ACCESS_OPS = ("affine.load", "affine.store")
READ_OPS = {"affine.load": 0, "memref.load": 0}
WRITE_OPS = {"affine.store": 1, "memref.store": 1}

ADD_OP = "arith.addi"
SUB_OP = "arith.subi"
MUL_OP = "arith.muli"
CONSTANT_OP = "arith.constant"
APPLY_OP = "affine.apply"
VALUE_PRESERVING_OPS = (
    "arith.index_cast",
    "arith.index_castui",
    "arith.extsi",
    "arith.extui",
)

EVERY_DISTANCE = "every distance"
UNDECLARED_PREMISE = "premise not declared (s.dependence(..., because=...))"


@dataclass(frozen=True)
class Claim:
    """One ``s.dependence`` call, as the legality rule sees it."""

    target: str
    dep_type: str = "inter"
    direction: str = None
    distance: int = None
    dependent: bool = False
    dep_class: str = None
    because: str = None

    @property
    def pragma(self):
        """The claim as Vitis reads it, in the emitter's field order."""
        fields = [f"variable={self.target}"]
        if self.dep_class is not None:
            fields.append(self.dep_class)
        fields.append(self.dep_type)
        if self.direction is not None:
            fields.append(self.direction)
        if self.distance is not None:
            fields.append(f"distance={self.distance}")
        fields.append("true" if self.dependent else "false")
        return " ".join(fields)

    def covers(self, direction):
        return self.direction is None or self.direction == direction

    def denies(self, distance):
        """Whether the claim says that no dependence runs this many iterations
        apart. Only a denied distance can make the claim false."""
        if self.dep_type == "intra":
            return distance == 0 and not self.dependent
        if distance < 1:
            return False
        if not self.dependent:
            return True
        return self.distance is not None and distance < self.distance

    @property
    def nearest_denied_distance(self):
        """The shortest distance this claim denies, which is the one to report
        for a pair of accesses that alias however far apart they are."""
        for distance in range(0, (self.distance or 1) + 1):
            if self.denies(distance):
                return distance
        return None

    def repair(self, found):
        nearest = min(proof.distance for proof in found)
        if self.dep_type == "intra":
            return (
                "legal if: dependent=True, which keeps the two accesses ordered "
                "within an iteration; or a body in which they cannot reach the "
                "same element."
            )
        direction = "" if self.direction is None else f", direction={self.direction!r}"
        return (
            f'legal if: dep_type="inter", dependent=True, distance={nearest}'
            f"{direction} -- the loop still pipelines at II={nearest}; or a "
            "kernel in which these two accesses cannot reach the same element. "
            f"No claim that denies distance {nearest} is legal here."
        )


class Loops:
    """Positional names for loops, so that a linear form's terms are hashable.
    MLIR operations compare but do not hash, and a term has to key on one."""

    def __init__(self):
        self._loops = []

    def name(self, loop):
        for position, other in enumerate(self._loops):
            if other == loop:
                return position
        self._loops.append(loop)
        return len(self._loops) - 1

    def op(self, position):
        return self._loops[position]

    def axis(self, position):
        loop = self._loops[position]
        if "loop_name" in loop.attributes:
            return StringAttr(loop.attributes["loop_name"]).value
        return f"loop{position}"


@dataclass(frozen=True)
class Linear:
    """``constant + sum(coefficient * induction variable)``: the only shape of
    subscript this rule can reason about. Anything else is ``None``, which is
    the accepting answer."""

    constant: int
    terms: tuple = ()

    def coefficient(self, loop):
        for other, coefficient in self.terms:
            if other == loop:
                return coefficient
        return 0

    @property
    def support(self):
        return tuple(loop for loop, _ in self.terms)


def _linear(constant, coefficients):
    return Linear(constant, tuple(sorted((k, v) for k, v in coefficients.items() if v)))


def add(left, right):
    if left is None or right is None:
        return None
    coefficients = dict(left.terms)
    for loop, coefficient in right.terms:
        coefficients[loop] = coefficients.get(loop, 0) + coefficient
    return _linear(left.constant + right.constant, coefficients)


def scale(form, factor):
    if form is None:
        return None
    return _linear(form.constant * factor, {loop: c * factor for loop, c in form.terms})


def multiply(left, right):
    if left is None or right is None:
        return None
    if not right.terms:
        return scale(left, right.constant)
    if not left.terms:
        return scale(right, left.constant)
    return None


def _integer_value(attribute):
    try:
        return IntegerAttr(attribute).value
    except (ValueError, TypeError):
        return None


def _induction_variable_term(value, block, loops):
    owner = block.owner
    if owner.name not in LOOP_OPS or not block.arguments:
        return None
    if value != block.arguments[0]:
        return None
    return Linear(0, ((loops.name(owner), 1),))


def form_of_value(value, loops):
    """The linear form of a computed index, or None when the computation is not
    affine in the enclosing induction variables."""
    owner = value.owner
    if isinstance(owner, Block):
        return _induction_variable_term(value, owner, loops)
    op = owner.operation
    if op.name == CONSTANT_OP:
        constant = _integer_value(op.attributes["value"])
        return None if constant is None else Linear(constant)
    if op.name in VALUE_PRESERVING_OPS:
        return form_of_value(op.operands[0], loops)
    if op.name == ADD_OP:
        return add(*(form_of_value(operand, loops) for operand in op.operands))
    if op.name == SUB_OP:
        left, right = (form_of_value(operand, loops) for operand in op.operands)
        return add(left, scale(right, -1))
    if op.name == MUL_OP:
        return multiply(*(form_of_value(operand, loops) for operand in op.operands))
    if op.name == APPLY_OP:
        return form_of_affine_map(op.attributes["map"], op.operands, loops)[0]
    return None


def form_of_expression(expression, operands, dims):
    if AffineConstantExpr.isinstance(expression):
        return Linear(AffineConstantExpr(expression).value)
    if AffineDimExpr.isinstance(expression):
        return operands[AffineDimExpr(expression).position]
    if AffineSymbolExpr.isinstance(expression):
        return operands[dims + AffineSymbolExpr(expression).position]
    if AffineAddExpr.isinstance(expression):
        binary = AffineAddExpr(expression)
        return add(
            form_of_expression(binary.lhs, operands, dims),
            form_of_expression(binary.rhs, operands, dims),
        )
    if AffineMulExpr.isinstance(expression):
        binary = AffineMulExpr(expression)
        return multiply(
            form_of_expression(binary.lhs, operands, dims),
            form_of_expression(binary.rhs, operands, dims),
        )
    return None


def form_of_affine_map(attribute, operands, loops):
    affine_map = AffineMapAttr(attribute).value
    forms = [form_of_value(operand, loops) for operand in operands]
    return tuple(
        form_of_expression(result, forms, affine_map.n_dims)
        for result in affine_map.results
    )


def _constant_bound(attribute):
    affine_map = AffineMapAttr(attribute).value
    if affine_map.n_dims or affine_map.n_symbols or len(affine_map.results) != 1:
        return None
    result = affine_map.results[0]
    if not AffineConstantExpr.isinstance(result):
        return None
    return AffineConstantExpr(result).value


def _bounds(loop):
    if loop.name == "affine.for":
        return (
            _constant_bound(loop.attributes["lowerBoundMap"]),
            _constant_bound(loop.attributes["upperBoundMap"]),
            _integer_value(loop.attributes["step"]),
        )
    forms = [form_of_value(operand, Loops()) for operand in loop.operands[:3]]
    if any(form is None or form.terms for form in forms):
        return None, None, None
    return tuple(form.constant for form in forms)


def step_of(loop):
    """How far the induction variable moves in one iteration, which is what
    separates an iteration distance from a subscript offset."""
    return _bounds(loop)[2]


def trip_count(loop):
    """The loop's constant trip count in iterations, or None when the bounds are
    not constants -- which means an execution may run it as many times as it
    likes, so it rules no witness out."""
    lower, upper, step = _bounds(loop)
    if lower is None or upper is None or not step:
        return None
    return max(0, -(-(upper - lower) // step))


def _render(form, loops):
    if form is None:
        return "?"
    parts = [
        loops.axis(loop) if coefficient == 1 else f"{coefficient}*{loops.axis(loop)}"
        for loop, coefficient in form.terms
    ]
    if form.constant or not parts:
        parts.append(str(form.constant))
    return " + ".join(parts).replace("+ -", "- ")


@dataclass(frozen=True)
class Access:
    """One read or write of the claimed array, with the loops that enclose it."""

    op: object
    is_write: bool
    order: int
    loops: tuple
    guarded: bool
    subscripts: tuple

    @property
    def kind(self):
        return "write" if self.is_write else "read"

    @property
    def support(self):
        return tuple(
            loop
            for form in self.subscripts
            if form is not None
            for loop in form.support
        )

    def describe(self, target, loops):
        subscripts = ", ".join(_render(form, loops) for form in self.subscripts)
        return f"{self.kind:5s} {target}[{subscripts}]   ({self.op.name})"


def _children(op):
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield child.operation


def _access(op, memref, enclosing, guarded, order, loops):
    position = READ_OPS.get(op.name, WRITE_OPS.get(op.name))
    if position is None or op.operands[position] != memref:
        return None
    indices = list(op.operands)[position + 1 :]
    if op.name in AFFINE_ACCESS_OPS:
        subscripts = form_of_affine_map(op.attributes["map"], indices, loops)
    else:
        subscripts = tuple(form_of_value(index, loops) for index in indices)
    return Access(op, op.name in WRITE_OPS, order, enclosing, guarded, subscripts)


def accesses_of(loop, memref, loops):
    """Every read and write of ``memref`` inside ``loop``, in program order,
    tagged with the loops that enclose it and whether a guard stands over it."""
    found = []

    def visit(op, enclosing, guarded):
        for child in _children(op):
            if child.name in LOOP_OPS:
                visit(child, enclosing + (loops.name(child),), guarded)
            elif child.name in GUARD_OPS:
                visit(child, enclosing, True)
            else:
                access = _access(child, memref, enclosing, guarded, len(found), loops)
                if access is not None:
                    found.append(access)

    visit(loop, (loops.name(loop),), False)
    return found


def aliasing_distance(earlier, later, axis, axis_step):
    """The inter-iteration distance at which these two accesses provably reach
    the same element, for every iteration of the surrounding loops:
    ``EVERY_DISTANCE`` when every distance does, ``None`` when no distance is
    provable. ``None`` is the accepting answer, and says nothing about whether
    a dependence is in fact there."""
    if not axis_step:
        return None
    required = None
    for before, after in zip(earlier.subscripts, later.subscripts):
        if before is None or after is None:
            return None
        for loop in set(before.support) | set(after.support):
            if loop != axis and before.coefficient(loop) != after.coefficient(loop):
                return None
        stride = after.coefficient(axis)
        if before.coefficient(axis) != stride:
            return None
        offset = before.constant - after.constant
        if stride == 0:
            if offset:
                return None
            continue
        travel = stride * axis_step
        if offset % travel:
            return None
        if required is not None and required != offset // travel:
            return None
        required = offset // travel
    return EVERY_DISTANCE if required is None else required


def _direction(earlier, later, distance):
    if distance < 0 or (distance == 0 and earlier.order >= later.order):
        return None
    if earlier.is_write and later.is_write:
        return "WAW"
    if earlier.is_write:
        return "RAW"
    if later.is_write:
        return "WAR"
    return None


def _witnessable(earlier, later, distance, axis, loops):
    """Whether an execution exists that runs both accesses this many iterations
    apart. A subscript that strides over an inner loop needs that loop to have
    the same iteration space in both outer iterations, which constant bounds
    give and nothing else here establishes."""
    for loop in set(earlier.loops) | set(later.loops):
        if loop != axis and trip_count(loops.op(loop)) == 0:
            return False
    inside = set(earlier.loops) | set(later.loops)
    for loop in set(earlier.support) | set(later.support):
        if loop != axis and loop in inside and not trip_count(loops.op(loop)):
            return False
    trips = trip_count(loops.op(axis))
    return trips is None or trips > distance


@dataclass(frozen=True)
class Proof:
    """A dependence the compiler can prove: a pair of accesses that reach the
    same element a fixed number of iterations apart."""

    direction: str
    distance: int
    axis: str
    earlier: Access
    later: Access
    target: str
    loops: object

    def __str__(self):
        return (
            f"a {self.direction} dependence runs at distance {self.distance} on "
            f"axis {self.axis}, between\n"
            f"      {self.earlier.describe(self.target, self.loops)}\n"
            f"      {self.later.describe(self.target, self.loops)}"
        )


def _proof(earlier, later, claim, axis, loops):
    if earlier.guarded or later.guarded:
        return None
    distance = aliasing_distance(earlier, later, axis, step_of(loops.op(axis)))
    if distance is None:
        return None
    if distance == EVERY_DISTANCE:
        distance = claim.nearest_denied_distance
        if distance is None:
            return None
    direction = _direction(earlier, later, distance)
    if direction is None or not claim.covers(direction):
        return None
    if not claim.denies(distance):
        return None
    if not _witnessable(earlier, later, distance, axis, loops):
        return None
    return Proof(
        direction, distance, loops.axis(axis), earlier, later, claim.target, loops
    )


def proofs(loop, memref, claim):
    """Every dependence on ``memref`` inside ``loop`` that is provable at a
    distance ``claim`` denies. Empty means only that no proof was found, which
    the rule treats as legal -- see the module docstring."""
    loops = Loops()
    axis = loops.name(loop.operation)
    accesses = accesses_of(loop.operation, memref, loops)
    return [
        proof
        for earlier in accesses
        for later in accesses
        if (proof := _proof(earlier, later, claim, axis, loops)) is not None
    ]


class DependenceError(AlloError):
    """A ``dependence`` claim the IR disproves."""

    def __init__(self, where, claim, found, after=None):
        self.claim = claim
        self.proofs = found
        blame = f" after s.{after}()" if after else ""
        body = "\n    ".join(str(proof) for proof in found)
        super().__init__(
            f"dependence: `{claim.pragma}` on {where} is provably false{blame}:"
            f"\n    {body}\n{claim.repair(found)}"
        )


class UndeclaredPremise(UserWarning):
    """An accepted claim whose author has not said what makes it true."""


@dataclass(frozen=True)
class Obligation:
    """What an accepted claim still rests on. The rule refuses a provable lie
    and confirms nothing, so every claim it accepts leaves one of these."""

    where: str
    pragma: str
    premise: str = None

    def __str__(self):
        return f"{self.where}: {self.pragma}\n    {self.premise or UNDECLARED_PREMISE}"


def check(loop, memref, claim, where, after=None):
    found = proofs(loop, memref, claim)
    if found:
        raise DependenceError(where, claim, found, after=after)


def accept(loop, memref, claim, where, stacklevel=3):
    """Run the rule, then record what it could not settle."""
    check(loop, memref, claim, where)
    if claim.because is None:
        warnings.warn(
            f"dependence: `{claim.pragma}` on {where} is accepted but not "
            "proven. The rule refuses only a claim it can disprove, and this "
            "one it cannot; no simulator can see a violation either. Say what "
            "makes it true with s.dependence(..., because=...) -- discharging "
            "it is yours, outside the compiler.",
            UndeclaredPremise,
            stacklevel=stacklevel,
        )
    return Obligation(where, claim.pragma, claim.because)


def _functions(module):
    for op in module.body.operations:
        if op.operation.name == FUNC_OP:
            yield op.operation


def _descendants(op):
    for child in _children(op):
        yield child
        yield from _descendants(child)


def _axis_of(loop):
    if "loop_name" in loop.attributes:
        return StringAttr(loop.attributes["loop_name"]).value
    return "?"


def _claim_of(entry):
    def text(key):
        return StringAttr(entry[key]).value if key in entry else None

    return Claim(
        target=text("variable"),
        dep_type=text("type") or "inter",
        direction=text("direction"),
        distance=(
            IntegerAttr(entry["distance"]).value if "distance" in entry else None
        ),
        dependent="dependent" in entry and BoolAttr(entry["dependent"]).value,
        dep_class=text("class"),
        because=text("because"),
    )


def _memref_of(func, entry):
    if "arg_index" in entry:
        arguments = list(func.regions[0].blocks[0].arguments)
        index = IntegerAttr(entry["arg_index"]).value
        return arguments[index] if index < len(arguments) else None
    name = StringAttr(entry["variable"]).value
    for op in _descendants(func):
        if op.name in ALLOC_OPS and "name" in op.attributes:
            if StringAttr(op.attributes["name"]).value == name:
                return op.results[0]
    return None


def recorded_claims(module):
    """Every claim already in the IR, read back from the same loop attributes
    the emitter reads, so a re-check needs nothing the schedule remembers."""
    for func in _functions(module):
        where = StringAttr(func.attributes["sym_name"]).value
        for op in _descendants(func):
            if op.name not in LOOP_OPS or "dependence" not in op.attributes:
                continue
            for attribute in ArrayAttr(op.attributes["dependence"]):
                entry = DictAttr(attribute)
                memref = _memref_of(func, entry)
                if memref is not None:
                    yield f"{where}:{_axis_of(op)}", op, memref, _claim_of(entry)


def recheck(module, after=None):
    """Re-run the rule over every claim in the module. A later primitive can
    turn a claim that was unprovable into one that is provably false."""
    for where, loop, memref, claim in recorded_claims(module):
        check(loop, memref, claim, where, after=after)
