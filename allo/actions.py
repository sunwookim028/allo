# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An instruction as a composition of per-unit Actions.

An instruction is not a primitive here. It is an ordered list of effects, each
one belonging to a named unit and spending a named port of that unit, over
state the machine declares. Everything a consumer usually restates -- which
units an opcode reaches, how many steps each of them spends on it, which rows
of which memory it reads before it writes, and what value comes out -- is a
query over that one list.

The rule this module enforces is one-sided, in the shape ``allo.dependence``
uses: it refuses a composition it can disprove (a port no unit has, a value
nothing produces, a value crossing units with no channel to cross on, a lane
permutation that drops or duplicates an operand) and it confirms nothing else.
What an accepted composition still rests on comes back as an
:class:`Obligation` -- a reassociated reduction under rounding arithmetic, a
predicate the model cannot settle, a program contract no composition can
enforce. See ``docs/source/developer/actions.rst``.

Nothing here knows about any particular machine. ``examples/accelerator/
tinytpu_vitis`` builds a :class:`Machine` out of its ISA spec; the doc page
builds a second one, with an adder-tree reduction unit, out of nothing at all.
"""

from dataclasses import dataclass, field, replace

_UNSET = object()

READ = "read"
WRITE = "write"
COMPUTE = "compute"
EMIT = "emit"
RECEIVE = "receive"
KINDS = (READ, WRITE, COMPUTE, EMIT, RECEIVE)

PER_ROW = "row"
PER_INSTRUCTION = "instruction"

EXACT = "exact"
ROUNDING = "rounding"

#: What an action's claim is WORTH. Three different things get written the
#: same way in every design this project has read, and conflating them is how
#: a declaration nothing refuses comes to be mistaken for a rule.
DESCRIPTION = "description"      # nothing refuses a violation of it
CHECKED = "checked"              # something refuses a program that violates it
GUARANTEED = "guaranteed"        # the structure cannot violate it
STATUSES = (DESCRIPTION, CHECKED, GUARANTEED)

#: What a memory does when two writes reach one element in one cycle.
DEFINED = "defined"              # the memory arbitrates; last writer wins
ORED = "or"                      # one-hot mux, no arbiter: a program contract
UNDEFINED = "undefined"          # left undefined, as on the FPGA

_TOUCHES_STATE = (READ, WRITE)
_MOVES_VALUE = (EMIT, RECEIVE)


class ActionError(Exception):
    """A composition the rule disproves."""

    def __init__(self, machine, found, after=None):
        self.machine = machine
        self.violations = found
        blame = f" after {after}" if after else ""
        body = "\n".join(str(v) for v in found)
        super().__init__(
            f"{len(found)} illegal action(s) in {machine!r}{blame}:\n{body}"
        )


@dataclass(frozen=True)
class Violation:
    rule: str
    where: str
    found: str
    repair: str

    def __str__(self):
        pad = " " * 26
        return f"{self.rule:24s} {self.where}\n{pad}{self.found}\n{pad}repair: {self.repair}"


@dataclass(frozen=True)
class Obligation:
    """What an accepted composition still rests on. The rule refuses a
    provable lie and confirms nothing, so every composition it accepts leaves
    these behind."""

    where: str
    claim: str
    premise: str = None
    discharged_by: str = None

    def __str__(self):
        held = self.discharged_by or "nothing in this model"
        return f"{self.where}: {self.claim}\n    {self.premise or 'no premise declared'}\n    discharged by: {held}"


@dataclass(frozen=True)
class Port:
    """One resource a unit spends, bounded per cycle.

    A port is what makes a work count derivable rather than declared: a unit's
    cost for an instruction is its busiest port's item count, so ``vadd``
    costing two accumulator steps per row is a consequence of the accumulator
    having one read port and ``vadd`` reading twice.

    ``physical`` is items per CYCLE; a step is ``Unit.ii`` cycles, so the
    capacity of a port per step is ``physical * ii``. Keeping the two apart is
    what lets an unpipelined unit do two reads a step through one port."""

    name: str
    physical: int = 1
    carries: str = None

    def capacity(self, ii=1):
        return self.physical * max(1, ii)


@dataclass(frozen=True)
class Unit:
    """One unit of the machine, the ports it spends, and how often it steps.

    ``ii`` is the initiation interval of the unit's own work loop: the cycles
    between consecutive steps. It is here because a port budget without an II
    decides a two-store loop wrongly in BOTH directions -- see the doc page's
    account of ``s.memory_ports``.

    ``elastic`` says what happens when an issue needs more cycles than one
    step: a unit with a flat work loop takes another step (TinyTPU's
    accumulator), and a unit whose body must retire every ``ii`` cycles is
    REFUSED instead. Which of the two a unit is decides whether contention is
    a cost or an illegality, and neither answer is right for both."""

    name: str
    ports: tuple = ()
    ii: int = 1
    elastic: bool = True
    note: str = None

    def port(self, name):
        for p in self.ports:
            if p.name == name:
                return p
        return None


@dataclass(frozen=True)
class State:
    """Something an action reads or writes: a memory, or the outside world.

    ``bank`` is the LAYOUT MAP, an expression over the row index ``r``: it is
    what ``Partition.Cyclic`` and ``Partition.Block`` each do, written down.
    A rule that counts accesses without composing this map decides a
    block-partitioned design wrongly, because both stores of an even/odd pair
    land in bank 0 under ``r // half`` and in different banks under ``r % 2``.
    """

    name: str
    rows: str = None
    owner: str = None
    lanes: str = None
    banks: str = "1"
    bank: str = "0"
    read_ports: int = 1
    write_ports: int = 1
    collision: str = DEFINED
    note: str = None

    def bank_of(self, row, environment):
        env = dict(environment)
        env["r"] = row
        env["rows"] = evaluate(self.rows, env, None)
        return evaluate(self.bank, env, 0)


@dataclass(frozen=True)
class Action:
    """One per-unit effect of one instruction.

    ``base``, ``count``, ``rows`` and ``when`` are expressions over the
    instruction's operand names and the machine's parameters, so an action is
    written once and resolved per issue. ``into``/``args`` name values, which
    is how a composition says that the accumulator's write is the array's
    product and not something it invented.

    ``at`` is the effect's DECLARED LATENCY: the cycles after its inputs are
    ready at which it lands. With it an action is an entry in a calendar --
    "this unit writes element f(i) at offset d from issue, repeating every
    II" -- rather than a predicate over resources, and a calendar is what
    makes both failure directions of a count impossible by construction. It
    is also the model's weakest point: a calendar is only as right as the
    latency it books, and a declared span has to be MEASURED against the
    hardware, by a probe that scans downward so that it fails when the RTL
    turns out to be faster than declared, not only when it is slower."""

    unit: str
    kind: str
    port: str = None
    state: str = None
    base: str = None
    count: str = "1"
    per: str = PER_ROW
    when: str = None
    into: str = None
    args: tuple = ()
    compute: str = None
    lanes: str = None
    role: str = None
    at: int = 0
    status: str = DESCRIPTION
    enforced_by: str = None

    @property
    def touches_state(self):
        return self.kind in _TOUCHES_STATE


@dataclass(frozen=True)
class Contract:
    """A property of PROGRAMS that no single instruction's composition can
    establish. It is carried here so that it is reported as an obligation
    rather than forgotten."""

    name: str
    statement: str
    discharged_by: str = None


@dataclass(frozen=True)
class Instruction:
    """One instruction, as the ordered actions it composes."""

    name: str
    actions: tuple = ()
    operands: tuple = ()
    rows: str = "nr"
    note: str = None

    def of(self, unit):
        return tuple(a for a in self.actions if a.unit == unit)

    @property
    def units(self):
        seen = []
        for a in self.actions:
            if a.unit not in seen:
                seen.append(a.unit)
        return tuple(seen)


@dataclass(frozen=True)
class Effect:
    """One action, resolved for one issue: which cycle, which row, which value.

    ``cycle`` is relative to the issue, and ``step`` is the unit's own work
    step it falls in -- the number a per-unit work count sums."""

    cycle: int
    step: int
    unit: str
    port: str
    kind: str
    state: str = None
    row: int = None
    count: int = 1
    role: str = None
    into: str = None
    args: tuple = ()
    compute: str = None
    lanes: tuple = None
    action: Action = None

    def __str__(self):
        where = f"{self.state}[{self.row}]" if self.state else (self.compute or self.port)
        return f"cycle {self.cycle} {self.unit}.{self.port or self.kind} {self.kind} {where}"


def _steps(span, ii):
    """How many steps of ``ii`` cycles a span of cycles occupies."""
    return max(1, -(-span // max(1, ii)))


def evaluate(expression, environment, fill=_UNSET):
    """An expression over operand names and machine parameters.

    Restricted to the names the caller supplies: a spec file is data this
    repository already `eval`s for the same reason (`isa_encoding.header`).
    An unbound name is an error unless ``fill`` is given, in which case every
    unbound name takes that value -- per NAME, not per expression, so that the
    legality rule can still tell ``2 * d`` from ``2 * d + 1`` with no issue in
    hand. Collapsing the two hides exactly the conflict it looks for."""
    if expression is None:
        return None
    if isinstance(expression, int):
        return expression
    code = compile(expression, "<action>", "eval")
    env = dict(environment)
    if fill is not _UNSET:
        for name in code.co_names:
            env.setdefault(name, fill)
    return int(eval(code, {"__builtins__": {}}, env))  # noqa: S307


def holds(expression, environment):
    """A predicate over the same names. ``None`` (no predicate) always holds;
    a predicate whose names are not all bound is UNSETTLED, which is an
    obligation rather than a refusal."""
    if expression is None:
        return True
    try:
        return bool(eval(expression, {"__builtins__": {}}, dict(environment)))  # noqa: S307
    except (NameError, TypeError):
        return None


def lane_order(expression, width, environment):
    """The leaf a source lane lands on, for every lane.

    MiniTPU's reduction tree feeds leaf ``sublane * NUM_LANES + lane``, which
    is deliberately not lane-major and REASSOCIATES the sum. An action that
    can declare an effect but not its operand mapping cannot describe that
    unit at all, so the mapping is part of the action."""
    if expression is None:
        return None
    env = dict(environment)
    out = []
    for i in range(width):
        env["i"] = i
        out.append(evaluate(expression, env))
    return tuple(out)


@dataclass
class Machine:
    """A set of units, the state they own, and the instructions they compose.

    Checked when it is built, and re-checked by :meth:`recheck` after anything
    changes it -- the same discipline ``s.encodable_on`` follows, where a
    budget is re-verified after every later primitive so the error names what
    broke it."""

    name: str
    units: tuple = ()
    states: tuple = ()
    instructions: tuple = ()
    parameters: dict = field(default_factory=dict)
    arithmetic: str = EXACT
    contracts: tuple = ()
    _checked: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        self.recheck()

    def __repr__(self):
        return (f"Machine({self.name!r}, {len(self.units)} units, "
                f"{len(self.instructions)} instructions)")

    # ------------------------------------------------------------ lookup ---
    def unit(self, name):
        for u in self.units:
            if u.name == name:
                return u
        return None

    def state(self, name):
        for s in self.states:
            if s.name == name:
                return s
        return None

    def instruction(self, name):
        for i in self.instructions:
            if i.name == name:
                return i
        return None

    def units_of(self, name):
        """Which units an instruction reaches. DERIVED: the dispatch table and
        the unit list a document prints are the same fact as the actions."""
        return self.instruction(name).units

    # ---------------------------------------------------------- schedule ---
    def _environment(self, extra=None):
        env = dict(self.parameters)
        env.update(extra or {})
        return env

    def _lanes_of(self, action, env):
        state = self.state(action.state) if action.state else None
        width = evaluate(state.lanes, env) if state is not None and state.lanes else 0
        return lane_order(action.lanes, width or 0, env)

    def _resources(self, action, unit, row, env):
        """Every resource one effect books, and how many of it there is per
        CYCLE. A unit's port is its own; a memory's read and write ports are
        the MEMORY's, shared by every unit that touches it, which is the only
        reason two units can collide at all."""
        out = []
        port = action.port or action.kind
        model = self.unit(unit)
        if model is not None and model.port(port) is not None:
            out.append((("port", unit, port), model.port(port).physical))
        if action.touches_state:
            state = self.state(action.state)
            if state is not None:
                bank = state.bank_of(row or 0, env)
                ports = state.write_ports if action.kind == WRITE \
                    else state.read_ports
                out.append((("state", state.name, bank, action.kind), ports))
        return out

    def _book(self, unit, ii, actions, row, env, origin, calendar, loose=False):
        """Place actions on the unit's CYCLE timeline, booking each resource
        as it goes: an Action is an entry in a calendar, not a predicate over
        resources. Composing in time is what makes a count's two failure
        directions impossible rather than merely watched for.

        Returns ``(effects, span)``: span is the cycles this pass occupies."""
        placed, ready, span = [], {}, 0
        for action in actions:
            if holds(action.when, env) is False:
                continue
            count = evaluate(action.count, env, *((1,) if loose else ())) or 0
            if count <= 0:
                continue
            base = evaluate(action.base, env, *((0,) if loose else ())) \
                if action.base is not None else None

            at = max([ready.get(a, 0) for a in action.args] + [0]) \
                + max(0, action.at)
            last = at
            for item in range(count):
                resources = self._resources(action, unit, 
                                            None if base is None else base + row,
                                            env)
                cycle = at + item
                while any(len(calendar.get((key, origin + cycle), ())) >= cap
                          for key, cap in resources):
                    cycle += 1
                effect = Effect(
                    cycle=origin + cycle, step=(origin + cycle) // max(1, ii),
                    unit=unit,
                    port=action.port or action.kind, kind=action.kind,
                    state=action.state,
                    row=None if base is None else base + row,
                    role=action.role, into=action.into, args=action.args,
                    compute=action.compute, lanes=self._lanes_of(action, env),
                    action=action)
                for key, _cap in resources:
                    calendar.setdefault((key, origin + cycle), []).append(effect)
                placed.append(effect)
                last = max(last, cycle)
                at = cycle
            if action.into is not None:
                # Available in the cycle it is produced in: a value crosses an
                # action boundary combinationally unless the action declares a
                # latency with `at=`, which is why `vadd`'s second read and its
                # write share one cycle and a reduction tree's root does not.
                ready[action.into] = last
            span = max(span, last + 1)
        return placed, span

    def calendar(self, name, environment=None, loose=False):
        """Every resource this instruction books, at every cycle, across every
        unit. ``{(resource, cycle): [Effect]}``."""
        instruction = self.instruction(name)
        env = self._environment(environment)
        booked = {}
        self._issue(instruction, env, booked, loose)
        return booked

    def _issue(self, instruction, env, calendar, loose=False):
        """Place one issue of one instruction, unit by unit.

        Each unit is placed against a calendar OF ITS OWN, so contention
        inside a unit is resolved by moving the effect later -- that is what a
        unit's own scheduler does. The results are then merged into
        ``calendar``, where contention BETWEEN units survives, because no unit
        can reschedule itself around another without an arbiter. That
        asymmetry is the whole reason a composition can be illegal when each
        of its actions is legal.

        Returns ``{unit: (effects, steps, span)}``, span being the cycles one
        row costs."""
        rows = evaluate(instruction.rows, env, *((1,) if loose else ())) \
            if instruction.rows else 1
        if loose:
            rows = min(rows, 2)
        out = {}
        for unit in instruction.units:
            model = self.unit(unit)
            ii = max(1, model.ii if model is not None else 1)
            mine = instruction.of(unit)
            head = [a for a in mine if a.per == PER_INSTRUCTION]
            body = [a for a in mine if a.per != PER_INSTRUCTION]
            mine_only = {}
            effects, cycle, width = [], 0, None
            if head:
                placed, span = self._book(unit, ii, head, 0, env, cycle,
                                          mine_only, loose)
                effects += placed
                width = span
                cycle += _steps(span, ii) * ii
            for row in range(max(0, rows) if body else 0):
                placed, span = self._book(unit, ii, body, row, env, cycle,
                                          mine_only, loose)
                effects += placed
                if row == 0 or width is None:
                    width = span
                cycle += _steps(span, ii) * ii
            for key, hits in mine_only.items():
                calendar.setdefault(key, []).extend(hits)
            out[unit] = (tuple(effects), cycle // ii, width or 0)
        return out

    def schedule(self, name, unit, environment=None, loose=False):
        """The effects one unit contributes to one issue, in cycle order.

        Per-instruction actions come first and per-row actions follow, row by
        row, because a unit's flat loop runs them that way."""
        instruction = self.instruction(name)
        env = self._environment(environment)
        return self._issue(instruction, env, {}, loose).get(unit, ((), 0, 0))[0]

    def work(self, unit, name, environment=None):
        """How many steps ``unit`` spends on one issue of ``name``.

        This is the number a per-unit work count in an instruction-memory
        header carries, and the number a dispatcher rewrites a row count to.
        Neither is declared anywhere in this model: both are this."""
        if unit not in self.units_of(name):
            return 0
        instruction = self.instruction(name)
        env = self._environment(environment)
        return self._issue(instruction, env, {}).get(unit, ((), 0, 0))[1]

    def effects(self, name, environment=None, units=None):
        """Every resolved effect of one issue, per unit, in cycle order."""
        instruction = self.instruction(name)
        env = self._environment(environment)
        issued = self._issue(instruction, env, {})
        out = []
        for unit in (units or instruction.units):
            out.extend(issued.get(unit, ((), 0, 0))[0])
        return tuple(out)

    def reads(self, name, environment=None):
        return tuple(e for e in self.effects(name, environment) if e.kind == READ)

    def writes(self, name, environment=None):
        return tuple(e for e in self.effects(name, environment) if e.kind == WRITE)

    # --------------------------------------------------------- the rule ---
    def violations(self):
        return [v for rule in RULES for v in rule(self)]

    def recheck(self, after=None):
        found = self.violations()
        if found:
            raise ActionError(self, found, after=after)
        self._checked = True
        return self

    def with_instruction(self, instruction, after=None):
        """A machine with one more instruction, re-checked. Adding an
        instruction is this call; nothing else in a consumer changes."""
        blame = after or f"add {instruction.name!r}"
        try:
            return replace(self, instructions=self.instructions + (instruction,))
        except ActionError as e:
            raise ActionError(self, e.violations, after=blame) from None

    def obligations(self):
        """What the rule accepted without proving.

        Three sources: a reduction whose lane order reassociates under an
        arithmetic that is not associative, a predicate the model cannot
        settle, and a contract that is a property of programs rather than of
        any one composition."""
        out = []
        env = self._environment()
        for instruction in self.instructions:
            for a in instruction.actions:
                if a.lanes is not None:
                    width = 0
                    state = self.state(a.state) if a.state else None
                    if state is not None and state.lanes:
                        width = evaluate(state.lanes, env)
                    order = lane_order(a.lanes, width, env)
                    if order and tuple(order) != tuple(range(width)):
                        out.append(Obligation(
                            where=f"{instruction.name}/{a.unit}",
                            claim=f"{a.compute or a.kind} folds lanes in order "
                                  f"{list(order)}, not lane-major",
                            premise=(
                                "equal to the lane-major fold only if the "
                                f"arithmetic is associative; this machine's is "
                                f"{self.arithmetic!r}"),
                            discharged_by=("associativity of exact integer "
                                           "addition" if self.arithmetic == EXACT
                                           else None)))
                if holds(a.when, env) is None:
                    out.append(Obligation(
                        where=f"{instruction.name}/{a.unit}",
                        claim=f"the effect is predicated on {a.when!r}",
                        premise="the predicate names an operand, so it is "
                                "settled per issue and not here"))
                if a.status == DESCRIPTION:
                    out.append(Obligation(
                        where=f"{instruction.name}/{a.unit}",
                        claim=f"{a.kind} of "
                              f"{a.state or a.compute or a.port} is a "
                              f"DESCRIPTION",
                        premise="nothing refuses a program or a design that "
                                "violates it"))
                state = self.state(a.state) if a.state else None
                if a.kind == WRITE and state is not None and state.collision == ORED:
                    out.append(Obligation(
                        where=f"{instruction.name}/{a.unit}",
                        claim=f"writes to {state.name} are OR-ed, not "
                              f"arbitrated",
                        premise="two units writing one element in one cycle "
                                "are merged by a one-hot mux with no arbiter, "
                                "so the schedule must keep them apart",
                        discharged_by=a.enforced_by))
        for c in self.contracts:
            out.append(Obligation(where=self.name, claim=c.statement,
                                  premise=f"contract {c.name}",
                                  discharged_by=c.discharged_by))
        return tuple(out)


# ------------------------------------------------------------- the rules ---
def unknown_name_violations(machine):
    """A unit, port or state an action names and the machine does not have.
    Provably false, and the cheapest drift there is."""
    units = {u.name for u in machine.units}
    states = {s.name for s in machine.states}
    out = []
    for i in machine.instructions:
        for a in i.actions:
            where = f"{i.name}/{a.unit}"
            if a.kind not in KINDS:
                out.append(Violation("unknown kind", where, f"kind {a.kind!r}",
                                     f"one of {', '.join(KINDS)}"))
            if a.unit not in units:
                out.append(Violation("unknown unit", where,
                                     f"no unit {a.unit!r}",
                                     "declare the unit, or name one that exists"))
                continue
            port = a.port or a.kind
            if machine.unit(a.unit).port(port) is None:
                out.append(Violation(
                    "unknown port", where,
                    f"unit {a.unit!r} has no port {port!r} "
                    f"(it has {[p.name for p in machine.unit(a.unit).ports]})",
                    "declare the port on the unit, or spend one it has"))
            if a.touches_state and a.state not in states:
                out.append(Violation("unknown state", where,
                                     f"no state {a.state!r}",
                                     "declare the memory this action touches"))
    return out


def value_flow_violations(machine):
    """A value consumed and never produced, produced twice, or crossing from
    one unit to another with no port carrying it.

    The last is the rule that makes a composition a MACHINE rather than a
    wish: the accumulator may take the array's product only because a channel
    is declared between them."""
    out = []
    for i in machine.instructions:
        producer, seen = {}, {}
        for a in i.actions:
            if a.into is None:
                continue
            if a.into in seen:
                out.append(Violation(
                    "value defined twice", f"{i.name}/{a.unit}",
                    f"{a.into!r} is also produced by {seen[a.into]!r}",
                    "name the second value something else"))
            seen[a.into] = a.unit
            producer[a.into] = a
        emits = {(a.unit, a.port) for a in i.actions if a.kind == EMIT}
        receives = {(a.unit, a.port) for a in i.actions if a.kind == RECEIVE}
        for a in i.actions:
            for name in a.args:
                if name not in producer:
                    out.append(Violation(
                        "undefined value", f"{i.name}/{a.unit}",
                        f"{name!r} is consumed and nothing produces it",
                        "produce it with into=, or consume a value that exists"))
                    continue
                source = producer[name].unit
                if source == a.unit:
                    continue
                shared = {p for u, p in emits if u == source} \
                    & {p for u, p in receives if u == a.unit}
                out.append(Violation(
                    "value crosses units", f"{i.name}/{a.unit}",
                    f"{name!r} is produced in {source!r} and consumed in "
                    f"{a.unit!r} directly"
                    + (f"; the channel {sorted(shared)} between them carries "
                       f"values but this action does not go through it"
                       if shared else ", and no channel joins them"),
                    f"emit it from {source!r} and receive it in {a.unit!r} "
                    f"over a shared port, then consume what the receive names"))
    return out


def status_violations(machine):
    """An action must say what its claim is WORTH.

    Three things get written the same way everywhere this project has looked:
    a description nothing refuses, a claim something refuses a program for
    breaking, and a guarantee the structure makes. A model that lets them be
    written identically has not helped anyone. ``CHECKED`` therefore has to
    name its enforcer, exactly as ``s.dependence(..., because=)`` has to name
    its premise. The DEFAULT is ``DESCRIPTION``, because that is what an
    undecorated declaration is worth, and each one comes back as an obligation
    rather than being silently counted as a rule."""
    out = []
    for i in machine.instructions:
        for a in i.actions:
            where = f"{i.name}/{a.unit}"
            if a.status not in STATUSES:
                out.append(Violation(
                    "unknown status", where, f"status={a.status!r}",
                    f"one of {', '.join(STATUSES)}"))
            elif a.status == CHECKED and not a.enforced_by:
                out.append(Violation(
                    "checked claim with no checker", where,
                    f"{a.kind} of {a.state or a.compute or a.port!r} claims to "
                    f"be checked and names nothing that checks it",
                    "name the checker with enforced_by=, or say "
                    "status=DESCRIPTION and accept that nothing refuses it"))
    return out


def lane_map_violations(machine):
    """A lane mapping that drops or duplicates an operand.

    A reduction's leaf order may be any permutation and still be a reduction;
    it may not be a function that sends two lanes to one leaf, which silently
    loses a term."""
    out = []
    env = machine._environment()  # pylint: disable=protected-access
    for i in machine.instructions:
        for a in i.actions:
            if a.lanes is None:
                continue
            state = machine.state(a.state) if a.state else None
            width = evaluate(state.lanes, env) if state is not None and state.lanes else None
            if not width:
                out.append(Violation(
                    "lane map without a width", f"{i.name}/{a.unit}",
                    f"lanes={a.lanes!r} but {a.state!r} declares no lane count",
                    "give the state a `lanes` expression"))
                continue
            order = lane_order(a.lanes, width, env)
            if sorted(order) != list(range(width)):
                out.append(Violation(
                    "lane map is not a permutation", f"{i.name}/{a.unit}",
                    f"lanes={a.lanes!r} sends {width} lanes to {sorted(set(order))}",
                    "a reduction's leaf order must be a permutation of its "
                    "operands; anything else drops or duplicates a term"))
    return out


def calendar_violations(machine):
    """The calendar the instruction books, read back.

    An Action is an entry in a calendar over time, not a predicate over
    resources, and this is where that pays. Three things it decides that a
    count cannot:

    * a BANK, not a memory: the layout map is composed with each access's own
      row, so a block-partitioned pair that both land in bank 0 costs two
      cycles where ``banks * ports >= stores`` sees one;
    * an INITIATION INTERVAL: a step is ``ii`` cycles, so a sequential pair
      through one port fits in one step of an unpipelined unit, where a count
      refuses it;
    * TWO UNITS: a memory's ports belong to the memory, and a unit can only
      reschedule itself. Two units writing one element in one cycle is the
      case where each action is legal alone and the composition is not.

    Inside one unit, contention moves an effect later; whether that is a cost
    or an illegality is what ``Unit.elastic`` says. A flat work loop spends
    another step; a body that must retire every ``ii`` cycles cannot, and is
    refused with the resource that forced it named."""
    out = []
    env = machine._environment()  # pylint: disable=protected-access
    for i in machine.instructions:
        calendar = {}
        issued = machine._issue(i, env, calendar, loose=True)  # pylint: disable=protected-access
        for unit, (effects, _steps_taken, span) in sorted(issued.items()):
            model = machine.unit(unit)
            if model is None or model.elastic or span <= max(1, model.ii):
                continue
            window = [e for e in effects if e.cycle < span]
            binding = []
            for key, booked in _by_resource(machine, window, env).items():
                if len(booked) > _capacity(machine, key, unit) * model.ii:
                    binding.append((key, booked))
            blamed = " and ".join(
                f"{_name_resource(machine, key)} is booked {len(hits)} time(s)"
                + (f" -- rows {[h.row for h in hits if h.row is not None]}"
                   if any(h.row is not None for h in hits) else "")
                for key, hits in sorted(binding, key=lambda kv: str(kv[0])))
            out.append(Violation(
                "does not fit the initiation interval", f"{i.name}/{unit}",
                f"one row needs {span} cycle(s) but {unit!r} retires every "
                f"{model.ii}; " + (blamed or "no single resource binds"),
                "bank the memory so these rows differ, raise the unit's II, "
                "add a port, or make the unit elastic so it spends a step"))
        for (resource, cycle), hits in sorted(calendar.items(),
                                              key=lambda kv: str(kv[0])):
            if resource[0] != "state":
                continue
            _, name, bank, kind = resource
            state = machine.state(name)
            ports = state.write_ports if kind == WRITE else state.read_ports
            units = sorted({h.unit for h in hits})
            if kind == WRITE and state.collision == UNDEFINED:
                clash = [h for h in hits
                         if sum(1 for o in hits if o.row == h.row) > 1]
                if clash:
                    out.append(Violation(
                        "undefined write collision", f"{i.name}/{units}",
                        f"two writes reach {name}[{clash[0].row}] in cycle "
                        f"{cycle}, and {name} leaves a same-element collision "
                        f"undefined",
                        "separate them in the calendar, send them to different "
                        "elements, or give the memory a defined collision "
                        "behaviour"))
            if len(hits) > ports and len(units) > 1:
                out.append(Violation(
                    f"{kind} port shared by units", f"{i.name}/{units}",
                    f"{units} each {kind} {name} bank {bank} in cycle {cycle}; "
                    f"the port belongs to {name}, not to either unit, and "
                    f"there are {ports} of it",
                    "declare a latency on one of them so they book different "
                    "cycles, bank them apart, or put an arbiter between them "
                    "-- neither unit can reschedule around the other"))
    return out


def _by_resource(machine, effects, env):
    out = {}
    for e in effects:
        for key, _cap in machine._resources(  # pylint: disable=protected-access
                e.action, e.unit, e.row, env):
            out.setdefault(key, []).append(e)
    return out


def _capacity(machine, key, unit):
    if key[0] == "state":
        state = machine.state(key[1])
        return state.write_ports if key[3] == WRITE else state.read_ports
    model = machine.unit(key[1])
    port = model.port(key[2]) if model else None
    return port.physical if port else 1


def _name_resource(machine, key):
    if key is None:
        return "no resource"
    if key[0] == "state":
        state = machine.state(key[1])
        return (f"{key[1]} bank {key[2]} {key[3]} port (bank map "
                f"{state.bank!r}, {state.write_ports if key[3] == WRITE else state.read_ports} port(s))")
    return f"{key[1]}'s {key[2]} port"


RULES = (
    unknown_name_violations,
    status_violations,
    value_flow_violations,
    lane_map_violations,
    calendar_violations,
)


def check(machine, after=None):
    """Run the rule. Raises :class:`ActionError` on a composition it can
    disprove; returns the obligations it could not settle."""
    found = machine.violations()
    if found:
        raise ActionError(machine, found, after=after)
    return machine.obligations()
