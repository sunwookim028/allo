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

Nothing here knows about any particular machine. ``examples/tinytpu`` builds a
:class:`Machine` out of its ISA spec; the doc page builds a second one, with an
adder-tree reduction unit, out of nothing at all.
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
class Channel:
    """One stream between two units: what an ``emit`` and a ``receive`` spend.

    It is here because a lane map needs a WIDTH and nothing else, and the
    model used to read a width off addressed state alone. MiniTPU's fold
    reads a vector register file, so its width is a property of a memory;
    ours arrives on a FIFO, and the packed word was declared a one-row
    ``State`` purely to get past the check. A channel is not a memory -- it
    has no rows, no bank map and no collision behaviour -- and a fold does
    not need any of those. It needs to know how many operands one word
    carries, which is the ONE field here.

    ``lanes`` and ``lane_bits`` are what ``allo.compose.Channel`` declares;
    the bit width of the word is derived from them there. Nothing in this
    model reads the bit width."""

    name: str
    lanes: str = None
    lane_bits: str = None
    depth: str = None
    carries: str = None
    endpoints: tuple = ()


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

    ``offset`` is the lane offset within the element -- the column block a
    ``dma_ld`` takes out of a DRAM row. Without it an action can say which
    ROW it touches and not which part of it, which is the same gap as a
    reduction tree whose leaf order cannot be stated.

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
    offset: str = None
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
class Cost:
    """What one unit spends on one issue: a head, a row, and a row count.

    Kept apart rather than summed because a header word sometimes counts the
    head (how many ``mm``s) and sometimes the rows (how many wavefront rows),
    and the two are different questions about the same instruction."""

    unit: str
    rows: int
    head_steps: int
    row_steps: int
    head_latency: int = 0
    row_latency: int = 0
    ii: int = 1
    head: tuple = ()
    row: tuple = ()

    @property
    def steps(self):
        """What the unit's own work loop counts: one step per initiation.

        A step is the unit accepting a row, not the row being finished. The
        two were the same number until an action declared a latency, and
        keeping them the same is what made a pipelined unit's work count
        wrong by its depth."""
        return self.head_steps + self.rows * self.row_steps

    @property
    def latency(self):
        """Cycles from the first effect of one issue to its last: the head's
        own span, plus one row per initiation, plus the last row's span.

        This is the number a designer means by "how long does it take", and
        it is NOT the number a work count carries. Reporting one for the
        other is the defect this pair replaces."""
        rows = max(0, self.rows) if self.row_steps else 0
        if not rows:
            return self.head_latency
        return ((self.head_steps + (rows - 1) * self.row_steps) * self.ii
                + self.row_latency)

    def items(self, port):
        return (sum(1 for e in self.head if e.port == port)
                + self.rows * sum(1 for e in self.row if e.port == port))


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
    offset: int = None
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
    channels: tuple = ()
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

    def channel(self, name):
        for c in self.channels:
            if c.name == name:
                return c
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

    def _value_channels(self, instruction):
        """For every ``(unit, value)``, the channel its contents arrived on.

        A value that enters a unit through a ``receive`` keeps its channel
        through the computes that consume it, so a fold can be asked how wide
        its operand is without the operand being addressed state. This is a
        query over the composition, not a new field on an action."""
        out = {}
        if instruction is None:
            return out
        for a in instruction.actions:
            if a.into is None:
                continue
            channel = self.channel(a.port) if a.port else None
            if channel is not None and a.kind in _MOVES_VALUE:
                out[(a.unit, a.into)] = channel.name
                continue
            for src in a.args:
                if (a.unit, src) in out:
                    out[(a.unit, a.into)] = out[(a.unit, src)]
                    break
        return out

    def width_of(self, action, env, instruction=None):
        """How many operands the thing this action folds carries.

        Resolved from, in order: the STATE the action names, a CHANNEL it
        names, or the channel its operand arrived on. The third is what lets
        a unit whose operand is a FIFO word declare a leaf order at all --
        before it, the packed word had to be declared a one-row memory, which
        is a memory nothing addresses."""
        state = self.state(action.state) if action.state else None
        if state is not None and state.lanes:
            return evaluate(state.lanes, env)
        channel = self.channel(action.state) if action.state else None
        if channel is None:
            arrived = self._value_channels(instruction)
            for src in action.args:
                named = arrived.get((action.unit, src))
                if named:
                    channel = self.channel(named)
                    break
        if channel is not None and channel.lanes:
            return evaluate(channel.lanes, env)
        return None

    def _lanes_of(self, action, env, instruction=None):
        width = self.width_of(action, env, instruction)
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

    def _book(self, unit, ii, actions, row, env, origin, calendar,
              loose=False, instruction=None):
        """Place actions on the unit's CYCLE timeline, booking each resource
        as it goes: an Action is an entry in a calendar, not a predicate over
        resources. Composing in time is what makes a count's two failure
        directions impossible rather than merely watched for.

        Returns ``(effects, span, initiation)``. The two numbers are
        DIFFERENT QUESTIONS about the same placement and this model used to
        answer both with the first:

        * ``span`` is the LATENCY -- the cycles from the first effect of one
          row to the last, which is what a declared ``at=`` lengthens;
        * ``initiation`` is the RATE -- the cycles that must pass before the
          same actions can be placed again, which is set by the busiest
          resource they book and by nothing else. A value in flight occupies
          no port.

        Charging the span per row is what made ``reduce_tree`` cost 80 steps
        where the hardware does 16: a pipelined unit with a two-cycle fold
        retires one word a cycle, and its own docstring says the probe's
        back-to-back run and its one-word run agree. Neither number is
        declared: the latency is the actions' ``at=``, the rate is the ports.
        """
        placed, ready, span = [], {}, 0
        booked = {}
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
            first, last = None, at
            for _ in range(count):
                resources = self._resources(
                    action, unit, None if base is None else base + row, env)
                cycle = at
                while any(len(calendar.get((key, origin + cycle), ())) >= cap
                          for key, cap in resources):
                    cycle += 1
                effect = Effect(
                    cycle=origin + cycle, step=(origin + cycle) // max(1, ii),
                    unit=unit,
                    port=action.port or action.kind, kind=action.kind,
                    state=action.state,
                    row=None if base is None else base + row,
                    offset=evaluate(action.offset, env,
                                     *((0,) if loose else ()))
                    if action.offset else None,
                    role=action.role, into=action.into, args=action.args,
                    compute=action.compute,
                    lanes=self._lanes_of(action, env, instruction),
                    action=action)
                for key, cap in resources:
                    calendar.setdefault((key, origin + cycle), []).append(effect)
                    have, _ = booked.get(key, (0, cap))
                    booked[key] = (have + 1, cap)
                placed.append(effect)
                first = cycle if first is None else first
                last, at = cycle, cycle + 1
            if action.into is not None:
                # Available from the cycle its FIRST item lands, not its last:
                # a value crosses an action boundary combinationally unless the
                # action declares a latency with `at=`, and a multi-item action
                # is a STREAM, so its consumer starts with it rather than after
                # it. That is why `spm` spends T+1 steps on an `mm` -- one spad
                # read and one `wcol` put per step -- and not 2T+1.
                ready[action.into] = first if first is not None else last
            span = max(span, last + 1)
        initiation = max([-(-n // max(1, cap)) for n, cap in booked.values()]
                         + [1 if placed else 0])
        return placed, span, initiation

    def profile(self, name, environment=None):
        """Per-unit cost of one issue, WITHOUT materialising every row.

        Every row of an instruction has the same shape -- only the element it
        touches moves -- so the head and one row are placed and the rest is
        arithmetic. This is what a header count and a dispatch rewrite are
        computed from, and it is why computing them costs the same whether
        ``nr`` is 1 or 127."""
        instruction = self.instruction(name)
        env = self._environment(environment)
        rows = evaluate(instruction.rows, env) if instruction.rows else 1
        out = {}
        for unit in instruction.units:
            model = self.unit(unit)
            ii = max(1, model.ii if model is not None else 1)
            mine = instruction.of(unit)
            head = [a for a in mine if a.per == PER_INSTRUCTION]
            body = [a for a in mine if a.per != PER_INSTRUCTION]
            hits, hspan, hinit = self._book(
                unit, ii, head, 0, env, 0, {}, instruction=instruction) \
                if head else ((), 0, 0)
            rits, rspan, rinit = self._book(
                unit, ii, body, 0, env, 0, {}, instruction=instruction) \
                if body else ((), 0, 0)
            out[unit] = Cost(
                unit=unit, rows=max(0, rows),
                head_steps=_steps(hinit, ii) if hinit else 0,
                row_steps=_steps(rinit, ii) if rinit else 0,
                head_latency=hspan, row_latency=rspan, ii=ii,
                head=tuple(hits), row=tuple(rits))
        return out

    def work(self, unit, name, environment=None):
        """How many steps ``unit`` spends on one issue of ``name``.

        This is the number a per-unit work count in an instruction-memory
        header carries, and the number a dispatcher rewrites a row count to.
        Neither is declared anywhere in this model: both are this."""
        cost = self.profile(name, environment).get(unit)
        return cost.steps if cost else 0

    def items(self, unit, port, name, environment=None):
        """How many items one PORT of one unit carries for one issue. A header
        word that counts something narrower than a unit's work names it."""
        cost = self.profile(name, environment).get(unit)
        return cost.items(port) if cost else 0

    def row_span(self, state, name, environment=None):
        """The highest element of ``state`` one issue touches, plus one --
        the span a burst has to cover."""
        best = 0
        for cost in self.profile(name, environment).values():
            for effect in cost.head:
                if effect.state == state and effect.row is not None:
                    best = max(best, effect.row + 1)
            for effect in cost.row:
                if effect.state == state and effect.row is not None:
                    best = max(best, effect.row + cost.rows)
        return best

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

        Returns ``{unit: (effects, steps, initiation)}``: the third number is
        the cycles between one row and the next, NOT the cycles one row takes.
        Rows overlap exactly as the unit's pipeline overlaps them, so a row
        whose last effect lands after the next row starts books the resource
        it is still holding and the next row is pushed off it -- which is the
        only honest way to place a pipelined unit on a shared calendar."""
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
                placed, span, init = self._book(
                    unit, ii, head, 0, env, cycle, mine_only, loose,
                    instruction=instruction)
                effects += placed
                width = init
                cycle += (_steps(init, ii) * ii) if init else 0
            for row in range(max(0, rows) if body else 0):
                placed, span, init = self._book(
                    unit, ii, body, row, env, cycle, mine_only, loose,
                    instruction=instruction)
                effects += placed
                if row == 0 or width is None:
                    width = init
                cycle += (_steps(init, ii) * ii) if init else 0
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
                    width = self.width_of(a, env, instruction) or 0
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
    """A value consumed and never produced, produced twice in one unit, or
    consumed in a unit other than the one that produced it.

    Values are scoped PER UNIT, because that is what a unit is: a value leaves
    one unit only through an ``emit`` and enters another only through a
    ``receive``, and the two ends may name it the same thing without being the
    same value. The rule that makes a composition a MACHINE rather than a wish
    is the last one -- the accumulator may take the array's product only
    because a channel is declared between them, and a composition that skips
    the channel is refused."""
    out = []
    for i in machine.instructions:
        producer = {}
        for a in i.actions:
            if a.into is None:
                continue
            key = (a.unit, a.into)
            if key in producer:
                out.append(Violation(
                    "value defined twice", f"{i.name}/{a.unit}",
                    f"{a.into!r} is produced twice in {a.unit!r}",
                    "name the second value something else"))
            producer[key] = a
        emits = {(a.unit, a.port) for a in i.actions if a.kind == EMIT}
        receives = {(a.unit, a.port) for a in i.actions if a.kind == RECEIVE}
        for a in i.actions:
            for name in a.args:
                if (a.unit, name) in producer:
                    continue
                if a.kind == RECEIVE:
                    # A receive's argument is a cross-unit value by
                    # definition: it names what arrives on the channel. What
                    # is checked instead is that a matching emit exists.
                    senders = {u for u, n in producer if n == name}
                    ports = {p for u, p in
                             {(x.unit, x.port) for x in i.actions
                              if x.kind == EMIT} if u in senders}
                    if a.port not in ports:
                        out.append(Violation(
                            "receive with no sender", f"{i.name}/{a.unit}",
                            f"{a.unit!r} receives {name!r} on {a.port!r} and "
                            f"nothing emits it there",
                            f"emit {name!r} on {a.port!r} from the unit that "
                            f"produces it"))
                    continue
                elsewhere = sorted({u for u, n in producer if n == name})
                if not elsewhere:
                    out.append(Violation(
                        "undefined value", f"{i.name}/{a.unit}",
                        f"{name!r} is consumed in {a.unit!r} and nothing "
                        f"produces it there or anywhere",
                        "produce it with into=, or consume a value that exists"))
                    continue
                shared = {p for u, p in emits if u in elsewhere} \
                    & {p for u, p in receives if u == a.unit}
                out.append(Violation(
                    "value crosses units", f"{i.name}/{a.unit}",
                    f"{name!r} is produced in {elsewhere} and consumed in "
                    f"{a.unit!r} without entering it"
                    + (f"; the channel {sorted(shared)} joins them but this "
                       f"action does not take its value from the receive"
                       if shared else ", and no channel joins them"),
                    f"emit it from {elsewhere[0]!r} and receive it in "
                    f"{a.unit!r} over a shared port, then consume what the "
                    f"receive names"))
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
            width = machine.width_of(a, env, i)
            if not width:
                out.append(Violation(
                    "lane map without a width", f"{i.name}/{a.unit}",
                    f"lanes={a.lanes!r} and nothing says how wide the operand "
                    f"is: {a.state!r} is not a state or a channel with a "
                    f"`lanes` count, and no channel delivered {list(a.args)}",
                    "give the state or the channel a `lanes` expression, or "
                    "take the operand from a receive on a channel that has "
                    "one"))
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
        for unit, (effects, _steps_taken, initiation) in sorted(issued.items()):
            model = machine.unit(unit)
            if model is None or model.elastic or initiation <= max(1, model.ii):
                continue
            window = [e for e in effects if e.cycle < initiation]
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
                f"one row can be started only every {initiation} cycle(s) "
                f"but {unit!r} retires every {model.ii}; "
                + (blamed or "no single resource binds"),
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


# ------------------------------------------ derived from a composed region ---
#: What a unit spends on one element of a memory it addresses. The Action
#: layer names a memory port ``<state>.<kind>``; nothing chooses that name
#: twice.
def _state_port(state, kind):
    return f"{state}.{kind}"


def structure(architecture, name=None):
    """The units, ports, states and channels a composed ``Architecture``
    already implies -- built from it, not declared a second time.

    ``allo.compose`` and this module were two descriptions of one machine,
    and the second was a second declaration of every unit, which is the
    failure class the Action layer exists to remove. This is the join. A
    composed architecture declares, and ``Architecture._check`` enforces,
    every channel a unit touches and which end it is; ``Unit.arrays`` reads
    the memories it addresses off the same AST the declaration is checked
    against. Both become ports here.

    WHAT DOES NOT COME ACROSS, and this is the measured result rather than a
    caveat (``tests/ip/test_derived_ports.py`` pins each one):

    * a COMPUTE port -- ``alu``, ``mac``, ``mux`` -- has no structural
      counterpart. A unit that multiplies declares no channel and no memory
      for it, so nothing in the composition implies it. Arithmetic is what
      the Action layer adds, and it is the layer's reason to exist.
    * a port's CAPACITY. ``physical=2`` on the accumulator's ALU is the claim
      that the unit chains two lane operations in one step; it is a hardware
      fact that has to be measured, and no structural declaration carries it.
    * the GRAIN, where the two models disagree about what a unit is. A
      compose unit is a KERNEL, replicated by ``instances``; an Action unit
      is a DISPATCH DOMAIN, one work counter fed one row count. They are the
      same thing for every unit instantiated once and a different thing for
      every unit that is not: TinyTPU's ``pe`` and ``wld`` are ``T x T``
      kernels wired by five chains, and the Action model's ``array`` is the
      whole mesh with the chains inside it. Such a unit is reported by
      :func:`projection` and is not derived.

    Returns a :class:`Machine` with no instructions: the structure, ready for
    a composition to be written against it.
    """
    states, seen = [], {}
    for memory in architecture.memories:
        seen[memory.name] = _rows_of(memory.dtype)
    units, channels = [], []
    for channel in architecture.channels:
        ends = tuple(u.name for u in architecture.units
                     if channel.name in u.reads + u.writes)
        channels.append(Channel(
            name=channel.name,
            lanes=getattr(channel, "lanes", "") or None,
            lane_bits=getattr(channel, "lane_bits", "") or None,
            depth=channel.depth, carries=channel.carries or None,
            endpoints=ends))
    for unit in architecture.units:
        ports = [Port(c) for c in _ordered(unit.reads + unit.writes)]
        for state, use in unit.arrays().items():
            rows = use["rows"] or seen.get(state)
            if state not in {s.name for s in states}:
                states.append(State(name=state, rows=rows, owner=unit.name))
            for kind in (READ, WRITE):
                if use[kind]:
                    ports.append(Port(_state_port(state, kind)))
        units.append(Unit(name=unit.name, ports=tuple(ports),
                          note="derived from the composed region"))
    return Machine(name=name or architecture.name, units=tuple(units),
                   states=tuple(states), channels=tuple(channels),
                   parameters={k: v for k, v in architecture.parameters.items()
                               if isinstance(v, int)})


def _ordered(names):
    out = []
    for n in names:
        if n not in out:
            out.append(n)
    return tuple(out)


def _rows_of(dtype):
    """``int8[MAXDIM * MAXDIM]`` -> ``MAXDIM * MAXDIM``."""
    import ast as _ast  # noqa: PLC0415  -- one call, at the boundary
    node = _ast.parse(dtype, mode="eval").body
    return _ast.unparse(node.slice) if isinstance(node, _ast.Subscript) else None


def projection(architecture, machine, aggregate=None):
    """Every port and state of ``machine`` that the composed architecture
    implies, every one it does not, and every one the architecture implies
    that the machine has not got.

    This is both the measurement of how much of a hand-written machine the
    composed architecture already replaces, and the check on it: where the two
    models DO agree, a disagreement is drift and this reports it, so the second
    declaration stops being independent even where it cannot be removed.

    ``{unit: {"derived": [...], "only_declared": [...],
    "only_derived": [...], "grain": str or None}}``
    """
    implied = structure(architecture)
    aggregate = dict(aggregate or {})
    members = {m for names in aggregate.values() for m in names}
    out, covered = {}, set()
    for unit in machine.units:
        declared = [p.name for p in unit.ports]
        parts = aggregate.get(unit.name, (unit.name,))
        mine = [implied.unit(p) for p in parts]
        composed = [u for u in architecture.units if u.name in parts]
        if not any(m is not None for m in mine):
            out[unit.name] = {
                "derived": [], "only_declared": declared, "only_derived": [],
                "grain": "no unit of this name is composed, and it is not "
                         "declared to aggregate any"}
            continue
        covered.update(parts)
        have = {p.name for m in mine if m is not None for p in m.ports}
        shape = [f"{u.name} as {' x '.join(u.instances)}" for u in composed
                 if u.instances != ("1",)]
        out[unit.name] = {
            "derived": [p for p in declared if p in have],
            "only_declared": [p for p in declared if p not in have],
            "only_derived": sorted(have - set(declared)),
            "grain": None if len(parts) == 1 and not shape else
                     f"{' + '.join(parts)}, composed as {', '.join(shape)}, "
                     f"modelled here as one dispatch domain"}
    for unit in implied.units:
        if unit.name in covered:
            continue
        if machine.unit(unit.name) is None:
            out[unit.name] = {
                "derived": [], "only_declared": [],
                "only_derived": [p.name for p in unit.ports],
                "grain": "composed, and the Action model has no unit of this "
                         "name"}
    for name in members:
        if name in out and name in covered:
            out.pop(name, None)
    return out
