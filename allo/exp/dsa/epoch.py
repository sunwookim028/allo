# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The denotational layer: a compiled program read as a sequence of epochs.

``drafts/schedule-isa-summary.md`` (v2, §3) gives the model this module lands:

    ⟦instr⟧ = an **update** of the machine configuration (λ, σ)
    epoch   = an instruction segment ending at the one that runs K
    config  = the fold of every ⟦instr⟧ up to and including that one
    ⟦epoch⟧ = K(config)         — defined when the machine's validity admits it
    program = epoch₁ ; epoch₂ ; …   composed with ∘

**The operator is an update, not a join, and that is a correction** (v2 §3.1 as
first written said ``⟦instr⟧ : PartialConfig`` and took ⊔ over an epoch's own
instructions). A configuration field is a *register*: a machine that reconfigures
between layers writes the same field twice with different values, which ⊔ calls ⊥
and the hardware calls a layer boundary. Under ⊔ a real MINISA stream — one
``Set*VNLayout`` group followed by several ``ExecuteMapping`` — cannot be cut into
epochs at all: the first executor's segment joins to something total, and the
second executor's segment carries no layout fields whatever. Folding instead of
joining fixes both, and it is the move §3.3 already made for λ-as-data ("a mover
is λ's delta encoding") applied one level up, to λ-as-configuration.

Two entries in v2 §2.2's P1–P4 ledger move with it: **P1** ("a prefix means
something") gets *stronger* — a prefix always denotes the current configuration —
and **P3** ("order carries information") stops being a theorem, because two writes
to one register do not commute.

Operation and schedule ISAs are one model under this reading; what varies is the
epoch's granularity, and both ends are here. A machine whose every instruction
carries a *total* configuration has one ``EmitRecord`` per ``Epoch`` — the
operation end, and every machine in the corpus, where the fold is the identity and
nothing persists. A machine that **configures before it executes** (MINISA's
``Set*VNLayout … ExecuteMapping``) declares its setters with ``ISA.configures``,
and their assignment then reaches every epoch until it is overwritten — so a setter
need not sit next to the instruction it configures, and one setter configures many.

An ``Epoch`` is derived from the serialized program alone — the mnemonic plus the
emitted fields — never from the planner's in-memory state. That is deliberate:
the denotational view says what the *emitted* program means, which is exactly
what a checker must consume (refactor target 3).

``σ`` also lives here (refactor target 2), minimally: ``schedule`` ASAP-places
the epoch sequence onto the microarchitecture's ``@unit``s and fills each
``Epoch.sigma`` with its ``(PE, t)``. The instruction word of an operation ISA
carries no σ — it is *derived*, from stream order, region-overlap dependences
and the units' ``(ii, depth)`` — which is why ``epochs`` yields ``sigma=None``
and ``schedule`` is a separate step. A machine whose word does carry σ's fields
(MINISA's ``ExecuteMapping`` has a ``t``) would *read* them instead; same type,
different provenance.

Statements of the model that are executable here, and enforced:

- **Totality.** A compiled emit binds every parameter — offsets from the
  allocator, shapes from ``linsolve``, residence from unification or routing, α
  from the source, schedule params from ``configure``. ``epochs`` asserts it:
  a partial configuration is not an epoch. What a *configuring* instruction binds
  is not its epoch's alone — it is installed, and the fold carries it forward — so
  the one thing still refused is a stream that **ends** configuring: those updates
  reach no kernel run, and a program whose tail computes nothing was mis-assembled.
- **The λ-fragment.** The configuration determines, per operand, which addresses
  hold which logical elements: a ``Region``, computed from the emitted numbers by
  the same ``access_map`` the solvers used. The inter-epoch interface condition —
  what Stage 2b's unification solves for and ``plan``'s movers repair — is that
  consecutive epochs agree on these regions for the values flowing between them.
- **Events, and constraints 2–4** (v2 §3.4). ``depends`` derives the program's
  dependence edges from region overlap on the serialized addresses; its RAW edges
  are exactly the (write event → read event) pairs the reachability relation
  ``R`` quantifies over. With ``schedule``'s σ, no-overbooking (2) and dependence
  (3) are evaluable for the first time; the communication-latency term of (3) is
  discharged structurally, because movement is explicit mover epochs whose own
  ``(pe, t)`` the RAW chain runs through.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .errors import AssemblyError
from .core import (
    _access_chain,
    _resolve,
    access_map,
    access_names,
    buffer_weights,
    compute_params,
    param_roles,
    residence,
)


@dataclass(frozen=True)
class Region:
    """One operand's λ-fragment: where its data sits.

    ``base`` is the placement coordinate (one component per address axis of the
    buffer) and ``map`` the residence — ``(size, stride)`` per spanning dim,
    strides in elements relative to ``base``. Together with the buffer they name
    the address set the epoch reads or writes; the interface and capacity
    constraints (v2 §3.4) quantify over exactly these."""

    buffer: str
    base: tuple
    map: tuple
    # Whether the epoch *writes* this region. Carried on the region rather than
    # recovered from the instruction's arity, because a multi-instruction epoch's
    # regions come from several instructions and there is no single arity to ask.
    writes: bool = False

    @property
    def placement(self) -> tuple:
        """Where the data sits, with no claim about who touches it — the half the
        inter-epoch interface condition compares. A producer's write and its
        consumer's read are the same placement and differ exactly in ``writes``."""
        return (self.buffer, self.base, self.map)


@dataclass(frozen=True)
class Config:
    """One epoch's total configuration: every instruction parameter, by name,
    under the role the model gives it (v2 §4.1 — a parameter's cell says who
    determines it, and by emission time every cell is filled).

    ``offsets`` + ``residence`` are the epoch's λ half (where operands sit, how
    they are packed); ``schedule`` is the chosen configuration a schedule-ISA
    machine carries in its instruction word; ``alpha`` is the one channel a
    source value enters that word; ``shapes`` are the solved extents."""

    offsets: dict
    shapes: dict
    residence: dict  # stride + ordering params: how the operands are packed
    alpha: dict
    schedule: dict


@dataclass(frozen=True)
class Sigma:
    """σ's value at one epoch: where and when it runs.

    ``pe`` is the unit that executes it — the machine's spatial axis, taken from
    ``ISA.bind`` (unbound instructions share the one implicit ``"pe"``). The
    epoch occupies its unit's issue slots over ``[start, start + issue)``; its
    writes are available at ``finish = start + issue + depth``. Whether these are
    cycles or unit steps is the ``Schedule``'s ``in_cycles``."""

    pe: str
    start: float
    issue: float
    finish: float


@dataclass(frozen=True)
class Epoch:
    """One epoch: the mnemonic naming the kernel run, its total ``Config``, and
    the per-operand ``Region``s (in access order, sources then destinations).

    ``sigma`` is its place in space-time — ``None`` as read off the wire (an
    operation ISA's instruction word carries no σ), filled by ``schedule``."""

    name: str
    config: Config
    regions: tuple
    sigma: Sigma | None = None
    # ``(mnemonic, Config)`` per instruction *issued* in this epoch's segment, the
    # one that runs the kernel **last**. One entry is the operation-ISA case;
    # several is a machine that configures before it executes (``ISA.configures``).
    # Note this is the segment's own instructions, not everything that configured
    # it: a setter installed three layers ago is in ``config``, not here.
    members: tuple = ()

    @property
    def run(self) -> Config:
        """The configuration of the instruction that runs the kernel — the fields in
        its own word, as opposed to the fold with what the machine already had
        installed. σ is placed against this one: a configuring instruction
        contributes meaning, not work."""
        return self.members[-1][1] if self.members else self.config


_FIELDS = ("offsets", "shapes", "residence", "alpha", "schedule")


def epochs(isa, emits) -> list[Epoch]:
    """Read a compiled emit stream as its epoch sequence — the program's
    ∘-composition (v2 §3.2).

    An epoch is a segment ending at the instruction that runs the kernel, and its
    configuration is the **fold** of every update up to and including that one. One
    emit is one epoch wherever every instruction's own assignment is already total
    — the operation-ISA degenerate case, and every machine in the corpus, where the
    installed configuration stays empty and the fold is the identity.

    An instruction the ISA declares as **configuring** (``ISA.configures``) writes
    configuration *registers*: its assignment is installed and reaches every later
    epoch until some instruction assigns the same field again. That is why a second
    write is an ordinary layer boundary rather than a contradiction, and why one
    ``Set*VNLayout`` group configures the whole run of ``ExecuteMapping``s that
    follows it — the shape a real MINISA trace has.

    A field written *later* wins, so the fold is a function by construction and
    there is nothing left to refuse about it. Note the consequence for naming: a
    ``Config`` is keyed by parameter name, so two configuration registers are two
    registers only if their params are named apart."""
    out = []
    pending: list = []
    # The machine's installed configuration: what a kernel run finds already set.
    installed: dict = {f: {} for f in _FIELDS}
    for rec in emits:
        spec = isa._ops[rec.name].spec
        names = access_names(spec)
        roles, _ = param_roles(spec)
        alpha_names = compute_params(spec)
        # Totality: only a compiled stream reads as epochs. A hand-written
        # (@oracle) stream makes no such promise — it may omit schedule fields
        # the simulator ignores — and is refused here rather than half-read.
        assert len(rec.addr) == len(names) and all(
            v is not None for v in rec.addr
        ), f"{rec.name}: partial address list — not a total configuration"
        assert len(rec.compute) == len(alpha_names), f"{rec.name}: partial α list"
        assert len(rec.schedule) == len(
            spec.schedule_domains
        ), f"{rec.name}: partial schedule fields — not a total configuration"
        groups: dict = {"offset": {}, "shape": {}, "stride": {}, "layout": {}}
        for i, value in enumerate(rec.addr):
            groups[roles[i]][names[i]] = value
        config = Config(
            offsets=groups["offset"],
            shapes=groups["shape"],
            residence=groups["stride"] | groups["layout"],
            alpha=dict(zip(alpha_names, rec.compute)),
            schedule=dict(zip(spec.schedule_domains, rec.schedule)),
        )
        pending.append((rec.name, config, _regions(spec, rec.addr)))
        if spec.configures:
            # A configuration write: installed, and live until overwritten.
            for name in _FIELDS:
                installed[name] = installed[name] | getattr(config, name)
            continue
        out.append(_join(pending, installed))
        pending = []
    if pending:
        raise AssemblyError(
            f"the stream ends with {[n for n, _c, _r in pending]}, which only "
            f"configure — no kernel run ever reads what they install, so the "
            f"program's last segment computes nothing"
        )
    return out


def _join(members: list, installed: dict) -> Epoch:
    """One epoch: the configuration its kernel runs under, and the events issued.

    The configuration is what the machine already had installed, overlaid with the
    fields the running instruction carries in its own word — an instruction's word
    is not a register, so it configures its own run and nothing after it. The
    *regions* are the segment's own, and only the segment's: a configuration write
    is one event at one point in the stream, however long its effect lasts, and
    replaying it into every later epoch would invent dependences that are not there.
    """
    name, own, _regions_of_run = members[-1]
    joined = Config(**{f: installed[f] | getattr(own, f) for f in _FIELDS})
    return Epoch(
        name,
        joined,
        tuple(r for _n, _c, rs in members for r in rs),
        members=tuple((n, c) for n, c, _r in members),
    )


def _regions(spec, addr: list) -> tuple:
    """Each operand's ``Region`` under the emitted parameter values — the epoch's
    λ-fragment, recovered from the serialized numbers by the same ``access_map``
    the solvers used, so the reading and the solving cannot drift apart."""
    params = dict(enumerate(addr))
    n_src = len(spec.sources)
    out = []
    for pos, stripped, root in _access_chain(spec):
        pattern = stripped[0] if stripped else root
        base = tuple(_resolve(b, params) for b in root.basis)
        out.append(
            Region(
                root.buffer.name,
                base,
                residence(access_map(pattern, params)),
                pos >= n_src,
            )
        )
    return tuple(out)


# ==========================================================================#
# σ, minimally: dependences from region overlap + an ASAP placement onto units
# ==========================================================================#


@dataclass(frozen=True)
class Dep:
    """One dependence edge ``src -> dst`` (epoch indices, ``src`` earlier in the
    stream), derived from region overlap in ``buffer``.

    A dependence here is an **address** fact of the serialized program — no value
    identities survive emission, and none are needed: WAR/WAW edges are real
    because the allocator reuses slots (the linear order was load-bearing, and
    these edges say exactly where), and the RAW edges are the (write event →
    read event) pairs constraint 4's reachability relation quantifies over."""

    src: int
    dst: int
    kind: str  # "raw" | "war" | "waw"
    buffer: str


@dataclass(frozen=True)
class Schedule:
    """A σ for the whole program: every epoch placed (``Epoch.sigma`` filled),
    plus the dependence edges the placement respects.

    ``in_cycles`` says the time base: ``True`` when every emitted instruction is
    bound to a ``@unit`` with a declared ``ISA.latency`` — σ is then in cycles
    and ``makespan`` sits between ``bottleneck_cycles()`` (everything overlaps)
    and ``cycles()`` (nothing does), the point in that bracket this schedule
    actually achieves. ``False`` degrades every epoch to one unit step rather
    than reporting made-up cycles; σ's ordinal content — constraints 2 and 3 —
    is unchanged by the degradation, only the units of measure are."""

    epochs: tuple
    deps: tuple
    in_cycles: bool

    @property
    def makespan(self) -> float:
        return max((e.sigma.finish for e in self.epochs), default=0.0)


def _extent(isa, region: Region) -> tuple[int, int]:
    """The element interval ``[lo, hi]`` (inclusive) a region touches, flattening
    its base coordinate by the buffer's own axis weights.

    An **over-approximation**: a strided region's holes are inside the interval,
    so overlap may report a dependence where the element sets are disjoint. For
    deriving an order that is the sound direction — a false edge only serializes;
    a missed one would reorder a real conflict. Exact emptiness of strided sets
    is the checker's business (refactor target 3, where Presburger machinery
    arrives for constraint reasons of its own)."""
    weight = buffer_weights(isa.buffers[region.buffer])
    lo = sum(c * w for c, w in zip(region.base, weight))
    span = 0
    for size, stride in region.map:
        assert isinstance(stride, int) and stride >= 0, f"unresolved map in {region}"
        span += (size - 1) * stride
    return lo, lo + span


def region_elements(isa, region: Region) -> frozenset:
    """The exact element addresses a region touches — the checker's substrate.

    A compiled program is fully concrete, so a region is a *finite* set and
    enumerating it is the decision procedure: intersection, difference and
    membership below are exact, not approximate. This is also the one seam a
    Presburger backend would replace — the day something **symbolic** needs
    checking (a parametric mapping family, refactor targets 4–5), sets become
    quasi-affine relations and this function's callers are the interface; until
    then ISL would buy nothing a ``frozenset`` does not already give exactly."""
    weight = buffer_weights(isa.buffers[region.buffer])
    addrs = {sum(c * w for c, w in zip(region.base, weight))}
    for size, stride in region.map:
        addrs = {a + i * stride for a in addrs for i in range(size)}
    return frozenset(addrs)


def depends(isa, eps: list, exact: bool = False) -> tuple:
    """The dependence edges among ``eps``, from region overlap alone.

    For every earlier/later pair whose regions intersect in one buffer with at
    least one side writing: write→read is RAW, read→write WAR, write→write WAW.
    Reads never conflict. Sources precede destinations in ``Epoch.regions``, so
    which side writes is read off the instruction's own arity.

    ``exact`` refines each interval hit by exact element-set intersection
    (``region_elements``), so interleaved strided accesses whose bounding
    intervals overlap but whose elements are disjoint contribute no edge. The
    default stays conservative: for *deriving* an order (``schedule``) a false
    edge only serializes, while a checker wants the exact set — an
    over-approximate edge there rejects legal schedules, exactness removes the
    incompleteness (never the soundness, which both modes have)."""
    sides = [
        [(r.writes, r.buffer, *_extent(isa, r), r) for r in e.regions] for e in eps
    ]
    memo: dict = {}

    def elems(r: Region) -> frozenset:
        if r not in memo:
            memo[r] = region_elements(isa, r)
        return memo[r]

    out = []
    for j in range(len(eps)):
        for i in range(j):
            hits = set()
            for wi, bi, lo_i, hi_i, ri in sides[i]:
                for wj, bj, lo_j, hi_j, rj in sides[j]:
                    if bi != bj or not (wi or wj) or lo_i > hi_j or lo_j > hi_i:
                        continue
                    if exact and not (elems(ri) & elems(rj)):
                        continue
                    hits.add((("waw" if wj else "raw") if wi else "war", bi))
            out += [Dep(i, j, kind, buf) for kind, buf in sorted(hits)]
    return tuple(out)


def pe_names(isa, eps: list, instance: list) -> list:
    """σ's spatial axis when a lowering imported one: the unit an instruction is
    bound to, refined by the instance it was placed on.

    ``instance`` is per *emit* — the granularity the planner records it at — while
    an epoch may join several, so each epoch takes the instance of the member that
    runs its kernel. ``None`` entries are instructions the compiler placed itself.
    With no instance anywhere this is exactly what ``schedule`` derives on its own,
    so the import *adds* spatial information rather than replacing the machine's."""
    if not any(k for k in instance):
        return None
    out, k = [], 0
    for e in eps:
        k += max(len(e.members), 1)
        spec = isa._ops[e.name].spec
        unit = spec.unit.func_name if spec.unit is not None else "pe"
        out.append(f"{unit}#{instance[k - 1] or 0}")
    return out


def schedule(isa, eps: list, pes: list | None = None) -> Schedule:
    """σ's minimal existence: place each epoch as soon as its dependences and its
    unit allow (ASAP over the emit order), never reordering the stream's choices
    — this derives the σ the emitted program *has*, it does not search for a
    better one.

    An epoch starts at ``max(latest predecessor finish, its unit's next free
    issue slot)``; the unit is released after ``issue`` (back-to-back issue, the
    ``unit_cycles`` depth-paid-once model) while the value arrives ``depth``
    later. The degenerate case is a theorem, not a mode: with no ``@unit``s every
    epoch shares one PE at unit time, and ASAP reproduces ``start = stream
    index`` — the linear order, recovered as σ.

    ``pes`` overrides the spatial axis per epoch. An operation ISA's word names
    no instance — ``ISA.bind`` gives one unit per *mnemonic*, so every invocation
    of an instruction lands on it — but an imported mapping does say which
    instance runs what (``mapping.assemble``). Supplying it separates epochs the
    derivation would have serialized, and is the one part of σ this frontend
    cannot recover on its own."""
    deps = depends(isa, eps)
    preds: dict = {}
    for d in deps:
        preds.setdefault(d.dst, set()).add(d.src)
    in_cycles = all(
        (s := isa._ops[e.name].spec).unit is not None and s.unit_latency.declared
        for e in eps
    )
    placed: list = []
    free: dict = {}
    for j, e in enumerate(eps):
        spec = isa._ops[e.name].spec
        pe = (
            pes[j]
            if pes is not None
            else (spec.unit.func_name if spec.unit is not None else "pe")
        )
        if in_cycles:
            names = access_names(spec)
            shapes = {names.index(n): v for n, v in e.run.shapes.items()}
            lat = spec.unit_latency
            issue = lat.ii * spec.trips_at(shapes, e.run.schedule)
            depth = float(lat.depth)
        else:
            issue, depth = 1.0, 0.0
        ready = max((placed[i].sigma.finish for i in preds.get(j, ())), default=0.0)
        start = max(ready, free.get(pe, 0.0))
        free[pe] = start + issue
        placed.append(replace(e, sigma=Sigma(pe, start, issue, start + issue + depth)))
    return Schedule(tuple(placed), deps, in_cycles)
