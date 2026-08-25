# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the RTL tests."""

from __future__ import annotations

import re

from allo.backend.rtl import RTL, ScheduleResult, default_device


# Operator latencies keyed by (kind, first argument dtype). The dtype and not
# just its width, since a float and an integer multiply are both `mul` at 32
# bits. `optype` may be a plain string for a core that binds by MLIR mnemonic
# rather than by abstract kind.
def _key(op):
    kind = getattr(op.optype, "value", op.optype)
    return (kind, op.parse_argument_annotations()[0].name)


# The default clock the chaining scheduler cuts to. A test that picks a clock to
# make a chain fit or not fit derives the period from these rather than
# restating the device's numbers.
PERIOD_NS = 1000.0 / default_device.default_freq_mhz

# What one register-to-register hop costs before any logic. Paid once per cycle
# rather than once per operator, so a chain of n operators costs
# `REG_NS + n * comb_step_ns(...)` and not `n * comb_ns(...)`.
REG_NS = default_device.reg_delay_ns


def _row_need(timing) -> float:
    """What one operator row needs for a cycle of its own, the same max the
    compiler's derate walk takes: its input cone above the register floor, its
    output cone, and the period its internal stages are warranted at."""
    return max(REG_NS + timing.in_delay_ns, timing.out_delay_ns, timing.min_period_ns)


def _op_row(kind: str, dtype: str, latency: int):
    """The device's operator row for one kind, signature and depth."""
    return next(
        o
        for o in default_device.operators
        if _key(o) == (kind, dtype) and o.timing.latency == latency
    )


def period_need(kind: str, dtype: str, latency: int) -> float:
    """What one row needs for a cycle of its own; see :func:`_row_need`."""
    return _row_need(_op_row(kind, dtype, latency).timing)


# The library binds the shortest candidate of a kind and signature that fits the
# clock, so that is the latency a test predicts a schedule against. A row whose
# cone or warranted period exceeds the period is not a candidate.
_LAT: dict[tuple[str, str], int] = {}
for _o in default_device.operators:
    if _row_need(_o.timing) > PERIOD_NS:
        continue
    _k = _key(_o)
    _LAT[_k] = min(_LAT.get(_k, _o.timing.latency), _o.timing.latency)

FADD = FSUB = _LAT[("add", "float32")]  # floating-point add/sub latency (cycles)
FMUL = _LAT[("mul", "float32")]  # floating-point multiply latency
FDIV = _LAT[("div", "float32")]  # floating-point divide latency
IMUL = _LAT[("mul", "int32")]  # 32-bit integer multiply (a DSP core, not comb)
IMUL64 = _LAT[("mul", "int64")]  # 64-bit integer multiply
IDIV = _LAT[("divsi", "int32")]  # 32-bit signed integer divide
MEM = default_device.storage["lutram"].read_latency  # default read/write
MEM_URAM = default_device.storage["uram"].read_latency


# A device delay row is a curve over operand width, so a caller names the width
# it means. 32 is the default because these kernels are i32.
def comb_ns(kind: str, width: int = 32) -> float:
    """What ``kind`` costs on a path starting at a register, the register floor
    included. Evaluated by the compiler's own cost evaluator."""
    return default_device.comb_delay(kind, width)


def comb_step_ns(kind: str, width: int = 32) -> float:
    """What ``kind`` adds to a path that already left a register, the spacing the
    chaining solve leaves between two chained operators."""
    return max(0.0, comb_ns(kind, width) - REG_NS)


# A memory-carried accumulate (`M[x] += ...`) closes a distance-1 recurrence
# read -> add -> write, but store->load forwarding hands the next read the
# store's datum on an address match, so the II is read + add rather than the
# full round trip. The read's datum reaches the adder inside the read's own
# cycle only where the read's cone and the adder's input cone fit the period
# together; where they do not the chain breaks and the recurrence pays one more
# cycle. A scalar-carried accumulate keeps the partial in a register, so its II
# is just the add latency.
def mem_reduce_ii(storage: str = "lutram") -> int:
    """The II of a float accumulate carried through one storage row."""
    row = default_device.storage[storage]
    arrival = row.read_delay_ns + _op_row("add", "float32", FADD).timing.in_delay_ns
    return row.read_latency + FADD + (0 if arrival <= PERIOD_NS else 1)


MEM_REDUCE_II = mem_reduce_ii()


def _to_rtl(kernel, **kw) -> RTL:
    """Export ``kernel`` to the RTL backend."""
    return kernel.schedule().export("rtl", **kw)


def _sched(kernel, **kw) -> ScheduleResult:
    """Schedule ``kernel`` through the RTL backend."""
    return _to_rtl(kernel, **kw).schedule()


def _latency(kernel, **kw):
    """Whole-kernel latency (cycles) of ``kernel`` scheduled on its own; ``None``
    when a trip count is not statically known."""
    return _sched(kernel, **kw).func(kernel.__name__).latency


def _iis(regions):
    """Sorted IIs of ``regions``; a dynamic-trip sequential wrapper (``ii`` is
    ``None``) is skipped."""
    return sorted(r.interval for r in regions if r.interval is not None)


def _impls(result):
    """The IP modules the schedule binds ops to, across every region."""
    return {o.impl for r in result.regions(wrappers=True) for o in r.ops if o.impl}


def _outer(func, kind):
    """``func``'s own outermost regions (depth 0) of ``kind``.

    A region nested in a container is reported at a greater depth, so this is
    the caller's own top-level structure rather than anything a child of it
    contributes."""
    return [r for r in func.regions if r.kind == kind and r.depth == 0]


# --- structural reading of the scheduled DCP IR ------------------------------


def _walk(root, name=None):
    """Every op nested under ``root`` in program order, optionally only those
    named ``name`` (a full op name, e.g. ``"allo.dcp.select"``)."""
    op = getattr(root, "operation", root)
    found = []
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                if name is None or child.operation.name == name:
                    found.append(child.operation)
                found += _walk(child, name)
    return found


def _attr(op, name):
    """``op``'s ``name`` attribute as a plain Python value; a unit attribute,
    which carries none, reads as itself."""
    a = op.attributes[name]
    return getattr(a, "value", a)


class _Scope:
    """A span of DCP IR read op by op: the whole module, or one kernel."""

    def __init__(self, root):
        self.root = root

    def ops(self, name=None):
        return _walk(self.root, name)

    def count(self, name):
        return len(_walk(self.root, name))

    def has(self, name):
        return bool(_walk(self.root, name))

    def attrs(self, op_name, attr):
        """The value of ``attr`` on every ``op_name`` op that carries it."""
        return [_attr(o, attr) for o in self.ops(op_name) if attr in o.attributes]


class Dcp(_Scope):
    """The scheduled DCP module, read through the MLIR bindings.

    The printed IR is a debugging aid, not an interface: nothing holds its
    layout still. A structural assertion asks the module for its ops and their
    attributes instead, and reaches for :class:`ScheduleResult` first for
    anything the schedule already reports.
    """

    def __init__(self, rtl):
        super().__init__(rtl.dcp_module)
        self.kernels = {
            _attr(op, "sym_name"): DcpFunc(op)
            for op in _walk(self.root, "allo.dcp.module")
        }

    def func(self, name):
        assert name in self.kernels, f"no kernel {name!r} in {list(self.kernels)}"
        return self.kernels[name]


class DcpFunc(_Scope):
    """One scheduled kernel (an ``allo.dcp.module``).

    Its callees are separate top-level kernels rather than nested bodies, so an
    assertion made here is about this kernel's own span and cannot be satisfied
    by one of its children.
    """

    def __init__(self, op):
        super().__init__(op)
        self.name = _attr(op, "sym_name")

    def callees(self, *, spawned=None):
        """The callee of each ``dcp.instance``, in program order. ``spawned``
        selects only the `await` spawns (``True``) or only the sequenced calls
        (``False``); an `allo.async` carrier is what marks a spawn."""
        return [
            _attr(i, "callee")
            for i in self.ops("allo.dcp.instance")
            if spawned is None or ("allo.async" in i.attributes) == spawned
        ]

    def arg_attrs(self, name):
        """The value of argument attribute ``name`` on each argument carrying
        it, in argument order."""
        if "arg_attrs" not in self.root.attributes:
            return []
        return [d[name] for d in self.root.attributes["arg_attrs"] if name in d]

    def accesses(self, memref):
        """The scheduled loads and stores that reach ``memref`` (an SSA value),
        in program order."""
        return [
            o
            for o in self.ops()
            if o.name in ("allo.dcp.load", "allo.dcp.store")
            and any(v == memref for v in o.operands)
        ]


# --- structural reading of the emitted RTL -----------------------------------

_DEF = re.compile(r"^%([\w.$-]+) = (.+)$")
_COMPREG = re.compile(
    r'^seq\.compreg(\.ce)? (?:name "([^"]*)" )?%([\w.$-]+), %[\w.$-]+(?:, %([\w.$-]+))?'
)
_MUX = re.compile(r"^comb\.mux (?:bin )?%([\w.$-]+), %([\w.$-]+), %([\w.$-]+)")
_HINT = re.compile(r'sv\.namehint = "([^"]+)"')
_OPERAND = re.compile(r"%([\w.$-]+)")


class Mod:
    """The ops of one ``hw.module``, indexed for structural assertions.

    Text-level rather than a real parse: the tests that use it are locks on the
    *shape* of a small, named piece of the emitted hardware (a stall shell, a
    controller), so what they need is the def of each SSA value, its namehint,
    and the register list rather than an IR data structure.
    """

    def __init__(self, mlir, name):
        body, seen = [], False
        for line in mlir.splitlines():
            s = line.strip()
            if s.startswith(f"hw.module @{name}("):
                seen = True
                continue
            if seen:
                if s == "}":
                    break
                body.append(s)
        assert seen, f"no hw.module @{name} in the emitted module"
        # The module body verbatim, for what the per-op index cannot hold: a
        # multi-result op (an `hw.instance`) defines no single value and so has
        # no entry in `defs`.
        self.text = "\n".join(body)
        self.defs, self.hint, self.regs, self.ce = {}, {}, [], {}
        for s in body:
            m = _DEF.match(s)
            if not m:
                continue
            res, rhs = m.group(1), m.group(2)
            self.defs[res] = rhs
            h = _HINT.search(rhs)
            if h:
                self.hint[res] = h.group(1)
            r = _COMPREG.match(rhs)
            if r:
                self.regs.append((r.group(2) or res, res, r.group(3)))
                if r.group(1):  # a `seq.compreg.ce` names its enable directly
                    self.ce[res] = r.group(4)

    def hinted(self, name):
        """The single SSA value labelled ``sv.namehint = name``."""
        hits = [v for v, h in self.hint.items() if h == name]
        assert len(hits) == 1, f"expected one {name!r}, got {hits}"
        return hits[0]

    def hints_like(self, pattern):
        return sorted({h for h in self.hint.values() if re.search(pattern, h)})

    def signal(self, name):
        """The SSA value carrying ``name``.

        A named register prints AS its name (CIRCT takes the SSA name from
        ``seq.compreg``'s ``name`` attribute); named combinational logic carries
        an ``sv.namehint`` instead. Callers of a control signal should not have
        to know which of the two the emitter happened to build.
        """
        return name if name in self.defs else self.hinted(name)

    def regions_with(self, suffix):
        """The region ids for which an ``r<N>_<suffix>`` signal exists."""
        pat = re.compile(rf"^r(\d+)_{suffix}$")
        names = set(self.defs) | set(self.hint.values())
        return sorted(int(m.group(1)) for m in map(pat.match, names) if m)

    def operands(self, v):
        return _OPERAND.findall(self.defs.get(v, ""))

    def mux(self, v):
        """``(sel, t, f)`` of ``v`` when it is a 2:1 mux, else ``None``."""
        m = _MUX.match(self.defs.get(v, ""))
        return m.groups() if m else None

    def enable_of(self, reg, inp):
        """The enable selecting ``reg``'s next value, or None if unconditional.

        A ``seq.compreg.ce`` names its enable as an operand; a plain register
        spelled as a self-hold (``compreg(mux(en, in, reg))``) yields the mux
        select.
        """
        if reg in self.ce:
            return self.ce[reg]
        m = self.mux(inp)
        return m[0] if m and m[2] == reg else None

    def cone(self, root, limit=64):
        """The SSA values reachable from ``root`` through comb logic.

        Leaves are module ports, instance results and registers, anything with
        no combinational def in this module. They are IN the result, since
        "does this signal reach `start`" is exactly the sort of question a
        control-structure lock asks.
        """
        seen, work = set(), [root]
        while work and len(seen) < limit:
            v = work.pop()
            if v in seen:
                continue
            seen.add(v)
            rhs = self.defs.get(v, "")
            if not rhs or rhs.startswith("seq."):  # a register ends the cone
                continue
            work += _OPERAND.findall(rhs)
        return seen

    def reg_named(self, label):
        hits = [(r, i) for lb, r, i in self.regs if lb == label]
        assert len(hits) == 1, f"expected one {label!r} register, got {hits}"
        return hits[0]


def _one_region(m):
    """The single done-driven region of `m` (the one that emits an `r<N>_fire`)."""
    ids = m.regions_with("fire")
    assert len(ids) == 1, f"expected one done-driven region, got {ids}"
    return ids[0]


def _hold_done(m, region):
    """The set-pulse of region `region`'s done latch.

    `holdDone` is `done = compreg(mux(start, false, mux(set, true, done)))`:
    cleared by the region start so a retriggered region re-edges, set by the
    completion pulse. Returns `set`, having checked the shape.
    """
    reg, inp = m.reg_named(f"r{region}_done")
    clear = m.mux(inp)
    assert clear and clear[1].startswith("false"), f"r{region}_done not cleared: {inp}"
    hold = m.mux(clear[2])
    assert (
        hold and hold[1].startswith("true") and hold[2] == reg
    ), f"r{region}_done is not a hold latch: {clear[2]}"
    return hold[0]
