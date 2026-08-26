# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The device schema for the RTL backend: what a device declares, how a cost is
expressed, and how both reach the IR."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from typing import NamedTuple

from ...lang.ip import OperatorIP, OperatorType
from .sim.ip_models import OpDesc, Ty


class Realization(NamedTuple):
    """Device-supplied project content beyond the emitted RTL: source and build
    files keyed by name, plus the extern modules it cannot build."""

    files: dict[str, str]
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class Resource:
    """A device resource: a named counter with a capacity.

    Resource names are the device's own vocabulary; the compiler only adds and
    multiplies them. ``capacity`` is a price input, not a constraint (regions
    schedule independently). ``weight`` scales the derived scarcity price.
    """

    name: str
    capacity: int
    weight: float = 1.0


@dataclass(frozen=True)
class Cost:
    """What one realization spends of one resource, as a function of one of the
    realization's parameters (an operand width, a mux's fan-in).

    Build these through :func:`Const` and friends. The shape is structural and
    only the coefficients are measured; a non-structural shape belongs in a
    :func:`Table`.
    """

    form: str
    coeffs: tuple[float, ...]
    #: The two arms of a :func:`Piecewise`, and empty for every other form.
    arms: tuple[Cost, ...] = ()

    def _attr(self):
        """This cost as an ``#allo.cost``, in whatever context is current."""
        from ..._mlir.dialects.allo import CostAttr

        return CostAttr.get(
            self.form,
            [float(c) for c in self.coeffs],
            [a._attr() for a in self.arms],
        )


def Const(value: float) -> Cost:
    """A fixed amount, whatever the parameter."""
    return Cost("const", (float(value),))


def Linear(coeff: float, base: float = 0.0) -> Cost:
    """``base + coeff * p``."""
    return Cost("linear", (float(base), float(coeff)))


def Quadratic(coeff: float) -> Cost:
    """``coeff * p * p``, the shape of a divider."""
    return Cost("quadratic", (float(coeff),))


def Step(threshold: float, below_coeff: float, above: float) -> Cost:
    """``p < threshold ? below_coeff * p : above``: a shift-register cliff, where
    a chain past the threshold stops being flip-flops and stops growing."""
    return Cost("step", (float(threshold), float(below_coeff), float(above)))


def Table(points: dict[int, float]) -> Cost:
    """Measured point by point. A parameter between two points takes the lower
    one's value, and one outside the first and last is not measured at all.

    A staircase, which fits a quantity that really steps: a multiply takes a
    whole number of DSP slices, never 1.7 of one.

    Sampling a continuous quantity into one under-states it at every parameter
    between two points, and nothing downstream re-checks a timing model: a
    48-bit divide reading the 32-bit delay row is 45% short.
    """
    if not points:
        raise ValueError("a cost table needs at least one point")
    flat: list[float] = []
    for p in sorted(points):
        flat += [float(p), float(points[p])]
    return Cost("table", tuple(flat))


def Interp(points: dict[int, float]) -> Cost:
    """The same measured points as a :func:`Table`, read continuously: a
    parameter between two points takes their linear interpolation, and one
    outside the first and last is not measured at all.

    For a quantity continuous in its parameter, such as an operator's delay in
    its operand width.
    """
    if not points:
        raise ValueError("a cost table needs at least one point")
    flat: list[float] = []
    for p in sorted(points):
        flat += [float(p), float(points[p])]
    return Cost("interp", tuple(flat))


def Piecewise(bp: float, below: Cost, above: Cost) -> Cost:
    """``p < breakpoint ? below(p) : above(p)``, with arms of any form.

    The general form of :func:`Step`.
    """
    return Cost("piecewise", (float(bp),), (below, above))


#: What one realization spends: ``(resource name, one cost factor per parameter
#: of its kind)`` pairs, as ``#allo.res_use`` carries. One pair is one product
#: term; a resource that names several is spent their sum.
Spend = tuple[tuple[str, tuple[Cost, ...]], ...]


def _terms(cost: Cost | Sequence) -> tuple[tuple[Cost, ...], ...]:
    """A ``uses`` value as product terms. A :class:`Cost`, or a sequence of one
    per parameter, is ONE term; a sequence of those is a sum of them."""
    if isinstance(cost, Cost):
        return ((cost,),)
    seq = tuple(cost)
    if seq and isinstance(seq[0], Cost):
        return (seq,)
    return tuple(tuple(term) for term in seq)


#: Cache of built `uses` attributes, keyed by the declaration they came from.
_COST_ATTRS: dict[Spend, object] = {}

#: The same for single delay costs (a `dcp.comb` row's delay), which are not a
#: `Spend` and cannot share the cache above.
_DELAY_ATTRS: dict[Cost, object] = {}


@lru_cache(maxsize=None)
def _scratch_context():
    """The context the evaluation-only attributes below are uniqued in, one per
    process. Not the RTL module's context: an attribute holds its context alive,
    and :meth:`Device.price` is called with no module in reach."""
    from ..._mlir.ir import Context
    from ..._mlir.dialects.allo import register_dialect

    ctx = Context()
    register_dialect(ctx)
    return ctx


def _res_use_array(spent: Spend, scope: str = ""):
    """``spent`` as an ``#allo.res_use`` array, in whatever context is current.
    ``scope`` names the device symbol a reference from outside the device's
    region has to reach through, giving ``@u55c::@lut`` rather than ``@lut``."""
    from ..._mlir.ir import ArrayAttr, SymbolRefAttr
    from ..._mlir.dialects.allo import ResourceUseAttr

    return ArrayAttr.get(
        [
            ResourceUseAttr.get(
                SymbolRefAttr.get([scope, name] if scope else [name]),
                [c._attr() for c in factors],
            )
            for name, factors in spent
        ]
    )


def _res_use_attr(spent: Spend):
    """The ``#allo.res_use`` array for ``spent``, for evaluation only."""
    attr = _COST_ATTRS.get(spent)
    if attr is None:
        with _scratch_context():
            attr = _COST_ATTRS[spent] = _res_use_array(spent)
    return attr


def _cost_attr(cost: Cost):
    """The ``#allo.cost`` for one cost, for evaluation only."""
    attr = _DELAY_ATTRS.get(cost)
    if attr is None:
        with _scratch_context():
            attr = _DELAY_ATTRS[cost] = cost._attr()
    return attr


def _measured_over(cost: Cost) -> str:
    """A cost's measured points, for a diagnostic."""
    if cost.form in {"table", "interp"}:
        return f"{cost.form} measured over {cost.coeffs[0]:g}..{cost.coeffs[-2]:g}"
    return repr(cost)


def _unmeasured(uses: Spend, params: Sequence[int]) -> str:
    """Which cost of ``uses`` the compiler's evaluator declined to read at
    ``params``, named for a diagnostic."""
    for name, factors in uses:
        if len(factors) != len(params):
            continue  # a lone Tiled, which reads the whole tuple
        for factor, param in zip(factors, params):
            if _cost_attr(factor).evaluate(int(param)) is None:
                return f"the cost of {name!r} is a {_measured_over(factor)}"
    return "a cost is not measured"


def Tiled(bits_per_tile: int, offset: float = 0.0) -> Cost:
    """``ceil((depth * width + offset) / bits_per_tile)``: the shape of a tiled
    memory.

    Standing alone it reads the whole parameter tuple: a block-RAM tile holds so
    many bits however the array is cut, which puts the product inside the
    ceiling and does not separate. As one of a full set of factors it tiles its
    own parameter instead, ``ceil((p + offset) / bits_per_tile)``.

    ``offset`` is how many of the parameter's items the tiles do not hold."""
    if bits_per_tile <= 0:
        raise ValueError("a tile holds a positive number of bits")
    if offset:
        return Cost("tiled", (float(bits_per_tile), float(offset)))
    return Cost("tiled", (float(bits_per_tile),))


@dataclass(frozen=True)
class Storage:  # pylint: disable=too-many-instance-attributes
    """A storage realization: one buildable structure an array can live in.

    Not a resource but something the device builds out of resources, with its
    own timing and ports; ``uses`` names what it spends. ``allo.bind.storage
    impl=`` names one. An array with no explicit choice takes
    :meth:`Device.set_default_storage`, else the cheapest pinnable row at the
    least access latency.
    """

    name: str
    read_latency: int
    write_latency: int
    read_delay_ns: float
    write_delay_ns: float
    # The non-memory row: one cell per element, no address, no port limit. A
    # completely partitioned array resolves here; a device declares at most one.
    is_scatter: bool = False
    # A constant lookup built out of logic: no address bus, no port limit. Only
    # a read-only array declared with contents resolves here; a device declares
    # at most one.
    is_table: bool = False
    # Ports of each direction per instance, `None` for no limit. Per instance,
    # not per array: the compiler decides how many instances hold an array,
    # every copy taking every write.
    inst_reads: int | None = None
    inst_writes: int | None = None
    # Ports per instance shared across directions, each serving one read or
    # write per cycle. `None` where the directions are independent structures,
    # as a LUT RAM's write port and its one addressed read are.
    inst_ports: int | None = None
    # Vendor attribute pinning an array to this structure, stamped on the emitted
    # declaration (Xilinx `ram_style = "block"`). `None` leaves the choice to
    # the synthesizer.
    ram_style: str | None = None
    # Whether the structure powers up holding contents. False for one that powers
    # up undefined (an UltraRAM); a compile-time-initialized array cannot bind
    # there.
    can_init: bool = True
    # Whether a read returns the OLD contents under a same-cycle write to the
    # same element, in hardware and not merely in RTL simulation (a LUT RAM's
    # asynchronous read). A block RAM's cross-port same-address collision is
    # undefined in silicon and must not carry this.
    read_first: bool = False
    # What it spends, over `(depth, width)`: each term is two cost factors or
    # one `Tiled`.
    uses: Spend = ()
    # Read delay (ns) over the array's depth, and the factor its width scales it
    # by, for a row whose cone grows with the array (a constant table's does, an
    # addressed row's does not). `read_delay_ns` is this curve at the reference
    # shape.
    read_delay_depth: Cost | None = None
    read_delay_width: Cost | None = None


@dataclass(frozen=True)
class StreamTiming:
    """Get/put timing of a stream channel (a FIFO, not bound array storage)."""

    read_latency: int
    write_latency: int
    read_delay_ns: float
    write_delay_ns: float


class CombKind(Enum):
    """A combinational operator kind whose chaining delay a device may characterize."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    REM = "rem"
    NEG = "neg"  # `arith.negf` only: a float sign flip, not an integer negate
    # `arith.minsi`/`minui`/`maxsi`/`maxui`, a compare feeding a multiplexer. A
    # fabric with no row prices them at the default 0.1 ns and free.
    MIN = "min"
    MAX = "max"
    CMP = "cmp"
    AND = "and"
    OR = "or"
    XOR = "xor"
    SHL = "shl"
    SHR = "shr"
    SELECT = "select"
    INT_CAST = "icast"
    INT_FLOAT_CAST = "ifcast"
    FLOAT_CAST = "fcast"


# One attribute per kind of thing a part declares, so the count tracks the
# vocabulary rather than any coupling between them.
# pylint: disable=too-many-instance-attributes
class Device:
    """A hardware platform: what it has (resources) and what it can realize
    (storage structures, native operator kinds, operator IPs, multiplexers,
    delay chains), plus a default synthesis frequency. Built fluently through
    ``add_resource`` / ``add_storage`` / ``set_comb_delay`` / ``add_operator``
    and the ``set_*_uses`` declarations."""

    def __init__(
        self,
        name: str,
        *,
        part: str = "",
        fabric: str = "",
        grade: str = "",
    ):
        self.name = name
        # Identity only; nothing in the compiler switches on these.
        self.part = part
        self.fabric = fabric
        self.grade = grade
        # Native chaining delays: kind -> ns as a function of the operand width.
        self.comb: dict[str, Cost] = {}
        # What a register-to-register path with no operator in it costs (ns).
        self.reg_delay_ns: float = 0.0
        self.resources: dict[str, Resource] = {}
        self.comb_uses: dict[str, Spend] = {}  # comb kind -> what it spends
        self.operator_uses: dict[str, Spend] = {}  # IP symbol -> what it spends
        # IP symbol -> which of the realizer's alternative builds of that core
        # this row was measured on, absent where it is the default one.
        self.operator_variant: dict[str, str] = {}
        # What a multiplexer spends. This and the chain rows are unnamed
        # structures the emitter builds, one row each.
        self.mux_uses: Spend = ()
        # Routed marginal delay (ns) of a one-hot select over its fan-in at a
        # 32-bit reference width, and the unitless factor its width scales it by.
        # None on an uncharacterized device, priced by a conservative formula.
        self.mux_delay: Cost | None = None
        self.mux_delay_width: Cost | None = None
        self.chain_uses: Spend = ()
        # A reset-free chain, extractable as a shift register; the ledger's
        # `reset` flag picks between the two rows.
        self.chain_uses_norst: Spend = ()
        self.storage: dict[str, Storage] = {}
        # Routed logic sites per modeled LUT. Area rows count LUT instances;
        # post-route combining packs several into one site, so an estimate
        # quoted against a utilization report scales by this.
        self.lut_packing: float = 1.0
        # A name, not a handle, so redeclaring a row keeps the default valid.
        self.default_storage: str | None = None
        self.stream_timing: StreamTiming | None = None
        # Built-in and user `@operator_ip` cores. `operator_uses` above is keyed
        # on their `symbol`.
        self.operators: list[OperatorIP] = []
        self.default_freq_mhz: float = 100.0
        # Builds the extern operator modules the emitter instantiates, called as
        # ``realizer(interfaces, device)``. ``None`` leaves the externs as black
        # boxes.
        self.realizer: Callable[..., Realization] | None = None

    def _spend(
        self,
        what: str,
        params: str,
        uses: dict[Resource, Cost | Sequence] | None,
    ) -> Spend:
        """``uses`` as ``(resource name, factors)`` pairs, one per product term,
        checked against the parameter tuple ``params`` of the realization's
        kind: one factor per parameter, or the single :func:`Tiled` that reads
        them together. A resource whose value is a sequence of terms is spent
        their sum."""
        arity = len(params.split(","))
        spent: list[tuple[str, tuple[Cost, ...]]] = []
        for resource, cost in (uses or {}).items():
            if self.resources.get(resource.name) is not resource:
                raise ValueError(
                    f"{resource.name!r} is not a resource of device {self.name!r}"
                )
            for factors in _terms(cost):
                whole_tuple = len(factors) == 1 and factors[0].form == "tiled"
                if len(factors) != arity and not whole_tuple:
                    raise ValueError(
                        f"{what} is characterized by ({params}), so each term of "
                        f"its cost of {resource.name!r} is {arity} factor(s) or "
                        "one Tiled"
                    )
                spent.append((resource.name, factors))
        return tuple(spent)

    def price(self, uses: Spend, params: Sequence[int]) -> dict[str, int]:
        """What one instance of a realization spends at ``params``.

        Goes through the compiler's own ``CostAttr::evaluate``, so a consumer
        outside the compiler (``benchmark/area.py``) reads the same measured
        model the scheduler will, rather than a second copy of the shapes.

        Raises where a cost was not measured at its parameter: a `Table` and an
        `Interp` are read and never extrapolated."""
        if not uses:
            return {}
        from ..._mlir.dialects.allo import ResourceUseAttr

        spent = ResourceUseAttr.evaluate_all(_res_use_attr(uses), list(params))
        if spent is None:
            raise ValueError(
                f"{_unmeasured(uses, params)}, so it does not price "
                f"{tuple(params)}; measure it there, or price the realization "
                "at a parameter it covers"
            )
        return dict(spent)

    def comb_delay(self, kind: CombKind | str, width: int) -> float:
        """The chaining delay (ns) of a native operator kind at ``width`` bits,
        including the register floor the measurement saw.

        Evaluated by the compiler's own ``CostAttr::evaluate``, as :meth:`price`
        is, so a reader outside the compiler reads the same curve the scheduler
        does. Returns 0.0 where the device declares no row, and raises at a
        width the row was not measured at.
        """
        name = kind.value if isinstance(kind, CombKind) else kind
        cost = self.comb.get(name)
        if cost is None:
            return 0.0
        delay = _cost_attr(cost).evaluate(int(width))
        if delay is None:
            raise ValueError(
                f"the {self.name} {name!r} delay is a {_measured_over(cost)} "
                f"and was not measured at {width} bits"
            )
        return delay

    def add_resource(
        self, name: str, capacity: int, *, weight: float = 1.0
    ) -> Resource:
        """Declare a resource this device has ``capacity`` of, and return the
        handle a cost refers to. ``weight`` scales the price the compiler
        derives from scarcity; a schedule-time ``resource_weights`` map
        composes onto it multiplicatively."""
        if name in self.resources:
            raise ValueError(f"resource {name!r} already declared")
        if capacity <= 0:
            raise ValueError(f"resource {name!r} must have a positive capacity")
        if weight <= 0.0:
            raise ValueError(f"resource {name!r} must have a positive weight")
        r = Resource(name, int(capacity), float(weight))
        self.resources[name] = r
        return r

    # pylint: disable-next=too-many-arguments
    def add_storage(
        self,
        name: str,
        *,
        read_latency: int,
        write_latency: int,
        read_delay_ns: float = 0.0,
        write_delay_ns: float = 0.0,
        is_scatter: bool = False,
        is_table: bool = False,
        inst_reads: int | None = None,
        inst_writes: int | None = None,
        inst_ports: int | None = None,
        ram_style: str | None = None,
        can_init: bool = True,
        read_first: bool = False,
        uses: dict[Resource, Cost | Sequence] | None = None,
        read_delay_depth: Cost | None = None,
        read_delay_width: Cost | None = None,
    ) -> Storage:
        """Declare a storage realization and return the handle ``bind_storage``
        and :meth:`set_default_storage` refer to.

        Redeclaring a name REPLACES the row: retuning one primitive of a copied
        device is the normal way to build a variant, and the default, being a
        name, keeps pointing at whatever is declared under it.

        ``is_scatter`` marks the row that is not a memory at all: one cell per
        element, which is what a completely partitioned array becomes. A device
        marks at most one, and one that marks none cannot hold a complete
        partition.

        ``is_table`` marks the constant lookup built out of logic, which is what
        a read-only array declared with contents becomes. A device marks at most
        one, and one that marks none holds every such array in a memory that
        powers on with the contents. A table's read delay grows with the array,
        so the row declares ``read_delay_depth``; a table too deep to close at
        the target clock is held in a memory instead.

        ``inst_reads`` / ``inst_writes`` are the ports of each direction one
        instance has, omitted where there is no limit; a scatter row declares
        neither, having no addressed port to count. ``inst_ports`` is how many
        one instance has altogether, declared wherever the two directions draw
        on one pool as a block RAM's two ports do. None of the three bounds an
        array: how many instances hold one is the compiler's to decide, every
        copy taking every write.

        ``ram_style`` is the vendor attribute that pins an array to this
        structure, stamped on the emitted declaration. Omit it and the
        synthesizer chooses instead.

        ``can_init`` is whether the structure comes up holding contents. Clear
        it for one that powers up undefined, as an UltraRAM does; an array
        declared with compile-time contents is then refused there.

        ``read_first`` marks a structure whose read returns the OLD contents
        under a same-cycle write to the same element, in hardware and not
        merely in RTL simulation (a LUT RAM's asynchronous read). The
        scheduler relaxes write-after-read ordering to the read's sampling
        cycle only on a marked row; a block RAM's cross-port same-address
        collision is undefined in silicon, so leave it unmarked.

        Storage carries two parameters, ``(depth, width)``, so a ``uses`` term
        is a pair of costs, or the single :func:`Tiled` that reads them
        together.
        """
        if read_latency < 0 or write_latency < 0:
            raise ValueError(f"storage {name!r}: latency must be non-negative")
        if read_delay_ns < 0 or write_delay_ns < 0:
            raise ValueError(f"storage {name!r}: delay must be non-negative")
        limits = (
            ("inst_reads", inst_reads),
            ("inst_writes", inst_writes),
            ("inst_ports", inst_ports),
        )
        for role, limit in limits:
            if limit is not None and limit < 1:
                raise ValueError(f"storage {name!r}: {role} must be at least one port")
            if limit is not None and (is_scatter or is_table):
                raise ValueError(
                    f"storage {name!r} is not addressed, so it has no "
                    f"{role} to declare"
                )
        for role, limit in limits[:2]:
            if inst_ports is not None and limit is not None and limit > inst_ports:
                raise ValueError(
                    f"storage {name!r}: {role}={limit} exceeds inst_ports={inst_ports},"
                    " but an access of either direction takes one port of the pool"
                )
        if is_scatter and is_table:
            raise ValueError(
                f"storage {name!r} is one structure: `is_scatter` is a cell per "
                "element and `is_table` a constant lookup"
            )
        if is_table and not can_init:
            raise ValueError(
                f"storage {name!r} holds compile-time contents, so it cannot be "
                "one that powers up undefined"
            )
        for mark, what in ((is_scatter, "is_scatter"), (is_table, "is_table")):
            other = next(
                (
                    s
                    for s in self.storage.values()
                    if getattr(s, what) and s.name != name
                ),
                None,
            )
            if mark and other is not None:
                raise ValueError(
                    f"device {self.name!r} already marks {other.name!r} "
                    f"{what}; a device has at most one such storage"
                )
        s = Storage(
            name=name,
            read_latency=int(read_latency),
            write_latency=int(write_latency),
            read_delay_ns=float(read_delay_ns),
            write_delay_ns=float(write_delay_ns),
            is_scatter=bool(is_scatter),
            is_table=bool(is_table),
            inst_reads=None if inst_reads is None else int(inst_reads),
            inst_writes=None if inst_writes is None else int(inst_writes),
            inst_ports=None if inst_ports is None else int(inst_ports),
            ram_style=ram_style,
            can_init=bool(can_init),
            read_first=bool(read_first),
            uses=self._spend(f"storage {name!r}", "depth, width", uses),
            read_delay_depth=read_delay_depth,
            read_delay_width=read_delay_width,
        )
        self.storage[name] = s
        return s

    def set_storage_uses(
        self, name: str, uses: dict[Resource, Cost | Sequence]
    ) -> Device:
        """What one storage realization spends, over ``(depth, width)``. Apart
        from :meth:`add_storage` so that a device's timing and its area can be
        declared apart, the way a combinational kind's are."""
        s = self.storage.get(name)
        if s is None:
            raise ValueError(f"{name!r} is not a storage of device {self.name!r}")
        self.storage[name] = replace(
            s, uses=self._spend(f"storage {name!r}", "depth, width", uses)
        )
        return self

    def set_default_storage(self, storage: Storage) -> Device:
        """Hold every array with no ``bind_storage`` here, whatever it costs.

        Optional: left unset, the compiler picks the cheapest row on this part
        that it can pin the array to, among those at the least access latency.

        Takes a realization, so defaulting to a :class:`Resource` is a type
        error rather than a name that fails to resolve much later."""
        if not isinstance(storage, Storage):
            raise TypeError(
                f"the default must be a storage realization, got "
                f"{type(storage).__name__}"
            )
        if self.storage.get(storage.name) is not storage:
            raise ValueError(
                f"{storage.name!r} is not a storage of device {self.name!r}"
            )
        self.default_storage = storage.name
        return self

    def set_stream_timing(
        self,
        read_latency: int,
        write_latency: int,
        read_delay_ns: float = 0.0,
        write_delay_ns: float = 0.0,
    ) -> Device:
        """Get/put timing of a stream channel."""
        if read_latency < 0 or write_latency < 0:
            raise ValueError("stream latency must be non-negative")
        if read_delay_ns < 0 or write_delay_ns < 0:
            raise ValueError("stream delay must be non-negative")
        self.stream_timing = StreamTiming(
            read_latency=int(read_latency),
            write_latency=int(write_latency),
            read_delay_ns=float(read_delay_ns),
            write_delay_ns=float(write_delay_ns),
        )
        return self

    def set_comb_delay(
        self,
        kind: CombKind,
        delay_ns: Cost | float,
        uses: dict[Resource, Cost | Sequence] | None = None,
    ) -> Device:
        """Set the combinational chaining delay of a native operator kind, and
        optionally what one instance of it spends. A comb kind carries one
        parameter, its operand width, and both the delay and each cost are
        functions of it. A bare number is the constant function."""
        if not isinstance(kind, CombKind):
            raise TypeError(f"kind must be a CombKind, got {kind!r}")
        if not isinstance(delay_ns, Cost):
            if delay_ns < 0:
                raise ValueError(f"comb delay for {kind.value!r} must be non-negative")
            delay_ns = Const(float(delay_ns))
        self.comb[kind.value] = delay_ns
        if uses:
            self.comb_uses[kind.value] = self._spend(
                f"comb kind {kind.value!r}", "width", uses
            )
        return self

    def set_operator_uses(
        self, operator: OperatorIP, uses: dict[Resource, Cost | Sequence]
    ) -> Device:
        """What one instance of an operator IP spends. Its parameter is the
        operand width, as a native operator kind's is, even though the IP's
        signature already fixes that width: the arity follows the realization's
        kind so that one rule covers every row."""
        if operator not in self.operators:
            raise ValueError(
                f"{operator.symbol!r} is not an operator of device {self.name!r}"
            )
        self.operator_uses[operator.symbol] = self._spend(
            f"operator {operator.symbol!r}", "width", uses
        )
        return self

    def set_operator_variant(self, operator: OperatorIP, variant: str) -> Device:
        """Which of the realizer's alternative builds of this core the row was
        measured on. The realizer names the builds."""
        if operator not in self.operators:
            raise ValueError(
                f"{operator.symbol!r} is not an operator of device {self.name!r}"
            )
        self.operator_variant[operator.symbol] = variant
        return self

    def set_mux_uses(self, uses: dict[Resource, Sequence]) -> Device:
        """What one select over ``k`` sources of ``width`` bits spends."""
        self.mux_uses = self._spend("a multiplexer", "fan-in, width", uses)
        return self

    def set_mux_delay(self, delay_ns: Cost, width_factor: Cost | None = None) -> Device:
        """The routed marginal delay of that select in ns as a function of its
        fan-in, measured at a 32-bit reference width. ``width_factor`` is the
        unitless function of the actual width that scales it."""
        if not isinstance(delay_ns, Cost):
            raise TypeError("mux delay must be a Cost over the fan-in")
        if width_factor is not None and not isinstance(width_factor, Cost):
            raise TypeError("mux width factor must be a Cost over the width")
        self.mux_delay = delay_ns
        self.mux_delay_width = width_factor
        return self

    def set_chain_uses(self, uses: dict[Resource, Sequence]) -> Device:
        """What one ``depth``-stage, ``width``-bit delay chain spends when it
        carries a synchronous reset, which is what a control run holds."""
        self.chain_uses = self._spend("a delay chain", "depth, width", uses)
        return self

    def set_chain_uses_norst(self, uses: dict[Resource, Sequence]) -> Device:
        """What the same chain spends with no reset, the form every value and
        pulse run is emitted in and the row ``dcp.chain`` carries."""
        self.chain_uses_norst = self._spend(
            "a reset-free delay chain", "depth, width", uses
        )
        return self

    def set_register_floor(self, delay_ns: float) -> Device:
        """The register-to-register floor (ns): a source flip-flop's clock-to-out
        plus the routing every path pays, measured with nothing between the
        registers.

        Every combinational delay this device declares includes it. A cycle pays
        it once however many operators chain within it, so the scheduler charges
        a comb row its whole delay where a chain ends and the delay less this
        floor where a successor extends the chain. It defaults to zero.
        """
        if delay_ns < 0:
            raise ValueError("the register floor must be non-negative")
        self.reg_delay_ns = float(delay_ns)
        return self

    def set_default_frequency(self, freq_mhz: float) -> Device:
        if freq_mhz <= 0:
            raise ValueError("default frequency must be positive")
        self.default_freq_mhz = float(freq_mhz)
        return self

    def set_lut_packing(self, sites_per_lut: float) -> Device:
        """Routed logic sites per modeled LUT, measured as routed sites divided
        by LUT instances over real designs. Scales the quoted estimate only;
        realizations are compared in unpacked instance counts."""
        if not 0.0 < sites_per_lut <= 1.0:
            raise ValueError("lut packing is a fraction of instances kept")
        self.lut_packing = float(sites_per_lut)
        return self

    def add_operator(self, operator: OperatorIP) -> Device:
        """Declare a core this device offers."""
        if not isinstance(operator, OperatorIP):
            raise TypeError(f"expected an operator IP, got {type(operator).__name__}")
        symbol = operator.symbol
        if any(o.symbol == symbol for o in self.operators):
            raise ValueError(
                f"device {self.name!r} already declares an operator {symbol!r}; two "
                "`dcp.operator`s under one symbol is a symbol table error, and a "
                "core differing in kind, signature or latency is named apart on "
                "its own (see OperatorIP.symbol)"
            )
        self.operators.append(operator)
        return self

    def add_operators(self, *ips: OperatorIP) -> Device:
        for operator in ips:
            self.add_operator(operator)
        return self

    def remove_operator(self, operator: OperatorIP | str) -> Device:
        """Drop a core, named by handle or by symbol, and what it spends with
        it. Raises where the device does not declare it."""
        symbol = operator.symbol if isinstance(operator, OperatorIP) else operator
        found = next((o for o in self.operators if o.symbol == symbol), None)
        if found is None:
            raise ValueError(f"{symbol!r} is not an operator of device {self.name!r}")
        self.operators.remove(found)
        self.operator_uses.pop(symbol, None)
        self.operator_variant.pop(symbol, None)
        return self

    def validate(self) -> Device:
        """Check the device is complete enough to compile and to price against,
        and return it. Call it where a device is built, so a part missing a row
        fails there rather than inside a compile.
        """
        if not self.resources:
            raise ValueError(f"device {self.name!r} declares no resources")
        scatter = [s.name for s in self.storage.values() if s.is_scatter]
        if len(scatter) != 1:
            raise ValueError(
                f"device {self.name!r} marks {len(scatter)} scatter storages; it "
                "needs exactly one, since that is what a completely partitioned "
                "array and an array that failed RAM inference both become"
            )
        if self.stream_timing is None:
            raise ValueError(f"device {self.name!r} declares no stream timing")
        # The estimator prices every mux and delay chain through the one
        # whole-device row, so an absent row reads as free rather than as
        # unmodelled.
        for what, spent in (
            ("mux", self.mux_uses),
            ("chain", self.chain_uses),
            ("reset-free chain", self.chain_uses_norst),
        ):
            if not spent:
                raise ValueError(
                    f"device {self.name!r} declares no {what} cost; every {what} "
                    "in the design would then price at nothing"
                )
        table = next((s for s in self.storage.values() if s.is_table), None)
        if table is not None and table.read_delay_depth is None:
            raise ValueError(
                f"device {self.name!r} declares the constant table {table.name!r} "
                "without a `read_delay_depth`; its cone is what decides how deep "
                "a table may be, and one delay for every depth would let a table "
                "no clock can close be chosen"
            )
        return self

    def copy(self) -> Device:
        """An independent copy, so extending it does not mutate this device. The
        timing and IP objects are shared, never mutated."""
        d = Device(self.name, part=self.part, fabric=self.fabric, grade=self.grade)
        d.comb = dict(self.comb)
        d.resources = dict(self.resources)
        d.comb_uses = dict(self.comb_uses)
        d.operator_uses = dict(self.operator_uses)
        d.operator_variant = dict(self.operator_variant)
        d.mux_uses = self.mux_uses
        d.chain_uses = self.chain_uses
        d.chain_uses_norst = self.chain_uses_norst
        d.storage = dict(self.storage)
        d.lut_packing = self.lut_packing
        d.default_storage = self.default_storage
        d.stream_timing = self.stream_timing
        d.operators = list(self.operators)
        d.default_freq_mhz = self.default_freq_mhz
        d.reg_delay_ns = self.reg_delay_ns
        d.realizer = self.realizer
        return d


# --- operator behavioral descriptors (for cosim) ---------------------------


def _ty(dtype) -> Ty:
    """A behavioral-model :class:`Ty` from an allo scalar dtype."""
    return Ty(
        name=dtype.name,
        width=dtype.primitive_width,
        is_float=dtype.is_float(),
        signed=getattr(dtype, "signed", False),
    )


def operator_descs(operators: Sequence[OperatorIP]) -> list[OpDesc]:
    """The device operators as behavioral :class:`OpDesc` descriptors, the cosim
    source of truth for each extern IP's kind, latency and dtypes. ``name`` is
    the operator's symbol, the extern module the emitter instantiates and what
    the model joins on."""
    out = []
    for op in operators:
        kind = (
            op.optype.value if isinstance(op.optype, OperatorType) else str(op.optype)
        )
        rets = op.parse_return_annotation()
        out.append(
            OpDesc(
                name=op.symbol,
                kind=kind,
                latency=op.timing.latency,
                arg_types=tuple(_ty(a) for a in op.parse_argument_annotations()),
                ret_type=_ty(rets[0]),
                c_expr=op.c_model,
            )
        )
    return out


# --- injection into the scheduled module -----------------------------------


def inject_operators(module, device: Device):
    """Inject each device operator as a module-level ``dcp.operator`` symbol the
    scheduler and reifier match concrete ``arith.*``/``math.*`` ops onto. The
    ``sym_name`` is the operator's :attr:`~allo.lang.ip.OperatorIP.symbol` and
    the stem of the RTL module name the emitter instantiates. One declaration
    can cover several distinct pieces of hardware, so the emitter appends
    whatever else distinguishes them (a float compare's predicate:
    ``cmp_f32_f32_u1_l1`` -> ``cmp_f32_f32_u1_l1_ogt``).

    The resources an IP spends are the device's, but this op is not in the
    device's symbol table, so its references reach through the device symbol
    (``@u55c::@lut``) and resolve from where they are written."""
    if not device.operators:
        return
    from ..._mlir.ir import (
        InsertionPoint,
        Location,
        TypeAttr,
        FloatAttr,
        F32Type,
    )
    from ..._mlir.dialects.allo import DCPathOperatorOp, StallContractAttr
    from ...compiler.utils import generate_function_type

    with module.context as ctx, Location.unknown():
        f32ty = F32Type.get()
        insert = InsertionPoint.at_block_begin(module.body)
        for op in device.operators:
            kind = (
                op.optype.value
                if isinstance(op.optype, OperatorType)
                else str(op.optype)
            )
            sig = generate_function_type(
                ctx, op.parse_argument_annotations(), op.parse_return_annotation()
            )
            t = op.timing
            # A pipelined IP's style, else the clock-enable default.
            stall = StallContractAttr.get(t.style or "ce", ctx)
            DCPathOperatorOp(
                sym_name=op.symbol,
                kind=kind,
                signature=TypeAttr.get(sig),
                latency=t.latency,
                in_delay=FloatAttr.get(f32ty, t.in_delay_ns),
                out_delay=FloatAttr.get(f32ty, t.out_delay_ns),
                min_period=FloatAttr.get(f32ty, t.min_period_ns),
                pipelined=t.pipelined,
                stall=stall,
                uses=_uses_attr(device.operator_uses.get(op.symbol), device.name),
                fed_width=op.fed_width,
                ip=insert,
            )


def _uses_attr(spent, scope: str = ""):
    """``uses`` as a ``#allo.res_use`` array, or None when nothing is declared:
    an undeclared cost spends nothing, it is not a zero."""
    return _res_use_array(spent, scope) if spent else None


def inject_device(module, device: Device, weights: dict[str, float] | None = None):
    """Inject the device technology tables as a module-level ``dcp.device`` op:
    the per-kind combinational chaining delays and the storage model, which
    override the built-in library defaults. Target frequency is not injected: it
    is a per-run scheduling parameter, not technology data. ``weights`` are the
    schedule-time resource price multipliers, composed onto each resource's own
    declared weight."""
    from ..._mlir.ir import (
        InsertionPoint,
        Location,
        FloatAttr,
        IntegerAttr,
        F32Type,
        IntegerType,
    )
    from ..._mlir.dialects.allo import (
        OpKindAttr,
        DCPathChainOp,
        DCPathCombOp,
        DCPathDeviceOp,
        DCPathMuxOp,
        DCPathResourceOp,
        DCPathStorageOp,
        DCPathStreamTimingOp,
    )

    with module.context, Location.unknown():
        f32ty = F32Type.get()
        i64 = IntegerType.get_signless(64)

        def _port_limit(ty, n):
            return None if n is None else IntegerAttr.get(ty, n)

        def _timing(t) -> dict:
            return {
                "rd_latency": IntegerAttr.get(i64, t.read_latency),
                "rd_delay": FloatAttr.get(f32ty, t.read_delay_ns),
                "wr_latency": IntegerAttr.get(i64, t.write_latency),
                "wr_delay": FloatAttr.get(f32ty, t.write_delay_ns),
            }

        dev = DCPathDeviceOp(
            sym_name=device.name,
            reg_delay=FloatAttr.get(f32ty, device.reg_delay_ns),
            ip=InsertionPoint.at_block_begin(module.body),
        )
        # The body declares what the device HAS and what it can REALIZE, each a
        # symbol the others refer to. One op to inject, one to erase.
        body = dev.regions[0].blocks.append()
        with InsertionPoint(body):
            for name in weights or {}:
                if name not in device.resources:
                    raise ValueError(
                        f"resource_weights names {name!r}, which device "
                        f"{device.name!r} does not declare"
                    )
            for r in device.resources.values():
                w = r.weight * (weights or {}).get(r.name, 1.0)
                if w <= 0.0:
                    raise ValueError(f"resource {r.name!r} weight must be positive")
                DCPathResourceOp(
                    sym_name=r.name,
                    capacity=IntegerAttr.get(i64, r.capacity),
                    weight=None if w == 1.0 else FloatAttr.get(f32ty, w),
                )
            for kind, delay in device.comb.items():
                DCPathCombOp(
                    kind=OpKindAttr.get(kind),
                    delay=delay._attr(),
                    uses=_uses_attr(device.comb_uses.get(kind)),
                )
            for s in device.storage.values():
                DCPathStorageOp(
                    sym_name=s.name,
                    is_default=s.name == device.default_storage,
                    is_scatter=s.is_scatter,
                    is_table=s.is_table,
                    inst_reads=_port_limit(i64, s.inst_reads),
                    inst_writes=_port_limit(i64, s.inst_writes),
                    inst_ports=_port_limit(i64, s.inst_ports),
                    ram_style=s.ram_style,
                    no_init=not s.can_init,
                    read_first=s.read_first,
                    uses=_uses_attr(s.uses),
                    rd_delay_depth=(
                        s.read_delay_depth._attr() if s.read_delay_depth else None
                    ),
                    rd_delay_width=(
                        s.read_delay_width._attr() if s.read_delay_width else None
                    ),
                    **_timing(s),
                )
            if device.mux_uses:
                DCPathMuxOp(
                    uses=_uses_attr(device.mux_uses),
                    delay=device.mux_delay._attr() if device.mux_delay else None,
                    delay_width=(
                        device.mux_delay_width._attr()
                        if device.mux_delay_width
                        else None
                    ),
                )
            # Every chain a schedule pays for carries a value or an activation
            # pulse, neither of which holds a reset.
            if device.chain_uses_norst:
                DCPathChainOp(uses=_uses_attr(device.chain_uses_norst))
            if device.stream_timing is not None:
                DCPathStreamTimingOp(**_timing(device.stream_timing))


__all__ = [
    "Device",
    "Resource",
    "Storage",
    "StreamTiming",
    "CombKind",
    "Cost",
    "Const",
    "Linear",
    "Quadratic",
    "Step",
    "Table",
    "Interp",
    "Piecewise",
    "Tiled",
    "operator_descs",
    "inject_device",
    "inject_operators",
]
