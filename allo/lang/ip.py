# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from enum import Enum
from dataclasses import dataclass, replace
from typing import TypeVar, ParamSpec, overload, Literal
from collections.abc import Callable

from .kernel import Kernel

P = ParamSpec("P")
R = TypeVar("R")


class OperatorType(Enum):
    """The abstract operator kinds an IP can characterize."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    REM = "rem"
    MAX = "max"
    MAXNUM = "maxnum"
    MIN = "min"
    MINNUM = "minnum"
    CEILDIV = "ceildiv"
    FLOORDIV = "floordiv"
    NEG = "neg"
    CMP = "cmp"
    AND = "and"
    OR = "or"
    XOR = "xor"
    SHL = "shl"
    SHR = "shr"
    SELECT = "select"
    INT_CAST = "icast"  # sext / zext / trunc / index_cast
    INT_FLOAT_CAST = "ifcast"  # si/ui-to-fp, fp-to-si/ui
    FLOAT_CAST = "fcast"  # extf / truncf


@dataclass
class Timing:
    # number of cycles to complete the operation
    latency: int
    # time in nanoseconds to receive the input data
    in_delay_ns: float
    # time in nanoseconds to send the output data
    out_delay_ns: float
    # whether the operation can be pipelined
    pipelined: bool
    # pipelining style: free running, elastic, or clock enable
    style: Literal["free", "elastic", "ce"] | None = None
    # least clock period (ns) the core's internal stages hold; 0 claims
    # nothing and leaves the boundary cones as the only gate
    min_period_ns: float = 0.0


def verify_timing(timing: Timing):
    if timing.latency < 0:
        raise ValueError("Latency must be non-negative.")
    if timing.in_delay_ns < 0:
        raise ValueError("Input delay must be non-negative.")
    if timing.out_delay_ns < 0:
        raise ValueError("Output delay must be non-negative.")
    if timing.min_period_ns < 0:
        raise ValueError("Minimum period must be non-negative.")
    if timing.pipelined:
        if timing.style is None:
            raise ValueError("Pipelined operations must specify a style.")
        if timing.style not in ("free", "elastic", "ce"):
            raise ValueError("Pipeling style must be one of 'free', 'elastic', 'ce'")
    elif timing.style is not None:
        raise ValueError("Non-pipelined operations cannot specify a style.")


class IP(Kernel[P, R]):
    """An external hardware block: a signature, a timing contract, and an
    optional behavioral model for cosim."""

    def __init__(
        self,
        fn: Callable[P, R],
        latency: int = 1,
        in_delay_ns: float = 0.0,
        out_delay_ns: float = 0.0,
        pipelined: bool = False,
        style: Literal["free", "elastic", "ce"] | None = None,
        min_period_ns: float = 0.0,
    ):
        super().__init__(fn, mapping=())
        if self.is_async:
            raise TypeError("External IPs cannot be asynchronous.")
        self.timing = Timing(
            latency=latency,
            in_delay_ns=in_delay_ns,
            out_delay_ns=out_delay_ns,
            pipelined=pipelined,
            style=style,
            min_period_ns=min_period_ns,
        )
        # An optional user behavioral model for cosim: a C expression over the
        # operands `a`, `b`, ... computing the result (see `add_c_model`). None
        # falls back to the built-in expression for the operator's kind.
        self.c_model: str | None = None
        verify_timing(self.timing)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.fn(*args, **kwargs)

    def compile(self):
        raise RuntimeError(
            "External IPs cannot be compiled standalone. They must be used within a kernel."
        )

    def schedule(self):
        raise NotImplementedError(
            "External IPs cannot be scheduled standalone. They must be used within a kernel."
        )

    @property
    def module(self):
        raise NotImplementedError(
            "External IPs do not have a module. They must be used within a kernel."
        )

    def add_c_model(self, expr: str) -> "IP[P, R]":
        """Attach a cosim behavioral model: a C expression over the operands
        ``a``, ``b``, ... (positional) computing the result -- e.g.
        ``add_c_model("a + b")`` or ``add_c_model("std::erf(a)")``. It overrides
        the built-in expression the operator's ``optype`` would otherwise use, so
        it is how a user characterizes an operator kind the backend has no
        built-in model for. Returns ``self`` for chaining."""
        if not isinstance(expr, str):
            raise TypeError(f"add_c_model expects a C expression string, got {expr!r}")
        self.c_model = expr
        return self

    def add_rtl_model(self, *arg, **kwargs):
        raise NotImplementedError("add_rtl_model is not implemented yet")


#: How a dtype family abbreviates in an operator symbol. A family the table does
#: not name keeps its full dtype name.
_FAMILY_TAG = {"float": "f", "bfloat": "bf", "int": "i", "uint": "u"}


def _dtype_tag(dtype) -> str:
    name = dtype.name
    family = name.rstrip("0123456789")
    width = name[len(family) :]
    tag = _FAMILY_TAG.get(family)
    return f"{tag}{width}" if tag and width else name


# An operator core has no schedule, module, or RTL model, so the base's
# raising implementations stand.
# pylint: disable-next=abstract-method
class OperatorIP(IP[P, R]):
    """An IP the compiler matches onto concrete ``arith``/``math`` ops instead
    of one a kernel instantiates by name.

    ``optype`` is the abstract kind it realizes; every op of that kind whose
    signature it fits binds to it. A device may declare several such cores per
    (kind, signature), which are candidates the operator library chooses
    between.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        fn: Callable[P, R],
        optype: OperatorType | str,
        mnemonic: str | None = None,
        latency: int = 1,
        in_delay_ns: float = 0.0,
        out_delay_ns: float = 0.0,
        pipelined: bool = False,
        style: Literal["free", "elastic", "ce"] | None = None,
        fed_width: int | None = None,
        min_period_ns: float = 0.0,
    ):
        super().__init__(
            fn,
            latency=latency,
            in_delay_ns=in_delay_ns,
            out_delay_ns=out_delay_ns,
            pipelined=pipelined,
            style=style,
            min_period_ns=min_period_ns,
        )
        self.optype = optype
        # Restricts the row to ops whose operands carry no more significant
        # bits, the core having been measured with extension-fed inputs.
        self.fed_width = fed_width
        # The readable base of the symbol, not what makes it unique. Defaults
        # to the kind's own string.
        self.mnemonic = mnemonic or (
            optype.value if isinstance(optype, OperatorType) else str(optype)
        )

    @property
    def symbol(self) -> str:
        """The ``dcp.operator`` symbol this core injects under, which is also
        the extern RTL module name the emitter instantiates and the key the
        cosim behavioral model joins on.

        Every axis a core specializes along is a field, so that distinct
        hardware never shares a symbol::

            <mnemonic>_<arg tag>..._<result tag>_l<latency>

        * ``mnemonic`` is the kind, defaulting to ``optype``'s own string
          (``add``, ``cmp``, ``ifcast``, or an advanced op's ``sqrt``).
        * one tag per argument and then one for the result, so the field count
          is the arity and a unary core cannot collide with a binary one. Tags
          are dtypes rather than widths, keeping ``bf16`` apart from a future
          ``f16``. The result tag is carried even where it repeats an argument,
          since a cast differs only there (``ifcast_i32_f32_l3`` against
          ``ifcast_f32_i32_l3``).
        * ``_l<latency>`` last, since a core pipelined differently is different
          hardware with a different area. No dtype tag begins with ``l``, so the
          suffix cannot be read as one.

        A float compare is not fully determined here: the predicate belongs to
        the op and not to the IP, so the emitter appends it to the module name
        (``cmp_f32_f32_u1_l1_ogt``).
        """
        rets = self.parse_return_annotation()
        assert len(rets) == 1, f"operator IP {self.func_name!r} returns one scalar"
        tags = [_dtype_tag(t) for t in (*self.parse_argument_annotations(), rets[0])]
        return f"{self.mnemonic}_{'_'.join(tags)}_l{self.timing.latency}"

    def retimed(
        self,
        latency: int,
        in_delay_ns: float,
        min_period_ns: float,
        out_delay_ns: float,
    ) -> "OperatorIP[P, R]":
        """A copy of this core pipelined to ``latency``, carrying the timing
        measured at that depth. The symbol follows the new latency, so the same
        core at two depths gets two names. All three delays are required, each
        belonging to the depth it was measured at."""
        core = copy.copy(self)
        core.timing = replace(
            self.timing,
            latency=latency,
            in_delay_ns=in_delay_ns,
            min_period_ns=min_period_ns,
            out_delay_ns=out_delay_ns,
        )
        verify_timing(core.timing)
        return core


@overload
def operator_ip(
    fn: Callable[P, R],
    *,
    optype: OperatorType | str,
    mnemonic: str | None = None,
    latency: int = 1,
    in_delay_ns: float = 0.0,
    out_delay_ns: float = 0.0,
    pipelined: bool = False,
    style: Literal["free", "elastic", "ce"] | None = None,
    fed_width: int | None = None,
    min_period_ns: float = 0.0,
) -> OperatorIP[P, R]: ...


@overload
def operator_ip(
    *,
    optype: OperatorType | str,
    mnemonic: str | None = None,
    latency: int = 1,
    in_delay_ns: float = 0.0,
    out_delay_ns: float = 0.0,
    pipelined: bool = False,
    style: Literal["free", "elastic", "ce"] | None = None,
    fed_width: int | None = None,
    min_period_ns: float = 0.0,
) -> Callable[[Callable[P, R]], OperatorIP[P, R]]: ...


def operator_ip(fn=None, **kw):
    """Declare an operator core: an external block the compiler binds ops of
    ``optype`` onto. The body is ``...``; the parameters declare the signature,
    which with ``optype`` and ``latency`` selects and names the core (see
    :attr:`OperatorIP.symbol`)."""

    def build(f):
        assert callable(f), "The first argument must be a callable function."
        return OperatorIP(f, **kw)

    return build(fn) if fn is not None else build
