# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral models for the extern IP operators the emitter instantiates"""

from __future__ import annotations

from dataclasses import dataclass

from ..interface import Interfaces, Operator
from ....lang.ip import OperatorType

# --- descriptors (from the device operator table) --------------------------


@dataclass(frozen=True)
class Ty:
    """One operand/result dtype, as the behavioral model needs to see it."""

    name: str  # allo dtype name: float32 / float64 / bfloat16 / int32 / uint32 ...
    width: int  # bit width
    is_float: bool
    signed: bool  # meaningful for integers only


@dataclass(frozen=True)
class OpDesc:
    """One device operator IP. ``name`` is the operator's ``sym_name`` and the
    extern module's base name. ``c_expr`` is a user ``add_c_model`` C expression
    over the operands ``a``, ``b``, ...; ``None`` selects the built-in model."""

    name: str
    kind: str  # abstract kind: add/sub/mul/div/rem/cmp/ifcast/fcast/<math mnemonic>
    latency: int
    arg_types: tuple[Ty, ...]
    ret_type: Ty
    c_expr: str | None = None


def _operand(k: int) -> str:
    """Operand ``k``'s data port name, which the built-in models are written over."""
    return chr(ord("a") + k)


# --- the externs, as the port manifest declares them ------------------------


@dataclass(frozen=True)
class _Extern:
    """An instantiated extern operator module: the descriptor plus the realized
    port shape from the manifest. Each port carries its role, so the clock, the
    optional clock enable and the result are found structurally, not by name.
    Every width is read off the descriptor; the manifest supplies only the
    names."""

    name: str  # the extern module's RTL name
    ports: tuple[tuple[str, int, Operator.Role], ...]  # (name, width, role) in order
    pred: str  # compare predicate; "" if none
    desc: OpDesc

    def __post_init__(self):
        d = self.desc
        ins = self._of_role(Operator.Role.DATA)
        assert len(ins) == len(d.arg_types), (
            f"operator '{d.name}': extern has {len(ins)} data ports but the "
            f"descriptor declares {len(d.arg_types)} operands"
        )
        for k, ((pn, pw), ty) in enumerate(zip(ins, d.arg_types)):
            assert pw == ty.width, (
                f"operator '{d.name}': extern port '{pn}' is {pw} bits but the "
                f"descriptor declares operand {ty.name} of {ty.width} bits"
            )
            assert pn == _operand(k), (
                f"operator '{d.name}': data port {k} is named '{pn}'; the "
                "behavioral models are written over a, b, c in operand order"
            )
        assert len(self._of_role(Operator.Role.CLK)) == 1
        assert len(self._of_role(Operator.Role.CE)) <= 1
        outs = self._of_role(Operator.Role.OUT)
        assert len(outs) == 1, f"operator '{d.name}': extern has {len(outs)} outputs"
        assert outs[0][1] == d.ret_type.width, (
            f"operator '{d.name}': extern output '{outs[0][0]}' is {outs[0][1]} "
            f"bits but the descriptor declares {d.ret_type.name} of "
            f"{d.ret_type.width} bits"
        )

    def _of_role(self, role: Operator.Role) -> list[tuple[str, int]]:
        return [(n, w) for n, w, r in self.ports if r == role]

    @property
    def has_ce(self) -> bool:
        return bool(self._of_role(Operator.Role.CE))

    @property
    def clk(self) -> str:
        return self._of_role(Operator.Role.CLK)[0][0]

    @property
    def out(self) -> str:
        return self._of_role(Operator.Role.OUT)[0][0]


def _plan(interfaces: Interfaces, descs) -> list[_Extern]:
    """The extern operators instantiated across every emitted module, each joined
    to its device descriptor by the ``impl`` the manifest names. One entry per
    module, since several kernels may share one behavioral model."""
    by_name = {d.name: d for d in descs}
    seen: dict[str, _Extern] = {}
    for iface in interfaces.values():
        for op in iface.operators:
            if op.module in seen:
                continue
            desc = by_name.get(op.impl)
            assert desc, f"extern operator '{op.impl}' has no device operator"
            ports = tuple((p.name, p.width, p.role) for p in op.ports)
            seen[op.module] = _Extern(op.module, ports, op.predicate, desc)
    return list(seen.values())


# --- built-in behavior, and which language it is written in -----------------


@dataclass(frozen=True)
class _Model:
    """How one operator kind computes, one realization per domain. ``sv`` is a
    SystemVerilog expression over the operand ports, exact at any width; ``c`` is
    a C expression the DPI evaluates; a domain the kind has no meaning in stays
    ``None``. ``sv`` covers both signednesses (the ports carry the dtype's own),
    ``svu`` overriding it where the unsigned core differs. ``{w}`` expands to the
    result width, ``{ret}`` to the result's C scalar, ``{cmp}`` to the compare
    predicate's expression."""

    sv: str | None = None
    c: str | None = None
    svu: str | None = None


_MODELS: dict[str, _Model] = {
    "add": _Model("a + b", "a + b"),
    "sub": _Model("a - b", "a - b"),
    "mul": _Model("a * b", "a * b"),
    # Fused multiply-add, exact at the shared width: the expression
    # self-truncates to the result like the plain multiply.
    "muladd": _Model("a * b + c", "a * b + c"),
    "div": _Model("a / b", "a / b"),
    "rem": _Model("a % b", "std::fmod(a, b)"),
    # `max`/`min` propagate a NaN operand; `maxnum`/`minnum` return the other
    # one, which is what fmax/fmin already do.
    "max": _Model(
        "a > b ? a : b", "std::isnan(a) || std::isnan(b) ? a + b : std::fmax(a, b)"
    ),
    "min": _Model(
        "a < b ? a : b", "std::isnan(a) || std::isnan(b) ? a + b : std::fmin(a, b)"
    ),
    "maxnum": _Model(c="std::fmax(a, b)"),
    "minnum": _Model(c="std::fmin(a, b)"),
    # The signed correction turns on whether the operand signs agree; unsigned
    # operands agree by construction. `$signed` keeps the size cast from making
    # the whole expression, division included, unsigned.
    "ceildiv": _Model(
        sv="a / b + $signed({w}'(a % b != 0 && (a < 0) == (b < 0)))",
        svu="a / b + {w}'(a % b != 0)",
    ),
    "floordiv": _Model(
        sv="a / b - $signed({w}'(a % b != 0 && (a < 0) != (b < 0)))",
        svu="a / b",
    ),
    "neg": _Model("-a", "-a"),
    "cmp": _Model("{cmp}", "{cmp}"),
    "and": _Model(sv="a & b"),
    "or": _Model(sv="a | b"),
    "xor": _Model(sv="a ^ b"),
    "shl": _Model(sv="a << b"),
    # `>>>` fills with the sign bit only when its left operand is signed, so one
    # operator covers both the arithmetic and the logical shift.
    "shr": _Model(sv="a >>> b"),
    "select": _Model("a ? b : c", "a ? b : c"),
    # A size cast extends by the operand port's own signedness or truncates,
    # which is exactly sext / zext / trunc.
    "icast": _Model(sv="{w}'(a)"),
    "ifcast": _Model(c="({ret})a"),
    "fcast": _Model(c="({ret})a"),
    # advanced unary math, keyed by mnemonic rather than by OperatorType
    "sqrt": _Model(c="std::sqrt(a)"),
    "exp": _Model(c="std::exp(a)"),
    "log": _Model(c="std::log(a)"),
    "sin": _Model(c="std::sin(a)"),
    "cos": _Model(c="std::cos(a)"),
    "tan": _Model(c="std::tan(a)"),
    "tanh": _Model(c="std::tanh(a)"),
    # Integer divide and remainder bind by MLIR mnemonic rather than by kind,
    # since `i32` is signless and the abstract `div` cannot say which arithmetic
    # is meant. One expression serves both: the operand ports are declared
    # `signed` from the core's dtypes, and SystemVerilog `/` and `%` follow their
    # operands, truncating toward zero and taking the dividend's sign exactly as
    # `arith.divsi` / `arith.remsi` do.
    "divsi": _Model(sv="a / b"),
    "divui": _Model(sv="a / b"),
    "remsi": _Model(sv="a % b"),
    "remui": _Model(sv="a % b"),
    "abs": _Model("a < 0 ? -a : a", "std::fabs(a)", svu="a"),
    "absf": _Model(c="std::fabs(a)"),
    "floor": _Model(c="std::floor(a)"),
    "ceil": _Model(c="std::ceil(a)"),
}

assert {t.value for t in OperatorType} <= _MODELS.keys(), (
    "every OperatorType needs a behavioral model; missing: "
    f"{sorted({t.value for t in OperatorType} - _MODELS.keys())}"
)

# Compare predicate -> the expression it stands for. An ordered (o*) and an
# unordered (u*) float relation map to the same operator, since cosim inputs are
# NaN-free; the NaN tests and the two constants are not `a <op> b`.
_FCMP = {
    "oeq": "a == b",
    "one": "a != b",
    "ogt": "a > b",
    "oge": "a >= b",
    "olt": "a < b",
    "ole": "a <= b",
    "ueq": "a == b",
    "une": "a != b",
    "ugt": "a > b",
    "uge": "a >= b",
    "ult": "a < b",
    "ule": "a <= b",
    "uno": "std::isnan(a) || std::isnan(b)",
    "ord": "!std::isnan(a) && !std::isnan(b)",
    "true": "true",
    "false": "false",
}
_ICMP = {
    "eq": "a == b",
    "ne": "a != b",
    "slt": "a < b",
    "sle": "a <= b",
    "sgt": "a > b",
    "sge": "a >= b",
    "ult": "a < b",
    "ule": "a <= b",
    "ugt": "a > b",
    "uge": "a >= b",
}


def _icmp(pred: str, signed: bool) -> str:
    """The expression for integer predicate ``pred`` over operands read as
    ``signed``. The predicate names the signedness it wants, so a core where it
    disagrees with the operand dtype is caught here."""
    expr = _ICMP.get(pred)
    assert expr is not None, f"unsupported integer-compare predicate '{pred}'"
    if pred not in ("eq", "ne"):
        assert (pred[0] == "s") == signed, (
            f"integer-compare predicate '{pred}' does not match the operand "
            f"dtype, which is {'signed' if signed else 'unsigned'}"
        )
    return expr


def _no_model(desc: OpDesc, domain: str) -> NotImplementedError:
    return NotImplementedError(
        f"operator '{desc.name}': no {domain} behavioral model for kind "
        f"'{desc.kind}'; attach one with @ip.add_c_model(\"<C expression>\")"
    )


def _float_op(desc: OpDesc) -> bool:
    """Whether a float appears anywhere in the signature, the result included."""
    return any(t.is_float for t in (*desc.arg_types, desc.ret_type))


def _via_dpi(desc: OpDesc) -> bool:
    """Whether this core computes in C rather than in RTL. A user model is a C
    expression and always does; a float operator does; an integer one does not,
    since SystemVerilog carries integer width and signedness exactly and C does
    not."""
    return desc.c_expr is not None or _float_op(desc)


# --- native models: a SystemVerilog expression over the operand ports -------


def _sv_expr(desc: OpDesc, pred: str, signed: bool) -> str:
    """The built-in SystemVerilog expression for an integer ``desc``."""
    model = _MODELS.get(desc.kind, _Model())
    expr = model.sv if signed or model.svu is None else model.svu
    if expr is None:
        raise _no_model(desc, "integer")
    return expr.format(w=desc.ret_type.width, cmp=_icmp(pred, signed) if pred else "")


# --- DPI models: a C expression over operands decoded from the port bits ----

#: The float formats allo has (see ``APFloat``), each naming its runtime codec.
#: A format absent here has no bit layout the model knows and is a hard error.
_FLOAT_FMT = {
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float64": "f64",
}


def _cscalar(ty: Ty) -> str:
    """The C scalar a value of ``ty`` is held in. Every integer widens to 64
    bits, so a width the C types cannot name (a 48-bit accumulator) is still
    exact; only the load and the store know the real width."""
    if ty.is_float:
        return "double" if ty.name == "float64" else "float"  # f16/bf16 in float
    return "int64_t" if ty.signed else "uint64_t"


def _operand_ctype(desc: OpDesc) -> str:
    """The one C type every operand is evaluated in. It is not the result's: a
    core whose result is wider than its operands (an i32 x i32 -> i64 multiplier)
    has to compute in a type wide enough to hold it, which C's own promotions do
    not give."""
    tys = (*desc.arg_types, desc.ret_type)
    if _float_op(desc):
        return "double" if any(t.name == "float64" for t in tys) else "float"
    return "int64_t" if any(t.signed for t in desc.arg_types) else "uint64_t"


# `allo_ld_*` / `allo_st_*` are the codecs in `_DPI_RUNTIME` at the end of this
# file, the one place a dtype's bit layout is written down.
def _load(ty: Ty, ctype: str, raw: str, name: str) -> str:
    """A C statement binding operand ``name``, at the evaluation type ``ctype``,
    to the value the port bits ``raw`` encode."""
    if ty.is_float:
        return f"{ctype} {name} = allo_ld_{_FLOAT_FMT[ty.name]}({raw});"
    fn = "allo_ld_int" if ty.signed else "allo_ld_uint"
    return f"{ctype} {name} = {fn}({raw}, {ty.width});"


def _store(ty: Ty, dst: str, val: str) -> str:
    """A C statement writing typed value ``val`` into the result port ``dst``."""
    if ty.is_float:
        return f"allo_st_{_FLOAT_FMT[ty.name]}({dst}, {val});"
    return f"allo_st_int({dst}, {ty.width}, (uint64_t){val});"


def _c_expr(desc: OpDesc, pred: str) -> str:
    """The built-in C expression for a float ``desc``, over the bound operands."""
    expr = _MODELS.get(desc.kind, _Model()).c
    if expr is None:
        raise _no_model(desc, "float")
    cmp = _FCMP.get(pred, "")
    if desc.kind == "cmp":
        assert cmp, f"unsupported float-compare predicate '{pred}'"
    return expr.format(ret=_cscalar(desc.ret_type), cmp=cmp)


def _dpi_name(e: _Extern) -> str:
    """A DPI function name unique per behavior (the operator + its predicate)."""
    return f"allo_op_{e.desc.name}" + (f"_{e.pred}" if e.pred else "")


def _dpi_body(e: _Extern) -> str:
    """One operator's DPI function body: bind each operand from its port bits,
    evaluate, write the result back."""
    d = e.desc
    for ty in (*d.arg_types, d.ret_type):
        if ty.is_float:
            assert ty.name in _FLOAT_FMT, f"unknown float format '{ty.name}'"
        elif ty.width > 64:
            raise NotImplementedError(
                f"operator '{d.name}': a {ty.width}-bit integer does not fit the "
                "64-bit C value a user model computes in; drop the add_c_model "
                "to take the native SystemVerilog model, which has no such limit"
            )
    ctype = _operand_ctype(d)
    assert not (ctype == "double" and d.ret_type.name in {"float16", "bfloat16"}), (
        f"operator '{d.name}': a {d.ret_type.name} result computed in double "
        "would round twice; model it from a float operand instead"
    )
    binds = "".join(
        f"  {_load(t, ctype, f'p{k}', _operand(k))}\n"
        for k, t in enumerate(d.arg_types)
    )
    # A user expression is pasted as written; only a built-in model is a template.
    expr = d.c_expr if d.c_expr is not None else _c_expr(d, e.pred)
    return (
        f"{binds}  {_cscalar(d.ret_type)} _r = ({expr});\n"
        f"  {_store(d.ret_type, 'r', '_r')}"
    )


_DPI_C = """\
// Auto-generated DPI-C behavioral models for the extern IP operators.

#include <cstdint>
#include <cstring>
#include <cmath>

#include "svdpi.h"

{runtime}
{functions}
"""

_DPI_OP = """\
extern "C" void {name}({params}) {{
{body}
}}
"""


def dpi_c(interfaces: Interfaces, descs) -> str:
    """C implementations of the DPI operators the instantiated externs need, or
    the empty string when every one of them models natively."""
    fns: dict[str, str] = {}
    for e in _plan(interfaces, descs):
        name = _dpi_name(e)
        if not _via_dpi(e.desc) or name in fns:
            continue
        params = ", ".join(
            [f"const svBitVecVal *p{k}" for k in range(len(e.desc.arg_types))]
            + ["svBitVecVal *r"]
        )
        fns[name] = _DPI_OP.format(name=name, params=params, body=_dpi_body(e))
    if not fns:
        return ""
    return _DPI_C.format(runtime=_DPI_RUNTIME, functions="\n".join(fns.values()))


# --- the emitted SystemVerilog ----------------------------------------------

# A shift register `latency` deep in front of the operator's value. `decl` and
# `sample` are all the two realization paths differ by: a native model is a
# combinational wire, while a DPI call has to land in `t` with a blocking
# assignment first, since a nonblocking read of the output that same call just
# wrote would skip a stage.
_SV_OP = """\
module {name}({ports});
{decl}  reg [{msb}:0] p [0:{last}];
  integer i;
  always @(posedge {clk}) {guard}begin
{sample}    for (i = 1; i < {latency}; i = i + 1) p[i] <= p[i - 1];
  end
  assign {out} = p[{last}];
endmodule
"""

_SV_MODELS = """\
// Auto-generated behavioral models for the extern IP operators (ip_models)

{imports}

{modules}
"""


def _dpi_import(e: _Extern) -> str:
    """The import declaration for ``e``'s DPI function. Its widths come from the
    descriptor, the same side the C is generated from."""
    args = ", ".join(
        f"input bit [{t.width - 1}:0] p{k}" for k, t in enumerate(e.desc.arg_types)
    )
    return (
        f'import "DPI-C" function void {_dpi_name(e)}({args}, '
        f"output bit [{e.desc.ret_type.width - 1}:0] r);"
    )


def _sv_module(e: _Extern) -> str:
    """One extern's behavioral module."""
    d = e.desc
    assert d.latency >= 1, f"operator '{d.name}' needs latency >= 1 for a shift model"
    msb = d.ret_type.width - 1
    native = not _via_dpi(d)
    # A native port carries the dtype's signedness, so the expression does not
    # restate it.
    ports = [
        f"input{' signed' if native and t.signed else ''} "
        f"[{t.width - 1}:0] {_operand(k)}"
        for k, t in enumerate(d.arg_types)
    ]
    ports.append(f"input {e.clk}")
    if e.has_ce:
        ports.append("input ce")
    ports.append(f"output [{msb}:0] {e.out}")

    if native:
        expr = _sv_expr(d, e.pred, any(t.signed for t in d.arg_types))
        decl, sample = f"  wire [{msb}:0] f = {expr};\n", "    p[0] <= f;\n"
    else:
        args = ", ".join([_operand(k) for k in range(len(d.arg_types))] + ["t"])
        decl = f"  reg [{msb}:0] t;\n"
        sample = f"    {_dpi_name(e)}({args});\n    p[0] <= t;\n"

    return _SV_OP.format(
        name=e.name,
        ports=", ".join(ports),
        decl=decl,
        msb=msb,
        last=d.latency - 1,
        clk=e.clk,
        guard="if (ce) " if e.has_ce else "",
        sample=sample,
        latency=d.latency,
        out=e.out,
    )


def sv_models(interfaces: Interfaces, descs) -> str:
    """SystemVerilog behavioral models + DPI import decls for the instantiated
    extern IP operators."""
    plan = _plan(interfaces, descs)
    if not plan:
        return ""
    imports = {_dpi_name(e): _dpi_import(e) for e in plan if _via_dpi(e.desc)}
    return _SV_MODELS.format(
        imports="\n".join(imports.values()),
        modules="\n".join(_sv_module(e) for e in plan),
    )


# --- the C runtime the generated DPI functions call into --------------------

# Every load and store goes through these codecs, so a port of any width is
# decoded the same way and each dtype's bit layout is stated exactly once. The
# narrowing float conversions round to nearest even, as the hardware does.
_DPI_RUNTIME = """\
// A packed SystemVerilog `bit [W-1:0]` reaches C as `svBitVecVal[]`: 32-bit
// words, word k holding bits [32k+31 : 32k].

template <typename U>
static inline U allo_ld_raw(const svBitVecVal *v, unsigned w) {
  U x = 0;
  for (unsigned i = 0; i * 32 < w; ++i)
    x |= (U)(uint32_t)v[i] << (32 * i);
  const unsigned n = sizeof(U) * 8;
  return w < n ? (U)(x & ((((U)1) << w) - 1)) : x;
}

static inline void allo_st_raw(svBitVecVal *v, unsigned w, uint64_t x) {
  for (unsigned i = 0; i * 32 < w; ++i)
    v[i] = (svBitVecVal)(uint32_t)(x >> (32 * i));
  const unsigned tail = w & 31;  // the top word carries only the port's bits
  if (tail)
    v[(w - 1) / 32] &= (svBitVecVal)((1u << tail) - 1);
}

// An integer operand widens to 64 bits by its OWN signedness, so the expression
// below it sees the value the port carries whatever the port's width.
static inline int64_t allo_ld_int(const svBitVecVal *v, unsigned w) {
  const uint64_t x = allo_ld_raw<uint64_t>(v, w);
  const unsigned sh = 64 - w;
  return sh ? ((int64_t)(x << sh)) >> sh : (int64_t)x;
}
static inline uint64_t allo_ld_uint(const svBitVecVal *v, unsigned w) {
  return allo_ld_raw<uint64_t>(v, w);
}
static inline void allo_st_int(svBitVecVal *v, unsigned w, uint64_t x) {
  allo_st_raw(v, w, x);
}

static inline float allo_ld_f32(const svBitVecVal *v) {
  const uint32_t u = (uint32_t)allo_ld_raw<uint64_t>(v, 32);
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static inline void allo_st_f32(svBitVecVal *v, float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  allo_st_raw(v, 32, u);
}
static inline double allo_ld_f64(const svBitVecVal *v) {
  const uint64_t u = allo_ld_raw<uint64_t>(v, 64);
  double d;
  memcpy(&d, &u, 8);
  return d;
}
static inline void allo_st_f64(svBitVecVal *v, double d) {
  uint64_t u;
  memcpy(&u, &d, 8);
  allo_st_raw(v, 64, u);
}

// binary16: widen exactly, narrow round-to-nearest-even.
static inline float allo_ld_f16(const svBitVecVal *v) {
  const uint16_t h = (uint16_t)allo_ld_raw<uint64_t>(v, 16);
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, u;
  if (exp == 0) {
    if (man == 0) {
      u = sign;
    } else {  // subnormal half, a normal float once renormalized
      exp = 127 - 15 + 1;
      while (!(man & 0x400u)) {
        man <<= 1;
        --exp;
      }
      u = sign | (exp << 23) | ((man & 0x3FFu) << 13);
    }
  } else if (exp == 0x1Fu) {
    u = sign | 0x7F800000u | (man << 13);
  } else {
    u = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static inline void allo_st_f16(svBitVecVal *v, float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  const uint32_t sign = (u >> 16) & 0x8000u, be = (u >> 23) & 0xFFu;
  uint32_t man = u & 0x7FFFFFu, h;
  int32_t exp = (int32_t)be - 127 + 15;
  if (be == 0xFFu) {  // inf, or a NaN kept quiet
    h = sign | 0x7C00u | (man ? 0x200u : 0u);
  } else if (exp >= 0x1F) {
    h = sign | 0x7C00u;
  } else if (exp < -10) {
    h = sign;
  } else if (exp <= 0) {  // subnormal half: shift the whole significand down
    man |= 0x800000u;
    const unsigned sh = (unsigned)(14 - exp);
    h = sign | ((man + (1u << (sh - 1)) - 1u + ((man >> sh) & 1u)) >> sh);
  } else {
    const uint32_t r = (man + 0x0FFFu + ((man >> 13) & 1u)) >> 13;
    exp += (int32_t)(r >> 10);  // rounding may carry into the exponent
    h = exp >= 0x1F ? (sign | 0x7C00u)
                    : (sign | ((uint32_t)exp << 10) | (r & 0x3FFu));
  }
  allo_st_raw(v, 16, h);
}

// bfloat16: the top 16 bits of a binary32, round-to-nearest-even downward.
static inline float allo_ld_bf16(const svBitVecVal *v) {
  const uint32_t u = (uint32_t)allo_ld_raw<uint64_t>(v, 16) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static inline void allo_st_bf16(svBitVecVal *v, float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  const bool nan = (u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu);
  allo_st_raw(v, 16,
              nan ? ((u >> 16) | 0x40u)
                  : ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16));
}
"""
