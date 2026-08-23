# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vivado IP core recipes, one per operator archetype, shared by every Xilinx fabric."""

from __future__ import annotations

import re
from typing import NamedTuple

from ....lang.ip import OperatorIP
from ..device import Device, Realization
from ..interface import Interfaces, Operator
from . import ip


class VivadoCore(NamedTuple):
    """The Vivado build of one operator archetype.

    ``shape`` is the config fragment the archetype's signature fixes. ``no_dsp``
    is the DSP-free fragment, non-empty exactly where the default build spends
    DSPs; a row whose area carries no ``dsp`` count is built with it. A full
    configuration applies ``CE_BASE[core]``, ``shape``, ``no_dsp`` if asked, then
    ``LATENCY[core]``, in that order. ``operation`` is the constant driven on the
    core's operation channel where the shape leaves one; a compare's comes from
    the predicate instead.
    """

    # `create_ip -name`: floating_point / mult_gen / div_gen; or `rtl`, which
    # generates no core: the shim computes `shape` (an SV expression over the
    # data ports) behind the row's latency and synthesis infers the DSPs.
    core: str
    shape: str  # `Key=Value` pairs, comma separated
    no_dsp: str = ""  # DSP-free fragment, empty where the core has no such knob
    operation: str = ""  # operation-channel constant, as binary tdata bits


#: The clock-enable half of the emitter's `ce` stall contract, per core family:
#: no back-pressure, a clock enable, no result ready. Applied first.
CE_BASE = {
    "floating_point": (
        "Flow_Control=NonBlocking,Has_ACLKEN=true,Has_RESULT_TREADY=false"
    ),
    "mult_gen": "ClockEnable=true",
    "div_gen": "ACLKEN=true",
}

#: How each core family pins its pipeline depth, applied last: `set_property
#: -dict` applies in list order and changing `Operation_Type` resets latency to
#: the new type's default. `C_Latency` is disabled until `Maximum_Latency` is
#: false; left alone a floating-point core builds at its maximum depth.
LATENCY = {
    "floating_point": "Maximum_Latency=false,C_Latency={lat}",
    "mult_gen": "PipeStages={lat}",
    "div_gen": "latency_configuration=Manual,latency={lat}",
}

_FP = "floating_point"

_NO_DSP = "C_Mult_Usage=No_Usage"

# The floating_point core offers Half/Single/Double/Custom and no bfloat16, so
# bfloat16 is spelled as a custom format: 8 exponent bits and 8 fraction (7
# stored plus the hidden one). Single is the default and needs no fragment.
_DOUBLE = (
    "A_Precision_Type=Double,C_A_Exponent_Width=11,C_A_Fraction_Width=53,"
    "Result_Precision_Type=Double,C_Result_Exponent_Width=11,"
    "C_Result_Fraction_Width=53"
)
_BF16 = (
    "A_Precision_Type=Custom,C_A_Exponent_Width=8,C_A_Fraction_Width=8,"
    "Result_Precision_Type=Custom,C_Result_Exponent_Width=8,"
    "C_Result_Fraction_Width=8"
)
# IEEE half is a native precision (exponent 5, fraction 11 with the hidden bit).
_HALF = (
    "A_Precision_Type=Half,C_A_Exponent_Width=5,C_A_Fraction_Width=11,"
    "Result_Precision_Type=Half,C_Result_Exponent_Width=5,"
    "C_Result_Fraction_Width=11"
)

# One Programmable compare core serves every predicate, which the instantiating
# wrapper drives as a constant on the core's operation channel.
_CMP = "Operation_Type=Compare,C_Compare_Operation=Programmable"

# The Add_Subtract shape is the "Both" core: an operation channel selects add
# against subtract, so the pair shares one measured core.
_ADD = "00000000"
_SUB = "00000001"

RECIPES: dict[OperatorIP, VivadoCore] = {
    ip.fadd: VivadoCore(_FP, "Operation_Type=Add_Subtract", _NO_DSP, _ADD),
    ip.fsub: VivadoCore(_FP, "Operation_Type=Add_Subtract", _NO_DSP, _SUB),
    ip.fmul: VivadoCore(_FP, "Operation_Type=Multiply", _NO_DSP),
    ip.fdiv: VivadoCore(_FP, "Operation_Type=Divide"),
    ip.fsqrt: VivadoCore(_FP, "Operation_Type=Square_root"),
    ip.fcmp: VivadoCore(_FP, _CMP),
    ip.dadd: VivadoCore(_FP, f"Operation_Type=Add_Subtract,{_DOUBLE}", _NO_DSP, _ADD),
    ip.dsub: VivadoCore(_FP, f"Operation_Type=Add_Subtract,{_DOUBLE}", _NO_DSP, _SUB),
    ip.dmul: VivadoCore(_FP, f"Operation_Type=Multiply,{_DOUBLE}", _NO_DSP),
    ip.ddiv: VivadoCore(_FP, f"Operation_Type=Divide,{_DOUBLE}"),
    ip.dsqrt: VivadoCore(_FP, f"Operation_Type=Square_root,{_DOUBLE}"),
    ip.dcmp: VivadoCore(_FP, f"{_CMP},{_DOUBLE}"),
    ip.bfadd: VivadoCore(_FP, f"Operation_Type=Add_Subtract,{_BF16}", "", _ADD),
    ip.bfsub: VivadoCore(_FP, f"Operation_Type=Add_Subtract,{_BF16}", "", _SUB),
    ip.bfmul: VivadoCore(_FP, f"Operation_Type=Multiply,{_BF16}", _NO_DSP),
    ip.hadd: VivadoCore(_FP, f"Operation_Type=Add_Subtract,{_HALF}", _NO_DSP, _ADD),
    ip.hsub: VivadoCore(_FP, f"Operation_Type=Add_Subtract,{_HALF}", _NO_DSP, _SUB),
    ip.hmul: VivadoCore(_FP, f"Operation_Type=Multiply,{_HALF}", _NO_DSP),
    ip.hdiv: VivadoCore(_FP, f"Operation_Type=Divide,{_HALF}"),
    ip.hcmp: VivadoCore(_FP, f"{_CMP},{_HALF}"),
    ip.i2f: VivadoCore(
        _FP,
        "Operation_Type=Fixed_to_float,A_Precision_Type=Int32,"
        "C_A_Exponent_Width=32,C_A_Fraction_Width=0",
    ),
    ip.f2i: VivadoCore(
        _FP,
        "Operation_Type=Float_to_fixed,Result_Precision_Type=Int32,"
        "C_Result_Exponent_Width=32,C_Result_Fraction_Width=0",
    ),
    ip.fcvt: VivadoCore(
        _FP,
        "Operation_Type=Float_to_float,"
        "A_Precision_Type=Single,C_A_Exponent_Width=8,C_A_Fraction_Width=24,"
        "Result_Precision_Type=Double,C_Result_Exponent_Width=11,"
        "C_Result_Fraction_Width=53",
    ),
    ip.bf2f: VivadoCore(
        _FP,
        "Operation_Type=Float_to_float,"
        "A_Precision_Type=Custom,C_A_Exponent_Width=8,C_A_Fraction_Width=8,"
        "Result_Precision_Type=Single,C_Result_Exponent_Width=8,"
        "C_Result_Fraction_Width=24",
    ),
}


# The fabric build of a multiply: an adder tree in LUTs, not DSP columns. It
# repeats a key the shape already sets; in a `set_property -dict` list the last
# value for a key wins.
_LUT_MULT = "Multiplier_Construction=Use_LUTs"


# `arith.muli` on iN returns the low N bits, so the core is asked for exactly
# those: without `Use_Custom_Output_Width` the width bounds are ignored and the
# core builds the full 2N-bit product (36 DSPs at 32x32 against a handful).
def _mult(width: int) -> VivadoCore:
    return VivadoCore(
        "mult_gen",
        f"PortAWidth={width},PortBWidth={width},"
        "PortAType=Signed,PortBType=Signed,"
        "Multiplier_Construction=Use_Mults,OptGoal=Speed,"
        f"Use_Custom_Output_Width=true,OutputWidthHigh={width - 1},"
        "OutputWidthLow=0",
        _LUT_MULT,
    )


# One divider core computes quotient and remainder together; the div and rem
# archetypes of one width and signedness share the recipe and the wrapper
# slices the packed result.
def _div(width: int, sign: str) -> VivadoCore:
    return VivadoCore(
        "div_gen",
        f"dividend_and_quotient_width={width},divisor_width={width},"
        f"remainder_type=Remainder,operand_sign={sign},algorithm_type=Radix2",
    )


for _w in (8, 16, 32, 64):
    _mul = getattr(ip, f"imul{_w}")
    assert _mul.parse_argument_annotations()[0].primitive_width == _w
    RECIPES[_mul] = _mult(_w)
    for _stem in ("idiv", "udiv", "irem", "urem"):
        _a = getattr(ip, f"{_stem}{_w}")
        assert _a.parse_argument_annotations()[0].primitive_width == _w
        RECIPES[_a] = _div(_w, "Unsigned" if _a.optype.endswith("ui") else "Signed")

del _a, _mul, _stem, _w

# No LogiCORE builds a fused multiply-add at this width, so the `rtl` recipe
# lets synthesis infer the DSP cascade with the addend absorbed.
RECIPES[ip.imuladd32] = VivadoCore("rtl", "a * b + c")

# The narrow core under `imulw33`'s 64-bit signature: the 33x33 product's low
# 64 bits, which the shim feeds the extended operands' low 33 bits.
RECIPES[ip.imulw33] = VivadoCore(
    "mult_gen",
    "PortAWidth=33,PortBWidth=33,"
    "PortAType=Signed,PortBType=Signed,"
    "Multiplier_Construction=Use_Mults,OptGoal=Speed,"
    "Use_Custom_Output_Width=true,OutputWidthHigh=63,OutputWidthLow=0",
)

_RECIPE_BY_NAME = {a.func_name: r for a, r in RECIPES.items()}

# Operation-channel opcodes of the Programmable compare core (PG060). An
# unordered relation takes its ordered opcode, the NaN-free contract the cosim
# models state.
_CMP_OPCODE = {
    "uno": 0b00000100,
    "lt": 0b00001100,
    "eq": 0b00010100,
    "le": 0b00011100,
    "gt": 0b00100100,
    "ne": 0b00101100,
    "ge": 0b00110100,
}


def _cmp_opcode(predicate: str) -> int | None:
    if predicate == "uno":
        return _CMP_OPCODE["uno"]
    if predicate[:1] in {"o", "u"}:
        return _CMP_OPCODE.get(predicate[1:])
    return None  # ord / true / false have no single opcode


class Generated(NamedTuple):
    """What the extern operator modules of a design need to synthesize with
    real cores: the wrapper Verilog, the core-generation script, and what no
    recipe covers. A ``missing`` module stays a black box and synthesizes to
    nothing."""

    shims: str  # one wrapper module per extern operator module
    ip_tcl: str  # `create_ip` script building every core, deduplicated
    cores: tuple[str, ...]  # impl symbols, one `<impl>_core` each
    missing: tuple[str, ...]  # `module: reason` for what cannot be built


def _config(recipe: VivadoCore, latency: int, no_dsp: bool = False) -> str:
    """The full core configuration, `Key=Value` comma separated, in apply
    order: clock-enable base, shape, the DSP-free fragment if the row was
    measured without DSPs, the pipeline depth last."""
    assert latency >= 1, "a zero-latency core is not an instanced IP"
    parts = [CE_BASE[recipe.core], recipe.shape]
    if no_dsp:
        assert recipe.no_dsp, "this core has no DSP-free build"
        parts.append(recipe.no_dsp)
    parts.append(LATENCY[recipe.core].format(lat=latency))
    return ",".join(parts)


def _header(op: Operator) -> str:
    decls = []
    for p in op.ports:
        width = f" [{p.width - 1}:0]" if p.width > 1 else ""
        decls.append(f"  {'input' if p.is_input else 'output'}{width} {p.name}")
    return f"module {op.module}(\n" + ",\n".join(decls) + "\n);\n"


def _split(op: Operator):
    """Data ports, the clock, the enable expression, and the result port. A
    free-running module has no `ce` port and the core's enable ties high."""
    data = [p for p in op.ports if p.role is Operator.Role.DATA]
    clk = next(p.name for p in op.ports if p.role is Operator.Role.CLK)
    ce = next((p.name for p in op.ports if p.role is Operator.Role.CE), "1'b1")
    out = next(p for p in op.ports if p.role is Operator.Role.OUT)
    return data, clk, ce, out


def _fp_shim(op: Operator, recipe: VivadoCore, opcode: int | None) -> str:
    """The floating_point core: one AXI channel per operand, named as the
    operand is; `tvalid` ties high (non-blocking flow control computes every
    cycle) and the result `tvalid` is ignored. A core whose shape leaves an
    operation channel gets its constant, ``opcode`` on a compare. The core pads
    `tdata` to a byte boundary, so a narrower result (a compare's single bit)
    slices a wire."""
    data, clk, ce, out = _split(op)
    conns = [f".aclk({clk})", f".aclken({ce})"]
    for p in data:
        conns.append(f".s_axis_{p.name}_tvalid(1'b1)")
        conns.append(f".s_axis_{p.name}_tdata({p.name})")
    operation = f"{opcode:08b}" if opcode is not None else recipe.operation
    if operation:
        conns.append(".s_axis_operation_tvalid(1'b1)")
        conns.append(f".s_axis_operation_tdata(8'b{operation})")
    padded = (out.width + 7) // 8 * 8
    body, tail, sink = "", "", out.name
    if padded != out.width:
        body = f"  wire [{padded - 1}:0] result;\n"
        tail = f"  assign {out.name} = result[{out.width - 1}:0];\n"
        sink = "result"
    conns.append(".m_axis_result_tvalid()")
    conns.append(f".m_axis_result_tdata({sink})")
    joined = ",\n    ".join(conns)
    return (
        f"{_header(op)}{body}"
        f"  {op.impl}_core u (\n    {joined}\n  );\n{tail}endmodule\n"
    )


def _mult_shim(op: Operator, arche: OperatorIP) -> str:
    """A `fed_width` core's ports are narrower than the module's: the operands
    are extensions, so their low bits are the whole value."""
    data, clk, ce, out = _split(op)
    a, b = data
    sl = f"[{arche.fed_width - 1}:0]" if arche.fed_width else ""
    return (
        f"{_header(op)}"
        f"  {op.impl}_core u (.CLK({clk}), .CE({ce}), "
        f".A({a.name}{sl}), .B({b.name}{sl}), .P({out.name}));\n"
        "endmodule\n"
    )


def _div_shim(op: Operator, arche: OperatorIP) -> str:
    """The div_gen core packs quotient above remainder in `dout`, each field
    padded to a byte boundary; the wrapper takes the half its mnemonic means."""
    data, clk, ce, out = _split(op)
    dividend, divisor = data
    w = out.width
    assert w % 8 == 0 and dividend.width == w, "field padding needs byte widths"
    upper = f"[{2 * w - 1}:{w}]" if arche.optype.startswith("div") else ""
    return (
        f"{_header(op)}"
        f"  wire [{2 * w - 1}:0] dout;\n"
        f"  {op.impl}_core u (\n"
        f"    .aclk({clk}),\n"
        f"    .aclken({ce}),\n"
        f"    .s_axis_dividend_tvalid(1'b1),\n"
        f"    .s_axis_dividend_tdata({dividend.name}),\n"
        f"    .s_axis_divisor_tvalid(1'b1),\n"
        f"    .s_axis_divisor_tdata({divisor.name}),\n"
        f"    .m_axis_dout_tvalid(),\n"
        f"    .m_axis_dout_tdata(dout)\n"
        f"  );\n"
        f"  assign {out.name} = dout{upper or f'[{w - 1}:0]'};\n"
        "endmodule\n"
    )


def _rtl_shim(op: Operator, arche: OperatorIP, expr: str) -> str:
    """An ``rtl`` recipe's whole build, for the one shape it carries,
    ``x * y + z`` over the data ports: an input stage, the product on registers
    of its own, the addend delayed alongside, and the add as the last stage,
    where the final slice's ALU absorbs it. The product needs its own registers
    for synthesis to pipeline the DSP cascade; summed before its first register
    the whole cascade lands in one cycle."""
    m = re.fullmatch(r"(\w+) \* (\w+) \+ (\w+)", expr)
    assert m, "an rtl recipe builds `x * y + z` today"
    x, y, z = m.groups()
    data, clk, ce, out = _split(op)
    w = out.width
    lat = arche.timing.latency
    assert lat >= 3, "the staged product needs input, product and add stages"
    widths = {p.name: p.width for p in data}
    regs = [f"  reg [{widths[n] - 1}:0] {n}_q;" for n in (x, y)]
    shifts = [f"      {n}_q <= {n};" for n in (x, y)]
    prod, addend = "m0", z
    regs.append(f"  reg [{w - 1}:0] m0;")
    shifts.append(f"      m0 <= {x}_q * {y}_q;")
    for k in range(1, lat - 2):
        regs.append(f"  reg [{w - 1}:0] m{k};")
        shifts.append(f"      m{k} <= m{k - 1};")
        prod = f"m{k}"
    for k in range(lat - 1):
        regs.append(f"  reg [{widths[z] - 1}:0] {z}_d{k};")
        shifts.append(f"      {z}_d{k} <= {addend};")
        addend = f"{z}_d{k}"
    regs.append(f"  reg [{w - 1}:0] r;")
    shifts.append(f"      r <= {prod} + {addend};")
    body = "\n".join(regs) + "\n"
    body += f"  always @(posedge {clk}) begin\n"
    body += f"    if ({ce}) begin\n" + "\n".join(shifts) + "\n    end\n"
    body += f"  end\n  assign {out.name} = r;\n"
    return f"{_header(op)}{body}endmodule\n"


def _ip_tcl(part: str, cores: dict[str, tuple[str, str]]) -> str:
    """One `create_ip` block per core, rooted at the script's own directory so
    the script runs from anywhere; an existing `.xci` is reused."""
    lines = [
        "set ipdir [file join [file dirname [file normalize [info script]]] ip]",
        "file mkdir $ipdir",
        f"create_project -in_memory -part {part}",
        "set_property target_language Verilog [current_project]",
    ]
    for impl, (kind, cfg) in sorted(cores.items()):
        props = " ".join(
            f"CONFIG.{k} {v}" for k, v in (p.split("=", 1) for p in cfg.split(","))
        )
        xci = f"$ipdir/{impl}_core/{impl}_core.xci"
        lines += [
            f"if {{![file exists {xci}]}} {{",
            f"  create_ip -name {kind} -vendor xilinx.com"
            f" -library ip -module_name {impl}_core -dir $ipdir",
            f"  set_property -dict [list {props}] [get_ips {impl}_core]",
            "} else {",
            f"  read_ip {xci}",
            "}",
            f"set_property generate_synth_checkpoint false [get_files {xci}]",
            f"generate_target synthesis [get_ips {impl}_core]",
        ]
    return "\n".join(lines) + "\n"


def generate(interfaces: Interfaces, device: Device) -> Generated:
    """The wrappers and cores for every extern operator module the manifest
    declares, resolved against ``device``: the impl symbol names the device
    row, whose archetype names the recipe and whose area decides the DSP-free
    build."""
    by_symbol = {o.symbol: o for o in device.operators}
    shims: dict[str, str] = {}
    cores: dict[str, tuple[str, str]] = {}  # impl -> (core kind, config)
    missing: list[str] = []
    for iface in interfaces.values():
        for op in iface.operators:
            if op.module in shims:
                continue
            arche = by_symbol.get(op.impl)
            recipe = _RECIPE_BY_NAME.get(arche.func_name) if arche else None
            if recipe is None:
                what = "device operator" if arche is None else "recipe"
                missing.append(f"{op.module}: no {what} for '{op.impl}'")
                continue
            opcode = _cmp_opcode(op.predicate) if op.predicate else None
            if op.predicate and opcode is None:
                missing.append(
                    f"{op.module}: predicate '{op.predicate}' has no compare opcode"
                )
                continue
            if recipe.core == "rtl":
                shims[op.module] = _rtl_shim(op, arche, recipe.shape)
                continue
            no_dsp = bool(recipe.no_dsp) and not any(
                n == "dsp" for n, _ in device.operator_uses.get(op.impl, ())
            )
            cores[op.impl] = (
                recipe.core,
                _config(recipe, arche.timing.latency, no_dsp),
            )
            if recipe.core == "floating_point":
                shims[op.module] = _fp_shim(op, recipe, opcode)
            elif recipe.core == "mult_gen":
                shims[op.module] = _mult_shim(op, arche)
            else:
                assert recipe.core == "div_gen"
                shims[op.module] = _div_shim(op, arche)
    return Generated(
        shims="\n".join(shims.values()),
        ip_tcl=_ip_tcl(device.part, cores),
        cores=tuple(sorted(cores)),
        missing=tuple(missing),
    )


def realize(interfaces: Interfaces, device: Device) -> Realization:
    """The fabrics' ``Device.realizer``: :func:`generate` folded into the
    neutral shape a scaffold writes."""
    g = generate(interfaces, device)
    files = {}
    if g.shims:
        files["shims.v"] = g.shims
    if g.cores:
        files["gen_ip.tcl"] = g.ip_tcl
    return Realization(files=files, missing=g.missing)
