#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A TinyTPU program as text: one line per instruction, named operands, AGU
terms, indented by loop depth.

Everything it knows comes from `isa_encoding`, which is generated from
`isa_spec.json`, so an opcode added to the spec is listed with its operand
names and nothing here changes.

    from examples.tinytpu.disasm import disassemble
    print(disassemble(program))
"""

from examples.tinytpu.isa_encoding import (
    OP_ENDLOOP, OP_LOOP, OPCODE_NAME, OPERAND_NAME, decode, decode_agu,
    operands,
)

FIELDS = ("f0", "f1", "f2", "f3")


def line(w0, w1):
    """One instruction, without its pc or indentation."""
    op, nr, f0, f1, f2, f3 = decode(w0)
    if op == OP_LOOP:
        return f"loop x{nr}"
    if op == OP_ENDLOOP:
        return "endloop"
    named = operands(op, f0, f1, f2, f3, nr)
    named.pop("nr", None)
    text = " ".join([OPCODE_NAME[op]] +
                    [f"{name}={value}" for name, value in named.items()])
    # An AGU term targets a field by position; name it the way the operand is.
    field_name = [name or field for name, field in zip(OPERAND_NAME[op], FIELDS)]
    terms = [f"{field_name[target - 1]}+=iv{level}*{stride}"
             for target, level, stride in decode_agu(w1) if target]
    if terms:
        text += "   agu: " + ", ".join(terms)
    return text


def disassemble(prog, indent="  "):
    """The whole program, one numbered line per instruction."""
    lines, depth = [], 0
    for pc, (w0, w1) in enumerate(prog):
        if decode(w0)[0] == OP_ENDLOOP:
            depth -= 1
        lines.append(f"{indent}{pc:3d}  {'  ' * depth}{line(w0, w1)}")
        if decode(w0)[0] == OP_LOOP:
            depth += 1
    return "\n".join(lines)
