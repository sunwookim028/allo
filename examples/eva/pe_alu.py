"""Ultra-simple PE: a pure ALU.  (op1, op2, opcode) -> result.

No state, no sequencer, no register file, no ISA fetch/decode -- just the
arithmetic. The agent builds the wrapper around this (where op1/op2 come from,
how the opcode is chosen, where the result goes, sequencing, flow control).
"""
from allo.ir.types import float16, int32

Ty = float16
OP_ADD, OP_SUB, OP_MULT, OP_MOV = 0, 1, 2, 3
OP_GEQ, OP_LT = 8, 9


def alu_pe(op1: Ty, op2: Ty, opcode: int32) -> Ty:
    res: Ty = 0
    if opcode == OP_ADD:    res = op1 + op2
    elif opcode == OP_SUB:  res = op1 - op2
    elif opcode == OP_MULT: res = op1 * op2
    elif opcode == OP_GEQ:
        if op1 >= op2: res = 1.0
        else:          res = -1.0
    elif opcode == OP_LT:
        if op1 < op2:  res = 1.0
        else:          res = -1.0
    else:                   res = op1   # MOV / passthrough
    return res
