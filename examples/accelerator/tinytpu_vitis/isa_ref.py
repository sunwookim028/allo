# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The instruction set as numpy: what a TinyTPU-isa program MEANS.

`bench_isa.py` can only check GEMM, because its gold is `A @ B`. Any other
program -- and so any unit field a GEMM never varies (a nonzero DRAM row, a
`vadd` whose destination is not its source) -- had no reference at all. This
is that reference: one pass over the same resolved dynamic stream the
sequencer issues.

    C_expected = run(prog, A, B, C_before)

**It is built on the spec, not on the design, and it has no per-opcode arm.**
It interprets the ACTIONS: each instruction is an ordered list of per-unit
effects, and this walks them -- a read fetches an element, a compute applies
one of the primitives below, a write stores one. Every ISA fact it uses --
opcode numbers, which field carries which operand, the bit layout, the loop
stack, the AGU, the accumulator width, the output conversion, and now which
memory each instruction touches and in what order -- comes from
`isa_encoding`, which `gen_isa.py` generates from `isa_spec.json`.

What that buys is measurable: an instruction that recombines the primitives
already in `PRIMITIVES` needs no edit to this file at all. An instruction that
needs arithmetic none of them does needs exactly one entry, and the model
refuses it until the entry exists rather than computing something wrong.

What it still takes from the design is the program VALIDATOR, `check_program`.
That is a guard, not an oracle: it refuses a program the hardware would run to
a wrong answer, so this model never has to invent a value for an unwritten row.
`C_before` is carried through untouched outside what the program's `mvout`s
name, which is how a sentinel check falls out of comparing the whole of `C`.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from examples.accelerator.tinytpu_vitis.isa_encoding import (  # noqa: E402
    ACTIONS, MAXDIM, NAR, NVR, SPAD_ROWS, T,
    ACC_DTYPE, OPERAND_DTYPE,
    acc, expand, machine, operands, to_operand,
)
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    check_program,
)


def _select(*values):
    """The one of a mutually exclusive pair of predicated reads that fired."""
    live = [v for v in values if v is not None]
    if len(live) != 1:
        raise ValueError(f"`select` saw {len(live)} live operands, not 1")
    return live[0]


#: Every arithmetic an action may name. An instruction composed only of these
#: needs nothing added here; one that needs new arithmetic needs one entry,
#: and until it exists `run` raises rather than guessing.
PRIMITIVES = {
    "select": _select,
    # PE(i, j) holds lane j of weight row i and taps lane i of the activation
    # word, so output column j is sum_i act[i] * W[i][j]. One `acc()` at the
    # end although the array accumulates sequentially down the column:
    # wraparound addition is a ring homomorphism, so the order the spec fixes
    # cannot be observed in THIS configuration. A configuration whose
    # accumulate rounds would have to fold in the spec's order term by term.
    "matmul": lambda act, w: act @ w,
    "acc_add": lambda base, addend: acc((0 if base is None else base) + addend),
    "add": lambda left, right: acc(left + right),
    "max0": lambda v: np.maximum(v, 0),
    "to_operand": to_operand,
}


class _State:
    """The machine's memories, and the only place an element is addressed."""

    def __init__(self, A, B, C):
        self.mem = {
            "A": np.asarray(A, OPERAND_DTYPE).reshape(MAXDIM, MAXDIM),
            "B": np.asarray(B, OPERAND_DTYPE).reshape(MAXDIM, MAXDIM),
            "C": np.array(C, OPERAND_DTYPE).reshape(MAXDIM, MAXDIM),
            "spad": np.zeros((SPAD_ROWS, T), np.int64),
            "vr": np.zeros((NVR, T), np.int64),
            "ar": np.zeros((NAR, T), np.int64),
        }

    def read(self, state, row, offset):
        if offset is None:
            return self.mem[state][row]
        return self.mem[state][row, offset:offset + T]

    def write(self, state, row, offset, value):
        if offset is None:
            self.mem[state][row] = value
        else:
            self.mem[state][row, offset:offset + T] = value


def _resolve(expression, env):
    if expression is None:
        return None
    return int(eval(expression, {"__builtins__": {}}, env))  # noqa: S307


def _live(action, env):
    if action.when is None:
        return True
    return bool(eval(action.when, {"__builtins__": {}}, env))  # noqa: S307


def _apply(action, values, state, env, row):
    if not _live(action, env):
        return
    base = _resolve(action.base, env)
    offset = _resolve(action.offset, env)
    count = _resolve(action.count, env) or 1
    if action.kind == "read":
        if action.per == "instruction" and count > 1:
            value = np.stack([state.read(action.state, base + k, offset)
                              for k in range(count)])
        else:
            value = state.read(action.state, base + row, offset)
        values[action.into] = value
    elif action.kind == "write":
        state.write(action.state, base + row, offset, values[action.args[0]])
    elif action.kind == "compute":
        primitive = PRIMITIVES.get(action.compute)
        if primitive is None:
            raise NotImplementedError(
                f"the actions name a `{action.compute}` and this reference "
                f"model has no primitive for it; add one to PRIMITIVES")
        values[action.into] = primitive(*(values.get(a) for a in action.args))
    elif action.kind in ("emit", "receive"):
        if action.into is not None:
            values[action.into] = values.get(action.args[0])


def run(prog, A, B, C):
    """Execute `prog` on flat operand-format `A`, `B`, `C` (MAXDIM*MAXDIM
    each). Returns the new `C`; the arguments are not modified."""
    check_program(prog)
    state = _State(A, B, C)
    names = {i.name: i for i in machine().instructions}
    for op, nr, f0, f1, f2, f3 in expand(prog):
        env = dict(operands(op, f0, f1, f2, f3, nr), T=T, MAXDIM=MAXDIM)
        actions = ACTIONS[op]
        if not actions:
            continue
        values = {}
        for action in actions:
            if action.per == "instruction":
                _apply(action, values, state, env, 0)
        body = [a for a in actions if a.per != "instruction"]
        for r in range(nr if names else 0):
            for action in body:
                _apply(action, values, state, env, r)
    return state.mem["C"].reshape(-1)


# The lanes above are int64 so that no intermediate wraps before `acc()` folds
# the value into the spec's accumulator format. That is only sound while the
# spec's accumulator fits in an int64.
assert np.iinfo(np.int64).min < np.iinfo(ACC_DTYPE).min and \
    np.iinfo(ACC_DTYPE).max < np.iinfo(np.int64).max, (
        "the spec's accumulator no longer fits inside this model's int64 lanes")
