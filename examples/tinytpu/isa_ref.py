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

import ast
import operator
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from examples.tinytpu.isa_encoding import (  # noqa: E402
    ACTIONS, MAXDIM, NAR, NVR, SPAD_ROWS, T,
    ACC_DTYPE, OPERAND_DTYPE,
    acc, expand, machine, operands, to_operand,
)
from examples.tinytpu.microarch_isa import (  # noqa: E402
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


#: The whole expression language the spec's actions are written in: a name, an
#: integer, and the operators below. Every `base`, `offset`, `count` and `when`
#: in `isa_spec.json` is one of `ar_s1`, `col_block * T`, `T + 1`,
#: `(mode & 1) == 0`, `acc == 1` -- and that is the complete list, measured off
#: the spec rather than assumed.
#:
#: WHY THIS IS NOT `eval`. These strings come from `isa_spec.json`, and the
#: spec is a candidate's to edit now (`chia_agent/design.py`). `eval` with
#: emptied builtins still reaches `().__class__.__bases__[0].__subclasses__()`,
#: and from there the frame of the frozen check that called it -- where
#: `gate_runner.py`'s nonce lives. A whitelist over the AST has no such reach:
#: an attribute, a subscript, a call or a comprehension is a ValueError naming
#: the construct, so an expression that tries is a REFUSED program rather than
#: an escape. `chia_agent/spec_policy.py` refuses the same constructs in the
#: spec at edit time; this is the half that cannot be edited around.
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.BitAnd: operator.and_,
           ast.BitOr: operator.or_, ast.BitXor: operator.xor,
           ast.LShift: operator.lshift, ast.RShift: operator.rshift}
_CMPOPS = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos,
             ast.Invert: operator.invert, ast.Not: operator.not_}


def _arith(node, env):
    """One node of a spec expression, evaluated. Anything else raises."""
    if isinstance(node, ast.Expression):
        return _arith(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"the spec names {node.id!r}, which this "
                             f"instruction's operands do not define")
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_arith(node.left, env),
                                      _arith(node.right, env))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_arith(node.operand, env))
    if isinstance(node, ast.Compare) and all(
            type(op) in _CMPOPS for op in node.ops):
        left, out = _arith(node.left, env), True
        for op, right in zip(node.ops, node.comparators):
            right = _arith(right, env)
            out = out and _CMPOPS[type(op)](left, right)
            left = right
        return out
    if isinstance(node, ast.BoolOp):
        values = [_arith(v, env) for v in node.values]
        return (all(values) if isinstance(node.op, ast.And) else any(values))
    raise ValueError(f"{type(node).__name__} is not in the expression language "
                     f"the spec's actions may use (names, ints, arithmetic, "
                     f"bitwise and comparison operators)")


def expression(text, env):
    """A spec expression's value. Importable, so the policy and the reference
    model agree by construction about what an expression may be."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"the spec expression {text!r} does not parse: {exc}")
    return _arith(tree, env)


def _resolve(expr, env):
    if expr is None:
        return None
    return int(expression(expr, env))


def _live(action, env):
    if action.when is None:
        return True
    return bool(expression(action.when, env))


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
