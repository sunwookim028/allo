# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The mapper, and the rule by which it picks. FROZEN.

The agent in the co-design loop edits the hardware and its encoding. It may not
edit this file, and it cannot import it: `spec_policy.py` refuses any
`examples.*` import other than the two spec modules, `evaluate.py` composes this
file from git rather than from disk, and the tool surface refuses every path
except the two bare spec file names.

What this file is
=================

A stand-in for ACT's loop-nest mapper, reduced to the part that is reusable.
ACT is Kai Shao's work (https://github.com/kkkaishao/allo, branch ``act``); see
`docs/source/extensions/act.rst` for the audit that decided to **cite it, not
integrate it**, and `ATTRIBUTION.md` at tag `chia-codesign-final` for the terms.
No ACT code is copied here. What is reused is its *interface*: a mapping is a
tuple of `Loop(rank, factor, level, spatial)`, outermost-first, so a real
`allo.exp.dsa.mapping.Mapping.loops` plugs straight into `select()`.

Three reasons the rest of ACT is not here, each measured in that page: its
mapspace enumerator is dead code that nothing can construct a `Binding` for; its
lowering fully unrolls the nest and collapses an operand address to a scalar
offset, destroying exactly the structure an AGU needs; and its cost model,
`(makespan, emits)`, models no instruction memory, so a loop scored on it would
reward a program our imem cannot hold.

The inner loop is EXHAUSTIVE
============================

For one hardware candidate, `search()` enumerates the whole mapspace and asks
the candidate's encoder about every nest. It does not search cleverly; it does
not stop early. That is deliberate. A non-exhaustive inner search would turn
"which hardware runs the better program" into "whose search got luckier", and
the point of the loop is the first question. Keeping the mapspace small enough
to enumerate is therefore a requirement, not a convenience: at 16x16x16 it is
1,226 nests and one second of pure python.

The selection rule, stated
==========================

Among the nests the candidate can encode at a shape, `select()` takes the
minimum of

    (instruction FETCHES, static instruction words, nest string)

Both counts are *static* properties of the program -- a count of fetches and a
count of 64-bit words -- not cycle estimates, and the nest string only breaks
ties so the choice is deterministic.

`fetches()` below counts what the sequencer actually fetches, **control flow
included**. `microarch_isa.expand` deliberately does not: it runs the loops and
yields only the data issues, because its job is to tell each unit how much work
it will be sent, and `LOOP`/`ENDLOOP` are sent to no unit. They are still
fetched and dispatched, so a rule built on `len(expand(prog))` alone
under-charges a loop-heavy nest by roughly a factor of two and can misrank two
close candidates. (Measured at all five shapes on the shipped design: charging
the fetches changes no pick, because the margins are wide -- 59 against 112
against 214 at 16x16x16. The rule charges them anyway; being right for a reason
is cheaper than being right by luck.)
The nest must be encodable at BOTH relu settings (`cosim.py`'s correctness
testbench runs `gemm.relu`), and it is ranked by the `relu=False` program, which
is the one the scored testbench runs.

This ranking is the only place a proxy is allowed, and it never leaves this
file: `evaluate.py` reports cosim cycles and csynth resources, never a modelled
cycle count. The failure mode this avoids is named in act.rst -- a program
scored on `(makespan, emits)`, or on the five-point regression
`74.5 + 21.70 x dynamic`, optimises the model and its residual.

What comes out
==============

`search()` returns, per shape, how many nests were enumerated, how many became
encodable, and **which constraint refused the rest**. That histogram is the
co-design signal: at 16x16x16 on the shipped design, 1,150 of 1,226 are refused
by one static instruction field (`acc`), 17 by `AGU_TERMS=3`, 2 by the
accumulator RAW distance, and 54 by the encoder's own limitation. A hardware
change moves those numbers before it moves a cycle count, and it may move them
without ever moving a cycle count -- which is a result, not a failure.
"""

from __future__ import annotations

import itertools
import os
import re
import sys
from collections import namedtuple

#: ACT's `mapping.Loop`, by field name only, so the two sides plug together.
#: A `spatial` loop is Timeloop's fanout across instances; this machine has one
#: array and no instance index, so a spatial loop is refused rather than
#: silently serialised.
Loop = namedtuple("Loop", "rank factor level spatial")
Loop.__new__.__defaults__ = (False,)

DRAM, ARRAY = "dram", "array"
RANKS = ("M", "K", "N")
#: Emitted slots each rank's residual is split over, before permutation. Two is
#: what a two-level mapper offers and is what keeps the space enumerable.
SLOTS = 2

#: How a refusal is named in the report. Ordered; first match wins. The bucket
#: a refusal lands in is a label, never part of the objective -- the number that
#: matters, `encodable`, is structural (the encoder returned a program and the
#: assembler took it).
CAUSES = (
    ("agu-terms", r"address terms"),
    ("loop-depth", r"loop stack"),
    ("acc-peel", r"^acc-peel:"),
    ("emitter", r"^emitter:"),
    ("intrinsic", r"^intrinsic:"),
    ("shape", r"^shape:"),
    ("coverage", r"^coverage:"),
    ("spatial", r"^spatial:"),
    ("imem", r"IMEM_SIZE"),
    ("accumulator-raw-distance", r"accumulator|AR_RAW_DIST|distance"),
)


def _design():
    """The candidate's two modules. Imported here, not at module scope, so this
    file can be read and reasoned about without a built `allo`."""
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
    from examples.accelerator.tinytpu_vitis import isa_dsl, microarch_isa
    return isa_dsl, microarch_isa


# ---------------------------------------------------------------- the space --
def canonical(M, K, N, T):
    """The nest describing the tiling `isa_dsl.gemm_program` hardcodes.

    `gemm_from_nest(canonical(...))` must re-emit `gemm_program` -- and hence
    `microarch_isa.gemm_program_handwritten` -- word for word. That identity is
    the seam, and `codesign_gate.py` asserts it on every candidate: an encoder
    that has drifted from the program the frozen gates verify is rejected.
    """
    return (Loop("N", N // T, DRAM), Loop("K", K // T, DRAM),
            Loop("M", M, ARRAY), Loop("K", T, ARRAY), Loop("N", T, ARRAY))


def nests(M, K, N, T, slots=SLOTS):
    """Every nest a two-level loop-nest mapper would offer for this GEMM.

    Each rank's residual after the intrinsic tile is split over `slots` emitted
    slots and the result permuted -- the factorisations x permutations a mapper
    enumerates, with the innermost level pinned to the intrinsic. No spatial
    slots: one array, no instance index.
    """
    def divisors(n):
        return [d for d in range(1, n + 1) if n % d == 0]

    def splits(n, parts):
        if parts == 1:
            return [(n,)]
        return [(d,) + r for d in divisors(n) for r in splits(n // d, parts - 1)]

    seen = set()
    for Mt in divisors(M):
        tail = (Loop("M", Mt, ARRAY), Loop("K", T, ARRAY), Loop("N", T, ARRAY))
        residual = {"M": M // Mt, "K": K // T, "N": N // T}
        for combo in itertools.product(*(splits(residual[r], slots)
                                         for r in RANKS)):
            factors = dict(zip(RANKS, combo))
            loops = [Loop(r, f, DRAM) for r in RANKS for f in factors[r] if f != 1]
            for perm in itertools.permutations(loops):
                nest = tuple(perm) + tail
                if nest not in seen:
                    seen.add(nest)
                    yield nest


def coverage(nest, M, K, N):
    """The one legality rule that belongs to the MAPPER, not to the machine:
    the loop factors must cover the extents exactly, and no loop may be spatial.

    Frozen on purpose. Everything else a nest can be refused for is a property
    of the hardware or of its encoding, and the agent may move it; this cannot
    be moved, because a nest that does not cover the iteration space computes
    something other than the GEMM.
    """
    extents = {"M": M, "K": K, "N": N}
    for loop in nest:
        if loop.rank not in extents:
            raise ValueError(f"coverage: unknown rank {loop.rank!r}")
        if loop.spatial:
            raise ValueError(
                f"spatial: loop {loop.rank!r} fans out across instances, and "
                f"this machine has one array and no instance index")
    for rank, extent in extents.items():
        got = 1
        for loop in nest:
            if loop.rank == rank:
                got *= loop.factor
        if got != extent:
            raise ValueError(
                f"coverage: rank {rank} factors to {got}, extent {extent}")


def cause_of(exc):
    """Which constraint refused a nest, as one of `CAUSES` or `other: ...`."""
    text = str(exc) or type(exc).__name__
    for label, pattern in CAUSES:
        if re.search(pattern, text, re.M):
            return label
    return "other: " + text.split("\n")[0][:60]


def describe(nest):
    """A nest as a short string: the emitted loops, then the intrinsic rows."""
    inner = nest[-1].level
    emitted = ">".join(f"{l.rank}{l.factor}" for l in nest
                       if l.level != inner) or "-"
    rows = [l.factor for l in nest if l.level == inner and l.rank == "M"]
    return f"{emitted} rows={rows[0] if rows else '?'}"


# --------------------------------------------------------------- the search --
def fetches(prog, microarch_isa):
    """How many instructions the sequencer FETCHES to run `prog`, control flow
    included -- `expand`'s data issues plus every `LOOP` and `ENDLOOP`.

    `expand` exists to tell each unit its work count and so yields nothing for
    control flow; a ranking that used it alone would treat a loop as free. This
    mirrors `_trace`'s walk rather than reusing it, because what is wanted here
    is the count of fetches and not the resolved operands.
    """
    OP_LOOP, OP_ENDLOOP = microarch_isa.OP_LOOP, microarch_isa.OP_ENDLOOP
    pc, stack, guard, n = 0, [], 0, 0
    while pc < len(prog):
        guard += 1
        if guard > 1 << 22:
            raise RuntimeError("program does not terminate")
        w0, _ = prog[pc]
        op = w0 & 0x3F
        if op == OP_LOOP:
            stack.append([pc + 1, 0, (w0 >> 54) & 0xFF])
            pc += 1
            n += 1
        elif op == OP_ENDLOOP:
            frame = stack[-1]
            frame[1] += 1
            n += 1
            if frame[1] < frame[2]:
                pc = frame[0]
            else:
                stack.pop()
                pc += 1
        else:
            pc += 1
    return n + len(microarch_isa.expand(prog))


def encode(nest, M, K, N, relu, isa_dsl, microarch_isa):
    """The candidate's answer for one nest: a program and its static costs.

    Raises whatever the candidate raises. The nest counts as encodable only if
    the encoder produced a program AND the assembler took it -- which is where
    `IMEM_SIZE` is enforced, so a program the instruction memory cannot hold is
    refused here rather than silently truncated.
    """
    prog = isa_dsl.gemm_from_nest(nest, M, K, N, relu)
    words = microarch_isa.assemble(prog)          # check_program + IMEM_SIZE
    return prog, fetches(prog, microarch_isa), len(words)


def search(M, K, N, slots=SLOTS):
    """Enumerate the whole mapspace for one shape against this hardware.

    Returns a dict with `total`, `encodable`, `refused` (cause -> count) and
    `ranked`: the encodable nests, best first, by the rule in this module's
    docstring. EXHAUSTIVE -- see the docstring.
    """
    isa_dsl, microarch_isa = _design()
    ok, refused = [], {}
    for nest in nests(M, K, N, microarch_isa.T, slots):
        try:
            coverage(nest, M, K, N)
        except ValueError as e:
            refused[cause_of(e)] = refused.get(cause_of(e), 0) + 1
            continue
        try:
            prog, dyn, words = encode(nest, M, K, N, False, isa_dsl, microarch_isa)
            # The nest must also be sayable with relu: cosim.py's correctness
            # testbench runs gemm.relu on the same RTL.
            encode(nest, M, K, N, True, isa_dsl, microarch_isa)
        except Exception as e:                    # noqa: BLE001 -- the candidate's
            cause = cause_of(e)
            refused[cause] = refused.get(cause, 0) + 1
            continue
        ok.append((dyn, words, describe(nest), nest, prog))
    ok.sort(key=lambda r: (r[0], r[1], r[2]))
    return {"total": len(ok) + sum(refused.values()),
            "encodable": len(ok),
            "refused": dict(sorted(refused.items(), key=lambda kv: -kv[1])),
            "ranked": ok}


def select(M, K, N, slots=SLOTS):
    """The best nest this hardware can run at this shape, by the stated rule."""
    found = search(M, K, N, slots)
    if not found["ranked"]:
        raise RuntimeError(
            f"mapspace {M}x{K}x{N}: this hardware can encode none of "
            f"{found['total']} nests ({found['refused']})")
    return found["ranked"][0][3], found


def report(shapes, slots=SLOTS, out=print):
    """Per shape: the counts, the refusal histogram, and the chosen nest.

    Returns {shape: {total, encodable, refused, chosen, fetches, words}} --
    the co-design signal `evaluate.py` records next to the measured cycles.
    """
    summary = {}
    for (M, K, N) in shapes:
        found = search(M, K, N, slots)
        tag = f"{M}x{K}x{N}"
        out(f"MAPSPACE {tag}: {found['encodable']}/{found['total']} encodable")
        for cause, n in found["refused"].items():
            out(f"MAPSPACE {tag}: refused {n:6d}  {cause}")
        for dyn, words, name, _nest, _prog in found["ranked"][:8]:
            out(f"MAPSPACE {tag}: nest {name:22s} {dyn:5d} fetches {words:4d} words")
        if not found["ranked"]:
            out(f"MAPSPACE {tag}: CHOSEN none")
            summary[tag] = {"total": found["total"], "encodable": 0,
                            "refused": found["refused"], "chosen": None}
            continue
        dyn, words, name, _nest, _prog = found["ranked"][0]
        out(f"MAPSPACE {tag}: CHOSEN {name} ({dyn} fetches, {words} words)")
        summary[tag] = {"total": found["total"], "encodable": found["encodable"],
                        "refused": found["refused"], "chosen": name,
                        "fetches": dyn, "words": words}
    return summary
