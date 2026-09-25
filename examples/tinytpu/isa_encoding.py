# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GENERATED from isa_spec.json by gen_isa.py -- DO NOT EDIT.

The instruction set as Python: the constants, the encoder, a decoder that
names operands the way the spec names them, the sequencer's control-flow
and address resolution, the instruction-memory header, and the numerics of
the active arithmetic configuration.

Edit isa_spec.json and run `python gen_isa.py --write`. `--check`
regenerates this file and fails on any difference, so drift is caught
rather than discovered.
"""

import os

import numpy as np

from allo.actions import (
    Action as _A, Contract as _Contract, Instruction as _Instruction,
    Machine as _Machine, Port as _Port, State as _State, Unit as _Unit)

_MACHINE = None

# ---------------------------------------------------------- parameters ---
#: Each parameter is selected by its own environment variable, and its
#: default must be the one the design uses -- `gen_isa.py --check`
#: re-imports both modules under several environments and compares.
T = int(os.environ.get("TPU_T", 4))   # SIMD width, and the array dimension: the array is T*T processing elements
MAXDIM = int(os.environ.get("TPU_MAXDIM", 64))   # largest M, K, N supported by one build
QD = int(os.environ.get("TPU_QD", 16))   # stream depth on every point-to-point channel

VW = T * 8   # packed operand word: T operand lanes
AW = T * 32   # packed accumulator word: T accumulator lanes
WPR = MAXDIM // T   # packed words per DRAM row; the legal range of col_block
OPERAND_ROWS = (MAXDIM // T) * MAXDIM   # the highest operand row the shipped GEMM's layout names; what the operand memories are sized from
TEST_WINDOW = 64   # the fixed row window the stress harness's fuzz programs address, independently of MAXDIM; a floor under the operand memories
BUS_BYTES = 64   # the m_axi beat the build aligns to (align_value 64)
MAXROWS = 127   # the largest nr the 8-bit field admits under the encoding rule

NHDR = 8
IWORDS = 2
LOOP_DEPTH = 4
AGU_TERMS = 3
AR_RAW_DIST = 4
SPAD_ROWS = int(os.environ.get("TPU_SPAD", max(TEST_WINDOW, OPERAND_ROWS)))   # scratchpad depth in packed words
NVR = int(os.environ.get("TPU_NVR", max(TEST_WINDOW, OPERAND_ROWS)))   # operand vector registers
NAR = int(os.environ.get("TPU_NAR", max(128, TEST_WINDOW, 2 * MAXDIM + 8)))   # accumulator vector registers
IMEM_SIZE = int(os.environ.get("TPU_IMEM", NHDR + IWORDS * 24))   # instruction memory depth in 64-bit words
DMA_WORDS = int(os.environ.get("TPU_DMA_WORDS", min(WPR, max(1, BUS_BYTES // (VW // 8))) if os.environ.get("TPU_DMA_WIDEN") == "1" else 1))   # packed words the operand burst moves per loop iteration

#: (name, minimum, maximum or None) per build parameter, from the spec.
PARAMETER_RANGE = {
    "T": (4, None),
    "MAXDIM": (4, None),
    "SPAD_ROWS": (1, 2048),
    "NVR": (1, 2048),
    "NAR": (1, 2048),
    "IMEM_SIZE": (8, None),
    "QD": (2, None),
    "DMA_WORDS": (1, None),
}
#: The same parameters BY VALUE, so that `check_parameters` reads them
#: from a name it declares rather than through `globals()`. The spec
#: policy denies `globals` -- it is one of the ways candidate code
#: reaches a module namespace it was not given -- and this module is
#: editable now (`chia_agent/design.py`), so the generated form must
#: pass the policy it is held to.
PARAMETER_VALUES = {
    "T": T,
    "MAXDIM": MAXDIM,
    "SPAD_ROWS": SPAD_ROWS,
    "NVR": NVR,
    "NAR": NAR,
    "IMEM_SIZE": IMEM_SIZE,
    "QD": QD,
    "DMA_WORDS": DMA_WORDS,
}

def check_parameters():
    """Raise if a build parameter is outside the range the spec admits,
    or breaks one of its cross-parameter constraints."""
    for name, (lo, hi) in PARAMETER_RANGE.items():
        v = PARAMETER_VALUES[name]
        if v < lo or (hi is not None and v > hi):
            raise ValueError(f"{name}={v} outside the spec range {lo}..{hi}")
    if not (T >= 4):
        raise ValueError("T >= 4: a packed operand word must hold the two 16-bit counts spm sends down wcol")
    if not (MAXDIM % T == 0):
        raise ValueError("MAXDIM % T == 0: a DRAM row must be a whole number of packed words")
    if not (OPERAND_ROWS <= 2048):
        raise ValueError("OPERAND_ROWS <= 2048: an address field carries 11 usable bits, and the layout numbers its rows 0 .. OPERAND_ROWS-1, so the highest address must be <= 2047")
    if not (AR_RAW_DIST <= T):
        raise ValueError("AR_RAW_DIST <= T: a T-row GEMM must satisfy the accumulator distance contract")
    if not (IMEM_SIZE % 8 == 0):
        raise ValueError("IMEM_SIZE % 8 == 0: the program prefetch moves 8 words per iteration")
    if not (IMEM_SIZE >= NHDR + IWORDS):
        raise ValueError("IMEM_SIZE >= NHDR + IWORDS: imem must hold the header and at least one instruction")
    if not (SPAD_ROWS <= 2048 and NVR <= 2048 and NAR <= 2048):
        raise ValueError("SPAD_ROWS <= 2048 and NVR <= 2048 and NAR <= 2048: a 12-bit address field carries 0..2047 under the encoding rule, and a memory of depth D is addressed 0..D-1, so D <= 2048 -- the same count-versus-address distinction as the addressing ceiling")
    if not (SPAD_ROWS >= OPERAND_ROWS and NVR >= OPERAND_ROWS):
        raise ValueError("SPAD_ROWS >= OPERAND_ROWS and NVR >= OPERAND_ROWS: an operand memory smaller than the layout addresses assembles and gives wrong answers")
    if not (NAR >= 2 * MAXDIM + 2):
        raise ValueError("NAR >= 2 * MAXDIM + 2: AR_C is MAXDIM rows and AR_P another MAXDIM from MAXDIM+1")
    if not (DMA_WORDS >= 1):
        raise ValueError("DMA_WORDS >= 1: the operand burst moves at least one packed word per iteration")


# ------------------------------------------------------------- opcodes ---
OP_NOP = 0
OP_DMA_LD = 1
OP_DMA_ST = 2
OP_VLD = 3
OP_MM = 4
OP_VADD = 5
OP_VRELU = 6
OP_MVOUT = 7
OP_LOOP = 8
OP_ENDLOOP = 9
OP_VADDRELU = 10

OPCODE_NAME = {
    OP_NOP: "nop",
    OP_DMA_LD: "dma_ld",
    OP_DMA_ST: "dma_st",
    OP_VLD: "vld",
    OP_MM: "mm",
    OP_VADD: "vadd",
    OP_VRELU: "vrelu",
    OP_MVOUT: "mvout",
    OP_LOOP: "loop",
    OP_ENDLOOP: "endloop",
    OP_VADDRELU: "vaddrelu",
}
RETIRED = frozenset({OP_DMA_ST})

#: The operand each field carries, per opcode, in f0..f3 order. `None`
#: means the opcode does not use that field.
OPERAND_NAME = {
    OP_NOP: (None, None, None, None),
    OP_DMA_LD: ("mode", "dram_row0", "col_block", "dst_row0"),
    OP_DMA_ST: (None, None, None, None),
    OP_VLD: ("vr0", "spad0", None, None),
    OP_MM: ("vr_a", "ar0", "acc", "spad_w"),
    OP_VADD: ("ar_d", "ar_s1", "ar_s2", None),
    OP_VRELU: ("ar_d", "ar_s", None, None),
    OP_MVOUT: ("ar0", "dram_row0", "col_block", None),
    OP_LOOP: (None, None, None, None),
    OP_ENDLOOP: (None, None, None, None),
    OP_VADDRELU: ("ar_d", "ar_s1", "ar_s2", None),
}

#: Values `op` field of an opcode admits, where the spec restricts them.
LEGAL_VALUES = {
    (OP_DMA_LD, "mode"): (0, 1, 2, 3),
    (OP_MM, "acc"): (0, 1),
}

DMA_SRC_B = 1   # 0 = source A, 1 = source B
DMA_TO_VR = 2   # 0 = destination spad, 2 = destination vr


# ------------------------------------------------------------ the word ---
#: (name, lo bit, width) for every field of instruction word 0.
FIELDS = (
    ("op", 0, 6),
    ("f0", 6, 12),
    ("f1", 18, 12),
    ("f2", 30, 12),
    ("f3", 42, 12),
    ("nr", 54, 8),
)
WORD_BITS = 64
SPARE_BITS = (62, 63)

#: (name, lo bit, width) within one 19-bit AGU term.
AGU_SUBFIELDS = (
    ("target", 0, 4),
    ("level", 4, 3),
    ("stride", 7, 12),
)
AGU_TERM_BITS = 19

AGU_F0 = 1
AGU_F1 = 2
AGU_F2 = 3
AGU_F3 = 4
AGU_TARGET_FIELD = {0: None, 1: 'f0', 2: 'f1', 3: 'f2', 4: 'f3'}


def usable_max(width):
    """The spec's encoding rule: an N-bit field safely carries
    0 .. 2^(N-1) - 1, because the top bit was once a sign bit after
    extraction. See isa_spec.json, "encoding_rule"."""
    return 2 ** (width - 1) - 1


def _get(word, lo, width):
    return (int(word) >> lo) & ((1 << width) - 1)


def encode(op, f0=0, f1=0, f2=0, f3=0, nr=0):
    """Instruction word 0. Refuses a field outside the encoding rule."""
    vals = {"op": op, "f0": f0, "f1": f1, "f2": f2, "f3": f3, "nr": nr}
    word = 0
    for name, lo, width in FIELDS:
        v = int(vals[name])
        if not 0 <= v <= usable_max(width):
            raise ValueError(
                f"{name}={v} does not fit {width - 1} usable bits of a "
                f"{width}-bit field (the spec's encoding rule)")
        word |= v << lo
    return word


def encode_agu(*terms):
    """Instruction word 1: up to AGU_TERMS (target, level, stride)."""
    if len(terms) > AGU_TERMS:
        raise ValueError(f"at most {AGU_TERMS} address terms")
    word = 0
    for i, (target, level, stride) in enumerate(terms):
        if target not in AGU_TARGET_FIELD:
            raise ValueError(f"AGU target {target} is not a field")
        if not 0 <= level < LOOP_DEPTH:
            raise ValueError(f"AGU level {level} outside the loop stack")
        vals = {"target": target, "level": level, "stride": int(stride)}
        for name, lo, width in AGU_SUBFIELDS:
            v = vals[name]
            if not 0 <= v <= usable_max(width):
                raise ValueError(
                    f"AGU {name}={v} does not fit {width - 1} usable bits")
            word |= v << (AGU_TERM_BITS * i + lo)
    return word


def decode(w0):
    """-> (op, nr, f0, f1, f2, f3), the raw fields of word 0."""
    v = {name: _get(w0, lo, width) for name, lo, width in FIELDS}
    return v["op"], v["nr"], v["f0"], v["f1"], v["f2"], v["f3"]


def decode_agu(w1):
    """-> [(target, level, stride)] for every term, unused ones included."""
    out = []
    for i in range(AGU_TERMS):
        base = AGU_TERM_BITS * i
        out.append(tuple(_get(w1, base + lo, width)
                         for _, lo, width in AGU_SUBFIELDS))
    return out


def operands(op, f0, f1, f2, f3, nr):
    """The resolved fields of one instruction, BY THE SPEC'S NAMES.

    This is how a consumer reads an instruction without knowing a bit
    position or which `fN` an operand happens to live in: `operands(...)
    ["spad_w"]`, not `f3`. Reassigning an operand to another field in
    isa_spec.json moves every such reader with it."""
    d = {"rows": nr, "nr": nr}
    for name, v in zip(OPERAND_NAME[op], (f0, f1, f2, f3)):
        if name is not None:
            d[name] = v
    return d


def dma_source_is_b(mode):
    """dma_ld mode bit 0: the source matrix."""
    return bool(mode & DMA_SRC_B)


def dma_dest_is_vr(mode):
    """dma_ld mode bit 1: the destination memory."""
    return bool(mode & DMA_TO_VR)


#: (name, predicate over (MAXDIM, T), the question it answers) per
#: encoding ceiling on MAXDIM. See isa_spec.json, "maxdim_ceilings".
MAXDIM_CEILINGS = {
    "addressing": (lambda m, t: (m // t) * m <= 2048,
     "How large a MAXDIM can the shipped GEMM's operand layout ADDRESS? A property of the layout and the address field, independent of the shape being run.",
     "an address field carries 11 usable bits (encoding_rule), so the highest operand ADDRESS must be <= 2047. The layout names OPERAND_ROWS rows, numbered 0 .. OPERAND_ROWS-1, so the bound is OPERAND_ROWS <= 2048 -- the design's import-time assert says the same thing. Writing it as OPERAND_ROWS <= 2047 is off by one and understates the ceiling at T=8 (120 instead of 128)."),
    "cubic_header": (lambda m, t: m ** 3 // t ** 2 + m ** 2 // t <= 32767,
     "How large a MAXDIM can a CUBIC GEMM's header count promise accu? A property of the WORKLOAD as well as the encoding -- a non-cubic shape gives a different count, so this ceiling moves with the program.",
     "a header count is read back through a 15-bit slice (imem.count_slice_width), and accu's iteration count for an MxMxM GEMM is M**3/T**2 + M**2/T"),
}


def maxdim_ceiling(name, t=None, limit=1 << 14):
    """The largest MAXDIM this encoding admits under one ceiling.

    COMPUTED, not typed, and the search runs over MULTIPLES OF T,
    because `MAXDIM % T == 0` is an assertion of this ISA. Solving
    either inequality over the reals gives a number no build can use:
    that is how `MAXDIM <= 90` came to be written in three documents
    when the answer at T=4 is 88.

    The two ceilings answer different questions and neither corrects
    the other -- `MAXDIM_CEILINGS[name][1]` says which."""
    t = T if t is None else t
    ok = MAXDIM_CEILINGS[name][0]
    best = None
    for m in range(t, limit + 1, t):
        if ok(m, t):
            best = m
    return best


#: {name: (need(M,K,N,T), allowed(M,K,N,T), rule)} per program limit.
PROGRAM_LIMITS = {
    "build_extent": (lambda M, K, N, T: max(M, K, N),
     lambda M, K, N, T: MAXDIM,
     "No dimension may exceed the build's MAXDIM."),
    "rows_per_instruction.shipped_layout": (lambda M, K, N, T: max(M, K),
     lambda M, K, N, T: MAXROWS,
     'The shipped tiled-GEMM layout issues one dma_ld of K rows per B column block and one mm of M rows per tile, so both M and K travel in `nr`.'),
    "rows_per_instruction.split_operand_load": (lambda M, K, N, T: M,
     lambda M, K, N, T: MAXROWS,
     'Splitting the B load into several dma_lds removes K from the row count; only M still travels in `nr`. The instruction set permits this -- the shipped generator simply does not emit it.'),
    "header_counts.accu": (lambda M, K, N, T: (K // T) * M * (N // T) + M * (N // T),
     lambda M, K, N, T: 32767,
     "accu's per-unit work count is read back through a 15-bit slice. It is the largest header count, and it depends on the SHAPE, not only on MAXDIM."),
}


def gemm_limits(M, K, N):
    """Which program limit refuses an M x K x N GEMM on this build, and
    BY HOW MUCH.

    -> [(name, need, allowed, over)] with `over` positive where the
    limit refuses. The margin is the point: "short by one in nr" names
    a design target, "cannot express" does not. Note that a limit named
    `.shipped_layout` is a property of the layout the generator emits,
    not of the instruction set -- compare it with the
    `.split_operand_load` row before concluding a shape is
    inexpressible."""
    out = []
    for name, (need, allowed, _) in PROGRAM_LIMITS.items():
        n, a = need(M, K, N, T), allowed(M, K, N, T)
        out.append((name, n, a, max(0, n - a)))
    return out


def agu_reach(base, stride, trip):
    """Every value a field takes over a loop of `trip` iterations.

    An AGU term is ADDITIVELY MONOTONE: the sequencer adds
    `iv[level] * stride` to the field and writes the sum back, with no
    predication, no saturation and no wrap. So the values a field takes
    over a loop are an arithmetic progression, and nothing can bend it
    back inside a bound. See isa_spec.json, "derived_properties"."""
    return [base + i * stride for i in range(trip)]


def agu_legal_trip(op, operand, base=0, stride=1, limit=1 << 12):
    """The largest trip count for which an AGU term on `operand` of `op`
    stays inside the values the spec admits for that operand.

    `None` means the operand has no restricted value set, so only the
    encoding rule bounds it. Otherwise the answer follows from
    `agu_reach` alone: a field whose legal values are a finite set is
    drivable from a loop for exactly as long as the progression stays
    inside that set. It is NOT a question of whether the field may be an
    AGU target -- every operand field may (AGU_TARGET_FIELD), and the
    sequencer does not know which fields have restricted values."""
    legal = LEGAL_VALUES.get((op, operand))
    if legal is None:
        return None
    t = 0
    while t < limit and base + t * stride in legal:
        t += 1
    return t


def agu_terms_used(w1):
    """How many of the AGU_TERMS an instruction word 1 already spends."""
    return sum(1 for target, _, _ in decode_agu(w1) if target)


# ----------------------------------------------- control flow and AGU ---
def trace(prog):
    """Run the program's control flow and address generation exactly as
    the spec says the sequencer does, yielding one
    `(pc, ivs, op, nr, f0, f1, f2, f3)` per instruction ISSUED.

    The loop stack is a LIFO of {body_start, iv, trip} 4 levels deep and
    the back edge is tested AFTER the body; each AGU term with a nonzero
    target adds `iv[level] * stride` to the field it names, in term
    order. Derived from isa_spec.json, not from the design."""
    pc = 0
    stack = []
    iv_now = [0] * LOOP_DEPTH
    guard = 0
    while pc < len(prog):
        guard += 1
        if guard > 1 << 22:
            raise AssertionError("program does not terminate")
        w0, w1 = prog[pc]
        op, nr, f0, f1, f2, f3 = decode(w0)
        if op == OP_LOOP:
            iv_now[len(stack)] = 0
            stack.append([pc + 1, 0, nr])
            pc += 1
        elif op == OP_ENDLOOP:
            fr = stack[-1]
            fr[1] += 1
            if fr[1] < fr[2]:
                iv_now[len(stack) - 1] = fr[1]
                pc = fr[0]
            else:
                stack.pop()
                pc += 1
        else:
            f = [f0, f1, f2, f3]
            for target, level, stride in decode_agu(w1):
                if target:
                    f[target - 1] += iv_now[level] * stride
            yield (pc, tuple(iv_now[:len(stack)]), op, nr, *f)
            pc += 1


def expand(prog):
    """`trace` without the provenance: one `(op, nr, f0, f1, f2, f3)` per
    dynamic issue."""
    return [e[2:] for e in trace(prog)]


# ------------------------------------------------- units and actions ---
#: Each unit as its ports and its step rate, isa_spec.json "units".
UNITS = (
    ('sequencer', 1, True, (('fetch', 1), ('dispatch', 5))),
    ('dma_ld', 1, True, (('dram.read', 1), ('mux', 1), ('dma2sp', 1), ('dma2vr', 1))),
    ('spm', 1, True, (('spad.read', 1), ('spad.write', 1), ('dma2sp', 1), ('sp2vr', 1), ('wcol', 1))),
    ('vru', 1, True, (('vr.read', 1), ('vr.write', 1), ('acol', 1), ('sp2vr', 1), ('dma2vr', 1))),
    ('array', 1, True, (('wcol', 1), ('acol', 1), ('instructions', 1), ('mac', 1), ('cw', 1))),
    ('accu', 1, True, (('cw', 1), ('ar.read', 1), ('ar.write', 1), ('alu', 2), ('ac2sp', 1))),
    ('dma_st', 1, True, (('ac2sp', 1), ('dram.write', 1))),
)

#: Each memory as the model sees it: depth, lanes, ports, bank map.
STATES = (
    ('spad', 'SPAD_ROWS', 'T', 'spm', 1, 1, 'defined'),
    ('vr', 'NVR', 'T', 'vru', 1, 1, 'defined'),
    ('ar', 'NAR', 'T', 'accu', 1, 1, 'defined'),
    ('imem', 'IMEM_SIZE', None, 'sequencer', 1, 1, 'defined'),
    ('A', 'MAXDIM * MAXDIM', None, 'dma_ld', 1, 1, 'defined'),
    ('B', 'MAXDIM * MAXDIM', None, 'dma_ld', 1, 1, 'defined'),
    ('C', 'MAXDIM * MAXDIM', None, 'dma_st', 1, 1, 'defined'),
)

#: Every instruction as the per-unit effects it composes. THIS IS THE
#: ONE STATEMENT: the units an opcode reaches, the rows it reads before
#: it writes, and every per-unit work count in the header below are
#: queries over this table, not separate declarations.
ACTIONS = {
    0: (
    ),
    1: (
        _A('dma_ld', 'read', port='dram.read', state='A', base='dram_row0', when='(mode & 1) == 0', into='from_a', role='the source rows', offset='col_block * T'),
        _A('dma_ld', 'read', port='dram.read', state='B', base='dram_row0', when='(mode & 1) == 1', into='from_b', role='the source rows', offset='col_block * T'),
        _A('dma_ld', 'compute', port='mux', compute='select', args=('from_a', 'from_b'), into='beat'),
        _A('dma_ld', 'emit', port='dma2sp', args=('beat',), when='(mode & 2) == 0'),
        _A('dma_ld', 'emit', port='dma2vr', args=('beat',), when='(mode & 2) == 2'),
        _A('spm', 'receive', port='dma2sp', into='landing', when='(mode & 2) == 0', args=('beat',)),
        _A('spm', 'write', port='spad.write', state='spad', base='dst_row0', args=('landing',), when='(mode & 2) == 0'),
        _A('vru', 'receive', port='dma2vr', into='landing', when='(mode & 2) == 2', args=('beat',)),
        _A('vru', 'write', port='vr.write', state='vr', base='dst_row0', args=('landing',), when='(mode & 2) == 2'),
    ),
    2: (
    ),
    3: (
        _A('spm', 'read', port='spad.read', state='spad', base='spad0', into='word', role='the row to copy'),
        _A('spm', 'emit', port='sp2vr', args=('word',)),
        _A('vru', 'receive', port='sp2vr', into='word_in', args=('word',)),
        _A('vru', 'write', port='vr.write', state='vr', base='vr0', args=('word_in',)),
    ),
    4: (
        _A('spm', 'read', port='spad.read', state='spad', base='spad_w', count='T', per='instruction', into='weights', role='weights'),
        _A('spm', 'emit', port='wcol', count='T + 1', per='instruction', args=('weights',)),
        _A('array', 'receive', port='wcol', count='T + 1', per='instruction', into='weights_in', args=('weights',)),
        _A('array', 'emit', port='instructions', per='instruction', args=('weights_in',)),
        _A('vru', 'read', port='vr.read', state='vr', base='vr_a', into='activation', role='activations'),
        _A('vru', 'emit', port='acol', args=('activation',)),
        _A('array', 'receive', port='acol', into='activation_in', args=('activation',)),
        _A('array', 'compute', port='mac', compute='matmul', args=('activation_in', 'weights_in'), into='psum'),
        _A('array', 'emit', port='cw', args=('psum',)),
        _A('accu', 'receive', port='cw', into='psum_in', args=('psum',)),
        _A('accu', 'read', port='ar.read', state='ar', base='ar0', when='acc == 1', into='carried', role='the accumulate base'),
        _A('accu', 'compute', port='alu', compute='acc_add', args=('carried', 'psum_in'), into='total'),
        _A('accu', 'write', port='ar.write', state='ar', base='ar0', args=('total',)),
    ),
    5: (
        _A('accu', 'read', port='ar.read', state='ar', base='ar_s1', into='left', role='a source'),
        _A('accu', 'read', port='ar.read', state='ar', base='ar_s2', into='right', role='a source'),
        _A('accu', 'compute', port='alu', compute='add', args=('left', 'right'), into='total'),
        _A('accu', 'write', port='ar.write', state='ar', base='ar_d', args=('total',)),
    ),
    6: (
        _A('accu', 'read', port='ar.read', state='ar', base='ar_s', into='before', role='a source'),
        _A('accu', 'compute', port='alu', compute='max0', args=('before',), into='after'),
        _A('accu', 'write', port='ar.write', state='ar', base='ar_d', args=('after',)),
    ),
    7: (
        _A('accu', 'read', port='ar.read', state='ar', base='ar0', into='value', role='the value to retire'),
        _A('accu', 'compute', port='alu', compute='to_operand', args=('value',), into='clipped'),
        _A('accu', 'emit', port='ac2sp', args=('clipped',)),
        _A('dma_st', 'receive', port='ac2sp', into='clipped_in', args=('clipped',)),
        _A('dma_st', 'write', port='dram.write', state='C', base='dram_row0', args=('clipped_in',), offset='col_block * T'),
    ),
    8: (
        _A('sequencer', 'compute', port='fetch', compute='push_loop', per='instruction', into='frame'),
    ),
    9: (
        _A('sequencer', 'compute', port='fetch', compute='pop_loop', per='instruction', into='frame'),
    ),
    10: (
        _A('accu', 'read', port='ar.read', state='ar', base='ar_s1', into='left', role='a source'),
        _A('accu', 'read', port='ar.read', state='ar', base='ar_s2', into='right', role='a source'),
        _A('accu', 'compute', port='alu', compute='add', args=('left', 'right'), into='total'),
        _A('accu', 'compute', port='alu', compute='max0', args=('total',), into='rectified'),
        _A('accu', 'write', port='ar.write', state='ar', base='ar_d', args=('rectified',)),
    ),
}

#: How many rows one issue of each opcode runs.
ROWS_EXPRESSION = {
    0: '1',
    1: 'nr',
    2: '1',
    3: 'nr',
    4: 'nr',
    5: 'nr',
    6: 'nr',
    7: 'nr',
    8: '1',
    9: '1',
    10: 'nr',
}

#: Properties of PROGRAMS that no instruction can establish on its own.
#: The model reports them as obligations rather than forgetting them.
CONTRACTS = (
    ('write_before_read', {"rule": 'Every ar row read by an accumulating mm, by either source of a vadd, by a vrelu source or by an mvout must have been written earlier in the SAME program, by an overwriting mm, a vadd or a vrelu. Every vr row an mm reads as activations, and every spad row it reads as weights, must hold data a dma_ld put there -- directly, or into spad and then through a vld. vld is a pure copy and may copy an unwritten spad row; the copy is then unwritten too, and consuming it in an mm is an error. nr >= 1 on every data op: a unit fetches an instruction whenever its row counter runs out, so a zero-row instruction is fetched as if it had one row. It desynchronises the unit; it is not a no-op. Every resolved field must be within 0 .. 2047, the range the encoding rule admits.', "enforced_by": 'microarch_isa.check_program, which microarch_isa.assemble calls, so a violating program cannot be assembled'}),
    ('accumulator_raw_distance', {"rule": 'A read of an ar row must come at least AR_RAW_DIST accu iterations after the write it depends on.', "enforced_by": 'microarch_isa.check_program'}),
)


def machine():
    """This ISA as an `allo.actions.Machine`, built once.

    The model is machine-independent and lives in `allo/actions.py`;
    everything specific to this ISA is the three tables above, which
    `gen_isa.py` writes out of `isa_spec.json`."""
    global _MACHINE
    if _MACHINE is None:
        _MACHINE = _Machine(
            name="TinyTPU-isa",
            units=tuple(
                _Unit(n, ports=tuple(_Port(p, physical=w)
                                     for p, w in ports),
                      ii=ii, elastic=el)
                for n, ii, el, ports in UNITS),
            states=tuple(
                _State(n, rows=depth, lanes=lanes, owner=owner,
                       read_ports=rp, write_ports=wp, collision=col)
                for n, depth, lanes, owner, rp, wp, col in STATES),
            instructions=tuple(
                _Instruction(OPCODE_NAME[op], actions=acts,
                             rows=ROWS_EXPRESSION[op])
                for op, acts in ACTIONS.items()),
            parameters={"T": T, "MAXDIM": MAXDIM, "SPAD_ROWS": SPAD_ROWS,
                        "NVR": NVR, "NAR": NAR, "IMEM_SIZE": IMEM_SIZE,
                        "AR_RAW_DIST": AR_RAW_DIST},
            arithmetic="exact" if PRODUCT_EXACT else "rounding",
            contracts=tuple(
                _Contract(name, c["rule"] if "rule" in c
                          else "; ".join(c.get("rules", ())),
                          discharged_by=c.get("enforced_by"))
                for name, c in CONTRACTS),
        )
    return _MACHINE


def units_of(op):
    """Which units an opcode reaches. DERIVED, and the dispatch table
    in the sequencer is held to it."""
    return machine().units_of(OPCODE_NAME[op])


def effects(op, f0, f1, f2, f3, nr):
    """Every resolved effect of one issue: which unit, which cycle,
    which row of which memory, and what value. A validator and a
    reference model are both walks over this."""
    return machine().effects(OPCODE_NAME[op],
                             operands(op, f0, f1, f2, f3, nr))


def work(unit, op, f0, f1, f2, f3, nr):
    """The steps one unit spends on one issue. What a header count sums
    and what the sequencer rewrites `nr` to."""
    return machine().work(unit, OPCODE_NAME[op],
                          operands(op, f0, f1, f2, f3, nr))


def dispatch_rewrites():
    """Where a unit's own work count differs from the instruction's row
    count, so the sequencer has to hand it a rewritten `nr`. DERIVED:
    `spm` taking T+1 on an `mm` and `accu` taking 2*nr on a `vadd` are
    consequences of the ports, not entries in a table."""
    out = []
    for op, acts in ACTIONS.items():
        if not acts:
            continue
        names = [n for n in OPERAND_NAME[op] if n]
        probe = dict.fromkeys(names, 0)
        probe.update({"nr": 3, "acc": 1, "mode": 0})
        for unit in units_of(op):
            steps = machine().work(unit, OPCODE_NAME[op], probe)
            if steps and steps != probe["nr"]:
                out.append((OPCODE_NAME[op], unit, steps, probe["nr"]))
    return tuple(out)


# -------------------------------------------------------- imem header ---
#: What each header word counts, isa_spec.json "imem".entries[].work.
HEADER_WORK = (
    (0, (0, 16), "n_instr", {'kind': 'static_instructions'}),
    (1, (0, 16), "dma_ld_rows", {'unit': 'dma_ld'}),
    (2, (0, 16), "spm_rows", {'unit': 'spm'}),
    (3, (0, 16), "vru_words", {'unit': 'vru'}),
    (4, (0, 16), "mm_count", {'unit': 'array', 'port': 'instructions'}),
    (4, (16, 32), "mm_rows", {'unit': 'array', 'port': 'mac'}),
    (5, (0, 16), "accu_iterations", {'unit': 'accu'}),
    (6, (0, 16), "dma_st_rows", {'unit': 'dma_st'}),
    (7, (0, 16), "a_span", {'kind': 'row_span', 'state': 'A'}),
    (7, (16, 32), "b_span", {'kind': 'row_span', 'state': 'B'}),
)


def header(prog):
    """The NHDR header words of `imem`, computed from the ACTIONS.

    Every count but the static instruction count is DYNAMIC: it is a sum
    over the issues `trace` produces, because each unit loops over the
    work it is really sent. A unit promised the wrong number hangs.

    Nothing here knows that `spm` charges an `mm` T+1 iterations or that
    `accu` charges a `vadd` two steps a row. Those were sentences in the
    spec until the units declared their ports; now they are what the
    model computes from one `mm` and one `vadd`."""
    m = machine()
    ev = list(expand(prog))
    words = [0] * NHDR
    for index, (lo, hi), name, job in HEADER_WORK:
        total = 0
        if job.get("kind") == "static_instructions":
            total = len(prog)
        else:
            for op, nr, f0, f1, f2, f3 in ev:
                i = operands(op, f0, f1, f2, f3, nr)
                what = OPCODE_NAME[op]
                if job.get("kind") == "row_span":
                    total = max(total, m.row_span(job["state"], what, i))
                elif "port" in job:
                    total += m.items(job["unit"], job["port"], what, i)
                else:
                    total += m.work(job["unit"], what, i)
        if not 0 <= total <= usable_max(hi - lo):
            raise ValueError(
                f"header {name}={total} does not fit {hi - lo - 1} usable "
                f"bits of its {hi - lo}-bit slice")
        words[index] |= total << lo
    return words


def image(prog):
    """The whole instruction-memory image: header, then two words each."""
    words = header(prog)
    for w0, w1 in prog:
        words.append(int(w0))
        words.append(int(w1))
    if len(words) > IMEM_SIZE:
        raise ValueError(f"{len(words)} words > IMEM_SIZE={IMEM_SIZE}")
    return words


# ----------------------------------------------------------- numerics ---
#: The active arithmetic configuration, isa_spec.json "numerics".
NUMERICS = 'int8'
OPERAND_BITS = 8
OPERAND_MIN = -128
OPERAND_MAX = 127
OPERAND_DTYPE = np.int8
ACC_BITS = 32
ACC_MIN = -2147483648
ACC_MAX = 2147483647
ACC_DTYPE = np.int32
PRODUCT_EXACT = True
ACC_OVERFLOW = "wraparound, two's complement, at 32 bits"
OUTPUT_CONVERSION = 'saturate'


def acc(x):
    """A value in the accumulator format: wraparound, two's complement, at 32 bits."""
    m = 1 << ACC_BITS
    return ((np.asarray(x, np.int64) - ACC_MIN) % m + ACC_MIN).astype(np.int64)


def product(a, b):
    """One operand-by-operand product, in the accumulator format.

    Exact: int8 x int8 is bounded by 128*128 = 16384, so a 16-bit product is exact and the multiplier stays narrow."""
    return acc(np.asarray(a, np.int64) * np.asarray(b, np.int64))


def to_operand(x):
    """The accumulator-to-operand conversion `mvout` applies: saturate to
    -128 .. 127, no rounding."""
    return np.clip(x, OPERAND_MIN, OPERAND_MAX)

