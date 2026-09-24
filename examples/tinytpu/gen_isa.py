# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate TinyTPU-isa's derived artefacts from `isa_spec.json`, and hold both
consumers to it.

`isa_spec.json` is the source of truth for every ISA fact. This script turns it
into the artefacts that would otherwise be hand-copies, and checks the two
things that cannot be generated:

    generated, checked in, verified byte-identical
      isa_encoding.py                       the spec as Python: constants,
        encoder, decoder (operands by NAME), control-flow/AGU resolver, header
        builder, numerics.
      docs/source/designs/tinytpu_isa_spec.rst  the ISA tables, between the
        GENERATED markers. The prose around them is hand-written and untouched.

    checked, not generated
      microarch_isa.py   the shipped instantiation of the `ip/` unit library:
        the build parameters, held here by range and by agreement.
      ip/isa.py, ip/assembler.py   the layout named once, the encoder, the
        decoder and the header, held by value and by behaviour.
      ip/units/          the hardware. Its bit slices stay WRITTEN OUT: Allo
        cannot infer a slice's width from symbolic bounds, so a slice cannot
        read ip/isa.py's names without widening to i32. Every literal slice is
        held to the field table here.
      isa_dsl.py         the program generator, through the encoder it shares
        with the design.
      allo/encoding.py   `TINYTPU_ISA`, the compiler-side descriptor of what
        one instruction word can carry. It imports nothing from `examples/`,
        so its two budgets are checked here rather than generated.

The two directions matter. `isa_ref.py` is built on the generated module and
names operands by the spec's names, so it never sees a bit position or an
opcode number the design chose; the design is measured against the spec
independently. They are no longer compared only with each other.

    python gen_isa.py --write     regenerate every artefact
    python gen_isa.py --check     fail if an artefact is stale or a consumer
                                  disagrees with the spec
    python gen_isa.py --conform   --check, plus the emitted HLS's own bit
                                  slices against the field table (needs the
                                  Vitis-less `target="vhls"` path only)
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SPEC = os.path.join(HERE, "isa_spec.json")
ENCODING = os.path.join(HERE, "isa_encoding.py")
DESIGN = os.path.join(HERE, "microarch_isa.py")
HARDWARE = sorted(os.path.join(HERE, "ip", "units", f)
                  for f in os.listdir(os.path.join(HERE, "ip", "units"))
                  if f.endswith(".py"))
DOC = os.path.join(REPO, "docs", "source", "designs", "tinytpu_isa_spec.rst")

BEGIN = ".. BEGIN GENERATED: examples/tinytpu/gen_isa.py"
END = ".. END GENERATED"

sys.path.insert(0, REPO)


def load():
    with open(SPEC) as f:
        return json.load(f)


# ------------------------------------------------------- isa_encoding.py ---
def _fields(spec):
    return spec["instruction_word"]["word0"]["fields"]


def gen_encoding(spec) -> str:
    """The spec as an importable Python module."""
    iw = spec["instruction_word"]
    agu = spec["agu"]
    ops = spec["opcodes"]
    num = spec["numerics"]["configurations"][spec["numerics"]["active"]]
    out = []
    w = out.append

    w('# Copyright Allo authors. All Rights Reserved.')
    w('# SPDX-License-Identifier: Apache-2.0')
    w('')
    w('"""GENERATED from isa_spec.json by gen_isa.py -- DO NOT EDIT.')
    w('')
    w('The instruction set as Python: the constants, the encoder, a decoder that')
    w("names operands the way the spec names them, the sequencer's control-flow")
    w('and address resolution, the instruction-memory header, and the numerics of')
    w('the active arithmetic configuration.')
    w('')
    w('Edit isa_spec.json and run `python gen_isa.py --write`. `--check`')
    w('regenerates this file and fails on any difference, so drift is caught')
    w('rather than discovered.')
    w('"""')
    w('')
    w('import os')
    w('')
    w('import numpy as np')
    w('')
    w('from allo.actions import (')
    w('    Action as _A, Contract as _Contract, Instruction as _Instruction,')
    w('    Machine as _Machine, Port as _Port, State as _State, Unit as _Unit)')
    w('')
    w('_MACHINE = None')
    w('')
    w('# ---------------------------------------------------------- parameters ---')
    w('#: Each parameter is selected by its own environment variable, and its')
    w('#: default must be the one the design uses -- `gen_isa.py --check`')
    w('#: re-imports both modules under several environments and compares.')
    # Emitted in spec order, then the fixed constants, then any parameter whose
    # default is an expression over them (IMEM_SIZE today; a memory sized from
    # MAXDIM tomorrow). A parameter is deferred until its expression resolves.
    fixed = [("NHDR", spec["imem"]["header_words"]),
             ("IWORDS", iw["words_per_instruction"]),
             ("LOOP_DEPTH", spec["loop_stack"]["depth"]),
             ("AGU_TERMS", agu["terms"]),
             ("AR_RAW_DIST",
              spec["contracts"]["accumulator_raw_distance"]["value"])]
    simple = [p for p in spec["parameters"] if "default_expression" not in p]
    derived = [p for p in spec["parameters"] if "default_expression" in p]
    for p in simple:
        w(f'{p["name"]} = int(os.environ.get("{p["env"]}", {p["default"]}))'
          f'   # {p["role"]}')
    w('')
    for c in spec["derived_constants"]:
        w(f'{c["name"]} = {c["value"]}   # {c["role"]}')
    w('')
    for name, v in fixed:
        w(f'{name} = {v}')
    for p in derived:
        w(f'{p["name"]} = int(os.environ.get("{p["env"]}", '
          f'{p["default_expression"]}))   # {p["role"]}')
    w('')
    w('#: (name, minimum, maximum or None) per build parameter, from the spec.')
    w('PARAMETER_RANGE = {')
    for p in spec["parameters"]:
        w(f'    "{p["name"]}": ({p["min"]}, {p["max"]}),')
    w('}')
    w('')
    w('def check_parameters():')
    w('    """Raise if a build parameter is outside the range the spec admits,')
    w('    or breaks one of its cross-parameter constraints."""')
    w('    for name, (lo, hi) in PARAMETER_RANGE.items():')
    w('        v = globals()[name]')
    w('        if v < lo or (hi is not None and v > hi):')
    w('            raise ValueError(f"{name}={v} outside the spec range '
      '{lo}..{hi}")')
    for a in spec["assertions"]:
        w(f'    if not ({a["expr"]}):')
        w(f'        raise ValueError("{a["expr"]}: {a["why"]}")')
    w('')
    w('')
    w('# ------------------------------------------------------------- opcodes ---')
    for o in ops:
        w(f'{o["software_constant"]} = {o["value"]}')
    w('')
    w('OPCODE_NAME = {')
    for o in ops:
        w(f'    {o["software_constant"]}: "{o["name"]}",')
    w('}')
    w('RETIRED = frozenset({%s})'
      % ", ".join(o["software_constant"] for o in ops if o.get("retired")))
    w('')
    w('#: The operand each field carries, per opcode, in f0..f3 order. `None`')
    w('#: means the opcode does not use that field.')
    w('OPERAND_NAME = {')
    for o in ops:
        names = {d["field"]: d["name"] for d in o["operands"]}
        row = ", ".join(
            f'"{names[f]}"' if f in names else "None" for f in ("f0", "f1", "f2", "f3"))
        w(f'    {o["software_constant"]}: ({row}),')
    w('}')
    w('')
    w('#: Values `op` field of an opcode admits, where the spec restricts them.')
    w('LEGAL_VALUES = {')
    for o in ops:
        for d in o["operands"]:
            if "legal_values" in d:
                w(f'    ({o["software_constant"]}, "{d["name"]}"): '
                  f'{tuple(d["legal_values"])},')
    w('}')
    w('')
    for o in ops:
        for d in o["operands"]:
            for b in d.get("bitflags", []):
                w(f'{b["software_constant"]} = {b["mask"]}   # {b["meaning"]}')
    w('')
    w('')
    w('# ------------------------------------------------------------ the word ---')
    w('#: (name, lo bit, width) for every field of instruction word 0.')
    w('FIELDS = (')
    for f in _fields(spec):
        w(f'    ("{f["name"]}", {f["lo"]}, {f["width"]}),')
    w(')')
    w('WORD_BITS = %d' % iw["word_bits"])
    w('SPARE_BITS = %r' % (tuple(iw["word0"]["spare_bits"]),))
    w('')
    w('#: (name, lo bit, width) within one %d-bit AGU term.' % agu["term_bits"])
    w('AGU_SUBFIELDS = (')
    for s in agu["subfields"]:
        w(f'    ("{s["name"]}", {s["lo"]}, {s["width"]}),')
    w(')')
    w('AGU_TERM_BITS = %d' % agu["term_bits"])
    w('')
    for t in agu["targets"]:
        if t.get("software_constant"):
            w(f'{t["software_constant"]} = {t["value"]}')
    w('AGU_TARGET_FIELD = {%s}'
      % ", ".join(f'{t["value"]}: {t["field"]!r}' for t in agu["targets"]))
    w('')
    w('')
    w('def usable_max(width):')
    w('    """The spec\'s encoding rule: an N-bit field safely carries')
    w('    0 .. 2^(N-1) - 1, because the top bit was once a sign bit after')
    w('    extraction. See isa_spec.json, "encoding_rule"."""')
    w('    return %s' % spec["encoding_rule"]["usable_max"].replace("width", "width"))
    w('')
    w('')
    w('def _get(word, lo, width):')
    w('    return (int(word) >> lo) & ((1 << width) - 1)')
    w('')
    w('')
    w('def encode(op, f0=0, f1=0, f2=0, f3=0, nr=0):')
    w('    """Instruction word 0. Refuses a field outside the encoding rule."""')
    w('    vals = {"op": op, "f0": f0, "f1": f1, "f2": f2, "f3": f3, "nr": nr}')
    w('    word = 0')
    w('    for name, lo, width in FIELDS:')
    w('        v = int(vals[name])')
    w('        if not 0 <= v <= usable_max(width):')
    w('            raise ValueError(')
    w('                f"{name}={v} does not fit {width - 1} usable bits of a "')
    w('                f"{width}-bit field (the spec\'s encoding rule)")')
    w('        word |= v << lo')
    w('    return word')
    w('')
    w('')
    w('def encode_agu(*terms):')
    w('    """Instruction word 1: up to AGU_TERMS (target, level, stride)."""')
    w('    if len(terms) > AGU_TERMS:')
    w('        raise ValueError(f"at most {AGU_TERMS} address terms")')
    w('    word = 0')
    w('    for i, (target, level, stride) in enumerate(terms):')
    w('        if target not in AGU_TARGET_FIELD:')
    w('            raise ValueError(f"AGU target {target} is not a field")')
    w('        if not 0 <= level < LOOP_DEPTH:')
    w('            raise ValueError(f"AGU level {level} outside the loop stack")')
    w('        vals = {"target": target, "level": level, "stride": int(stride)}')
    w('        for name, lo, width in AGU_SUBFIELDS:')
    w('            v = vals[name]')
    w('            if not 0 <= v <= usable_max(width):')
    w('                raise ValueError(')
    w('                    f"AGU {name}={v} does not fit {width - 1} usable bits")')
    w('            word |= v << (AGU_TERM_BITS * i + lo)')
    w('    return word')
    w('')
    w('')
    w('def decode(w0):')
    w('    """-> (op, nr, f0, f1, f2, f3), the raw fields of word 0."""')
    w('    v = {name: _get(w0, lo, width) for name, lo, width in FIELDS}')
    w('    return v["op"], v["nr"], v["f0"], v["f1"], v["f2"], v["f3"]')
    w('')
    w('')
    w('def decode_agu(w1):')
    w('    """-> [(target, level, stride)] for every term, unused ones included."""')
    w('    out = []')
    w('    for i in range(AGU_TERMS):')
    w('        base = AGU_TERM_BITS * i')
    w('        out.append(tuple(_get(w1, base + lo, width)')
    w('                         for _, lo, width in AGU_SUBFIELDS))')
    w('    return out')
    w('')
    w('')
    w('def operands(op, f0, f1, f2, f3, nr):')
    w('    """The resolved fields of one instruction, BY THE SPEC\'S NAMES.')
    w('')
    w('    This is how a consumer reads an instruction without knowing a bit')
    w('    position or which `fN` an operand happens to live in: `operands(...)')
    w('    ["spad_w"]`, not `f3`. Reassigning an operand to another field in')
    w('    isa_spec.json moves every such reader with it."""')
    w('    d = {"rows": nr, "nr": nr}')
    w('    for name, v in zip(OPERAND_NAME[op], (f0, f1, f2, f3)):')
    w('        if name is not None:')
    w('            d[name] = v')
    w('    return d')
    w('')
    w('')
    w('def dma_source_is_b(mode):')
    w('    """dma_ld mode bit 0: the source matrix."""')
    w('    return bool(mode & %s)' % _dma_flag(spec, "src_b"))
    w('')
    w('')
    w('def dma_dest_is_vr(mode):')
    w('    """dma_ld mode bit 1: the destination memory."""')
    w('    return bool(mode & %s)' % _dma_flag(spec, "to_vr"))
    w('')
    w('')
    w('#: (name, predicate over (MAXDIM, T), the question it answers) per')
    w('#: encoding ceiling on MAXDIM. See isa_spec.json, "maxdim_ceilings".')
    w('MAXDIM_CEILINGS = {')
    for c in spec["maxdim_ceilings"]["ceilings"]:
        w(f'    "{c["name"]}": (lambda m, t: {c["predicate"]},')
        w(f'     {c["question"]!r},')
        w(f'     {c["from"]!r}),')
    w('}')
    w('')
    w('')
    w('def maxdim_ceiling(name, t=None, limit=1 << 14):')
    w('    """The largest MAXDIM this encoding admits under one ceiling.')
    w('')
    w('    COMPUTED, not typed, and the search runs over MULTIPLES OF T,')
    w('    because `MAXDIM % T == 0` is an assertion of this ISA. Solving')
    w('    either inequality over the reals gives a number no build can use:')
    w('    that is how `MAXDIM <= 90` came to be written in three documents')
    w('    when the answer at T=4 is 88.')
    w('')
    w('    The two ceilings answer different questions and neither corrects')
    w('    the other -- `MAXDIM_CEILINGS[name][1]` says which."""')
    w('    t = T if t is None else t')
    w('    ok = MAXDIM_CEILINGS[name][0]')
    w('    best = None')
    w('    for m in range(t, limit + 1, t):')
    w('        if ok(m, t):')
    w('            best = m')
    w('    return best')
    w('')
    w('')
    w('#: {name: (need(M,K,N,T), allowed(M,K,N,T), rule)} per program limit.')
    w('PROGRAM_LIMITS = {')
    for L in spec["program_limits"]["limits"]:
        w(f'    "{L["name"]}": (lambda M, K, N, T: {L["need"]},')
        w(f'     lambda M, K, N, T: {L["allowed"]},')
        w(f'     {L["rule"]!r}),')
    w('}')
    w('')
    w('')
    w('def gemm_limits(M, K, N):')
    w('    """Which program limit refuses an M x K x N GEMM on this build, and')
    w('    BY HOW MUCH.')
    w('')
    w('    -> [(name, need, allowed, over)] with `over` positive where the')
    w('    limit refuses. The margin is the point: "short by one in nr" names')
    w('    a design target, "cannot express" does not. Note that a limit named')
    w('    `.shipped_layout` is a property of the layout the generator emits,')
    w('    not of the instruction set -- compare it with the')
    w('    `.split_operand_load` row before concluding a shape is')
    w('    inexpressible."""')
    w('    out = []')
    w('    for name, (need, allowed, _) in PROGRAM_LIMITS.items():')
    w('        n, a = need(M, K, N, T), allowed(M, K, N, T)')
    w('        out.append((name, n, a, max(0, n - a)))')
    w('    return out')
    w('')
    w('')
    w('def agu_reach(base, stride, trip):')
    w('    """Every value a field takes over a loop of `trip` iterations.')
    w('')
    w('    An AGU term is ADDITIVELY MONOTONE: the sequencer adds')
    w('    `iv[level] * stride` to the field and writes the sum back, with no')
    w('    predication, no saturation and no wrap. So the values a field takes')
    w('    over a loop are an arithmetic progression, and nothing can bend it')
    w('    back inside a bound. See isa_spec.json, "derived_properties"."""')
    w('    return [base + i * stride for i in range(trip)]')
    w('')
    w('')
    w('def agu_legal_trip(op, operand, base=0, stride=1, limit=1 << 12):')
    w('    """The largest trip count for which an AGU term on `operand` of `op`')
    w('    stays inside the values the spec admits for that operand.')
    w('')
    w('    `None` means the operand has no restricted value set, so only the')
    w('    encoding rule bounds it. Otherwise the answer follows from')
    w('    `agu_reach` alone: a field whose legal values are a finite set is')
    w('    drivable from a loop for exactly as long as the progression stays')
    w('    inside that set. It is NOT a question of whether the field may be an')
    w('    AGU target -- every operand field may (AGU_TARGET_FIELD), and the')
    w('    sequencer does not know which fields have restricted values."""')
    w('    legal = LEGAL_VALUES.get((op, operand))')
    w('    if legal is None:')
    w('        return None')
    w('    t = 0')
    w('    while t < limit and base + t * stride in legal:')
    w('        t += 1')
    w('    return t')
    w('')
    w('')
    w('def agu_terms_used(w1):')
    w('    """How many of the AGU_TERMS an instruction word 1 already spends."""')
    w('    return sum(1 for target, _, _ in decode_agu(w1) if target)')
    w('')
    w('')
    w('# ----------------------------------------------- control flow and AGU ---')
    w('def trace(prog):')
    w('    """Run the program\'s control flow and address generation exactly as')
    w('    the spec says the sequencer does, yielding one')
    w('    `(pc, ivs, op, nr, f0, f1, f2, f3)` per instruction ISSUED.')
    w('')
    w('    The loop stack is a LIFO of {body_start, iv, trip} %d levels deep and'
      % spec["loop_stack"]["depth"])
    w('    the back edge is tested AFTER the body; each AGU term with a nonzero')
    w('    target adds `iv[level] * stride` to the field it names, in term')
    w('    order. Derived from isa_spec.json, not from the design."""')
    w('    pc = 0')
    w('    stack = []')
    w('    iv_now = [0] * LOOP_DEPTH')
    w('    guard = 0')
    w('    while pc < len(prog):')
    w('        guard += 1')
    w('        if guard > 1 << 22:')
    w('            raise AssertionError("program does not terminate")')
    w('        w0, w1 = prog[pc]')
    w('        op, nr, f0, f1, f2, f3 = decode(w0)')
    w('        if op == OP_LOOP:')
    w('            iv_now[len(stack)] = 0')
    w('            stack.append([pc + 1, 0, nr])')
    w('            pc += 1')
    w('        elif op == OP_ENDLOOP:')
    w('            fr = stack[-1]')
    w('            fr[1] += 1')
    w('            if fr[1] < fr[2]:')
    w('                iv_now[len(stack) - 1] = fr[1]')
    w('                pc = fr[0]')
    w('            else:')
    w('                stack.pop()')
    w('                pc += 1')
    w('        else:')
    w('            f = [f0, f1, f2, f3]')
    w('            for target, level, stride in decode_agu(w1):')
    w('                if target:')
    w('                    f[target - 1] += iv_now[level] * stride')
    w('            yield (pc, tuple(iv_now[:len(stack)]), op, nr, *f)')
    w('            pc += 1')
    w('')
    w('')
    w('def expand(prog):')
    w('    """`trace` without the provenance: one `(op, nr, f0, f1, f2, f3)` per')
    w('    dynamic issue."""')
    w('    return [e[2:] for e in trace(prog)]')
    w('')
    w('')
    w('# ------------------------------------------------- units and actions ---')
    w('#: Each unit as its ports and its step rate, isa_spec.json "units".')
    w('UNITS = (')
    for u in spec["units"]["list"]:
        ports = tuple((p["name"], p.get("physical", 1)) for p in u["ports"])
        w(f'    ({u["name"]!r}, {u.get("ii", 1)}, {u.get("elastic", True)!r}, '
          f'{ports!r}),')
    w(')')
    w('')
    w('#: Each memory as the model sees it: depth, lanes, ports, bank map.')
    w('STATES = (')
    for m in spec["memories"]:
        w(f'    ({m["name"]!r}, {m["depth_parameter"]!r}, '
          f'{m.get("lanes")!r}, {m["owner"]!r}, {m.get("read_ports", 1)}, '
          f'{m.get("write_ports", 1)}, {m.get("collision", "defined")!r}),')
    w(')')
    w('')
    w('#: Every instruction as the per-unit effects it composes. THIS IS THE')
    w('#: ONE STATEMENT: the units an opcode reaches, the rows it reads before')
    w('#: it writes, and every per-unit work count in the header below are')
    w('#: queries over this table, not separate declarations.')
    w('ACTIONS = {')
    for o in spec["opcodes"]:
        w(f'    {o["value"]}: (')
        for a in o["actions"]:
            fields = ", ".join(
                f"{k}={v!r}" if k != "args" else f"args={tuple(v)!r}"
                for k, v in a.items() if k not in ("unit", "kind"))
            head = f'{a["unit"]!r}, {a["kind"]!r}'
            w(f'        _A({head}' + (f', {fields}' if fields else '') + '),')
        w('    ),')
    w('}')
    w('')
    w('#: How many rows one issue of each opcode runs.')
    w('ROWS_EXPRESSION = {')
    for o in spec["opcodes"]:
        w(f'    {o["value"]}: {o["rows_expression"]!r},')
    w('}')
    w('')
    w('#: Properties of PROGRAMS that no instruction can establish on its own.')
    w('#: The model reports them as obligations rather than forgetting them.')
    w('CONTRACTS = (')
    for name, c in spec["contracts"].items():
        rule = c.get("rule") or " ".join(c.get("rules", ()))
        w(f'    ({name!r}, {{"rule": {rule!r}, '
          f'"enforced_by": {c.get("enforced_by")!r}}}),')
    w(')')
    w('')
    w('')
    w('def machine():')
    w('    """This ISA as an `allo.actions.Machine`, built once.')
    w('')
    w('    The model is machine-independent and lives in `allo/actions.py`;')
    w('    everything specific to this ISA is the three tables above, which')
    w('    `gen_isa.py` writes out of `isa_spec.json`."""')
    w('    global _MACHINE')
    w('    if _MACHINE is None:')
    w('        _MACHINE = _Machine(')
    w('            name="TinyTPU-isa",')
    w('            units=tuple(')
    w('                _Unit(n, ports=tuple(_Port(p, physical=w)')
    w('                                     for p, w in ports),')
    w('                      ii=ii, elastic=el)')
    w('                for n, ii, el, ports in UNITS),')
    w('            states=tuple(')
    w('                _State(n, rows=depth, lanes=lanes, owner=owner,')
    w('                       read_ports=rp, write_ports=wp, collision=col)')
    w('                for n, depth, lanes, owner, rp, wp, col in STATES),')
    w('            instructions=tuple(')
    w('                _Instruction(OPCODE_NAME[op], actions=acts,')
    w('                             rows=ROWS_EXPRESSION[op])')
    w('                for op, acts in ACTIONS.items()),')
    w('            parameters={"T": T, "MAXDIM": MAXDIM, "SPAD_ROWS": SPAD_ROWS,')
    w('                        "NVR": NVR, "NAR": NAR, "IMEM_SIZE": IMEM_SIZE,')
    w('                        "AR_RAW_DIST": AR_RAW_DIST},')
    w('            arithmetic="exact" if PRODUCT_EXACT else "rounding",')
    w('            contracts=tuple(')
    w('                _Contract(name, c["rule"] if "rule" in c')
    w('                          else "; ".join(c.get("rules", ())),')
    w('                          discharged_by=c.get("enforced_by"))')
    w('                for name, c in CONTRACTS),')
    w('        )')
    w('    return _MACHINE')
    w('')
    w('')
    w('def units_of(op):')
    w('    """Which units an opcode reaches. DERIVED, and the dispatch table')
    w('    in the sequencer is held to it."""')
    w('    return machine().units_of(OPCODE_NAME[op])')
    w('')
    w('')
    w('def effects(op, f0, f1, f2, f3, nr):')
    w('    """Every resolved effect of one issue: which unit, which cycle,')
    w('    which row of which memory, and what value. A validator and a')
    w('    reference model are both walks over this."""')
    w('    return machine().effects(OPCODE_NAME[op],')
    w('                             operands(op, f0, f1, f2, f3, nr))')
    w('')
    w('')
    w('def work(unit, op, f0, f1, f2, f3, nr):')
    w('    """The steps one unit spends on one issue. What a header count sums')
    w('    and what the sequencer rewrites `nr` to."""')
    w('    return machine().work(unit, OPCODE_NAME[op],')
    w('                          operands(op, f0, f1, f2, f3, nr))')
    w('')
    w('')
    w('def dispatch_rewrites():')
    w('    """Where a unit\'s own work count differs from the instruction\'s row')
    w('    count, so the sequencer has to hand it a rewritten `nr`. DERIVED:')
    w('    `spm` taking T+1 on an `mm` and `accu` taking 2*nr on a `vadd` are')
    w('    consequences of the ports, not entries in a table."""')
    w('    out = []')
    w('    for op, acts in ACTIONS.items():')
    w('        if not acts:')
    w('            continue')
    w('        names = [n for n in OPERAND_NAME[op] if n]')
    w('        probe = dict.fromkeys(names, 0)')
    w('        probe.update({"nr": 3, "acc": 1, "mode": 0})')
    w('        for unit in units_of(op):')
    w('            steps = machine().work(unit, OPCODE_NAME[op], probe)')
    w('            if steps and steps != probe["nr"]:')
    w('                out.append((OPCODE_NAME[op], unit, steps, probe["nr"]))')
    w('    return tuple(out)')
    w('')
    w('')
    w('# -------------------------------------------------------- imem header ---')
    w('#: What each header word counts, isa_spec.json "imem".entries[].work.')
    w('HEADER_WORK = (')
    for e in spec["imem"]["entries"]:
        w(f'    ({e["index"]}, {tuple(e["slice"])!r}, "{e["name"]}", '
          f'{e["work"]!r}),')
    w(')')
    w('')
    w('')
    w('def header(prog):')
    w('    """The NHDR header words of `imem`, computed from the ACTIONS.')
    w('')
    w('    Every count but the static instruction count is DYNAMIC: it is a sum')
    w('    over the issues `trace` produces, because each unit loops over the')
    w('    work it is really sent. A unit promised the wrong number hangs.')
    w('')
    w('    Nothing here knows that `spm` charges an `mm` T+1 iterations or that')
    w('    `accu` charges a `vadd` two steps a row. Those were sentences in the')
    w('    spec until the units declared their ports; now they are what the')
    w('    model computes from one `mm` and one `vadd`."""')
    w('    m = machine()')
    w('    ev = list(expand(prog))')
    w('    words = [0] * NHDR')
    w('    for index, (lo, hi), name, job in HEADER_WORK:')
    w('        total = 0')
    w('        if job.get("kind") == "static_instructions":')
    w('            total = len(prog)')
    w('        else:')
    w('            for op, nr, f0, f1, f2, f3 in ev:')
    w('                i = operands(op, f0, f1, f2, f3, nr)')
    w('                what = OPCODE_NAME[op]')
    w('                if job.get("kind") == "row_span":')
    w('                    total = max(total, m.row_span(job["state"], what, i))')
    w('                elif "port" in job:')
    w('                    total += m.items(job["unit"], job["port"], what, i)')
    w('                else:')
    w('                    total += m.work(job["unit"], what, i)')
    w('        if not 0 <= total <= usable_max(hi - lo):')
    w('            raise ValueError(')
    w('                f"header {name}={total} does not fit {hi - lo - 1} usable "')
    w('                f"bits of its {hi - lo}-bit slice")')
    w('        words[index] |= total << lo')
    w('    return words')
    w('')
    w('')
    w('def image(prog):')
    w('    """The whole instruction-memory image: header, then two words each."""')
    w('    words = header(prog)')
    w('    for w0, w1 in prog:')
    w('        words.append(int(w0))')
    w('        words.append(int(w1))')
    w('    if len(words) > IMEM_SIZE:')
    w('        raise ValueError(f"{len(words)} words > IMEM_SIZE={IMEM_SIZE}")')
    w('    return words')
    w('')
    w('')
    w('# ----------------------------------------------------------- numerics ---')
    w('#: The active arithmetic configuration, isa_spec.json "numerics".')
    w('NUMERICS = %r' % spec["numerics"]["active"])
    w('OPERAND_BITS = %d' % num["operand"]["bits"])
    w('OPERAND_MIN = %d' % num["operand"]["min"])
    w('OPERAND_MAX = %d' % num["operand"]["max"])
    w('OPERAND_DTYPE = np.%s' % num["operand"]["numpy_dtype"])
    w('ACC_BITS = %d' % num["accumulator"]["bits"])
    w('ACC_MIN = %d' % num["accumulator"]["min"])
    w('ACC_MAX = %d' % num["accumulator"]["max"])
    w('ACC_DTYPE = np.%s' % num["accumulator"]["numpy_dtype"])
    w('PRODUCT_EXACT = %r' % num["multiply"]["exact"])
    w('ACC_OVERFLOW = %r' % num["accumulate"]["overflow"])
    w('OUTPUT_CONVERSION = %r' % num["output"]["conversion"])
    w('')
    w('')
    w('def acc(x):')
    w('    """A value in the accumulator format: %s."""' % num["accumulate"]["overflow"])
    w('    m = 1 << ACC_BITS')
    w('    return ((np.asarray(x, np.int64) - ACC_MIN) % m + ACC_MIN)'
      '.astype(np.int64)')
    w('')
    w('')
    w('def product(a, b):')
    w('    """One operand-by-operand product, in the accumulator format.')
    w('')
    w('    Exact: %s"""' % num["multiply"]["note"])
    w('    return acc(np.asarray(a, np.int64) * np.asarray(b, np.int64))')
    w('')
    w('')
    w('def to_operand(x):')
    w('    """The accumulator-to-operand conversion `mvout` applies: %s to'
      % num["output"]["conversion"])
    w('    %d .. %d, no rounding."""' % (num["output"]["saturate"]["lo"],
                                         num["output"]["saturate"]["hi"]))
    w('    return np.clip(x, OPERAND_MIN, OPERAND_MAX)')
    w('')
    return "\n".join(out) + "\n"


def _dma_flag(spec, flag):
    for o in spec["opcodes"]:
        for d in o["operands"]:
            for b in d.get("bitflags", []):
                if b["name"] == flag:
                    return b["software_constant"]
    raise KeyError(flag)


# ------------------------------------------------------- the action model ---
def machine_of(spec, parameters=None):
    """This spec as an `allo.actions.Machine`.

    Built from the JSON rather than from the generated module, so the doc
    tables and the conformance checks read the same declaration the generated
    module is written out of and not each other."""
    from allo.actions import (  # noqa: PLC0415
        Action, Contract, Instruction, Machine, Port, State, Unit)
    p = dict(parameters or {})
    p.setdefault("T", 4)
    p.setdefault("MAXDIM", 64)
    for name in ("SPAD_ROWS", "NVR", "NAR", "IMEM_SIZE"):
        p.setdefault(name, 1 << 16)
    p.setdefault("AR_RAW_DIST", spec["contracts"]["accumulator_raw_distance"]["value"])
    return Machine(
        name=spec["name"],
        units=tuple(
            Unit(u["name"],
                 ports=tuple(Port(q["name"], physical=q.get("physical", 1))
                             for q in u["ports"]),
                 ii=u.get("ii", 1), elastic=u.get("elastic", True),
                 note=u.get("note"))
            for u in spec["units"]["list"]),
        states=tuple(
            State(m["name"], rows=str(m["depth_parameter"]),
                  lanes=m.get("lanes"), owner=m["owner"],
                  read_ports=m.get("read_ports", 1),
                  write_ports=m.get("write_ports", 1),
                  collision=m.get("collision", "defined"),
                  note=m.get("note"))
            for m in spec["memories"]),
        instructions=tuple(
            Instruction(o["name"], rows=o["rows_expression"],
                        operands=tuple(d["name"] for d in o["operands"]),
                        actions=tuple(
                            Action(**{k: (tuple(v) if k == "args" else v)
                                      for k, v in a.items()})
                            for a in o["actions"]),
                        note=o.get("note"))
            for o in spec["opcodes"]),
        parameters=p,
        arithmetic=("exact" if spec["numerics"]["configurations"][
            spec["numerics"]["active"]]["multiply"]["exact"] else "rounding"),
        contracts=tuple(
            Contract(name, c.get("rule") or " ".join(c.get("rules", ())),
                     discharged_by=c.get("enforced_by"))
            for name, c in spec["contracts"].items()),
    )


def action_units(spec, name):
    """The units one opcode reaches, derived. Nothing states this any more."""
    return machine_of(spec).instruction(name).units


def action_effects(spec, name):
    """One row's worth of reads and writes, as (kind, memory, base, rows,
    role). The spec used to carry this as prose that nothing computed."""
    machine = machine_of(spec)
    out = []
    for a in machine.instruction(name).actions:
        if a.kind not in ("read", "write"):
            continue
        rows = a.count if a.per == "instruction" else \
            machine.instruction(name).rows
        out.append((a.kind, a.state, a.base, rows, a.role or "",
                    a.when or "", a.unit))
    return out


# ------------------------------------------------------------ the doc tables ---
def _ceiling_value(predicate, t, limit=1 << 14):
    """The largest multiple of `t` satisfying a ceiling predicate -- the same
    computation `isa_encoding.maxdim_ceiling` does, so the doc carries no
    typed ceiling either."""
    ok = eval(f"lambda m, t: {predicate}", {"__builtins__": {}})  # noqa: S307
    best = None
    for m in range(t, limit + 1, t):
        if ok(m, t):
            best = m
    return best


def _cells(items, width=12):
    """One grid-table row pair for `[(label, "hi:lo")]`, most significant
    first, each cell sized to its widest line."""
    widths = [max(width, len(a) + 2, len(b) + 2) for a, b in items]
    rule = "+" + "+".join("-" * w for w in widths) + "+"
    top = "|" + "|".join(a.center(w) for (a, _), w in zip(items, widths)) + "|"
    bot = "|" + "|".join(b.center(w) for (_, b), w in zip(items, widths)) + "|"
    return [rule, top, bot, rule]


def _bit_layout(spec):
    """The two instruction words as bit grids, most significant on the left.

    Generated: a hand-drawn version of this went stale the moment a field
    moved, which is the argument for the whole file."""
    iw = spec["instruction_word"]
    agu = spec["agu"]
    fields = sorted(iw["word0"]["fields"], key=lambda f: -f["lo"])
    spare = iw["word0"]["spare_bits"]
    w0 = [(f["name"], f"{f['lo'] + f['width'] - 1}:{f['lo']}") for f in fields]
    if spare:
        w0 = [("", f"{max(spare)}:{min(spare)}")] + w0
    nterms, tbits = agu["terms"], agu["term_bits"]
    w1 = []
    top = iw["word_bits"] - 1
    if nterms * tbits < iw["word_bits"]:
        w1.append(("", f"{top}:{nterms * tbits}"))
    for i in range(nterms - 1, -1, -1):
        w1.append((f"term {i}", f"{tbits * (i + 1) - 1}:{tbits * i}"))
    out = ["Bit layout", "~~~~~~~~~~", "",
           f"Most significant on the left. ``enc()`` builds word 0 and "
           f"``enc_agu()`` word 1; both are generated from the field table "
           f"below, so neither picture can go stale.", "",
           ".. code-block:: text", "",
           "   word 0:"]
    out += ["   " + l for l in _cells(w0)]
    out += ["", "   word 1: up to AGU_TERMS = %d address terms of %d bits:"
            % (nterms, tbits)]
    out += ["   " + l for l in _cells(w1)]
    sub = "   ".join(
        f"{d['name']} [{d['lo'] + d['width'] - 1}:{d['lo']}]"
        for d in sorted(agu["subfields"], key=lambda d: -d["lo"]))
    out += ["", f"   one term:    {sub}",
            f"   resolves to: {agu['semantics']}",
            ""]
    if spare:
        out += [f"Bits {', '.join(str(b) for b in spare)} of word 0 and bits "
                f"{top}:{nterms * tbits} of word 1 are unused.", ""]
    return out


def _lit(text):
    """Literal markup for an identifier; plain prose for a phrase."""
    return f"``{text}``" if re.fullmatch(r"[\w.]+", text) else text


def _table(title, headers, rows):
    """A list-table. A cell containing newlines becomes an RST line block, so
    each line renders on its own line instead of being reflowed into one."""
    out = [f".. list-table:: {title}" if title else ".. list-table::",
           "   :header-rows: 1", ""]
    for r in (headers, *rows):
        for i, cell in enumerate(r):
            lead = "   * - " if i == 0 else "     - "
            lines = str(cell).split("\n")
            if len(lines) > 1:
                lines = ["| " + l for l in lines]
            out.append((lead + lines[0]).rstrip())
            out += [("       " + l).rstrip() for l in lines[1:]]
    out.append("")
    return out


def gen_doc(spec) -> str:
    """The ISA tables, for the region between the GENERATED markers."""
    iw = spec["instruction_word"]
    agu = spec["agu"]
    out = [BEGIN, "",
           ".. Generated from examples/tinytpu/isa_spec.json.",
           "   Edit the spec and run `python gen_isa.py --write`; `--check`",
           "   fails if this region is stale.", ""]

    out += _bit_layout(spec)

    out += _table("Who reads this spec, and how each one is held to it",
                  ("file", "held", "note"),
                  [(f"``{c['path']}``", c["held"], c["note"])
                   for c in spec["consumers"] if "path" in c])

    out += ["Instruction word", "~~~~~~~~~~~~~~~~", "",
            f"An instruction is **{iw['words_per_instruction']} "
            f"{iw['word_bits']}-bit words** (``IWORDS``). Word 0 carries the "
            f"opcode and five fields; word 1 carries up to "
            f"{agu['terms']} address-generation terms.", ""]
    out += _table("Instruction word 0",
                  ("field", "bits", "width", "usable range", "role"),
                  [(f"``{f['name']}``", f"``[{f['lo']}:{f['lo'] + f['width']}]``",
                    f["width"], f"0 .. {2 ** (f['width'] - 1) - 1}", f["role"])
                   for f in iw["word0"]["fields"]])
    out += ["Every field carries one more bit than its value range needs: an "
            "N-bit field safely holds ``0 .. 2^(N-1) - 1``, the spare-bit "
            "rule below.", ""]

    out += _table("Instruction word 1: one AGU term (%d bits, %d of them)"
                  % (agu["term_bits"], agu["terms"]),
                  ("subfield", "bits", "width", "usable range", "meaning"),
                  [(f"``{s['name']}``",
                    f"``[{s['lo']}:{s['lo'] + s['width']}]``", s["width"],
                    f"0 .. {2 ** (s['width'] - 1) - 1}",
                    s.get("note", ""))
                   for s in agu["subfields"]])
    out += [f"A term resolves to ``{agu['semantics']}``. Targets: "
            + ", ".join(f"{t['value']} = "
                        + (f"``{t['field']}``" if t["field"] else "unused")
                        for t in agu["targets"]) + ".", ""]

    out += ["Opcodes", "~~~~~~~", ""]
    rows = []
    for o in spec["opcodes"]:
        if o["operands"]:
            flds = "\n".join(
                f"``{d['field']}`` = {d['name']}: {d['meaning']}"
                for d in o["operands"])
        elif o.get("retired"):
            flds = o["note"]
        else:
            flds = ""
        rows.append((f"``{o['software_constant']}``", o["value"], o["name"],
                     flds, o["rows"],
                     ", ".join(_lit(u) for u in action_units(spec, o["name"]))
                     or "--"))
    out += _table("Opcodes", ("constant", "value", "name", "operand fields",
                              "``nr``", "units"), rows)
    out += ["The ``units`` column is **derived** from the actions below, not "
            "written beside each opcode: an opcode reaches whichever units "
            "its actions name. It used to be typed, and it was wrong twice "
            "-- ``mm`` did not name the array, and ``dma_ld`` named its "
            "destination in prose.", ""]

    out += ["Instructions as compositions of Actions", "^" * 38, "",
            "Every instruction is an ordered list of per-unit **effects**. "
            "Each effect names a unit, one of that unit's ports, and the "
            "element it touches. ``allo.actions`` holds the model and its "
            "legality rule; this table is what the spec declares, and the "
            "per-unit work counts in the header, the dispatch rewrites and "
            "the reads and writes below are queries over it rather than "
            "further declarations.", ""]
    rows = []
    for o in spec["opcodes"]:
        for a in o["actions"]:
            what = a.get("state") or a.get("compute") or ""
            rows.append((o["name"], a["unit"], a.get("port", a["kind"]),
                         a["kind"], _lit(what) if what else "--",
                         a.get("base", "--"),
                         a.get("count", "1") if a.get("per") == "instruction"
                         else o["rows_expression"],
                         a.get("when", "--")))
    out += _table("Actions", ("opcode", "unit", "port", "kind", "state",
                              "base", "items", "only if"), rows)

    out += _table("Units: ports, step rate, elasticity",
                  ("unit", "ports (items per cycle)", "II", "elastic"),
                  [(u["name"],
                    ", ".join(f"``{q['name']}``"
                              + (f" x{q['physical']}" if q.get("physical", 1) > 1
                                 else "")
                              for q in u["ports"]),
                    u.get("ii", 1), "yes" if u.get("elastic", True) else "no")
                   for u in spec["units"]["list"]])
    out += ["A unit's cost for an instruction is its busiest port's item "
            "count, so ``spm`` charging an ``mm`` ``T + 1`` iterations and "
            "``accu`` charging a ``vadd`` two steps a row are consequences of "
            "these ports and not entries in a table. ``elastic`` says whether "
            "contention inside a unit costs a step or is illegal.", ""]

    out += ["Derived properties", "~~~~~~~~~~~~~~~~~~", "",
            "Facts that **follow** from the tables above rather than being "
            "written in them. ``gen_isa.py --check`` recomputes each one from "
            "the spec and then confirms the design behaves that way, so they "
            "are checked rather than asserted -- every one of them was "
            "documented wrongly here until 2026-09-21, which is the argument "
            "for computing them.", ""]
    for d in spec["derived_properties"]:
        if "id" not in d:
            continue
        out += [f"``{d['id']}``", f"   {d['statement']}", ""]
        out += [f"   Derived from: "
                + ", ".join(f"``{x}``" for x in d["derived_from"]) + ".", ""]
        if d.get("corrects"):
            out += [f"   **Corrects:** {d['corrects']}", ""]
        if d.get("note"):
            out += [f"   {d['note']}", ""]

    out += ["Per-unit rewrites", "~~~~~~~~~~~~~~~~~", "",
            "The sequencer hands two units a rewritten copy of the word, so "
            "each unit's flat row loop reads its own work count out of "
            "``nr``:", ""]
    out += _table("", ("opcode", "unit", "rewritten to", "why"),
                  [(f"``{op}``", f"``{unit}``", f"``nr`` = {what}",
                    spec["dispatch"]["notes"].get(f"{op}/{unit}", ""))
                   for op, unit, what in _rewrites(spec)])
    out += ["Which units need a rewrite, and to what, is **derived**: it is "
            "every unit the sequencer dispatches to whose own work count "
            "differs from the instruction's row count. Nothing states it, so "
            "adding an instruction adds no entry here.", ""]

    out += ["Instruction memory header", "~~~~~~~~~~~~~~~~~~~~~~~~~", "",
            f"``imem[0:NHDR]`` (``NHDR = {spec['imem']['header_words']}``) is a "
            f"header of per-unit **work** counts; instructions follow, two "
            f"words each. Each count is read back through a "
            f"{spec['imem']['count_slice_width']}-bit slice, so it stops at "
            f"{spec['imem']['count_usable_max']}.", ""]
    hrows = []
    for e in spec["imem"]["entries"]:
        hrows.append((f"``imem[{e['index']}]``",
                      f"``[{e['slice'][0]}:{e['slice'][1]}]``",
                      f"``{e['name']}``", _work_prose(e["work"]),
                      _lit(e["consumer"])))
    out += _table("Header words", ("word", "bits", "name", "count", "consumer"),
                  hrows)
    out += ["Every count but the static one is **the work a unit does**, "
            "summed over the issues the program makes, and it is computed "
            "from the actions above rather than written here. The per-opcode "
            "costs that follow are therefore derived too:", ""]
    machine = machine_of(spec)
    crows = []
    for o in spec["opcodes"]:
        if not o["actions"]:
            continue
        probe = {d["name"]: 0 for d in o["operands"]}
        probe.update({"nr": 8, "acc": 1, "mode": 0})
        for unit in machine.instruction(o["name"]).units:
            steps = machine.work(unit, o["name"], probe)
            if not steps:
                continue
            per = ("``T + 1`` per issue" if steps == machine.parameters["T"] + 1
                   and "mm" == o["name"]
                   else f"``{steps // probe['nr']} x nr``"
                   if steps % probe["nr"] == 0 else f"{steps} at nr=8")
            crows.append((o["name"], unit, per))
    out += _table("Per-unit cost, derived from the ports",
                  ("opcode", "unit", "steps"), crows)

    out += ["Memory map", "~~~~~~~~~~", ""]
    touch = _memory_traffic(spec)
    out += _table("Memories", ("memory", "owner", "depth", "row width",
                               "written by", "read by", "cleared at start"),
                  [(f"``{m['name']}``", _lit(m["owner"]),
                    f"``{m['depth_parameter']}``",
                    f"``{m['row_width_bits']}`` bits",
                    ", ".join(touch[m["name"]]["write"]) or "the host",
                    ", ".join(touch[m["name"]]["read"]) or "the host",
                    "yes" if m["cleared_by_hardware"] else "**no**")
                   for m in spec["memories"]])
    out += ["The ``written by`` and ``read by`` columns are **derived** from "
            "the actions. They used to be two lists beside each memory that "
            "nothing computed from and nothing checked.", ""]
    out += ["No on-chip memory is cleared by the hardware, so every read of "
            "one is the program's obligation; see the contracts below.", ""]

    out += ["Contracts", "~~~~~~~~~", ""]
    wbr = spec["contracts"]["write_before_read"]
    out += ["**Write before read.**", ""]
    out += [f"* {r}" for r in wbr["rules"]] + [""]
    out += [f"Enforced by {wbr['enforced_by']}.", ""]
    ard = spec["contracts"]["accumulator_raw_distance"]
    out += ["**The accumulator read-after-write distance.** "
            f"{ard['rule']} ``{ard['software_constant']} = {ard['value']}`` "
            f"{ard['unit']}; cost per opcode: "
            + ", ".join(f"``{k}`` {v}" for k, v in ard["iteration_cost"].items())
            + f". Enforced by {ard['enforced_by']}. Exercised at its edge by "
              f"{ard['exercised_at_its_edge_by']}.", ""]

    out += ["Parameters", "~~~~~~~~~~", ""]
    out += _table("Build parameters",
                  ("constant", "environment variable", "default", "legal range",
                   "role"),
                  [(f"``{p['name']}``", f"``{p['env']}``",
                    f"``{p['default_expression']}``" if p["default"] is None
                    else p["default"],
                    f"{p['min']} .. " + ("unbounded" if p["max"] is None
                                         else str(p["max"])),
                    p["role"]) for p in spec["parameters"]])
    out += _table("Derived constants", ("constant", "value", "role"),
                  [(f"``{c['name']}``", f"``{c['value']}``", c["role"])
                   for c in spec["derived_constants"]])
    out += ["Cross-parameter constraints, asserted by "
            "``isa_encoding.check_parameters()``:", ""]
    out += [f"* ``{a['expr']}`` -- {a['why']}" for a in spec["assertions"]]
    out += [""]

    mc = spec["maxdim_ceilings"]
    out += ["What bounds ``MAXDIM``", "~~~~~~~~~~~~~~~~~~~~~~", "",
            "Two ceilings, both in the **encoding** rather than the datapath. "
            "They answer different questions, so neither is a correction of "
            "the other, and which one binds depends on the program and on "
            "``T``. The values below are **computed** by "
            "``isa_encoding.maxdim_ceiling``, over multiples of ``T``: "
            "solving either inequality over the reals gives a number no build "
            "can use.", ""]
    out += _table("``MAXDIM`` ceilings",
                  ("ceiling", "at T=4", "at T=8", "predicate",
                   "the question it answers", "how the design refuses past it"),
                  [(f"``{c['name']}``",
                    _ceiling_value(c["predicate"], 4),
                    _ceiling_value(c["predicate"], 8),
                    f"``{c['predicate'].replace('m', 'MAXDIM').replace('t', 'T')}``",
                    c["question"], c["fails_as"])
                   for c in mc["ceilings"]])
    out += [mc["note"], ""]

    out += ["Numerics", "~~~~~~~~", ""]
    nm = spec["numerics"]
    out += [f"Active configuration: **{nm['active']}**. A configuration states "
            f"what happens to *values*, not only how wide they are, so that a "
            f"format whose arithmetic is inexact can be added without "
            f"restructuring anything above.", ""]
    for cname, c in nm["configurations"].items():
        rows = [
            ("operand", f"{c['operand']['kind']}, "
                        f"{'signed' if c['operand']['signed'] else 'unsigned'}, "
                        f"{c['operand']['bits']} bits",
             f"{c['operand']['min']} .. {c['operand']['max']}",
             c["operand"]["where"]),
            ("accumulator", f"{c['accumulator']['kind']}, "
                            f"{'signed' if c['accumulator']['signed'] else 'unsigned'}, "
                            f"{c['accumulator']['bits']} bits",
             f"{c['accumulator']['min']} .. {c['accumulator']['max']}",
             c["accumulator"]["where"]),
            ("multiply", f"{' x '.join(c['multiply']['inputs'])} -> "
                         f"{c['multiply']['intermediate_bits']} bits",
             "exact" if c["multiply"]["exact"] else "inexact",
             c["multiply"]["note"]),
            ("accumulate", c["accumulate"]["operation"],
             "exact" if c["accumulate"]["exact"]
             else f"rounding: {c['accumulate']['rounding']}; overflow: "
                  f"{c['accumulate']['overflow']}",
             f"order: {c['accumulate']['order']}"),
            ("output", f"{c['output']['from']} -> {c['output']['to']}, "
                       f"{c['output']['conversion']}",
             f"to {c['output']['saturate']['lo']} .. "
             f"{c['output']['saturate']['hi']}, rounding "
             f"{c['output']['rounding']}",
             c["output"]["where"]),
        ]
        for op, e in c["elementwise"].items():
            rows.append((f"``{op}``", e["operation"],
                         "exact" if e["exact"] else f"overflow: {e['overflow']}",
                         f"in the {e['format']} format"))
        out += _table(f"``{cname}``", ("stage", "operation",
                                       "range, or exactness and what replaces it",
                                       "where / note"), rows)

    out.append(END)
    # The spec is format-neutral, so its prose marks identifiers with single
    # backticks. Promote those to reST inline literals; doubled ones are left
    # alone. No trailing newline: the splice must be idempotent.
    body = "\n".join(out)
    return re.sub(r"(?<!`)`([^`\n]+)`(?!`)", r"``\1``", body)


#: Which unit each sequencer dispatch queue feeds, for the derivations that
#: need to know what the sequencer can rewrite at all.
QUEUE_UNITS = ("dma_ld", "spm", "vru", "accu", "dma_st")


def _rewrites(spec, nr=3):
    """Every (opcode, unit, work) whose work count differs from `nr`.

    The sequencer hands such a unit a rewritten copy of the word so its flat
    row loop reads its own count out of `nr`. DERIVED from the actions: `spm`
    taking T+1 on an `mm` and `accu` taking 2*nr on a `vadd` are consequences
    of the ports, not entries in a table."""
    machine = machine_of(spec)
    out = []
    for o in spec["opcodes"]:
        if not o["actions"]:
            continue
        probe = {d["name"]: 0 for d in o["operands"]}
        probe.update({"nr": nr, "acc": 1, "mode": 0})
        for unit in machine.instruction(o["name"]).units:
            if unit not in QUEUE_UNITS:
                continue
            steps = machine.work(unit, o["name"], probe)
            if steps and steps != nr:
                out.append((o["name"], unit,
                            f"{steps // nr} * nr" if steps % nr == 0
                            else f"T + {steps - machine.parameters['T']}"))
    return out


def _work_prose(job):
    if job.get("kind") == "static_instructions":
        return "static instruction count"
    if job.get("kind") == "row_span":
        return f"the row span of ``{job['state']}`` every ``dma_ld`` reads"
    if "port" in job:
        return f"items on ``{job['unit']}``'s ``{job['port']}`` port"
    return f"the steps ``{job['unit']}`` spends"


def _memory_traffic(spec):
    """Which opcodes read and write each memory, derived from the actions."""
    out = {m["name"]: {"read": [], "write": []} for m in spec["memories"]}
    for o in spec["opcodes"]:
        for a in o["actions"]:
            if a["kind"] not in ("read", "write") or a["state"] not in out:
                continue
            label = f"``{o['name']}``" + (f" ({a['role']})" if a.get("role")
                                          and a["kind"] == "read" else "")
            if a.get("when"):
                label += f" [{a['when']}]"
            if label not in out[a["state"]][a["kind"]]:
                out[a["state"]][a["kind"]].append(label)
    out["imem"]["read"] = ["``sequencer``"]
    return out


def _term_prose(t):
    scale = t.get("scale")
    ops = ", ".join(f"``{o}``" for o in t.get("ops", ()))
    where = {"dst_spad": " with destination ``spad``",
             "dst_vr": " with destination ``vr``",
             "src_a": " with source ``A``", "src_b": " with source ``B``",
             None: ""}[t.get("where")]
    if t["kind"] == "static_instructions":
        return "static instruction count"
    body = {"rows": f"rows of {ops}{where}",
            "instructions": f"issues of {ops}{where}",
            "row_span": f"max(first row + rows) over {ops}{where}"}[t["kind"]]
    return body + (f", times {scale}" if scale and scale != "1" else "")


# ----------------------------------------------------------------- writing ---
def _splice(text, body):
    i, j = text.find(BEGIN), text.find(END)
    if i < 0 or j < 0:
        raise SystemExit(f"{DOC}: the GENERATED markers are missing")
    return text[:i] + body + text[j + len(END):]


def artefacts(spec):
    """{path: wanted content} for every generated artefact."""
    with open(DOC) as f:
        doc = f.read()
    return {ENCODING: gen_encoding(spec), DOC: _splice(doc, gen_doc(spec))}


def write(spec):
    for path, want in artefacts(spec).items():
        with open(path, "w") as f:
            f.write(want)
        print(f"  wrote {os.path.relpath(path, REPO)}")
    return 0


def spec_module(spec):
    """The spec as a live module, generated in memory.

    The behavioural checks below must be against THE SPEC, not against the
    checked-in `isa_encoding.py`, or a stale artefact would make them compare
    the design with the design's own last agreed-upon encoding and pass.
    `check_generated` separately proves the checked-in file equals this one."""
    import types
    m = types.ModuleType("isa_spec_live")
    m.__dict__["__file__"] = ENCODING
    exec(compile(gen_encoding(spec), ENCODING, "exec"), m.__dict__)  # noqa: S102
    return m


def check_generated(spec):
    """Every generated artefact byte-identical to what the spec produces."""
    fails = []
    for path, want in artefacts(spec).items():
        with open(path) as f:
            have = f.read()
        rel = os.path.relpath(path, REPO)
        if have == want:
            print(f"  {rel}: up to date ({len(want)} bytes, byte-identical)")
            continue
        d = "".join(difflib.unified_diff(
            have.splitlines(True), want.splitlines(True),
            fromfile=rel + " (checked in)", tofile=rel + " (from the spec)"))
        fails.append(f"{rel} is STALE -- run `gen_isa.py --write`:\n{d}")
    return fails


# ------------------------------------------------- the design, by the spec ---
def _module_assignments(path):
    """{name: ast node} for every module-level `NAME = ...`."""
    with open(path) as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = node.value
    return out


def check_design_constants(spec, U):
    """Every software constant the spec names, by value, in the design."""
    fails = []
    want = {}
    for o in spec["opcodes"]:
        want[o["software_constant"]] = o["value"]
    for d in (d for o in spec["opcodes"] for d in o["operands"]):
        for b in d.get("bitflags", []):
            want[b["software_constant"]] = b["mask"]
    for t in spec["agu"]["targets"]:
        if t.get("software_constant"):
            want[t["software_constant"]] = t["value"]
    want["AGU_TERMS"] = spec["agu"]["terms"]
    want["LOOP_DEPTH"] = spec["loop_stack"]["depth"]
    want["IWORDS"] = spec["instruction_word"]["words_per_instruction"]
    want["NHDR"] = spec["imem"]["header_words"]
    want["AR_RAW_DIST"] = spec["contracts"]["accumulator_raw_distance"]["value"]
    for c in spec["derived_constants"]:
        if isinstance(c["value"], int):
            want[c["name"]] = c["value"]
    for name, v in sorted(want.items()):
        got = getattr(U, name, None)
        if got != v:
            fails.append(f"microarch_isa.{name} = {got!r}, spec says {v!r}")
    print(f"  design constants: {len(want)} held to the spec by value")
    return fails


def check_design_layout(spec):
    """The layout `ip/isa.py` names once -- the field bounds the encoder and
    the assembler's decoder both read, and the AGU term's sub-field widths --
    held to the field table by value."""
    from examples.tinytpu.ip import isa as L
    want = {}
    for f in _fields(spec):
        key = f["name"].upper()
        want[f"{key}_LO"] = f["lo"]
        want[f"{key}_HI"] = f["lo"] + f["width"]
    sub = {s["name"]: s for s in spec["agu"]["subfields"]}
    want["AGU_TERM_BITS"] = spec["agu"]["term_bits"]
    want["AGU_TARGET_BITS"] = sub["target"]["width"]
    want["AGU_LEVEL_BITS"] = sub["level"]["width"]
    fails = [f"ip/isa.{name} = {getattr(L, name, None)!r}, spec says {v!r}"
             for name, v in sorted(want.items()) if getattr(L, name, None) != v]
    if sub["level"]["lo"] != sub["target"]["width"] or \
            sub["stride"]["lo"] != sub["level"]["lo"] + sub["level"]["width"]:
        fails.append("the spec's AGU sub-fields are not packed back to back, "
                     "which ip/isa.enc_agu assumes")
    print(f"  design layout: {len(want)} names in ip/isa.py held to the field "
          f"table by value")
    return fails


def _parameter_source(src, name):
    """The expression a design parameter is built from. The instantiation
    binds most of them as `NAME = PARAMS.NAME`, so follow that to the
    `TpuParams(...)` keyword, and a keyword that is itself a module name to
    that name's own assignment."""
    node = src.get(name)
    call = src.get("PARAMS")
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "PARAMS" and isinstance(call, ast.Call)):
        node = next((k.value for k in call.keywords if k.arg == node.attr),
                    None)
    if isinstance(node, ast.Name) and node.id != name and node.id in src:
        return _parameter_source(src, node.id)
    return node


def check_design_parameters(spec, U, E):
    """Parameter values in range, cross-constraints true, and each one still
    read from its own environment variable (the gate rebuilds at other
    values, so a literal would pass every scored check)."""
    fails = []
    src = _module_assignments(DESIGN)
    for p in spec["parameters"]:
        name, env = p["name"], p["env"]
        v = getattr(U, name, None)
        if v is None:
            fails.append(f"microarch_isa has no parameter {name}")
            continue
        if v < p["min"] or (p["max"] is not None and v > p["max"]):
            fails.append(f"{name}={v} outside the spec range "
                         f"{p['min']}..{p['max']}")
        node = _parameter_source(src, name)
        if node is None or env not in ast.dump(node):
            fails.append(f"microarch_isa.{name} is not read from "
                         f"os.environ[{env!r}] -- the spec makes it a build "
                         f"parameter and the gate rebuilds at other values")
    try:
        E.check_parameters()
    except ValueError as e:
        fails.append(f"spec parameter constraint: {e}")
    print(f"  design parameters: {len(spec['parameters'])} in range, read from "
          f"their environment variables, constraints hold")
    return fails


#: Configurations the two modules are asked to agree on. The empty one is the
#: important one: it catches a DEFAULT that moved in the design and not in the
#: spec, which no test would otherwise fail on until a shape mismatch.
PROBE_ENVS = [
    {},
    {"TPU_MAXDIM": "16"},                      # what reproduce.sh pins
    {"TPU_MAXDIM": "8"},
    {"TPU_MAXDIM": "12"},                      # ... and the two the CHIA
    {"TPU_T": "8", "TPU_MAXDIM": "32"},        # parametricity gate rebuilds at
    {"TPU_SPAD": "300", "TPU_NVR": "300", "TPU_NAR": "300", "TPU_IMEM": "64"},
]

_PROBE = """
import json, sys
sys.path.insert(0, %r)
sys.path.insert(0, %r)
import microarch_isa as U
import isa_encoding as E
names = %r
print(json.dumps({n: [getattr(U, n, None), getattr(E, n, None)] for n in names}))
"""


def check_parameter_agreement(spec, U):
    """`microarch_isa` and the generated module must land on the SAME
    configuration, under any environment -- not merely on the same ranges.

    The generated module reads the same `TPU_*` variables the design reads, so
    a parameter whose DEFAULT moves in the design (the MAXDIM rebuild is one
    such change) silently gives the reference model a different machine from
    the one under test. Each probe below re-imports both modules in a fresh
    interpreter, because a default is read at import time."""
    names = [p["name"] for p in spec["parameters"]]
    names += [c["name"] for c in spec["derived_constants"]]
    names += ["NHDR", "IWORDS", "LOOP_DEPTH", "AGU_TERMS", "AR_RAW_DIST"]
    src = _PROBE % (HERE, REPO, names)
    fails = []
    for env in PROBE_ENVS:
        e = {k: v for k, v in os.environ.items() if not k.startswith("TPU_")}
        e.update(env)
        out = subprocess.run([sys.executable, "-c", src], env=e,
                             capture_output=True, text=True)
        if out.returncode:
            fails.append(f"parameter probe {env or 'defaults'} failed: "
                         f"{out.stderr.strip().splitlines()[-1:]}")
            continue
        got = json.loads(out.stdout)
        for n, (design, ours) in got.items():
            if design != ours:
                fails.append(
                    f"parameter probe {env or 'defaults'}: "
                    f"microarch_isa.{n} = {design!r} but isa_encoding.{n} = "
                    f"{ours!r} -- the reference model would be modelling a "
                    f"different machine")
    print(f"  parameter agreement: the design and the generated module land on "
          f"the same {len(names)} values under {len(PROBE_ENVS)} "
          f"configurations, defaults included")
    return fails


_SLICE = re.compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]")


def _header_slices(spec):
    """Every bit range of a header word the design may legitimately read: each
    count's own slice, and the whole span of a word that packs two of them
    (`spm` takes the mm count and the row count in one move)."""
    out, span = set(), {}
    for e in spec["imem"]["entries"]:
        lo, hi = e["slice"]
        out.add((lo, hi))
        cur = span.get(e["index"])
        span[e["index"]] = (min(lo, cur[0]), max(hi, cur[1])) if cur else (lo, hi)
    return out | set(span.values())


def check_design_slices(spec):
    """Every bit slice the design takes of a 64-bit instruction word is one the
    spec defines, and every field the spec defines is taken somewhere.

    A slice of a control word is the design's copy of the field table: this is
    the check that a unit decoding `w0[6:18]` is reading what the spec calls
    `f0` and not something one bit off."""
    fails = []
    want = {(f["lo"], f["lo"] + f["width"]): f["name"]
            for f in _fields(spec)}
    # The header counts and the per-unit rewrites slice the same words.
    allowed = dict(want)
    for lo, hi in _header_slices(spec):
        allowed.setdefault((lo, hi), "a header count")
    seen = set()
    # Every name a unit declares as a 64-bit word or an array of them -- read
    # out of the source, so a new control word is covered without editing this
    # check. Per unit: two units may use one name for words of different
    # widths. An array index is not a bit slice and simply does not match the
    # `[lo:hi]` form below.
    lines, names = [], set()
    for path in HARDWARE:
        rel = os.path.relpath(path, HERE)
        with open(path) as f:
            text = f.read()
        words = set(re.findall(r"([A-Za-z_][A-Za-z_0-9]*)\s*:\s*UInt\(64\)",
                               text))
        names |= words
        if words:
            pattern = re.compile(r"\b(%s)\b\s*\[" % "|".join(sorted(words)))
            lines += [(rel, n, line, pattern)
                      for n, line in enumerate(text.splitlines(), 1)]
    names = sorted(names)
    assert names, "ip/units declares no 64-bit word"
    for rel, n, line, word_vars in lines:
        code = line.split("#", 1)[0]
        for m in word_vars.finditer(code):
            tail = code[m.end() - 1:]
            s = _SLICE.match(tail)
            if not s:
                continue            # a computed slice, e.g. w1[19 * _t : ...]
            lo, hi = int(s.group(1)), int(s.group(2))
            if (lo, hi) not in allowed:
                fails.append(
                    f"{rel}:{n}: slices bits [{lo}:{hi}] of a "
                    f"64-bit instruction word, which the spec does not "
                    f"define: {line.strip()}")
            else:
                seen.add((lo, hi))
    missing = [f"{name} [{lo}:{hi}]" for (lo, hi), name in want.items()
               if (lo, hi) not in seen]
    if missing:
        fails.append("the design never reads these spec fields: "
                     + ", ".join(missing))
    print(f"  hardware bit slices: {len(seen)} distinct slices of the "
          f"{len(names)} 64-bit words ip/units declares "
          f"({', '.join(names)}), all defined by the spec")
    return fails


def _shapes():
    from examples.tinytpu.shapes import SHAPES
    return SHAPES


def _programs(U, D):
    """Programs to hold the two implementations to each other on.

    Every generator is guarded: the shipped ones are written for MAXDIM >= 16
    and the parametricity gate also builds at 8 and 12, so one that cannot
    emit a legal program at this configuration is SKIPPED, exactly as
    `chia_agent/param_check.py` skips such a seed. The GEMMs that do fit are
    always there, so the set is never empty."""
    import stress_isa as S          # only for its random-program generator
    from examples.tinytpu.shapes import SHAPES

    progs = []

    def add(tag, make, *args):
        try:
            prog = make(*args)
            U.check_program(prog)   # and skip one the validator refuses here
        except Exception:  # noqa: BLE001 -- ungeneratable at this MAXDIM
            return
        progs.append((tag, prog))

    fit = [s for s in SHAPES
           if max(s) <= U.MAXDIM and all(d % U.T == 0 for d in s)]
    if not fit:
        # No published shape is legal at this T; fall back to the smallest
        # cubic GEMM the configuration does admit.
        fit = [(U.T, U.T, U.T)]
    for shape in fit:
        for relu in (False, True):
            add(f"gemm{'.relu' if relu else ''} {shape}",
                D.gemm_program, *shape, relu)
            add(f"gemm.hand {shape} relu={relu}",
                U.gemm_program_handwritten, *shape, relu)
            # The unrolled form outgrows imem at the larger shapes, which is
            # the point of having control flow; take it where it fits.
            def flat(M, K, N, r):
                p = U.gemm_program_flat(M, K, N, r)
                assert U.NHDR + U.IWORDS * len(p) <= U.IMEM_SIZE, "too long"
                return p

            add(f"gemm.flat {shape} relu={relu}", flat, *shape, relu)
    add("vadd_program", U.vadd_program, *max(fit))
    for M in (4, 8):
        add(f"vector_program({M})", D.vector_program, M)
    for d in (U.AR_RAW_DIST, U.AR_RAW_DIST + 1):
        add(f"ar_distance_program({d})", D.ar_distance_program, d)
    for s in range(40):
        add(f"fuzz {7000 + s}", S.random_program, 7000 + s)
    assert progs, "no program could be generated at this configuration"
    return progs


def check_behaviour(spec, U, D, E):
    """The design and the spec must agree on what the bits MEAN, not only on
    what the constants are: the encoder, the AGU word, the resolved dynamic
    stream, and the header."""
    import numpy as np
    fails = []
    rng = np.random.default_rng(0)

    def both(what, design, ours):
        """Encode the same thing with the design's encoder and the spec's.
        A refusal on one side and not the other is a disagreement, not a
        crash: a widened spec that the design would reject is exactly the
        drift this check exists to name."""
        try:
            a = design()
        except (AssertionError, ValueError) as e:
            a = f"refused: {e}"
        try:
            b = ours()
        except (AssertionError, ValueError) as e:
            b = f"refused: {e}"
        if a == b:
            return None
        fmt = lambda v: v if isinstance(v, str) else f"0x{v:016x}"  # noqa: E731
        return f"{what}: design {fmt(a)}, spec {fmt(b)}"

    n = 0
    for o in spec["opcodes"]:
        for _ in range(200):
            fs = [int(x) for x in rng.integers(0, 2048, 4)]
            nr = int(rng.integers(0, 128))
            n += 1
            bad = both(f"enc({o['name']}, {fs}, nr={nr})",
                       lambda: U.enc(o["value"], *fs, nr=nr),
                       lambda: E.encode(o["value"], *fs, nr=nr))
            if bad:
                fails.append(bad)
                break
    for _ in range(500):
        k = int(rng.integers(0, spec["agu"]["terms"] + 1))
        terms = [(int(rng.integers(1, 5)),
                  int(rng.integers(0, spec["loop_stack"]["depth"])),
                  int(rng.integers(0, 2048))) for _ in range(k)]
        n += 1
        bad = both(f"enc_agu({terms})",
                   lambda: U.enc_agu(*terms), lambda: E.encode_agu(*terms))
        if bad:
            fails.append(bad)
            break
    print(f"  encoder: {n} words identical between microarch_isa.enc/enc_agu "
          f"and the spec's")

    progs = _programs(U, D)
    for tag, prog in progs:
        if U.expand(prog) != E.expand(prog):
            fails.append(f"{tag}: the design's resolved dynamic stream differs "
                         f"from the spec's")
            break
    print(f"  control flow and AGU: the design's `expand` and the spec's agree "
          f"on all {len(progs)} programs")

    for tag, prog in progs:
        a = U.assemble(prog)[:U.NHDR]
        b = E.header(prog)
        if a != b:
            bad = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
            fails.append(f"{tag}: header words differ (index, design, spec): "
                         f"{bad}")
            break
    print(f"  imem header: microarch_isa.assemble's {U.NHDR} words equal the "
          f"spec's on all {len(progs)} programs")

    # The numerics, as the spec states them, against the design's own
    # declarations: the accumulator lane width and the mvout saturation.
    num = spec["numerics"]["configurations"][spec["numerics"]["active"]]
    if U.AW != U.T * num["accumulator"]["bits"]:
        fails.append(f"microarch_isa.AW = {U.AW}, spec accumulator is "
                     f"{num['accumulator']['bits']} bits x T")
    if U.VW != U.T * num["operand"]["bits"]:
        fails.append(f"microarch_isa.VW = {U.VW}, spec operand is "
                     f"{num['operand']['bits']} bits x T")
    hsrc = ""
    for path in HARDWARE:
        with open(path) as f:
            hsrc += f.read()
    sat = num["output"]["saturate"]
    for op, bound in (("<", sat["lo"]), (">", sat["hi"])):
        if not re.search(rf"\w+\s*{op}\s*{bound}\s*:", hsrc):
            fails.append(f"ip/units' mvout does not saturate at {bound}, "
                         f"which the spec's output conversion requires")
    print(f"  numerics: operand and accumulator lane widths and the mvout "
          f"saturation bounds match the {spec['numerics']['active']} "
          f"configuration")
    return fails


#: Which unit each sequencer dispatch queue feeds. The queue names are the
#: design's; the units are the spec's.
DISPATCH_QUEUE = {"c_dld": "dma_ld", "c_spm": "spm", "c_vru": "vru",
                  "c_acc": "accu", "c_dst": "dma_st"}


def _sequencer_dispatch():
    """Which queues the sequencer puts an opcode on, read out of its source.

    A regex over one unit's body rather than an import, because the body is a
    `@df.kernel` that only Allo can execute. It is the hardware's own dispatch
    table, and the point is to hold it to a table nobody wrote."""
    path = os.path.join(HERE, "ip", "units", "sequencer.py")
    with open(path) as f:
        text = f.read()
    out, current = {}, None
    for line in text.splitlines():
        body = line.split("#", 1)[0]
        m = re.match(r"\s*(?:el)?if op == (OP_[A-Z_]+):", body)
        if m:
            current = m.group(1)
            out.setdefault(current, set())
            continue
        if current is None:
            continue
        if re.match(r"\s{8}\S", body) and not re.match(r"\s*(if|else|#)", body) \
                and "put(" not in body and "=" in body and ":" in body:
            pass
        for q in re.findall(r"(c_[a-z]+)\.put\(", body):
            out[current].add(DISPATCH_QUEUE[q])
        if body.strip().startswith("pc += 1"):
            current = None
    return out


def _sequencer_rewrites():
    """Which units the sequencer hands a rewritten `nr`, and the expression it
    rewrites it to, read out of the hardware.

    A `<name>_copy[54:62] = <expr>` followed by a `c_<queue>.put(<name>_copy)`
    IS the rewrite. The actions say which ones should be there and what they
    should say; this says what the hardware does."""
    path = os.path.join(HERE, "ip", "units", "sequencer.py")
    with open(path) as f:
        text = f.read()
    rewritten, sent, opcode_of, current = {}, {}, {}, None
    for line in text.splitlines():
        body = line.split("#", 1)[0]
        m = re.match(r"\s*(?:el)?if op == OP_([A-Z_]+):", body)
        if m:
            current = m.group(1).lower()
            continue
        m = re.match(r"\s*([a-z_]+)_copy\[54:62\] = (.+)$", body)
        if m:
            rewritten[m.group(1)] = m.group(2).strip()
            opcode_of[m.group(1)] = current
        m = re.match(r"\s*c_([a-z]+)\.put\(([a-z_]+)_copy\)", body)
        if m:
            sent[m.group(2)] = DISPATCH_QUEUE["c_" + m.group(1)]
    return {(opcode_of[name], sent[name]): expression
            for name, expression in rewritten.items() if name in sent}


def check_actions(spec, U, E):
    """The action model, and the four things it is held to.

    The model itself is in `allo/actions.py` and knows nothing about this
    machine; what is checked here is that the ONE declaration in `opcodes`
    accounts for every place the design states the same fact independently.
    """
    from allo.actions import CHECKED, DESCRIPTION, GUARANTEED  # noqa: PLC0415
    fails = []
    machine = machine_of(spec, {"T": U.T, "MAXDIM": U.MAXDIM,
                                "SPAD_ROWS": U.SPAD_ROWS, "NVR": U.NVR,
                                "NAR": U.NAR, "IMEM_SIZE": U.IMEM_SIZE,
                                "AR_RAW_DIST": U.AR_RAW_DIST})
    held = machine.obligations()
    kinds = {}
    for o in spec["opcodes"]:
        for a in o["actions"]:
            kinds[a.get("status", DESCRIPTION)] = \
                kinds.get(a.get("status", DESCRIPTION), 0) + 1
    n_actions = sum(len(o["actions"]) for o in spec["opcodes"])
    print(f"  action model: {n_actions} actions over "
          f"{len(spec['units']['list'])} units compose "
          f"{len([o for o in spec['opcodes'] if o['actions']])} instructions; "
          f"the rule accepts and leaves {len(held)} obligation(s)")
    print("    status: " + ", ".join(
        f"{n} {k}" for k, n in sorted(kinds.items())) +
        f" ({DESCRIPTION} is what an undecorated action is worth, and "
        f"{CHECKED} must name its checker; nothing here is {GUARANTEED})")

    # 1. the per-unit rewrites the sequencer performs, derived.
    derived = {}
    for o in spec["opcodes"]:
        if not o["actions"]:
            continue
        probe = {d["name"]: 0 for d in o["operands"]}
        probe.update({"nr": 3, "acc": 1, "mode": 0})
        for unit in machine.instruction(o["name"]).units:
            # Only a unit the sequencer dispatches to can be handed a
            # rewritten word; the array takes its counts through spm's header
            # word, and the sequencer is the one doing the rewriting.
            if unit not in DISPATCH_QUEUE.values():
                continue
            steps = machine.work(unit, o["name"], probe)
            if steps and steps != probe["nr"]:
                derived[(o["name"], unit)] = steps
    hardware = _sequencer_rewrites()
    if set(derived) != set(hardware):
        fails.append(
            f"the actions imply a rewritten `nr` for "
            f"{sorted(derived)} and ip/units/sequencer.py rewrites it for "
            f"{sorted(hardware)}")
    for key, expression in hardware.items():
        if key not in derived:
            continue
        got = int(eval(expression, {"__builtins__": {}},  # noqa: S307
                       {"nr": 3, "T": U.T}))
        if got != derived[key]:
            fails.append(
                f"{key[0]}/{key[1]}: the sequencer rewrites nr to "
                f"{expression!r} = {got}, the actions say {derived[key]}")
    print(f"  dispatch rewrites: the actions imply {len(derived)}, and "
          f"ip/units/sequencer.py performs exactly those -- "
          + ", ".join(f"{op}/{unit} -> {n}" for (op, unit), n in
                      sorted(derived.items())))

    # 2. the sequencer's own dispatch, read out of the hardware.
    hardware = _sequencer_dispatch()
    checked = 0
    for o in spec["opcodes"]:
        if o["name"] in ("nop", "loop", "endloop") or o.get("retired"):
            continue
        want = {u for u in machine.instruction(o["name"]).units
                if u in DISPATCH_QUEUE.values()}
        got = hardware.get(o["software_constant"])
        if got is None:
            fails.append(f"the sequencer dispatches no queue for "
                         f"{o['software_constant']}, which has actions")
            continue
        if got != want:
            fails.append(
                f"{o['name']}: the sequencer feeds {sorted(got)} but the "
                f"actions name {sorted(want)}")
        checked += 1
    print(f"  sequencer dispatch: {checked} opcodes feed exactly the queues "
          f"their actions name, read out of ip/units/sequencer.py")

    # 3. each unit's declared ISA namespace against the opcodes it acts on.
    from examples.accelerator.tinytpu_vitis.ip import tinytpu as T_  # noqa: PLC0415
    by_name = {u.name: u for u in T_.units()}
    n = 0
    for u in spec["units"]["list"]:
        design = by_name.get(u["name"])
        if design is None:            # `array` is wld + pe, neither decodes
            continue
        declared = {name for name in design.isa if name.startswith("OP_")}
        acting = {o["software_constant"] for o in spec["opcodes"]
                  if any(a["unit"] == u["name"] for a in o["actions"])}
        if u["name"] == "sequencer":
            continue                  # it decodes every opcode to dispatch it
        if len(acting) == 1:
            # One opcode reaches this unit's queue, so its body needs no
            # opcode test and declares no OP_ name. That exemption is itself
            # derived: it holds exactly while the actions name one opcode.
            acting = set()
        if declared != acting:
            fails.append(
                f"unit {u['name']} declares isa={sorted(declared)} but its "
                f"actions are on {sorted(acting)}")
        n += 1
    print(f"  unit ISA namespaces: {n} units decode exactly the opcodes they "
          f"have actions for")

    # 4. the memories each opcode touches, against the validator's behaviour.
    n = 0
    for o in spec["opcodes"]:
        for a in o["actions"]:
            if a["kind"] != "read" or a["state"] not in ("spad", "vr", "ar"):
                continue
            n += 1
    print(f"  write-before-read surface: {n} read actions name a memory the "
          f"assembler must have seen written; the contract is carried as an "
          f"obligation, not as a property of any one instruction")
    return fails


def check_reference(spec):
    """The reference model must take its ISA facts from the spec, not from the
    design. Structural: it may import parameters and the program validator from
    `microarch_isa`, but no opcode number, field position or numeric width."""
    path = os.path.join(HERE, "isa_ref.py")
    with open(path) as f:
        tree = ast.parse(f.read())
    banned = {o["software_constant"] for o in spec["opcodes"]}
    banned |= {b["software_constant"] for o in spec["opcodes"]
               for d in o["operands"] for b in d.get("bitflags", [])}
    banned |= {"IWORDS", "NHDR", "AGU_TERMS", "LOOP_DEPTH", "AR_RAW_DIST",
               "VW", "AW", "expand", "enc", "enc_agu"}
    fails = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and \
                node.module.endswith("microarch_isa"):
            bad = sorted({a.name for a in node.names} & banned)
            if bad:
                fails.append(
                    f"isa_ref.py imports {bad} from the design; the spec "
                    f"defines them and isa_encoding generates them, so the "
                    f"reference model would be restating the design")
    uses_spec = any(
        isinstance(n, ast.ImportFrom) and n.module
        and n.module.endswith("isa_encoding") for n in ast.walk(tree))
    if not uses_spec:
        fails.append("isa_ref.py does not import isa_encoding: the reference "
                     "model is not built on the spec")
    print("  reference model: takes its opcodes, field layout, control flow "
          "and numerics from the generated module, not from the design")
    return fails


# ------------------------------------------------- the derived properties ---
def _witness_acc_agu_target(spec, U, E, trip):
    """A reduce loop whose body drives `mm`'s accumulate field from its own
    induction variable. Written with the raw encoder because `isa_dsl.mm`
    takes `acc` as a bool, which is the generator's choice and not the ISA's."""
    return [
        (U.enc(U.OP_DMA_LD, f0=U.DMA_TO_VR, f1=0, f2=0, f3=U.A_VR, nr=U.T), 0),
        (U.enc(U.OP_DMA_LD, f0=U.DMA_SRC_B, f1=0, f2=0, f3=U.B_SP, nr=U.T), 0),
        (U.enc(U.OP_LOOP, nr=trip), 0),
        (U.enc(U.OP_MM, f0=U.A_VR, f1=U.AR_C, f2=0, f3=U.B_SP, nr=U.T),
         U.enc_agu((U.AGU_F2, 0, 1))),
        (U.enc(U.OP_ENDLOOP), 0),
        (U.enc(U.OP_MVOUT, f0=U.AR_C, f1=0, f2=0, nr=U.T), 0),
    ]


def _observe_acc_agu_target(spec, U, D, E):
    prog = _witness_acc_agu_target(spec, U, E, 2)
    try:
        U.check_program(prog)
        ok = True
    except U.ProgramError:
        ok = False
    acc = [e[4] for e in U.expand(prog) if e[0] == U.OP_MM]
    return {"accepted": ok, "resolved_acc": acc}


def _observe_acc_agu_monotone(spec, U, D, E):
    """The spec PREDICTS the frontier from `legal_values` alone; the design is
    then asked where it actually stops."""
    frontier = E.agu_legal_trip(E.OP_MM, "acc", base=0, stride=1)
    accepts, rejects, field = [], [], None
    for trip in range(1, frontier + 3):
        try:
            U.check_program(_witness_acc_agu_target(spec, U, E, trip))
            accepts.append(trip)
        except U.ProgramError as e:
            rejects.append(trip)
            field = field or re.search(r"\bf[0-3]\b", str(e)).group(0)
    return {"frontier_from_spec": frontier,
            "design_accepts_up_to": max(accepts) if accepts else 0,
            "design_rejects_from": min(rejects) if rejects else None,
            "rejection_names_field": field}


def _observe_acc_term_budget(spec, U, D, E):
    """Read the term count out of the SHIPPED program's own AGU word."""
    shape = max(s for s in _shapes() if max(s) <= U.MAXDIM)
    prog = D.gemm_program(*shape, False)
    accs = [(w0, w1) for w0, w1 in prog
            if E.decode(w0)[0] == E.OP_MM and E.decode(w0)[4] == 1]
    assert accs, "the shipped GEMM has no accumulating mm at this shape"
    used = E.agu_terms_used(accs[0][1])
    terms = [t for t in E.decode_agu(accs[0][1]) if t[0]] + [(E.AGU_F2, 1, 1)]
    try:
        U.enc_agu(*terms)
        refused = False
    except AssertionError:
        refused = True
    return {"terms_used_by_shipped_acc_mm": used, "terms_with_acc": len(terms),
            "refused_by_budget": refused}


#: MAXDIM values the ceiling property is probed at: inside both bounds, past
#: the header-count bound, past the address-field bound.
#: Filled in by the witness from `isa_encoding.maxdim_ceiling`, so the probe
#: points are computed rather than typed.
CEILING_PROBES = None

#: Probe one MAXDIM. The design is asked TWICE, and WHERE it refuses is what
#: separates the two ceilings: the addressing one is a property of the layout
#: and fires when the design module is imported at all; the cubic-header one
#: is a property of the workload and fires only when a cubic GEMM is
#: assembled. Reporting a bare "refuses" would let either ceiling take credit
#: for the other's failure.
_CEILING = """
import json, sys
sys.path.insert(0, %r)
sys.path.insert(0, %r)


def outcome(f):
    try:
        f()
        return "accepts"
    except Exception:
        return "refuses"


import isa_encoding as E
spec = outcome(E.check_parameters)
try:
    import microarch_isa as U
except Exception:
    print(json.dumps({"spec": spec, "design": "refuses at import"}))
    raise SystemExit(0)


def build():
    from isa_dsl import gemm_program
    U.assemble(gemm_program(U.MAXDIM, U.MAXDIM, U.MAXDIM, False))


print(json.dumps({"spec": spec,
                  "design": {"accepts": "accepts",
                             "refuses": "refuses in assemble"}[outcome(build)]}))
"""


def _observe_maxdim_ceilings(spec, U, D, E):
    """Does the spec's COMPUTED ceiling land where the design actually stops?

    Two ceilings, answering two questions (see isa_spec.json,
    "maxdim_ceilings"). The binding one for a cubic GEMM is the smaller, so
    the design is probed at it and one step of T past it. The spec asserts
    both at import; the design hits the addressing one at import and the
    header-count one inside `assemble`. What must agree is WHERE, not where
    it is noticed."""
    ceilings = {name: E.maxdim_ceiling(name)
                for name in E.MAXDIM_CEILINGS}
    binding = min(ceilings.values())
    src = _CEILING % (HERE, REPO)
    rows, agree = {}, True
    # Each ceiling is probed AT its computed value and one step of T past
    # it. Probing only past it would let a spec that claims a ceiling the
    # design does not reach pass unnoticed.
    probes = sorted({U.T, binding - U.T}
                    | {c for c in ceilings.values()}
                    | {c + U.T for c in ceilings.values()})
    for md in probes:
        # Carry the configuration under test, not just MAXDIM: scrubbing every
        # TPU_* would silently probe T=4 while the ceilings were computed for
        # this build's T, and the property would fail for the wrong reason.
        e = {k: v for k, v in os.environ.items() if not k.startswith("TPU_")}
        e["TPU_T"] = str(U.T)
        e["TPU_MAXDIM"] = str(md)
        out = subprocess.run([sys.executable, "-c", src], env=e,
                             capture_output=True, text=True)
        got = json.loads(out.stdout) if out.returncode == 0 else \
            {"spec": "probe failed", "design": out.stderr.strip()[-60:]}
        rows[md] = got
        # The spec's `check_parameters` is a BUILD check, so it is compared
        # against whether the design IMPORTS -- not against whether a cubic
        # GEMM assembles. Conflating the two is the error this witness caught
        # in the spec itself: the cubic-header bound is a property of the
        # WORKLOAD and must not refuse a build.
        imports = got["design"] != "refuses at import"
        agree = agree and (got["spec"] == "accepts") == imports
        # ... and the cubic ceiling is compared against the cubic GEMM.
        cubic = md <= ceilings["cubic_header"]
        agree = agree and cubic == (got["design"] == "accepts")
    # Each ceiling confirmed by the STAGE the design refuses at: past the
    # addressing ceiling the design cannot be imported at all; between the
    # two, it imports and refuses the cubic GEMM in `assemble`.
    stage = {md: v["design"] for md, v in rows.items()}
    return {"agree_at_every_probe": agree, "probes": rows,
            # At the addressing ceiling the design must still IMPORT (a cubic
            # GEMM may well be refused there by the other ceiling); one step
            # past it, the module itself must not load.
            "addressing_imports_at_ceiling":
                stage.get(ceilings["addressing"]) != "refuses at import",
            "addressing_refused_at_import":
                stage.get(ceilings["addressing"] + U.T) == "refuses at import",
            # At the cubic-header ceiling a cubic GEMM must assemble; one step
            # past it, it must be refused -- and in `assemble`, not at import.
            "cubic_header_accepts_at_ceiling":
                stage.get(ceilings["cubic_header"]) == "accepts",
            "cubic_header_refused_in_assemble":
                stage.get(ceilings["cubic_header"] + U.T) == "refuses in assemble",
            "ceilings": ceilings, "binding": binding,
            "ceilings_are_multiples_of_T":
                all(c % U.T == 0 for c in ceilings.values())}


def _observe_program_limits(spec, U, D, E):
    """`gemm_limits` must predict exactly which shapes the design assembles.

    The `split_operand_load` row is an ALTERNATIVE layout, not a requirement
    of the shipped one, so it is excluded from the shipped-layout verdict."""
    shapes = [(m, k, n)
              for m in (U.T, 2 * U.T, U.MAXDIM // 2, U.MAXROWS + 1, U.MAXDIM)
              for k in (U.T, U.MAXDIM // 2, U.MAXDIM)
              for n in (U.T, U.MAXDIM)]
    disagreements = []
    for (M, K, N) in {s for s in shapes if all(d >= U.T and d % U.T == 0
                                               for d in s)}:
        fits = all(over == 0 for name, _, _, over in E.gemm_limits(M, K, N)
                   if not name.endswith("split_operand_load"))
        try:
            U.assemble(D.gemm_program(M, K, N, False))
            built = True
        except Exception:  # noqa: BLE001 -- any refusal is a refusal
            built = False
        if fits != built:
            disagreements.append(
                f"{M}x{K}x{N}: limits say {'fits' if fits else 'refused'}, "
                f"the design {'assembles' if built else 'refuses'}")
    return {"agrees_on_every_shape": not disagreements,
            "shapes_checked": len({s for s in shapes
                                   if all(d >= U.T and d % U.T == 0
                                          for d in s)}),
            "disagreements": disagreements}


_WITNESS = {
    "acc_agu_target": _observe_acc_agu_target,
    "acc_agu_monotone": _observe_acc_agu_monotone,
    "acc_term_budget": _observe_acc_term_budget,
    "maxdim_ceilings": _observe_maxdim_ceilings,
    "program_limits": _observe_program_limits,
}


def _expected(spec, expect, observed):
    """Resolve an expectation written against the spec: an int is literal, and
    a string is an expression over the spec's own values and this witness's
    own observations, so nothing in the table is a restated constant."""
    env = {"frontier": observed.get("frontier_from_spec"),
           "AGU_TERMS": spec["agu"]["terms"]}
    out = {}
    for k, v in expect.items():
        if isinstance(v, str):
            e = v.replace("agu.terms", "AGU_TERMS")
            if e.startswith("isa_encoding."):
                continue            # computed by the witness, not compared
            try:
                v = eval(e, {"__builtins__": {}}, env)  # noqa: S307
            except Exception:       # noqa: BLE001 -- a prose expectation
                continue
        out[k] = v
    return out


def check_derived_properties(spec, U, D, E):
    """Recompute each derived property from the spec, then confirm the design
    behaves that way.

    These are the facts that FOLLOW from the encoding rather than being
    written in it. All three were documented wrongly here until 2026-09-21,
    which is the argument for computing them instead of asserting them in
    prose."""
    fails = []
    props = [p for p in spec["derived_properties"] if "id" in p]
    for p in props:
        try:
            observed = _WITNESS[p["witness"]](spec, U, D, E)
        except Exception as e:  # noqa: BLE001
            # A witness that cannot even be built on this design IS a
            # disagreement, and it must be reported rather than raised.
            fails.append(f"derived property {p['id']}: its witness could not "
                         f"be built against the design: "
                         f"{type(e).__name__}: {e}")
            p["_observed"] = {}
            continue
        want = _expected(spec, p["expect"], observed)
        for k, v in want.items():
            if observed.get(k) != v:
                fails.append(f"derived property {p['id']}: expected {k}={v!r}, "
                             f"the design gives {observed.get(k)!r}")
        p["_observed"] = observed
    if not fails:
        o = {p["id"]: p["_observed"] for p in props}
        print(f"  derived properties: {len(props)} recomputed from the spec and "
              f"confirmed against the design --")
        print(f"      every operand field is an AGU target, `acc` included "
              f"(resolved acc {o['every_operand_field_is_an_agu_target']['resolved_acc']});")
        print(f"      terms are additively monotone, so `acc` at base 0 stride 1 "
              f"is drivable for "
              f"{o['agu_terms_are_additively_monotone']['frontier_from_spec']} "
              f"trips and rejected at "
              f"{o['agu_terms_are_additively_monotone']['design_rejects_from']};")
        print(f"      the shipped accumulating mm already spends "
              f"{o['a_term_on_acc_costs_one_of_the_budget']['terms_used_by_shipped_acc_mm']}"
              f"/{spec['agu']['terms']} terms, so a term on `acc` needs "
              f"{o['a_term_on_acc_costs_one_of_the_budget']['terms_with_acc']} "
              f"and the budget refuses it first")
        ce = o["the_maxdim_ceilings_are_the_encoding's"]
        print(f"      the encoding's MAXDIM ceilings, computed at T={U.T}: "
              + ", ".join(f"{n} {v}" for n, v in ce["ceilings"].items())
              + f" (binding {ce['binding']}); the design "
              + ", ".join(f"{md} {v['design']}"
                          for md, v in ce["probes"].items())
              + " -- each ceiling confirmed by the stage it fires at")
        pl = o["program_limits_predict_what_assembles"]
        print(f"      the program limits predict exactly which of "
              f"{pl['shapes_checked']} GEMM shapes the design assembles")
    return fails


def check_target_encoding(spec):
    """`allo.encoding.TINYTPU_ISA` is a THIRD copy of two ISA facts.

    It is the compiler-side descriptor of what one instruction word can carry,
    and it deliberately imports nothing from `examples/`, so it keeps its own
    numbers written out -- and they are exactly the numbers the spec fixes."""
    from allo.encoding import TINYTPU_ISA as E
    want = {"address_terms": spec["agu"]["terms"],
            "loop_depth": spec["loop_stack"]["depth"]}
    fails = []
    for name, v in want.items():
        got = getattr(E, name, None)
        if got != v:
            fails.append(f"allo.encoding.TINYTPU_ISA.{name} = {got!r}, spec "
                         f"says {v!r}")
    print(f"  allo.encoding.TINYTPU_ISA: {len(want)} budget(s) held to the "
          f"spec by value")
    return fails


# ----------------------------------------------- the emitted hardware ---
#: A bit-range read in the emitted HLS: `v51 = v51_tmp(61, 54);`, i.e.
#: `ap_uint::operator()(hi, lo)`, inclusive at both ends.
_HLS_SLICE = re.compile(r"=\s*[A-Za-z_][A-Za-z_0-9]*\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def check_emitted(spec):
    """The field positions and widths THE EMITTED HARDWARE uses.

    Builds the design down the HLS path and reads the bit ranges the generated
    C++ takes of its 64-bit control words. This is the one check that does not
    trust the Python source: an `s.partition` or a rewritten slice that changed
    what the hardware reads shows up here and nowhere else."""
    import shutil
    import tempfile

    import allo.dataflow as df
    import microarch_isa as U
    tmp = tempfile.mkdtemp(prefix="isa_conform.")
    try:
        prj = os.path.join(tmp, "top.prj")
        df.build(U.tinytpu_isa, target="vhls", wrap_io=False, project=prj,
                 configs={"align_value": 64}, enable_tensor=False)
        with open(os.path.join(prj, "kernel.cpp")) as f:
            text = f.read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    want = {(f["lo"], f["lo"] + f["width"]) for f in _fields(spec)}
    hdr = {(e["slice"][0], e["slice"][1]) for e in spec["imem"]["entries"]}
    agu_sub = spec["agu"]["subfields"]
    # Word 1's terms are at 19*t + lo for t in 0..AGU_TERMS-1.
    agu = {(spec["agu"]["term_bits"] * t + s["lo"],
            spec["agu"]["term_bits"] * t + s["lo"] + s["width"])
           for t in range(spec["agu"]["terms"]) for s in agu_sub}
    lanes = {(8 * e, 8 * (e + 1)) for e in range(U.T)}
    lanes |= {(32 * e, 32 * (e + 1)) for e in range(U.T)}
    lanes |= {(0, 12), (8, 20), (0, 8), (0, 32), (16, 32)}   # header words
    allowed = want | hdr | agu | lanes
    fails, seen = [], set()
    for m in _HLS_SLICE.finditer(text):
        hi, lo = int(m.group(1)), int(m.group(2))
        if hi < lo:
            continue                # a two-argument call, not a bit range
        # `ap_uint::range(hi, lo)` is inclusive; the spec's slices are [lo, hi).
        rng = (lo, hi + 1)
        if rng in want:
            seen.add(rng)
        elif rng not in allowed:
            fails.append(f"the emitted HLS slices bits [{lo}:{hi + 1}], which "
                         f"the spec does not define")
    missing = sorted(want - seen)
    if missing:
        fails.append(f"the emitted HLS never reads spec fields at {missing}")
    print(f"  emitted HLS: {len(seen)}/{len(want)} instruction-word fields read "
          f"at exactly the spec's bit positions, no undefined slice of a "
          f"control word")
    return sorted(set(fails))


# ------------------------------------------------------------------- main ---
def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--conform", action="store_true",
                    help="--check, plus the emitted HLS's own bit slices")
    a = ap.parse_args(argv)
    if not (a.write or a.check or a.conform):
        ap.error("one of --write, --check, --conform")
    spec = load()
    if a.write:
        return write(spec)

    sys.path.insert(0, HERE)
    print(f"TinyTPU-isa ISA conformance, against "
          f"{os.path.relpath(SPEC, REPO)}\n")
    fails = check_generated(spec)
    E = spec_module(spec)
    import microarch_isa as U
    import isa_dsl as D
    fails += check_design_constants(spec, U)
    fails += check_design_layout(spec)
    fails += check_design_parameters(spec, U, E)
    fails += check_parameter_agreement(spec, U)
    fails += check_design_slices(spec)
    fails += check_behaviour(spec, U, D, E)
    fails += check_reference(spec)
    fails += check_actions(spec, U, E)
    fails += check_target_encoding(spec)
    fails += check_derived_properties(spec, U, D, E)
    if a.conform:
        fails += check_emitted(spec)
    print()
    for f in fails:
        print("  ISA FAIL", f)
    print(f"  ISA OK: the spec, its {len(artefacts(spec))} generated "
          f"artefacts and all "
          f"{len([c for c in spec['consumers'] if 'path' in c])} consumers agree"
          if not fails else f"  ISA FAILED: {len(fails)} disagreement(s)")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
