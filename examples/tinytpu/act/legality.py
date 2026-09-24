# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Is a submitted program legal, and if not, what should the generator change?

Four layers, reported in the order a program fails them: the ISA contracts
(`microarch_isa.check_program`), the encoding limits `assemble` asserts, the
channel protocol (`kpn_model`), and the spec's own write window. Prose:
docs/source/extensions/act_specs.rst."""

import os
import re
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from examples.tinytpu import kpn_model  # noqa: E402
from examples.tinytpu.microarch_isa import (  # noqa: E402
    AGU_TERMS, AR_RAW_DIST, IMEM_SIZE, LOOP_DEPTH, MAXDIM, NAR, NVR,
    OP_MVOUT, SPAD_ROWS, T, ProgramError, assemble, check_program,
    expand,
)
from examples.tinytpu.ip.isa import OPCODE_NAMES  # noqa: E402
from examples.tinytpu.act import spec as spec_mod  # noqa: E402
from examples.tinytpu.isa_encoding import (  # noqa: E402
    FIELDS, HEADER_WORK, usable_max,
)

# The two encoding ceilings these rules quote, from the generated spec module
# rather than typed out: what a header count can promise a unit, and what an
# AGU-resolved operand field can address.
COUNT_BITS = min(hi - lo for _, (lo, hi), _, _ in HEADER_WORK)
COUNT_MAX = usable_max(COUNT_BITS)
FIELD_MAX = usable_max(dict((n, w) for n, _, w in FIELDS)["f3"]) + 1


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: str
    statement: str
    remedy: str


ISA_RULES = (
    Rule("ar.raw_distance", "accu iteration(s) after it was written",
         f"every accumulator read must come at least AR_RAW_DIST="
         f"{AR_RAW_DIST} accu iterations after the write it depends on, "
         f"because `accu` runs at II=1 on a dependence claim that is only "
         f"true at that distance",
         f"raise the offending instruction's row count to at least "
         f"{AR_RAW_DIST} and retire only the rows the spec asks for; or put "
         f"the write and the read in different ar regions and put at least "
         f"{AR_RAW_DIST} iterations of other accu work between them. An "
         f"`nr`-row instruction's first row is read {AR_RAW_DIST} iterations "
         f"later only if nr >= {AR_RAW_DIST}; a `vadd` counts two iterations "
         f"per row."),
    Rule("mem.write_before_read", "before any instruction wrote them",
         "spad, vr and ar are not cleared between programs, so a row that "
         "this program never wrote holds whatever the last one left",
         "load the rows with `dma_ld` (or `vld`) before the instruction that "
         "consumes them. If these rows are a pad you wanted to be zero, note "
         "that no opcode zeroes a row: `dma_ld` is the only writer of spad "
         "and vr and it can only copy DRAM the host filled, so a zero pad has "
         "to be part of the spec's calling convention."),
    Rule("mem.out_of_range", "outside 0..",
         f"the on-chip memories are spad {SPAD_ROWS} rows, vr {NVR}, "
         f"ar {NAR}, and an out-of-range row is silent corruption in RTL",
         "reduce the base, the row count, or the AGU stride that walked the "
         "address past the end; the message names the resolved span."),
    Rule("dram.operand_range", f"outside the {MAXDIM}x{MAXDIM} operand",
         f"A and B are {MAXDIM}x{MAXDIM} int8 images and a column block is "
         f"T={T} lanes wide, so rows must stay below {MAXDIM} and the column "
         f"block below {MAXDIM // T}",
         "split the load into several `dma_ld`s, or lower the DRAM row base "
         "the AGU resolved to."),
    Rule("dram.result_range", f"outside the {MAXDIM}x{MAXDIM} result",
         f"C is a {MAXDIM}x{MAXDIM} int8 image written a whole T={T}-lane "
         f"word at a time",
         "lower the `mvout` row base or row count; if the spec's output does "
         "not reach that far, the extra rows are a clobber, not a rounding."),
    Rule("enc.field_range", "range `enc` admits",
         f"the sequencer writes an AGU-resolved sum back into an operand "
         f"field, so a base plus its terms must stay below {FIELD_MAX}",
         f"shrink the stride, the trip count, or the base of the address that "
         f"overflowed; a layout that needs more than {FIELD_MAX} rows of one "
         f"memory cannot be encoded at all."),
    Rule("enc.zero_rows", "desynchronises the unit's flat row loop",
         "every unit is one flat loop over rows and fetches an instruction "
         "when its row counter runs out, so a zero-row instruction is fetched "
         "as if it had one row",
         "drop the instruction instead of emitting it with nr=0; an empty "
         "tile is an empty tile at generation time, not a no-op at run time."),
    Rule("loop.depth", f"nesting exceeds LOOP_DEPTH={LOOP_DEPTH}",
         f"the sequencer's loop stack is {LOOP_DEPTH} levels deep",
         f"flatten two levels into one by folding their strides, or unroll "
         f"the outermost level; a nest deeper than {LOOP_DEPTH} has no "
         f"encoding."),
    Rule("loop.trip", "trip count 0 still runs the body once",
         "the back edge is tested after the body, so a trip count below 1 "
         "still executes it",
         "guard the loop at generation time and emit nothing when the trip "
         "count is zero."),
    Rule("loop.unbalanced", "endloop with no open loop",
         "loop and endloop must balance for the sequencer's stack to unwind",
         "emit the missing endloop, or drop the stray one."),
    Rule("loop.unclosed", "never closed",
         "loop and endloop must balance for the sequencer's stack to unwind",
         "close every loop you open before the program ends."),
    Rule("loop.stale_iv", "stale iv_now",
         "an AGU term names a loop level by number, and the sequencer "
         "resolves it against whatever that level last held",
         "only let an address walk a loop that is open around the "
         "instruction; if you need the value after the loop, recompute the "
         "base instead."),
    Rule("op.unknown", "not an opcode this machine executes",
         "nine opcodes exist and dma_st is retired",
         "re-encode with one of nop, dma_ld, vld, mm, vadd, vrelu, mvout, "
         "loop, endloop; results leave through mvout."),
    Rule("op.mm_acc_field", "must be 0 (overwrite) or 1 (accumulate)",
         "mm's f2 selects whether the accumulator is overwritten or added to",
         "set f2 to 0 on the first k-tile of an output tile and 1 on the "
         "rest; there is no predicate on the induction variable, which is why "
         "the first tile is peeled."),
    Rule("op.dma_ld_field", "must be source (0 A, 1 B)",
         "dma_ld's f0 is a source bit and a destination bit",
         "use 0 or 1 for A or B, plus 2 to send it to the vregs instead of "
         "the scratchpad."),
    Rule("prog.empty", "empty program",
         "the sequencer fetches at least one instruction",
         "emit at least one instruction."),
)

ENCODING_RULES = (
    Rule("imem.overflow", f"words > IMEM_SIZE={IMEM_SIZE}",
         f"instruction memory is {IMEM_SIZE} 64-bit words, header included, "
         f"at two words per instruction",
         "roll the repeated work into `loop`/`endloop` with AGU terms instead "
         "of unrolling it; that is what the loop stack is for."),
    Rule("hdr.count_overflow", f"does not fit {COUNT_BITS - 1} bits",
         f"each per-unit work count in the header is read back through a "
         f"{COUNT_BITS}-bit slice",
         f"split the workload into several programs; one program cannot "
         f"promise a unit more than {COUNT_MAX} work items."),
    Rule("hdr.array_overflow", "array counts overflow",
         "the array's mm count and mm row count share one header word",
         "split the workload into several programs."),
    Rule("dma.span", "dma_ld row span",
         f"the header tells `dma_ld` how many DRAM rows of A and of B to "
         f"burst before the first instruction arrives, and the operands are "
         f"{MAXDIM} rows",
         "lower the resolved DRAM row base or row count of the `dma_ld` that "
         "reaches furthest."),
)

PROTOCOL_RULE = Rule(
    "kpn.protocol", "",
    "each unit loops over the work count the header promises it, so a unit "
    "sent more than its count leaves tokens behind and one sent less waits "
    "forever",
    "this is an assembler/microarchitecture mismatch rather than a program "
    "bug: report it with the blocked process below rather than working "
    "around it.")

SPEC_RULES = (
    Rule("spec.write_window", "",
         "the spec's output region, plus the rest of its column block when "
         "write_window is column_block, is the only part of C a program may "
         "write; everything else must arrive as the host left it",
         "lower the `mvout` row count to the spec's output rows and retire "
         "only the column blocks the output covers. Padding rows for the "
         "accumulator's distance contract is fine -- pad the `mm`, not the "
         "`mvout`."),
    Rule("spec.output_incomplete", "",
         "every byte of the spec's output region has to be written by some "
         "`mvout`",
         "add the missing `mvout`s, or widen the row count or column blocks "
         "of the ones you have."),
)

ALL_RULES = {r.name: r for r in
             ISA_RULES + ENCODING_RULES + SPEC_RULES + (PROTOCOL_RULE,)}


@dataclass(frozen=True)
class Rejection:
    layer: str
    rule: Rule
    reason: str
    context: str

    def __str__(self):
        head = f"REJECTED [{self.layer} / {self.rule.name}]"
        return "\n".join([head,
                          f"  what happened: {self.reason}",
                          f"  the rule:      {self.rule.statement}",
                          f"  what to change: {self.rule.remedy}"]
                         + ([f"  in context:\n{self.context}"]
                            if self.context else []))


def instruction_line(prog, pc):
    w0, w1 = prog[pc]
    op = w0 & 0x3F
    fields = [(w0 >> sh) & 0xFFF for sh in (6, 18, 30, 42)]
    terms = []
    for t in range(AGU_TERMS):
        target = (w1 >> (19 * t)) & 0xF
        if target:
            terms.append(f"f{target - 1}+=iv{(w1 >> (19 * t + 4)) & 0x7}"
                         f"*{(w1 >> (19 * t + 7)) & 0xFFF}")
    return (f"    {pc:3d}  {OPCODE_NAMES.get(op, f'op{op}'):8s} "
            f"nr={(w0 >> 54) & 0xFF:3d} "
            f"f={fields}  {' '.join(terms)}")


def around(prog, pc, span=2):
    lo, hi = max(0, pc - span), min(len(prog), pc + span + 1)
    return "\n".join(instruction_line(prog, i) + ("   <-- here" if i == pc else "")
                     for i in range(lo, hi))


def _classify(rules, text, prog):
    for rule in rules:
        if rule.pattern and rule.pattern in text:
            pc = re.search(r"instruction (\d+)", text)
            ctx = around(prog, int(pc.group(1))) if pc and prog else ""
            return rule, ctx
    return None, ""


def write_footprint(prog):
    """The bytes of C every `mvout` in the dynamic stream touches."""
    touched = np.zeros((MAXDIM, MAXDIM), bool)
    for op, nr, f0, f1, f2, f3 in expand(prog):
        if op == OP_MVOUT:
            touched[f1:f1 + nr, f2 * T:(f2 + 1) * T] = True
    return touched


def spec_footprint(sp):
    """`(allowed, required)` masks of C for one spec."""
    allowed = np.zeros((MAXDIM, MAXDIM), bool)
    required = np.zeros((MAXDIM, MAXDIM), bool)
    wrs, wcs = spec_mod.write_window(sp)
    allowed[wrs, wcs] = True
    _, rs, cs = spec_mod.region(sp, sp["output"])
    required[rs, cs] = True
    return allowed, required


def check(sp, prog):
    """The first `Rejection` this program earns, or None."""
    try:
        check_program(prog)
    except ProgramError as e:
        rule, ctx = _classify(ISA_RULES, str(e), prog)
        return Rejection("isa", rule or Rule("isa.contract", "", "an ISA "
                                             "contract check_program enforces",
                                             "read the message above"),
                         str(e), ctx)
    try:
        assemble(prog)
    except AssertionError as e:
        rule, ctx = _classify(ENCODING_RULES, str(e), prog)
        return Rejection("encoding",
                         rule or Rule("encoding.limit", "", "an encoding limit "
                                      "`assemble` asserts", "read the message "
                                      "above"), str(e), ctx)
    good, report = kpn_model.run(prog)
    if not good:
        return Rejection("protocol", PROTOCOL_RULE, report[0],
                         "\n".join("    " + line for line in report[1:]))
    if sp is None:
        return None
    touched = write_footprint(prog)
    allowed, required = spec_footprint(sp)
    stray = touched & ~allowed
    if stray.any():
        rows = sorted(set(np.nonzero(stray)[0].tolist()))
        cols = sorted(set(np.nonzero(stray)[1].tolist()))
        return Rejection("spec", ALL_RULES["spec.write_window"],
                         f"{int(stray.sum())} bytes of C outside the spec's "
                         f"write window are written: rows {rows}, columns "
                         f"{cols}", "")
    missing = required & ~touched
    if missing.any():
        rows = sorted(set(np.nonzero(missing)[0].tolist()))
        cols = sorted(set(np.nonzero(missing)[1].tolist()))
        return Rejection("spec", ALL_RULES["spec.output_incomplete"],
                         f"{int(missing.sum())} bytes of the spec's output "
                         f"region are never written: rows {rows}, columns "
                         f"{cols}", "")
    return None


if __name__ == "__main__":
    print(f"{len(ALL_RULES)} rules, by layer:")
    for layer, rules in (("isa", ISA_RULES), ("encoding", ENCODING_RULES),
                         ("protocol", (PROTOCOL_RULE,)), ("spec", SPEC_RULES)):
        for r in rules:
            print(f"  {layer:8s} {r.name}")
