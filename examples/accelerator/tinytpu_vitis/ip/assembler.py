# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The assembler: run the program's control flow at assembly time, reject what
the hardware cannot run, and emit the per-unit work counts the header carries.

The units loop over their own work count, so the header must carry the DYNAMIC
count, and the assembler is where the dispatch rules of `sequencer`, the
flattened row loops of the other units and the accumulator's dependence claim
are all made true. A unit promised the wrong count hangs; a program that reads
an unwritten row, or an `ar` row too soon after writing it, is a wrong answer.
Both are rejected here. See ``docs/source/designs/tinytpu_isa.rst``.
"""

from collections import namedtuple

from .isa import (AGU_TERMS, LOOP_DEPTH, OPCODE_NAMES, OP_DMA_LD, OP_DMA_ST,
                  OP_ENDLOOP, OP_LOOP, OP_MM, OP_MVOUT, OP_NOP, OP_VADD,
                  OP_VLD, OP_VRELU, NHDR, DMA_SRC_B, DMA_TO_VR)

# The minimum read-after-write distance, in `accu` iterations, that makes the
# `inter false` dependence claim on `ar` true. Measured in RTL cosim with
# `isa_dsl.ar_distance_program(d)`: d=1 -> 4 cells wrong, d=2 -> 20 wrong,
# d=3, 4, 5 -> 0. This is the first safe distance plus one of margin.
AR_RAW_DIST = 4

#: One instruction the sequencer will issue, with its address fields resolved.
Issue = namedtuple("Issue", "op nr f0 f1 f2 f3")
#: The same, with where it came from: `pc` its static slot, `ivs` the live
#: induction variables of the loops open around it.
TracedIssue = namedtuple("TracedIssue", "pc ivs op nr f0 f1 f2 f3")


class ProgramError(ValueError):
    """A program the hardware would run to a wrong answer, or a hang."""


class Assembler:
    """Bound to one parameter set: memory sizes bound every address it checks.

    `check` is static and exact rather than conservative -- the program has no
    data-dependent control flow, so `trace` visits every issue the sequencer
    will make, with the addresses it will resolve.
    """

    def __init__(self, params, ar_raw_dist=AR_RAW_DIST):
        self.p = params
        self.ar_raw_dist = ar_raw_dist
        assert ar_raw_dist <= params.T, (
            "a T-row GEMM must satisfy the accumulator contract")
        self.size = {"spad": params.SPAD_ROWS, "vr": params.NVR,
                     "ar": params.NAR}

    def expand(self, prog):
        """One `Issue` per instruction the sequencer will issue, with the AGU
        resolved exactly as it resolves it."""
        return [Issue(*traced[2:]) for traced in self.trace(prog)]

    def trace(self, prog):
        """`expand`, with each issue's origin: the one place the assembler
        mirrors the sequencer's control flow."""
        pc = 0
        stack = []
        live_iv = [0] * LOOP_DEPTH
        guard = 0
        while pc < len(prog):
            guard += 1
            if guard > 1 << 22:
                raise AssertionError("program does not terminate")
            control_word, agu_word = prog[pc]
            op = control_word & 0x3F
            if op == OP_LOOP:
                live_iv[len(stack)] = 0
                stack.append([pc + 1, 0, (control_word >> 54) & 0xFF])
                pc += 1
            elif op == OP_ENDLOOP:
                body, iteration, trip = frame = stack[-1]
                frame[1] = iteration = iteration + 1
                if iteration < trip:
                    live_iv[len(stack) - 1] = iteration
                    pc = body
                else:
                    stack.pop()
                    pc += 1
            else:
                fields = [(control_word >> sh) & 0xFFF
                          for sh in (6, 18, 30, 42)]
                for term in range(AGU_TERMS):
                    base = 19 * term
                    target = (agu_word >> base) & 0xF
                    level = (agu_word >> (base + 4)) & 0x7
                    stride = (agu_word >> (base + 7)) & 0xFFF
                    if target != 0:
                        fields[target - 1] += live_iv[level] * stride
                yield TracedIssue(pc, tuple(live_iv[:len(stack)]), op,
                                  (control_word >> 54) & 0xFF, *fields)
                pc += 1

    # pylint: disable=too-many-branches, too-many-statements, too-many-locals
    def check(self, prog):
        """Reject a program the hardware cannot run correctly.

        Raises `ProgramError` naming the static instruction, the loop
        iteration, and the rows; returns None."""
        p = self.p
        if not prog:
            raise ProgramError("empty program")
        depth = 0
        for pc, (control_word, agu_word) in enumerate(prog):
            op = control_word & 0x3F
            name = OPCODE_NAMES.get(op)
            where = f"instruction {pc} ({name or f'opcode {op}'})"
            if name is None or op == OP_DMA_ST:
                raise ProgramError(f"{where}: not an opcode this machine executes")
            if op == OP_LOOP:
                if depth >= LOOP_DEPTH:
                    raise ProgramError(f"{where}: nesting exceeds LOOP_DEPTH={LOOP_DEPTH}")
                if (control_word >> 54) & 0xFF < 1:
                    raise ProgramError(f"{where}: trip count 0 still runs the body once")
                depth += 1
            elif op == OP_ENDLOOP:
                if depth == 0:
                    raise ProgramError(f"{where}: endloop with no open loop")
                depth -= 1
            for term in range(AGU_TERMS):
                target = (agu_word >> (19 * term)) & 0xF
                level = (agu_word >> (19 * term + 4)) & 0x7
                if target == 0:
                    continue
                if (op in (OP_LOOP, OP_ENDLOOP, OP_NOP) or target > 4
                        or level >= depth):
                    raise ProgramError(
                        f"{where}: AGU term {term} targets field {target - 1} "
                        f"with loop level {level}, but {depth} loop(s) are open "
                        f"here -- the sequencer would use a stale "
                        f"live_iv[{level}]")
        if depth:
            raise ProgramError(f"{depth} loop(s) never closed")

        written = {"spad": [False] * p.SPAD_ROWS, "vr": [False] * p.NVR,
                   "ar": [False] * p.NAR}
        # The accu step each `ar` row was last written on, and how many steps
        # the program has issued so far: the distance contract counts in steps.
        ar_written_at = [-self.ar_raw_dist] * p.NAR
        accu_step = 0
        size = self.size

        for pc, ivs, op, nr, f0, f1, f2, f3 in self.trace(prog):
            where = (f"instruction {pc} ({OPCODE_NAMES[op]}"
                     + (f", loop ivs {list(ivs)}" if ivs else "") + ")")

            def span(mem, base, n):
                if base < 0 or base + n > size[mem]:
                    raise ProgramError(f"{where}: {mem} rows {base}..{base + n - 1} "
                                       f"outside 0..{size[mem] - 1}")
                return range(base, base + n)

            def need(mem, rows, what):
                bad = [row for row in rows if not written[mem][row]]
                if bad:
                    raise ProgramError(
                        f"{where}: reads {mem} row(s) {bad} as {what} before any "
                        f"instruction wrote them. {mem} is not cleared by the "
                        f"hardware; see the write-before-read contract.")

            if op == OP_NOP:
                continue
            for v, fld in ((f0, "f0"), (f1, "f1"), (f2, "f2"), (f3, "f3")):
                if v >= 1 << 11:
                    raise ProgramError(f"{where}: AGU-resolved {fld}={v} is "
                                       f"outside the 0..2047 range `enc` admits")
            if nr < 1:
                raise ProgramError(f"{where}: nr=0 desynchronises the unit's "
                                   f"flat row loop; drop the instruction instead")

            def ar_read(row, step, what):
                need("ar", [row], what)
                if step - ar_written_at[row] < self.ar_raw_dist:
                    raise ProgramError(
                        f"{where}: reads ar row {row} as {what} "
                        f"{step - ar_written_at[row]} accu step(s) after it was "
                        f"written; the accumulator's dependence claim needs "
                        f">= AR_RAW_DIST={self.ar_raw_dist} (see the "
                        f"accumulator distance contract)")

            def ar_write(row, step):
                written["ar"][row] = True
                ar_written_at[row] = step

            if op == OP_DMA_LD:
                if f0 not in (0, 1, 2, 3):
                    raise ProgramError(f"{where}: f0={f0}, must be source (0 A, 1 B) "
                                       f"| destination (0 spad, 2 vr)")
                if f2 >= p.WPR or f1 + nr > p.MAXDIM:
                    raise ProgramError(f"{where}: DRAM rows {f1}..{f1 + nr - 1}, "
                                       f"col block {f2} outside the "
                                       f"{p.MAXDIM}x{p.MAXDIM} operand")
                memory = "vr" if f0 & DMA_TO_VR else "spad"
                for row in span(memory, f3, nr):
                    written[memory][row] = True
            elif op == OP_VLD:
                # A copy of an unwritten row is unwritten too.
                for dest, source in zip(span("vr", f0, nr),
                                        span("spad", f1, nr)):
                    written["vr"][dest] = written["spad"][source]
            elif op == OP_MM:
                if f2 not in (0, 1):
                    raise ProgramError(f"{where}: f2={f2}, must be 0 (overwrite) or 1 (accumulate)")
                need("spad", span("spad", f3, p.T), "weights")
                need("vr", span("vr", f0, nr), "activations")
                # Row by row, one accu step each, as `accu` runs it.
                for i, row in enumerate(span("ar", f1, nr)):
                    if f2 == 1:
                        ar_read(row, accu_step + i, "the accumulate base")
                    ar_write(row, accu_step + i)
                accu_step += nr
            elif op == OP_VADD:
                first = span("ar", f1, nr)
                second = span("ar", f2, nr)
                dest = span("ar", f0, nr)
                for i in range(nr):              # two accu steps per row
                    ar_read(first[i], accu_step + 2 * i, "a source")
                    ar_read(second[i], accu_step + 2 * i + 1, "a source")
                    ar_write(dest[i], accu_step + 2 * i + 1)
                accu_step += 2 * nr
            elif op == OP_VRELU:
                source, dest = span("ar", f1, nr), span("ar", f0, nr)
                for i in range(nr):
                    ar_read(source[i], accu_step + i, "a source")
                    ar_write(dest[i], accu_step + i)
                accu_step += nr
            elif op == OP_MVOUT:
                for i, row in enumerate(span("ar", f0, nr)):
                    ar_read(row, accu_step + i, "the value to retire")
                accu_step += nr
                if f2 >= p.WPR or f1 + nr > p.MAXDIM:
                    raise ProgramError(f"{where}: C rows {f1}..{f1 + nr - 1}, col "
                                       f"block {f2} outside the "
                                       f"{p.MAXDIM}x{p.MAXDIM} result")

    def assemble(self, prog, check=True):
        """Two words per instruction, behind a header of dynamic per-unit
        counts.

            imem[0] static instruction count   imem[4] mm count | mm rows << 16
            imem[1] dma_ld  rows               imem[5] accu   steps
            imem[2] spm     rows               imem[6] dma_st rows
            imem[3] vru     words              imem[7] A rows | B rows << 16

        These are WORK counts, not instruction counts, with the two per-unit
        adjustments the flattened bodies make: `spm` charges an `mm` T + 1
        iterations whatever its own row count, `accu` charges a `vadd` two
        steps per row, and a `dma_ld` goes to `spm` or to `vru` by its
        destination bit, never both.

        `check=False` exists only so a test can put a known-bad program on the
        machine and watch it fail; nothing that ships passes it.
        """
        p = self.p
        if check:
            self.check(prog)
        issues = self.expand(prog)

        def rows(*ops):
            return sum(issue.nr for issue in issues if issue.op in ops)

        def count(*ops):
            return sum(1 for issue in issues if issue.op in ops)

        def dram_span(source):
            # The DRAM row span `dma_ld` must burst for one operand matrix: the
            # highest row any of its `dma_ld`s names, after the AGU is resolved.
            return max([issue.f1 + issue.nr for issue in issues
                        if issue.op == OP_DMA_LD
                        and (issue.f0 & DMA_SRC_B) == source] + [0])

        a_span, b_span = dram_span(0), dram_span(1)
        assert a_span <= p.MAXDIM and b_span <= p.MAXDIM, (
            f"dma_ld row span {a_span}/{b_span} exceeds MAXDIM={p.MAXDIM}")

        n_mm = count(OP_MM)
        mm_rows = rows(OP_MM)
        assert n_mm < (1 << 15) and mm_rows < (1 << 15), "array counts overflow"
        rows_to_vr = sum(issue.nr for issue in issues
                         if issue.op == OP_DMA_LD and issue.f0 & DMA_TO_VR)
        rows_to_spad = rows(OP_DMA_LD) - rows_to_vr
        header = [len(prog),
                  rows(OP_DMA_LD),
                  rows_to_spad + rows(OP_VLD) + n_mm * (p.T + 1),
                  rows_to_vr + rows(OP_VLD) + mm_rows,
                  n_mm | (mm_rows << 16),
                  rows(OP_MM, OP_VRELU, OP_MVOUT) + 2 * rows(OP_VADD),
                  rows(OP_MVOUT),
                  a_span | (b_span << 16)]
        assert len(header) == NHDR
        # Every count is read back through a 16-bit slice, which used to
        # extract to a signed ap_int<16>, so the usable range stops at 2^15 - 1.
        for count_field in header[1:4] + header[5:7]:
            assert 0 <= count_field < (1 << 15), (
                f"header count {count_field} does not fit 15 bits")
        words = list(header)
        for control_word, agu_word in prog:
            words.append(int(control_word))
            words.append(int(agu_word))
        assert len(words) <= p.IMEM_SIZE, (
            f"{len(words)} words > IMEM_SIZE={p.IMEM_SIZE}")
        return words
