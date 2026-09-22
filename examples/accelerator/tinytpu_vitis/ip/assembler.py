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

from .isa import (AGU_TERMS, LOOP_DEPTH, OPCODE_NAMES, OP_DMA_LD, OP_DMA_ST,
                  OP_ENDLOOP, OP_LOOP, OP_MM, OP_MVOUT, OP_NOP, OP_VADD,
                  OP_VLD, OP_VRELU, NHDR, DMA_SRC_B, DMA_TO_VR)

# The minimum read-after-write distance, in `accu` iterations, that makes the
# `inter false` dependence claim on `ar` true. Measured in RTL cosim with
# `isa_dsl.ar_distance_program(d)`: d=1 -> 4 cells wrong, d=2 -> 20 wrong,
# d=3, 4, 5 -> 0. This is the first safe distance plus one of margin.
AR_RAW_DIST = 4


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
        """One `(opcode, rows, f0, f1, f2, f3)` tuple per instruction the
        sequencer will issue, with the AGU resolved as it resolves it."""
        return [e[2:] for e in self.trace(prog)]

    def trace(self, prog):
        """`expand`, with where each dynamic issue came from: yields
        `(pc, ivs, op, nr, f0, f1, f2, f3)`, `ivs` the live induction
        variables. The one place the assembler mirrors the sequencer."""
        pc = 0
        stack = []
        iv_now = [0] * LOOP_DEPTH
        guard = 0
        while pc < len(prog):
            guard += 1
            if guard > 1 << 22:
                raise AssertionError("program does not terminate")
            w0, w1 = prog[pc]
            op = w0 & 0x3F
            if op == OP_LOOP:
                trip = (w0 >> 54) & 0xFF
                iv_now[len(stack)] = 0
                stack.append([pc + 1, 0, trip])
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
                f = [(w0 >> sh) & 0xFFF for sh in (6, 18, 30, 42)]
                for t in range(AGU_TERMS):
                    base = 19 * t
                    tw = (w1 >> base) & 0xF
                    lw = (w1 >> (base + 4)) & 0x7
                    st = (w1 >> (base + 7)) & 0xFFF
                    if tw != 0:
                        f[tw - 1] += iv_now[lw] * st
                yield (pc, tuple(iv_now[:len(stack)]), op, (w0 >> 54) & 0xFF,
                       f[0], f[1], f[2], f[3])
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
        for pc, (w0, w1) in enumerate(prog):
            op = w0 & 0x3F
            name = OPCODE_NAMES.get(op)
            where = f"instruction {pc} ({name or f'opcode {op}'})"
            if name is None or op == OP_DMA_ST:
                raise ProgramError(f"{where}: not an opcode this machine executes")
            if op == OP_LOOP:
                if depth >= LOOP_DEPTH:
                    raise ProgramError(f"{where}: nesting exceeds LOOP_DEPTH={LOOP_DEPTH}")
                if (w0 >> 54) & 0xFF < 1:
                    raise ProgramError(f"{where}: trip count 0 still runs the body once")
                depth += 1
            elif op == OP_ENDLOOP:
                if depth == 0:
                    raise ProgramError(f"{where}: endloop with no open loop")
                depth -= 1
            for t in range(AGU_TERMS):
                tw = (w1 >> (19 * t)) & 0xF
                lw = (w1 >> (19 * t + 4)) & 0x7
                if tw == 0:
                    continue
                if op in (OP_LOOP, OP_ENDLOOP, OP_NOP) or tw > 4 or lw >= depth:
                    raise ProgramError(
                        f"{where}: AGU term {t} targets field {tw - 1} with loop "
                        f"level {lw}, but {depth} loop(s) are open here -- the "
                        f"sequencer would use a stale iv_now[{lw}]")
        if depth:
            raise ProgramError(f"{depth} loop(s) never closed")

        written = {"spad": [False] * p.SPAD_ROWS, "vr": [False] * p.NVR,
                   "ar": [False] * p.NAR}
        ar_wrote = [-self.ar_raw_dist] * p.NAR  # accu iteration of the last write
        it = 0                            # accu iterations issued so far
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
                bad = [r for r in rows if not written[mem][r]]
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

            def ar_read(row, at, what):
                # the distance contract: `at` is the accu iteration of the read
                need("ar", [row], what)
                if at - ar_wrote[row] < self.ar_raw_dist:
                    raise ProgramError(
                        f"{where}: reads ar row {row} as {what} "
                        f"{at - ar_wrote[row]} accu iteration(s) after it was "
                        f"written; the accumulator's dependence claim needs "
                        f">= AR_RAW_DIST={self.ar_raw_dist} (see the "
                        f"accumulator distance contract)")

            def ar_write(row, at):
                written["ar"][row] = True
                ar_wrote[row] = at

            if op == OP_DMA_LD:
                if f0 not in (0, 1, 2, 3):
                    raise ProgramError(f"{where}: f0={f0}, must be source (0 A, 1 B) "
                                       f"| destination (0 spad, 2 vr)")
                if f2 >= p.WPR or f1 + nr > p.MAXDIM:
                    raise ProgramError(f"{where}: DRAM rows {f1}..{f1 + nr - 1}, "
                                       f"col block {f2} outside the "
                                       f"{p.MAXDIM}x{p.MAXDIM} operand")
                for r in span("vr" if f0 & DMA_TO_VR else "spad", f3, nr):
                    written["vr" if f0 & DMA_TO_VR else "spad"][r] = True
            elif op == OP_VLD:
                src = span("spad", f1, nr)
                for d, s in zip(span("vr", f0, nr), src):
                    written["vr"][d] = written["spad"][s]
            elif op == OP_MM:
                if f2 not in (0, 1):
                    raise ProgramError(f"{where}: f2={f2}, must be 0 (overwrite) or 1 (accumulate)")
                need("spad", span("spad", f3, p.T), "weights")
                need("vr", span("vr", f0, nr), "activations")
                dst = span("ar", f1, nr)
                for i, r in enumerate(dst):      # row by row, as `accu` runs it
                    if f2 == 1:
                        ar_read(r, it + i, "the accumulate base")
                    ar_write(r, it + i)
                it += nr
            elif op == OP_VADD:
                s1, s2, dst = (span("ar", f1, nr), span("ar", f2, nr),
                               span("ar", f0, nr))
                for i in range(nr):              # two iterations per row
                    ar_read(s1[i], it + 2 * i, "a source")
                    ar_read(s2[i], it + 2 * i + 1, "a source")
                    ar_write(dst[i], it + 2 * i + 1)
                it += 2 * nr
            elif op == OP_VRELU:
                src, dst = span("ar", f1, nr), span("ar", f0, nr)
                for i in range(nr):
                    ar_read(src[i], it + i, "a source")
                    ar_write(dst[i], it + i)
                it += nr
            elif op == OP_MVOUT:
                for i, r in enumerate(span("ar", f0, nr)):
                    ar_read(r, it + i, "the value to retire")
                it += nr
                if f2 >= p.WPR or f1 + nr > p.MAXDIM:
                    raise ProgramError(f"{where}: C rows {f1}..{f1 + nr - 1}, col "
                                       f"block {f2} outside the "
                                       f"{p.MAXDIM}x{p.MAXDIM} result")

    def assemble(self, prog, check=True):
        """Two words per instruction, behind a header of dynamic per-unit
        counts.

            imem[0] static instruction count   imem[4] mm count | mm rows << 16
            imem[1] dma_ld  rows               imem[5] accu   iterations
            imem[2] spm     rows               imem[6] dma_st rows
            imem[3] vru     rows               imem[7] A rows | B rows << 16

        `check=False` exists only so a test can put a known-bad program on the
        machine and watch it fail; nothing that ships passes it.
        """
        p = self.p
        if check:
            self.check(prog)
        ev = self.expand(prog)

        def rows(*ops):
            return sum(e[1] for e in ev if e[0] in ops)

        def count(*ops):
            return sum(1 for e in ev if e[0] in ops)

        def span(src):
            # The DRAM row span `dma_ld` must burst for one source matrix: the
            # highest row any of its `dma_ld`s names, after the AGU is resolved.
            return max([e[3] + e[1] for e in ev
                        if e[0] == OP_DMA_LD and (e[2] & DMA_SRC_B) == src] + [0])

        a_span, b_span = span(0), span(1)
        assert a_span <= p.MAXDIM and b_span <= p.MAXDIM, (
            f"dma_ld row span {a_span}/{b_span} exceeds MAXDIM={p.MAXDIM}")

        n_mm = count(OP_MM)
        mm_rows = rows(OP_MM)
        assert n_mm < (1 << 15) and mm_rows < (1 << 15), "array counts overflow"
        ld_vr = sum(e[1] for e in ev if e[0] == OP_DMA_LD and e[2] & DMA_TO_VR)
        ld_sp = rows(OP_DMA_LD) - ld_vr
        hdr = [len(prog),
               rows(OP_DMA_LD),
               ld_sp + rows(OP_VLD) + n_mm * (p.T + 1),
               ld_vr + rows(OP_VLD) + mm_rows,
               n_mm | (mm_rows << 16),
               rows(OP_MM, OP_VRELU, OP_MVOUT) + 2 * rows(OP_VADD),
               rows(OP_MVOUT),
               a_span | (b_span << 16)]
        assert len(hdr) == NHDR
        # Every count is read back through a 16-bit slice, which used to
        # extract to a signed ap_int<16>, so the usable range stops at 2^15 - 1.
        for h in hdr[1:4] + hdr[5:7]:
            assert 0 <= h < (1 << 15), f"header count {h} does not fit 15 bits"
        words = list(hdr)
        for w0, w1 in prog:
            words.append(int(w0))
            words.append(int(w1))
        assert len(words) <= p.IMEM_SIZE, (
            f"{len(words)} words > IMEM_SIZE={p.IMEM_SIZE}")
        return words
