# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A nest in, a TinyTPU-isa program out, with the machine's refusals named."""

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from act.mapspace import divisors  # noqa: E402
from act.nest import (  # noqa: E402
    Refused, check_coverage, check_sequential, order, peel_intrinsic, refuse,
)
from act.target import Target, register  # noqa: E402
from examples.accelerator.tinytpu_vitis import isa_ref  # noqa: E402
from examples.accelerator.tinytpu_vitis.act_machine import MACHINE  # noqa: E402
from examples.accelerator.tinytpu_vitis.isa_dsl import (  # noqa: E402
    NestError, Program, Ref,
)
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    AR_C, AR_RAW_DIST, A_VR, B_SP, MAXDIM, MAXROWS, NAR, NVR, ProgramError,
    SPAD_ROWS, T, assemble, expand,
)

EPILOGUE = {
    "relu": lambda k, dst, src, rows: k.vrelu(dst, src, rows),
}


@dataclass(frozen=True)
class Roles:
    row: str
    reduce: str
    column: str


@dataclass(frozen=True)
class Placement:
    tensor: str
    row: int
    col_block: int
    local: int


def roles_of(workload):
    if len(workload.reduce) != 1 or len(workload.free) != 2:
        raise Refused(
            "shape-class",
            f"{workload.name} reduces {list(workload.reduce)} over free ranks "
            f"{list(workload.free)}; this machine contracts exactly one rank "
            f"into a two-dimensional result")
    reduce = workload.reduce[0]
    for act, wgt in workload.contractions:
        left = set(workload.operand(act).ranks)
        right = set(workload.operand(wgt).ranks)
        if reduce not in left or reduce not in right:
            raise Refused(
                "shape-class",
                f"{workload.name}: contraction {act}x{wgt} does not carry "
                f"{reduce} in both operands")
        rows = left - {reduce}
        cols = right - {reduce}
        if len(rows) != 1 or len(cols) != 1 or rows == cols:
            raise Refused(
                "shape-class",
                f"{workload.name}: contraction {act}x{wgt} is not "
                f"rows x reduce, reduce x columns")
    first = workload.contractions[0]
    row = (set(workload.operand(first[0]).ranks) - {reduce}).pop()
    column = (set(workload.operand(first[1]).ranks) - {reduce}).pop()
    return Roles(row, reduce, column)


def placements(workload, extents, roles):
    out, vr, spad, columns = {}, A_VR, B_SP, 0
    blocks = extents[roles.reduce] // T
    for name in dict.fromkeys(a for a, _ in workload.contractions):
        out[name] = Placement("A", len(out) * extents[roles.row], 0, vr)
        vr += blocks * MAXDIM
    rows = len(out)
    for name in dict.fromkeys(w for _, w in workload.contractions):
        out[name] = Placement("B", 0, columns, spad)
        columns += extents[roles.column] // T
        spad += (extents[roles.column] // T) * MAXDIM
    if vr > NVR or spad > SPAD_ROWS:
        raise Refused(
            "capacity",
            f"{rows} activations need {vr} of NVR={NVR} vector registers and "
            f"the weights need {spad} of SPAD_ROWS={SPAD_ROWS} scratchpad rows")
    if columns * T > MAXDIM or max(p.row for p in out.values()) >= MAXDIM:
        raise Refused(
            "capacity",
            f"the operands do not fit one {MAXDIM}x{MAXDIM} DRAM image: "
            f"{columns} column blocks of weights")
    return out


def accumulators(workload):
    if MAXDIM * len(workload.contractions) > NAR:
        raise Refused(
            "capacity",
            f"{len(workload.contractions)} partial sums need "
            f"{MAXDIM * len(workload.contractions)} of NAR={NAR} rows")
    return {term: AR_C + i * MAXDIM
            for i, term in enumerate(workload.contractions)}


def mixed_radix(open_loops, rank, unit):
    out, step = [], unit
    for loop_rank, iv, factor in reversed(open_loops):
        if loop_rank == rank:
            out.append((iv, step))
            step *= factor
    return out


def walked(base, terms, scale=1):
    address = Ref(base)
    for iv, stride in terms:
        address = address.at(iv, stride * scale)
    return address


class TinyTpu(Target):
    name = "tinytpu-isa"
    machine = MACHINE

    def intrinsics(self, workload, extents):
        roles = roles_of(workload)
        for rows in divisors(extents[roles.row]):
            if rows <= min(MAXROWS, MAXDIM):
                yield {roles.row: rows, roles.reduce: T, roles.column: T}

    def lower(self, workload, extents, nest):
        roles = roles_of(workload)
        for rank, extent in extents.items():
            if extent > MAXDIM:
                raise Refused(
                    "shape", f"rank {rank}={extent} exceeds MAXDIM={MAXDIM}")
        check_sequential(nest)
        check_coverage(nest, extents)
        emitted, intrinsic = peel_intrinsic(nest)
        rows = self._intrinsic_rows(roles, intrinsic)
        reduce_at = [i for i, l in enumerate(emitted) if l.rank == roles.reduce]
        refuse(self._structural(nest, roles, rows, emitted, reduce_at))
        return self._emit(workload, extents, nest, roles, rows,
                          tuple(emitted[:len(emitted) - len(reduce_at)]),
                          emitted[reduce_at[0]].factor if reduce_at else 1)

    def _intrinsic_rows(self, roles, intrinsic):
        if intrinsic.get(roles.reduce) != T or intrinsic.get(roles.column) != T:
            raise Refused(
                "intrinsic",
                f"one `mm` performs a {T}x{T} weight block, the nest asks for "
                f"{intrinsic.get(roles.reduce)}x{intrinsic.get(roles.column)}")
        rows = intrinsic.get(roles.row, 1)
        if not 1 <= rows <= MAXROWS:
            raise Refused(
                "intrinsic", f"rows={rows} exceeds MAXROWS={MAXROWS}")
        return rows

    def _structural(self, nest, roles, rows, emitted, reduce_at):
        out = []
        if len(reduce_at) > 1:
            out.append((
                "acc-peel",
                f"{order(nest)} splits {roles.reduce} across "
                f"{len(reduce_at)} emitted loops, so the first partial sum is "
                f"not a peelable prefix, and `acc` is an instruction field "
                f"with no predicate on an induction variable"))
        elif reduce_at and reduce_at[0] != len(emitted) - 1:
            out.append((
                "acc-peel",
                f"{order(nest)} does not put {roles.reduce} innermost among "
                f"the emitted loops, which peeling the first partial sum "
                f"requires"))
        if rows < AR_RAW_DIST:
            out.append((
                "ar-distance",
                f"rows={rows} makes the accumulating `mm` re-read its own row "
                f"within AR_RAW_DIST={AR_RAW_DIST} accu iterations"))
        return out

    def _emit(self, workload, extents, nest, roles, rows, outer, reduce_trips):
        where = placements(workload, extents, roles)
        into = accumulators(workload)
        blocks = extents[roles.reduce] // T
        k = Program(f"{workload.name} {order(nest)}")
        row_loops = [l for l in outer if l.rank == roles.row]

        def stage_weights():
            for _, weight in workload.contractions:
                spot = where[weight]
                with k.loop(extents[roles.column] // T, "w") as block:
                    k.dma_ld(src=1, dram_row=spot.row,
                             col_block=Ref(spot.col_block).at(block, 1),
                             spad=Ref(spot.local).at(block, MAXDIM),
                             rows=extents[roles.reduce])

        def stage_activations(row_terms):
            for activation, _ in workload.contractions:
                spot = where[activation]
                with k.loop(blocks, "a") as block:
                    k.dma_ld(src=0,
                             dram_row=walked(spot.row, row_terms),
                             col_block=Ref(spot.col_block).at(block, 1),
                             vr=Ref(spot.local).at(block, MAXDIM), rows=rows)

        def body(open_loops):
            row_terms = mixed_radix(open_loops, roles.row, rows)
            column_terms = mixed_radix(open_loops, roles.column, 1)
            for term in workload.contractions:
                activation, weight = term
                k.mm(where[activation].local, into[term],
                     walked(where[weight].local, column_terms, MAXDIM),
                     rows=rows, acc=False)
                if reduce_trips > 1:
                    with k.loop(reduce_trips - 1, "r") as block:
                        k.mm(Ref(where[activation].local + MAXDIM)
                             .at(block, MAXDIM),
                             into[term],
                             walked(where[weight].local + T, column_terms,
                                    MAXDIM).at(block, T),
                             rows=rows, acc=True)
            result = into[workload.contractions[0]]
            for term in workload.contractions[1:]:
                k.vadd(result, result, into[term], rows=rows)
            for op in workload.epilogue:
                EPILOGUE[op](k, result, result, rows)
            k.mvout(result, dram_row=walked(0, row_terms),
                    col_block=walked(0, column_terms), rows=rows)

        def walk(loops, open_loops):
            if not loops:
                body(open_loops)
                return
            head = loops[0]
            with k.loop(head.factor, f"{head.rank}{len(open_loops)}") as iv:
                deeper = open_loops + [(head.rank, iv, head.factor)]
                if head.rank == roles.row and not any(
                        l.rank == roles.row for l in loops[1:]):
                    stage_activations(mixed_radix(deeper, roles.row, rows))
                walk(loops[1:], deeper)

        if not row_loops:
            stage_activations([])
        stage_weights()
        try:
            walk(list(outer), [])
            program = k.emit()
            assemble(program)
        except NestError as refusal:
            raise Refused(refusal.code or "nest", str(refusal)) from refusal
        except ProgramError as refusal:
            raise Refused("machine", str(refusal)) from refusal
        except AssertionError as refusal:
            raise Refused("resources", str(refusal)) from refusal
        return program

    def steps(self, program):
        return self.machine.steps(expand(program))

    def emits(self, program):
        return len(expand(program))

    def report(self, program):
        return {"static": len(program), "words": len(assemble(program)),
                "dynamic": self.emits(program)}

    def images(self, workload, extents, seed=0):
        rng = np.random.default_rng(seed)
        roles = roles_of(workload)
        where = placements(workload, extents, roles)
        images = {"A": np.zeros((MAXDIM, MAXDIM), np.int8),
                  "B": np.zeros((MAXDIM, MAXDIM), np.int8)}
        values = {}
        for name, spot in where.items():
            tensor = workload.operand(name)
            shape = tensor.extent(extents)
            values[name] = rng.integers(-4, 5, shape).astype(np.int8)
            images[spot.tensor][spot.row:spot.row + shape[0],
                                spot.col_block * T:spot.col_block * T
                                + shape[1]] = values[name]
        return images, values

    def verify(self, workload, extents, program, seed=0):
        roles = roles_of(workload)
        images, values = self.images(workload, extents, seed)
        got = isa_ref.run(program, images["A"].reshape(-1),
                          images["B"].reshape(-1),
                          np.zeros(MAXDIM * MAXDIM, np.int8))
        gold = np.clip(workload.evaluate(values), -128, 127).astype(np.int8)
        rows, columns = extents[roles.row], extents[roles.column]
        return np.array_equal(
            got.reshape(MAXDIM, MAXDIM)[:rows, :columns], gold)


TINYTPU = register(TinyTpu())
