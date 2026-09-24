# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What to compute: an einsum over named ranks plus a pointwise epilogue."""

from dataclasses import dataclass, field

import numpy as np

LETTERS = "abcdefghijklmnopqrstuvwxyz"

POINTWISE = {
    "relu": lambda x: np.maximum(x, 0),
}


class SpecError(Exception):
    """A workload spec that does not describe a computation."""


@dataclass(frozen=True)
class Tensor:
    name: str
    ranks: tuple

    def extent(self, extents):
        return tuple(extents[r] for r in self.ranks)


@dataclass(frozen=True)
class Workload:
    name: str
    ranks: tuple
    reduce: tuple
    operands: tuple
    result: Tensor
    contractions: tuple
    epilogue: tuple = ()

    def __post_init__(self):
        known = set(self.ranks)
        if len(known) != len(self.ranks):
            raise SpecError(f"{self.name}: repeated rank in {self.ranks}")
        for t in self.operands + (self.result,):
            unknown = set(t.ranks) - known
            if unknown:
                raise SpecError(
                    f"{self.name}: tensor {t.name} indexes {sorted(unknown)}, "
                    f"which is not in ranks {list(self.ranks)}")
        if set(self.reduce) - known:
            raise SpecError(
                f"{self.name}: reduce {list(self.reduce)} is not a subset of "
                f"ranks {list(self.ranks)}")
        if set(self.result.ranks) != set(self.free):
            raise SpecError(
                f"{self.name}: result {self.result.name} indexes "
                f"{list(self.result.ranks)}, but the free ranks are "
                f"{list(self.free)}")
        for act, wgt in self.contractions:
            covered = set(self.operand(act).ranks) | set(self.operand(wgt).ranks)
            if covered != known:
                raise SpecError(
                    f"{self.name}: contraction {act}x{wgt} covers "
                    f"{sorted(covered)}, not the whole iteration space "
                    f"{sorted(known)}")
        for op in self.epilogue:
            if op not in POINTWISE:
                raise SpecError(
                    f"{self.name}: unknown epilogue op {op!r}; add it to "
                    f"allo.act.workload.POINTWISE and to the target's op table")

    @property
    def free(self):
        return tuple(r for r in self.ranks if r not in self.reduce)

    def operand(self, name):
        for t in self.operands:
            if t.name == name:
                return t
        raise SpecError(f"{self.name}: no operand named {name!r}")

    def subscript(self, tensor):
        return "".join(LETTERS[self.ranks.index(r)] for r in tensor.ranks)

    def evaluate(self, values):
        out = None
        for act, wgt in self.contractions:
            a, w = self.operand(act), self.operand(wgt)
            term = np.einsum(
                f"{self.subscript(a)},{self.subscript(w)}->"
                f"{self.subscript(self.result)}",
                values[act].astype(np.int64), values[wgt].astype(np.int64))
            out = term if out is None else out + term
        for op in self.epilogue:
            out = POINTWISE[op](out)
        return out
