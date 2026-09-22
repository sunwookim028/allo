# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A machine as data: spaces, units, and one opcode row per instruction kind."""

from dataclasses import dataclass

from act.schedule import Region, Step


class MachineError(Exception):
    """A machine declaration that does not describe the instruction stream."""


@dataclass(frozen=True)
class Space:
    name: str
    rows: int


@dataclass(frozen=True)
class Opcode:
    code: int
    name: str
    loads: tuple
    reads: tuple = ()
    writes: tuple = ()


@dataclass(frozen=True)
class Machine:
    name: str
    spaces: tuple
    opcodes: tuple

    def space(self, name):
        for s in self.spaces:
            if s.name == name:
                return s
        raise MachineError(f"{self.name}: no space {name!r}")

    def opcode(self, code):
        for op in self.opcodes:
            if op.code == code:
                return op
        raise MachineError(f"{self.name}: opcode {code} is not declared")

    @property
    def units(self):
        return tuple(sorted({u for op in self.opcodes for u, _ in op.loads}))

    def step(self, index, issue):
        op = self.opcode(issue[0])
        fields = dict(zip(("nr", "f0", "f1", "f2", "f3"), issue[1:]))
        busy = ((u, int(w(**fields))) for u, w in op.loads)
        return Step(index=index,
                    loads=tuple((u, w) for u, w in busy if w > 0),
                    reads=tuple(regions(op.reads, fields)),
                    writes=tuple(regions(op.writes, fields)))

    def steps(self, stream):
        return tuple(self.step(i, issue) for i, issue in enumerate(stream))


def regions(accesses, fields):
    for access in accesses:
        got = access(**fields)
        if got is not None:
            yield got if isinstance(got, Region) else Region(*got)
