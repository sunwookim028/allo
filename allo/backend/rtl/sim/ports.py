# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind numpy kernel arguments to the emitted module's ports"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..interface import FIFO, Memory, ModuleInterface, RegisterFile
from ....lang.core import BufferType, StreamType
from ...marshal import (
    RTL_ABI,
    HostType,
    element_type,
    host_type,
    scalar_from_bits,
    scalar_to_bits,
    to_bits,
)


def bank_elements(shape, axes: tuple[Memory.Axis, ...], bank: int) -> np.ndarray:
    """The flat element indices bank ``bank`` holds, in in-bank offset order.

    The host-side mirror of ``allo::BankLayout``: a cyclic axis of factor ``F``
    puts element ``i`` in bank ``i % F`` at local ``i // F``; a block axis in
    bank ``i // extent`` at ``i % extent``, ``extent = ceil(dim / F)``; a skew
    axis in bank ``(sum of all subscripts) % F``, keeping ``i_d // F`` on its
    distribution dimension ``d``. Axes compose in mixed radix in the order the
    emitter applied them, so the inverse walks them in reverse. ``-1`` marks a
    padding word, a bank slot with no element behind it.
    """
    bank_shape = list(shape)
    peeled = []  # (dim, factor, kind, extent), in the order the emitter applied
    for a in axes:
        extent = -(-bank_shape[a.dim] // a.factor)
        peeled.append((a.dim, a.factor, a.kind, extent))
        bank_shape[a.dim] = extent
    # `bank` in mixed radix over the axis factors, most significant first.
    digits, rest = [], bank
    for _, factor, _, _ in reversed(peeled):
        digits.append(rest % factor)
        rest //= factor
    digits.reverse()
    # Rebuild each original coordinate from this bank's local grid, undoing the
    # axes in reverse.
    coord = list(np.indices(bank_shape))
    for (dim, factor, kind, extent), digit in zip(reversed(peeled), reversed(digits)):
        if kind == "block":
            coord[dim] = coord[dim] + digit * extent
        elif kind == "cyclic":
            coord[dim] = coord[dim] * factor + digit
        else:
            # Skew: `i_d` is the one subscript in [q*F, q*F+F) whose total sum
            # lands on this bank, so the residue is the digit less the others.
            others = sum(coord[k] for k in range(len(shape)) if k != dim)
            coord[dim] = coord[dim] * factor + (digit - others) % factor
    flat, stride, valid = 0, 1, True
    for k in reversed(range(len(shape))):
        flat = flat + coord[k] * stride
        valid = valid & (coord[k] < shape[k])
        stride *= shape[k]
    return np.where(valid, flat, -1).reshape(-1)


@dataclass
class Mem:
    """One backing array behind an external kernel argument (one bank of it when
    the argument is partitioned), with the manifest's :class:`Memory` interfaces
    that read from / write to it. A group, not one interface, so ``arg``/``bank``
    are its identity. ``elements`` is this bank's flat index per in-bank offset,
    where the host's layout meets the RTL's address arithmetic."""

    arg: int
    host: HostType  # element type as it crosses, port width included
    size: int  # elements in this bank (== the flattened argument when unbanked)
    bank: int = 0  # which bank of the argument (0 when unbanked)
    elements: np.ndarray | None = None  # flat index per offset (None = unbanked)
    readers: list[Memory] = field(default_factory=list)
    writers: list[Memory] = field(default_factory=list)

    @property
    def writeback(self) -> bool:
        return bool(self.writers)

    def slice_in(self, array: np.ndarray) -> np.ndarray:
        """This bank's flat uint bit pattern of ``array`` (its own elements for a
        partitioned argument, the whole array otherwise). A padding slot reads 0.
        """
        bits = to_bits(array, self.host)
        if self.elements is None:
            return bits
        return np.where(self.elements >= 0, bits[np.maximum(self.elements, 0)], 0)

    def scatter_out(self, array: np.ndarray, values: np.ndarray) -> None:
        """Write this bank's ``values`` back into ``array`` at its own elements,
        skipping padding slots."""
        if self.elements is None:
            array[...] = values.reshape(array.shape)
            return
        live = self.elements >= 0
        array.reshape(-1)[self.elements[live]] = values[live]


@dataclass
class RegFile:
    """A completely-partitioned argument, held at the boundary as one port per
    element rather than an addressed memory: the read side is a held assignment,
    the write side commits on the edge its ``we`` is high."""

    port: RegisterFile
    host: HostType


def plan_regfiles(interface: ModuleInterface, arg_types) -> list[RegFile]:
    """One :class:`RegFile` per completely-partitioned argument."""
    out = []
    for rf in interface.registers:
        host, size = _elem(arg_types[rf.arg])
        assert size == len(rf.elements), (
            "the manifest declares a port per element, so the count must equal "
            "the flattened argument"
        )
        out.append(RegFile(rf, host))
    return out


def _elem(arg_type) -> tuple[HostType, int]:
    """(element host type, flattened size) for a buffer argument."""
    assert isinstance(arg_type, BufferType), "memory port on a non-buffer argument"
    return element_type(arg_type.dtype, RTL_ABI), int(np.prod(arg_type.shape))


def plan_mems(interface: ModuleInterface, arg_types) -> list[Mem]:
    """Group the interface's read/write ports into one :class:`Mem` per
    (argument, bank): a partitioned argument yields one array per bank."""
    mems: dict[tuple[int, int], Mem] = {}

    def entry(port: Memory) -> Mem:
        key = (port.arg, port.bank)
        if key not in mems:
            host, total = _elem(arg_types[port.arg])
            assert port.width == host.value_bits, (
                f"argument {port.arg}: the manifest declares a {port.width}-bit "
                f"element but the dtype carries {host.value_bits} bits"
            )
            elements = None
            if port.factor > 1:
                # One slot per in-bank offset, padding included, which is
                # exactly the RTL bank's address space.
                elements = bank_elements(port.shape, port.axes, port.bank)
                total = int(elements.size)
            mems[key] = Mem(port.arg, host, total, bank=port.bank, elements=elements)
        return mems[key]

    for acc in interface.reads:
        for r in acc:
            entry(r).readers.append(r)
    for acc in interface.writes:
        for w in acc:
            entry(w).writers.append(w)
    return list(mems.values())


@dataclass
class StreamCh:
    """One FIFO channel bound to a kernel stream argument: an input the host
    feeds token-by-token, or an output it drains."""

    port: FIFO
    host: HostType


def plan_streams(interface: ModuleInterface, arg_types) -> list[StreamCh]:
    """One :class:`StreamCh` per stream port, in interface order."""
    out = []
    for s in interface.streams:
        arg_type = arg_types[s.arg]
        assert isinstance(arg_type, StreamType), "stream port on a non-stream argument"
        out.append(StreamCh(s, element_type(arg_type.base_type, RTL_ABI)))
    return out


def scalar_bits(value, arg_type) -> int:
    """The bit pattern of a scalar argument at its port width."""
    return scalar_to_bits(value, host_type(arg_type, RTL_ABI))


def from_scalar_bits(bits: int, res_type):
    """A result port's bit pattern as the numpy scalar of the return type."""
    return scalar_from_bits(bits, host_type(res_type, RTL_ABI))
