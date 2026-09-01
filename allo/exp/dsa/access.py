# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Access-pattern builders (traced).

These return ``PatternExpr`` nodes describing how an instruction addresses a
buffer. ``strided`` and ``layout`` are rooted at a buffer; ``expand`` / ``collapse``
relayout a source pattern. Counts / strides / basis / shape / ordering entries may be
ints or ``IndexExpr`` (address params). An access changes only where data *lives*
(its address map), never the logical tensor — which is why a ``layout``'s dimension
ordering belongs here while a value-reordering transpose lives in ``prim``
(semantics).

There is no blocked (``tiled``) builder: ``strided`` composed with ``reshape``
already expresses a sub-block. The MLIR-side ``allo.patterns.tiled`` stays, unused
by this frontend.
"""

from __future__ import annotations

from .core import IndexExpr, PatternExpr, _as_list, as_permutation, prod_dims
from .errors import AcceleratorDescriptionError


def strided(buffer, basis, counts, strides) -> PatternExpr:
    """An affine stream: ``counts`` slots per axis at ``basis``, stepping ``strides``
    slots. ``basis`` / ``counts`` / ``strides`` are ints or ``IndexExpr`` (address
    params), scalar or per-axis. The lowest-level access builder."""
    return PatternExpr(
        "strided",
        buffer=buffer,
        basis=_as_list(basis),
        counts=_as_list(counts),
        strides=_as_list(strides),
    )


def contiguous(buffer, basis, counts) -> PatternExpr:
    """A unit-stride block — the common special case of ``strided`` (``counts`` is
    in slots, same as ``strided``)."""
    return strided(buffer, basis, counts, 1)


def view(buffer, basis, shape) -> PatternExpr:
    """A contiguous region of ``buffer`` at ``basis`` seen as a tensor of ``shape``
    (an int for a 1-D run, a tuple for an N-D tile). Sugar for ``contiguous`` of
    ``prod(shape)`` slots, reshaped to ``shape``. Word-addressable (scalar) buffers
    only — multi-element-slot buffers (vector/tile) use ``contiguous`` directly."""
    if buffer.slot_size != 1:
        raise AcceleratorDescriptionError(
            f"view: buffer '{buffer.name}' has multi-element slots; use contiguous()"
        )
    dims = [shape] if isinstance(shape, (int, IndexExpr)) else list(shape)
    pat = contiguous(buffer, basis, prod_dims(dims))
    return pat if len(dims) == 1 else pat.reshape(dims)


def layout(buffer, offset, sizes, order=None) -> PatternExpr:
    """A dense tensor of ``sizes`` based at ``offset``, its dims packed in ``order``.

    ``order`` lists the logical dims outermost (slowest-varying) first, so the strides
    are the suffix products taken in that order: ``(0, 1, 2)`` is row-major and
    ``(2, 0, 1)`` puts dim 2 outermost — MINISA's ``Set*VNLayout`` ordering, and
    ordinary channel-last. It may be

    - omitted, meaning row-major (the host ABI's own packing);
    - an explicit permutation, when the hardware fixes the packing; or
    - an **address param**, when the instruction encodes it. That param is then
      *solved*: an ordering never shows up in a visible shape, so what pins it is the
      residence the value's other accesses describe (see ``core.access_map``).

    The visible shape is ``sizes`` in logical order whatever the ordering — the
    packing changes where the data lives, not which tensor it is. Word-addressable,
    flat buffers only: a layout linearizes elements, so a multi-element slot or a
    multi-extent address space already fixes part of the packing."""
    if buffer.address_rank != 1:
        raise AcceleratorDescriptionError(
            f"layout: buffer '{buffer.name}' has {buffer.address_rank} extents; a "
            f"layout packs a tensor into one flat address space"
        )
    if buffer.slot_size != 1:
        raise AcceleratorDescriptionError(
            f"layout: buffer '{buffer.name}' has multi-element slots, which already "
            f"fix the innermost packing"
        )
    dims = _as_list(sizes)
    if order is None:
        order = tuple(range(len(dims)))
    elif not isinstance(order, IndexExpr):
        order = as_permutation(order, len(dims), "layout")
    return PatternExpr(
        "layout", buffer=buffer, basis=[offset], counts=dims, order=order
    )


def expand(source: PatternExpr, reassociation, shape) -> PatternExpr:
    """Split ``source``'s dims into more dims (value-preserving), e.g. a flat run
    into a tile. ``reassociation`` groups output dims under each source dim; ``shape``
    is the resulting visible shape. Usually reached via ``PatternExpr.reshape``."""
    return PatternExpr(
        "expand",
        source=source,
        reassociation=[list(g) for g in reassociation],
        output_shape=_as_list(shape),
    )


def collapse(source: PatternExpr, reassociation) -> PatternExpr:
    """Merge ``source``'s dims into fewer dims (value-preserving), the inverse of
    ``expand``. ``reassociation`` groups the source dims folded into each output dim.
    Usually reached via ``PatternExpr.reshape``."""
    return PatternExpr(
        "collapse",
        source=source,
        reassociation=[list(g) for g in reassociation],
    )
