# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Access-pattern builders (traced).

These return ``PatternExpr`` nodes describing how an instruction addresses a
buffer. ``strided`` / ``tiled`` are rooted at a buffer; ``expand`` / ``collapse``
relayout a source pattern. Counts / strides / basis / shape entries may be ints or
``IndexExpr`` (address params). Access is value-transparent (a reshape/affine
view); value-reordering relayouts like transpose live in ``prim`` (semantics).
"""

from __future__ import annotations

from .core import IndexExpr, PatternExpr, _as_list


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
    assert (
        buffer.slot_size == 1
    ), f"view: buffer '{buffer.name}' has multi-element slots; use contiguous()"
    dims = [shape] if isinstance(shape, (int, IndexExpr)) else list(shape)
    count = dims[0]
    for d in dims[1:]:
        count = count * d  # IndexExpr-aware (keeps symbolic tile dims symbolic)
    pat = contiguous(buffer, basis, count)
    return pat if len(dims) == 1 else pat.reshape(dims)


def tiled(buffer, basis, counts, strides, tile_sizes) -> PatternExpr:
    """A blocked stream: like ``strided`` but each visited point is a ``tile_sizes``
    block, so the visible tensor gains the tile dims. For loading 2-D sub-blocks."""
    return PatternExpr(
        "tiled",
        buffer=buffer,
        basis=_as_list(basis),
        counts=_as_list(counts),
        strides=_as_list(strides),
        tile_sizes=_as_list(tile_sizes),
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
